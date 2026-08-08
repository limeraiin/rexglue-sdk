/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// [NR-SC] Native-renderer phase 5, increment 5-2: the ucode -> DXBC cache.
// See include/rex/graphics/nr_shader_cache.h for what this owns and why.

#include "rex/graphics/nr_shader_cache.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace rex {
namespace graphics {
namespace nr {

namespace {

// Open addressing with linear probing at a load factor of at most 1/2. A key
// that cannot be placed within the probe limit is refused and counted rather
// than silently displacing another key: the 4b-2 lesson, where a table that
// quietly overflowed made a distinct-count read as coverage it did not have.
constexpr uint32_t kProbeLimit = 32;

uint64_t Mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t KeyHash(uint32_t stage, uint64_t ucode_hash, uint64_t modification) {
  return Mix64(ucode_hash ^ Mix64(modification + stage + 1));
}

uint32_t NextPow2(uint32_t v) {
  uint32_t p = 1;
  while (p < v) {
    p <<= 1;
  }
  return p;
}

// One ucode blob, however many modifications it has been translated under.
struct UcodeRec {
  uint64_t hash;
  uint32_t stage_mask;
  uint32_t mods;
};

struct Cache {
  NrShaderTranslateFn translate = nullptr;
  void* ctx = nullptr;
  uint32_t max_entries = 0;
  uint64_t max_bytes = 0;

  std::vector<NrShaderEntry> entries;
  std::vector<uint32_t> slots;  // index + 1, 0 = empty

  std::vector<UcodeRec> ucodes;
  std::vector<uint32_t> ucode_slots;

  NrShaderCacheStats stats;
};

Cache g_cache;

// Counts one {ucode, stage} sighting and, when the key is new, one more
// modification for that ucode. Returns nothing: this is bookkeeping for the
// AOT question, never a lookup path.
void NoteUcode(uint64_t ucode_hash, uint32_t stage) {
  if (g_cache.ucode_slots.empty()) {
    return;
  }
  const uint32_t mask = uint32_t(g_cache.ucode_slots.size()) - 1;
  uint32_t slot = uint32_t(Mix64(ucode_hash)) & mask;
  for (uint32_t i = 0; i < kProbeLimit; ++i) {
    uint32_t& ref = g_cache.ucode_slots[slot];
    if (ref) {
      UcodeRec& rec = g_cache.ucodes[ref - 1];
      if (rec.hash == ucode_hash) {
        ++rec.mods;
        rec.stage_mask |= 1u << stage;
        if (rec.mods > g_cache.stats.mods_max) {
          g_cache.stats.mods_max = rec.mods;
          g_cache.stats.mods_max_hash = rec.hash;
        }
        return;
      }
    } else {
      if (g_cache.ucodes.size() >= g_cache.ucodes.capacity()) {
        ++g_cache.stats.probe_ovf;
        return;
      }
      g_cache.ucodes.push_back(UcodeRec{ucode_hash, 1u << stage, 1});
      ref = uint32_t(g_cache.ucodes.size());
      ++g_cache.stats.ucodes;
      if (stage == kNrShaderStagePixel) {
        ++g_cache.stats.ps_ucodes;
      } else {
        ++g_cache.stats.vs_ucodes;
      }
      if (g_cache.stats.mods_max < 1) {
        g_cache.stats.mods_max = 1;
        g_cache.stats.mods_max_hash = ucode_hash;
      }
      return;
    }
    slot = (slot + 1) & mask;
  }
  ++g_cache.stats.probe_ovf;
}

}  // namespace

void NrShaderCacheConfigure(NrShaderTranslateFn fn, void* ctx, uint32_t max_entries,
                            uint64_t max_bytes) {
  if (max_entries < 16) {
    max_entries = 16;
  }
  const bool resize = g_cache.max_entries != max_entries;
  g_cache.translate = fn;
  g_cache.ctx = ctx;
  g_cache.max_bytes = max_bytes;
  if (!resize) {
    return;
  }
  g_cache.max_entries = max_entries;
  g_cache.entries.clear();
  g_cache.entries.shrink_to_fit();
  // Reserved once and never exceeded, so an entry pointer handed to a draw
  // stays valid for the life of the cache. Every insert checks size against
  // max_entries first, so this vector never reallocates.
  g_cache.entries.reserve(max_entries);
  g_cache.slots.assign(NextPow2(max_entries * 2), 0);
  g_cache.ucodes.clear();
  g_cache.ucodes.shrink_to_fit();
  g_cache.ucodes.reserve(max_entries);
  g_cache.ucode_slots.assign(NextPow2(max_entries * 2), 0);
  g_cache.stats = NrShaderCacheStats{};
}

bool NrShaderCacheConfigured() { return g_cache.translate != nullptr && g_cache.max_entries != 0; }

void NrShaderCacheReset() {
  g_cache.entries.clear();
  std::fill(g_cache.slots.begin(), g_cache.slots.end(), 0u);
  g_cache.ucodes.clear();
  std::fill(g_cache.ucode_slots.begin(), g_cache.ucode_slots.end(), 0u);
  g_cache.stats = NrShaderCacheStats{};
}

NrShaderEntry* NrShaderCacheLookup(uint32_t stage, uint64_t ucode_hash,
                                   const uint32_t* ucode_dwords, uint32_t ucode_dword_count,
                                   uint64_t modification) {
  if (!NrShaderCacheConfigured() || stage >= kNrShaderStageCount) {
    return nullptr;
  }
  NrShaderCacheStats& st = g_cache.stats;
  ++st.lookups;

  const uint32_t mask = uint32_t(g_cache.slots.size()) - 1;
  uint32_t slot = uint32_t(KeyHash(stage, ucode_hash, modification)) & mask;
  uint32_t* free_slot = nullptr;
  for (uint32_t i = 0; i < kProbeLimit; ++i) {
    uint32_t& ref = g_cache.slots[slot];
    if (!ref) {
      free_slot = &ref;
      break;
    }
    NrShaderEntry& entry = g_cache.entries[ref - 1];
    if (entry.stage == stage && entry.ucode_hash == ucode_hash &&
        entry.modification == modification) {
      ++st.hits;
      if (!entry.valid) {
        ++st.invalid_hits;
      }
      return &entry;
    }
    slot = (slot + 1) & mask;
  }
  if (!free_slot) {
    ++st.probe_ovf;
    ++st.refused;
    return nullptr;
  }
  if (g_cache.entries.size() >= g_cache.max_entries) {
    ++st.refused;
    return nullptr;
  }

  // Miss: translate now, on this thread. Misses are first sightings of a
  // {shader, modification} pair, so this is the hitch a native renderer pays
  // once per pair; it is timed rather than assumed.
  ++st.misses;
  std::vector<uint8_t> dxbc;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = ucode_dwords && ucode_dword_count &&
                  g_cache.translate(g_cache.ctx, stage, ucode_hash, ucode_dwords,
                                    ucode_dword_count, modification, &dxbc);
  const uint64_t ns =
      uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - t0)
                   .count());
  ++st.translations;
  st.translate_ns_total += ns;
  if (ns > st.translate_ns_max) {
    st.translate_ns_max = ns;
    st.translate_ns_max_hash = ucode_hash;
  }
  if (!ok) {
    dxbc.clear();
    ++st.translate_fail;
  } else if (st.dxbc_bytes + dxbc.size() > g_cache.max_bytes) {
    // Out of byte budget: refuse without recording the key, so the cache never
    // reports a shader it cannot hand back.
    ++st.byte_refused;
    ++st.refused;
    return nullptr;
  }

  NrShaderEntry entry;
  entry.ucode_hash = ucode_hash;
  entry.modification = modification;
  entry.stage = stage;
  entry.valid = ok;
  entry.dxbc = std::move(dxbc);
  st.dxbc_bytes += entry.dxbc.size();
  g_cache.entries.push_back(std::move(entry));
  *free_slot = uint32_t(g_cache.entries.size());
  ++st.entries;
  if (stage == kNrShaderStagePixel) {
    ++st.ps_entries;
  } else {
    ++st.vs_entries;
  }
  NoteUcode(ucode_hash, stage);
  return &g_cache.entries.back();
}

uint32_t NrShaderCacheVerify(NrShaderEntry* entry, const uint8_t* theirs, uint64_t theirs_size) {
  if (!entry) {
    return kNrShaderVerifyAlready;
  }
  NrShaderCacheStats& st = g_cache.stats;
  if (entry->verified) {
    return kNrShaderVerifyAlready;
  }
  if (!theirs) {
    // Their translation has not happened yet (async compilation). Leave the
    // key unverified so a later draw settles it.
    ++st.pending;
    return kNrShaderVerifyPending;
  }
  entry->verified = true;
  ++st.verified;
  if (!entry->valid) {
    ++st.valid_ne;
    return kNrShaderVerifyValidNe;
  }

  const uint64_t ours_size = uint64_t(entry->dxbc.size());
  const uint64_t common = ours_size < theirs_size ? ours_size : theirs_size;
  uint64_t diff_at = common;
  for (uint64_t i = 0; i < common; ++i) {
    if (entry->dxbc[size_t(i)] != theirs[size_t(i)]) {
      diff_at = i;
      break;
    }
  }
  const bool sizes_equal = ours_size == theirs_size;
  if (sizes_equal && diff_at == common) {
    entry->agreed = true;
    ++st.agreed;
    return kNrShaderVerifyAgreed;
  }
  if (!st.have_first_ne) {
    st.have_first_ne = true;
    st.first_ne_hash = entry->ucode_hash;
    st.first_ne_modification = entry->modification;
    st.first_ne_stage = entry->stage;
    st.first_ne_offset = diff_at;
    st.first_ne_ours = diff_at < ours_size ? entry->dxbc[size_t(diff_at)] : 0xFFFFFFFFu;
    st.first_ne_theirs = diff_at < theirs_size ? theirs[size_t(diff_at)] : 0xFFFFFFFFu;
    st.first_ne_ours_size = ours_size;
    st.first_ne_theirs_size = theirs_size;
  }
  if (!sizes_equal) {
    ++st.size_ne;
    return kNrShaderVerifySizeNe;
  }
  ++st.bytes_ne;
  return kNrShaderVerifyBytesNe;
}

const NrShaderCacheStats& NrShaderCacheGetStats() { return g_cache.stats; }

void NrShaderCacheEndWindow() {
  NrShaderCacheStats& st = g_cache.stats;
  st.lookups = st.hits = st.misses = st.refused = st.pending = st.invalid_hits = 0;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
