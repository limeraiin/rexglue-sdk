/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <rex/graphics/nr_shader_db.h>

#include <cstdio>
#include <cstring>

#include <rex/hash.h>  // XXH3, header-only (XXH_INLINE_ALL)

namespace rex {
namespace graphics {
namespace nr {

namespace {

constexpr uint32_t kMagicMask = 0xFFFFFF00u;
constexpr uint32_t kMagicBase = 0x102A1100u;  // ...00 pixel, ...01 vertex
constexpr size_t kContainerHeader = 36;       // 9 big-endian dwords
constexpr uint32_t kConstantTableSize = 0x1C;  // D3DXSHADER_CONSTANTTABLE
constexpr uint32_t kConstantRecordSize = 20;   // D3DXSHADER_CONSTANTINFO
// Leading bytes that bucket the prefix query. Xenos ucode opens with a
// control-flow header, so blobs of the same shader agree here long before
// they diverge; the bucket is only a filter, every hit is memcmp-verified.
constexpr uint32_t kPrefixBytes = 32;

uint32_t BE32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

uint16_t BE16(const uint8_t* p) {
  return uint16_t((uint32_t(p[0]) << 8) | uint32_t(p[1]));
}

// An asciiz string at `offset` inside [base, base + region). Returns nullptr
// unless the string is entirely inside the region -- an unterminated run to
// the end of a mapped file would otherwise read past it.
const char* BoundedString(const uint8_t* base, uint32_t region, uint32_t offset) {
  if (offset >= region) return nullptr;
  for (uint32_t i = offset; i < region; ++i) {
    if (base[i] == 0) return reinterpret_cast<const char*>(base + offset);
  }
  return nullptr;
}

// Fills the constant-table fields of `out`, or leaves them cleared and
// reports which way it failed. `c` is the container base, `virtual_size` its
// metadata region.
enum class CtabResult { kOk, kEmpty, kAbsent, kMalformed };

CtabResult ParseConstantTableHeader(const uint8_t* c, uint32_t virtual_size,
                                    uint32_t ctab_offset,
                                    ShaderContainer* out) {
  if (!ctab_offset || uint64_t(ctab_offset) + 4 > virtual_size) {
    return CtabResult::kAbsent;
  }
  const uint32_t declared = BE32(c + ctab_offset);
  const uint8_t* base = c + ctab_offset + 4;
  const uint32_t available = virtual_size - (ctab_offset + 4);
  if (declared < kConstantTableSize || declared > available) {
    return CtabResult::kMalformed;
  }
  if (BE32(base) != kConstantTableSize) return CtabResult::kMalformed;

  const uint32_t creator_offset = BE32(base + 0x04);
  const uint32_t constants = BE32(base + 0x0C);
  const uint32_t info_offset = BE32(base + 0x10);
  const uint32_t target_offset = BE32(base + 0x18);
  if (constants &&
      (info_offset < kConstantTableSize ||
       uint64_t(info_offset) + uint64_t(constants) * kConstantRecordSize >
           declared)) {
    return CtabResult::kMalformed;
  }

  out->ctab = base;
  out->ctab_bytes = declared;
  out->constant_count = constants;
  out->version_token = BE32(base + 0x08);
  out->target = BoundedString(base, declared, target_offset);
  out->creator = BoundedString(base, declared, creator_offset);
  return constants ? CtabResult::kOk : CtabResult::kEmpty;
}

}  // namespace

uint64_t HashUcode(const uint8_t* ucode, uint32_t bytes) {
  return XXH3_64bits(ucode, bytes);
}

const char* RegisterSetName(uint16_t register_set) {
  switch (ShaderRegisterSet(register_set)) {
    case ShaderRegisterSet::kBool:
      return "bool";
    case ShaderRegisterSet::kInt4:
      return "int4";
    case ShaderRegisterSet::kFloat4:
      return "float4";
    case ShaderRegisterSet::kSampler:
      return "sampler";
    default:
      return "set?";
  }
}

void ScanShaderDatabase(const uint8_t* data, size_t size,
                        std::vector<ShaderContainer>* out,
                        ShaderDbStats* stats) {
  if (!data || !out) return;
  ShaderDbStats local;
  ShaderDbStats& st = stats ? *stats : local;

  for (size_t i = 0; i + kContainerHeader < size;) {
    const uint8_t* c = data + i;
    const uint32_t flags = BE32(c);
    const uint32_t virtual_size = BE32(c + 4);
    const uint32_t physical_size = BE32(c + 8);
    const uint64_t span = uint64_t(virtual_size) + physical_size;
    // The two trailing header dwords are always zero in a real container and
    // are what keeps a bare magic match (common inside texture data) from
    // producing false positives.
    const bool header_ok = (flags & kMagicMask) == kMagicBase &&
                           span >= kContainerHeader && span <= size - i &&
                           BE32(c + 28) == 0 && BE32(c + 32) == 0;
    if (!header_ok) {
      ++i;  // byte stride: containers are not aligned in the database
      continue;
    }

    ShaderContainer container;
    container.file_offset = i;
    container.pixel = (flags & 1) == 0;

    // Ucode via the Shader struct at +shaderOffset: {physicalOffset, size}.
    bool ucode_ok = false;
    const uint32_t shader_offset = BE32(c + 24);
    if (uint64_t(shader_offset) + 24 <= virtual_size) {
      const uint32_t ucode_offset = BE32(c + shader_offset);
      const uint32_t ucode_size = BE32(c + shader_offset + 4);
      if (ucode_size && (ucode_size & 3) == 0 &&
          uint64_t(ucode_offset) + ucode_size <= physical_size) {
        container.ucode = c + virtual_size + ucode_offset;
        container.ucode_bytes = ucode_size;
        container.ucode_hash = HashUcode(container.ucode, ucode_size);
        ucode_ok = true;
      }
    }
    if (!ucode_ok) {
      ++st.bad_ucode_bounds;
      i += size_t(span);
      continue;
    }

    switch (ParseConstantTableHeader(c, virtual_size, BE32(c + 16),
                                     &container)) {
      case CtabResult::kOk:
        break;
      case CtabResult::kEmpty:
        ++st.ctab_empty;
        break;
      case CtabResult::kAbsent:
        ++st.ctab_absent;
        break;
      case CtabResult::kMalformed:
        ++st.ctab_malformed;
        break;
    }
    if (container.ctab) {
      if (BE32(c + 16) + 4 + container.ctab_bytes == shader_offset) {
        ++st.ctab_ends_at_shader;
      }
      // The version token witnesses the stage independently of flags bit 0.
      const uint32_t token_stage = container.version_token >> 16;
      if (token_stage == 0xFFFFu || token_stage == 0xFFFEu) {
        if ((token_stage == 0xFFFFu) != container.pixel) {
          ++st.stage_token_mismatch;
        }
      }
    }

    ++st.containers;
    (container.pixel ? st.pixel : st.vertex)++;
    out->push_back(container);
    i += size_t(span);
  }
}

bool ParseShaderSymbols(const ShaderContainer& container,
                        std::vector<ShaderSymbol>* out) {
  if (!out) return false;
  out->clear();
  if (!container.ctab) return false;
  const uint8_t* base = container.ctab;
  const uint32_t region = container.ctab_bytes;
  if (region < kConstantTableSize) return false;
  const uint32_t count = container.constant_count;
  if (!count) return true;
  const uint32_t info_offset = BE32(base + 0x10);
  if (uint64_t(info_offset) + uint64_t(count) * kConstantRecordSize > region) {
    return false;
  }
  out->reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* rec = base + info_offset + i * kConstantRecordSize;
    ShaderSymbol symbol;
    symbol.name = BoundedString(base, region, BE32(rec));
    symbol.register_set = BE16(rec + 4);
    symbol.register_index = BE16(rec + 6);
    symbol.register_count = BE16(rec + 8);
    out->push_back(symbol);
  }
  return true;
}

// ---- ShaderDbIndex ---------------------------------------------------------

bool ShaderDbIndex::LoadFile(const char* path) {
  loaded_ = false;
  data_.clear();
  containers_.clear();
  by_hash_.clear();
  by_prefix_.clear();
  stats_ = ShaderDbStats();
  if (!path || !*path) return false;

  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  const long size = ftell(f);
  if (size <= 0) {
    fclose(f);
    return false;
  }
  fseek(f, 0, SEEK_SET);
  // Braces, not parens: `data(size_t(size))` declares a function.
  std::vector<uint8_t> data;
  data.resize(size_t(size));
  const size_t read = fread(data.data(), 1, data.size(), f);
  fclose(f);
  if (read != data.size()) return false;
  return LoadBytes(std::move(data));
}

bool ShaderDbIndex::LoadBytes(std::vector<uint8_t> data) {
  loaded_ = false;
  containers_.clear();
  by_hash_.clear();
  by_prefix_.clear();
  stats_ = ShaderDbStats();
  data_ = std::move(data);
  return Index();
}

bool ShaderDbIndex::Index() {
  ScanShaderDatabase(data_.data(), data_.size(), &containers_, &stats_);
  for (size_t i = 0; i < containers_.size(); ++i) {
    const ShaderContainer& c = containers_[i];
    auto it = by_hash_.find(c.ucode_hash);
    if (it != by_hash_.end()) {
      ++it->second.containers;
      continue;
    }
    Entry e;
    e.hash = c.ucode_hash;
    e.ucode_bytes = c.ucode_bytes;
    e.containers = 1;
    e.pixel = c.pixel;
    e.first_container = i;
    by_hash_.emplace(c.ucode_hash, e);
    if (c.ucode_bytes >= kPrefixBytes) {
      by_prefix_.emplace(XXH3_64bits(c.ucode, kPrefixBytes), c.ucode_hash);
    }
  }
  loaded_ = true;
  return true;
}

const ShaderDbIndex::Entry* ShaderDbIndex::Lookup(uint64_t hash) const {
  auto it = by_hash_.find(hash);
  return it == by_hash_.end() ? nullptr : &it->second;
}

ShaderDbIndex::PrefixMatch ShaderDbIndex::FindByPrefix(const uint8_t* ucode,
                                                       uint32_t bytes) const {
  PrefixMatch best;
  if (!ucode || bytes < kPrefixBytes) return best;
  const uint64_t bucket = XXH3_64bits(ucode, kPrefixBytes);
  auto range = by_prefix_.equal_range(bucket);
  // Rank: a container that agrees over its ENTIRE length is the better
  // witness of "same shader, different declared length" than one that merely
  // shares a truncated head; among equals, the longer agreement wins.
  uint64_t best_rank = 0;
  for (auto it = range.first; it != range.second; ++it) {
    auto entry = by_hash_.find(it->second);
    if (entry == by_hash_.end()) continue;
    const ShaderContainer& c = containers_[entry->second.first_container];
    const uint32_t common = c.ucode_bytes < bytes ? c.ucode_bytes : bytes;
    if (std::memcmp(c.ucode, ucode, common) != 0) continue;
    const uint64_t rank =
        (uint64_t(c.ucode_bytes <= bytes ? 1 : 0) << 32) | common;
    if (best.found && rank <= best_rank) continue;
    best_rank = rank;
    best.found = true;
    best.hash = entry->second.hash;
    best.db_bytes = c.ucode_bytes;
    best.equal_bytes = common;
    best.pixel = entry->second.pixel;
  }
  return best;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
