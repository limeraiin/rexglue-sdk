/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_draw_cache.h"

// [NR-CACHE] See the header for what this is. Threading model is the
// registry's, for the registry's reasons: one guest-thread writer at raised
// IRQL, one command-processor reader, no synchronisation, every output a rate
// and a torn read one sample. The one ordering rule: on a slot CLAIM (empty
// or displaced) the writer invalidates the key first and stores the new key
// last, and the reader re-checks the key after copying, so the race window
// reads as a miss, never as another draw's arguments.

namespace rex {
namespace graphics {
namespace nr {

namespace {

constexpr uint32_t kPhysMask = 0x1FFFFFFF;
constexpr uint32_t kTableBits = 20;
constexpr uint32_t kTableSize = 1u << kTableBits;  // 24MB of DrawRecord

DrawRecord g_tab[kTableSize] = {};  // addr == 0 means empty
uint32_t g_seq = 0;
CacheStats g_stats = {};

// Granule dirty-epochs (see the header). Keyless by design.
constexpr uint32_t kGranuleShift = 14;  // 16KB
constexpr uint32_t kEpochSlots = 8192;
uint32_t g_gran_epoch[kEpochSlots] = {};

// [NR-STORE] The state-payload ring arena. Sizing: the city stores ~195k
// records/s at ~150 body dwords each (~115 payload + run headers), ~115MB/s,
// so 64MB holds ~0.55s of traffic; a live city record re-stores every ~10
// frames (~0.17s), a ~3x margin. A record whose payload the ring has lapped
// reads as STALE at compose time (header mismatch) and is counted, never
// mis-diffed. Block layout: {addr, gen, ndw} header + ndw body dwords,
// contiguous (the ring wraps to 0 rather than splitting a block).
constexpr uint32_t kStateArenaDwords = 16u << 20;  // 64MB
constexpr uint32_t kStateHdrDwords = 3;
constexpr uint32_t kStateMaxBody = 4096;  // worst captured group ~2.8k dwords
uint32_t g_state_arena[kStateArenaDwords];
uint32_t g_state_head = 0;
uint32_t g_state_gen = 0;

// Packet headers are word-aligned; multiplicative hash, top bits.
inline uint32_t Slot(uint32_t addr) {
  return ((addr >> 2) * 2654435761u) >> (32 - kTableBits);
}

}  // namespace

bool LookupDraw(uint32_t phys_addr, DrawRecord* out) {
  const uint32_t addr = phys_addr & kPhysMask;
  if (!addr) return false;
  const DrawRecord& slot = g_tab[Slot(addr)];
  if (slot.addr != addr) return false;
  *out = slot;
  // Re-check after the copy: a concurrent displacement of this slot by a
  // different address can otherwise hand back a half-written record under the
  // old key. A same-address upsert racing the copy can still mix old and new
  // ARGUMENTS -- that is the accepted one-sample tear, same as v2.
  return out->addr == addr;
}

uint64_t SumRangeEpoch(uint32_t phys_addr, uint32_t bytes) {
  if (!bytes) return 0;
  const uint32_t p = phys_addr & kPhysMask;
  const uint32_t first = p >> kGranuleShift;
  const uint32_t last = (p + bytes - 1) >> kGranuleShift;
  uint64_t sum = 0;
  for (uint32_t g = first; g <= last; ++g) {
    sum += g_gran_epoch[g & (kEpochSlots - 1)];
  }
  return sum;
}

uint64_t EpochActivity() { return g_seq; }

const uint32_t* AcquireState(const DrawRecord& rec, uint32_t* out_ndw) {
  if (!rec.state_gen) return nullptr;
  const uint32_t off = rec.state_off;
  if (off + kStateHdrDwords > kStateArenaDwords) return nullptr;
  const uint32_t* blk = &g_state_arena[off];
  const uint32_t ndw = blk[2];
  if (blk[0] != rec.addr || blk[1] != rec.state_gen) return nullptr;
  if (ndw == 0 || ndw > kStateMaxBody ||
      off + kStateHdrDwords + ndw > kStateArenaDwords) {
    return nullptr;  // header torn mid-overwrite: same stale class
  }
  *out_ndw = ndw;
  return blk + kStateHdrDwords;
}

bool VerifyState(const DrawRecord& rec) {
  if (!rec.state_gen || rec.state_off + kStateHdrDwords > kStateArenaDwords) {
    return false;
  }
  const uint32_t* blk = &g_state_arena[rec.state_off];
  return blk[0] == rec.addr && blk[1] == rec.state_gen;
}

const CacheStats& GetCacheStats() { return g_stats; }

void ResetCacheStats() { g_stats = CacheStats{}; }

void ResetCache() {
  for (uint32_t i = 0; i < kTableSize; ++i) g_tab[i].addr = 0;
  for (uint32_t i = 0; i < kEpochSlots; ++i) g_gran_epoch[i] = 0;
  g_seq = 0;
  g_state_head = 0;
  g_state_gen = 0;
  ResetCacheStats();
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex

extern "C" void rex_nr_record_draw_args_state(uint32_t guest_addr, uint32_t rid,
                                              uint32_t prim, uint32_t start,
                                              uint32_t count, uint32_t state_off,
                                              uint32_t state_gen, uint32_t vs,
                                              uint32_t ps) {
  using namespace rex::graphics::nr;
  const uint32_t addr = guest_addr & kPhysMask;
  if (!addr) return;
  DrawRecord* slot = &g_tab[Slot(addr)];

  if (slot->addr == addr) {
    // The upsert: this packet address re-recorded. Under the city's
    // patch-in-place model this is the common case for stable buffers.
    ++g_stats.replaced;
  } else {
    if (slot->addr) ++g_stats.evictions;
    // Invalidate before mutating so a concurrent lookup misses instead of
    // reading a mixed record under either key.
    slot->addr = 0;
  }
  slot->seq = ++g_seq;
  slot->rid = rid;
  slot->prim = prim;
  slot->start = start;
  slot->count = count;
  slot->state_off = state_off;
  slot->state_gen = state_gen;
  slot->vs = vs;
  slot->ps = ps;
  slot->addr = addr;  // key last: the record becomes visible complete
  g_gran_epoch[(addr >> kGranuleShift) & (kEpochSlots - 1)] += 1;
  ++g_stats.recorded;
  if (state_gen) ++g_stats.state_stored;
}

extern "C" void rex_nr_record_draw_args(uint32_t guest_addr, uint32_t rid,
                                        uint32_t prim, uint32_t start,
                                        uint32_t count) {
  // Args-only path: clears the state fields so a re-record that could not
  // capture never leaves the record pointing at a payload for stale bytes.
  rex_nr_record_draw_args_state(guest_addr, rid, prim, start, count, 0, 0, 0,
                                0);
}

extern "C" uint32_t* rex_nr_state_alloc(uint32_t guest_addr, uint32_t body_ndw,
                                        uint32_t* out_off, uint32_t* out_gen) {
  using namespace rex::graphics::nr;
  const uint32_t addr = guest_addr & kPhysMask;
  if (!addr) return nullptr;
  if (!body_ndw || body_ndw > kStateMaxBody) {
    ++g_stats.state_ovf;
    return nullptr;
  }
  const uint32_t need = kStateHdrDwords + body_ndw;
  uint32_t off = g_state_head;
  if (off + need > kStateArenaDwords) {
    off = 0;
    ++g_stats.state_wraps;
  }
  g_state_head = off + need;
  uint32_t* blk = &g_state_arena[off];
  // Header first: a reader holding an older record that pointed here sees the
  // new {addr, gen} and reads its own payload as stale instead of as ours.
  blk[0] = addr;
  blk[1] = ++g_state_gen;
  blk[2] = body_ndw;
  *out_off = off;
  *out_gen = g_state_gen;
  g_stats.state_dwords += body_ndw;
  return blk + kStateHdrDwords;
}
