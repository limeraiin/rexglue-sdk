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

const CacheStats& GetCacheStats() { return g_stats; }

void ResetCacheStats() { g_stats = CacheStats{}; }

void ResetCache() {
  for (uint32_t i = 0; i < kTableSize; ++i) g_tab[i].addr = 0;
  g_seq = 0;
  ResetCacheStats();
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex

extern "C" void rex_nr_record_draw_args(uint32_t guest_addr, uint32_t rid,
                                        uint32_t prim, uint32_t start,
                                        uint32_t count) {
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
  slot->addr = addr;  // key last: the record becomes visible complete
  ++g_stats.recorded;
}
