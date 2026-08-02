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
// and a torn read one sample.

namespace rex {
namespace graphics {
namespace nr {

namespace {

constexpr uint32_t kGranuleShift = 14;  // 16KB, matching the registry
constexpr uint32_t kTableSize = 8192;   // direct-mapped granule index
constexpr uint32_t kPhysMask = 0x1FFFFFFF;

// The pool holds recording history, not live state: a record only has to
// survive from the moment it is written until the last execution of its
// buffer (or until the buffer is re-recorded, whichever comes first). At the
// city's ~93k recorded draws/s, 2^20 records is ~11 seconds of history --
// far beyond the measured re-record cadence of a few frames. Buffers that
// outlive it show up in the 'evicted' bucket, which is a number worth having:
// it is the lifetime distribution the real per-buffer cache must be sized for.
constexpr uint32_t kPoolBits = 20;
constexpr uint32_t kPoolSize = 1u << kPoolBits;
constexpr uint32_t kPoolMask = kPoolSize - 1;

struct GranuleEntry {
  uint32_t key;    // (physical address >> kGranuleShift) + 1; 0 means empty
  uint32_t head;   // pool index (unmasked) of the granule's first record
  uint32_t count;  // records since the recorder entered this granule
};

DrawRecord g_pool[kPoolSize] = {};
GranuleEntry g_tab[kTableSize] = {};
uint32_t g_pool_widx = 0;  // unmasked; mask on access
uint32_t g_cur_key = 0;    // granule the recorder is currently writing inside
uint32_t g_seq = 0;
CacheStats g_stats = {};

}  // namespace

CacheQueryResult QueryDraws(uint32_t phys_addr, uint32_t bytes, DrawRecord* out,
                            uint32_t max_out) {
  CacheQueryResult r = {};
  if (!bytes) return r;
  const uint32_t p = phys_addr & kPhysMask;
  const uint32_t first = p >> kGranuleShift;
  const uint32_t last = (p + bytes - 1) >> kGranuleShift;
  uint32_t prev_seq = 0;
  bool have_prev = false;
  for (uint32_t g = first; g <= last; ++g) {
    const GranuleEntry& e = g_tab[g & (kTableSize - 1)];
    if (e.key != g + 1) continue;
    // Snapshot count once; the writer may append while we walk, and a bounded
    // stale walk is one sample while an unbounded one is a hang.
    uint32_t n = e.count;
    if (n > kPoolSize) n = kPoolSize;
    const uint32_t head = e.head;
    for (uint32_t k = 0; k < n; ++k) {
      const DrawRecord& rec = g_pool[(head + k) & kPoolMask];
      // A slot the pool ring has overwritten belongs to some other granule
      // now. Report the loss instead of returning another recording's draw.
      if ((rec.addr >> kGranuleShift) != g) {
        ++r.evicted;
        continue;
      }
      // The exact filter that the counting registry could not do: edge
      // granules also hold the ring-neighbour's records, and this excludes
      // them by address instead of summing them in.
      if (rec.addr < p || rec.addr - p >= bytes) continue;
      if (have_prev && rec.seq < prev_seq) ++r.seq_inversions;
      prev_seq = rec.seq;
      have_prev = true;
      if (r.n < max_out) {
        out[r.n++] = rec;
      } else {
        ++r.truncated;
      }
    }
  }
  return r;
}

uint32_t TrimStaleFront(DrawRecord* recs, uint32_t n) {
  if (n < 2) return n;
  uint32_t first = n - 1;
  while (first > 0 && recs[first - 1].seq + 1 == recs[first].seq) {
    --first;
  }
  if (first) {
    for (uint32_t i = first; i < n; ++i) recs[i - first] = recs[i];
    n -= first;
  }
  return n;
}

const CacheStats& GetCacheStats() { return g_stats; }

void ResetCacheStats() { g_stats = CacheStats{}; }

void ResetCache() {
  for (uint32_t i = 0; i < kTableSize; ++i) g_tab[i] = GranuleEntry{};
  // The pool itself need not be cleared: nothing references a record except
  // through a granule entry, and those are gone. Tests that reset repeatedly
  // would otherwise pay 24MB of writes each time.
  g_pool_widx = 0;
  g_cur_key = 0;
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
  const uint32_t key = (addr >> kGranuleShift) + 1;
  GranuleEntry* e = &g_tab[(key - 1) & (kTableSize - 1)];

  if (key != g_cur_key) {
    // The recorder just arrived here: whatever the granule held is a previous
    // recording being overwritten. Same single rule as the registry -- this IS
    // the invalidate-on-re-record policy, and no boundary detection exists
    // because none can (Ch.9: boundaries advance the pointer monotonically).
    if (e->key && e->key != key) ++g_stats.collisions;
    e->key = key;
    e->head = g_pool_widx;
    e->count = 0;
    g_cur_key = key;
    ++g_stats.granule_resets;
  } else if (e->key != key) {
    // Same granule as the last draw, but a direct-map conflict displaced the
    // slot in between. Reclaim it; the other granule's records are orphaned
    // and will read as evicted, never as ours.
    ++g_stats.collisions;
    e->key = key;
    e->head = g_pool_widx;
    e->count = 0;
  }

  DrawRecord* rec = &g_pool[g_pool_widx & kPoolMask];
  rec->addr = addr;
  rec->seq = ++g_seq;
  rec->rid = rid;
  rec->prim = prim;
  rec->start = start;
  rec->count = count;
  ++g_pool_widx;
  ++e->count;
  ++g_stats.recorded;
}
