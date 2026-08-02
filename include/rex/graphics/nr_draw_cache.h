/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_DRAW_CACHE_H_
#define REX_GRAPHICS_NR_DRAW_CACHE_H_

#include <cstdint>

// [NR-CACHE] Exact draw-record cache: native-renderer build-out increment 1.
//
// The draw registry (nr_draw_registry.h) reduced the renderer's cache to
// counting and answered the gate: every executed draw is attributable to a
// recorded one (Run 3, PASS). This unit is the same cache with the counting
// replaced by the actual records a renderer will replay: for every draw the
// guest recorder emits, the hook stores WHERE it was written plus the draw
// arguments it was recorded with (recorder id, primitive type, start index,
// index count). At execution time the command processor asks for the records
// inside the buffer's range and compares them, per draw and in order, against
// the DRAW_INDX packets the buffer actually contains.
//
// What this changes versus the registry, and why:
//
//   * Records carry their exact address, so a query FILTERS to [ptr, ptr+bytes)
//     instead of summing whole granules. Run 3's one systematic artifact -- the
//     21.9% "registry long" bucket caused by edge granules shared with ring
//     neighbours -- is eliminated by construction, not corrected for.
//
//   * The comparison is per-draw argument equality, not a count. If it holds,
//     the hook layer demonstrably captures enough to regenerate the buffer's
//     draw stream, which is the data path the real renderer replays.
//
// What stays, because it was load-bearing and proven:
//
//   * Granules for the index, entering-a-granule-invalidates, no boundary
//     detection anywhere (Ch.9: a segment boundary advances the write pointer
//     monotonically and cannot be told from an ordinary advance).
//
//   * Both sides masked to 0x1FFFFFFF; the registry's alias tally remains the
//     check on that assumption (the game hook feeds both units).
//
//   * Deliberately unsynchronised, same argument as the registry: recording
//     runs on a guest thread inside the engine's raised-IRQL section, a mutex
//     would risk the known lock inversion, and a torn read costs one sample.
//
// Storage is a single fixed pool written in recording order. A granule's
// records are contiguous in the pool (the recorder walks forward; re-entering
// a granule resets it), so the granule table stores {head, count} into the
// pool. The pool can wrap over records of a recording that is still live; a
// query detects that (the slot's address no longer belongs to the granule) and
// reports it as 'evicted' rather than returning another recording's draws.
// The eviction rate is itself a measurement: it says how long recordings stay
// live, which sizes the real per-buffer cache.

namespace rex {
namespace graphics {
namespace nr {

// One recorded draw, exactly as the guest hook saw it.
struct DrawRecord {
  uint32_t addr;   // masked physical address of the draw's packet group start
  uint32_t seq;    // global recording order (rises within a clean recording)
  uint32_t rid;    // which of the three DRAW_INDX recorders fired (0/1/2)
  uint32_t prim;   // r4: primitive type (4 = TRIANGLELIST on this title)
  uint32_t start;  // r6: start index (the recorder scales it x2/x4 to bytes)
  uint32_t count;  // r7: index count (recorder splits packets above 65535)
};

struct CacheQueryResult {
  uint32_t n;               // records written to out[], in recording order
  uint32_t truncated;       // in-range records that did not fit out[]
  uint32_t evicted;         // slots the pool overwrote before this query
  uint32_t seq_inversions;  // recording-order breaks (torn / mixed recording)
};

// Return the records recorded into [phys_addr, phys_addr + bytes), oldest
// first. Cost is one pass over the granules the range covers plus their
// records. Safe to call while recording continues.
CacheQueryResult QueryDraws(uint32_t phys_addr, uint32_t bytes,
                            DrawRecord* out, uint32_t max_out);

// Drop stale-epoch survivors from the front of a query result, in place, and
// return the new count. The current recording pass wrote its draws in one
// forward sweep by a single writer, so its records are CONSECUTIVE in seq
// walking the range's tail backwards; a record in front that breaks the run
// belongs to an older layout, surviving in a granule the new pass wrote only
// state into -- no draw header lands there, so draw-event-keyed invalidation
// cannot reset it, and its seq rises, so the torn detector rightly stays
// quiet. Measured live before this existed: menu buffers read "long, args=0",
// a perfect current-pass match plus a stale front. This is the snapshot
// admission rule the renderer's per-buffer cache uses.
uint32_t TrimStaleFront(DrawRecord* recs, uint32_t n);

struct CacheStats {
  uint64_t recorded;        // draws stored since the last reset
  uint64_t granule_resets;  // invalidations (recorder entering a granule)
  uint64_t collisions;      // granules displaced by a direct-map conflict
};

const CacheStats& GetCacheStats();
void ResetCacheStats();

// Drop everything. Tests only; in the game the cache persists like the
// renderer's own would.
void ResetCache();

}  // namespace nr
}  // namespace graphics
}  // namespace rex

// Called once per recorded draw from the guest D3D9 recorder hook
// ([NARUTO-NRMAP], naruto-recomp/src/record_map.cpp), alongside
// rex_nr_record_draw. Same constraints: guest thread, raised IRQL, so no
// logging, no allocation, no locks -- and none are used.
extern "C" void rex_nr_record_draw_args(uint32_t guest_addr, uint32_t rid,
                                        uint32_t prim, uint32_t start,
                                        uint32_t count);

#endif  // REX_GRAPHICS_NR_DRAW_CACHE_H_
