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

// [NR-CACHE] Exact draw-record cache, v3: native-renderer build-out.
//
// For every draw the guest D3D9 recorder emits, the hook stores the draw's
// arguments (recorder id, primitive type, start index, index count) keyed by
// the EXACT physical address of the draw's own DRAW_INDX packet header --
// located by the post-call forward parse in record_map.cpp, exact for 98.6%
// of draws. At execution time the command processor walks the buffer's
// packets (a renderer must anyway, for RT/clear/resolve recovery) and JOINS
// each 0x22 packet to its hook context by header address: LookupDraw(addr).
//
// Why address-keyed (v3), replacing the ordered-granule store (v2): the city
// run `naruto_291` proved THE CITY RECORDER PATCHES BUFFERS IN PLACE --
// individual draws of a stable buffer re-recorded in arbitrary order, not a
// forward sweep. That killed both v2 assumptions at once: pool order is not
// address order under patching (args ~55%), and entering-a-granule-resets-it
// kills a patched granule's untouched neighbours (short 25-33%). The join
// needs neither: insert is an upsert (a re-record of the same packet address
// replaces its record, wherever in the buffer and whenever it happens), and
// there is NO invalidation heuristic at all -- a forward sweep is just
// patching in address order, so menus, forest and city unify. Records of an
// abandoned buffer layout simply never join; they surface as lookup misses,
// not as wrong answers.
//
// What stays from v2, because it was load-bearing and proven:
//
//   * Both sides masked to 0x1FFFFFFF; the registry's alias tally remains the
//     check on that assumption (the game hook feeds both units).
//
//   * Deliberately unsynchronised: recording runs on a guest thread inside
//     the engine's raised-IRQL section, a mutex would risk the known lock
//     inversion, and a torn read costs one sample. The writer stores the key
//     last on a slot claim and the reader re-checks the key after copying, so
//     a race reads as a miss, never as another draw's arguments.
//
// Storage is one direct-mapped hash of 2^20 records (24MB static). Live
// draws measure ~36k in the city (~102 buffers x ~350 draws), a ~3.5% load;
// a different address hashing onto an occupied slot displaces it (counted
// 'evictions' -- the displaced draw will re-miss and re-record), a same-
// address re-record is the upsert (counted 'replaced' -- this is the city's
// patch-in-place, and its rate measures the patch cadence).

namespace rex {
namespace graphics {
namespace nr {

// One recorded draw, exactly as the guest hook saw it.
struct DrawRecord {
  uint32_t addr;   // masked physical address of the draw's DRAW_INDX header
  uint32_t seq;    // global recording order (diagnostic: record age)
  uint32_t rid;    // which of the three DRAW_INDX recorders fired (0/1/2)
  uint32_t prim;   // r4: primitive type (4 = TRIANGLELIST on this title)
  uint32_t start;  // start index (the recorder scales it x2/x4 to bytes)
  uint32_t count;  // index count (recorder splits packets above 65535)
};

// Join one executed DRAW_INDX packet to its recorded draw: look up the record
// whose packet header lives at phys_addr (any alias; masked internally).
// Copies the record into *out and returns true on a hit. A miss means the
// packet was never recorded while the hook was attached, was recorded under
// an abandoned layout at a different address, or was displaced by a
// direct-map collision -- never that some other draw's record was returned.
// Safe to call while recording continues.
bool LookupDraw(uint32_t phys_addr, DrawRecord* out);

// Granule dirty-epochs, for the per-buffer snapshot layer (nr_buffer_cache.h,
// increment 2): every recorded draw also bumps a counter for its 16KB granule
// of physical address, so summing the counters over a range yields a value
// that RISES whenever any draw is recorded into that range. A snapshot
// admitted when the range summed to E is still current iff it still sums to
// E. The map is a keyless direct-mapped 8192-slot counter array: counters
// only increment and a record's own granule is always inside its range's sum,
// so a real patch can never be hidden -- aliasing (two granules 128MB apart
// sharing a slot, or a neighbour's draws in a shared edge granule) only ever
// ADDS dirt, i.e. a false invalidation, which the consumer measures as
// re-admissions with unchanged content.
uint64_t SumRangeEpoch(uint32_t phys_addr, uint32_t bytes);

struct CacheStats {
  uint64_t recorded;   // draws stored since the last reset
  uint64_t replaced;   // upserts: same-address re-records (patch-in-place rate)
  uint64_t evictions;  // records displaced by a direct-map conflict
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
// rex_nr_record_draw. guest_addr is the draw's own DRAW_INDX packet header
// (the post-call parse result). Same constraints: guest thread, raised IRQL,
// so no logging, no allocation, no locks -- and none are used.
extern "C" void rex_nr_record_draw_args(uint32_t guest_addr, uint32_t rid,
                                        uint32_t prim, uint32_t start,
                                        uint32_t count);

#endif  // REX_GRAPHICS_NR_DRAW_CACHE_H_
