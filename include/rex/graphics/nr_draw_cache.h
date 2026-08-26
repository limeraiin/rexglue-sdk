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
//
// [NR-STORE] N-9-3d extends the record with the CAPTURED STATE PAYLOAD: the
// type0 surface the recorder body flushed for this draw, predicted from the
// pre-hook dirty masks + device shadows (the N-9-3 capture, validated exact
// at title and city). The payload itself is variable-size and lives in a
// separate ring arena (see rex_nr_state_alloc below); the record carries only
// {offset, generation}. state_gen == 0 means "no payload" (store off, rid 0,
// parse fallback, or an alloc refusal) and the compose side counts it.
struct DrawRecord {
  uint32_t addr;   // masked physical address of the draw's DRAW_INDX header
  uint32_t seq;    // global recording order (diagnostic: record age)
  // [NR-5B-2b] liveness stamp for the retained store's reclamation: the
  // reader (LookupDraw) writes the current recording seq here on every hit
  // (unsynchronised u32; a torn stamp costs one sweep round). The writer's
  // incremental sweep expires records neither stored nor joined for a
  // horizon -- forward-advancing phases (boot/intro ~45k NEW addrs/s,
  // naruto_681) abandon addresses at rates no fixed arena survives.
  uint32_t last_use;
  uint32_t rid;    // which of the three DRAW_INDX recorders fired (0/1/2)
  uint32_t prim;   // r4: primitive type (4 = TRIANGLELIST on this title)
  uint32_t start;  // start index (the recorder scales it x2/x4 to bytes)
  uint32_t count;  // index count (recorder splits packets above 65535)
  uint32_t state_off;  // dword offset of the payload block in the state arena
  uint32_t state_gen;  // arena generation stamp; 0 = no state payload
  uint32_t vs;     // shader object pointers at record time (dev+12684/12688):
  uint32_t ps;     //   per-draw shader identity (city churn is 69%/59%)
  // [NR-5C] reader-owned (command processor) skip-unchanged stamps: the
  // state_gen this record last FULLY applied under, and the apply sequence
  // it was stamped at (see g_nrb_apply_seq). The writer clears applied_gen
  // on every upsert, so a re-recorded payload can never read as applied.
  uint32_t applied_gen;
  uint32_t applied_seq;
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

// Nonzero once the guest recorder hook has ever recorded a draw (i.e. the
// epoch counters are actually being fed). A consumer that would SKIP work on
// an unchanged epoch sum must refuse while this is zero: a dead hook reads
// every range as eternally unchanged.
uint64_t EpochActivity();

struct CacheStats {
  uint64_t recorded;   // draws stored since the last reset
  uint64_t replaced;   // upserts: same-address re-records (patch-in-place rate)
  uint64_t evictions;  // records displaced by a direct-map conflict
  // [NR-STORE] state-payload store traffic ([NR-5B-2]: retained free-list
  // allocator -- payload lifetime == record lifetime, no ring laps).
  uint64_t state_stored;  // payload blocks published
  uint64_t state_dwords;  // body dwords written
  uint64_t state_ovf;     // alloc refusals: block over the size cap
  uint64_t state_freed;   // blocks freed by record replace/evict/clear
  uint64_t state_nomem;   // alloc refusals: arena exhausted (expect 0)
  uint64_t state_badfree; // frees refused on an implausible header (leak if >0)
  uint64_t state_expired; // records reclaimed by the dormancy sweep
  // [NR-5B-2b] policy witness: a JOIN HIT on a record dormant past the
  // horizon (it would have been expired had the sweep reached it). Must
  // read ~0 before 5b-3 makes payloads load-bearing.
  uint64_t dormant_rejoin;
  // Gauges (survive ResetCacheStats): current allocator state.
  uint64_t state_live;    // blocks currently allocated
  uint64_t state_bump_dw; // virgin-space watermark, dwords
};

// [NR-STORE] Read one record's state payload on the execute side. Validates
// the arena block's {addr, gen} header against the record BEFORE handing out
// the body; the caller must call VerifyState AFTER consuming it and discard
// everything if that fails (the writer wrapped the arena over the block
// mid-read). Returns null (and the caller counts a stale payload) when the
// block was already overwritten. Same unsynchronised model as LookupDraw.
const uint32_t* AcquireState(const DrawRecord& rec, uint32_t* out_ndw);
bool VerifyState(const DrawRecord& rec);

// [NR-5C] Stamp the record at phys_addr as fully applied under {gen, seq}
// (see DrawRecord::applied_gen). Reader-side unsynchronised u32 stores, same
// model as LookupDraw's last_use stamp; a stamp racing a same-address upsert
// can only pair an OLD gen with the slot, which reads as not-applied.
void StampApplied(uint32_t phys_addr, uint32_t gen, uint32_t seq);

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

// [NR-STORE] N-9-3d: the record store carries the captured state payload.
//
// Protocol, all on the guest recorder thread (raised IRQL: no logging, no
// allocation, no locks -- the arena is a static ring):
//
//   1. rex_nr_state_alloc(addr, body_ndw, &off, &gen) claims a contiguous
//      block in the ring, stamps its {addr, gen, ndw} header, and returns a
//      host pointer to the body (null = over the size cap; counted). The
//      generation is globally monotonic, so a reader can always tell "this
//      block still belongs to this record" from the header alone.
//   2. The hook fills the body: [0] = nruns, then per run {reg, cnt} followed
//      by cnt payload dwords IN RAW GUEST BYTE ORDER (memcpy straight from
//      the device shadows; synthetic values byte-swapped at store time), in
//      the body's emission order -- exactly the CapPredict output plus the
//      draw loop's own {0x2102, base_vertex} single for rid 1/2.
//   3. rex_nr_record_draw_args_state publishes the block by upserting the
//      full record (args + state_off/state_gen + shader ptrs), key last.
//
// rex_nr_record_draw_args (above) remains the args-only path and CLEARS the
// state fields -- a re-record that could not capture (parse fallback, rid 0)
// must not leave the record pointing at a payload for different bytes.
extern "C" uint32_t* rex_nr_state_alloc(uint32_t guest_addr, uint32_t body_ndw,
                                        uint32_t* out_off, uint32_t* out_gen);
extern "C" void rex_nr_record_draw_args_state(uint32_t guest_addr, uint32_t rid,
                                              uint32_t prim, uint32_t start,
                                              uint32_t count, uint32_t state_off,
                                              uint32_t state_gen, uint32_t vs,
                                              uint32_t ps);

#endif  // REX_GRAPHICS_NR_DRAW_CACHE_H_
