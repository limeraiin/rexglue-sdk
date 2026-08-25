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

// [NR-STORE] The state-payload store. Was a ring arena (N-9-3d..5a); now
// [NR-5B-2] a size-class free-list allocator, because live suppression
// needs RETENTION: a STABLE buffer never re-records, so its records'
// payloads must live exactly as long as the records -- under the ring they
// lapped (~800 stale/s at city), which reads as a counted fallback while
// packets exist and as a draw with NO constants once they do not.
//
// Shape: blocks of {addr, gen, ndw} header + body, carved from the static
// arena at power-of-two class sizes (32..8192 dwords); a virgin bump
// pointer serves a class until its free list has a block. Lifetime: the
// record-table upsert FREES the previous payload of the slot it overwrites
// (replace, evict, and the args-only clear) AFTER the new record is
// published. Freeing stamps gen = 0 (never a valid generation) and threads
// the free list through blk[0], so a reader still holding the old record
// fails Acquire/Verify deterministically -- the same stale/torn classes as
// before, now rare instead of structural. Single guest-thread writer, no
// locks, no OS allocation, exactly as before.
//
// Sizing: ~36k live city draws x ~150 body dwords ~= 22MB live; 64MB with
// pow2 rounding leaves >2x headroom. Exhaustion returns null (counted
// nomem) and the record publishes args-only -- the compose side already
// treats that as a fallback.
// [NR-5B-2b] Sizing, corrected after the naruto_680 city exhaustion: the
// retained set spans EVERY address recorded since boot (menus + village +
// city -- the ring used to reclaim those by lapping), and pow2 classes
// wasted up to ~67% per block (a ~153-dw body burned a 256-dw block, the
// full-invalidation class a 4096-dw one). 256MB + quarter-step classes
// (waste <= 25%) with live/bump counters so the real live set is a
// measured number. Committed on touch, so the resident cost is the bump
// watermark, not the reservation.
constexpr uint32_t kStateArenaDwords = 64u << 20;  // 256MB
constexpr uint32_t kStateHdrDwords = 3;
constexpr uint32_t kStateMaxBody = 4096;  // worst captured group ~2.8k dwords
// Quarter-step class sizes: {2^k, 1.25*2^k, 1.5*2^k, 1.75*2^k} per octave.
constexpr uint32_t kStateClassSizes[] = {
    32,   40,   48,   56,   64,   80,   96,   112,  128,  160,  192,
    224,  256,  320,  384,  448,  512,  640,  768,  896,  1024, 1280,
    1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192};
constexpr uint32_t kStateClasses =
    uint32_t(sizeof(kStateClassSizes) / sizeof(kStateClassSizes[0]));
uint32_t g_state_arena[kStateArenaDwords];
uint32_t g_state_bump = 0;                    // virgin space watermark
uint32_t g_state_free[kStateClasses] = {};    // head offset + 1 (0 = empty)
uint32_t g_state_gen = 0;

// Class index for a block that must hold `need` dwords (header included).
// Alloc and free MUST round identically; both call this.
inline uint32_t StateClassFor(uint32_t need) {
  uint32_t c = 0;
  while (kStateClassSizes[c] < need) ++c;
  return c;
}

// [NR-5B-2b] Dormancy horizon and sweep. The clock is g_seq (recordings):
// no OS clock on the raised-IRQL writer, and the horizon self-scales with
// activity -- ~2s at the city's ~150k stores/s, ~a minute on a quiet
// title. A record neither re-stored nor joined for the horizon is expired
// by the writer's incremental sweep (whole record cleared, payload freed):
// the consumer joins every live buffer every frame, so dormancy means the
// address is abandoned (the recorder moved on) or its buffer stopped
// executing -- either way a later miss is a counted fallback, and the
// dormant_rejoin witness measures exactly how often that gamble would
// lose. Sweep pace: kStateSweepPerStore slots per store call ~= the whole
// 1M-slot table per second at city rates.
constexpr uint32_t kStateDormantSeq = 300000;
constexpr uint32_t kStateSweepPerStore = 8;
uint32_t g_state_sweep_cursor = 0;

// Free one block: kill its generation first (readers fail deterministically),
// then thread it onto its class list via blk[0]. The class comes from the
// block's own ndw -- alloc and free run on the one writer thread, so the
// writer reads back exactly the ndw it stamped.
inline void StateFree(uint32_t off) {
  if (off + kStateHdrDwords > kStateArenaDwords) {
    ++g_stats.state_badfree;  // never stamped by us: refuse (a leak if >0)
    return;
  }
  uint32_t* blk = &g_state_arena[off];
  const uint32_t ndw = blk[2];
  if (!ndw || ndw > kStateMaxBody) {
    ++g_stats.state_badfree;
    return;
  }
  blk[1] = 0;  // gen 0: never valid, Acquire/Verify now always refuse
  const uint32_t cls = StateClassFor(kStateHdrDwords + ndw);
  blk[0] = g_state_free[cls];
  g_state_free[cls] = off + 1;
  --g_stats.state_live;
}

// Packet headers are word-aligned; multiplicative hash, top bits.
inline uint32_t Slot(uint32_t addr) {
  return ((addr >> 2) * 2654435761u) >> (32 - kTableBits);
}

}  // namespace

bool LookupDraw(uint32_t phys_addr, DrawRecord* out) {
  const uint32_t addr = phys_addr & kPhysMask;
  if (!addr) return false;
  DrawRecord& slot = g_tab[Slot(addr)];
  if (slot.addr != addr) return false;
  *out = slot;
  // Re-check after the copy: a concurrent displacement of this slot by a
  // different address can otherwise hand back a half-written record under the
  // old key. A same-address upsert racing the copy can still mix old and new
  // ARGUMENTS -- that is the accepted one-sample tear, same as v2.
  if (out->addr != addr) return false;
  // [NR-5B-2b] liveness stamp (reader-side u32 store, torn = one sweep
  // round). A hit on a record already dormant past the horizon is the
  // policy witness: it would have been expired, and 5b-3 must see ~0.
  const uint32_t now = g_seq;
  if (uint32_t(now - out->last_use) > kStateDormantSeq &&
      uint32_t(now - out->seq) > kStateDormantSeq) {
    ++g_stats.dormant_rejoin;
  }
  slot.last_use = now;
  return true;
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

void ResetCacheStats() {
  // state_live and state_bump_dw are GAUGES (current allocator state), not
  // window rates: they survive the per-second reset.
  const uint64_t live = g_stats.state_live;
  const uint64_t bump = g_stats.state_bump_dw;
  g_stats = CacheStats{};
  g_stats.state_live = live;
  g_stats.state_bump_dw = bump;
}

void ResetCache() {
  for (uint32_t i = 0; i < kTableSize; ++i) g_tab[i].addr = 0;
  for (uint32_t i = 0; i < kEpochSlots; ++i) g_gran_epoch[i] = 0;
  g_seq = 0;
  g_state_bump = 0;
  for (uint32_t c = 0; c < kStateClasses; ++c) g_state_free[c] = 0;
  g_state_gen = 0;
  g_state_sweep_cursor = 0;
  g_stats = CacheStats{};  // full reset, gauges included
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

  // [NR-5B-2] payload lifetime == record lifetime: whatever block this slot
  // pointed at before (replace, evict, or an args-only clear) is freed AFTER
  // the new record is published, so a reader that copied the old record can
  // only ever read a deterministic stale, never a recycled body under a
  // live-looking header.
  const uint32_t old_off = slot->state_off;
  const uint32_t old_gen = slot->state_gen;

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
  slot->last_use = g_seq;
  slot->rid = rid;
  slot->prim = prim;
  slot->start = start;
  slot->count = count;
  slot->state_off = state_off;
  slot->state_gen = state_gen;
  slot->vs = vs;
  slot->ps = ps;
  slot->addr = addr;  // key last: the record becomes visible complete
  if (old_gen) {
    StateFree(old_off);
    ++g_stats.state_freed;
  }
  g_gran_epoch[(addr >> kGranuleShift) & (kEpochSlots - 1)] += 1;
  ++g_stats.recorded;
  if (state_gen) ++g_stats.state_stored;
  // [NR-5B-2b] the incremental dormancy sweep rides every store call.
  const uint32_t now = g_seq;
  for (uint32_t k = 0; k < kStateSweepPerStore; ++k) {
    DrawRecord* s = &g_tab[g_state_sweep_cursor];
    g_state_sweep_cursor = (g_state_sweep_cursor + 1) & (kTableSize - 1);
    if (!s->addr) continue;
    if (uint32_t(now - s->last_use) <= kStateDormantSeq ||
        uint32_t(now - s->seq) <= kStateDormantSeq) {
      continue;
    }
    const uint32_t doff = s->state_off;
    const uint32_t dgen = s->state_gen;
    s->addr = 0;  // invalidate first: a racing lookup misses, never mixes
    s->state_gen = 0;
    if (dgen) {
      StateFree(doff);
      ++g_stats.state_freed;
    }
    ++g_stats.state_expired;
  }
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
  // [NR-5B-2] free-list first, virgin bump second. The block is NOT freed
  // here on failure paths: the caller publishes {off, gen} via the record
  // upsert, and the upsert frees the slot's PREVIOUS block -- payload
  // lifetime == record lifetime.
  const uint32_t need = kStateHdrDwords + body_ndw;
  const uint32_t cls = StateClassFor(need);
  const uint32_t block_dw = kStateClassSizes[cls];
  uint32_t off;
  if (g_state_free[cls]) {
    off = g_state_free[cls] - 1;
    g_state_free[cls] = g_state_arena[off];  // pop (next threaded via blk[0])
  } else if (g_state_bump + block_dw <= kStateArenaDwords) {
    off = g_state_bump;
    g_state_bump += block_dw;
    g_stats.state_bump_dw = g_state_bump;
  } else {
    ++g_stats.state_nomem;
    return nullptr;
  }
  ++g_stats.state_live;
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
