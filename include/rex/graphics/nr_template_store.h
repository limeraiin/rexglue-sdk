/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_TEMPLATE_STORE_H_
#define REX_GRAPHICS_NR_TEMPLATE_STORE_H_

#include <cstdint>

#include "rex/graphics/nr_context.h"

// [NR-TMPL] N-2 rung 0: the record-time span-template store and its
// execute-time compare gate. Read naruto-recomp/NEXT-AGENT-5.md "N-1
// CLOSE-OUT" section 6 first.
//
// N-1 proved offline that record-time semantic state reconstructs the walk's
// decode (city: 99.83-99.99% with a frame-granular join; menu: 100.00%).
// N-2's bet is the ONLINE version: a template built by ONE parse of the
// just-written span on the guest producer thread (whose measured 70% idle
// pays for it) reproduces, at execute time, exactly what the PM4 walk decodes
// from that span -- across re-records (the hooks rebuild), by-ref constants
// (re-read at emit), and predicated tiling (resolved against the live bin
// state). This unit is the store plus the non-consuming gate that measures
// that claim; no consumption, no behavior change, default OFF.
//
// The template IS the walker's own memo op stream (nr_context.h, [NR-WM])
// recorded over the span bytes, plus a copy of those bytes: replaying the ops
// over the copy through CtxMemoNext runs the walker's own decode paths, so
// build-side semantics cannot drift from the walk. Differences from the
// 5-4-8 memo (which is execute-time, byte-identity-gated, bin-keyed):
//   - built at RECORD time from the producer's span, keyed by the span's
//     first packet address (the city-proven join key), not by buffer,
//   - bin-AGNOSTIC: the build declines bulk ranges for predicated packets so
//     they stay packet-ops and predication resolves against the bin state in
//     effect at replay (see the CtxMemoNext [NR-TMPL] notes),
//   - values come from the STORED byte copy (fixed at record; a re-record
//     re-builds through the same hook), by-ref ranges from live guest memory
//     at apply, per [[bindings-inline-constants-byref]].

namespace rex {
namespace graphics {
namespace nr {

struct TmplStats {
  // Build side (guest producer thread).
  uint64_t feed = 0;            // spans offered by the record hooks
  uint64_t built = 0;           // templates parsed + published
  uint64_t rebuilt = 0;         // published over an existing key
  uint64_t unchanged = 0;       // byte-identical re-record: publish skipped
  uint64_t feed_reject = 0;     // zero/backwards/oversize span
  uint64_t parse_fail = 0;      // neither cursor convention lands exactly
  uint64_t ops_overflow = 0;    // op stream exceeded the build cap
  uint64_t arena_wraps = 0;     // arena filled: epoch bumped, store restarted
  uint64_t slot_evict = 0;      // probe chain full: a live template displaced
  // Compare side (CP thread).
  uint64_t bufs = 0;            // depth-1 buffer executions compared
  uint64_t bufs_aborted = 0;    // emission log overflow: buffer not compared
  uint64_t spans_hit = 0;       // lookup hit at a packet boundary
  uint64_t spans_eq = 0;        // ...replayed with every emission equal
  uint64_t spans_ne = 0;        // ...replayed with at least one mismatch
  uint64_t spans_stale = 0;     // ...bytes differ from the live buffer
  uint64_t stale_hdr_eq = 0;    // ...stale with the FIRST HEADER intact: the
                                //    layout survived, payload changed -- the
                                //    smell of a re-record the feed missed
  uint64_t stale_hdr_ne = 0;    // ...stale with a different first header: the
                                //    ring recycled this region (dead key)
  uint64_t spans_cross = 0;     // ...span extends past the buffer end
  uint64_t dwords_covered = 0;  // buffer dwords inside hit spans
  uint64_t dwords_gap = 0;      // buffer dwords walked packet-by-packet
  uint64_t gap_pkts = 0;        // packets in gaps (no template starts there)
  uint64_t emi_eq = 0;          // emissions equal (all kinds)
  uint64_t emi_ne = 0;          // emissions unequal
  uint64_t emi_ne_reg = 0;      //   by kind, for the drilldown
  uint64_t emi_ne_range = 0;
  uint64_t emi_ne_shader = 0;
  uint64_t emi_ne_stop = 0;
  uint64_t a_uncovered = 0;     // live emissions in gaps (never compared)
  uint64_t a_extra = 0;         // live emissions a hit span did not produce
  uint64_t b_extra = 0;         // template emissions the live walk lacked
  uint64_t lookup_stale = 0;    // slot epoch/key changed mid-read (race)
  // First mismatch this report window (the gate's deliverable is a NAME).
  uint32_t ne_armed = 0;  // 0 => the next mismatch fills the fields below
  uint32_t ne_kind = 0;
  uint32_t ne_dw = 0;      // absolute dword in the live buffer
  uint32_t ne_key = 0;     // span key (first packet phys address)
  uint32_t ne_a_reg = 0, ne_a_val = 0;  // live side (reg/opcode, value/dword)
  uint32_t ne_b_reg = 0, ne_b_val = 0;  // template side
  // Stale-diff attribution: every differing dword of a stale span is
  // attributed to its covering packet's class (live framing first, stored as
  // the fallback). A span increments each class it touches once. The point:
  // "stale" must decompose into NAMED writer classes (the kick-time
  // finalize-in-place machinery: placeholders flipped to SET_BIN / WAIT /
  // EVENT_WRITE / window packets, and in-place constant patches) -- anything
  // outside those is a surprise the su_* fields name.
  uint64_t stale_cls[16] = {};
  // First stale span this report window (same arming discipline).
  uint32_t st_armed = 0;
  uint32_t st_key = 0;
  uint32_t st_idx = 0;     // first differing dword index within the span
  uint32_t st_stored = 0;  // stored dword at st_idx (host order)
  uint32_t st_live = 0;    // live dword at st_idx (host order)
  // First SURPRISING stale diff (a class outside the finalize/patch set).
  uint32_t su_armed = 0;
  uint32_t su_cls = 0;
  uint32_t su_key = 0;
  uint32_t su_idx = 0;
  uint32_t su_stored = 0;
  uint32_t su_live = 0;
};

// stale_cls indices.
enum TmplStaleCls : uint32_t {
  kTmplScPlaceholder = 0,  // zero / type-2 / lone-fence dwords
  kTmplScWin,              // 0x2080-0x2082 bin window (per-tile)
  kTmplScScissorCopy,      // 0x200E-0x200F, 0x2318-0x2321 (per-tile)
  kTmplScAlu,              // 0x4000-0x47FF constant payloads (patch class)
  kTmplScFetch,            // 0x4800-0x48FF fetch payloads (patch class)
  kTmplScBoolLoop,         // 0x4900+
  kTmplScRegOther,         // any other register target
  kTmplScSetBin,           // SET_BIN mask/select family
  kTmplScWait,             // WAIT_REG_MEM 0x3C
  kTmplScEvent,            // EVENT_WRITE family 0x46/0x5A
  kTmplScDraw,             // a DRAW packet changed (surprise)
  kTmplScOtherOp,          // any other type-3 opcode
  kTmplScUnframed,         // neither side's framing lands exactly
  kTmplScDead,             // >50% of the span's dwords differ: the ring
                           // recycled this region, the key is dead, and
                           // per-dword classes would be meaningless
  kTmplScCount
};

// Compare one executed depth-1 indirect buffer against the store.
// Observation-only: private walker, private contexts, nothing shared with
// the live modes. `raw` = TranslatePhysical(ptr); `mem_read` resolves by-ref
// constant loads (the command processor passes its own CtxMemRead).
void TmplCompareBuffer(const uint8_t* raw, uint32_t ptr, uint32_t count,
                       uint64_t bin_select, uint64_t bin_mask,
                       CtxMemReadFn mem_read, void* mem_user);

TmplStats* TmplStatsPtr();

}  // namespace nr
}  // namespace graphics
}  // namespace rex

// One cvar runs the whole gate: the game's record hooks latch this at startup
// (NarutoApp::OnPostInitLogging) to arm the span feed, and the command
// processor latches it to arm the compare. First call initializes the store
// (arena + build scratch) on the calling (host) thread, before guest threads
// exist -- the producer-thread feed itself never allocates.
extern "C" bool rex_nr_tmpl_active();

// The span feed, called from the game's wrapper hooks (template_capture.cpp)
// on the guest producer thread for every D3D9-block call that moved the
// recorder write pointer: parse [entry_va, end_va) once, publish the
// template. Same thread rules as every record-side hook: no logging, no host
// mutexes, no allocation.
extern "C" void rex_nr_tmpl_span(uint32_t entry_va, uint32_t end_va,
                                 uint8_t* base);

#endif  // REX_GRAPHICS_NR_TEMPLATE_STORE_H_
