/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>  // [NR-ORC] oracle-dump directory creation
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/perf/counter.h>
#include <rex/chrono/clock.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/nr_buffer_cache.h>
#include <rex/graphics/nr_context.h>
#include <rex/graphics/nr_detile.h>
#include <rex/graphics/nr_draw_cache.h>
#include <rex/graphics/nr_state_walk.h>
#include <rex/graphics/nr_draw_registry.h>
#include <rex/graphics/nr_regfile.h>
#include <rex/graphics/nr_resource.h>
#include <rex/graphics/nr_shader_db.h>
#include <rex/graphics/nr_template_store.h>  // [NR-TMPL] N-2 rung 0
#include <rex/hash.h>  // [NR-RUSE] XXH3, header-only (XXH_INLINE_ALL)
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/sampler_info.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory.h>
#include <rex/memory/ring_buffer.h>
#include <rex/stream.h>
#include <rex/system/kernel_state.h>
#include <rex/system/user_module.h>

REXCVAR_DEFINE_BOOL(vsync, true, "GPU", "Enable vertical sync");

REXCVAR_DEFINE_BOOL(clear_memory_page_state, true, "GPU",
                    "Refresh page-valid state from GPU-written memory at frame end. "
                    "Disable for minor CPU overhead reduction, but may break memory coherency.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(occlusion_query_enable, true, "GPU", "Enable host occlusion query handling")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_worker_profile, false, "GPU",
                    "Diagnostic: log a ~1s breakdown of the command-processor "
                    "thread's wall time (starvation-spin vs real ExecutePrimaryBuffer "
                    "work). For perf triage on Release builds (no Tracy). Off by default.");

REXCVAR_DEFINE_BOOL(gpu_split_profile, false, "GPU",
                    "Diagnostic [GPU-SPLIT]: clean (no per-packet timing) ~1s split of "
                    "ExecutePrimaryBuffer time into draw work (IssueDraw) vs the type-0 "
                    "register/constant-write firehose vs other. Decides whether "
                    "parallelizing draws can help heavy-forest fps. Launch-time only "
                    "(read once at worker start, like gpu_worker_profile). Off by default.");

// [N7] The re-baseline instrument -- NATIVE-PATH.md rung N-7. Every per-draw
// and per-thread number this project owns predates de-tile, which cut executed
// draws per frame from ~9.4k to ~3.7k. The standing finding that the guest
// producer is not the wall was measured against a consumer 2.5x heavier than
// today's, so it has to be re-taken before N-8..N-12 are ordered against it.
//
// It reports three things once a second, all cheap enough to leave on:
//  - fps and draws per PRESENTED frame, on ONE line, so a reading buckets
//    itself (single-run heavy-band fps is noise up to 3.4 at mismatched load).
//  - the command-processor thread's own wall clock, split into real
//    ExecutePrimaryBuffer work and starvation spin. This is what
//    gpu_worker_profile measured, minus the per-packet timing that inflated it
//    ~20%: the bracket is two clock reads per worker wake (~50/s at city load).
//    exec% is CAP-INDEPENDENT, which is the point -- the heavy city now sits at
//    the 60 cap, where a 20% consumer win reads as "60".
//  - the ring backlog at each wake. That is the producer/consumer verdict
//    measured at the seam: a deep ring means the guest is running ahead of us
//    and we are the wall; a shallow one means the guest is pacing us.
// Mode 2 additionally cycles the clean draw-vs-register-write split 5 s on /
// 5 s off IN PLACE, so one drive prices both halves at matched draws per frame
// and says whether N-8 (state) or N-9 (draws) is the larger pool.
REXCVAR_DEFINE_INT32(gpu_n7, 1, "GPU",
                     "[n7] Re-baseline report once a second: fps, draws per "
                     "frame, command-processor exec vs spin, ring backlog. "
                     "0 = off, 1 = report (default, ~free), 2 = report + cycle "
                     "CLEAN/SPLIT/SKIP 5 s each in place, 3 = report + cycle "
                     "the N-8b constant census OFF/CENSUS/CENSUS+DEDUPE 5 s "
                     "each in place, 4 = report + cycle the N-8c "
                     "synthetic range split CLEAN/ALUx4/FETCHx4/ALLx4 5 s "
                     "each in place (pixel-correct; the slope is the "
                     "marginal cost of one range), 5 = report + cycle the "
                     "N-8f coalescer CLEAN/COALESCE 5 s each in place, 6 = "
                     "report + cycle the N-9 plan replay CLEAN/PROG 5 s "
                     "each in place (decode-exact; pair adjacent phases).");

// [PM4-CENSUS] Naruto §11.7 draw-emission census. Settles where the ~10k
// heavy-forest draws/frame come from: counts DRAW_INDX vs DRAW_INDX_2 packets,
// split by whether each is inside a PM4_INDIRECT_BUFFER (a pre-recorded /
// replayed command buffer) vs inline in the primary ring, plus indirect-buffer
// count/size per second. If most draws are inside big indirect buffers =>
// pre-recorded replay (intercept at buffer build). If inline in the primary
// ring => a per-frame guest draw loop (find + collapse it). '[pm4-census]',
// ~1s report from the cmd-proc worker. Off by default; launch-time.
REXCVAR_DEFINE_BOOL(gpu_pm4_census, false, "GPU",
                    "Diagnostic [pm4-census]: count DRAW_INDX vs DRAW_INDX_2 packets, split "
                    "by inside-INDIRECT_BUFFER vs primary-ring, + indirect-buffer count/size, "
                    "~1s. Decides pre-recorded-replay vs per-frame draw loop. Off by default.");

// [PM4-IB-DUMP] Naruto §11.9: one-shot hex dump of the first sizable indirect
// buffer, to reverse the per-draw packet structure (the DRAW_INDX 0x22 header
// format, the SET_CONSTANT transform packets, the VB fetch constant) so the
// guest DRAW_INDX recorder / the instancing intercept can be designed. Decode
// offline. '[pm4-ib-dump]'. Off by default.
REXCVAR_DEFINE_BOOL(gpu_pm4_ib_dump, false, "GPU",
                    "Diagnostic [pm4-ib-dump]: one-shot hex dump of the first indirect buffer "
                    ">=512 dwords, to reverse the per-draw PM4 structure. Off by default.");

// [NR-IBL] Native-renderer gate: the replay half of the record/replay
// correlation. Reports every indirect buffer executed in the last second by
// address, with its length, its draw count and how many times it ran. Pair it
// with the game-side 'nr_record_map' cvar ([nrmap]), which reports the address
// ranges the guest RECORDED draws into, then correlate offline with
// tools/correlate-record-replay.py. Off by default.
REXCVAR_DEFINE_BOOL(gpu_ib_ledger, false, "GPU",
                    "Diagnostic [nr-ibl]: per-indirect-buffer execution ledger (address, "
                    "dwords, draws, exec count), ~1s. The replay half of the native-renderer "
                    "record/replay gate. Off by default.");

// [NR-CACHE] Native-renderer build-out increment 1: exact draw-record cache
// validation. The gate (Run 3) proved every executed draw is attributable to a
// recorded one by COUNT; this compares the actual cached records -- recorder
// id, primitive type, start, index count -- per draw and in order against the
// DRAW_INDX packets each executed buffer really contains. If the sequences
// match, the hook layer captures enough to regenerate the buffer's draw stream,
// which is exactly what the renderer will replay. Requires the game-side
// nr_record_map hook; implies gpu_ib_ledger (the scoring lives on its walk).
// Costs an O(dwords) buffer walk per scored execution, so diagnostic-only.
REXCVAR_DEFINE_BOOL(gpu_nr_cache, false, "GPU",
                    "Diagnostic [nr-cache]: per-draw comparison of the native-renderer "
                    "draw-record cache against each executed indirect buffer's DRAW_INDX "
                    "packets. Needs nr_record_map; implies gpu_ib_ledger. Off by default.");

// [NR-CMP] N-9-3d: the record store carries the captured state payload
// (nr_rec_store on the game side writes it at the hook); at execution the
// join walk accumulates each draw group's in-scope type0 spans and diffs the
// record's composed payload against the buffer's actual bytes. The city gate
// for the increment: diverge=0 outside the counted skip/torn classes.
// Perf-neutral by construction (both paths run; naruto_605 rule).
REXCVAR_DEFINE_BOOL(gpu_nr_compose, false, "GPU",
                    "Diagnostic [nr-cmp]: compose each joined draw's state from its "
                    "stored capture payload and diff against the executed buffer's "
                    "type0 packets. Needs game-side nr_rec_store; implies "
                    "gpu_nr_cache. Off by default.");

// [NR-4A] N-9-4a: apply constants FROM THE RECORD at the draw stop. The
// walker defers each group's inline constant-window ranges (ALU/fetch/
// bool-loop) and -- since N-9-4b-ext -- the captured STATE-window ranges
// (Nr4aStateWindow: the 40040 field families that arrive as bulk ranges;
// mirror-crossing runs, the below-0x2000 coher pair, watched registers and
// by-ref loads stay immediate) and, at the draw stop, applies the
// record's composed runs instead when the record is valid AND its run shape
// exactly matches the deferred list; any disagreement falls back to the
// deferred packet ranges, counted by reason. This is the N-9-5 execution
// shape (records supply the values, packets only verify) proven live while
// the packets still exist. Correctness-gated (naruto_605 rule), not fps.
REXCVAR_DEFINE_BOOL(gpu_nr_rec_apply, false, "GPU",
                    "N-9-4a: apply each draw's constant state from its stored "
                    "record at the draw stop (packet ranges deferred, exact "
                    "shape match, fallback counted). Needs game-side "
                    "nr_rec_store. Reports [nr-4a]. Off by default.");

// [NR-4B] N-9-4b: price the value-verify. The verify (memcmp of composed
// bytes vs the deferred packet payloads) is the ONLY remaining read of the
// covered payload on composed groups; N-9-5 deletes it together with the
// packets. This cvar makes that deletion priceable NOW, in place:
//   1  = verify every composed group (the shipping 4a behavior; default)
//   N  = verify 1-in-N composed groups (the rest apply unverified)
//   0  = never verify
//  -1  = CYCLE: 10 s full-verify / 10 s verify-off, phase logged, so one
//        city drive pairs the honest fps/us-per-draw delta at matched dpf
//        ([[price-stages-in-place]]).
// ⚠ MEASUREMENT MODE ONLY away from 1: an unverified composed group can
// apply a one-generation-behind record (the naruto_663 post-hook torn
// window, ~0.2/s at city) and flash for one draw until the next re-record.
// Counters stay honest either way: ver/unver split, fb_val among verified.
REXCVAR_DEFINE_INT32(gpu_nr_rec_verify, 1, "GPU",
                     "N-9-4b: 4a value-verify rate. 1 = every composed group "
                     "(default), N = 1-in-N sampled, 0 = off, -1 = 10 s "
                     "on/off in-place cycle for pricing. Values != 1 are "
                     "measurement-only (torn records may flash).");

// [NR-5C] The N-9-5 fps increment, lever (a): compose reads the record body
// IN PLACE in the retained arena instead of memcpying every payload byte to
// scratch per execution. The retained store makes tears deterministic (free
// stamps gen=0, a re-claimed block re-stamps {addr,gen} first), so the
// in-place walk re-validates with VerifyState after the parse and after the
// applies; a tear restarts the compose once with a forced scratch copy
// (counted). The only copy left is the ~6-dword draw entry, which must
// outlive the compose (the stop issues from it).
REXCVAR_DEFINE_INT32(gpu_nr_arena_apply, 0, "GPU",
                     "N-9-5c lever (a): apply composed groups directly from "
                     "the retained record arena (no scratch memcpy). 0 = "
                     "off, 1 = on.");

// [NR-5C] Lever (b): skip re-applying UNCHANGED record runs. A stable city
// record re-executes ~2.66x/frame x many frames with identical content; if
// its payload generation is unchanged since its last full apply AND no other
// write touched its registers since (the 32-reg bucket epoch map, see
// command_processor.h), the register file already holds exactly these values
// and the apply is pure re-touching. Fetch-constant runs are NEVER skipped
// (the residency/texture invalidations are correctness, not bookkeeping --
// the N8B precedent). Mode 2 also skips the per-dword mirror/state-class
// runs (walker mirror and register file move together through the stamped
// chokepoints, so the same cleanliness predicate covers both).
REXCVAR_DEFINE_INT32(gpu_nr_apply_skip, 0, "GPU",
                     "N-9-5c lever (b): skip clean record applies. 0 = off, "
                     "1 = float/bool inline runs, 2 = + per-dword state "
                     "runs. Fetch always applies.");

// [NR-5C] The one-drive pricing cycler: 10 s per phase, in place --
// phase 0 = both levers off, 1 = arena only, 2 = arena + skip 2. Overrides
// the two cvars above while set. Pair [n7] fps/us-per-draw at matched dpf
// across adjacent phases ([[price-stages-in-place]]).
REXCVAR_DEFINE_BOOL(gpu_nr_apply_cycle, false, "GPU",
                    "N-9-5c: cycle {off, arena, arena+skip} 10 s each in "
                    "place for the paired fps read.");

// [NR-5C-ATTR] The compose bracket, armable ON ITS OWN. naruto_716 priced the
// compose at ~39% of the SUPPRESSED CP second, but the bracket only existed
// while the 5c levers/cycler were armed, so the sup-vs-pkt DELTA (the ~15 fps
// that keeps suppression opt-in) has never been read. With this on, every
// composed 0x22 stop is timed and BINNED BY ITS OWN RECORD (suppressed record
// vs packet-sourced record), so one `--nr_rec_supcover -1` drive attributes
// the delta per phase even when a 1 Hz window straddles a phase flip. The
// drawstop dispatch bracket ([n7] skip, gpu_nr_skip_profile) is binned the
// same way, so compose / drawstop / walk+bulk are all readable per phase.
// [NR-6] N-9-6: THE REPEAT-BAND COMPOSE SKIP. naruto_717 measured that the
// guest emits every draw once per EDRAM tile band (3x256 rows at city), and
// de-tile drops bands 2-3 -- but it drops them at the DISPATCH, which sits
// BELOW the compose. So two thirds of every compose second (240 ms/s of a
// 1000 ms CP second, ~65% of 300-355k composes/s vs 110-134k issued draws/s)
// applies constants to draws that are then thrown away. This drops the
// group's deferred ranges at a repeat-band stop instead of composing or
// flushing them: the work unit disappears rather than moving.
// Soundness rests on N-9-1's census: 288 ALU constants (72 float4s) are
// RE-EMITTED PER DRAW, so a later kept draw carries its own and never reads a
// dropped band's. Modes escalate by class so a visual defect names its own
// cause; the predicate is hoisted above the compose and CROSS-CHECKED against
// the dispatch-site test every stop (`pred_ne` must stay 0).
// DEFAULT 1 since 2026-08-26 (naruto_720 paired + naruto_722 fixed-mode
// visual drive, user-confirmed zero flashing over a whole session): mode 1
// is -13 to -16% CP us/draw at city and modes 2/3 add nothing measurable on
// top of it, so the conservative class is also the fast one. ⚠ Modes 2 and 3
// are NOT exonerated: one of them produced an alternating black-frame strobe
// in the naruto_720 cycle. They buy no fps -- do not re-enable without a
// reason. ⚠ A probe that measures compose COVERAGE (nr-cmp/nrcap gates)
// must set this to 0, or its denominator loses every repeat-band stop.
REXCVAR_DEFINE_INT32(gpu_nr_band_skip, 1, "GPU",
                     "N-9-6: at a repeat-band draw stop, DROP the deferred "
                     "ranges instead of composing them. 0 = off, 1 = ALU + "
                     "bool/loop (DEFAULT, gated), 2 = + fetch, 3 = + state "
                     "windows (2 and 3 buy nothing and one of them flashes).");
// One in-place cycler, 10 s per phase: 0 off / 1 / 2 / 3. The visual gate
// rides it ([[visual-defect-needs-a-visual-bisect]]) -- the phase is logged
// so the eye and the log agree on what was live.
REXCVAR_DEFINE_BOOL(gpu_nr_band_cycle, false, "GPU",
                    "N-9-6: cycle band-skip {off, alu, +fetch, +state} 10 s "
                    "each in place for the paired fps + visual read.");

// [NR-97] N-9-7 phase A: stop APPLYING a repeat band's register stream.
//
// The premise is measured, not assumed: the guest RECORDS the scene once and
// the ring EXECUTES the same buffers once per EDRAM tile band (N-9-1 read
// record:execute = 2.66x, ~195k recorded/s against ~510k executed pkts/s), so
// the 3x is entirely on our side of the fence. De-tile already drops bands 2
// and 3 at the DISPATCH and N-9-6 dropped their deferred compose, but their
// whole register stream is still decoded AND applied -- ~30M dwords/s written
// into a file whose only reader for those registers is a draw we then throw
// away.
//
// ⚠⚠ THE FIRST CUT DROPPED THE VALUES TOO, AND THE CITY REFUTED IT
// (naruto_742: whole-screen flashing with random textures in the ON phase).
// Its premise was "band 1 already wrote these values, so a repeat band's
// applies are re-writes of what is there". FALSE for shader constants: a city
// band run leaves ~160 pixel-shader float constants (0x4099-0x409F,
// 0x4424-0x4432) different from how it found them, which is a fullscreen
// post-pass quad sampling with the wrong constants. The soundness instrument
// had already named that exact class before the drive; the drive only
// confirmed the size.
//
// SO THE LEVER WAS INVERTED, AND N-8b IS WHY IT STILL PAYS. That measurement
// priced a range at ~72 ns of which ~4 ns is the data copy: the other ~95% is
// the per-range dirty-tail evaluation and side effects (cbuffer up_to_date,
// TextureFetchConstantsWritten, InvalidateVertexBufferResidencyRange). So a
// repeat band now KEEPS the value store -- the register file stays
// byte-identical to a full apply, which makes the whole 742 class impossible
// by construction -- and skips only the side effects, because every draw in
// that band is thrown away and nothing else reads a dirty flag until the run
// ends. When it ends, the ranges the run touched are re-applied ONCE through
// the normal path from their CURRENT values, so every side effect fires with
// the right value at the right time. ~150 ranges instead of ~1.5M/s.
//
// The band execution keeps its parse, every delegate, and every write below
// 0x2000 unchanged. Three traps, each already paid for once:
//   naruto_503  a buffer may never be skipped WHOLESALE: its delegates carry
//               the EVENT_WRITE fences the producer spin-waits on (86.3% of
//               city groups end in one), and the guest dies in ~1 s.
//   naruto_504  writes below 0x2000 are the guest's own progress window
//               (scratch / COHER / callback); dead in ~1.5 s without them.
//   naruto_505  never stop updating the register that is your own trigger.
//               GENERALISED here: the keep-list is the whole 27-slot recovery
//               mirror, not just the bin trio, because CtxWriteReg stores a
//               mirrored slot into g_ctx_state whether or not reg_fn runs --
//               dropping the LIVE write alone would make the mirror and the
//               register file disagree and feed the de-tile learner a stale
//               surface signature, which is the same self-latch in new
//               clothes. The mirror is 27 dwords; it costs nothing to keep.
// The PARSE deliberately stays: N-9a already refuted memoizing it at its
// strongest design point ([[subtraction-buckets-mislabel]] -- the parse was
// never the cost). This removes the TRAFFIC, which is the half that measured.
// DEFAULT 1 since 2026-08-26 (naruto_746 + naruto_747, two user city drives on
// different routes, paired in place at matched dpf, 0 fatals): -6.7 to -9.3%
// CP us/draw, and the user could not tell the phases apart in either drive.
// The register file is byte-identical to a full apply by construction, so the
// naruto_742 flashing class cannot recur; lostdraws read 0 in every window.
REXCVAR_DEFINE_INT32(gpu_nr_band_walk, 1, "GPU",
                     "N-9-7: for a buffer execution de-tile calls a repeat "
                     "band, STORE its register values but skip the per-range "
                     "side effects, and re-apply the touched ranges once when "
                     "the band run ends. Parse, delegates and every write "
                     "below 0x2000 run unchanged. 0 = off, 1 = on.");
// One in-place cycler, 10 s per phase ([[price-stages-in-place]]). The census
// runs in BOTH phases, so the OFF phase prices exactly what the ON phase did.
REXCVAR_DEFINE_BOOL(gpu_nr_band_walk_cycle, false, "GPU",
                    "N-9-7: cycle the band walk-skip 10 s off / 10 s on in "
                    "place for the paired fps read at matched draws/frame.");

REXCVAR_DEFINE_BOOL(gpu_nr_compose_prof, false, "GPU",
                    "[nr-5c] Diagnostic: time the record compose per draw "
                    "stop, split suppressed vs packet-sourced. Read beside "
                    "gpu_nr_skip_profile under a supcover cycle. Off by "
                    "default.");

// [NR-5A] N-9-5a: WHOLE-DRAW source coherence at the composed stop -- the
// torn-window precondition for guest-emission suppression. Today's 4a mixes
// sources on a composed draw: constants/state from the record, by-ref 0x2F
// loads and the draw arguments from the packets. That mix is exactly the
// naruto_663 hazard once packets stop existing (a record one re-record
// behind its packets pairs OLD values with NEW args). Coherent mode makes a
// composed draw single-source: the walker defers composable by-ref ranges
// too, the stop applies them FROM the record's sequence markers (re-reading
// guest memory, position preserved), and the draw issues from the record's
// stored draw-packet dwords -- all still verified against the packets with
// counted fallbacks, so behavior is bit-identical while packets exist. When
// they stop existing (N-9-5b), a one-generation-behind record then yields
// the PREVIOUS coherent draw, never a mixed one. Implies gpu_nr_rec_apply.
REXCVAR_DEFINE_BOOL(gpu_nr_rec_coherent, false, "GPU",
                    "N-9-5a: composed draws take by-refs (from markers) and "
                    "draw args (from the record's draw entry) from the record "
                    "too -- single-source, verified, fallback counted. Needs "
                    "game-side nr_rec_store; implies gpu_nr_rec_apply. Off by "
                    "default.");

// [NR-5B-3i] N-9-5b-3: the BY-REF SOURCE under live suppression. F88 (the
// 0x2F emitter) is deliberately NOT shimmed -- markers are parsed FROM the
// emitted group, so suppressing it amputates by-refs entirely (naruto_689).
// So a suppressed group's REAL 0x2F packets are still in the ring, the walker
// still defers them (null be) in stream order, and the stop still flushes
// them from their LIVE address. Mode 0 then walks the record and applies the
// SAME by-refs a second time from the address the RECORD remembers -- and a
// record one recording behind names last frame's constant memory, so the
// stale copy wins the write. That is the identical failure mode the fetch
// half proved (a stale pointer re-read at execute time), in the class that
// paints objects. Mode 1 drops the record's markers (the live packets are
// the only source, applied at the flush). Mode 2 keeps the marker's ORDER
// but takes reg/count/address from the live deferred by-ref it pairs with,
// so the interleave with the record's runs survives without the staleness.
// `mk_addr_ne` prices the hypothesis numerically in EVERY mode.
REXCVAR_DEFINE_INT32(gpu_nr_sup_marker, 0, "GPU",
                     "N-9-5b-3: by-ref source on a SUPPRESSED group. 0 = from "
                     "the record's marker address (stale-capable), 1 = drop "
                     "the markers (live 0x2F packets only), 2 = live address "
                     "at the marker's recorded position, -1 = cycle 0/1/2 5 s "
                     "each in place. Reports '[nr-5b] mk'.");

// [NR-5B-3j] N-9-5b-3: WHAT A FALLBACK MEANS ONCE THE PACKETS ARE GONE.
// Every fallback class in 4a/5a was correct by construction: decline the
// record, flush the packet ranges, render from the packets. Under live
// suppression the packets for the suppressed classes DO NOT EXIST, so a
// declined compose now issues the draw against whatever the register file
// happened to be left holding -- a silent, per-draw wrong-constants defect
// that the counters still report in the reassuring language of the packet
// era. naruto_703 measured the pool at city: miss ~450/s + shape ~450/s +
// foreign ~220/s = ~1100/s against 455k composes, ~22 executions a frame,
// and the game side confirms armed == live (every draw suppressed), so all
// of it lands on draws with no constants. Mode 1 SKIPS the dispatch instead
// of issuing it: if the flashing objects become briefly ABSENT objects at
// the same cadence, the fallback pool is the ALU defect and the fix is
// structural (the arm must be revocable), not a rate reduction -- which is
// why four fallback-reduction drives moved nothing.
REXCVAR_DEFINE_INT32(gpu_nr_sup_fallback, 0, "GPU",
                     "N-9-5b-3: draw whose compose DECLINED while live "
                     "suppression is on. 0 = issue it anyway (today, stale "
                     "registers), 1 = skip the dispatch, -1 = cycle 0/1 5 s "
                     "each in place. Reports '[nr-5b] fb'.");

// [NR-5B-3k] N-9-5b-3: GENERATION, NOT ADDRESS. The city re-records 96-98%
// of its draws IN PLACE every frame. Under live suppression a draw's
// constants exist only in its record, so if the guest replaces that record
// while the consumer is still behind, the consumer composes the NEWER
// recording onto the OLDER draw. Every measurement fits: addr_ne 0 (the
// address is right), torn 0 and fb_val 0 (the read is clean), argsne 0 (the
// 5a draw-entry anchor passes, because an in-place re-record usually keeps
// identical draw bytes -- same prim, same count, same VB -- while the
// constants differ, so the anchor is structurally blind to this class), and
// city-only + intermittent + per-object matches the in-place population
// exactly. This probe stamps the comparison the anchor cannot make: the
// record's own recording seq against the seq the executing buffer STARTED
// at. rec.seq ahead of that means the record belongs to a later frame than
// the draw being replayed. Mode 1 skips those draws, so a flash that turns
// into a brief ABSENCE at the same cadence proves the class by eye the way
// the fetch half was proved.
REXCVAR_DEFINE_INT32(gpu_nr_sup_gen, 0, "GPU",
                     "N-9-5b-3: record-generation probe under suppression. "
                     "0 = count only, 1 = skip draws whose record is AHEAD of "
                     "the executing buffer, -1 = cycle 0/1 5 s each in place. "
                     "Reports '[nr-5b] gen'.");

// [NR-BUF] Increment 2: per-buffer draw-list snapshots (nr_buffer_cache.h).
// Measures the renderer's actual serving model -- admit a snapshot on a clean
// join, serve it while the range's dirty-epoch holds, re-admit after patches
// -- and, probe-mode, VERIFIES every would-be serve against a live join
// (stale serves must be ~0). Implies gpu_nr_cache.
REXCVAR_DEFINE_BOOL(gpu_nr_bufcache, false, "GPU",
                    "Diagnostic [nr-buf]: per-buffer snapshot cache (admit/serve/"
                    "invalidate) with live verification of every served snapshot. "
                    "Needs nr_record_map; implies gpu_nr_cache. Off by default.");

// [NR-STATE] Increment 3: RT/clear/resolve/viewport recovery census on the
// same packet walk (nr_state_walk.h). Classifies every register write and
// type-3 opcode in the executed buffers, flags copy-mode (resolve) draws and
// inherited-state draws, and splits the [nr-cache] misses by mode.
REXCVAR_DEFINE_BOOL(gpu_nr_state, false, "GPU",
                    "Diagnostic [nr-state]: register/opcode census + RT/resolve/"
                    "viewport context recovery over executed indirect buffers. "
                    "Needs nr_record_map; implies gpu_nr_cache. Off by default.");

// [NR-CTX] Increment 4a: the RUNNING state context (nr_context.h). The city
// census (increment 3) proved buffers inherit recovery state from ring order
// (rt_set 19%, mode_inh 40%), so the replay carries a persistent mirror of
// the recovery register set + active shaders across buffers in execution
// order. This probe builds exactly that context, measures per-draw context
// completeness WITH carry, attributes carry vs in-buffer state, censuses the
// live shader working set, and validates the mirror against the live
// register file at every buffer entry (the ground-truth divergence gate).
REXCVAR_DEFINE_BOOL(gpu_nr_ctx, false, "GPU",
                    "Diagnostic [nr-ctx]: carried state context across executed "
                    "indirect buffers, with register-file ground-truth validation. "
                    "Implies gpu_nr_cache. Off by default.");

// [NR-RES] Increment 4b-3: the resource census. 4b-1 closed the input model
// and 4b-2 the shader question; what a draw still needs is its RESOURCES,
// and that half has never been measured. Mirrors the ALU/fetch/bool/loop
// constant files off the same context walk, validates the mirror against the
// live register file at buffer entry, and splits every write by whether it
// arrived inline in the packet or by reference to guest memory.
REXCVAR_DEFINE_BOOL(gpu_nr_res, false, "GPU",
                    "Diagnostic [nr-res]: vertex/texture fetch and constant "
                    "state recovered from the buffer stream, with "
                    "register-file ground truth. Implies gpu_nr_ctx. "
                    "Off by default.");

// [NR-DRAW] Increment 4c: the shadow register file, compared AT EACH DRAW.
//
// Every earlier increment mirrored a hand-picked slice of state and gated it.
// This mirrors the WHOLE register file from the same walk and asks the same
// question over all of it, which does two things no slice can:
//
//   - it removes the choosing. A draw issued from a complete shadow that
//     equals the live file is identical to the emulated draw whatever it
//     reads, so no enumeration of "which registers matter" has to be right.
//   - its failures ARE the port scope. A register the live file has that the
//     stream never wrote must come from the D3D9 hook layer instead, and the
//     named list of those has only ever been reasoned about, never measured.
//
// It also puts the walk in LOCKSTEP with execution (CtxWalkNextDraw), which
// is the only way to read per-draw state honestly: the whole-buffer walk runs
// at buffer entry, so anything asked afterwards sees the buffer's END state.
// That is a real correction, not a refinement -- 4b-3's per-draw coverage
// check ran at execution time and therefore read each fetch slot's LAST type
// in the buffer rather than the type in effect at the draw.
REXCVAR_DEFINE_BOOL(gpu_nr_draw, false, "GPU",
                    "Diagnostic [nr-draw]: full shadow register file walked in "
                    "lockstep with execution and compared against the live "
                    "register file at every draw. Implies gpu_nr_ctx. "
                    "Off by default.");

// [NR-ISSUE] Increment 4d: ISSUE the draw from the shadow instead of merely
// comparing it. 4c closed the input model (diverge=0 over the whole register
// file at full city load); this is the first step that consumes it. At each
// lockstep stop the shadow is composed with the live file (defined registers
// from the walk's own decoded values, everything else -- the four named
// externs, the two side-effect ports, dead registers -- from live, i.e. "read
// it once"), the walk's shader refs are resolved through LoadShader exactly as
// IM_LOAD does, and the backend issues the draw against that private file
// through the real pipeline/texture/render-target caches via the precord
// SetRegisterFile machinery. If the frame is unchanged, the recovered state is
// sufficient END TO END, not just as numbers.
REXCVAR_DEFINE_BOOL(gpu_nr_issue, false, "GPU",
                    "[nr-issue]: issue draws from the increment-4c shadow "
                    "register file (walk-recovered state) instead of the live "
                    "one. Implies gpu_nr_draw. D3D12 only (other backends arm "
                    "but never issue). Off by default.");

// Bisection: a mismatch found with everything shadow-issued is narrowed by
// halving the ordinal range. Ordinals count lockstep geometry stops since
// boot and are printed on the [nr-issue] line.
REXCVAR_DEFINE_INT32(gpu_nr_issue_from, 0, "GPU",
                     "[nr-issue]: first shadow-issue ordinal (lockstep draw "
                     "stops since boot). Draws before it use the live path.");
REXCVAR_DEFINE_INT32(gpu_nr_issue_count, -1, "GPU",
                     "[nr-issue]: number of draws to shadow-issue from "
                     "gpu_nr_issue_from on; -1 = unbounded.");

// [NR-FX] Phase 5-4-0: the first rung of "the walk replaces the executor".
// The walk has maintained every register VALUE since 4c; what the executor
// still owns is WriteRegister's dirty-tracking TAIL (float cbuffer dirty
// gated on the active shader's constant map, bool/loop dirty, fetch: cbuffer
// dirty + texture-cache fetch notification + vertex-residency invalidation).
// Skipping the executor without reproducing that tail would leave every
// rebuilt binding stale-but-plausible, so before anything is skipped the walk
// proves it can DRIVE the subsystems: its decoded write stream fires the same
// tail while the executor still runs. Every effect is idempotent with the
// executor's own firing (the walk is in lockstep, so both fire between the
// same two draws under the same active-shader maps), which makes the gate
// "everything unchanged": pixel-identical, all [nr-*] gates unmoved, and
// rebuild rates NOT inflated -- over-invalidation would show there.
REXCVAR_DEFINE_BOOL(gpu_nr_walk_effects, false, "GPU",
                    "[nr-fx] Phase 5-4-0: fire WriteRegister's dirty-tracking "
                    "side effects from the lockstep walk's decoded write "
                    "stream, in addition to the executor's own (idempotent). "
                    "Implies gpu_nr_draw. Off by default.");

// [NR-PKT] Phase 5-4-1: the non-draw packet census. Before 5-4-2 can skip the
// executor for a depth-1 buffer, every packet class the executor handles that
// is neither a register write nor a draw must be either transcribed into the
// walk or made a per-buffer REFUSE class. This measures which classes actually
// occur, at what rate, in how many depth-1 buffer executions, and whether
// those buffers carry draws (a class confined to draw-less buffers costs the
// skip nothing). Census first; the list is measured, not guessed. Also counts
// type-0/type-1 register writes that hit the STATEFUL ports (scratch writeback
// / COHER_STATUS_HOST RMW / the DC_LUT gamma machine -- the precord
// must-not-defer set): their WriteRegister behavior is not idempotent and not
// walk-reproducible, so a skipped buffer containing one must refuse.
REXCVAR_DEFINE_BOOL(gpu_nr_pkt_census, false, "GPU",
                    "[nr-pkt] Phase 5-4-1: census of every packet class the "
                    "executor dispatches inside depth-1 indirect buffers, "
                    "split stream/non-stream/stateful, with per-class buffer "
                    "and draw-buffer counts. Diagnostic only, off by default.");

// [NR-SKP] Phase 5-4-2: the skip -- the first increment that can MOVE fps.
// For eligible depth-1 indirect buffers the walk becomes the ONLY decoder:
// every packet it understands (the register/constant stream, shader loads,
// bins, no-ops, INVALIDATE_STATE) is applied through the full virtual
// WriteRegister from the walk's decoded write stream, and every packet it
// does not (draws + the 5-4-1 closed delegate list, ~4% of dispatches at
// city load) is dispatched to the executor's own handler at the walk cursor
// via a span reader. Draws therefore still run ExecutePacketType3Draw ->
// IssueDraw with the lockstep arm (shadow file + walk-resolved shaders, the
// proven gpu_nr_issue seam); the executor's per-packet framing for the other
// ~96% of dispatches never runs. Eligibility: D3D12, no precord capture, no
// open trace, no gpu_nr_issue_from/count bisection window.
REXCVAR_DEFINE_BOOL(gpu_nr_skip, true, "GPU",
                    "[nr-skp] Phase 5-4-2: for eligible depth-1 indirect "
                    "buffers, skip the executor's packet loop and run the "
                    "lockstep walk as the only decoder (draws + non-stream "
                    "packets delegated to the executor's own handlers). "
                    "Implies gpu_nr_issue and gpu_nr_walk_effects. ON by "
                    "default.");

// [NR-SPP] 5-4-4 step 0b: prices the skip path's halves. Whole-buffer bracket
// minus the two stop-dispatch brackets = the walk decode + bulk range applies;
// drawstop minus [gpu-split]'s IssueDraw bracket = the per-draw delegation
// round-trip (packet re-parse + Type3Draw framing). ~2 timestamps per stop
// (~770k/s at city), same self-cost class as the gpu_split_profile draw
// bracket it is meant to be read beside.
REXCVAR_DEFINE_BOOL(gpu_nr_skip_profile, false, "GPU",
                    "[nr-spp] Diagnostic: time the skip path (whole buffer / "
                    "draw-stop dispatch / delegated-stop dispatch; remainder = "
                    "walk + bulk applies). Read beside gpu_split_profile. "
                    "Launch-time only. Off by default.");

// [NR-SKP] Phase 5-4-4a: the framing kill. At a draw stop the walk has
// ALREADY applied the packet's own register payload (VGT_DRAW_INITIATOR +
// DMA base/size go through the full virtual WriteRegister before the stop
// returns), so the delegated re-dispatch -- span RingBuffer construction +
// ExecutePacket + re-reading the same dwords + re-writing the same registers
// -- is pure framing (measured 0.58us/draw at city peak, 13.6% of the CP
// second, [nr-spp] naruto_362). With this on, a draw stop derives its args
// straight from the buffer bytes (the same reads ExecutePacketType3Draw
// performs) and calls the extracted tail directly; the lockstep arm +
// gpu_nr_issue seam run unchanged inside it. Any packet shape the direct
// decode does not cover falls back to the delegated dispatch, so every odd
// case keeps the proven handler.
REXCVAR_DEFINE_BOOL(gpu_nr_skip_direct, true, "GPU",
                    "[nr-skp] Phase 5-4-4a: under gpu_nr_skip, issue draw "
                    "stops by direct call from the walk instead of the "
                    "per-draw delegated packet re-parse. Odd packet shapes "
                    "fall back to delegation. Default ON since the 5-4-4a "
                    "city validation (naruto_364, pixel-perfect at 13M "
                    "draws).");

// [NR-PB] N-2-2 item 0: the widened bulk apply. NrWalkRegRange used to accept
// only the three pure constant windows; every multi-register write to the
// state file (0x2000+) fell to the per-dword virtual WriteRegister, which for
// a plain register does nothing beyond the value store (both backend
// overrides add dirty tails ONLY for the constant windows; state registers
// are consumed by draw-time reads of the register file). All stateful ports
// -- scratch writeback 0x578-0x57F (+UMSK/ADDR 0x1DC/0x1DD), COHER 0xA31,
// the DC_LUT machine 0x1922-0x1925 -- sit below 0x2000, so "base >= 0x2000,
// wholly outside the constant windows, inside the register file" is a static
// proof of plainness. City measured 2.84M such writes/s slow-pathed by packet
// shape ([nr-bfc] pdwsf plain=99.5%). Priced 2026-08-20: sound; this cvar is
// the prerequisite the N-2-2 flat apply plan compiles against.
// [NR-PLAN] N-2-2 item 2: the consuming swap. The record-time flat apply plan
// (nr_template_store.cpp / nr_context.h [NR-PLAN]) REPLACES the PM4 parse for
// every span it covers -- one op per packet, zero header decode, values read
// live. Requires gpu_nr_skip (the walk must be the only decoder) plus
// gpu_nr_tmpl + gpu_nr_tmpl_plan (which arm the record-time feed and put the
// store in plan shape). Turning it on DISABLES the [nr-tmpl] compare gate:
// the gate's own cost would poison any fps read, and the two answer different
// questions.
REXCVAR_DEFINE_BOOL(gpu_nr_tmpl_swap, false, "GPU",
                    "[nr-swap] N-2-2 item 2: replay the record-time flat "
                    "apply plan per span instead of parsing the PM4. "
                    "Requires gpu_nr_skip + gpu_nr_tmpl + gpu_nr_tmpl_plan. "
                    "Off by default.");

// [NR-RUSE-BLK] N-2-2 follow-up. The 2026-08-20 menu A/B measured that 53%
// of the `walk+rng` pool is the reuse bookkeeping, not PM4 decode and not the
// register store: every applied register dword reaches NrRuseStaleMark, and
// NrRuseRange loops it per dword with only the `stale_cnt == 0` O(1) escape.
// City applies ~122 range dwords per draw, so that loop runs ~34M times a
// second to discover, almost always, that nothing changed and nothing was
// stale. This makes that discovery O(1) per 64 dwords instead of O(64): a
// window is skipped only when NO dword in it changed (diff bitmap) AND no
// register in the blocks it spans is currently stale (a per-64-register
// stale count), which is exactly the condition under which every one of
// those 64 NrRuseStaleMark calls would have been a no-op.
// [NR-ISSUE-LIVE] N-2-2 follow-up (2026-08-20). Under the skip the replay
// file is a strict mirror of the live register file: every walk write goes
// through the full virtual WriteRegister into register_file_ FIRST and is
// then memcpy'd into g_nri_file (~122 range dwords + 3.2 singles per draw at
// city), and the one-time seed with an unfed shadow is a pure live-file
// copy. Pointing nr_issue_file_ at the live file removes that second copy.
// Gated to the exact configuration where the identity holds by construction:
// skip on (lockstep), verify off (shadow unfed), reuse off (the only other
// g_nri_file reader). Delegate-side register writes bypass the mirror, so
// live is strictly MORE current, never less.
REXCVAR_DEFINE_BOOL(gpu_nr_issue_live, false, "GPU",
                    "[nr-issue] point the draw-issue register source at the "
                    "live register file instead of the mirrored replay file. "
                    "Requires skip + verify off + reuse off. Default off.");

REXCVAR_DEFINE_BOOL(gpu_nr_reuse_blk, false, "GPU",
                    "[nr-ruse] block-skip the per-dword stale feed for "
                    "register runs that neither changed nor hold a stale "
                    "register. Requires the reuse machinery. Off by default.");

// [NR-TILE] N-4: price the EDRAM TILE LOOP. The heavy city renders the whole
// scene THREE TIMES a frame, once per EDRAM tile band: the oracle census
// (tools/n4-tile-census.py, 648 heavy frames) reads 93.8% of executions as
// the same draw packet run x3, each run carrying a different
// PA_SC_WINDOW_OFFSET (0, -256, -512) and window scissor band, with
// per-tile draw counts identical to the packet (2466 / 7 / 542 in every
// band). 62.5% of all draw executions exist only to emulate a 10 MB EDRAM.
// This probe measures what removing them is worth, by skipping work for
// every band but the first (window offset != 0):
//   1 = skip the DRAW dispatch (resolves and the walk still run) -> the
//       ceiling of a tile-pass draw-package replay.
//   2 = skip the whole depth-1 BUFFER (walk included) -> the ceiling of
//       true de-tiling (render once into a full-size target).
// PIXELS ARE WRONG BY DESIGN in bands 2-3; fps and the [nr-tile] counters
// are the measurement. Mode 2 additionally leaves the register file in band
// 1's end state, so treat any visual beyond band 1 as meaningless.
// Mode 3 CYCLES 0 -> 1 -> 2 every 10 s in place, so ONE drive to the heavy
// spot prices all three conditions at matched draws/frame - which is the
// standing rule for every fps read here, and the thing three separate
// drives to "the same spot" can never guarantee.
REXCVAR_DEFINE_INT32(gpu_nr_tile_probe, 0, "GPU",
                     "DEV [nr-tile] N-4: skip repeat EDRAM-tile-band work to "
                     "price de-tiling. 1 = skip their draws, 2 = skip their "
                     "draws AND their device-state applies, 3 = cycle 0/1/2 "
                     "every 10s at one spot. Wrong pixels by design. 0 = off.");


// [NR-TIL] N-4-1: the BUILD the tile probe priced. Every band but the first
// re-executes the same draw packets in the same order with only the tile
// window changed, so a band-2/3 draw's native command span equals the band-1
// span it repeats except for viewport, scissor and the system-constants NDC.
// Record the base band's span per stream position, then serve the repeat
// bands from it: the derivation half of the draw (shader analysis, primitive
// processing, pipeline configure, the whole binding recompose and upload) is
// removed, and the bin-dependent set is recomputed live.
// Pixel-identical BY CONSTRUCTION - unlike gpu_nr_tile_probe, which is the
// same population with the draws simply dropped.
// DEFAULT ON since 2026-08-21: city-gated at 100.0000% over ~1.2M compares at
// two loads, then a full user drive (naruto_518) replayed 8.79M repeat-band
// draws = 97.5% of the band with ambig / absent / nomatch / segcap all zero,
// no wrong pixels and 0 fatals, for +2.4 fps (+6.7%) at matched 9.2k dpf.
REXCVAR_DEFINE_INT32(gpu_nr_tile_replay, 1, "GPU",
                     "[nr-til] N-4-1: serve repeat EDRAM-tile-band draws from "
                     "the base band recorded native span (viewport, scissor "
                     "and NDC recomputed per band). 1 = on, 2 = CYCLE on/off "
                     "every 10s at one spot so a city A/B is matched by "
                     "construction. Requires the skip, verify off and the "
                     "bindings swap. 0 = off.");
// [NR-TILP] N-4-2: its OWN cvar, not gpu_draw_profile. gpu_draw_profile also
// arms [nr-texp], whose four stamps live INSIDE RequestTextures - i.e. inside
// the tex stage this split is trying to price. A stage profile must not pay
// for another profile nested in one of its stages.
REXCVAR_DEFINE_BOOL(gpu_nr_tile_prof, false, "GPU",
                    "DEV [nr-tilp] N-4-2: stage split of the tile replay HEAD "
                    "(pre/tex/rt/vp/sys/res) per replayed draw. Attribution "
                    "only - the stamps are a few percent at city replay "
                    "rates, so never read fps from a run with this on.");
// [NR-TILSYS] N-4-2b piece 1: the system constants of a repeat-band draw are
// byte-identical to its base band recording outside three viewport-derived
// fields -- ndc_scale, ndc_offset and (point lists only)
// point_screen_diameter_to_ndc_radius. The N-4-1 compare gate read that at
// 100.0000% over ~1.2M city compares and ~1.2M menu ones, so the replay can
// restore the recorded snapshot and patch those three live instead of
// re-deriving the whole struct. Priced at 0.106 us/rep at the heavy city.
// DEFAULT ON since 2026-08-21: city-gated at ne=0 / UNDER=0 over 8.92M
// compares at ~420k draws/s (naruto_537), then a matched in-place A/B
// (naruto_538) read the sys stage 0.099 -> 0.044 us/rep.
REXCVAR_DEFINE_INT32(gpu_nr_tile_sys, 1, "GPU",
                     "[nr-tilsys] N-4-2b: serve a tile-replay draw's system "
                     "constants from the base band snapshot (3 viewport "
                     "fields recomputed) instead of re-deriving them. "
                     "1 = on (default), 2 = VERIFY (derive fast into a temp, run the "
                     "real derivation, compare every byte), 3 = CYCLE "
                     "on/off every 10 s in place. Requires "
                     "gpu_nr_tile_replay. 0 = off.");
// [NR-TILTEX] N-4-2b piece 2: the biggest head stage of a replayed draw is
// its texture request (0.31 us/rep at the heavy city, 77% of it the request
// itself and the rest the SRV-key valve). The three tile bands execute the
// same PM4, so they rewrite the same fetch constants and re-derive the same
// bindings - but the binding LIVE at ordinal i of a repeat band is the one
// the base band left at its LAST ordinal, so the derivation cannot just be
// skipped. It is restored from what the base band recorded, once the live
// fetch constants are shown to match.
// DEFAULT ON since 2026-08-21: city-gated at ne=0 with miss=0 over 8.43M
// restores (naruto_537), then a matched in-place A/B (naruto_538) read the
// tex stage 0.299 -> 0.107 us/rep. City carries 2.34 fetch slots per draw
// against menu's 1.00, so the win is BIGGER at the city than at the menu.
REXCVAR_DEFINE_INT32(gpu_nr_tile_tex, 1, "GPU",
                     "[nr-tiltex] N-4-2b: restore a tile-replay draw's "
                     "texture bindings from the base band recording when the "
                     "fetch constants match, instead of re-deriving them. "
                     "1 = on (default), 2 = VERIFY (restore, then force a real "
                     "re-derivation and compare), 3 = CYCLE on/off every "
                     "10 s in place (recording stays on in both halves). "
                     "Requires "
                     "gpu_nr_tile_replay. 0 = off.");
REXCVAR_DEFINE_BOOL(gpu_nr_tmpl_swap_probe, false, "GPU",
                    "DEV [nr-swap] N-2-2: run the swap's store lookup and "
                    "guard pass per span but NOT the replay, so the cost "
                    "half is measured alone against the same baseline. "
                    "Requires gpu_nr_tmpl_swap. Off by default.");

// [NR-DETILE] N-5: the EDRAM tile loop, removed rather than made cheaper.
// The offline census (tools/n5-detile-census.py) proved the repeat bands are
// pure duplicates of band 1 and that the three resolve destinations are one
// contiguous texture, so band 1 rasterising and resolving the full frame
// height makes bands 2 and 3 dead work. Needs the expanded EDRAM
// (xenos::kEdramExpandLog2) - a 720-row 4xMSAA colour+depth pair does not fit
// a 10 MB one. See include/rex/graphics/nr_detile.h.
// [NR-DETILE] N-5 drive-2 bisect. The three band-1 resolve taps are the three
// moments the frame is snapshotted; extending exactly one of them at a time
// says which extension duplicates the overlay, with no theory involved.
REXCVAR_DEFINE_INT32(gpu_nr_detile_tap, -1, "GPU",
                     "[nr-detile] Bisect: -1 extends every band-1 resolve tap "
                     "to the full frame height (normal), 0..7 extends ONLY "
                     "that tap and leaves the others at band height, -2 cycles "
                     "8 s per phase through tap 0 / 1 / 2 / all so one drive "
                     "names the tap. Wrong pixels by design except at -1.");

// [NR-DETILE] N-5 drive-3 bisect. The reader census proved the visible rows
// below the band line come only from tap 2's write of the post-pass input, so
// the duplicated overlay is drawn into EDRAM rows 256+ by one of band 1's
// WIDENED draw segments. Refusing the widening per segment names it.
// [NR-DETILE] N-5 drive-4 experiment. Capture frame 2500 resolve #62 copies
// the fullscreen msaa=0 post RT into the bottom band's region of the
// post-pass input; under de-tiling the EDRAM it reads is no longer band-local.
REXCVAR_DEFINE_INT32(gpu_nr_detile_tail, -1, "GPU",
                     "[nr-detile] Bisect: -1 executes tail rewrites (normal, "
                     "counted and logged), 0 SKIPS the resolves that write a "
                     "repeat band's region of a band-1 tap destination from "
                     "outside the tiled pass, -2 cycles 8 s per phase between "
                     "the two so one drive says whether the tail rewrite "
                     "draws the duplicated overlay.");

// [NR-DETILE] N-6-2. The widen edits PA_SC_WINDOW_SCISSOR only, but the host
// viewport rectangle scissors too. Prints band 1's own viewport intent beside
// the scissor, gated to the tiled pass, once a second.
// [NR-DETILE] N-6-3. The Xenos tiles by PREDICATION: the app records one
// command buffer, each band sets PA_SC_WINDOW_OFFSET and a BIN_MASK, and every
// object is preceded by a SET_BIN_SELECT carrying the bins its bounding box
// touches. A predicated packet runs only where (bin_select & bin_mask) != 0.
// If that is what the guest does here, band 1's draw stream is ALREADY
// restricted to objects that touch band 1, and no amount of scissor widening
// can make the rest appear - which is exactly what the city EDRAM readback
// showed. This counts the predicate outcome, split by band.
REXCVAR_DEFINE_BOOL(gpu_nr_detile_bin_probe, false, "GPU",
                    "[nr-detile] Count predicated packets that pass and fail, "
                    "split by band, once a second");

REXCVAR_DEFINE_BOOL(gpu_nr_detile_vp_probe, false, "GPU",
                    "[nr-detile] Log the guest viewport and scissor of the "
                    "band-1 draws de-tile widens, once a second");

REXCVAR_DEFINE_INT32(gpu_nr_detile_widen_seg, -1, "GPU",
                     "[nr-detile] Bisect: -1 widens every band-1 draw segment "
                     "to the full frame (normal), N widens only segments below "
                     "N (2 = world + refraction widened, overlay left at band "
                     "height), -2 cycles 8 s per phase between ALL and 2 so "
                     "one drive says whether the overlay segment draws the "
                     "duplicate. HUD below the band line missing by design "
                     "except at -1.");

REXCVAR_DEFINE_INT32(gpu_nr_detile, 1, "GPU",
                     "[nr-detile] N-5: render the frame ONCE instead of once "
                     "per EDRAM tile band - band 1 rasterises and resolves the "
                     "full frame height and the repeat bands are not executed. "
                     "Self-arming from the previous frame, with a "
                     "resolve-destination guard that disarms on violation. "
                     "0 = off, 1 = on, 2 = cycle 10 s on / 10 s off IN PLACE "
                     "so one drive reads both halves at matched load. "
                     "DEFAULT ON since the N-6-5 city gate: correct picture "
                     "confirmed by drive, 56-61 fps at 2500-3000 band-1 draws "
                     "per frame against ~43 shipping, and the EDRAM readback "
                     "reports every sampled row r0..r704 changing every "
                     "sample. Kept as a cvar for the A/B and the in-place "
                     "cycle, not as a thing anyone has to set.");

// [nr-pb] Plain-register bulk apply (N-2-2 item 0) is UNCONDITIONAL under
// the skip -- was `gpu_nr_plain_bulk` default ON since the naruto_485/486
// city gates (diverge=0, 68% of per-dword dwords captured, walk pool
// 1.26 -> 1.17 us/draw); cvar removed 2026-08-24 (user directive:
// confirmed wins are code, not flags). Stateful ports and mirror-window
// ranges keep the per-dword path, decided in NrWalkRegRange itself.

// [N8F] Per-draw effect coalescing -- the rung N-8c priced at a ~7% floor
// (naruto_601: 25.4 applies/draw -> 8.7, ~20 ns each). A walk-decoded range
// >= 0x2000 stores its VALUES immediately (plain copy_and_swap into the
// register file and the issue mirror -- no virtual dispatch, no per-range
// dirty evaluation) and its touched span is accumulated; the class side
// effects fire ONCE per merged span at the draw stop, which is when their
// only consumers (cbuffer upload, texture cache, residency) actually read
// them. Register-file readers between ranges always see live values, so
// REG_RMW / COND_WRITE / WAIT_REG_MEM delegates need no flush. The gpu_n7 5
// cycler prices it in place; the cvar is the standing switch.
REXCVAR_DEFINE_BOOL(gpu_nr_coalesce, false, "GPU",
                    "[n8f] Under gpu_nr_skip, defer constant/plain range "
                    "side effects to the draw stop and fire them once per "
                    "merged span (values are stored immediately). Vetoed "
                    "(with a log line) under verify/ruse/memo/tile/instance/"
                    "dedupe modes and on backends without the effects "
                    "override. Pixel-identical by construction.");
REXCVAR_DECLARE(bool, gpu_instance);
REXCVAR_DECLARE(bool, gpu_dedupe_constants);

// [N9] Whole-buffer flat-plan replay. Every skip-eligible depth-1 buffer
// compiles ONCE (its first execution) into the N-2-2 CtxPlanOp stream --
// one op per packet, everything the walk DERIVES precomputed, everything it
// READS still read from the live bytes -- and every later execution replays
// that plan through the proven interpreter instead of re-decoding the PM4.
// Validity is the structural guard set, checked per execution: an in-place
// value patch (the city recorder's constant traffic) passes untouched, a
// re-record fails a guard and recompiles in the same execution. This is the
// N-9 wedge at the naruto_603 decode bucket (~29% of the CP second exists
// only to re-derive unchanged packet structure, ~2.68x per recording).
// Distinct from the three parked shapes by mechanism, not by hope:
// per-BUFFER lookup (not the template store's per-span 138 ns), structural
// validity (not the memo's 17%-coverage byte gate), full flat replay (not
// CtxMemoNext's 2/3 re-parse). The gpu_n7 6 cycler prices it in place.
REXCVAR_DEFINE_BOOL(gpu_nr_prog, false, "GPU",
                    "[n9] Under gpu_nr_skip, replay each depth-1 buffer's "
                    "compiled flat plan instead of re-decoding its PM4 "
                    "(structure memoized per buffer+length, values and "
                    "delegates always live, structural guards checked every "
                    "execution). Vetoed with a log line under "
                    "ruse/bfc/memo/tmpl-swap/tile modes.");

// [NR-WM] Phase 5-4-8: the walk memo. A skip-driven buffer whose bytes are
// identical to its previous compared execution (the ruse shadow compare, on
// anyway under the fast path) replays the walker's recorded emission stream
// -- bulk ranges, per-dword packets, shader loads, bin updates, draw and
// delegate stops -- through the SAME apply callbacks instead of re-parsing
// the PM4. Keyed per (buffer, entry bin state) so predicated tiles stay
// exact; anything per-dword or stateful re-parses its one packet, so the
// semantics cannot drift. Recording free-rides on a walk that happens anyway
// (the economics the 5-4-7-2 span swap lacked). Under gpu_nr_verify the memo
// is not consumed: each eligible execution records fresh and byte-compares
// against the stored stream ([nr-wm] ne must stay 0).
REXCVAR_DEFINE_BOOL(gpu_nr_walk_memo, false, "GPU",
                    "[nr-wm] Phase 5-4-8: replay the walker's recorded "
                    "decode stream on byte-identical buffer replays instead "
                    "of re-parsing the PM4. Requires gpu_nr_skip + the reuse "
                    "machinery. Default off until city-gated.");

// [NR-RUSE] Phase 5-4-5 inc 0: the replay state-reuse pricing probe. Buffers
// are recorded once and replayed many times (once per bin = 3x/frame, and
// across frames); a draw whose full input state is byte-identical to its
// previous execution re-derives and re-uploads everything today. This probe
// prices the reuse ceiling WITHOUT changing behavior: per skip-driven buffer,
// a shadow copy names the first changed dword vs the previous replay
// (same-frame vs cross-frame split); entry-state component shadows over the
// replay file (ctl minus the per-tile bin trio, float/fetch/bool-loop files,
// walker shader refs) attribute inherited-state churn; an order-sensitive
// chain digest of every by-ref LOAD_ALU payload (the only draw input that
// bypasses the buffer bytes) is checked per draw key. A draw classifies
// REUSABLE when its byte prefix, by-ref digest and entry core all match its
// previous execution. Decision gate in NEXT-AGENT-5.md "5-4-5".
REXCVAR_DEFINE_BOOL(gpu_nr_reuse_probe, false, "GPU",
                    "[nr-ruse] Phase 5-4-5 inc 0: price per-draw input-state "
                    "identity across buffer replays (reuse ceiling for the "
                    "native replay's state engine). Requires gpu_nr_skip. "
                    "Diagnostic only, off by default.");

// [NR-RUSE] Phase 5-4-5-2: the fast path's master switch. Turns the reuse
// probe's v2 verdict into the driver of the backend's per-draw state reuse
// (the D3D12 side latches its own eligibility beside the bindings swap).
// Implies gpu_nr_reuse_probe: the verdict machinery is the classification
// source. Was DEFAULT ON 2026-08-13..2026-08-20 (the 5-4-5 pricing: +3-4 fps
// at matched heavy load, pre-item-0 / pre-skip_direct stack).
// ⛔ DEFAULT FLIPPED OFF 2026-08-20 by a paired city A/B (naruto_495 on vs
// naruto_499 off, matched 9.4k draws/frame, user drive, pixel-clean):
// OFF is +8.6 fps (26.95 -> 35.51). The verdict feed costs ~0.72 us/draw
// (122 range dwords/draw through NrRuseStaleMark at ~5.9ns each) and the
// fast path's UpdateBindings skip no longer pays for it under the current
// stack (drawstop measured LOWER with reuse off: 1.894 -> 1.713 us/draw --
// skip_direct + the bindings swap already made per-draw recompose cheap).
REXCVAR_DEFINE_BOOL(gpu_nr_reuse_fast, false, "GPU",
                    "[nr-ruse] Phase 5-4-5-2: reuse a draw's previously "
                    "derived state when the v2 verdict proves its inputs "
                    "unchanged (bindings restored from the bundle instead of "
                    "recomposed). Implies gpu_nr_reuse_probe and requires "
                    "the bindings swap + bundle gate machinery. Default off "
                    "since 2026-08-20 (city A/B: the verdict feed costs more "
                    "than the reuse saves).");

// [NR-BFC] Phase 5-4-6-0: the buffer-level native-replay census. For every
// skip-driven buffer execution, split candidate (the measured bufid rule:
// wholly byte-identical + same-frame + entry-core-eq, byref checked at end)
// vs non-candidate, and price each 5-4-6 design branch: delegate schedule
// composition (refuse classes: REG_RMW / COND_WRITE / nested IB / XE_SWAP /
// WAIT on a non-COHER register), native span composition from the deferred
// list (whitelist violations = barriers/copies/clears/dispatches/queries),
// bin-fixup site counts (vp / sci / sys-CBV), RT-update body runs, written
// register set size, stateful-port writes, submission-boundary crossings
// (zero-copy span validity), by-ref volume. Design + decision gates in
// NEXT-AGENT-5.md "5-4-6".
REXCVAR_DEFINE_BOOL(gpu_nr_bufreplay_census, false, "GPU",
                    "[nr-bfc] Phase 5-4-6-0: census pricing the buffer-level "
                    "native replay (the 60 lever) over the bufid candidate "
                    "set. Requires gpu_nr_skip + the reuse machinery. "
                    "Diagnostic only, off by default.");

// [NR-DSP] Phase 5-4-7-0: the DRAW-level half of the same question, and the
// bigger one -- v2 calls 71-77% of prev'd draws reusable while only 24% sit
// in wholly byte-identical buffers, so a per-draw native span replay serves
// ~3x the population of the buffer-level mechanism. This probe captures each
// draw's emitted native command span and, at its next execution under a
// reusable verdict, compares fresh vs stored: equal, equal-except-the-fields
// a replay would patch (root view addresses / descriptor-table bases), or a
// real difference (named by command class). Design + gate: NEXT-AGENT-5.md
// "THE ENDGAME".
REXCVAR_DEFINE_BOOL(gpu_nr_drawspan_census, false, "GPU",
                    "[nr-dsp] Phase 5-4-7-0: does a reusable draw re-emit a "
                    "byte-identical native command span? Prices per-draw "
                    "span replay. Requires gpu_nr_skip + the reuse "
                    "machinery. Diagnostic only, off by default.");

// [NR-SPR] Phase 5-4-7-1: the production half of the naruto_423 verdict. The
// census compared deduped spans (a command is emitted only when it CHANGES);
// a real replay needs a CONTEXT-FREE recording it can memcpy under any
// dedupe state. This cvar makes every bracketed draw emit its tail state in
// full (pipeline, root signature, all root parameters, topology -- the
// fixed-function block deliberately stays deduped: vp/sci/sys are the
// bin-dependent set the replay keeps LIVE, per the 5-4-6 fixup design),
// stores the first whitelist-clean span per draw key with its patch-site
// offsets, and gates every later reusable execution's fresh emission against
// that FIXED recording. City gate = ne=0 lenne=0. NOT passive: the forced
// re-emits reach the real command list (redundant same-value sets, the
// precord-1a-proven class); do not co-run with the censuses when reading
// them. Consumption (memcpy + patch + skip the derivation) is the next
// increment behind its own cvar.
REXCVAR_DEFINE_BOOL(gpu_nr_span_replay, false, "GPU",
                    "[nr-spr] Phase 5-4-7-1: record context-free per-draw "
                    "native spans and gate the patched replay prediction "
                    "against fresh emission. Compare-only: fresh always "
                    "draws. Requires gpu_nr_skip + the reuse machinery. "
                    "Off by default.");

// [NR-SPW] Phase 5-4-7-2: THE CONSUMING SWAP. A reusable+same-frame draw
// whose key holds a whitelist-clean recording (city-gated ne=0 lenne=0 over
// 3.37M compares, naruto_425) skips the whole derivation half of
// IssueDrawImpl -- shader analysis, primitive processing, ConfigurePipeline,
// UpdateBindings, topology/IB/draw emission -- and instead runs a LIVE HEAD
// (RT-cache update, vertex/index residency, texture requests, barriers,
// fixed-function state, system constants: the bin-dependent + safety-
// critical set), then memcpys the recording into the deferred stream and
// patches the 7 root-CBV GPU addresses from the current member state (the
// bundle restore supplies them, exactly like the 5-4-5 fast path). Any
// precondition failure falls through to the full proven path (everything
// the head did is idempotent there). D3D12 + bindless + bindings-swap +
// reuse-fast + verify-off only; under gpu_nr_verify the cvar degrades to
// the 5-4-7-1 compare gate. ⚠ Under the swap [nr-swp] swapped<draws by
// design (replayed draws never enter UpdateBindings) -- the live gates are
// [nr-spw] rep/fallback accounting + [nr-skp] + pixel identity.
REXCVAR_DEFINE_BOOL(gpu_nr_span_swap, false, "GPU",
                    "[nr-spw] Phase 5-4-7-2: REPLAY recorded per-draw native "
                    "spans instead of re-deriving (memcpy + patch + skip the "
                    "derivation). Implies the 5-4-7-1 record machinery. "
                    "Requires gpu_nr_skip + reuse-fast + the bindings swap; "
                    "verify-off. Off by default.");

// [NR-SPD] Phase 5-4-7-3: the DEDUPED-span revival of 5-4-7-2 (its handoff's
// option (a), rung 1 of the 2026-08-17 EV ranking). The 4x-measured fps loss
// of 5-4-7-2 was the RECORD side: every replay paid a preceding FORCED
// context-free re-emit (+~20 elements emitted + scanned + ~1KB stored) and
// city keys alternate reusable/miss with period ~2, so rep/rec stuck at 1.2.
// This mode records the DEDUPED span from the NORMAL emission instead --
// recording free-rides on work that happens anyway (the 5-4-8 economics, by
// construction) -- and gates replay on EMISSION-CONTEXT match: the entry
// dedupe-member state (pipeline/rootsig/topology/root-up-to-date mask/cbuffer
// addresses+flags/shared-memory flavor) is snapshotted per recording and
// memcmp'd at replay; the recording's captured EXIT context is applied to the
// members after a replay so chains keep matching. Store traffic is gated to
// keys that have proven a reusable execution (the 5-4-8 stability lesson:
// record only the population that will replay). Modifies BOTH modes: with
// gpu_nr_span_replay = the compare gate (fresh vs stored under ctx match,
// city gate = ne=0 lenne=0), with gpu_nr_span_swap = consumption. The store
// is cleared when this mode flips (deduped and context-free recordings are
// not comparable). Census basis: naruto_423 -- deduped spans byte-identical
// under reuse at 99.88%, so the context usually matches; the gate names the
// residual.
REXCVAR_DEFINE_BOOL(gpu_nr_span_dedup, false, "GPU",
                    "[nr-spd] Phase 5-4-7-3: record DEDUPED per-draw spans "
                    "from normal emission (free record) and gate the replay "
                    "on emission-context match. Modifies gpu_nr_span_replay "
                    "(compare) and gpu_nr_span_swap (consume). Off by "
                    "default.");

// [NR-RUSE-EP] The epoch shortcut is REFUTED as a byte-equivalence at city
// (naruto_410 verify run: ep_ne ~45% of clean predictions -- same-frame bin
// repeats DO carry patches whose recorder-hook epoch bump trails the byte
// writes). With the shortcut on, those replays classify against a stale
// shadow and the fast path serves the PREVIOUS execution's state where the
// executor would re-derive from the live (torn) bytes. One city pixel A/B
// (naruto_409) survived that deviation, but it is a deviation. Keep OFF
// unless deliberately trading exactness for the compare cost.
REXCVAR_DEFINE_BOOL(gpu_nr_ruse_epoch, false, "GPU",
                    "[nr-ruse] skip the buffer entry byte compare when the "
                    "granule dirty-epoch sum is unchanged. UNSOUND at city "
                    "(ep_ne ~45%%): trades executor-exact classification "
                    "for compare cost. Off by default.");

// [NR-VERIFY] Phase 5-4-4a inc 2: the per-draw VERIFY work is real CP cost --
// RegShadowSweep (256 reg compares/draw), the per-draw shader re-hash in the
// arm (LoadShader x2 = XXH3 over the whole ucode), the 4b-3 census feeds and
// its per-draw coverage, the d3d12 compare passes (sysconst memcmp, the
// residency predict/compare lists, the 5-1 state-mirror compare) -- and every
// one of those gates has already proven its subsystem city-clean. ON (the
// default) keeps everything exactly as shipped. OFF is the perf config:
// every MIRROR a swap consumes keeps updating (sysconst derivation, the
// texture-descriptor map hooks, the g_nri_file replay state), every
// compare/census pass on the draw path is skipped, and the arm memoizes the
// resolved Shader* per walked IM_LOAD (the executor also resolves at
// IM_LOAD, not per draw). Flipping back ON mid-run re-seeds the drifted
// mirrors (d3d12 latch) so the gates re-arm honestly. The live gates under
// OFF are the swaps' own counters: [nr-skp] direct==draws, [nr-swp]
// swapped==draws fallback=0, [nr-issue] issued==armed.
REXCVAR_DEFINE_BOOL(gpu_nr_verify, false, "GPU",
                    "[nr-*] Phase 5-4-4a inc 2: run the per-draw compare/"
                    "census passes of the settled native-renderer gates. OFF "
                    "(default since the 2026-08-10 city A/B: +10-15 fps at "
                    "matched load, peak 351k draws/s) = perf config, mirrors "
                    "and swaps keep running with compares skipped. Set "
                    "--gpu_nr_verify true for a gate-validation run.");

// [NR-SDB] Increment 4b-2: the shader-database probe. The offline census
// proved the whole 3,320-shader corpus in xeshader.sdb translates, and the
// index keys each blob by the runtime's own shader key -- but "the corpus
// translates" only becomes "the corpus can be translated ahead of time" if
// the shaders the game LOADS are in it under that same key. This measures
// exactly that, and splits every miss into "same shader, different declared
// length" (a key problem) and "not in the corpus at all" (a corpus problem).
// Independent of the ledger: it hooks the shader-load packets, not the walk.
REXCVAR_DEFINE_BOOL(gpu_nr_shaderdb, false, "GPU",
                    "Diagnostic [nr-sdb]: check every shader the game loads "
                    "against the offline shader database, keyed by the "
                    "runtime's own shader hash. Off by default.");

REXCVAR_DEFINE_STRING(gpu_nr_shaderdb_path, "", "GPU",
                      "Shader database for [nr-sdb]. Empty resolves to "
                      "<game_data_root>/xeshader.sdb.");

REXCVAR_DEFINE_STRING(gpu_nr_shaderdb_dump, "", "GPU",
                      "File for [nr-sdb] to write the raw ucode of every "
                      "distinct shader MISSING from the database, so its "
                      "real source can be found offline. Empty disables.");

// [NR-ORC] N-1 template-capture oracle dump. A capture run pairs this with
// the game-side nr_tcap record dump; tools/nr-template-analyze.py joins the
// two offline by packet address and iterates the record->decode translation
// model with no further live runs. A private walker decodes each depth-1
// indirect buffer to the file BEFORE any live mode touches it, so the dump is
// independent of skip/ruse/memo and changes no behavior. Capture-only cost.
// [NR-TMPL] N-2 rung 0: gpu_nr_tmpl is DEFINED in nr_template_store.cpp
// (one flag arms the game-side feed and the compare); declared here for the
// disarm diagnostic only.
REXCVAR_DECLARE(bool, gpu_nr_tmpl);

REXCVAR_DEFINE_BOOL(gpu_nr_oracle_dump, false, "GPU",
                    "DEV [nr-orc]: dump every executed depth-1 indirect "
                    "buffer's decoded stream (register writes, shader loads, "
                    "draw and delegate stops, raw bytes on change) to "
                    "gpu_nr_oracle_path. Pair with the game's nr_tcap. Off by "
                    "default.");

REXCVAR_DEFINE_STRING(gpu_nr_oracle_path, "D:\\nr-capture\\oracle.bin", "GPU",
                      "Output file for the [nr-orc] oracle dump.");

REXCVAR_DEFINE_BOOL(gpu_rb_incremental_readptr, false, "GPU",
                    "Write the ring-buffer read pointer back to the guest incrementally "
                    "as ExecutePrimaryBuffer consumes packets (about every "
                    "read_ptr_update_freq_ quadwords), instead of only once after the whole "
                    "batch. Matches real X360 GPU behavior. The guest render thread's ring "
                    "flow-control is a db16cyc spin-wait built around that incremental "
                    "feedback; without it, a heavy scene stalls the producer ~18ms/frame "
                    "waiting for a full drain (measured: guest sub_8212F708). This turns the "
                    "lock-step drain into a producer/consumer pipeline. Off by default (A/B).")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_STRING(readback_resolve, "none", "GPU",
                      "Controls CPU readback of render-to-texture resolve results.\n"
                      " none: Disable readback (default)\n"
                      " fast: Read previous frame (delayed, copy every frame)\n"
                      " some: Read previous frame (delayed, copy on cache miss)\n"
                      " full: Immediate sync readback (accurate but stalls)")
    .allowed({"none", "fast", "some", "full"})
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_resolve_half_pixel_offset, false, "GPU",
                    "When draw resolution scaling is active, sample from the center of each "
                    "scaled block during resolve readback downscale")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_memexport, true, "GPU",
                    "Enable CPU readback of shader memexport writes for guest memory "
                    "coherency (can reduce correctness issues, but may add GPU/CPU sync cost)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_memexport_fast, true, "GPU",
                    "Use fast double-buffered memexport readback when possible, with "
                    "automatic fallback to full synchronous readback")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(query_occlusion_fake_sample_count, 1000, "GPU",
                     "Fake sample count for occlusion queries")
    .range(1, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(async_shader_compilation, true, "GPU",
                    "Compile shaders and create pipelines asynchronously in background "
                    "threads. This reduces stutter but may cause brief visual artifacts while "
                    "pipelines are being prepared.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics {

using namespace rex::graphics::xenos;

// [NR-TILP] N-4-2 outer stamps: the drawstop bracket (this TU) around the
// d3d12 tile replay (that TU), so one run prices what a repeat draw spends
// OUTSIDE NrTileReplayTry. ostop_t0 is stamped at the direct drawstop
// bracket entry; armed keeps a stale stamp from reaching a draw that came
// in by any other path (delegated fallback, viz-killed predecessor);
// orep/orep_t1 are set at a successful replay's return so the bracket exit
// can close the post segment. CP thread only; extern'd by the d3d12 TU.
bool g_tile_oprof = false;
bool g_tile_ostop_armed = false;
bool g_tile_orep = false;
std::chrono::steady_clock::time_point g_tile_ostop_t0;
std::chrono::steady_clock::time_point g_tile_orep_t1;
uint64_t g_tile_ns_opost = 0;
uint64_t g_tile_n_opost = 0;

namespace {

ReadbackResolveMode ParseReadbackResolveMode(std::string_view value) {
  if (value == "fast") {
    return ReadbackResolveMode::kFast;
  }
  if (value == "some") {
    return ReadbackResolveMode::kSome;
  }
  if (value == "full") {
    return ReadbackResolveMode::kFull;
  }
  return ReadbackResolveMode::kDisabled;
}

}  // namespace

CommandProcessor::CommandProcessor(GraphicsSystem* graphics_system,
                                   system::KernelState* kernel_state)
    : memory_(graphics_system->memory()),
      kernel_state_(kernel_state),
      graphics_system_(graphics_system),
      register_file_(graphics_system_->register_file()),
      trace_writer_(graphics_system->memory()->physical_membase()),
      worker_running_(true),
      write_ptr_index_event_(rex::thread::Event::CreateAutoResetEvent(false)),
      write_ptr_index_(0) {
  assert_not_null(write_ptr_index_event_);
}

CommandProcessor::~CommandProcessor() = default;

bool CommandProcessor::Initialize() {
  // Initialize the gamma ramps to their default (linear) values - taken from
  // what games set when starting with the sRGB (return value 1)
  // VdGetCurrentDisplayGamma.
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t value = i * 0x3FF / 0xFF;
    reg::DC_LUT_30_COLOR& gamma_ramp_entry = gamma_ramp_256_entry_table_[i];
    gamma_ramp_entry.color_10_blue = value;
    gamma_ramp_entry.color_10_green = value;
    gamma_ramp_entry.color_10_red = value;
  }
  for (uint32_t i = 0; i < 128; ++i) {
    reg::DC_LUT_PWL_DATA gamma_ramp_entry = {};
    gamma_ramp_entry.base = (i * 0xFFFF / 0x7F) & ~UINT32_C(0x3F);
    gamma_ramp_entry.delta = i < 0x7F ? 0x200 : 0;
    for (uint32_t j = 0; j < 3; ++j) {
      gamma_ramp_pwl_rgb_[i][j] = gamma_ramp_entry;
    }
  }

  worker_running_ = true;
  worker_thread_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
        WorkerThreadMain();
        return 0;
      }));
  worker_thread_->set_name("GPU Commands");
  worker_thread_->Create();

  return true;
}

void CommandProcessor::Shutdown() {
  EndTracing();

  worker_running_ = false;
  write_ptr_index_event_->Set();
  worker_thread_->Wait(0, 0, 0, nullptr);
  worker_thread_.reset();
}

void CommandProcessor::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                               uint32_t title_id, bool blocking) {}

void CommandProcessor::RequestFrameTrace(const std::filesystem::path& root_path) {
  if (trace_state_ == TraceState::kStreaming) {
    REXGPU_ERROR("Streaming trace; cannot also trace frame.");
    return;
  }
  if (trace_state_ == TraceState::kSingleFrame) {
    REXGPU_ERROR("Frame trace already pending; ignoring.");
    return;
  }
  trace_state_ = TraceState::kSingleFrame;
  trace_frame_path_ = root_path;
}

void CommandProcessor::BeginTracing(const std::filesystem::path& root_path) {
  if (trace_state_ == TraceState::kStreaming) {
    REXGPU_ERROR("Streaming already active; ignoring request.");
    return;
  }
  if (trace_state_ == TraceState::kSingleFrame) {
    REXGPU_ERROR("Frame trace pending; ignoring streaming request.");
    return;
  }
  // Streaming starts on the next primary buffer execute.
  trace_state_ = TraceState::kStreaming;
  trace_stream_path_ = root_path;
}

void CommandProcessor::EndTracing() {
  if (!trace_writer_.is_open()) {
    return;
  }
  assert_true(trace_state_ == TraceState::kStreaming);
  trace_state_ = TraceState::kDisabled;
  trace_writer_.Close();
}

void CommandProcessor::RestoreRegisters(uint32_t first_register, const uint32_t* register_values,
                                        uint32_t register_count, bool execute_callbacks) {
  if (first_register > RegisterFile::kRegisterCount ||
      RegisterFile::kRegisterCount - first_register < register_count) {
    REXGPU_WARN(
        "CommandProcessor::RestoreRegisters out of bounds (0x{:X} registers "
        "starting with 0x{:X}, while a total of 0x{:X} registers are stored)",
        register_count, first_register, RegisterFile::kRegisterCount);
    if (first_register > RegisterFile::kRegisterCount) {
      return;
    }
    register_count =
        std::min(uint32_t(RegisterFile::kRegisterCount) - first_register, register_count);
  }
  if (execute_callbacks) {
    for (uint32_t i = 0; i < register_count; ++i) {
      WriteRegister(first_register + i, register_values[i]);
    }
  } else {
    std::memcpy(register_file_->values + first_register, register_values,
                sizeof(uint32_t) * register_count);
  }
}

void CommandProcessor::RestoreGammaRamp(const reg::DC_LUT_30_COLOR* new_gamma_ramp_256_entry_table,
                                        const reg::DC_LUT_PWL_DATA* new_gamma_ramp_pwl_rgb,
                                        uint32_t new_gamma_ramp_rw_component) {
  std::memcpy(gamma_ramp_256_entry_table_, new_gamma_ramp_256_entry_table,
              sizeof(reg::DC_LUT_30_COLOR) * 256);
  std::memcpy(gamma_ramp_pwl_rgb_, new_gamma_ramp_pwl_rgb, sizeof(reg::DC_LUT_PWL_DATA) * 3 * 128);
  gamma_ramp_rw_component_ = new_gamma_ramp_rw_component;
  OnGammaRamp256EntryTableValueWritten();
  OnGammaRampPWLValueWritten();
}

void CommandProcessor::CallInThread(std::function<void()> fn) {
  if (pending_fns_.empty() && system::XThread::IsInThread(worker_thread_.get())) {
    fn();
  } else {
    pending_fns_.push(std::move(fn));
  }
}

void CommandProcessor::ClearCaches() {}

void CommandProcessor::InvalidateGpuMemory() {}

ReadbackResolveMode CommandProcessor::GetReadbackResolveMode(
    bool legacy_readback_resolve_enabled) const {
  ReadbackResolveMode shared_mode = ParseReadbackResolveMode(REXCVAR_GET(readback_resolve));
  bool shared_mode_overrides_legacy = shared_mode != ReadbackResolveMode::kDisabled ||
                                      rex::cvar::HasNonDefaultValue("readback_resolve");
  if (shared_mode_overrides_legacy) {
    return shared_mode;
  }
  return legacy_readback_resolve_enabled ? ReadbackResolveMode::kFast
                                         : ReadbackResolveMode::kDisabled;
}

bool CommandProcessor::IsReadbackMemexportEnabled(bool legacy_backend_flag) const {
  if (legacy_readback_memexport_cvar_name_ &&
      rex::cvar::HasNonDefaultValue(legacy_readback_memexport_cvar_name_)) {
    return legacy_backend_flag;
  }
  return REXCVAR_GET(readback_memexport);
}

void CommandProcessor::SetDesiredSwapPostEffect(SwapPostEffect swap_post_effect) {
  if (swap_post_effect_desired_ == swap_post_effect) {
    return;
  }
  swap_post_effect_desired_ = swap_post_effect;
  CallInThread([this, swap_post_effect]() { swap_post_effect_actual_ = swap_post_effect; });
}

// [N8B] see the header: the constant-census phase the `gpu_n7 3` cycler drives.
int g_n7_n8b_phase = -1;

// [NR-5C] definitions for the header externs (see command_processor.h):
// the register-write epoch map every backend write path stamps.
bool g_nrb_track = false;
uint32_t g_nrb_apply_seq = 0;
uint32_t g_nrb_reg_epoch[kNrbBuckets] = {};

namespace {
// [GPU-EXEC-PROFILE] Per-category timing, broken out of gpu_worker_profile's total
// ExecutePrimaryBuffer time, to find WHERE the cmd-proc thread's 100% goes:
// register/state writes (type0) vs type3 (draws+constants) vs the IssueDraw host
// call specifically. Only the worker thread runs ExecutePacket/IssueDraw, so
// thread_local is correct + lock-free. Gate set from WorkerThreadMain; ~0 cost off.
bool g_exec_prof = false;
thread_local uint64_t g_exec_type_ns[4] = {0, 0, 0, 0};
thread_local uint64_t g_exec_type_cnt[4] = {0, 0, 0, 0};
thread_local uint64_t g_issuedraw_ns = 0;
thread_local uint64_t g_draw_cnt = 0;
// [GPU-SPLIT] Clean draw-vs-firehose split (gpu_split_profile). Reuses the clean
// g_issuedraw_ns/g_draw_cnt brackets (330k/s, ~1.6% self-inflation) and adds a
// whole-call ExecutePacketType0 timer (the register/constant-write firehose).
// Does NOT enable the per-packet g_exec_prof timing, so it stays ~clean.
bool g_split_prof = false;
thread_local uint64_t g_type0_split_ns = 0;

// [N8C] The splitter and the census. See NrSkipApplyRegRangeBody for the
// apply itself.
//
// N-8b measured the constant path and found the bytes are not the cost:
// 59.5 M applied dwords a second is 238 MB/s of memcpy, ~3 ms of a CPU
// second, and a dedupe that removed a quarter of them moved fps 0.7%. What
// is expensive is that those dwords arrive in 4.14 M RANGES a second - 24.3
// per draw, 14.4 dwords each, ~72 ns apiece of which ~4 ns is the data. So
// the rung is fewer ranges, and two numbers decide it:
//
//   1. what one range COSTS. Measured by SPLITTING: apply each range as k
//      equal sub-ranges instead of one. The register file, the cbuffer
//      invalidations and the fetch invalidations all end identical (the
//      split unions to the same set), so it is correctness-preserving; the
//      only thing that changes is the range count. Three points on a line,
//      and the SLOPE is the marginal cost of a range - exactly the quantity
//      a merge recovers. Splitting the whole body, not just the store, is
//      deliberate: a merge saves everything downstream of the decode.
//   2. how many ranges a merge could actually remove. Measured by the
//      contiguity census below, on the UNSPLIT stream.
//
// Split factor per class, driven by the `gpu_n7 4` in-place cycler.
int g_n8c_split_alu = 1, g_n8c_split_fetch = 1, g_n8c_split_other = 1;

// [N9] Whole-buffer plan replay state. CP thread only.
bool g_n9_eligible = false;  // latched once per worker init, veto logged
bool g_n9_on = false;        // live switch: cvar, or the gpu_n7 6 phase
// Window counters for the [n9] 1Hz line ([[count-refusals-every-granularity]]:
// buffers AND the draws inside them, so a per-buffer refusal cannot hide).
uint64_t g_n9_bufs = 0;        // buffer executions served from a plan
uint64_t g_n9_walk_bufs = 0;   // buffer executions walked live with n9 on
uint64_t g_n9_draws = 0;       // draw stops emitted by the plan interpreter
uint64_t g_n9_walk_draws = 0;  // draw stops from the live walk with n9 on

// [N8F] Coalescer state. Only the command-processor thread touches it (same
// single-thread argument as every other g_* here). Spans are class-tagged so
// a merge can never manufacture a range that straddles a constant-window
// boundary (the effects dispatch classifies by address).
bool g_n8f_eligible = false;  // latched once per worker init, veto logged
bool g_n8f_on = false;        // live switch: cvar, or the gpu_n7 5 phase
struct N8fSpan {
  uint32_t base;  // first register
  uint32_t end;   // exclusive
  uint8_t cls;    // 0 alu, 1 fetch, 2 bool/loop (plain spans are not tracked)
};
constexpr uint32_t kN8fSpans = 24;  // [n8e]: union averages 8.7 spans/draw
N8fSpan g_n8f_spans[kN8fSpans];
uint32_t g_n8f_nspans = 0;
uint64_t g_n8f_rng = 0;       // ranges deferred through the coalescer
uint64_t g_n8f_applies = 0;   // merged spans flushed (the new apply count)
uint64_t g_n8f_flushes = 0;   // flushes that had at least one span
uint64_t g_n8f_ovf = 0;       // span-table overflow: effects fired inline
uint64_t g_n8f_straddle = 0;  // window straddlers sent to the legacy apply

// [N8D] The contiguity census. A range that begins exactly where the previous
// one ended is a range a look-ahead merge could absorb; if it is also
// contiguous in guest memory the merged apply is a single memcpy, and if it
// is not, it is two memcpys but still one dispatch and one side-effect
// evaluation. Both are counted because they are different rungs.
// cls: 0 = alu, 1 = fetch, 2 = everything else.
uint64_t g_n8d_rng[3] = {}, g_n8d_creg[3] = {}, g_n8d_cmem[3] = {};
uint64_t g_n8d_repeat[3] = {};
// Per CLASS, not one global cursor: the walk interleaves classes inside a
// draw, so a single cursor breaks the chain on every alu-state-alu sequence
// and reports "nothing is contiguous" when plenty is.
uint32_t g_n8d_prev_end[3] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
uint32_t g_n8d_prev_base[3] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
const uint32_t* g_n8d_prev_mem_end[3] = {nullptr, nullptr, nullptr};
// Ranges the BODY actually applied, i.e. AFTER any synthetic split. The
// slope wants this against us/draw; g_n8d_rng is the pre-split walk count and
// is deliberately unaffected by the probe.
uint64_t g_n8d_applied = 0;

// [N8E] The census that matters if adjacent-merge is dead. Adjacency asks
// "can I merge this range with the one that arrived just before it", and the
// answer at the intro is 0.0%. The better question does not depend on
// emission order at all: **if every constant range in a draw were
// accumulated and applied ONCE at the draw stop, how many applies would there
// be?** That is what a native renderer does - it does not apply constants
// incrementally, it composes the buffer once per draw - and the ceiling is
// the union of the draw's intervals, not their adjacency.
// Per class, a small set of disjoint [lo,hi) intervals, insert-merged as each
// range arrives and counted at the draw stop. kMax caps the work; an overflow
// is COUNTED, never silently dropped, because a silent cap reads as coverage.
constexpr int kN8eMax = 24;
struct N8eSet {
  uint32_t lo[kN8eMax], hi[kN8eMax];
  int n = 0;
  bool overflow = false;
};
N8eSet g_n8e[3];
// SAMPLED 1 draw in 16. The insert-merge is ~10 comparisons per range and
// there are 17-24 ranges a draw, which unsampled is 1-2% of the command
// processor - a constant offset cancels in the split SLOPE, but it would also
// make this run's absolute us/draw incomparable with naruto_596. A 1/16
// sample costs ~0.1% and the ratio it reports is unbiased.
constexpr uint64_t kN8eSample = 16;
bool g_n8e_active = true;
uint64_t g_n8e_seen = 0;
uint64_t g_n8e_draws = 0;
uint64_t g_n8e_rng[3] = {}, g_n8e_grp[3] = {}, g_n8e_dw[3] = {}, g_n8e_udw[3] = {};
uint64_t g_n8e_overflow = 0;

inline void N8eAdd(int cls, uint32_t base, uint32_t n) {
  if (!g_n8e_active) return;
  N8eSet& s = g_n8e[cls];
  ++g_n8e_rng[cls];
  g_n8e_dw[cls] += n;
  uint32_t lo = base, hi = base + n;
  // absorb every interval this one touches or abuts
  int w = 0;
  for (int i = 0; i < s.n; ++i) {
    if (s.hi[i] < lo || s.lo[i] > hi) {
      s.lo[w] = s.lo[i]; s.hi[w] = s.hi[i]; ++w;
    } else {
      if (s.lo[i] < lo) lo = s.lo[i];
      if (s.hi[i] > hi) hi = s.hi[i];
    }
  }
  s.n = w;
  if (s.n < kN8eMax) {
    s.lo[s.n] = lo; s.hi[s.n] = hi; ++s.n;
  } else {
    s.overflow = true;
  }
}

inline void N8eDrawBoundary() {
  const bool was = g_n8e_active;
  g_n8e_active = (++g_n8e_seen % kN8eSample) == 0;
  if (!was) return;
  ++g_n8e_draws;
  for (int c = 0; c < 3; ++c) {
    N8eSet& s = g_n8e[c];
    g_n8e_grp[c] += uint32_t(s.n);
    for (int i = 0; i < s.n; ++i) g_n8e_udw[c] += s.hi[i] - s.lo[i];
    if (s.overflow) ++g_n8e_overflow;
    s.n = 0;
    s.overflow = false;
  }
}
// A non-sampled draw never accumulated anything, so the sets are already
// empty and the early return above is safe.

inline int N8cClass(uint32_t base) {
  if (base >= 0x4800 && base < 0x4900) return 1;   // fetch constants
  if (base >= 0x4000 && base < 0x4800) return 0;   // ALU float constants
  return 2;
}

// [N8] Where the walk's applied register dwords actually GO. N-7 priced the
// walk plus bulk applies at 30% of the command-processor second, and N-8 was
// written as "hook the 91 D3D9 render-state setters". That premise is only
// worth building if the state registers are actually where the dwords are:
// [nr-skp] reports ~13M applied dwords a second at the city, and if 95% of
// them are shader constants then the whole setter family is a rounding error
// and N-8 has to be about constants instead. A pool is not a cost until it is
// decomposed, so decompose it before designing anything.
//
// Classes are the Xenos register map: below 0x2000 is the guest's own
// GPU-progress block (scratch / COHER / callback), 0x2000-0x3FFF is plain
// state, 0x4000 is the ALU float constant file, 0x4800 the fetch constants,
// 0x4900 the bool/loop constants.
// The per-register histogram is kept for the sub-0x4000 half only: that is the
// part a device-level state mirror could ever replace, and it is a small
// fraction of the dwords, so the per-dword increment is affordable there.
uint64_t g_n8_dw_low = 0, g_n8_dw_state = 0, g_n8_dw_alu = 0;
uint64_t g_n8_dw_fetch = 0, g_n8_dw_boolloop = 0;
uint64_t g_n8_rng = 0, g_n8_pdw = 0;
constexpr uint32_t kN8HistRegs = 0x4000;
uint32_t* g_n8_hist = nullptr;  // lazily allocated, 64 KB

inline void N8Classify(uint32_t base, uint32_t n, bool per_dword) {
  if (per_dword) ++g_n8_pdw; else ++g_n8_rng;
  if (base >= 0x4900) { g_n8_dw_boolloop += n; return; }
  if (base >= 0x4800) { g_n8_dw_fetch += n; return; }
  if (base >= 0x4000) { g_n8_dw_alu += n; return; }
  if (base >= 0x2000) g_n8_dw_state += n; else g_n8_dw_low += n;
  if (!g_n8_hist) {
    g_n8_hist = new uint32_t[kN8HistRegs]();
  }
  const uint32_t end = (base + n < kN8HistRegs) ? base + n : kN8HistRegs;
  for (uint32_t r = base; r < end; ++r) ++g_n8_hist[r];
}

// [N7] Where the 19% outside the skip buffers goes. exec minus the skip
// buffers was the second-biggest bucket of the N-7 drive and it was
// un-itemised. These two brackets split it: every ExecuteIndirectBuffer (so
// non-skip buffers = ib minus the skip path's own buf) and the swap.
uint64_t g_n7_ib_ns = 0, g_n7_ib_calls = 0;
uint64_t g_n7_swap_ns = 0, g_n7_swap_calls = 0;

// [N7] Executed draws, unconditional. One increment at the single IssueDraw
// call site; draws de-tile removed never reach it, so this counts what the
// consumer actually did -- the number every rung below has to move.
uint64_t g_n7_draws = 0;

// [PM4-CENSUS] Only the cmd-proc worker thread executes packets, so plain
// globals (no atomics) -- same pattern as g_issuedraw_ns / g_split_prof. Depth
// is tracked unconditionally (cheap, keeps it correct if the cvar toggles); the
// counters are gated on g_pm4_census.
bool g_pm4_census = false;
uint32_t g_pm4_ib_depth = 0;               // PM4_INDIRECT_BUFFER nesting depth
uint64_t g_pm4_draw_indx_primary = 0;      // DRAW_INDX (0x22) inline in the primary ring
uint64_t g_pm4_draw_indx_indirect = 0;     // DRAW_INDX inside an indirect buffer
uint64_t g_pm4_draw_indx2_primary = 0;     // DRAW_INDX_2 (0x36) inline
uint64_t g_pm4_draw_indx2_indirect = 0;    // DRAW_INDX_2 inside an indirect buffer
uint64_t g_pm4_ib_count = 0;               // indirect buffers executed
uint64_t g_pm4_ib_dwords = 0;              // total dwords across indirect buffers
bool g_pm4_ib_dump = false;                // [PM4-IB-DUMP] one-shot IB hex dump

// [NR-IBL] Native-renderer gate: per-indirect-buffer execution ledger.
//
// The gate asks whether every EXECUTED draw is attributable to exactly one
// RECORDED draw. The guest side ([nrmap], src/record_map.cpp) reports the
// address ranges it recorded draws into; this side reports the address ranges
// the GPU actually executed and how many times each. Correlating the two
// answers whether the record/replay model holds, which decides the whole port.
//
// Keyed on the buffer address, so a buffer replayed N times in a frame shows up
// once with execs=N. Draws are counted by walking the buffer, but only the first
// time an address is seen (the walk is O(dwords), the point is to keep the
// steady-state cost at one hash probe per execution).
bool g_ib_ledger = false;
struct IbLedgerEntry {
  uint32_t ptr;       // buffer address as passed to ExecuteIndirectBuffer
  uint32_t dwords;    // its length
  uint32_t draws;     // DRAW_INDX (0x22) packets -- what the guest hooks see
  uint32_t draws2;    // DRAW_INDX_2 (0x36) packets -- emitted by other code paths
  uint64_t execs;     // times executed since the last report
  uint32_t reg;       // [NR-REG] draws the registry held at the last execution
};
constexpr uint32_t kIbLedgerSize = 1024;  // power of two; ~102 live in the city
IbLedgerEntry g_ib_ledger_tab[kIbLedgerSize] = {};
uint32_t g_ib_ledger_used = 0;
uint64_t g_ib_ledger_evictions = 0;  // distinct buffers that did not fit

// [NR-REG] Per-execution scoring against the draw registry (nr_draw_registry.h,
// which holds the registry itself and explains the design). Every time a buffer
// containing draws is about to run, ask the registry how many draws were
// recorded into that address range and compare it against the count walked out
// of the buffer's own DRAW_INDX packets. Agreement means every executed draw is
// attributable to a recorded one, which is the gate the whole port depends on.
uint64_t g_reg_queries = 0;       // executions of a buffer that contains draws
uint64_t g_reg_r_zero = 0;        // registry had nothing for the range
uint64_t g_reg_r_under = 0;       // held some, but fewer than the buffer's own count
uint64_t g_reg_r_one = 0;         // matched within 5 percent -- the PASS bucket
uint64_t g_reg_r_over = 0;        // held more (a re-record the GPU has not run yet)
uint64_t g_reg_draws_registry = 0;
uint64_t g_reg_draws_ledger = 0;
uint64_t g_reg_gran_hit = 0, g_reg_gran_miss = 0;
uint64_t g_reg_split = 0;         // buffers caught with a half-overwritten recording

// [NR-CACHE] Per-execution JOIN of the buffer's DRAW_INDX packets against the
// address-keyed draw-record cache (nr_draw_cache.h, v3). The consumer walks
// the buffer's packets and looks each 0x22 up by its own header address --
// LookupDraw -- which is exactly the join the real renderer performs. No
// order assumption, no invalidation heuristic (the city recorder patches
// buffers in place, in arbitrary per-draw order); a miss is a packet the
// hook never recorded at that address (torn window, pre-attach, abandoned
// layout, or a direct-map displacement), never another draw's record.
bool g_nr_cache = false;
uint64_t g_nrc_execs = 0;       // scored executions (draw-carrying, depth 1)
uint64_t g_nrc_execs_full = 0;  //   of which every packet hit with args equal
uint64_t g_nrc_pkts = 0;        // DRAW_INDX packets walked
uint64_t g_nrc_hit = 0;         //   joined to a record at their exact address
uint64_t g_nrc_miss = 0;        //   no record at that address
uint64_t g_nrc_arg_eq = 0;      //   hits whose (prim, count) match the packet
uint64_t g_nrc_prim_ne = 0;     //   primitive-type mismatches
uint64_t g_nrc_cnt_ne = 0;      //   index-count mismatches
uint64_t g_nrc_rid0 = 0;        //   rid-0 hits: draws from device state, no args
// First few misses and mismatches each window, logged verbatim at report time
// so a systematic difference (a bad address base, an argument-format skew)
// names itself instead of being a rate. miss=1 => the rec_* fields are absent.
struct NrcSample {
  uint32_t buf, addr, i, pkt_prim, pkt_cnt;
  uint32_t rec_rid, rec_prim, rec_start, rec_cnt;
  uint32_t miss;
};
constexpr uint32_t kNrcSamples = 4;
NrcSample g_nrc_samp[kNrcSamples];
uint32_t g_nrc_samp_n = 0;

// [NR-CMP] N-9-3d compose+diff state. The join walk (below) additionally
// collects, per draw group, the in-scope type0 spans the buffer actually
// carries (a span = one type0 packet: register, count, payload offset in the
// buffer); the join loop then diffs each hit's stored capture payload against
// them, ordered scan-forward exactly like the record-side CapDiff
// (naruto-recomp/src/record_map.cpp) so the two instruments agree on what a
// divergence is. Scope must stay identical to the game-side CapInScope.
// (The per-packet arrays live below, after kNrbMaxPkts is declared.)
bool g_nr_compose = false;
struct CmpSpan {
  uint32_t reg, cnt;
  uint32_t off;  // dword index of the payload in the walked buffer
};
constexpr uint32_t kCmpMaxSpans = 32768;  // city ~2.8k spans/buffer walk
CmpSpan g_cmp_spans[kCmpMaxSpans];
constexpr uint8_t kCmpFlagTainted = 1;  // span scratch overflowed in-group
constexpr uint8_t kCmpFlagFirst = 2;    // first draw in the buffer (open
                                        // framing may pollute its group)
// Window counters. Diff verdicts mirror [nrcap]: run_eq/val_eq vs the
// diverge classes; the skip classes are everything the diff refused, each
// with its own name so coverage stays quotable (a gate over an unquoted
// coverage is vacuous).
uint64_t g_cmp_groups = 0;      // groups diffed
uint64_t g_cmp_pred_runs = 0;   // runs composed from records
uint64_t g_cmp_emit_runs = 0;   // in-scope type0 spans walked in diffed groups
uint64_t g_cmp_run_eq = 0;      //   reg+count matched in order
uint64_t g_cmp_val_eq = 0;      //     and payload bytes equal
uint64_t g_cmp_val_ne = 0;      //     payload differed
uint64_t g_cmp_pred_only = 0;   // composed, never emitted
uint64_t g_cmp_emit_only = 0;   // emitted, never composed
uint64_t g_cmp_no_state = 0;    // hit whose record has no payload (rid 0,
                                //   parse fallback at record, alloc refusal)
uint64_t g_cmp_stale = 0;       // payload lapped by the arena ring
uint64_t g_cmp_torn = 0;        // payload overwritten mid-diff; discarded
uint64_t g_cmp_skip_prevmiss = 0;  // previous packet missed the join (65535-
                                   //   split groups register only their last)
uint64_t g_cmp_skip_ovf = 0;       // group tainted by span-scratch overflow
uint64_t g_cmp_skip_sup = 0;       // [NR-5B-3] suppressed group: no spans to diff
uint64_t g_cmp_outside = 0;        // spans before the body floor: between-body
                                   //   emissions (standalone coher flushes)
// [NR-CMP] N-9-4a ordering census: by-ref LOAD_ALU_CONSTANT (0x2F) packets
// interleave with the captured type0 runs inside a group. N-9-5's suppressed
// bodies force the composed record to be applied AT THE DRAW STOP, which is
// order-safe iff no 0x2F target range ever overlaps the record's runs. This
// measures exactly that: overlap = an 0x2F range intersecting a record run
// in the same group; ilv_overlap = the killer subset where covered spans
// also FOLLOW the 0x2F (true interleave, order actually flips).
struct CmpBref {
  uint32_t reg, cnt;
  uint32_t spans_before;  // span top at insert: covered spans after => interleaved
  uint32_t off;           // [NR-4B-EXT] packet header byte offset (floor filter)
  uint32_t raw[3];        // [NR-4B-EXT] the 0x2F's own dw1..dw3, raw guest bytes
};
constexpr uint32_t kCmpMaxBref = 8192;
CmpBref g_cmp_bref[kCmpMaxBref];
uint64_t g_cmp_bref_pkts = 0;       // 0x2F packets census'd (scored groups)
uint64_t g_cmp_bref_groups = 0;     // diffed groups containing >=1 of them
uint64_t g_cmp_bref_type[6] = {};   // ALU/FETCH/BOOL/LOOP/REGISTERS/other
uint64_t g_cmp_bref_overlap = 0;    // 0x2F range intersects a record run
uint64_t g_cmp_bref_ilv = 0;        //   and covered spans follow it in-group
uint64_t g_cmp_bref_drop = 0;       // census scratch overflowed (completeness)
uint32_t g_cmp_bref_samp[3] = {};   // first overlap: reg, cnt, armed
// [NR-4B-EXT] by-ref MARKER validation: the record body now carries each
// group's 0x2F packets as tagged marker entries in sequence (game-side
// CapStore, from the winning parse). The diff scores the stored markers
// against the group's actually-executed 0x2F packets: same count, same
// dwords ({addr|swap, byte_off, count} -- memcmp raw), same position among
// the record's runs. This is the N-9-5 schema gate for by-refs: a suppressed
// body must re-emit exactly these from the record.
constexpr uint32_t kCmpMaxMk = 24;
uint64_t g_cmp_mk_pred = 0;     // marker entries in scored bodies
uint64_t g_cmp_mk_emit = 0;     // in-floor 0x2F packets in scored groups
uint64_t g_cmp_mk_eq = 0;       // paired markers, dwords equal
uint64_t g_cmp_mk_ne = 0;       // paired markers, dwords differ
uint64_t g_cmp_mk_pos_ne = 0;   // paired equal, but position among runs differs
uint64_t g_cmp_mk_cnt_ne = 0;   // groups where marker count != 0x2F count
uint64_t g_cmp_mk_ovf = 0;      // groups skipped: marker scratch cap
uint32_t g_cmp_mk_samp[5] = {};  // first anomaly: kind(1 val,2 pos,3 cnt),
                                 // pred_pos, emit_pos, dw1; [4]=armed
uint64_t g_cmp_first_div = 0;   // diverging groups that were first-in-buffer
uint64_t g_cmp_div_groups = 0;  // groups with any divergence at all
// First divergence each window, verbatim.
struct CmpSample {
  uint32_t kind;  // 1 pred_only, 2 emit_only, 3 val_ne
  uint32_t reg, pred_cnt, emit_cnt, buf, pkt;
};
CmpSample g_cmp_samp;
bool g_cmp_samp_armed = false;

// Identical to the game-side CapInScope (record_map.cpp): the type0 windows
// the capture predicts. Anything else in a group (below-0x2000 progress
// writes, VGT draw payload regs outside these windows) is out of scope on
// BOTH sides, so it can never read as a divergence.
inline bool CmpInScope(uint32_t reg) {
  return (reg >= 0x4000 && reg < 0x48C0) || reg == 0x5000 ||
         (reg >= 0x2200 && reg < 0x220C) ||  // [dev+16] & 0xFFF
         (reg >= 0x2180 && reg < 0x2185) ||  // [dev+16] bits 12-16
         (reg >= 0x2000 && reg < 0x2010) ||  // [dev+16] bits 42-57 (+0x2007)
         (reg >= 0x2100 && reg < 0x2115) ||  // [dev+16] bits 21-41 (+0x2102)
         (reg >= 0x2280 && reg < 0x2295) ||  // [dev+24] bits 34-54
         (reg >= 0x2300 && reg < 0x2326) ||  // [dev+32] bits 0-37
         (reg >= 0x2380 && reg < 0x2388) ||  // [dev+32] bits 38-45
         (reg >= 0x2388 && reg < 0x23A0) ||  // sub_82140100 payload family
         (reg >= 0x4900 && reg < 0x4928) ||  // bit-56 wholesale bool/loop
         reg == 0xA31 || reg == 0xA2F;       // 40100 coherency preamble
}

void CmpSampleSet(uint32_t kind, uint32_t reg, uint32_t pcnt, uint32_t ecnt,
                  uint32_t buf, uint32_t pkt) {
  if (g_cmp_samp_armed) return;
  g_cmp_samp = {kind, reg, pcnt, ecnt, buf, pkt};
  g_cmp_samp_armed = true;
}

// [NR-BUF] Increment-2 scoring: how executions would be SERVED by the
// per-buffer snapshot cache, and whether any serve would be stale (the gate:
// g_nrb_vne must stay ~0 -- a stale serve means the epoch invalidation rule
// missed a patch, the exact failure class that killed v2). The walk cost is
// unchanged in probe mode -- serving executions are re-walked anyway to
// verify -- so served% here is the fraction of walks the real renderer
// SKIPS, not a probe speedup.
bool g_nr_bufcache = false;
uint64_t g_nrb_execs = 0;    // scored executions (draw-carrying, depth 1)
uint64_t g_nrb_served = 0;   //   valid snapshot present (renderer replays it)
uint64_t g_nrb_vok = 0;      //     served and verified identical
uint64_t g_nrb_vne = 0;      //     served but STALE -- must be ~0
uint64_t g_nrb_absent = 0;   //   no snapshot yet for this address
uint64_t g_nrb_dirty = 0;    //   invalidated: draws re-recorded in range
uint64_t g_nrb_resized = 0;  //   invalidated: buffer length changed
uint64_t g_nrb_draws_served = 0;  // draws inside served executions
uint64_t g_nrb_draws_total = 0;   // draws inside all scored executions
uint64_t g_nrb_walk_ovf = 0;      // walks over the packet scratch bound
// One executed DRAW_INDX as walked, for the join + snapshot layers.
constexpr uint32_t kNrbMaxPkts = 4096;  // city ~350/buffer; order of magnitude
nr::PacketRef g_nrb_pkts[kNrbMaxPkts];
// [NR-CMP] Per collected DRAW_INDX: its group's span range and flags.
uint32_t g_cmp_pkt_lo[kNrbMaxPkts];
uint32_t g_cmp_pkt_hi[kNrbMaxPkts];
uint8_t g_cmp_pkt_flags[kNrbMaxPkts];
// [NR-CMP] N-9-4a: the group's by-ref (0x2F) range list, same lifecycle.
uint32_t g_cmp_pkt_blo[kNrbMaxPkts];
uint32_t g_cmp_pkt_bhi[kNrbMaxPkts];

// [NR-STATE] Increment-3 window counters: the state census, aggregated over
// the executions scored this window, plus the per-draw context splits and the
// join-miss attribution the flags make possible.
bool g_nr_state = false;
uint64_t g_nrs_execs = 0;
uint64_t g_nrs_regs[nr::kRegClassCount] = {};
uint64_t g_nrs_ops[128] = {};
uint64_t g_nrs_t0 = 0, g_nrs_t1 = 0, g_nrs_t2 = 0, g_nrs_t3 = 0;
uint64_t g_nrs_draws = 0;          // 0x22 draws flagged
uint64_t g_nrs_copy = 0;           // draws under kCopy (resolves/clears)
uint64_t g_nrs_mode_inherited = 0; // draws before any in-buffer RB_MODECONTROL
uint64_t g_nrs_rt_set = 0;         // draws with an RT reg written earlier in-buffer
uint64_t g_nrs_vport_set = 0;      // same, viewport
uint64_t g_nrs_desync = 0;         // walks whose flag count != join pkt count
uint64_t g_nrs_mode_mem = 0;       // RB_MODECONTROL covered by a memory load
// Join-miss attribution: what was the mode context of each [nr-cache] miss?
uint64_t g_nrs_miss_copy = 0, g_nrs_miss_inherited = 0, g_nrs_miss_plain = 0;
// Last resolve setup seen, reported once per window.
uint32_t g_nrs_copy_control = 0, g_nrs_copy_dest = 0, g_nrs_copy_dest_info = 0;
// Unclassified registers, merged across the window's walks.
struct NrsOtherReg {
  uint32_t reg;
  uint64_t count;
};
NrsOtherReg g_nrs_other[16] = {};
uint8_t g_nrs_flags[kNrbMaxPkts];

// [NR-CTX] Increment-4a state: the running context itself (CP-thread-only,
// like everything else here) plus the window counters. The context is NOT
// reset per window -- it is the persistent thing being measured.
bool g_nr_ctx = false;
bool g_nr_res = false;   // [NR-RES] increment 4b-3
bool g_nr_draw = false;  // [NR-DRAW] increment 4c
nr::StateContext g_ctx_state;
uint16_t g_ctx_flags[kNrbMaxPkts];
uint64_t g_ctx_execs = 0;   // depth-1 buffers walked into the context
uint64_t g_ctx_depth2 = 0;  // nested executions NOT walked (hole if nonzero)
uint64_t g_ctx_draws = 0;
uint64_t g_ctx_rt_def = 0, g_ctx_vp_def = 0, g_ctx_mode_def = 0,
         g_ctx_sh_def = 0;
uint64_t g_ctx_full = 0;  // all four groups defined at the draw
uint64_t g_ctx_copy = 0;  // draws under EFFECTIVE kCopy (carry included)
uint64_t g_ctx_rt_carry = 0, g_ctx_vp_carry = 0, g_ctx_mode_carry = 0,
         g_ctx_sh_carry = 0;
uint64_t g_ctx_flags_ovf = 0;
uint64_t g_ctx_checks = 0, g_ctx_diverge = 0;
uint64_t g_ctx_mem_loads = 0, g_ctx_poisoned = 0;
uint64_t g_ctx_im_loads = 0, g_ctx_im_imms = 0;
// Divergence attribution: which mirrored registers disagree with the live
// register file at buffer entry, plus verbatim first samples.
struct CtxDivReg {
  uint32_t reg;
  uint64_t count;
};
CtxDivReg g_ctx_div_top[8] = {};
struct CtxDivSample {
  uint32_t reg, ours, live, buf;
};
constexpr uint32_t kCtxDivSamples = 4;
CtxDivSample g_ctx_div_samp[kCtxDivSamples];
uint32_t g_ctx_div_samp_n = 0;
// Distinct shaders this window: open-addressed set keyed on (addr, size,
// immediate). Sizes the corpus a native replay must have translated (the
// offline database caps the honest answer at 3,320 unique).
//
// 4b-1's city run OVERFLOWED this at 4,096 slots (set_ovf 179-480/s, 0 in
// the menu), so its distinct count was an undercount and could only be
// quoted as ">= 2,634". At ~3.3k distinct per second against a 16-probe
// give-up, a 4,096-slot table is already at 80% load. 65,536 slots keeps the
// load under 6% (512 KB, cleared once per window) so the number can be read
// literally.
constexpr uint32_t kCtxShaderSetSize = 65536;  // power of two
uint64_t g_ctx_shader_set[kCtxShaderSetSize];
uint64_t g_ctx_sh_distinct = 0, g_ctx_sh_set_ovf = 0;
// [NR-RING] Increment 4b-0: out-of-stream writes captured into the context
// by the WriteRegister tap, split by where they came from (depth 0 = primary
// ring, depth >= 2 = nested IB), plus which registers. With the tap, the
// walk+observer pair must drive `diverge` to ~0; any residual names a write
// path that bypasses the command processor entirely (e.g. guest MMIO).
uint64_t g_ctx_ext_ring = 0, g_ctx_ext_nested = 0;
CtxDivReg g_ctx_ext_top[8] = {};
// [NR-RING] REG_RMW (0x21) and COND_WRITE (0x45) write registers through
// WriteRegister but are NOT decoded by the context walker -- inside a
// depth-1 buffer they are a walker blind spot (the 4a "outside the stream"
// diverge signature fits them exactly: too rare to crack the increment-3
// top-ops list). The handlers tag their WriteRegister calls so the tap can
// capture and attribute them at EVERY depth.
bool g_nr_in_rmw = false, g_nr_in_cond = false;
uint64_t g_ctx_ext_rmw = 0, g_ctx_ext_cond = 0;

// [NR-WSAMP] Increment 4b-1: the write sampler. 4b-0 eliminated every
// out-of-stream write path (the tap fired zero times while diverge held at
// ~480/s) and its samples showed the context holding IMPOSSIBLE values where
// the live file held plausible ones -- so the walker itself was writing them.
// This pairs a walker-side sampler (which packet produced each write to a
// watched register) with an executor-side twin on the same registers (which
// PM4 opcode produced it downstream): one frame of both, diffed, names the
// misdecoded bytes. The watch set is the 4a divergence signature.
constexpr uint32_t kNrWatchRegs[] = {0x2080, 0x2081, 0x2082, 0x2319};
inline bool NrWatched(uint32_t reg) {
  for (uint32_t r : kNrWatchRegs) {
    if (r == reg) return true;
  }
  return false;
}
struct NrWalkSample {
  uint32_t buf, dw, hdr, arg, reg, value;
};
struct NrExecSample {
  uint32_t pkt, depth, reg, value;
};
constexpr uint32_t kNrSampleMax = 8;
NrWalkSample g_nr_wsamp[kNrSampleMax];
NrExecSample g_nr_esamp[kNrSampleMax];
uint32_t g_nr_wsamp_n = 0, g_nr_esamp_n = 0;
uint64_t g_nr_wsamp_tot = 0, g_nr_esamp_tot = 0;
uint64_t g_ctx_nop0 = 0, g_ctx_sc2 = 0;
// [NR-TILE] Predicated tiling: packets (and draws) the bin check rejected,
// and SET_BIN_* packets found INSIDE buffers. Naruto draws 720p as three
// EDRAM tiles, replaying the same buffer once per bin.
uint64_t g_ctx_pred = 0, g_ctx_pred_draws = 0, g_ctx_bin_pkts = 0;
uint64_t g_ctx_pred_draws_run = 0;
// Packet the executor is currently running, so a sampled write names its
// source: 0x100 | type for types 0-2, 0x200 | opcode for type 3.
uint32_t g_nr_cur_pkt = 0;
uint32_t g_nr_walk_buf = 0;  // buffer being walked, for the walker samples

void NrCtxWatch(void*, uint32_t reg, uint32_t value, uint32_t dw, uint32_t hdr,
                uint32_t arg) {
  if (!NrWatched(reg)) return;
  ++g_nr_wsamp_tot;
  if (g_nr_wsamp_n < kNrSampleMax) {
    g_nr_wsamp[g_nr_wsamp_n++] = {g_nr_walk_buf, dw, hdr, arg, reg, value};
  }
}

void NrExecWatch(uint32_t reg, uint32_t value) {
  ++g_nr_esamp_tot;
  if (g_nr_esamp_n < kNrSampleMax) {
    g_nr_esamp[g_nr_esamp_n++] = {g_nr_cur_pkt, g_pm4_ib_depth, reg, value};
  }
}

// [NR-VERIFY] inc 2: base-side latch of gpu_nr_verify + the arm's shader
// memo (valid only while verify is off; invalidated by every walked
// IM_LOAD/IM_LOAD_IMMEDIATE via CtxShaderSeen below).
bool g_nr_verify_base = true;
bool g_nri_sh_dirty = true;
nr::ShaderRef g_nri_vs_ref = {}, g_nri_ps_ref = {};
Shader* g_nri_vs_cached = nullptr;
Shader* g_nri_ps_cached = nullptr;

void CtxShaderSeen(void*, const nr::ShaderRef& ref) {
  // [NR-VERIFY] inc 2: any walked shader load invalidates the arm's memoized
  // Shader* pair -- the ucode behind an unchanged ref may have been patched,
  // and the re-IM_LOAD is exactly the event that publishes that.
  g_nri_sh_dirty = true;
  uint64_t key = (uint64_t(ref.addr) << 17) ^ (uint64_t(ref.size_dwords) << 1) ^
                 (ref.immediate ? 1u : 0u);
  if (!key) key = 1;
  uint32_t i = uint32_t((key * 2654435761ull) >> 16) & (kCtxShaderSetSize - 1);
  for (uint32_t probe = 0; probe < 16; ++probe) {
    uint64_t& slot = g_ctx_shader_set[(i + probe) & (kCtxShaderSetSize - 1)];
    if (slot == key) return;
    if (!slot) {
      slot = key;
      ++g_ctx_sh_distinct;
      return;
    }
  }
  ++g_ctx_sh_set_ovf;
}

uint32_t CtxMemRead(void* user, uint32_t phys) {
  auto* mem = static_cast<memory::Memory*>(user);
  return __builtin_bswap32(
      *reinterpret_cast<const uint32_t*>(mem->TranslatePhysical(phys)));
}

// ---- [NR-ORC] N-1 template-capture oracle dump ------------------------------
// The consumer half of the N-1 capture (record half: the game's nr_tcap).
// A PRIVATE walker decodes each executed depth-1 indirect buffer into a flat
// binary stream: every register write per-dword (no range_fn, so constants
// arrive as individual writes), the vs/ps refs after each shader load, every
// draw and delegate stop with its dword index (packet address = ptr + dword*4,
// the join key against nr_tcap's located DRAW headers), and the raw buffer
// bytes whenever their hash changes. Nothing here touches the live walker,
// g_ctx_state, or any mode latch: the dump is observation-only and runs
// before them. CP-thread only, plain fwrite on a large stdio buffer -- this
// is a capture-run cost, not a play-mode path.

bool g_nr_orc = false;
// [NR-TMPL] N-2 rung 0: compare latch (store + gate in nr_template_store.cpp).
bool g_nr_tmpl = false;
FILE* g_orc_file = nullptr;
bool g_orc_failed = false;
nr::StateContext g_orc_ctx;  // private mirror; never the live g_ctx_state
uint64_t g_orc_bufs = 0, g_orc_regs = 0, g_orc_draws = 0, g_orc_delegs = 0;
uint64_t g_orc_bytes_dumped = 0, g_orc_nested = 0;

constexpr uint32_t kOrcTagBufBegin = 0xB1000001;
constexpr uint32_t kOrcTagReg = 0xB1000002;
constexpr uint32_t kOrcTagShader = 0xB1000003;
constexpr uint32_t kOrcTagDraw = 0xB1000004;
constexpr uint32_t kOrcTagDeleg = 0xB1000005;
constexpr uint32_t kOrcTagBufEnd = 0xB1000006;
constexpr uint32_t kOrcTagRange = 0xB1000007;
constexpr uint32_t kOrcFileMagic = 0x4E4F4331;  // "NOC1"

// Per-buffer byte hashes: raw bytes ride the dump only when they changed.
constexpr uint32_t kOrcHashSlots = 8192;  // open addressing; ptr==0 empty
struct OrcHashEntry {
  uint32_t ptr;
  uint64_t h;
};
OrcHashEntry g_orc_hash[kOrcHashSlots] = {};

inline void OrcWrite(const void* p, size_t n) { fwrite(p, 1, n, g_orc_file); }

void OrcRegWrite(void*, uint32_t reg, uint32_t value, bool from_memory) {
  const uint32_t rec[4] = {kOrcTagReg, reg, value, from_memory ? 1u : 0u};
  OrcWrite(rec, sizeof(rec));
  ++g_orc_regs;
}

// Bulk ranges are consumed as ranges so the dump keeps their POSITION: for an
// inline range, `phys` is the payload's physical address inside the buffer,
// which is what segments a draw group against nr_tcap's recorder spans. The
// values are normalized to host order (inline payloads arrive big-endian; a
// by-ref range is read through CtxMemRead, which already swaps). REG records
// keep covering whatever the walker does not offer in bulk (mirrored-slot
// ranges, one-reg and truncated shapes), exactly the walker's own split.
bool OrcRegRange(void* user, uint32_t base, const uint32_t* values_be,
                 uint32_t n, uint32_t phys, bool from_memory) {
  uint32_t rec[6] = {kOrcTagRange, base, n, phys, from_memory ? 1u : 0u, 0};
  OrcWrite(rec, sizeof(rec));
  uint32_t chunk[256];
  uint32_t done = 0;
  while (done < n) {
    const uint32_t m = std::min(n - done, 256u);
    for (uint32_t i = 0; i < m; ++i) {
      chunk[i] = values_be
                     ? __builtin_bswap32(values_be[done + i])
                     : CtxMemRead(user, phys + (done + i) * 4);
    }
    OrcWrite(chunk, m * 4);
    done += m;
  }
  g_orc_regs += n;
  return true;
}

void OrcShaderSeen(void*, const nr::ShaderRef&) {
  // ShaderRef does not carry the stage; the walker has already updated the
  // private context, so dump BOTH current refs and let the analyzer diff.
  const nr::ShaderRef& vs = g_orc_ctx.vs;
  const nr::ShaderRef& ps = g_orc_ctx.ps;
  const uint32_t rec[7] = {kOrcTagShader,
                           vs.addr,
                           vs.size_dwords,
                           uint32_t(vs.immediate) | (uint32_t(vs.valid) << 8),
                           ps.addr,
                           ps.size_dwords,
                           uint32_t(ps.immediate) | (uint32_t(ps.valid) << 8)};
  OrcWrite(rec, sizeof(rec));
}

void NrOracleOpen() {
  if (g_orc_file || g_orc_failed) return;
  const std::string path = REXCVAR_GET(gpu_nr_oracle_path);
  std::error_code ec;
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent, ec);
  g_orc_file = fopen(path.c_str(), "wb");
  if (!g_orc_file) {
    g_orc_failed = true;
    REXGPU_INFO("[nr-orc] DISARMED: cannot open {}", path);
    return;
  }
  setvbuf(g_orc_file, nullptr, _IOFBF, 4 << 20);
  const uint32_t hdr[4] = {kOrcFileMagic, 1 /*version*/, 0, 0};
  OrcWrite(hdr, sizeof(hdr));
  REXGPU_INFO("[nr-orc] ON -> {}", path);
}

void NrOracleDumpBuffer(const uint8_t* raw, uint32_t ptr, uint32_t count,
                        uint64_t bin_select, uint64_t bin_mask, uint32_t swap,
                        memory::Memory* mem) {
  if (!g_orc_file) return;
  ++g_orc_bufs;

  // FNV-1a over the buffer dwords: cheap, and only gates raw-byte inclusion.
  uint64_t h = 1469598103934665603ull;
  const size_t nbytes = size_t(count) * 4;
  for (size_t i = 0; i < nbytes; ++i) {
    h = (h ^ raw[i]) * 1099511628211ull;
  }
  bool dump_bytes = true;
  {
    uint32_t s = (ptr >> 2) * 2654435761u % kOrcHashSlots;
    for (uint32_t probe = 0; probe < kOrcHashSlots; ++probe) {
      OrcHashEntry& e = g_orc_hash[(s + probe) % kOrcHashSlots];
      if (e.ptr == ptr) {
        dump_bytes = e.h != h;
        e.h = h;
        break;
      }
      if (!e.ptr) {
        e.ptr = ptr;
        e.h = h;
        break;
      }
    }
  }

  const uint32_t bb[12] = {kOrcTagBufBegin,
                           ptr,
                           count,
                           uint32_t(bin_select),
                           uint32_t(bin_select >> 32),
                           uint32_t(bin_mask),
                           uint32_t(bin_mask >> 32),
                           swap,
                           uint32_t(h),
                           uint32_t(h >> 32),
                           dump_bytes ? 1u : 0u,
                           0};
  OrcWrite(bb, sizeof(bb));
  if (dump_bytes) {
    OrcWrite(raw, nbytes);
    g_orc_bytes_dumped += nbytes;
  }

  static uint16_t orc_flags[4096];
  nr::CtxWalkStats st;
  nr::CtxWalker w;
  nr::CtxWalkBegin(&w, raw, count, ptr, &g_orc_ctx, orc_flags, 4096, &st,
                   CtxMemRead, mem, OrcShaderSeen, nullptr, nullptr, nullptr,
                   bin_select, bin_mask, OrcRegWrite, nullptr, nullptr,
                   nullptr);
  // Ranges as ranges: keeps the payload position in the dump (see OrcRegRange).
  w.range_fn = OrcRegRange;
  w.range_user = mem;
  nr::CtxDrawStop stop;
  while (nr::CtxWalkNextStop(&w, &stop)) {
    if (stop.delegate) {
      const uint32_t rec[3] = {kOrcTagDeleg, stop.opcode, stop.dword};
      OrcWrite(rec, sizeof(rec));
      ++g_orc_delegs;
      // A nested indirect buffer is recorded as a stop but NOT entered: its
      // content is outside this buffer's bytes and outside the N-1 class.
      if ((stop.opcode & 0x7F) == 0x3F) ++g_orc_nested;
      nr::CtxWalkSkipDelegated(&w);
    } else {
      const uint32_t rec[5] = {kOrcTagDraw, stop.opcode, stop.dword, stop.flags,
                               stop.index};
      OrcWrite(rec, sizeof(rec));
      ++g_orc_draws;
    }
  }
  const uint32_t be[2] = {kOrcTagBufEnd, 0};
  OrcWrite(be, sizeof(be));

  if ((g_orc_bufs & 0xFFF) == 0) {
    REXGPU_INFO(
        "[nr-orc] bufs={} draws={} delegs={} regs={} raw={}MB nested={}",
        g_orc_bufs, g_orc_draws, g_orc_delegs, g_orc_regs,
        g_orc_bytes_dumped >> 20, g_orc_nested);
  }
}

// [NR-RING] Called from CommandProcessor::WriteRegister (CP thread, same as
// every other g_ctx_* touch) for writes landing in the 0x2xxx probe window
// at depth != 1, PLUS depth-1 writes issued by the tagged blind-spot ops
// (REG_RMW / COND_WRITE, which the walker does not decode). Plain depth-1
// writes stay excluded: the entry walk already applied them ahead of
// execution, and re-applying here would mask a walker decode bug the
// diverge check exists to catch.
void NrCtxExternalWrite(uint32_t reg, uint32_t value) {
  if (nr::CtxApplyExternalWrite(&g_ctx_state, reg, value) < 0) return;
  if (g_nr_in_rmw) {
    ++g_ctx_ext_rmw;
  } else if (g_nr_in_cond) {
    ++g_ctx_ext_cond;
  } else if (g_pm4_ib_depth) {
    ++g_ctx_ext_nested;
  } else {
    ++g_ctx_ext_ring;
  }
  for (auto& d : g_ctx_ext_top) {
    if (d.count && d.reg == reg) {
      ++d.count;
      break;
    }
    if (!d.count) {
      d.reg = reg;
      d.count = 1;
      break;
    }
  }
}

// Walk a PM4 buffer and count its draw packets, split by opcode: DRAW_INDX
// (0x22) is what the guest D3D9 recorder emits and therefore what the registry
// can possibly know about, while DRAW_INDX_2 (0x36) comes from paths that are
// not hooked. Keeping them apart stops the unhooked draw type from reading as a
// failed gate. Guest memory is big-endian.
void CountBufferDraws(const uint8_t* raw, uint32_t dwords, uint32_t* out_draw_indx,
                      uint32_t* out_draw_indx2, uint64_t bin_select,
                      uint64_t bin_mask) {
  uint32_t d1 = 0, d2 = 0;
  // [NR-TILE] Count what this pass EXECUTES: predicated packets the bin check
  // rejects are skipped, exactly as the command processor skips them.
  nr::CtxBinState bin{bin_select, bin_mask};
  for (uint32_t j = 0; j < dwords;) {
    const uint32_t hdr = __builtin_bswap32(*(const uint32_t*)(raw + j * 4));
    if (!hdr) {  // one-dword no-op, per ExecutePacket; NOT a type-0 packet
      ++j;
      continue;
    }
    const uint32_t ty = hdr >> 30, cnt = ((hdr >> 16) & 0x3FFF) + 1;
    if (ty == 3) {
      const uint32_t op = (hdr >> 8) & 0x7F;
      if (nr::CtxPredicatedOut(bin, hdr)) {
        j += 1 + cnt;
        continue;
      }
      nr::CtxApplyBinPacket(
          &bin, op,
          (j + 1 < dwords) ? __builtin_bswap32(*(const uint32_t*)(raw + (j + 1) * 4)) : 0,
          (j + 2 < dwords) ? __builtin_bswap32(*(const uint32_t*)(raw + (j + 2) * 4)) : 0);
      if (op == 0x22) ++d1;
      else if (op == 0x36) ++d2;
      j += 1 + cnt;
    } else if (ty == 0) {
      j += 1 + cnt;
    } else {
      ++j;
    }
  }
  *out_draw_indx = d1;
  *out_draw_indx2 = d2;
}

// [NR-DRAW] Increment 4c: the shadow register file and the lockstep walk.
//
// The walker is a single global because it is only ever driven at IB depth 1
// on the command-processor thread, exactly like every other g_ctx_* here. A
// nested execution does not touch it (and is counted, not absorbed).
static_assert(nr::kRegShadowCount == RegisterFile::kRegisterCount,
              "the shadow must be the same shape as the register file it "
              "mirrors -- increment 4c issues a draw by copying one into the "
              "other");
nr::RegShadow g_reg_shadow;
nr::RegShadowStats g_reg_stats;
nr::CtxWalker g_ctx_walker;
nr::CtxWalkStats g_ctx_walk_stats;
// A walk is in progress and must be finished at the end of the buffer.
bool g_ctx_walk_active = false;
// ... and is still in step with the executor, so a stop can be trusted. These
// are separate on purpose: a desync must stop the COMPARING without
// abandoning the walk, or the running context would be left half-applied and
// the next buffer would start from a state no replay would ever have.
bool g_ctx_walk_lockstep = false;
// Lockstep integrity. A stop whose opcode is not the one the executor is
// running, or a draw the walk has already run out of, means the two decoders
// disagree about which packets execute -- which would make every value read
// at the next stop come from the wrong moment. Counted, and comparing stops
// for the rest of that buffer rather than reporting from a desynced walk.
uint64_t g_nrd_stops = 0, g_nrd_desync = 0, g_nrd_skipped_depth = 0;
uint64_t g_nrd_buffers = 0;
// Findings, tallied by register: the deliverable is a NAMED list, not a rate.
struct NrdReg {
  uint32_t reg;
  uint64_t count;
  uint32_t last_ours, last_live;
};
constexpr uint32_t kNrdTop = 16;
NrdReg g_nrd_div_top[kNrdTop] = {};
NrdReg g_nrd_ext_top[kNrdTop] = {};

void NrdTally(NrdReg* top, uint32_t reg, uint32_t ours, uint32_t live) {
  for (uint32_t i = 0; i < kNrdTop; ++i) {
    if (top[i].count && top[i].reg == reg) {
      ++top[i].count;
      top[i].last_ours = ours;
      top[i].last_live = live;
      return;
    }
    if (!top[i].count) {
      top[i] = {reg, 1, ours, live};
      return;
    }
  }
}

// ★ The THIRD category, found by the first menu run and confirmed against the
// executor's own code rather than assumed. These registers are not state the
// stream carries; they are ports the EXECUTOR itself writes as a side effect
// of executing, so the shadow holding the packet's value while the live file
// holds something else is correct behaviour on both sides:
//
//   COHER_STATUS_HOST  WriteRegister ORs in bit 31 on every write, and
//                      MakeCoherent clears the status once it has acted. Live
//                      is a CONSUMED value: a coherency handshake, not a
//                      value a draw reads. (command_processor.cpp, the
//                      XE_GPU_REG_COHER_STATUS_HOST case.)
//   DC_LUT_RW_INDEX    the gamma-ramp upload AUTO-INCREMENTS it through a
//                      nested WriteRegister after each entry, so live runs
//                      ahead of whatever the packet set. A write cursor.
//
// Counted apart rather than suppressed, and still named on the report, so
// that `diverge` means "the walk decoded something wrong" and a NEW register
// appearing in the city cannot hide inside an exemption.
bool NrdSideEffectReg(uint32_t reg) {
  return reg == 0x0A31 /* COHER_STATUS_HOST */ ||
         reg == 0x1922 /* DC_LUT_RW_INDEX */;
}
uint64_t g_nrd_sfx = 0;
NrdReg g_nrd_sfx_top[kNrdTop] = {};

void NrdFinding(void*, nr::RegShadowFinding what, uint32_t reg, uint32_t ours,
                uint32_t live) {
  if (what == nr::kRegShadowDiverge && NrdSideEffectReg(reg)) {
    ++g_nrd_sfx;
    NrdTally(g_nrd_sfx_top, reg, ours, live);
    return;
  }
  NrdTally(what == nr::kRegShadowDiverge ? g_nrd_div_top : g_nrd_ext_top, reg,
           ours, live);
}

// [NR-FX] Phase 5-4-0. CP-thread-only. When on, NrWalkRegWrite forwards every
// decoded write to the backend's NrWalkWriteEffects (the WriteRegister dirty
// tail, no value store). Counters + the [nr-fx] line live backend-side, next
// to the effects they count.
bool g_nr_walk_fx = false;

// [NR-SKP] Phase 5-4-2. CP-thread-only, like everything else here.
// g_nr_skip_bufactive is true exactly while NrSkipExecuteBuffer drives the
// walker for the current buffer: NrWalkRegWrite then routes every decoded
// write through the FULL virtual WriteRegister (the executor's apply never
// runs), instead of 5-4-0's effects-only firing.
bool g_nr_skip = false;
bool g_nr_skip_bufactive = false;
// [NR-TILE] N-4 probe latch + counters (CP thread only). See the cvar.
int g_nr_tile_probe = 0;
// [NR-DETILE] N-5 mode: 0 = off, 1 = on, 2 = cycle in place (see the cvar).
int g_nr_detile_mode = 0;
bool g_nr_tile_cycle = false;     // mode 3: walk 0/1/2 in place
bool g_nr_tile_bufskip = false;   // mode 2: this buffer is a repeat band
uint64_t g_tile_draws_skipped = 0, g_tile_draws_kept = 0;
uint64_t g_tile_bufs_skipped = 0, g_tile_bufs_kept = 0;
// The draw handshake: at a draw stop the skip loop stores the stop and
// delegates the draw packet; ExecutePacketType3Draw consumes it instead of
// advancing the walk (which already sits past this draw).
bool g_nr_skip_draw_pending = false;
nr::CtxDrawStop g_nr_skip_stop = {};

// [NR-4A] N-9-4a state (CP thread only; see the gpu_nr_rec_apply cvar).
// The defer list holds one GROUP's inline constant-window ranges (resolved
// at every stop, so it never spans a group); pointers reference the walked
// buffer's bytes, stable for the duration of the execution.
bool g_nr4a_on = false;         // latched cvar
bool g_nr4a_bufactive = false;  // this buffer defers (per-buffer mode gate)
struct Nr4aDef {
  uint32_t base, n, phys;
  const uint32_t* be;
};
constexpr uint32_t kNr4aMaxDef = 256;  // a group carries ~12 constant ranges
Nr4aDef g_nr4a_def[kNr4aMaxDef];
uint32_t g_nr4a_ndef = 0;
bool g_nr4a_group_ovf = false;  // defer list overflowed: group must fall back
uint32_t g_nr4a_scratch[4099];  // record body copy (validated before apply)
// [NR-5C] the fps-increment levers (see the cvars). The three header
// externs (g_nrb_track/apply_seq/reg_epoch) are defined OUTSIDE this
// anonymous namespace, next to g_n7_n8b_phase.
bool g_nr5c_arena_on = false;   // lever (a) live value (cycler-owned if cycling)
int32_t g_nr5c_skip_mode = 0;   // lever (b) live value
bool g_nr5c_cycle = false;
int32_t g_nr5c_phase = -1;
// window counters, reported [nr-5c] 1 Hz
uint64_t g_nr5c_inplace = 0, g_nr5c_copies = 0;      // compose source split
uint64_t g_nr5c_tear = 0, g_nr5c_tear_pkt = 0;       // post-apply VerifyState fails
uint64_t g_nr5c_restart = 0;                         // forced-copy restarts taken
uint64_t g_nr5c_skip_rng = 0, g_nr5c_skip_dw = 0;    // inline runs skipped clean
uint64_t g_nr5c_skip_pdw_rng = 0, g_nr5c_skip_pdw_dw = 0;  // per-dword runs skipped
uint64_t g_nr5c_dirty_rng = 0;   // skippable runs blocked: buckets touched since
uint64_t g_nr5c_genmiss = 0;     // records ineligible: applied_gen != state_gen
uint64_t g_nr5c_stamped = 0;     // StampApplied calls (full-apply completions)
// [NR-5C-ATTR] compose-time bracket, binned [0] = packet-sourced record,
// [1] = suppressed record (the draw whose emission the shims ate). Armed by
// gpu_nr_compose_prof OR by any 5c lever/cycler (the 716 behaviour).
uint64_t g_nr5c_ns[2] = {}, g_nr5c_calls[2] = {};
// The [n7] 1 Hz gate and the [nr-*] 1 Hz gate are INDEPENDENT clocks: each
// window owner needs its own accumulator or one line reports the other's
// window ([[price-stages-in-place]] applies to windows too). These are the
// n7 line's copy, zeroed there.
uint64_t g_n7_cmp_ns[2] = {}, g_n7_cmp_calls[2] = {};
bool g_nr5c_prof = false;     // bracket armed (latched cvar || levers)
bool g_nr5c_cur_sup = false;  // the stop in flight composed a SUPPRESSED record

// [NR-6] N-9-6 repeat-band compose skip (see the cvars).
int32_t g_nr6_mode = 0;
bool g_nr6_cycle = false;
int32_t g_nr6_phase = -1;
uint64_t g_nr6_stops = 0;      // 0x22 stops the hoisted predicate called a band
uint64_t g_nr6_groups = 0;     // ...of which had a deferred group to drop
uint64_t g_nr6_drop_rng = 0, g_nr6_drop_dw = 0;  // ranges/dwords never applied
uint64_t g_nr6_keep_rng = 0;   // ranges the mode still applied at a band stop
// ⚠ THE SAFETY COUNTER. The predicate is read BEFORE the compose and again at
// the dispatch site (where DetileObserve has since run and may have tripped
// the guard). A disagreement means constants were dropped for a draw the
// dispatch would have kept: the drop wins (the invariant is "a draw whose
// constants we dropped never renders"), and this counts how often.
uint64_t g_nr6_pred_ne = 0;
// [NR-6] the suppression-safe path (naruto_724): dwords the mirror pre-pass
// applied before re-testing the band, and bands caught by that LATE test --
// the ones the hoisted read could not see because their surface signature
// was still inside the record.
uint64_t g_nr6_pre_dw = 0, g_nr6_late = 0, g_nr6_pre_mk = 0;

// [NR-97] N-9-7 phase A: the repeat-band walk skip (see the cvars).
int32_t g_n97_mode = 0;
bool g_n97_cycle = false;
int32_t g_n97_phase = -1;
// Per buffer execution, set at entry from de-tile's own predicate.
bool g_n97_armed = false;      // the mode (or its cycler) owns this execution
bool g_n97_bufband = false;    // entry state says this execution is a band
bool g_n97_filter = false;     // the band filter is installed (both phases)
bool g_n97_bufdrop = false;    // ...and the phase is live, so applies DROP
bool g_n97_bufmixed = false;   // ...and a stop inside it disagreed (once)
bool g_n97_decided = false;    // the first draw stop has ruled on this one
uint32_t g_n97_bufkey = 0;
uint32_t g_n97_stops_seen = 0, g_n97_stops_band = 0;
uint64_t g_n97_bufs_eband = 0;  // executions whose ENTRY state read as a band
uint64_t g_n97_bufs_fband = 0;  // ...whose FIRST DRAW STOP read as a band
uint64_t g_n97_bufs_entry = 0;  // ...activated at entry (memo said clean)
uint64_t g_n97_late_band = 0;   // band stops after a non-band first stop:
                                // the win first-stop-decides leaves behind
uint64_t g_n97_pre_dw = 0;      // dwords applied before a mid-buffer activation
uint64_t g_n97_pre_mark = 0;    // ...its snapshot, taken at buffer entry
uint64_t g_n97_dropped_draws = 0;  // mixed draws the activation forced us to
                                   // drop (their state was already gone)
// (SS) THE SOUNDNESS INSTRUMENT. The whole increment rests on one claim: a
// repeat band's applies leave the register file where band 1 already left it,
// so dropping them is invisible to every reader downstream. That is checkable,
// not merely arguable.
//
// ⚠ THE FIRST CUT OF THIS ASKED THE WRONG QUESTION (naruto_733). Comparing
// each dropped dword against the register it is about to overwrite reads 14.7%
// "would change" -- and that number is meaningless, because it measures churn
// INSIDE the execution (a buffer re-loads 72 float4s per draw across ~220
// draws, so of course consecutive writes differ). It says nothing about
// whether the execution's NET effect differs from what was already there.
//
// ⚠ AND THE SECOND CUT ASKED IT AT THE WRONG BOUNDARY (naruto_734). Per
// EXECUTION the answer is not zero and cannot be: band 2 runs buffers A,B,C
// in order, so B starts from A's end state rather than from its own, and
// SQ_PROGRAM_CNTL (0x2180, the first offender the counter named) legitimately
// moves. The claim was never about one buffer. It is about the RUN: band 1
// ends the frame's first pass at some state, band 2 replays the same buffers
// with the same bytes and therefore ends at the SAME state, and so does band
// 3 -- so the whole band-2+3 sequence is a round trip, and only the round trip
// nets to zero.
//
// So the sweep boundary is the FRAME. Every dropped register is remembered at
// its FIRST touch in the frame with the value it held BEFORE that touch, and
// the list is swept at the swap: net_chg = registers the frame's band
// executions left different from how they found them, and it must read 0.
// Covers 100% of the dropped class including by-ref ranges -- the sweep never
// needs the written value, only the register index and the live file.
// ⚠ AND THE THIRD CUT HAD A FALSE POSITIVE (naruto_737): reading the "after"
// value off the LIVE file at the swap blames the band for everything the
// post-pass did. The frame is [band 1][band 2][band 3][post-pass + UI], and
// the post-pass rebinds shaders and textures after the last band execution --
// which is exactly the register profile the counter named (SQ_PROGRAM_CNTL,
// the 0x4800 fetch window, VGT_DRAW_INITIATOR, a few ALU constants, all with
// the same before->after every single frame, i.e. a steady oscillation, not
// drift). So the "after" value is snapshotted at the end of each band
// execution instead, and the sweep compares entry against the last band
// execution's snapshot. Nothing after the bands can reach it now.
// Sampled 1 frame in 8 so it cannot bias the paired fps read it rides along
// with ([[price-stages-in-place]]).
RegisterFile* g_n97_rf = nullptr;
memory::Memory* g_n97_mem = nullptr;  // by-ref payloads, stamped at entry
bool g_n97_sample = false;
std::vector<uint32_t> g_n97_touched;  // per register: index into tlist, +1
std::vector<uint32_t> g_n97_tlist;    // ...the registers, to sweep and clear
std::vector<uint32_t> g_n97_tentry;   // ...their value BEFORE the first touch
std::vector<uint32_t> g_n97_tlast;    // ...and as the last band execution left
                                      //    it, which is the "after" that
                                      //    belongs to the bands alone
std::vector<uint8_t> g_n97_exec_seen;  // touched in THIS execution
std::vector<uint32_t> g_n97_exec_list;
uint64_t g_n97_net_chg = 0;   // registers a band execution NET-changed
uint64_t g_n97_net_same = 0;  // ...left exactly as it found them
uint64_t g_n97_net_frames = 0;  // sampled frames swept
uint64_t g_n97_net_dirty = 0;   // ...of which changed at least one register
uint64_t g_n97_frame_n = 0;     // frames seen, for the 1-in-8 sample
// (SS) The other way a dropped apply could be READ instead of merely
// overwritten: a delegate inside the band execution itself. Every delegate
// still runs -- that is trap naruto_503 and it is not negotiable -- but three
// classes READ the register file the drop has stopped feeding: REG_RMW
// (read-modify-write), COND_WRITE (compares a register) and WAIT_REG_MEM on a
// register poll. The EVENT_WRITE family and MEM_WRITE do not. Counted per
// class so the risk has a size instead of an argument; deleg_read must be 0,
// or the keep-list needs the registers those delegates touch.
// WAIT_REG_MEM is the one that needs splitting rather than counting: it waits
// on MEMORY when its wait_info carries bit 0x10, and on a REGISTER otherwise.
// A register wait on COHER_STATUS_HOST (0x0A31) is below 0x2000 and therefore
// still fed; a wait on anything at or above 0x2000 would be spinning on a
// value this mode has stopped writing, which is a hang, not a glitch.
// wait_other must read 0.
// THE DEFERRED SIDE-EFFECT SET: which registers a band run stored without
// firing their side effects, so the run's end can fire them once from the
// CURRENT values. Last-writer-wins IS the run's net effect, and re-reading the
// live file means that if something after the band moved a register first, the
// re-apply carries the newer value rather than a stale one.
//
// ⚠ THE FIRST CUT OF THIS WAS AN EXACT {base,n} LIST AND IT OVERFLOWED
// (naruto_743: `pend ovf=22340` in one window). The city does not reuse ~150
// ranges, it emits thousands of distinct bases, so an exact list is unbounded
// in the one direction that matters: a range that overflows the cap is stored
// but never re-applied, which resurrects the naruto_742 class for exactly the
// registers the cap dropped. COVERAGE cannot overflow, so this is a bitmap of
// 64-register BLOCKS instead. Marking costs (n/64)+1 stores per range rather
// than one hash probe, the flush emits one apply per CONTIGUOUS RUN of marked
// blocks, and the worst case is the whole register file: 321 blocks, bounded,
// always complete. Over-covering by up to 63 registers per block is harmless -
// those registers are re-applied with the values they already hold, so the
// only effect is that their (correct) side effects fire too.
constexpr uint32_t kN97Blk = 64;
constexpr uint32_t kN97Blocks = (RegisterFile::kRegisterCount + kN97Blk - 1) / kN97Blk;
uint8_t g_n97_blk[kN97Blocks] = {};
uint32_t g_n97_blk_n = 0;  // marked blocks, so an empty run costs one compare
std::vector<uint32_t> g_n97_pend_scratch;
uint64_t g_n97_flush_runs = 0, g_n97_flush_rng = 0, g_n97_flush_dw = 0;
uint64_t g_n97_store_rng = 0, g_n97_store_dw = 0;

uint64_t g_n97_deleg = 0, g_n97_deleg_read = 0;
uint64_t g_n97_wait_mem = 0, g_n97_wait_coher = 0, g_n97_wait_other = 0;
uint32_t g_n97_wait_reg = 0xFFFFFFFFu;
uint64_t g_n97_deleg_op[128] = {};
uint32_t g_n97_chg_reg = 0xFFFFFFFFu;  // the first register that disagreed
// A silent count is ambiguous between "a real class" and "my predicate is
// wrong", so the sweep NAMES what it found: the distinct registers, with the
// value they held before the frame's band executions and after.
constexpr uint32_t kN97ChgSamples = 24;
struct N97ChgRec {
  uint32_t reg, before, after;
  uint64_t n;
};
N97ChgRec g_n97_chg_rec[kN97ChgSamples] = {};
uint32_t g_n97_chg_rec_n = 0;
uint64_t g_n97_chg_rec_ovf = 0;
// The write-by-write read is kept beside it, clearly labelled, because it is
// the number that looks alarming and it must stay visible next to the one that
// actually decides ([[subtraction-buckets-mislabel]] in miniature).
uint64_t g_n97_chg_dw = 0, g_n97_same_dw = 0, g_n97_skip_dw = 0;
uint64_t g_n97_bufs = 0;        // depth-1 executions seen
uint64_t g_n97_bufs_drop = 0;   // ...and were actually walk-skipped
uint64_t g_n97_bufs_ref = 0;    // ...refused: a past execution came out mixed
uint64_t g_n97_bufs_mixed = 0;  // ...that turned out mixed in THIS window
// (SS) THE SAFETY COUNTER. A "mixed" stop is a draw inside a band execution
// whose OWN registers say it is not a repeat band -- its state was already
// dropped, so it must not render, and the buffer is refused from here on.
// This is the one shape that can lose a real draw; it must read 0, and it is
// counted in BOTH cycler phases so the OFF phase prices the risk the ON phase
// took ([[count-refusals-every-granularity]]).
uint64_t g_n97_mixed_stops = 0;
uint64_t g_n97_stops = 0;  // draw stops inside band executions (all classes)
// The traffic itself: what a band execution would stop applying (drop) and
// what it must keep applying (below 0x2000 + the recovery mirror).
uint64_t g_n97_drop_rng = 0, g_n97_drop_dw = 0, g_n97_drop_pdw = 0;
uint64_t g_n97_keep_rng = 0, g_n97_keep_dw = 0, g_n97_keep_pdw = 0;

// Per-buffer memory of what the LAST execution looked like. Direct-mapped and
// lossy on purpose: a collision costs at most one buffer's optimisation or one
// dropped draw before it re-detects, and the city records 40-55k NEW buffer
// addresses a second (N-9-5b-2), so an unbounded set is not an option.
//   kN97Clean  every draw stop of the last execution was a repeat band, so the
//              NEXT one may activate at entry and lose no prefix.
//   kN97Mixed  a stop disagreed: this buffer writes the band registers
//              mid-stream, and it is refused outright from here on.
constexpr uint32_t kN97MemoBits = 13;
enum : uint8_t { kN97Unknown = 0, kN97Clean = 1, kN97Mixed = 2 };
struct N97Memo {
  uint32_t key;
  uint8_t state;
};
N97Memo g_n97_memo[1u << kN97MemoBits] = {};
inline N97Memo* N97MemoSlot(uint32_t key) {
  return &g_n97_memo[(key * 2654435761u) >> (32 - kN97MemoBits)];
}
inline uint8_t N97MemoGet(uint32_t key) {
  const N97Memo* m = N97MemoSlot(key);
  return m->key == key ? m->state : uint8_t(kN97Unknown);
}
inline void N97MemoSet(uint32_t key, uint8_t st) {
  N97Memo* m = N97MemoSlot(key);
  m->key = key;
  m->state = st;
}

// [NR-5C] all buckets of [base, base+n) untouched since applied_seq
// (wrap-safe signed compare; the seq bumps ~2x per draw stop).
inline bool NrbRangeClean(uint32_t base, uint32_t n, uint32_t applied_seq) {
  const uint32_t b1 = (base + n - 1) >> kNrbBucketShift;
  for (uint32_t b = base >> kNrbBucketShift; b <= b1 && b < kNrbBuckets; ++b) {
    if (int32_t(g_nrb_reg_epoch[b] - applied_seq) > 0) return false;
  }
  return true;
}
// [NR-5C] inline runs eligible for the skip: wholly inside the float or
// bool/loop constant windows. Fetch is deliberately excluded -- its texture/
// residency invalidations are correctness ([N8B] fetch-never-deduped rule).
inline bool Nr5cSkippableInline(uint32_t base, uint32_t n) {
  const uint32_t end = base + n - 1;
  return (base >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
          end <= XE_GPU_REG_SHADER_CONSTANT_511_W) ||
         (base >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
          end <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31);
}
// [nr-4a] window counters.
uint64_t g_4a_composed = 0;     // draw stops whose constants applied FROM THE RECORD
uint64_t g_4a_comp_runs = 0, g_4a_comp_dw = 0;
uint64_t g_4a_flush_draw = 0;   // draw stops that fell back to packet ranges
uint64_t g_4a_flush_d2 = 0;     // 0x36 stops (never recorded; packet ranges)
uint64_t g_4a_flush_deleg = 0;  // delegate stops with pending ranges
uint64_t g_4a_flush_end = 0;    // end-of-buffer trailing ranges
uint64_t g_4a_fb_miss = 0, g_4a_fb_nostate = 0, g_4a_fb_stale = 0,
         g_4a_fb_torn = 0, g_4a_fb_shape = 0, g_4a_fb_ovf = 0;
uint64_t g_4a_def_ranges = 0;   // ranges deferred
uint64_t g_4a_def_state = 0;    // [NR-4B-EXT] ...of which state-window ranges
uint64_t g_4a_mk_runs = 0;      // [NR-4B-EXT] by-ref marker entries seen in bodies
uint64_t g_4a_ovf_ranges = 0;   // ranges applied immediately on list overflow
// First shape mismatch each window: [0]=armed, [1]=record const runs (nf),
// [2]=deferred ranges (ndef), [3]=first mismatched index, [4]/[5]=record
// reg/cnt there, [6]/[7]=deferred base/n there (0xFFFFFFFF = past the end).
uint32_t g_4a_shape_samp[8] = {};
// Full lists for that first mismatch (up to 8 each): reg,cnt pairs.
uint32_t g_4a_shape_rec[16] = {};
uint32_t g_4a_shape_def[16] = {};
// [NR-4B] by-ref witness for the shape_ne open question: the from_memory
// (0x2F-class) const-window ranges the CURRENT group applied before its
// stop. If the record's "missing" inline runs show up here, the executed
// group carries the files BY-REF while the record predicts them inline
// (the full-invalidation class); if this is empty at a shape_ne, the
// executed group and the record disagree for another reason entirely.
uint32_t g_4a_byref_n = 0;          // ranges this group
uint32_t g_4a_byref_rng[8] = {};    // first 4 {base,n}
uint32_t g_4a_shape_byref[9] = {};  // sampled at the first shape_ne: n + 4 pairs
uint32_t g_4a_shape_span[3] = {};   // {rec pre_span, stop addr, def0 phys}
// [NR-4A] leak detector: per-dword virtual applies landing in the constant
// windows while a 4a group is pending -- the walker took the per-dword path
// for a packet the defer branch never saw (names the shape_ne mechanism).
uint64_t g_4a_pdw_const = 0;
// [NR-4A] by-ref order guard: a from_memory range overlapped a pending
// deferred range and forced an early flush (stream order restored; the
// group then falls back at the stop). The naruto_663 flashing class.
uint64_t g_4a_byref_conflict = 0;
// [NR-4A] value-verify: shape matched but composed bytes differed from the
// packet bytes (the one-generation-behind torn window); fallback taken.
uint64_t g_4a_fb_val = 0;
// [NR-4B] verify pricing (see the gpu_nr_rec_verify cvar). g_nr4b_verify_n
// is the LIVE rate (the cycler rewrites it); g_nr4b_cycle arms the 10 s
// phase walk in the worker loop. ver/unver split the composed groups by
// whether their memcmp ran; fb_val can only fire among ver.
int32_t g_nr4b_verify_n = 1;
bool g_nr4b_cycle = false;
uint32_t g_nr4b_vctr = 0;  // sampling counter (composed groups)
uint64_t g_4a_ver = 0, g_4a_unver = 0;
// [NR-5A] coherent-compose state (CP thread only; see gpu_nr_rec_coherent).
bool g_nr5_coh = false;              // latched cvar
uint64_t g_nr5_composed = 0;         // coherent composes (subset of composed)
uint64_t g_nr5_def_byref = 0;        // by-ref ranges deferred
uint64_t g_nr5_mk_applied = 0;       // markers applied from the record
uint64_t g_nr5_mk_skipped = 0;       // markers decoding out of scope (per-dword class)
uint64_t g_nr5_draw_issued = 0;      // draws issued from record draw entries
uint64_t g_nr5_fb_args = 0;          // draw entry != packet bytes (torn) -> fallback
uint64_t g_nr5_fb_nodraw = 0;        // record carries no draw entry -> fallback
uint64_t g_nr5_fb_mk = 0;            // marker/by-ref sequence mismatch -> fallback
// The composed draw's packet dwords, pointing into g_nr4a_scratch (stable
// until the next compose); consumed by NrSkipDrawDirect in the same stop.
const uint32_t* g_nr5_draw_src = nullptr;
uint32_t g_nr5_draw_ndw = 0;
// [NR-5B-3] suppressed-group counters.
uint64_t g_nr5b_sup_composed = 0;  // suppressed groups applied from record
uint64_t g_nr5b_sup_argsne = 0;    // record draw entry != kept packet (diag)
uint64_t g_nr5b_sup_refuse = 0;    // suppressed record, coherent off/no entry
uint64_t g_nr5b_pdw_dw = 0;        // mirror/watched-class dwords applied per-dword
uint64_t g_nr5b_pred_stops = 0;    // predicated-out apply-only stops surfaced
uint64_t g_nr5b_sup_foreign = 0;   // record at a recycled address != this draw
// [0]=armed, [1]=pkt addr, [2]=rec_prim<<16|pkt_prim, [3]=rec cnt, [4]=pkt cnt
uint32_t g_nr5b_foreign_samp[5] = {};

// [NR-5B-3i] by-ref source mode on suppressed groups (see gpu_nr_sup_marker).
// g_nr5b_mk_mode is the LIVE value; g_nr5b_mk_cycle arms the 5 s cycler.
int32_t g_nr5b_mk_mode = 0;
bool g_nr5b_mk_cycle = false;
uint64_t g_nr5b_mk_stale = 0;    // applied from the record's remembered address
uint64_t g_nr5b_mk_live = 0;     // applied from the live deferred by-ref
uint64_t g_nr5b_mk_drop = 0;     // marker dropped (redundant with the flush)
uint64_t g_nr5b_mk_addr_ne = 0;  // paired, record address != live address
uint64_t g_nr5b_mk_unpaired = 0; // composable marker with no live by-ref left
uint64_t g_nr5b_mk_paired = 0;   // composable markers paired with a live by-ref

// [NR-5B-3j] fallback-under-suppression (see gpu_nr_sup_fallback).
int32_t g_nr5b_fbmode = 0;
bool g_nr5b_fbcycle = false;
// Latched the first time a SUPPRESSED record is seen: from then on the run is
// known to be emitting suppressed groups, so a declined compose is known to
// be a draw with no packet-side constants. Latched, never cleared -- live
// suppression is a launch flag, and a per-window read would go blind exactly
// in the windows where every compose declined.
bool g_nr5b_sup_live = false;
uint64_t g_nr5b_fb_skipped = 0;  // draws whose dispatch mode 1 dropped
uint64_t g_nr5b_stop_nodef = 0;  // 0x22 stops with NOTHING deferred

// [NR-5B-3k] record-generation probe (see gpu_nr_sup_gen).
int32_t g_nr5b_genmode = 0;
bool g_nr5b_gencycle = false;
uint32_t g_nr5b_exec_seq0 = 0;   // recording seq when THIS buffer began
bool g_nr5b_gen_skip = false;    // this stop's record is ahead; skip if armed
uint64_t g_nr5b_gen_ahead = 0;   // composes whose record post-dates the buffer
uint64_t g_nr5b_gen_seen = 0;    // suppressed composes examined
uint64_t g_nr5b_gen_skipped = 0; // draws mode 1 dropped
int32_t g_nr5b_gen_maxlead = 0;  // largest (rec.seq - exec_seq0) seen
uint64_t g_nr5b_gen_leadsum = 0; // for the mean lead of the ahead class
// [NR-5B-3k] LIVENESS for the ahead counter above. ahead=0 is only a result
// if the comparison is actually running: lag = exec_seq0 - rec.seq, how many
// recordings OLD the consumed record is. A sane, varying, small-positive lag
// says the probe is live and the zero is real; lag pinned at 0, or absurd,
// says the zero is vacuous ([[swapped-out-gates-vacuous]]).
uint64_t g_nr5b_lagsum = 0;
int32_t g_nr5b_lagmin = 0x7FFFFFFF;
int32_t g_nr5b_lagmax = -0x7FFFFFFF;

// [NR-5B-3l] RUN OVERLAP. Every gate so far compared record VALUES against
// packet values and passed. None ever tested the ORDER the record applies
// them in. Under suppression the observed emission positions collapse (the
// packets are gone), so the record's run order is the PREDICTION's order --
// VS, PS, fields, fetch -- not necessarily the order the guest emitted. That
// only matters if two runs in one group touch the same registers: then the
// last writer decides the value, and a reordered pair silently paints the
// wrong constants. This census counts overlapping run pairs per record and
// how many of them actually disagree on the overlapping dwords. Sampled
// 1-in-256: the test is O(runs^2) against ~450k composes a second.
uint64_t g_nr5b_ovl_ctr = 0;
uint64_t g_nr5b_ovl_samples = 0;
uint64_t g_nr5b_ovl_pairs = 0;
uint64_t g_nr5b_ovl_valne = 0;
uint32_t g_nr5b_ovl_samp[5] = {};  // first disagreeing pair: {set,regA,cntA,regB,cntB}
// [NR-5B-DBG] swap-time register-file divergence probe.
uint32_t g_nr5dbg_hash[4] = {};
uint32_t g_nr5dbg_key[8] = {};
void Nr4aFlush(CommandProcessor* cp);

// The deferable predicate: full fit inside ONE of the three pure constant
// windows -- exactly NrWalkRegRange's bulk-window test, and exactly the
// record-run filter in Nr4aTryCompose. One predicate on both sides means
// the shape match can never see a gap or a duplicate.
inline bool Nr4aConstWindow(uint32_t base, uint32_t n) {
  const uint32_t end = base + n - 1;
  return (base >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
          end <= XE_GPU_REG_SHADER_CONSTANT_511_W) ||
         (base >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
          end <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) ||
         (base >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
          end <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31);
}
// [NR-4B-EXT] N-9-4b-ext: the STATE half of the record's captured surface --
// the 40040 field-group windows (+ the 0x5000 fetch tail) the capture gate
// proved exact at both scales. A range is deferable/composable only when the
// walker would actually OFFER it in bulk: wholly inside one captured window
// AND not touching the 27-reg recovery mirror (mirror-crossing runs arrive
// per-dword and stay immediate -- the mirror needs their values live). The
// register windows here are CapPredict's emission windows (record_map.cpp);
// the CmpInScope windows over in [nr-cmp] are the same list plus the
// below-0x2000 coher pair, which must never be deferred (stateful ports).
inline bool Nr4aStateWindow(uint32_t base, uint32_t n) {
  const uint32_t end = base + n;  // exclusive
  const bool in_win =
      (base >= 0x2000u && end <= 0x2010u) ||   // [dev+16] bits 42-57
      (base >= 0x2100u && end <= 0x2115u) ||   // [dev+16] bits 21-41 + 0x2102
      (base >= 0x2180u && end <= 0x2185u) ||   // [dev+16] bits 12-16
      (base >= 0x2200u && end <= 0x220Cu) ||   // [dev+16] & 0xFFF
      (base >= 0x2280u && end <= 0x2295u) ||   // [dev+24] bits 34-54
      (base >= 0x2300u && end <= 0x2326u) ||   // [dev+32] bits 0-37
      (base >= 0x2380u && end <= 0x23A0u) ||   // [dev+32] 38-45 + 0x2388 fam
      (base >= 0x5000u && end <= 0x5003u);     // fetch-flush fixed tail
  if (!in_win) return false;
  // Mirror the walker's offer test exactly (CtxRangeOfferable): >= 0x2322 is
  // always offerable, below it the mirror-overlap test decides.
  return base >= 0x2322u || !nr::CtxRangeTouchesMirrorRegs(base, n);
}
// One predicate for both sides of the shape match: what the walker defers ==
// what the compose filter expects from the record. Body MARKER entries
// (by-ref 0x2F, tag bit 31) fail this naturally and are skipped.
inline bool Nr4aComposable(uint32_t base, uint32_t n) {
  return Nr4aConstWindow(base, n) || Nr4aStateWindow(base, n);
}
// Window counters for the [nr-skp] 1Hz line.
uint64_t g_skp_bufs = 0;         // skip-driven depth-1 buffer executions
uint64_t g_skp_fb = 0;           // cvar on but buffer refused (trace/backend)
uint64_t g_skp_draws = 0;        // draw packets delegated
uint64_t g_skp_deleg = 0;        // non-draw packets delegated
uint64_t g_skp_exec_fail = 0;    // delegated dispatch returned false
uint64_t g_skp_arm_orphan = 0;   // draw stop the handler never consumed
uint64_t g_skp_deleg_op[128] = {};
// [NR-SKP] Phase 5-4-3: the range-level apply. rng counts constant ranges
// bulk-applied through NrSkipApplyRegRange (rng_dw their dwords); pdw counts
// the residual decoded writes still taking the per-dword virtual
// WriteRegister under the skip. The fps hypothesis is rng_dw >> pdw at city.
uint64_t g_skp_rng = 0;
uint64_t g_skp_rng_dw = 0;
uint64_t g_skp_pdw = 0;
// [NR-SKP] Phase 5-4-4a: the direct draw path. direct counts draw stops
// issued by direct call (no delegated re-parse); dfb counts the ones the
// direct decode refused (odd packet shape) that fell back to delegation.
// Gate under the cvar: direct == draws, dfb == 0.
bool g_nr_skip_direct = false;
uint64_t g_skp_direct = 0;
uint64_t g_skp_direct_fb = 0;
// [NR-PB] N-2-2 item 0: widened plain-register bulk apply (see the cvar).
// plain_rng/plain_dw count the ranges the widening accepted that the old
// constant-window gate would have refused into the per-dword path.
bool g_nr_plain_bulk = false;
uint64_t g_skp_plain_rng = 0;
uint64_t g_skp_plain_dw = 0;
// [NR-PLAN] N-2-2 item 2: the consuming swap latch.
bool g_nr_tmpl_swap = false;
bool g_nr_tmpl_swap_probe = false;
// [NR-SPP] 5-4-4 step 0b: skip-path timing (CP-thread-only, gated on
// gpu_nr_skip_profile; one bool test per stop when off).
bool g_skp_prof = false;
uint64_t g_spp_buf_ns = 0;    // whole NrSkipExecuteBuffer
uint64_t g_spp_draw_ns = 0;   // ExecutePacket at draw stops (incl. IssueDraw)
uint64_t g_spp_deleg_ns = 0;  // ExecutePacket at delegated stops
// [NR-5C-ATTR] the draw-stop dispatch bracket split by the record the stop
// composed from ([0] packet-sourced, [1] suppressed). Filled only while the
// compose bracket is armed; the sum stays == g_spp_draw_ns.
uint64_t g_spp_draw_bin_ns[2] = {}, g_spp_draw_bin_n[2] = {};

// [NR-WM] Phase 5-4-8: walk-memo master latch + window counters.
bool g_nr_wm = false;
uint64_t g_wm_rep = 0;     // buffers replayed from a memo stream
uint64_t g_wm_rec = 0;     // buffers recorded (no stream for the key yet)
uint64_t g_wm_cmp = 0;     // verify mode: parse + compare against the stream
uint64_t g_wm_ne = 0;      // verify mode: stream mismatches (gate: 0)
uint64_t g_wm_inval = 0;   // buffers invalidated (bytes changed)
uint64_t g_wm_refuse = 0;  // buffers marked never-memo (nested IB in record)
uint64_t g_wm_cap = 0;     // commits that hit the byte cap (store cleared)
uint64_t g_wm_abort = 0;   // recordings dropped on an executor abort

// [NR-PKT] Phase 5-4-1: the non-draw packet census. CP-thread-only. Counted
// at the executor's own dispatch, AFTER its predicate skip, so the tallies
// are execution truth (a predicated-out packet never runs and is not listed
// -- the same rule the 4b-1 walkers and the join list obey).
bool g_nr_pkt = false;
uint64_t g_pkt_op[128];            // type-3 dispatches inside depth-1 IBs
uint64_t g_pkt_op_bufs[128];       // depth-1 executions containing the op
uint64_t g_pkt_op_drawbufs[128];   // ...that also executed a draw
uint64_t g_pkt_t0_pkts = 0, g_pkt_t0_regs = 0, g_pkt_t1_pkts = 0,
         g_pkt_t2_pkts = 0;
// Stateful-port register writes (0 scratch / 1 COHER_STATUS_HOST / 2 DC_LUT),
// the precord must-not-defer enumeration: WriteRegister behavior beyond the
// value store that is NOT idempotent (memory writeback / RMW / a write
// cursor), so the walk cannot re-fire it and a skipped buffer cannot carry it.
uint64_t g_pkt_sfx_writes[3] = {};
uint64_t g_pkt_sfx_bufs[3] = {};
uint64_t g_pkt_sfx_drawbufs[3] = {};
uint64_t g_pkt_bufs = 0, g_pkt_drawbufs = 0;
// Current depth-1 buffer execution.
uint64_t g_pkt_buf_ops[2] = {};    // 128-bit op-presence set
uint32_t g_pkt_buf_sfx = 0;        // 3-bit class-presence set
bool g_pkt_buf_draw = false;
bool g_pkt_buf_open = false;

void NrPktBufBegin() {
  g_pkt_buf_ops[0] = g_pkt_buf_ops[1] = 0;
  g_pkt_buf_sfx = 0;
  g_pkt_buf_draw = false;
  g_pkt_buf_open = true;
}

void NrPktBufEnd() {
  if (!g_pkt_buf_open) return;
  g_pkt_buf_open = false;
  ++g_pkt_bufs;
  if (g_pkt_buf_draw) ++g_pkt_drawbufs;
  for (uint32_t w = 0; w < 2; ++w) {
    uint64_t bits = g_pkt_buf_ops[w];
    while (bits) {
      const uint32_t b = static_cast<uint32_t>(__builtin_ctzll(bits));
      bits &= bits - 1;
      const uint32_t op = w * 64 + b;
      ++g_pkt_op_bufs[op];
      if (g_pkt_buf_draw) ++g_pkt_op_drawbufs[op];
    }
  }
  for (uint32_t c = 0; c < 3; ++c) {
    if (g_pkt_buf_sfx & (1u << c)) {
      ++g_pkt_sfx_bufs[c];
      if (g_pkt_buf_draw) ++g_pkt_sfx_drawbufs[c];
    }
  }
}

// Classify the register writes of a type-0/type-1 packet against the three
// stateful classes. `writes` is how many stores land on each register in
// [lo, hi] (a one-reg type-0 packet stores `count` times to one register --
// the DC_LUT gamma machine advances per STORE, so the multiplicity matters).
void NrPktRegRange(uint32_t lo, uint32_t hi, uint64_t writes) {
  auto hit = [&](uint32_t rlo, uint32_t rhi) -> uint64_t {
    if (lo > rhi || hi < rlo) return 0;
    return (std::min(hi, rhi) - std::max(lo, rlo) + 1) * writes;
  };
  uint64_t n;
  if ((n = hit(XE_GPU_REG_SCRATCH_REG0, XE_GPU_REG_SCRATCH_REG7)) != 0) {
    g_pkt_sfx_writes[0] += n;
    g_pkt_buf_sfx |= 1u << 0;
  }
  if ((n = hit(XE_GPU_REG_COHER_STATUS_HOST, XE_GPU_REG_COHER_STATUS_HOST)) != 0) {
    g_pkt_sfx_writes[1] += n;
    g_pkt_buf_sfx |= 1u << 1;
  }
  n = hit(XE_GPU_REG_DC_LUT_RW_INDEX, XE_GPU_REG_DC_LUT_RW_INDEX) +
      hit(XE_GPU_REG_DC_LUT_SEQ_COLOR, XE_GPU_REG_DC_LUT_SEQ_COLOR) +
      hit(XE_GPU_REG_DC_LUT_PWL_DATA, XE_GPU_REG_DC_LUT_PWL_DATA) +
      hit(XE_GPU_REG_DC_LUT_30_COLOR, XE_GPU_REG_DC_LUT_30_COLOR);
  if (n) {
    g_pkt_sfx_writes[2] += n;
    g_pkt_buf_sfx |= 1u << 2;
  }
}

// The walk-transcribed set: packet classes the 4a-4c decoder already applies
// (constants, shader loads, draws, the bin family, and the one-dword/NOP
// skips). Everything else on the report is a transcribe-or-refuse decision.
bool NrPktStreamOp(uint32_t op) {
  switch (op) {
    case PM4_NOP:                  // 0x10
    case PM4_DRAW_INDX:            // 0x22
    case PM4_DRAW_INDX_2:          // 0x36
    case PM4_IM_LOAD:              // 0x27
    case PM4_IM_LOAD_IMMEDIATE:    // 0x2b
    case PM4_SET_CONSTANT:         // 0x2d
    case PM4_LOAD_ALU_CONSTANT:    // 0x2f
    case PM4_SET_CONSTANT2:        // 0x55
    case PM4_SET_SHADER_CONSTANTS: // 0x56
    case PM4_SET_BIN_MASK:         // 0x50
    case PM4_SET_BIN_SELECT:       // 0x51
    case PM4_SET_BIN_MASK_LO:      // 0x60
    case PM4_SET_BIN_MASK_HI:      // 0x61
    case PM4_SET_BIN_SELECT_LO:    // 0x62
    case PM4_SET_BIN_SELECT_HI:    // 0x63
      return true;
    default:
      return false;
  }
}

const char* NrPktOpName(uint32_t op) {
  switch (op) {
    case PM4_ME_INIT: return "ME_INIT";
    case PM4_INTERRUPT: return "INTERRUPT";
    case PM4_XE_SWAP: return "XE_SWAP";
    case PM4_INDIRECT_BUFFER: return "INDIRECT_BUFFER";
    case PM4_INDIRECT_BUFFER_PFD: return "INDIRECT_BUFFER_PFD";
    case PM4_WAIT_REG_MEM: return "WAIT_REG_MEM";
    case PM4_WAIT_FOR_IDLE: return "WAIT_FOR_IDLE";
    case PM4_REG_RMW: return "REG_RMW";
    case PM4_REG_TO_MEM: return "REG_TO_MEM";
    case PM4_MEM_WRITE: return "MEM_WRITE";
    case PM4_COND_WRITE: return "COND_WRITE";
    case PM4_COND_EXEC: return "COND_EXEC";
    case PM4_EVENT_WRITE: return "EVENT_WRITE";
    case PM4_EVENT_WRITE_SHD: return "EVENT_WRITE_SHD";
    case PM4_EVENT_WRITE_CFL: return "EVENT_WRITE_CFL";
    case PM4_EVENT_WRITE_EXT: return "EVENT_WRITE_EXT";
    case PM4_EVENT_WRITE_ZPD: return "EVENT_WRITE_ZPD";
    case PM4_VIZ_QUERY: return "VIZ_QUERY";
    case PM4_INVALIDATE_STATE: return "INVALIDATE_STATE";
    case PM4_CONTEXT_UPDATE: return "CONTEXT_UPDATE";
    case PM4_SET_STATE: return "SET_STATE";
    case PM4_LOAD_CONSTANT_CONTEXT: return "LOAD_CONSTANT_CONTEXT";
    case PM4_IM_STORE: return "IM_STORE";
    case PM4_MEM_WRITE_CNTR: return "MEM_WRITE_CNTR";
    case PM4_DRAW_INDX_BIN: return "DRAW_INDX_BIN";
    case PM4_DRAW_INDX_2_BIN: return "DRAW_INDX_2_BIN";
    default: return "?";
  }
}

// [NR-ISSUE] Increment 4d. CP-thread-only, like everything else here.
bool g_nr_issue = false;
int64_t g_nri_from = 0;
int64_t g_nri_count = -1;
uint64_t g_nri_ordinal = 0;      // lockstep geometry stops since boot (cumulative)
uint64_t g_nri_armed = 0;        // window: draws armed for the backend
uint64_t g_nri_copy_armed = 0;   // window: kCopy draws (resolves) armed --
                                 // shadow-issued like everything else since
                                 // 4f, split out so the report keeps naming
                                 // the resolve population
uint64_t g_nri_sh_invalid = 0;   // window: walk shader refs not yet valid
uint64_t g_nri_sh_mismatch = 0;  // window: walk-resolved shader != live active
                                 // (issued from the walk's anyway -- that IS
                                 // the recovered path; nonzero names a 4b-2
                                 // tracking hole)
// [NR-ISSUE] Increment 4e: THE REPLAY REGISTER FILE -- the persistent file
// draws are issued from, maintained INCREMENTALLY. 4d rebuilt it per draw
// (RegShadowCompose over 20,483 registers + an 82 KB copy into the backend's
// private file, per draw, on the Ch.9 bottleneck thread) and the city paid
// 4x its fps for that (naruto_318: ~7.8 vs 30.1). A replay never needs the
// rebuild: the walk already decodes every write in execution order, so the
// file is seeded ONCE (compose: shadow values where the stream has written,
// live elsewhere) and each decoded write is applied as it happens. The six
// registers the stream never carries (4 externs + 2 ports, the 4c closure)
// are refreshed from live at each arm -- "read it once", performed per draw
// because it is six loads, not a policy.
RegisterFile g_nri_file;
bool g_nri_seeded = false;
// [NR-ISSUE-LIVE] latched per config epoch: mirror writes are skipped and
// the arm points nr_issue_file_ at the live file.
bool g_nri_live = false;
constexpr uint32_t kNriNonStreamRegs[] = {
    0x0081,  // SCLK_PWRMGT_CNTL2_REG   extern, power
    0x01C5,  // CP_RB_WPTR              extern, ring bookkeeping
    0x0A31,  // COHER_STATUS_HOST       sfx port (coherency handshake)
    0x0D00,  // SQ_GPR_MANAGEMENT       extern, THE draw-state init constant
    0x1922,  // DC_LUT_RW_INDEX         sfx port (gamma write cursor)
    0x21F9,  // VGT_EVENT_INITIATOR     extern, event trigger
};

// Applies whatever is left of the buffer and folds the walk's per-draw flags
// and stats into the window counters. Called at buffer entry when the shadow
// is off (the pre-4c whole-buffer behaviour) and after execution when it is
// on, so the two modes tally identically.
void CtxFinishWalk() {
  const uint32_t nf = nr::CtxWalkFinish(&g_ctx_walker);
  const nr::CtxWalkStats& cst = g_ctx_walk_stats;
  ++g_ctx_execs;
  g_ctx_nop0 += cst.nop0;
  g_ctx_sc2 += cst.set_const2;
  g_ctx_pred += cst.pred_skipped;
  g_ctx_pred_draws += cst.pred_draws;
  g_ctx_pred_draws_run += cst.pred_draws_run;
  g_ctx_bin_pkts += cst.bin_pkts;
  if (cst.draws22 > nf) ++g_ctx_flags_ovf;
  g_ctx_mem_loads += cst.mem_loads;
  g_ctx_poisoned += cst.mem_poisoned;
  g_ctx_im_loads += cst.im_loads;
  g_ctx_im_imms += cst.im_load_imms;
  g_ctx_draws += nf;
  for (uint32_t i = 0; i < nf; ++i) {
    const uint16_t f = g_ctx_flags[i];
    const bool rt = f & nr::kCtxDrawRtDef;
    const bool vp = f & nr::kCtxDrawVportDef;
    const bool md = f & nr::kCtxDrawModeDef;
    const bool sh = f & nr::kCtxDrawShadersDef;
    if (rt) ++g_ctx_rt_def;
    if (vp) ++g_ctx_vp_def;
    if (md) ++g_ctx_mode_def;
    if (sh) ++g_ctx_sh_def;
    if (rt && vp && md && sh) ++g_ctx_full;
    if (f & nr::kCtxDrawCopy) ++g_ctx_copy;
    if (f & nr::kCtxDrawRtCarried) ++g_ctx_rt_carry;
    if (f & nr::kCtxDrawVportCarried) ++g_ctx_vp_carry;
    if (f & nr::kCtxDrawModeCarried) ++g_ctx_mode_carry;
    if (f & nr::kCtxDrawShadersCarried) ++g_ctx_sh_carry;
  }
}

// Registers compared per draw. The file is 20,483 registers and the city
// issues ~200k draws a second, so a full compare per draw would cost more
// than the thing being measured; a rolling slice sweeps the whole file
// thousands of times a second instead. A divergence persists until the
// register is rewritten, so this catches one within a fraction of a
// millisecond rather than instantly -- which is the honest trade and is
// stated on the report line as sweeps/s.
constexpr uint32_t kNrdSweepPerDraw = 256;

// [NR-RES] Increment 4b-3: the resource census. Rides the increment-4a
// context walk (one decoder, per the 4b-1 lesson) and mirrors the four
// constant files a draw's resources live in. Ground truth is the same gate
// that settled 4a and 4b-1: at buffer entry the mirror must equal the live
// register file.
nr::ResourceContext g_res_state;
nr::ResStats g_res_stats;
constexpr uint32_t kResDivSamples = 6;
nr::ResDivergence g_res_div_samp[kResDivSamples];
uint32_t g_res_div_samp_n = 0;
// Distinct fetch base addresses seen at draw time this window: how many
// distinct buffers a replay would have to bind.
constexpr uint32_t kResAddrSetSize = 8192;
uint32_t g_res_addr_set[kResAddrSetSize];
uint64_t g_res_addr_distinct = 0, g_res_addr_ovf = 0;

// [NR-RUSE] Phase 5-4-5 inc 0: the replay state-reuse pricing probe. All
// state below is command-processor-thread-only. A draw's inputs are exactly:
// the buffer bytes up to its packet end (prefix), every by-ref LOAD_ALU
// payload applied before it (chain digest), and the replay-file state at
// buffer entry (component shadows, bin trio split out because the ring
// rewrites it per tile). See NEXT-AGENT-5.md "5-4-5" for the model.
bool g_nr_ruse = false;
bool g_nr_ruse_v0 = false;  // the v0 digest/prefix reporting rides the probe
                            // cvar only; the fast path needs just v2
bool g_nr_ruse_ep = false;  // [NR-RUSE-EP] opt-in epoch compare shortcut
struct RuseBuf {
  uint32_t dwords = 0;      // buffer size at last replay
  uint32_t last_swap = 0;   // swap_counter() at last replay
  bool have_entry = false;  // entry shadows captured (g_nri_file was seeded)
  // [NR-RUSE-EP] granule dirty-epoch sum over the buffer's range at last
  // replay (the bufcache scheme): unchanged sum = no recorded packet touched
  // the range since, so the byte compare would find nothing. Soundness is
  // the same empirical claim the bufcache shipped on (vne=0); the ep_ne
  // counter re-proves it on every verify run.
  uint64_t last_epoch = 0;
  bool have_epoch = false;
  std::vector<uint8_t> shadow;  // raw buffer bytes at last replay
  // Entry-state shadows (host-order dwords from g_nri_file at buffer entry).
  std::vector<uint32_t> e_ctl;             // 0x2000..0x2FFF (4096 dwords)
  std::vector<uint32_t> e_flt;             // 0x4000..0x47FF (2048)
  std::vector<uint32_t> e_ftc;             // 0x4800..0x48BF (192)
  std::vector<uint32_t> e_bl;              // 0x4900..0x4927 (40)
  uint32_t e_bin[3] = {0, 0, 0};           // the per-tile trio 0x2080-0x2082
  nr::ShaderRef e_vs = {}, e_ps = {};      // walker refs at entry
  std::vector<uint32_t> byref_prev;        // by-ref value stream, prev replay
  // [NR-BFC] 5-4-6-0: native-stream generation at this buffer's last replay
  // (a crossing kills the zero-copy span; the census prices how often).
  uint64_t bfc_submission = 0;
  bool bfc_have_sub = false;
};
struct RuseDraw {
  uint32_t key = 0;                    // full packet phys addr (slot validity)
  uint8_t used = 0;
  uint64_t digest = 0;                 // v0: by-ref chain digest at the stop
  uint32_t vs_addr = 0, ps_addr = 0;   // walker shader refs at the stop
  uint32_t vs_size = 0, ps_size = 0;
};
std::unordered_map<uint32_t, RuseBuf> g_ruse_bufs;  // phys ptr -> buf
// Per-draw records in a direct map (the join cache's proven shape): the
// city classifies ~270k draws/s, so a hash map find per stop is real cost.
// Collisions overwrite (counted) and read as first-seen, never as stale.
constexpr uint32_t kRuseDrawMapBits = 20;
std::vector<RuseDraw> g_ruse_draws_m;  // sized 1<<kRuseDrawMapBits on arm
inline RuseDraw* RuseDrawSlot(uint32_t key) {
  return &g_ruse_draws_m[(key >> 2) & ((1u << kRuseDrawMapBits) - 1)];
}
uint64_t g_ruse_w_coll = 0;  // slot collisions (info)
size_t g_ruse_shadow_bytes = 0;
constexpr size_t kRuseShadowCap = 128ull << 20;  // clear-all backstop
// Per-replay classification, set at buffer entry, read at each draw stop.
uint64_t g_ruse_byref_h = 0;          // chain digest of by-ref payloads so far
uint32_t g_ruse_first_diff = 0;       // first differing dword vs prev replay
bool g_ruse_have_prev = false;        // prev replay comparable (size+entry ok)
// [NR-WM] whole-buffer byte identity vs the previous COMPARED execution --
// the walk memo's validity gate (the shadow updates only here, so identity
// chains across executions exactly as the memo needs).
bool g_ruse_bytes_ident = false;
bool g_ruse_same_frame = false;
bool g_ruse_entry_core_eq = false;    // all components eq (bin trio excluded)
bool g_ruse_bin_ne = false;           // ONLY the bin trio differed
uint8_t g_ruse_entry_ne_mask = 0;     // bit 0..4 = ctl/flt/ftc/bl/sh
// [NR-RUSE] v2: the stale-register set. Layout is stable under in-place
// patching, so a changed source dword changes exactly the register it
// writes, and only until a later write from UNCHANGED bytes re-converges
// the value stream. A draw is v2-reusable when the set is empty (and its
// shader refs match its previous execution). Entry diffs seed the set;
// walk applies mark (changed source) or clear (unchanged source) it.
std::vector<uint64_t> g_ruse_diff;      // changed-dword bitmap, this replay
// [NR-RUSE-EP] when set, this replay's diff bitmap is all-zero BY
// CONSTRUCTION and g_ruse_diff may be stale-sized -- every reader must
// short-circuit on this flag before touching the bitmap.
bool g_ruse_diff_empty = false;
std::vector<uint32_t> g_ruse_byref_cur; // by-ref value stream, this replay
std::vector<uint8_t> g_ruse_stale;      // per-register stale flag (0x5003)
std::vector<uint32_t> g_ruse_stale_list;  // regs ever marked this replay
uint32_t g_ruse_stale_cnt = 0;
// [NR-RUSE-BLK] stale registers per 64-register block: the O(1) summary that
// lets a clean window be skipped whole. Maintained in lockstep with
// g_ruse_stale -- if these two ever disagree the skip is unsound, so every
// site that touches one touches the other.
std::vector<uint8_t> g_ruse_stale_blk;
bool g_nr_ruse_blk = false;
uint64_t g_ruse_blk_skip = 0;  // dwords skipped whole
uint64_t g_ruse_blk_mark = 0;  // dwords still marked one by one
uint32_t g_ruse_stale_sample = 0xFFFFFFFFu;  // last reg marked stale
RuseBuf* g_ruse_cur_buf = nullptr;      // this replay's buffer entry
const uint8_t* g_ruse_raw = nullptr;    // this replay's buffer bytes
uint32_t g_ruse_dwords = 0;
bool g_ruse_v2_ok = false;              // buffer+entry comparable for v2
bool g_ruse_deleg_poison = false;       // reg-writing delegate ran this replay
uint32_t g_ruse_pkt_dw = 0xFFFFFFFFu;   // per-packet payload cursor
uint32_t g_ruse_pkt_m = 0;
// [NR-RUB] 5-4-5-1: the pending stop's verdict, read by the backend's
// bundle gate while it derives this draw (same thread, synchronous).
uint32_t g_ruse_stop_key = 0;
bool g_ruse_stop_r2 = false;
bool g_ruse_stop_sf = false;
bool g_ruse_stop_valid = false;
// [NR-RUF-V2B] the miss was STALE-ONLY: prev comparable, packet span clean,
// shaders equal, no delegate poison -- only the stale-register set blocked
// reuse. The backend may upgrade such a miss when every stale register is a
// float constant neither active shader reads (bitmap-packed packs never
// carry it). The stale set itself is published via NrRuseStaleRegs.
bool g_ruse_stop_stale_only = false;
// Window counters (reset each 1Hz report).
uint64_t g_ruse_w_bufs = 0, g_ruse_w_new = 0, g_ruse_w_id = 0, g_ruse_w_ch = 0;
uint64_t g_ruse_w_sf = 0, g_ruse_w_sf_id = 0, g_ruse_w_xf = 0, g_ruse_w_xf_id = 0;
uint64_t g_ruse_w_chdw = 0;  // changed dwords summed over changed replays
uint64_t g_ruse_w_entry_eq = 0, g_ruse_w_entry_binonly = 0;
uint64_t g_ruse_w_entry_ne[5] = {};  // ctl flt ftc bl sh
uint32_t g_ruse_w_ctl_diff_reg = 0xFFFFFFFFu;  // sample: first ctl reg that ne'd
uint64_t g_ruse_w_draws = 0, g_ruse_w_first = 0, g_ruse_w_reuse = 0;
uint64_t g_ruse_w_reuse_sf = 0, g_ruse_w_reuse_xf = 0, g_ruse_w_reuse_binonly = 0;
uint64_t g_ruse_w_miss_pfx = 0, g_ruse_w_miss_byref = 0;
uint64_t g_ruse_w_miss_entry[5] = {};  // first-fail attribution, ctl..sh
uint64_t g_ruse_w_evict = 0, g_ruse_w_cost_ns = 0;
// [NR-RUSE-EP] epoch gate: replays whose epoch sum matched the previous
// replay (compare skippable), and -- verify runs only -- how often a
// clean-predicted replay actually had changed bytes (the soundness gate,
// must stay 0).
uint64_t g_ruse_w_ep_clean = 0, g_ruse_w_ep_ne = 0;
// v2 window counters.
uint64_t g_ruse_w_first2 = 0, g_ruse_w_reuse2 = 0;
uint64_t g_ruse_w_reuse2_sf = 0, g_ruse_w_reuse2_xf = 0;
// [NR-BUFREPLAY] 60-lever candidate census (v0 reporting only): draws whose
// buffer replay is WHOLLY byte-identical, same-frame, entry-core-eq -- the
// population a buffer-level native replay could skip walk+issue for; the
// second counter additionally requires the draw's by-ref prefix digest eq
// (a whole-buffer skip must also prove its by-ref reads unchanged).
uint64_t g_ruse_w_bufid_draws = 0, g_ruse_w_bufid_byref = 0;
uint64_t g_ruse_w_miss2_pkt = 0, g_ruse_w_miss2_stale = 0;
uint64_t g_ruse_w_stale_sum = 0, g_ruse_w_miss2_sh = 0, g_ruse_w_miss2_deleg = 0;
uint64_t g_ruse_w_pdw_cons = 0;  // per-dword applies with no offset mapping

// [NR-BFC] Phase 5-4-6-0: the buffer-level native-replay census
// (CP-thread-only). One BfcPerBuf per skip-driven buffer execution, folded
// at buffer end into the candidate or non-candidate aggregate. Candidate =
// the measured bufid rule (bytes identical + same-frame + entry-core-eq +
// whole-buffer byref-eq), evaluated BEFORE NrRuseBufEnd swaps the by-ref
// stream. Design + decision gates: NEXT-AGENT-5.md "5-4-6".
bool g_nr_bfc = false;
// [NR-DSP] Phase 5-4-7-0: per-draw native span probe (backend-side storage).
bool g_nr_dsp = false;
// [NR-SPR] Phase 5-4-7-1: context-free span record + replay-prediction gate.
bool g_nr_spr = false;
struct BfcPerBuf {
  // Delegate schedule (the replay must still run these at their stops).
  uint32_t deleg = 0;
  uint32_t deleg_events = 0;      // EVENT_WRITE family + INTERRUPT
  uint32_t deleg_wait_mem = 0;    // WAIT_REG_MEM, memory poll
  uint32_t deleg_wait_coher = 0;  // WAIT_REG_MEM on COHER_STATUS_HOST
  uint32_t deleg_wait_other = 0;  // WAIT_REG_MEM on any other register
  uint32_t deleg_memw = 0;        // MEM_WRITE
  uint32_t deleg_other = 0;
  // bit0 REG_RMW, bit1 COND_WRITE, bit2 nested IB, bit3 XE_SWAP,
  // bit4 WAIT on a non-COHER register -- any bit refuses the buffer.
  uint32_t refuse_mask = 0;
  uint32_t deleg_before_draw = 0;  // schedule position: before the 1st draw
  bool saw_draw = false;
  // Walk apply stream: end-state size + fixup-input classes.
  uint32_t rng = 0, pdw = 0;
  uint32_t dw_float = 0, dw_fetch = 0, dw_bl = 0, dw_ctl = 0, dw_other = 0;
  uint32_t pdw_stateful = 0;  // per-dword writes outside constants + ctl
  // [NR-BFC] gap 2: the sub-0x2000 writes split by whether they are the 2
  // known side-effect ports (which a replay must fire in ORDER) or plain
  // registers (which the bulk end-state can carry).
  uint32_t pdw_port = 0;
  uint32_t pdw_plain = 0;
  uint32_t frags = 0;         // non-contiguous apply fragments
  uint32_t last_end = 0xFFFFFFFFu;
  uint32_t draws = 0;
};
BfcPerBuf g_bfc_buf;
// Distinct written registers this buffer (end-state size): bitmap + dirty
// word list so the per-buffer clear is O(written), never O(file).
std::vector<uint64_t> g_bfc_wset;
std::vector<uint32_t> g_bfc_wset_words;
uint32_t g_bfc_wset_distinct = 0;
struct BfcAgg {
  uint64_t bufs = 0, draws = 0;
  uint64_t deleg = 0, dl_events = 0, dl_wait_mem = 0, dl_wait_coher = 0,
           dl_wait_other = 0, dl_memw = 0, dl_other = 0, dl_before_draw = 0;
  uint64_t refuse_bufs = 0;
  uint64_t refuse_by[5] = {};  // rmw / condw / nested ib / xe_swap / wait-reg
  uint64_t span_el = 0, sp_draw = 0, sp_pso = 0, sp_sys = 0, sp_cbv = 0,
           sp_root = 0, sp_ia = 0, sp_vp = 0, sp_sci = 0, sp_om_rt = 0,
           sp_om_misc = 0, sp_bar = 0, sp_copy = 0, sp_clear = 0,
           sp_disp = 0, sp_query = 0, sp_heaps = 0, sp_marker = 0,
           sp_other = 0;
  uint64_t viol_bufs = 0;  // bufs whose span holds any non-whitelist command
  uint64_t viol_draws = 0;  // and the DRAWS they carry (the honest weight)
  uint64_t pdw_port = 0, pdw_plain = 0;
  uint32_t vp_max = 0, sci_max = 0, sys_max = 0;
  uint64_t rt_runs = 0;
  uint32_t rt_max = 0;
  uint64_t sub_cross = 0;  // replays whose prev replay was another submission
  uint64_t wset = 0;
  uint32_t wset_max = 0;
  uint64_t frags = 0, dw_ctl = 0, dw_other = 0, pdw_stateful = 0;
  uint64_t byref_dw = 0;
};
BfcAgg g_bfc_cand, g_bfc_non;
uint64_t g_bfc_backend_missing = 0;
uint32_t g_bfc_wait_reg_sample = 0xFFFFFFFFu;

inline void NrBfcMarkWritten(uint32_t reg, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t r = reg + i;
    if (r >= RegisterFile::kRegisterCount) break;
    uint64_t& w = g_bfc_wset[r >> 6];
    const uint64_t bit = 1ull << (r & 63);
    if (!(w & bit)) {
      if (!w) g_bfc_wset_words.push_back(r >> 6);
      w |= bit;
      ++g_bfc_wset_distinct;
    }
  }
}

// One walk apply (bulk range or per-dword) into the census.
inline void NrBfcApply(uint32_t base, uint32_t n, bool per_dword) {
  BfcPerBuf& b = g_bfc_buf;
  if (per_dword) ++b.pdw; else ++b.rng;
  if (base != b.last_end) ++b.frags;
  b.last_end = base + n;
  if (base >= 0x4000 && base + n <= 0x4800) {
    b.dw_float += n;
  } else if (base >= 0x4800 && base + n <= 0x48C0) {
    b.dw_fetch += n;
  } else if (base >= 0x4900 && base + n <= 0x4928) {
    b.dw_bl += n;
  } else if (base >= 0x2000 && base + n <= 0x3000) {
    b.dw_ctl += n;
  } else {
    b.dw_other += n;
    if (per_dword) {
      b.pdw_stateful += n;
      // The 4c closure: exactly two ports carry side effects on write.
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t r = base + i;
        if (r == XE_GPU_REG_COHER_STATUS_HOST || r == XE_GPU_REG_DC_LUT_RW_INDEX) {
          ++b.pdw_port;
        } else {
          ++b.pdw_plain;
        }
      }
    }
  }
  NrBfcMarkWritten(base, n);
}

// Buffer begin: reset the per-buffer state (bitmap cleared by dirty list).
inline void NrBfcBufReset() {
  g_bfc_buf = BfcPerBuf{};
  const size_t words = (RegisterFile::kRegisterCount + 63) / 64;
  if (g_bfc_wset.size() != words) {
    g_bfc_wset.assign(words, 0);
    g_bfc_wset_words.clear();
  } else {
    for (uint32_t w : g_bfc_wset_words) g_bfc_wset[w] = 0;
    g_bfc_wset_words.clear();
  }
  g_bfc_wset_distinct = 0;
}

// Buffer end: classify candidate vs not, collect the backend sample, fold.
void NrBfcFold(CommandProcessor* cp) {
  RuseBuf* rb = g_ruse_cur_buf;
  const bool bytes_id =
      g_ruse_have_prev && g_ruse_first_diff >= g_ruse_dwords;
  const bool byref_eq = rb && g_ruse_byref_cur == rb->byref_prev;
  const bool cand = bytes_id && g_ruse_same_frame && g_ruse_entry_core_eq &&
                    byref_eq;
  CommandProcessor::NrBfcBackendSample s;
  const bool have_backend = cp->NrBfcBufEnd(&s);
  if (!have_backend) ++g_bfc_backend_missing;
  bool sub_cross = false;
  if (rb && have_backend) {
    if (rb->bfc_have_sub && s.submission_id != rb->bfc_submission) {
      sub_cross = true;
    }
    rb->bfc_submission = s.submission_id;
    rb->bfc_have_sub = true;
  }
  const BfcPerBuf& b = g_bfc_buf;
  BfcAgg& a = cand ? g_bfc_cand : g_bfc_non;
  ++a.bufs;
  a.draws += b.draws;
  a.deleg += b.deleg;
  a.dl_events += b.deleg_events;
  a.dl_wait_mem += b.deleg_wait_mem;
  a.dl_wait_coher += b.deleg_wait_coher;
  a.dl_wait_other += b.deleg_wait_other;
  a.dl_memw += b.deleg_memw;
  a.dl_other += b.deleg_other;
  a.dl_before_draw += b.deleg_before_draw;
  if (b.refuse_mask) ++a.refuse_bufs;
  for (uint32_t i = 0; i < 5; ++i) {
    if (b.refuse_mask & (1u << i)) ++a.refuse_by[i];
  }
  if (have_backend) {
    a.span_el += s.span_elements;
    a.sp_draw += s.cmd_draw;
    a.sp_pso += s.cmd_pso;
    a.sp_sys += s.cmd_sys_cbv;
    a.sp_cbv += s.cmd_root_cbv;
    a.sp_root += s.cmd_root_other;
    a.sp_ia += s.cmd_ia;
    a.sp_vp += s.cmd_vp;
    a.sp_sci += s.cmd_sci;
    a.sp_om_rt += s.cmd_om_rt;
    a.sp_om_misc += s.cmd_om_misc;
    a.sp_bar += s.cmd_barrier;
    a.sp_copy += s.cmd_copy;
    a.sp_clear += s.cmd_clear;
    a.sp_disp += s.cmd_dispatch;
    a.sp_query += s.cmd_query;
    a.sp_heaps += s.cmd_heaps;
    a.sp_marker += s.cmd_marker;
    a.sp_other += s.cmd_other;
    if (s.cmd_barrier + s.cmd_copy + s.cmd_clear + s.cmd_dispatch +
        s.cmd_query + s.cmd_other) {
      ++a.viol_bufs;
      a.viol_draws += b.draws;
    }
    a.vp_max = std::max(a.vp_max, s.cmd_vp);
    a.sci_max = std::max(a.sci_max, s.cmd_sci);
    a.sys_max = std::max(a.sys_max, s.cmd_sys_cbv);
    a.rt_runs += s.rt_body_runs;
    a.rt_max = std::max(a.rt_max, uint32_t(s.rt_body_runs));
    if (sub_cross) ++a.sub_cross;
  }
  a.wset += g_bfc_wset_distinct;
  a.wset_max = std::max(a.wset_max, g_bfc_wset_distinct);
  a.frags += b.frags;
  a.dw_ctl += b.dw_ctl;
  a.dw_other += b.dw_other;
  a.pdw_stateful += b.pdw_stateful;
  a.pdw_port += b.pdw_port;
  a.pdw_plain += b.pdw_plain;
  a.byref_dw += g_ruse_byref_cur.size();
}

// Chain the by-ref payload stream (order-sensitive: same payloads in the same
// order give the same digest at every draw prefix).
inline void NrRuseFeedByrefRange(uint32_t base, const uint32_t* values_be,
                                 uint32_t n) {
  const uint32_t head[2] = {base, n};
  const uint64_t h = XXH3_64bits_withSeed(head, sizeof(head), g_ruse_byref_h);
  g_ruse_byref_h =
      XXH3_64bits_withSeed(values_be, size_t(n) * sizeof(uint32_t), h);
}
inline void NrRuseFeedByrefDword(uint32_t reg, uint32_t value) {
  const uint32_t pair[2] = {reg, value};
  g_ruse_byref_h = XXH3_64bits_withSeed(pair, sizeof(pair), g_ruse_byref_h);
}

// v2: mark or clear one register's staleness from one applied write. The bin
// trio is never tracked (expected to differ per tile; the mechanism
// re-derives viewport/scissor/sys per execution), and neither are the
// non-stream regs (outside the 4c stream closure by definition).
inline void NrRuseStaleMark(uint32_t reg, bool changed) {
  if (reg >= RegisterFile::kRegisterCount) return;
  if (reg >= XE_GPU_REG_PA_SC_WINDOW_OFFSET &&
      reg <= XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR) {
    return;
  }
  for (uint32_t r : kNriNonStreamRegs) {
    if (r == reg) return;
  }
  if (changed) {
    if (!g_ruse_stale[reg]) {
      g_ruse_stale[reg] = 1;
      ++g_ruse_stale_blk[reg >> 6];
      g_ruse_stale_list.push_back(reg);
      ++g_ruse_stale_cnt;
      g_ruse_stale_sample = reg;
    }
  } else if (g_ruse_stale[reg]) {
    g_ruse_stale[reg] = 0;
    --g_ruse_stale_blk[reg >> 6];
    --g_ruse_stale_cnt;
  }
}

// [NR-RUSE-BLK] The `cnt` diff bits starting at buffer dword `dw`, low bits
// first. Caller guarantees dw + cnt <= g_ruse_dwords and cnt <= 64, so the
// out-of-range "reads as changed" rule of NrRuseDiffBit cannot apply.
inline uint64_t NrRuseDiffMaskN(uint32_t dw, uint32_t cnt) {
  if (g_ruse_diff_empty) return 0;
  const uint32_t w = dw >> 6, sh = dw & 63;
  uint64_t m = g_ruse_diff[w] >> sh;
  // The next word is needed only when the window actually reaches into it.
  // g_ruse_diff holds exactly (g_ruse_dwords + 63) / 64 words, so testing the
  // LAST dword's word index -- not `sh` alone -- is what keeps this in bounds
  // for a window ending in the final partial word.
  if (sh && ((dw + cnt - 1) >> 6) != w) {
    m |= g_ruse_diff[w + 1] << (64 - sh);
  }
  return cnt == 64 ? m : (m & ((uint64_t(1) << cnt) - 1));
}

// Changed-dword test against this replay's diff bitmap; out of range reads
// as changed (conservative).
inline bool NrRuseDiffBit(uint32_t dw) {
  if (dw >= g_ruse_dwords) return true;
  if (g_ruse_diff_empty) return false;
  return ((g_ruse_diff[dw >> 6] >> (dw & 63)) & 1ull) != 0;
}

// v2: one per-dword walk apply. The source offset is reconstructable for
// every shape the walker decodes per-dword (payload dwords arrive in order,
// so a per-packet cursor names each write's source dword); anything else is
// conservatively stale.
void NrRuseDword(uint32_t reg, uint32_t value, bool from_memory) {
  if (!g_ruse_cur_buf) return;
  if (from_memory) {
    const size_t idx = g_ruse_byref_cur.size();
    const std::vector<uint32_t>& prev = g_ruse_cur_buf->byref_prev;
    const bool changed =
        !g_ruse_v2_ok || idx >= prev.size() || prev[idx] != value;
    g_ruse_byref_cur.push_back(value);
    NrRuseStaleMark(reg, changed);
    return;
  }
  const uint32_t dw = g_ctx_walker.cur_dw;
  const uint32_t hdr = g_ctx_walker.cur_hdr;
  if (dw != g_ruse_pkt_dw) {
    g_ruse_pkt_dw = dw;
    g_ruse_pkt_m = 0;
  }
  const uint32_t m = g_ruse_pkt_m++;
  const uint32_t ty = hdr >> 30;
  uint32_t start;
  if (ty == 0 || ty == 1) {
    start = dw + 1;
  } else if (ty == 3) {
    const uint32_t op = (hdr >> 8) & 0x7F;
    if (op == 0x2D || op == 0x55 || op == 0x56 || op == 0x22) {
      start = dw + 2;  // typed/raw index dword (or the 0x22 viz token) first
    } else if (op == 0x36) {
      start = dw + 1;
    } else {
      ++g_ruse_w_pdw_cons;
      NrRuseStaleMark(reg, true);
      return;
    }
  } else {
    ++g_ruse_w_pdw_cons;
    NrRuseStaleMark(reg, true);
    return;
  }
  NrRuseStaleMark(reg, NrRuseDiffBit(start + m));
}

// v2: one bulk range apply (inline source inside the buffer, or by-ref).
void NrRuseRange(uint32_t base, const uint32_t* be, uint32_t n,
                 bool from_memory) {
  if (!g_ruse_cur_buf) return;
  if (from_memory) {
    const std::vector<uint32_t>& prev = g_ruse_cur_buf->byref_prev;
    for (uint32_t i = 0; i < n; ++i) {
      const size_t idx = g_ruse_byref_cur.size();
      const uint32_t v = be[i];
      const bool changed =
          !g_ruse_v2_ok || idx >= prev.size() || prev[idx] != v;
      g_ruse_byref_cur.push_back(v);
      NrRuseStaleMark(base + i, changed);
    }
    return;
  }
  const uint8_t* p = reinterpret_cast<const uint8_t*>(be);
  if (p < g_ruse_raw ||
      p + size_t(n) * 4 > g_ruse_raw + size_t(g_ruse_dwords) * 4) {
    // Inline ranges always point into the buffer; keep a conservative arm
    // anyway so a future walker change cannot silently break the model.
    for (uint32_t i = 0; i < n; ++i) NrRuseStaleMark(base + i, true);
    return;
  }
  const uint32_t o0 = uint32_t((p - g_ruse_raw) / 4);
  if (g_ruse_stale_cnt == 0) {
    // [NR-RUSE-EP] no diffs at all this replay: whole-span no-op, O(1).
    // (Also the guard that keeps the block scan below off a stale-sized
    // bitmap -- the epoch-clean path never allocates one.)
    if (g_ruse_diff_empty) return;
    // Fast path: nothing stale and nothing changed in the span => no effect.
    bool any = false;
    uint32_t i = 0;
    while (i < n && !any) {
      const uint32_t dw = o0 + i;
      if ((dw & 63) == 0 && i + 64 <= n && dw < g_ruse_dwords) {
        if (g_ruse_diff[dw >> 6]) {
          any = true;
        } else {
          i += 64;
        }
      } else if (NrRuseDiffBit(dw)) {
        any = true;
      } else {
        ++i;
      }
    }
    if (!any) return;
  }
  if (!g_nr_ruse_blk) {
    for (uint32_t i = 0; i < n; ++i) {
      NrRuseStaleMark(base + i, NrRuseDiffBit(o0 + i));
    }
    return;
  }
  // [NR-RUSE-BLK] Skip a window whole only when every NrRuseStaleMark it
  // would make is provably a no-op: no dword in it changed (so no register
  // can become stale) and no register in the 64-blocks it spans is currently
  // stale (so no register can be cleared). The excluded registers the mark
  // returns early on are covered for free -- skipping them is what the mark
  // does. The stale set after this loop is bit-identical either way.
  uint32_t i = 0;
  while (i < n) {
    const uint32_t reg = base + i;
    const uint32_t rem = n - i;
    const uint32_t blk = rem < 64 ? rem : 64;
    if (o0 + i + blk <= g_ruse_dwords &&
        reg + blk <= RegisterFile::kRegisterCount &&
        !g_ruse_stale_blk[reg >> 6] &&
        !g_ruse_stale_blk[(reg + blk - 1) >> 6] &&
        !NrRuseDiffMaskN(o0 + i, blk)) {
      g_ruse_blk_skip += blk;
      i += blk;
      continue;
    }
    ++g_ruse_blk_mark;
    NrRuseStaleMark(reg, NrRuseDiffBit(o0 + i));
    ++i;
  }
}

// One entry-state component: full diff scan (each differing register seeds
// the v2 stale set), shadow refreshed in place. Returns true when equal.
// `skip_lo`/`skip_n` cut the bin trio out of the ctl compare (it is compared
// separately and never tracked).
bool NrRuseEntryComponent(std::vector<uint32_t>& shad, const uint32_t* live,
                          uint32_t count, uint32_t base_reg, uint32_t skip_lo,
                          uint32_t skip_n, bool* had_prev, uint32_t* first_ne) {
  if (shad.size() != count) {
    shad.assign(live, live + count);
    *had_prev = false;
    return true;
  }
  bool eq = true;
  // u64-stride, drill on mismatch (the skip window only matters inside a
  // mismatching block; an equal block needs nothing).
  const uint64_t* a64 = reinterpret_cast<const uint64_t*>(live);
  const uint64_t* s64 = reinterpret_cast<const uint64_t*>(shad.data());
  const uint32_t words = count / 2;
  const auto drill = [&](uint32_t i) {
    if (skip_n && i >= skip_lo && i < skip_lo + skip_n) return;
    if (shad[i] != live[i]) {
      if (eq && first_ne && *first_ne == 0xFFFFFFFFu) *first_ne = base_reg + i;
      eq = false;
      NrRuseStaleMark(base_reg + i, true);
      shad[i] = live[i];
    }
  };
  for (uint32_t w = 0; w < words; ++w) {
    if (a64[w] == s64[w]) continue;
    drill(w * 2);
    drill(w * 2 + 1);
  }
  if (count & 1) drill(count - 1);
  return eq;
}

// Buffer entry: shadow-compare the bytes, compare the entry components,
// arm the per-replay classification flags. Runs only under the skip (the
// walk is the only decoder there, so `raw` is exactly what will be decoded).
void NrRuseBufEntry(uint32_t ptr, const uint8_t* raw, uint32_t count,
                    uint32_t swap_now, uint64_t epoch, bool epoch_live) {
  // [NR-RUSE] Probe-gated self-timer; see NrRuseDrawStop.
  const bool cost_timed = g_nr_ruse_v0;
  const auto t0 = cost_timed ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  ++g_ruse_w_bufs;
  g_ruse_byref_h = 0;
  g_ruse_have_prev = false;
  g_ruse_bytes_ident = false;
  g_ruse_first_diff = 0;
  // v2 per-replay reset: only the regs actually marked are cleared (a full
  // 20KB memset per replay was measurable at city rates).
  if (g_ruse_stale.size() != RegisterFile::kRegisterCount) {
    g_ruse_stale.assign(RegisterFile::kRegisterCount, 0);
    g_ruse_stale_blk.assign((RegisterFile::kRegisterCount + 63) / 64, 0);
  } else {
    // [NR-RUSE-BLK] the block counts follow the same sparse reset; a listed
    // register may already have been cleared, so decrement only on the
    // transition, exactly as the mark does.
    for (uint32_t r : g_ruse_stale_list) {
      if (g_ruse_stale[r]) {
        g_ruse_stale[r] = 0;
        --g_ruse_stale_blk[r >> 6];
      }
    }
  }
  g_ruse_stale_list.clear();
  g_ruse_stale_cnt = 0;
  if (g_ruse_draws_m.size() != (size_t(1) << kRuseDrawMapBits)) {
    g_ruse_draws_m.assign(size_t(1) << kRuseDrawMapBits, RuseDraw{});
  }
  // [NR-RUSE-EP] the diff bitmap is only allocated on the compare path;
  // until then this replay reads as diff-free (new buffers have no prev to
  // diff against, epoch-clean replays are diff-free by the gate).
  g_ruse_diff_empty = true;
  g_ruse_byref_cur.clear();
  g_ruse_pkt_dw = 0xFFFFFFFFu;
  g_ruse_pkt_m = 0;
  g_ruse_deleg_poison = false;
  g_ruse_raw = raw;
  g_ruse_dwords = count;
  g_ruse_v2_ok = false;
  if (g_ruse_shadow_bytes > kRuseShadowCap) {
    g_ruse_bufs.clear();
    for (RuseDraw& d : g_ruse_draws_m) d.used = 0;
    g_ruse_shadow_bytes = 0;
    ++g_ruse_w_evict;
  }
  RuseBuf& b = g_ruse_bufs[ptr & 0x1FFFFFFFu];
  g_ruse_cur_buf = &b;
  const size_t bytes = size_t(count) * sizeof(uint32_t);
  bool have_prev = b.dwords == count && b.shadow.size() == bytes;
  // [NR-RUSE-EP] epoch gate: same granule-epoch sum as the previous replay
  // of this buffer means no recorded packet touched its range since -- the
  // byte compare would find nothing. Trusted only when the epoch stream is
  // live (the guest recorder hook has ever bumped it; a dead hook reads
  // constant-zero and must not classify anything clean). Verify runs still
  // compare and count ep_ne (must stay 0).
  const bool ep_clean =
      have_prev && epoch_live && b.have_epoch && b.last_epoch == epoch;
  b.last_epoch = epoch;
  b.have_epoch = true;
  if (!have_prev) {
    g_ruse_shadow_bytes += bytes - b.shadow.size();
    b.shadow.assign(raw, raw + bytes);
    b.dwords = count;
    ++g_ruse_w_new;
    g_ruse_first_diff = 0;  // nothing comparable: no draw may claim a prefix
  } else if (ep_clean && g_nr_ruse_ep && !g_nr_verify_base) {
    // Compare skipped wholesale: bytes unchanged, shadow already equal, the
    // diff bitmap stays unallocated (g_ruse_diff_empty short-circuits every
    // reader).
    ++g_ruse_w_ep_clean;
    g_ruse_first_diff = count;
    g_ruse_bytes_ident = true;
    const bool sf = b.last_swap == swap_now;
    if (sf) ++g_ruse_w_sf; else ++g_ruse_w_xf;
    g_ruse_same_frame = sf;
    ++g_ruse_w_id;
    if (sf) ++g_ruse_w_sf_id; else ++g_ruse_w_xf_id;
  } else {
    // Changed-dword bitmap + first diff + count vs the previous replay.
    // u64-stride with drill-on-mismatch: the all-equal case (most replays)
    // is a straight vectorizable compare.
    g_ruse_diff.assign((count + 63) / 64, 0);
    g_ruse_diff_empty = false;
    uint32_t first = count, changed = 0;
    const uint32_t* a = reinterpret_cast<const uint32_t*>(raw);
    const uint32_t* s = reinterpret_cast<const uint32_t*>(b.shadow.data());
    const uint64_t* a64 = reinterpret_cast<const uint64_t*>(raw);
    const uint64_t* s64 = reinterpret_cast<const uint64_t*>(b.shadow.data());
    const uint32_t words = count / 2;
    for (uint32_t w = 0; w < words; ++w) {
      if (a64[w] == s64[w]) continue;
      for (uint32_t i = w * 2; i < w * 2 + 2; ++i) {
        if (a[i] != s[i]) {
          g_ruse_diff[i >> 6] |= 1ull << (i & 63);
          if (first == count) first = i;
          ++changed;
        }
      }
    }
    if ((count & 1) && a[count - 1] != s[count - 1]) {
      g_ruse_diff[(count - 1) >> 6] |= 1ull << ((count - 1) & 63);
      if (first == count) first = count - 1;
      ++changed;
    }
    g_ruse_first_diff = first;
    const bool sf = b.last_swap == swap_now;
    if (sf) ++g_ruse_w_sf; else ++g_ruse_w_xf;
    g_ruse_same_frame = sf;
    if (changed) {
      ++g_ruse_w_ch;
      g_ruse_w_chdw += changed;
      std::memcpy(b.shadow.data(), raw, bytes);
    } else {
      ++g_ruse_w_id;
      if (sf) ++g_ruse_w_sf_id; else ++g_ruse_w_xf_id;
      // Identical after a full compare: diff-free for the walk's readers
      // (same O(1) per-range no-op the epoch-clean path gets).
      g_ruse_diff_empty = true;
      g_ruse_bytes_ident = true;
    }
    // [NR-RUSE-EP] verify-run soundness gate: a clean-predicted replay with
    // changed bytes would have been a stale skip. Must stay 0.
    if (ep_clean) {
      ++g_ruse_w_ep_clean;
      if (changed) ++g_ruse_w_ep_ne;
    }
  }
  b.last_swap = swap_now;
  // Entry components, only meaningful once the replay file is live.
  g_ruse_entry_ne_mask = 0;
  g_ruse_bin_ne = false;
  g_ruse_entry_core_eq = false;
  bool comp_prev = false;
  if (g_nr_issue && g_nri_seeded) {
    const uint32_t* f = g_nri_file.values;
    comp_prev = b.have_entry;
    uint32_t ctl_ne = 0xFFFFFFFFu;
    // ctl 0x2000..0x2FFF with the bin trio (0x2080-0x2082) cut out.
    if (!NrRuseEntryComponent(b.e_ctl, f + 0x2000, 0x1000, 0x2000,
                              XE_GPU_REG_PA_SC_WINDOW_OFFSET - 0x2000, 3,
                              &comp_prev, &ctl_ne)) {
      g_ruse_entry_ne_mask |= 1u << 0;
      if (g_ruse_w_ctl_diff_reg == 0xFFFFFFFFu && ctl_ne != 0xFFFFFFFFu) {
        g_ruse_w_ctl_diff_reg = ctl_ne;
      }
    }
    if (!NrRuseEntryComponent(b.e_flt, f + 0x4000, 0x800, 0x4000, 0, 0,
                              &comp_prev, nullptr)) {
      g_ruse_entry_ne_mask |= 1u << 1;
    }
    if (!NrRuseEntryComponent(b.e_ftc, f + 0x4800, 0xC0, 0x4800, 0, 0,
                              &comp_prev, nullptr)) {
      g_ruse_entry_ne_mask |= 1u << 2;
    }
    if (!NrRuseEntryComponent(b.e_bl, f + 0x4900, 0x28, 0x4900, 0, 0,
                              &comp_prev, nullptr)) {
      g_ruse_entry_ne_mask |= 1u << 3;
    }
    const nr::ShaderRef& vs = g_ctx_state.vs;
    const nr::ShaderRef& ps = g_ctx_state.ps;
    if (vs.addr != b.e_vs.addr || vs.size_dwords != b.e_vs.size_dwords ||
        vs.valid != b.e_vs.valid || ps.addr != b.e_ps.addr ||
        ps.size_dwords != b.e_ps.size_dwords || ps.valid != b.e_ps.valid) {
      g_ruse_entry_ne_mask |= 1u << 4;
      b.e_vs = vs;
      b.e_ps = ps;
    }
    g_ruse_bin_ne = f[XE_GPU_REG_PA_SC_WINDOW_OFFSET] != b.e_bin[0] ||
                    f[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL] != b.e_bin[1] ||
                    f[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR] != b.e_bin[2];
    b.e_bin[0] = f[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
    b.e_bin[1] = f[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL];
    b.e_bin[2] = f[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR];
    g_ruse_entry_core_eq = comp_prev && g_ruse_entry_ne_mask == 0;
    if (comp_prev) {
      if (g_ruse_entry_ne_mask == 0) {
        ++g_ruse_w_entry_eq;
        if (g_ruse_bin_ne) ++g_ruse_w_entry_binonly;
      } else {
        for (uint32_t i = 0; i < 5; ++i) {
          if (g_ruse_entry_ne_mask & (1u << i)) ++g_ruse_w_entry_ne[i];
        }
      }
    }
    g_ruse_have_prev = have_prev && comp_prev;
    // v2: bytes + entry comparable is enough (entry diffs are IN the stale
    // set; shader identity is checked per draw). Entry ne does not block.
    g_ruse_v2_ok = have_prev && comp_prev;
    b.have_entry = true;
  } else {
    b.have_entry = false;
  }
  if (cost_timed) {
    g_ruse_w_cost_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
  }
}

// One draw stop: classify this execution against the draw's previous one,
// under both rules (v0 prefix, v2 stale set).
void NrRuseDrawStop(uint32_t ptr, const uint8_t* raw, uint32_t count,
                    uint32_t stop_dword) {
  // [NR-RUSE] The self-timer is a PROBE, and it used to run unconditionally:
  // two steady_clock reads (QueryPerformanceCounter on Windows) on every draw
  // of every buffer, in the default perf config, to fill a counter only the
  // probe's own report ever prints. At city draw rates that is ~500k clock
  // reads a second sitting inside the walk pool. Gated on the probe now, so
  // g_ruse_w_cost_ns reads 0 when the probe is off -- which is honest: with
  // the probe off, nothing is measuring.
  const bool cost_timed = g_nr_ruse_v0;
  const auto t0 = cost_timed ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  ++g_ruse_w_draws;
  const uint32_t hdr = __builtin_bswap32(
      *reinterpret_cast<const uint32_t*>(raw + size_t(stop_dword) * 4));
  const uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;
  uint32_t end = stop_dword + 1 + cnt;
  if (end > count) end = count;
  const uint32_t key = (ptr & 0x1FFFFFFFu) + stop_dword * 4;
  const uint64_t digest = g_ruse_byref_h;
  const nr::ShaderRef& vs = g_ctx_state.vs;
  const nr::ShaderRef& ps = g_ctx_state.ps;
  RuseDraw& d = *RuseDrawSlot(key);
  const bool key_prev = d.used && d.key == key;
  if (d.used && d.key != key) ++g_ruse_w_coll;
  bool byref_eq = false, sh_eq = false;
  if (key_prev) {
    byref_eq = d.digest == digest;
    sh_eq = d.vs_addr == vs.addr && d.vs_size == vs.size_dwords &&
            d.ps_addr == ps.addr && d.ps_size == ps.size_dwords;
  }
  d.key = key;
  d.used = 1;
  d.digest = digest;
  d.vs_addr = vs.addr;
  d.vs_size = vs.size_dwords;
  d.ps_addr = ps.addr;
  d.ps_size = ps.size_dwords;
  // v0: the prefix rule (sound floor). Reporting only; skipped when only
  // the fast path wants the verdict.
  if (g_nr_ruse_v0) {
    // [NR-BUFREPLAY] 60-lever candidate census (see the counters).
    if (g_ruse_have_prev && g_ruse_first_diff >= g_ruse_dwords &&
        g_ruse_same_frame && g_ruse_entry_core_eq) {
      ++g_ruse_w_bufid_draws;
      if (key_prev && byref_eq) ++g_ruse_w_bufid_byref;
    }
    if (!g_ruse_have_prev || !key_prev) {
      ++g_ruse_w_first;
    } else if (g_ruse_first_diff < end) {
      ++g_ruse_w_miss_pfx;
    } else if (!byref_eq) {
      ++g_ruse_w_miss_byref;
    } else if (!g_ruse_entry_core_eq) {
      for (uint32_t i = 0; i < 5; ++i) {
        if (g_ruse_entry_ne_mask & (1u << i)) {
          ++g_ruse_w_miss_entry[i];
          break;
        }
      }
    } else {
      ++g_ruse_w_reuse;
      if (g_ruse_same_frame) ++g_ruse_w_reuse_sf; else ++g_ruse_w_reuse_xf;
      if (g_ruse_bin_ne) ++g_ruse_w_reuse_binonly;
    }
  }
  // v2: the stale-register rule. The draw's own packet span still gets a
  // direct byte check (the 0x22 viz token is consumed but is not a register
  // write, so the stale set alone cannot see it change).
  bool reusable2 = false;
  bool stale_only = false;
  if (!g_ruse_v2_ok || !key_prev) {
    ++g_ruse_w_first2;
  } else if (g_ruse_deleg_poison) {
    ++g_ruse_w_miss2_deleg;
  } else {
    bool pkt_changed = false;
    if (!g_ruse_diff_empty) {
      for (uint32_t i = stop_dword; i < end && !pkt_changed; ++i) {
        pkt_changed = NrRuseDiffBit(i);
      }
    }
    if (pkt_changed) {
      ++g_ruse_w_miss2_pkt;
    } else if (g_ruse_stale_cnt) {
      ++g_ruse_w_miss2_stale;
      g_ruse_w_stale_sum += g_ruse_stale_cnt;
      // [NR-RUF-V2B] every other reuse condition holds; only the stale set
      // blocked. Shader equality is REQUIRED here: the upgrade consults the
      // active shaders' constant bitmaps, which the previous execution must
      // share for its stored bundle to be layout-identical.
      stale_only = sh_eq;
    } else if (!sh_eq) {
      ++g_ruse_w_miss2_sh;
    } else {
      reusable2 = true;
      ++g_ruse_w_reuse2;
      if (g_ruse_same_frame) ++g_ruse_w_reuse2_sf; else ++g_ruse_w_reuse2_xf;
    }
  }
  // [NR-RUB] publish the verdict for the backend's bundle gate: the issue of
  // this draw runs synchronously after this classification.
  g_ruse_stop_key = key;
  g_ruse_stop_r2 = reusable2;
  g_ruse_stop_sf = g_ruse_same_frame;
  g_ruse_stop_stale_only = stale_only;
  g_ruse_stop_valid = true;
  if (cost_timed) {
    g_ruse_w_cost_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
  }
}

}  // namespace-local helpers continue below

// [NR-RUB] 5-4-5-1: the backend's bundle gate asks for the pending draw's
// reuse verdict. Returns false when the probe is off or no stop is pending.
bool NrRuseCurrentDraw(uint32_t* key, bool* reusable2, bool* same_frame) {
  if (!g_nr_ruse || !g_ruse_stop_valid) return false;
  *key = g_ruse_stop_key;
  *reusable2 = g_ruse_stop_r2;
  *same_frame = g_ruse_stop_sf;
  return true;
}

// [NR-RUF-V2B] 5-4-5-2b: verdict + the stale-only miss flag. A stale-only
// miss is upgradeable by the backend when the stale set intersects nothing
// the draw's derivation reads (see NrRuseStaleRegs).
bool NrRuseCurrentDrawEx(uint32_t* key, bool* reusable2, bool* same_frame,
                         bool* stale_only) {
  if (!NrRuseCurrentDraw(key, reusable2, same_frame)) return false;
  *stale_only = g_ruse_stop_stale_only;
  return true;
}

// [NR-RUF-V2B] Copy the pending stop's CURRENT stale-register set (up to
// `max` entries) and return its true size; a return > max means the caller
// got a truncated copy and must refuse. Valid only between a draw stop's
// classification and the walk resuming -- the backend's issue runs there
// synchronously. g_ruse_stale_list accumulates every reg EVER marked this
// replay, so filter by the live flag.
uint32_t NrRuseStaleRegs(uint32_t* out, uint32_t max) {
  if (!g_nr_ruse || !g_ruse_stop_valid) return 0;
  uint32_t n = 0;
  for (uint32_t r : g_ruse_stale_list) {
    if (!g_ruse_stale[r]) continue;
    if (n < max) out[n] = r;
    ++n;
    if (n >= g_ruse_stale_cnt) break;  // all live entries found
  }
  return n;
}

namespace {

// Buffer end: this replay's by-ref stream becomes the previous one.
void NrRuseBufEnd() {
  if (!g_ruse_cur_buf) return;
  g_ruse_cur_buf->byref_prev.swap(g_ruse_byref_cur);
  g_ruse_cur_buf = nullptr;
  g_ruse_raw = nullptr;
  g_ruse_dwords = 0;
}

// [NR-RES]/[NR-DRAW] Every register write the ONE walk decodes, fanned out to
// whichever censuses are on. Both consume the same stream by design: a second
// decoder is the drift 4b-1 had to repair across four walkers.
void NrWalkRegWrite(void* user, uint32_t reg, uint32_t value, bool from_memory) {
  if (g_nr_res) {
    nr::ResApplyWrite(&g_res_state, &g_res_stats, reg, value, from_memory);
  }
  // [NR-VERIFY] inc 2: the shadow's only consumers are the per-draw sweep
  // (verify) and the one-time issue seed; an unfed shadow makes the seed a
  // pure live-file copy, which at a lockstep arm is the same values.
  if (g_nr_draw && g_nr_verify_base) {
    nr::RegShadowApplyWrite(&g_reg_shadow, &g_reg_stats, reg, value);
  }
  // [NR-ISSUE] Increment 4e: the replay file takes every decoded write as it
  // happens (execution order -- the walk is in lockstep). Same range rule as
  // the shadow: out-of-range is dropped, never clamped onto a real register.
  if (g_nr_issue && g_nri_seeded && !g_nri_live &&
      reg < RegisterFile::kRegisterCount) {
    g_nri_file.values[reg] = value;
  }
  // [NR-SKP] Phase 5-4-2: while the skip loop drives this buffer, the
  // executor's apply never runs, so the walk's decoded write goes through the
  // FULL virtual WriteRegister -- value store, dirty tail, the stateful
  // scratch/COHER/DC_LUT machines, and the instancing/dedupe semantics, all
  // identical to the executor's by construction. NrWalkWriteEffects is NOT
  // fired on top (WriteRegister's own tail already ran).
  if (g_nr_skip_bufactive && user) {
    ++g_skp_pdw;
    // [NR-4A] leak detector: a constant-window dword arriving per-dword
    // while a group is pending never reached the defer branch (measured 0;
    // if it ever fires, flush so ordering can never silently invert).
    if (g_nr4a_bufactive && g_nr4a_ndef && Nr4aComposable(reg, 1)) {
      // pdw_const stays a CONST-window leak detector (measured 0); state
      // per-dword writes here are EXPECTED (mirror-crossing runs, one-reg
      // packets) and by construction never overlap a deferred range -- the
      // loop below is the belt-and-braces order guard for both classes.
      if (!from_memory && Nr4aConstWindow(reg, 1)) ++g_4a_pdw_const;
      for (uint32_t i = 0; i < g_nr4a_ndef; ++i) {
        const Nr4aDef& d = g_nr4a_def[i];
        if (reg >= d.base && reg < d.base + d.n) {
          ++g_4a_byref_conflict;
          Nr4aFlush(static_cast<CommandProcessor*>(user));
          break;
        }
      }
    }
    N8Classify(reg, 1, true);
    // [NR-RUSE] the one-reg by-ref fallback must feed the digest too, or a
    // patched single-dword LOAD_ALU would classify as reusable; every
    // per-dword apply also drives the v2 stale set.
    if (g_nr_ruse) {
      if (g_nr_ruse_v0 && from_memory) NrRuseFeedByrefDword(reg, value);
      NrRuseDword(reg, value, from_memory);
    }
    // [NR-BFC] per-dword applies feed the census too (stateful ports and
    // odd shapes only ever arrive here).
    if (g_nr_bfc) NrBfcApply(reg, 1, true);
    static_cast<CommandProcessor*>(user)->NrSkipApplyRegWrite(reg, value);
    return;
  }
  // [NR-FX] Phase 5-4-0: additionally fire the backend's dirty-tracking tail
  // for this write. The walk is in lockstep (gpu_nr_walk_effects implies
  // gpu_nr_draw), so the executor has ALREADY applied this exact write and
  // fired the same effects between the same two draws; re-firing marks the
  // same bindings dirty before the same draw, which is what makes this a
  // strict no-op to validate. `user` is the command processor that began the
  // walk (reg_user at CtxWalkBegin).
  if (g_nr_walk_fx && user) {
    static_cast<CommandProcessor*>(user)->NrWalkWriteEffects(reg);
  }
}

// [NR-SKP] Phase 5-4-3: the walker's bulk range consumer, active ONLY while
// the skip loop drives the buffer (every other mode returns false and the
// walker's per-dword path runs bit-identically to a walk with no range_fn).
// Accepts a range only when it sits wholly inside ONE of the three pure
// constant windows -- exactly the windows the D3D12 WriteRegistersFromMem
// override bulk-handles with one dirty-tail evaluation per range. Everything
// else (the REGISTERS file, stateful ports, mixed ranges) refuses here and
// keeps the per-dword virtual WriteRegister, COHER's OR-write included.
bool NrWalkRegRange(void* user, uint32_t base, const uint32_t* values_be,
                    uint32_t n, uint32_t phys, bool from_memory) {
  if (!g_nr_skip_bufactive || !user || !n) return false;
  const uint32_t end = base + n - 1;
  if (!Nr4aConstWindow(base, n)) {
    // [NR-PB] N-2-2 item 0: a range of PLAIN registers -- inside the register
    // file, at or above 0x2000 (every stateful port is below), and wholly
    // outside the constant windows (a range that INTERSECTS one would need
    // that window's dirty tail for part of itself; full-fit inside was
    // matched above). NrSkipApplyRegRange routes it to the bulk plain store.
    // The walker has already refused mirror-window overlaps
    // (CtxRangeOfferable), so the 27 mirrored slots and the 4 watched
    // registers never arrive here.
    if (!(g_nr_plain_bulk && base >= 0x2000u &&
          end < RegisterFile::kRegisterCount &&
          (end < XE_GPU_REG_SHADER_CONSTANT_000_X ||
           base > XE_GPU_REG_SHADER_CONSTANT_LOOP_31))) {
      return false;
    }
    // [NR-4B-EXT] an inline STATE range inside the record's captured windows:
    // defer to the stop, same as a constant range (Nr4aStateWindow replicates
    // the walker's own offer test, so the record-side filter can never expect
    // a range that would not have arrived here).
    if (g_nr4a_bufactive && !from_memory && Nr4aStateWindow(base, n)) {
      if (g_nr4a_ndef < kNr4aMaxDef) {
        g_nr4a_def[g_nr4a_ndef++] = {base, n, phys, values_be};
        ++g_4a_def_ranges;
        ++g_4a_def_state;
        return true;
      }
      g_nr4a_group_ovf = true;
      ++g_4a_ovf_ranges;
    } else if (g_nr5_coh && g_nr4a_bufactive && from_memory &&
               Nr4aStateWindow(base, n)) {
      // [NR-5A] a composable by-ref into the state windows: defer with a
      // null be (the tag) so the stop can pair it with the record's marker
      // and apply it in the record's sequence, re-read from guest memory.
      if (g_nr4a_ndef < kNr4aMaxDef) {
        g_nr4a_def[g_nr4a_ndef++] = {base, n, phys, nullptr};
        ++g_4a_def_ranges;
        ++g_nr5_def_byref;
        return true;
      }
      g_nr4a_group_ovf = true;
      ++g_4a_ovf_ranges;
    } else if (g_nr4a_bufactive && from_memory && g_nr4a_ndef) {
      // A by-ref load into the REGISTERS window (0x2F type 4) about to apply
      // while ranges are pending: same stream-order guard as the constant
      // branch below -- flush before it applies, group falls back at the stop.
      for (uint32_t i = 0; i < g_nr4a_ndef; ++i) {
        const Nr4aDef& d = g_nr4a_def[i];
        if (base < d.base + d.n && d.base < base + n) {
          ++g_4a_byref_conflict;
          Nr4aFlush(static_cast<CommandProcessor*>(user));
          break;
        }
      }
    }
  } else if (g_nr4a_bufactive && !from_memory) {
    // [NR-4A] an inline constant-window range: defer to the group's stop,
    // where the record's composed runs replace it (or it flushes verbatim).
    if (g_nr4a_ndef < kNr4aMaxDef) {
      g_nr4a_def[g_nr4a_ndef++] = {base, n, phys, values_be};
      ++g_4a_def_ranges;
      return true;
    }
    g_nr4a_group_ovf = true;
    ++g_4a_ovf_ranges;
  } else if (g_nr4a_bufactive && from_memory) {
    // [NR-4B] witness every by-ref const-window range of the group (the
    // shape_ne sample prints them; reset at every stop/flush).
    if (g_4a_byref_n < 4) {
      g_4a_byref_rng[g_4a_byref_n * 2] = base;
      g_4a_byref_rng[g_4a_byref_n * 2 + 1] = n;
    }
    ++g_4a_byref_n;
    // [NR-5A] coherent mode defers the by-ref itself (null-be tag); the stop
    // pairs it with the record's marker and applies it in record order, so
    // the 663-class overlap case composes instead of conflict-flushing.
    if (g_nr5_coh) {
      if (g_nr4a_ndef < kNr4aMaxDef) {
        g_nr4a_def[g_nr4a_ndef++] = {base, n, phys, nullptr};
        ++g_4a_def_ranges;
        ++g_nr5_def_byref;
        return true;
      }
      g_nr4a_group_ovf = true;
      ++g_4a_ovf_ranges;
      return static_cast<CommandProcessor*>(user)->NrSkipApplyRegRange(
          base, values_be, n, phys, from_memory);
    }
    if (!g_nr4a_ndef) {
      return static_cast<CommandProcessor*>(user)->NrSkipApplyRegRange(
          base, values_be, n, phys, from_memory);
    }
    // [NR-4A] a by-ref load about to apply while inline ranges are pending.
    // At city ~130/s of these OVERLAP a pending range with the by-ref
    // POSITIONED AFTER it (naruto_663: reg 0x43F0x16 per-draw constants) --
    // stream order says the by-ref wins, so deferring the inline past it
    // (or re-applying composed values at the stop) overwrites the by-ref's
    // FRESH guest-memory values with the record-time snapshot. That was the
    // naruto_663 flashing-black-squares regression. Restore stream order
    // exactly: flush the pending inline ranges BEFORE the by-ref applies;
    // the truncated defer list then shape-mismatches at the stop and the
    // group falls back to packets, which is always correct.
    for (uint32_t i = 0; i < g_nr4a_ndef; ++i) {
      const Nr4aDef& d = g_nr4a_def[i];
      if (base < d.base + d.n && d.base < base + n) {
        ++g_4a_byref_conflict;
        Nr4aFlush(static_cast<CommandProcessor*>(user));
        break;
      }
    }
  }
  return static_cast<CommandProcessor*>(user)->NrSkipApplyRegRange(
      base, values_be, n, phys, from_memory);
}

// [NR-4A] Apply the deferred packet ranges in decode order (the fallback and
// the non-draw resolution). Identical to the walk having applied them at
// decode time, just later within the same group -- order-safe because the
// deferable classes have no in-group coupling (see the cvar comment).
void Nr4aFlush(CommandProcessor* cp) {
  for (uint32_t i = 0; i < g_nr4a_ndef; ++i) {
    const Nr4aDef& d = g_nr4a_def[i];
    // [NR-5A] a null be is a deferred by-ref (coherent mode): apply it from
    // guest memory at its own phys, exactly as the immediate path would --
    // the list is in stream order, so the flush replays the group verbatim.
    cp->NrSkipApplyRegRange(d.base, d.be, d.n, d.phys, d.be == nullptr);
  }
  g_nr4a_ndef = 0;
  g_nr4a_group_ovf = false;
}

// [NR-6] N-9-6: the repeat-band group resolution. Everything the mode says a
// band-2/3 draw does not need is DROPPED (never applied, never composed); the
// rest is applied in stream order exactly as Nr4aFlush would, so a partial
// mode leaves the register file in the same order-dependent state the full
// flush does. Classification is by (base,n) only, so a coherent-mode by-ref
// entry (be == nullptr) rides with the class it belongs to.
// [NR-6] ONE class filter for both paths. ⚠ naruto_725: the suppressed path
// first shipped as "skip the whole remaining apply", which is mode 3
// semantics no matter what mode was set -- so a mode-1 run was silently
// executing the class that strobes. A band draw must drop exactly what the
// packet path drops, no more.
inline bool Nr6DropRange(uint32_t base, uint32_t n, int32_t mode) {
  if (mode <= 0) return false;
  if (Nr4aConstWindow(base, n)) {
    const bool is_fetch = base >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
                          base + n - 1 <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5;
    // Fetch is its own mode step: its residency/texture invalidations are the
    // one class this project has never deduped ([N8B] rule), so it escalates
    // separately even though this draw never renders.
    return is_fetch ? mode >= 2 : mode >= 1;
  }
  return mode >= 3;  // 40040 state windows
}

void Nr4aBandDrop(CommandProcessor* cp, int32_t mode) {
  for (uint32_t i = 0; i < g_nr4a_ndef; ++i) {
    const Nr4aDef& d = g_nr4a_def[i];
    const bool drop = Nr6DropRange(d.base, d.n, mode);
    if (drop) {
      ++g_nr6_drop_rng;
      g_nr6_drop_dw += d.n;
      continue;
    }
    ++g_nr6_keep_rng;
    cp->NrSkipApplyRegRange(d.base, d.be, d.n, d.phys, d.be == nullptr);
  }
  g_nr4a_ndef = 0;
  g_nr4a_group_ovf = false;
}

// [NR-4A] The compose attempt at a draw stop. Succeeds ONLY when the record
// is present, carries a payload, survives the arena validity round trip, and
// its constant-window run shape matches the deferred list EXACTLY (same
// ranges, same order) -- then the record's values are applied through the
// same bulk path the packets would have taken, and the packets' payload is
// never read. Any disagreement is a counted fallback; correctness never
// depends on the record being right. Known window: a record one re-record
// behind its packets (the measured 0.27% post-hook torn class) can pass the
// shape match and apply the PREVIOUS recording's values for one execution;
// the next record write heals it.
bool Nr4aTryCompose(CommandProcessor* cp, uint32_t pkt_addr, const uint8_t* raw,
                    uint32_t buf_dwords, uint32_t stop_dword) {
  if (g_nr4a_group_ovf) {
    ++g_4a_fb_ovf;
    return false;
  }
  nr::DrawRecord rec;
  uint32_t ndw = 0;
  uint32_t nruns = 0;
  uint32_t f_reg[kNr4aMaxDef], f_cnt[kNr4aMaxDef], f_off[kNr4aMaxDef];
  // [NR-5A] coherent-mode entry attributes: a by-ref (marker-derived) entry
  // carries its guest memory address in f_phys and applies from memory.
  uint32_t f_phys[kNr4aMaxDef];
  uint8_t f_byref[kNr4aMaxDef];
  uint32_t draw_off = 0, draw_ndw = 0;
  bool draw_sup = false;
  bool draw_bytes_eq = false;
  uint32_t nf = 0, p = 2;
  // [NR-5C] apply-seq bracket: bump at entry and again at EVERY exit, so a
  // write outside this compose's own apply pass (walk per-dword between
  // stops, ring inline, delegates, executor fallback) always stamps the
  // epoch map strictly later than the applied_seq a record stores here --
  // a same-seq stamp can only be this compose's own apply.
  struct Nr5cSeqBracket {
    Nr5cSeqBracket() { ++g_nrb_apply_seq; }
    ~Nr5cSeqBracket() { ++g_nrb_apply_seq; }
  } nr5c_seq_bracket;
  // [NR-5C] lever (a): rb is the record body -- the arena block itself when
  // apply-from-arena is on (no scratch memcpy; VerifyState re-checked after
  // the parse and after the applies, a tear restarts once with force_copy),
  // the validated scratch copy otherwise.
  const uint32_t* rb = g_nr4a_scratch;
  bool force_copy = false;
  uint32_t arena_restarts = 0;
  bool apply_valid = true;
nr5c_restart:
  // [NR-5B-3] identity-retry loop. A suppressed record whose draw bytes
  // mismatch the kept packet is usually the SAME draw mid-re-record: the
  // store completes microseconds after the body writes the packet, and the
  // consumer racing that window used to lose the draw's constants for one
  // execution (naruto_693: argsne 0, foreign 17-196/s, samples = changed
  // counts at the same address). Spin ~a microsecond and re-fetch -- the
  // fresh record arrives and the group composes CURRENT. A record that
  // never converges (true recycling: a different draw now lives at the
  // address) exits foreign below. Same spin-on-the-producer pattern the
  // consumer already lives by at the ring.
  for (uint32_t att = 0;; ++att) {
    if (!nr::LookupDraw(pkt_addr, &rec)) {
      ++g_4a_fb_miss;
      return false;
    }
    if (!rec.state_gen) {
      ++g_4a_fb_nostate;
      return false;
    }
    ndw = 0;
    const uint32_t* body = nr::AcquireState(rec, &ndw);
    if (!body) {
      ++g_4a_fb_stale;
      return false;
    }
    if (g_nr5c_arena_on && !force_copy) {
      rb = body;
      ++g_nr5c_inplace;
    } else {
      std::memcpy(g_nr4a_scratch, body, size_t(ndw) * 4);
      rb = g_nr4a_scratch;
      ++g_nr5c_copies;
    }
    if (!nr::VerifyState(rec)) {
      ++g_4a_fb_torn;
      return false;
    }
    nruns = rb[0];
    nf = 0;
    p = 2;
    draw_off = 0;
    draw_ndw = 0;
    draw_sup = false;
    for (uint32_t k = 0; k < nruns; ++k) {
    if (p + 2 > ndw) {
      ++g_4a_fb_torn;
      return false;
    }
    const uint32_t reg = rb[p];
    const uint32_t cnt = rb[p + 1];
    if (!cnt || p + 2 + cnt > ndw) {
      ++g_4a_fb_torn;
      return false;
    }
    if (reg & 0x80000000u) {
      // [NR-5A] draw entry (bit31|bit30): the draw packet's raw dwords.
      // [NR-5B-3] tag bit 0 = the covered emission was SUPPRESSED at
      // record time: no packets to shape-match or verify against.
      if (reg & 0x40000000u) {
        draw_off = p + 2;
        draw_ndw = cnt;
        draw_sup = (reg & 1u) != 0;
        p += 2 + cnt;
        continue;
      }
      // [NR-4B-EXT] by-ref marker entry (tag bit 31, payload = the 0x2F's
      // own three dwords, raw guest order). Non-coherent: carried, not
      // composed (the walker applied the packet itself in stream order).
      // Coherent: decode exactly as the walker decodes the packet and pair
      // it with the deferred by-ref at this sequence position.
      ++g_4a_mk_runs;
      if (g_nr5_coh && cnt == 3 && p + 5 <= ndw) {
        const uint32_t d1 = __builtin_bswap32(rb[p + 2]);
        const uint32_t d2 = __builtin_bswap32(rb[p + 3]);
        const uint32_t d3 = __builtin_bswap32(rb[p + 4]);
        static constexpr uint32_t kBrefBase[5] = {0x4000u, 0x4800u, 0x4900u,
                                                  0x4908u, 0x2000u};
        const uint32_t btype = (d2 >> 16) & 0xFF;
        const uint32_t mcnt = d3 & 0xFFF;
        const uint32_t mreg =
            btype < 5 ? kBrefBase[btype] + (d2 & 0x7FF) : 0xFFFFFFFFu;
        if (btype < 5 && mcnt && Nr4aComposable(mreg, mcnt)) {
          if (nf >= kNr4aMaxDef) {
            ++g_4a_fb_shape;
            return false;
          }
          f_reg[nf] = mreg;
          f_cnt[nf] = mcnt;
          f_off[nf] = 0;
          f_phys[nf] = d1 & 0x1FFFFFFFu;
          f_byref[nf] = 1;
          ++nf;
        } else {
          // Out-of-scope decode: its packet took the per-dword path and was
          // never deferred; skip it in the sequence match.
          ++g_nr5_mk_skipped;
        }
      }
      p += 2 + cnt;
      continue;
    }
    if (Nr4aComposable(reg, cnt)) {
      if (nf >= kNr4aMaxDef) {
        ++g_4a_fb_shape;
        return false;
      }
      f_reg[nf] = reg;
      f_cnt[nf] = cnt;
      f_off[nf] = p + 2;
      f_phys[nf] = 0;
      f_byref[nf] = 0;
      ++nf;
    }
    p += 2 + cnt;
  }
  if (p != ndw) {
    ++g_4a_fb_torn;
    return false;
  }
  // [NR-5C] the in-place parse read live arena memory: re-validate before
  // anything derived from it is used. A tear restarts once with a forced
  // scratch copy (no applies have happened yet).
  if (rb != g_nr4a_scratch && !nr::VerifyState(rec)) {
    ++g_nr5c_tear;
    if (arena_restarts++ == 0) {
      ++g_nr5c_restart;
      force_copy = true;
      goto nr5c_restart;
    }
    ++g_4a_fb_torn;
    return false;
  }
  if (!draw_sup) break;
  draw_bytes_eq =
      draw_ndw != 0 && stop_dword + draw_ndw <= buf_dwords &&
      __builtin_memcmp(rb + draw_off, raw + stop_dword * 4,
                       size_t(draw_ndw) * 4) == 0;
  if (draw_bytes_eq || att >= 3) break;
  for (volatile uint32_t spin = 0; spin < 600; ++spin) {
  }
  }  // identity-retry loop
  // [NR-5C-ATTR] the compose bracket's bin for THIS stop, known as soon as the
  // record parsed. A compose that declines before the parse (miss/torn/ovf)
  // stays in the pkt bin -- those classes are counted by name in [nr-4a], so
  // the mis-binned time is bounded by a printed rate, not a guess.
  g_nr5c_cur_sup = draw_sup;
  // [NR-5C] lever (b) eligibility, shared by both apply paths below: the
  // record's payload generation is unchanged since its last full apply on
  // this thread. Per-run cleanliness (bucket epochs) is checked at each run.
  const bool rec_clean = g_nr5c_skip_mode > 0 && rec.applied_gen != 0 &&
                         rec.applied_gen == rec.state_gen;
  if (g_nr5c_skip_mode > 0 && !rec_clean) ++g_nr5c_genmiss;
  // [NR-5B-3] SUPPRESSED group: the covered runs exist ONLY in the record.
  // No shape match (there is no deferred list to match: the walker saw only
  // the kept classes), no value verify (no packet bytes). Flush the kept
  // deferred ranges in stream order, then apply the whole record in its
  // recorded order -- runs from the payload, by-ref markers re-read from
  // guest memory -- and issue the draw from the record. A one-generation-
  // behind record here is the COHERENT previous draw by 5a's construction.
  if (draw_sup) {
    g_nr5b_sup_live = true;
    // [NR-5B-3k] Is this record NEWER than the buffer being replayed? A
    // positive lead means the guest re-recorded this draw in place after the
    // execution started, so its constants belong to a later frame. Signed
    // difference: both are wrap-around u32 recording counters.
    {
      const int32_t lead = int32_t(rec.seq - g_nr5b_exec_seq0);
      const int32_t lag = -lead;
      g_nr5b_lagsum += uint64_t(uint32_t(lag));
      if (lag < g_nr5b_lagmin) g_nr5b_lagmin = lag;
      if (lag > g_nr5b_lagmax) g_nr5b_lagmax = lag;
      ++g_nr5b_gen_seen;
      if (lead > 0) {
        ++g_nr5b_gen_ahead;
        g_nr5b_gen_leadsum += uint64_t(lead);
        if (lead > g_nr5b_gen_maxlead) g_nr5b_gen_maxlead = lead;
        g_nr5b_gen_skip = g_nr5b_genmode == 1;
      }
    }
    if (!g_nr5_coh || !draw_ndw) {
      ++g_nr5b_sup_refuse;
      return false;
    }
    // Identity guard: after the retries, a record whose draw bytes still
    // mismatch is either the same draw with stale bytes (prim and count
    // match -- stale constants are benign) or a FOREIGN record left at a
    // recycled ring address -- applying a foreign record's constants
    // painted the naruto_691 whole-screen flashing. Foreign => a miss.
    if (!draw_bytes_eq) {
      uint32_t pkt_prim = 0xFFFFFFFFu, pkt_cnt = 0xFFFFFFFFu;
      if (stop_dword + 2 < buf_dwords) {
        const uint32_t init = __builtin_bswap32(
            *reinterpret_cast<const uint32_t*>(raw + (stop_dword + 2) * 4));
        pkt_prim = init & 0x3Fu;
        pkt_cnt = init >> 16;
      }
      if ((rec.prim & 0x3Fu) != pkt_prim || rec.count != pkt_cnt) {
        ++g_nr5b_sup_foreign;
        if (!g_nr5b_foreign_samp[0]) {
          g_nr5b_foreign_samp[1] = pkt_addr;
          g_nr5b_foreign_samp[2] = (rec.prim << 16) | (pkt_prim & 0xFFFFu);
          g_nr5b_foreign_samp[3] = rec.count;
          g_nr5b_foreign_samp[4] = pkt_cnt;
          g_nr5b_foreign_samp[0] = 1;
        }
        return false;
      }
    }
    // [NR-5B-3i] The group's LIVE by-refs. F88 is never shimmed, so the real
    // 0x2F packets are still in the ring and the walker deferred the
    // composable ones (null be) in stream order. Snapshot them before the
    // flush clears the list: they are the FRESH source for exactly the
    // constants the record's markers also name, and the record's copy of the
    // address can be a recording behind.
    struct { uint32_t base, n, phys; } live_br[kNr4aMaxDef];
    uint32_t nlive = 0, live_i = 0;
    for (uint32_t i = 0; i < g_nr4a_ndef; ++i) {
      if (g_nr4a_def[i].be == nullptr) {
        live_br[nlive].base = g_nr4a_def[i].base;
        live_br[nlive].n = g_nr4a_def[i].n;
        live_br[nlive].phys = g_nr4a_def[i].phys;
        ++nlive;
      }
    }
    if (g_nr5b_mk_mode == 2) {
      // Mode 2 holds the by-refs back so each applies at ITS marker's
      // recorded position, interleaved with the record's runs; only the
      // packet-sourced (inline) deferred ranges flush here.
      for (uint32_t i = 0; i < g_nr4a_ndef; ++i) {
        const Nr4aDef& d = g_nr4a_def[i];
        if (d.be) cp->NrSkipApplyRegRange(d.base, d.be, d.n, d.phys, false);
      }
      g_nr4a_ndef = 0;
      g_nr4a_group_ovf = false;
    } else {
        Nr4aFlush(cp);
    }
    // [NR-5B-3l] order census, sampled. f_[0..nf) is this record's composable
    // run list in RECORD order (markers carry no inline payload and are
    // skipped: their source was settled by gpu_nr_sup_marker). Two inline
    // runs whose register ranges intersect are order-sensitive by definition,
    // and if their overlapping dwords also differ, the order the record
    // replays them in decides what the draw sees.
    if ((g_nr5b_ovl_ctr++ & 255u) == 0u) {
      ++g_nr5b_ovl_samples;
      for (uint32_t i = 0; i < nf; ++i) {
        if (f_byref[i]) continue;
        for (uint32_t j = i + 1; j < nf; ++j) {
          if (f_byref[j]) continue;
          const uint32_t lo = f_reg[i] > f_reg[j] ? f_reg[i] : f_reg[j];
          const uint32_t hi_i = f_reg[i] + f_cnt[i];
          const uint32_t hi_j = f_reg[j] + f_cnt[j];
          const uint32_t hi = hi_i < hi_j ? hi_i : hi_j;
          if (lo >= hi) continue;
          ++g_nr5b_ovl_pairs;
          bool ne = false;
          for (uint32_t r = lo; r < hi && !ne; ++r) {
            ne = rb[f_off[i] + (r - f_reg[i])] !=
                 rb[f_off[j] + (r - f_reg[j])];
          }
          if (ne) {
            ++g_nr5b_ovl_valne;
            if (!g_nr5b_ovl_samp[0]) {
              g_nr5b_ovl_samp[0] = 1;
              g_nr5b_ovl_samp[1] = f_reg[i];
              g_nr5b_ovl_samp[2] = f_cnt[i];
              g_nr5b_ovl_samp[3] = f_reg[j];
              g_nr5b_ovl_samp[4] = f_cnt[j];
            }
          }
        }
      }
    }
    // [NR-6] ⚠⚠ THE SUPPRESSION ORDERING TRAP (naruto_724, band-shaped
    // glitch in the middle third of the screen). Under suppression
    // RB_SURFACE_INFO / RB_COLOR_INFO / RB_DEPTH_INFO (0x2000..0x2004) are
    // MIRROR class, so they are not deferrable and arrive FROM THIS RECORD,
    // inside this compose. The band predicate hoisted above the compose
    // therefore read the PREVIOUS draw's surface signature and misclassified
    // ~24 draws/s (`pred_ne` 6-25/s here, 0 in every packet-path window):
    // their constants were dropped AND the draw was skipped, which is a
    // band's worth of missing draws.
    bool nr6_band_drop = false;
    // The fix is ordering, not a retreat: apply the mirror/state per-dword
    // class FIRST, then re-test the band on CURRENT registers. The pre-pass
    // is idempotent (the full loop below re-applies the same values in
    // record order, and mirror-class and composable ranges are disjoint by
    // construction) and small (~27k dwords/s at city against 32M).
    // The six registers the predicate reads -- RB_SURFACE_INFO(0x2000),
    // RB_COLOR_INFO(0x2001), RB_DEPTH_INFO(0x2004), PA_SC_WINDOW_OFFSET
    // (0x2080) and both scissors (0x2081/2) -- are ALL inside the recovery
    // mirror ({0x2000,6} RT + {0x2080,3} scissor in nr_context's kRanges), so
    // a composable range can never touch them (Nr4aStateWindow rejects any
    // mirror-crossing run outright). That is what makes this pre-pass
    // complete: the ONLY record items that can move the predicate are
    // non-composable runs and non-composable by-ref MARKERS, and both are
    // replayed here. Missing the markers is a silent misclassification that
    // pred_ne cannot catch, because by then both predicate reads sit on the
    // same incomplete state.
    if (g_nr6_mode > 0 && g_nr_detile_mode) {
      uint32_t s = 2;
      for (uint32_t k = 0; k < nruns; ++k) {
        const uint32_t sreg = rb[s];
        const uint32_t scnt = rb[s + 1];
        if (sreg & 0x80000000u) {
          // A by-ref marker. Mirror-class ones (btype 4 => 0x2000 base) apply
          // per dword FROM GUEST MEMORY in the main loop whenever the marker
          // mode leaves them in (mode 0); modes 1 and 2 drop them precisely
          // because the walker already applied them at parse, so the register
          // file is current either way and only mode 0 needs the replay.
          if (!(sreg & 0x40000000u) && scnt == 3 && g_nr5b_mk_mode == 0) {
            const uint32_t d2 = __builtin_bswap32(rb[s + 3]);
            const uint32_t d3 = __builtin_bswap32(rb[s + 4]);
            static constexpr uint32_t kPreBase[5] = {0x4000u, 0x4800u, 0x4900u,
                                                     0x4908u, 0x2000u};
            const uint32_t btype = (d2 >> 16) & 0xFF;
            const uint32_t mcnt = d3 & 0xFFF;
            if (btype == 4 && mcnt) {
              const uint32_t mreg = kPreBase[4] + (d2 & 0x7FF);
              if (mreg >= 0x2000u && !Nr4aComposable(mreg, mcnt)) {
                const uint32_t maddr = __builtin_bswap32(rb[s + 2]) & 0x1FFFFFFFu;
                const uint32_t* src = cp->NrTranslatePhys(maddr);
                if (src) {
                  for (uint32_t m = 0; m < mcnt; ++m) {
                    nr::CtxExternalWrite(&g_ctx_walker, mreg + m,
                                         __builtin_bswap32(src[m]), true);
                  }
                  ++g_nr6_pre_mk;
                }
              }
            }
          }
          s += 2 + scnt;
          continue;
        }
        if (sreg >= 0x2000u && !Nr4aComposable(sreg, scnt)) {
          for (uint32_t m = 0; m < scnt; ++m) {
            nr::CtxExternalWrite(&g_ctx_walker, sreg + m,
                                 __builtin_bswap32(rb[s + 2 + m]), false);
          }
          g_nr6_pre_dw += scnt;
        }
        s += 2 + scnt;
      }
      const nr::DetileRegs n6r{
          cp->NrPeekReg(XE_GPU_REG_PA_SC_WINDOW_OFFSET),
          cp->NrPeekReg(XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL),
          cp->NrPeekReg(XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR),
          cp->NrPeekReg(XE_GPU_REG_RB_SURFACE_INFO),
          cp->NrPeekReg(XE_GPU_REG_RB_COLOR_INFO),
          cp->NrPeekReg(XE_GPU_REG_RB_DEPTH_INFO)};
      if (nr::DetileIsRepeatBand(n6r)) {
        // A repeat band, decided on state this compose just made current.
        // The body still runs: only the classes Nr6DropRange names are
        // skipped, exactly as on the packet path. ⚠ Do NOT skip the whole
        // body (naruto_725): state and fetch runs a later KEPT draw inherits
        // live in this record too, and under suppression the record is their
        // ONLY source -- the guest's dirty tracking will not re-emit them.
        // The draw itself is left to the dispatch site's predicate, which now
        // reads the same registers and drops it too.
        ++g_nr6_late;
        nr6_band_drop = true;
      }
    }
    // Apply the WHOLE body in record order, per class exactly as the packet
    // world applied it: Composable runs through the bulk range path, and
    // mirror/watched-class runs (>= 0x2000, not Composable) PER DWORD
    // through the walk's own write path (CtxExternalWrite: reg_fn + context
    // mirror + watch). Dropping those is what blinded de-tile in naruto_685
    // (scissor/surface-info never applied, repeat bands invisible, dpf
    // 92 -> 155). Below-0x2000 runs (the 40100 preamble synths) stay
    // skipped: the kept ring bytes still carry them, and COHER is stateful.
    uint32_t q = 2;
    for (uint32_t k = 0; k < nruns; ++k) {
      const uint32_t reg = rb[q];
      const uint32_t cnt = rb[q + 1];
      if (reg & 0x80000000u) {
        if (!(reg & 0x40000000u) && cnt == 3) {
          // by-ref marker: re-read guest memory (address per mk_mode below).
          const uint32_t d1 = __builtin_bswap32(rb[q + 2]);
          const uint32_t d2 = __builtin_bswap32(rb[q + 3]);
          const uint32_t d3 = __builtin_bswap32(rb[q + 4]);
          static constexpr uint32_t kBrefBase[5] = {0x4000u, 0x4800u, 0x4900u,
                                                    0x4908u, 0x2000u};
          const uint32_t btype = (d2 >> 16) & 0xFF;
          uint32_t mcnt = d3 & 0xFFF;
          if (btype < 5 && mcnt) {
            uint32_t mreg = kBrefBase[btype] + (d2 & 0x7FF);
            uint32_t maddr = d1 & 0x1FFFFFFFu;
            // [NR-5B-3i] pick the by-ref's SOURCE. A composable marker pairs,
            // in sequence, with the live deferred by-ref the walker took from
            // this group's own 0x2F packets; a non-composable one has no
            // deferred twin (the walker applied it per-dword the moment it
            // parsed the packet), so under any non-zero mode it is pure
            // duplication. The address diff is counted in every mode -- it is
            // the numeric form of the stale-marker hypothesis.
            // Pair on {reg,count}, not on position alone: a drifted sequence
            // must not hand a marker some other constant's address. The scan
            // is forward-only and bounded by the group's own by-ref list
            // (~14 at city), so it stays O(group).
            // [NR-6] band draw: a marker naming a dropped class is dropped
            // with it (the pre-pass above already replayed the mirror-class
            // markers, which are never in a droppable class).
            if (nr6_band_drop && Nr6DropRange(mreg, mcnt, g_nr6_mode)) {
              ++g_nr6_drop_rng;
              g_nr6_drop_dw += mcnt;
              q += 2 + cnt;
              continue;
            }
            const bool composable = Nr4aComposable(mreg, mcnt);
            bool paired = false;
            if (composable) {
              for (uint32_t j = live_i; j < nlive; ++j) {
                if (live_br[j].base != mreg || live_br[j].n != mcnt) continue;
                ++g_nr5b_mk_paired;
                if (live_br[j].phys != maddr) ++g_nr5b_mk_addr_ne;
                if (g_nr5b_mk_mode == 2) {
                  maddr = live_br[j].phys;
                  ++g_nr5b_mk_live;
                }
                live_i = j + 1;
                paired = true;
                break;
              }
              if (!paired) ++g_nr5b_mk_unpaired;
            }
            // Mode 1 drops every marker whose live twin exists; mode 2
            // drops only the per-dword class (already applied at parse) and
            // re-sources the rest. An UNPAIRED composable marker has no live
            // source at all, so it still applies from the record in both
            // modes -- dropping it would delete the constant outright.
            if ((g_nr5b_mk_mode == 1 && (paired || !composable)) ||
                (g_nr5b_mk_mode == 2 && !composable)) {
              // Redundant: the live packet already applied (at the flush for a
              // composable range, immediately for a per-dword one).
              ++g_nr5b_mk_drop;
              q += 2 + cnt;
              continue;
            }
            if (g_nr5b_mk_mode == 0) ++g_nr5b_mk_stale;
            if (composable) {
              cp->NrSkipApplyRegRange(mreg, nullptr, mcnt, maddr, true);
              ++g_nr5_mk_applied;
            } else if (mreg >= 0x2000u) {
              const uint32_t* src = cp->NrTranslatePhys(maddr);
              if (src) {
                for (uint32_t m = 0; m < mcnt; ++m) {
                  nr::CtxExternalWrite(&g_ctx_walker, mreg + m,
                                       __builtin_bswap32(src[m]), true);
                }
                ++g_nr5_mk_applied;
              }
            }
            g_4a_comp_dw += mcnt;
          }
        }
        q += 2 + cnt;
        continue;
      }
      if (Nr4aComposable(reg, cnt)) {
        // [NR-6] this draw is a repeat band the dispatch will drop: the
        // classes the mode names never need to reach the register file.
        if (nr6_band_drop && Nr6DropRange(reg, cnt, g_nr6_mode)) {
          ++g_nr6_drop_rng;
          g_nr6_drop_dw += cnt;
          q += 2 + cnt;
          continue;
        }
        // [NR-5C] lever (b): a gen-stable record run whose buckets nobody
        // touched since its last apply is already in the register file.
        bool skip = false;
        if (rec_clean && Nr5cSkippableInline(reg, cnt)) {
          if (NrbRangeClean(reg, cnt, rec.applied_seq)) {
            skip = true;
            ++g_nr5c_skip_rng;
            g_nr5c_skip_dw += cnt;
          } else {
            ++g_nr5c_dirty_rng;
          }
        }
        if (!skip) {
          cp->NrSkipApplyRegRange(reg, rb + q + 2, cnt, 0, false);
        }
      } else if (reg >= 0x2000u) {
        if (rec_clean && g_nr5c_skip_mode >= 2 &&
            NrbRangeClean(reg, cnt, rec.applied_seq)) {
          // mirror/state class: register file and walker mirror move
          // together through the stamped chokepoints, so clean means both
          // already hold these values.
          ++g_nr5c_skip_pdw_rng;
          g_nr5c_skip_pdw_dw += cnt;
        } else {
          for (uint32_t m = 0; m < cnt; ++m) {
            nr::CtxExternalWrite(&g_ctx_walker, reg + m,
                                 __builtin_bswap32(rb[q + 2 + m]), false);
          }
          g_nr5b_pdw_dw += cnt;
        }
      }
      g_4a_comp_dw += cnt;
      q += 2 + cnt;
    }
    g_4a_comp_runs += nf;
    // [NR-5C] the draw entry must outlive the compose (the stop issues from
    // it), so copy it out of the arena; then re-validate the whole in-place
    // apply pass. A tear restarts once with a forced scratch copy -- the
    // stable pass re-applies over whatever the torn pass wrote.
    if (rb != g_nr4a_scratch) {
      if (draw_bytes_eq && draw_ndw) {
        std::memcpy(g_nr4a_scratch, rb + draw_off, size_t(draw_ndw) * 4);
      }
      if (!nr::VerifyState(rec)) {
        ++g_nr5c_tear;
        if (arena_restarts++ == 0) {
          ++g_nr5c_restart;
          force_copy = true;
          goto nr5c_restart;
        }
        apply_valid = false;   // never stamp a torn pass as applied
        draw_bytes_eq = false; // and never issue the draw from torn bytes
      }
    }
    // The draw issues from the record ONLY when the record's draw bytes
    // equal the kept packet's. On a torn (one-behind, same-draw: identity
    // guard above) record the PACKET is the current truth: an old
    // VGT_DMA_BASE points at per-frame-recycled index memory -- garbage
    // geometry ([[h3-overlap-live-guest-memory]]). Stale-by-one CONSTANTS
    // are visually benign; stale POINTERS are not.
    if (draw_bytes_eq) {
      // in-place pass copied the draw entry to scratch[0]; copy pass left
      // the whole body in scratch.
      g_nr5_draw_src = rb != g_nr4a_scratch ? g_nr4a_scratch
                                            : g_nr4a_scratch + draw_off;
      g_nr5_draw_ndw = draw_ndw;
    } else {
      ++g_nr5b_sup_argsne;  // draw issues from the current packet bytes
    }
    // [NR-6] a band draw applied only PART of the record, so it must never
    // claim the record is in the register file (lever (b) would then skip a
    // later kept draw's real applies).
    if (g_nrb_track && apply_valid && !nr6_band_drop) {
      nr::StampApplied(pkt_addr, rec.state_gen, g_nrb_apply_seq);
      ++g_nr5c_stamped;
    }
    ++g_4a_composed;
    ++g_nr5_composed;
    ++g_nr5b_sup_composed;
    return true;
  }
  const auto shape_fail = [&](uint32_t idx) {
    ++g_4a_fb_shape;
    if (!g_4a_shape_samp[0]) {
      g_4a_shape_samp[0] = 1;
      g_4a_shape_samp[1] = nf;
      g_4a_shape_samp[2] = g_nr4a_ndef;
      g_4a_shape_samp[3] = idx;
      g_4a_shape_samp[4] = idx < nf ? f_reg[idx] : 0xFFFFFFFFu;
      g_4a_shape_samp[5] = idx < nf ? f_cnt[idx] : 0xFFFFFFFFu;
      g_4a_shape_samp[6] =
          idx < g_nr4a_ndef ? g_nr4a_def[idx].base : 0xFFFFFFFFu;
      g_4a_shape_samp[7] = idx < g_nr4a_ndef ? g_nr4a_def[idx].n : 0xFFFFFFFFu;
      for (uint32_t s = 0; s < 8; ++s) {
        g_4a_shape_rec[s * 2] = s < nf ? f_reg[s] : 0;
        g_4a_shape_rec[s * 2 + 1] = s < nf ? f_cnt[s] : 0;
        g_4a_shape_def[s * 2] = s < g_nr4a_ndef ? g_nr4a_def[s].base : 0;
        g_4a_shape_def[s * 2 + 1] = s < g_nr4a_ndef ? g_nr4a_def[s].n : 0;
      }
      // [NR-4B] what the group applied BY-REF before this stop.
      g_4a_shape_byref[0] = g_4a_byref_n;
      for (uint32_t s = 0; s < 8; ++s) {
        g_4a_shape_byref[1 + s] = s < 8 ? g_4a_byref_rng[s] : 0;
      }
      // [NR-4B] segment-split witness: the record's own body extent
      // (pre_span, bytes from body start to the 0x22) vs where this stop
      // and the first deferred payload actually sit. A pre_span far larger
      // than (stop - def0_phys) means the body began in a PREVIOUS ring
      // segment and this group is only its tail.
      g_4a_shape_span[0] = rb[1];  // record pre_span
      g_4a_shape_span[1] = pkt_addr;
      g_4a_shape_span[2] = g_nr4a_ndef ? g_nr4a_def[0].phys : 0;
    }
  };
  if (nf != g_nr4a_ndef) {
    shape_fail(nf < g_nr4a_ndef ? nf : g_nr4a_ndef);
    return false;
  }
  for (uint32_t i = 0; i < nf; ++i) {
    if (f_reg[i] != g_nr4a_def[i].base || f_cnt[i] != g_nr4a_def[i].n) {
      shape_fail(i);
      return false;
    }
    // [NR-5A] kind + address must agree too: a marker pairs only with a
    // deferred by-ref (null be) at the same guest address, an inline run
    // only with a deferred packet range. A mismatch is a marker-sequence
    // divergence (the torn/segmentation classes), counted separately.
    const bool def_byref = g_nr4a_def[i].be == nullptr;
    if (def_byref != (f_byref[i] != 0) ||
        (def_byref && f_phys[i] != g_nr4a_def[i].phys)) {
      ++g_nr5_fb_mk;
      return false;
    }
  }
  // [NR-5A] the draw entry is the coherence anchor: the composed draw's own
  // packet dwords must equal the executed packet's bytes, or the record is
  // one re-record behind (the torn window) and the whole group falls back.
  // Runs unconditionally in coherent mode (it is ~24 bytes and it is the
  // check that makes single-source issue safe).
  if (g_nr5_coh) {
    if (!draw_ndw) {
      ++g_nr5_fb_nodraw;
      return false;
    }
    if (stop_dword + draw_ndw > buf_dwords ||
        __builtin_memcmp(rb + draw_off, raw + stop_dword * 4,
                         size_t(draw_ndw) * 4) != 0) {
      ++g_nr5_fb_args;
      return false;
    }
  }
  // Value-verify: the composed bytes must equal the packet bytes, or the
  // record is one re-record behind its packets (the post-hook torn window)
  // and applying it would put PREVIOUS-recording constants on this draw.
  // With this check 4a is incapable of diverging from packet execution;
  // N-9-5 removes the check together with the packets.
  // [NR-4B] the check is rate-gated so its removal can be priced in place
  // (gpu_nr_rec_verify); an unverified group skips the ONLY payload read.
  bool verify = true;
  const int32_t vn = g_nr4b_verify_n;
  if (vn == 0) {
    verify = false;
  } else if (vn > 1) {
    verify = (g_nr4b_vctr++ % uint32_t(vn)) == 0;
  }
  if (verify) {
    ++g_4a_ver;
    for (uint32_t i = 0; i < nf; ++i) {
      // [NR-5A] a by-ref pair has no packet payload to compare (both sides
      // read the same guest memory); its verify is the {reg,cnt,phys} match
      // above. An inline run without packet bytes can only be a kind
      // mismatch, already refused there in coherent mode.
      if (f_byref[i]) continue;
      if (!g_nr4a_def[i].be ||
          __builtin_memcmp(rb + f_off[i], g_nr4a_def[i].be,
                           size_t(f_cnt[i]) * 4) != 0) {
        ++g_4a_fb_val;
        return false;
      }
    }
  } else {
    ++g_4a_unver;
  }
  for (uint32_t i = 0; i < nf; ++i) {
    if (f_byref[i]) {
      // [NR-5A] apply the by-ref from guest memory at the marker's address,
      // at its recorded position in the group's sequence.
      cp->NrSkipApplyRegRange(f_reg[i], nullptr, f_cnt[i], f_phys[i], true);
      ++g_nr5_mk_applied;
    } else if (rec_clean && Nr5cSkippableInline(f_reg[i], f_cnt[i]) &&
               NrbRangeClean(f_reg[i], f_cnt[i], rec.applied_seq)) {
      // [NR-5C] lever (b), packet phase: the run is verified equal to the
      // packets above AND unchanged since its last apply -- the register
      // file already holds it.
      ++g_nr5c_skip_rng;
      g_nr5c_skip_dw += f_cnt[i];
    } else {
      if (rec_clean && Nr5cSkippableInline(f_reg[i], f_cnt[i])) {
        ++g_nr5c_dirty_rng;
      }
      cp->NrSkipApplyRegRange(f_reg[i], rb + f_off[i], f_cnt[i], 0, false);
    }
    g_4a_comp_dw += f_cnt[i];
  }
  g_4a_comp_runs += nf;
  // [NR-5C] in-place pass: copy the draw entry out of the arena FIRST (the
  // stop issues from it after this function returns), then re-validate the
  // whole in-place read+apply. On a tear, refuse -- the caller flushes the
  // deferred packet ranges over whatever the torn pass wrote (the defer
  // list is intact until the clear below), which is the packet truth.
  if (rb != g_nr4a_scratch) {
    if (g_nr5_coh && draw_ndw) {
      std::memcpy(g_nr4a_scratch, rb + draw_off, size_t(draw_ndw) * 4);
    }
    if (!nr::VerifyState(rec)) {
      ++g_nr5c_tear_pkt;
      return false;
    }
  }
  ++g_4a_composed;
  // [NR-5A] arm the single-source draw issue: NrSkipDrawDirect reads the
  // draw's dwords from the record scratch (verified equal to the packet
  // above) instead of the buffer bytes. Cleared at every stop and on
  // consumption, so it can never leak across draws.
  if (g_nr5_coh && draw_ndw) {
    g_nr5_draw_src = rb != g_nr4a_scratch ? g_nr4a_scratch
                                          : g_nr4a_scratch + draw_off;
    g_nr5_draw_ndw = draw_ndw;
    ++g_nr5_composed;
  }
  if (g_nrb_track) {
    nr::StampApplied(pkt_addr, rec.state_gen, g_nrb_apply_seq);
    ++g_nr5c_stamped;
  }
  g_nr4a_ndef = 0;
  return true;
}

// [NR-TILE] N-4 probe mode 2: apply only the SYSTEM/SYNC half of a repeat
// tile band's register stream (everything below 0x2000: the scratch,
// COHER and callback blocks the GUEST polls for progress). 0x2000 and above
// - device state, the shader constant files, the fetch constants - is read
// by this renderer alone, so a band whose draws are all skipped never needs
// it, and band 1 rewrites it next frame from the same buffers.
// ⚠ Dropping the whole stream is what hung the guest inside one second in
// naruto_503 (buffer dropped outright, delegates lost with it) and again in
// naruto_504 (delegates kept, applies dropped): the producer spins on
// values these low registers carry.
// ⚠ AND the bin trio (0x2080-0x2082) must ride along even though it sits
// above 0x2000: PA_SC_WINDOW_OFFSET is what NAMES the band, and the guest
// writes it from inside the buffers as well as from the ring (the N-2-0
// in-place finalize class). Dropping it self-latched the probe in
// naruto_505 - once a band left the offset non-zero, every later buffer
// looked like a repeat band and every draw in the frame was skipped
// (`bufs skipped=734 kept=0`).
void NrTileRegWrite(void* user, uint32_t reg, uint32_t value,
                    bool from_memory) {
  if (reg < 0x2000u || (reg >= 0x2080u && reg <= 0x2082u)) {
    NrWalkRegWrite(user, reg, value, from_memory);
  }
}

bool NrTileRegRange(void* user, uint32_t base, const uint32_t* values_be,
                    uint32_t n, uint32_t phys, bool from_memory) {
  if (base < 0x2000u) {
    return NrWalkRegRange(user, base, values_be, n, phys, from_memory);
  }
  if (base <= 0x2082u && base + n > 0x2080u) {
    return false;  // straddles the bin trio: per-dword, where the filter
                   // above keeps exactly those three registers live
  }
  return true;  // consumed and dropped: never falls back to per-dword
}

// [NR-97] N-9-7 phase A: the keep-list. Everything the guest itself reads, and
// everything the de-tile learner reads -- nothing else.
inline bool N97KeepReg(uint32_t reg) {
  // The guest's own GPU-progress window: scratch, COHER, the callback block.
  // Load-bearing, naruto_504.
  if (reg < 0x2000u) return true;
  // Above the last mirrored slot there is nothing to protect (the constant
  // files and the fetch window start far past it), so the common case is two
  // compares and out.
  if (reg >= 0x2322u) return false;
  // The 27-slot recovery mirror. CtxWriteReg stores these into g_ctx_state
  // whether or not reg_fn runs, so dropping the live write would desynchronise
  // the mirror from the register file -- and the six that de-tile's predicate
  // and observer read (RB_SURFACE/COLOR/DEPTH_INFO, PA_SC_WINDOW_OFFSET and
  // both window scissors) plus RB_MODECONTROL and RB_COPY_DEST_BASE all sit
  // inside it. This is naruto_505 stated as a class instead of as three
  // registers.
  return nr::CtxRangeTouchesMirrorRegs(reg, 1);
}

// (SS) Remember a dropped register at its FIRST touch, with the value it held
// BEFORE the write lands, so the buffer-end sweep can ask the net question.
inline void N97Touch(uint32_t reg) {
  if (reg >= RegisterFile::kRegisterCount) return;
  if (!g_n97_touched[reg]) {
    g_n97_tlist.push_back(reg);
    g_n97_tentry.push_back(g_n97_rf->values[reg]);
    g_n97_tlast.push_back(g_n97_rf->values[reg]);
    g_n97_touched[reg] = uint32_t(g_n97_tlist.size());  // index + 1
  }
  if (!g_n97_exec_seen[reg]) {
    g_n97_exec_seen[reg] = 1;
    g_n97_exec_list.push_back(reg);
  }
}

// (SS) Close a band execution: whatever it left in the registers it touched is
// the newest "after" the BANDS are answerable for. Bounded by what this one
// execution touched, so a long run stays linear.
void N97ExecEnd() {
  for (uint32_t reg : g_n97_exec_list) {
    // The sweep clears g_n97_touched, so a slot of 0 means this execution's
    // list outlived its sweep. It cannot happen on one thread (touches and
    // this call are inside the same buffer execution), and a 0 here would
    // index tlast at -1, so it is checked rather than assumed.
    const uint32_t slot = g_n97_touched[reg];
    if (slot) g_n97_tlast[slot - 1] = g_n97_rf->values[reg];
    g_n97_exec_seen[reg] = 0;
  }
  g_n97_exec_list.clear();
}

// The write-by-write read (churn, NOT the verdict -- see the globals) plus the
// touch recording that feeds the one that is.
void N97CountChanged(uint32_t base, const uint32_t* be, uint32_t n) {
  if (!g_n97_rf || base + n > RegisterFile::kRegisterCount) {
    g_n97_skip_dw += n;
    return;
  }
  const uint32_t* live = &g_n97_rf->values[base];
  for (uint32_t i = 0; i < n; ++i) {
    N97Touch(base + i);
    if (!be) {
      ++g_n97_skip_dw;  // by-ref: no packet value to compare, net sweep covers
    } else if (__builtin_bswap32(be[i]) == live[i]) {
      ++g_n97_same_dw;
    } else {
      ++g_n97_chg_dw;
    }
  }
}

// (SS) The verdict, swept at the swap. Did this frame's band executions leave
// any register different from how they found it?
void N97NetSweep() {
  if (g_n97_tlist.empty()) return;
  ++g_n97_net_frames;
  bool dirty = false;
  for (size_t i = 0; i < g_n97_tlist.size(); ++i) {
    const uint32_t reg = g_n97_tlist[i];
    g_n97_touched[reg] = 0;
    if (g_n97_tlast[i] == g_n97_tentry[i]) {
      ++g_n97_net_same;
    } else {
      ++g_n97_net_chg;
      dirty = true;
      if (g_n97_chg_reg == 0xFFFFFFFFu) g_n97_chg_reg = reg;
      uint32_t k = 0;
      for (; k < g_n97_chg_rec_n; ++k) {
        if (g_n97_chg_rec[k].reg == reg) break;
      }
      if (k < g_n97_chg_rec_n) {
        ++g_n97_chg_rec[k].n;
      } else if (g_n97_chg_rec_n < kN97ChgSamples) {
        g_n97_chg_rec[g_n97_chg_rec_n++] =
            N97ChgRec{reg, g_n97_tentry[i], g_n97_tlast[i], 1};
      } else {
        ++g_n97_chg_rec_ovf;
      }
    }
  }
  if (dirty) ++g_n97_net_dirty;
  g_n97_tlist.clear();
  g_n97_tentry.clear();
  g_n97_tlast.clear();
}

// Mark the blocks a deferred range covers. Cannot fail and cannot overflow,
// which is the whole point.
inline void N97Pending(uint32_t base, uint32_t n) {
  const uint32_t first = base / kN97Blk;
  const uint32_t last = (base + n - 1) / kN97Blk;
  for (uint32_t b = first; b <= last && b < kN97Blocks; ++b) {
    if (!g_n97_blk[b]) {
      g_n97_blk[b] = 1;
      ++g_n97_blk_n;
    }
  }
}

// The per-dword half. The value is STORED either way; only the side effects
// are deferred, and only while the phase is live.
void N97RegWrite(void* user, uint32_t reg, uint32_t value, bool from_memory) {
  if (N97KeepReg(reg)) {
    ++g_n97_keep_pdw;
    NrWalkRegWrite(user, reg, value, from_memory);
    return;
  }
  ++g_n97_drop_pdw;
  if (g_n97_sample && !g_n97_bufdrop && g_n97_rf &&
      reg < RegisterFile::kRegisterCount) {
    N97Touch(reg);
    if (value == g_n97_rf->values[reg]) {
      ++g_n97_same_dw;
    } else {
      ++g_n97_chg_dw;
    }
  }
  if (!g_n97_bufdrop) {
    NrWalkRegWrite(user, reg, value, from_memory);
    return;
  }
  // Value now, effects at the end of the run. The store is the whole reason
  // the naruto_742 class cannot come back: the register file is left exactly
  // where a full apply would have left it.
  if (reg < RegisterFile::kRegisterCount) {
    g_n97_rf->values[reg] = value;
    if (g_nr_issue && g_nri_seeded && !g_nri_live) {
      g_nri_file.values[reg] = value;
    }
    N97Pending(reg, 1);
    ++g_n97_store_rng;
    ++g_n97_store_dw;
  }
}

// The bulk half. A range only reaches range_fn once CtxRangeOfferable has
// proved it clear of all 27 mirrored slots, and a range crossing 0x2000 upward
// would touch the RT slot {0x2000,6} and so is never offered: a range arriving
// here is therefore wholly below 0x2000 or wholly clear of everything we keep.
// No straddle case exists -- unlike NrTileRegRange, which had to defend a
// three-register keep-list sitting in the middle of the state block.
bool N97RegRange(void* user, uint32_t base, const uint32_t* values_be,
                 uint32_t n, uint32_t phys, bool from_memory) {
  if (base < 0x2000u) {
    ++g_n97_keep_rng;
    g_n97_keep_dw += n;
    return NrWalkRegRange(user, base, values_be, n, phys, from_memory);
  }
  ++g_n97_drop_rng;
  g_n97_drop_dw += n;
  if (g_n97_sample && !g_n97_bufdrop) N97CountChanged(base, values_be, n);
  if (!g_n97_bufdrop) {
    return NrWalkRegRange(user, base, values_be, n, phys, from_memory);
  }
  // Value now, effects at the end of the run. `be` is big-endian exactly as
  // the apply path copy_and_swaps it; a null one is a by-ref load whose
  // payload lives in guest memory at `phys`.
  const uint32_t* be = values_be;
  if (!be) {
    be = g_n97_mem ? g_n97_mem->TranslatePhysical<uint32_t*>(phys) : nullptr;
    if (!be) return false;  // cannot store it: let the normal path have it
  }
  if (base + n > RegisterFile::kRegisterCount) return false;
  // The same SIMD swap the normal apply path uses -- a scalar loop here would
  // hand back part of what the deferral just won (~144 MB/s at city rates).
  uint32_t* dst = g_n97_rf->values + base;
  memory::copy_and_swap(dst, be, n);
  if (g_nr_issue && g_nri_seeded && !g_nri_live) {
    std::memcpy(&g_nri_file.values[base], dst, n * sizeof(uint32_t));
  }
  N97Pending(base, n);
  ++g_n97_store_rng;
  g_n97_store_dw += n;
  return true;  // consumed: stored, side effects deferred to the run's end
}

// (SS) The run ended: re-apply every deferred range through the NORMAL path,
// from the values now in the register file. Idempotent (the store already put
// them there), so its only job is to fire the dirty tails and the texture /
// residency invalidations exactly once each, with the right values, before
// anything that reads them. On pending overflow the list is short, not wrong:
// ranges past the cap were stored but never listed, so the cap is chosen far
// above the ~150 distinct ranges a city run actually uses and the overflow is
// counted so silence can never be mistaken for coverage.
void N97FlushPending(CommandProcessor* cp) {
  if (!g_n97_blk_n) return;
  ++g_n97_flush_runs;
  // Nothing below 0x2000 can be marked (the keep-list takes every one of those
  // writes, and re-applying a stateful port -- COHER's OR-write above all --
  // would be actively wrong), but the bound is asserted rather than assumed.
  const uint32_t first_blk = 0x2000u / kN97Blk;
  uint32_t b = first_blk;
  while (b < kN97Blocks) {
    if (!g_n97_blk[b]) {
      ++b;
      continue;
    }
    uint32_t e = b;
    while (e < kN97Blocks && g_n97_blk[e]) {
      g_n97_blk[e] = 0;
      ++e;
    }
    const uint32_t base = b * kN97Blk;
    uint32_t n = (e - b) * kN97Blk;
    if (base + n > RegisterFile::kRegisterCount) {
      n = RegisterFile::kRegisterCount - base;
    }
    if (g_n97_pend_scratch.size() < n) g_n97_pend_scratch.resize(n);
    memory::copy_and_swap(g_n97_pend_scratch.data(), g_n97_rf->values + base, n);
    cp->NrSkipApplyRegRange(base, g_n97_pend_scratch.data(), n, 0, false);
    ++g_n97_flush_rng;
    g_n97_flush_dw += n;
    b = e;
  }
  // Anything below the floor was never legal to mark; clear it so a stray bit
  // cannot survive into the next run.
  for (uint32_t i = 0; i < first_blk; ++i) g_n97_blk[i] = 0;
  g_n97_blk_n = 0;
}

// Arm the filter for the rest of this execution. Called either at buffer entry
// (the memo says this buffer's last execution was all-band) or at its first
// draw stop. Installing the filter LAZILY is what keeps a non-band buffer at
// exactly its old cost: the walker reads reg_fn/range_fn per call, so nothing
// but a band execution ever pays the extra branch.
void N97Activate(bool at_entry) {
  if (!at_entry) {
    // What the prefix before the first stop already applied -- the part a
    // buffer-entry decision would have removed and this one cannot.
    // ⚠ g_skp_pdw / g_skp_rng_dw are ZEROED by the 1 Hz report, so a window
    // boundary landing inside this buffer makes the subtraction wrap. Drop
    // that sample rather than add 2^64 to the total.
    const uint64_t now = g_skp_pdw + g_skp_rng_dw;
    if (now >= g_n97_pre_mark) g_n97_pre_dw += now - g_n97_pre_mark;
  }
  g_n97_filter = true;
  g_n97_bufdrop = g_n97_mode > 0 && N97MemoGet(g_n97_bufkey) != kN97Mixed;
  if (g_n97_bufdrop) {
    ++g_n97_bufs_drop;
    // The 4a defer path assumes decode-order applies that are about to stop
    // happening; pending ranges are already resolved at this stop, so simply
    // standing it down for the rest of the buffer is exact.
    g_nr4a_bufactive = false;
  } else if (N97MemoGet(g_n97_bufkey) == kN97Mixed) {
    ++g_n97_bufs_ref;
  }
  // The census runs in BOTH cycler phases: the filter is installed either way
  // and only DROPS when the phase is live, so the OFF phase prices exactly the
  // traffic (and the risk) the ON phase took ([[price-stages-in-place]]).
  g_ctx_walker.reg_fn = N97RegWrite;
  g_ctx_walker.range_fn = N97RegRange;
}

void ResAddrSeen(uint32_t addr) {
  if (!addr) return;
  uint32_t i = (addr * 2654435761u) >> 15;
  for (uint32_t probe = 0; probe < 8; ++probe) {
    uint32_t& slot = g_res_addr_set[(i + probe) & (kResAddrSetSize - 1)];
    if (slot == addr) return;
    if (!slot) {
      slot = addr;
      ++g_res_addr_distinct;
      return;
    }
  }
  ++g_res_addr_ovf;
}

void NrResDraw(void*) {
  nr::ResObserveDraw(&g_res_state, &g_res_stats);
  // Only the constants that are actually live vertex fetches, straight off
  // the mask: at ~200k draws/s a full scan per draw would be probe cost, not
  // measurement.
  for (uint32_t w = 0; w < 3; ++w) {
    uint32_t mask = g_res_state.vfetch_live_mask[w];
    while (mask) {
      const uint32_t bit = uint32_t(__builtin_ctz(mask));
      mask &= mask - 1;
      ResAddrSeen(
          nr::ResDecodeVertexFetch(&g_res_state, w * 32 + bit).base);
    }
  }
}

uint32_t ResReadLive(void* user, uint32_t reg) {
  return static_cast<const RegisterFile*>(user)->values[reg];
}

// [NR-RES] Coverage: of the fetch constants a draw's own shaders reference,
// how many are recoverable from the walk-derived mirror? This is the strict
// question the occupancy counters cannot answer, and it is the last input a
// native draw needs. Counted at EXECUTION time, right after IssueDraw, for
// two reasons: the shaders are analyzed by then (analysis happens inside it),
// and no packet has run since the draw, so the mirror still holds exactly
// that draw's state.
uint64_t g_res_cov_draws = 0, g_res_cov_full = 0, g_res_cov_unanalyzed = 0;
uint64_t g_res_vref = 0, g_res_vref_undef = 0, g_res_vref_type = 0;
uint64_t g_res_tref = 0, g_res_tref_undef = 0, g_res_tref_type = 0;
struct ResCovSample {
  uint32_t slot;
  uint32_t dword0;  // the constant's first dword, VERBATIM
  uint8_t texture;  // 0 = vertex view, 1 = texture view
  uint8_t pixel;    // which shader referenced it
  uint8_t reason;   // nr::ResCoverage
};
constexpr uint32_t kResCovSamples = 6;
ResCovSample g_res_cov_samp[kResCovSamples];
uint32_t g_res_cov_samp_n = 0;
// ⚠ Samples are capped at 6 a window and taken in binding order, so they can
// suggest "it is always slot 0" without ever having measured it. That caveat
// was written down last session and then reasoned from anyway. A full
// histogram costs one increment per failure and settles it.
uint64_t g_res_vref_fail_slot[nr::kResFetchVertexSlots] = {};
uint64_t g_res_tref_fail_slot[nr::kResFetchTextureSlots] = {};

void ResCoverShader(const Shader* shader, bool pixel, bool* all_covered) {
  if (!shader || !shader->is_ucode_analyzed()) return;
  for (const auto& vb : shader->vertex_bindings()) {
    const nr::ResCoverage c =
        nr::ResVertexFetchCoverage(&g_res_state, vb.fetch_constant);
    ++g_res_vref;
    if (c == nr::kResCoverLive) continue;
    *all_covered = false;
    (c == nr::kResCoverUndefined ? g_res_vref_undef : g_res_vref_type)++;
    if (vb.fetch_constant < nr::kResFetchVertexSlots) {
      ++g_res_vref_fail_slot[vb.fetch_constant];
    }
    if (g_res_cov_samp_n < kResCovSamples) {
      g_res_cov_samp[g_res_cov_samp_n++] = {
          vb.fetch_constant,
          nr::ResFetchDword0(&g_res_state, vb.fetch_constant, false), 0,
          uint8_t(pixel), uint8_t(c)};
    }
  }
  for (const auto& tb : shader->texture_bindings()) {
    const nr::ResCoverage c =
        nr::ResTextureFetchCoverage(&g_res_state, tb.fetch_constant);
    ++g_res_tref;
    if (c == nr::kResCoverLive) continue;
    *all_covered = false;
    (c == nr::kResCoverUndefined ? g_res_tref_undef : g_res_tref_type)++;
    if (tb.fetch_constant < nr::kResFetchTextureSlots) {
      ++g_res_tref_fail_slot[tb.fetch_constant];
    }
    if (g_res_cov_samp_n < kResCovSamples) {
      g_res_cov_samp[g_res_cov_samp_n++] = {
          tb.fetch_constant,
          nr::ResFetchDword0(&g_res_state, tb.fetch_constant, true), 1,
          uint8_t(pixel), uint8_t(c)};
    }
  }
}

// [NR-SDB] Increment 4b-2: the shader-database probe.
//
// The offline index keys every container by XXH3-64 over its raw big-endian
// ucode, which is what PipelineCache::LoadShader computes for a shader read
// from guest memory -- provided the guest declares the same length the
// container stores. The whole AOT-translation plan rests on that "provided",
// and it has never been measured against a running game, so nothing is built
// on the key until this probe answers: of the shaders the game actually
// loads, how many are in the database, under the SAME key?
//
// A miss is not one thing, so the probe splits it. `prefix` = the database
// holds a blob agreeing with this one over their common length: the same
// shader under a different declared size, which an AOT cache keyed by the
// container hash would miss even though it holds the shader (fixable by
// keying on the common prefix). `absent` = the corpus genuinely does not
// hold this shader, which would mean shaders arrive from somewhere other
// than xeshader.sdb and the corpus is not closed. The two demand opposite
// responses, so a single "miss%" would be useless.
//
// Distinct accounting is the headline: per-load rates are dominated by
// whichever shaders the current scene rebinds most, while the corpus
// question is about the SET.
nr::ShaderDbIndex* g_sdb_index = nullptr;
bool g_sdb_ready = false, g_sdb_failed = false;
uint64_t g_sdb_loads = 0, g_sdb_hits = 0, g_sdb_misses = 0;
uint64_t g_sdb_distinct = 0, g_sdb_distinct_hit = 0;
uint64_t g_sdb_distinct_prefix = 0, g_sdb_distinct_absent = 0;
uint64_t g_sdb_stage_mismatch = 0, g_sdb_set_ovf = 0, g_sdb_zero = 0;
// Distinct-blob set over the whole session (NOT per window): the corpus
// question is cumulative, and clearing it each second would re-count the
// same shaders forever. 65,536 slots against a 3,320-shader corpus.
constexpr uint32_t kSdbSeenSize = 65536;
uint64_t g_sdb_seen[kSdbSeenSize];
struct SdbMissSample {
  uint64_t hash;
  uint32_t dwords;
  uint32_t guest_addr;
  uint32_t db_bytes;
  uint32_t equal_bytes;
  bool pixel;
  bool prefix;
};
constexpr uint32_t kSdbMissSamples = 6;
SdbMissSample g_sdb_miss_samp[kSdbMissSamples];
uint32_t g_sdb_miss_samp_n = 0;
// Raw-ucode dump of the misses. A miss that is "absent" says the corpus is
// not closed but says nothing about where the shader DID come from, and the
// address alone cannot answer that after the process exits -- so the bytes
// go to disk, where they can be searched for in the XEX, the AI2C module and
// the BigFile. Record: {'NRSD', bytes, hash, stage} then the raw big-endian
// ucode, so the file is self-describing and appendable.
FILE* g_sdb_dump = nullptr;
uint32_t g_sdb_dumped = 0;
constexpr uint32_t kSdbDumpMax = 4096;

void SdbDumpMiss(uint64_t hash, bool pixel, const uint8_t* ucode,
                 uint32_t bytes) {
  if (!g_sdb_dump || g_sdb_dumped >= kSdbDumpMax) return;
  const uint32_t tag = 0x4E525344u;  // 'NRSD'
  const uint32_t stage = pixel ? 1u : 0u;
  fwrite(&tag, 4, 1, g_sdb_dump);
  fwrite(&bytes, 4, 1, g_sdb_dump);
  fwrite(&hash, 8, 1, g_sdb_dump);
  fwrite(&stage, 4, 1, g_sdb_dump);
  fwrite(ucode, 1, bytes, g_sdb_dump);
  ++g_sdb_dumped;
  // Flushed per record: the probe is normally ended by killing the process,
  // and a buffered dump would lose exactly the tail that motivated the run.
  fflush(g_sdb_dump);
}

// True when `hash` had not been seen before (the caller then classifies it).
bool SdbSeenFirstTime(uint64_t hash) {
  uint64_t key = hash ? hash : 1;
  uint32_t i = uint32_t((key * 2654435761ull) >> 16) & (kSdbSeenSize - 1);
  for (uint32_t probe = 0; probe < 16; ++probe) {
    uint64_t& slot = g_sdb_seen[(i + probe) & (kSdbSeenSize - 1)];
    if (slot == key) return false;
    if (!slot) {
      slot = key;
      return true;
    }
  }
  ++g_sdb_set_ovf;
  return false;
}

// Called for every shader the executor loads (both IM_LOAD forms), on the
// command-processor thread. `host_address` is the guest ucode in place: the
// SAME bytes and the SAME length the pipeline cache is about to hash, so the
// probe cannot drift from the key it is validating.
void NrSdbObserve(xenos::ShaderType shader_type, uint32_t guest_address,
                  const uint32_t* host_address, uint32_t dword_count) {
  if (!g_sdb_ready || !host_address) return;
  if (!dword_count) {
    ++g_sdb_zero;
    return;
  }
  const uint8_t* ucode = reinterpret_cast<const uint8_t*>(host_address);
  const uint32_t bytes = dword_count * 4;
  const uint64_t hash = nr::HashUcode(ucode, bytes);
  const bool pixel = shader_type == xenos::ShaderType::kPixel;

  ++g_sdb_loads;
  const nr::ShaderDbIndex::Entry* entry = g_sdb_index->Lookup(hash);
  if (entry) {
    ++g_sdb_hits;
  } else {
    ++g_sdb_misses;
  }
  if (!SdbSeenFirstTime(hash)) return;

  ++g_sdb_distinct;
  if (entry) {
    ++g_sdb_distinct_hit;
    // The runtime's map is keyed by the hash ALONE, so a blob the database
    // files under the other stage would translate as the wrong stage.
    if (entry->pixel != pixel) ++g_sdb_stage_mismatch;
    return;
  }
  const nr::ShaderDbIndex::PrefixMatch pm =
      g_sdb_index->FindByPrefix(ucode, bytes);
  if (pm.found) {
    ++g_sdb_distinct_prefix;
  } else {
    ++g_sdb_distinct_absent;
  }
  SdbDumpMiss(hash, pixel, ucode, bytes);
  if (g_sdb_miss_samp_n < kSdbMissSamples) {
    g_sdb_miss_samp[g_sdb_miss_samp_n++] = {hash,       dword_count,
                                            guest_address, pm.db_bytes,
                                            pm.equal_bytes, pixel,
                                            pm.found};
  }
}

}  // namespace

void CommandProcessor::WorkerThreadMain() {
  if (!SetupContext()) {
    rex::FatalError("Unable to setup command processor internal state");
    return;
  }

  // [GPU-WORKER-PROFILE] Opt-in (gpu_worker_profile cvar) breakdown of where this
  // single command-processor thread's wall time actually goes: the starvation
  // spin (waiting on the guest producer) vs real ExecutePrimaryBuffer work.
  // Resolves the spin-vs-compute question on Release builds, which have no Tracy
  // zones. ~zero cost when the cvar is off (one bool test per frame region).
  // Only this thread touches these counters, so plain locals — no atomics.
  const bool kProfile = REXCVAR_GET(gpu_worker_profile);
  g_exec_prof = kProfile;
  // [GPU-SPLIT] independent clean split mode (no per-packet timing).
  const bool kSplit = REXCVAR_GET(gpu_split_profile);
  g_split_prof = kSplit;
  // [PM4-CENSUS] independent draw-emission census (no timing; just packet counts).
  const bool kCensus = REXCVAR_GET(gpu_pm4_census);
  g_pm4_census = kCensus;
  // [NR-BUF]/[NR-STATE] imply [NR-CACHE] implies the ledger: each layer's
  // scoring lives on the previous one's per-buffer walk in
  // ExecuteIndirectBuffer.
  const bool kNrBuf = REXCVAR_GET(gpu_nr_bufcache);
  g_nr_bufcache = kNrBuf;
  const bool kNrState = REXCVAR_GET(gpu_nr_state);
  g_nr_state = kNrState;
  // [NR-VERIFY] inc 2: the master verify latch. Applied here to the
  // compare-only base probes; the d3d12 side latches its own copy per frame.
  const bool kNrVerify = REXCVAR_GET(gpu_nr_verify);
  g_nr_verify_base = kNrVerify;
  if (kNrVerify) g_nri_sh_dirty = true;  // memo never trusted across verify
  // [NR-RES] rides the 4a context walk, so it implies it. Compare-only
  // census: forced off by the perf config.
  const bool kNrRes = REXCVAR_GET(gpu_nr_res) && kNrVerify;
  g_nr_res = kNrRes;
  // [NR-SKP] 5-4-2: the skip is the consumer of BOTH the issue seam and the
  // walk-driven effects, so it implies them (and through them the lockstep
  // walk and the running context).
  const bool kNrSkip = REXCVAR_GET(gpu_nr_skip);
  g_nr_skip = kNrSkip;
  // [NR-TILE] N-4: the tile probe rides the skip (the draw skip sits in the
  // skip loop). Announced, because a run with wrong pixels must never be
  // mistaken for a normal one in a log.
  g_nr_tile_probe = kNrSkip ? REXCVAR_GET(gpu_nr_tile_probe) : 0;
  g_nr_tile_cycle = g_nr_tile_probe == 3;
  if (g_nr_tile_cycle) {
    g_nr_tile_probe = 0;
    REXGPU_INFO(
        "[nr-tile] PROBE ARMED mode=3 CYCLE - 10s off, 10s repeat-band draws "
        "skipped, 10s repeat-band buffers skipped, repeating. Drive to the "
        "heavy city and HOLD STILL. Pixels outside band 1 are wrong by "
        "design in phases 1 and 2.");
  } else if (g_nr_tile_probe) {
    REXGPU_INFO(
        "[nr-tile] PROBE ARMED mode={} - {} of every EDRAM tile band but the "
        "first is SKIPPED. Pixels outside band 1 are wrong by design.",
        g_nr_tile_probe, g_nr_tile_probe == 1 ? "the draws" : "the buffers");
  }
  // [NR-DETILE] N-5: rides the skip, because the observe-and-skip seam is the
  // walk's own draw-stop loop. The per-frame value is recomputed at the swap
  // (mode 2 cycles there); this only announces the mode and gates it.
  g_nr_detile_mode = kNrSkip ? REXCVAR_GET(gpu_nr_detile) : 0;
  if (g_nr_detile_mode && REXCVAR_GET(gpu_nr_detile_tap) >= 0) {
    REXGPU_INFO(
        "[nr-detile] TAP BISECT: only band-1 resolve tap {} is extended to "
        "the full frame height. Wrong pixels by design.",
        REXCVAR_GET(gpu_nr_detile_tap));
  }
  if (g_nr_detile_mode) {
    REXGPU_INFO(
        "[nr-detile] {}: the frame is rendered once instead of once per EDRAM "
        "tile band (arming from the first frame that shows the bands).",
        g_nr_detile_mode == 2 ? "CYCLE 10s ON / 10s OFF" : "ON");
  }
  // [NR-SPP] skip-path timing probe rides the skip.
  const bool kSkpProf = REXCVAR_GET(gpu_nr_skip_profile) && kNrSkip;
  g_skp_prof = kSkpProf;
  // [NR-SKP] 5-4-4a: the direct draw path only exists under the skip.
  g_nr_skip_direct = REXCVAR_GET(gpu_nr_skip_direct) && kNrSkip;
  // [NR-PB] N-2-2 item 0: the plain bulk apply only exists under the skip
  // (NrWalkRegRange is dead everywhere else). Unconditional there.
  g_nr_plain_bulk = kNrSkip;
  // [NR-RUSE] 5-4-5 inc 0: the reuse pricing probe rides the skip (the walk
  // is the only decoder there, so buffer bytes ARE the input stream).
  // 5-4-5-2: the fast path consumes the verdict, so it implies the probe
  // machinery; the v0 digest/prefix reporting runs only under the probe
  // cvar itself (fast-only runs pay just the v2 verdict).
  g_nr_ruse_v0 = REXCVAR_GET(gpu_nr_reuse_probe) && kNrSkip;
  g_nr_ruse = (g_nr_ruse_v0 || REXCVAR_GET(gpu_nr_reuse_fast)) && kNrSkip;
  g_nr_ruse_blk = REXCVAR_GET(gpu_nr_reuse_blk) && g_nr_ruse;
  // [NR-WM] 5-4-8: the walk memo needs the skip (the walk is the only
  // decoder) and the ruse byte compare (its validity gate). Rising edge
  // clears the store: while the latch was off, byte changes went unobserved
  // by the invalidator, so no stored stream survives the gap.
  {
    const bool wm = REXCVAR_GET(gpu_nr_walk_memo) && kNrSkip && g_nr_ruse;
    if (wm && !g_nr_wm) nr::CtxMemoClear();
    g_nr_wm = wm;
  }
  // [NR-RUSE-EP] the unsound-at-city compare shortcut, deliberate opt-in.
  g_nr_ruse_ep = REXCVAR_GET(gpu_nr_ruse_epoch);
  // [NR-BFC] 5-4-6-0: the buffer-replay census needs the ruse machinery for
  // candidacy (bytes/entry/byref facts) and only measures skip-driven
  // buffers.
  g_nr_bfc = REXCVAR_GET(gpu_nr_bufreplay_census) && kNrSkip && g_nr_ruse;
  // [NR-DSP] 5-4-7-0: same requirements (the verdict is the population).
  g_nr_dsp = REXCVAR_GET(gpu_nr_drawspan_census) && kNrSkip && g_nr_ruse;
  // [NR-SPR] 5-4-7-1: same requirements again -- the reuse verdict IS the
  // replay gate (naruto_423: input identity implies emission identity).
  // [NR-SPW] 5-4-7-2: the consuming swap implies the record machinery (its
  // store is the replay source); the backend latches its own consume gate.
  g_nr_spr = (REXCVAR_GET(gpu_nr_span_replay) || REXCVAR_GET(gpu_nr_span_swap)) &&
             kNrSkip && g_nr_ruse;
  // [NR-ISSUE] consumes the lockstep shadow, so it implies it.
  const bool kNrIssue = REXCVAR_GET(gpu_nr_issue) || kNrSkip;
  g_nr_issue = kNrIssue;
  g_nri_from = REXCVAR_GET(gpu_nr_issue_from);
  g_nri_count = REXCVAR_GET(gpu_nr_issue_count);
  // [NR-ISSUE-LIVE] only in the lockstep-identity configuration; any other
  // mode keeps the mirror (walk-ahead arms, the verify shadow compose, and
  // the reuse entry components all read g_nri_file).
  g_nri_live = REXCVAR_GET(gpu_nr_issue_live) && kNrSkip &&
               !g_nr_verify_base && !g_nr_ruse;
  // [NR-ISSUE] Increment 4e: while issue is off the replay file receives no
  // writes, so a later enable must reseed rather than trust a stale file.
  if (!kNrIssue) g_nri_seeded = false;
  // [NR-FX] 5-4-0: walk-driven side effects REQUIRE the lockstep walk --
  // fired from a buffer-entry whole-buffer walk they would run ahead of the
  // executor and over-invalidate (correct output, inflated rebuilds).
  const bool kNrWalkFx = REXCVAR_GET(gpu_nr_walk_effects) || kNrSkip;
  g_nr_walk_fx = kNrWalkFx;
  // [NR-PKT] 5-4-1: executor-side census, independent of the walk.
  const bool kNrPkt = REXCVAR_GET(gpu_nr_pkt_census);
  g_nr_pkt = kNrPkt;
  // [NR-ORC] N-1 oracle dump: independent of every mode (private walker,
  // observation only). Opens the file here, on the thread that writes it.
  g_nr_orc = REXCVAR_GET(gpu_nr_oracle_dump);
  if (g_nr_orc) NrOracleOpen();
  // [NR-TMPL] N-2 rung 0: record-time span-template store + execute-time
  // compare gate (nr_template_store.cpp). Observation only, like the oracle.
  // The same call arms the game-side span feed at its own startup latch.
  g_nr_tmpl = rex_nr_tmpl_active();
  if (g_nr_tmpl) {
    REXGPU_INFO("[nr-tmpl] ON: span-template store armed (compare at depth-1)");
  } else if (REXCVAR_GET(gpu_nr_tmpl)) {
    REXGPU_INFO("[nr-tmpl] DISARMED: store arena allocation failed");
  }
  // [NR-PLAN] N-2-2 item 2: the consuming swap. Every precondition is a
  // REFUSAL that names itself, never a silent no-op: an armed-but-inert swap
  // would read as "the plan is fps-neutral" when it never ran.
  g_nr_tmpl_swap = REXCVAR_GET(gpu_nr_tmpl_swap) && kNrSkip && g_nr_tmpl &&
                   nr::TmplPlanMode();
  g_nr_tmpl_swap_probe =
      REXCVAR_GET(gpu_nr_tmpl_swap_probe) && g_nr_tmpl_swap;
  if (g_nr_tmpl_swap_probe) {
    REXGPU_INFO(
        "[nr-swap] COST PROBE: lookup + guards per span, no replay (this run "
        "measures the swap's overhead alone, never its win)");
  }
  if (REXCVAR_GET(gpu_nr_tmpl_swap)) {
    if (g_nr_tmpl_swap) {
      REXGPU_INFO(
          "[nr-swap] ON: the flat apply plan replaces the parse per span "
          "(compare gate disabled)");
    } else {
      REXGPU_INFO(
          "[nr-swap] REFUSED: skip={} tmpl={} planMode={} -- the swap is "
          "inert, do not read a perf number from this run",
          kNrSkip, g_nr_tmpl, nr::TmplPlanMode());
    }
  }
  // [N8F] Per-draw effect coalescing. Every veto is a mode whose machinery
  // reads or replays the per-range apply stream (ruse/memo/verify/res), or
  // adds a per-range side effect the deferred flush does not reproduce
  // (gpu_instance's old-value compare, gpu_dedupe_constants' early return),
  // or is a wrong-pixels diagnostic (tile probes). Declines are logged, not
  // silent.
  g_n8f_eligible = kNrSkip && NrCoalesceEligible() && !g_nr_verify_base &&
                   !g_nr_res && !g_nr_ruse && !g_nr_wm && !g_nr_tmpl_swap &&
                   !g_nr_tile_probe && !g_nr_tile_cycle &&
                   !REXCVAR_GET(gpu_instance) &&
                   !REXCVAR_GET(gpu_dedupe_constants);
  g_n8f_on = g_n8f_eligible && REXCVAR_GET(gpu_nr_coalesce);
  if (REXCVAR_GET(gpu_nr_coalesce) && !g_n8f_eligible) {
    REXGPU_INFO(
        "[n8f] REFUSED: skip={} backend={} verify={} res={} ruse={} wm={} "
        "tmplswap={} tile={} instance={} dedupe={} -- the coalescer is "
        "inert, do not read a perf number from this run",
        kNrSkip, NrCoalesceEligible(), g_nr_verify_base, g_nr_res, g_nr_ruse,
        g_nr_wm, g_nr_tmpl_swap, g_nr_tile_probe || g_nr_tile_cycle,
        REXCVAR_GET(gpu_instance), REXCVAR_GET(gpu_dedupe_constants));
  } else if (g_n8f_on) {
    REXGPU_INFO(
        "[n8f] ON: range side effects deferred to the draw stop, one apply "
        "per merged span (values always live)");
  }
  // [N9] Whole-buffer plan replay. Vetoes are the modes that record, replay
  // or byte-compare the walk's own emission stream (ruse/bfc/memo/tmpl-swap)
  // or change what the stream means (tile probes). gpu_nr_verify is NOT a
  // veto: the interpreter fires the same callbacks, so the verify passes are
  // exactly the plan's divergence gate. Refusals are logged, never silent.
  g_n9_eligible = kNrSkip && !g_nr_ruse && !g_nr_bfc && !g_nr_wm &&
                  !g_nr_tmpl_swap && !g_nr_tile_probe && !g_nr_tile_cycle;
  g_n9_on = g_n9_eligible && REXCVAR_GET(gpu_nr_prog);
  if (REXCVAR_GET(gpu_nr_prog) && !g_n9_eligible) {
    REXGPU_INFO(
        "[n9] REFUSED: skip={} ruse={} bfc={} wm={} tmplswap={} tile={} -- "
        "the plan replay is inert, do not read a perf number from this run",
        kNrSkip, g_nr_ruse, g_nr_bfc, g_nr_wm, g_nr_tmpl_swap,
        g_nr_tile_probe || g_nr_tile_cycle);
  } else if (g_n9_on) {
    REXGPU_INFO(
        "[n9] ON: depth-1 buffers replay their compiled flat plan (values "
        "and delegates live, structural guards checked per execution)");
  }
  // [NR-DRAW] rides the same walk, and turns it into a lockstep one.
  const bool kNrDraw = REXCVAR_GET(gpu_nr_draw) || kNrIssue || kNrWalkFx;
  g_nr_draw = kNrDraw;
  const bool kNrCtx = REXCVAR_GET(gpu_nr_ctx) || kNrRes || kNrDraw;
  g_nr_ctx = kNrCtx;
  // [NR-SDB] Load the shader database once, here, on the thread that will
  // query it -- never lazily inside the packet handler, where a 6 MB read
  // would land in the middle of a measured frame.
  const bool kNrSdb = REXCVAR_GET(gpu_nr_shaderdb);
  if (kNrSdb && !g_sdb_ready && !g_sdb_failed) {
    std::string sdb_path = REXCVAR_GET(gpu_nr_shaderdb_path);
    if (sdb_path.empty()) {
      const std::string root = rex::cvar::Query<std::string>("game_data_root");
      if (!root.empty()) sdb_path = root + "/xeshader.sdb";
    }
    g_sdb_index = new nr::ShaderDbIndex();
    if (!sdb_path.empty() && g_sdb_index->LoadFile(sdb_path.c_str())) {
      g_sdb_ready = true;
      const std::string dump_path = REXCVAR_GET(gpu_nr_shaderdb_dump);
      if (!dump_path.empty()) {
        g_sdb_dump = fopen(dump_path.c_str(), "wb");
        REXGPU_INFO("[nr-sdb] miss ucode dump -> {} ({})", dump_path,
                    g_sdb_dump ? "open" : "FAILED TO OPEN");
      }
      REXGPU_INFO(
          "[nr-sdb] loaded {} ({} bytes): {} containers, {} unique blobs "
          "(pixel={} vertex={}, bad-bounds={})",
          sdb_path, g_sdb_index->file_bytes(), g_sdb_index->stats().containers,
          g_sdb_index->unique_count(), g_sdb_index->stats().pixel,
          g_sdb_index->stats().vertex, g_sdb_index->stats().bad_ucode_bounds);
    } else {
      // A probe that silently measures nothing is worse than no probe.
      g_sdb_failed = true;
      delete g_sdb_index;
      g_sdb_index = nullptr;
      REXGPU_ERROR(
          "[nr-sdb] cannot read shader database '{}' - probe disabled (set "
          "gpu_nr_shaderdb_path)",
          sdb_path.empty() ? "<unset>" : sdb_path);
    }
  }
  // [N8D] The record/replay-gate probes are OPT-IN now. `|| kNrCtx` used to
  // ride here, and kNrCtx is forced true by the skip chain, so every DEFAULT
  // run paid the whole [NR-IBL]+[nrc] block per buffer execution: a registry
  // range query, a CountBufferDraws re-walk (always, at city -- the registry
  // is empty on default runs, so sum!=truth every time), and the full join
  // walk. That was the [n7] "executor fallback" 10% of the CP second
  // (naruto_593: ib 902ms vs skip buf 800ms, fb=0, n==bufs -- nothing ever
  // declined the skip; the gap was this block). The gate is PASS and its
  // consumers are counters only, so it runs when its own cvars ask.
  // [NR-CMP] N-9-3d compose+diff rides the join walk, so it implies the
  // cache probe (both opt-in diagnostics; the default path stays clean --
  // the naruto_602 lesson about implied probes is about DEFAULT flags).
  const bool kNrCompose = REXCVAR_GET(gpu_nr_compose);
  g_nr_compose = kNrCompose;
  // [NR-4A] independent of the diagnostic join walk: it joins via LookupDraw
  // at the stop itself. Needs the game-side nr_rec_store hook feeding the
  // record store; without it every draw counts a no_state/miss fallback.
  // [NR-5A] coherent compose implies the 4a apply it extends.
  g_nr5_coh = REXCVAR_GET(gpu_nr_rec_coherent);
  g_nr4a_on = REXCVAR_GET(gpu_nr_rec_apply) || g_nr5_coh;
  if (g_nr4a_on) {
    REXGPU_INFO(
        "[nr-4a] ON: draw-stop constant apply from stored records "
        "(shape-matched, fallback counted; needs game-side nr_rec_store)");
  }
  if (g_nr5_coh) {
    REXGPU_INFO(
        "[nr-5a] ON: coherent compose -- by-refs from markers, draw args "
        "from the record's draw entry, single-source per composed draw");
  }
  // [NR-5B-3i] by-ref source on suppressed groups. -1 arms the 5 s cycler
  // and starts at 0 so the first phase is today's behaviour (the baseline
  // the eye compares against, in place -- [[visual-defect-needs-visual-bisect]]).
  {
    const int32_t m = REXCVAR_GET(gpu_nr_sup_marker);
    g_nr5b_mk_cycle = g_nr5_coh && m == -1;
    g_nr5b_mk_mode = m < 0 ? 0 : (m > 2 ? 0 : m);
    if (g_nr5_coh && m != 0) {
      REXGPU_INFO(
          "[nr-5b] suppressed by-ref source: {}",
          g_nr5b_mk_cycle  ? "CYCLE 0/1/2 5 s each in place"
          : g_nr5b_mk_mode == 1 ? "1 = drop markers (live 0x2F packets only)"
                                : "2 = live address at the marker's position");
    }
  }
  // [NR-5B-3j] declined-compose policy under live suppression. -1 arms the
  // 5 s cycler and starts at 0 so the first phase is today's behaviour.
  {
    const int32_t g = REXCVAR_GET(gpu_nr_sup_gen);
    g_nr5b_gencycle = g_nr5_coh && g == -1;
    g_nr5b_genmode = g == 1 ? 1 : 0;
    if (g_nr5_coh && g != 0) {
      REXGPU_INFO("[nr-5b] record-generation probe: {}",
                  g_nr5b_gencycle ? "CYCLE count/SKIP-AHEAD 5 s each in place"
                                  : "SKIP draws whose record is AHEAD");
    }
  }
  {
    const int32_t f = REXCVAR_GET(gpu_nr_sup_fallback);
    g_nr5b_fbcycle = g_nr5_coh && f == -1;
    g_nr5b_fbmode = f == 1 ? 1 : 0;
    if (g_nr5_coh && f != 0) {
      REXGPU_INFO("[nr-5b] declined compose under suppression: {}",
                  g_nr5b_fbcycle ? "CYCLE issue/SKIP 5 s each in place"
                                 : "SKIP the dispatch");
    }
  }
  // [NR-4B] verify-rate latch. -1 arms the 10 s cycler (starts FULL so the
  // first phase is the baseline); any other value is the fixed live rate.
  {
    const int32_t v = REXCVAR_GET(gpu_nr_rec_verify);
    g_nr4b_cycle = g_nr4a_on && v == -1;
    g_nr4b_verify_n = v < 0 ? 1 : v;
    if (g_nr4a_on && v != 1) {
      REXGPU_INFO(
          "[nr-4b] verify pricing armed: {} (MEASUREMENT MODE - torn "
          "records may flash while verify is off/sampled)",
          g_nr4b_cycle ? "10s FULL / 10s OFF cycle" : "fixed rate");
    }
  }
  // [NR-5C] the fps-increment levers. The cycler owns the live values while
  // armed (phase flips in the 1 Hz block below); tracking runs in every
  // phase so records accumulate applied stamps before a skip phase starts.
  {
    g_nr5c_arena_on = REXCVAR_GET(gpu_nr_arena_apply) != 0;
    const int32_t s = REXCVAR_GET(gpu_nr_apply_skip);
    g_nr5c_skip_mode = s > 0 ? s : 0;
    g_nr5c_cycle = REXCVAR_GET(gpu_nr_apply_cycle);
    g_nrb_track = g_nr5c_skip_mode > 0 || g_nr5c_cycle;
    // [NR-5C-ATTR] the bracket is armable on its own now; any lever still
    // implies it, so the 716 reading protocol is unchanged.
    g_nr5c_prof = REXCVAR_GET(gpu_nr_compose_prof) || g_nr5c_arena_on ||
                  g_nr5c_skip_mode > 0 || g_nr5c_cycle;
    if (g_nr5c_arena_on || g_nr5c_skip_mode || g_nr5c_cycle) {
      REXGPU_INFO("[nr-5c] levers armed: arena={} skip={}{}", g_nr5c_arena_on,
                  g_nr5c_skip_mode,
                  g_nr5c_cycle ? " CYCLE off/arena/arena+skip 10 s each" : "");
    }
    // [NR-6] the repeat-band compose skip. Off unless asked for; the cycler
    // owns the live mode while armed.
    {
      const int32_t m = REXCVAR_GET(gpu_nr_band_skip);
      g_nr6_mode = m > 0 ? (m > 3 ? 3 : m) : 0;
      g_nr6_cycle = REXCVAR_GET(gpu_nr_band_cycle);
      if (g_nr6_mode || g_nr6_cycle) {
        REXGPU_INFO(
            "[nr-6] repeat-band compose skip ARMED: mode={}{} -- a band-2/3 "
            "draw's deferred ranges are DROPPED, not composed. pred_ne must "
            "stay 0.",
            g_nr6_mode,
            g_nr6_cycle ? " CYCLE off/alu/+fetch/+state 10 s each" : "");
      }
    }
    // [NR-97] N-9-7 phase A: the repeat-band walk skip. Rides the skip (the
    // walk is the only decoder there) and de-tile (its predicate IS the band
    // test, so an unarmed or guard-tripped de-tile disarms this outright).
    {
      const int32_t m = REXCVAR_GET(gpu_nr_band_walk);
      g_n97_mode = (m > 0 && kNrSkip && g_nr_detile_mode) ? 1 : 0;
      g_n97_cycle = REXCVAR_GET(gpu_nr_band_walk_cycle) && kNrSkip &&
                    g_nr_detile_mode;
      if (m > 0 && !(kNrSkip && g_nr_detile_mode)) {
        REXGPU_WARN(
            "[nr-97] DISARMED: needs the skip walk and gpu_nr_detile (the "
            "band predicate is de-tile's own).");
      }
      if (g_n97_mode || g_n97_cycle) {
        REXGPU_INFO(
            "[nr-97] repeat-band WALK SKIP armed: mode={}{} -- a buffer "
            "execution de-tile calls a repeat band keeps its parse, its "
            "delegates and every write below 0x2000, and applies NOTHING "
            "above it but the 27-slot recovery mirror. mixed must stay 0.",
            g_n97_mode,
            g_n97_cycle ? " CYCLE 10 s off / 10 s on in place" : "");
      }
    }
    if (g_nr5c_prof) {
      REXGPU_INFO(
          "[nr-5c] compose bracket ARMED (per stop, split sup/pkt){}",
          REXCVAR_GET(gpu_nr_skip_profile)
              ? " + [n7] skip drawstop split -- supcover-cycle attribution"
              : " (add --gpu_nr_skip_profile true for walk/drawstop ms)");
    }
  }
  const bool kNrCache =
      REXCVAR_GET(gpu_nr_cache) || kNrBuf || kNrState || kNrCompose;
  g_nr_cache = kNrCache;
  const bool kIbLedger = REXCVAR_GET(gpu_ib_ledger) || kNrCache;  // [NR-IBL] record/replay gate
  g_ib_ledger = kIbLedger;
  g_pm4_ib_dump = REXCVAR_GET(gpu_pm4_ib_dump);  // [PM4-IB-DUMP] one-shot IB structure dump
  // [N7] The spin/exec bracket is UNCONDITIONAL now. At city load the worker
  // wakes ~50 times a second, so it is ~100 clock reads per second: it cannot
  // move a frame rate, and it is the only thing that answers "how much of the
  // consumer's wall clock is real work" without the per-packet timing that
  // inflated gpu_worker_profile by ~20%.
  const bool kTimeExec = true;
  using prof_clock = std::chrono::steady_clock;
  uint64_t prof_spin_ns = 0, prof_exec_ns = 0;
  uint64_t prof_spin_loops = 0, prof_event_waits = 0, prof_exec_calls = 0,
           prof_starves = 0;
  uint64_t last_spin_ns = 0, last_exec_ns = 0, last_spin_loops = 0,
           last_event_waits = 0, last_exec_calls = 0, last_starves = 0;
  uint64_t last_type_ns[4] = {0, 0, 0, 0}, last_type_cnt[4] = {0, 0, 0, 0};
  uint64_t last_issuedraw_ns = 0, last_draw_cnt = 0;
  uint64_t last_type0_split_ns = 0;
  auto prof_last_report = prof_clock::now();
  const int32_t kN7 = REXCVAR_GET(gpu_n7);
  uint64_t n7_backlog_sum = 0, n7_backlog_max = 0, n7_wakes = 0;
  uint64_t n7_last_frames = 0, n7_last_draws = 0;
  uint64_t n7_last_spin_ns = 0, n7_last_exec_ns = 0;
  uint64_t n7_last_exec_calls = 0, n7_last_starves = 0;
  uint64_t n7_last_issuedraw_ns = 0, n7_last_type0_ns = 0, n7_last_draw_cnt = 0;
  uint64_t n7_last_ib_ns = 0, n7_last_swap_ns = 0;
  uint64_t n7_last_ib_calls = 0, n7_last_swap_calls = 0;
  auto n7_last_report = prof_clock::now();
  const auto n7_t0 = n7_last_report;

  while (worker_running_) {
    while (!pending_fns_.empty()) {
      auto fn = std::move(pending_fns_.front());
      pending_fns_.pop();
      fn();
    }

    uint32_t write_ptr_index = write_ptr_index_.load();
    if (write_ptr_index == 0xBAADF00D || read_ptr_index_ == write_ptr_index) {
      SCOPE_profile_cpu_i("gpu", "rex::graphics::CommandProcessor::Stall");
      // We've run out of commands to execute.
      // We spin here waiting for new ones, as the overhead of waiting on our
      // event is too high.
      auto spin_t0 = prof_clock::now();  // [N7] unconditional, ~50 reads/s
      prof_starves++;
      PrepareForWait();
      uint32_t loop_count = 0;
      do {
        // Block on the write-pointer event once a short spin fails to find
        // work. The event is Set() the instant the guest submits commands
        // (UpdateWritePointer), so this wakes immediately on new work — the
        // OS wakeup latency (sub-ms with timeBeginPeriod(1)) is negligible vs
        // the frame and is absorbed because the guest is the bottleneck, not
        // this consumer. The old 500-iteration busy-spin pegged a full core
        // (the guest feeds work at sub-ms cadence, so it spun continuously)
        // — wasteful heat that throttles real-work cores on a laptop.
        constexpr uint32_t kStarvedSpinYields = 64;
        if (loop_count > kStarvedSpinYields) {
          const int wait_time_ms = 5;
          if (kProfile) prof_event_waits++;
          rex::thread::Wait(write_ptr_index_event_.get(), true,
                            std::chrono::milliseconds(wait_time_ms));
        }

        rex::thread::MaybeYield();
        loop_count++;
        if (kProfile) prof_spin_loops++;
        write_ptr_index = write_ptr_index_.load();
      } while (worker_running_ && pending_fns_.empty() &&
               (write_ptr_index == 0xBAADF00D || read_ptr_index_ == write_ptr_index));
      ReturnFromWait();
      prof_spin_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          prof_clock::now() - spin_t0).count();
      if (!worker_running_ || !pending_fns_.empty()) {
        continue;
      }
    }
    assert_true(read_ptr_index_ != write_ptr_index);

    // Execute. Note that we handle wraparound transparently.
    {
      // [N7] Ring backlog at the moment we wake, in dwords. Deep = the guest is
      // ahead of us and we are the wall; shallow = the guest is pacing us.
      if (kN7) {
        const uint32_t rb_dwords = uint32_t(primary_buffer_size_ / sizeof(uint32_t));
        const uint32_t backlog =
            rb_dwords ? ((write_ptr_index - read_ptr_index_) & (rb_dwords - 1)) : 0u;
        n7_backlog_sum += backlog;
        if (backlog > n7_backlog_max) n7_backlog_max = backlog;
        ++n7_wakes;
      }
      auto exec_t0 = kTimeExec ? prof_clock::now() : prof_clock::time_point{};
      read_ptr_index_ = ExecutePrimaryBuffer(read_ptr_index_, write_ptr_index);
      if (kTimeExec) {
        prof_exec_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            prof_clock::now() - exec_t0).count();
        prof_exec_calls++;
      }
    }

    // TODO(benvanik): use reader->Read_update_freq_ and only issue after moving
    //     that many indices.
    if (read_ptr_writeback_ptr_) {
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(read_ptr_writeback_ptr_),
                                       read_ptr_index_);
    }

    // [NR-TILE] N-4 mode 3: walk the probe through 0/1/2 in place, 10s each,
    // so one drive prices all three conditions at matched draws/frame. Only
    // armed runs read the clock.
    if (g_nr_tile_cycle) {
      static auto tile_t0 = prof_clock::now();
      const int phase =
          int((std::chrono::duration_cast<std::chrono::seconds>(
                   prof_clock::now() - tile_t0)
                   .count() /
               10) %
              3);
      if (phase != g_nr_tile_probe) {
        g_nr_tile_probe = phase;
        REXGPU_INFO("[nr-tile] mode -> {} ({})", phase,
                    phase == 0   ? "OFF, pixels correct"
                    : phase == 1 ? "repeat-band DRAWS skipped"
                                 : "repeat-band BUFFERS skipped");
      }
    }

    // [NR-4B] verify-price cycle: 10 s FULL verify / 10 s verify OFF, in
    // place, phases logged so the log pairs fps/us-per-draw at matched dpf.
    if (g_nr4b_cycle) {
      static auto nr4b_t0 = prof_clock::now();
      const int phase =
          int((std::chrono::duration_cast<std::chrono::seconds>(
                   prof_clock::now() - nr4b_t0)
                   .count() /
               10) %
              2);
      const int32_t want = phase == 0 ? 1 : 0;
      if (want != g_nr4b_verify_n) {
        g_nr4b_verify_n = want;
        REXGPU_INFO("[nr-4b] verify -> {}", want ? "FULL" : "OFF");
      }
    }

    // [NR-5C] the fps-increment pricing cycle: 10 s per phase, in place --
    // 0 = both levers off (baseline), 1 = apply-from-arena, 2 = arena +
    // skip-unchanged (mode 2). Pair [n7] fps at matched dpf across phases.
    if (g_nr5c_cycle) {
      static auto nr5c_t0 = prof_clock::now();
      const int32_t phase =
          int32_t((std::chrono::duration_cast<std::chrono::seconds>(
                       prof_clock::now() - nr5c_t0)
                       .count() /
                   10) %
                  3);
      if (phase != g_nr5c_phase) {
        g_nr5c_phase = phase;
        g_nr5c_arena_on = phase >= 1;
        g_nr5c_skip_mode = phase == 2 ? 2 : 0;
        REXGPU_INFO("[nr-5c] PHASE {} - {}", phase,
                    phase == 0   ? "OFF (baseline: scratch memcpy, full applies)"
                    : phase == 1 ? "ARENA (apply in place, no memcpy)"
                                 : "ARENA+SKIP (clean record runs skipped)");
      }
    }

    // [NR-6] the repeat-band drop cycle: 10 s per class, in place, phase
    // logged so the eye and the log agree ([[visual-defect-needs-a-visual-
    // bisect]]). 0 = off (baseline), 1 = ALU + bool/loop dropped, 2 = +
    // fetch, 3 = + state windows.
    if (g_nr6_cycle) {
      static auto nr6_t0 = prof_clock::now();
      const int32_t phase =
          int32_t((std::chrono::duration_cast<std::chrono::seconds>(
                       prof_clock::now() - nr6_t0)
                       .count() /
                   10) %
                  4);
      if (phase != g_nr6_phase) {
        g_nr6_phase = phase;
        g_nr6_mode = phase;
        REXGPU_INFO("[nr-6] PHASE {} - {}", phase,
                    phase == 0   ? "OFF (baseline: repeat bands compose)"
                    : phase == 1 ? "DROP alu+bool (fetch/state still apply)"
                    : phase == 2 ? "DROP alu+bool+fetch"
                                 : "DROP alu+bool+fetch+state (everything)");
      }
    }

    // [NR-97] the repeat-band walk-skip cycle: 10 s off, 10 s on, in place.
    // Two phases only -- this lever has no classes, it either applies a band's
    // register stream or it does not. The census counts in BOTH phases, so the
    // OFF phase reports exactly the traffic and the risk the ON phase carried
    // ([[price-stages-in-place]]).
    if (g_n97_cycle) {
      static auto n97_t0 = prof_clock::now();
      const int32_t phase =
          int32_t((std::chrono::duration_cast<std::chrono::seconds>(
                       prof_clock::now() - n97_t0)
                       .count() /
                   10) %
                  2);
      if (phase != g_n97_phase) {
        g_n97_phase = phase;
        g_n97_mode = phase;
        REXGPU_INFO("[nr-97] PHASE {} - {}", phase,
                    phase == 0 ? "OFF (baseline: repeat bands apply in full)"
                               : "ON (repeat-band applies above 0x2000 DROPPED)");
      }
    }

    // [NR-5B-3i] the ALU-half visual bisect: 5 s per by-ref source, in place,
    // phase logged so the log names what the eye was looking at.
    if (g_nr5b_mk_cycle) {
      static auto nr5bmk_t0 = prof_clock::now();
      const int32_t phase =
          int32_t((std::chrono::duration_cast<std::chrono::seconds>(
                       prof_clock::now() - nr5bmk_t0)
                       .count() /
                   5) %
                  3);
      if (phase != g_nr5b_mk_mode) {
        g_nr5b_mk_mode = phase;
        REXGPU_INFO("[nr-5b] mk source -> {} ({})", phase,
                    phase == 0   ? "RECORD address (today)"
                    : phase == 1 ? "DROP markers"
                                 : "LIVE address, record position");
      }
    }

    // [NR-5B-3j] the fallback bisect: 5 s issuing declined draws (today) /
    // 5 s skipping them, in place. Flashing that turns into ABSENCE on the
    // beat names the pool.
    if (g_nr5b_fbcycle) {
      static auto nr5bfb_t0 = prof_clock::now();
      const int32_t phase =
          int32_t((std::chrono::duration_cast<std::chrono::seconds>(
                       prof_clock::now() - nr5bfb_t0)
                       .count() /
                   5) %
                  2);
      if (phase != g_nr5b_fbmode) {
        g_nr5b_fbmode = phase;
        REXGPU_INFO("[nr-5b] declined draw -> {}",
                    phase ? "SKIPPED" : "ISSUED (today)");
      }
    }

    // [NR-5B-3k] the generation bisect: 5 s counting only / 5 s skipping the
    // ahead class, in place. Flashing that becomes ABSENCE on the beat names
    // the class the 5a draw anchor cannot see.
    if (g_nr5b_gencycle) {
      static auto nr5bgen_t0 = prof_clock::now();
      const int32_t phase =
          int32_t((std::chrono::duration_cast<std::chrono::seconds>(
                       prof_clock::now() - nr5bgen_t0)
                       .count() /
                   5) %
                  2);
      if (phase != g_nr5b_genmode) {
        g_nr5b_genmode = phase;
        REXGPU_INFO("[nr-5b] ahead record -> {}",
                    phase ? "SKIPPED" : "COMPOSED (today)");
      }
    }

    // [N7] The re-baseline line. Prints on a DEFAULT run, so every drive from
    // here on carries its own ceiling and nothing about N-8..N-12 has to be
    // judged against a number taken before de-tile.
    if (kN7) {
      const auto n7_now = prof_clock::now();
      const uint64_t n7_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      n7_now - n7_last_report).count();
      if (n7_wall_ns >= 1000000000ull) {
        n7_last_report = n7_now;
        const double wall_ms = n7_wall_ns / 1e6;
        const uint64_t frames_now = swap_counter_.load(std::memory_order_relaxed);
        const uint64_t frames = frames_now - n7_last_frames;
        const uint64_t draws = g_n7_draws - n7_last_draws;
        const double spin_ms = (prof_spin_ns - n7_last_spin_ns) / 1e6;
        const double exec_ms = (prof_exec_ns - n7_last_exec_ns) / 1e6;
        const double fps = 1000.0 * double(frames) / wall_ms;
        const double exec_pct = 100.0 * exec_ms / wall_ms;
        // What this consumer implies it could deliver: it spent exec_pct of the
        // wall clock producing `fps` frames, so fps/exec_pct*100 is where IT
        // becomes the wall. Only meaningful below ~99% -- at 100% the consumer
        // IS the wall and the ceiling is the fps printed beside it.
        const double ceiling = exec_pct > 1.0 ? fps * 100.0 / exec_pct : 0.0;
        REXGPU_INFO(
            "[n7] fps={:.1f} dpf={} | cp exec={:.1f}% spin={:.1f}% -> "
            "ceiling~{:.0f}fps | wakes={} backlog avg={}dw max={}dw | "
            "draws/s={} exec_calls={} starves={}",
            fps, frames ? (draws / frames) : draws, exec_pct,
            100.0 * spin_ms / wall_ms, ceiling, n7_wakes,
            n7_wakes ? (n7_backlog_sum / n7_wakes) : uint64_t(0), n7_backlog_max,
            draws, prof_exec_calls - n7_last_exec_calls,
            prof_starves - n7_last_starves);
        // The split, when this window was a SPLIT-ON phase. draw and exec are
        // clean brackets; type0 carries ~3-8% of its own self-inflation, so
        // other% is a floor. This is the number that orders N-8 against N-9.
        if (g_split_prof && exec_ms > 0.0) {
          const double draw_ms = (g_issuedraw_ns - n7_last_issuedraw_ns) / 1e6;
          const double t0_ms = (g_type0_split_ns - n7_last_type0_ns) / 1e6;
          const double other_ms = exec_ms - draw_ms - t0_ms;
          REXGPU_INFO(
              "[n7] split exec={:.1f}ms draw={:.1f}ms({:.0f}%) "
              "type0/regwrite={:.1f}ms({:.0f}%) other={:.1f}ms({:.0f}%) draws={}",
              exec_ms, draw_ms, 100.0 * draw_ms / exec_ms, t0_ms,
              100.0 * t0_ms / exec_ms, other_ms, 100.0 * other_ms / exec_ms,
              g_draw_cnt - n7_last_draw_cnt);
        }
        // The skip path's own halves, when this window was a SKIP-ON phase.
        // buf = the whole skip-driven buffer; drawstop and delegstop are the
        // two dispatch brackets; the remainder is the walk decode plus the
        // bulk range applies. Under gpu_nr_skip the register firehose does NOT
        // go through ExecutePacketType0 any more -- it is the walk -- which is
        // why the split line's type0 bucket reads 0 and this line is the one
        // that prices state decoding.
        if (g_skp_prof && g_spp_buf_ns) {
          const double b_ms = g_spp_buf_ns / 1e6;
          const double d_ms = g_spp_draw_ns / 1e6;
          const double g_ms = g_spp_deleg_ns / 1e6;
          // [NR-5C-ATTR] the compose sits INSIDE the buffer bracket and
          // OUTSIDE both stop brackets, so it is part of this line's
          // remainder: print it and the true walk+bulk (remainder - compose)
          // so no one has to subtract two lines by hand.
          const double c_ms =
              g_nr5c_prof ? double(g_n7_cmp_ns[0] + g_n7_cmp_ns[1]) / 1e6 : 0.0;
          REXGPU_INFO(
              "[n7] skip buf={:.1f}ms drawstop={:.1f}ms delegstop={:.1f}ms "
              "walk+bulk={:.1f}ms (wall={:.0f}ms, exec={:.1f}ms)",
              b_ms, d_ms, g_ms, b_ms - d_ms - g_ms - c_ms, wall_ms, exec_ms);
          if (g_nr5c_prof) {
            REXGPU_INFO(
                "[n7] skip   compose={:.1f}ms sup={:.1f}ms({} calls, "
                "{:.0f}ns) pkt={:.1f}ms({} calls, {:.0f}ns) | drawstop "
                "sup={:.1f}ms({}) pkt={:.1f}ms({})",
                c_ms, double(g_n7_cmp_ns[1]) / 1e6, g_n7_cmp_calls[1],
                g_n7_cmp_calls[1]
                    ? double(g_n7_cmp_ns[1]) / double(g_n7_cmp_calls[1])
                    : 0.0,
                double(g_n7_cmp_ns[0]) / 1e6, g_n7_cmp_calls[0],
                g_n7_cmp_calls[0]
                    ? double(g_n7_cmp_ns[0]) / double(g_n7_cmp_calls[0])
                    : 0.0,
                double(g_spp_draw_bin_ns[1]) / 1e6, g_spp_draw_bin_n[1],
                double(g_spp_draw_bin_ns[0]) / 1e6, g_spp_draw_bin_n[0]);
          }
        }
        g_spp_buf_ns = g_spp_draw_ns = g_spp_deleg_ns = 0;
        g_spp_draw_bin_ns[0] = g_spp_draw_bin_ns[1] = 0;
        g_spp_draw_bin_n[0] = g_spp_draw_bin_n[1] = 0;
        g_n7_cmp_ns[0] = g_n7_cmp_ns[1] = 0;
        g_n7_cmp_calls[0] = g_n7_cmp_calls[1] = 0;
        // [N7] the ring remainder, itemised. ib = every indirect buffer;
        // swap = the presenter and frame-end work; inline = what is left in
        // ExecutePrimaryBuffer, which is the primary ring's own packets.
        // The swap packet is emitted INSIDE an indirect buffer, so its
        // bracket NESTS in the ib bracket rather than sitting beside it.
        // Subtracting both from exec printed a negative inline figure in the
        // first build; outside-ib is exec - ib, and swap is a sub-item of ib.
        {
          const double ib_ms = (g_n7_ib_ns - n7_last_ib_ns) / 1e6;
          const double sw_ms = (g_n7_swap_ns - n7_last_swap_ns) / 1e6;
          const double inl_ms = exec_ms - ib_ms;
          REXGPU_INFO(
              "[n7] ring exec={:.1f}ms | indirect_buffers={:.1f}ms({:.0f}%) n={} "
              "| of which swap={:.1f}ms({:.0f}%) n={} | outside ib={:.1f}ms({:.0f}%)",
              exec_ms, ib_ms, exec_ms > 0 ? 100.0 * ib_ms / exec_ms : 0.0,
              g_n7_ib_calls - n7_last_ib_calls, sw_ms,
              exec_ms > 0 ? 100.0 * sw_ms / exec_ms : 0.0,
              g_n7_swap_calls - n7_last_swap_calls, inl_ms,
              exec_ms > 0 ? 100.0 * inl_ms / exec_ms : 0.0);
          n7_last_ib_ns = g_n7_ib_ns; n7_last_swap_ns = g_n7_swap_ns;
          n7_last_ib_calls = g_n7_ib_calls; n7_last_swap_calls = g_n7_swap_calls;
        }
        // [N8] the applied-dword census. This is the number that says whether
        // N-8 is about render state at all.
        {
          const uint64_t tot = g_n8_dw_low + g_n8_dw_state + g_n8_dw_alu +
                               g_n8_dw_fetch + g_n8_dw_boolloop;
          if (tot) {
            const double pc = 100.0 / double(tot);
            REXGPU_INFO(
                "[n8] applied dwords/s={} | alu_const={:.1f}% fetch_const={:.1f}% "
                "bool/loop={:.1f}% STATE(0x2000-0x3FFF)={:.1f}% low(<0x2000)={:.1f}% "
                "| ranges={} per_dword={} | state dwords/frame={}",
                tot, g_n8_dw_alu * pc, g_n8_dw_fetch * pc, g_n8_dw_boolloop * pc,
                g_n8_dw_state * pc, g_n8_dw_low * pc, g_n8_rng, g_n8_pdw,
                frames ? (g_n8_dw_state + g_n8_dw_low) / frames : 0);
            // The top state registers: the N-8 target list, named. If these are
            // the D3D9 shadow words (RB_DEPTHCONTROL 0x2200, RB_BLENDCONTROL0
            // 0x2201, RB_COLOR_MASK 0x2104) the setter family is the right
            // wedge; if they are VGT/PA/SQ plumbing it is not.
            if (g_n8_hist) {
              char hb[512];
              int hn = 0;
              for (int k = 0; k < 10; ++k) {
                uint32_t best = 0, best_r = 0;
                for (uint32_t r = 0; r < kN8HistRegs; ++r) {
                  if (g_n8_hist[r] > best) { best = g_n8_hist[r]; best_r = r; }
                }
                if (!best) break;
                const RegisterInfo* ri = RegisterFile::GetRegisterInfo(best_r);
                hn += snprintf(hb + hn, sizeof(hb) - hn, " %s(%04X)=%u",
                               ri && ri->name ? ri->name : "?", best_r, best);
                g_n8_hist[best_r] = 0;
                if (hn >= int(sizeof(hb)) - 48) break;
              }
              REXGPU_INFO("[n8] top state registers by applied dwords:{}", hb);
              for (uint32_t r = 0; r < kN8HistRegs; ++r) g_n8_hist[r] = 0;
            }
          }
          g_n8_dw_low = g_n8_dw_state = g_n8_dw_alu = 0;
          g_n8_dw_fetch = g_n8_dw_boolloop = 0;
          g_n8_rng = g_n8_pdw = 0;
        }
        // [N8D] the contiguity census, and with it the merge ceiling. A range
        // that begins where the previous one ended is one a look-ahead merge
        // could absorb, so `groups` is what the range count would fall to.
        // The split probe does NOT pollute this: the census runs on the walk's
        // range, before the split.
        {
          uint64_t tr = 0, tc = 0, tm = 0, tp = 0;
          for (int c = 0; c < 3; ++c) {
            tr += g_n8d_rng[c]; tc += g_n8d_creg[c];
            tm += g_n8d_cmem[c]; tp += g_n8d_repeat[c];
          }
          if (tr) {
            static const char* const kC[3] = {"alu", "fetch", "other"};
            const double pc = 100.0 / double(tr);
            const uint64_t groups = tr - tc;
            char cb[320];
            int cn = 0;
            for (int c = 0; c < 3; ++c) {
              cn += snprintf(cb + cn, sizeof(cb) - cn,
                             " %s=%.0f%%/%.0f%%", kC[c],
                             g_n8d_rng[c] ? 100.0 * double(g_n8d_creg[c]) / double(g_n8d_rng[c]) : 0.0,
                             g_n8d_rng[c] ? 100.0 * double(g_n8d_cmem[c]) / double(g_n8d_rng[c]) : 0.0);
            }
            REXGPU_INFO(
                "[n8d] split(alu{},ftc{},oth{}) walk ranges/draw={:.2f} "
                "APPLIED ranges/draw={:.2f} | contiguous reg={:.1f}% "
                "reg+mem={:.1f}% repeat-base={:.1f}% -> merge groups/draw="
                "{:.2f} (ceiling -{:.0f}% of ranges) | by class reg%/mem%:{}",
                g_n8c_split_alu, g_n8c_split_fetch, g_n8c_split_other,
                draws ? double(tr) / double(draws) : 0.0,
                draws ? double(g_n8d_applied) / double(draws) : 0.0,
                tc * pc, tm * pc, tp * pc,
                draws ? double(groups) / double(draws) : 0.0, tc * pc, cb);
          }
          for (int c = 0; c < 3; ++c) {
            g_n8d_rng[c] = g_n8d_creg[c] = g_n8d_cmem[c] = g_n8d_repeat[c] = 0;
          }
          g_n8d_applied = 0;
        }
        // [N8E] the per-draw union: what the range count would fall to if a
        // draw's constant ranges were accumulated and applied ONCE at the draw
        // stop. This is order-independent, so it survives the adjacency
        // census reading zero.
        if (g_n8e_draws) {
          static const char* const kC[3] = {"alu", "fetch", "other"};
          const double d = double(g_n8e_draws);
          char eb[384];
          int en = 0;
          double tr = 0, tg = 0;
          for (int c = 0; c < 3; ++c) {
            tr += double(g_n8e_rng[c]); tg += double(g_n8e_grp[c]);
            en += snprintf(eb + en, sizeof(eb) - en,
                           " %s %.2f->%.2f (dw %.0f->%.0f)", kC[c],
                           double(g_n8e_rng[c]) / d, double(g_n8e_grp[c]) / d,
                           double(g_n8e_dw[c]) / d, double(g_n8e_udw[c]) / d);
            if (en >= int(sizeof(eb)) - 64) break;
          }
          REXGPU_INFO(
              "[n8e] per-draw union: ranges {:.2f} -> applies {:.2f} "
              "(-{:.0f}%) | overflowed draws={} |{}",
              tr / d, tg / d, tr > 0 ? 100.0 * (1.0 - tg / tr) : 0.0,
              g_n8e_overflow, eb);
          g_n8e_draws = g_n8e_overflow = 0;
          for (int c = 0; c < 3; ++c) {
            g_n8e_rng[c] = g_n8e_grp[c] = g_n8e_dw[c] = g_n8e_udw[c] = 0;
          }
        }
        // [N8F] the coalescer's own economics, printed for any window it ran
        // in. deferred = ranges taken by the fast path; applies = merged
        // spans flushed (the number that replaced them); ovf/straddle must
        // stay ~0 or the span table / window model is wrong.
        if (g_n8f_rng) {
          REXGPU_INFO(
              "[n8f] deferred rng/draw={:.2f} -> effect applies/draw={:.2f} "
              "(-{:.0f}%) | flushes={} ovf={} straddle={}",
              draws ? double(g_n8f_rng) / double(draws) : 0.0,
              draws ? double(g_n8f_applies + g_n8f_ovf) / double(draws) : 0.0,
              g_n8f_rng ? 100.0 * (1.0 - double(g_n8f_applies + g_n8f_ovf) /
                                             double(g_n8f_rng))
                        : 0.0,
              g_n8f_flushes, g_n8f_ovf, g_n8f_straddle);
          g_n8f_rng = g_n8f_applies = g_n8f_flushes = g_n8f_ovf = 0;
          g_n8f_straddle = 0;
        }
        n7_last_frames = frames_now;
        n7_last_draws = g_n7_draws;
        n7_last_spin_ns = prof_spin_ns;
        n7_last_exec_ns = prof_exec_ns;
        n7_last_exec_calls = prof_exec_calls;
        n7_last_starves = prof_starves;
        n7_last_issuedraw_ns = g_issuedraw_ns;
        n7_last_type0_ns = g_type0_split_ns;
        n7_last_draw_cnt = g_draw_cnt;
        n7_backlog_sum = n7_backlog_max = n7_wakes = 0;
        // Mode 2: cycle the split IN PLACE, 5 s per phase, toggled here so a
        // report window never straddles a phase boundary.
        // Three phases, 5 s each, ONE profiler at a time: a stage profiler
        // running inside another one overstates it ([[nested-profile-inflates-
        // stage]]), and fps must be read from the phase with neither.
        //   0 CLEAN  - fps, occupancy, backlog, nothing armed
        //   1 SPLIT  - IssueDraw vs the rest of exec (~2% self-cost)
        //   2 SKIP   - the skip path's halves: walk+bulk vs the two dispatches
        //              (~2 timestamps per stop, ~3% at city load)
        if (kN7 == 2 && !kSplit && !kSkpProf) {
          const int64_t el = std::chrono::duration_cast<std::chrono::seconds>(
                                 n7_now - n7_t0).count();
          const int phase = int((el / 5) % 3);
          static const char* const kPhaseName[3] = {"CLEAN", "SPLIT", "SKIP"};
          static int n7_phase_prev = -1;
          if (phase != n7_phase_prev) {
            n7_phase_prev = phase;
            g_split_prof = (phase == 1);
            g_skp_prof = (phase == 2);
            REXGPU_INFO(
                "[n7] PHASE {} - {} (read fps from CLEAN; the other two carry "
                "their own ~2-3% self-cost).",
                kPhaseName[phase],
                phase == 0   ? "nothing armed"
                : phase == 1 ? "IssueDraw bracket armed"
                             : "skip-path brackets armed");
          }
        }
        // Mode 4: N-8c, the synthetic-split probe. Four 5 s phases IN PLACE.
        // Every phase is pixel-correct - splitting a range into k sub-ranges
        // leaves the register file and every invalidation identical, only the
        // range COUNT changes. Read us/draw per phase, subtract CLEAN, and
        // divide by the added ranges per draw ([n8d] prints ranges/draw in
        // every phase): that quotient IS the marginal cost of a range, per
        // class, and it is the exact quantity a look-ahead merge recovers.
        //   0 CLEAN     nothing split
        //   1 ALU x4    the 9.3 alu ranges/draw become ~37
        //   2 FETCH x4  the 4.2 fetch ranges/draw become ~17
        //   3 ALL x4    linearity check: should be ~ the sum of 1 and 2 plus
        //               the other classes. If it is not, the cost is not
        //               per-range and this whole rung is wrong.
        if (kN7 == 4) {
          const int64_t el = std::chrono::duration_cast<std::chrono::seconds>(
                                 n7_now - n7_t0).count();
          const int phase = int((el / 5) % 4);
          static const char* const kSplitName[4] = {"CLEAN", "ALU x4",
                                                    "FETCH x4", "ALL x4"};
          static int split_phase_prev = -1;
          if (phase != split_phase_prev) {
            split_phase_prev = phase;
            g_n8c_split_alu = (phase == 1 || phase == 3) ? 4 : 1;
            g_n8c_split_fetch = (phase == 2 || phase == 3) ? 4 : 1;
            g_n8c_split_other = (phase == 3) ? 4 : 1;
            REXGPU_INFO(
                "[n7] N8C PHASE {} - every phase is pixel-correct; the SLOPE "
                "of us/draw against ranges/draw is the answer.",
                kSplitName[phase]);
          }
        }
        // Mode 3: the N-8b constant census, three 5 s phases IN PLACE. This
        // one is not a profiler, it is a LEVER plus its own measurement, so
        // the three phases separate the three things a single number cannot:
        //   0 OFF           - no compare, the honest fps baseline
        //   1 CENSUS        - compare and count, act on nothing: this is what
        //                     the compare itself costs, and it is the number
        //                     that says whether a dedupe can ever pay
        //   2 CENSUS+DEDUPE - the whole-range early return fires, skipping the
        //                     register copy AND the cbuffer re-upload
        // fps is read from OFF against DEDUPE; CENSUS is the tax line between.
        if (kN7 == 3) {
          const int64_t el = std::chrono::duration_cast<std::chrono::seconds>(
                                 n7_now - n7_t0).count();
          const int phase = int((el / 5) % 3);
          static const char* const kN8bName[3] = {"OFF", "CENSUS",
                                                  "CENSUS+DEDUPE"};
          static int n8b_phase_prev = -1;
          if (phase != n8b_phase_prev) {
            n8b_phase_prev = phase;
            g_n7_n8b_phase = phase;
            REXGPU_INFO(
                "[n7] N8B PHASE {} - {} (fps: compare OFF against "
                "CENSUS+DEDUPE; CENSUS is the compare's own tax).",
                kN8bName[phase],
                phase == 0   ? "no constant compare at all"
                : phase == 1 ? "compare and count, act on nothing"
                             : "unchanged constant ranges skipped entirely");
          }
        }
        // Mode 5: N-8f, per-draw effect coalescing, two 5 s phases IN PLACE.
        // Both phases are pixel-correct (values are stored identically; only
        // when the side effects fire changes, and every consumer of them is
        // the draw), so the paired fps delta IS the rung's worth.
        //   0 CLEAN     the legacy per-range applies
        //   1 COALESCE  values stored at decode, effects once per merged
        //               span at the draw stop
        if (kN7 == 5) {
          const int64_t el = std::chrono::duration_cast<std::chrono::seconds>(
                                 n7_now - n7_t0).count();
          const int phase = int((el / 5) % 2);
          static int n8f_phase_prev = -1;
          if (phase != n8f_phase_prev) {
            n8f_phase_prev = phase;
            if (g_n8f_eligible) {
              g_n8f_on = phase == 1;
              REXGPU_INFO(
                  "[n7] N8F PHASE {} - both phases pixel-correct; pair "
                  "adjacent phases for the fps answer.",
                  phase == 1 ? "COALESCE" : "CLEAN");
            } else if (n8f_phase_prev == phase) {
              REXGPU_INFO("[n8f] cycler armed but INELIGIBLE (see the boot "
                          "[n8f] line) - phases will not fire");
            }
          }
        }
        // Mode 6: N-9, whole-buffer plan replay, two 5 s phases IN PLACE.
        // Both phases are decode-exact (the interpreter fires the same
        // callbacks the parse would, values read live), so the paired fps
        // delta IS the rung's worth. Store survives CLEAN phases, so PROG
        // phases after the first serve warm.
        //   0 CLEAN  the live per-packet parse
        //   1 PROG   compiled flat plans, guards checked per execution
        if (kN7 == 6) {
          const int64_t el = std::chrono::duration_cast<std::chrono::seconds>(
                                 n7_now - n7_t0).count();
          const int phase = int((el / 5) % 2);
          static int n9_phase_prev = -1;
          if (phase != n9_phase_prev) {
            n9_phase_prev = phase;
            if (g_n9_eligible) {
              g_n9_on = phase == 1;
              REXGPU_INFO(
                  "[n7] N9 PHASE {} - both phases decode-exact; pair "
                  "adjacent phases for the fps answer.",
                  phase == 1 ? "PROG" : "CLEAN");
            } else {
              REXGPU_INFO("[n9] cycler armed but INELIGIBLE (see the boot "
                          "[n9] line) - phases will not fire");
            }
          }
        }
      }
    }

    // [GPU-WORKER-PROFILE] / [GPU-SPLIT] / [PM4-CENSUS] ~1s wall-time report.
    // [N8D] kNrSkip keeps the per-second [nr-skp]/[nr-ctx] watchdog lines on
    // default runs -- they used to ride the always-on kIbLedger, and losing
    // exec_fail/orphan/diverge visibility is a debuggability regression.
    // Inner prints stay gated on their own flags, so this adds no new noise.
    if (kProfile || kSplit || kCensus || kIbLedger || kNrSdb || kSkpProf ||
        kNrSkip) {
      auto now = prof_clock::now();
      uint64_t wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             now - prof_last_report).count();
      if (wall_ns >= 1000000000ull) {
        double wall_ms = wall_ns / 1e6;
        double spin_ms = (prof_spin_ns - last_spin_ns) / 1e6;
        double exec_ms = (prof_exec_ns - last_exec_ns) / 1e6;
        if (kProfile) {
          REXGPU_INFO(
              "[gpu-instr] wall={:.0f}ms spin={:.1f}ms({:.0f}%) exec={:.1f}ms({:.0f}%) "
              "exec_calls={} starves={} spin_loops={} event_waits={}",
              wall_ms, spin_ms, 100.0 * spin_ms / wall_ms, exec_ms,
              100.0 * exec_ms / wall_ms, prof_exec_calls - last_exec_calls,
              prof_starves - last_starves, prof_spin_loops - last_spin_loops,
              prof_event_waits - last_event_waits);
          REXGPU_INFO(
              "[gpu-exec] reg(t0)={:.1f}ms/{} t3={:.1f}ms/{} of-which-issuedraw={:.1f}ms/{} "
              "t1={:.1f}ms t2={:.1f}ms",
              (g_exec_type_ns[0] - last_type_ns[0]) / 1e6, g_exec_type_cnt[0] - last_type_cnt[0],
              (g_exec_type_ns[3] - last_type_ns[3]) / 1e6, g_exec_type_cnt[3] - last_type_cnt[3],
              (g_issuedraw_ns - last_issuedraw_ns) / 1e6, g_draw_cnt - last_draw_cnt,
              (g_exec_type_ns[1] - last_type_ns[1]) / 1e6,
              (g_exec_type_ns[2] - last_type_ns[2]) / 1e6);
        }
        if (kSplit && exec_ms > 0.0) {
          // Clean split of exec time: draw work (IssueDraw) vs type-0 register/
          // constant-write firehose vs other (non-draw type3: constant-loads,
          // indirect-buffer, waits, parse). draw+exec are ~clean; type0 carries
          // ~3-8% self-inflation from its own 1.6M/s brackets, so other% is the
          // floor for non-draw cost. Decides: parallelize draws vs cut firehose.
          double draw_ms = (g_issuedraw_ns - last_issuedraw_ns) / 1e6;
          double type0_ms = (g_type0_split_ns - last_type0_split_ns) / 1e6;
          double other_ms = exec_ms - draw_ms - type0_ms;
          REXGPU_INFO(
              "[gpu-split] exec={:.1f}ms  draw={:.1f}ms({:.0f}%)  type0/regwrite={:.1f}ms({:.0f}%)  "
              "other={:.1f}ms({:.0f}%)  draws={} (type0 ~self-inflated; draw+exec clean)",
              exec_ms, draw_ms, 100.0 * draw_ms / exec_ms, type0_ms,
              100.0 * type0_ms / exec_ms, other_ms, 100.0 * other_ms / exec_ms,
              g_draw_cnt - last_draw_cnt);
        }
        if (kCensus) {
          // [PM4-CENSUS] per-second draw-emission census (counters reset each report).
          const uint64_t di_p = g_pm4_draw_indx_primary, di_i = g_pm4_draw_indx_indirect;
          const uint64_t d2_p = g_pm4_draw_indx2_primary, d2_i = g_pm4_draw_indx2_indirect;
          const uint64_t ibc = g_pm4_ib_count, ibd = g_pm4_ib_dwords;
          const uint64_t total = di_p + di_i + d2_p + d2_i;
          const double sec = wall_ns / 1e9;
          REXGPU_INFO(
              "[pm4-census] draws/s={:.0f} | DRAW_INDX p={} i={} | DRAW_INDX_2 p={} i={} | "
              "in-indirect={:.0f}% | indirect_buffers/s={:.0f} avg_dwords={}",
              sec > 0 ? total / sec : 0.0, di_p, di_i, d2_p, d2_i,
              total ? 100.0 * double(di_i + d2_i) / double(total) : 0.0,
              sec > 0 ? ibc / sec : 0.0, ibc ? (ibd / ibc) : uint64_t(0));
          g_pm4_draw_indx_primary = 0; g_pm4_draw_indx_indirect = 0;
          g_pm4_draw_indx2_primary = 0; g_pm4_draw_indx2_indirect = 0;
          g_pm4_ib_count = 0; g_pm4_ib_dwords = 0;
        }
        if (kIbLedger) {
          // [NR-IBL] One line per distinct indirect buffer executed this second,
          // busiest first, plus the totals the gate actually compares against.
          // Sort by index (the table is small and this runs once a second).
          uint32_t order[kIbLedgerSize];
          uint32_t n = 0;
          uint64_t tot_exec = 0, tot_draw_exec = 0, tot_draw2_exec = 0;
          for (uint32_t i = 0; i < kIbLedgerSize; ++i) {
            if (!g_ib_ledger_tab[i].execs) continue;
            order[n++] = i;
            tot_exec += g_ib_ledger_tab[i].execs;
            tot_draw_exec += g_ib_ledger_tab[i].execs * g_ib_ledger_tab[i].draws;
            tot_draw2_exec += g_ib_ledger_tab[i].execs * g_ib_ledger_tab[i].draws2;
          }
          for (uint32_t a = 0; a + 1 < n; ++a) {
            for (uint32_t b = a + 1; b < n; ++b) {
              if (g_ib_ledger_tab[order[b]].execs > g_ib_ledger_tab[order[a]].execs) {
                uint32_t t = order[a]; order[a] = order[b]; order[b] = t;
              }
            }
          }
          const double sec = wall_ns / 1e9;
          // draws_executed counts DRAW_INDX only, which is what the guest hooks
          // see; DRAW_INDX_2 is reported beside it so a mismatch caused by the
          // draw type we do not hook cannot be mistaken for a failed gate.
          REXGPU_INFO("[nr-ibl] distinct_buffers={} executions={} draws_executed={} "
                      "({:.0f}/s) draw_indx2={} overflow={}",
                      n, tot_exec, tot_draw_exec + tot_draw2_exec,
                      sec > 0 ? (tot_draw_exec + tot_draw2_exec) / sec : 0.0,
                      tot_draw2_exec, g_ib_ledger_evictions);
          const uint32_t kShow = 256;  // all of them; the correlator needs every buffer
          for (uint32_t k = 0; k < n && k < kShow; ++k) {
            const IbLedgerEntry& e = g_ib_ledger_tab[order[k]];
            REXGPU_INFO("[nr-ibl]   buf={:08X}..{:08X} dwords={} draws={}+{} execs={} reg={}",
                        e.ptr, e.ptr + e.dwords * 4, e.dwords, e.draws, e.draws2,
                        e.execs, e.reg);
          }
          if (n > kShow) REXGPU_INFO("[nr-ibl]   ... {} more not shown", n - kShow);

          // [NR-REG] The gate verdict. Every scored execution asked the registry
          // how many draws were recorded into the range it was about to run; the
          // buffer's own DRAW_INDX packets say how many are really there. 'one'
          // is the pass bucket. Read it together with alias and split: a wrong
          // address alias shows up as zero everywhere, and a recording caught
          // half-overwritten shows up as split.
          {
            const uint64_t q = g_reg_queries;
            const double pc = q ? 100.0 / double(q) : 0.0;
            REXGPU_INFO("[nr-reg] queries={} one={:.1f}% under={:.1f}% over={:.1f}% "
                        "zero={:.1f}% split={:.1f}%",
                        q, g_reg_r_one * pc, g_reg_r_under * pc, g_reg_r_over * pc,
                        g_reg_r_zero * pc, g_reg_split * pc);
            const nr::Stats& rs = nr::GetStats();
            REXGPU_INFO("[nr-reg]   draws registry={} ledger={} ratio={:.3f} | "
                        "granules hit={} miss={} | recorded={} collisions={}",
                        g_reg_draws_registry, g_reg_draws_ledger,
                        g_reg_draws_ledger
                            ? double(g_reg_draws_registry) / double(g_reg_draws_ledger)
                            : 0.0,
                        g_reg_gran_hit, g_reg_gran_miss, rs.recorded, rs.collisions);
            // Which physical alias the recorder's pointers arrived through. If
            // this is not the alias the buffer addresses live in, the mask is
            // wrong and every other number here is meaningless.
            for (uint32_t a = 0; a < 8; ++a) {
              if (rs.alias[a]) {
                REXGPU_INFO("[nr-reg]   alias {:08X} -> {} draws", a << 29, rs.alias[a]);
              }
            }
            nr::ResetStats();
            g_reg_queries = 0;
            g_reg_r_one = g_reg_r_under = g_reg_r_over = g_reg_r_zero = 0;
            g_reg_split = 0;
            g_reg_draws_registry = g_reg_draws_ledger = 0;
            g_reg_gran_hit = g_reg_gran_miss = 0;
          }
          // [NR-CACHE] The per-packet join verdict. 'hit' is the build-out
          // bucket: every executed DRAW_INDX found its hook context at its
          // own header address (success bar: city hit >= 95%, arg_eq ~100%).
          // Samples name any systematic difference.
          // ([NR-CMP] snapshots the stats first: the block below resets them
          // and the compose report still needs the store counters.)
          const nr::CacheStats nr_cs_win = nr::GetCacheStats();
          if (g_nr_cache && g_nrc_execs) {
            const uint64_t known = g_nrc_hit - g_nrc_rid0;
            REXGPU_INFO(
                "[nr-cache] execs={} full={:.1f}% | pkts={} hit={:.1f}% "
                "miss={} | args eq={:.2f}% prim_ne={} cnt_ne={} rid0={}",
                g_nrc_execs, 100.0 * double(g_nrc_execs_full) / double(g_nrc_execs),
                g_nrc_pkts,
                g_nrc_pkts ? 100.0 * double(g_nrc_hit) / double(g_nrc_pkts) : 0.0,
                g_nrc_miss,
                known ? 100.0 * double(g_nrc_arg_eq) / double(known) : 0.0,
                g_nrc_prim_ne, g_nrc_cnt_ne, g_nrc_rid0);
            const nr::CacheStats& cs = nr::GetCacheStats();
            REXGPU_INFO("[nr-cache]   cache recorded={} replaced={} evictions={}",
                        cs.recorded, cs.replaced, cs.evictions);
            for (uint32_t s = 0; s < g_nrc_samp_n; ++s) {
              const NrcSample& sm = g_nrc_samp[s];
              if (sm.miss) {
                REXGPU_INFO(
                    "[nr-cache]   MISS buf={:08X} pkt[{}]@{:08X} (prim={} n={})",
                    sm.buf, sm.i, sm.addr, sm.pkt_prim, sm.pkt_cnt);
              } else {
                REXGPU_INFO(
                    "[nr-cache]   MISMATCH buf={:08X} pkt[{}]@{:08X} "
                    "pkt(prim={} n={}) rec(rid={} prim={} start={} n={})",
                    sm.buf, sm.i, sm.addr, sm.pkt_prim, sm.pkt_cnt, sm.rec_rid,
                    sm.rec_prim, sm.rec_start, sm.rec_cnt);
              }
            }
            // [NR-CMP] moved the store stats into its own line below when
            // compose is on; ResetCacheStats() here clears both either way.
            nr::ResetCacheStats();
            g_nrc_execs = g_nrc_execs_full = 0;
            g_nrc_pkts = g_nrc_hit = g_nrc_miss = 0;
            g_nrc_arg_eq = g_nrc_prim_ne = g_nrc_cnt_ne = g_nrc_rid0 = 0;
            g_nrc_samp_n = 0;
          }
          // [NR-CMP] N-9-3d compose+diff verdict. The gate: DIVERGE
          // (val_ne + pred_only + emit_only) must read 0 at the city outside
          // the counted skip classes; scored% quotes the coverage the gate
          // actually ran over ([[swapped-out-gates-vacuous]]). stale= is the
          // arena ring lapping a record that stopped re-recording; torn= is
          // the known post-hook re-record window plus mid-diff overwrites.
          if (g_nr_compose && (g_cmp_groups || g_cmp_no_state || g_cmp_stale ||
                               g_cmp_torn || g_cmp_skip_prevmiss ||
                               g_cmp_skip_ovf)) {
            const uint64_t refused = g_cmp_no_state + g_cmp_stale + g_cmp_torn +
                                     g_cmp_skip_prevmiss + g_cmp_skip_ovf;
            const uint64_t offered = g_cmp_groups + refused;
            REXGPU_INFO(
                "[nr-cmp] groups={} scored={:.2f}% | runs pred={} emit={} "
                "run_eq={:.2f}% val_eq={:.2f}% | DIVERGE val_ne={} "
                "pred_only={} emit_only={} div_groups={} (first_in_buf={})",
                g_cmp_groups,
                offered ? 100.0 * double(g_cmp_groups) / double(offered) : 0.0,
                g_cmp_pred_runs, g_cmp_emit_runs,
                g_cmp_emit_runs
                    ? 100.0 * double(g_cmp_run_eq) / double(g_cmp_emit_runs)
                    : 0.0,
                g_cmp_emit_runs
                    ? 100.0 * double(g_cmp_val_eq) / double(g_cmp_emit_runs)
                    : 0.0,
                g_cmp_val_ne, g_cmp_pred_only, g_cmp_emit_only,
                g_cmp_div_groups, g_cmp_first_div);
            const nr::CacheStats& ss = nr_cs_win;
            REXGPU_INFO(
                "[nr-cmp]   skipped: no_state={} stale={} torn={} "
                "prevmiss={} ovf={} sup={} outside_runs={} | store: stored={} "
                "dwords={} ovf={} freed={} nomem={} badfree={} expired={} "
                "dormant_rejoin={} | live={} bump={}MB",
                g_cmp_no_state, g_cmp_stale, g_cmp_torn, g_cmp_skip_prevmiss,
                g_cmp_skip_ovf, g_cmp_skip_sup, g_cmp_outside, ss.state_stored,
                ss.state_dwords, ss.state_ovf, ss.state_freed, ss.state_nomem,
                ss.state_badfree, ss.state_expired, ss.dormant_rejoin,
                ss.state_live, (ss.state_bump_dw * 4) >> 20);
            // [N-9-4a] The ordering-census verdict: OVERLAP must read 0 for
            // the draw-stop compose apply (the N-9-5 shape) to be order-safe.
            REXGPU_INFO(
                "[nr-cmp]   byref: pkts={} groups={} type "
                "alu/fetch/bool/loop/reg/other={}/{}/{}/{}/{}/{} | OVERLAP={} "
                "interleaved={} drop={}",
                g_cmp_bref_pkts, g_cmp_bref_groups, g_cmp_bref_type[0],
                g_cmp_bref_type[1], g_cmp_bref_type[2], g_cmp_bref_type[3],
                g_cmp_bref_type[4], g_cmp_bref_type[5], g_cmp_bref_overlap,
                g_cmp_bref_ilv, g_cmp_bref_drop);
            if (g_cmp_bref_samp[2]) {
              REXGPU_INFO("[nr-cmp]   first overlap: reg={:#06X} cnt={}",
                          g_cmp_bref_samp[0], g_cmp_bref_samp[1]);
              g_cmp_bref_samp[2] = 0;
            }
            // [NR-4B-EXT] the marker-schema verdict: pred==emit, eq==pred,
            // ne/pos_ne/cnt_ne all 0 means a suppressed body could re-emit
            // this group's 0x2F packets from the record exactly.
            if (g_cmp_mk_pred || g_cmp_mk_emit || g_cmp_mk_ovf) {
              REXGPU_INFO(
                  "[nr-cmp]   mk: pred={} emit={} eq={} NE={} POS_NE={} "
                  "CNT_NE={} ovf={}",
                  g_cmp_mk_pred, g_cmp_mk_emit, g_cmp_mk_eq, g_cmp_mk_ne,
                  g_cmp_mk_pos_ne, g_cmp_mk_cnt_ne, g_cmp_mk_ovf);
              if (g_cmp_mk_samp[4]) {
                static const char* kMkKinds[4] = {"?", "val_ne", "pos_ne",
                                                  "cnt_ne"};
                REXGPU_INFO(
                    "[nr-cmp]   mk first: {} pred_pos={} emit_pos={} "
                    "dw1={:08X}",
                    kMkKinds[g_cmp_mk_samp[0] < 4 ? g_cmp_mk_samp[0] : 0],
                    g_cmp_mk_samp[1], g_cmp_mk_samp[2], g_cmp_mk_samp[3]);
                g_cmp_mk_samp[4] = 0;
              }
            }
            if (g_cmp_samp_armed) {
              static const char* kCmpKinds[4] = {"?", "pred_only", "emit_only",
                                                 "val_ne"};
              REXGPU_INFO(
                  "[nr-cmp]   first: {} reg={:#06X} pred_cnt={} emit_cnt={} "
                  "buf={:08X} pkt@{:08X}",
                  kCmpKinds[g_cmp_samp.kind < 4 ? g_cmp_samp.kind : 0],
                  g_cmp_samp.reg, g_cmp_samp.pred_cnt, g_cmp_samp.emit_cnt,
                  g_cmp_samp.buf, g_cmp_samp.pkt);
              g_cmp_samp_armed = false;
            }
            g_cmp_groups = g_cmp_pred_runs = g_cmp_emit_runs = 0;
            g_cmp_run_eq = g_cmp_val_eq = g_cmp_val_ne = 0;
            g_cmp_pred_only = g_cmp_emit_only = 0;
            g_cmp_no_state = g_cmp_stale = g_cmp_torn = 0;
            g_cmp_skip_prevmiss = g_cmp_skip_ovf = g_cmp_outside = 0;
            g_cmp_skip_sup = 0;
            g_cmp_div_groups = g_cmp_first_div = 0;
            g_cmp_bref_pkts = g_cmp_bref_groups = 0;
            for (uint32_t t = 0; t < 6; ++t) g_cmp_bref_type[t] = 0;
            g_cmp_bref_overlap = g_cmp_bref_ilv = g_cmp_bref_drop = 0;
            g_cmp_mk_pred = g_cmp_mk_emit = g_cmp_mk_eq = 0;
            g_cmp_mk_ne = g_cmp_mk_pos_ne = g_cmp_mk_cnt_ne = g_cmp_mk_ovf = 0;
          }
          // [NR-BUF] The serving verdict. served= is the fraction of
          // executions the renderer replays from a snapshot without
          // walking; vne= is the stale-serve gate and must stay ~0;
          // same= prices the epoch scheme's false invalidations.
          if (g_nr_bufcache && g_nrb_execs) {
            const double pc = 100.0 / double(g_nrb_execs);
            REXGPU_INFO(
                "[nr-buf] execs={} served={:.1f}% (vok={} VNE={}) absent={:.1f}% "
                "dirty={:.1f}% resized={:.1f}% ovf={} | draws served={:.1f}%",
                g_nrb_execs, g_nrb_served * pc, g_nrb_vok, g_nrb_vne,
                g_nrb_absent * pc, g_nrb_dirty * pc, g_nrb_resized * pc,
                g_nrb_walk_ovf,
                g_nrb_draws_total ? 100.0 * double(g_nrb_draws_served) /
                                        double(g_nrb_draws_total)
                                  : 0.0);
            const nr::BufCacheStats& bs = nr::GetBufCacheStats();
            REXGPU_INFO(
                "[nr-buf]   snaps admit={} same={} reject={} toobig={} "
                "displaced={} live={}",
                bs.admissions, bs.admissions_same, bs.rejects, bs.toobig,
                bs.displaced, bs.live);
            nr::ResetBufCacheStats();
            g_nrb_execs = g_nrb_served = g_nrb_vok = g_nrb_vne = 0;
            g_nrb_absent = g_nrb_dirty = g_nrb_resized = 0;
            g_nrb_draws_served = g_nrb_draws_total = 0;
            g_nrb_walk_ovf = 0;
          }
          // [NR-STATE] The recovery census. copy= is the resolve/clear rate
          // (draws under kCopy); mode_inh= are draws whose EDRAM mode the
          // buffer inherits from ring state; rt/vport= how many draws had
          // that state established IN the buffer. The miss split tests
          // whether the join misses are precisely the unhooked resolve
          // draws. other= registers name any classification hole.
          if (g_nr_state && g_nrs_execs) {
            const double dp =
                g_nrs_draws ? 100.0 / double(g_nrs_draws) : 0.0;
            REXGPU_INFO(
                "[nr-state] execs={} draws={} copy={} mode_inh={:.1f}% "
                "rt_set={:.1f}% vport_set={:.1f}% desync={} mode_mem={} | pkts "
                "t0={} t1={} t2={} t3={}",
                g_nrs_execs, g_nrs_draws, g_nrs_copy,
                g_nrs_mode_inherited * dp, g_nrs_rt_set * dp,
                g_nrs_vport_set * dp, g_nrs_desync, g_nrs_mode_mem, g_nrs_t0,
                g_nrs_t1, g_nrs_t2, g_nrs_t3);
            const uint64_t* rc = g_nrs_regs;
            REXGPU_INFO(
                "[nr-state]   regs const={} fetch={} bool={} rt={} vport={} "
                "sciss={} copy={} clear={} mode={} dev={} coher={} other={}",
                rc[nr::kRegShaderConst], rc[nr::kRegFetchConst],
                rc[nr::kRegBoolLoop], rc[nr::kRegRenderTarget],
                rc[nr::kRegViewport], rc[nr::kRegScissor], rc[nr::kRegCopy],
                rc[nr::kRegClearValue], rc[nr::kRegModeControl],
                rc[nr::kRegDeviceState], rc[nr::kRegCoher], rc[nr::kRegOther]);
            REXGPU_INFO(
                "[nr-state]   miss split copy={} inherited={} plain={} | last "
                "resolve ctl={:08X} dest={:08X} info={:08X}",
                g_nrs_miss_copy, g_nrs_miss_inherited, g_nrs_miss_plain,
                g_nrs_copy_control, g_nrs_copy_dest, g_nrs_copy_dest_info);
            // Top type-3 opcodes beyond the draws, and any unclassified regs.
            {
              char ops[160];
              int n = 0;
              for (uint32_t pass = 0; pass < 6; ++pass) {
                uint32_t best = 0, best_op = 0;
                for (uint32_t op = 0; op < 128; ++op) {
                  if (op == 0x22 || op == 0x36) continue;
                  if (g_nrs_ops[op] > best) {
                    best = uint32_t(g_nrs_ops[op]);
                    best_op = op;
                  }
                }
                if (!best) break;
                n += snprintf(ops + n, sizeof(ops) - n, " %02X=%u", best_op,
                              best);
                g_nrs_ops[best_op] = 0;  // window resets below anyway
                if (n >= int(sizeof(ops)) - 12) break;
              }
              char oth[160];
              int m = 0;
              for (const auto& o : g_nrs_other) {
                if (!o.count) break;
                m += snprintf(oth + m, sizeof(oth) - m, " %04X=%llu", o.reg,
                              (unsigned long long)o.count);
                if (m >= int(sizeof(oth)) - 16) break;
              }
              REXGPU_INFO("[nr-state]   ops:{} | other:{}", n ? ops : " none",
                          m ? oth : " none");
            }
            g_nrs_execs = g_nrs_draws = g_nrs_copy = 0;
            g_nrs_mode_inherited = g_nrs_rt_set = g_nrs_vport_set = 0;
            g_nrs_desync = g_nrs_mode_mem = 0;
            g_nrs_t0 = g_nrs_t1 = g_nrs_t2 = g_nrs_t3 = 0;
            for (uint32_t c = 0; c < nr::kRegClassCount; ++c) g_nrs_regs[c] = 0;
            for (uint32_t op = 0; op < 128; ++op) g_nrs_ops[op] = 0;
            g_nrs_miss_copy = g_nrs_miss_inherited = g_nrs_miss_plain = 0;
            for (auto& o : g_nrs_other) o = NrsOtherReg{};
          }
          // [NR-CTX] The carried-context verdict. def%: with carry, how many
          // draws have each recovery group fully established (bar: near 100%
          // once past the first frames -- anything else means a state source
          // this stream cannot see). diverge: mirrored regs disagreeing with
          // the live register file at buffer entry (bar ~0; nonzero = state
          // arrives outside depth-1 IBs, and the reg names itself). carry%:
          // draws depending on cross-buffer carry (menu ~0, city high --
          // increment 3's inheritance, now resolved instead of just counted).
          if (g_nr_ctx && g_ctx_execs) {
            const double dp = g_ctx_draws ? 100.0 / double(g_ctx_draws) : 0.0;
            REXGPU_INFO(
                "[nr-ctx] execs={} draws={} | def rt={:.1f}% vp={:.1f}% "
                "mode={:.1f}% sh={:.1f}% full={:.1f}% | copy={} | carry "
                "rt={:.1f}% vp={:.1f}% mode={:.1f}% sh={:.1f}%",
                g_ctx_execs, g_ctx_draws, g_ctx_rt_def * dp, g_ctx_vp_def * dp,
                g_ctx_mode_def * dp, g_ctx_sh_def * dp, g_ctx_full * dp,
                g_ctx_copy, g_ctx_rt_carry * dp, g_ctx_vp_carry * dp,
                g_ctx_mode_carry * dp, g_ctx_sh_carry * dp);
            REXGPU_INFO(
                "[nr-ctx]   validate checks={} diverge={} | shaders im={} "
                "imm={} distinct={} set_ovf={} | mem_ld={} poison={} depth2={} "
                "ovf={}",
                g_ctx_checks, g_ctx_diverge, g_ctx_im_loads, g_ctx_im_imms,
                g_ctx_sh_distinct, g_ctx_sh_set_ovf, g_ctx_mem_loads,
                g_ctx_poisoned, g_ctx_depth2, g_ctx_flags_ovf);
            // [NR-RING] What the observer captured out-of-stream (ring /
            // nested) or through the walker's blind-spot ops (rmw / cond).
            if (g_ctx_ext_ring || g_ctx_ext_nested || g_ctx_ext_rmw ||
                g_ctx_ext_cond) {
              char ex[160];
              int n = 0;
              for (const auto& d : g_ctx_ext_top) {
                if (!d.count) break;
                n += snprintf(ex + n, sizeof(ex) - n, " %04X=%llu", d.reg,
                              (unsigned long long)d.count);
                if (n >= int(sizeof(ex)) - 16) break;
              }
              REXGPU_INFO(
                  "[nr-ctx]   ring obs: ring={} nested={} rmw={} cond={} "
                  "regs:{}",
                  g_ctx_ext_ring, g_ctx_ext_nested, g_ctx_ext_rmw,
                  g_ctx_ext_cond, n ? ex : " none");
            }
            // [NR-WSAMP] The one-frame diff: what the WALK wrote into the
            // watched registers (with the packet bytes it decoded them from)
            // against what the EXECUTOR wrote into the same registers
            // downstream. Equal streams mean the walk mirrors execution;
            // walker-only writes with impossible values name a misdecode, and
            // executor-only writes name an opcode the walk does not handle.
            if (g_nr_wsamp_tot || g_nr_esamp_tot || g_ctx_nop0 || g_ctx_sc2 ||
                g_ctx_pred) {
              REXGPU_INFO(
                  "[nr-ctx]   wsamp: walk={} exec={} | nop0={} setconst2={} | "
                  "pred_skip={} pred_draws={}/{} bin_pkts={}",
                  g_nr_wsamp_tot, g_nr_esamp_tot, g_ctx_nop0, g_ctx_sc2,
                  g_ctx_pred, g_ctx_pred_draws, g_ctx_pred_draws_run,
                  g_ctx_bin_pkts);
              for (uint32_t s = 0; s < g_nr_wsamp_n; ++s) {
                const NrWalkSample& w = g_nr_wsamp[s];
                REXGPU_INFO(
                    "[nr-ctx]   WSAMP buf={:08X} dw={} hdr={:08X} arg={:08X} "
                    "reg={:04X} val={:08X}",
                    w.buf, w.dw, w.hdr, w.arg, w.reg, w.value);
              }
              for (uint32_t s = 0; s < g_nr_esamp_n; ++s) {
                const NrExecSample& e = g_nr_esamp[s];
                REXGPU_INFO(
                    "[nr-ctx]   ESAMP pkt={:03X} depth={} reg={:04X} "
                    "val={:08X}",
                    e.pkt, e.depth, e.reg, e.value);
              }
            }
            if (g_ctx_diverge) {
              char dv[160];
              int n = 0;
              for (const auto& d : g_ctx_div_top) {
                if (!d.count) break;
                n += snprintf(dv + n, sizeof(dv) - n, " %04X=%llu", d.reg,
                              (unsigned long long)d.count);
                if (n >= int(sizeof(dv)) - 16) break;
              }
              REXGPU_INFO("[nr-ctx]   diverge regs:{}", n ? dv : " none");
              for (uint32_t s = 0; s < g_ctx_div_samp_n; ++s) {
                const CtxDivSample& sm = g_ctx_div_samp[s];
                REXGPU_INFO(
                    "[nr-ctx]   DIVERGE buf={:08X} reg={:04X} ctx={:08X} "
                    "live={:08X}",
                    sm.buf, sm.reg, sm.ours, sm.live);
              }
            }
            g_ctx_execs = g_ctx_depth2 = g_ctx_draws = 0;
            g_ctx_rt_def = g_ctx_vp_def = g_ctx_mode_def = g_ctx_sh_def = 0;
            g_ctx_full = g_ctx_copy = 0;
            g_ctx_rt_carry = g_ctx_vp_carry = g_ctx_mode_carry = 0;
            g_ctx_sh_carry = 0;
            g_ctx_checks = g_ctx_diverge = 0;
            g_ctx_mem_loads = g_ctx_poisoned = 0;
            g_ctx_im_loads = g_ctx_im_imms = 0;
            g_ctx_flags_ovf = 0;
            for (auto& d : g_ctx_div_top) d = CtxDivReg{};
            g_ctx_div_samp_n = 0;
            g_ctx_ext_ring = g_ctx_ext_nested = 0;
            g_ctx_ext_rmw = g_ctx_ext_cond = 0;
            g_nr_wsamp_n = g_nr_esamp_n = 0;
            g_nr_wsamp_tot = g_nr_esamp_tot = 0;
            g_ctx_nop0 = g_ctx_sc2 = 0;
            g_ctx_pred = g_ctx_pred_draws = g_ctx_bin_pkts = 0;
            g_ctx_pred_draws_run = 0;
            for (auto& d : g_ctx_ext_top) d = CtxDivReg{};
            for (uint32_t i = 0; i < kCtxShaderSetSize; ++i) {
              g_ctx_shader_set[i] = 0;
            }
            g_ctx_sh_distinct = g_ctx_sh_set_ovf = 0;
          }
          // [NR-RES] The resource verdict. diverge is the gate (must be ~0,
          // same meaning as 4a's): nonzero means resource state reaches a
          // draw from somewhere this buffer stream does not carry, and the
          // sample names the register and file. by_mem% is the design
          // number: constants that arrive by reference to guest memory
          // cannot be baked into a per-buffer translation, because buffers
          // are replayed for many frames after they are recorded.
          if (g_nr_res && g_res_stats.draws) {
            const nr::ResStats& r = g_res_stats;
            const double dp = 100.0 / double(r.draws);
            const double wp = r.writes ? 100.0 / double(r.writes) : 0.0;
            REXGPU_INFO(
                "[nr-res] draws={} writes={} by_mem={:.1f}% | per-file "
                "alu={} fetch={} bool={} loop={} | by_mem alu={} fetch={}",
                r.draws, r.writes, r.writes_from_memory * wp,
                r.writes_by_file[nr::kResFileAlu],
                r.writes_by_file[nr::kResFileFetch],
                r.writes_by_file[nr::kResFileBool],
                r.writes_by_file[nr::kResFileLoop],
                r.writes_from_memory_by_file[nr::kResFileAlu],
                r.writes_from_memory_by_file[nr::kResFileFetch]);
            REXGPU_INFO(
                "[nr-res]   per draw: vfetch={:.1f}/{} carried={:.1f} | "
                "tfetch={:.1f}/{} carried={:.1f} | alu_def={:.0f}/{} "
                "all_alu_carried={:.1f}% | addrs={} addr_ovf={}",
                double(r.vfetch_live) / double(r.draws),
                nr::kResFetchVertexSlots,
                double(r.vfetch_carried) / double(r.draws),
                double(r.tfetch_live) / double(r.draws),
                nr::kResFetchTextureSlots,
                double(r.tfetch_carried) / double(r.draws),
                double(r.alu_defined_at_draw) / double(r.draws),
                nr::kResAluCount, r.draws_all_alu_carried * dp,
                g_res_addr_distinct, g_res_addr_ovf);
            REXGPU_INFO(
                "[nr-res]   checks={} diverge={} | by file alu={} fetch={} "
                "bool={} loop={}",
                r.checks, r.diverge, r.diverge_by_file[nr::kResFileAlu],
                r.diverge_by_file[nr::kResFileFetch],
                r.diverge_by_file[nr::kResFileBool],
                r.diverge_by_file[nr::kResFileLoop]);
            for (uint32_t s = 0; s < g_res_div_samp_n; ++s) {
              const nr::ResDivergence& d = g_res_div_samp[s];
              REXGPU_INFO(
                  "[nr-res]   DIVERGE reg={:04X} ({}) ours={:08X} live={:08X}",
                  d.reg, nr::ResFileName(nr::ResSlotFile(
                             uint32_t(nr::ResSlot(d.reg)))),
                  d.ours, d.live);
            }
            // Coverage: the strict per-draw question. cov% must be ~100 for
            // the resource inputs of a native draw to be considered closed.
            if (g_res_cov_draws) {
              const double cp = 100.0 / double(g_res_cov_draws);
              // lockstep= says WHEN the mirror was read, and it changes what
              // the numbers mean. Off, the walk has already consumed the whole
              // buffer, so a fetch slot reports its LAST type in that buffer
              // rather than the one in effect at this draw: `undef` still
              // holds (definedness is sticky and cross-buffer) but
              // `wrongtype` is measuring the wrong moment. On (gpu_nr_draw),
              // the walk stops at each draw and the numbers are per-draw.
              REXGPU_INFO(
                  "[nr-res]   cover draws={} full={:.2f}% unanalyzed={} | "
                  "vref={} undef={} wrongtype={} | tref={} undef={} "
                  "wrongtype={} | lockstep={}",
                  g_res_cov_draws, g_res_cov_full * cp, g_res_cov_unanalyzed,
                  g_res_vref, g_res_vref_undef, g_res_vref_type, g_res_tref,
                  g_res_tref_undef, g_res_tref_type, kNrDraw ? 1 : 0);
              // Which slots actually fail, counted rather than sampled.
              std::string vslots, tslots;
              for (uint32_t i = 0; i < nr::kResFetchVertexSlots; ++i) {
                if (g_res_vref_fail_slot[i]) {
                  vslots += fmt::format("{}:{} ", i, g_res_vref_fail_slot[i]);
                }
              }
              for (uint32_t i = 0; i < nr::kResFetchTextureSlots; ++i) {
                if (g_res_tref_fail_slot[i]) {
                  tslots += fmt::format("{}:{} ", i, g_res_tref_fail_slot[i]);
                }
              }
              if (!vslots.empty() || !tslots.empty()) {
                REXGPU_INFO("[nr-res]   fail-by-slot vfetch[{}] tfetch[{}]",
                            vslots.empty() ? "-" : vslots.c_str(),
                            tslots.empty() ? "-" : tslots.c_str());
              }
              for (uint32_t s = 0; s < g_res_cov_samp_n; ++s) {
                const ResCovSample& c = g_res_cov_samp[s];
                REXGPU_INFO(
                    "[nr-res]   MISS {} slot={} from {} ({}) dword0={:08X} "
                    "type={}",
                    c.texture ? "tfetch" : "vfetch", c.slot,
                    c.pixel ? "ps" : "vs",
                    c.reason == nr::kResCoverUndefined ? "undefined"
                                                       : "wrong type",
                    c.dword0, c.dword0 & 3u);
              }
            }
            g_res_cov_draws = g_res_cov_full = g_res_cov_unanalyzed = 0;
            g_res_vref = g_res_vref_undef = g_res_vref_type = 0;
            g_res_tref = g_res_tref_undef = g_res_tref_type = 0;
            g_res_cov_samp_n = 0;
            for (uint32_t i = 0; i < nr::kResFetchVertexSlots; ++i) {
              g_res_vref_fail_slot[i] = 0;
            }
            for (uint32_t i = 0; i < nr::kResFetchTextureSlots; ++i) {
              g_res_tref_fail_slot[i] = 0;
            }
            g_res_stats = nr::ResStats{};
            g_res_div_samp_n = 0;
            for (uint32_t i = 0; i < kResAddrSetSize; ++i) {
              g_res_addr_set[i] = 0;
            }
            g_res_addr_distinct = g_res_addr_ovf = 0;
          }
          // Clear for the next window. Addresses are re-walked when they
          // reappear, which is what keeps a stale draw count from surviving a
          // buffer being re-recorded at the same address with different content.
          for (uint32_t i = 0; i < kIbLedgerSize; ++i) g_ib_ledger_tab[i] = IbLedgerEntry{};
          g_ib_ledger_used = 0;
          g_ib_ledger_evictions = 0;
        }
        // [NR-4A] N-9-4a verdict (outside the ledger gate: 4a runs with no
        // other probe on). composed= draws whose constants applied FROM THE
        // RECORD; every fallback names its reason. The gate: composed ~= 0x22
        // stops, fb classes at the known residual rates (miss ~0.2%, torn
        // ~0.3% city), shape ~0, and the frame visually identical.
        if (g_nr4a_on &&
            (g_4a_composed || g_4a_flush_draw || g_4a_flush_d2 ||
             g_4a_flush_deleg || g_4a_flush_end || g_4a_def_ranges)) {
          const uint64_t stops = g_4a_composed + g_4a_flush_draw;
          REXGPU_INFO(
              "[nr-4a] composed={} ({:.2f}% of 0x22 stops, {} runs {} dw) | "
              "flush: draw_fb={} d2={} deleg={} end={} | deferred={} "
              "(state {}) mk={} ovf_ranges={}",
              g_4a_composed,
              stops ? 100.0 * double(g_4a_composed) / double(stops) : 0.0,
              g_4a_comp_runs, g_4a_comp_dw, g_4a_flush_draw, g_4a_flush_d2,
              g_4a_flush_deleg, g_4a_flush_end, g_4a_def_ranges,
              g_4a_def_state, g_4a_mk_runs, g_4a_ovf_ranges);
          REXGPU_INFO(
              "[nr-4a]   fb: miss={} no_state={} stale={} torn={} shape={} "
              "val={} ovf={} | byref_conflict_flush={}",
              g_4a_fb_miss, g_4a_fb_nostate, g_4a_fb_stale, g_4a_fb_torn,
              g_4a_fb_shape, g_4a_fb_val, g_4a_fb_ovf, g_4a_byref_conflict);
          // [NR-5A] coherent-compose verdict. The gate: composed tracks the
          // 4a composed count, draw_issued == composed, fb args/nodraw/mk at
          // the known torn/segmentation residual rates, mk_applied at the
          // deferred by-ref rate.
          if (g_nr5_coh) {
            REXGPU_INFO(
                "[nr-5a] composed={} draw_issued={} | byref def={} "
                "mk_applied={} mk_skipped={} | fb: args={} nodraw={} mk={}",
                g_nr5_composed, g_nr5_draw_issued, g_nr5_def_byref,
                g_nr5_mk_applied, g_nr5_mk_skipped, g_nr5_fb_args,
                g_nr5_fb_nodraw, g_nr5_fb_mk);
            g_nr5_composed = g_nr5_draw_issued = 0;
            g_nr5_def_byref = g_nr5_mk_applied = g_nr5_mk_skipped = 0;
            g_nr5_fb_args = g_nr5_fb_nodraw = g_nr5_fb_mk = 0;
            // [NR-5B-DBG] swap-aligned register-file fingerprint: diff two
            // title runs (suppressed vs not) to name the diverging class.
            REXGPU_INFO(
                "[nr-rf] st={:08X} av={:08X} ap={:08X} fx={:08X} | surf={:08X} "
                "col={:08X} dep={:08X} woff={:08X} scis={:08X} dctl={:08X} "
                "mode={:08X} vte={:08X}",
                g_nr5dbg_hash[0], g_nr5dbg_hash[1], g_nr5dbg_hash[2],
                g_nr5dbg_hash[3], g_nr5dbg_key[0], g_nr5dbg_key[1],
                g_nr5dbg_key[2], g_nr5dbg_key[3], g_nr5dbg_key[4],
                g_nr5dbg_key[5], g_nr5dbg_key[6], g_nr5dbg_key[7]);
            // [NR-5B-3] suppressed-group verdict. The gate: sup_composed
            // tracks the game-side live= rate; argsne at the known torn
            // rate (diagnostic: those draws rendered the previous coherent
            // generation); refuse 0.
            if (g_nr5b_sup_composed || g_nr5b_sup_refuse) {
              REXGPU_INFO(
                  "[nr-5b] sup_composed={} argsne={} foreign={} refuse={} "
                  "pdw_dw={} pred_stops={}",
                  g_nr5b_sup_composed, g_nr5b_sup_argsne, g_nr5b_sup_foreign,
                  g_nr5b_sup_refuse, g_nr5b_pdw_dw, g_nr5b_pred_stops);
              // [NR-5B-3i] the by-ref source line. addr_ne is the hypothesis
              // itself: paired markers whose recorded address differs from
              // the live 0x2F's. A non-trivial rate at mode 0 means the ALU
              // class is being painted from last frame's constant memory.
              {
                const uint64_t paired = g_nr5b_mk_paired;
                REXGPU_INFO(
                    "[nr-5b]   mk mode={} stale={} live={} drop={} "
                    "addr_ne={} ({:.3f}%) unpaired={}",
                    g_nr5b_mk_mode, g_nr5b_mk_stale, g_nr5b_mk_live,
                    g_nr5b_mk_drop, g_nr5b_mk_addr_ne,
                    paired ? 100.0 * double(g_nr5b_mk_addr_ne) / double(paired)
                           : 0.0,
                    g_nr5b_mk_unpaired);
                g_nr5b_mk_stale = g_nr5b_mk_live = g_nr5b_mk_drop = 0;
                g_nr5b_mk_addr_ne = g_nr5b_mk_unpaired = g_nr5b_mk_paired = 0;
              }
              // [NR-5B-3j] the declined-compose pool: what it costs and what
              // mode 1 dropped. nodef = stops that never reached a compose.
              REXGPU_INFO("[nr-5b]   fb mode={} declined={} skipped={} "
                          "nodef={}",
                          g_nr5b_fbmode,
                          g_4a_fb_miss + g_4a_fb_shape + g_4a_fb_torn +
                              g_4a_fb_val + g_4a_fb_ovf + g_4a_fb_stale +
                              g_nr5b_sup_foreign,
                          g_nr5b_fb_skipped, g_nr5b_stop_nodef);
              g_nr5b_fb_skipped = g_nr5b_stop_nodef = 0;
              // [NR-5B-3k] the generation verdict. ahead% is the share of
              // suppressed composes whose record post-dates the buffer being
              // replayed; lead is how many recordings newer.
              REXGPU_INFO(
                  "[nr-5b]   gen mode={} seen={} ahead={} ({:.3f}%) "
                  "meanlead={:.1f} maxlead={} skipped={}",
                  g_nr5b_genmode, g_nr5b_gen_seen, g_nr5b_gen_ahead,
                  g_nr5b_gen_seen
                      ? 100.0 * double(g_nr5b_gen_ahead) / double(g_nr5b_gen_seen)
                      : 0.0,
                  g_nr5b_gen_ahead
                      ? double(g_nr5b_gen_leadsum) / double(g_nr5b_gen_ahead)
                      : 0.0,
                  g_nr5b_gen_maxlead, g_nr5b_gen_skipped);
              // [NR-5B-3k] liveness for that zero, and [3l] the order census.
              REXGPU_INFO(
                  "[nr-5b]   lag mean={:.1f} min={} max={} | ovl samp={} "
                  "pairs={} valne={}",
                  g_nr5b_gen_seen ? double(g_nr5b_lagsum) /
                                        double(g_nr5b_gen_seen)
                                  : 0.0,
                  g_nr5b_lagmin == 0x7FFFFFFF ? 0 : g_nr5b_lagmin,
                  g_nr5b_lagmax == -0x7FFFFFFF ? 0 : g_nr5b_lagmax,
                  g_nr5b_ovl_samples, g_nr5b_ovl_pairs, g_nr5b_ovl_valne);
              if (g_nr5b_ovl_samp[0]) {
                REXGPU_INFO(
                    "[nr-5b]   ovl samp: regA={:04X}x{} regB={:04X}x{}",
                    g_nr5b_ovl_samp[1], g_nr5b_ovl_samp[2],
                    g_nr5b_ovl_samp[3], g_nr5b_ovl_samp[4]);
                g_nr5b_ovl_samp[0] = 0;
              }
              g_nr5b_lagsum = 0;
              g_nr5b_lagmin = 0x7FFFFFFF;
              g_nr5b_lagmax = -0x7FFFFFFF;
              g_nr5b_ovl_samples = g_nr5b_ovl_pairs = g_nr5b_ovl_valne = 0;
              g_nr5b_gen_seen = g_nr5b_gen_ahead = g_nr5b_gen_skipped = 0;
              g_nr5b_gen_leadsum = 0;
              g_nr5b_gen_maxlead = 0;
              if (g_nr5b_foreign_samp[0]) {
                REXGPU_INFO(
                    "[nr-5b]   foreign samp: pkt={:08X} rec_prim={} pkt_prim={} "
                    "rec_cnt={} pkt_cnt={}",
                    g_nr5b_foreign_samp[1], g_nr5b_foreign_samp[2] >> 16,
                    g_nr5b_foreign_samp[2] & 0xFFFFu, g_nr5b_foreign_samp[3],
                    g_nr5b_foreign_samp[4]);
                g_nr5b_foreign_samp[0] = 0;
              }
              g_nr5b_sup_composed = g_nr5b_sup_argsne = g_nr5b_sup_foreign = 0;
              g_nr5b_sup_refuse = g_nr5b_pdw_dw = g_nr5b_pred_stops = 0;
            }
          }
          // [NR-5C] the fps-increment levers, 1 Hz. tear must stay ~0 (a
          // restart heals it); dirty vs skip is the overlap-reality answer
          // the design was waiting on; genmiss tracks the re-record rate.
          // [NR-6] the repeat-band drop, 1 Hz. stops = draws the hoisted
          // predicate called a band; drop rng/dw = the applies that no longer
          // happen at all. pred_ne is the safety counter and must read 0.
          if (g_nr6_stops || g_nr6_drop_rng || g_nr6_pred_ne || g_nr6_late) {
            REXGPU_INFO(
                "[nr-6] mode={} band stops={} groups={} | dropped rng={} "
                "dw={} | kept rng={} | late(sup)={} pre_dw={} pre_mk={} | pred_ne={}",
                g_nr6_mode, g_nr6_stops, g_nr6_groups, g_nr6_drop_rng,
                g_nr6_drop_dw, g_nr6_keep_rng, g_nr6_late, g_nr6_pre_dw,
                g_nr6_pre_mk, g_nr6_pred_ne);
            g_nr6_stops = g_nr6_groups = 0;
            g_nr6_drop_rng = g_nr6_drop_dw = g_nr6_keep_rng = 0;
            g_nr6_pred_ne = g_nr6_late = g_nr6_pre_dw = g_nr6_pre_mk = 0;
          }
          // [NR-5C-ATTR] the compose line prints on its OWN gate now: the
          // bracket arms without any lever (gpu_nr_compose_prof), and the
          // sup/pkt split is the attribution drive's primary read.
          if (g_nr5c_calls[0] || g_nr5c_calls[1]) {
            const auto line = [](const char* tag, uint64_t calls, uint64_t ns) {
              REXGPU_INFO("[nr-5c]   compose {} {} calls, {:.0f} ns/call, "
                          "{:.1f} ms/s",
                          tag, calls,
                          calls ? double(ns) / double(calls) : 0.0,
                          double(ns) / 1e6);
            };
            line("SUP", g_nr5c_calls[1], g_nr5c_ns[1]);
            line("pkt", g_nr5c_calls[0], g_nr5c_ns[0]);
            g_nr5c_ns[0] = g_nr5c_ns[1] = 0;
            g_nr5c_calls[0] = g_nr5c_calls[1] = 0;
          }
          if (g_nr5c_inplace || g_nr5c_copies ||
              g_nr5c_skip_rng || g_nr5c_dirty_rng || g_nr5c_tear ||
              g_nr5c_tear_pkt || g_nr5c_genmiss) {
            REXGPU_INFO(
                "[nr-5c] arena={} skip={} | inplace={} copy={} tear={} "
                "tearpkt={} restart={} | skip rng={} dw={} pdw={} pdw_dw={} "
                "dirty={} genmiss={} stamped={}",
                g_nr5c_arena_on ? 1 : 0, g_nr5c_skip_mode, g_nr5c_inplace,
                g_nr5c_copies, g_nr5c_tear, g_nr5c_tear_pkt, g_nr5c_restart,
                g_nr5c_skip_rng, g_nr5c_skip_dw, g_nr5c_skip_pdw_rng,
                g_nr5c_skip_pdw_dw, g_nr5c_dirty_rng, g_nr5c_genmiss,
                g_nr5c_stamped);
            g_nr5c_inplace = g_nr5c_copies = 0;
            g_nr5c_tear = g_nr5c_tear_pkt = g_nr5c_restart = 0;
            g_nr5c_skip_rng = g_nr5c_skip_dw = 0;
            g_nr5c_skip_pdw_rng = g_nr5c_skip_pdw_dw = 0;
            g_nr5c_dirty_rng = g_nr5c_genmiss = g_nr5c_stamped = 0;
          }
          // [NR-4B] only printed when the verify is rate-gated (ver+unver
          // == composed + val fallbacks; unver groups applied unread).
          if (g_4a_unver || g_nr4b_verify_n != 1) {
            REXGPU_INFO(
                "[nr-4b]   verify={} ver={} unver={} (phase {})",
                g_nr4b_verify_n, g_4a_ver, g_4a_unver,
                g_nr4b_verify_n == 1 ? "FULL"
                : g_nr4b_verify_n == 0 ? "OFF"
                                       : "SAMPLED");
          }
          if (g_4a_shape_samp[0]) {
            REXGPU_INFO(
                "[nr-4a]   first shape_ne: rec_runs={} deferred={} at[{}] "
                "rec={:#06X}x{} def={:#06X}x{} | pdw_const={}",
                g_4a_shape_samp[1], g_4a_shape_samp[2], g_4a_shape_samp[3],
                g_4a_shape_samp[4], g_4a_shape_samp[5], g_4a_shape_samp[6],
                g_4a_shape_samp[7], g_4a_pdw_const);
            REXGPU_INFO(
                "[nr-4a]   rec: {:04X}x{} {:04X}x{} {:04X}x{} {:04X}x{} "
                "{:04X}x{} {:04X}x{} {:04X}x{} {:04X}x{}",
                g_4a_shape_rec[0], g_4a_shape_rec[1], g_4a_shape_rec[2],
                g_4a_shape_rec[3], g_4a_shape_rec[4], g_4a_shape_rec[5],
                g_4a_shape_rec[6], g_4a_shape_rec[7], g_4a_shape_rec[8],
                g_4a_shape_rec[9], g_4a_shape_rec[10], g_4a_shape_rec[11],
                g_4a_shape_rec[12], g_4a_shape_rec[13], g_4a_shape_rec[14],
                g_4a_shape_rec[15]);
            REXGPU_INFO(
                "[nr-4a]   def: {:04X}x{} {:04X}x{} {:04X}x{} {:04X}x{} "
                "{:04X}x{} {:04X}x{} {:04X}x{} {:04X}x{}",
                g_4a_shape_def[0], g_4a_shape_def[1], g_4a_shape_def[2],
                g_4a_shape_def[3], g_4a_shape_def[4], g_4a_shape_def[5],
                g_4a_shape_def[6], g_4a_shape_def[7], g_4a_shape_def[8],
                g_4a_shape_def[9], g_4a_shape_def[10], g_4a_shape_def[11],
                g_4a_shape_def[12], g_4a_shape_def[13], g_4a_shape_def[14],
                g_4a_shape_def[15]);
            REXGPU_INFO(
                "[nr-4b]   byref in that group: n={} {:04X}x{} {:04X}x{} "
                "{:04X}x{} {:04X}x{}",
                g_4a_shape_byref[0], g_4a_shape_byref[1], g_4a_shape_byref[2],
                g_4a_shape_byref[3], g_4a_shape_byref[4], g_4a_shape_byref[5],
                g_4a_shape_byref[6], g_4a_shape_byref[7],
                g_4a_shape_byref[8]);
            REXGPU_INFO(
                "[nr-4b]   span: rec pre_span={} stop={:08X} def0_phys={:08X} "
                "(stop-def0={})",
                g_4a_shape_span[0], g_4a_shape_span[1], g_4a_shape_span[2],
                g_4a_shape_span[2]
                    ? int64_t(g_4a_shape_span[1]) - int64_t(g_4a_shape_span[2])
                    : 0);
            g_4a_shape_samp[0] = 0;
          }
          g_4a_pdw_const = 0;
          g_4a_composed = g_4a_comp_runs = g_4a_comp_dw = 0;
          g_4a_flush_draw = g_4a_flush_d2 = 0;
          g_4a_flush_deleg = g_4a_flush_end = 0;
          g_4a_fb_miss = g_4a_fb_nostate = g_4a_fb_stale = 0;
          g_4a_fb_torn = g_4a_fb_shape = g_4a_fb_ovf = 0;
          g_4a_fb_val = g_4a_byref_conflict = 0;
          g_4a_def_ranges = g_4a_ovf_ranges = 0;
          g_4a_def_state = g_4a_mk_runs = 0;
          g_4a_ver = g_4a_unver = 0;
        }
        // [NR-97] N-9-7 phase A, 1 Hz. Deliberately OUTSIDE the [nr-4a] gate
        // this file's neighbours sit in: N-9-7 lands on the DEFAULT path,
        // where 4a is off, and a report that only prints under an opt-in flag
        // reads as "the lever did nothing" ([[reporter-gate-must-follow-
        // implication]]).
        // `band` is the share of depth-1 executions whose ENTRY state de-tile
        // calls a repeat band -- the premise (~2/3) read at the buffer instead
        // of at the draw. `drop` is what the lever actually took. `dw` drop vs
        // keep is the traffic itself: the drop half stops happening when
        // mode=1, the keep half is the guest's progress window plus the
        // recovery mirror and never stops. mixed is the safety counter and
        // must read 0.
        if (g_n97_bufs || g_n97_mixed_stops) {
          const uint64_t tot_dw = g_n97_drop_dw + g_n97_drop_pdw +
                                  g_n97_keep_dw + g_n97_keep_pdw;
          const uint64_t drop_dw = g_n97_drop_dw + g_n97_drop_pdw;
          REXGPU_INFO(
              "[nr-97] mode={} bufs={} | band entry={} 1st={} ({}% of bufs) | "
              "act={} (atentry={}) refused={} mixedbufs={} lateband={} | "
              "dw drop={} keep={} pre={} ({}% of band traffic dropped) | "
              "rng drop={} keep={} pdw drop={} keep={} | mixed={} lostdraws={}",
              g_n97_mode, g_n97_bufs, g_n97_bufs_eband, g_n97_bufs_fband,
              g_n97_bufs ? g_n97_bufs_fband * 100 / g_n97_bufs : 0,
              g_n97_bufs_drop, g_n97_bufs_entry, g_n97_bufs_ref,
              g_n97_bufs_mixed, g_n97_late_band, drop_dw,
              g_n97_keep_dw + g_n97_keep_pdw, g_n97_pre_dw,
              tot_dw ? drop_dw * 100 / tot_dw : 0, g_n97_drop_rng,
              g_n97_keep_rng, g_n97_drop_pdw, g_n97_keep_pdw,
              g_n97_mixed_stops, g_n97_dropped_draws);
          // The lever itself: ranges STORED without their side effects, and
          // the far smaller set re-applied WITH them when each run ended.
          // flush rng / store rng is the whole reduction, in one ratio.
          if (g_n97_store_rng || g_n97_flush_rng) {
            REXGPU_INFO(
                "[nr-97]   store rng={} dw={} | flush runs={} rng={} dw={} "
                "({:.3f}% of stored ranges re-applied, {:.1f} per run)",
                g_n97_store_rng, g_n97_store_dw, g_n97_flush_runs,
                g_n97_flush_rng, g_n97_flush_dw,
                g_n97_store_rng
                    ? 100.0 * double(g_n97_flush_rng) / double(g_n97_store_rng)
                    : 0.0,
                g_n97_flush_runs
                    ? double(g_n97_flush_rng) / double(g_n97_flush_runs)
                    : 0.0);
          }
          g_n97_store_rng = g_n97_store_dw = 0;
          g_n97_flush_runs = g_n97_flush_rng = g_n97_flush_dw = 0;
          // (SS) the soundness read, 1-in-16 sampled, OFF-phase only.
          // net_chg IS THE VERDICT: registers a repeat-band execution left
          // different from how it found them. It must be 0. `churn` is the
          // write-by-write number and is expected to be large and to mean
          // nothing on its own -- it is printed only so it cannot be
          // rediscovered later and mistaken for the verdict.
          if (g_n97_net_frames || g_n97_net_chg) {
            const uint64_t regs = g_n97_net_chg + g_n97_net_same;
            const uint64_t churn = g_n97_chg_dw + g_n97_same_dw;
            REXGPU_INFO(
                "[nr-97]   NET net_chg={} of {} regs over {} sampled frames "
                "(dirty frames={}) first={:04X} | churn {}/{} dw uncmp={}",
                g_n97_net_chg, regs, g_n97_net_frames, g_n97_net_dirty,
                g_n97_chg_reg == 0xFFFFFFFFu ? 0u : g_n97_chg_reg,
                g_n97_chg_dw, churn, g_n97_skip_dw);
          }
          if (g_n97_chg_rec_n) {
            char buf[512];
            size_t at = 0;
            for (uint32_t k = 0; k < g_n97_chg_rec_n && at + 40 < sizeof(buf);
                 ++k) {
              at += size_t(snprintf(buf + at, sizeof(buf) - at,
                                    "%s%04X:%08X->%08X x%llu", k ? " " : "",
                                    g_n97_chg_rec[k].reg,
                                    g_n97_chg_rec[k].before,
                                    g_n97_chg_rec[k].after,
                                    (unsigned long long)g_n97_chg_rec[k].n));
            }
            buf[sizeof(buf) - 1] = 0;
            REXGPU_INFO("[nr-97]   NET regs ({} distinct, +{} unlisted): {}",
                        g_n97_chg_rec_n, g_n97_chg_rec_ovf, buf);
          }
          if (g_n97_deleg) {
            char db[256];
            size_t da = 0;
            for (uint32_t op = 0; op < 128 && da + 24 < sizeof(db); ++op) {
              if (!g_n97_deleg_op[op]) continue;
              da += size_t(snprintf(db + da, sizeof(db) - da, " %02X=%llu", op,
                                    (unsigned long long)g_n97_deleg_op[op]));
            }
            db[sizeof(db) - 1] = 0;
            REXGPU_INFO(
                "[nr-97]   band delegates={} reg-reading={} | wait mem={} "
                "low={} OTHER={} (first {:04X}) |{}",
                g_n97_deleg, g_n97_deleg_read, g_n97_wait_mem,
                g_n97_wait_coher, g_n97_wait_other,
                g_n97_wait_reg == 0xFFFFFFFFu ? 0u : g_n97_wait_reg, db);
            g_n97_deleg = g_n97_deleg_read = 0;
            g_n97_wait_mem = g_n97_wait_coher = g_n97_wait_other = 0;
            g_n97_wait_reg = 0xFFFFFFFFu;
            for (uint32_t op = 0; op < 128; ++op) g_n97_deleg_op[op] = 0;
          }
          g_n97_chg_rec_n = 0;
          g_n97_chg_rec_ovf = 0;
          g_n97_net_chg = g_n97_net_same = g_n97_net_frames = 0;
          g_n97_net_dirty = 0;
          g_n97_chg_dw = g_n97_same_dw = g_n97_skip_dw = 0;
          g_n97_chg_reg = 0xFFFFFFFFu;
          g_n97_bufs = g_n97_bufs_drop = 0;
          g_n97_bufs_eband = g_n97_bufs_fband = g_n97_bufs_entry = 0;
          g_n97_bufs_ref = g_n97_bufs_mixed = g_n97_stops = 0;
          g_n97_late_band = g_n97_pre_dw = g_n97_dropped_draws = 0;
          g_n97_drop_rng = g_n97_drop_dw = g_n97_drop_pdw = 0;
          g_n97_keep_rng = g_n97_keep_dw = g_n97_keep_pdw = 0;
          g_n97_mixed_stops = 0;
        }
        // [NR-SDB] The corpus verdict. Per-load hit% says how often a bind
        // resolves against the database; the DISTINCT line is the one the
        // AOT plan rests on -- of every distinct blob loaded this session,
        // how many the database holds under the runtime's own key, and for
        // the rest whether the shader is there under another length
        // (prefix) or not there at all (absent). Samples name the miss.
        if (kNrSdb && g_sdb_ready && g_sdb_loads) {
          const double lp = 100.0 / double(g_sdb_loads);
          const double dp = g_sdb_distinct ? 100.0 / double(g_sdb_distinct) : 0.0;
          REXGPU_INFO(
              "[nr-sdb] loads={} hit={:.2f}% miss={} | distinct={} "
              "hit={:.2f}% prefix={} absent={} | corpus={}/{} dumped={} "
              "stage_mismatch={} zero={} set_ovf={}",
              g_sdb_loads, g_sdb_hits * lp, g_sdb_misses, g_sdb_distinct,
              g_sdb_distinct_hit * dp, g_sdb_distinct_prefix,
              g_sdb_distinct_absent, g_sdb_distinct_hit,
              g_sdb_index->unique_count(), g_sdb_dumped, g_sdb_stage_mismatch,
              g_sdb_zero, g_sdb_set_ovf);
          for (uint32_t s = 0; s < g_sdb_miss_samp_n; ++s) {
            const SdbMissSample& m = g_sdb_miss_samp[s];
            REXGPU_INFO(
                "[nr-sdb]   MISS {} hash={:016X} dwords={} addr={:08X} | {}",
                m.pixel ? "ps" : "vs", m.hash, m.dwords, m.guest_addr,
                m.prefix ? fmt::format("prefix db_bytes={} equal={}",
                                       m.db_bytes, m.equal_bytes)
                         : std::string("absent"));
          }
          // Per-load counters are per-window; the distinct set and its
          // classification are cumulative on purpose (see the set comment),
          // so they are NOT cleared here. Samples are one-shot per session
          // for the same reason: a repeated miss is not new evidence.
          g_sdb_loads = g_sdb_hits = g_sdb_misses = 0;
        }
        // [NR-DRAW] Increment 4c. Two numbers carry the verdict and they are
        // not the same kind of thing:
        //   diverge  the walk wrote a register and holds the wrong value.
        //            A decoder bug. Must be 0, like every gate before it.
        //   extern   the live file has a register the stream never wrote.
        //            NOT a bug: the boundary of the input model, and directly
        //            the list of D3D9 hooks a native replay needs instead.
        // Both are reported as named registers rather than rates, because a
        // rate here would be unactionable and the names are the deliverable.
        if (kNrDraw && g_nrd_stops) {
          const nr::RegShadowStats& r = g_reg_stats;
          REXGPU_INFO(
              "[nr-draw] stops={} buffers={} desync={} depth_skip={} | "
              "defined={} writes={}/{}ign | checks={} sweeps={} | "
              "diverge={} sfx={} extern={}",
              g_nrd_stops, g_nrd_buffers, g_nrd_desync, g_nrd_skipped_depth,
              r.defined_count, r.writes, r.writes_ignored, r.checks, r.sweeps,
              r.diverge - g_nrd_sfx, g_nrd_sfx, r.externs);
          for (uint32_t i = 0; i < kNrdTop && g_nrd_sfx_top[i].count; ++i) {
            const NrdReg& d = g_nrd_sfx_top[i];
            const RegisterInfo* info = RegisterFile::GetRegisterInfo(d.reg);
            REXGPU_INFO(
                "[nr-draw]   SFX     reg={:04X} ({}) x{} ours={:08X} "
                "live={:08X}",
                d.reg, info ? info->name : "?", d.count, d.last_ours,
                d.last_live);
          }
          for (uint32_t i = 0; i < kNrdTop && g_nrd_div_top[i].count; ++i) {
            const NrdReg& d = g_nrd_div_top[i];
            const RegisterInfo* info = RegisterFile::GetRegisterInfo(d.reg);
            REXGPU_INFO(
                "[nr-draw]   DIVERGE reg={:04X} ({}) x{} ours={:08X} "
                "live={:08X}",
                d.reg, info ? info->name : "?", d.count, d.last_ours,
                d.last_live);
          }
          for (uint32_t i = 0; i < kNrdTop && g_nrd_ext_top[i].count; ++i) {
            const NrdReg& e = g_nrd_ext_top[i];
            const RegisterInfo* info = RegisterFile::GetRegisterInfo(e.reg);
            REXGPU_INFO("[nr-draw]   EXTERN  reg={:04X} ({}) x{} live={:08X}",
                        e.reg, info ? info->name : "?", e.count, e.last_live);
          }
          // Per-window counters clear; the SHADOW does not. It is the
          // persistent thing being measured, exactly like the 4a context.
          g_reg_stats.checks = g_reg_stats.diverge = g_reg_stats.externs = 0;
          g_reg_stats.sweeps = g_reg_stats.compares = 0;
          g_reg_stats.writes = g_reg_stats.writes_ignored = 0;
          g_nrd_stops = g_nrd_desync = g_nrd_skipped_depth = 0;
          g_nrd_buffers = g_nrd_sfx = 0;
          for (uint32_t i = 0; i < kNrdTop; ++i) {
            g_nrd_div_top[i] = NrdReg{};
            g_nrd_ext_top[i] = NrdReg{};
            g_nrd_sfx_top[i] = NrdReg{};
          }
        }
        // [NR-ISSUE] Increment 4d. `issued` is the backend's count of draws
        // actually recorded from the composed shadow file; `armed` is the
        // base's. They must match (precord off); `precord_skip` is the named
        // reason when they do not. `ordinal` is cumulative for bisection via
        // gpu_nr_issue_from/count. sh_mismatch must be 0: the walk's resolved
        // shaders and the executor's actives are the same ucode or 4b-2's
        // tracking has a hole.
        if (kNrIssue && (g_nri_ordinal || g_nri_armed)) {
          REXGPU_INFO(
              "[nr-issue] issued={} armed={} copy_armed={} sh_invalid={} "
              "sh_mismatch={} precord_skip={} ordinal={}",
              nr_issue_issued_, g_nri_armed, g_nri_copy_armed,
              g_nri_sh_invalid, g_nri_sh_mismatch, nr_issue_precord_skips_,
              g_nri_ordinal);
          nr_issue_issued_ = 0;
          nr_issue_precord_skips_ = 0;
          g_nri_armed = g_nri_copy_armed = 0;
          g_nri_sh_invalid = g_nri_sh_mismatch = 0;
        }
        // [NR-SKP] Phase 5-4-2. bufs = skip-driven depth-1 buffer executions
        // this window; fb = buffers refused with the cvar on (trace open /
        // backend veto / bisection window); draws + deleg = packets handed to
        // the executor's own handlers (everything else was walked natively);
        // exec_fail and orphan must stay 0. Top delegated ops are named so
        // the 5-4-1 closed list stays checkable live.
        if (kNrSkip && (g_skp_bufs || g_skp_fb)) {
          char ops[160];
          int n = 0;
          for (uint32_t pass = 0; pass < 6; ++pass) {
            uint64_t best = 0;
            uint32_t best_op = 0;
            for (uint32_t op = 0; op < 128; ++op) {
              if (g_skp_deleg_op[op] > best) {
                best = g_skp_deleg_op[op];
                best_op = op;
              }
            }
            if (!best) break;
            n += snprintf(ops + n, sizeof(ops) - n, " %s=%llu",
                          NrPktOpName(best_op), (unsigned long long)best);
            g_skp_deleg_op[best_op] = 0;
            if (n >= int(sizeof(ops)) - 24) break;
          }
          REXGPU_INFO(
              "[nr-skp] bufs={} fb={} draws={} direct={} dfb={} deleg={} "
              "exec_fail={} orphan={} rng={}/{}dw pdw={} plain={}/{}dw |{}",
              g_skp_bufs, g_skp_fb, g_skp_draws, g_skp_direct, g_skp_direct_fb,
              g_skp_deleg, g_skp_exec_fail, g_skp_arm_orphan, g_skp_rng,
              g_skp_rng_dw, g_skp_pdw, g_skp_plain_rng, g_skp_plain_dw,
              n ? ops : " deleg none");
          // [NR-TILE] N-4 probe: what the EDRAM tile loop costs. `skipped`
          // is the repeat-band work dropped this window, `kept` is band 1
          // plus every offset-0 auxiliary pass. skipped/(skipped+kept)
          // should read ~2/3 at heavy city if the census transfers.
          if (g_nr_tile_probe || g_nr_tile_cycle) {
            REXGPU_INFO(
                "[nr-tile] mode={} draws skipped={} kept={} | bufs "
                "skipped={} kept={}",
                g_nr_tile_probe, g_tile_draws_skipped, g_tile_draws_kept,
                g_tile_bufs_skipped, g_tile_bufs_kept);
            g_tile_draws_skipped = g_tile_draws_kept = 0;
            g_tile_bufs_skipped = g_tile_bufs_kept = 0;
          }
          // [N9] whole-buffer plan replay. bufs rep vs walk and draws rep
          // vs walk are the coverage at both granularities; gfail is
          // structural drift (recompiled in place), refuse/marks name the
          // buffers that never replay. cmp counts compiles this window.
          if (g_n9_on || kN7 == 6) {
            nr::CtxProgStats* ps = nr::CtxProgStatsPtr();
            static nr::CtxProgStats n9_last{};
            REXGPU_INFO(
                "[n9] bufs rep={} walk={} | draws rep={} walk={} | cmp={} "
                "gfail={} refexec={} marks={} retry={} | store bufs={} "
                "{}KB evict={} | guards={}",
                g_n9_bufs, g_n9_walk_bufs, g_n9_draws, g_n9_walk_draws,
                ps->compiles - n9_last.compiles, ps->gfails - n9_last.gfails,
                ps->refused_execs - n9_last.refused_execs,
                ps->refuse_marks - n9_last.refuse_marks,
                ps->retries - n9_last.retries, ps->bufs, ps->bytes >> 10,
                ps->evicts - n9_last.evicts,
                ps->guards_checked - n9_last.guards_checked);
            n9_last = *ps;
            g_n9_bufs = g_n9_walk_bufs = 0;
            g_n9_draws = g_n9_walk_draws = 0;
          }
          // [NR-PLAN] N-2-2 item 2: what the swap actually replaced. `spans`
          // and `dw` are the removed parse; `gap` is the packets still
          // parsed live; `miss` is where no template starts. Coverage is
          // dw/(dw+gap dwords) read against the [nr-tmpl] gate's own.
          if (g_nr_tmpl_swap) {
            nr::TmplSwapStats* sw = nr::TmplSwapStatsPtr();
            REXGPU_INFO(
                "[nr-swap] spans={} dw={} gapPkts={} miss={} cross={} "
                "dead={} demoted={} lkStale={}",
                sw->spans, sw->dwords, sw->gap_pkts, sw->miss, sw->cross,
                sw->dead, sw->demoted, sw->lk_stale);
            *sw = nr::TmplSwapStats{};
          }
          // [NR-RUSE-BLK] coverage of the block skip: `skip` is dwords the
          // stale feed never touched, `mark` is dwords it still walked one
          // by one. A low skip share means the city's ranges genuinely
          // change, and the per-dword feed is irreducible as written.
          if (g_nr_ruse_blk) {
            const uint64_t tot = g_ruse_blk_skip + g_ruse_blk_mark;
            REXGPU_INFO("[nr-ruse-blk] skip={} mark={} ({}% skipped)",
                        g_ruse_blk_skip, g_ruse_blk_mark,
                        tot ? g_ruse_blk_skip * 100 / tot : 0);
            g_ruse_blk_skip = g_ruse_blk_mark = 0;
          }
          g_skp_bufs = g_skp_fb = g_skp_draws = g_skp_deleg = 0;
          g_skp_direct = g_skp_direct_fb = 0;
          g_skp_exec_fail = g_skp_arm_orphan = 0;
          g_skp_rng = g_skp_rng_dw = g_skp_pdw = 0;
          g_skp_plain_rng = g_skp_plain_dw = 0;
          for (uint32_t op = 0; op < 128; ++op) g_skp_deleg_op[op] = 0;
        }
        // [NR-WM] 5-4-8: the walk memo. Gate under verify: ne=0. Perf mode:
        // rep is the coverage, fb (declined replayed ranges) must stay 0.
        if (g_nr_wm && (g_wm_rep || g_wm_rec || g_wm_cmp || g_wm_inval ||
                        g_wm_refuse || g_wm_ne)) {
          nr::CtxMemoStats* ms = nr::CtxMemoStatsPtr();
          REXGPU_INFO(
              "[nr-wm] rep={} rec={} cmp={} ne={} inval={} refuse={} cap={} "
              "abort={} fb={} | bufs={} streams={} {}KB",
              g_wm_rep, g_wm_rec, g_wm_cmp, g_wm_ne, g_wm_inval, g_wm_refuse,
              g_wm_cap, g_wm_abort, ms->fallbacks, ms->bufs, ms->streams,
              ms->bytes >> 10);
          ms->fallbacks = 0;
          g_wm_rep = g_wm_rec = g_wm_cmp = g_wm_ne = 0;
          g_wm_inval = g_wm_refuse = g_wm_cap = g_wm_abort = 0;
        }
        // [NR-RUSE] 5-4-5 inc 0. Line 1 = buffer replays: id/ch = byte
        // identical vs changed vs previous replay of the same buffer, sf/xf
        // = same-frame (bin) vs cross-frame splits with their identical
        // counts, entry = inherited-state component stability at buffer
        // entry (bin trio excluded; binonly = ONLY the trio moved). Line 2 =
        // the per-draw verdict: reuse = prefix-clean AND by-ref digest eq
        // AND entry core eq vs this draw's previous execution; miss
        // attribution is first-fail (pfx > byref > ctl/flt/ftc/bl/sh).
        if (g_nr_ruse && (g_ruse_w_bufs || g_ruse_w_draws)) {
          REXGPU_INFO(
              "[nr-ruse] bufs={} new={} id={} ch={} chdw={} sf={}/{} "
              "xf={}/{} ep={}/{} | entry eq={} binonly={} "
              "ne c/f/t/b/s={}/{}/{}/{}/{} "
              "ctl1st={:04X} | live={} shad={}MB evict={} cost={:.1f}ms",
              g_ruse_w_bufs, g_ruse_w_new, g_ruse_w_id, g_ruse_w_ch,
              g_ruse_w_chdw, g_ruse_w_sf_id, g_ruse_w_sf, g_ruse_w_xf_id,
              g_ruse_w_xf, g_ruse_w_ep_clean, g_ruse_w_ep_ne,
              g_ruse_w_entry_eq, g_ruse_w_entry_binonly,
              g_ruse_w_entry_ne[0], g_ruse_w_entry_ne[1], g_ruse_w_entry_ne[2],
              g_ruse_w_entry_ne[3], g_ruse_w_entry_ne[4],
              g_ruse_w_ctl_diff_reg == 0xFFFFFFFFu ? 0u : g_ruse_w_ctl_diff_reg,
              g_ruse_bufs.size(), g_ruse_shadow_bytes >> 20, g_ruse_w_evict,
              g_ruse_w_cost_ns / 1e6);
          const uint64_t with_prev = g_ruse_w_draws - g_ruse_w_first;
          REXGPU_INFO(
              "[nr-ruse]   draws={} 1st={} reuse={} ({:.1f}% of prev'd) "
              "sf={} xf={} binonly={} | miss pfx={} byref={} "
              "c/f/t/b/s={}/{}/{}/{}/{}",
              g_ruse_w_draws, g_ruse_w_first, g_ruse_w_reuse,
              with_prev ? 100.0 * g_ruse_w_reuse / with_prev : 0.0,
              g_ruse_w_reuse_sf, g_ruse_w_reuse_xf, g_ruse_w_reuse_binonly,
              g_ruse_w_miss_pfx, g_ruse_w_miss_byref, g_ruse_w_miss_entry[0],
              g_ruse_w_miss_entry[1], g_ruse_w_miss_entry[2],
              g_ruse_w_miss_entry[3], g_ruse_w_miss_entry[4]);
          const uint64_t with_prev2 = g_ruse_w_draws - g_ruse_w_first2;
          REXGPU_INFO(
              "[nr-ruse]   v2 reuse2={} ({:.1f}% of prev'd) sf={} xf={} | "
              "miss2 pkt={} stale={} (avg {:.1f}) sh={} deleg={} 1st2={} "
              "pdwcons={} stale1st={:04X} | bufid={}/{}",
              g_ruse_w_reuse2,
              with_prev2 ? 100.0 * g_ruse_w_reuse2 / with_prev2 : 0.0,
              g_ruse_w_reuse2_sf, g_ruse_w_reuse2_xf, g_ruse_w_miss2_pkt,
              g_ruse_w_miss2_stale,
              g_ruse_w_miss2_stale
                  ? double(g_ruse_w_stale_sum) / g_ruse_w_miss2_stale
                  : 0.0,
              g_ruse_w_miss2_sh, g_ruse_w_miss2_deleg, g_ruse_w_first2,
              g_ruse_w_pdw_cons,
              g_ruse_stale_sample == 0xFFFFFFFFu ? 0u : g_ruse_stale_sample,
              g_ruse_w_bufid_draws, g_ruse_w_bufid_byref);
          g_ruse_w_bufs = g_ruse_w_new = g_ruse_w_id = g_ruse_w_ch = 0;
          g_ruse_w_chdw = 0;
          g_ruse_w_ep_clean = g_ruse_w_ep_ne = 0;
          g_ruse_w_sf = g_ruse_w_sf_id = g_ruse_w_xf = g_ruse_w_xf_id = 0;
          g_ruse_w_entry_eq = g_ruse_w_entry_binonly = 0;
          for (uint32_t i = 0; i < 5; ++i) {
            g_ruse_w_entry_ne[i] = g_ruse_w_miss_entry[i] = 0;
          }
          g_ruse_w_ctl_diff_reg = 0xFFFFFFFFu;
          g_ruse_w_draws = g_ruse_w_first = g_ruse_w_reuse = 0;
          g_ruse_w_reuse_sf = g_ruse_w_reuse_xf = g_ruse_w_reuse_binonly = 0;
          g_ruse_w_miss_pfx = g_ruse_w_miss_byref = 0;
          g_ruse_w_evict = 0;
          g_ruse_w_cost_ns = 0;
          g_ruse_w_first2 = g_ruse_w_reuse2 = 0;
          g_ruse_w_reuse2_sf = g_ruse_w_reuse2_xf = 0;
          g_ruse_w_miss2_pkt = g_ruse_w_miss2_stale = g_ruse_w_stale_sum = 0;
          g_ruse_w_miss2_sh = g_ruse_w_miss2_deleg = g_ruse_w_pdw_cons = 0;
          g_ruse_w_bufid_draws = g_ruse_w_bufid_byref = 0;
          g_ruse_stale_sample = 0xFFFFFFFFu;
        }
        // [NR-BFC] 5-4-6-0: buffer-replay census, candidate vs non-candidate
        // aggregates. Line 1 = delegate schedule + refuse classes; line 2 =
        // native span composition (viol = whitelist-violating buffers whose
        // span a replay could NOT memcpy verbatim); line 3 = maintenance
        // costs (RT body runs, submission crossings, end-state size, by-ref
        // volume). Decision gates in NEXT-AGENT-5.md "5-4-6".
        if (g_nr_bfc && (g_bfc_cand.bufs || g_bfc_non.bufs)) {
          const auto bfc_print = [](const char* tag, const BfcAgg& a) {
            if (!a.bufs) return;
            REXGPU_INFO(
                "[nr-bfc] {} bufs={} draws={} | deleg={} ev={} wcoh={} "
                "wmem={} woth={} memw={} oth={} befdraw={} | refuse bufs={} "
                "rmw/cw/ib/swap/wreg={}/{}/{}/{}/{}",
                tag, a.bufs, a.draws, a.deleg, a.dl_events, a.dl_wait_coher,
                a.dl_wait_mem, a.dl_wait_other, a.dl_memw, a.dl_other,
                a.dl_before_draw, a.refuse_bufs, a.refuse_by[0],
                a.refuse_by[1], a.refuse_by[2], a.refuse_by[3],
                a.refuse_by[4]);
            REXGPU_INFO(
                "[nr-bfc]   span el={} draw={} pso={} sys={} (mx {}) cbv={} "
                "root={} ia={} vp={} (mx {}) sci={} (mx {}) rt_set={} om={} "
                "heaps={} mrk={} | viol bufs={} draws={} bar={} cpy={} clr={} "
                "dsp={} qry={} oth={}",
                a.span_el, a.sp_draw, a.sp_pso, a.sp_sys, a.sys_max, a.sp_cbv,
                a.sp_root, a.sp_ia, a.sp_vp, a.vp_max, a.sp_sci, a.sci_max,
                a.sp_om_rt, a.sp_om_misc, a.sp_heaps, a.sp_marker,
                a.viol_bufs, a.viol_draws, a.sp_bar, a.sp_copy, a.sp_clear,
                a.sp_disp, a.sp_query, a.sp_other);
            REXGPU_INFO(
                "[nr-bfc]   rt_runs={} (mx {}) subx={} | wset={} (mx {}) "
                "frags={} ctl={} othdw={} pdwsf={} (port={} plain={}) "
                "byref={}",
                a.rt_runs, a.rt_max, a.sub_cross, a.wset, a.wset_max,
                a.frags, a.dw_ctl, a.dw_other, a.pdw_stateful, a.pdw_port,
                a.pdw_plain, a.byref_dw);
          };
          bfc_print("CAND", g_bfc_cand);
          bfc_print("non ", g_bfc_non);
          if (g_bfc_backend_missing) {
            REXGPU_INFO("[nr-bfc] backend_missing={} (no D3D12 sample)",
                        g_bfc_backend_missing);
          }
          if (g_bfc_wait_reg_sample != 0xFFFFFFFFu) {
            REXGPU_INFO("[nr-bfc] wreg1st={:04X}", g_bfc_wait_reg_sample);
          }
          g_bfc_cand = BfcAgg{};
          g_bfc_non = BfcAgg{};
          g_bfc_backend_missing = 0;
          g_bfc_wait_reg_sample = 0xFFFFFFFFu;
        }
        // [NR-SPP] 5-4-4 step 0b lived here and was DEAD CODE: the [n7] 1 Hz
        // block above (a separate, earlier gate) zeroes g_spp_* unconditionally
        // every window, so this print could never see a non-zero buffer time.
        // The live line is `[n7] skip buf=... drawstop=... walk+bulk=...`,
        // which carries the same brackets plus the 5c-attr sup/pkt split.
        // ([[reporter-gate-must-follow-implication]] -- two reporters, one
        // counter set, whoever zeroes first owns it.)
        // [NR-PKT] Phase 5-4-1. The header carries the buffer split and the
        // stateful-port classes; one sub-line per NON-STREAM op actually
        // seen, with how many buffer executions carry it and how many of
        // those draw -- the transcribe-or-refuse decision needs exactly
        // those two numbers. Stream-op totals are folded into one line as a
        // cross-check against the walk's own stats.
        if (kNrPkt && g_pkt_bufs) {
          REXGPU_INFO(
              "[nr-pkt] bufs={} drawbufs={} | t0={}pkts/{}regs t1={} t2={} | "
              "sfx scratch={}/{}b/{}db coher={}/{}b/{}db dclut={}/{}b/{}db",
              g_pkt_bufs, g_pkt_drawbufs, g_pkt_t0_pkts, g_pkt_t0_regs,
              g_pkt_t1_pkts, g_pkt_t2_pkts, g_pkt_sfx_writes[0],
              g_pkt_sfx_bufs[0], g_pkt_sfx_drawbufs[0], g_pkt_sfx_writes[1],
              g_pkt_sfx_bufs[1], g_pkt_sfx_drawbufs[1], g_pkt_sfx_writes[2],
              g_pkt_sfx_bufs[2], g_pkt_sfx_drawbufs[2]);
          REXGPU_INFO(
              "[nr-pkt]   stream nop={} draw22={} draw36={} setc={} setc2={} "
              "shc={} ldalu={} im={} imm={} bin={}",
              g_pkt_op[PM4_NOP], g_pkt_op[PM4_DRAW_INDX],
              g_pkt_op[PM4_DRAW_INDX_2], g_pkt_op[PM4_SET_CONSTANT],
              g_pkt_op[PM4_SET_CONSTANT2], g_pkt_op[PM4_SET_SHADER_CONSTANTS],
              g_pkt_op[PM4_LOAD_ALU_CONSTANT], g_pkt_op[PM4_IM_LOAD],
              g_pkt_op[PM4_IM_LOAD_IMMEDIATE],
              g_pkt_op[PM4_SET_BIN_MASK] + g_pkt_op[PM4_SET_BIN_SELECT] +
                  g_pkt_op[PM4_SET_BIN_MASK_LO] + g_pkt_op[PM4_SET_BIN_MASK_HI] +
                  g_pkt_op[PM4_SET_BIN_SELECT_LO] +
                  g_pkt_op[PM4_SET_BIN_SELECT_HI]);
          for (uint32_t op = 0; op < 128; ++op) {
            if (!g_pkt_op[op] || NrPktStreamOp(op)) continue;
            REXGPU_INFO("[nr-pkt]   OP 0x{:02X} {} n={} bufs={} drawbufs={}",
                        op, NrPktOpName(op), g_pkt_op[op], g_pkt_op_bufs[op],
                        g_pkt_op_drawbufs[op]);
          }
          for (uint32_t op = 0; op < 128; ++op) {
            g_pkt_op[op] = g_pkt_op_bufs[op] = g_pkt_op_drawbufs[op] = 0;
          }
          for (uint32_t c = 0; c < 3; ++c) {
            g_pkt_sfx_writes[c] = g_pkt_sfx_bufs[c] = g_pkt_sfx_drawbufs[c] = 0;
          }
          g_pkt_t0_pkts = g_pkt_t0_regs = g_pkt_t1_pkts = g_pkt_t2_pkts = 0;
          g_pkt_bufs = g_pkt_drawbufs = 0;
        }
        prof_last_report = now;
        last_type0_split_ns = g_type0_split_ns;
        last_spin_ns = prof_spin_ns;
        last_exec_ns = prof_exec_ns;
        last_spin_loops = prof_spin_loops;
        last_event_waits = prof_event_waits;
        last_exec_calls = prof_exec_calls;
        last_starves = prof_starves;
        for (int i = 0; i < 4; i++) {
          last_type_ns[i] = g_exec_type_ns[i];
          last_type_cnt[i] = g_exec_type_cnt[i];
        }
        last_issuedraw_ns = g_issuedraw_ns;
        last_draw_cnt = g_draw_cnt;
      }
    }

    // FIXME: We're supposed to process the WAIT_UNTIL register at this point,
    // but no games seem to actually use it.
  }

  ShutdownContext();
}

void CommandProcessor::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;

  thread::Fence fence;
  CallInThread([&fence]() {
    fence.Signal();
    thread::Thread::GetCurrentThread()->Suspend();
  });

  fence.Wait();
}

void CommandProcessor::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;

  worker_thread_->thread()->Resume();
}

bool CommandProcessor::Save(::rex::stream::ByteStream* stream) {
  assert_true(paused_);

  stream->Write<uint32_t>(primary_buffer_ptr_);
  stream->Write<uint32_t>(primary_buffer_size_);
  stream->Write<uint32_t>(read_ptr_index_);
  stream->Write<uint32_t>(read_ptr_update_freq_);
  stream->Write<uint32_t>(read_ptr_writeback_ptr_);
  stream->Write<uint32_t>(write_ptr_index_.load());

  return true;
}

bool CommandProcessor::Restore(::rex::stream::ByteStream* stream) {
  assert_true(paused_);

  primary_buffer_ptr_ = stream->Read<uint32_t>();
  primary_buffer_size_ = stream->Read<uint32_t>();
  read_ptr_index_ = stream->Read<uint32_t>();
  read_ptr_update_freq_ = stream->Read<uint32_t>();
  read_ptr_writeback_ptr_ = stream->Read<uint32_t>();
  write_ptr_index_.store(stream->Read<uint32_t>());

  return true;
}

bool CommandProcessor::SetupContext() {
  return true;
}

void CommandProcessor::ShutdownContext() {}

void CommandProcessor::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  read_ptr_index_ = 0;
  primary_buffer_ptr_ = ptr;
  primary_buffer_size_ = uint32_t(1) << (size_log2 + 3);
}

void CommandProcessor::EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) {
  // CP_RB_RPTR_ADDR Ring Buffer Read Pointer Address 0x70C
  // ptr = RB_RPTR_ADDR, pointer to write back the address to.
  read_ptr_writeback_ptr_ = ptr;
  // CP_RB_CNTL Ring Buffer Control 0x704
  // block_size = RB_BLKSZ, log2 of number of quadwords read between updates of
  //              the read pointer.
  read_ptr_update_freq_ = uint32_t(1) << block_size_log2 >> 2;
}

void CommandProcessor::UpdateWritePointer(uint32_t value) {
  write_ptr_index_ = value;
  write_ptr_index_event_->Set();
}

uint32_t CommandProcessor::ReadRegisterValue(uint32_t index) const {
  if (index < RegisterFile::kRegisterCount) {
    return register_file_->values[index];
  }
  auto it = extended_register_values_.find(index);
  return it != extended_register_values_.end() ? it->second : 0;
}

void CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  RegisterFile& regs = *register_file_;
  if (index >= RegisterFile::kRegisterCount) {
    auto [it, inserted] = extended_register_values_.insert_or_assign(index, value);
    (void)it;
    if (inserted) {
      REXGPU_WARN(
          "CommandProcessor::WriteRegister index out of bounds: {} (stored as extended register)",
          index);
    }
    return;
  }

  // Volatile for the WAIT_REG_MEM loop.
  const_cast<volatile uint32_t&>(regs.values[index]) = value;
  // [NR-5C] overlap tracking: every register-file write dirties its bucket.
  NrbStampRange(index, 1);

  // [NR-RING] Increment 4b-0: the ring observer. The 4a city verdict proved
  // per-frame swap state (PA_SC_WINDOW_OFFSET/SCISSOR_TL/BR +
  // RB_COPY_DEST_BASE) is written OUTSIDE the depth-1 IB stream, so the
  // running context also captures out-of-stream writes to its mirrored regs:
  // depth 0 = primary ring, depth >= 2 = nested IB. Depth-1 writes were
  // already applied ahead of execution by the entry walk. Cost when the
  // probe is off: one predictable branch on a bool.
  if (g_nr_ctx && index - 0x2000u <= 0x321u) {
    // [NR-WSAMP] executor-side twin: every write to a watched register, at
    // EVERY depth and from every opcode, purely observed (no context
    // mutation, so the diverge check it exists to explain stays intact).
    if (NrWatched(index)) NrExecWatch(index, value);
    if (g_pm4_ib_depth != 1 || g_nr_in_rmw || g_nr_in_cond) {
      NrCtxExternalWrite(index, value);
    }
  }
  // [PERF] GetRegisterInfo() is a ~20k-case switch, and WriteRegister sits on
  // the command-processor hot path (~1-2M calls/sec). Its only use here is to
  // gate an unknown-register DEBUG log that is off during normal play, so only
  // run the lookup when GPU debug logging is actually enabled. Short-circuits
  // to a cheap logger load + level compare in retail.
  if (auto* gpu_log = ::rex::GetLoggerRaw(::rex::log::gpu());
      gpu_log && gpu_log->should_log(spdlog::level::debug) && !regs.GetRegisterInfo(index)) {
    REXGPU_DEBUG("GPU: Write to unknown register ({:04X} = {:08X})", index, value);
  }

  // Scratch register writeback.
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    uint32_t scratch_reg = index - XE_GPU_REG_SCRATCH_REG0;
    if ((1 << scratch_reg) & regs.values[XE_GPU_REG_SCRATCH_UMSK]) {
      // Enabled - write to address.
      uint32_t scratch_addr = regs.values[XE_GPU_REG_SCRATCH_ADDR];
      uint32_t mem_addr = scratch_addr + (scratch_reg * 4);
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(mem_addr), value);
    }
  } else {
    switch (index) {
      // If this is a COHER register, set the dirty flag.
      // This will block the command processor the next time it WAIT_REG_MEMs
      // and allow us to synchronize the memory.
      case XE_GPU_REG_COHER_STATUS_HOST: {
        const_cast<volatile uint32_t&>(regs.values[index]) |= UINT32_C(0x80000000);
      } break;

      case XE_GPU_REG_DC_LUT_RW_INDEX: {
        // Reset the sequential read / write component index (see the M56
        // DC_LUT_SEQ_COLOR documentation).
        gamma_ramp_rw_component_ = 0;
      } break;

      case XE_GPU_REG_DC_LUT_SEQ_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // DC_LUT_SEQ_COLOR is in the red, green, blue order, but the write
        // enable mask is blue, green, red.
        bool write_gamma_ramp_component = (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
                                           (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          // Bits 0:5 are hardwired to zero.
          uint32_t gamma_ramp_seq_color = regs.Get<reg::DC_LUT_SEQ_COLOR>().seq_color >> 6;
          switch (gamma_ramp_rw_component_) {
            case 0:
              gamma_ramp_entry.color_10_red = gamma_ramp_seq_color;
              break;
            case 1:
              gamma_ramp_entry.color_10_green = gamma_ramp_seq_color;
              break;
            case 2:
              gamma_ramp_entry.color_10_blue = gamma_ramp_seq_color;
              break;
          }
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          ++new_gamma_ramp_rw_index.rw_index;
          WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                        rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_PWL_DATA: {
        // Should be in the PWL writing mode.
        assert_not_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // Bit 7 of the index is ignored for PWL.
        uint32_t gamma_ramp_rw_index_pwl = gamma_ramp_rw_index.rw_index & 0x7F;
        // DC_LUT_PWL_DATA is likely in the red, green, blue order because
        // DC_LUT_SEQ_COLOR is, but the write enable mask is blue, green, red.
        bool write_gamma_ramp_component = (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
                                           (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_PWL_DATA& gamma_ramp_entry =
              gamma_ramp_pwl_rgb_[gamma_ramp_rw_index_pwl][gamma_ramp_rw_component_];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_PWL_DATA>();
          // Bits 0:5 are hardwired to zero.
          gamma_ramp_entry.base = gamma_ramp_value.base & ~UINT32_C(0x3F);
          gamma_ramp_entry.delta = gamma_ramp_value.delta & ~UINT32_C(0x3F);
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          // TODO(Triang3l): Should this increase beyond 7 bits for PWL?
          // Direct3D 9 explicitly sets rw_index to 0x80 after writing the last
          // PWL entry. However, the DC_LUT_RW_INDEX documentation says that for
          // PWL, the bit 7 is ignored.
          new_gamma_ramp_rw_index.rw_index = (gamma_ramp_rw_index.rw_index & ~UINT32_C(0x7F)) |
                                             ((gamma_ramp_rw_index_pwl + 1) & 0x7F);
          WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                        rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRampPWLValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_30_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        uint32_t gamma_ramp_write_enable_mask = regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] & 0b111;
        if (gamma_ramp_write_enable_mask) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_30_COLOR>();
          if (gamma_ramp_write_enable_mask & 0b001) {
            gamma_ramp_entry.color_10_blue = gamma_ramp_value.color_10_blue;
          }
          if (gamma_ramp_write_enable_mask & 0b010) {
            gamma_ramp_entry.color_10_green = gamma_ramp_value.color_10_green;
          }
          if (gamma_ramp_write_enable_mask & 0b100) {
            gamma_ramp_entry.color_10_red = gamma_ramp_value.color_10_red;
          }
        }
        // TODO(Triang3l): Should this reset the component write index? If this
        // increase is assumed to behave like a full DC_LUT_RW_INDEX write, it
        // probably should. Currently this also calls WriteRegister for
        // DC_LUT_RW_INDEX, which resets gamma_ramp_rw_component_ as well.
        gamma_ramp_rw_component_ = 0;
        reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
        ++new_gamma_ramp_rw_index.rw_index;
        WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                      rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        if (gamma_ramp_write_enable_mask) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;
    }
  }
}

// [NR-PB] N-2-2 item 0: plain bulk store. The caller proved the range free of
// stateful ports, constant windows, extended registers and mirror/watch slots
// (NrWalkRegRange + CtxRangeOfferable), so the per-dword virtual
// WriteRegister would have been n value stores and nothing else. Same
// copy_and_swap the constant windows use; same thread as every register
// consumer (WAIT_REG_MEM polls from this thread), so the volatile per-dword
// store buys nothing here.
void CommandProcessor::WriteRegisterRangePlain(uint32_t base, uint32_t* values_be, uint32_t n) {
  memory::copy_and_swap(register_file_->values + base, values_be, n);
  NrbStampRange(base, n);  // [NR-5C] overlap tracking
}

void CommandProcessor::WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  for (uint32_t i = 0; i < num_registers; ++i) {
    uint32_t data = memory::load_and_swap<uint32_t>(base + i);
    WriteRegister(start_index + i, data);
  }
}

void CommandProcessor::WriteRegisterRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                                  uint32_t num_registers) {
  if (!num_registers) {
    return;
  }
  memory::RingBuffer::ReadRange range = ring->BeginRead(size_t(num_registers) * sizeof(uint32_t));
  if (range.first_length != 0) {
    uint32_t first_count = uint32_t(range.first_length / sizeof(uint32_t));
    WriteRegistersFromMem(base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.first)),
                          first_count);
    base += first_count;
  }
  if (range.second_length != 0) {
    WriteRegistersFromMem(base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.second)),
                          uint32_t(range.second_length / sizeof(uint32_t)));
  }
  ring->EndRead(range);
}

void CommandProcessor::WriteALURangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                             uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4000, num_registers);
}

void CommandProcessor::WriteFetchRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                               uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4800, num_registers);
}

void CommandProcessor::WriteBoolRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                              uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4900, num_registers);
}

void CommandProcessor::WriteLoopRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                              uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4908, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                                   uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x2000, num_registers);
}

void CommandProcessor::WriteALURangeFromMem(uint32_t start_index, uint32_t* base,
                                            uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4000, base, num_registers);
}

void CommandProcessor::WriteFetchRangeFromMem(uint32_t start_index, uint32_t* base,
                                              uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4800, base, num_registers);
}

void CommandProcessor::WriteBoolRangeFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4900, base, num_registers);
}

void CommandProcessor::WriteLoopRangeFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4908, base, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromMem(uint32_t start_index, uint32_t* base,
                                                  uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x2000, base, num_registers);
}

void CommandProcessor::MakeCoherent() {
  SCOPE_profile_cpu_f("gpu");

  // Status host often has 0x01000000 or 0x03000000.
  // This is likely toggling VC (vertex cache) or TC (texture cache).
  // Or, it also has a direction in here maybe - there is probably
  // some way to check for dest coherency (what all the COHER_DEST_BASE_*
  // registers are for).
  // Best docs I've found on this are here:
  // https://web.archive.org/web/20160711162346/https://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/10/R6xx_R7xx_3D.pdf
  // https://cgit.freedesktop.org/xorg/driver/xf86-video-radeonhd/tree/src/r6xx_accel.c?id=3f8b6eccd9dba116cc4801e7f80ce21a879c67d2#n454

  // Volatile because this may be called from the WAIT_REG_MEM loop.
  volatile uint32_t* regs_volatile = register_file_->values;
  auto status_host = rex::memory::Reinterpret<reg::COHER_STATUS_HOST>(
      uint32_t(regs_volatile[XE_GPU_REG_COHER_STATUS_HOST]));
  uint32_t base_host = regs_volatile[XE_GPU_REG_COHER_BASE_HOST];
  uint32_t size_host = regs_volatile[XE_GPU_REG_COHER_SIZE_HOST];

  if (!status_host.status) {
    return;
  }

  const char* action = "N/A";
  if (status_host.vc_action_ena && status_host.tc_action_ena) {
    action = "VC | TC";
  } else if (status_host.tc_action_ena) {
    action = "TC";
  } else if (status_host.vc_action_ena) {
    action = "VC";
  }

  // TODO(benvanik): notify resource cache of base->size and type.
  REXGPU_TRACE("Make {:08X} -> {:08X} ({}b) coherent, action = {}", base_host,
               base_host + size_host, size_host, action);

  // Mark coherent.
  regs_volatile[XE_GPU_REG_COHER_STATUS_HOST] = 0;
}

void CommandProcessor::PrepareForWait() {
  trace_writer_.Flush();
}

void CommandProcessor::ReturnFromWait() {}

uint32_t CommandProcessor::ExecutePrimaryBuffer(uint32_t read_index, uint32_t write_index) {
  SCOPE_profile_cpu_f("gpu");

  // If we have a pending trace stream open it now. That way we ensure we get
  // all commands.
  if (!trace_writer_.is_open() && trace_state_ == TraceState::kStreaming) {
    uint32_t title_id =
        kernel_state_->GetExecutableModule() ? kernel_state_->GetExecutableModule()->title_id() : 0;
    auto file_name = fmt::format("{:08X}_stream.xtr", title_id);
    auto path = trace_stream_path_ / file_name;
    trace_writer_.Open(path, title_id);
    InitializeTrace();
  }

  // Adjust pointer base.
  uint32_t start_ptr = primary_buffer_ptr_ + read_index * sizeof(uint32_t);
  start_ptr = (primary_buffer_ptr_ & ~0x1FFFFFFF) | (start_ptr & 0x1FFFFFFF);
  uint32_t end_ptr = primary_buffer_ptr_ + write_index * sizeof(uint32_t);
  end_ptr = (primary_buffer_ptr_ & ~0x1FFFFFFF) | (end_ptr & 0x1FFFFFFF);

  trace_writer_.WritePrimaryBufferStart(start_ptr, write_index - read_index);

  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(primary_buffer_ptr_), primary_buffer_size_);
  reader.set_read_offset(read_index * sizeof(uint32_t));
  reader.set_write_offset(write_index * sizeof(uint32_t));

  // [GPU-RB-INCREMENTAL-RPTR] On real X360 the GPU writes the ring read pointer
  // back to the guest every read_ptr_update_freq_ quadwords as it drains the
  // ring. The guest render thread's ring flow-control (a db16cyc spin-wait,
  // guest sub_8212F708/sub_8212DF78) is built around that incremental feedback:
  // it fills the ring, then spins until the read pointer advances enough to free
  // the space it needs. Writing the read pointer back only ONCE, after the whole
  // batch (see below), starves that feedback -> in a heavy scene the producer
  // spins ~18ms/frame waiting for a full drain (measured: sub_8212F708 = ~98% of
  // the guest render thread). Emitting the writeback mid-batch turns the
  // lock-step drain into a producer/consumer pipeline. (Was Xenia's TODO.)
  const bool incremental_rptr =
      REXCVAR_GET(gpu_rb_incremental_readptr) && read_ptr_writeback_ptr_ != 0;
  uint32_t wb_stride_bytes = 0;
  uint32_t last_wb_offset = reader.read_offset();
  if (incremental_rptr) {
    // read_ptr_update_freq_ is in quadwords (8 bytes). Clamp so a degenerate
    // CP_RB_CNTL can't make us write back every packet (cache ping-pong with the
    // spinning guest) or effectively never.
    uint32_t stride = read_ptr_update_freq_ ? (read_ptr_update_freq_ << 3) : 512u;
    if (stride < 256u) stride = 256u;
    if (stride > 8192u) stride = 8192u;
    wb_stride_bytes = stride;
  }
  const uint32_t rb_mask = primary_buffer_size_ - 1;  // primary_buffer_size_ is a power of 2

  do {
    if (!ExecutePacket(&reader)) {
      // This probably should be fatal - but we're going to continue anyways.
      REXGPU_ERROR("**** PRIMARY RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
    if (incremental_rptr) {
      uint32_t cur = reader.read_offset();
      if (((cur - last_wb_offset) & rb_mask) >= wb_stride_bytes) {
        memory::store_and_swap<uint32_t>(
            memory_->TranslatePhysical(read_ptr_writeback_ptr_), cur / sizeof(uint32_t));
        last_wb_offset = cur;
      }
    }
  } while (reader.read_count());

  OnPrimaryBufferEnd();

  trace_writer_.WritePrimaryBufferEnd();

  return write_index;
}

void CommandProcessor::ExecuteIndirectBuffer(uint32_t ptr, uint32_t count) {
  SCOPE_profile_cpu_f("gpu");
  // [N7] one bracket per indirect buffer (~5.8k/s at city). Subtracting the
  // skip path's own buf total leaves the buffers that DECLINED the skip.
  const auto n7_ib_t0 = std::chrono::steady_clock::now();
  struct N7IbEnd {
    std::chrono::steady_clock::time_point t0;
    ~N7IbEnd() {
      g_n7_ib_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0).count();
      ++g_n7_ib_calls;
    }
  } n7_ib_end{n7_ib_t0};

  trace_writer_.WriteIndirectBufferStart(ptr, count * sizeof(uint32_t));

  // [PM4-CENSUS] track indirect-buffer nesting + size. Depth is unconditional
  // (cheap; keeps it correct if gpu_pm4_census toggles mid-buffer); the census
  // tallies are gated on the cvar. Draw packets executed below (via ExecutePacket
  // -> ExecutePacketType3) will see g_pm4_ib_depth > 0 and count as "indirect".
  ++g_pm4_ib_depth;
  if (g_pm4_census) { ++g_pm4_ib_count; g_pm4_ib_dwords += count; }
  // [NR-PKT] 5-4-1: open the per-execution census record for depth-1 buffers.
  if (g_nr_pkt && g_pm4_ib_depth == 1) NrPktBufBegin();

  // [NR-ORC] N-1: dump this buffer's decoded stream through a private walker
  // BEFORE any live mode reads it. Depth-1 only: a nested buffer appears in
  // its parent's dump as a delegate stop and is out of the N-1 class.
  if (g_nr_orc && g_pm4_ib_depth == 1) {
    NrOracleDumpBuffer(memory_->TranslatePhysical(ptr), ptr, count, bin_select_,
                       bin_mask_, swap_counter(), memory_);
  }

  // [NR-TMPL] N-2 rung 0: compare this buffer's walk against the record-time
  // span templates. Observation only, depth-1 only, before any live mode --
  // exactly the oracle's placement, for the oracle's reasons.
  if (g_nr_tmpl && !g_nr_tmpl_swap && g_pm4_ib_depth == 1) {
    nr::TmplCompareBuffer(memory_->TranslatePhysical(ptr), ptr, count,
                          bin_select_, bin_mask_, CtxMemRead, memory_);
    nr::TmplStats* ts = nr::TmplStatsPtr();
    if (ts->bufs && (ts->bufs & 0x7FF) == 0) {
      REXGPU_INFO(
          "[nr-tmpl] bufs={} (abort={} big={}) spans hit={} eq={} ne={} "
          "stale={} (eq={} ne={} dead={} hdrEq={} hdrNe={}) cross={} "
          "cover={}/{}dw gapPkts={} emi eq={} ne={} (reg={} rng={} sh={} "
          "stop={} byref={} aExtra={} bExtra={} aUncov={}) lkStale={} | "
          "feed={} built={} rebuilt={} same={} rej={} pfail={} wraps={} "
          "evict={}",
          ts->bufs, ts->bufs_aborted, ts->bufs_toobig, ts->spans_hit,
          ts->spans_eq, ts->spans_ne, ts->spans_stale, ts->spans_stale_eq,
          ts->spans_stale_ne, ts->spans_dead, ts->stale_hdr_eq,
          ts->stale_hdr_ne, ts->spans_cross, ts->dwords_covered,
          ts->dwords_covered + ts->dwords_gap, ts->gap_pkts, ts->emi_eq,
          ts->emi_ne, ts->emi_ne_reg, ts->emi_ne_range, ts->emi_ne_shader,
          ts->emi_ne_stop, ts->emi_ne_byref, ts->a_extra, ts->b_extra,
          ts->a_uncovered, ts->lookup_stale, ts->feed, ts->built,
          ts->rebuilt, ts->unchanged, ts->feed_reject, ts->parse_fail,
          ts->arena_wraps, ts->slot_evict);
      // [NR-TMPL] rung 1: the finalize class, from both sides -- what the
      // template recorded as placeholder windows, what those windows decoded
      // to at replay, and what the in-place patches did to the ops -- plus
      // the producer's mid-execution writes into the buffer we just compared.
      REXGPU_INFO(
          "[nr-tmpl] finalize: scanOps={} scanDw={} scanPkts={} scanOver={} "
          "opDrift={} (rangeDemoted={}) resync catchup={} ahead={} "
          "spanOver={} | neBy scan={} ahead={} catchup={} over={} plain={} | "
          "mutBufs={} mutDw={}",
          ts->scan_ops, ts->scan_dw, ts->scan_pkts, ts->scan_over,
          ts->op_drift, ts->op_drift_range, ts->rep_catchup, ts->rep_ahead,
          ts->span_overrun, ts->ne_scan, ts->ne_ahead, ts->ne_catchup,
          ts->ne_over, ts->ne_plain, ts->bufs_mutated, ts->mut_dwords);
      // [NR-PLAN] N-2-2: the flat apply plan's own economics. planSpans is
      // how many spans the compare replayed FROM a plan (vs the memo stream);
      // demoted is how many ops a failing structure guard sent back to a live
      // re-parse -- the plan's equivalent of rangeDemoted, and never zero at
      // city if the guards are doing their job.
      if (ts->plan_built || ts->plan_spans) {
        REXGPU_INFO(
            "[nr-plan] built={} fail={} ops={} guards={} (ops/span={} "
            "guards/span={}) | spans={} guarded={} demoted={}",
            ts->plan_built, ts->plan_fail, ts->plan_ops, ts->plan_guards,
            ts->plan_built ? ts->plan_ops / ts->plan_built : 0,
            ts->plan_built ? ts->plan_guards / ts->plan_built : 0,
            ts->plan_spans, ts->plan_guard_spans, ts->plan_demoted);
      }
      if (ts->mu_armed) {
        REXGPU_INFO(
            "[nr-tmpl] first-mutation: dw{} snapshot=0x{:08X} live=0x{:08X}",
            ts->mu_dw, ts->mu_before, ts->mu_after);
        ts->mu_armed = 0;
      }
      if (ts->ne_armed) {
        REXGPU_INFO(
            "[nr-tmpl] first-ne: span=0x{:08X} tmpl kind={} dw={} "
            "reg=0x{:04X} val=0x{:08X} | live kind={} dw={} reg=0x{:04X} "
            "val=0x{:08X} | hdr live=0x{:08X} stored=0x{:08X}",
            ts->ne_key, ts->ne_kind, ts->ne_dw, ts->ne_b_reg, ts->ne_b_val,
            ts->ne_a_kind, ts->ne_a_dw, ts->ne_a_reg, ts->ne_a_val,
            ts->ne_hdr_live, ts->ne_hdr_stored);
        ts->ne_armed = 0;  // re-arm: one named mismatch per report window
      }
      if (ts->st_armed) {
        REXGPU_INFO(
            "[nr-tmpl] first-stale: span=0x{:08X} diff@dw{} stored=0x{:08X} "
            "live=0x{:08X}",
            ts->st_key, ts->st_idx, ts->st_stored, ts->st_live);
        ts->st_armed = 0;
      }
      REXGPU_INFO(
          "[nr-tmpl] stale-cls (spans touching): plc={} win={} sciscp={} "
          "alu={} ftc={} bl={} regOther={} setbin={} wait={} evw={} draw={} "
          "otherOp={} unframed={}",
          ts->stale_cls[nr::kTmplScPlaceholder], ts->stale_cls[nr::kTmplScWin],
          ts->stale_cls[nr::kTmplScScissorCopy], ts->stale_cls[nr::kTmplScAlu],
          ts->stale_cls[nr::kTmplScFetch], ts->stale_cls[nr::kTmplScBoolLoop],
          ts->stale_cls[nr::kTmplScRegOther], ts->stale_cls[nr::kTmplScSetBin],
          ts->stale_cls[nr::kTmplScWait], ts->stale_cls[nr::kTmplScEvent],
          ts->stale_cls[nr::kTmplScDraw], ts->stale_cls[nr::kTmplScOtherOp],
          ts->stale_cls[nr::kTmplScUnframed]);
      REXGPU_INFO("[nr-tmpl] stale-dead (recycled region, >50% diff): {}",
                  ts->stale_cls[nr::kTmplScDead]);
      if (ts->su_armed) {
        REXGPU_INFO(
            "[nr-tmpl] first-surprise: cls={} span=0x{:08X} diff@dw{} "
            "stored=0x{:08X} live=0x{:08X}",
            ts->su_cls, ts->su_key, ts->su_idx, ts->su_stored, ts->su_live);
        ts->su_armed = 0;
      }
    }
  }

  // [NR-SKP] 5-4-2: set inside the [NR-CTX] depth-1 branch below when this
  // buffer execution runs walk-only; consumed where the executor loop would
  // otherwise run.
  bool nr_skip_buf = false;

  // [NR-CTX] Increment 4a: update the RUNNING state context, in execution
  // order, for EVERY top-level buffer -- state-only buffers carry state the
  // next buffer's draws depend on, so this cannot ride the ledger's
  // draw-carrying filter. It runs BEFORE the buffer's packets execute, so
  // the live register file still holds the carried (pre-buffer) state:
  // comparing every defined mirrored slot against it is the ground-truth
  // check that recovery state only ever arrives through this stream. Nested
  // indirect buffers are skipped and counted; a nonzero count names a
  // context hole instead of absorbing it.
  if (g_nr_ctx) {
    if (g_pm4_ib_depth != 1) {
      ++g_ctx_depth2;
    } else {
      for (uint32_t s = 0; s < nr::kCtxRegCount; ++s) {
        if (!g_ctx_state.defined[s]) continue;
        ++g_ctx_checks;
        const uint32_t reg = nr::CtxSlotReg(s);
        const uint32_t live = register_file_->values[reg];
        if (live != g_ctx_state.values[s]) {
          ++g_ctx_diverge;
          for (auto& d : g_ctx_div_top) {
            if (d.count && d.reg == reg) {
              ++d.count;
              break;
            }
            if (!d.count) {
              d.reg = reg;
              d.count = 1;
              break;
            }
          }
          if (g_ctx_div_samp_n < kCtxDivSamples) {
            g_ctx_div_samp[g_ctx_div_samp_n++] = {reg, g_ctx_state.values[s],
                                                  live, ptr};
          }
        }
      }
      // [NR-RES] Increment 4b-3: the resource half, on the SAME walk. Its
      // ground-truth compare runs here for the same reason the recovery one
      // does -- before the buffer executes, the live file holds exactly the
      // carried state a ring-order replay would have.
      if (g_nr_res) {
        nr::ResDivergence rsamp[kResDivSamples];
        const uint32_t rn = nr::ResCompareLive(
            &g_res_state, &g_res_stats, ResReadLive, register_file_,
            g_res_div_samp_n < kResDivSamples ? rsamp : nullptr,
            kResDivSamples - g_res_div_samp_n);
        for (uint32_t i = 0; i < rn && g_res_div_samp_n < kResDivSamples; ++i) {
          g_res_div_samp[g_res_div_samp_n++] = rsamp[i];
        }
        nr::ResBeginBuffer(&g_res_state);
      }
      g_nr_walk_buf = ptr;
      // [NR-FX] 5-4-0: `this` rides as reg_user so the walk's write stream can
      // reach the backend's NrWalkWriteEffects; unused by the other consumers.
      const bool has_reg_fn = g_nr_res || g_nr_draw || g_nr_walk_fx;
      nr::CtxWalkBegin(&g_ctx_walker, memory_->TranslatePhysical(ptr), count,
                       ptr, &g_ctx_state, g_ctx_flags, kNrbMaxPkts,
                       &g_ctx_walk_stats, CtxMemRead, memory_, CtxShaderSeen,
                       nullptr, NrCtxWatch, nullptr, bin_select_, bin_mask_,
                       has_reg_fn ? NrWalkRegWrite : nullptr, this,
                       g_nr_res ? NrResDraw : nullptr, nullptr);
      // [NR-SKP] 5-4-2: the skip decision, per buffer. The walk (begun above)
      // becomes the ONLY decoder; the executor's packet loop below is
      // replaced by NrSkipExecuteBuffer. Refusals are counted, never silent:
      // an open trace needs the executor's per-packet trace events, a
      // bisection window (gpu_nr_issue_from/count) would leave unarmed draws
      // running with stale active shaders, and the backend vetoes precord
      // capture / non-D3D12.
      if (g_nr_skip) {
        if (g_nri_from == 0 && g_nri_count < 0 && !trace_writer_.is_open() &&
            NrSkipBackendEligible()) {
          nr_skip_buf = true;
          // [NR-TILE] N-4 probe mode 2: a buffer submitted for a repeat tile
          // band keeps its PARSE and its DELEGATES but loses its draws and
          // its register applies - which is what buffer-level tile replay
          // would remove, plus a parse the template plan would remove too.
          // ⚠ The first cut of this probe dropped the whole buffer and hung
          // the guest inside one second (naruto_503): the delegates carry
          // the EVENT_WRITE fences the producer spin-waits on, so a buffer
          // may never be skipped wholesale.
          g_nr_tile_bufskip =
              g_nr_tile_probe == 2 &&
              register_file_->values[XE_GPU_REG_PA_SC_WINDOW_OFFSET] != 0;
          // [NR-SKP] 5-4-3: only a skip-driven buffer gets the bulk range
          // consumer (set after CtxWalkBegin zeroed the fields); every other
          // mode keeps the walker's per-dword path bit-identically.
          g_ctx_walker.range_fn = NrWalkRegRange;
          g_ctx_walker.range_user = this;
          if (g_nr_tile_bufskip) {
            ++g_tile_bufs_skipped;
            g_ctx_walker.reg_fn = NrTileRegWrite;
            g_ctx_walker.range_fn = NrTileRegRange;
          } else if (g_nr_tile_probe == 2) {
            ++g_tile_bufs_kept;
          }
          // [NR-97] N-9-7 phase A: the band decision.
          // ⚠ THE ENTRY STATE IS NOT THE BAND (naruto_731, and it is the same
          // shape as naruto_505). PA_SC_WINDOW_OFFSET is not reset between
          // buffers, so a buffer that FOLLOWS a band execution inherits its
          // offset and reads as a band at entry while its own draws are band
          // 1. The title census caught it immediately: 41% of executions read
          // band at entry, and 305 of their 427 draw stops disagreed. The
          // predicate that matters is the one at a DRAW STOP -- de-tile's own
          // -- so the entry read is only ever a hint here, and it is trusted
          // to activate early ONLY when the last execution of this exact
          // buffer had every one of its stops in a band (the memo). Otherwise
          // the first draw stop rules, and the prefix before it is applied
          // normally (counted as pre_dw: at ~220 draws per city buffer that
          // prefix is a couple of hundredths of the execution).
          g_n97_armed = (g_n97_mode || g_n97_cycle) && !g_nr_tile_bufskip;
          g_n97_bufband = false;
          g_n97_filter = false;
          g_n97_bufdrop = false;
          g_n97_bufmixed = false;
          g_n97_decided = false;
          g_n97_stops_seen = g_n97_stops_band = 0;
          g_n97_bufkey = ptr & 0x1FFFFFFFu;
          // [NR-97] The band run ends here if this execution cannot be a band.
          // The entry hint OVER-approximates (it is true for any buffer that
          // merely follows a band, see the naruto_731 note), so a false hint
          // is proof: a real band execution always has it true. A buffer that
          // slips through with a true hint flushes at its first non-band draw
          // stop instead, which is still before anything renders.
          g_n97_pre_mark = g_skp_pdw + g_skp_rng_dw;
          g_n97_rf = register_file_;
          g_n97_mem = memory_;
          if (g_n97_touched.size() != RegisterFile::kRegisterCount) {
            g_n97_touched.assign(RegisterFile::kRegisterCount, 0);
            g_n97_exec_seen.assign(RegisterFile::kRegisterCount, 0);
            g_n97_tlist.reserve(16384);
            g_n97_tentry.reserve(16384);
            g_n97_tlast.reserve(16384);
            g_n97_exec_list.reserve(4096);
          }
          if (g_n97_armed) {
            ++g_n97_bufs;
            const uint32_t* n97_rf = register_file_->values;
            const nr::DetileRegs n97_regs{
                n97_rf[XE_GPU_REG_PA_SC_WINDOW_OFFSET],
                n97_rf[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL],
                n97_rf[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR],
                n97_rf[XE_GPU_REG_RB_SURFACE_INFO],
                n97_rf[XE_GPU_REG_RB_COLOR_INFO],
                n97_rf[XE_GPU_REG_RB_DEPTH_INFO]};
            g_n97_bufband = nr::DetileIsRepeatBand(n97_regs);
            if (g_n97_bufband) ++g_n97_bufs_eband;
            if (!g_n97_bufband && g_n97_blk_n) N97FlushPending(this);
            if (g_n97_bufband && N97MemoGet(g_n97_bufkey) == kN97Clean) {
              // Both the entry hint and this buffer's own history agree:
              // activate before the first packet and lose no prefix.
              ++g_n97_bufs_entry;
              N97Activate(true);
            }
          }
        } else {
          ++g_skp_fb;
        }
      }
      // [NR-DRAW] Increment 4c: with the shadow on, the walk advances one draw
      // at a time from the execution path below, so that every per-draw
      // question is asked at that draw's moment instead of at the buffer's
      // end. Without it, finish here and keep the pre-4c behaviour exactly.
      // Under the skip the lockstep flags stay CLEAR: the skip loop owns the
      // walker and hands each draw stop over explicitly.
      if (nr_skip_buf) {
        ++g_nrd_buffers;
      } else if (g_nr_draw) {
        g_ctx_walk_active = true;
        g_ctx_walk_lockstep = true;
        ++g_nrd_buffers;
      } else {
        CtxFinishWalk();
      }
    }
  }

  // [NR-IBL] record this execution against the buffer's address.
  if (g_ib_ledger) {
    uint32_t h = (ptr >> 2) * 2654435761u;
    uint32_t i = h & (kIbLedgerSize - 1);
    IbLedgerEntry* e = nullptr;
    for (uint32_t probe = 0; probe < kIbLedgerSize; ++probe) {
      IbLedgerEntry* c = &g_ib_ledger_tab[(i + probe) & (kIbLedgerSize - 1)];
      if (c->execs && c->ptr == ptr) { e = c; break; }
      if (!c->execs) {  // free slot: claim it and count the draws once
        c->ptr = ptr;
        c->dwords = count;
        c->draws = 0;
        c->draws2 = 0;
        c->reg = 0;
        CountBufferDraws(memory_->TranslatePhysical(ptr), count, &c->draws,
                         &c->draws2, bin_select_, bin_mask_);
        ++g_ib_ledger_used;
        e = c;
        break;
      }
    }
    if (e) {
      ++e->execs;
      // [NR-REG] Ask the registry what it recorded into the range about to run,
      // and compare against the count walked out of the buffer's own packets.
      // Only buffers that actually draw are scored; the engine also replays
      // state-only buffers, which would otherwise pad every rate with matches
      // that mean nothing.
      if (e->draws) {
        // Length comes from THIS execution's packet, never from what the ledger
        // remembers. A buffer re-recorded shorter at the same address leaves the
        // granules past its new end holding the previous pass, so asking at the
        // stale length would sum two recordings. tools/nr-registry-test.cpp
        // pins both halves of that behaviour.
        e->dwords = count;
        const nr::QueryResult q = nr::QueryRange(ptr, count * 4);
        const uint32_t sum = q.draws;
        // The ledger's draw count was walked the first time this address was
        // seen in the window, but the engine re-records a city buffer roughly
        // every third frame, so by now it can be stale -- and scoring a fresh
        // registry against a stale truth would invent disagreements. Re-walk
        // whenever the two differ. Agreement needs no walk (a cached count that
        // equals the current recording is not stale in the value being
        // compared), so the O(dwords) cost is paid only where the answer is
        // actually in doubt, which keeps the probe from perturbing the very
        // record/execute interleaving it is measuring.
        uint32_t truth = e->draws, truth2 = e->draws2;
        if (sum != truth) {
          CountBufferDraws(memory_->TranslatePhysical(ptr), count, &truth,
                           &truth2, bin_select_, bin_mask_);
          e->draws = truth;
          e->draws2 = truth2;
        }
        e->reg = sum;
        // The re-walk can find the buffer no longer draws at all, in which case
        // there is nothing to score against and counting it either way would be
        // a made-up number.
        if (truth) {
          ++g_reg_queries;
          g_reg_gran_hit += q.granules_hit;
          g_reg_gran_miss += q.granules_miss;
          g_reg_draws_registry += sum;
          g_reg_draws_ledger += truth;
          // Recording order must rise across a buffer written in one pass, so
          // any inversion means the registry was caught holding half of a
          // re-record. A real hazard for the renderer rather than a probe
          // artifact, so it is counted rather than smoothed away.
          if (q.seq_inversions) ++g_reg_split;
          if (sum == 0) {
            ++g_reg_r_zero;
          } else if (sum * 100u < truth * 95u) {
            ++g_reg_r_under;
          } else if (sum * 100u > truth * 105u) {
            ++g_reg_r_over;
          } else {
            ++g_reg_r_one;
          }
        }
        // [NR-CACHE]/[NR-BUF] Walk this buffer's DRAW_INDX packets once into
        // the packet scratch, then score the per-packet join (increment 1)
        // and the snapshot cache (increment 2) from it. Per
        // ExecutePacketType3_DRAW_INDX the packet is header, viz-query
        // token, VGT_DRAW_INITIATOR (prim = bits 0..5, num_indices = bits
        // 16..31). DRAW_INDX_2 (0x36) comes from unhooked paths and is not
        // scored, matching the ledger's `draws` count. Depth-1 only, keeping
        // the rates comparable across runs.
        if (g_nr_cache && truth && g_pm4_ib_depth == 1) {
          const uint8_t* raw = memory_->TranslatePhysical(ptr);
          // The snapshot layer's dirty-epoch is read BEFORE the walk: a
          // patch racing the walk then reads as dirty at the next
          // execution, never as a stale serve.
          const uint64_t epoch =
              g_nr_bufcache ? nr::SumRangeEpoch(ptr, count * 4) : 0;
          uint32_t npkt = 0;
          bool ovf = false;
          // [NR-TILE] The join list is what a replay would execute, so it
          // obeys the same predicate + bin rules the executor does: a
          // predicated-out draw never runs and must not be listed.
          nr::CtxBinState join_bin{bin_select_, bin_mask_};
          // [NR-CMP] Group accumulation: a draw's group is everything since
          // the previous draw-class packet (0x22 or 0x36, predicated or not
          // -- the bytes belong to that draw's recording either way). The
          // recorded payload's own in-group order is the body's emission
          // order, which the buffer preserves, so ordered matching holds.
          uint32_t cmp_nspan = 0, cmp_group_lo = 0;
          uint32_t cmp_nbref = 0, cmp_bgroup_lo = 0;
          bool cmp_tainted = false, cmp_first22 = true;
          for (uint32_t j = 0; j < count;) {
            const uint32_t hdr =
                __builtin_bswap32(*(const uint32_t*)(raw + j * 4));
            if (!hdr) {  // one-dword no-op, per ExecutePacket
              ++j;
              continue;
            }
            const uint32_t ty = hdr >> 30, cnt = ((hdr >> 16) & 0x3FFF) + 1;
            if (ty == 3) {
              const uint32_t op = (hdr >> 8) & 0x7F;
              if (nr::CtxPredicatedOut(join_bin, hdr)) {
                if (g_nr_compose && (op == 0x22 || op == 0x36)) {
                  // A predicated-out draw still consumed its group's bytes.
                  cmp_group_lo = cmp_nspan;
                  cmp_bgroup_lo = cmp_nbref;
                  cmp_tainted = false;
                }
                j += 1 + cnt;
                continue;
              }
              nr::CtxApplyBinPacket(
                  &join_bin, op,
                  (j + 1 < count) ? __builtin_bswap32(
                                        *(const uint32_t*)(raw + (j + 1) * 4))
                                  : 0,
                  (j + 2 < count) ? __builtin_bswap32(
                                        *(const uint32_t*)(raw + (j + 2) * 4))
                                  : 0);
              if (op == 0x22 && j + 2 < count) {
                if (npkt >= kNrbMaxPkts) {
                  ovf = true;
                  break;
                }
                const uint32_t init = __builtin_bswap32(
                    *(const uint32_t*)(raw + (j + 2) * 4));
                if (g_nr_compose) {
                  g_cmp_pkt_lo[npkt] = cmp_group_lo;
                  g_cmp_pkt_hi[npkt] = cmp_nspan;
                  g_cmp_pkt_blo[npkt] = cmp_bgroup_lo;
                  g_cmp_pkt_bhi[npkt] = cmp_nbref;
                  g_cmp_pkt_flags[npkt] =
                      (cmp_tainted ? kCmpFlagTainted : 0) |
                      (cmp_first22 ? kCmpFlagFirst : 0);
                  cmp_group_lo = cmp_nspan;
                  cmp_bgroup_lo = cmp_nbref;
                  cmp_tainted = false;
                  cmp_first22 = false;
                }
                g_nrb_pkts[npkt++] = {ptr + j * 4, init & 0x3F, init >> 16};
              } else if (g_nr_compose && op == 0x36) {
                cmp_group_lo = cmp_nspan;
                cmp_bgroup_lo = cmp_nbref;
                cmp_tainted = false;
              } else if (g_nr_compose && op == 0x2F && j + 3 < count) {
                // [N-9-4a census] by-ref load: payload = {addr, offset_type,
                // size}; target register range per the executor's own
                // Write*RangeFromMem bases.
                const uint32_t ot = __builtin_bswap32(
                    *(const uint32_t*)(raw + (j + 2) * 4));
                const uint32_t sz = __builtin_bswap32(
                                        *(const uint32_t*)(raw + (j + 3) * 4)) &
                                    0xFFF;
                const uint32_t btype = (ot >> 16) & 0xFF;
                static constexpr uint32_t kBrefBase[5] = {0x4000u, 0x4800u,
                                                          0x4900u, 0x4908u,
                                                          0x2000u};
                ++g_cmp_bref_pkts;
                ++g_cmp_bref_type[btype < 5 ? btype : 5];
                if (btype < 5 && sz) {
                  if (cmp_nbref < kCmpMaxBref) {
                    CmpBref& br = g_cmp_bref[cmp_nbref++];
                    br.reg = kBrefBase[btype] + (ot & 0x7FF);
                    br.cnt = sz;
                    br.spans_before = cmp_nspan;
                    br.off = j * 4;
                    // Raw dw1..dw3 verbatim (guest byte order), for the
                    // marker memcmp against the record body.
                    __builtin_memcpy(br.raw, raw + (j + 1) * 4, 12);
                  } else {
                    ++g_cmp_bref_drop;
                  }
                }
              }
              j += 1 + cnt;
            } else if (ty == 0) {
              if (g_nr_compose) {
                const uint32_t reg = hdr & 0x7FFF;
                if (CmpInScope(reg)) {
                  if (cmp_nspan < kCmpMaxSpans) {
                    g_cmp_spans[cmp_nspan++] = {reg, cnt, j + 1};
                  } else {
                    cmp_tainted = true;
                  }
                }
              }
              j += 1 + cnt;
            } else {
              ++j;
            }
          }
          if (ovf) {
            ++g_nrb_walk_ovf;
          } else {
            // Increment 3: the state census + per-draw context flags, from
            // the unit's own walk of the same buffer. Flags align with the
            // join list by construction (same walk rules, same packet
            // subset); a count mismatch means a decode divergence and is
            // counted, never guessed through.
            bool state_flags_ok = false;
            if (g_nr_state) {
              static nr::StateWalkResult s_swr;
              const uint32_t nf = nr::WalkBufferState(raw, count, &s_swr,
                                                      g_nrs_flags, kNrbMaxPkts);
              ++g_nrs_execs;
              for (uint32_t c = 0; c < nr::kRegClassCount; ++c) {
                g_nrs_regs[c] += s_swr.reg_writes[c];
              }
              for (uint32_t op = 0; op < 128; ++op) {
                g_nrs_ops[op] += s_swr.type3_ops[op];
              }
              g_nrs_t0 += s_swr.type0_pkts;
              g_nrs_t1 += s_swr.type1_pkts;
              g_nrs_t2 += s_swr.type2_pkts;
              g_nrs_t3 += s_swr.type3_pkts;
              g_nrs_copy += s_swr.copy_draws;
              g_nrs_mode_inherited += s_swr.inherited_draws;
              g_nrs_mode_mem += s_swr.mode_loads_from_mem;
              if (s_swr.reg_writes[nr::kRegCopy]) {
                g_nrs_copy_control = s_swr.last_copy_control;
                g_nrs_copy_dest = s_swr.last_copy_dest_base;
                g_nrs_copy_dest_info = s_swr.last_copy_dest_info;
              }
              for (const auto& o : s_swr.other_top) {
                if (!o.count) break;
                for (auto& g : g_nrs_other) {
                  if (g.count && g.reg == o.reg) {
                    g.count += o.count;
                    break;
                  }
                  if (!g.count) {
                    g.reg = o.reg;
                    g.count = o.count;
                    break;
                  }
                }
              }
              if (nf == npkt) {
                state_flags_ok = true;
                g_nrs_draws += nf;
                for (uint32_t i = 0; i < nf; ++i) {
                  if (g_nrs_flags[i] & nr::kDrawFlagRtSet) ++g_nrs_rt_set;
                  if (g_nrs_flags[i] & nr::kDrawFlagVportSet) ++g_nrs_vport_set;
                }
              } else {
                ++g_nrs_desync;
              }
            }
            // Increment 1: the per-packet join rates.
            ++g_nrc_execs;
            bool full = true;
            bool cmp_prev_missed = false;  // [NR-CMP] split-group guard
            for (uint32_t i = 0; i < npkt; ++i) {
              const nr::PacketRef& pk = g_nrb_pkts[i];
              ++g_nrc_pkts;
              nr::DrawRecord rec;
              if (!nr::LookupDraw(pk.addr, &rec)) {
                ++g_nrc_miss;
                full = false;
                cmp_prev_missed = true;
                // The miss-attribution the increment-3 flags enable: is the
                // unhooked draw a resolve (copy mode), a draw whose mode the
                // buffer inherits, or a plain geometry miss (torn window)?
                if (state_flags_ok) {
                  const uint8_t f = g_nrs_flags[i];
                  if (f & nr::kDrawFlagCopyMode) {
                    ++g_nrs_miss_copy;
                  } else if (f & nr::kDrawFlagModeInherited) {
                    ++g_nrs_miss_inherited;
                  } else {
                    ++g_nrs_miss_plain;
                  }
                }
                if (g_nrc_samp_n < kNrcSamples) {
                  g_nrc_samp[g_nrc_samp_n++] = {ptr, pk.addr, i,
                                                pk.prim, pk.count, 0,
                                                0,   0,       0,
                                                1};
                }
              } else {
                ++g_nrc_hit;
                bool args_ok = true;
                // rid 0 (sub_821375A0) draws entirely from device state
                // and takes no draw arguments; there is nothing to
                // compare, and counting it as a mismatch would misread
                // the hook layer.
                if (rec.rid == 0) {
                  ++g_nrc_rid0;
                } else {
                  const bool prim_eq = (rec.prim & 0x3F) == pk.prim;
                  const bool cnt_eq = rec.count == pk.count;
                  if (prim_eq && cnt_eq) {
                    ++g_nrc_arg_eq;
                  } else {
                    if (!prim_eq) ++g_nrc_prim_ne;
                    if (!cnt_eq) ++g_nrc_cnt_ne;
                    full = false;
                    args_ok = false;
                    if (g_nrc_samp_n < kNrcSamples) {
                      g_nrc_samp[g_nrc_samp_n++] = {
                          ptr,     pk.addr,  i,         pk.prim, pk.count,
                          rec.rid, rec.prim, rec.start, rec.count,
                          0};
                    }
                  }
                }
                // [NR-CMP] N-9-3d: compose this draw's state from the stored
                // capture payload and diff it against the group's actual
                // in-scope type0 spans, ordered scan-forward exactly like the
                // record-side CapDiff. Every refusal is a named counter so
                // the gate's coverage stays quotable.
                if (g_nr_compose) {
                  if (!rec.state_gen) {
                    ++g_cmp_no_state;
                  } else if (cmp_prev_missed) {
                    // A 65535-split group registers only its LAST packet; the
                    // earlier splits miss the join and this group's span range
                    // holds only the final split's writes.
                    ++g_cmp_skip_prevmiss;
                  } else if (g_cmp_pkt_flags[i] & kCmpFlagTainted) {
                    ++g_cmp_skip_ovf;
                  } else if (!args_ok) {
                    // The known torn re-record window (record stored after
                    // the packet): the args already read mixed, so the
                    // payload cannot be paired with these bytes either.
                    ++g_cmp_torn;
                  } else {
                    uint32_t ndw = 0;
                    const uint32_t* body = nr::AcquireState(rec, &ndw);
                    if (!body) {
                      ++g_cmp_stale;
                    } else {
                      // Parse the body's run table into random-access form
                      // (the scan-forward match needs to look ahead).
                      constexpr uint32_t kCmpMaxRuns = 192;
                      uint32_t r_reg[kCmpMaxRuns], r_cnt[kCmpMaxRuns],
                          r_off[kCmpMaxRuns];
                      const uint32_t nruns = body[0];
                      // body[1] = pre_span: bytes from the recorder body's
                      // start to the DRAW_INDX header. Spans BELOW that
                      // floor were emitted between bodies (standalone
                      // coherency flushes) and belong to no record.
                      const uint32_t pre_span = ndw >= 2 ? body[1] : 0;
                      const uint32_t pkt_byte = pk.addr - ptr;
                      const uint32_t floor_byte =
                          pkt_byte > pre_span ? pkt_byte - pre_span : 0;
                      // [NR-4B-EXT] marker entries (tag bit 31) separate from
                      // the run table: they diff against the group's 0x2F
                      // packets, never against type0 spans.
                      uint32_t mk_pos[kCmpMaxMk], mk_boff[kCmpMaxMk];
                      uint32_t nmk = 0;
                      bool mk_ovf = false;
                      uint32_t nr_parsed = 0, p = 2;
                      bool cmp_body_sup = false;
                      bool malformed = nruns > kCmpMaxRuns + kCmpMaxMk;
                      for (uint32_t k = 0; !malformed && k < nruns; ++k) {
                        if (p + 2 > ndw || body[p + 1] == 0 ||
                            p + 2 + body[p + 1] > ndw) {
                          malformed = true;
                          break;
                        }
                        if (body[p] & 0x80000000u) {
                          // [NR-5A] draw entries (bit30) carry the draw
                          // packet's dwords; not part of the span/marker diff.
                          // [NR-5B-3] tag bit 0 = suppressed: the group's
                          // covered spans do not exist -- skip the diff.
                          if (body[p] & 0x40000000u) {
                            if (body[p] & 1u) cmp_body_sup = true;
                            p += 2 + body[p + 1];
                            continue;
                          }
                          if (nmk < kCmpMaxMk && body[p + 1] == 3) {
                            mk_pos[nmk] = nr_parsed;
                            mk_boff[nmk] = p + 2;
                            ++nmk;
                          } else {
                            mk_ovf = true;
                          }
                          p += 2 + body[p + 1];
                          continue;
                        }
                        if (nr_parsed >= kCmpMaxRuns) {
                          malformed = true;
                          break;
                        }
                        r_reg[nr_parsed] = body[p];
                        r_cnt[nr_parsed] = body[p + 1];
                        r_off[nr_parsed] = p + 2;
                        p += 2 + body[p + 1];
                        ++nr_parsed;
                      }
                      if (malformed || p != ndw) {
                        // A well-formed header over an inconsistent body can
                        // only be a mid-overwrite read: torn, discard.
                        ++g_cmp_torn;
                      } else if (cmp_body_sup) {
                        // [NR-5B-3] suppressed group: its covered spans were
                        // never emitted -- the diff has nothing to pair.
                        ++g_cmp_skip_sup;
                      } else {
                        uint64_t l_run_eq = 0, l_val_eq = 0, l_val_ne = 0,
                                 l_pred_only = 0, l_emit_only = 0;
                        CmpSample l_samp = {};
                        bool l_samp_set = false;
                        auto note = [&](uint32_t kind, uint32_t reg,
                                        uint32_t pcnt, uint32_t ecnt) {
                          if (l_samp_set) return;
                          l_samp = {kind, reg, pcnt, ecnt, ptr, pk.addr};
                          l_samp_set = true;
                        };
                        const uint32_t lo = g_cmp_pkt_lo[i];
                        const uint32_t hi = g_cmp_pkt_hi[i];
                        uint32_t pi = 0;
                        uint32_t l_outside = 0;
                        for (uint32_t s = lo; s < hi; ++s) {
                          const CmpSpan& sp = g_cmp_spans[s];
                          if ((sp.off - 1) * 4 < floor_byte) {
                            // The span's packet HEADER (off-1) sits before
                            // the body start: a between-body emission, not
                            // this record's.
                            ++l_outside;
                            continue;
                          }
                          uint32_t look = pi;
                          while (look < nr_parsed &&
                                 !(r_reg[look] == sp.reg &&
                                   r_cnt[look] == sp.cnt)) {
                            ++look;
                          }
                          if (look < nr_parsed) {
                            for (; pi < look; ++pi) {
                              ++l_pred_only;
                              note(1, r_reg[pi], r_cnt[pi], 0);
                            }
                            ++l_run_eq;
                            if (__builtin_memcmp(
                                    raw + sp.off * 4,
                                    (const uint8_t*)(body + r_off[pi]),
                                    (size_t)sp.cnt * 4) == 0) {
                              ++l_val_eq;
                            } else {
                              ++l_val_ne;
                              note(3, sp.reg, r_cnt[pi], sp.cnt);
                            }
                            ++pi;
                          } else {
                            ++l_emit_only;
                            note(2, sp.reg, 0, sp.cnt);
                          }
                        }
                        for (; pi < nr_parsed; ++pi) {
                          ++l_pred_only;
                          note(1, r_reg[pi], r_cnt[pi], 0);
                        }
                        // [NR-4B-EXT] marker diff: pair the body's stored
                        // by-ref markers, in order, against the group's
                        // in-floor 0x2F packets. Outside spans form a prefix
                        // of the group (byte order + fixed floor), so the
                        // in-floor position of a packet is its span index
                        // minus lo minus the outside count. All body reads
                        // happen here, before VerifyState; commit below.
                        uint64_t l_mk_eq = 0, l_mk_ne = 0, l_mk_pos_ne = 0;
                        uint32_t l_mk_emit = 0, mi = 0;
                        uint32_t mk_samp[4] = {};
                        {
                          const uint32_t mblo = g_cmp_pkt_blo[i];
                          const uint32_t mbhi = g_cmp_pkt_bhi[i];
                          for (uint32_t b = mblo; b < mbhi; ++b) {
                            const CmpBref& br = g_cmp_bref[b];
                            if (br.off < floor_byte) continue;
                            const uint32_t sp_rel =
                                br.spans_before > lo ? br.spans_before - lo : 0;
                            const uint32_t emit_pos =
                                sp_rel > l_outside ? sp_rel - uint32_t(l_outside)
                                                   : 0;
                            ++l_mk_emit;
                            if (mi < nmk) {
                              if (__builtin_memcmp(body + mk_boff[mi], br.raw,
                                                   12) != 0) {
                                ++l_mk_ne;
                                if (!mk_samp[0]) {
                                  mk_samp[0] = 1;
                                  mk_samp[1] = mk_pos[mi];
                                  mk_samp[2] = emit_pos;
                                  mk_samp[3] = br.raw[0];
                                }
                              } else if (mk_pos[mi] != emit_pos) {
                                ++l_mk_pos_ne;
                                if (!mk_samp[0]) {
                                  mk_samp[0] = 2;
                                  mk_samp[1] = mk_pos[mi];
                                  mk_samp[2] = emit_pos;
                                  mk_samp[3] = br.raw[0];
                                }
                              } else {
                                ++l_mk_eq;
                              }
                              ++mi;
                            }
                          }
                        }
                        const bool l_mk_cnt_ne = l_mk_emit != nmk;
                        if (l_mk_cnt_ne && !mk_samp[0]) {
                          mk_samp[0] = 3;
                          mk_samp[1] = nmk;
                          mk_samp[2] = l_mk_emit;
                        }
                        // Commit only if the payload survived the diff
                        // un-overwritten; otherwise everything above was
                        // read against recycled arena bytes.
                        if (!nr::VerifyState(rec)) {
                          ++g_cmp_torn;
                        } else {
                          ++g_cmp_groups;
                          // [NR-4B-EXT] marker verdict commit.
                          if (mk_ovf) {
                            ++g_cmp_mk_ovf;
                          } else {
                            g_cmp_mk_pred += nmk;
                            g_cmp_mk_emit += l_mk_emit;
                            g_cmp_mk_eq += l_mk_eq;
                            g_cmp_mk_ne += l_mk_ne;
                            g_cmp_mk_pos_ne += l_mk_pos_ne;
                            if (l_mk_cnt_ne) ++g_cmp_mk_cnt_ne;
                            if (mk_samp[0] && !g_cmp_mk_samp[4]) {
                              g_cmp_mk_samp[0] = mk_samp[0];
                              g_cmp_mk_samp[1] = mk_samp[1];
                              g_cmp_mk_samp[2] = mk_samp[2];
                              g_cmp_mk_samp[3] = mk_samp[3];
                              g_cmp_mk_samp[4] = 1;
                            }
                          }
                          g_cmp_pred_runs += nr_parsed;
                          g_cmp_emit_runs += (hi - lo) - l_outside;
                          g_cmp_outside += l_outside;
                          // [N-9-4a census] does any of this group's by-ref
                          // (0x2F) target ranges overlap the record's runs?
                          const uint32_t blo = g_cmp_pkt_blo[i];
                          const uint32_t bhi = g_cmp_pkt_bhi[i];
                          if (bhi > blo) {
                            ++g_cmp_bref_groups;
                            for (uint32_t b = blo; b < bhi; ++b) {
                              const CmpBref& br = g_cmp_bref[b];
                              bool ovl = false;
                              for (uint32_t k = 0; k < nr_parsed && !ovl;
                                   ++k) {
                                ovl = br.reg < r_reg[k] + r_cnt[k] &&
                                      r_reg[k] < br.reg + br.cnt;
                              }
                              if (ovl) {
                                ++g_cmp_bref_overlap;
                                if (hi > br.spans_before) ++g_cmp_bref_ilv;
                                if (!g_cmp_bref_samp[2]) {
                                  g_cmp_bref_samp[0] = br.reg;
                                  g_cmp_bref_samp[1] = br.cnt;
                                  g_cmp_bref_samp[2] = 1;
                                }
                              }
                            }
                          }
                          g_cmp_run_eq += l_run_eq;
                          g_cmp_val_eq += l_val_eq;
                          g_cmp_val_ne += l_val_ne;
                          g_cmp_pred_only += l_pred_only;
                          g_cmp_emit_only += l_emit_only;
                          if (l_val_ne | l_pred_only | l_emit_only) {
                            ++g_cmp_div_groups;
                            if (g_cmp_pkt_flags[i] & kCmpFlagFirst) {
                              ++g_cmp_first_div;
                            }
                            if (l_samp_set) {
                              CmpSampleSet(l_samp.kind, l_samp.reg,
                                           l_samp.pred_cnt, l_samp.emit_cnt,
                                           l_samp.buf, l_samp.pkt);
                            }
                          }
                        }
                      }
                    }
                  }
                }
                cmp_prev_missed = false;
              }
            }
            if (full) ++g_nrc_execs_full;
            // Increment 2: how this execution would be SERVED. A valid
            // snapshot is what the renderer replays without walking; in
            // probe mode it is verified against a live join instead --
            // any difference is a stale serve, the must-be-zero gate.
            if (g_nr_bufcache) {
              ++g_nrb_execs;
              g_nrb_draws_total += npkt;
              const nr::BufSnapshot* snap = nullptr;
              const nr::BufQuery q =
                  nr::QuerySnapshot(ptr, count, epoch, &snap);
              if (q == nr::BufQuery::kValid) {
                ++g_nrb_served;
                g_nrb_draws_served += npkt;
                if (nr::VerifySnapshot(snap, g_nrb_pkts, npkt)) {
                  ++g_nrb_vok;
                } else {
                  ++g_nrb_vne;
                }
              } else {
                if (q == nr::BufQuery::kAbsent) {
                  ++g_nrb_absent;
                } else if (q == nr::BufQuery::kDirty) {
                  ++g_nrb_dirty;
                } else {
                  ++g_nrb_resized;
                }
                // Not servable: try to (re)admit from this walk's clean
                // join; rejection means the buffer is mid-patch and the
                // next execution retries. Outcomes are the unit's stats.
                nr::AdmitFromPackets(ptr, count, g_nrb_pkts, npkt, epoch);
              }
            }
          }
        }
      }
    } else {
      ++g_ib_ledger_evictions;
    }
  }

  // [PM4-IB-DUMP] one-shot decode + hex of the first sizable indirect buffer.
  // ~37/frame in the forest, so gate on the launch-time bool first (short-
  // circuits before the static once s_ib_dumped is set). PM4 in guest memory is
  // big-endian -> bswap to host order for decode/display.
  {
    static int s_ib_dumps = 0;
    if (g_pm4_ib_dump && s_ib_dumps < 3 && count >= 64) {
      const uint8_t* raw = memory_->TranslatePhysical(ptr);
      auto rd = [&](uint32_t k) { return __builtin_bswap32(*(const uint32_t*)(raw + k * 4)); };
      // Walk the whole buffer: count draws, and measure consecutive same-VB runs
      // (the foliage signature -- reg 0x4800 fetch-const, first payload dword, is
      // the VB base/key; identical across a run of the same mesh). Skip menu
      // sync/state buffers by requiring a substantial draw count.
      uint32_t draw_count = 0, first_draw_i = 0, last_vb = 0;
      uint32_t run = 0, run_vb = 0, longest_run = 0, longest_vb = 0, num_runs = 0;
      for (uint32_t j = 0; j < count;) {
        uint32_t h = rd(j), ty = h >> 30;
        if (ty == 3) {
          uint32_t op = (h >> 8) & 0x7F, cnt = ((h >> 16) & 0x3FFF) + 1;
          if (op == 0x22 || op == 0x36) {
            if (!draw_count) first_draw_i = j;
            ++draw_count;
            if (run && last_vb == run_vb) { ++run; }
            else { if (run > longest_run) { longest_run = run; longest_vb = run_vb; } run = 1; run_vb = last_vb; ++num_runs; }
          }
          j += 1 + cnt;
        } else if (ty == 0) {
          uint32_t reg = h & 0x7FFF, cnt = ((h >> 16) & 0x3FFF) + 1;
          if (reg == 0x4800 && j + 1 < count) last_vb = rd(j + 1);
          j += 1 + cnt;
        } else j++;
      }
      if (run > longest_run) { longest_run = run; longest_vb = run_vb; }
      if (draw_count >= 150) {  // a substantial geometry buffer (menu geom buffers ~<=73)
        ++s_ib_dumps;
        REXGPU_INFO("[pm4-ib-dump] GEOM IB @ {:08X} count={} draws={} | consecutive same-VB runs: "
                    "longest={} (VB={:08X}) num_runs={} -- decode from dw{}:",
                    ptr, count, draw_count, longest_run, longest_vb, num_runs,
                    first_draw_i > 24 ? first_draw_i - 24 : 0);
        const uint32_t start = first_draw_i > 24 ? first_draw_i - 24 : 0;
        uint32_t i = start, pk = 0, cur_vb = 0;
        while (i < count && pk < 100) {
          uint32_t hdr = rd(i), type = hdr >> 30;
          if (type == 3) {
            uint32_t op = (hdr >> 8) & 0x7F, cnt = ((hdr >> 16) & 0x3FFF) + 1;
            if (op == 0x22 || op == 0x36)
              REXGPU_INFO("[pm4-ib-dump]   [{:04}] DRAW op=0x{:02X} cnt={} VB={:08X}", i, op, cnt, cur_vb);
            else
              REXGPU_INFO("[pm4-ib-dump]   [{:04}] type3 op=0x{:02X} cnt={} p0={:08X}",
                          i, op, cnt, i + 1 < count ? rd(i + 1) : 0);
            i += 1 + cnt;
          } else if (type == 0) {
            uint32_t reg = hdr & 0x7FFF, cnt = ((hdr >> 16) & 0x3FFF) + 1;
            if (reg == 0x4800 && i + 1 < count) cur_vb = rd(i + 1);
            REXGPU_INFO("[pm4-ib-dump]   [{:04}] type0 cnt={} reg=0x{:04X}{}", i, cnt, reg,
                        reg == 0x4800 ? " (VB)" : reg == 0x4000 ? " (xform)" : "");
            i += 1 + cnt;
          } else { i++; }
          pk++;
        }
      }
    }
  }

  // Execute commands!
  if (nr_skip_buf) {
    // [NR-SKP] 5-4-2: walk-only execution -- the executor loop below never
    // runs for this buffer.
    NrSkipExecuteBuffer(ptr, count);
  } else {
    memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
    reader.set_write_offset(count * sizeof(uint32_t));
    do {
      if (!ExecutePacket(&reader)) {
        // Return up a level if we encounter a bad packet.
        REXGPU_ERROR("**** INDIRECT RINGBUFFER: Failed to execute packet.");
        assert_always();
        break;
      }
    } while (reader.read_count());
  }

  // [NR-DRAW] The lockstep walk ends where the buffer does: apply whatever
  // trailing state follows the last draw, then tally. Guarded on the flag
  // rather than on g_nr_draw so a mid-buffer cvar flip cannot finish a walk
  // that was never begun.
  if (g_ctx_walk_active && g_pm4_ib_depth == 1) {
    g_ctx_walk_active = false;
    g_ctx_walk_lockstep = false;
    CtxFinishWalk();
  }

  // [NR-PKT] 5-4-1: fold this execution's op/class sets into the tallies.
  if (g_nr_pkt && g_pm4_ib_depth == 1) NrPktBufEnd();

  --g_pm4_ib_depth;

  trace_writer_.WriteIndirectBufferEnd();
}

// [NR-PLAN] N-2-2 item 2: the swap's stop source. The plan covers a SPAN, not
// a buffer, so unlike every earlier consumer this cannot let the walker run
// free to the end: it must re-offer the store at every packet boundary. Three
// states, in order of frequency -- a plan is attached and drives; a plan is
// available at the cursor and gets attached; neither, so one packet is parsed
// live (the gap class, ~1 packet per span at menu) and the offer repeats.
namespace {
bool NrSwapNextStop(uint32_t pbase, const uint8_t* raw,
                    nr::CtxDrawStop* stop) {
  static uint32_t probe_until = 0;
  if (!g_ctx_walker.cursor) probe_until = 0;
  for (;;) {
    if (g_ctx_walker.plan) {
      if (nr::CtxWalkNextStop(&g_ctx_walker, stop)) return true;
      nr::CtxPlanEnd(&g_ctx_walker);  // span exhausted: offer the next
      continue;
    }
    if (g_ctx_walker.cursor >= g_ctx_walker.dwords) return false;
    if (g_nr_tmpl_swap_probe) {
      // Cost half alone: pay the lookup, then walk the span live. The
      // suppression window keeps the offer RATE identical to the real swap's
      // (one per span, plus one per gap packet), which is the whole point.
      if (g_ctx_walker.cursor >= probe_until) {
        const uint32_t nd = nr::TmplPlanProbe(pbase, g_ctx_walker.cursor,
                                              g_ctx_walker.dwords, raw);
        if (nd) probe_until = g_ctx_walker.cursor + nd;
      }
    } else if (nr::TmplPlanAttach(&g_ctx_walker, pbase, g_ctx_walker.cursor,
                                  g_ctx_walker.dwords, raw)) {
      continue;
    }
    if (!g_nr_tmpl_swap_probe) ++nr::TmplSwapStatsPtr()->gap_pkts;
    if (nr::CtxWalkStepOne(&g_ctx_walker, stop)) return true;
  }
}
}  // namespace

// [NR-SKP] Phase 5-4-2: one eligible depth-1 buffer, walk-only. The walker
// (already begun by ExecuteIndirectBuffer) decodes the register/constant
// stream, shader loads, bins and no-ops natively -- every decoded write
// reaches the full virtual WriteRegister through NrWalkRegWrite while
// g_nr_skip_bufactive is set -- and surfaces everything else as a stop:
// draws are handed to ExecutePacketType3Draw via the pending-stop handshake
// (the proven lockstep arm + gpu_nr_issue seam run unchanged), delegate
// packets (the 5-4-1 closed list plus anything unknown) run the executor's
// own handler at the walk cursor through a span reader.
void CommandProcessor::NrSkipExecuteBuffer(uint32_t ptr, uint32_t count) {
  ++g_skp_bufs;
  g_nr_skip_bufactive = true;
  // [NR-SPP] whole-buffer bracket.
  const bool spp = g_skp_prof;
  const auto spp_buf_t0 = spp ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  const uint8_t* raw = memory_->TranslatePhysical(ptr);
  // [NR-RUSE] 5-4-5 inc 0: arm this replay's classification before any
  // packet is decoded (shadow compare + entry components + digest reset).
  // The dirty-epoch is read BEFORE the walk (a patch racing the walk reads
  // as dirty at the next execution, never as a stale serve).
  if (g_nr_ruse) {
    NrRuseBufEntry(ptr, raw, count, swap_counter(),
                   nr::SumRangeEpoch(ptr, count * 4),
                   nr::EpochActivity() != 0);
  }
  // [NR-5B-3k] the generation reference: every record written from here on
  // belongs to a LATER frame than the one this buffer is replaying. Read
  // once per execution, before any packet is decoded, so a re-record racing
  // the walk lands on the far side of the comparison.
  g_nr5b_exec_seq0 = uint32_t(nr::EpochActivity());
  // [NR-BFC] 5-4-6-0: census bracket. Candidacy is read from the ruse facts
  // at buffer end; the backend latches its stream anchor here.
  if (g_nr_bfc) {
    NrBfcBufReset();
    NrBfcBufBegin();
  }
  // [NR-WM] 5-4-8: mode selection. Byte-identical to the previous compared
  // execution + a stored stream for this entry bin state => replay (parse
  // skipped); stored stream under verify => parse + compare (ne gate); no
  // stream => record (free-rides on the parse). Changed bytes invalidate.
  // The epoch shortcut is excluded outright: its "identical" is unsound at
  // city (ep_ne 45%), and the memo must never trust it.
  // NB: the walker's watch_fn (NrCtxWatch, installed unconditionally) is
  // safe under the memo: every op that can fire it re-parses its packet
  // through CtxWalkStep, which sets the packet context itself -- refusing on
  // watch_fn here would (and once did) refuse every buffer vacuously.
  bool wm_recording = false, wm_replaying = false, wm_compare = false;
  const nr::CtxMemoStream* wm_stream = nullptr;
  uint64_t wm_sel = 0, wm_msk = 0;
  if (g_nr_wm && !g_nr_ruse_ep) {
    const uint32_t bufkey = ptr & 0x1FFFFFFFu;
    wm_sel = g_ctx_walker.bin.select;
    wm_msk = g_ctx_walker.bin.mask;
    if (!g_ruse_bytes_ident) {
      if (nr::CtxMemoInvalidate(bufkey)) ++g_wm_inval;
    } else {
      wm_stream = nr::CtxMemoFind(bufkey, count, wm_sel, wm_msk);
    }
    if (wm_stream && !g_nr_verify_base) {
      nr::CtxMemoReplayBegin(&g_ctx_walker, wm_stream);
      wm_replaying = true;
      ++g_wm_rep;
    } else if (wm_stream) {
      nr::CtxMemoRecordBegin(&g_ctx_walker);
      wm_compare = wm_recording = g_ctx_walker.rec != nullptr;
      if (wm_compare) ++g_wm_cmp;
    } else if (g_ruse_bytes_ident && !nr::CtxMemoRefused(bufkey)) {
      // Record ONLY a buffer that just proved byte-stable across one gap.
      // The city recorder patches buffers in place constantly; recording
      // every stream-less execution (the v1 policy) spent ~750 commits/s +
      // ~500 invalidations/s on churners whose streams never served one
      // replay (naruto_447, store 207MB). Stability-gated, a churner costs
      // nothing and a stable buffer records once, one execution later than
      // v1 (the lost replay is the cheap half of record-once/replay-many).
      nr::CtxMemoRecordBegin(&g_ctx_walker);
      wm_recording = g_ctx_walker.rec != nullptr;
      if (wm_recording) ++g_wm_rec;
    }
  }
  // [N9] Whole-buffer plan: one store lookup per buffer execution, then the
  // interpreter drives every CtxWalkNextStop below with zero header decode
  // for the common kinds. A guard failure recompiled inside Attach; a
  // refusal fell through to the live walk, counted. The eligibility latch
  // already excluded ruse/bfc/wm/tmpl-swap, so the mode checks here are
  // belt-and-braces against a mid-run cvar flip.
  bool n9_active = false;
  if (g_n9_on && !wm_recording && !wm_replaying && !g_nr_ruse && !g_nr_bfc) {
    n9_active =
        nr::CtxProgAttach(&g_ctx_walker, ptr & 0x1FFFFFFFu, count, raw);
    if (n9_active) {
      ++g_n9_bufs;
    } else {
      ++g_n9_walk_bufs;
    }
  }
  nr::CtxDrawStop stop;
  bool aborted = false;
  bool dsp_open = false;  // [NR-DSP] a draw's span bracket is open
  bool spr_open = false;  // [NR-SPR] a draw's record/compare bracket is open
  // [NR-PLAN] N-2-2 item 2: the memo attaches w->rec and records through
  // CtxWalkStep, which the plan replay does not call -- a stream recorded
  // under the swap would be silently short. One of the two, never both.
  // [N9] an attached whole-buffer plan replaces the span swap outright.
  const bool swap =
      g_nr_tmpl_swap && !wm_recording && !wm_replaying && !n9_active;
  const uint32_t swap_pbase = ptr & 0x1FFFFFFFu;
  // [NR-4A] arm the record-apply deferral for this buffer. Every mode that
  // reads or replaces the per-range apply stream vetoes it (their digests /
  // recordings assume decode-order applies); the tile modes swap range_fn
  // outright and never reach the defer branch anyway.
  g_nr4a_bufactive = g_nr4a_on && !wm_recording && !wm_replaying &&
                     !n9_active && !swap && !g_nr_tile_bufskip && !g_nr_ruse &&
                     !g_nr_bfc && !g_n8f_on;
  g_nr4a_ndef = 0;
  g_nr4a_group_ovf = false;
  // [NR-5B-3] with the stop-resolving compose active, predicated-out draws
  // must surface as apply-only stops: under suppression their group state
  // exists only in the record, and in the packet world it applied at decode
  // regardless of the predicate (naruto_686's corruption was this class).
  g_ctx_walker.surface_pred_draws = g_nr4a_bufactive;
  while (swap ? NrSwapNextStop(swap_pbase, raw, &stop)
              : nr::CtxWalkNextStop(&g_ctx_walker, &stop)) {
    // The delegated dispatch re-checks the predicate against the CP's own
    // bin members (including the predicated-XE_SWAP rule), and a delegated
    // nested buffer executes against them: they must hold the walk's CURRENT
    // bin state, not the buffer-entry one.
    bin_select_ = g_ctx_walker.bin.select;
    bin_mask_ = g_ctx_walker.bin.mask;
    // [NR-TIL] N-4-1: name every stop by its STREAM POSITION. The tile
    // replay keys a recording by position, not by content, because the
    // repeat bands re-execute the same packets in the same order; two
    // register loads that are already in hand make the key.
    nr_tile_draw_addr_ = (ptr & 0x1FFFFFFFu) + stop.dword * 4;
    // [NR-4A] resolve the group's deferred constant ranges at EVERY stop --
    // a 0x22 draw composes from its record (fallback counted by reason);
    // everything else (0x36 resolves, delegates -- REG_RMW and COND_WRITE
    // write registers, nested buffers run the executor loop) flushes the
    // packet ranges first so it reads exactly the state decode-order applies
    // would have left.
    // [NR-5A] the record-sourced draw arm never survives a stop boundary.
    g_nr5_draw_src = nullptr;
    // [NR-5C-ATTR] per-stop bin, cleared for EVERY stop: a stop that never
    // reaches the compose (nothing deferred, delegate, 0x36) must not inherit
    // the previous stop's tag when the drawstop bracket bins below.
    g_nr5c_cur_sup = false;
    // [NR-6] N-9-6: is this stop a draw de-tile is about to throw away? The
    // test is HOISTED above the compose, which is legal because every register
    // it reads (window offset, both scissors, surface/color/depth info) is
    // mirror-class and therefore never deferrable -- the walk has already
    // applied them. DetileObserve stays at the dispatch site (observing early
    // would re-order the band learner; naruto_505's self-latching trap), so
    // the only way the two reads can disagree is a guard trip in between --
    // counted as pred_ne, and the drop wins.
    // ⚠ naruto_724: the hoisted read is sound ONLY while no record can write
    // the surface signature between it and the dispatch. That holds on the
    // packet path (pred_ne 0 over 81 windows) and FAILS under suppression,
    // where 0x2000..0x2004 come from the record. Once a suppressed record has
    // been seen, the hoist stands down and the LATE test inside the compose
    // (after the mirror pre-pass) owns the decision instead.
    bool nr6_band = false;
    if (g_nr6_mode > 0 && g_nr_detile_mode && !stop.delegate &&
        stop.opcode == 0x22 && g_nr4a_bufactive && !g_nr5b_sup_live) {
      const uint32_t* n6_rf = register_file_->values;
      const nr::DetileRegs n6_regs{n6_rf[XE_GPU_REG_PA_SC_WINDOW_OFFSET],
                                   n6_rf[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL],
                                   n6_rf[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR],
                                   n6_rf[XE_GPU_REG_RB_SURFACE_INFO],
                                   n6_rf[XE_GPU_REG_RB_COLOR_INFO],
                                   n6_rf[XE_GPU_REG_RB_DEPTH_INFO]};
      nr6_band = nr::DetileIsRepeatBand(n6_regs);
      if (nr6_band) ++g_nr6_stops;
    }
    // [NR-5B-3j] set when THIS stop's compose declined while suppression is
    // live: the draw is about to render with no constants of its own.
    bool nr5b_fb_decline = false;
    g_nr5b_gen_skip = false;
    if (g_nr4a_bufactive && (g_nr4a_ndef || g_nr4a_group_ovf)) {
      if (nr6_band) {
        // [NR-6] a draw de-tile will drop: resolve the group by DROPPING what
        // the mode says it does not need. No compose, no flush -- the work
        // unit is gone, which is the only shape that has ever moved fps here.
        ++g_nr6_groups;
        Nr4aBandDrop(this, g_nr6_mode);
      } else if (!stop.delegate && stop.opcode == 0x22) {
        // [NR-5C] compose-time bracket while the levers/cycler are armed:
        // splits "the compose second" from everything else per phase, so a
        // null fps delta still names where the sup cost does NOT live.
        // [NR-5C-ATTR] armable on its own (gpu_nr_compose_prof) and binned by
        // the record's own SUPPRESSED tag, so a supcover cycle attributes the
        // sup-vs-pkt delta even across a phase flip inside one window.
        const bool nr5c_timed = g_nr5c_prof;
        const auto nr5c_t0 = nr5c_timed
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
        const bool nr5c_ok =
            Nr4aTryCompose(this, nr_tile_draw_addr_, raw, count, stop.dword);
        if (nr5c_timed) {
          const int bin = g_nr5c_cur_sup ? 1 : 0;
          const uint64_t d =
              uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - nr5c_t0)
                           .count());
          g_nr5c_ns[bin] += d;
          ++g_nr5c_calls[bin];
          g_n7_cmp_ns[bin] += d;
          ++g_n7_cmp_calls[bin];
        }
        if (!nr5c_ok) {
          ++g_4a_flush_draw;
          Nr4aFlush(this);
          nr5b_fb_decline = g_nr5b_sup_live;
        }
      } else {
        if (stop.delegate) {
          ++g_4a_flush_deleg;
        } else {
          ++g_4a_flush_d2;
        }
        Nr4aFlush(this);
      }
    } else if (g_nr5b_sup_live && g_nr4a_bufactive && !stop.delegate &&
               stop.opcode == 0x22) {
      // [NR-5B-3j] the SILENT hole: a draw stop with nothing deferred never
      // reaches the compose at all, so it is in none of the fallback classes
      // -- yet under suppression its group's constants were eaten by the
      // shims just the same. Counted here so the pool has a size before
      // anything is designed around it ([[count-refusals-every-granularity]]).
      ++g_nr5b_stop_nodef;
    }
    // [NR-4B] every stop ends a group: the by-ref witness resets even when
    // nothing was deferred (a pure by-ref group has ndef == 0).
    g_4a_byref_n = 0;
    // [NR-5B-3] apply-only stop: a predicated-out draw whose group state the
    // compose above resolved (composed from the record, or flushed). The
    // draw itself never executes -- nothing below (detile, census brackets,
    // dispatch) may see it.
    if (!stop.delegate && stop.pred_out) {
      ++g_nr5b_pred_stops;
      g_nr5_draw_src = nullptr;
      continue;
    }
    if (!stop.delegate) {
      // [NR-DETILE] N-5. OBSERVE FIRST, THEN DECIDE: the band pattern is
      // re-learned every frame from the packets themselves, including the ones
      // whose dispatch we are about to drop. Observing after the skip would
      // disarm the mode after one frame - the same self-latching trap that
      // made tile probe mode 2 skip every band in naruto_505.
      // Dropping only the DISPATCH is what makes this safe: the walk has
      // already applied this packet's register payload, delegates (and with
      // them the EVENT_WRITE fences the producer spin-waits on) are untouched,
      // and nothing below 0x2000 is skipped.
      {
        const uint32_t* dt_rf = register_file_->values;
        const nr::DetileRegs dt_regs{dt_rf[XE_GPU_REG_PA_SC_WINDOW_OFFSET],
                                     dt_rf[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL],
                                     dt_rf[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR],
                                     dt_rf[XE_GPU_REG_RB_SURFACE_INFO],
                                     dt_rf[XE_GPU_REG_RB_COLOR_INFO],
                                     dt_rf[XE_GPU_REG_RB_DEPTH_INFO]};
        const bool dt_is_resolve =
            (dt_rf[XE_GPU_REG_RB_MODECONTROL] & 0x7) == uint32_t(xenos::EdramMode::kCopy);
        nr::DetileObserve(dt_regs, dt_is_resolve, dt_rf[XE_GPU_REG_RB_COPY_DEST_BASE]);
        // [NR-6] the cross-check: the hoisted read (above the compose) and
        // this one must agree. If they do not, the hoisted one already threw
        // this draw's constants away, so the draw MUST NOT render -- the drop
        // wins and the disagreement is counted, never silent.
        const bool dt_repeat = nr::DetileIsRepeatBand(dt_regs);
        if (nr6_band && !dt_repeat) ++g_nr6_pred_ne;
        // [NR-97] the band rule, evaluated at the only place it is sound: a
        // DRAW STOP, against de-tile's own predicate.
        //  - the first stop DECIDES the execution (unless the memo already
        //    activated it at entry),
        //  - a later stop that disagrees means this buffer writes the band
        //    registers mid-stream (the N-2-0 in-place finalize class). Its
        //    state is already gone, so the draw must NOT render, and the
        //    buffer is refused from here on: the loss is bounded to one draw
        //    per detection instead of one per frame.
        if (g_n97_armed) {
          ++g_n97_stops_seen;
          if (dt_repeat) ++g_n97_stops_band;
          if (!g_n97_decided) {
            g_n97_decided = true;
            ++g_n97_stops;
            if (dt_repeat) {
              ++g_n97_bufs_fband;
              if (!g_n97_filter) N97Activate(false);
            }
          } else if (dt_repeat && !g_n97_filter) {
            // A band stop in an execution whose first stop was band 1: the win
            // first-stop-decides leaves on the table. Sized, never guessed.
            ++g_n97_late_band;
          }
          if (g_n97_filter && !dt_repeat) {
            // A stop inside a band execution whose own registers say band 1:
            // the buffer writes the band registers mid-stream (the N-2-0
            // in-place finalize class). Its state is CORRECT -- the store put
            // it there -- so nothing is lost and nothing is dropped. Only the
            // deferred side effects are owed, so the run ends here, before
            // this draw renders, and the rest of the buffer takes the normal
            // path. This is the whole point of storing instead of dropping:
            // the naruto_742 lost-draw class simply does not exist any more.
            ++g_n97_mixed_stops;
            if (!g_n97_bufmixed) {
              g_n97_bufmixed = true;
              ++g_n97_bufs_mixed;
              N97MemoSet(g_n97_bufkey, kN97Mixed);
            }
            if (g_n97_bufdrop) {
              N97FlushPending(this);
              g_n97_bufdrop = false;  // stand down for the rest of the buffer
            }
          }
        }
        if (dt_repeat || nr6_band) {
          if (dt_is_resolve) {
            nr::DetileCountSkippedResolve();
          } else {
            nr::DetileCountSkippedDraw();
          }
          continue;
        }
        // [NR-DETILE] N-5 drive-4 experiment: the tail rewrite (the resolve
        // that copies the fullscreen post RT back into a repeat band's region
        // of the post-pass input) is the prime suspect for the duplicated
        // overlay. Counted always; dropped only under gpu_nr_detile_tail 0.
        if (dt_is_resolve &&
            nr::DetileIsTailRewrite(dt_regs,
                                    dt_rf[XE_GPU_REG_RB_COPY_DEST_BASE])) {
          continue;
        }
      }
      // [NR-TILE] N-4 probe mode 1: skip the draw dispatch for every repeat
      // tile band. Only opcode 0x22 is dropped, so the copy-mode draws that
      // carry the resolves (all 0x36) still run and band 1 still presents.
      // The walk already applied this packet's payload, so dropping the
      // dispatch leaves the register file exactly as a normal execution
      // would - this is the same shape as a viz-killed draw.
      if (g_nr_tile_probe == 1 && stop.opcode == 0x22) {
        if (register_file_->values[XE_GPU_REG_PA_SC_WINDOW_OFFSET] != 0) {
          ++g_tile_draws_skipped;
          continue;
        }
        ++g_tile_draws_kept;
      }
      // Mode 2 drops BOTH draw kinds in a repeat band: with the applies gone
      // the copy-mode 0x36 resolves would read band 1's stale RB_COPY_* and
      // write somewhere meaningless.
      if (g_nr_tile_bufskip) {
        ++g_tile_draws_skipped;
        continue;
      }
      // [NR-5B-3j] the fallback experiment. Placed HERE, after DetileObserve
      // and before the dispatch, for the same reason tile probe 1 is: the
      // band pattern must keep being learned from every packet, including
      // the ones whose dispatch we drop (naruto_505's self-latching trap).
      // Only the dispatch goes; the walk's applies, the delegates and their
      // EVENT_WRITE fences, and everything below 0x2000 are untouched.
      if (nr5b_fb_decline && g_nr5b_fbmode == 1) {
        ++g_nr5b_fb_skipped;
        continue;
      }
      // [NR-5B-3k] the generation bisect. Same position and same reasoning as
      // the fallback skip above: dispatch only, everything else intact.
      if (g_nr5b_gen_skip) {
        ++g_nr5b_gen_skipped;
        continue;
      }
      // [N8F] The draw is about to dispatch (direct or delegated): fire the
      // deferred effects now, so IssueDraw's consumers (cbuffer dirty flags,
      // texture cache, residency) read exactly what per-range applies would
      // have left. Detile/tile-skipped draws never reach here and keep
      // accumulating, which is the point.
      if (g_n8f_nspans) N8fFlush();
      // Draw stop: the walk already sits past this draw's packet, so the
      // handler must consume THIS stop instead of advancing the walk.
      g_nr_skip_stop = stop;
      g_nr_skip_draw_pending = true;
      ++g_skp_draws;
      // [N9] attribute the draw to its stop source (plan still attached =
      // the interpreter emitted it, drift catch-up parses included).
      if (g_n9_on) {
        if (g_ctx_walker.plan) {
          ++g_n9_draws;
        } else {
          ++g_n9_walk_draws;
        }
      }
      // [NR-BFC] schedule position: delegates counted so far were pre-draw.
      if (g_nr_bfc) {
        g_bfc_buf.saw_draw = true;
        ++g_bfc_buf.draws;
      }
      // [NR-RUSE] classify BOTH direct and delegated draw stops (this is
      // before the direct/delegate split on purpose).
      if (g_nr_ruse) NrRuseDrawStop(ptr, raw, count, stop.dword);
      // [NR-DSP] 5-4-7-0: open this draw's native-span bracket AFTER the
      // classification (the verdict selects the measured population) and
      // close it at both dispatch exits below.
      if (g_nr_dsp) {
        uint32_t dsp_key = 0;
        bool dsp_r2 = false, dsp_sf = false;
        if (NrRuseCurrentDraw(&dsp_key, &dsp_r2, &dsp_sf)) {
          NrDspDrawBegin(dsp_key, dsp_r2 && dsp_sf);
          dsp_open = true;
        }
      }
      // [NR-SPR] 5-4-7-1: open after the census bracket (Begin FORCES the
      // tail-state re-emit, so the census -- if co-run -- would read the
      // forced spans; keep gate reads single-probe).
      if (g_nr_spr) {
        uint32_t spr_key = 0;
        bool spr_r2 = false, spr_sf = false;
        if (NrRuseCurrentDraw(&spr_key, &spr_r2, &spr_sf)) {
          NrSprDrawBegin(spr_key, spr_r2 && spr_sf);
          spr_open = true;
        }
      }
      // [NR-SKP] 5-4-4a: direct issue -- the walk already applied this
      // packet's register payload, so the delegated re-dispatch below is
      // pure framing. The tail consumes the pending stop exactly as the
      // delegated handler would (same lockstep arm, same orphan semantics
      // on a viz-killed draw); an odd packet shape refuses and falls
      // through to the proven delegated path.
      if (g_nr_skip_direct) {
        const auto spp_direct_t0 = spp ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        // [NR-TILP] arm the outer stamp for this stop; IssueDrawImpl's tile
        // block consumes it (and clears armed) before BeginDraw.
        if (g_tile_oprof) {
          g_tile_ostop_t0 = std::chrono::steady_clock::now();
          g_tile_ostop_armed = true;
          g_tile_orep = false;
        }
        if (NrSkipDrawDirect(stop.opcode, stop.dword, raw, count)) {
          // [NR-TILP] close opost (replay return -> bracket exit) for a draw
          // the replay served; drop the arm a viz-killed draw never consumed.
          g_tile_ostop_armed = false;
          if (g_tile_orep) {
            g_tile_ns_opost += uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - g_tile_orep_t1)
                    .count());
            ++g_tile_n_opost;
            g_tile_orep = false;
          }
          if (spp) {
            const uint64_t d =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - spp_direct_t0)
                    .count();
            g_spp_draw_ns += d;
            // [NR-5C-ATTR] same bin as the compose that fed this stop.
            if (g_nr5c_prof) {
              g_spp_draw_bin_ns[g_nr5c_cur_sup ? 1 : 0] += d;
              ++g_spp_draw_bin_n[g_nr5c_cur_sup ? 1 : 0];
            }
          }
          ++g_skp_direct;
          if (g_nr_skip_draw_pending) {
            // Viz-killed draw: the tail returned before the lockstep block,
            // same as the delegated path. Drop the stop so it cannot leak.
            g_nr_skip_draw_pending = false;
            ++g_skp_arm_orphan;
          }
          if (spr_open) {
            NrSprDrawEnd();
            spr_open = false;
          }
          if (dsp_open) {
            NrDspDrawEnd();
            dsp_open = false;
          }
          continue;
        }
        ++g_skp_direct_fb;
        // [NR-TILP] the direct decode refused before the arm was consumed.
        g_tile_ostop_armed = false;
      }
    } else {
      ++g_skp_deleg;
      ++g_skp_deleg_op[stop.opcode & 0x7F];
      // (SS) [NR-97] what a band execution still delegates, and whether any of
      // it READS the registers this mode stopped writing.
      if (g_n97_filter) {
        const uint32_t n97_dop = stop.opcode & 0x7F;
        ++g_n97_deleg;
        ++g_n97_deleg_op[n97_dop];
        if (n97_dop == PM4_WAIT_REG_MEM) {
          // Same decode the [nr-bfc] census uses: dword 1 is wait_info, dword
          // 2 is the poll address.
          const auto n97_rd = [&](uint32_t i) {
            return uint32_t(__builtin_bswap32(
                *reinterpret_cast<const uint32_t*>(raw + size_t(i) * 4)));
          };
          const uint32_t wait_info =
              stop.dword + 1 < count ? n97_rd(stop.dword + 1) : 0;
          const uint32_t poll =
              stop.dword + 2 < count ? n97_rd(stop.dword + 2) : 0;
          if (wait_info & 0x10) {
            ++g_n97_wait_mem;
          } else if (poll < 0x2000u) {
            ++g_n97_wait_coher;
          } else {
            ++g_n97_wait_other;
            ++g_n97_deleg_read;
            if (g_n97_wait_reg == 0xFFFFFFFFu) g_n97_wait_reg = poll;
          }
        } else if (n97_dop == PM4_REG_RMW || n97_dop == PM4_COND_WRITE ||
                   n97_dop == PM4_INDIRECT_BUFFER) {
          ++g_n97_deleg_read;
        }
      }
      // [N8F] A delegated nested indirect buffer runs the executor's own
      // packet loop, whose draws consume the deferred effects: flush first.
      // Every other delegate reads register VALUES at most (live already) or
      // memory, never the dirty tails. depth2=0 at city, so this never fires
      // in practice.
      if (g_n8f_nspans && (stop.opcode & 0x7F) == PM4_INDIRECT_BUFFER) {
        N8fFlush();
      }
      // [NR-WM] a nested indirect buffer's content sits outside the byte
      // gate and can steer bin state through its own SET_BIN packets: this
      // buffer never gets a memo (measured zero among byte-identical city
      // buffers, naruto_420 gate 1).
      if (wm_recording && (stop.opcode & 0x7F) == PM4_INDIRECT_BUFFER) {
        nr::CtxMemoRecordAbandon(&g_ctx_walker);
        wm_recording = wm_compare = false;
        nr::CtxMemoRefuse(ptr & 0x1FFFFFFFu);
        ++g_wm_refuse;
      }
      // [NR-BFC] delegate schedule census: what a buffer-level replay must
      // still run at its stops, and the refuse classes that disqualify a
      // buffer outright (register-writing delegates + XE_SWAP + WAIT on a
      // register the recorded stateful-port schedule cannot pre-establish).
      if (g_nr_bfc) {
        BfcPerBuf& bb = g_bfc_buf;
        ++bb.deleg;
        if (!bb.saw_draw) ++bb.deleg_before_draw;
        switch (stop.opcode & 0x7F) {
          case PM4_REG_RMW:
            bb.refuse_mask |= 1u << 0;
            ++bb.deleg_other;
            break;
          case PM4_COND_WRITE:
            bb.refuse_mask |= 1u << 1;
            ++bb.deleg_other;
            break;
          case PM4_INDIRECT_BUFFER:
            bb.refuse_mask |= 1u << 2;
            ++bb.deleg_other;
            break;
          case PM4_XE_SWAP:
            bb.refuse_mask |= 1u << 3;
            ++bb.deleg_other;
            break;
          case PM4_WAIT_REG_MEM: {
            const auto bfc_rd = [&](uint32_t i) {
              return uint32_t(__builtin_bswap32(*reinterpret_cast<const uint32_t*>(
                  raw + size_t(i) * 4)));
            };
            const uint32_t wait_info =
                stop.dword + 1 < count ? bfc_rd(stop.dword + 1) : 0;
            const uint32_t poll =
                stop.dword + 2 < count ? bfc_rd(stop.dword + 2) : 0;
            if (wait_info & 0x10) {
              ++bb.deleg_wait_mem;
            } else if (poll == XE_GPU_REG_COHER_STATUS_HOST) {
              ++bb.deleg_wait_coher;
            } else {
              ++bb.deleg_wait_other;
              bb.refuse_mask |= 1u << 4;
              g_bfc_wait_reg_sample = poll;
            }
          } break;
          case PM4_MEM_WRITE:
            ++bb.deleg_memw;
            break;
          case PM4_EVENT_WRITE:
          case PM4_EVENT_WRITE_SHD:
          case PM4_EVENT_WRITE_CFL:
          case PM4_EVENT_WRITE_EXT:
          case PM4_EVENT_WRITE_ZPD:
          case PM4_INTERRUPT:
            ++bb.deleg_events;
            break;
          default:
            ++bb.deleg_other;
            break;
        }
      }
      // [NR-RUSE] v2: a delegated packet that can write registers runs
      // OUTSIDE the walk's apply stream, so the stale set cannot see it --
      // poison the rest of this replay (counted per draw as miss2 deleg).
      if (g_nr_ruse) {
        const uint32_t dop = stop.opcode & 0x7F;
        if (dop == 0x3F /*INDIRECT_BUFFER*/ || dop == 0x45 /*COND_WRITE*/ ||
            dop == 0x21 /*REG_RMW*/) {
          g_ruse_deleg_poison = true;
        }
      }
    }
    // One packet through the executor's own dispatch. The reader spans from
    // the packet header to the buffer's end; ExecutePacket consumes exactly
    // header + count dwords of it.
    const uint32_t span_bytes = (count - stop.dword) * uint32_t(sizeof(uint32_t));
    memory::RingBuffer reader(
        const_cast<uint8_t*>(raw) + stop.dword * sizeof(uint32_t), span_bytes);
    reader.set_write_offset(span_bytes);
    const auto spp_stop_t0 = spp ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    const bool ok = ExecutePacket(&reader);
    if (spp) {
      const uint64_t d = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - spp_stop_t0)
                             .count();
      (stop.delegate ? g_spp_deleg_ns : g_spp_draw_ns) += d;
      // [NR-5C-ATTR] the delegated draw-stop path, binned by the record the
      // compose used (delegates never compose and stay in the pkt bin).
      if (g_nr5c_prof && !stop.delegate) {
        g_spp_draw_bin_ns[g_nr5c_cur_sup ? 1 : 0] += d;
        ++g_spp_draw_bin_n[g_nr5c_cur_sup ? 1 : 0];
      }
    }
    if (g_nr_skip_draw_pending) {
      // The handler returned before reaching the lockstep block (short
      // packet / invalid source select). The draw did not run in either
      // model; drop the stop so it cannot leak onto a later draw.
      g_nr_skip_draw_pending = false;
      ++g_skp_arm_orphan;
    }
    // [NR-SPR]/[NR-DSP] close the brackets for a draw that took the
    // delegated path (inner bracket first).
    if (spr_open) {
      NrSprDrawEnd();
      spr_open = false;
    }
    if (dsp_open) {
      NrDspDrawEnd();
      dsp_open = false;
    }
    if (!ok) {
      ++g_skp_exec_fail;
      // Mirror the executor loop's abort: the rest of the buffer does not
      // run, so the walk must not apply it either -- the mirror and the
      // (aborted) live state stay the same thing. A false return during
      // shutdown (WAIT_REG_MEM short-circuit) is expected and quiet.
      if (worker_running_) {
        REXGPU_ERROR("**** NR-SKIP: Failed to execute delegated packet.");
        assert_always();
      }
      aborted = true;
      break;
    }
    if (stop.delegate) {
      nr::CtxWalkSkipDelegated(&g_ctx_walker);
      // The delegate may have changed the bin members (SET_BIN inside a
      // nested indirect buffer); resume the walk from the executor's truth.
      g_ctx_walker.bin = nr::CtxBinState{bin_select_, bin_mask_};
    }
  }
  // [NR-PLAN] Detach before anything else drives the walker: CtxFinishWalk
  // under an attached plan would stop at the span end and leave the rest of
  // the buffer unapplied.
  nr::CtxPlanEnd(&g_ctx_walker);
  if (aborted) g_ctx_walker.cursor = g_ctx_walker.dwords;
  // [NR-WM] finalize: commit a clean recording, compare under verify (ne
  // re-records so the store self-heals while naming the divergence), drop
  // aborted recordings (a partially applied buffer must not be trusted).
  if (wm_recording) {
    if (aborted) {
      nr::CtxMemoRecordAbandon(&g_ctx_walker);
      ++g_wm_abort;
    } else if (wm_compare) {
      uint32_t first_ne = 0;
      if (!nr::CtxMemoRecordMatches(&g_ctx_walker, wm_stream, &first_ne)) {
        ++g_wm_ne;
        if (g_wm_ne <= 3) {
          REXGPU_WARN("[nr-wm] STREAM NE buf={:08X} op={} (re-recorded)",
                      ptr, first_ne);
        }
        nr::CtxMemoRecordCommit(&g_ctx_walker, ptr & 0x1FFFFFFFu, count,
                                wm_sel, wm_msk);
      } else {
        nr::CtxMemoRecordAbandon(&g_ctx_walker);
      }
    } else if (!nr::CtxMemoRecordCommit(&g_ctx_walker, ptr & 0x1FFFFFFFu,
                                        count, wm_sel, wm_msk)) {
      ++g_wm_cap;
    }
  } else if (wm_replaying) {
    nr::CtxMemoReplayEnd(&g_ctx_walker);
  }
  // [NR-BFC] fold BEFORE NrRuseBufEnd swaps the by-ref stream: the
  // whole-buffer byref compare must see cur vs prev, not cur vs itself.
  if (g_nr_bfc) NrBfcFold(this);
  // [NR-RUSE] this replay's by-ref stream becomes the previous one.
  if (g_nr_ruse) NrRuseBufEnd();
  // Fold this buffer's stats/flags exactly as the lockstep path does, and
  // leave the CP's bin members holding the walk's end state (an in-buffer
  // SET_BIN would otherwise be lost to the packets that follow).
  CtxFinishWalk();
  // [N8F] Trailing ranges (applied by CtxFinishWalk above) and anything an
  // abort left pending: flush before the buffer ends, so the next consumer
  // -- whatever it is -- starts from fired effects.
  if (g_n8f_nspans) N8fFlush();
  // [NR-4A] trailing constant ranges past the last stop (CtxFinishWalk just
  // decoded them into the defer list) and anything an abort left pending:
  // they carry state to the NEXT buffer in ring order, so they must land.
  if (g_nr4a_bufactive && (g_nr4a_ndef || g_nr4a_group_ovf)) {
    ++g_4a_flush_end;
    Nr4aFlush(this);
  }
  g_nr4a_bufactive = false;
  g_4a_byref_n = 0;
  bin_select_ = g_ctx_walker.bin.select;
  bin_mask_ = g_ctx_walker.bin.mask;
  if (spp) {
    g_spp_buf_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - spp_buf_t0)
                        .count();
  }
  g_nr_skip_bufactive = false;
  // [NR-97] fold this execution into the buffer's memo, then disarm. The
  // filter stayed installed through CtxFinishWalk above on purpose: trailing
  // ranges past the last stop belong to the same execution. The walker's
  // reg_fn/range_fn are re-set at every CtxWalkBegin, so nothing to restore.
  if (g_n97_armed && !g_n97_bufmixed && g_n97_stops_seen &&
      g_n97_stops_band == g_n97_stops_seen) {
    N97MemoSet(g_n97_bufkey, kN97Clean);
  }
  // (SS) snapshot what THIS band execution left, before anything after the
  // bands can touch it.
  if (g_n97_sample && g_n97_filter && !g_n97_bufdrop) N97ExecEnd();
  g_n97_armed = g_n97_bufband = g_n97_bufmixed = false;
  g_n97_filter = g_n97_bufdrop = false;
  g_n97_decided = false;
}

// [NR-SKP] Phase 5-4-3: one walk-decoded constant range, applied at range
// level. The virtual WriteRegistersFromMem is the executor's own proven bulk
// path (SET_CONSTANT/LOAD_ALU_CONSTANT ride it in every non-skip run): the
// D3D12 override does one copy_and_swap value store and ONE dirty-tail
// evaluation for the whole range -- the float check against the ACTIVE shader
// constant maps is sound once per range because the maps only change at
// draws, and a range never crosses a draw; the fetch window keeps
// per-CONSTANT granularity inside TextureFetchConstantsWritten /
// InvalidateVertexBufferResidencyRange (the 5-3b-3 residency mirror hook
// funnels through the per-slot invalidation) -- instead of a virtual
// WriteRegister per dword. Probe mirrors are then fed from the stored
// host-order values in tight loops, storing exactly what the per-dword
// NrWalkRegWrite feeds would have. Precord capture cannot be live here
// (NrSkipBackendEligible vetoes it), so WriteRegistersFromMem's segment
// branch is dead by construction.
bool CommandProcessor::NrSkipApplyRegRange(uint32_t base,
                                           const uint32_t* values_be,
                                           uint32_t n, uint32_t phys,
                                           bool from_memory) {
  uint32_t* be = const_cast<uint32_t*>(values_be);
  if (!be) {
    be = memory_->TranslatePhysical<uint32_t*>(phys);
    if (!be) return false;
  }
  const int cls = N8cClass(base);
  // [N8D] census on the range as the WALK decoded it, before any split.
  {
    ++g_n8d_rng[cls];
    if (base == g_n8d_prev_base[cls]) {
      ++g_n8d_repeat[cls];
    } else if (base == g_n8d_prev_end[cls]) {
      ++g_n8d_creg[cls];
      if (be == g_n8d_prev_mem_end[cls]) ++g_n8d_cmem[cls];
    }
    g_n8d_prev_base[cls] = base;
    g_n8d_prev_end[cls] = base + n;
    g_n8d_prev_mem_end[cls] = be + n;
    N8eAdd(cls, base, n);
  }
  // [N8C] the synthetic split.
  const int k = cls == 0 ? g_n8c_split_alu
              : cls == 1 ? g_n8c_split_fetch
                         : g_n8c_split_other;
  if (k > 1 && n >= uint32_t(k)) {
    const uint32_t chunk = n / uint32_t(k);
    uint32_t off = 0;
    bool ok = true;
    for (int i = 0; i < k; ++i) {
      const uint32_t m = (i == k - 1) ? (n - off) : chunk;
      ok = NrSkipApplyRegRangeBody(base + off, be + off, m, from_memory) && ok;
      off += m;
    }
    return ok;
  }
  return NrSkipApplyRegRangeBody(base, be, n, from_memory);
}

bool CommandProcessor::NrSkipApplyRegRangeBody(uint32_t base, uint32_t* be,
                                               uint32_t n, bool from_memory) {
  // [N8F] Coalesced fast path: values now, effects at the draw. Eligibility
  // vetoed every mode below that reads the per-range apply stream, so their
  // feeds are safely skipped here. A window straddler returns false and
  // falls through to the legacy apply (counted, never silent).
  if (g_n8f_on && N8fApplyRange(base, be, n)) {
    return true;
  }
  // [NR-RUSE] by-ref payloads bypass the buffer bytes; chain them so a draw's
  // prefix digest covers everything the packet dwords cannot. Every range
  // also drives the v2 stale set (inline via the diff bitmap, by-ref via the
  // per-buffer value-stream shadow).
  if (g_nr_ruse) {
    if (g_nr_ruse_v0 && from_memory) NrRuseFeedByrefRange(base, be, n);
    NrRuseRange(base, be, n, from_memory);
  }
  // [NR-PB] N-2-2 item 0: a plain state range (see NrWalkRegRange) must NOT
  // go through the virtual WriteRegistersFromMem -- for a non-constant range
  // both backend overrides fall through to the base per-dword loop, which is
  // exactly the slow path the widening exists to remove. The plain store is
  // its own virtual: value copy + the D3D12 gpu_instance dirty semantic.
  const uint32_t pb_end = base + n - 1;
  if (pb_end < XE_GPU_REG_SHADER_CONSTANT_000_X ||
      base > XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    WriteRegisterRangePlain(base, be, n);
    ++g_skp_plain_rng;
    g_skp_plain_dw += n;
  } else {
    WriteRegistersFromMem(base, be, n);
  }
  const uint32_t* host = &register_file_->values[base];
  if (g_nr_res) {
    nr::ResApplyRange(&g_res_state, &g_res_stats, base, host, n, from_memory);
  }
  if (g_nr_draw && g_nr_verify_base) {
    nr::RegShadowApplyRange(&g_reg_shadow, &g_reg_stats, base, host, n);
  }
  if (g_nr_issue && g_nri_seeded && !g_nri_live) {
    std::memcpy(&g_nri_file.values[base], host, n * sizeof(uint32_t));
  }
  // [NR-BFC] bulk ranges feed the census (end-state size + class split).
  if (g_nr_bfc) NrBfcApply(base, n, false);
  N8Classify(base, n, false);
  ++g_n8d_applied;
  ++g_skp_rng;
  g_skp_rng_dw += n;
  return true;
}

// [N8F] The coalesced range apply. Stores the values exactly where the legacy
// path would have (register file via copy_and_swap, issue mirror via memcpy),
// keeps every counter the legacy body feeds so the [nr-skp]/[n8]/[n8d] lines
// stay comparable across the cycler's phases, and defers only the class side
// effects. Plain ranges have no deferred effect at all (gpu_instance is
// vetoed), so they take the store and nothing else -- that is the whole win
// for the 10.9 plain applies/draw.
bool CommandProcessor::N8fApplyRange(uint32_t base, uint32_t* be, uint32_t n) {
  const uint32_t end_index = base + n - 1;
  uint8_t cls;
  bool plain = false;
  if (end_index < XE_GPU_REG_SHADER_CONSTANT_000_X ||
      base > XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    plain = true;
    cls = 0xFF;
  } else if (base >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
             end_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    cls = 0;
  } else if (base >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
             end_index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    cls = 1;
  } else if (base >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
             end_index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    cls = 2;
  } else {
    // Straddles a window boundary (or lands in the 0x48C0..0x48FF hole):
    // the legacy apply owns the shape.
    ++g_n8f_straddle;
    return false;
  }
  memory::copy_and_swap(register_file_->values + base, be, n);
  NrbStampRange(base, n);  // [NR-5C] overlap tracking
  if (g_nr_issue && g_nri_seeded && !g_nri_live) {
    std::memcpy(&g_nri_file.values[base], &register_file_->values[base],
                n * sizeof(uint32_t));
  }
  ++g_n8f_rng;
  N8Classify(base, n, false);
  ++g_n8d_applied;
  ++g_skp_rng;
  g_skp_rng_dw += n;
  if (plain) {
    ++g_skp_plain_rng;
    g_skp_plain_dw += n;
    return true;
  }
  // Accumulate [base, base+n) into the span set: absorb every same-class
  // span it overlaps or abuts (a widened span can newly touch an
  // already-scanned one, so absorbing restarts the scan; the set is <= 24
  // entries and multi-absorbs are rare).
  uint32_t nb = base, ne = base + n;
  bool absorbed = true;
  while (absorbed) {
    absorbed = false;
    for (uint32_t i = 0; i < g_n8f_nspans; ++i) {
      const N8fSpan& s = g_n8f_spans[i];
      if (s.cls == cls && nb <= s.end && ne >= s.base) {
        if (s.base < nb) nb = s.base;
        if (s.end > ne) ne = s.end;
        g_n8f_spans[i] = g_n8f_spans[--g_n8f_nspans];
        absorbed = true;
        break;
      }
    }
  }
  if (g_n8f_nspans == kN8fSpans) {
    // No slot: fire this (merged) span's effects immediately. Values are
    // stored, so this is only an early effect, never a lost one.
    NrApplyRangeEffects(nb, ne - nb);
    ++g_n8f_ovf;
    return true;
  }
  g_n8f_spans[g_n8f_nspans++] = {nb, ne, cls};
  return true;
}

// [N8F] Fire the deferred effects, one virtual call per merged span. Called
// before every draw/resolve dispatch, before a delegated nested indirect
// buffer, at an executor abort, and at buffer end -- the four places after
// which something other than the walk's own range stream may consume them.
void CommandProcessor::N8fFlush() {
  const uint32_t k = g_n8f_nspans;
  if (!k) return;
  for (uint32_t i = 0; i < k; ++i) {
    NrApplyRangeEffects(g_n8f_spans[i].base,
                        g_n8f_spans[i].end - g_n8f_spans[i].base);
  }
  g_n8f_applies += k;
  ++g_n8f_flushes;
  g_n8f_nspans = 0;
}

// [NR-SKP] Phase 5-4-4a: one draw stop, issued by direct call. The walk has
// already applied this packet's register payload (CtxApplyDrawPayload ->
// full virtual WriteRegister) before the stop returned, so re-writing
// VGT_DRAW_INITIATOR / VGT_DMA_BASE / VGT_DMA_SIZE would store the same
// values the register file already holds; only the executor's arg derivation
// remains, reproduced here from the same buffer dwords its reader would
// consume, bounds check for bounds check. Returns false -- nothing ran --
// whenever the delegated handler would have done anything beyond the plain
// issue: truncated buffer (Type3's overflow abort), short packet (the
// "packet too small" false return), immediate/invalid source select (the
// logged drop). The caller then delegates, so those shapes keep the proven
// path bit for bit.
const uint32_t* CommandProcessor::NrTranslatePhys(uint32_t phys) {
  return memory_->TranslatePhysical<const uint32_t*>(phys);
}

bool CommandProcessor::NrSkipDrawDirect(uint32_t opcode, uint32_t dword,
                                        const uint8_t* raw, uint32_t count) {
  // [NR-5A] coherent compose armed this stop: the draw's dwords come from
  // the record (verified byte-equal to the packet), so a composed draw is
  // single-source end to end. Consumed here; never survives the call.
  const uint32_t* rec_src = g_nr5_draw_src;
  const uint32_t rec_ndw = g_nr5_draw_ndw;
  g_nr5_draw_src = nullptr;
  if (rec_src) ++g_nr5_draw_issued;
  const auto rd = [raw, rec_src, rec_ndw, dword](uint32_t k) {
    if (rec_src && k >= dword && k - dword < rec_ndw) {
      return __builtin_bswap32(rec_src[k - dword]);
    }
    return __builtin_bswap32(*reinterpret_cast<const uint32_t*>(raw + k * 4));
  };
  const uint32_t hdr = rd(dword);
  const uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;
  const uint32_t end = dword + 1 + cnt;
  if (end > count) return false;
  // DRAW_INDX leads with a viz-query token (read and unused by the handler);
  // DRAW_INDX_2 does not.
  const uint32_t p = dword + 1 + (opcode == 0x22 ? 1u : 0u);
  if (p >= end) return false;
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = rd(p);
  bool is_indexed = false;
  IndexBufferInfo index_buffer_info;
  switch (vgt_draw_initiator.source_select) {
    case xenos::SourceSelect::kDMA: {
      if (p + 2 >= end) return false;
      is_indexed = true;
      const uint32_t vgt_dma_base = rd(p + 1);
      reg::VGT_DMA_SIZE vgt_dma_size;
      vgt_dma_size.value = rd(p + 2);
      uint32_t index_size_bytes = vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16
                                      ? sizeof(uint16_t)
                                      : sizeof(uint32_t);
      index_buffer_info.guest_base = vgt_dma_base & ~(index_size_bytes - 1);
      index_buffer_info.endianness = vgt_dma_size.swap_mode;
      index_buffer_info.format = vgt_draw_initiator.index_size;
      index_buffer_info.length = vgt_dma_size.num_words * index_size_bytes;
      index_buffer_info.count = vgt_draw_initiator.num_indices;
    } break;
    case xenos::SourceSelect::kAutoIndex:
      index_buffer_info.guest_base = 0;
      index_buffer_info.length = 0;
      break;
    default:
      return false;
  }
  ExecutePacketType3DrawTail(
      vgt_draw_initiator.value, is_indexed ? &index_buffer_info : nullptr,
      opcode == 0x22 ? "PM4_DRAW_INDX" : "PM4_DRAW_INDX_2", opcode);
  return true;
}

void CommandProcessor::ExecutePacket(uint32_t ptr, uint32_t count) {
  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
  reader.set_write_offset(count * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      REXGPU_ERROR("**** ExecutePacket: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());
}

bool CommandProcessor::ExecutePacket(memory::RingBuffer* reader) {
  const uint32_t packet = reader->ReadAndSwap<uint32_t>();
  const uint32_t packet_type = packet >> 30;
  if (packet == 0) {
    trace_writer_.WritePacketStart(uint32_t(reader->read_ptr() - 4), 1);
    trace_writer_.WritePacketEnd();
    return true;
  }

  if (packet == 0xCDCDCDCD) {
    REXGPU_WARN("GPU packet is CDCDCDCD - probably read uninitialized memory!");
  }

  // [NR-WSAMP] name the packet a sampled register write came from.
  if (g_nr_ctx) {
    g_nr_cur_pkt = packet_type == 3 ? (0x200u | ((packet >> 8) & 0x7F))
                                    : (0x100u | packet_type);
  }

  if (!g_exec_prof) {
    switch (packet_type) {
      case 0x00:
        return ExecutePacketType0(reader, packet);
      case 0x01:
        return ExecutePacketType1(reader, packet);
      case 0x02:
        return ExecutePacketType2(reader, packet);
      case 0x03:
        return ExecutePacketType3(reader, packet);
      default:
        assert_unhandled_case(packet_type);
        return false;
    }
  }
  // [GPU-EXEC-PROFILE] timed dispatch (accumulate per packet type).
  const auto exec_t0 = std::chrono::steady_clock::now();
  bool r;
  switch (packet_type) {
    case 0x00:
      r = ExecutePacketType0(reader, packet);
      break;
    case 0x01:
      r = ExecutePacketType1(reader, packet);
      break;
    case 0x02:
      r = ExecutePacketType2(reader, packet);
      break;
    case 0x03:
      r = ExecutePacketType3(reader, packet);
      break;
    default:
      assert_unhandled_case(packet_type);
      return false;
  }
  g_exec_type_ns[packet_type] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - exec_t0).count();
  g_exec_type_cnt[packet_type]++;
  return r;
}

bool CommandProcessor::ExecutePacketType0(memory::RingBuffer* reader, uint32_t packet) {
  // Type-0 packet.
  // Write count registers in sequence to the registers starting at
  // (base_index << 2).
  // [GPU-SPLIT] whole-call timer = the register/constant-write firehose bucket.
  const auto t0_split = g_split_prof ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};

  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType0 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }

  trace_writer_.WritePacketStart(uint32_t(reader->read_ptr() - 4), 1 + count);

  uint32_t base_index = (packet & 0x7FFF);
  uint32_t write_one_reg = (packet >> 15) & 0x1;
  // [NR-PKT] 5-4-1: type-0 register writes inside depth-1 buffers -- packet
  // count, register-store count, and the stateful-port classification. A
  // one-reg packet stores `count` times to ONE register (multiplicity kept).
  if (g_nr_pkt && g_pm4_ib_depth == 1) {
    ++g_pkt_t0_pkts;
    g_pkt_t0_regs += count;
    if (write_one_reg) {
      NrPktRegRange(base_index, base_index, count);
    } else {
      NrPktRegRange(base_index, base_index + count - 1, 1);
    }
  }
  for (uint32_t m = 0; m < count; m++) {
    uint32_t reg_data = reader->ReadAndSwap<uint32_t>();
    uint32_t target_index = write_one_reg ? base_index : base_index + m;
    WriteRegister(target_index, reg_data);
  }

  trace_writer_.WritePacketEnd();
  if (g_split_prof) {
    g_type0_split_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0_split).count();
  }
  return true;
}

bool CommandProcessor::ExecutePacketType1(memory::RingBuffer* reader, uint32_t packet) {
  // Type-1 packet.
  // Contains two registers of data. Type-0 should be more common.
  trace_writer_.WritePacketStart(uint32_t(reader->read_ptr() - 4), 3);
  uint32_t reg_index_1 = packet & 0x7FF;
  uint32_t reg_index_2 = (packet >> 11) & 0x7FF;
  // [NR-PKT] 5-4-1: type-1 writes inside depth-1 buffers.
  if (g_nr_pkt && g_pm4_ib_depth == 1) {
    ++g_pkt_t1_pkts;
    NrPktRegRange(reg_index_1, reg_index_1, 1);
    NrPktRegRange(reg_index_2, reg_index_2, 1);
  }
  uint32_t reg_data_1 = reader->ReadAndSwap<uint32_t>();
  uint32_t reg_data_2 = reader->ReadAndSwap<uint32_t>();
  WriteRegister(reg_index_1, reg_data_1);
  WriteRegister(reg_index_2, reg_data_2);
  trace_writer_.WritePacketEnd();
  return true;
}

bool CommandProcessor::ExecutePacketType2(memory::RingBuffer* reader, uint32_t packet) {
  // Type-2 packet.
  // No-op. Do nothing.
  // [NR-PKT] 5-4-1: counted so the census is exhaustive over packet types.
  if (g_nr_pkt && g_pm4_ib_depth == 1) ++g_pkt_t2_pkts;
  trace_writer_.WritePacketStart(uint32_t(reader->read_ptr() - 4), 1);
  trace_writer_.WritePacketEnd();
  return true;
}

bool CommandProcessor::ExecutePacketType3(memory::RingBuffer* reader, uint32_t packet) {
  // Type-3 packet.
  uint32_t opcode = (packet >> 8) & 0x7F;
  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  auto data_start_offset = reader->read_offset();

  // [PM4-CENSUS] count draw packets, split by inside-indirect-buffer vs primary.
  if (g_pm4_census) {
    if (opcode == PM4_DRAW_INDX) {
      if (g_pm4_ib_depth) ++g_pm4_draw_indx_indirect; else ++g_pm4_draw_indx_primary;
    } else if (opcode == PM4_DRAW_INDX_2) {
      if (g_pm4_ib_depth) ++g_pm4_draw_indx2_indirect; else ++g_pm4_draw_indx2_primary;
    }
  }

  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType3 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }

  // To handle nesting behavior when tracing we special case indirect buffers.
  if (opcode == PM4_INDIRECT_BUFFER) {
    trace_writer_.WritePacketStart(uint32_t(reader->read_ptr() - 4), 2);
  } else {
    trace_writer_.WritePacketStart(uint32_t(reader->read_ptr() - 4), 1 + count);
  }

  // & 1 == predicate - when set, we do bin check to see if we should execute
  // the packet. Only type 3 packets are affected.
  // We also skip predicated swaps, as they are never valid (probably?).
  if (packet & 1) {
    bool any_pass = (bin_select_ & bin_mask_) != 0;
    if (REXCVAR_GET(gpu_nr_detile_bin_probe)) {
      // Band 1 is the pass with no window offset; the repeat bands carry a
      // negative one. Split there, because the question is precisely whether
      // band 1 is missing packets that the repeat bands run.
      const uint32_t win = register_file_->values[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
      const int32_t win_y = (int32_t(win << 1) >> 16) / 2;
      const int band = win_y ? 1 : 0;
      static uint64_t s_pred[2] = {}, s_fail[2] = {}, s_draw[2] = {}, s_draw_fail[2] = {};
      static uint64_t s_sel = 0, s_mask = 0;
      static std::chrono::steady_clock::time_point s_last{};
      ++s_pred[band];
      s_fail[band] += any_pass ? 0 : 1;
      const bool is_draw = opcode == PM4_DRAW_INDX || opcode == PM4_DRAW_INDX_2;
      if (is_draw) {
        ++s_draw[band];
        s_draw_fail[band] += any_pass ? 0 : 1;
        s_sel = bin_select_;
        s_mask = bin_mask_;
      }
      const auto now = std::chrono::steady_clock::now();
      if (now - s_last >= std::chrono::seconds(1)) {
        s_last = now;
        REXGPU_INFO(
            "[nr-detile-bin] band0: pkt={} fail={} ({:.1f}%) draws={} dfail={} ({:.1f}%) | "
            "repeat: pkt={} fail={} ({:.1f}%) draws={} dfail={} ({:.1f}%) | "
            "last draw select={:016X} mask={:016X}",
            s_pred[0], s_fail[0], s_pred[0] ? 100.0 * double(s_fail[0]) / double(s_pred[0]) : 0.0,
            s_draw[0], s_draw_fail[0],
            s_draw[0] ? 100.0 * double(s_draw_fail[0]) / double(s_draw[0]) : 0.0,
            s_pred[1], s_fail[1], s_pred[1] ? 100.0 * double(s_fail[1]) / double(s_pred[1]) : 0.0,
            s_draw[1], s_draw_fail[1],
            s_draw[1] ? 100.0 * double(s_draw_fail[1]) / double(s_draw[1]) : 0.0,
            s_sel, s_mask);
      }
    }
    if (!any_pass || opcode == PM4_XE_SWAP) {
      reader->AdvanceRead(count * sizeof(uint32_t));
      trace_writer_.WritePacketEnd();
      return true;
    }
  }

  // [NR-PKT] 5-4-1: census at the dispatch, after the predicate skip above,
  // so only packets that actually execute are tallied.
  if (g_nr_pkt && g_pm4_ib_depth == 1) {
    const uint32_t op7 = opcode & 0x7F;
    ++g_pkt_op[op7];
    g_pkt_buf_ops[op7 >> 6] |= 1ull << (op7 & 63);
    if (op7 == PM4_DRAW_INDX || op7 == PM4_DRAW_INDX_2) g_pkt_buf_draw = true;
  }

  bool result = false;
  switch (opcode) {
    case PM4_ME_INIT:
      result = ExecutePacketType3_ME_INIT(reader, packet, count);
      break;
    case PM4_NOP:
      result = ExecutePacketType3_NOP(reader, packet, count);
      break;
    case PM4_INTERRUPT:
      result = ExecutePacketType3_INTERRUPT(reader, packet, count);
      break;
    case PM4_XE_SWAP:
      result = ExecutePacketType3_XE_SWAP(reader, packet, count);
      break;
    case PM4_INDIRECT_BUFFER:
    case PM4_INDIRECT_BUFFER_PFD:
      result = ExecutePacketType3_INDIRECT_BUFFER(reader, packet, count);
      break;
    case PM4_WAIT_REG_MEM:
      result = ExecutePacketType3_WAIT_REG_MEM(reader, packet, count);
      break;
    case PM4_REG_RMW:
      result = ExecutePacketType3_REG_RMW(reader, packet, count);
      break;
    case PM4_REG_TO_MEM:
      result = ExecutePacketType3_REG_TO_MEM(reader, packet, count);
      break;
    case PM4_MEM_WRITE:
      result = ExecutePacketType3_MEM_WRITE(reader, packet, count);
      break;
    case PM4_COND_WRITE:
      result = ExecutePacketType3_COND_WRITE(reader, packet, count);
      break;
    case PM4_EVENT_WRITE:
      result = ExecutePacketType3_EVENT_WRITE(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_SHD:
      result = ExecutePacketType3_EVENT_WRITE_SHD(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_EXT:
      result = ExecutePacketType3_EVENT_WRITE_EXT(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_ZPD:
      result = ExecutePacketType3_EVENT_WRITE_ZPD(reader, packet, count);
      break;
    case PM4_DRAW_INDX:
      result = ExecutePacketType3_DRAW_INDX(reader, packet, count);
      break;
    case PM4_DRAW_INDX_2:
      result = ExecutePacketType3_DRAW_INDX_2(reader, packet, count);
      break;
    case PM4_SET_CONSTANT:
      result = ExecutePacketType3_SET_CONSTANT(reader, packet, count);
      break;
    case PM4_SET_CONSTANT2:
      result = ExecutePacketType3_SET_CONSTANT2(reader, packet, count);
      break;
    case PM4_LOAD_ALU_CONSTANT:
      result = ExecutePacketType3_LOAD_ALU_CONSTANT(reader, packet, count);
      break;
    case PM4_SET_SHADER_CONSTANTS:
      result = ExecutePacketType3_SET_SHADER_CONSTANTS(reader, packet, count);
      break;
    case PM4_IM_LOAD:
      result = ExecutePacketType3_IM_LOAD(reader, packet, count);
      break;
    case PM4_IM_LOAD_IMMEDIATE:
      result = ExecutePacketType3_IM_LOAD_IMMEDIATE(reader, packet, count);
      break;
    case PM4_INVALIDATE_STATE:
      result = ExecutePacketType3_INVALIDATE_STATE(reader, packet, count);
      break;
    case PM4_VIZ_QUERY:
      result = ExecutePacketType3_VIZ_QUERY(reader, packet, count);
      break;

    case PM4_SET_BIN_MASK_LO: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | value;
      result = true;
    } break;
    case PM4_SET_BIN_MASK_HI: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (bin_mask_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
      result = true;
    } break;
    case PM4_SET_BIN_SELECT_LO: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | value;
      result = true;
    } break;
    case PM4_SET_BIN_SELECT_HI: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (bin_select_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
      result = true;
    } break;
    case PM4_SET_BIN_MASK: {
      assert_true(count == 2);
      uint64_t val_hi = reader->ReadAndSwap<uint32_t>();
      uint64_t val_lo = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (val_hi << 32) | val_lo;
      result = true;
    } break;
    case PM4_SET_BIN_SELECT: {
      assert_true(count == 2);
      uint64_t val_hi = reader->ReadAndSwap<uint32_t>();
      uint64_t val_lo = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (val_hi << 32) | val_lo;
      result = true;
    } break;
    case PM4_CONTEXT_UPDATE: {
      assert_true(count == 1);
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      REXGPU_INFO("GPU context update = {:08X}", value);
      assert_true(value == 0);
      result = true;
      break;
    }
    case PM4_WAIT_FOR_IDLE: {
      // This opcode is used by 5454084E while going / being ingame.
      assert_true(count == 1);
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      REXGPU_INFO("GPU wait for idle = {:08X}", value);
      result = true;
      break;
    }

    default:
      REXGPU_INFO("Unimplemented GPU OPCODE: 0x{:02X}\t\tCOUNT: {}\n", opcode, count);
      assert_always();
      reader->AdvanceRead(count * sizeof(uint32_t));
      break;
  }

  trace_writer_.WritePacketEnd();
  if (opcode == PM4_XE_SWAP) {
    // End the trace writer frame.
    if (trace_writer_.is_open()) {
      trace_writer_.WriteEvent(EventCommand::Type::kSwap);
      trace_writer_.Flush();
      if (trace_state_ == TraceState::kSingleFrame) {
        trace_state_ = TraceState::kDisabled;
        trace_writer_.Close();
      }
    } else if (trace_state_ == TraceState::kSingleFrame) {
      // New trace request - we only start tracing at the beginning of a frame.
      uint32_t title_id = kernel_state_->GetExecutableModule()->title_id();
      auto file_name = fmt::format("{:08X}_{}.xtr", title_id, counter_ - 1);
      auto path = trace_frame_path_ / file_name;
      trace_writer_.Open(path, title_id);
      InitializeTrace();
    }
  }

  assert_true(reader->read_offset() ==
              (data_start_offset + (count * sizeof(uint32_t))) % reader->capacity());
  return result;
}

bool CommandProcessor::ExecutePacketType3_ME_INIT(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // initialize CP's micro-engine
  me_bin_.clear();
  for (uint32_t i = 0; i < count; i++) {
    me_bin_.push_back(reader->ReadAndSwap<uint32_t>());
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3_NOP(memory::RingBuffer* reader, uint32_t packet,
                                              uint32_t count) {
  // skip N 32-bit words to get to the next packet
  // No-op, ignore some data.
  reader->AdvanceRead(count * sizeof(uint32_t));
  return true;
}

bool CommandProcessor::ExecutePacketType3_INTERRUPT(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // generate interrupt from the command stream
  uint32_t cpu_mask = reader->ReadAndSwap<uint32_t>();
  for (int n = 0; n < 6; n++) {
    if (cpu_mask & (1 << n)) {
      if (graphics_system_) {
        graphics_system_->DispatchInterruptCallback(1, n);
      }
    }
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_XE_SWAP(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

#ifdef REXGLUE_ENABLE_PERF_COUNTERS
  {
    static uint64_t last_frame_tick = 0;
    uint64_t now = rex::chrono::Clock::QueryHostTickCount();
    if (last_frame_tick) {
      uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
      int64_t dt_us = static_cast<int64_t>((now - last_frame_tick) * 1000000 / freq);
      PROFILE_FRAME_TIME_US(dt_us);
      PROFILE_FPS(freq / (now - last_frame_tick));
    }
    last_frame_tick = now;
  }
#endif
  rex::perf::Profiler::Flip();

  // Xenia-specific VdSwap hook.
  // VdSwap will post this to tell us we need to swap the screen/fire an
  // interrupt.
  // 63 words here, but only the first has any data.
  uint32_t magic = reader->ReadAndSwap<memory::fourcc_t>();
  assert_true(magic == kSwapSignature);

  // TODO(benvanik): only swap frontbuffer ptr.
  uint32_t frontbuffer_ptr = reader->ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_width = reader->ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_height = reader->ReadAndSwap<uint32_t>();
  reader->AdvanceRead((count - 4) * sizeof(uint32_t));

  {
    // [N7] the swap itself: presenter, post-effect, frame-end RT work.
    const auto n7_sw_t0 = std::chrono::steady_clock::now();
    IssueSwap(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);
    g_n7_swap_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - n7_sw_t0).count();
    ++g_n7_swap_calls;
  }

  ++counter_;
  // Pure presented-frame tally (real swaps only, no vblank pollution).
  swap_counter_.fetch_add(1, std::memory_order_relaxed);

  // [NR-5B-DBG] register-file divergence probe: class hashes + the key
  // visibility registers, snapshotted AT THE SWAP (frame-aligned, so two
  // title runs -- suppressed vs not -- diff meaningfully). Printed 1 Hz in
  // the [nr-5a] block. Cheap (one XOR pass over ~5k dwords per frame) and
  // only when the 4a compose is on.
  if (g_nr4a_on) {
    const uint32_t* rv = register_file_->values;
    auto hash_range = [rv](uint32_t lo, uint32_t hi) {
      uint32_t h = 0;
      for (uint32_t r = lo; r < hi; ++r) h ^= rv[r] * (r | 1u);
      return h;
    };
    g_nr5dbg_hash[0] = hash_range(0x2000u, 0x2400u);   // render state
    g_nr5dbg_hash[1] = hash_range(0x4000u, 0x4400u);   // ALU VS
    g_nr5dbg_hash[2] = hash_range(0x4400u, 0x4800u);   // ALU PS
    g_nr5dbg_hash[3] = hash_range(0x4800u, 0x4930u);   // fetch + bool/loop
    static constexpr uint32_t kKeys[8] = {
        0x2000u,  // RB_SURFACE_INFO
        0x2001u,  // RB_COLOR_INFO
        0x2002u,  // RB_DEPTH_INFO
        0x2080u,  // PA_SC_WINDOW_OFFSET
        0x200Eu,  // PA_SC_SCREEN_SCISSOR_TL
        0x2200u,  // RB_DEPTHCONTROL
        0x2208u,  // RB_MODECONTROL
        0x210Fu,  // PA_CL_VTE_CNTL family
    };
    for (uint32_t k = 0; k < 8; ++k) g_nr5dbg_key[k] = rv[kKeys[k]];
  }

  // [NR-DETILE] N-5: the frame boundary. Promote this frame's observation of
  // the tile bands into the next frame's arming, and report once a second.
  {
    // Mode 2 cycles the lever 10 s on / 10 s off IN PLACE, so one drive reads
    // both halves at matched draws-per-frame. A cross-run fps comparison at
    // this scale is noise up to 3.4 fps; an in-place cycle is not.
    bool dt_on = g_nr_detile_mode == 1;
    if (g_nr_detile_mode == 2) {
      static const auto dt_t0 = std::chrono::steady_clock::now();
      const auto dt_el = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - dt_t0)
                             .count();
      dt_on = ((dt_el / 10) & 1) != 0;
    }
    static bool dt_was_on = false;
    if (dt_on != dt_was_on) {
      dt_was_on = dt_on;
      REXGPU_INFO("[nr-detile] PHASE {} - the EDRAM tile bands are {}.",
                  dt_on ? "ON" : "OFF",
                  dt_on ? "collapsed into one full-frame pass"
                        : "all executed (control)");
    }
    nr::DetileSetEnabled(dt_on);
    // [NR-DETILE] tap bisect. -2 cycles 8 s per phase through "only tap 0",
    // "only tap 1", "only tap 2", "all taps", so ONE drive says which band-1
    // resolve extension introduces the duplicated overlay. Phases 0-2 have
    // wrong pixels by design.
    {
      int32_t dt_tap = REXCVAR_GET(gpu_nr_detile_tap);
      if (dt_tap == -2) {
        static const auto tap_t0 = std::chrono::steady_clock::now();
        const auto tap_el = std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now() - tap_t0)
                                .count();
        const int32_t phase = int32_t((tap_el / 8) % 4);
        dt_tap = phase < 3 ? phase : -1;
      }
      static int32_t dt_tap_prev = -3;
      if (dt_tap != dt_tap_prev) {
        dt_tap_prev = dt_tap;
        if (dt_tap < 0) {
          REXGPU_INFO(
              "[nr-detile] TAP PHASE ALL - every band-1 resolve tap extended "
              "to the full frame (the normal mode).");
        } else {
          REXGPU_INFO(
              "[nr-detile] TAP PHASE {} - ONLY band-1 tap {} is extended; the "
              "other taps resolve their band only. Wrong pixels by design.",
              dt_tap, dt_tap);
        }
      }
      nr::DetileSetTapFilter(dt_tap);
    }
    // [NR-DETILE] widen-segment bisect. -2 cycles 8 s between "widen every
    // band-1 segment" and "widen only the world + refraction segments", so
    // ONE drive says whether the duplicated overlay is drawn by the widened
    // overlay segment. With the overlay left at band height, HUD elements
    // below the band line are missing by design.
    {
      int32_t dt_wseg = REXCVAR_GET(gpu_nr_detile_widen_seg);
      if (dt_wseg == -2) {
        static const auto wseg_t0 = std::chrono::steady_clock::now();
        const auto wseg_el = std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::steady_clock::now() - wseg_t0)
                                 .count();
        dt_wseg = ((wseg_el / 8) & 1) ? 2 : -1;
      }
      static int32_t dt_wseg_prev = -3;
      if (dt_wseg != dt_wseg_prev) {
        dt_wseg_prev = dt_wseg;
        if (dt_wseg < 0) {
          REXGPU_INFO(
              "[nr-detile] WIDEN PHASE ALL - every band-1 draw segment widened "
              "to the full frame (the normal mode).");
        } else {
          REXGPU_INFO(
              "[nr-detile] WIDEN PHASE {} - only band-1 segments below {} are "
              "widened; the overlay segment stays at band height. HUD below "
              "the band line missing by design.",
              dt_wseg, dt_wseg);
        }
      }
      nr::DetileSetWidenSegLimit(dt_wseg);
    }
    // [NR-DETILE] tail-rewrite bisect latch. -2 cycles 8 s execute / skip.
    {
      int32_t dt_tail = REXCVAR_GET(gpu_nr_detile_tail);
      if (dt_tail == -2) {
        static const auto tail_t0 = std::chrono::steady_clock::now();
        const auto tail_el = std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::steady_clock::now() - tail_t0)
                                 .count();
        dt_tail = ((tail_el / 8) & 1) ? 0 : -1;
      }
      static int32_t dt_tail_prev = -3;
      if (dt_tail != dt_tail_prev) {
        dt_tail_prev = dt_tail;
        REXGPU_INFO(
            "[nr-detile] TAIL PHASE {} - tail rewrites into the tap "
            "destinations are {}.",
            dt_tail == 0 ? "SKIP" : "EXEC",
            dt_tail == 0 ? "dropped" : "executed (normal)");
      }
      nr::DetileSetTailMode(dt_tail);
    }
    // (SS) [NR-97] the frame is the sweep boundary -- see the instrument's
    // comment. Swept BEFORE DetileFrameEnd so the reading belongs to the frame
    // that just ran, and only in the census (non-dropping) phase, where the
    // applies still happen and the live file is the thing being questioned.
    // [NR-97] nothing may be owed across a frame boundary.
    if (g_n97_blk_n) N97FlushPending(this);
    if (g_n97_sample) N97NetSweep();
    ++g_n97_frame_n;
    // 1 frame in 16. The census phase is the BASELINE half of the paired fps
    // read, so every cost it pays that the ON half does not overstates the
    // win: the touch bitmap runs on each dropped dword, and this keeps it to
    // ~6% of frames while still sampling several a second.
    g_n97_sample = g_n97_mode == 0 && (g_n97_frame_n & 15) == 0;
    const nr::DetileFrameVerdict dtv = nr::DetileFrameEnd();
    if (dtv.guard_tripped) {
      REXGPU_ERROR(
          "[nr-detile] GUARD TRIPPED: the repeat bands' resolve destinations "
          "are not evenly spaced strips of one texture. De-tiling disarmed.");
    } else if (dtv.is_armed != dtv.was_armed) {
      REXGPU_INFO("[nr-detile] {}: {} rows per band, {} rows per frame.",
                  dtv.is_armed ? "ARMED" : "DISARMED", dtv.band_rows, dtv.full_height);
    }
    static auto dt_last = std::chrono::steady_clock::now();
    const auto dt_now = std::chrono::steady_clock::now();
    if (dt_now - dt_last >= std::chrono::seconds(1)) {
      dt_last = dt_now;
      const nr::DetileStats& s = nr::DetileGetStats();
      static nr::DetileStats prev{};
      if (g_nr_detile_mode) {
        // ⚠ `extended` must equal `taps`. Every band-1 resolve tap of the
        // tiled pass has to reach the full frame height; the ones that do not
        // leave their destination's lower rows unwritten, which is exactly the
        // wrong frame city drive 1 produced (extended=1 against taps=3).
        // ON and OFF frames are both counted so a matched A/B has two halves.
        REXGPU_INFO(
            "[nr-detile] fps on={} off={} band={}x{}rows | skipped draws={} "
            "resolves={} | scwiden={} wrefuse={} extended={} of taps={} | "
            "dest_stride={:#x} guard_fail={} own={}rows",
            s.frames - prev.frames, s.frames_off - prev.frames_off,
            (s.band_rows ? (s.full_height + s.band_rows - 1) / s.band_rows : 0),
            s.band_rows, s.draws_skipped - prev.draws_skipped,
            s.resolves_skipped - prev.resolves_skipped,
            s.scissor_widened - prev.scissor_widened,
            s.widen_refused - prev.widen_refused,
            s.resolves_extended - prev.resolves_extended, s.taps - prev.taps,
            s.dest_stride, s.guard_fail, s.height_used);
        char tap_buf[512];
        if (nr::DetileFormatTaps(tap_buf, sizeof(tap_buf))) {
          REXGPU_INFO("[nr-detile] band-1 taps: {}", tap_buf);
        }
        char tail_buf[160];
        if (nr::DetileFormatTail(tail_buf, sizeof(tail_buf))) {
          REXGPU_INFO("[nr-detile] tail: {}", tail_buf);
        }
        // The tail bisect's first two city runs logged NO tail line at all,
        // including at real heavy-city load. `seen=0` cannot distinguish "the
        // live frame has no tail rewrite" from "the predicate is wrong", so
        // the census names every resolve the observe site refused and lists
        // the destinations that reached the last gate.
        char rescen_buf[4096];
        if (nr::DetileFormatResolveCensus(rescen_buf, sizeof(rescen_buf))) {
          REXGPU_INFO("[nr-detile] resolves: {}", rescen_buf);
        }
        char rt_buf[768];
        if (nr::DetileFormatRenderTargets(rt_buf, sizeof(rt_buf))) {
          REXGPU_INFO("[nr-detile] edram: {}", rt_buf);
        }
      }
      prev = s;
    }
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_INDIRECT_BUFFER(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // indirect buffer dispatch
  uint32_t list_ptr = CpuToGpu(reader->ReadAndSwap<uint32_t>());
  uint32_t list_length = reader->ReadAndSwap<uint32_t>();
  assert_zero(list_length & ~0xFFFFF);
  list_length &= 0xFFFFF;
  ExecuteIndirectBuffer(GpuToCpu(list_ptr), list_length);
  return true;
}

bool CommandProcessor::ExecutePacketType3_WAIT_REG_MEM(memory::RingBuffer* reader, uint32_t packet,
                                                       uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // wait until a register or memory location is a specific value

  uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t ref = reader->ReadAndSwap<uint32_t>();
  uint32_t mask = reader->ReadAndSwap<uint32_t>();
  uint32_t wait = reader->ReadAndSwap<uint32_t>();

  bool is_memory = (wait_info & 0x10) != 0;

  bool matched = false;
  do {
    uint32_t value = 0;
    if (is_memory) {
      value =
          *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(poll_reg_addr & ~uint32_t(0x3)));
      trace_writer_.WriteMemoryRead(CpuToGpu(poll_reg_addr & ~uint32_t(0x3)), sizeof(uint32_t));
      value = xenos::GpuSwap(value, static_cast<xenos::Endian>(poll_reg_addr & 0x3));
    } else {
      value = ReadRegisterValue(poll_reg_addr);
      if (poll_reg_addr == XE_GPU_REG_COHER_STATUS_HOST) {
        MakeCoherent();
        value = ReadRegisterValue(poll_reg_addr);
      }
    }
    switch (wait_info & 0x7) {
      case 0x0:  // Never.
        matched = false;
        break;
      case 0x1:  // Less than reference.
        matched = (value & mask) < ref;
        break;
      case 0x2:  // Less than or equal to reference.
        matched = (value & mask) <= ref;
        break;
      case 0x3:  // Equal to reference.
        matched = (value & mask) == ref;
        break;
      case 0x4:  // Not equal to reference.
        matched = (value & mask) != ref;
        break;
      case 0x5:  // Greater than or equal to reference.
        matched = (value & mask) >= ref;
        break;
      case 0x6:  // Greater than reference.
        matched = (value & mask) > ref;
        break;
      case 0x7:  // Always
        matched = true;
        break;
    }
    if (!matched) {
      // Wait.
      if (wait >= 0x100) {
        PrepareForWait();
        if (!REXCVAR_GET(vsync)) {
          // User wants it fast and dangerous.
          rex::thread::MaybeYield();
        } else {
          rex::thread::Sleep(std::chrono::milliseconds(wait / 0x100));
        }
        rex::thread::SyncMemory();
        ReturnFromWait();

        if (!worker_running_) {
          // Short-circuited exit.
          return false;
        }
      } else {
        rex::thread::MaybeYield();
      }
    }
  } while (!matched);

  return true;
}

bool CommandProcessor::ExecutePacketType3_REG_RMW(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // register read/modify/write
  // ? (used during shader upload and edram setup)
  uint32_t rmw_info = reader->ReadAndSwap<uint32_t>();
  uint32_t and_mask = reader->ReadAndSwap<uint32_t>();
  uint32_t or_mask = reader->ReadAndSwap<uint32_t>();
  uint32_t value = register_file_->values[rmw_info & 0x1FFF];
  if ((rmw_info >> 31) & 0x1) {
    // & reg
    value &= register_file_->values[and_mask & 0x1FFF];
  } else {
    // & imm
    value &= and_mask;
  }
  if ((rmw_info >> 30) & 0x1) {
    // | reg
    value |= register_file_->values[or_mask & 0x1FFF];
  } else {
    // | imm
    value |= or_mask;
  }
  // [NR-RING] Tag this write so the context tap can attribute it: the
  // walker does not decode REG_RMW, so at depth 1 this is its blind spot.
  if (g_nr_ctx) g_nr_in_rmw = true;
  WriteRegister(rmw_info & 0x1FFF, value);
  g_nr_in_rmw = false;
  return true;
}

bool CommandProcessor::ExecutePacketType3_REG_TO_MEM(memory::RingBuffer* reader, uint32_t packet,
                                                     uint32_t count) {
  // Copy Register to Memory (?)
  // Count is 2, assuming a Register Addr and a Memory Addr.

  uint32_t reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t mem_addr = reader->ReadAndSwap<uint32_t>();

  uint32_t reg_val = ReadRegisterValue(reg_addr);

  auto endianness = static_cast<xenos::Endian>(mem_addr & 0x3);
  mem_addr &= ~0x3;
  reg_val = GpuSwap(reg_val, endianness);
  memory::store(memory_->TranslatePhysical(mem_addr), reg_val);
  trace_writer_.WriteMemoryWrite(CpuToGpu(mem_addr), 4);

  return true;
}

bool CommandProcessor::ExecutePacketType3_MEM_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  uint32_t write_addr = reader->ReadAndSwap<uint32_t>();
  for (uint32_t i = 0; i < count - 1; i++) {
    uint32_t write_data = reader->ReadAndSwap<uint32_t>();

    auto endianness = static_cast<xenos::Endian>(write_addr & 0x3);
    auto addr = write_addr & ~0x3;
    write_data = GpuSwap(write_data, endianness);
    memory::store(memory_->TranslatePhysical(addr), write_data);
    trace_writer_.WriteMemoryWrite(CpuToGpu(addr), 4);
    write_addr += 4;
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3_COND_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                     uint32_t count) {
  // conditional write to memory or register
  uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t ref = reader->ReadAndSwap<uint32_t>();
  uint32_t mask = reader->ReadAndSwap<uint32_t>();
  uint32_t write_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t write_data = reader->ReadAndSwap<uint32_t>();
  uint32_t value;
  if (wait_info & 0x10) {
    // Memory.
    auto endianness = static_cast<xenos::Endian>(poll_reg_addr & 0x3);
    poll_reg_addr &= ~0x3;
    trace_writer_.WriteMemoryRead(CpuToGpu(poll_reg_addr), 4);
    value = memory::load<uint32_t>(memory_->TranslatePhysical(poll_reg_addr));
    value = GpuSwap(value, endianness);
  } else {
    // Register.
    value = ReadRegisterValue(poll_reg_addr);
  }
  bool matched = false;
  switch (wait_info & 0x7) {
    case 0x0:  // Never.
      matched = false;
      break;
    case 0x1:  // Less than reference.
      matched = (value & mask) < ref;
      break;
    case 0x2:  // Less than or equal to reference.
      matched = (value & mask) <= ref;
      break;
    case 0x3:  // Equal to reference.
      matched = (value & mask) == ref;
      break;
    case 0x4:  // Not equal to reference.
      matched = (value & mask) != ref;
      break;
    case 0x5:  // Greater than or equal to reference.
      matched = (value & mask) >= ref;
      break;
    case 0x6:  // Greater than reference.
      matched = (value & mask) > ref;
      break;
    case 0x7:  // Always
      matched = true;
      break;
  }
  if (matched) {
    // Write.
    if (wait_info & 0x100) {
      // Memory.
      auto endianness = static_cast<xenos::Endian>(write_reg_addr & 0x3);
      write_reg_addr &= ~0x3;
      write_data = GpuSwap(write_data, endianness);
      memory::store(memory_->TranslatePhysical(write_reg_addr), write_data);
      trace_writer_.WriteMemoryWrite(CpuToGpu(write_reg_addr), 4);
    } else {
      // Register.
      // [NR-RING] Tag: COND_WRITE register writes are the walker's other
      // blind-spot op (see REG_RMW).
      if (g_nr_ctx) g_nr_in_cond = true;
      WriteRegister(write_reg_addr, write_data);
      g_nr_in_cond = false;
    }
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // generate an event that creates a write to memory when completed
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  if (count == 1) {
    // Just an event flag? Where does this write?
  } else {
    // Write to an address.
    assert_always();
    reader->AdvanceRead((count - 1) * sizeof(uint32_t));
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_SHD(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // generate a VS|PS_done event
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  uint32_t value = reader->ReadAndSwap<uint32_t>();

  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  uint32_t data_value;
  if ((initiator >> 31) & 0x1) {
    // Write counter (GPU vblank counter?).
    data_value = counter_;
  } else {
    // Write value.
    data_value = value;
  }
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;
  data_value = GpuSwap(data_value, endianness);
  memory::store(memory_->TranslatePhysical(address), data_value);
  trace_writer_.WriteMemoryWrite(CpuToGpu(address), 4);
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_EXT(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // generate a screen extent event
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;

  // Let us hope we can fake this.
  // This callback tells the driver the xy coordinates affected by a previous
  // drawcall.
  // https://www.google.com/patents/US20060055701
  uint16_t extents[] = {
      0 >> 3,                                    // min x
      xenos::kTexture2DCubeMaxWidthHeight >> 3,  // max x
      0 >> 3,                                    // min y
      xenos::kTexture2DCubeMaxWidthHeight >> 3,  // max y
      0,                                         // min z
      1,                                         // max z
  };
  assert_true(endianness == xenos::Endian::k8in16);
  memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(address), extents,
                                     rex::countof(extents));
  trace_writer_.WriteMemoryWrite(CpuToGpu(address), sizeof(extents));
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // Set by D3D as BE but struct ABI is LE
  const uint32_t kQueryFinished = rex::byte_swap(0xFFFFFEED);
  assert_true(count == 1);
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);

  // Occlusion queries:
  // This command is send on query begin and end.
  // As a workaround report some fixed amount of passed samples.
  auto fake_sample_count = REXCVAR_GET(query_occlusion_fake_sample_count);
  if (fake_sample_count >= 0) {
    auto* pSampleCounts = memory_->TranslatePhysical<xe_gpu_depth_sample_counts*>(
        register_file_->values[XE_GPU_REG_RB_SAMPLE_COUNT_ADDR]);
    if (!pSampleCounts) {
      return true;
    }
    // 0xFFFFFEED is written to this two locations by D3D only on D3DISSUE_END
    // and used to detect a finished query.
    bool is_end_via_z_pass =
        pSampleCounts->ZPass_A == kQueryFinished && pSampleCounts->ZPass_B == kQueryFinished;
    // Older versions of D3D also checks for ZFail (4D5307D5).
    bool is_end_via_z_fail =
        pSampleCounts->ZFail_A == kQueryFinished && pSampleCounts->ZFail_B == kQueryFinished;
    std::memset(pSampleCounts, 0, sizeof(xe_gpu_depth_sample_counts));
    if (is_end_via_z_pass || is_end_via_z_fail) {
      pSampleCounts->ZPass_A = fake_sample_count;
      pSampleCounts->Total_A = fake_sample_count;
    }
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3Draw(memory::RingBuffer* reader, uint32_t packet,
                                              const char* opcode_name, uint32_t draw_opcode,
                                              uint32_t viz_query_condition,
                                              uint32_t count_remaining) {
  // if viz_query_condition != 0, this is a conditional draw based on viz query.
  // This ID matches the one issued in PM4_VIZ_QUERY
  // uint32_t viz_id = viz_query_condition & 0x3F;
  // when true, render conditionally based on query result
  // uint32_t viz_use = viz_query_condition & 0x100;

  assert_not_zero(count_remaining);
  if (!count_remaining) {
    REXGPU_ERROR("{}: Packet too small, can't read VGT_DRAW_INITIATOR", opcode_name);
    return false;
  }
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = reader->ReadAndSwap<uint32_t>();
  --count_remaining;
  WriteRegister(XE_GPU_REG_VGT_DRAW_INITIATOR, vgt_draw_initiator.value);

  bool draw_succeeded = true;
  // TODO(Triang3l): Remove IndexBufferInfo and replace handling of all this
  // with PrimitiveProcessor when the old Vulkan renderer is removed.
  bool is_indexed = false;
  IndexBufferInfo index_buffer_info;
  switch (vgt_draw_initiator.source_select) {
    case xenos::SourceSelect::kDMA: {
      // Indexed draw.
      is_indexed = true;

      // Two separate bounds checks so if there's only one missing register
      // value out of two, one uint32_t will be skipped in the command buffer,
      // not two.
      assert_not_zero(count_remaining);
      if (!count_remaining) {
        REXGPU_ERROR("{}: Packet too small, can't read VGT_DMA_BASE", opcode_name);
        return false;
      }
      uint32_t vgt_dma_base = reader->ReadAndSwap<uint32_t>();
      --count_remaining;
      WriteRegister(XE_GPU_REG_VGT_DMA_BASE, vgt_dma_base);
      reg::VGT_DMA_SIZE vgt_dma_size;
      assert_not_zero(count_remaining);
      if (!count_remaining) {
        REXGPU_ERROR("{}: Packet too small, can't read VGT_DMA_SIZE", opcode_name);
        return false;
      }
      vgt_dma_size.value = reader->ReadAndSwap<uint32_t>();
      --count_remaining;
      WriteRegister(XE_GPU_REG_VGT_DMA_SIZE, vgt_dma_size.value);

      uint32_t index_size_bytes = vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16
                                      ? sizeof(uint16_t)
                                      : sizeof(uint32_t);
      // The base address must already be word-aligned according to the R6xx
      // documentation, but for safety.
      index_buffer_info.guest_base = vgt_dma_base & ~(index_size_bytes - 1);
      index_buffer_info.endianness = vgt_dma_size.swap_mode;
      index_buffer_info.format = vgt_draw_initiator.index_size;
      index_buffer_info.length = vgt_dma_size.num_words * index_size_bytes;
      index_buffer_info.count = vgt_draw_initiator.num_indices;
    } break;
    case xenos::SourceSelect::kImmediate: {
      // TODO(Triang3l): VGT_IMMED_DATA.
      REXGPU_ERROR(
          "{}: Using immediate vertex indices, which are not supported yet. "
          "Report the game to Xenia developers!",
          opcode_name, uint32_t(vgt_draw_initiator.source_select));
      draw_succeeded = false;
      assert_always();
    } break;
    case xenos::SourceSelect::kAutoIndex: {
      // Auto draw.
      index_buffer_info.guest_base = 0;
      index_buffer_info.length = 0;
    } break;
    default: {
      // Invalid source selection.
      draw_succeeded = false;
      assert_unhandled_case(vgt_draw_initiator.source_select);
    } break;
  }

  // Skip to the next command, for example, if there are immediate indexes that
  // we don't support yet.
  reader->AdvanceRead(count_remaining * sizeof(uint32_t));

  if (draw_succeeded) {
    // [NR-SKP] 5-4-4a: the body lives in ExecutePacketType3DrawTail so the
    // skip's direct draw path (NrSkipDrawDirect) runs the exact same code.
    ExecutePacketType3DrawTail(vgt_draw_initiator.value,
                               is_indexed ? &index_buffer_info : nullptr,
                               opcode_name, draw_opcode);
  }

  // If read the packed correctly, but merely couldn't execute it (because of,
  // for instance, features not supported by the host), don't terminate command
  // buffer processing as that would leave rendering in a way more inconsistent
  // state than just a single dropped draw command.
  return true;
}

// [NR-SKP] Phase 5-4-4a: the post-parse half of ExecutePacketType3Draw,
// extracted VERBATIM (only the wrapper changed: the initiator arrives as a
// value, index_buffer_info as a pointer that is already null for the
// non-indexed case; the inner brace block preserves the extracted body
// byte-for-byte). Both the packet handler and the skip's direct draw path
// run this body, so the two paths cannot drift.
void CommandProcessor::ExecutePacketType3DrawTail(
    uint32_t vgt_draw_initiator_value, IndexBufferInfo* index_buffer_info,
    const char* opcode_name, uint32_t draw_opcode) {
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = vgt_draw_initiator_value;
  {
    auto viz_query = register_file_->Get<reg::PA_SC_VIZ_QUERY>();
    if (!(viz_query.viz_query_ena && viz_query.kill_pix_post_hi_z)) {
      // TODO(Triang3l): Don't drop the draw call completely if the vertex
      // shader has memexport.
      // TODO(Triang3l || JoelLinn): Handle this properly in the render
      // backends.

      bool major_mode_explicit =
          xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type);
      // [GPU-EXEC-PROFILE]/[GPU-SPLIT] time the host backend draw recording
      // specifically (clean: ~330k brackets/s). Fires for either profile.
      // [NR-DRAW] Increment 4c: advance the walk to THIS draw and compare the
      // shadow against the live register file, both read at the same moment.
      // Before IssueDraw rather than after, because the executor has by now
      // applied every register write preceding the draw and nothing else has
      // run: the two sides are looking at the same state or the walk is wrong.
      // Depth-1 only, for the same reason the walk itself is: a nested buffer
      // is not walked, so its draws must not advance the outer walk.
      // [NR-SKP] 5-4-2: under the skip this handler runs as a DELEGATED
      // packet -- the walk already stopped at exactly this draw and sits past
      // it, so the pre-made stop is consumed instead of advancing the walk.
      const bool nr_skip_draw = g_nr_skip_draw_pending;
      if ((g_ctx_walk_lockstep || nr_skip_draw) && g_pm4_ib_depth == 1) {
        nr::CtxDrawStop stop;
        bool have_stop;
        if (nr_skip_draw) {
          stop = g_nr_skip_stop;
          g_nr_skip_draw_pending = false;
          have_stop = true;
        } else {
          have_stop = nr::CtxWalkNextDraw(&g_ctx_walker, &stop);
        }
        if (!have_stop || stop.opcode != draw_opcode) {
          // The walk and the executor disagree about which packets run. Every
          // value read from here on would be from the wrong moment, so stop
          // COMPARING -- but keep the walk itself alive, so the buffer's state
          // still reaches the running context and the next buffer starts from
          // the truth rather than from a half-applied one.
          ++g_nrd_desync;
          g_ctx_walk_lockstep = false;
        } else {
          ++g_nrd_stops;
          // [NR-VERIFY] inc 2: the rolling shadow-vs-live compare is pure
          // verify (256 reg compares per draw).
          if (g_nr_verify_base) {
            nr::RegShadowSweep(&g_reg_shadow, &g_reg_stats, kNrdSweepPerDraw,
                               register_file_->values, NrdFinding, nullptr);
          }
          // [NR-ISSUE] Increment 4d: arm this draw to be issued from the
          // shadow. Increment 4f: copy-mode draws (resolves) arm too --
          // IssueCopy's state reads now honor the active draw file (the one
          // stray direct read fixed; RenderTargetCache::Resolve always read
          // the repointed member) -- counted apart so the report still splits
          // geometry from resolves. Draws whose walk shader refs are not yet
          // valid (boot moments before carry is established) stay live: a
          // partially recovered issue would blur what an A/B mismatch means.
          if (g_nr_issue) {
            const uint64_t ord = g_nri_ordinal++;
            if (ord >= uint64_t(g_nri_from) &&
                (g_nri_count < 0 ||
                 ord < uint64_t(g_nri_from) + uint64_t(g_nri_count))) {
              if (register_file_->Get<reg::RB_MODECONTROL>().edram_mode ==
                  xenos::EdramMode::kCopy) {
                ++g_nri_copy_armed;
              }
              if (!g_ctx_state.vs.valid || !g_ctx_state.ps.valid) {
                ++g_nri_sh_invalid;
              } else {
                // Resolve the walk's own shader refs exactly as IM_LOAD does.
                // An immediate shader's ref addresses the ucode inside the
                // buffer itself, which TranslatePhysical reaches the same way.
                // Same bytes => same hash => the same Shader* the live path
                // holds, unless the walk's shader tracking is wrong -- counted
                // apart, and the walk's shaders are used either way: that IS
                // the recovered path, and it exercises the 4b-2 result.
                // [NR-VERIFY] inc 2: under the perf config the resolved
                // Shader* is memoized -- shader refs only change at IM_LOAD
                // packets (the executor also resolves at IM_LOAD, not per
                // draw), and every walked IM_LOAD invalidates the memo via
                // CtxShaderSeen, including a re-IM_LOAD of patched ucode at
                // the same address. Re-hashing the whole ucode twice per draw
                // is verify cost.
                Shader* vs;
                Shader* ps;
                const bool nri_sh_memo_ok =
                    !g_nr_verify_base && !g_nri_sh_dirty && g_nri_vs_cached &&
                    g_nri_ps_cached &&
                    g_nri_vs_ref.addr == g_ctx_state.vs.addr &&
                    g_nri_vs_ref.size_dwords == g_ctx_state.vs.size_dwords &&
                    g_nri_ps_ref.addr == g_ctx_state.ps.addr &&
                    g_nri_ps_ref.size_dwords == g_ctx_state.ps.size_dwords;
                if (nri_sh_memo_ok) {
                  vs = g_nri_vs_cached;
                  ps = g_nri_ps_cached;
                } else {
                  vs = LoadShader(
                      xenos::ShaderType::kVertex, g_ctx_state.vs.addr,
                      memory_->TranslatePhysical<uint32_t*>(g_ctx_state.vs.addr),
                      g_ctx_state.vs.size_dwords);
                  ps = LoadShader(
                      xenos::ShaderType::kPixel, g_ctx_state.ps.addr,
                      memory_->TranslatePhysical<uint32_t*>(g_ctx_state.ps.addr),
                      g_ctx_state.ps.size_dwords);
                  g_nri_vs_cached = vs;
                  g_nri_ps_cached = ps;
                  g_nri_vs_ref = g_ctx_state.vs;
                  g_nri_ps_ref = g_ctx_state.ps;
                  g_nri_sh_dirty = false;
                }
                // [NR-SKP] Under the skip the executor never sees IM_LOAD, so
                // the active members hold the previous draw's shaders: the
                // compare would count real shader changes as mismatches.
                // Instead the walk IS the truth -- converge the members to
                // what the last walked IM_LOAD would have set, so any
                // fallback path that follows reads coherent state.
                if (nr_skip_draw) {
                  active_vertex_shader_ = vs;
                  active_pixel_shader_ = ps;
                } else if (vs != active_vertex_shader_ ||
                           ps != active_pixel_shader_) {
                  ++g_nri_sh_mismatch;
                }
                // [NR-ISSUE] Increment 4e: seed the replay file ONCE (the
                // shadow already holds every stream write up to this stop, so
                // the compose loses nothing to the pre-seed window), then
                // rely on the incremental applies. Per arm, only the six
                // non-stream registers are re-read from live.
                if (g_nri_live) {
                  // [NR-ISSUE-LIVE] the live file IS the replay state in
                  // lockstep; no seed, no mirror, no non-stream re-read.
                  nr_issue_file_ = register_file_;
                } else {
                if (!g_nri_seeded) {
                  nr::RegShadowCompose(&g_reg_shadow, register_file_->values,
                                       g_nri_file.values);
                  g_nri_seeded = true;
                }
                for (uint32_t r : kNriNonStreamRegs) {
                  g_nri_file.values[r] = register_file_->values[r];
                }
                nr_issue_file_ = &g_nri_file;
                }
                nr_issue_vertex_shader_ = vs;
                nr_issue_pixel_shader_ = ps;
                nr_issue_shaders_active_ = true;
                nr_issue_armed_ = true;
                ++g_nri_armed;
              }
            }
          }
        }
      } else if (g_nr_draw && g_pm4_ib_depth != 1) {
        ++g_nrd_skipped_depth;
      }
      ++g_n7_draws;  // [N7] executed draws, unconditional
      N8eDrawBoundary();  // [N8E] close this draw's constant-interval union
      const bool kTimeDraw = g_exec_prof || g_split_prof;
      const auto draw_t0 = kTimeDraw ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      const bool draw_succeeded =
          IssueDraw(vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
                    index_buffer_info, major_mode_explicit);
      // [NR-ISSUE] Disarm unconditionally: the handshake is one draw wide.
      if (nr_issue_armed_ || nr_issue_shaders_active_) {
        nr_issue_armed_ = false;
        nr_issue_shaders_active_ = false;
        nr_issue_file_ = nullptr;
        nr_issue_vertex_shader_ = nullptr;
        nr_issue_pixel_shader_ = nullptr;
      }
      if (kTimeDraw) {
        g_issuedraw_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - draw_t0).count();
        g_draw_cnt++;
      }
      // [NR-RES] Coverage of this draw's own shader references. Here rather
      // than in the walk: the shaders are analyzed only once IssueDraw has
      // run, and nothing has written state since the draw.
      if (g_nr_res) {
        const Shader* vs = active_vertex_shader();
        const Shader* ps = active_pixel_shader();
        if ((vs && vs->is_ucode_analyzed()) || (ps && ps->is_ucode_analyzed())) {
          bool covered = true;
          ++g_res_cov_draws;
          ResCoverShader(vs, false, &covered);
          ResCoverShader(ps, true, &covered);
          if (covered) ++g_res_cov_full;
        } else {
          ++g_res_cov_unanalyzed;
        }
      }
      if (!draw_succeeded) {
        auto vgt_output_path_cntl = register_file_->Get<reg::VGT_OUTPUT_PATH_CNTL>();
        auto vgt_hos_cntl = register_file_->Get<reg::VGT_HOS_CNTL>();
        auto rb_modecontrol = register_file_->Get<reg::RB_MODECONTROL>();
        REXGPU_ERROR(
            "{}({}, {}, {}): Failed in backend "
            "(major_mode={}, explicit_major={}, path_select={}, tess_mode={}, edram_mode={})",
            opcode_name, static_cast<uint32_t>(vgt_draw_initiator.num_indices),
            uint32_t(vgt_draw_initiator.prim_type), uint32_t(vgt_draw_initiator.source_select),
            uint32_t(vgt_draw_initiator.major_mode), uint32_t(major_mode_explicit),
            uint32_t(vgt_output_path_cntl.path_select), uint32_t(vgt_hos_cntl.tess_mode),
            uint32_t(rb_modecontrol.edram_mode));
      }
    }
  }
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  // "initiate fetch of index buffer and draw"
  // Generally used by Xbox 360 Direct3D 9 for kDMA and kAutoIndex sources.
  // With a viz query token as the first one.
  uint32_t count_remaining = count;
  assert_not_zero(count_remaining);
  if (!count_remaining) {
    REXGPU_ERROR("PM4_DRAW_INDX: Packet too small, can't read the viz query token");
    return false;
  }
  uint32_t viz_query_condition = reader->ReadAndSwap<uint32_t>();
  --count_remaining;
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX", 0x22, viz_query_condition,
                                count_remaining);
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX_2(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // "draw using supplied indices in packet"
  // Generally used by Xbox 360 Direct3D 9 for kAutoIndex source.
  // No viz query token.
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX_2", 0x36, 0, count);
}

bool CommandProcessor::ExecutePacketType3_SET_CONSTANT(memory::RingBuffer* reader, uint32_t packet,
                                                       uint32_t count) {
  // load constant into chip and to memory
  // PM4_REG(reg) ((0x4 << 16) | (GSL_HAL_SUBBLOCK_OFFSET(reg)))
  //                                     reg - 0x2000
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t count_registers = count - 1;
  switch (type) {
    case 0:  // ALU
      WriteALURangeFromRing(reader, index, count_registers);
      break;
    case 1:  // FETCH
      WriteFetchRangeFromRing(reader, index, count_registers);
      break;
    case 2:  // BOOL
      WriteBoolRangeFromRing(reader, index, count_registers);
      break;
    case 3:  // LOOP
      WriteLoopRangeFromRing(reader, index, count_registers);
      break;
    case 4:  // REGISTERS
      WriteREGISTERSRangeFromRing(reader, index, count_registers);
      break;
    default:
      assert_always();
      reader->AdvanceRead((count - 1) * sizeof(uint32_t));
      return true;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_SET_CONSTANT2(memory::RingBuffer* reader, uint32_t packet,
                                                        uint32_t count) {
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  WriteRegisterRangeFromRing(reader, index, count - 1);
  return true;
}

bool CommandProcessor::ExecutePacketType3_LOAD_ALU_CONSTANT(memory::RingBuffer* reader,
                                                            uint32_t packet, uint32_t count) {
  // load constants from memory
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  address &= 0x3FFFFFFF;
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t size_dwords = reader->ReadAndSwap<uint32_t>();
  size_dwords &= 0xFFF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t* xlat_address = memory_->TranslatePhysical<uint32_t*>(address);
  switch (type) {
    case 0:  // ALU
      trace_writer_.WriteMemoryRead(CpuToGpu(address), size_dwords * 4);
      WriteALURangeFromMem(index, xlat_address, size_dwords);
      break;
    case 1:  // FETCH
      trace_writer_.WriteMemoryRead(CpuToGpu(address), size_dwords * 4);
      WriteFetchRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 2:  // BOOL
      trace_writer_.WriteMemoryRead(CpuToGpu(address), size_dwords * 4);
      WriteBoolRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 3:  // LOOP
      trace_writer_.WriteMemoryRead(CpuToGpu(address), size_dwords * 4);
      WriteLoopRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 4:  // REGISTERS
      trace_writer_.WriteMemoryRead(CpuToGpu(address), size_dwords * 4);
      WriteREGISTERSRangeFromMem(index, xlat_address, size_dwords);
      break;
    default:
      assert_always();
      return true;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_SET_SHADER_CONSTANTS(memory::RingBuffer* reader,
                                                               uint32_t packet, uint32_t count) {
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  WriteRegisterRangeFromRing(reader, index, count - 1);
  return true;
}

bool CommandProcessor::ExecutePacketType3_IM_LOAD(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (pointer-based)
  uint32_t addr_type = reader->ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(addr_type & 0x3);
  uint32_t addr = addr_type & ~0x3;
  uint32_t start_size = reader->ReadAndSwap<uint32_t>();
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);

  trace_writer_.WriteMemoryRead(CpuToGpu(addr), size_dwords * 4);
  // [NR-SDB] Same pointer and length the pipeline cache is about to hash.
  NrSdbObserve(shader_type, addr, memory_->TranslatePhysical<uint32_t*>(addr),
               size_dwords);
  auto shader =
      LoadShader(shader_type, addr, memory_->TranslatePhysical<uint32_t*>(addr), size_dwords);
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_IM_LOAD_IMMEDIATE(memory::RingBuffer* reader,
                                                            uint32_t packet, uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (code embedded in packet)
  uint32_t dword0 = reader->ReadAndSwap<uint32_t>();
  uint32_t dword1 = reader->ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(dword0);
  uint32_t start_size = dword1;
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);
  assert_true(reader->read_count() >= size_dwords * 4);
  assert_true(count - 2 >= size_dwords);
  // [NR-SDB] Inline ucode: the same bytes, addressed in the packet stream.
  NrSdbObserve(shader_type, uint32_t(reader->read_ptr()),
               reinterpret_cast<uint32_t*>(reader->read_ptr()), size_dwords);
  auto shader = LoadShader(shader_type, uint32_t(reader->read_ptr()),
                           reinterpret_cast<uint32_t*>(reader->read_ptr()), size_dwords);
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  reader->AdvanceRead(size_dwords * sizeof(uint32_t));
  return true;
}

bool CommandProcessor::ExecutePacketType3_INVALIDATE_STATE(memory::RingBuffer* reader,
                                                           uint32_t packet, uint32_t count) {
  // selective invalidation of state pointers
  /*uint32_t mask =*/reader->ReadAndSwap<uint32_t>();
  // driver_->InvalidateState(mask);
  return true;
}

bool CommandProcessor::ExecutePacketType3_VIZ_QUERY(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  // begin/end initiator for viz query extent processing
  // https://www.google.com/patents/US20050195186
  assert_true(count == 1);

  uint32_t dword0 = reader->ReadAndSwap<uint32_t>();

  uint32_t id = dword0 & 0x3F;
  uint32_t end = dword0 & 0x100;
  if (!end) {
    // begin a new viz query @ id
    // On hardware this clears the internal state of the scan converter (which
    // is different to the register)
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, VIZQUERY_START);
    REXGPU_INFO("Begin viz query ID {:02X}", id);
  } else {
    // end the viz query
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, VIZQUERY_END);
    REXGPU_INFO("End viz query ID {:02X}", id);
    // The scan converter writes the internal result back to the register here.
    // We just fake it and say it was visible in case it is read back.
    if (id < 32) {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_0] |= uint32_t(1) << id;
    } else {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_1] |= uint32_t(1) << (id - 32);
    }
  }

  return true;
}

void CommandProcessor::InitializeTrace() {
  // Write the initial register values, to be loaded directly into the
  // RegisterFile since all registers, including those that may have side
  // effects on setting, will be saved.
  trace_writer_.WriteRegisters(0, register_file_->values, RegisterFile::kRegisterCount, false);

  trace_writer_.WriteGammaRamp(gamma_ramp_256_entry_table(), gamma_ramp_pwl_rgb(),
                               gamma_ramp_rw_component_);
}

}  // namespace rex::graphics
