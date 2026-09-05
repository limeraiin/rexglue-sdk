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
#include <cfloat>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/perf/counter.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/graphics/d3d12/graphics_system.h>
#include <rex/graphics/d3d12/hitch_probe.h>
#include <rex/graphics/d3d12/shader.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/nr_bindings.h>
#include <rex/graphics/nr_draw_cache.h>  // [GKEY] rung-2 census join

// [SKC] the rung-2 skip census tracker lives in command_processor.cpp (global scope).
extern bool g_skc_on;
extern uint8_t g_spp_stop_kind;  // [NR-SPP3] 1 = this stop was a pool hit
static bool g_skc_disarmed = false;         // [SKC] gpu_pool_skip_census_secs elapsed
static uint32_t g_skc_city_windows = 0;     // [SKC] armed windows at city scale
extern uint8_t g_skc_taint[];
extern uint32_t g_skc_seen[];
extern uint32_t g_skc_gen;
extern uint32_t g_skc_chg[];
extern uint32_t g_skc_chg_n;
extern bool g_skc_chg_ovf;
extern uint32_t g_skc_ntaint;
void SkcWriteBE(uint32_t base, const uint32_t* be, uint32_t n, const uint32_t* rf);

namespace rex::graphics {
// [GKEY-UC] defined in src/graphics/command_processor.cpp (the walk's shader refs).
void NrCtxShaderRef(int which, uint32_t* addr, uint32_t* ndw, uint8_t* immediate,
                    uint8_t* valid);
}  // namespace rex::graphics
#include <rex/graphics/nr_descriptors.h>
#include <rex/graphics/nr_detile.h>
#include <rex/graphics/nr_residency.h>
#include <rex/graphics/nr_sys_constants.h>
#include <rex/graphics/nr_native_pso.h>
#include <rex/graphics/registers.h>
#include <rex/hash.h>  // [NR-RUB] XXH3, header-only (XXH_INLINE_ALL)
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/kernel/xboxkrnl/video.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/ui/d3d12/d3d12_presenter.h>
#include <rex/ui/d3d12/d3d12_util.h>

REXCVAR_DEFINE_BOOL(d3d12_bindless, true, "GPU/D3D12", "Use bindless resources where available")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(d3d12_readback_memexport, false, "GPU/D3D12",
                    "Read data written by memory export in shaders on the CPU")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(d3d12_readback_resolve, false, "GPU/D3D12",
                    "Read render-to-texture results on the CPU")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(d3d12_submit_on_primary_buffer_end, true, "GPU/D3D12",
                    "Submit command list when PM4 primary buffer ends")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_dedupe_constants, false, "GPU/D3D12",
                    "Skip shader-constant register writes whose value is unchanged, "
                    "avoiding needless constant-buffer re-uploads on the command-"
                    "processor thread. Experimental perf lever (Ch.9); default off.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_draw_profile, false, "GPU/D3D12",
                    "Diagnostic: per-second breakdown of where IssueDraw's CPU time goes "
                    "(prim-process / RT-update / pipeline / textures / bindings vs total), "
                    "timed per-draw (cheap, ~a few %). Off by default. Ch.9 cmd-proc triage.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [BATCH-CENSUS] The draw-count question, measured where it matters: of the
// issued draws in a heavy city frame, how many share a {geometry+shader} batch
// key with their NEIGHBOUR (mergeable without reordering) and how many repeat
// ANYWHERE in the frame (poolable with reordering)? The forest answer was
// 92.6% ([[draw-stream-instanceable]]); the city number decides whether
// draw-count reduction is the 2x lever. Key = the inst-probe batch key
// (prim + counts + VS/PS ucode hashes + IB identity + bound-VB hash) - an
// UPPER BOUND on mergeability (RT/viewport state is not in the key). Runs at
// the IssueDrawImpl success seam, outside every [gpu-draw] phase bracket;
// [DRAW-POOL] rung 1b-2 (per-instance pixel-shader constants): a key whose
// pooled draws differed in PS constants (the `vfy ps` refusal) opens its
// next batch with the instanced pixel shader variant, and every occurrence
// appends its PS constants to a second instance block. Unconditional since
// 2026-09-02 (naruto_775 RTX: hit 22.9 -> 29.4%, PS refusals -92%; naruto_776
// Intel: refusals -93%, fps within noise; eye clean; cvars deleted).
// its own cost is bracketed as `bc` on [gpu-draw2]. Logs '[batch-census]' 1 Hz.
REXCVAR_DEFINE_BOOL(gpu_batch_census, false, "GPU/D3D12",
                    "Diagnostic: 1 Hz census of issued draws sharing a batch key with a "
                    "neighbour (adjacent runs) and within the frame (poolable repeats). "
                    "Off by default; '[batch-census]'.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// [GKEY] Rung-2 census (ISSUEDRAW-HANDOFF.md section 4 B, first increment).
// Pair with the game's nr_gkey_census: the guest recorder hook hashes the
// draw pool's inputs from the device shadows and publishes them with the
// record; this side mirrors the hash over the register file at IssueDraw
// and joins by packet address. Counters only. Logs '[gkey]' 1 Hz.
// [SKC] rung-2 skip census: taint the registers a pooled hit's group changed
// and count the plain draws that read one. Counters only, '[pool-skip]'.
REXCVAR_DEFINE_BOOL(gpu_pool_skip_census, false, "GPU/D3D12",
                    "Diagnostic: rung-2 skip census - would dropping a pooled hit's "
                    "register applies change what a later draw reads? Off by default; "
                    "'[pool-skip]'.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_pool_skip_census_secs, 0, "GPU/D3D12",
                     "Diagnostic: keep gpu_pool_skip_census armed for this many CITY "
                     "windows (>= 100k received draws/s), then disarm for the rest of "
                     "the run so the same drive can price the apply path clean. 0 = "
                     "never disarm.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_gkey_census, false, "GPU/D3D12",
                    "Diagnostic: rung-2 census - join the guest hook's shadow-side hash "
                    "of the draw pool's inputs (game cvar nr_gkey_census) against the "
                    "register file at IssueDraw. Off by default; '[gkey]'.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_parallel_record, false, "GPU/D3D12",
                    "[GPU-PRECORD] Phase 1a correctness probe (Ch.9 cmd-proc parallel-record "
                    "track): every N draws, force a full command-list state re-emit into the "
                    "single deferred list, modelling a parallel-segment boundary. Output MUST "
                    "stay pixel-identical (worst case: redundant commands, slightly slower). "
                    "Off by default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_precord_capture, false, "GPU/D3D12",
                    "[GPU-PRECORD] Phase 1b-0 capture/replay correctness probe (Ch.9 cmd-proc "
                    "parallel-record track): defer each draw's recording into a per-segment "
                    "{register snapshot + ordered write/draw log}, then replay the log on the "
                    "same thread (rewinding the register file to the snapshot) into the deferred "
                    "command list at each flush boundary. Output MUST stay pixel-identical. "
                    "Foundation for moving segment recording off-thread. Use alone (do not "
                    "combine with gpu_parallel_record/gpu_instance). Off by default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_precord_localrf, false, "GPU/D3D12",
                    "[GPU-PRECORD] Phase 1b-1b correctness probe (requires gpu_precord_capture): "
                    "replay each captured segment against a PRIVATE local register file (all "
                    "draw-path holders repointed to it) instead of rewinding the shared one. "
                    "Exercises the 1b-1a register-file decoupling on the same thread. Output MUST "
                    "stay pixel-identical to gpu_precord_capture. Off by default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_precord_thread, false, "GPU/D3D12",
                    "[GPU-PRECORD] Phase 1b-1b worker (requires gpu_precord_capture; implies "
                    "gpu_precord_localrf): run each segment's local-register-file replay on a "
                    "dedicated worker thread while the parse thread blocks until it finishes "
                    "(Model C: correctness/plumbing first, no parse/worker overlap yet). Output "
                    "MUST stay pixel-identical. Off by default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_precord_overlap, false, "GPU/D3D12",
                    "[GPU-PRECORD] Phase 1b-1c Inc 5 overlap (requires gpu_precord_thread): "
                    "let the parse thread run AHEAD of the replay worker -- at a 256-draw "
                    "segment boundary the captured slot is POSTed to the worker and the parse "
                    "thread keeps capturing the next segment into the other slot instead of "
                    "blocking (Model C). Backpressure bounds the lead to 1 segment (2 slots); "
                    "true flush points (swap/copy/end-submission/markers/stateful writes) still "
                    "drain the worker for ordering. This is the first parse/worker overlap -> "
                    "the first fps signal. Output MUST stay pixel-identical. Off by default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [GPU-PRECORD H3-PROBE] Phase 1b-1c Inc 6 overlap correctness probe (diagnostic only,
// no behavior change). For each replayed draw, re-hash the guest index buffer and
// compare to the hash captured at parse time; count mismatches (= the guest overwrote
// the IB in the parse->replay lead window). Reports '[gpu-h3]' checked/mismatch/rate
// once/sec. Run it under +thread (Model C, expect ~0) vs +overlap (expect elevated) to
// confirm + quantify the H3 hazard and pick the fix. See naruto-recomp/H3-ROOT-CAUSE.md.
REXCVAR_DEFINE_BOOL(gpu_precord_h3_probe, false, "GPU/D3D12",
                    "[GPU-PRECORD] Phase 1b-1c Inc 6 H3 diagnostic: count replayed draws "
                    "whose guest index buffer was overwritten between capture and replay. "
                    "Off by default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [GPU-DRAW-DUMP] Native-renderer R&D (Ch.9 path B): dump the per-draw "draw
// stream" -- VB/IB guest addresses + sizes, VS/PS ucode hashes, primitive type,
// vertex/index counts -- for the first ~12k draws after enabling, then auto-stops
// so log volume stays bounded (~one heavy frame). This is the data model a native
// renderer would replay: IssueDraw is the ONE seam that sees every world draw
// (the guest scene-graph traversal only covers the HUD). Logs '[gpu-drawdump]'.
REXCVAR_DEFINE_BOOL(gpu_draw_dump, false, "GPU/D3D12",
                    "Native-renderer R&D: dump the per-draw geometry/shader stream "
                    "(VB/IB guest addrs, VS/PS hashes, prim+counts) for the first ~12k "
                    "draws after enabling, then auto-stops. Off by default; '[gpu-drawdump]'.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [INST-PROBE] Instancing feasibility spike (Ch.9, optimize-Xenia route): the
// gpu_draw_dump capture proved one heavy frame's ~12k draws collapse to ~884
// distinct {geometry+shader} batch keys (92.6% redundant) -- the top 3 foliage
// meshes alone are drawn ~5,862x. To turn that into GPU instancing we must confirm
// that across a run of identical-key draws ONLY a per-instance transform constant
// block differs (and which register holds it). When on, for the first ~12k draws
// after enabling this groups draws by batch key and tracks which float constant
// registers ever differ from the first draw of that key, then dumps a per-key
// report ('[inst-probe]'). Arm it IN the heavy scene (console: 'gpu_instance_probe
// true'), same as gpu_draw_dump. Read-only + auto-bounded; cmd-proc thread only.
REXCVAR_DEFINE_BOOL(gpu_instance_probe, false, "GPU/D3D12",
                    "Instancing R&D: group draws by {geometry+shader} batch key and "
                    "report which float constant registers vary per-instance, for the "
                    "first ~12k draws after enabling, then auto-stops. Off by default; "
                    "'[inst-probe]'.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [GPU-INST] Consecutive-draw GPU instancing (Ch.9 draw-volume optimization).
// When on, a run of back-to-back draws that are identical except for their
// vertex float constants (the per-instance transform) is coalesced into a
// single DrawIndexedInstanced; the vertex shader reads each instance's float
// constants via SV_InstanceID. Default OFF (experimental); '[gpu-inst]'.
// Default ON as of 2026-07-26 (was off after ch11 blamed it for tester
// black-screen flashing; that flashing is now attributed to permissive
// gpu_allow_invalid_upload_range, which defaults to reject). always_persist
// keeps the value written to naruto.toml either way, so anyone who does see
// flashing can flip this to false without knowing the key exists.
REXCVAR_DEFINE_BOOL(gpu_instance, false, "GPU/D3D12",
                    "Coalesce consecutive identical-except-transform draws into one "
                    "instanced draw (vertex-shader SV_InstanceID per-instance constants). "
                    "OFF by default: the NR-era city A/B measured the merge capped at 7.5% "
                    "and fps neutral. Reports '[gpu-inst]'.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload)
    .always_persist();

// [NR-NPSO] Phase 5-3a. Defined with the mapping, in pipeline_cache.cpp.
REXCVAR_DECLARE(bool, gpu_nr_native_pso);
REXCVAR_DECLARE(bool, gpu_nr_native_pso_bind);
// [NR-VERIFY] inc 2: defined base-side (command_processor.cpp).
REXCVAR_DECLARE(bool, gpu_nr_detile_vp_probe);
REXCVAR_DECLARE(bool, gpu_nr_verify);
REXCVAR_DECLARE(bool, gpu_nr_reuse_fast);
REXCVAR_DECLARE(bool, gpu_nr_span_swap);
REXCVAR_DECLARE(bool, gpu_nr_span_dedup);
REXCVAR_DECLARE(bool, gpu_nr_span_replay);
// [NR-TIL] N-4-1: the tile-pass replay + the probe it must never co-run with.
REXCVAR_DECLARE(int32_t, gpu_nr_tile_replay);
REXCVAR_DECLARE(bool, gpu_nr_tile_prof);
REXCVAR_DECLARE(int32_t, gpu_nr_tile_probe);
REXCVAR_DECLARE(int32_t, gpu_nr_tile_sys);
REXCVAR_DECLARE(int32_t, gpu_nr_tile_tex);
// [NR-RUF-V2B] guard: the CPU vertex-shader extent path reads float
// constants outside the bitmap-packed packs -- refuse upgrades under it.
REXCVAR_DECLARE(bool, execute_unclipped_draw_vs_on_cpu);

// [NR-BND] Phase 5-3b-0: the bindings mirror's gate.
REXCVAR_DEFINE_BOOL(gpu_nr_bindings, false, "GPU/D3D12",
                    "[NR-BND] Phase 5-3b-0 bindings mirror: recompose the guest constant "
                    "buffers (float VS/PS, bool/loop, fetch) from the draw register file "
                    "with the native renderer's own transcription and byte-compare them "
                    "against every emulated upload inside UpdateBindings; check the "
                    "root-signature selection; census the binding-layout UIDs. Reports "
                    "'[nr-bnd]' once/sec. Diagnostic only, off by default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [NR-SYS] Phase 5-3b-1: the system-constants mirror's gate.
// [NR-SYS] Phase 5-3b-1 system-constants mirror: the bindings swap uploads
// from it. Unconditional (cvar deleted 2026-09-02).

// [NR-DSC] Phase 5-3b-2: the descriptor/sampler mirror's gate.
REXCVAR_DEFINE_BOOL(gpu_nr_desc, false, "GPU/D3D12",
                    "[NR-DSC] Phase 5-3b-2 descriptor/sampler mirror: re-derive sampler "
                    "parameters and texture SRV keys from the draw register file with "
                    "the native renderer's own transcription, mirror the bindless "
                    "sampler-heap index map, and byte-compare our compose of the "
                    "descriptor-indices constant buffers against every emulated rebuild "
                    "inside UpdateBindings. Reports '[nr-dsc]' once/sec. Diagnostic "
                    "only, off by default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [NR-SWP] Phase 5-3b swap: our own UpdateBindings for eligible draws.
// [NR-SWP] Phase 5-3b swap: each draw's bindings are assembled by the native
// renderer's own UpdateBindings, per-draw fallback counted. Unconditional
// (cvar deleted 2026-09-02).

// [NR-LEAN] Phase 5-4-4b inc 2b: skip the emulated sysconst derivation.
// Default ON 2026-08-11 after the city A/B (naruto_380 vs 374: drawstop
// 1.94 -> 1.81us/draw at ~330k draws/s, user: better, all gates zero); the
// verify latch already forces it off for gate runs.
// [NR-LEAN] 5-4-4b inc 2b: under the swap with verify off, the emulated
// UpdateSystemConstantValues body is skipped per draw. Unconditional (cvar
// deleted 2026-09-02).

// [NR-RUB] Phase 5-4-5-1: the replay-reuse bundle gate. For every swapped
// draw with a reuse-probe identity, the derived binding outputs (the four
// guest cbuffer packs, both descriptor-indices composes, sampler params and
// heap indices, root signature) are captured per draw key in a frame-scoped
// bundle; when the base-side v2 verdict says this execution's inputs are
// byte-identical to the previous one, the fresh derivation is compared
// against the stored bundle. ne=0 proves the reuse model BY CONSUMPTION
// before any fast path skips a derivation. System constants are hashed and
// counted separately (bin-dependent NDC fields expected). Pack/di composes
// stage in cached memory and publish with one memcpy under this gate --
// never read an upload heap back.
REXCVAR_DEFINE_BOOL(gpu_nr_ruse_bundle, false, "GPU/D3D12",
                    "[NR-RUB] Phase 5-4-5-1: capture each swapped draw's derived binding "
                    "outputs per draw key and byte-compare them against the previous "
                    "execution whenever the gpu_nr_reuse_probe v2 verdict says the "
                    "draw's inputs are unchanged. Requires the bindings swap and the "
                    "reuse probe. Reports '[nr-rub]' once/sec. Diagnostic only, off by "
                    "default.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [NR-RUF-V2B] Phase 5-4-5-2b: upgrade a STALE-ONLY v2 miss (prev
// comparable, packet span clean, shaders equal -- only the stale-register
// set blocked) to reusable when every stale register is a float constant
// (0x4000-0x47FF) that NEITHER active shader reads. Sound at the byte
// level, not just consumption: the float packs are BITMAP-PACKED (5-3b-0),
// so a constant outside both bitmaps never enters any derived or uploaded
// artifact, and the 5-4-5-1 byte-compare gate stays exact over upgraded
// draws. Fetch/bool-loop/ctl stale regs always refuse (their packs and the
// pipeline description consume them whole). Refused whole-frame when the
// CPU vertex-shader extent path is on (guest-memory reads outside the
// packs), and per draw on dynamic float addressing or memexport stream
// constants (the CP itself reads those float regs).
REXCVAR_DEFINE_BOOL(gpu_nr_reuse_v2b, false, "GPU/D3D12",
                    "[NR-RUF-V2B] Phase 5-4-5-2b: treat a v2 stale-only miss as "
                    "reusable when every stale register is a float constant neither "
                    "active shader reads (bitmap-packed packs cannot carry it). "
                    "Requires the reuse machinery (bundle compare or fast path). "
                    "Off by default until city-gated.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [NR-RSY] Phase 5-3b-3: the residency + descriptor-allocation mirror's gate.
// [NR-RSY] Phase 5-3b-3 residency mirror: the bindings swap requires it.
// Unconditional (cvar deleted 2026-09-02).

// [NR-TILP] N-4-2 outer stamps, defined in graphics/command_processor.cpp
// (the drawstop-bracket side of the seam). See the definitions for the
// protocol; this TU consumes the arm at the tile block, commits on replay
// success, and reports/resets the opost accumulators.
namespace rex::graphics {
extern bool g_tile_oprof;
extern bool g_tile_ostop_armed;
extern bool g_tile_orep;
extern std::chrono::steady_clock::time_point g_tile_ostop_t0;
extern std::chrono::steady_clock::time_point g_tile_orep_t1;
extern uint64_t g_tile_ns_opost;
extern uint64_t g_tile_n_opost;
}  // namespace rex::graphics

// [occ] the occlusion census: every issued draw inside a pipeline-statistics
// query and an occlusion query (1 Hz beside [pool]/[geo]). Names the share
// of the city's primitives that reach zero visible samples at draw time (the
// occlusion lever's lower-bound ceiling), the clipper's frustum share, the
// VS invocation count (prices the vertex-fetch path) and the fill volume.
// [cull] increment 2: frustum refusal from object-space bounds carried by the
// shader's own clip transform (nr_cull.h). 0 off, 1 verify (every draw still
// issued, the verdict tagged into the [occ] query and contradicted by the
// clipper count), 2 skip, 3 cycle off / skip / verify 10 s each. Drive 805
// verified skip (0 visible samples in 107767 out verdicts): default 2.
REXCVAR_DEFINE_INT32(gpu_cull, 2, "GPU/D3D12",
                     "[cull] bounds frustum culling: 0 off (kill switch), 1 verify, 2 skip.");
// [hiz] increment 3: the Hi-Z cull. Every cull-eligible plain draw in the
// frustum becomes an ExecuteIndirect whose arguments a checkpoint compute
// pass writes: the checkpoint reads the bound depth buffer into a per-tile
// (min, max) Hi-Z and tests the next K draws' screen rects and nearest
// depth against it, zeroing the InstanceCount of the hidden ones. Drive 813
// (Intel 720p city, 3200-3599 received dpf): off 13.9 fps, skip 15.1 fps,
// draw class 47.8 -> 40.4 ms/frame with 27% of the frame's indices hidden
// at 2 builds/frame: SHIPPED as the default (skip, tested draws bypass the
// draw pool, rebuild every 800 received draws). 0 off (kill switch), 1
// verify (draws kept, the GPU verdict judged against the [occ] query: WRONG
// = hidden verdict with visible samples, must be 0), 2 skip.
REXCVAR_DEFINE_INT32(gpu_hiz, 2, "GPU/D3D12",
                     "[hiz] Hi-Z occlusion culling: 0 off, 1 verify, 2 skip (default).");
// [hiz-cycle] Drive 843 (Intel 720p): direct draws with NO Hi-Z under a 1x1
// scissor were 5 ms/fr cheaper than the Hi-Z ExecuteIndirect form hiding
// 21% of the indices; the Hi-Z may be a net loss on Intel now that the pool
// is off there and the plain draws are draw-only. Seconds per phase, two
// phases in place: the gpu_hiz mode / off. Read the [hiz] phase= windows
// with geo-sum. 0 = no cycling. Temporary: delete with the drive that reads it.
REXCVAR_DEFINE_INT32(gpu_hiz_cycle, 0, "GPU/D3D12",
                     "[hiz-cycle] Hi-Z re-price A/B, seconds per phase (0 = off).");
REXCVAR_DEFINE_INT32(gpu_hiz_k, 1024, "GPU/D3D12",
                     "[hiz] draws tested per checkpoint (window size, 1..1024).");
REXCVAR_DEFINE_INT32(gpu_hiz_rebuild, 800, "GPU/D3D12",
                     "[hiz] received draws after the last build or invalidation before the Hi-Z "
                     "is (re)built; until then checkpoints test against the stale Hi-Z (still "
                     "conservative) or, when invalid, hide nothing.");
REXCVAR_DEFINE_INT32(gpu_occ_census, 2, "GPU/D3D12",
                     "[occ] per-draw pipeline-statistics + occlusion query census: 0 off, 1 "
                     "always (halves the Intel fps), 2 only in a verify phase of [cull]/[hiz].");
// [gpu-census] N-10c: the naruto_636 iGPU drive proved the Intel city wall
// is the GPU (700-775 ms/s submission-fence blocking); this names what the
// GPU second is spent on before anything is designed against it.
REXCVAR_DEFINE_BOOL(d3d12_gpu_census, true, "GPU/D3D12",
                    "[gpu-census] N-10c: price the GPU second by work class with "
                    "timestamp queries (draw / RTC transfer / resolve / texture upload / "
                    "shared-memory upload / swap), reported once a second. A few hundred "
                    "bottom-of-pipe timestamps a second - kill switch only.");

namespace rex::graphics::d3d12 {

// [hitch] N-10b hitch probe counters (see hitch_probe.h). Written here (fence
// waits) and in pipeline_cache.cpp (PSO creates, EndSubmission wait, library
// outcomes); reported once a second from EndSubmission's frame close.
HitchProbe g_hitch_probe;

namespace {
// [PERF/CONST-DEDUPE] Cached once per frame from the gpu_dedupe_constants cvar.
// Only the command-processor thread reads/writes these, so plain globals (no
// atomics). The hot WriteRegister path reads this bool instead of the cvar
// accessor (which carries a static-local guard check per call).
bool g_dedupe_constants = false;

// [N8B] THE CONSTANT-REDUNDANCY CENSUS.
// N-8's dword census said 82.7% of the command processor's 63.9 M applied
// register dwords a second are ALU shader constants: 288 dwords, 72 float4s,
// re-uploaded on EVERY draw, copied into the register file and then copied
// again into a cbuffer. A pool is not a cost, so before designing a zero-copy
// constant path, ask the cheap question first -- how many of those dwords are
// re-writes of a value that is already there?
//   dword-level sameness = the ceiling for ANY dedupe.
//   range-level sameness = what the whole-range early return below actually
//     removes, and it removes the cbuffer re-upload too, which is the SECOND
//     copy and lives inside IssueDraw's 42%.
// Class index: 0 = ALU float (0x4000), 1 = bool/loop (0x4900), 2 = fetch
// (0x4800 -- censused, never deduped: the texture-cache and vertex-residency
// invalidations are correctness, not bookkeeping).
//
// Mode, set once a frame in IssueSwap:
//   0 = off          -- no compare at all, this is the clean fps baseline
//   1 = census only  -- compare and count, act on nothing. Prices the compare.
//   2 = census+act   -- the whole-range early return fires (the dedupe).
// `gpu_n7 3` cycles the three 5 s each IN PLACE ([[price-stages-in-place]]),
// so one drive gives the redundancy fractions, the compare's own cost, and the
// net win, all at matched draws per frame.
int g_n8b_mode = 0;
uint64_t g_n8b_rng[3] = {}, g_n8b_rng_same[3] = {};
uint64_t g_n8b_dw[3] = {}, g_n8b_dw_same[3] = {};
uint64_t g_n8b_pdw[3] = {}, g_n8b_pdw_same[3] = {};
uint64_t g_n8b_skip_rng = 0, g_n8b_skip_dw = 0;

// [N8C] The SECOND copy, counted. The constants are copied into the register
// file by the walk and then composed again into a per-draw upload-heap
// cbuffer by NrUpdateBindings. That second copy is the part that lives inside
// IssueDraw's 42%, and it only happens when a binding went stale. These four
// counters plus the per-draw denominator say how many cbuffer re-uploads a
// draw actually costs -- which is what an N-8b dedupe has to move, and the
// only way an fps-neutral result can be explained rather than shrugged at.
// Free (one increment on a path that is already writing kilobytes), so they
// are unconditional and reported whenever [n7] is on, in EVERY phase.
// Index: 0 = float vertex, 1 = float pixel, 2 = bool/loop, 3 = fetch.
uint64_t g_n8c_up[4] = {};
uint64_t g_n8c_binds = 0;

// One bulk constant range: count how many of its dwords already match the
// register file, and report whether ALL of them did. base[] is guest-endian,
// the register file is host-endian, so swap-compare -- same load the copy
// below would do.
inline bool N8bCensusRange(int cls, const uint32_t* be, const uint32_t* cur, uint32_t n) {
  uint32_t same = 0;
  for (uint32_t i = 0; i < n; ++i) {
    same += (memory::load_and_swap<uint32_t>(be + i) == cur[i]) ? 1u : 0u;
  }
  ++g_n8b_rng[cls];
  g_n8b_dw[cls] += n;
  g_n8b_dw_same[cls] += same;
  if (same != n) {
    return false;
  }
  ++g_n8b_rng_same[cls];
  return true;
}

// [PERF/DRAW-PROFILE] Per-draw phase timing (cmd-proc thread only -> plain
// globals). Refreshed once/frame in IssueSwap; reported there when on.
// Index: 0=prim 1=rt-update 2=pipeline 3=textures 4=bindings 5=total IssueDraw
//        6=BeginSubmission 7=draw-tail(residency+barriers+draw) 8=fixed-fn+sysconst.
bool g_draw_prof = false;
bool g_tile_prof = false;  // [NR-TILP] N-4-2 head stage split
// [GPU-DRAW] 0..8 = the 9 named phases; 9..11 = tail sub-brackets
// (9 = vertex-residency loop, 10 = primitive topology, 11 = index-buffer
// view + barriers + deferred emission + memexport finish), added 5-4-4b so
// a city run can name the dominant slice of the tail without guessing.
// 12..15 = `other` sub-brackets (5-4-4b inc 3): 12 = head (entry ->
// BeginSubmission: shader-ucode analysis, rasterization checks, instanced
// merge, pipeline lock), 13 = mods (prim -> rt-update: interpolator mask +
// shader modification selection), 14 = trans (rt-update -> pso:
// GetOrCreateTranslation + bound-RT query), 15 = vp (tex -> ff+sc: native
// pso lookup/bind + viewport cache key + scissor), 16 = copy (the IssueCopy
// early-return path: resolves ride IssueDrawImpl's total bracket but close
// no phase bracket, so without this they hide in rest). rest = other - sum,
// printed so the split provably covers the slice.
// 17..19 = vp sub-brackets (17 = NrNativePipeline lookup + pipeline bind,
// 18 = viewport cache key build/compare + host viewport info, 19 = scissor).
// 20..26 = `rest` sub-brackets (ISSUEDRAW census follow-up: rest was 62.7 ms/s
// at the city in naruto_751, the largest single item, and unnamed - exactly
// the remainder shape [[subtraction-buckets-mislabel]] warns about). 20 = cen
// (BeginSubmission -> prim: census class, precord segment bookkeeping, rsy
// setup), 21 = ruse (trans -> pso: the RUF-V2B upgrade verdict + NrRubFind
// fast-pso probe), 22 = pret (pso -> tex: async-compile check +
// GetD3D12PipelineByHandle + pipeline-lock release + texture-mask derivation),
// 23 = rsp (ff+sc -> bind: the NR-BND root-signature predict block), 24 = rply
// (a tile/span replay-served draw: total from entry to its return - it closes
// NO phase bracket, so before this it hid whole in rest), 25 = tilr (post-tail
// NrTileRecordEnd/CompareEnd), 26 = bc (the batch-key census's own per-draw
// work when gpu_batch_census rides the same drive). rr = rest - sum, printed
// on [gpu-draw2] so the split provably covers the slice.
uint64_t g_draw_ns[27] = {};
// [GPU-DRAW2] early-return classes: a draw that leaves IssueDrawImpl before
// the tail contributes total but only partial phase brackets; the lost span is
// rest. Counters name the population: 0=tile-replay 1=span-replay 2=no-vs
// 3=no-effect (zero surface pitch / no rasterization) 4=zero host vertices
// 5=async-pso skip.
uint64_t g_draw_ret[6] = {};
// [GPU-DRAW] 5-4-4b inc 3: NrUpdateBindings sub-profile (g_draw_prof only,
// printed as [nr-bndp]). ns: 0=cbuffer composes 1=sampler-params derivation
// 2=srv-key freshness checks 3=sampler-heap find-or-allocate
// 4=descriptor-indices composes 5=root-parameter tail, 6=time inside the
// constant-pool Request calls alone (subset of 0 and 4 - prices the
// combined-allocation lean against the gathers). Counters:
// 0=DescSamplerParams evals 1=srv-value derivations 2=constant-pool
// Requests, 3..7=per-cbuffer recompose fires (sys/float_v/float_p/
// bool_loop/fetch). Names the cost driver inside bind before any lean.
uint64_t g_bind_ns[7] = {};
uint64_t g_bind_cnt[8] = {};
// [GPU-DRAW] residency-loop counters: slots visited / in_sync-bit hits /
// state-match re-arms (no request) / RequestRange calls / RequestRange
// calls whose exact {addr,size} was already requested since the last swap
// (a value-keyed front cache would eliminate exactly these). Names the
// inner cost driver of the res bracket instead of guessing it.
uint64_t g_draw_res_cnt[5] = {0, 0, 0, 0, 0};
// Per-frame set of requested ranges for the dup counter (probe-only).
std::unordered_set<uint64_t> g_draw_res_seen;
uint64_t g_draw_count = 0;
// [GPU-PRECORD] Phase 1a: cached once/frame in IssueSwap. Segment size = draws
// between forced full-state re-emits (correctness probe; tunable cvar comes in Phase 3).
bool g_parallel_record = false;
static constexpr uint32_t kParallelRecordSegmentDraws = 256;
// [GPU-PRECORD] Phase 1b-0: cached once/frame in IssueSwap. When on, draws are
// deferred into a per-segment event log and replayed at flush boundaries (segment
// size reuses kParallelRecordSegmentDraws).
bool g_precord_capture = false;
// [GPU-PRECORD] Phase 1b-1b: cached once/frame in IssueSwap. localrf ⇒ replay
// against a private local register file; thread ⇒ run that replay on the worker
// (implies localrf). Read on the parse thread (dispatch) only; the worker reads
// these via the parse thread's happens-before. Plain bools (cmd-proc thread).
bool g_precord_localrf = false;
bool g_precord_thread = false;
// [GPU-PRECORD] Phase 1b-1c Inc 5: overlap (parse runs ahead of the worker). Only
// meaningful with g_precord_thread; gates every Inc 5 behavioral change so
// gpu_precord_thread (Model C) stays the pixel-identical A/B reference. Read on the
// parse thread; the worker never consults it. Cached once/frame in IssueSwap.
bool g_precord_overlap = false;
// [GPU-PRECORD H3-PROBE] Phase 1b-1c Inc 6 overlap correctness probe (default OFF,
// cvar gpu_precord_h3_probe). For each replayed draw it re-hashes the guest index
// buffer and compares to the hash captured at parse time; a mismatch means the guest
// overwrote that IB in the parse->replay lead window (the H3 hazard). Counters are read
// once/sec in IssueSwap (parse thread) but bumped at replay on the worker thread under
// overlap, so they are atomic. The mismatch RATE decides the H3 fix: near-0 under
// +thread (Model C baseline) but elevated under +overlap == H3 confirmed + quantified,
// and a high rate means fences are frequent relative to draws (Option C would
// over-serialize -> favor Option A'). See naruto-recomp/H3-ROOT-CAUSE.md.
bool g_h3_probe = false;
std::atomic<uint64_t> g_h3_ib_checked{0};
std::atomic<uint64_t> g_h3_ib_mismatch{0};
// [GPU-PRECORD H3-PROBE] Inc 6: the vertex-buffer counterpart. The IB probe measured
// 0.0% overwrites in the heavy forest while foliage flickered, so the H3 corruption is
// the VERTEX buffer (or a texture); these quantify the VB hazard the same way.
std::atomic<uint64_t> g_h3_vb_checked{0};
std::atomic<uint64_t> g_h3_vb_mismatch{0};
// [GPU-DRAW-DUMP] Cached once/frame; counter auto-stops the dump after the cap so
// log volume stays bounded. Worker-thread-only, so plain globals (no atomics).
bool g_draw_dump = false;
uint64_t g_draw_dump_count = 0;
constexpr uint64_t kDrawDumpCap = 12000;

// [INST-PROBE] Instancing feasibility spike state. Cmd-proc(worker)-thread only =>
// plain globals, no atomics. Bounded: stops accumulating after kDrawDumpCap draws,
// then dumps a per-batch-key report exactly once.
bool g_inst_probe = false;
uint64_t g_inst_draw_count = 0;
bool g_inst_dumped = false;
// [INST-PROBE] city latch: issued draws this frame (unconditional ++ at the
// IssueDrawImpl success seam, reset each swap) and the once-latched "a city
// frame has been seen" flag. The probe capture is one-shot (kDrawDumpCap), so
// armed at launch it would spend its whole cap on the boot/menu stream; it
// stays dormant until a frame issues >= 2500 draws.
uint64_t g_frame_issued = 0;
bool g_inst_city = false;

// [GPU-INST] Cached gpu_instance enable (refreshed once/frame, cmd-proc thread
// only). g_instance_dirty tracks whether any register OUTSIDE the vertex
// float-constant range [SHADER_CONSTANT_000_X, SHADER_CONSTANT_256_X) has
// changed value since the last draw -- if not, the next draw differs only in
// its per-instance transform and can be merged. Stats count coalesced guest
// draws (in) vs emitted instanced draws (out).
bool g_instance = false;
bool g_instance_dirty = true;
uint64_t g_instance_draws_in = 0;
uint64_t g_instance_draws_out = 0;
// [GPU-INST] Flush-reason census (cmd-proc thread only): what limits batch
// length at a given scene. fail = why the merge check refused (dirty /
// shader / prim+count / ib / capacity), site = where a live batch was
// flushed (merge-fail / copy / submission-end+present), hist = flushed batch
// size buckets (1 / 2-4 / 5-16 / 17-64 / 65-256 / 257+), dirty_reg_census =
// which register FIRST broke the float-constants-only invariant (tallied at
// the merge-fail moment, cumulative).
uint32_t g_instance_dirty_first_reg = UINT32_MAX;
uint32_t g_inst_fail_reason = 0;
uint64_t g_inst_fail[5] = {};
uint64_t g_inst_flush_site[3] = {};
uint64_t g_inst_hist[6] = {};
std::unordered_map<uint32_t, uint64_t> g_inst_dirty_reg_census;

// [NR-NPSO] Phase 5-3a: latched once a frame beside the others, so the draw
// path never reads a cvar. g_nr_native_pso builds and checks our own pipeline
// object; g_nr_native_pso_bind is the swap, and implies the first.
bool g_nr_native_pso = false;
bool g_nr_native_pso_bind = false;

// [NR-BND] Phase 5-3b-0: latched once a frame beside the others; counters are
// cmd-proc-thread-only => plain globals. The *_ne counters are cumulative
// must-be-0 gates; volume counters are reported as per-second deltas by the
// 1Hz line. What each pair means: *_ne = the uploaded bytes differ from our
// recomposition; *_size_ne = the two sides disagree about how MANY bytes the
// shader's constant map packs (checked against the emulated write cursor, so
// it also catches a bitmap-iteration transcription slip).
bool g_nr_bindings = false;
struct NrBindProbe {
  uint64_t draws = 0;  // UpdateBindings calls observed under the gate
  uint64_t fv_up = 0, fv_size_ne = 0, fv_ne = 0;  // float VS uploads
  uint64_t fp_up = 0, fp_size_ne = 0, fp_ne = 0;  // float PS uploads
  uint64_t bl_up = 0, bl_ne = 0;                  // bool/loop uploads
  uint64_t fx_up = 0, fx_ne = 0;                  // fetch uploads
  uint64_t sys_up = 0;  // system-constant uploads (contents = a later peel)
  uint64_t div_up = 0, dip_up = 0;  // bindless descriptor-indices rebuilds
  uint64_t rs_checks = 0, rs_ne = 0, rs_tess = 0, rs_bindful = 0;
  uint64_t tex_v = 0, smp_v = 0, tex_p = 0, smp_p = 0;  // bindings, summed
  rex::graphics::nr::BindCensus layout_census;  // distinct layout-UID tuples
};
NrBindProbe g_nr_bind{};
// [NR-BND] Staging for the float-constant gate. The constant pool is an
// UPLOAD heap mapped with an empty read range: reading it back is
// write-combined-memory poison (~100x a cached read) and the first city run
// (naruto_335) measured exactly that -- the CP thread lost ~2/3 of its
// throughput to the memcmps. Under the gate the emulated pack loop writes
// HERE instead and one memcpy publishes to the heap, so the uploaded bytes
// are these bytes by construction and the compare never touches WC memory.
// CP-thread-only, and the two float blocks run sequentially, so one buffer.
alignas(16) uint8_t g_nr_bnd_staging[rex::graphics::nr::kBindFloatMaxBytes];

// [NR-BND] The 1Hz verdict line, emitted from the per-frame probe block so no
// clock is read on the draw path.
void NrBindReportIfDue() {
  if (!g_nr_bindings || !g_nr_bind.draws) {
    return;
  }
  static auto s_last = std::chrono::steady_clock::now();
  static NrBindProbe s_prev{};
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) {
    return;
  }
  s_last = now;
  const NrBindProbe& p = g_nr_bind;
  const uint64_t d = p.draws - s_prev.draws;
  const double per_draw = d ? double(d) : 1.0;
  REXGPU_INFO(
      "[nr-bnd] draws={} | up floatv={} floatp={} boolloop={} fetch={} sys={} "
      "idxv={} idxp={} | ne floatv={}/{} floatp={}/{} boolloop={} fetch={} | "
      "rootsig checks={} ne={} tess={} bindful={} | layouts={} ovf={} | "
      "bind/draw texv={:.1f} smpv={:.1f} texp={:.1f} smpp={:.1f}",
      d, p.fv_up - s_prev.fv_up, p.fp_up - s_prev.fp_up, p.bl_up - s_prev.bl_up,
      p.fx_up - s_prev.fx_up, p.sys_up - s_prev.sys_up, p.div_up - s_prev.div_up,
      p.dip_up - s_prev.dip_up, p.fv_ne, p.fv_size_ne, p.fp_ne, p.fp_size_ne,
      p.bl_ne, p.fx_ne, p.rs_checks - s_prev.rs_checks, p.rs_ne, p.rs_tess,
      p.rs_bindful, p.layout_census.count, p.layout_census.ovf,
      double(p.tex_v - s_prev.tex_v) / per_draw,
      double(p.smp_v - s_prev.smp_v) / per_draw,
      double(p.tex_p - s_prev.tex_p) / per_draw,
      double(p.smp_p - s_prev.smp_p) / per_draw);
  s_prev = p;
}

// [NR-SYS] Phase 5-3b-1: latched once a frame beside the others; counters are
// cmd-proc-thread-only => plain globals. mismatch is the cumulative must-be-0
// gate (a check whose struct memcmp differs); refused_rov counts derivations
// declined because the ROV path is out of the mirror's scope (must stay 0
// outside unit tests in this game); the cover_* counters say which conditional
// branches the run actually exercised, so a clean verdict can be quoted with
// its coverage instead of over-claimed (the 5-1 distinct_state lesson).
bool g_nr_sysconst = false;
struct NrSysProbe {
  uint64_t checks = 0;
  uint64_t mismatch = 0;
  uint64_t refused_rov = 0;
  uint64_t reseeds = 0;
  uint64_t cover_clip = 0;    // draws with user clip planes packed
  uint64_t cover_point = 0;   // point-list draws (point size fields written)
  uint64_t cover_atest = 0;   // alpha test enabled
  uint64_t cover_a2m = 0;     // alpha to mask enabled
  uint64_t cover_gamma = 0;   // draws with any gamma-convert flag set
  uint64_t cover_tex = 0;     // used-texture slots visited (signs RMW'd)
};
NrSysProbe g_nr_sys{};
// The persistent mirror. The emulated system_constants_ member is never
// initialized and many of its fields are written conditionally (sticky), so
// the mirror is SEEDED from it once when the gate arms; from that moment on
// every field the derivation writes is ours. Re-armed => re-seeded (the
// emulated struct kept evolving while the gate was off).
rex::graphics::nr::NrSysConstants g_nr_sys_state;
bool g_nr_sys_seeded = false;
// FIRST DIFFERENCE lines are capped per report window so a systematic
// mismatch cannot flood the log.
uint32_t g_nr_sys_samples_this_window = 0;
constexpr uint32_t kNrSysMaxSamplesPerWindow = 6;

// [NR-SYS] The texture-cache queries, reached through callbacks so the
// derivation module stays SDK-free.
uint8_t NrSysTextureSigns(void* ctx, uint32_t fetch_constant_index) {
  return static_cast<const TextureCache*>(ctx)->GetActiveTextureSwizzledSigns(
      fetch_constant_index);
}
bool NrSysTextureResScaled(void* ctx, uint32_t fetch_constant_index) {
  return static_cast<const TextureCache*>(ctx)->IsActiveTextureResolutionScaled(
      fetch_constant_index);
}

// [NR-SYS] The 1Hz verdict line, emitted from the per-frame probe block.
void NrSysReportIfDue() {
  if (!g_nr_sysconst || !g_nr_sys.checks) {
    return;
  }
  static auto s_last = std::chrono::steady_clock::now();
  static NrSysProbe s_prev{};
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) {
    return;
  }
  s_last = now;
  const NrSysProbe& p = g_nr_sys;
  REXGPU_INFO(
      "[nr-sys] checks={} | mismatch={} refused_rov={} reseeds={} | cover "
      "clip={} point={} atest={} a2m={} gammart={} tex={}",
      p.checks - s_prev.checks, p.mismatch, p.refused_rov, p.reseeds,
      p.cover_clip - s_prev.cover_clip, p.cover_point - s_prev.cover_point,
      p.cover_atest - s_prev.cover_atest, p.cover_a2m - s_prev.cover_a2m,
      p.cover_gamma - s_prev.cover_gamma, p.cover_tex - s_prev.cover_tex);
  s_prev = p;
  g_nr_sys_samples_this_window = 0;
}

// [NR-DSC] Phase 5-3b-2: latched once a frame beside the others; counters are
// cmd-proc-thread-only => plain globals. The *_ne counters are cumulative
// must-be-0 gates; volume counters are per-second deltas on the 1Hz line.
// Coverage split: smp_* = per-draw SamplerParameters derivations, key_* =
// per-draw texture SRV key derivations, smap_* = the bindless sampler-heap
// index mirror (seeded entries = allocated before the gate armed, like the
// 5-3b-1 seed; fresh = allocation index PREDICTED while armed), dc_* = the
// descriptor-indices cbuffer byte compares (vertex/pixel rebuilds).
bool g_nr_desc = false;
struct NrDescProbe {
  uint64_t smp_checks = 0, smp_ne = 0;
  uint64_t key_checks = 0, key_ne = 0;
  uint64_t key_invalid = 0;  // null-binding keys observed (coverage)
  uint64_t smap_match = 0, smap_fresh = 0, smap_seeded = 0, smap_ne = 0,
           smap_ovf = 0;
  uint64_t heap_switches = 0;  // bindless sampler heap overflowed + swapped
  uint64_t dc_v = 0, dc_p = 0, dc_ne_v = 0, dc_ne_p = 0;
  uint64_t dc_refused = 0;  // compose refusals (capacity/slot) - must stay 0
  uint64_t layout_bad = 0;  // one-shot bit-layout self-check failures
};
NrDescProbe g_nr_desc_probe{};
// The bindless sampler-heap mirror (params -> heap index + allocation
// counter). Reset + re-seeded on every rising edge of the gate and on a heap
// switch; entries then learn (seeded) or predict (fresh) per observation.
rex::graphics::nr::DescSamplerMap g_nr_desc_smap;
// Staging for the descriptor-indices compare: the emulated rebuild writes
// here instead of the upload heap and one memcpy publishes, so the compare
// never reads write-combined memory ([[upload-heap-readback-trap]]).
// 1024 dwords >> any shader's texture+sampler binding count.
constexpr uint32_t kNrDescStagingDwords = 1024;
uint32_t g_nr_desc_staging[kNrDescStagingDwords];
uint32_t g_nr_desc_ours[kNrDescStagingDwords];
// Scratch (slot, value) pair arrays for our compose (CP-thread-only).
uint32_t g_nr_desc_tex_slots[kNrDescStagingDwords];
uint32_t g_nr_desc_tex_vals[kNrDescStagingDwords];
uint32_t g_nr_desc_smp_slots[kNrDescStagingDwords];
uint32_t g_nr_desc_smp_vals[kNrDescStagingDwords];
// FIRST DIFFERENCE lines are capped per report window.
uint32_t g_nr_desc_samples_this_window = 0;
constexpr uint32_t kNrDescMaxSamplesPerWindow = 6;

// The module transcribes the xenos enum values as plain constants; pin them.
static_assert(uint32_t(xenos::FetchConstantType::kInvalidTexture) ==
              nr::kDescFetchTypeInvalidTexture);
static_assert(uint32_t(xenos::FetchConstantType::kTexture) ==
              nr::kDescFetchTypeTexture);
static_assert(uint32_t(xenos::DataDimension::k1D) == nr::kDescDim1D);
static_assert(uint32_t(xenos::DataDimension::k2DOrStacked) ==
              nr::kDescDim2DOrStacked);
static_assert(uint32_t(xenos::DataDimension::k3D) == nr::kDescDim3D);
static_assert(uint32_t(xenos::DataDimension::kCube) == nr::kDescDimCube);
static_assert(uint32_t(xenos::ClampMode::kClampToEdge) == nr::kDescClampToEdge);
static_assert(uint32_t(xenos::ClampMode::kClampToHalfway) ==
              nr::kDescClampToHalfway);
static_assert(uint32_t(xenos::ClampMode::kMirrorClampToEdge) ==
              nr::kDescMirrorClampToEdge);
static_assert(uint32_t(xenos::ClampMode::kMirrorClampToHalfway) ==
              nr::kDescMirrorClampToHalfway);
static_assert(uint32_t(xenos::ClampMode::kClampToBorder) ==
              nr::kDescClampToBorder);
static_assert(uint32_t(xenos::ClampMode::kMirrorClampToBorder) ==
              nr::kDescMirrorClampToBorder);
static_assert(uint32_t(xenos::TextureFilter::kPoint) == nr::kDescFilterPoint);
static_assert(uint32_t(xenos::TextureFilter::kLinear) == nr::kDescFilterLinear);
static_assert(uint32_t(xenos::TextureFilter::kBaseMap) ==
              nr::kDescFilterBaseMap);
static_assert(uint32_t(xenos::TextureFilter::kUseFetchConst) ==
              nr::kDescFilterUseFetchConst);
static_assert(uint32_t(xenos::AnisoFilter::kDisabled) == nr::kDescAnisoDisabled);
static_assert(uint32_t(xenos::AnisoFilter::kMax_16_1) == nr::kDescAnisoMax16);
static_assert(uint32_t(xenos::AnisoFilter::kUseFetchConst) ==
              nr::kDescAnisoUseFetchConst);
static_assert(uint32_t(xenos::XE_GPU_TEXTURE_SWIZZLE_0000) ==
              nr::kDescHostSwizzle0000);
static_assert(xenos::kTexture2DCubeMaxWidthHeight == nr::kDescMaxWidthHeight2D);
static_assert(sizeof(D3D12TextureCache::SamplerParameters) == sizeof(uint32_t));

// [NR-DSC] The host-format swizzle query, reached through a callback so the
// derivation module stays SDK-free (static config of the D3D12 backend).
uint32_t NrDescHostFormatSwizzle(void*, uint32_t base_format) {
  return D3D12TextureCache::NrHostFormatSwizzle(base_format);
}

// [NR-DSC] One-shot bit-LAYOUT self-check, run at every gate arm: pokes known
// values into the emulated bitfield structs and checks the packed dwords land
// where the module's shift transcription says they do (a static_assert cannot
// see bitfield placement). A failure refuses nothing but counts layout_bad,
// which must stay 0 - every byte compare after it would name itself anyway.
void NrDescLayoutSelfCheck() {
  {
    D3D12TextureCache::SamplerParameters sp{};
    sp.clamp_x = xenos::ClampMode(5);
    sp.clamp_y = xenos::ClampMode(3);
    sp.clamp_z = xenos::ClampMode(6);
    sp.border_color = xenos::BorderColor(2);
    sp.mag_linear = 1;
    sp.mip_linear = 1;
    sp.aniso_filter = xenos::AnisoFilter(3);
    sp.mip_min_level = 9;
    sp.mip_base_map = 1;
    const uint32_t expected = 5u | 3u << 3 | 6u << 6 | 2u << 9 | 1u << 11 |
                              1u << 13 | 3u << 14 | 9u << 17 | 1u << 21;
    if (sp.value != expected) {
      ++g_nr_desc_probe.layout_bad;
      REXGPU_ERROR("[nr-dsc] LAYOUT SamplerParameters value={:#x} expected={:#x}",
                   sp.value, expected);
    }
  }
  {
    D3D12TextureCache::TextureSRVKey k{};
    k.key.base_page = 0x1ABCD;
    k.key.dimension = xenos::DataDimension(3);
    k.key.width_minus_1 = 0x1F0F;
    k.key.height_minus_1 = 0x1234;
    k.key.tiled = 1;
    k.key.mip_page = 0x155AA;
    k.key.depth_or_array_size_minus_1 = 0x3F5;
    k.key.pitch = 0x1A5;
    k.key.mip_max_level = 0xB;
    k.key.format = xenos::TextureFormat(0x2A);
    k.key.endianness = xenos::Endian(1);
    k.key.is_valid = 1;
    uint32_t got[4];
    static_assert(sizeof(k.key) == sizeof(got));
    std::memcpy(got, &k.key, sizeof(got));
    const uint32_t expect[4] = {
        0x1ABCDu | 3u << 17 | 0x1F0Fu << 19,
        0x1234u | 1u << 13 | 0x155AAu << 15,
        0x3F5u | 0x1A5u << 10 | 0xBu << 19 | 0x2Au << 23 | 1u << 29,
        1u << 1};
    if (std::memcmp(got, expect, sizeof(got)) != 0) {
      ++g_nr_desc_probe.layout_bad;
      REXGPU_ERROR(
          "[nr-dsc] LAYOUT TextureKey got={:08x} {:08x} {:08x} {:08x} "
          "expected={:08x} {:08x} {:08x} {:08x}",
          got[0], got[1], got[2], got[3], expect[0], expect[1], expect[2],
          expect[3]);
    }
  }
}

// [NR-DSC] The 1Hz verdict line, emitted from the per-frame probe block.
void NrDescReportIfDue() {
  if (!g_nr_desc || !(g_nr_desc_probe.smp_checks + g_nr_desc_probe.key_checks)) {
    return;
  }
  static auto s_last = std::chrono::steady_clock::now();
  static NrDescProbe s_prev{};
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) {
    return;
  }
  s_last = now;
  const NrDescProbe& p = g_nr_desc_probe;
  REXGPU_INFO(
      "[nr-dsc] smp checks={} ne={} | key checks={} ne={} inv={} | smap "
      "match={} fresh={} seeded={} ne={} ovf={} heapswitch={} | dc v={} p={} "
      "ne={}/{} refused={} | layout_bad={}",
      p.smp_checks - s_prev.smp_checks, p.smp_ne,
      p.key_checks - s_prev.key_checks, p.key_ne,
      p.key_invalid - s_prev.key_invalid, p.smap_match - s_prev.smap_match,
      p.smap_fresh, p.smap_seeded, p.smap_ne, p.smap_ovf, p.heap_switches,
      p.dc_v - s_prev.dc_v, p.dc_p - s_prev.dc_p, p.dc_ne_v, p.dc_ne_p,
      p.dc_refused, p.layout_bad);
  s_prev = p;
  g_nr_desc_samples_this_window = 0;
}

// [NR-RSY] Phase 5-3b-3: latched once a frame beside the others; counters are
// cmd-proc-thread-only => plain globals. The *_ne counters are cumulative
// must-be-0 gates; volume counters are per-second deltas on the 1Hz line.
// Coverage split: vf_* = the per-draw vertex-buffer residency walk (predicted
// RequestRange list + post-loop state memcmp), ib_* = the index-buffer
// request derivation (the primitive processor's convert-vs-DMA decision is a
// declared input, counted by outcome), mx = memexport draws (ranges pass
// through uncompared - refused class, must stay 0), pool_* = the bindless
// view-descriptor allocator mirror (every persistent + one-use allocation
// PREDICTED), tmap_* = the per-texture SRV descriptor map mirror (fresh =
// find-or-create predicted while armed, seeded = pre-arm entries learned),
// srv_* = the texture SRV index VALUES in every descriptor-indices rebuild.
bool g_nr_res = false;
struct NrResProbe {
  uint64_t vf_draws = 0, vf_req = 0, vf_ne = 0, vf_state_ne = 0;
  uint64_t vf_abort_type = 0, vf_req_fail = 0, vf_pred_ovf = 0;
  uint64_t ib_checks = 0, ib_match = 0, ib_ne = 0;
  uint64_t ib_conv = 0, ib_auto = 0, ib_unexpected = 0;
  uint64_t mx_draws = 0;
  uint64_t pool_alloc = 0, pool_oneuse = 0, pool_release = 0, pool_ne = 0;
  uint64_t pool_refused = 0, pool_reseeds = 0;
  uint64_t tmap_match = 0, tmap_fresh = 0, tmap_seeded = 0, tmap_ne = 0,
           tmap_ovf = 0, tmap_evict = 0, tmap_invalid = 0;
  uint64_t srv_checks = 0, srv_ne = 0, srv_null = 0, srv_special = 0,
           srv_unknown = 0;
  uint64_t layout_bad = 0;
};
NrResProbe g_nr_res_probe{};
nr::ResVfetchMirror g_nr_res_vf;
nr::ResViewPool g_nr_res_pool;
nr::ResTexDescMap g_nr_res_tmap;
// Per-draw scratch (CP-thread-only). 128 > the 96-slot maximum.
constexpr uint32_t kNrResMaxRequests = 128;
nr::ResRange g_nr_res_vf_pred[kNrResMaxRequests];
uint32_t g_nr_res_vf_pred_count = 0;
nr::ResVfetchStatus g_nr_res_vf_pred_status = nr::kResVfetchOk;
nr::ResRange g_nr_res_vf_obs[kNrResMaxRequests];
uint32_t g_nr_res_vf_obs_count = 0;
uint32_t g_nr_res_vf_obs_ok = 0;
bool g_nr_res_vf_active = false;
bool g_nr_res_vf_allow = false;
uint32_t g_nr_res_vf_bitmap[3] = {0, 0, 0};
const uint32_t* g_nr_res_vf_fetch_regs = nullptr;
// The index-request observation for the current Process call.
bool g_nr_res_ib_seen = false;
uint32_t g_nr_res_ib_base = 0;
uint32_t g_nr_res_ib_len = 0;
bool g_nr_res_ib_result = false;
uint32_t g_nr_res_samples_this_window = 0;
constexpr uint32_t kNrResMaxSamplesPerWindow = 6;

// The module transcribes the raw values; pin them against the SDK enums.
static_assert(uint32_t(xenos::FetchConstantType::kInvalidVertex) ==
              nr::kResFetchTypeInvalidVertex);
static_assert(uint32_t(xenos::FetchConstantType::kVertex) ==
              nr::kResFetchTypeVertex);
static_assert(uint32_t(xenos::SourceSelect::kDMA) == nr::kResSourceSelectDma);
static_assert(uint32_t(xenos::IndexFormat::kInt16) == nr::kResIndexFormatInt16);
static_assert(uint32_t(xenos::IndexFormat::kInt32) == nr::kResIndexFormatInt32);
static_assert(uint32_t(xenos::FetchOpDimension::k1D) == nr::kResFetchDim1D);
static_assert(uint32_t(xenos::FetchOpDimension::k2D) == nr::kResFetchDim2D);
static_assert(uint32_t(xenos::FetchOpDimension::k3DOrStacked) ==
              nr::kResFetchDim3DOrStacked);
static_assert(uint32_t(xenos::FetchOpDimension::kCube) == nr::kResFetchDimCube);
static_assert(uint32_t(xenos::DataDimension::k1D) == nr::kResDataDim1D);
static_assert(uint32_t(xenos::DataDimension::k2DOrStacked) ==
              nr::kResDataDim2DOrStacked);
static_assert(uint32_t(xenos::DataDimension::k3D) == nr::kResDataDim3D);
static_assert(uint32_t(xenos::DataDimension::kCube) == nr::kResDataDimCube);
static_assert(xenos::kVertexFetchConstantCount == nr::kResVfetchSlots);

// [NR-RSY] One-shot bit-LAYOUT self-check, run at every gate arm (a
// static_assert cannot see bitfield placement). A failure counts layout_bad,
// which must stay 0.
void NrResLayoutSelfCheck() {
  {
    xenos::xe_gpu_vertex_fetch_t vf;
    vf.dword_0 = 0;
    vf.dword_1 = 0;
    vf.type = xenos::FetchConstantType::kVertex;
    vf.address = 0x2ABCDE1;
    vf.endian = xenos::Endian(2);
    vf.size = 0x123456;
    const uint32_t expect_0 = 3u | 0x2ABCDE1u << 2;
    const uint32_t expect_1 = 2u | 0x123456u << 2;
    if (vf.dword_0 != expect_0 || vf.dword_1 != expect_1) {
      ++g_nr_res_probe.layout_bad;
      REXGPU_ERROR("[nr-rsy] LAYOUT vertex_fetch {:08x} {:08x} expected {:08x} {:08x}",
                   vf.dword_0, vf.dword_1, expect_0, expect_1);
    }
  }
  {
    reg::VGT_DRAW_INITIATOR di;
    di.value = 0;
    di.prim_type = xenos::PrimitiveType(4);
    di.source_select = xenos::SourceSelect::kAutoIndex;
    di.index_size = xenos::IndexFormat::kInt32;
    di.num_indices = 0x1234;
    const uint32_t expect = 4u | 2u << 6 | 1u << 11 | 0x1234u << 16;
    if (di.value != expect) {
      ++g_nr_res_probe.layout_bad;
      REXGPU_ERROR("[nr-rsy] LAYOUT VGT_DRAW_INITIATOR {:08x} expected {:08x}", di.value,
                   expect);
    }
  }
  {
    reg::VGT_DMA_SIZE ds;
    ds.value = 0;
    ds.num_words = 0x123456;
    ds.swap_mode = xenos::Endian(1);
    const uint32_t expect = 0x123456u | 1u << 30;
    if (ds.value != expect) {
      ++g_nr_res_probe.layout_bad;
      REXGPU_ERROR("[nr-rsy] LAYOUT VGT_DMA_SIZE {:08x} expected {:08x}", ds.value, expect);
    }
  }
  {
    const uint32_t theirs = D3D12TextureCache::NrResPackSrvKey(true, 0xABC, 2);
    const uint32_t expect = nr::ResSrvDescriptorKey(true, 0xABC, 2);
    if (theirs != expect) {
      ++g_nr_res_probe.layout_bad;
      REXGPU_ERROR("[nr-rsy] LAYOUT SRVDescriptorKey {:08x} expected {:08x}", theirs, expect);
    }
  }
  for (uint32_t fetch_dim = 0; fetch_dim < 4; ++fetch_dim) {
    for (uint32_t data_dim = 0; data_dim < 4; ++data_dim) {
      const bool theirs = D3D12TextureCache::NrResDimensionsCompatible(fetch_dim, data_dim);
      if (nr::ResDimensionsCompatible(fetch_dim, data_dim) != theirs) {
        ++g_nr_res_probe.layout_bad;
        REXGPU_ERROR("[nr-rsy] LAYOUT DimensionsCompatible({}, {}) diverges", fetch_dim,
                     data_dim);
      }
    }
  }
}

// [NR-RSY] The 1Hz verdict line, emitted from the per-frame probe block.
void NrResReportIfDue() {
  if (!g_nr_res || !(g_nr_res_probe.vf_draws + g_nr_res_probe.ib_checks)) {
    return;
  }
  static auto s_last = std::chrono::steady_clock::now();
  static NrResProbe s_prev{};
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) {
    return;
  }
  s_last = now;
  const NrResProbe& p = g_nr_res_probe;
  REXGPU_INFO(
      "[nr-rsy] vf draws={} req={} ne={} state_ne={} abort={} fail={} ovf={} | "
      "ib checks={} match={} ne={} conv={} auto={} unexp={} | mx={} | pool "
      "alloc={} oneuse={} rel={} ne={} refuse={} reseeds={} | tmap match={} "
      "fresh={} seeded={} inv={} ne={} evict={} ovf={} | srv checks={} ne={} "
      "null={} spec={} unk={} | layout_bad={}",
      p.vf_draws - s_prev.vf_draws, p.vf_req - s_prev.vf_req, p.vf_ne,
      p.vf_state_ne, p.vf_abort_type, p.vf_req_fail, p.vf_pred_ovf,
      p.ib_checks - s_prev.ib_checks, p.ib_match - s_prev.ib_match, p.ib_ne,
      p.ib_conv - s_prev.ib_conv, p.ib_auto - s_prev.ib_auto, p.ib_unexpected,
      p.mx_draws, p.pool_alloc - s_prev.pool_alloc,
      p.pool_oneuse - s_prev.pool_oneuse, p.pool_release - s_prev.pool_release,
      p.pool_ne, p.pool_refused, p.pool_reseeds,
      p.tmap_match - s_prev.tmap_match, p.tmap_fresh, p.tmap_seeded,
      p.tmap_invalid, p.tmap_ne, p.tmap_evict, p.tmap_ovf,
      p.srv_checks - s_prev.srv_checks, p.srv_ne,
      p.srv_null - s_prev.srv_null, p.srv_special, p.srv_unknown,
      p.layout_bad);
  s_prev = p;
  g_nr_res_samples_this_window = 0;
}

// [NR-SWP] Phase 5-3b swap: latched once a frame; counters cmd-proc-thread-only.
// swapped = draws whose bindings OUR UpdateBindings assembled; fallback =
// eligible-mode draws refused mid-way (sampler-heap overflow - the emulated
// path then redoes the work on the same coherent state machine); srv_query =
// per-VALUE fallbacks to the emulated warm query inside our descriptor-indices
// compose (special view / unmapped texture - counted, never silent).
bool g_nr_swap = false;
struct NrSwapProbe {
  uint64_t swapped = 0, fallback = 0, srv_query = 0;
  uint64_t di_v = 0, di_p = 0;  // descriptor-indices rebuilds ours performed
  uint64_t smp_alloc = 0;       // sampler-heap slots OUR resolution allocated
  // [NR-LEAN] 5-4-4b inc 2b: draws whose emulated sysconst derivation was
  // skipped (mirror-only path) / fallback re-syncs of the member by memcpy.
  uint64_t sys_lean = 0, sys_lazy = 0;
};
NrSwapProbe g_nr_swap_probe{};

// [NR-VERIFY] Phase 5-4-4a inc 2: d3d12-side latch of gpu_nr_verify (per
// frame, like the probe latches below). OFF = perf config: the compare
// passes are skipped while every mirror a swap consumes keeps updating (the
// sysconst derivation, the texture-descriptor map hooks, the shared sampler
// allocator). A rising edge re-seeds the mirrors that drift while their
// compare/finish passes are off (vfetch + view pool + sysconst), so the
// gates re-arm honestly instead of reporting stale-mirror mismatches.
bool g_nr_verify = true;

// [NR-LEAN] 5-4-4b inc 2b: latched once a frame (requires the swap armed AND
// verify off). g_nr_sys_member_stale marks that the lean path skipped the
// emulated derivation since system_constants_ was last written; every
// consumer of the member re-syncs it from the mirror (byte-proven equivalent,
// sticky fields included) with one memcpy before reading.
bool g_nr_lean_sys = false;
bool g_nr_sys_member_stale = false;
// [NR-TILSYS] set by every derivation: did it change the mirror? The tile
// recorder stores it per draw.
bool g_nr_sys_changed = false;

// [NR-SWP] The 1Hz verdict line.
void NrSwapReportIfDue() {
  if (!g_nr_swap || !(g_nr_swap_probe.swapped + g_nr_swap_probe.fallback)) {
    return;
  }
  static auto s_last = std::chrono::steady_clock::now();
  static NrSwapProbe s_prev{};
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) {
    return;
  }
  s_last = now;
  const NrSwapProbe& p = g_nr_swap_probe;
  REXGPU_INFO(
      "[nr-swp] swapped={} fallback={} srv_query={} | di v={} p={} smp_alloc={} "
      "| lean={} lazy={}",
      p.swapped - s_prev.swapped, p.fallback, p.srv_query, p.di_v - s_prev.di_v,
      p.di_p - s_prev.di_p, p.smp_alloc, p.sys_lean - s_prev.sys_lean,
      p.sys_lazy);
  s_prev = p;
}

// [NR-RUB] Phase 5-4-5-1: the bundle store (frame-scoped: cleared when
// frame_current_ advances -- the city verdict says reuse is within-frame
// only, and a frame-scoped store bounds memory to one frame's distinct
// draws) + persistent staging mirrors of the current effective packs (the
// composes write upload memory directly; comparing requires CPU-side copies,
// [[upload-heap-readback-trap]]).
bool g_nr_rub = false;        // any bundle machinery (compare OR fast)
bool g_nr_rub_cmp = false;    // compare mode: staging + byte captures + gate
bool g_nr_rub_fast = false;   // [NR-RUF] 5-4-5-2: restore instead of derive
bool g_nr_ruf_v2b = false;    // [NR-RUF-V2B] 5-4-5-2b: stale-only upgrades
// [NR-RUF-V2B] this draw's upgrade verdict (computed once per draw in
// IssueDrawImpl before the first consumer; false unless an upgrade held).
bool g_ruf_v2b_up = false;
// City miss2 stale sets average 3-4 registers; a set bigger than this is
// refused (NrRuseStaleRegs reports true size past the cap).
constexpr uint32_t kNrRufV2bMaxStale = 32;
bool g_rub_stage_ok = false;  // staging coherent since arm (fallback clears)
struct NrRubBundle {
  uint32_t flt_v_bytes = 0, flt_p_bytes = 0;
  std::vector<uint8_t> flt_v, flt_p, bl, ftc;
  std::vector<uint32_t> di_v, di_p;
  std::vector<uint64_t> smp_v, smp_p;
  std::vector<uint32_t> si_v, si_p;
  uint64_t sys_hash = 0;
  void* rootsig = nullptr;
  // [NR-RUF] 5-4-5-2 restore state: valid same-frame only (the pool ring
  // recycles per frame; the city verdict says reuse IS within-frame).
  // packs_valid = the SMALL restore state is captured; the pack/di byte
  // copies above exist only for the compare gate (packs_have_bytes) -- the
  // fast path restores by ADDRESS and must never pay for them (the byte
  // captures were the measured net-negative of the first city A/B).
  bool packs_have_bytes = false;
  bool packs_valid = false;
  uint64_t frame = 0;
  D3D12_GPU_VIRTUAL_ADDRESS a_flt_v = 0, a_flt_p = 0, a_bl = 0, a_ftc = 0;
  D3D12_GPU_VIRTUAL_ADDRESS a_di_v = 0, a_di_p = 0;
  size_t smp_uid_v = 0, smp_uid_p = 0, tex_uid_v = 0, tex_uid_p = 0;
  std::vector<D3D12TextureCache::TextureSRVKey> keys_v, keys_p;
  // pso identity capture (compare only for now; the bypass is a later peel)
  bool pso_valid = false;
  void* pso_handle = nullptr;
  void* npso = nullptr;
};
// [NR-RUF] Direct-map find + bundle pool recycling: a hash-map find per
// draw (3-4 of them at city rates) was real cost, and the per-frame
// map.clear() destroyed and re-allocated every bundle's vectors each frame
// (~3.5k bundles x ~8 vectors at city). Slots invalidate by frame stamp
// (no per-frame sweep); the pool's bundles keep their vector capacity
// across frames (clear() only). A slot collision overwrites and the
// displaced key reads as no-bundle next time -- a re-derive, never a stale
// serve (same rule as the draw-record map).
constexpr uint32_t kRubMapBits = 20;
struct NrRubSlot {
  uint32_t key = 0;
  uint32_t idx = 0;
  uint64_t frame = ~0ull;
};
std::vector<NrRubSlot> g_rub_slots;   // sized 1<<kRubMapBits on first use
std::vector<NrRubBundle> g_rub_pool;  // capacity persists across frames
uint32_t g_rub_pool_used = 0;
uint64_t g_rub_frame = ~0ull;

inline NrRubBundle* NrRubFind(uint32_t key) {
  if (g_rub_slots.empty()) return nullptr;
  const NrRubSlot& s = g_rub_slots[(key >> 2) & ((1u << kRubMapBits) - 1)];
  if (s.key != key || s.frame != g_rub_frame) return nullptr;
  return &g_rub_pool[s.idx];
}

NrRubBundle* NrRubGetOrCreate(uint32_t key) {
  if (g_rub_slots.empty()) {
    g_rub_slots.resize(size_t(1) << kRubMapBits);
  }
  NrRubSlot& s = g_rub_slots[(key >> 2) & ((1u << kRubMapBits) - 1)];
  if (s.key == key && s.frame == g_rub_frame) return &g_rub_pool[s.idx];
  if (g_rub_pool_used == g_rub_pool.size()) {
    g_rub_pool.emplace_back();
  }
  NrRubBundle& b = g_rub_pool[g_rub_pool_used];
  // Recycle in place: vectors keep capacity, every validity flag resets.
  b.flt_v_bytes = b.flt_p_bytes = 0;
  b.flt_v.clear();
  b.flt_p.clear();
  b.bl.clear();
  b.ftc.clear();
  b.di_v.clear();
  b.di_p.clear();
  b.smp_v.clear();
  b.smp_p.clear();
  b.si_v.clear();
  b.si_p.clear();
  b.sys_hash = 0;
  b.rootsig = nullptr;
  b.packs_have_bytes = false;
  b.packs_valid = false;
  b.frame = 0;
  b.a_flt_v = b.a_flt_p = b.a_bl = b.a_ftc = b.a_di_v = b.a_di_p = 0;
  b.smp_uid_v = b.smp_uid_p = b.tex_uid_v = b.tex_uid_p = 0;
  b.keys_v.clear();
  b.keys_p.clear();
  b.pso_valid = false;
  b.pso_handle = nullptr;
  b.npso = nullptr;
  s.key = key;
  s.idx = g_rub_pool_used;
  s.frame = g_rub_frame;
  ++g_rub_pool_used;
  return &b;
}

// Frame advance: stale slots die by frame stamp, bundles are recycled.
inline void NrRubFrameReset(uint64_t frame) {
  g_rub_pool_used = 0;
  g_rub_frame = frame;
}

// Full disarm: release everything.
void NrRubRelease() {
  g_rub_slots.clear();
  g_rub_slots.shrink_to_fit();
  g_rub_pool.clear();
  g_rub_pool.shrink_to_fit();
  g_rub_pool_used = 0;
  g_rub_frame = ~0ull;
}
struct NrRubStage {
  std::vector<uint8_t> flt_v, flt_p, bl, ftc;
  uint32_t flt_v_bytes = 0, flt_p_bytes = 0;
  std::vector<uint32_t> di_v, di_p;
} g_rub_stage;
struct NrRubProbe {
  uint64_t draws = 0, checked = 0, captured = 0, nobundle = 0, nostop = 0,
           stage_off = 0;
  uint64_t ne_fltv = 0, ne_fltp = 0, ne_bl = 0, ne_ftc = 0;
  uint64_t ne_div = 0, ne_dip = 0, ne_smpv = 0, ne_smpp = 0, ne_siv = 0,
           ne_sip = 0, ne_rootsig = 0;
  uint64_t sys_eq = 0, sys_ne = 0;
  uint64_t pso_eq = 0, ne_pso = 0;      // pso-handle identity (IssueDrawImpl)
  uint64_t fast = 0, fast_miss = 0;     // [NR-RUF] restores / eligible misses
  uint64_t fast_pso = 0;                // [NR-RUF] ConfigurePipeline bypasses
  uint64_t v2b_up = 0, v2b_ref = 0;     // [NR-RUF-V2B] upgrades / refusals
} g_rub_probe;

// [NR-DSP] Phase 5-4-7-0: per-draw native span store + verdict counters.
// One slot per draw key (direct map, collisions overwrite and read as
// first-seen, never as stale). A slot holds the whole span inline: the city
// emits ~21 elements per draw, so 64 covers it with room and anything longer
// simply is not stored (counted).
// ⚠ The key is a packet ADDRESS, so its low bits alias hard across buffers
// (a plain key>>2 index collided 1,426/s at menu, silently costing coverage
// -- collisions read as first-seen, never as stale). Mix before indexing.
constexpr uint32_t kDspSlotBits = 17;
constexpr uint32_t kDspSlots = 1u << kDspSlotBits;
constexpr uint32_t kDspSlotElements = 64;
inline uint32_t NrDspSlotIndex(uint32_t key) {
  return uint32_t((uint64_t(key) * 2654435761ull) >> (32 - kDspSlotBits)) &
         (kDspSlots - 1);
}
struct DspSlot {
  uint32_t key = 0;
  uint32_t len = 0;
  uint8_t used = 0;
  uintmax_t data[kDspSlotElements];
};
std::vector<DspSlot> g_dsp_slots;
uint32_t g_dsp_key = 0;
bool g_dsp_reusable = false;
bool g_dsp_open = false;
size_t g_dsp_start = 0;
uint64_t g_dsp_gen = 0;
struct NrDspProbe {
  uint64_t draws = 0, compared = 0, eq = 0, eq_dyn = 0, real_ne = 0, len_ne = 0;
  uint64_t first_seen = 0, collision = 0, unstored = 0, reset = 0;
  uint64_t elements = 0, cmds = 0, view_sites = 0, dyn_view = 0, dyn_table = 0;
  uint64_t len_longer = 0, len_shorter = 0, len_delta = 0, len_and_real = 0;
  uint32_t first_real = 0xFFFFFFFFu;
};
NrDspProbe g_dsp_probe;

// [NR-SPR] Phase 5-4-7-1: the production span store. One slot per draw key
// (direct map, Fibonacci-mixed like the census -- packet addresses alias in
// the low bits); a slot holds the FIRST whitelist-clean context-free span
// this key emitted, plus its patch-site offsets, and is then FIXED: every
// later reusable execution compares fresh emission against that one
// recording, exactly the record-once/replay-many production shape (aging
// pool addresses must land in the eqdyn/patch class, never in ne).
// Collisions overwrite (read as first-seen, never as stale). 2^17 slots at
// 64 elements = ~73 MB, lazy -- store sizing is the known coverage lever
// (naruto_423 coll=23.7%), revisited when the swap increment owns a budget.
// [NR-SPD] 17 -> 18 (2026-08-17, first city consume A/B): at 2^17 the city
// evicted ~30k/s (coll ~= first_seen), losing ~20k/s of reusable draws their
// recording and burning ~30k/s of record traffic on refills. Payload is lazy
// (~196 MB virtual at 2^18); headers stay the hot 8B array (2 MB).
constexpr uint32_t kSprSlotBits = 18;
constexpr uint32_t kSprSlots = 1u << kSprSlotBits;
constexpr uint32_t kSprSlotElements = 64;
inline uint32_t NrSprSlotIndex(uint32_t key) {
  return uint32_t((uint64_t(key) * 2654435761ull) >> (32 - kSprSlotBits)) &
         (kSprSlots - 1);
}
// [NR-SPW v3] HOT/COLD SPLIT: the first city A/B pair read the swap 2-3 fps
// BELOW baseline while the replay path itself measured 0.78us -- the cost
// fit the STORE's DRAM traffic (every draw's Begin peeked a random ~856-byte
// slot across a ~112 MB array, and each record wrote ~1 KB cold). The
// 8-byte header array (1 MB, cache-warm) carries everything Begin's
// per-draw decision needs; the payload array is touched only at record and
// replay, which is compulsory traffic. Same index for both (identical
// semantics to the fused slot, only field placement moved).
struct SprHeader {
  uint32_t key = 0;
  uint8_t used = 0;
  uint8_t meta_valid = 0;
  uint8_t replay_worthy = 0;  // served a replay: a v2-miss re-records
                              // instead of invalidating, so keys that
                              // alternate reusable/miss keep replaying
                              // (invalidate-on-miss halved city coverage
                              // 49%->27% on exactly that population)
  uint8_t hot = 0;  // [NR-SPD] key has proven a reusable execution -- the
                    // only population whose recordings can ever serve a
                    // replay, so the only one dedup mode spends store
                    // traffic on (the 5-4-8 stability lesson). Survives
                    // used=0 invalidation (key field kept); reset on
                    // collision eviction.
};
// [NR-SPD] 5-4-7-3: the emission-context snapshot. A DEDUPED span's content
// depends on the CP's dedupe members at emission time (a command is emitted
// only when it CHANGES), so a recording is replayable exactly when the
// current members equal the recording's entry snapshot -- and after a replay
// the members must become the recording's EXIT snapshot so the next draw's
// context keeps matching (clearing them, as the context-free mode does,
// would force a re-record on every following candidate = the period-2 tax
// again). memcmp-compared whole: field order chosen so there are no padding
// holes on x64.
struct SprCtx {
  const void* guest_pipeline = nullptr;
  const void* external_pipeline = nullptr;
  const void* root_signature = nullptr;
  uint32_t topology = 0;  // D3D_PRIMITIVE_TOPOLOGY
  uint32_t ru2d = 0;      // current_graphics_root_up_to_date_
  // Order: fetch, float_v, float_p, system, bool_loop, di_v, di_p.
  uint64_t cb_addr[7] = {};
  uint8_t cb_utd[7] = {};
  uint8_t shm_uav = 2;  // 0/1 = flavor, 2 = nullopt
};
static_assert(sizeof(SprCtx) == 96, "SprCtx must be padding-free (memcmp'd)");
struct SprPayload {
  uint32_t len = 0;
  uint8_t view_count = 0;
  uint8_t ib_dma = 0;      // 1 = replay must RequestRange the guest IB
  uint8_t index_endian = 0;
  uint8_t pad = 0;
  uint16_t view_offsets[DeferredCommandList::kNrSprMaxViewSites];
  uint8_t view_roots[DeferredCommandList::kNrSprMaxViewSites] = {};
  const D3D12Shader* vs = nullptr;  // shader objects are never destroyed
  const D3D12Shader* ps = nullptr;
  uint32_t tex_mask = 0;
  uint32_t llci = 0;       // line loop closing index (pp result, reg-derived)
  uint32_t ib_base = 0;
  uint32_t ib_size = 0;
  uint64_t smp_heap_ptr = 0;  // record-time heap table bases: a recreated
  uint64_t view_heap_ptr = 0; // heap makes the recorded tables stale
  // [NR-SPD] dedup-mode extras: which roots the span re-establishes (the
  // coverage check's whitelist) + the entry/exit context snapshots.
  uint32_t root_mask = 0;
  SprCtx entry_ctx;
  SprCtx exit_ctx;
  uintmax_t data[kSprSlotElements];
};
std::vector<SprHeader> g_spr_headers;
std::vector<SprPayload> g_spr_payloads;
uint32_t g_spr_key = 0;
bool g_spr_reusable = false;
bool g_spr_open = false;
size_t g_spr_start = 0;
uint64_t g_spr_gen = 0;
// [NR-SPW] consume latch (per-frame, backend preconditions) + per-draw state.
bool g_nr_span_consume = false;
bool g_spr_forced = false;    // this bracket's fresh emission is context-free
bool g_spr_replayed = false;  // this bracket was served from the store
// [NR-SPD] 5-4-7-3 per-frame latch + per-draw state. In dedup mode nothing
// is ever forced; g_spr_record marks a fresh bracket whose span should be
// stored (reusable, or a key with reuse history -- misses of hot keys
// re-record for free, which is what keeps period-2 alternating keys
// replaying without the forced-re-emit tax).
bool g_nr_span_dedup = false;
bool g_spr_record = false;
SprCtx g_spr_entry_ctx;
// [NR-SPW] record-time capture of the IssueDrawImpl locals a replay's live
// head needs. Reset at bracket open, filled at the draw tail, consumed at
// store time in NrSprDrawEnd.
struct SprCapture {
  const D3D12Shader* vs = nullptr;
  const D3D12Shader* ps = nullptr;
  uint32_t tex_mask = 0;
  uint32_t llci = 0;
  uint32_t ib_base = 0;
  uint32_t ib_size = 0;
  uint8_t index_endian = 0;
  uint8_t ib_dma = 0;
  bool valid = false;    // the draw reached its emission tail
  bool refused = false;  // memexport / tessellated / non-kVertex / converted
                         // IB / instanced-batch start: never replayable
} g_spr_cap;
struct NrSprProbe {
  uint64_t draws = 0, compared = 0, eq = 0, eq_dyn = 0, real_ne = 0, len_ne = 0;
  uint64_t cmp_refused = 0;  // reusable + stored, but fresh span not clean
  uint64_t ctx_miss = 0;     // [NR-SPD] reusable + stored, entry context
                             // differs from the recording's snapshot (the
                             // dedup gate refusing -- re-recorded, not
                             // compared)
  uint64_t first_seen = 0, collision = 0, stored = 0, too_long = 0, empty = 0,
           reset = 0;
  uint64_t elements = 0, view_sites = 0, dyn_view = 0, dyn_table = 0;
  uint64_t ref_ff = 0, ref_bar = 0, ref_comp = 0, ref_heaps = 0, ref_other = 0;
  uint32_t first_real = 0xFFFFFFFFu;
};
NrSprProbe g_spr_probe;

// [NR-SPW] 5-4-7-2 swap accounting. cand = bracketed draws whose key held a
// replay-eligible recording at Begin; rep = actually replayed; the fb_*
// split names every fall-through to the full path. rep+fb == cand exactly.
struct NrSpwProbe {
  uint64_t cand = 0, rep = 0;
  uint64_t fb_batch = 0, fb_begin = 0, fb_bundle = 0, fb_heap = 0,
           fb_valve = 0, fb_rt = 0, fb_vf = 0, fb_ib = 0, fb_sys = 0;
  // [NR-SPD] dedup-mode fall-throughs: entry-context mismatch (the gate) +
  // post-head coverage refusal (a root the recording does not re-establish
  // went stale during the head).
  uint64_t fb_ctx = 0, fb_cover = 0;
  uint64_t rep_elements = 0, rep_patched = 0;
  // [NR-SPWP] stage split of the replay path, gpu_draw_profile-gated (the
  // clean-probe class: a handful of QPC stamps per REPLAYED draw only).
  // pre = lookups/begin/heap/bundle · tex = RequestTextures+valve ·
  // rt = RT-cache update · vp = viewport/scissor/UFFS · sys = mirror+upload
  // · rst = float-map+bundle restore · res = VB/IB residency+barriers ·
  // emit = memcpy+patch+member clears.
  uint64_t ns_pre = 0, ns_tex = 0, ns_rt = 0, ns_vp = 0, ns_sys = 0,
           ns_rst = 0, ns_res = 0, ns_emit = 0;
};
NrSpwProbe g_spw_probe;

void NrSpwReportIfDue() {
  if (!g_spw_probe.cand && !g_nr_span_consume) return;
  static auto s_last = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) return;
  s_last = now;
  const NrSpwProbe& p = g_spw_probe;
  const uint64_t fb = p.fb_batch + p.fb_begin + p.fb_bundle + p.fb_heap +
                      p.fb_valve + p.fb_rt + p.fb_vf + p.fb_ib + p.fb_sys +
                      p.fb_ctx + p.fb_cover;
  REXGPU_INFO(
      "[nr-spw] cand={} rep={} ({:.1f}%) el={:.1f} patch={:.2f} | fb={} "
      "(batch={} begin={} bundle={} heap={} valve={} rt={} vf={} ib={} "
      "sys={} ctx={} cover={})",
      p.cand, p.rep, p.cand ? 100.0 * double(p.rep) / double(p.cand) : 0.0,
      p.rep ? double(p.rep_elements) / double(p.rep) : 0.0,
      p.rep ? double(p.rep_patched) / double(p.rep) : 0.0, fb, p.fb_batch,
      p.fb_begin, p.fb_bundle, p.fb_heap, p.fb_valve, p.fb_rt, p.fb_vf,
      p.fb_ib, p.fb_sys, p.fb_ctx, p.fb_cover);
  if (g_draw_prof && p.rep) {
    const double r = double(p.rep) * 1000.0;  // ns -> us/draw
    REXGPU_INFO(
        "[nr-spwp] us/rep: pre={:.3f} tex={:.3f} rt={:.3f} vp={:.3f} "
        "sys={:.3f} rst={:.3f} res={:.3f} emit={:.3f} total={:.3f}",
        double(p.ns_pre) / r, double(p.ns_tex) / r, double(p.ns_rt) / r,
        double(p.ns_vp) / r, double(p.ns_sys) / r, double(p.ns_rst) / r,
        double(p.ns_res) / r, double(p.ns_emit) / r,
        double(p.ns_pre + p.ns_tex + p.ns_rt + p.ns_vp + p.ns_sys + p.ns_rst +
               p.ns_res + p.ns_emit) /
            r);
  }
  g_spw_probe = NrSpwProbe{};
}

void NrSprReportIfDue() {
  if (!g_spr_probe.draws) return;
  static auto s_last = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) return;
  s_last = now;
  const NrSprProbe& p = g_spr_probe;
  const double c = double(p.compared ? p.compared : 1);
  REXGPU_INFO(
      "[nr-spr] draws={} el={:.1f}/draw | cmp={} eq={} ({:.1f}%) eqdyn={} "
      "({:.1f}%) ne={} lenne={} real1st={} cmpref={} ctxmiss={} | "
      "replayable={:.1f}% | "
      "views={:.2f}/draw dynview={} dyntable={} | rec={} 1st={} coll={} "
      "long={} empty={} reset={} | refuse ff={} bar={} comp={} heaps={} "
      "other={}",
      p.draws, p.draws ? double(p.elements) / double(p.draws) : 0.0, p.compared,
      p.eq, 100.0 * double(p.eq) / c, p.eq_dyn, 100.0 * double(p.eq_dyn) / c,
      p.real_ne, p.len_ne,
      p.first_real == 0xFFFFFFFFu ? -1 : int32_t(p.first_real), p.cmp_refused,
      p.ctx_miss,
      100.0 * double(p.eq + p.eq_dyn) / c,
      p.compared ? double(p.view_sites) / c : 0.0, p.dyn_view, p.dyn_table,
      p.stored, p.first_seen, p.collision, p.too_long, p.empty, p.reset,
      p.ref_ff, p.ref_bar, p.ref_comp, p.ref_heaps, p.ref_other);
  g_spr_probe = NrSprProbe{};
}

// [NR-TIL] N-4-1 sizing. A heavy city base band is ~3.5k draws at ~43
// elements each (~1.2 MB of span), so the arena is generous but bounded;
// every cap refuses into the normal path and is counted, never silently.
constexpr uint32_t kTileMaxKeys = 16;           // texture bindings per stage
                                                // (8 refused ~10k draws/s in
                                                // the attract demo)
constexpr uint32_t kTileMaxSegs = 256;          // segments per frame (the
                                                // heavy city runs ~63
                                                // resolves a frame plus ~60
                                                // tiny shadow/cube passes,
                                                // all of them base-band)
constexpr uint32_t kTileMaxRecs = 32768;        // draws recorded per frame
constexpr uint32_t kTileMaxSpanElements = 512;  // elements in one draw's span
constexpr size_t kTileArenaMaxElements = 4u << 20;  // 32 MB
// ===========================================================================
// [NR-TIL] N-4-1: EDRAM TILE-PASS DRAW REPLAY.
//
// The heavy city renders the whole scene THREE times a frame, once per EDRAM
// tile band: the same draw packets, in the same order, with only
// PA_SC_WINDOW_OFFSET and the window scissor moved (0 / -256 / -512). The
// offline census (tools/n4-tile-census.py) reads 62.5% of all draw
// executions as repeat bands, and gpu_nr_tile_probe measured what removing
// their issue is worth: 36.9 -> 58 fps at the heavy city.
//
// This is the build the probe priced. Everything the CP derives for a repeat
// band's draw is bit-identical to the base band's - same shaders, same
// pipeline, same constants, same textures, same index buffer - because the
// same packets wrote the same registers. The ONLY band-dependent outputs are
// the viewport, the scissor and the system-constants NDC. So:
//   base band (window offset 0): record each draw's native command span,
//   anchored AFTER UpdateFixedFunctionState / UpdateSystemConstantValues so
//   the span holds no viewport or scissor set;
//   repeat band: recompute viewport / scissor / NDC live, then memcpy the
//   recorded span in and patch its system-constants root view.
// Pixel-identical by construction: same commands, same order, same state.
//
// Keyed by STREAM POSITION, not content. A segment (a run of draws between
// resolves, band changes and frames) records into a flat vector; the repeat
// band's segment finds its base by first-draw address and then walks the
// same ordinals. That is what makes the verdict free - N-2-2 measured a
// single hash-store lookup at 138 ns/span, which is more than the whole
// saving here.
//
// The per-draw head that remains (RT update, texture requests + key valve,
// viewport, sys upload, vertex/index residency) is the SAME head the 5-4-7-2
// span swap ran, and its stage profile (naruto_436/439/453 [nr-spwp]) prices
// it at ~0.9 us/draw against ~0.07 for the memcpy - so the head, not the
// emission, is what the next increment has to hoist to the band. v1 keeps it
// per draw because it is the sound version.
struct TileRec {
  uint32_t addr = 0;         // stream position (guard)
  uint32_t prim = 0;         // primitive type (guard)
  uint32_t index_count = 0;  // (guard)
  uint32_t ib_base = 0;      // guest index base or 0 (guard)
  uint32_t off = 0;          // element offset into g_tile_arena
  uint32_t len = 0;          // span length in elements
  uint32_t tex_mask = 0;
  uint32_t llci = 0;
  uint32_t ibr_base = 0;  // DMA index residency range
  uint32_t ibr_size = 0;
  uint32_t root_mask = 0;  // roots the span itself re-establishes
  const D3D12Shader* vs = nullptr;
  const D3D12Shader* ps = nullptr;
  uint64_t smp_heap_ptr = 0;
  uint64_t view_heap_ptr = 0;
  uint16_t view_offsets[DeferredCommandList::kNrSprMaxViewSites] = {};
  uint8_t view_roots[DeferredCommandList::kNrSprMaxViewSites] = {};
  uint8_t view_count = 0;
  uint8_t ib_dma = 0;
  uint8_t index_endian = 0;
  uint8_t valid = 0;  // 0 = this ordinal must fall through (the ordinal is
                      // still held so the bands stay in lockstep)
  // The pipeline-bind piece. A draw emits its pipeline set BEFORE the
  // viewport and scissor, so the recording is two pieces with the
  // bin-dependent set between them; this one is at most one command.
  uint8_t lenA = 0;
  uintmax_t dataA[8] = {};
  uint8_t nkeys_v = 0;
  uint8_t nkeys_p = 0;
  // [NR-TILSYS] did the BASE band's derivation actually move the system
  // constants at this ordinal? In lockstep the repeat band's sequence of
  // values is the same one, so this bit is the repeat band's dirty verdict
  // too - and it replaces a 464-byte compare per draw with a load.
  uint8_t sys_dirty = 0;
  // [NR-TILTEX] the derived texture bindings for this draw's fetch slots,
  // in g_tile_texsnaps. count 0 = nothing recorded, replay keeps the full
  // request.
  uint8_t tex_count = 0;
  uint32_t tex_off = 0;
  D3D12TextureCache::TextureSRVKey keys_v[kTileMaxKeys];
  D3D12TextureCache::TextureSRVKey keys_p[kTileMaxKeys];
  // Filled only in the compare mode: the system constants as the base band
  // left them, so the gate can prove that nothing outside the three
  // viewport-derived fields moves between bands.
  nr::NrSysConstants sys_snap;
  SprCtx entry_ctx;
  SprCtx exit_ctx;
};
struct TileSeg {
  uint32_t addr0 = 0;  // first draw's stream position = the segment's name
  uint32_t first = 0;  // index of its first record
  uint32_t count = 0;
};
std::vector<TileRec> g_tile_recs;
std::vector<TileSeg> g_tile_segs;
std::vector<uintmax_t> g_tile_arena;
size_t g_tile_arena_used = 0;
// Per-frame / per-segment walking state.
bool g_nr_tile_on = false;   // per-frame latch
uint32_t g_tile_frame = UINT32_MAX;
uint32_t g_tile_win = 0;     // the current segment's PA_SC_WINDOW_OFFSET
bool g_tile_seg_open = false;
// Ordinal 0 of the current repeat segment maps to this ordinal of its base
// segment (nonzero only when the rescue below found the base band running
// the two passes as one segment).
uint32_t g_tile_ord_bias = 0;
uint32_t g_tile_seg = 0;     // index into g_tile_segs
uint32_t g_tile_mode = 0;    // 0 = pass through, 1 = record, 2 = replay
uint32_t g_tile_ord = 0;     // next ordinal in this segment
uint32_t g_tile_cur_ord = 0; // this draw's ordinal
// This draw's shape guard, latched at Begin (the replay runs before the
// derivation, so these three are all the identity available for free).
uint32_t g_tile_prim = 0, g_tile_icount = 0, g_tile_ibbase = 0;
// Record bracket.
bool g_tile_rec_open = false;
// [NR-TIL-DORMANT] 2026-09-01 audit (naruto_754/755): under de-tile the
// repeat bands are dropped BELOW IssueDraw, so every draw here is a base
// band, the recorder records ~147k draws/s and the replay serves ZERO
// (ret tile=0, [gpu-draw2] tilr=27 ms/s + the ctx capture in head + the
// emission-tail capture, all for nothing). The latch: a frame that follows
// a frame with ZERO repeat-band draws takes a two-branch fast path in
// BeginDraw - no seg bookkeeping, no ctx capture, no record - and only
// keeps counting bands. The first repeat-band draw that appears (de-tile
// declined or disarmed) falls through to the FULL derivation (the designed
// refusal path, correct by construction) and wakes recording on the next
// frame boundary, restoring the +6.7% replay regime while bands persist.
// Never dormant under mode-3 verify (its compare population must be whole).
bool g_tile_dormant = false;
uint64_t g_tile_frame_rep0 = 0;  // p.repeat at the last frame boundary
// [NR-TIL] mode 3: compare instead of consume.
bool g_tile_verify = false;
// [NR-TILSYS] N-4-2b piece 1: 0 = off, 1 = serve the system constants from the
// base band snapshot, 2 = do that into a temp and compare it against the real
// derivation (the gate).
uint32_t g_tile_sys = 0;
// [NR-TILSYS] lockstep witness: the record whose (snapshot + live NDC) the
// mirror currently holds, and the NDC that was patched into it. When the next
// replay is the very next ordinal and neither moved, the mirror is ALREADY
// what that draw needs and the restore is a no-op.
// [NR-TILTEX] the per-frame binding snapshots the records index into.
std::vector<TextureCache::NrTexSnap> g_tile_texsnaps;
// A draw with more fetch slots than this records nothing and keeps the full
// request (counted, never silent).
constexpr uint32_t kTileTexMaxSlots = 12;
uint32_t g_tile_tex = 0;
size_t g_tile_sys_prev = SIZE_MAX;  // index into g_tile_recs, +1 biased
float g_tile_sys_ndc[8] = {};
bool g_tile_cmp_open = false;
uint32_t g_tile_cmp_idx = 0;
size_t g_tile_spanA_start = 0;  // pipeline-bind piece
size_t g_tile_spanA_end = 0;
size_t g_tile_span_start = 0;   // bindings + index buffer + draw
uint64_t g_tile_span_gen = 0;
SprCtx g_tile_entry_ctx;
// The pipeline pair as it stood at the TOP of the draw -- the point where
// the replay makes its decision. A texture load between there and the
// pipeline bind would break the recording symmetry, so both sides check it.
void* g_tile_top_guest_pso = nullptr;
ID3D12PipelineState* g_tile_top_ext_pso = nullptr;

struct NrTilProbe {
  uint64_t draws = 0;        // draws seen under the latch
  uint64_t base = 0;         // base-band draws (window offset 0)
  uint64_t repeat = 0;       // repeat-band draws
  uint64_t dormant = 0;      // draws that took the dormant fast path
  uint64_t recorded = 0;     // spans stored
  uint64_t cand = 0;         // repeat-band draws with a record at their ordinal
  uint64_t rep = 0;          // actually replayed
  uint64_t segs_rec = 0, segs_rep = 0, seg_nomatch = 0;
  // [NR-TIL] The segment-level population, which the per-draw fb_ counters
  // cannot see: a repeat draw whose SEGMENT never matched a base one never
  // becomes a candidate at all. nm_draws is how many draws that costs;
  // nm_empty / nm_none split the cause (a base segment with this name that
  // recorded no draw vs no base segment with this name), and seg_cap is the
  // push that the per-frame segment table refused.
  uint64_t nm_draws = 0, nm_empty = 0, nm_none = 0, seg_cap = 0;
  // The ordinal-bias rescue: the base band did not BREAK where this band
  // does, so the repeat segment's first draw sits INSIDE a longer base
  // segment. Found by address among the base records (unique = usable).
  uint64_t nm_bias = 0, nm_ambig = 0, nm_absent = 0;
  // Refusals, one counter per cause: a probe whose refusals are not counters
  // can go silent for the wrong reason.
  uint64_t fb_ord = 0, fb_norec = 0, fb_shape = 0, fb_batch = 0, fb_begin = 0,
           fb_texpso = 0,
           fb_heap = 0, fb_ctx = 0, fb_valve = 0, fb_rt = 0, fb_sys = 0,
           fb_vf = 0, fb_ib = 0;
  // Record-side refusals.
  uint64_t rr_gen = 0, rr_scan = 0, rr_class = 0, rr_len = 0, rr_arena = 0, rr_pso = 0,
           rr_keys = 0, rr_roots = 0, rr_cap = 0;
  uint64_t rep_elements = 0, sys_sets = 0;
  // Mode 3 compare gate.
  uint64_t cmp = 0, cmp_eq = 0, cmp_real = 0, cmp_dyn = 0, cmp_lenne = 0,
           cmp_psone = 0, cmp_ctxne = 0, cmp_sysne = 0, cmp_fbar = 0, cmp_fupl = 0;
  uint32_t cmp_first_real = 0xFFFFFFFFu;
  uint32_t cmp_sys_off = 0xFFFFFFFFu;
  uint64_t cmp_stale = 0;  // armed but the anchor or the frame moved
  // Why the entry gate refused: one bit per SprCtx field group (pipeline,
  // root signature, topology, root-up-to-date mask, shared-memory flavor,
  // then one per cbuffer up-to-date flag). A gate that only counts its
  // refusals cannot be fixed; this one names them.
  uint64_t ctx_why[12] = {};
  uint64_t ctx_first = 0;  // refusals on a segment first ordinal (expected)
  // [NR-TILP] N-4-2 brief: the stage split of the REPLAY HEAD, the thing
  // N-4-1 measured at ~1.44 of the ~1.75 us a repeat draw costs. Only the
  // 0.31 us derivation is gone; every stage below still runs per draw.
  // gpu_draw_profile-gated (a handful of QPC stamps per REPLAYED draw), and
  // the numbers are an ATTRIBUTION, not an fps read -- the stamps themselves
  // are ~3-4% at city replay rates.
  // pre = ordinal/shape/batch/begin/heap/ctx gates - tex = RequestTextures +
  // the SRV key valve - rt = render-target cache Update - vp = viewport key,
  // scissor and UpdateFixedFunctionState - sys = UpdateSystemConstantValues
  // + the pool upload - res = vfetch residency, IB range, UseForReading and
  // SubmitBarriers - emit = the memcpy, the view patch and member adoption.
  uint64_t ns_pre = 0, ns_tex = 0, ns_rt = 0, ns_vp = 0, ns_sys = 0,
           ns_res = 0, ns_emit = 0;
  // tex splits into the request (binding derivation, loads, barriers) and
  // the SRV-key valve that follows it - two different things to remove.
  uint64_t ns_treq = 0;
  uint64_t prof_rep = 0;  // replays that carried a full set of stamps
  // [NR-TILP] N-4-2 OUTER stamps, per replayed draw: ostop = direct drawstop
  // bracket entry -> tile block (arg recovery + NrSubmitDraw's register-file
  // repoint + IssueDrawImpl's prologue), obegin = NrTileBeginDraw itself.
  // opost (replay return -> bracket exit: the five restore SetRegisterFile
  // calls + the direct-issue tail) accumulates in the shared globals because
  // the bracket exit lives in the base TU.
  uint64_t ns_ostop = 0, ns_obegin = 0;
  uint64_t prof_orep = 0;  // replays that carried outer stamps
  // [NR-TILSYS] the snapshot-restore path: n = draws served, ne = draws whose
  // fast result differed from the real derivation (mode 2 only), off = first
  // differing byte, dne = draws where the fast dirty verdict differed from
  // the real one, dirty = draws that cleared cbuffer up_to_date, rov = draws
  // refused because the ROV path is live.
  uint64_t sf_n = 0, sf_ne = 0, sf_dne = 0, sf_dirty = 0, sf_rov = 0;
  // Direction of a dirty-verdict disagreement. over = the fast path uploaded
  // when it did not have to (a wasted constant buffer). under = it did NOT
  // upload when the real derivation would have - that one is a wrong frame,
  // and it has to be zero.
  uint64_t sf_dover = 0, sf_dunder = 0;
  uint64_t sf_nolock = 0, sf_noop = 0;
  // [NR-TILTEX] tf_hit = bindings restored, tf_miss = the fetch constants
  // moved (or a watch fired) so the full request ran, tf_norec = the base
  // band recorded no snapshot, tf_cap = a draw with too many fetch slots,
  // tf_slots = slots recorded.
  uint64_t tf_hit = 0, tf_miss = 0, tf_norec = 0, tf_cap = 0, tf_slots = 0;
  uint64_t tf_ne = 0;  // gate: a restored binding differed from the derived one
  // Gate only: restored draws that still ran the (tautological) SRV-key
  // valve and passed it. It never refuses - a refusal shows up as fb_valve.
  uint64_t tf_valve = 0;
  uint32_t sf_off = 0xFFFFFFFFu;
};
NrTilProbe g_tile_p;
// [NR-TILP] outer-stamp pendings (CP thread only): measured before ReplayTry
// runs, committed only by its success path -- same refusal rule as the inner
// stamps.
bool g_tile_pend_valid = false;
uint64_t g_tile_pend_ostop = 0, g_tile_pend_obegin = 0;
// [NR-TIL] the tile entry gate compares everything that decides WHETHER a
// command is emitted, and nothing that only decides its VALUE. The cbuffer
// ADDRESSES are the difference from the 5-4-7-3 gate: the first draw of a
// repeat band runs the full path and gets FRESH pool allocations, so from
// ordinal 2 on the addresses can never match the base band. They do not have
// to: the base band buffer at the recorded address is same-frame, still
// live, and holds byte-identical content, because the same packets wrote the
// same registers. What must match is the up-to-date flags, the root mask,
// the pipeline / root-signature / topology identity and the shared-memory
// flavor -- those decide which commands the recording OMITTED.
// [NR-TILSYS] N-4-2b piece 1. A repeat-band draw derives the same system
// constants as its base band draw except for what the VIEWPORT feeds:
// ndc_scale, ndc_offset, and - for point lists only -
// point_screen_diameter_to_ndc_radius, which divides the resolution scale by
// the viewport extent. Everything else was proven byte-identical by the
// N-4-1 compare gate (100.0000% over ~1.2M city compares), because it comes
// from registers the two bands share.
//
// So: restore the recorded snapshot, then write those three live. The
// per-band sticky rule is preserved exactly - the real derivation writes the
// point radius ONLY on a point-list draw, so a non-point draw keeps the
// value already live rather than importing the base band's.
//
// Returns whether the resulting state differs from what *state held on
// entry, which is the caller's cbuffer up-to-date verdict. The compare is
// bytewise on purpose: these floats can legitimately hold NaNs and still
// have to compare bit-exactly (the same rule the mirror gate uses).
bool NrTileSysFast(nr::NrSysConstants* state, size_t rec_index,
                   const nr::NrSysConstants& snap, bool rec_dirty,
                   const draw_util::ViewportInfo& vp, bool point_list,
                   uint32_t res_scale_x, uint32_t res_scale_y,
                   bool track) {
  static_assert(offsetof(nr::NrSysConstants, ndc_scale) == 128 &&
                    offsetof(nr::NrSysConstants, ndc_offset) == 144 &&
                    offsetof(nr::NrSysConstants,
                             point_screen_diameter_to_ndc_radius) == 168,
                "[nr-tilsys] the viewport-derived window moved");
  // The live viewport half, 32 bytes: ndc_scale, ndc_offset and the point
  // radius (written by the real derivation only for point lists, so a
  // non-point draw keeps whatever is live - the sticky rule, exactly).
  float ndc[8];
  std::memcpy(ndc, vp.ndc_scale, 3 * sizeof(float));
  std::memcpy(ndc + 3, vp.ndc_offset, 3 * sizeof(float));
  if (point_list) {
    ndc[6] = float(res_scale_x) / float(vp.xy_extent[0] ? vp.xy_extent[0] : 1u);
    ndc[7] = float(res_scale_y) / float(vp.xy_extent[1] ? vp.xy_extent[1] : 1u);
  } else {
    ndc[6] = state->point_screen_diameter_to_ndc_radius[0];
    ndc[7] = state->point_screen_diameter_to_ndc_radius[1];
  }
  // LOCKSTEP: the mirror holds the previous ordinal's snapshot with this
  // band's NDC already in it. Then "did anything move" is a recorded bit
  // plus a 32-byte compare - not a 464-byte one - and when nothing moved
  // there is nothing to write at all.
  const bool lockstep = track && rec_index != 0 &&
                        g_tile_sys_prev == rec_index - 1;
  if (track && !lockstep) ++g_tile_p.sf_nolock;
  const bool ndc_same =
      std::memcmp(g_tile_sys_ndc, ndc, sizeof(ndc)) == 0;
  bool changed;
  if (lockstep && !rec_dirty) {
    // The base band did not move the struct at this ordinal either, so the
    // only thing that can differ is the viewport half.
    if (ndc_same) {
      changed = false;
      if (track) ++g_tile_p.sf_noop;
    } else {
      std::memcpy(state->ndc_scale, ndc, 3 * sizeof(float));
      std::memcpy(state->ndc_offset, ndc + 3, 3 * sizeof(float));
      std::memcpy(state->point_screen_diameter_to_ndc_radius, ndc + 6,
                  2 * sizeof(float));
      changed = true;
    }
  } else {
    std::memcpy(state, &snap, sizeof(nr::NrSysConstants));
    std::memcpy(state->ndc_scale, ndc, 3 * sizeof(float));
    std::memcpy(state->ndc_offset, ndc + 3, 3 * sizeof(float));
    std::memcpy(state->point_screen_diameter_to_ndc_radius, ndc + 6,
                2 * sizeof(float));
    // Out of lockstep the previous state is unknown, so the verdict is the
    // conservative one: an upload that was not needed costs a buffer, a
    // skipped one that was needed draws the wrong frame.
    changed = rec_dirty || !lockstep || !ndc_same;
  }
  if (track) {
    g_tile_sys_prev = rec_index;
    std::memcpy(g_tile_sys_ndc, ndc, sizeof(ndc));
  }
  return changed;
}

// [NR-DETILE] N-5: the DRAW half of the band-1 widening. Band 1 rasterises the
// whole frame so the two repeat bands become dead work.
//
// ⚠ This deliberately does NOT live in draw_util::GetScissor. GetResolveInfo
// calls GetScissor too and uses it as a CLAMP; city drive 1 widened it there
// and thereby removed the clamp that keeps a resolve inside its band, which
// broke every frame below row 256. Draw and resolve want opposite things from
// the same register, so they get two separate edits.
//
// Called with the scissor in GUEST pixels, before the resolution scaling.
void NrDetileWidenDrawScissor(const RegisterFile& regs, draw_util::Scissor& scissor) {
  const nr::DetileRegs dt{regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET],
                          regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL],
                          regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR],
                          regs[XE_GPU_REG_RB_SURFACE_INFO],
                          regs[XE_GPU_REG_RB_COLOR_INFO],
                          regs[XE_GPU_REG_RB_DEPTH_INFO]};
  const uint32_t full_height = nr::DetileWidenFullHeight(dt, true);

  // [NR-DETILE] N-6-2. The widen edits PA_SC_WINDOW_SCISSOR and nothing else,
  // but the host VIEWPORT rectangle scissors as well - GetHostViewportInfo
  // derives it from PA_CL_VPORT alone and the call site passes
  // D3D12_VIEWPORT_BOUNDS_MAX as y_max, so nothing clamps it afterwards. If
  // band 1's guest viewport is only 256 rows tall, no scissor can make it
  // rasterise the other 464 and the widening is vacuous.
  //
  // Gated to the packet it names: full_height is non-zero only for a band-1
  // draw of the tiled pass. Prints the guest's own viewport intent, the
  // window offset that the 360 tiling scheme uses to slide bands, and the
  // scissor on both sides of the edit.
  if (full_height && REXCVAR_GET(gpu_nr_detile_vp_probe)) {
    static auto s_vp_last = std::chrono::steady_clock::now();
    static uint64_t s_vp_seen = 0;
    static uint64_t s_vp_vp_short = 0;
    static uint64_t s_vp_sci_short = 0;
    ++s_vp_seen;
    const auto vte = regs.Get<reg::PA_CL_VTE_CNTL>();
    const auto su = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
    const float yscale =
        vte.vport_y_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE) : 1.0f;
    const float yoffset =
        vte.vport_y_offset_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET) : 0.0f;
    const auto win = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
    const float win_y =
        su.vtx_window_offset_enable ? float(win.window_y_offset) : 0.0f;
    // What GetHostViewportInfo will produce for Y, before resolution scaling.
    const float vp_y0f = yoffset + win_y - std::abs(yscale);
    const float vp_y1f = yoffset + win_y + std::abs(yscale);
    const uint32_t vp_y0 = uint32_t(rex::clamp_float(vp_y0f, 0.0f, 16384.0f));
    const uint32_t vp_y1 = uint32_t(rex::clamp_float(vp_y1f, 0.0f, 16384.0f));
    if (vp_y1 < full_height) ++s_vp_vp_short;
    if (scissor.offset[1] + scissor.extent[1] < full_height) ++s_vp_sci_short;
    const auto now = std::chrono::steady_clock::now();
    if (now - s_vp_last >= std::chrono::seconds(1)) {
      s_vp_last = now;
      REXGPU_INFO(
          "[nr-detile-vp] seg={} full_height={} | vport y: scale={:.1f} offset={:.1f} "
          "winy={:.1f} (ena s={} o={} w={}) -> host viewport rows {}..{} | scissor rows "
          "{}..{} | seen={} vp_short={} sci_short={}",
          nr::DetileBand0Taps(), full_height, yscale, yoffset, win_y,
          uint32_t(vte.vport_y_scale_ena), uint32_t(vte.vport_y_offset_ena),
          uint32_t(su.vtx_window_offset_enable), vp_y0, vp_y1, scissor.offset[1],
          scissor.offset[1] + scissor.extent[1],
          static_cast<unsigned long long>(s_vp_seen),
          static_cast<unsigned long long>(s_vp_vp_short),
          static_cast<unsigned long long>(s_vp_sci_short));
    }
  }

  if (!full_height || scissor.offset[1] + scissor.extent[1] >= full_height) {
    return;
  }
  scissor.extent[1] = full_height - scissor.offset[1];
  nr::DetileCountScissorWidened();
}

bool NrTileCtxEq(const SprCtx& a, const SprCtx& b, uint32_t* why) {
  uint32_t w = 0;
  if (a.guest_pipeline != b.guest_pipeline ||
      a.external_pipeline != b.external_pipeline) {
    w |= 1u << 0;
  }
  if (a.root_signature != b.root_signature) w |= 1u << 1;
  if (a.topology != b.topology) w |= 1u << 2;
  if (a.ru2d != b.ru2d) w |= 1u << 3;
  if (a.shm_uav != b.shm_uav) w |= 1u << 4;
  for (int i = 0; i < 7; ++i) {
    if (a.cb_utd[i] != b.cb_utd[i]) w |= 1u << (5 + i);
  }
  *why = w;
  return w == 0;
}


void NrTilReportIfDue() {
  if (!g_nr_tile_on && !g_tile_p.draws) return;
  static auto s_last = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) return;
  s_last = now;
  const NrTilProbe& p = g_tile_p;
  const uint64_t fb = p.fb_ord + p.fb_norec + p.fb_shape + p.fb_batch +
                      p.fb_begin + p.fb_heap + p.fb_ctx + p.fb_valve +
                      p.fb_rt + p.fb_sys + p.fb_vf + p.fb_ib + p.fb_texpso;
  REXGPU_INFO(
      "[nr-til] draws={} dorm={} base={} repeat={} | rec={} segs {}/{} nomatch={} "
      "(empty={} none={} draws={} = {:.1f}% of repeat, segcap={} | bias={} "
      "ambig={} absent={}) | "
      "cand={} rep={} ({:.1f}% of repeat) el={:.1f} syssets={} | fb={} "
      "(ord={} norec={} shape={} batch={} begin={} heap={} ctx={} valve={} "
      "rt={} sys={} vf={} ib={} texpso={}) | rrefuse gen={} scan={} class={} "
      "len={} "
      "arena={} keys={} roots={} pso={} cap={} | ctxwhy pso={} rs={} "
      "topo={} ru2d={} shm={} utd={}/{}/{}/{}/{}/{}/{} first={} | "
      "arena={}KB recs={}",
      p.draws, p.dormant, p.base, p.repeat, p.recorded, p.segs_rep, p.segs_rec,
      p.seg_nomatch, p.nm_empty, p.nm_none, p.nm_draws,
      p.repeat ? 100.0 * double(p.nm_draws) / double(p.repeat) : 0.0, p.seg_cap,
      p.nm_bias, p.nm_ambig, p.nm_absent, p.cand, p.rep,
      p.repeat ? 100.0 * double(p.rep) / double(p.repeat) : 0.0,
      p.rep ? double(p.rep_elements) / double(p.rep) : 0.0, p.sys_sets, fb,
      p.fb_ord, p.fb_norec, p.fb_shape, p.fb_batch, p.fb_begin, p.fb_heap,
      p.fb_ctx, p.fb_valve, p.fb_rt, p.fb_sys, p.fb_vf, p.fb_ib, p.fb_texpso,
      p.rr_gen,
      p.rr_scan, p.rr_class, p.rr_len, p.rr_arena, p.rr_keys, p.rr_roots,
      p.rr_pso, p.rr_cap, p.ctx_why[0], p.ctx_why[1], p.ctx_why[2],
      p.ctx_why[3], p.ctx_why[4], p.ctx_why[5], p.ctx_why[6], p.ctx_why[7],
      p.ctx_why[8], p.ctx_why[9], p.ctx_why[10], p.ctx_why[11], p.ctx_first,
      (g_tile_arena.size() * sizeof(uintmax_t)) >> 10,
      g_tile_recs.size());
  if (p.tf_hit || p.tf_miss || p.tf_norec) {
    // [NR-TILTEX] hit% is the coverage; ne has to be 0 (gate mode only).
    const uint64_t tf_all = p.tf_hit + p.tf_miss + p.tf_norec;
    REXGPU_INFO(
        "[nr-tiltex] hit={} ({:.1f}%) miss={} norec={} | ne={} valveok={} "
        "slots/rec={:.2f} cap={}",
        p.tf_hit, tf_all ? 100.0 * double(p.tf_hit) / double(tf_all) : 0.0,
        p.tf_miss, p.tf_norec, p.tf_ne, p.tf_valve,
        p.recorded ? double(p.tf_slots) / double(p.recorded) : 0.0, p.tf_cap);
  }
  if (p.sf_n || p.sf_rov) {
    // [NR-TILSYS] ne has to be 0: the fast system constants must equal the
    // real derivation byte for byte, and their dirty verdicts must agree.
    // dirty% is how often the restore actually moves the struct (i.e. how
    // much of the upload traffic is real).
    REXGPU_INFO(
        "[nr-tilsys] n={} ne={} (1st byte {}) dirtyne={} (over={} "
        "UNDER={}) | dirty={} ({:.1f}%) nolock={} noop={} ({:.1f}%) rov={}",
        p.sf_n, p.sf_ne, p.sf_off == 0xFFFFFFFFu ? -1 : int32_t(p.sf_off),
        p.sf_dne, p.sf_dover, p.sf_dunder, p.sf_dirty,
        p.sf_n ? 100.0 * double(p.sf_dirty) / double(p.sf_n) : 0.0,
        p.sf_nolock, p.sf_noop,
        p.sf_n ? 100.0 * double(p.sf_noop) / double(p.sf_n) : 0.0, p.sf_rov);
  }
  if (p.cmp || p.cmp_stale) {
    // The gate line. eq is the number that has to be 100%: a repeat-band
    // draw whose freshly derived span, pipeline piece and system constants
    // all match the base band recording outside the three viewport-derived
    // NDC fields. dyn counts root views that differ ONLY in their GPU
    // address, which is expected and is what the replay reuses.
    REXGPU_INFO(
        "[nr-til] GATE cmp={} eq={} ({:.4f}%) | ne: span={} lenne={} pso={} "
        "sys={} (1st byte {}) | dyn={} fbar={} fupl={} ctxne={} stale={} "
        "1stcmd={}",
        p.cmp, p.cmp_eq, p.cmp ? 100.0 * double(p.cmp_eq) / double(p.cmp) : 0.0,
        p.cmp_real, p.cmp_lenne, p.cmp_psone, p.cmp_sysne,
        p.cmp_sys_off == 0xFFFFFFFFu ? -1 : int32_t(p.cmp_sys_off), p.cmp_dyn,
        p.cmp_fbar, p.cmp_fupl, p.cmp_ctxne, p.cmp_stale,
        p.cmp_first_real == 0xFFFFFFFFu ? -1 : int32_t(p.cmp_first_real));
  }
  if (g_tile_prof && p.prof_rep) {
    // us per REPLAYED draw. head = everything before the memcpy: what N-4-2
    // has to hoist to the band.
    const double r = double(p.prof_rep) * 1000.0;
    const uint64_t head =
        p.ns_pre + p.ns_tex + p.ns_rt + p.ns_vp + p.ns_sys + p.ns_res;
    REXGPU_INFO(
        "[nr-tilp] us/rep n={}: pre={:.3f} tex={:.3f} (req={:.3f} "
        "valve={:.3f}) rt={:.3f} vp={:.3f} "
        "sys={:.3f} res={:.3f} | head={:.3f} emit={:.3f} total={:.3f}",
        p.prof_rep, double(p.ns_pre) / r, double(p.ns_tex) / r,
        double(p.ns_treq) / r, double(p.ns_tex - p.ns_treq) / r,
        double(p.ns_rt) / r, double(p.ns_vp) / r, double(p.ns_sys) / r,
        double(p.ns_res) / r, double(head) / r, double(p.ns_emit) / r,
        double(head + p.ns_emit) / r);
  }
  if (g_tile_prof && (p.prof_orep || g_tile_n_opost)) {
    // [NR-TILP] N-4-2 outer split, us per replayed draw: ostop = drawstop
    // bracket entry -> tile block (arg recovery, register-file repoint,
    // IssueDrawImpl prologue), obegin = NrTileBeginDraw, opost = replay
    // return -> bracket exit (restore SetRegisterFiles + direct-issue tail).
    // ostop+obegin+opost + the [nr-tilp] total should reproduce N-4-1's
    // 1.44 us/repeat-draw arithmetic in ONE run.
    const double ro =
        p.prof_orep ? double(p.prof_orep) * 1000.0 : 1.0;
    REXGPU_INFO(
        "[nr-tilo] us/rep n={}: ostop={:.3f} obegin={:.3f} | opost n={} "
        "={:.3f}",
        p.prof_orep, double(p.ns_ostop) / ro, double(p.ns_obegin) / ro,
        g_tile_n_opost,
        g_tile_n_opost
            ? double(g_tile_ns_opost) / (double(g_tile_n_opost) * 1000.0)
            : 0.0);
    g_tile_ns_opost = 0;
    g_tile_n_opost = 0;
  }
  g_tile_p = NrTilProbe{};
}

void NrDspReportIfDue() {
  if (!g_dsp_probe.draws) return;
  static auto s_last = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) return;
  s_last = now;
  const NrDspProbe& p = g_dsp_probe;
  const double c = double(p.compared ? p.compared : 1);
  REXGPU_INFO(
      "[nr-dsp] draws={} el={:.1f}/draw | compared={} eq={} ({:.1f}%) "
      "eqdyn={} ({:.1f}%) ne={} lenne={} (long={} short={} avg{:.1f}el "
      "real={}) | replayable={:.1f}% | patch views={:.2f}/draw (dynview={} "
      "dyntable={}) | 1st={} coll={} unstored={} reset={} real1st={}",
      p.draws, p.draws ? double(p.elements) / double(p.draws) : 0.0, p.compared,
      p.eq, 100.0 * double(p.eq) / c, p.eq_dyn, 100.0 * double(p.eq_dyn) / c,
      p.real_ne, p.len_ne, p.len_longer, p.len_shorter,
      p.len_ne ? double(p.len_delta) / double(p.len_ne) : 0.0, p.len_and_real,
      100.0 * double(p.eq + p.eq_dyn) / c,
      p.compared ? double(p.view_sites) / c : 0.0, p.dyn_view, p.dyn_table,
      p.first_seen, p.collision, p.unstored, p.reset,
      p.first_real == 0xFFFFFFFFu ? -1 : int32_t(p.first_real));
  g_dsp_probe = NrDspProbe{};
}

void NrRubReportIfDue() {
  if (!g_nr_rub || !g_rub_probe.draws) return;
  static auto s_last = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) return;
  s_last = now;
  const NrRubProbe& p = g_rub_probe;
  REXGPU_INFO(
      "[nr-rub] draws={} chk={} cap={} nobundle={} nostop={} stageoff={} | "
      "ne fv/fp/bl/ft={}/{}/{}/{} di={}/{} smp={}/{} si={}/{} rs={} pso={} "
      "(eq={}) | sys eq={} ne={} | fast={} fastmiss={} fastpso={} "
      "v2b={}/{} | bundles={}",
      p.draws, p.checked, p.captured, p.nobundle, p.nostop, p.stage_off,
      p.ne_fltv, p.ne_fltp, p.ne_bl, p.ne_ftc, p.ne_div, p.ne_dip, p.ne_smpv,
      p.ne_smpp, p.ne_siv, p.ne_sip, p.ne_rootsig, p.ne_pso, p.pso_eq,
      p.sys_eq, p.sys_ne, p.fast, p.fast_miss, p.fast_pso, p.v2b_up,
      p.v2b_ref, g_rub_pool_used);
  g_rub_probe = NrRubProbe{};
}

// [NR-FX] Phase 5-4-0: walk-driven side-effect counters (CP thread only).
// Window deltas per class; the classes are the same three constant ranges the
// WriteRegister tail dispatches on. A zero line with the cvar on means the
// walk decoded no constant writes (menu idle), not a broken hook -- the
// executor's own write rates on the other [nr-*] lines say which.
struct NrFxProbe {
  uint64_t fl = 0, bl = 0, fetch = 0;
};
NrFxProbe g_nr_fx_probe{};

// [NR-FX] The 1Hz line.
void NrFxReportIfDue() {
  const NrFxProbe& p = g_nr_fx_probe;
  if (!(p.fl | p.bl | p.fetch)) {
    return;
  }
  static auto s_last = std::chrono::steady_clock::now();
  static NrFxProbe s_prev{};
  const auto now = std::chrono::steady_clock::now();
  if (now - s_last < std::chrono::seconds(1)) {
    return;
  }
  s_last = now;
  REXGPU_INFO("[nr-fx] float={} bl={} fetch={}", p.fl - s_prev.fl,
              p.bl - s_prev.bl, p.fetch - s_prev.fetch);
  s_prev = p;
}

// [NR-RSY] The primitive processor's index-request observer (installed while
// the gate is armed; the CP thread is the only caller of Process here).
void NrResIndexObserverThunk(void*, uint32_t base, uint32_t length, bool result) {
  g_nr_res_ib_seen = true;
  g_nr_res_ib_base = base;
  g_nr_res_ib_len = length;
  g_nr_res_ib_result = result;
}

// Per-{geometry+shader} batch key: how many draws share it, and which float
// constant registers ever differ across those draws (= the per-instance data a
// GPU-instanced draw would have to stream). A clean instancing candidate has a
// small CONTIGUOUS vertex-const delta (a transform matrix) and NO pixel delta.
struct InstKeyInfo {
  uint64_t count = 0;
  bool have_first = false;
  uint32_t prim = 0, idx = 0, vtx = 0, ib_base = 0, ib_cnt = 0, vb0 = 0;
  int ib_fmt = 0;
  uint64_t vs_hash = 0, ps_hash = 0;
  // Used-constant bitmaps copied from the shaders (only diff what's actually read).
  uint64_t vs_used[4] = {0, 0, 0, 0};
  uint64_t ps_used[4] = {0, 0, 0, 0};
  // First-draw snapshot of each used vec4 (256 vertex + 256 pixel float consts).
  float vs_first[256][4];
  float ps_first[256][4];
  // Which vec4 consts ever differed (bit-exact) from the first draw of this key.
  uint64_t vs_diff[4] = {0, 0, 0, 0};
  uint64_t ps_diff[4] = {0, 0, 0, 0};
};
std::unordered_map<uint64_t, InstKeyInfo> g_inst_keys;
constexpr size_t kInstMaxKeys = 8192;  // safety cap on distinct keys tracked

inline uint64_t inst_mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  return h;
}

// Decode a 256-bit (4x uint64) "varying vec4" mask into a compact list, collapsing
// contiguous runs to "c<lo>..c<hi>" so a 4-row transform reads as one range.
std::string inst_decode_mask(const uint64_t mask[4]) {
  std::vector<uint32_t> idxs;
  for (uint32_t w = 0; w < 4; ++w) {
    uint64_t bits = mask[w];
    uint32_t b;
    while (rex::bit_scan_forward(bits, &b)) {
      bits &= ~(1ull << b);
      idxs.push_back(w * 64 + b);
    }
  }
  if (idxs.empty()) return "(none)";
  std::string out;
  size_t i = 0;
  while (i < idxs.size()) {
    size_t j = i;
    while (j + 1 < idxs.size() && idxs[j + 1] == idxs[j] + 1) ++j;
    if (!out.empty()) out += ",";
    out += "c" + std::to_string(idxs[i]);
    if (j > i) out += ".." + ("c" + std::to_string(idxs[j]));
    i = j + 1;
  }
  return out;
}

// The batch key itself, shared by the one-shot inst-probe and the continuous
// [BATCH-CENSUS]: {prim, counts, VS/PS ucode hashes, IB identity, bound-VB
// hash}. Draws with equal keys are the same geometry through the same shaders;
// only constants (the per-instance transform) can differ.
uint64_t InstBatchKey(const RegisterFile& regs, const Shader* vs, const Shader* ps,
                      uint32_t prim, uint32_t index_count, uint32_t host_vtx,
                      uint32_t ib_base, int ib_fmt, uint32_t ib_cnt, uint32_t* vb0_out) {
  const Shader::ConstantRegisterMap& vcm = vs->constant_register_map();
  // Stable hash over the bound vertex buffers (+ a label VB addr).
  uint64_t vbhash = 0;
  uint32_t vb0 = 0;
  for (uint32_t i = 0; i < rex::countof(vcm.vertex_fetch_bitmap); ++i) {
    uint32_t bits = vcm.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(bits, &j)) {
      bits &= ~(uint32_t(1) << j);
      const xenos::xe_gpu_vertex_fetch_t vf = regs.GetVertexFetch(i * 32 + j);
      const uint32_t addr = uint32_t(vf.address) << 2;
      if (vb0 == 0) vb0 = addr;
      vbhash = inst_mix(vbhash, (uint64_t(addr) << 32) | (uint32_t(vf.size) << 2));
    }
  }
  if (vb0_out) *vb0_out = vb0;
  const uint64_t vs_hash = vs->ucode_data_hash();
  const uint64_t ps_hash = ps ? ps->ucode_data_hash() : 0;
  uint64_t key = 0;
  key = inst_mix(key, prim);
  key = inst_mix(key, index_count);
  key = inst_mix(key, host_vtx);
  key = inst_mix(key, vs_hash);
  key = inst_mix(key, ps_hash);
  key = inst_mix(key, (uint64_t(ib_base) << 32) ^ (uint64_t(uint32_t(ib_fmt)) << 16) ^ ib_cnt);
  key = inst_mix(key, vbhash);
  return key;
}

// Per-draw accumulation: compute this draw's batch key, snapshot/diff its used
// float constants against the first draw seen for that key.
void InstanceProbeDraw(const RegisterFile& regs, const Shader* vs, const Shader* ps,
                       uint32_t prim, uint32_t index_count, uint32_t host_vtx,
                       uint32_t ib_base, int ib_fmt, uint32_t ib_cnt) {
  if (vs == nullptr) return;
  const Shader::ConstantRegisterMap& vcm = vs->constant_register_map();
  uint32_t vb0 = 0;
  const uint64_t key = InstBatchKey(regs, vs, ps, prim, index_count, host_vtx, ib_base, ib_fmt,
                                    ib_cnt, &vb0);
  const uint64_t vs_hash = vs->ucode_data_hash();
  const uint64_t ps_hash = ps ? ps->ucode_data_hash() : 0;

  auto it = g_inst_keys.find(key);
  if (it == g_inst_keys.end()) {
    if (g_inst_keys.size() >= kInstMaxKeys) return;
    it = g_inst_keys.emplace(key, InstKeyInfo{}).first;
  }
  InstKeyInfo& ki = it->second;
  ki.count++;
  if (!ki.have_first) {
    ki.have_first = true;
    ki.prim = prim; ki.idx = index_count; ki.vtx = host_vtx;
    ki.ib_base = ib_base; ki.ib_fmt = ib_fmt; ki.ib_cnt = ib_cnt; ki.vb0 = vb0;
    ki.vs_hash = vs_hash; ki.ps_hash = ps_hash;
    std::memcpy(ki.vs_used, vcm.float_bitmap, sizeof(ki.vs_used));
    if (ps) {
      std::memcpy(ki.ps_used, ps->constant_register_map().float_bitmap, sizeof(ki.ps_used));
    }
    for (uint32_t w = 0; w < 4; ++w) {
      uint64_t e = ki.vs_used[w];
      uint32_t b;
      while (rex::bit_scan_forward(e, &b)) {
        e &= ~(1ull << b);
        std::memcpy(ki.vs_first[w * 64 + b],
                    &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + (w << 8) + (b << 2)],
                    4 * sizeof(float));
      }
    }
    for (uint32_t w = 0; w < 4; ++w) {
      uint64_t e = ki.ps_used[w];
      uint32_t b;
      while (rex::bit_scan_forward(e, &b)) {
        e &= ~(1ull << b);
        std::memcpy(ki.ps_first[w * 64 + b],
                    &regs[XE_GPU_REG_SHADER_CONSTANT_256_X + (w << 8) + (b << 2)],
                    4 * sizeof(float));
      }
    }
    return;
  }
  for (uint32_t w = 0; w < 4; ++w) {
    uint64_t e = ki.vs_used[w];
    uint32_t b;
    while (rex::bit_scan_forward(e, &b)) {
      e &= ~(1ull << b);
      if (std::memcmp(ki.vs_first[w * 64 + b],
                      &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + (w << 8) + (b << 2)],
                      4 * sizeof(float)) != 0) {
        ki.vs_diff[w] |= (1ull << b);
      }
    }
  }
  for (uint32_t w = 0; w < 4; ++w) {
    uint64_t e = ki.ps_used[w];
    uint32_t b;
    while (rex::bit_scan_forward(e, &b)) {
      e &= ~(1ull << b);
      if (std::memcmp(ki.ps_first[w * 64 + b],
                      &regs[XE_GPU_REG_SHADER_CONSTANT_256_X + (w << 8) + (b << 2)],
                      4 * sizeof(float)) != 0) {
        ki.ps_diff[w] |= (1ull << b);
      }
    }
  }
}

// Emit the per-key instancing report once, sorted by draw count.
void InstanceProbeDump() {
  std::vector<const InstKeyInfo*> v;
  v.reserve(g_inst_keys.size());
  for (auto& kv : g_inst_keys) v.push_back(&kv.second);
  std::sort(v.begin(), v.end(),
            [](const InstKeyInfo* a, const InstKeyInfo* b) { return a->count > b->count; });
  uint64_t total = 0;
  uint32_t clean = 0;  // keys with NO per-draw pixel/material delta (clean instancing)
  for (auto* k : v) {
    total += k->count;
    if (!(k->ps_diff[0] | k->ps_diff[1] | k->ps_diff[2] | k->ps_diff[3])) ++clean;
  }
  REXGPU_INFO(
      "[inst-probe] === {} draws over {} distinct batch keys; {} keys have NO pixel-const "
      "delta (clean instancing candidates) ===",
      total, v.size(), clean);
  const size_t report = v.size() < 60 ? v.size() : 60;
  for (size_t i = 0; i < report; ++i) {
    const InstKeyInfo* k = v[i];
    uint32_t vs_used_n = 0, ps_used_n = 0;
    for (uint32_t w = 0; w < 4; ++w) {
      vs_used_n += rex::bit_count(k->vs_used[w]);
      ps_used_n += rex::bit_count(k->ps_used[w]);
    }
    REXGPU_INFO(
        "[inst-probe] x{:5} prim{} idx{} vtx{} vs={:016X} ps={:016X} ib={:08X}:cnt{} vb={:08X} "
        "| vs_vary={} ({} used) ps_vary={} ({} used)",
        k->count, k->prim, k->idx, k->vtx, k->vs_hash, k->ps_hash, k->ib_base, k->ib_cnt, k->vb0,
        inst_decode_mask(k->vs_diff), vs_used_n, inst_decode_mask(k->ps_diff), ps_used_n);
  }
}

inline uint64_t prof_ns_since(std::chrono::steady_clock::time_point t0) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count());
}

// [BATCH-CENSUS] Continuous batch-key census at the IssueDrawImpl success
// seam (see the cvar comment). Two populations per second:
//  - adjacent: draws whose key equals the previous issued draw's key. A run of
//    length L contributes L-1 to `adj` and all L draws to `rundraws` - the
//    "share a key with a neighbour" number, mergeable with NO reordering.
//  - frame-global: keys seen >=2 times anywhere in the frame. `extra` =
//    sum(count-1) = draws removable by pooling WITH reordering (upper bound).
// A resolve or a frame boundary breaks a run (real merging cannot span them).
// CP-thread only, like every [gpu-draw] counter.
bool g_batch_census = false;
uint64_t g_bc_prev_key = 0;
uint32_t g_bc_run_len = 0;
uint64_t g_bc_draws = 0, g_bc_adj = 0, g_bc_runs = 0, g_bc_run_draws = 0;
uint64_t g_bc_maxrun = 0;  // window max, reset at each 1 Hz report
uint64_t g_bc_breaks = 0;
std::unordered_map<uint64_t, uint32_t> g_bc_frame_keys;
constexpr size_t kBcMaxFrameKeys = 32768;
uint64_t g_bc_frame_ovf = 0;
uint64_t g_bc_frames = 0, g_bc_keys = 0, g_bc_rep_draws = 0, g_bc_extra = 0;
uint64_t g_bc_maxkey = 0;  // window max, reset at each 1 Hz report

void BatchCensusRunEnd() {
  if (g_bc_run_len >= 2) {
    ++g_bc_runs;
    g_bc_run_draws += g_bc_run_len;
    if (g_bc_run_len > g_bc_maxrun) g_bc_maxrun = g_bc_run_len;
  }
  g_bc_run_len = 0;
}

void Bc2PassClose();
void Bc2FrameEnd();
void BatchCensusBreak() {
  BatchCensusRunEnd();
  ++g_bc_breaks;
  Bc2PassClose();
}

void BatchCensusDraw(uint64_t key) {
  ++g_bc_draws;
  if (g_bc_run_len != 0 && key == g_bc_prev_key) {
    ++g_bc_adj;
    ++g_bc_run_len;
  } else {
    BatchCensusRunEnd();
    g_bc_run_len = 1;
  }
  g_bc_prev_key = key;
  auto it = g_bc_frame_keys.find(key);
  if (it != g_bc_frame_keys.end()) {
    ++it->second;
  } else if (g_bc_frame_keys.size() < kBcMaxFrameKeys) {
    g_bc_frame_keys.emplace(key, 1u);
  } else {
    ++g_bc_frame_ovf;  // an overflowed key is UNDERcounted, never miscounted
  }
}

void BatchCensusFrameEnd() {
  BatchCensusRunEnd();
  ++g_bc_frames;
  g_bc_keys += g_bc_frame_keys.size();
  uint64_t rep = 0, extra = 0, mx = 0;
  for (const auto& kv : g_bc_frame_keys) {
    if (kv.second >= 2) {
      rep += kv.second;
      extra += kv.second - 1;
    }
    if (kv.second > mx) mx = kv.second;
  }
  g_bc_rep_draws += rep;
  g_bc_extra += extra;
  if (mx > g_bc_maxkey) g_bc_maxkey = mx;
  g_bc_frame_keys.clear();
  Bc2FrameEnd();
}

// [BATCH-CENSUS v2] The pooling question with the erosion steps separated.
// v1's key (geometry + shaders) is an UPPER BOUND on mergeability: two draws
// with the same key can still differ in bound textures, blend/depth state,
// viewport or render-target config, and a mesh drawn in two passes counts as
// a repeat. v2 folds ONE draw into three scopes with the FULL merge key
// (v1 + texture fetch constants of every used slot + the raster/blend/depth/
// viewport state + the pass key):
//   fr   = frame-global, full key      (v1 minus the texture/state erosion)
//   pass = within one render-target config, resolve to resolve
//   win  = within a run of ORDER-INDEPENDENT draws (blend off, z test AND
//          z write) of one pass; any other draw closes the window, because a
//          pooled draw is issued at the window's END and may not be
//          reordered across a blended or z-disabled draw.
// win extra is the number a host in-pass pooler can actually remove today;
// fr/pass extra bound what a smarter order treatment could add.
// One open-addressed table, epoch-tagged per scope: a slot is empty when its
// frame epoch is stale, so a frame is insert-only and reset is one increment.
// Two windows: `win` (strict: class opq only) and `win2` (opq + bzw, where
// bzw = blend ON but depth test AND depth write ON = a material the game
// left alpha blending enabled on; true transparents do not write depth).
// naruto_761 classified 97% of city draws as blended, which is that habit,
// not real transparency; win2 is the practical pool under the depth-write
// argument, win is the provable one.
// Third window `win3` (rung 1b-3 gate, 2026-09-02): win2's class rule with
// the key MINUS the texture fetch constants. win3 extra - win2 extra = the
// share per-instance texture descriptor indices would add to the pool.
struct Bc2Slot {
  uint64_t key;
  uint32_t ep_fr, ep_pass, ep_win[3];
  uint32_t n_fr, n_pass, n_win[3];
};
constexpr uint32_t kBc2Slots = 1u << 16;
constexpr uint32_t kBc2MaxProbe = 64;
std::vector<Bc2Slot> g_bc2_tab;
uint32_t g_bc2_ep_fr = 1, g_bc2_ep_pass = 1, g_bc2_ep_win[3] = {1, 1, 1};
uint64_t g_bc2_pass_key = 0;
bool g_bc2_pass_open = false;
uint32_t g_bc2_cur_win_draws[3] = {0, 0, 0};  // poolable draws in the open window
enum : uint32_t { kBc2Opq = 0, kBc2Bzw = 1, kBc2Zro = 2, kBc2Bld = 3 };
struct Bc2Counters {
  uint64_t extra_fr = 0, extra_pass = 0, extra_win[3] = {0, 0, 0};
  uint64_t passes = 0, wins[3] = {0, 0, 0}, win_blk[3] = {0, 0, 0}, win_draws[3] = {0, 0, 0},
           keys_win[3] = {0, 0, 0};
  uint64_t cls[4] = {0, 0, 0, 0};
  uint64_t ord[3][4] = {};  // pooled draw by ordinal: 2-3 / 4-9 / 10-49 / 50+
  uint64_t inst_vec4[3] = {0, 0, 0};  // per-instance vec4s streamed by pooled draws
  uint64_t ovf = 0;
};
Bc2Counters g_bc2;
uint64_t g_bc2_maxwin[3] = {0, 0, 0};  // window max key count, reset at each report
// What the city actually sets: top RB_BLENDCONTROL0 (masked) and normalized
// RB_DEPTHCONTROL values by draw count, per report window.
std::unordered_map<uint32_t, uint64_t> g_bc2_blend_hist, g_bc2_depth_hist;

Bc2Slot* Bc2Lookup(uint64_t key) {
  uint32_t i = uint32_t(key ^ (key >> 29) ^ (key >> 47)) & (kBc2Slots - 1);
  for (uint32_t p = 0; p < kBc2MaxProbe; ++p) {
    Bc2Slot& s = g_bc2_tab[i];
    if (s.ep_fr != g_bc2_ep_fr) {
      s.key = key;
      s.ep_fr = g_bc2_ep_fr;
      s.ep_pass = s.ep_win[0] = s.ep_win[1] = s.ep_win[2] = 0;
      s.n_fr = s.n_pass = s.n_win[0] = s.n_win[1] = s.n_win[2] = 0;
      return &s;
    }
    if (s.key == key) return &s;
    i = (i + 1) & (kBc2Slots - 1);
  }
  return nullptr;
}

void Bc2WinClose(uint32_t w, bool by_draw) {
  if (g_bc2_cur_win_draws[w]) {
    ++g_bc2.wins[w];
    if (by_draw) ++g_bc2.win_blk[w];
  }
  g_bc2_cur_win_draws[w] = 0;
  ++g_bc2_ep_win[w];
}

void Bc2PassClose() {
  Bc2WinClose(0, false);
  Bc2WinClose(1, false);
  Bc2WinClose(2, false);
  if (g_bc2_pass_open) ++g_bc2.passes;
  g_bc2_pass_open = false;
  ++g_bc2_ep_pass;
}

void Bc2WinFold(Bc2Slot* s, uint32_t w, uint32_t vs_float_count) {
  if (s->ep_win[w] != g_bc2_ep_win[w]) {
    s->ep_win[w] = g_bc2_ep_win[w];
    s->n_win[w] = 0;
  }
  ++g_bc2_cur_win_draws[w];
  ++g_bc2.win_draws[w];
  const uint32_t n = ++s->n_win[w];
  if (n == 1) {
    ++g_bc2.keys_win[w];
    return;
  }
  ++g_bc2.extra_win[w];
  g_bc2.inst_vec4[w] += vs_float_count;
  g_bc2.ord[w][n < 4 ? 0 : n < 10 ? 1 : n < 50 ? 2 : 3] += 1;
  if (n > g_bc2_maxwin[w]) g_bc2_maxwin[w] = n;
}

void Bc2Draw(uint64_t key, uint64_t key_notex, uint64_t pass_key, uint32_t cls,
             uint32_t vs_float_count, uint32_t blend0, uint32_t depthctl) {
  if (!g_bc2_pass_open || pass_key != g_bc2_pass_key) {
    Bc2PassClose();
    g_bc2_pass_key = pass_key;
    g_bc2_pass_open = true;
  }
  ++g_bc2.cls[cls];
  if (g_bc2_blend_hist.size() < 256) ++g_bc2_blend_hist[blend0];
  if (g_bc2_depth_hist.size() < 256) ++g_bc2_depth_hist[depthctl];
  const bool pool0 = cls == kBc2Opq;
  const bool pool1 = cls == kBc2Opq || cls == kBc2Bzw;
  if (!pool0) Bc2WinClose(0, true);
  if (!pool1) {
    Bc2WinClose(1, true);
    Bc2WinClose(2, true);
  }
  if (pool1) {
    // win3 first: its slot is another key; a table overflow undercounts.
    if (Bc2Slot* s3 = Bc2Lookup(key_notex)) Bc2WinFold(s3, 2, vs_float_count);
  }
  Bc2Slot* s = Bc2Lookup(key);
  if (!s) {
    ++g_bc2.ovf;  // undercounted, never miscounted
    return;
  }
  if (++s->n_fr >= 2) ++g_bc2.extra_fr;
  if (s->ep_pass != g_bc2_ep_pass) {
    s->ep_pass = g_bc2_ep_pass;
    s->n_pass = 0;
  }
  if (++s->n_pass >= 2) ++g_bc2.extra_pass;
  if (pool0) Bc2WinFold(s, 0, vs_float_count);
  if (pool1) Bc2WinFold(s, 1, vs_float_count);
}

// Top-4 of a value histogram as "v:share%" text, then clear it.
std::string Bc2HistTop(std::unordered_map<uint32_t, uint64_t>& h, uint64_t total) {
  std::vector<std::pair<uint32_t, uint64_t>> v(h.begin(), h.end());
  std::sort(v.begin(), v.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string out;
  for (size_t i = 0; i < v.size() && i < 4; ++i) {
    if (i) out += ' ';
    out += fmt::format("{:08x}:{:.1f}%", v[i].first,
                       total ? 100.0 * double(v[i].second) / double(total) : 0.0);
  }
  h.clear();
  return out.empty() ? "-" : out;
}

void Bc2FrameEnd() {
  Bc2PassClose();
  ++g_bc2_ep_fr;
  if (g_bc2_ep_fr == 0) {  // wrap: every slot must read stale
    for (Bc2Slot& s : g_bc2_tab) s.ep_fr = 0;
    g_bc2_ep_fr = 1;
  }
}

void Bc2Reset() {
  if (g_bc2_tab.empty()) g_bc2_tab.resize(kBc2Slots);
  for (Bc2Slot& s : g_bc2_tab) s.ep_fr = 0;
  g_bc2_ep_fr = g_bc2_ep_pass = g_bc2_ep_win[0] = g_bc2_ep_win[1] = g_bc2_ep_win[2] = 1;
  g_bc2_pass_key = 0;
  g_bc2_pass_open = false;
  g_bc2_cur_win_draws[0] = g_bc2_cur_win_draws[1] = g_bc2_cur_win_draws[2] = 0;
  g_bc2 = Bc2Counters{};
  g_bc2_maxwin[0] = g_bc2_maxwin[1] = g_bc2_maxwin[2] = 0;
  g_bc2_blend_hist.clear();
  g_bc2_depth_hist.clear();
}

void BatchCensusReset() {
  g_bc_prev_key = 0;
  g_bc_run_len = 0;
  g_bc_draws = g_bc_adj = g_bc_runs = g_bc_run_draws = g_bc_maxrun = 0;
  g_bc_breaks = 0;
  g_bc_frame_keys.clear();
  g_bc_frame_ovf = 0;
  g_bc_frames = g_bc_keys = g_bc_rep_draws = g_bc_extra = g_bc_maxkey = 0;
  Bc2Reset();
}

// ===================== [DRAW-POOL] rung 1 =====================
// The city draw stream repeats a full merge key (geometry + shaders + VB/IB +
// textures + blend/depth state + RT config) for 36% of its draws WITHIN a
// pass, interleaved, and every one of those repeats is order-independent
// (naruto_762). The pool turns each such repeat into one appended instance:
//   key time (before any derivation): compute the key; if a batch is open
//     for it in the current window and the draw's remaining state matches
//     the batch's snapshot byte for byte, append the vertex float constants
//     to the batch's instance block, raise InstanceCount in its argument
//     buffer, and return. That skips IssueDraw entirely.
//   emission (first occurrence, eligible, predictor says the key repeats):
//     one ExecuteIndirect with the instanced VS variant instead of a plain
//     draw; args + instance block reserved from the upload pool.
//   window close (any non-opaque draw, RT config change, resolve, frame
//     end, submission): one epoch increment; nothing to flush.
// Reordering: an appended instance renders at the FIRST occurrence's
// position. Legal because the pool class is order-independent by blend
// arithmetic (color ONE/ZERO, alpha ONE/ZERO or ZERO/ZERO) AND z-tested AND
// z-written, and no non-opaque draw sits between the two (it would have
// closed the window).
// Same-address hazards: a VB/IB/texture rewritten between the two draws at
// the same address would be a fence violation on real hardware too (the GPU
// consumes asynchronously), so the key's addresses identify contents.
bool g_pool_on = false;
// Drive 840 (Intel 720p, in-place A/B): the draw pool cost the Intel 17 ms
// of draw class per frame (61 -> 44 ms, 13.7 -> 17.4 fps). The iGPU prices a
// draw call at zero, so the pool's root-constant elements and its instanced
// vertex shader are pure cost there; on NVIDIA the pool is the CP lever
// (drive 817: -15 fps without it). Vendor-keyed code: on Intel no batch
// opens and no hit joins one; every draw is a plain draw.
bool g_pool_intel_off = false;

// [SKC] see command_processor.cpp (the tracker). Classes: 0 vs float, 1 ps
// float, 2 fetch (vb), 3 fetch (tex), 4 bool/loop, 5 state 0x2000-0x23FF,
// 6 below 0x2000, 7 other.
struct SkcStats {
  uint64_t hits = 0, hits_chg0 = 0, hits_ovf = 0, hit_only_alu = 0;
  uint64_t chg[8] = {};      // changed dwords on hits, per class (2 = any fetch)
  uint64_t hit_cls[8] = {};  // hits whose group changed the class
  uint64_t plain = 0, plain_taint = 0, taint_pop = 0;
  uint64_t rd[8] = {};       // plain draws reading a tainted register, per class
  uint64_t rd_regs[8] = {};  // tainted registers read, per class
};
SkcStats g_skc;
uint32_t g_skc_ntaint_cls[8] = {};
// Per-register histogram of tainted registers READ by plain draws (naruto_793:
// the state class carried 96% of the reading draws; name the registers).
// Float constants are binned on the vec4's first register. Cleared per window.
uint32_t g_skc_rd_hist[RegisterFile::kRegisterCount] = {};
inline int SkcClass(uint32_t idx) {
  if (idx >= XE_GPU_REG_SHADER_CONSTANT_000_X && idx <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    return idx < XE_GPU_REG_SHADER_CONSTANT_256_X ? 0 : 1;
  }
  if (idx >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 && idx <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    return 2;
  }
  if (idx >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 && idx <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    return 4;
  }
  if (idx < 0x2000) return 6;
  if (idx < 0x2400) return 5;
  return 7;
}
void SkcGroupReset() {
  g_skc_chg_n = 0;
  g_skc_chg_ovf = false;
  ++g_skc_gen;
}
void SkcHit() {
  ++g_skc.hits;
  if (!g_skc_chg_n) ++g_skc.hits_chg0;
  if (g_skc_chg_ovf) ++g_skc.hits_ovf;
  uint32_t clsmask = 0;
  for (uint32_t i = 0; i < g_skc_chg_n; ++i) {
    const uint32_t r = g_skc_chg[i];
    const int c = SkcClass(r);
    ++g_skc.chg[c];
    clsmask |= 1u << c;
    if (!g_skc_taint[r]) {
      g_skc_taint[r] = 1;
      ++g_skc_ntaint;
    }
  }
  for (int c = 0; c < 8; ++c) {
    if (clsmask & (1u << c)) ++g_skc.hit_cls[c];
  }
  if (clsmask && !(clsmask & ~3u)) ++g_skc.hit_only_alu;
  // Recount the taint population per class (cheap: only on hits).
  for (int c = 0; c < 8; ++c) g_skc_ntaint_cls[c] = 0;
  if (g_skc_ntaint) {
    for (uint32_t r = 0; r < RegisterFile::kRegisterCount; ++r) {
      if (g_skc_taint[r]) ++g_skc_ntaint_cls[SkcClass(r)];
    }
  }
}
void SkcPlain(const D3D12Shader* vs, const D3D12Shader* ps) {
  ++g_skc.plain;
  g_skc.taint_pop += g_skc_ntaint;
  if (!g_skc_ntaint || !vs) return;
  uint32_t rdmask = 0;
  const Shader::ConstantRegisterMap& vm = vs->constant_register_map();
  const Shader::ConstantRegisterMap* pm = ps ? &ps->constant_register_map() : nullptr;
  if (g_skc_ntaint_cls[0] | g_skc_ntaint_cls[1]) {
    for (uint32_t c = 0; c < 512; ++c) {
      const uint32_t r = XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * c;
      if (!(g_skc_taint[r] | g_skc_taint[r + 1] | g_skc_taint[r + 2] | g_skc_taint[r + 3])) continue;
      bool used;
      if (c < 256) {
        used = vm.float_dynamic_addressing || ((vm.float_bitmap[c >> 6] >> (c & 63)) & 1);
      } else {
        const uint32_t p = c - 256;
        used = pm && (pm->float_dynamic_addressing || ((pm->float_bitmap[p >> 6] >> (p & 63)) & 1));
      }
      if (used) {
        rdmask |= 1u << (c < 256 ? 0 : 1);
        ++g_skc_rd_hist[r];
        ++g_skc.rd_regs[c < 256 ? 0 : 1];
      }
    }
  }
  if (g_skc_ntaint_cls[2]) {
    const uint32_t tm = vs->GetUsedTextureMaskAfterTranslation() |
                        (ps ? ps->GetUsedTextureMaskAfterTranslation() : 0u);
    for (uint32_t d = 0; d < 192; ++d) {
      if (!g_skc_taint[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + d]) continue;
      const uint32_t vb = d / 2, tex = d / 6;
      if ((vm.vertex_fetch_bitmap[vb >> 5] >> (vb & 31)) & 1) {
        rdmask |= 1u << 2;
        ++g_skc.rd_regs[2];
        ++g_skc_rd_hist[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + d];
      }
      if ((tm >> tex) & 1) {
        rdmask |= 1u << 3;
        ++g_skc.rd_regs[3];
        ++g_skc_rd_hist[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + d];
      }
    }
  }
  if (g_skc_ntaint_cls[4]) {
    for (uint32_t r = XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031; r <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31; ++r) {
      if (g_skc_taint[r]) {
        rdmask |= 1u << 4;
        ++g_skc.rd_regs[4];
      }
    }
  }
  if (g_skc_ntaint_cls[5] | g_skc_ntaint_cls[6]) {
    for (uint32_t r = 0; r < 0x2400; ++r) {
      if (g_skc_taint[r]) {
        rdmask |= 1u << (r < 0x2000 ? 6 : 5);
        ++g_skc_rd_hist[r];
        ++g_skc.rd_regs[r < 0x2000 ? 6 : 5];
      }
    }
  }
  if (g_skc_ntaint_cls[7]) {
    for (uint32_t r = 0x2400; r < RegisterFile::kRegisterCount; ++r) {
      if (g_skc_taint[r] && SkcClass(r) == 7) {
        rdmask |= 1u << 7;
        ++g_skc.rd_regs[7];
      }
    }
  }
  if (rdmask) ++g_skc.plain_taint;
  for (int c = 0; c < 8; ++c) {
    if (rdmask & (1u << c)) ++g_skc.rd[c];
  }
}
constexpr uint32_t kPoolMaxPsVec4 = 64;
// Context registers that feed the system constants / fixed-function state
// and are NOT already in the key: compared byte for byte on a hit.
constexpr uint32_t kPoolRegRanges[][2] = {
    {0x2000, 0x2006}, {0x200E, 0x2010}, {0x2080, 0x208C}, {0x2100, 0x2115}, {0x2180, 0x2185},
    {0x2200, 0x220C}, {0x2280, 0x2295}, {0x2300, 0x2309}, {0x2312, 0x2313}, {0x2380, 0x23A0}};
constexpr uint32_t kPoolRegCount = 6 + 2 + 12 + 21 + 5 + 12 + 21 + 9 + 1 + 32;
struct PoolSnap {
  uint32_t regs[kPoolRegCount];
  uint32_t bool_loop[8 + 32];
  uint32_t ps_n;  // dwords of ps[] in use
  uint32_t ps[kPoolMaxPsVec4 * 4];
};
struct PoolBatch {
  uint64_t key;
  uint64_t gk;      // [GKEY] the opener's guest hash (0 = none); census only
  uint32_t gkc[6];  // [GKEY] its six components, for the hit attribution
  uint32_t epoch;
  uint32_t count, capacity, float_count;
  uint64_t float_bitmap[4];
  uint8_t* inst;        // upload-heap instance block (write only)
  uint32_t* count_ptr;  // InstanceCount inside the argument buffer
  // Rung 1b-2: the pixel shader's per-instance block (ps_inst batches only).
  bool ps_inst;
  uint32_t ps_float_count;
  uint64_t ps_bitmap[4];
  uint8_t* ps_block;
  PoolSnap snap;
  uint32_t occ_tag;  // [occ] this batch's census tag in the open submission
  // [hiz-pool] the per-instance form: hiz_base = the first argument slot of
  // the batch's capacity-long run (UINT32_MAX = the single-element form),
  // hiz_serial = the window the batch belongs to, idx_count = the host index
  // (vertex) count every element draws, hiz_dir = the depth test direction.
  uint32_t hiz_base, hiz_serial, idx_count;
  uint8_t hiz_dir;
  // [hiz-pool] the recorded ExecuteIndirect of the per-instance form, for
  // the exact-count patch at close (ei_offset SIZE_MAX = none).
  size_t ei_offset;
  uint64_t ei_gen, ei_arg_offset;
  ID3D12CommandSignature* ei_sig;
  ID3D12Resource* ei_buf;
};
std::vector<PoolBatch> g_pool_batches;  // arena, reused per window
DeferredCommandList* g_pool_dcl = nullptr;  // [hiz-pool] the list the batches record into
// [occ] the open submission's tags (a pool batch writes its final instance
// count into its tag at close, which always precedes the submission's
// resolve: kPoolCloseSubmit runs before the deferred list executes).
std::vector<OccCensusTag> g_occ_sub_tags;
uint32_t g_pool_batch_n = 0;
struct PoolSlot {
  uint64_t key;
  uint32_t epoch;
  uint32_t batch;
};
constexpr uint32_t kPoolSlots = 1u << 13;
constexpr uint32_t kPoolMaxProbe = 32;
std::vector<PoolSlot> g_pool_tab;
uint32_t g_pool_epoch = 1;
uint64_t g_pool_pass_key = 0;
bool g_pool_pass_open = false;
// The repeat predictor: a key's max count within one window of the previous
// frame decides whether its first occurrence opens a batch (and how big).
// A key that never repeated draws plainly and costs nothing.
struct PoolPred {
  uint64_t key;
  uint32_t frame, win_epoch;
  uint16_t win_count, frame_max, prev_max, pad;
  // Rung 1b-2: the last frame in which two pooled draws of this key differed
  // in pixel-shader constants; the key opens PS-instanced while it is fresh.
  uint32_t ps_delta_frame;
};
constexpr uint32_t kPoolPredSlots = 1u << 14;
std::vector<PoolPred> g_pool_pred;
uint32_t g_pool_frame = 1;
// Rung 1b-2: the predictor entry of the draw being issued (set by
// PoolPredict); its ps_delta_frame decides PS-instanced opens.
PoolPred* g_pool_last_pred = nullptr;
struct PoolStats {
  uint64_t draws = 0, opq = 0, hit = 0, opened = 0, first_in_batch = 0;
  uint64_t opened_ps = 0, hit_ps = 0;  // rung 1b-2: PS-instanced batches / their hits
  uint64_t ref_pred = 0, ref_cap = 0, ref_vfy_regs = 0, ref_vfy_ps = 0, ref_vfy_bl = 0,
           ref_elig = 0, ref_tile = 0, ref_alloc = 0, ref_psbig = 0, ref_ovf = 0;
  uint64_t close_class = 0, close_pass = 0, close_copy = 0, close_frame = 0, close_submit = 0;
  uint64_t ei_patched = 0, ei_unpatched = 0;  // [hiz-pool] exact-count patches at close
  uint64_t hist[5] = {0, 0, 0, 0, 0};  // closed batch sizes 1 / 2-3 / 4-9 / 10-49 / 50+
  uint64_t inst_bytes = 0, frames = 0;
  // [GEO] the iGPU's work unit is geometry, not draws (drive 766 reread +
  // tools/mdi-bench.cpp): host vertex/index counts issued per frame, plain
  // draws vs pooled hits, and received draws bucketed by index count
  // (<=16 / <=64 / <=256 / <=1k / <=4k / 4k+) with the index share per bucket.
  uint64_t geo_idx_plain = 0, geo_idx_hit = 0;
  uint64_t geo_hist[6] = {0, 0, 0, 0, 0, 0}, geo_hist_idx[6] = {0, 0, 0, 0, 0, 0};
};
PoolStats g_pool;
inline void GeoCount(uint32_t n, bool hit) {
  (hit ? g_pool.geo_idx_hit : g_pool.geo_idx_plain) += n;
  const int b = n <= 16 ? 0 : n <= 64 ? 1 : n <= 256 ? 2 : n <= 1024 ? 3 : n <= 4096 ? 4 : 5;
  ++g_pool.geo_hist[b];
  g_pool.geo_hist_idx[b] += n;
}

// [GKEY] rung-2 census window table (epoch-tagged like the pool tables;
// defined with the census below, cleared with them on an epoch wrap).
struct GkSlot {
  uint64_t gk;
  uint64_t hkey;
  uint32_t epoch;
  // [GKEY] HKNE attribution: the first occurrence's host-key inputs.
  uint64_t vs_hash, ps_hash;
  uint32_t ib_base, index_count, prim, addr;
};
std::vector<GkSlot> g_gk_tab;

enum PoolCloseReason { kPoolCloseClass, kPoolClosePass, kPoolCloseCopy, kPoolCloseFrame,
                       kPoolCloseSubmit };
void PoolClose(PoolCloseReason why) {
  for (uint32_t i = 0; i < g_pool_batch_n; ++i) {
    const uint32_t n = g_pool_batches[i].count;
    ++g_pool.hist[n < 2 ? 0 : n < 4 ? 1 : n < 10 ? 2 : n < 50 ? 3 : 4];
    // [hiz-pool] the exact count into the recorded ExecuteIndirect: the
    // batch takes no more instances from here, the list has not executed.
    PoolBatch& eb = g_pool_batches[i];
    if (eb.ei_offset != SIZE_MAX && g_pool_dcl) {
      if (g_pool_dcl->PatchExecuteIndirectMaxCount(eb.ei_offset, eb.ei_gen, eb.ei_sig, eb.ei_buf,
                                                   eb.ei_arg_offset, n)) {
        ++g_pool.ei_patched;
      } else {
        ++g_pool.ei_unpatched;
      }
      eb.ei_offset = SIZE_MAX;
    }
    // [occ] the batch's final instance count into its census tag.
    const uint32_t t = g_pool_batches[i].occ_tag;
    if (t < g_occ_sub_tags.size()) g_occ_sub_tags[t].inst = n;
  }
  if (g_pool_batch_n) {
    switch (why) {
      case kPoolCloseClass: ++g_pool.close_class; break;
      case kPoolClosePass: ++g_pool.close_pass; break;
      case kPoolCloseCopy: ++g_pool.close_copy; break;
      case kPoolCloseFrame: ++g_pool.close_frame; break;
      case kPoolCloseSubmit: ++g_pool.close_submit; break;
    }
  }
  g_pool_batch_n = 0;
  ++g_pool_epoch;
  if (g_pool_epoch == 0) {
    for (PoolSlot& s : g_pool_tab) s.epoch = 0;
    for (PoolPred& p : g_pool_pred) p.win_epoch = 0;
    for (auto& e : g_gk_tab) e.epoch = 0;  // [GKEY]
    g_pool_epoch = 1;
  }
  if (why == kPoolClosePass || why == kPoolCloseCopy || why == kPoolCloseFrame ||
      why == kPoolCloseSubmit) {
    g_pool_pass_open = false;
  }
}

void PoolReset() {
  if (g_pool_tab.empty()) g_pool_tab.resize(kPoolSlots);
  if (g_pool_pred.empty()) g_pool_pred.resize(kPoolPredSlots);
  if (g_pool_batches.empty()) g_pool_batches.resize(256);
  for (PoolSlot& s : g_pool_tab) s.epoch = 0;
  for (PoolPred& p : g_pool_pred) { p.key = 0; p.frame = 0; p.win_epoch = 0; }
  g_pool_batch_n = 0;
  g_pool_epoch = 1;
  g_pool_pass_open = false;
  g_pool_frame = 1;
}

PoolSlot* PoolLookup(uint64_t key) {
  uint32_t i = uint32_t(key ^ (key >> 29) ^ (key >> 47)) & (kPoolSlots - 1);
  for (uint32_t p = 0; p < kPoolMaxProbe; ++p) {
    PoolSlot& s = g_pool_tab[i];
    if (s.epoch != g_pool_epoch) {
      s.key = key;
      s.epoch = g_pool_epoch;
      s.batch = UINT32_MAX;
      return &s;
    }
    if (s.key == key) return &s;
    i = (i + 1) & (kPoolSlots - 1);
  }
  return nullptr;
}

// Counts this occurrence of the key in the current window/frame and returns
// the key's max window count from the PREVIOUS frame.
uint32_t PoolPredict(uint64_t key) {
  PoolPred& p = g_pool_pred[uint32_t(key ^ (key >> 31)) & (kPoolPredSlots - 1)];
  if (p.key != key) {
    p.key = key;
    p.frame = 0;
    p.prev_max = 0;
    p.frame_max = 0;
    p.win_epoch = 0;
    p.ps_delta_frame = 0;
  }
  g_pool_last_pred = &p;
  if (p.frame != g_pool_frame) {
    // A key absent for a whole frame forgets; one frame back is the model.
    p.prev_max = (p.frame + 1 == g_pool_frame) ? p.frame_max : 0;
    p.frame_max = 0;
    p.frame = g_pool_frame;
    p.win_epoch = 0;
  }
  if (p.win_epoch != g_pool_epoch) {
    p.win_epoch = g_pool_epoch;
    p.win_count = 0;
  }
  if (p.win_count < 0xFFFF) ++p.win_count;
  if (p.win_count > p.frame_max) p.frame_max = p.win_count;
  return p.prev_max;
}

// The pool class: order-independent by blend arithmetic on every written RT
// AND depth tested AND depth written, in the color+depth EDRAM mode.
bool PoolClassOpaque(const RegisterFile& regs) {
  if (regs.Get<reg::RB_MODECONTROL>().edram_mode != xenos::EdramMode::kColorDepth) return false;
  const reg::RB_DEPTHCONTROL dc = draw_util::GetNormalizedDepthControl(regs);
  if (!dc.z_enable || !dc.z_write_enable) return false;
  const uint32_t mask = regs[XE_GPU_REG_RB_COLOR_MASK];
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    if (!((mask >> (4 * i)) & 0xF)) continue;
    reg::RB_BLENDCONTROL bc;
    bc.value = regs[reg::RB_BLENDCONTROL::rt_register_indices[i]];
    if (bc.color_srcblend != xenos::BlendFactor::kOne ||
        bc.color_destblend != xenos::BlendFactor::kZero ||
        bc.color_comb_fcn != xenos::BlendOp::kAdd || bc.alpha_comb_fcn != xenos::BlendOp::kAdd) {
      return false;
    }
    const bool a_replace = bc.alpha_srcblend == xenos::BlendFactor::kOne &&
                           bc.alpha_destblend == xenos::BlendFactor::kZero;
    const bool a_const = bc.alpha_srcblend == xenos::BlendFactor::kZero &&
                         bc.alpha_destblend == xenos::BlendFactor::kZero;
    if (!a_replace && !a_const) return false;
  }
  return true;
}

// Everything a hit must match beyond the key. False = the pixel shader's
// constants cannot be snapshotted (dynamic addressing / too many).
bool PoolGather(const RegisterFile& regs, const Shader* ps, PoolSnap& out) {
  uint32_t k = 0;
  for (const auto& r : kPoolRegRanges) {
    for (uint32_t i = r[0]; i < r[1]; ++i) out.regs[k++] = regs[i];
  }
  for (uint32_t i = 0; i < 8; ++i) out.bool_loop[i] = regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 + i];
  for (uint32_t i = 0; i < 32; ++i) out.bool_loop[8 + i] = regs[XE_GPU_REG_SHADER_CONSTANT_LOOP_00 + i];
  out.ps_n = 0;
  if (!ps) return true;
  const Shader::ConstantRegisterMap& cm = ps->constant_register_map();
  if (cm.float_dynamic_addressing || cm.float_count > kPoolMaxPsVec4) return false;
  uint32_t* dst = out.ps;
  for (uint32_t word = 0; word < 4; ++word) {
    uint64_t entry = cm.float_bitmap[word];
    uint32_t idx;
    while (rex::bit_scan_forward(entry, &idx)) {
      entry &= ~(uint64_t(1) << idx);
      std::memcpy(dst, &regs[XE_GPU_REG_SHADER_CONSTANT_256_X + (word << 8) + (idx << 2)], 16);
      dst += 4;
    }
  }
  out.ps_n = uint32_t(dst - out.ps);
  return true;
}

// The instance block layout is the instanced VS's: the used vertex float
// constants, packed in bitmap order, float_count vec4s per instance.
void PoolPackInstance(const RegisterFile& regs, const uint64_t bitmap[4], uint8_t* dst,
                      uint32_t base_reg = XE_GPU_REG_SHADER_CONSTANT_000_X) {
  for (uint32_t word = 0; word < 4; ++word) {
    uint64_t entry = bitmap[word];
    uint32_t idx;
    while (rex::bit_scan_forward(entry, &idx)) {
      entry &= ~(uint64_t(1) << idx);
      std::memcpy(dst, &regs[base_reg + (word << 8) + (idx << 2)], 16);
      dst += 16;
    }
  }
}

// The hit path. True = this draw is now an instance of an open batch.
// The hit decision without the commit: the batch this draw would join, or
// null. [hiz-pool] the caller appends the instance's Hi-Z entry first (a
// per-instance batch whose window closed refuses; slot_out lets it reopen).
PoolBatch* PoolMatch(uint64_t key, const RegisterFile& regs, const Shader* ps,
                     PoolSlot** slot_out) {
  *slot_out = nullptr;
  PoolSlot* s = PoolLookup(key);
  if (!s) {
    ++g_pool.ref_ovf;
    return nullptr;
  }
  *slot_out = s;
  if (s->batch == UINT32_MAX) return nullptr;
  PoolBatch& b = g_pool_batches[s->batch];
  if (b.epoch != g_pool_epoch || b.key != key) return nullptr;
  if (b.count >= b.capacity) {
    ++g_pool.ref_cap;
    s->batch = UINT32_MAX;  // the next occurrence opens a fresh batch
    return nullptr;
  }
  PoolSnap now;
  if (!PoolGather(regs, ps, now)) {
    ++g_pool.ref_psbig;
    return nullptr;
  }
  if (std::memcmp(now.regs, b.snap.regs, sizeof(now.regs)) != 0) {
    ++g_pool.ref_vfy_regs;
    return nullptr;
  }
  if (std::memcmp(now.bool_loop, b.snap.bool_loop, sizeof(now.bool_loop)) != 0) {
    ++g_pool.ref_vfy_bl;
    return nullptr;
  }
  if (now.ps_n != b.snap.ps_n || std::memcmp(now.ps, b.snap.ps, now.ps_n * 4) != 0) {
    // Rung 1b-2: a PS-constant delta marks the key (its next batch opens
    // PS-instanced); a plain batch still refuses it.
    if (g_pool_last_pred) g_pool_last_pred->ps_delta_frame = g_pool_frame;
    if (!b.ps_inst) {
      ++g_pool.ref_vfy_ps;
      return nullptr;
    }
  }
  return &b;
}

// The commit of a hit: this draw's constants become the batch's next
// instance and the count in the argument buffer grows.
void PoolCommit(PoolBatch& b, const RegisterFile& regs) {
  PoolPackInstance(regs, b.float_bitmap, b.inst + size_t(b.count) * b.float_count * 16);
  if (b.ps_inst) {
    PoolPackInstance(regs, b.ps_bitmap, b.ps_block + size_t(b.count) * b.ps_float_count * 16,
                     XE_GPU_REG_SHADER_CONSTANT_256_X);
    g_pool.inst_bytes += size_t(b.ps_float_count) * 16;
    ++g_pool.hit_ps;
  }
  ++b.count;
  *b.count_ptr = b.count;
  g_pool.inst_bytes += size_t(b.float_count) * 16;
  ++g_pool.hit;
}

// ── [GKEY] rung-2 census ────────────────────────────────────────────────
// Joins the guest hook's shadow-side hash of the pool inputs (six 32-bit
// components, nr_draw_cache.h LookupGkey; game cvar nr_gkey_census) to the
// draw being issued, and answers with counters:
//   mirror  the same hash over the REGISTER FILE equals the shadows' (per
//           component: pkt+shaders, fetch, stA, stB, stC, bool/loop) --
//           the recorder CAN compute the pool's state from what it sees
//   shptr   the shader object pointer is a stable ucode identity (a
//           REHASH = same pointer, different ucode = an in-place patch)
//   cover   of the pool's hits, how many carry a guest hash equal to the
//           batch opener's: the hits rung 2 could take from a packet
//   safety  of the draws whose guest hash equals an earlier draw's in the
//           same window, how many did the host REFUSE and why; a
//           non-constant refusal (regs/bl/host-key) is a class the guest
//           would wrongly emit an append packet for
bool g_gk_census = false;
uint64_t g_gk_cur = 0;  // this draw's guest hash (0 = none), copied at open
// [GKEY] this draw's host-key inputs, for the HKNE sample (set by GkJoin).
uint64_t g_gk_cur_vs = 0, g_gk_cur_ps = 0;
uint32_t g_gk_cur_ib = 0, g_gk_cur_cnt = 0, g_gk_cur_prim = 0, g_gk_cur_addr = 0;
uint32_t g_gk_hkne_samples = 0;
uint32_t g_gk_cur_c[6] = {0, 0, 0, 0, 0, 0};
struct GkStats {
  uint64_t draws = 0, norec = 0, argsne = 0, badhdr = 0, nogk = 0, joined = 0;
  uint64_t mir_eq = 0, mir_ne[6] = {0, 0, 0, 0, 0, 0};
  uint64_t shp_new = 0, shp_rehash = 0;
  uint64_t hit = 0, hit_eq = 0, hit_ne = 0, hit_none = 0;
  uint64_t hit_ne_c[6] = {0, 0, 0, 0, 0, 0};  // which component split the hit
  uint64_t gkeq = 0, gkeq_hkne = 0, gkeq_hit = 0, gkeq_notopen = 0, gkeq_ref_ps = 0,
           gkeq_ref_regs = 0, gkeq_ref_bl = 0, gkeq_ref_cap = 0, gkeq_ref_other = 0;
  uint64_t tile = 0, tab_ovf = 0;
  uint64_t nomask = 0, mask_ne = 0;  // guest hashed the whole window / a stale mask
};
GkStats g_gk;
constexpr uint32_t kGkSlots = 1u << 13;
struct GkShp {
  uint32_t ptr;
  uint64_t hash;
};
constexpr uint32_t kGkShpSlots = 1u << 12;
std::vector<GkShp> g_gk_shp;

inline uint32_t GkFold(uint64_t h) { return uint32_t(h) ^ uint32_t(h >> 32); }
inline uint64_t GkRegs(uint64_t h, const RegisterFile& regs, uint32_t first, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) h = inst_mix(h, regs[first + i]);
  return h;
}
// The host mirror of the game's GkCompute (naruto-recomp src/record_map.cpp):
// identical inputs in identical order, read from the register file and the
// packet. Keep the two in lockstep.
void GkHostHash(const RegisterFile& regs, const uint32_t* pkt, uint32_t pkt_ndw,
                uint32_t vs_addr, uint32_t vs_ndw, uint32_t ps_addr, uint32_t ps_ndw,
                const uint32_t m[4], uint32_t c[6]) {
  uint64_t h = 0;
  for (uint32_t i = 0; i < pkt_ndw; ++i) h = inst_mix(h, pkt[i]);
  h = inst_mix(h, vs_addr);
  h = inst_mix(h, vs_ndw);
  h = inst_mix(h, ps_addr);
  h = inst_mix(h, ps_ndw);
  c[0] = GkFold(h);
  // c1 walks the fetch window the way the guest did: the used slots of the
  // mask it stored with the hash, or the whole window when it had none.
  if (m[0] | m[1] | m[2] | m[3]) {
    h = 0;
    for (uint32_t w = 0; w < 3; ++w) {
      uint32_t bits = m[w];
      uint32_t j;
      while (rex::bit_scan_forward(bits, &j)) {
        bits &= ~(uint32_t(1) << j);
        h = GkRegs(h, regs, 0x4800 + 2 * (w * 32 + j), 2);
      }
    }
    uint32_t bits = m[3];
    uint32_t t;
    while (rex::bit_scan_forward(bits, &t)) {
      bits &= ~(uint32_t(1) << t);
      h = GkRegs(h, regs, 0x4800 + 6 * t, 6);
    }
    c[1] = GkFold(h);
  } else {
    c[1] = GkFold(GkRegs(0, regs, 0x4800, 192));
  }
  h = GkRegs(0, regs, 0x2000, 6);
  h = GkRegs(h, regs, 0x200E, 2);
  h = GkRegs(h, regs, 0x2100, 21);
  h = GkRegs(h, regs, 0x2180, 5);
  c[2] = GkFold(h);
  h = GkRegs(0, regs, 0x2200, 12);
  h = GkRegs(h, regs, 0x2280, 21);
  c[3] = GkFold(h);
  h = GkRegs(0, regs, 0x2300, 9);
  h = inst_mix(h, regs[0x2312]);
  h = GkRegs(h, regs, 0x2380, 8);
  h = GkRegs(h, regs, 0x2388, 24);
  c[4] = GkFold(h);
  c[5] = GkFold(GkRegs(0, regs, 0x4900, 40));
}
// Keyed by the ucode's physical address (the IM_LOAD operand): REHASH =
// the same address carried different bytes = an in-place patch.
void GkShaderCheck(uint32_t ptr, uint64_t hash) {
  if (!ptr || g_gk_shp.empty()) return;
  GkShp& e = g_gk_shp[((ptr >> 2) * 0x9E3779B1u) >> 20];
  if (e.ptr == ptr) {
    if (e.hash != hash) {
      ++g_gk.shp_rehash;
      e.hash = hash;
    }
    return;
  }
  e.ptr = ptr;
  e.hash = hash;
  ++g_gk.shp_new;
}
GkSlot* GkLookup(uint64_t gk, bool* fresh) {
  if (g_gk_tab.empty()) return nullptr;
  uint32_t i = uint32_t(gk ^ (gk >> 29) ^ (gk >> 47)) & (kGkSlots - 1);
  for (uint32_t p = 0; p < kPoolMaxProbe; ++p) {
    GkSlot& e = g_gk_tab[i];
    if (e.epoch != g_pool_epoch) {
      e.gk = gk;
      e.epoch = g_pool_epoch;
      e.hkey = 0;
      *fresh = true;
      return &e;
    }
    if (e.gk == gk) {
      *fresh = false;
      return &e;
    }
    i = (i + 1) & (kGkSlots - 1);
  }
  return nullptr;
}
// [GKEY-UC] one-shot layout probe: where does the guest shader OBJECT keep
// its ucode? Scans the object for the ucode inline (dword offsets 0..2047)
// and for an embedded guest pointer whose target starts with the ucode, and
// for a dword equal to the ucode's dword/byte count. Logs the first 8 VS
// and 8 PS samples; the rung-2 hook needs the offsets to hash the ucode
// (drive 782: the object pointer is re-patched, REHASH 44.6%, HKNE 12).
uint32_t g_gk_ucprobe_n[2] = {0, 0};
void GkUcodeProbe(rex::memory::Memory* mem, uint32_t obj, const Shader* sh, const Shader* other,
                  int which) {
  if (!obj || !sh || g_gk_ucprobe_n[which] >= 8) return;
  ++g_gk_ucprobe_n[which];
  const uint32_t* uc = sh->ucode_dwords();
  const uint32_t ucn = uint32_t(sh->ucode_dword_count());
  if (!uc || ucn < 8) return;
  // naruto_785: the object's block holds the OTHER stage's ucode (the
  // device's +12684/+12688 are pixel/vertex, not vertex/pixel). Match both.
  const uint32_t* uo = other ? other->ucode_dwords() : nullptr;
  const uint32_t uon = other ? uint32_t(other->ucode_dword_count()) : 0;
  const uint32_t* o = mem->TranslateVirtual<const uint32_t*>(obj);
  if (!o) return;
  auto match_at = [&](const uint32_t* q) {
    for (uint32_t k = 0; k < 8; ++k) {
      if (rex::byte_swap(q[k]) != uc[k]) return false;
    }
    return true;
  };
  auto match_other = [&](const uint32_t* q) {
    if (!uo || uon < 8) return false;
    for (uint32_t k = 0; k < 8; ++k) {
      if (rex::byte_swap(q[k]) != uo[k]) return false;
    }
    return true;
  };
  // Is the host page behind a guest address committed? (a virtual-heap
  // pointer in the object may point at freed pages; the physical views are
  // always mapped.)
  auto readable = [&](const uint32_t* q, size_t bytes) {
    MEMORY_BASIC_INFORMATION mbi;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(q);
    const uint8_t* e = a + bytes;
    while (a < e) {
      if (!VirtualQuery(a, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
          (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) {
        return false;
      }
      a = reinterpret_cast<const uint8_t*>(mbi.BaseAddress) + mbi.RegionSize;
    }
    return true;
  };
  // The walk's own shader ref for this draw: where the executed ucode lives
  // (immediate = a copy inside the buffer) and how long it is.
  uint32_t ref_addr = 0, ref_ndw = 0;
  uint8_t ref_imm = 0, ref_valid = 0;
  NrCtxShaderRef(which, &ref_addr, &ref_ndw, &ref_imm, &ref_valid);
  int ref_off = -1;    // object dword whose masked value == the ref address
  int blk_delta = -1;  // ucode found relative to the object's physical block
  int blk_delta_other = -1;  // the OTHER stage's ucode, same block
  int ndw_off_other = -1, nbytes_off_other = -1;
  uint32_t blk_base = 0;
  {
    const uint32_t bidx = which ? 8 : 6;
    const uint32_t bv = rex::byte_swap(o[bidx]);
    if (bv >= 0xA0000000u) {
      blk_base = bv;
      const uint32_t* t = mem->TranslateVirtual<const uint32_t*>(bv);
      if (t && readable(t - 8192, 16384 * 4 + 32)) {
        for (int k = -8192; k < 8192; ++k) {
          if (blk_delta < 0 && match_at(t + k)) blk_delta = k * 4;
          if (blk_delta_other < 0 && match_other(t + k)) blk_delta_other = k * 4;
          if (blk_delta >= 0 && blk_delta_other >= 0) break;
        }
      }
    }
  }
  for (uint32_t i = 0; i < 256 && uon; ++i) {
    const uint32_t v = rex::byte_swap(o[i]);
    if (v == uon && ndw_off_other < 0) ndw_off_other = int(i * 4);
    if (v == uon * 4 && nbytes_off_other < 0) nbytes_off_other = int(i * 4);
  }
  int inline_off = -1, ptr_off = -1, ptr_delta = -1, ndw_off = -1, nbytes_off = -1;
  uint32_t ptr_val = 0;
  for (uint32_t i = 0; i < 2048 && inline_off < 0; ++i) {
    if (match_at(o + i)) inline_off = int(i * 4);
  }
  for (uint32_t i = 0; i < 256 && ptr_off < 0; ++i) {
    const uint32_t v = rex::byte_swap(o[i]);
    if (v == ucn && ndw_off < 0) ndw_off = int(i * 4);
    if (v == ucn * 4 && nbytes_off < 0) nbytes_off = int(i * 4);
    if (ref_addr && (v & 0x1FFFFFFFu) == ref_addr && v >= 0x40000000u && ref_off < 0) {
      ref_off = int(i * 4);
    }
    if (v < 0x40000000u || (v & 3) != 0 || v == 0xFFFF0000u) continue;
    const uint32_t* t = mem->TranslateVirtual<const uint32_t*>(v);
    if (!t || !readable(t, 4096 * 4 + 32)) continue;
    for (uint32_t k = 0; k < 4096; ++k) {
      if (match_at(t + k)) {
        ptr_off = int(i * 4);
        ptr_val = v;
        ptr_delta = int(k * 4);
        break;
      }
    }
  }
  std::string hdr;
  for (uint32_t i = 0; i < 32; ++i) hdr += fmt::format("{:08X} ", rex::byte_swap(o[i]));
  REXGPU_INFO("[gkey-uc] {} obj={:08X} ucode ndw={} hash={:016X} | walk ref addr={:08X} "
              "ndw={} imm={} valid={} ref_off={} | blk {:08X} delta={} OTHER delta={} "
              "(other ndw={} ndw_off={} nbytes_off={}) | inline_off={} "
              "ptr_off={} (-> {:08X} +{}) ndw_off={} nbytes_off={} | o[0..31]={}",
              which ? "PS" : "VS", obj, ucn, sh->ucode_data_hash(), ref_addr, ref_ndw,
              ref_imm, ref_valid, ref_off, blk_base, blk_delta, blk_delta_other, uon,
              ndw_off_other, nbytes_off_other, inline_off, ptr_off, ptr_val, ptr_delta,
              ndw_off, nbytes_off, hdr);
}

// Before the hit decision: join, mirror, shader identity. Returns the draw's
// guest hash (0 = no joinable record).
uint64_t GkJoin(rex::memory::Memory* mem, const RegisterFile& regs, const uint32_t* pkt_raw,
                uint32_t addr, uint32_t prim, uint32_t index_count, const D3D12Shader* vs,
                const D3D12Shader* ps) {
  ++g_gk.draws;
  nr::DrawRecord rec;
  if (!addr || !pkt_raw || !nr::LookupDraw(addr, &rec)) {
    ++g_gk.norec;
    return 0;
  }
  if (rec.rid == 0 || rec.prim != prim || (rec.count != index_count && rec.count <= 65535)) {
    ++g_gk.argsne;
    return 0;
  }
  uint32_t pkt[8];
  pkt[0] = rex::byte_swap(pkt_raw[0]);
  if ((pkt[0] >> 30) != 3 || ((pkt[0] >> 8) & 0x7F) != 0x22) {
    ++g_gk.badhdr;
    return 0;
  }
  uint32_t ndw = ((pkt[0] >> 16) & 0x3FFF) + 2;
  if (ndw > 8) ndw = 8;
  for (uint32_t i = 1; i < ndw; ++i) pkt[i] = rex::byte_swap(pkt_raw[i]);
  uint32_t gc[nr::kGkWords];
  if (!nr::LookupGkey(addr, gc)) {
    ++g_gk.nogk;
    return 0;
  }
  ++g_gk.joined;
  for (uint32_t i = 0; i < 6; ++i) g_gk_cur_c[i] = gc[i];
  // The walk's shader refs = the group's IM_LOAD operands = the identity
  // the hook hashed (naruto_785/786: the object pointers are not one).
  uint32_t vs_addr = 0, vs_ndw = 0, ps_addr = 0, ps_ndw = 0;
  uint8_t vs_imm = 0, vs_valid = 0, ps_imm = 0, ps_valid = 0;
  NrCtxShaderRef(0, &vs_addr, &vs_ndw, &vs_imm, &vs_valid);
  NrCtxShaderRef(1, &ps_addr, &ps_ndw, &ps_imm, &ps_valid);
  if (!vs_valid) vs_addr = vs_ndw = 0;
  if (!ps_valid) ps_addr = ps_ndw = 0;
  g_gk_cur_vs = vs->ucode_data_hash();
  g_gk_cur_ps = ps ? ps->ucode_data_hash() : 0;
  g_gk_cur_ib = (ndw >= 5) ? pkt[3] : 0;
  g_gk_cur_cnt = index_count;
  g_gk_cur_prim = prim;
  g_gk_cur_addr = addr;
  // Publish this pair's used-slot mask for the hook's next record of it,
  // and grade the mask the guest hashed with against today's.
  {
    const Shader::ConstantRegisterMap& vcm = vs->constant_register_map();
    uint32_t hm[4] = {vcm.vertex_fetch_bitmap[0], vcm.vertex_fetch_bitmap[1],
                      vcm.vertex_fetch_bitmap[2],
                      vs->GetUsedTextureMaskAfterTranslation() |
                          (ps ? ps->GetUsedTextureMaskAfterTranslation() : 0u)};
    nr::SetGkeySlots(vs_addr, ps_addr, hm);
    const uint32_t* gm = gc + 6;
    if (!(gm[0] | gm[1] | gm[2] | gm[3])) {
      ++g_gk.nomask;
    } else if (gm[0] != hm[0] || gm[1] != hm[1] || gm[2] != hm[2] || gm[3] != hm[3]) {
      ++g_gk.mask_ne;
    }
  }
  uint32_t hc[6];
  GkHostHash(regs, pkt, ndw, vs_addr, vs_ndw, ps_addr, ps_ndw, gc + 6, hc);
  bool all = true;
  for (uint32_t i = 0; i < 6; ++i) {
    if (hc[i] != gc[i]) {
      ++g_gk.mir_ne[i];
      all = false;
    }
  }
  if (all) ++g_gk.mir_eq;
  // naruto_785: the record's "vs" object (dev+12684) carries the PIXEL
  // ucode and "ps" (dev+12688) the vertex ucode; pair the identities so.
  GkShaderCheck(vs_addr, vs->ucode_data_hash());
  if (ps) GkShaderCheck(ps_addr, ps->ucode_data_hash());
  GkUcodeProbe(mem, rec.vs, vs, ps, 0);
  if (ps) GkUcodeProbe(mem, rec.ps, ps, vs, 1);
  uint64_t gk = 0;
  for (uint32_t i = 0; i < 6; ++i) gk = inst_mix(gk, gc[i]);
  return gk ? gk : 1;
}
// After the hit decision: coverage and the safety class. r[] = the vfy
// ps/regs/bl and cap refusal counters snapshotted before PoolTryHit.
void GkAfter(uint64_t gk, uint64_t hkey, bool hit, const uint64_t r[4]) {
  if (hit) {
    ++g_gk.hit;
    if (!gk) {
      ++g_gk.hit_none;
    } else {
      PoolSlot* s = PoolLookup(hkey);
      const PoolBatch* b = (s && s->batch != UINT32_MAX) ? &g_pool_batches[s->batch] : nullptr;
      if (b && b->gk == gk) {
        ++g_gk.hit_eq;
      } else {
        ++g_gk.hit_ne;
        if (b && b->gk) {
          for (uint32_t i = 0; i < 6; ++i) {
            if (b->gkc[i] != g_gk_cur_c[i]) ++g_gk.hit_ne_c[i];
          }
        }
      }
    }
  }
  if (!gk) return;
  bool fresh = false;
  GkSlot* e = GkLookup(gk, &fresh);
  if (!e) {
    ++g_gk.tab_ovf;
    return;
  }
  if (fresh) {
    e->hkey = hkey;
    e->vs_hash = g_gk_cur_vs;
    e->ps_hash = g_gk_cur_ps;
    e->ib_base = g_gk_cur_ib;
    e->index_count = g_gk_cur_cnt;
    e->prim = g_gk_cur_prim;
    e->addr = g_gk_cur_addr;
    return;
  }
  ++g_gk.gkeq;
  if (e->hkey != hkey) {
    ++g_gk.gkeq_hkne;
    if (g_gk_hkne_samples < 8) {
      ++g_gk_hkne_samples;
      REXGPU_INFO("[gkey] HKNE sample: first addr={:08X} vs={:016X} ps={:016X} ib={:08X} "
                  "cnt={} prim={} | this addr={:08X} vs={:016X} ps={:016X} ib={:08X} cnt={} "
                  "prim={} | gk={:016X}",
                  e->addr, e->vs_hash, e->ps_hash, e->ib_base, e->index_count, e->prim,
                  g_gk_cur_addr, g_gk_cur_vs, g_gk_cur_ps, g_gk_cur_ib, g_gk_cur_cnt,
                  g_gk_cur_prim, gk);
    }
    return;
  }
  if (hit) {
    ++g_gk.gkeq_hit;
  } else if (g_pool.ref_vfy_ps != r[0]) {
    ++g_gk.gkeq_ref_ps;
  } else if (g_pool.ref_vfy_regs != r[1]) {
    ++g_gk.gkeq_ref_regs;
  } else if (g_pool.ref_vfy_bl != r[2]) {
    ++g_gk.gkeq_ref_bl;
  } else if (g_pool.ref_cap != r[3]) {
    ++g_gk.gkeq_ref_cap;
  } else {
    PoolSlot* s = PoolLookup(hkey);
    if (s && s->batch == UINT32_MAX) ++g_gk.gkeq_notopen; else ++g_gk.gkeq_ref_other;
  }
}
}  // namespace

// Generated with `xb buildshaders`.
namespace shaders {
#include "../shaders/bytecode/d3d12_5_1/apply_gamma_pwl_cs.h"
#include "../shaders/bytecode/d3d12_5_1/apply_gamma_pwl_fxaa_luma_cs.h"
#include "../shaders/bytecode/d3d12_5_1/apply_gamma_table_cs.h"
#include "../shaders/bytecode/d3d12_5_1/apply_gamma_table_fxaa_luma_cs.h"
#include "../shaders/bytecode/d3d12_5_1/fxaa_cs.h"
#include "../shaders/bytecode/d3d12_5_1/fxaa_extreme_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_downscale_cs.h"
// [hiz] fxc, see src/graphics/shaders/hiz.cs.hlsl for the build lines.
#include "../shaders/bytecode/d3d12_5_1/hiz_build_1x_cs.h"
#include "../shaders/bytecode/d3d12_5_1/hiz_build_2x_cs.h"
#include "../shaders/bytecode/d3d12_5_1/hiz_build_4x_cs.h"
#include "../shaders/bytecode/d3d12_5_1/hiz_test_cs.h"
}  // namespace shaders

// [GPU-PRECORD] Phase 1b-1c Inc 5: per-thread replay flag (see the header). Each thread
// that runs a replay (the parse thread for inline modes, the worker for thread/overlap)
// gets its own copy, so the parse thread capturing under overlap always reads false.
thread_local bool D3D12CommandProcessor::precord_replaying_ = false;

D3D12CommandProcessor::D3D12CommandProcessor(D3D12GraphicsSystem* graphics_system,
                                             system::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state), deferred_command_list_(*this) {
  legacy_readback_memexport_cvar_name_ = "d3d12_readback_memexport";
}
D3D12CommandProcessor::~D3D12CommandProcessor() {
  // [GPU-PRECORD] Phase 1b-1b: ShutdownContext normally joins the replay worker;
  // join defensively here too so a joinable std::thread never reaches its
  // destructor (which would std::terminate). Idempotent.
  PrecordWorkerShutdown();
}

void D3D12CommandProcessor::UpdateDebugMarkersEnabled() {
  debug_markers_enabled_ = IsGpuDebugMarkersEnabled();
}

void D3D12CommandProcessor::PushDebugMarker(const char* format, ...) {
  if (!debug_markers_enabled_) {
    return;
  }
  char label[256];
  va_list args;
  va_start(args, format);
  vsnprintf(label, sizeof(label), format, args);
  va_end(args);
  PrecordFlush();  // [GPU-PRECORD] keep the marker ordered after pending draws
  deferred_command_list_.BeginDebugMarker(label);
}

void D3D12CommandProcessor::PopDebugMarker() {
  if (!debug_markers_enabled_) {
    return;
  }
  PrecordFlush();  // [GPU-PRECORD] keep the marker ordered after pending draws
  deferred_command_list_.EndDebugMarker();
}

void D3D12CommandProcessor::InsertDebugMarker(const char* format, ...) {
  if (!debug_markers_enabled_) {
    return;
  }
  char label[256];
  va_list args;
  va_start(args, format);
  vsnprintf(label, sizeof(label), format, args);
  va_end(args);
  PrecordFlush();  // [GPU-PRECORD] keep the marker ordered after pending draws
  deferred_command_list_.InsertDebugMarker(label);
}

void D3D12CommandProcessor::ClearCaches() {
  CommandProcessor::ClearCaches();
  InvalidateAllVertexBufferResidency();
  cache_clear_requested_ = true;
}

void D3D12CommandProcessor::InvalidateGpuMemory() {
  if (shared_memory_) {
    shared_memory_->InvalidateAllPages();
  }
}

void D3D12CommandProcessor::InvalidateAllVertexBufferResidency() {
  vertex_buffers_in_sync_[0] = 0;
  vertex_buffers_in_sync_[1] = 0;
  for (VertexBufferState& state : vertex_buffer_states_) {
    state.address = UINT32_MAX;
    state.size = UINT32_MAX;
  }
  // [NR-RSY] Phase 5-3b-3: the mirror transcribes the same invalidation.
  if (g_nr_res) {
    nr::ResVfetchInvalidateAll(&g_nr_res_vf);
  }
}

void D3D12CommandProcessor::InvalidateVertexBufferResidency(uint32_t vfetch_index) {
  if (vfetch_index >= vertex_buffer_states_.size()) {
    return;
  }
  vertex_buffers_in_sync_[vfetch_index >> 6] &= ~(uint64_t(1) << (vfetch_index & 63));
  // [NR-RSY] Phase 5-3b-3: per-slot invalidation, transcribed. (The range
  // form funnels through here, so this is the only per-slot hook needed.)
  if (g_nr_res) {
    nr::ResVfetchInvalidateOne(&g_nr_res_vf, vfetch_index);
  }
}

void D3D12CommandProcessor::InvalidateVertexBufferResidencyRange(uint32_t first_vfetch,
                                                                 uint32_t last_vfetch) {
  if (first_vfetch > last_vfetch) {
    std::swap(first_vfetch, last_vfetch);
  }
  if (first_vfetch >= vertex_buffer_states_.size()) {
    return;
  }
  last_vfetch = std::min(last_vfetch, uint32_t(vertex_buffer_states_.size() - 1));
  for (uint32_t vfetch_index = first_vfetch; vfetch_index <= last_vfetch; ++vfetch_index) {
    InvalidateVertexBufferResidency(vfetch_index);
  }
}

void D3D12CommandProcessor::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                                    uint32_t title_id, bool blocking) {
  CommandProcessor::InitializeShaderStorage(cache_root, title_id, blocking);
  pipeline_cache_->InitializeShaderStorage(cache_root, title_id, blocking);
}

void D3D12CommandProcessor::RequestFrameTrace(const std::filesystem::path& root_path) {
  // Capture with PIX if attached.
  if (GetD3D12Provider().GetGraphicsAnalysis() != nullptr) {
    pix_capture_requested_.store(true, std::memory_order_relaxed);
    return;
  }
  CommandProcessor::RequestFrameTrace(root_path);
}

void D3D12CommandProcessor::TracePlaybackWroteMemory(uint32_t base_ptr, uint32_t length) {
  shared_memory_->MemoryInvalidationCallback(base_ptr, length, true);
  primitive_processor_->MemoryInvalidationCallback(base_ptr, length, true);
}

void D3D12CommandProcessor::RestoreEdramSnapshot(const void* snapshot) {
  // Starting a new frame because descriptors may be needed.
  if (!BeginSubmission(true)) {
    return;
  }
  render_target_cache_->RestoreEdramSnapshot(snapshot);
}

bool D3D12CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader,
                                                               uint32_t packet, uint32_t count) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_) {
    return CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(reader, packet, count);
  }

  const uint32_t kQueryFinished = rex::byte_swap(0xFFFFFEED);
  assert_true(count == 1);
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);

  uint32_t sample_count_addr = register_file_->values[XE_GPU_REG_RB_SAMPLE_COUNT_ADDR];
  auto* sample_counts =
      memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(sample_count_addr);
  if (!sample_counts) {
    DisableHostOcclusionQueries();
    return true;
  }

  auto write_fallback_result = [sample_counts, kQueryFinished]() -> bool {
    auto fake_sample_count = REXCVAR_GET(query_occlusion_fake_sample_count);
    if (fake_sample_count < 0) {
      return true;
    }
    bool is_end_via_z_pass =
        sample_counts->ZPass_A == kQueryFinished && sample_counts->ZPass_B == kQueryFinished;
    bool is_end_via_z_fail =
        sample_counts->ZFail_A == kQueryFinished && sample_counts->ZFail_B == kQueryFinished;
    std::memset(sample_counts, 0, sizeof(xenos::xe_gpu_depth_sample_counts));
    if (is_end_via_z_pass || is_end_via_z_fail) {
      sample_counts->ZPass_A = fake_sample_count;
      sample_counts->Total_A = fake_sample_count;
    }
    return true;
  };

  bool is_end_via_z_pass =
      sample_counts->ZPass_A == kQueryFinished && sample_counts->ZPass_B == kQueryFinished;
  bool is_end_via_z_fail =
      sample_counts->ZFail_A == kQueryFinished && sample_counts->ZFail_B == kQueryFinished;
  bool is_end = is_end_via_z_pass || is_end_via_z_fail;

  if (!is_end) {
    if (active_occlusion_query_.valid &&
        active_occlusion_query_.sample_count_address != sample_count_addr) {
      DisableHostOcclusionQueries();
      return write_fallback_result();
    }
    if (!BeginGuestOcclusionQuery(sample_count_addr)) {
      return write_fallback_result();
    }
    return true;
  }

  if (!active_occlusion_query_.valid ||
      active_occlusion_query_.sample_count_address != sample_count_addr) {
    DisableHostOcclusionQueries();
    return write_fallback_result();
  }

  if (!EndGuestOcclusionQuery(sample_count_addr, sample_counts)) {
    return write_fallback_result();
  }

  return true;
}

bool D3D12CommandProcessor::PushTransitionBarrier(ID3D12Resource* resource,
                                                  D3D12_RESOURCE_STATES old_state,
                                                  D3D12_RESOURCE_STATES new_state,
                                                  UINT subresource) {
  if (old_state == new_state) {
    return false;
  }
  D3D12_RESOURCE_BARRIER barrier;
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Transition.pResource = resource;
  barrier.Transition.Subresource = subresource;
  barrier.Transition.StateBefore = old_state;
  barrier.Transition.StateAfter = new_state;
  barriers_.push_back(barrier);
  return true;
}

void D3D12CommandProcessor::PushAliasingBarrier(ID3D12Resource* old_resource,
                                                ID3D12Resource* new_resource) {
  D3D12_RESOURCE_BARRIER barrier;
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Aliasing.pResourceBefore = old_resource;
  barrier.Aliasing.pResourceAfter = new_resource;
  barriers_.push_back(barrier);
}

void D3D12CommandProcessor::PushUAVBarrier(ID3D12Resource* resource) {
  D3D12_RESOURCE_BARRIER barrier;
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.UAV.pResource = resource;
  barriers_.push_back(barrier);
}

void D3D12CommandProcessor::SubmitBarriers() {
  UINT barrier_count = UINT(barriers_.size());
  if (barrier_count != 0) {
    deferred_command_list_.D3DResourceBarrier(barrier_count, barriers_.data());
    barriers_.clear();
  }
}

ID3D12RootSignature* D3D12CommandProcessor::GetRootSignature(const DxbcShader* vertex_shader,
                                                             const DxbcShader* pixel_shader,
                                                             bool tessellated) {
  if (bindless_resources_used_) {
    return tessellated ? root_signature_bindless_ds_ : root_signature_bindless_vs_;
  }

  D3D12_SHADER_VISIBILITY vertex_visibility =
      tessellated ? D3D12_SHADER_VISIBILITY_DOMAIN : D3D12_SHADER_VISIBILITY_VERTEX;

  uint32_t texture_count_vertex =
      uint32_t(vertex_shader->GetTextureBindingsAfterTranslation().size());
  uint32_t sampler_count_vertex =
      uint32_t(vertex_shader->GetSamplerBindingsAfterTranslation().size());
  uint32_t texture_count_pixel =
      pixel_shader ? uint32_t(pixel_shader->GetTextureBindingsAfterTranslation().size()) : 0;
  uint32_t sampler_count_pixel =
      pixel_shader ? uint32_t(pixel_shader->GetSamplerBindingsAfterTranslation().size()) : 0;

  // Better put the pixel texture/sampler in the lower bits probably because it
  // changes often.
  uint32_t index = 0;
  uint32_t index_offset = 0;
  index |= texture_count_pixel << index_offset;
  index_offset += D3D12Shader::kMaxTextureBindingIndexBits;
  index |= sampler_count_pixel << index_offset;
  index_offset += D3D12Shader::kMaxSamplerBindingIndexBits;
  index |= texture_count_vertex << index_offset;
  index_offset += D3D12Shader::kMaxTextureBindingIndexBits;
  index |= sampler_count_vertex << index_offset;
  index_offset += D3D12Shader::kMaxSamplerBindingIndexBits;
  index |= uint32_t(vertex_visibility == D3D12_SHADER_VISIBILITY_DOMAIN) << index_offset;
  ++index_offset;
  assert_true(index_offset <= 32);

  // Try an existing root signature.
  auto it = root_signatures_bindful_.find(index);
  if (it != root_signatures_bindful_.end()) {
    return it->second;
  }

  // Create a new one.
  D3D12_ROOT_SIGNATURE_DESC desc;
  D3D12_ROOT_PARAMETER parameters[kRootParameter_Bindful_Count_Max];
  desc.NumParameters = kRootParameter_Bindful_Count_Base;
  desc.pParameters = parameters;
  desc.NumStaticSamplers = 0;
  desc.pStaticSamplers = nullptr;
  desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  // Base parameters.

  // Fetch constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_FetchConstants];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kFetchConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  // Vertex float constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_FloatConstantsVertex];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kFloatConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = vertex_visibility;
  }

  // Pixel float constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_FloatConstantsPixel];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kFloatConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
  }

  // System constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_SystemConstants];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kSystemConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  // Bool and loop constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_BoolLoopConstants];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kBoolLoopConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  // [hiz-pool] Instance base root constant (instancing vertex shaders).
  {
    auto& parameter = parameters[kRootParameter_Bindful_InstanceBase];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kInstanceBase);
    parameter.Constants.RegisterSpace = 0;
    parameter.Constants.Num32BitValues = 1;
    parameter.ShaderVisibility = vertex_visibility;
  }

  // Shared memory and, if ROVs are used, EDRAM.
  D3D12_DESCRIPTOR_RANGE shared_memory_and_edram_ranges[3];
  {
    auto& parameter = parameters[kRootParameter_Bindful_SharedMemoryAndEdram];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 2;
    parameter.DescriptorTable.pDescriptorRanges = shared_memory_and_edram_ranges;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    shared_memory_and_edram_ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    shared_memory_and_edram_ranges[0].NumDescriptors = 1;
    shared_memory_and_edram_ranges[0].BaseShaderRegister =
        uint32_t(DxbcShaderTranslator::SRVMainRegister::kSharedMemory);
    shared_memory_and_edram_ranges[0].RegisterSpace =
        uint32_t(DxbcShaderTranslator::SRVSpace::kMain);
    shared_memory_and_edram_ranges[0].OffsetInDescriptorsFromTableStart = 0;
    shared_memory_and_edram_ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    shared_memory_and_edram_ranges[1].NumDescriptors = 1;
    shared_memory_and_edram_ranges[1].BaseShaderRegister =
        UINT(DxbcShaderTranslator::UAVRegister::kSharedMemory);
    shared_memory_and_edram_ranges[1].RegisterSpace = 0;
    shared_memory_and_edram_ranges[1].OffsetInDescriptorsFromTableStart = 1;
    if (render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock) {
      ++parameter.DescriptorTable.NumDescriptorRanges;
      shared_memory_and_edram_ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
      shared_memory_and_edram_ranges[2].NumDescriptors = 1;
      shared_memory_and_edram_ranges[2].BaseShaderRegister =
          UINT(DxbcShaderTranslator::UAVRegister::kEdram);
      shared_memory_and_edram_ranges[2].RegisterSpace = 0;
      shared_memory_and_edram_ranges[2].OffsetInDescriptorsFromTableStart = 2;
    }
  }

  // Extra parameters.

  // Pixel textures.
  D3D12_DESCRIPTOR_RANGE range_textures_pixel;
  if (texture_count_pixel > 0) {
    auto& parameter = parameters[desc.NumParameters];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range_textures_pixel;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    range_textures_pixel.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range_textures_pixel.NumDescriptors = texture_count_pixel;
    range_textures_pixel.BaseShaderRegister =
        uint32_t(DxbcShaderTranslator::SRVMainRegister::kBindfulTexturesStart);
    range_textures_pixel.RegisterSpace = uint32_t(DxbcShaderTranslator::SRVSpace::kMain);
    range_textures_pixel.OffsetInDescriptorsFromTableStart = 0;
    ++desc.NumParameters;
  }

  // Pixel samplers.
  D3D12_DESCRIPTOR_RANGE range_samplers_pixel;
  if (sampler_count_pixel > 0) {
    auto& parameter = parameters[desc.NumParameters];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range_samplers_pixel;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    range_samplers_pixel.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    range_samplers_pixel.NumDescriptors = sampler_count_pixel;
    range_samplers_pixel.BaseShaderRegister = 0;
    range_samplers_pixel.RegisterSpace = 0;
    range_samplers_pixel.OffsetInDescriptorsFromTableStart = 0;
    ++desc.NumParameters;
  }

  // Vertex textures.
  D3D12_DESCRIPTOR_RANGE range_textures_vertex;
  if (texture_count_vertex > 0) {
    auto& parameter = parameters[desc.NumParameters];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range_textures_vertex;
    parameter.ShaderVisibility = vertex_visibility;
    range_textures_vertex.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range_textures_vertex.NumDescriptors = texture_count_vertex;
    range_textures_vertex.BaseShaderRegister =
        uint32_t(DxbcShaderTranslator::SRVMainRegister::kBindfulTexturesStart);
    range_textures_vertex.RegisterSpace = uint32_t(DxbcShaderTranslator::SRVSpace::kMain);
    range_textures_vertex.OffsetInDescriptorsFromTableStart = 0;
    ++desc.NumParameters;
  }

  // Vertex samplers.
  D3D12_DESCRIPTOR_RANGE range_samplers_vertex;
  if (sampler_count_vertex > 0) {
    auto& parameter = parameters[desc.NumParameters];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range_samplers_vertex;
    parameter.ShaderVisibility = vertex_visibility;
    range_samplers_vertex.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    range_samplers_vertex.NumDescriptors = sampler_count_vertex;
    range_samplers_vertex.BaseShaderRegister = 0;
    range_samplers_vertex.RegisterSpace = 0;
    range_samplers_vertex.OffsetInDescriptorsFromTableStart = 0;
    ++desc.NumParameters;
  }

  uint64_t root_signature_hash = 0;
  ID3D12RootSignature* root_signature =
      ui::d3d12::util::CreateRootSignature(GetD3D12Provider(), desc, &root_signature_hash);
  if (root_signature == nullptr) {
    REXGPU_ERROR(
        "Failed to create a root signature with {} pixel textures, {} pixel "
        "samplers, {} vertex textures and {} vertex samplers",
        texture_count_pixel, sampler_count_pixel, texture_count_vertex, sampler_count_vertex);
    return nullptr;
  }
  root_signatures_bindful_.emplace(index, root_signature);
  root_signature_hashes_.emplace(root_signature, root_signature_hash);
  return root_signature;
}

uint32_t D3D12CommandProcessor::GetRootBindfulExtraParameterIndices(
    const DxbcShader* vertex_shader, const DxbcShader* pixel_shader,
    RootBindfulExtraParameterIndices& indices_out) {
  uint32_t index = kRootParameter_Bindful_Count_Base;
  if (pixel_shader && !pixel_shader->GetTextureBindingsAfterTranslation().empty()) {
    indices_out.textures_pixel = index++;
  } else {
    indices_out.textures_pixel = RootBindfulExtraParameterIndices::kUnavailable;
  }
  if (pixel_shader && !pixel_shader->GetSamplerBindingsAfterTranslation().empty()) {
    indices_out.samplers_pixel = index++;
  } else {
    indices_out.samplers_pixel = RootBindfulExtraParameterIndices::kUnavailable;
  }
  if (!vertex_shader->GetTextureBindingsAfterTranslation().empty()) {
    indices_out.textures_vertex = index++;
  } else {
    indices_out.textures_vertex = RootBindfulExtraParameterIndices::kUnavailable;
  }
  if (!vertex_shader->GetSamplerBindingsAfterTranslation().empty()) {
    indices_out.samplers_vertex = index++;
  } else {
    indices_out.samplers_vertex = RootBindfulExtraParameterIndices::kUnavailable;
  }
  return index;
}

uint64_t D3D12CommandProcessor::RequestViewBindfulDescriptors(
    uint64_t previous_heap_index, uint32_t count_for_partial_update, uint32_t count_for_full_update,
    D3D12_CPU_DESCRIPTOR_HANDLE& cpu_handle_out, D3D12_GPU_DESCRIPTOR_HANDLE& gpu_handle_out) {
  assert_false(bindless_resources_used_);
  assert_true(submission_open_);
  uint32_t descriptor_index;
  uint64_t current_heap_index = view_bindful_heap_pool_->Request(
      frame_current_, previous_heap_index, count_for_partial_update, count_for_full_update,
      descriptor_index);
  if (current_heap_index == ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
    // There was an error.
    return ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
  }
  ID3D12DescriptorHeap* heap = view_bindful_heap_pool_->GetLastRequestHeap();
  if (view_bindful_heap_current_ != heap) {
    view_bindful_heap_current_ = heap;
    deferred_command_list_.SetDescriptorHeaps(view_bindful_heap_current_,
                                              sampler_bindful_heap_current_);
  }
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  cpu_handle_out = provider.OffsetViewDescriptor(
      view_bindful_heap_pool_->GetLastRequestHeapCPUStart(), descriptor_index);
  gpu_handle_out = provider.OffsetViewDescriptor(
      view_bindful_heap_pool_->GetLastRequestHeapGPUStart(), descriptor_index);
  return current_heap_index;
}

uint32_t D3D12CommandProcessor::RequestPersistentViewBindlessDescriptor() {
  assert_true(bindless_resources_used_);
  uint32_t descriptor_index;
  if (!view_bindless_heap_free_.empty()) {
    descriptor_index = view_bindless_heap_free_.back();
    view_bindless_heap_free_.pop_back();
  } else if (view_bindless_heap_allocated_ >= kViewBindlessHeapSize) {
    descriptor_index = UINT32_MAX;
  } else {
    descriptor_index = view_bindless_heap_allocated_++;
  }
  // [NR-RSY] Phase 5-3b-3: every persistent view allocation is predicted by
  // the pool mirror.
  if (g_nr_res) {
    ++g_nr_res_probe.pool_alloc;
    NrResPoolObserveAlloc(descriptor_index);
  }
  return descriptor_index;
}

void D3D12CommandProcessor::ReleaseViewBindlessDescriptorImmediately(uint32_t descriptor_index) {
  assert_true(bindless_resources_used_);
  // [NR-RSY] Phase 5-3b-3: the pool mirror follows every release.
  if (g_nr_res) {
    ++g_nr_res_probe.pool_release;
    NrResPoolObserveRelease(descriptor_index);
  }
  view_bindless_heap_free_.push_back(descriptor_index);
}

bool D3D12CommandProcessor::NrResArmed() const { return g_nr_res; }

void D3D12CommandProcessor::NrResObserveTexDescriptor(const void* texture, uint32_t srv_key,
                                                      bool hit, uint32_t index) {
  if (!g_nr_res) {
    return;
  }
  switch (nr::ResTexDescObserve(&g_nr_res_tmap,
                                uint64_t(reinterpret_cast<uintptr_t>(texture)), srv_key, hit,
                                index)) {
    case nr::kResTexDescMatch:
      ++g_nr_res_probe.tmap_match;
      break;
    case nr::kResTexDescFresh:
      // A refused creation (unsupported format/exhaustion) is learned so
      // lookups resolve it to the null fallback, but counted apart.
      if (index == UINT32_MAX) {
        ++g_nr_res_probe.tmap_invalid;
      } else {
        ++g_nr_res_probe.tmap_fresh;
      }
      break;
    case nr::kResTexDescSeeded:
      ++g_nr_res_probe.tmap_seeded;
      break;
    case nr::kResTexDescOverflow:
      ++g_nr_res_probe.tmap_ovf;
      break;
    default:
      ++g_nr_res_probe.tmap_ne;
      if (g_nr_res_samples_this_window < kNrResMaxSamplesPerWindow) {
        ++g_nr_res_samples_this_window;
        REXGPU_WARN("[nr-rsy] TMAP DIFF tex={:#x} key={:#x} hit={} theirs={}",
                    uint64_t(reinterpret_cast<uintptr_t>(texture)), srv_key, hit, index);
      }
      break;
  }
}

void D3D12CommandProcessor::NrResObserveTexDescriptorRelease(uint32_t index) {
  if (!g_nr_res) {
    return;
  }
  ++g_nr_res_probe.tmap_evict;
  nr::ResTexDescEvictIndex(&g_nr_res_tmap, index);
}

void D3D12CommandProcessor::NrResPoolReseed() {
  ++g_nr_res_probe.pool_reseeds;
  nr::ResViewPoolReset(&g_nr_res_pool, view_bindless_heap_allocated_,
                       view_bindless_heap_free_.data(),
                       uint32_t(view_bindless_heap_free_.size()), kViewBindlessHeapSize);
  if (g_nr_res_pool.refused) {
    ++g_nr_res_probe.pool_refused;
  }
}

void D3D12CommandProcessor::NrResPoolObserveAlloc(uint32_t actual_index) {
  if (g_nr_res_pool.refused) {
    ++g_nr_res_probe.pool_refused;
    return;
  }
  if (!nr::ResViewPoolObserveAlloc(&g_nr_res_pool, actual_index)) {
    ++g_nr_res_probe.pool_ne;
    if (g_nr_res_samples_this_window < kNrResMaxSamplesPerWindow) {
      ++g_nr_res_samples_this_window;
      uint32_t predicted = UINT32_MAX;
      nr::ResViewPoolPredictAlloc(&g_nr_res_pool, &predicted);
      REXGPU_WARN("[nr-rsy] POOL DIFF alloc ours={} theirs={} (free={} allocated={})", predicted,
                  actual_index, g_nr_res_pool.free_count, g_nr_res_pool.allocated);
    }
    // Re-sync so one divergence names itself once.
    NrResPoolReseed();
  }
}

void D3D12CommandProcessor::NrResPoolObserveRelease(uint32_t index) {
  nr::ResViewPoolObserveRelease(&g_nr_res_pool, index);
  if (g_nr_res_pool.refused) {
    ++g_nr_res_probe.pool_refused;
  }
}

void D3D12CommandProcessor::NrResVfetchSeedFromEmulated() {
  g_nr_res_vf.in_sync[0] = vertex_buffers_in_sync_[0];
  g_nr_res_vf.in_sync[1] = vertex_buffers_in_sync_[1];
  for (uint32_t i = 0; i < nr::kResVfetchSlots; ++i) {
    g_nr_res_vf.state_address[i] = vertex_buffer_states_[i].address;
    g_nr_res_vf.state_size[i] = vertex_buffer_states_[i].size;
  }
}

void D3D12CommandProcessor::NrResVfetchFinishDraw(uint32_t abort_reason) {
  // abort_reason: 0 = the loop completed, 1 = invalid fetch type, 2 = a
  // RequestRange failed (the emulated loop returned false).
  if (!g_nr_res_vf_active) {
    return;
  }
  g_nr_res_vf_active = false;
  ++g_nr_res_probe.vf_draws;
  g_nr_res_probe.vf_req += g_nr_res_vf_obs_count;
  bool ne = false;
  if (g_nr_res_vf_pred_count > kNrResMaxRequests) {
    ++g_nr_res_probe.vf_pred_ovf;  // cannot compare - impossible for 96 slots
  } else {
    const uint32_t pred_n = g_nr_res_vf_pred_count;
    const uint32_t obs_n = g_nr_res_vf_obs_count;
    const bool pred_abort = g_nr_res_vf_pred_status == nr::kResVfetchAbortInvalidType;
    switch (abort_reason) {
      case 0:
        ne = pred_abort || obs_n != pred_n;
        break;
      case 1:
        ne = !pred_abort || obs_n != pred_n;
        break;
      default:  // a failed request: an observed prefix of the prediction
        ne = obs_n > pred_n;
        break;
    }
    const uint32_t cmp_n = obs_n < pred_n ? obs_n : pred_n;
    for (uint32_t i = 0; i < cmp_n && !ne; ++i) {
      if (g_nr_res_vf_obs[i].start != g_nr_res_vf_pred[i].start ||
          g_nr_res_vf_obs[i].length != g_nr_res_vf_pred[i].length) {
        ne = true;
      }
    }
    if (ne) {
      ++g_nr_res_probe.vf_ne;
      if (g_nr_res_samples_this_window < kNrResMaxSamplesPerWindow) {
        ++g_nr_res_samples_this_window;
        REXGPU_WARN(
            "[nr-rsy] VF DIFF reason={} pred_n={} obs_n={} pred_abort={} first_pred={:08X}+{:X} "
            "first_obs={:08X}+{:X}",
            abort_reason, pred_n, obs_n, pred_abort,
            pred_n ? g_nr_res_vf_pred[0].start : 0, pred_n ? g_nr_res_vf_pred[0].length : 0,
            obs_n ? g_nr_res_vf_obs[0].start : 0, obs_n ? g_nr_res_vf_obs[0].length : 0);
      }
    }
  }
  // Apply the walk's mutations, then the mirror must equal the emulated
  // arrays bit for bit.
  nr::ResVfetchApply(&g_nr_res_vf, g_nr_res_vf_bitmap, g_nr_res_vf_fetch_regs,
                     g_nr_res_vf_allow, g_nr_res_vf_obs_ok);
  bool state_ne = g_nr_res_vf.in_sync[0] != vertex_buffers_in_sync_[0] ||
                  g_nr_res_vf.in_sync[1] != vertex_buffers_in_sync_[1];
  if (!state_ne) {
    for (uint32_t i = 0; i < nr::kResVfetchSlots; ++i) {
      if (g_nr_res_vf.state_address[i] != vertex_buffer_states_[i].address ||
          g_nr_res_vf.state_size[i] != vertex_buffer_states_[i].size) {
        state_ne = true;
        break;
      }
    }
  }
  if (state_ne) {
    ++g_nr_res_probe.vf_state_ne;
    if (g_nr_res_samples_this_window < kNrResMaxSamplesPerWindow) {
      ++g_nr_res_samples_this_window;
      REXGPU_WARN("[nr-rsy] VF STATE DIFF in_sync ours={:016X}/{:016X} theirs={:016X}/{:016X}",
                  g_nr_res_vf.in_sync[0], g_nr_res_vf.in_sync[1], vertex_buffers_in_sync_[0],
                  vertex_buffers_in_sync_[1]);
    }
    // Re-sync so one divergence names itself once.
    NrResVfetchSeedFromEmulated();
  }
}

bool D3D12CommandProcessor::RequestOneUseSingleViewDescriptors(
    uint32_t count, ui::d3d12::util::DescriptorCpuGpuHandlePair* handles_out) {
  assert_true(submission_open_);
  if (!count) {
    return true;
  }
  assert_not_null(handles_out);
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  if (bindless_resources_used_) {
    // Request separate bindless descriptors that will be freed when this
    // submission is completed by the GPU.
    if (count >
        kViewBindlessHeapSize - view_bindless_heap_allocated_ + view_bindless_heap_free_.size()) {
      return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t descriptor_index;
      if (!view_bindless_heap_free_.empty()) {
        descriptor_index = view_bindless_heap_free_.back();
        view_bindless_heap_free_.pop_back();
      } else {
        descriptor_index = view_bindless_heap_allocated_++;
      }
      // [NR-RSY] Phase 5-3b-3: one-use views draw from the same pool -
      // predicted too.
      if (g_nr_res) {
        ++g_nr_res_probe.pool_oneuse;
        NrResPoolObserveAlloc(descriptor_index);
      }
      view_bindless_one_use_descriptors_.push_back(
          std::make_pair(descriptor_index, submission_current_));
      handles_out[i] = std::make_pair(
          provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_, descriptor_index),
          provider.OffsetViewDescriptor(view_bindless_heap_gpu_start_, descriptor_index));
    }
  } else {
    // Request a range within the current heap for bindful resources path.
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_start;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_start;
    if (RequestViewBindfulDescriptors(ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid, count,
                                      count, cpu_handle_start, gpu_handle_start) ==
        ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
      return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
      handles_out[i] = std::make_pair(provider.OffsetViewDescriptor(cpu_handle_start, i),
                                      provider.OffsetViewDescriptor(gpu_handle_start, i));
    }
  }
  return true;
}

ui::d3d12::util::DescriptorCpuGpuHandlePair D3D12CommandProcessor::GetSystemBindlessViewHandlePair(
    SystemBindlessView view) const {
  assert_true(bindless_resources_used_);
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  return std::make_pair(
      provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_, uint32_t(view)),
      provider.OffsetViewDescriptor(view_bindless_heap_gpu_start_, uint32_t(view)));
}

ui::d3d12::util::DescriptorCpuGpuHandlePair
D3D12CommandProcessor::GetSharedMemoryUintPow2BindlessSRVHandlePair(
    uint32_t element_size_bytes_pow2) const {
  SystemBindlessView view;
  switch (element_size_bytes_pow2) {
    case 2:
      view = SystemBindlessView::kSharedMemoryR32UintSRV;
      break;
    case 3:
      view = SystemBindlessView::kSharedMemoryR32G32UintSRV;
      break;
    case 4:
      view = SystemBindlessView::kSharedMemoryR32G32B32A32UintSRV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      view = SystemBindlessView::kSharedMemoryR32UintSRV;
  }
  return GetSystemBindlessViewHandlePair(view);
}

ui::d3d12::util::DescriptorCpuGpuHandlePair
D3D12CommandProcessor::GetSharedMemoryUintPow2BindlessUAVHandlePair(
    uint32_t element_size_bytes_pow2) const {
  SystemBindlessView view;
  switch (element_size_bytes_pow2) {
    case 2:
      view = SystemBindlessView::kSharedMemoryR32UintUAV;
      break;
    case 3:
      view = SystemBindlessView::kSharedMemoryR32G32UintUAV;
      break;
    case 4:
      view = SystemBindlessView::kSharedMemoryR32G32B32A32UintUAV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      view = SystemBindlessView::kSharedMemoryR32UintUAV;
  }
  return GetSystemBindlessViewHandlePair(view);
}

ui::d3d12::util::DescriptorCpuGpuHandlePair
D3D12CommandProcessor::GetEdramUintPow2BindlessSRVHandlePair(
    uint32_t element_size_bytes_pow2) const {
  SystemBindlessView view;
  switch (element_size_bytes_pow2) {
    case 2:
      view = SystemBindlessView::kEdramR32UintSRV;
      break;
    case 3:
      view = SystemBindlessView::kEdramR32G32UintSRV;
      break;
    case 4:
      view = SystemBindlessView::kEdramR32G32B32A32UintSRV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      view = SystemBindlessView::kEdramR32UintSRV;
  }
  return GetSystemBindlessViewHandlePair(view);
}

ui::d3d12::util::DescriptorCpuGpuHandlePair
D3D12CommandProcessor::GetEdramUintPow2BindlessUAVHandlePair(
    uint32_t element_size_bytes_pow2) const {
  SystemBindlessView view;
  switch (element_size_bytes_pow2) {
    case 2:
      view = SystemBindlessView::kEdramR32UintUAV;
      break;
    case 3:
      view = SystemBindlessView::kEdramR32G32UintUAV;
      break;
    case 4:
      view = SystemBindlessView::kEdramR32G32B32A32UintUAV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      view = SystemBindlessView::kEdramR32UintUAV;
  }
  return GetSystemBindlessViewHandlePair(view);
}

uint64_t D3D12CommandProcessor::RequestSamplerBindfulDescriptors(
    uint64_t previous_heap_index, uint32_t count_for_partial_update, uint32_t count_for_full_update,
    D3D12_CPU_DESCRIPTOR_HANDLE& cpu_handle_out, D3D12_GPU_DESCRIPTOR_HANDLE& gpu_handle_out) {
  assert_false(bindless_resources_used_);
  assert_true(submission_open_);
  uint32_t descriptor_index;
  uint64_t current_heap_index = sampler_bindful_heap_pool_->Request(
      frame_current_, previous_heap_index, count_for_partial_update, count_for_full_update,
      descriptor_index);
  if (current_heap_index == ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
    // There was an error.
    return ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
  }
  ID3D12DescriptorHeap* heap = sampler_bindful_heap_pool_->GetLastRequestHeap();
  if (sampler_bindful_heap_current_ != heap) {
    sampler_bindful_heap_current_ = heap;
    deferred_command_list_.SetDescriptorHeaps(view_bindful_heap_current_,
                                              sampler_bindful_heap_current_);
  }
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  cpu_handle_out = provider.OffsetSamplerDescriptor(
      sampler_bindful_heap_pool_->GetLastRequestHeapCPUStart(), descriptor_index);
  gpu_handle_out = provider.OffsetSamplerDescriptor(
      sampler_bindful_heap_pool_->GetLastRequestHeapGPUStart(), descriptor_index);
  return current_heap_index;
}

ID3D12Resource* D3D12CommandProcessor::RequestScratchGPUBuffer(uint32_t size,
                                                               D3D12_RESOURCE_STATES state) {
  assert_true(submission_open_);
  assert_false(scratch_buffer_used_);
  if (!submission_open_ || scratch_buffer_used_ || size == 0) {
    return nullptr;
  }

  if (size <= scratch_buffer_size_) {
    PushTransitionBarrier(scratch_buffer_, scratch_buffer_state_, state);
    scratch_buffer_state_ = state;
    scratch_buffer_used_ = true;
    return scratch_buffer_;
  }

  size = rex::align(size, kScratchBufferSizeIncrement);

  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  D3D12_RESOURCE_DESC buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(buffer_desc, size,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  ID3D12Resource* buffer;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             provider.GetHeapFlagCreateNotZeroed(), &buffer_desc,
                                             state, nullptr, IID_PPV_ARGS(&buffer)))) {
    REXGPU_ERROR("Failed to create a {} MB scratch GPU buffer", size >> 20);
    return nullptr;
  }
  if (scratch_buffer_ != nullptr) {
    resources_for_deletion_.emplace_back(submission_current_, scratch_buffer_);
  }
  scratch_buffer_ = buffer;
  scratch_buffer_size_ = size;
  scratch_buffer_state_ = state;
  scratch_buffer_used_ = true;
  return scratch_buffer_;
}

void D3D12CommandProcessor::ReleaseScratchGPUBuffer(ID3D12Resource* buffer,
                                                    D3D12_RESOURCE_STATES new_state) {
  assert_true(submission_open_);
  assert_true(scratch_buffer_used_);
  scratch_buffer_used_ = false;
  if (buffer == scratch_buffer_) {
    scratch_buffer_state_ = new_state;
  }
}

void D3D12CommandProcessor::SetExternalPipeline(ID3D12PipelineState* pipeline) {
  if (current_external_pipeline_ != pipeline) {
    current_external_pipeline_ = pipeline;
    current_guest_pipeline_ = nullptr;
    deferred_command_list_.D3DSetPipelineState(pipeline);
  }
}

void D3D12CommandProcessor::SetExternalGraphicsRootSignature(ID3D12RootSignature* root_signature) {
  if (current_graphics_root_signature_ != root_signature) {
    current_graphics_root_signature_ = root_signature;
    inst_base_dirty_ = true;  // [hiz-pool]
    deferred_command_list_.D3DSetGraphicsRootSignature(root_signature);
  }
  // Force-invalidate because setting a non-guest root signature.
  current_graphics_root_up_to_date_ = 0;
}

void D3D12CommandProcessor::SetViewport(const D3D12_VIEWPORT& viewport) {
  ff_viewport_update_needed_ |= ff_viewport_.TopLeftX != viewport.TopLeftX;
  ff_viewport_update_needed_ |= ff_viewport_.TopLeftY != viewport.TopLeftY;
  ff_viewport_update_needed_ |= ff_viewport_.Width != viewport.Width;
  ff_viewport_update_needed_ |= ff_viewport_.Height != viewport.Height;
  ff_viewport_update_needed_ |= ff_viewport_.MinDepth != viewport.MinDepth;
  ff_viewport_update_needed_ |= ff_viewport_.MaxDepth != viewport.MaxDepth;
  if (ff_viewport_update_needed_) {
    ff_viewport_ = viewport;
    deferred_command_list_.RSSetViewport(ff_viewport_);
    ff_viewport_update_needed_ = false;
  }
}

void D3D12CommandProcessor::SetScissorRect(const D3D12_RECT& scissor_rect) {
  ff_scissor_update_needed_ |= ff_scissor_.left != scissor_rect.left;
  ff_scissor_update_needed_ |= ff_scissor_.top != scissor_rect.top;
  ff_scissor_update_needed_ |= ff_scissor_.right != scissor_rect.right;
  ff_scissor_update_needed_ |= ff_scissor_.bottom != scissor_rect.bottom;
  if (ff_scissor_update_needed_) {
    ff_scissor_ = scissor_rect;
    deferred_command_list_.RSSetScissorRect(ff_scissor_);
    ff_scissor_update_needed_ = false;
  }
}

void D3D12CommandProcessor::SetStencilReference(uint32_t stencil_ref) {
  ff_stencil_ref_update_needed_ |= ff_stencil_ref_ != stencil_ref;
  if (ff_stencil_ref_update_needed_) {
    ff_stencil_ref_ = stencil_ref;
    deferred_command_list_.D3DOMSetStencilRef(stencil_ref);
    ff_stencil_ref_update_needed_ = false;
  }
}

void D3D12CommandProcessor::SetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY primitive_topology) {
  if (primitive_topology_ != primitive_topology) {
    primitive_topology_ = primitive_topology;
    deferred_command_list_.D3DIASetPrimitiveTopology(primitive_topology);
  }
}

std::string D3D12CommandProcessor::GetWindowTitleText() const {
  std::ostringstream title;
  title << "Direct3D 12";
  if (render_target_cache_) {
    // Rasterizer-ordered views are a feature very rarely used as of 2020 and
    // that faces adoption complications (outside of Direct3D - on Vulkan - at
    // least), but crucial to Xenia - raise awareness of its usage.
    // https://github.com/KhronosGroup/Vulkan-Ecosystem/issues/27#issuecomment-455712319
    // "In Xenia's title bar "D3D12 ROV" can be seen, which was a surprise, as I
    //  wasn't aware that Xenia D3D12 backend was using Raster Order Views
    //  feature" - oscarbg in that issue.
    switch (render_target_cache_->GetPath()) {
      case RenderTargetCache::Path::kHostRenderTargets:
        title << " - RTV/DSV";
        break;
      case RenderTargetCache::Path::kPixelShaderInterlock:
        title << " - ROV";
        break;
      default:
        break;
    }
    uint32_t draw_resolution_scale_x =
        texture_cache_ ? texture_cache_->draw_resolution_scale_x() : 1;
    uint32_t draw_resolution_scale_y =
        texture_cache_ ? texture_cache_->draw_resolution_scale_y() : 1;
    if (draw_resolution_scale_x > 1 || draw_resolution_scale_y > 1) {
      title << ' ' << draw_resolution_scale_x << 'x' << draw_resolution_scale_y;
    }
  }
  return title.str();
}

bool D3D12CommandProcessor::SetupContext() {
  if (!CommandProcessor::SetupContext()) {
    REXGPU_ERROR("Failed to initialize base command processor context");
    return false;
  }
  InvalidateAllVertexBufferResidency();
  UpdateDebugMarkersEnabled();

  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  ID3D12CommandQueue* direct_queue = provider.GetDirectQueue();

  fence_completion_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (fence_completion_event_ == nullptr) {
    REXGPU_ERROR("Failed to create the fence completion event");
    return false;
  }
  if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&submission_fence_)))) {
    REXGPU_ERROR("Failed to create the submission fence");
    return false;
  }
  if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                 IID_PPV_ARGS(&queue_operations_since_submission_fence_)))) {
    REXGPU_ERROR(
        "Failed to create the fence for awaiting queue operations done since "
        "the latest submission");
    return false;
  }

  // Create the command list and one allocator because it's needed for a command
  // list.
  ID3D12CommandAllocator* command_allocator;
  if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            IID_PPV_ARGS(&command_allocator)))) {
    REXGPU_ERROR("Failed to create a command allocator");
    return false;
  }
  command_allocator_writable_first_ = new CommandAllocator;
  command_allocator_writable_first_->command_allocator = command_allocator;
  command_allocator_writable_first_->last_usage_submission = 0;
  command_allocator_writable_first_->next = nullptr;
  command_allocator_writable_last_ = command_allocator_writable_first_;
  if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, command_allocator,
                                       nullptr, IID_PPV_ARGS(&command_list_)))) {
    REXGPU_ERROR("Failed to create the graphics command list");
    return false;
  }
  // Initially in open state, wait until a deferred command list submission.
  command_list_->Close();
  // Optional - added in Creators Update (SDK 10.0.15063.0).
  command_list_->QueryInterface(IID_PPV_ARGS(&command_list_1_));

  bindless_resources_used_ = REXCVAR_GET(d3d12_bindless) &&
                             provider.GetResourceBindingTier() >= D3D12_RESOURCE_BINDING_TIER_2;

  // Get the draw resolution scale for the render target cache and the texture
  // cache.
  uint32_t draw_resolution_scale_x, draw_resolution_scale_y;
  bool draw_resolution_scale_not_clamped =
      TextureCache::GetConfigDrawResolutionScale(draw_resolution_scale_x, draw_resolution_scale_y);
  if (!D3D12TextureCache::ClampDrawResolutionScaleToMaxSupported(
          draw_resolution_scale_x, draw_resolution_scale_y, provider)) {
    draw_resolution_scale_not_clamped = false;
  }
  if (!draw_resolution_scale_not_clamped) {
    REXGPU_WARN(
        "The requested draw resolution scale is not supported by the device or "
        "the emulator, reducing to {}x{}",
        draw_resolution_scale_x, draw_resolution_scale_y);
  }

  shared_memory_ = std::make_unique<D3D12SharedMemory>(*this, *memory_, trace_writer_);
  if (!shared_memory_->Initialize()) {
    REXGPU_ERROR("Failed to initialize shared memory");
    return false;
  }

  // Initialize the render target cache before configuring binding - need to
  // know if using rasterizer-ordered views for the bindless root signature.
  render_target_cache_ = std::make_unique<D3D12RenderTargetCache>(
      *register_file_, *memory_, trace_writer_, draw_resolution_scale_x, draw_resolution_scale_y,
      *this, bindless_resources_used_);
  if (!render_target_cache_->Initialize()) {
    REXGPU_ERROR("Failed to initialize the render target cache");
    return false;
  }

  // Initialize resource binding.
  constant_buffer_pool_ = std::make_unique<ui::d3d12::D3D12UploadBufferPool>(
      provider, std::max(ui::d3d12::D3D12UploadBufferPool::kDefaultPageSize,
                         sizeof(float) * 4 * D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT));
  if (bindless_resources_used_) {
    D3D12_DESCRIPTOR_HEAP_DESC view_bindless_heap_desc;
    view_bindless_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    view_bindless_heap_desc.NumDescriptors = kViewBindlessHeapSize;
    view_bindless_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    view_bindless_heap_desc.NodeMask = 0;
    if (FAILED(device->CreateDescriptorHeap(&view_bindless_heap_desc,
                                            IID_PPV_ARGS(&view_bindless_heap_)))) {
      REXGPU_ERROR("Failed to create the bindless CBV/SRV/UAV descriptor heap");
      return false;
    }
    view_bindless_heap_cpu_start_ = view_bindless_heap_->GetCPUDescriptorHandleForHeapStart();
    view_bindless_heap_gpu_start_ = view_bindless_heap_->GetGPUDescriptorHandleForHeapStart();
    view_bindless_heap_allocated_ = uint32_t(SystemBindlessView::kCount);

    D3D12_DESCRIPTOR_HEAP_DESC sampler_bindless_heap_desc;
    sampler_bindless_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_bindless_heap_desc.NumDescriptors = kSamplerHeapSize;
    sampler_bindless_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_bindless_heap_desc.NodeMask = 0;
    if (FAILED(device->CreateDescriptorHeap(&sampler_bindless_heap_desc,
                                            IID_PPV_ARGS(&sampler_bindless_heap_current_)))) {
      REXGPU_ERROR("Failed to create the bindless sampler descriptor heap");
      return false;
    }
    sampler_bindless_heap_cpu_start_ =
        sampler_bindless_heap_current_->GetCPUDescriptorHandleForHeapStart();
    sampler_bindless_heap_gpu_start_ =
        sampler_bindless_heap_current_->GetGPUDescriptorHandleForHeapStart();
    sampler_bindless_heap_allocated_ = 0;
  } else {
    view_bindful_heap_pool_ = std::make_unique<ui::d3d12::D3D12DescriptorHeapPool>(
        device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kViewBindfulHeapSize);
    sampler_bindful_heap_pool_ = std::make_unique<ui::d3d12::D3D12DescriptorHeapPool>(
        device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, kSamplerHeapSize);
  }

  if (bindless_resources_used_) {
    // Global bindless resource root signatures.
    // No CBV or UAV descriptor ranges with any descriptors to be allocated
    // dynamically (via RequestPersistentViewBindlessDescriptor or
    // RequestOneUseSingleViewDescriptors) should be here, because they would
    // overlap the unbounded SRV range, which is not allowed on Nvidia Fermi!
    D3D12_ROOT_SIGNATURE_DESC root_signature_bindless_desc;
    D3D12_ROOT_PARAMETER
    root_parameters_bindless[kRootParameter_Bindless_Count];
    root_signature_bindless_desc.NumParameters = kRootParameter_Bindless_Count;
    root_signature_bindless_desc.pParameters = root_parameters_bindless;
    root_signature_bindless_desc.NumStaticSamplers = 0;
    root_signature_bindless_desc.pStaticSamplers = nullptr;
    root_signature_bindless_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    // Fetch constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_FetchConstants];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kFetchConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    // Vertex float constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_FloatConstantsVertex];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kFloatConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    }
    // Pixel float constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_FloatConstantsPixel];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kFloatConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    // Pixel shader descriptor indices.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_DescriptorIndicesPixel];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kDescriptorIndices);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    // Vertex shader descriptor indices.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_DescriptorIndicesVertex];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kDescriptorIndices);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    }
    // System constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_SystemConstants];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kSystemConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    // Bool and loop constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_BoolLoopConstants];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kBoolLoopConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    // Shared memory SRV and UAV.
    D3D12_DESCRIPTOR_RANGE root_shared_memory_view_ranges[2];
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_SharedMemory];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      parameter.DescriptorTable.NumDescriptorRanges =
          uint32_t(rex::countof(root_shared_memory_view_ranges));
      parameter.DescriptorTable.pDescriptorRanges = root_shared_memory_view_ranges;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      {
        auto& range = root_shared_memory_view_ranges[0];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = UINT(DxbcShaderTranslator::SRVMainRegister::kSharedMemory);
        range.RegisterSpace = UINT(DxbcShaderTranslator::SRVSpace::kMain);
        range.OffsetInDescriptorsFromTableStart = 0;
      }
      {
        auto& range = root_shared_memory_view_ranges[1];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = UINT(DxbcShaderTranslator::UAVRegister::kSharedMemory);
        range.RegisterSpace = 0;
        range.OffsetInDescriptorsFromTableStart = 1;
      }
    }
    // Sampler heap.
    D3D12_DESCRIPTOR_RANGE root_bindless_sampler_range;
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_SamplerHeap];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      // Will be appending.
      parameter.DescriptorTable.NumDescriptorRanges = 1;
      parameter.DescriptorTable.pDescriptorRanges = &root_bindless_sampler_range;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      root_bindless_sampler_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
      root_bindless_sampler_range.NumDescriptors = UINT_MAX;
      root_bindless_sampler_range.BaseShaderRegister = 0;
      root_bindless_sampler_range.RegisterSpace = 0;
      root_bindless_sampler_range.OffsetInDescriptorsFromTableStart = 0;
    }
    // View heap.
    D3D12_DESCRIPTOR_RANGE root_bindless_view_ranges[4];
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_ViewHeap];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      // Will be appending.
      parameter.DescriptorTable.NumDescriptorRanges = 0;
      parameter.DescriptorTable.pDescriptorRanges = root_bindless_view_ranges;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      // EDRAM.
      if (render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock) {
        assert_true(parameter.DescriptorTable.NumDescriptorRanges <
                    rex::countof(root_bindless_view_ranges));
        auto& range = root_bindless_view_ranges[parameter.DescriptorTable.NumDescriptorRanges++];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = UINT(DxbcShaderTranslator::UAVRegister::kEdram);
        range.RegisterSpace = 0;
        range.OffsetInDescriptorsFromTableStart = UINT(SystemBindlessView::kEdramR32UintUAV);
      }
      // Used UAV and SRV ranges must not overlap on Nvidia Fermi, so textures
      // have OffsetInDescriptorsFromTableStart after all static descriptors of
      // other types.
      // 2D array textures.
      {
        assert_true(parameter.DescriptorTable.NumDescriptorRanges <
                    rex::countof(root_bindless_view_ranges));
        auto& range = root_bindless_view_ranges[parameter.DescriptorTable.NumDescriptorRanges++];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = UINT_MAX;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = UINT(DxbcShaderTranslator::SRVSpace::kBindlessTextures2DArray);
        range.OffsetInDescriptorsFromTableStart = UINT(SystemBindlessView::kUnboundedSRVsStart);
      }
      // 3D textures.
      {
        assert_true(parameter.DescriptorTable.NumDescriptorRanges <
                    rex::countof(root_bindless_view_ranges));
        auto& range = root_bindless_view_ranges[parameter.DescriptorTable.NumDescriptorRanges++];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = UINT_MAX;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = UINT(DxbcShaderTranslator::SRVSpace::kBindlessTextures3D);
        range.OffsetInDescriptorsFromTableStart = UINT(SystemBindlessView::kUnboundedSRVsStart);
      }
      // Cube textures.
      {
        assert_true(parameter.DescriptorTable.NumDescriptorRanges <
                    rex::countof(root_bindless_view_ranges));
        auto& range = root_bindless_view_ranges[parameter.DescriptorTable.NumDescriptorRanges++];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = UINT_MAX;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = UINT(DxbcShaderTranslator::SRVSpace::kBindlessTexturesCube);
        range.OffsetInDescriptorsFromTableStart = UINT(SystemBindlessView::kUnboundedSRVsStart);
      }
    }
    // [hiz-pool] Instance base root constant (instancing vertex shaders).
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_InstanceBase];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      parameter.Constants.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kInstanceBase);
      parameter.Constants.RegisterSpace = 0;
      parameter.Constants.Num32BitValues = 1;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    }
    uint64_t root_signature_bindless_hash = 0;
    root_signature_bindless_vs_ = ui::d3d12::util::CreateRootSignature(
        provider, root_signature_bindless_desc, &root_signature_bindless_hash);
    if (!root_signature_bindless_vs_) {
      REXGPU_ERROR(
          "Failed to create the global root signature for bindless resources, "
          "the version for use without tessellation");
      return false;
    }
    root_parameters_bindless[kRootParameter_Bindless_FloatConstantsVertex].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_DOMAIN;
    root_parameters_bindless[kRootParameter_Bindless_DescriptorIndicesVertex].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_DOMAIN;
    root_signature_hashes_.emplace(root_signature_bindless_vs_, root_signature_bindless_hash);
    root_signature_bindless_ds_ = ui::d3d12::util::CreateRootSignature(
        provider, root_signature_bindless_desc, &root_signature_bindless_hash);
    if (!root_signature_bindless_ds_) {
      REXGPU_ERROR(
          "Failed to create the global root signature for bindless resources, "
          "the version for use with tessellation");
      return false;
    }
    root_signature_hashes_.emplace(root_signature_bindless_ds_, root_signature_bindless_hash);
  }

  primitive_processor_ = std::make_unique<D3D12PrimitiveProcessor>(
      *register_file_, *memory_, trace_writer_, *shared_memory_, *this);
  if (!primitive_processor_->Initialize()) {
    REXGPU_ERROR("Failed to initialize the geometric primitive processor");
    return false;
  }

  texture_cache_ =
      D3D12TextureCache::Create(*register_file_, *shared_memory_, draw_resolution_scale_x,
                                draw_resolution_scale_y, *this, bindless_resources_used_);
  if (!texture_cache_) {
    REXGPU_ERROR("Failed to initialize the texture cache");
    return false;
  }

  pipeline_cache_ = std::make_unique<PipelineCache>(
      *this, *register_file_, *render_target_cache_.get(), bindless_resources_used_);
  if (!pipeline_cache_->Initialize()) {
    REXGPU_ERROR("Failed to initialize the graphics pipeline cache");
    return false;
  }

  D3D12_HEAP_FLAGS heap_flag_create_not_zeroed = provider.GetHeapFlagCreateNotZeroed();

  // Create gamma ramp resources.
  gamma_ramp_256_entry_table_up_to_date_ = false;
  gamma_ramp_pwl_up_to_date_ = false;
  D3D12_RESOURCE_DESC gamma_ramp_buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(gamma_ramp_buffer_desc, (256 + 128 * 3) * 4,
                                          D3D12_RESOURCE_FLAG_NONE);
  // The first action will be uploading.
  gamma_ramp_buffer_state_ = D3D12_RESOURCE_STATE_COPY_DEST;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             heap_flag_create_not_zeroed, &gamma_ramp_buffer_desc,
                                             gamma_ramp_buffer_state_, nullptr,
                                             IID_PPV_ARGS(&gamma_ramp_buffer_)))) {
    REXGPU_ERROR("Failed to create the gamma ramp buffer");
    return false;
  }
  // The upload buffer is frame-buffered.
  gamma_ramp_buffer_desc.Width *= kQueueFrames;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesUpload,
                                             heap_flag_create_not_zeroed, &gamma_ramp_buffer_desc,
                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                             IID_PPV_ARGS(&gamma_ramp_upload_buffer_)))) {
    REXGPU_ERROR("Failed to create the gamma ramp upload buffer");
    return false;
  }
  if (FAILED(gamma_ramp_upload_buffer_->Map(
          0, nullptr, reinterpret_cast<void**>(&gamma_ramp_upload_buffer_mapping_)))) {
    REXGPU_ERROR("Failed to map the gamma ramp upload buffer");
    gamma_ramp_upload_buffer_mapping_ = nullptr;
    return false;
  }

  // Initialize compute pipelines for output with gamma ramp.
  D3D12_ROOT_PARAMETER
  apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kCount)];
  {
    D3D12_ROOT_PARAMETER& apply_gamma_root_parameter_constants =
        apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kConstants)];
    apply_gamma_root_parameter_constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    apply_gamma_root_parameter_constants.Constants.ShaderRegister = 0;
    apply_gamma_root_parameter_constants.Constants.RegisterSpace = 0;
    apply_gamma_root_parameter_constants.Constants.Num32BitValues =
        sizeof(ApplyGammaConstants) / sizeof(uint32_t);
    apply_gamma_root_parameter_constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE apply_gamma_root_descriptor_range_dest;
  apply_gamma_root_descriptor_range_dest.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  apply_gamma_root_descriptor_range_dest.NumDescriptors = 1;
  apply_gamma_root_descriptor_range_dest.BaseShaderRegister = 0;
  apply_gamma_root_descriptor_range_dest.RegisterSpace = 0;
  apply_gamma_root_descriptor_range_dest.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& apply_gamma_root_parameter_dest =
        apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kDestination)];
    apply_gamma_root_parameter_dest.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    apply_gamma_root_parameter_dest.DescriptorTable.NumDescriptorRanges = 1;
    apply_gamma_root_parameter_dest.DescriptorTable.pDescriptorRanges =
        &apply_gamma_root_descriptor_range_dest;
    apply_gamma_root_parameter_dest.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE apply_gamma_root_descriptor_range_source;
  apply_gamma_root_descriptor_range_source.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  apply_gamma_root_descriptor_range_source.NumDescriptors = 1;
  apply_gamma_root_descriptor_range_source.BaseShaderRegister = 1;
  apply_gamma_root_descriptor_range_source.RegisterSpace = 0;
  apply_gamma_root_descriptor_range_source.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& apply_gamma_root_parameter_source =
        apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kSource)];
    apply_gamma_root_parameter_source.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    apply_gamma_root_parameter_source.DescriptorTable.NumDescriptorRanges = 1;
    apply_gamma_root_parameter_source.DescriptorTable.pDescriptorRanges =
        &apply_gamma_root_descriptor_range_source;
    apply_gamma_root_parameter_source.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE apply_gamma_root_descriptor_range_ramp;
  apply_gamma_root_descriptor_range_ramp.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  apply_gamma_root_descriptor_range_ramp.NumDescriptors = 1;
  apply_gamma_root_descriptor_range_ramp.BaseShaderRegister = 0;
  apply_gamma_root_descriptor_range_ramp.RegisterSpace = 0;
  apply_gamma_root_descriptor_range_ramp.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& apply_gamma_root_parameter_gamma_ramp =
        apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kRamp)];
    apply_gamma_root_parameter_gamma_ramp.ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    apply_gamma_root_parameter_gamma_ramp.DescriptorTable.NumDescriptorRanges = 1;
    apply_gamma_root_parameter_gamma_ramp.DescriptorTable.pDescriptorRanges =
        &apply_gamma_root_descriptor_range_ramp;
    apply_gamma_root_parameter_gamma_ramp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_ROOT_SIGNATURE_DESC apply_gamma_root_signature_desc;
  apply_gamma_root_signature_desc.NumParameters = UINT(ApplyGammaRootParameter::kCount);
  apply_gamma_root_signature_desc.pParameters = apply_gamma_root_parameters;
  apply_gamma_root_signature_desc.NumStaticSamplers = 0;
  apply_gamma_root_signature_desc.pStaticSamplers = nullptr;
  apply_gamma_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(apply_gamma_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, apply_gamma_root_signature_desc);
  if (!apply_gamma_root_signature_) {
    REXGPU_ERROR("Failed to create the gamma ramp application root signature");
    return false;
  }
  *(apply_gamma_table_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::apply_gamma_table_cs, sizeof(shaders::apply_gamma_table_cs),
      apply_gamma_root_signature_.Get());
  if (!apply_gamma_table_pipeline_) {
    REXGPU_ERROR(
        "Failed to create the 256-entry table gamma ramp application compute "
        "pipeline");
    return false;
  }
  *(apply_gamma_table_fxaa_luma_pipeline_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateComputePipeline(device, shaders::apply_gamma_table_fxaa_luma_cs,
                                             sizeof(shaders::apply_gamma_table_fxaa_luma_cs),
                                             apply_gamma_root_signature_.Get());
  if (!apply_gamma_table_fxaa_luma_pipeline_) {
    REXGPU_ERROR(
        "Failed to create the 256-entry table gamma ramp application compute "
        "pipeline with perceptual luma output");
    return false;
  }
  *(apply_gamma_pwl_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::apply_gamma_pwl_cs, sizeof(shaders::apply_gamma_pwl_cs),
      apply_gamma_root_signature_.Get());
  if (!apply_gamma_pwl_pipeline_) {
    REXGPU_ERROR("Failed to create the PWL gamma ramp application compute pipeline");
    return false;
  }
  *(apply_gamma_pwl_fxaa_luma_pipeline_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateComputePipeline(device, shaders::apply_gamma_pwl_fxaa_luma_cs,
                                             sizeof(shaders::apply_gamma_pwl_fxaa_luma_cs),
                                             apply_gamma_root_signature_.Get());
  if (!apply_gamma_pwl_fxaa_luma_pipeline_) {
    REXGPU_ERROR(
        "Failed to create the PWL gamma ramp application compute pipeline with "
        "perceptual luma output");
    return false;
  }

  // Initialize compute pipelines for post-processing anti-aliasing.
  D3D12_ROOT_PARAMETER fxaa_root_parameters[UINT(FxaaRootParameter::kCount)];
  {
    D3D12_ROOT_PARAMETER& fxaa_root_parameter_constants =
        fxaa_root_parameters[UINT(ApplyGammaRootParameter::kConstants)];
    fxaa_root_parameter_constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    fxaa_root_parameter_constants.Constants.ShaderRegister = 0;
    fxaa_root_parameter_constants.Constants.RegisterSpace = 0;
    fxaa_root_parameter_constants.Constants.Num32BitValues =
        sizeof(FxaaConstants) / sizeof(uint32_t);
    fxaa_root_parameter_constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE fxaa_root_descriptor_range_dest;
  fxaa_root_descriptor_range_dest.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  fxaa_root_descriptor_range_dest.NumDescriptors = 1;
  fxaa_root_descriptor_range_dest.BaseShaderRegister = 0;
  fxaa_root_descriptor_range_dest.RegisterSpace = 0;
  fxaa_root_descriptor_range_dest.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& fxaa_root_parameter_dest =
        fxaa_root_parameters[UINT(FxaaRootParameter::kDestination)];
    fxaa_root_parameter_dest.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    fxaa_root_parameter_dest.DescriptorTable.NumDescriptorRanges = 1;
    fxaa_root_parameter_dest.DescriptorTable.pDescriptorRanges = &fxaa_root_descriptor_range_dest;
    fxaa_root_parameter_dest.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE fxaa_root_descriptor_range_source;
  fxaa_root_descriptor_range_source.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  fxaa_root_descriptor_range_source.NumDescriptors = 1;
  fxaa_root_descriptor_range_source.BaseShaderRegister = 0;
  fxaa_root_descriptor_range_source.RegisterSpace = 0;
  fxaa_root_descriptor_range_source.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& fxaa_root_parameter_source =
        fxaa_root_parameters[UINT(FxaaRootParameter::kSource)];
    fxaa_root_parameter_source.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    fxaa_root_parameter_source.DescriptorTable.NumDescriptorRanges = 1;
    fxaa_root_parameter_source.DescriptorTable.pDescriptorRanges =
        &fxaa_root_descriptor_range_source;
    fxaa_root_parameter_source.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_STATIC_SAMPLER_DESC fxaa_root_sampler;
  fxaa_root_sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
  fxaa_root_sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  fxaa_root_sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  fxaa_root_sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  fxaa_root_sampler.MipLODBias = 0.0f;
  fxaa_root_sampler.MaxAnisotropy = 1;
  fxaa_root_sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
  fxaa_root_sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
  fxaa_root_sampler.MinLOD = 0.0f;
  fxaa_root_sampler.MaxLOD = 0.0f;
  fxaa_root_sampler.ShaderRegister = 0;
  fxaa_root_sampler.RegisterSpace = 0;
  fxaa_root_sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC fxaa_root_signature_desc;
  fxaa_root_signature_desc.NumParameters = UINT(FxaaRootParameter::kCount);
  fxaa_root_signature_desc.pParameters = fxaa_root_parameters;
  fxaa_root_signature_desc.NumStaticSamplers = 1;
  fxaa_root_signature_desc.pStaticSamplers = &fxaa_root_sampler;
  fxaa_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(fxaa_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, fxaa_root_signature_desc);
  if (!fxaa_root_signature_) {
    REXGPU_ERROR("Failed to create the FXAA root signature");
    return false;
  }
  *(fxaa_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::fxaa_cs, sizeof(shaders::fxaa_cs), fxaa_root_signature_.Get());
  if (!fxaa_pipeline_) {
    REXGPU_ERROR("Failed to create the FXAA compute pipeline");
    return false;
  }
  *(fxaa_extreme_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::fxaa_extreme_cs, sizeof(shaders::fxaa_extreme_cs),
      fxaa_root_signature_.Get());
  if (!fxaa_pipeline_) {
    REXGPU_ERROR("Failed to create the extreme-quality FXAA compute pipeline");
    return false;
  }

  // Resolve downscale compute pipeline for scaled readback resolve.
  D3D12_ROOT_PARAMETER
  resolve_downscale_root_parameters[UINT(ResolveDownscaleRootParameter::kCount)];
  {
    D3D12_ROOT_PARAMETER& constants_parameter =
        resolve_downscale_root_parameters[UINT(ResolveDownscaleRootParameter::kConstants)];
    constants_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    constants_parameter.Constants.ShaderRegister = 0;
    constants_parameter.Constants.RegisterSpace = 0;
    constants_parameter.Constants.Num32BitValues =
        sizeof(ResolveDownscaleConstants) / sizeof(uint32_t);
    constants_parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE resolve_downscale_source_range;
  resolve_downscale_source_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  resolve_downscale_source_range.NumDescriptors = 1;
  resolve_downscale_source_range.BaseShaderRegister = 0;
  resolve_downscale_source_range.RegisterSpace = 0;
  resolve_downscale_source_range.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& source_parameter =
        resolve_downscale_root_parameters[UINT(ResolveDownscaleRootParameter::kSource)];
    source_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    source_parameter.DescriptorTable.NumDescriptorRanges = 1;
    source_parameter.DescriptorTable.pDescriptorRanges = &resolve_downscale_source_range;
    source_parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE resolve_downscale_destination_range;
  resolve_downscale_destination_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  resolve_downscale_destination_range.NumDescriptors = 1;
  resolve_downscale_destination_range.BaseShaderRegister = 0;
  resolve_downscale_destination_range.RegisterSpace = 0;
  resolve_downscale_destination_range.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& destination_parameter =
        resolve_downscale_root_parameters[UINT(ResolveDownscaleRootParameter::kDestination)];
    destination_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    destination_parameter.DescriptorTable.NumDescriptorRanges = 1;
    destination_parameter.DescriptorTable.pDescriptorRanges = &resolve_downscale_destination_range;
    destination_parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_ROOT_SIGNATURE_DESC resolve_downscale_root_signature_desc;
  resolve_downscale_root_signature_desc.NumParameters = UINT(ResolveDownscaleRootParameter::kCount);
  resolve_downscale_root_signature_desc.pParameters = resolve_downscale_root_parameters;
  resolve_downscale_root_signature_desc.NumStaticSamplers = 0;
  resolve_downscale_root_signature_desc.pStaticSamplers = nullptr;
  resolve_downscale_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(resolve_downscale_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, resolve_downscale_root_signature_desc);
  if (resolve_downscale_root_signature_) {
    *(resolve_downscale_pipeline_.ReleaseAndGetAddressOf()) =
        ui::d3d12::util::CreateComputePipeline(device, shaders::resolve_downscale_cs,
                                               sizeof(shaders::resolve_downscale_cs),
                                               resolve_downscale_root_signature_.Get());
  }
  if (!resolve_downscale_root_signature_ || !resolve_downscale_pipeline_) {
    resolve_downscale_pipeline_.Reset();
    resolve_downscale_root_signature_.Reset();
    REXGPU_WARN("Failed to initialize D3D12 resolve-downscale readback pipeline");
  }

  if (bindless_resources_used_) {
    // Create the system bindless descriptors once all resources are
    // initialized.
    // kNullRawSRV.
    ui::d3d12::util::CreateBufferRawSRV(
        device,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullRawSRV)),
        nullptr, 0);
    // kNullRawUAV.
    ui::d3d12::util::CreateBufferRawUAV(
        device,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullRawUAV)),
        nullptr, 0);
    // kNullTexture2DArray.
    D3D12_SHADER_RESOURCE_VIEW_DESC null_srv_desc;
    null_srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    null_srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
        D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
        D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0);
    null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    null_srv_desc.Texture2DArray.MostDetailedMip = 0;
    null_srv_desc.Texture2DArray.MipLevels = 1;
    null_srv_desc.Texture2DArray.FirstArraySlice = 0;
    null_srv_desc.Texture2DArray.ArraySize = 1;
    null_srv_desc.Texture2DArray.PlaneSlice = 0;
    null_srv_desc.Texture2DArray.ResourceMinLODClamp = 0.0f;
    device->CreateShaderResourceView(
        nullptr, &null_srv_desc,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullTexture2DArray)));
    // kNullTexture3D.
    null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    null_srv_desc.Texture3D.MostDetailedMip = 0;
    null_srv_desc.Texture3D.MipLevels = 1;
    null_srv_desc.Texture3D.ResourceMinLODClamp = 0.0f;
    device->CreateShaderResourceView(
        nullptr, &null_srv_desc,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullTexture3D)));
    // kNullTextureCube.
    null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    null_srv_desc.TextureCube.MostDetailedMip = 0;
    null_srv_desc.TextureCube.MipLevels = 1;
    null_srv_desc.TextureCube.ResourceMinLODClamp = 0.0f;
    device->CreateShaderResourceView(
        nullptr, &null_srv_desc,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullTextureCube)));
    // kSharedMemoryRawSRV.
    shared_memory_->WriteRawSRVDescriptor(provider.OffsetViewDescriptor(
        view_bindless_heap_cpu_start_, uint32_t(SystemBindlessView::kSharedMemoryRawSRV)));
    // kSharedMemoryR32UintSRV.
    shared_memory_->WriteUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kSharedMemoryR32UintSRV)),
        2);
    // kSharedMemoryR32G32UintSRV.
    shared_memory_->WriteUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kSharedMemoryR32G32UintSRV)),
        3);
    // kSharedMemoryR32G32B32A32UintSRV.
    shared_memory_->WriteUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(
            view_bindless_heap_cpu_start_,
            uint32_t(SystemBindlessView::kSharedMemoryR32G32B32A32UintSRV)),
        4);
    // kSharedMemoryRawUAV.
    shared_memory_->WriteRawUAVDescriptor(provider.OffsetViewDescriptor(
        view_bindless_heap_cpu_start_, uint32_t(SystemBindlessView::kSharedMemoryRawUAV)));
    // kSharedMemoryR32UintUAV.
    shared_memory_->WriteUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kSharedMemoryR32UintUAV)),
        2);
    // kSharedMemoryR32G32UintUAV.
    shared_memory_->WriteUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kSharedMemoryR32G32UintUAV)),
        3);
    // kSharedMemoryR32G32B32A32UintUAV.
    shared_memory_->WriteUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(
            view_bindless_heap_cpu_start_,
            uint32_t(SystemBindlessView::kSharedMemoryR32G32B32A32UintUAV)),
        4);
    // kEdramRawSRV.
    render_target_cache_->WriteEdramRawSRVDescriptor(provider.OffsetViewDescriptor(
        view_bindless_heap_cpu_start_, uint32_t(SystemBindlessView::kEdramRawSRV)));
    // kEdramR32UintSRV.
    render_target_cache_->WriteEdramUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32UintSRV)),
        2);
    // kEdramR32G32UintSRV.
    render_target_cache_->WriteEdramUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32G32UintSRV)),
        3);
    // kEdramR32G32B32A32UintSRV.
    render_target_cache_->WriteEdramUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32G32B32A32UintSRV)),
        4);
    // kEdramRawUAV.
    render_target_cache_->WriteEdramRawUAVDescriptor(provider.OffsetViewDescriptor(
        view_bindless_heap_cpu_start_, uint32_t(SystemBindlessView::kEdramRawUAV)));
    // kEdramR32UintUAV.
    render_target_cache_->WriteEdramUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32UintUAV)),
        2);
    // kEdramR32G32UintUAV.
    render_target_cache_->WriteEdramUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32G32UintUAV)),
        3);
    // kEdramR32G32B32A32UintUAV.
    render_target_cache_->WriteEdramUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32G32B32A32UintUAV)),
        4);
    // kGammaRampTableSRV.
    WriteGammaRampSRV(
        false, provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                             uint32_t(SystemBindlessView::kGammaRampTableSRV)));
    // kGammaRampPWLSRV.
    WriteGammaRampSRV(
        true, provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                            uint32_t(SystemBindlessView::kGammaRampPWLSRV)));
  }

  pix_capture_requested_.store(false, std::memory_order_relaxed);
  pix_capturing_ = false;
  occlusion_query_resources_available_ = InitializeOcclusionQueryResources();
  InitializeGpuCensusResources();
  InitializeOccCensusResources();
  nr_cull_.Initialize(memory_, shared_memory_.get());  // [cull]
  InitializeHizResources();                             // [hiz]

  // Just not to expose uninitialized memory.
  std::memset(&system_constants_, 0, sizeof(system_constants_));
  // [NR-LEAN] fresh member, nothing skipped yet.
  g_nr_sys_member_stale = false;

  return true;
}

void D3D12CommandProcessor::ShutdownContext() {
  // [GPU-PRECORD] Phase 1b-1b: stop the replay worker before tearing down the
  // subsystems it records into. No replay is in flight here (the worker only runs
  // while the parse thread is blocked inside PrecordFlush).
  PrecordWorkerShutdown();
  AwaitAllQueueOperationsCompletion();
  InvalidateAllVertexBufferResidency();
  ShutdownOcclusionQueryResources();
  ShutdownGpuCensusResources();
  ShutdownOccCensusResources();
  ShutdownHizResources();  // [hiz]
  nr_cull_.Shutdown();     // [cull]

  ui::d3d12::util::ReleaseAndNull(readback_buffer_);
  readback_buffer_size_ = 0;
  for (auto& resolve_readback_pair : readback_buffers_) {
    auto& readback = resolve_readback_pair.second;
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.buffers[i]) {
        if (readback.mapped_data[i]) {
          readback.buffers[i]->Unmap(0, nullptr);
          readback.mapped_data[i] = nullptr;
        }
        readback.buffers[i]->Release();
        readback.buffers[i] = nullptr;
      }
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
  }
  readback_buffers_.clear();
  for (auto& memexport_readback_pair : memexport_readback_buffers_) {
    auto& readback = memexport_readback_pair.second;
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.buffers[i]) {
        if (readback.mapped_data[i]) {
          readback.buffers[i]->Unmap(0, nullptr);
          readback.mapped_data[i] = nullptr;
        }
        readback.buffers[i]->Release();
        readback.buffers[i] = nullptr;
      }
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
  }
  memexport_readback_buffers_.clear();

  ui::d3d12::util::ReleaseAndNull(scratch_buffer_);
  scratch_buffer_size_ = 0;
  resolve_downscale_buffer_size_ = 0;
  resolve_downscale_buffer_.Reset();

  for (const std::pair<uint64_t, ID3D12Resource*>& resource_for_deletion :
       resources_for_deletion_) {
    resource_for_deletion.second->Release();
  }
  resources_for_deletion_.clear();

  fxaa_source_texture_submission_ = 0;
  fxaa_source_texture_.Reset();

  fxaa_extreme_pipeline_.Reset();
  fxaa_pipeline_.Reset();
  fxaa_root_signature_.Reset();
  resolve_downscale_pipeline_.Reset();
  resolve_downscale_root_signature_.Reset();

  apply_gamma_pwl_fxaa_luma_pipeline_.Reset();
  apply_gamma_pwl_pipeline_.Reset();
  apply_gamma_table_fxaa_luma_pipeline_.Reset();
  apply_gamma_table_pipeline_.Reset();
  apply_gamma_root_signature_.Reset();

  // Unmapping will be done implicitly by the destruction.
  gamma_ramp_upload_buffer_mapping_ = nullptr;
  gamma_ramp_upload_buffer_.Reset();
  gamma_ramp_buffer_.Reset();

  texture_cache_.reset();

  pipeline_cache_.reset();

  primitive_processor_.reset();

  // Shut down binding - bindless descriptors may be owned by subsystems like
  // the texture cache.

  // Root signatures are used by pipelines, thus freed after the pipelines.
  ui::d3d12::util::ReleaseAndNull(root_signature_bindless_ds_);
  ui::d3d12::util::ReleaseAndNull(root_signature_bindless_vs_);
  for (auto it : root_signatures_bindful_) {
    it.second->Release();
  }
  root_signatures_bindful_.clear();

  if (bindless_resources_used_) {
    texture_cache_bindless_sampler_map_.clear();
    for (const auto& sampler_bindless_heap_overflowed : sampler_bindless_heaps_overflowed_) {
      sampler_bindless_heap_overflowed.first->Release();
    }
    sampler_bindless_heaps_overflowed_.clear();
    sampler_bindless_heap_allocated_ = 0;
    ui::d3d12::util::ReleaseAndNull(sampler_bindless_heap_current_);
    view_bindless_one_use_descriptors_.clear();
    view_bindless_heap_free_.clear();
    ui::d3d12::util::ReleaseAndNull(view_bindless_heap_);
  } else {
    sampler_bindful_heap_pool_.reset();
    view_bindful_heap_pool_.reset();
  }
  constant_buffer_pool_.reset();

  render_target_cache_.reset();

  shared_memory_.reset();

  deferred_command_list_.Reset();
  ui::d3d12::util::ReleaseAndNull(command_list_1_);
  ui::d3d12::util::ReleaseAndNull(command_list_);
  ClearCommandAllocatorCache();

  frame_open_ = false;
  frame_current_ = 1;
  frame_completed_ = 0;
  std::memset(closed_frame_submissions_, 0, sizeof(closed_frame_submissions_));

  // First release the fences since they may reference fence_completion_event_.

  queue_operations_done_since_submission_signal_ = false;
  queue_operations_since_submission_fence_last_ = 0;
  ui::d3d12::util::ReleaseAndNull(queue_operations_since_submission_fence_);

  ui::d3d12::util::ReleaseAndNull(submission_fence_);
  submission_open_ = false;
  submission_current_ = 1;
  submission_completed_ = 0;

  if (fence_completion_event_) {
    CloseHandle(fence_completion_event_);
    fence_completion_event_ = nullptr;
  }

  device_removed_ = false;

  CommandProcessor::ShutdownContext();
}

// [GPU-PRECORD] Phase 1b-0: registers whose base WriteRegister has a
// non-idempotent side effect — a stateful sequence (the DC_LUT gamma-ramp write
// index/component machine) or a guest-memory / coherency effect (scratch
// writeback, COHER dirty bit). These must apply exactly once, live; deferring one
// into a replayed segment would double-apply it. (Same set the constant-dedupe
// path refuses to skip.) A write to one of these flushes the pending segment
// instead of being logged. Near-absent mid-frame in steady gameplay.
static inline bool PrecordRangeMustNotDefer(uint32_t start_index, uint32_t end_index) {
  auto overlaps = [&](uint32_t lo, uint32_t hi) { return start_index <= hi && end_index >= lo; };
  return overlaps(XE_GPU_REG_SCRATCH_REG0, XE_GPU_REG_SCRATCH_REG7) ||
         overlaps(XE_GPU_REG_COHER_STATUS_HOST, XE_GPU_REG_COHER_STATUS_HOST) ||
         overlaps(XE_GPU_REG_DC_LUT_RW_INDEX, XE_GPU_REG_DC_LUT_RW_INDEX) ||
         overlaps(XE_GPU_REG_DC_LUT_SEQ_COLOR, XE_GPU_REG_DC_LUT_SEQ_COLOR) ||
         overlaps(XE_GPU_REG_DC_LUT_PWL_DATA, XE_GPU_REG_DC_LUT_PWL_DATA) ||
         overlaps(XE_GPU_REG_DC_LUT_30_COLOR, XE_GPU_REG_DC_LUT_30_COLOR);
}

void D3D12CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  // [GPU-PRECORD] Phase 1b-0: while a deferred-draw segment is open, log this
  // write so PrecordFlush can replay it in order with the draws against the
  // rewound register file (reproducing the D3D12 cbuffer/texture invalidations).
  // A stateful register instead flushes the segment so it applies once, live.
  // [GPU-PRECORD] Phase 1b-1c Inc 5: gate on precord_segment_open_ ALONE (not also
  // !precord_replaying_) -- segment_open is false during every inline replay, so this
  // stays behavior-identical, but it keeps the now-thread_local replay flag off the hot
  // path AND ensures the parse thread's "should I defer?" test never reads the worker's
  // replay state under overlap. precord_deferred records that this write went into the
  // segment log, so the D3D12 side effects below can be skipped during pure overlap
  // capture (the worker reproduces them at replay via PrecordApplyWrite).
  bool precord_deferred = false;
  if (precord_segment_open_) {
    if (PrecordRangeMustNotDefer(index, index)) {
      PrecordFlush();
    } else {
      PrecordEvent ev;
      ev.kind = PrecordEvent::Kind::kWriteSingle;
      ev.a = index;
      ev.b = value;
      ev.c = 0;
      precord_slots_[precord_capture_slot_].events.push_back(ev);
      precord_deferred = true;
    }
  }

  // [GPU-INST] Track whether anything other than the vertex float constants
  // changes between draws. A value-changing write to any register outside the
  // vertex float-constant range breaks the "only the per-instance transform
  // differs" invariant, so the next draw must not merge into the open batch.
  // VGT_EVENT_INITIATOR is exempt: it is a write strobe (EVENT_WRITE family),
  // not state any draw consumes, and events already execute while a batch is
  // open (dirty never flushed the batch at the event) -- the city census
  // measured it as the top batch breaker (~88k/s, EVENT_WRITE/EXT alternate
  // initiator values every draw pair).
  if (g_instance && !g_instance_dirty &&
      (index < XE_GPU_REG_SHADER_CONSTANT_000_X || index >= XE_GPU_REG_SHADER_CONSTANT_256_X) &&
      index != XE_GPU_REG_VGT_EVENT_INITIATOR && register_file_->values[index] != value) {
    g_instance_dirty = true;
    g_instance_dirty_first_reg = index;
  }

  // [PERF/CONST-DEDUPE] A redundant shader-constant write (value unchanged)
  // leaves the GPU-side constant buffer already correct, so the only effect of
  // proceeding is marking the binding dirty -> a wasted full cbuffer re-upload
  // in UpdateBindings. Skip it for the pure constant ranges only (float and
  // bool/loop). Deliberately NOT applied to fetch constants (texture-cache /
  // vertex-residency invalidation is correctness-sensitive) or any non-constant
  // register (scratch writeback, COHER dirty bit, DC_LUT state machine).
  // [N8B] the per-dword half of the census rides the same test: mode 1 counts
  // without acting, mode 2 is the dedupe.
  if (g_n8b_mode &&
      ((index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
        index <= XE_GPU_REG_SHADER_CONSTANT_511_W) ||
       (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
        index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31))) {
    const int cls = index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 ? 1 : 0;
    ++g_n8b_pdw[cls];
    if (register_file_->values[index] == value) {
      ++g_n8b_pdw_same[cls];
      if (g_n8b_mode >= 2) {
        return;
      }
    }
  }

  // Value store into the shared register file (always -- the next segment's snapshot
  // must see it). Under overlap this is the ONLY thing pure capture does; the D3D12
  // side effects below belong to the worker at replay.
  CommandProcessor::WriteRegister(index, value);

  // [GPU-PRECORD] Phase 1b-1c Inc 5 (the crux): during overlap, a DEFERRED write's
  // D3D12 cbuffer/texture/vertex-residency invalidations must NOT run here on the parse
  // thread -- they touch shared subsystems (texture_cache_, the cbuffer bindings,
  // current_float_constant_map_) the worker owns while replaying an older segment.
  // They are reproduced against the repointed subsystems by PrecordApplyWrite at replay
  // time, so skipping them here keeps capture pure (H4/H5) with no lost effect. In every
  // non-overlap mode precord_deferred is treated as false below, so this is a no-op.
  if (!(g_precord_overlap && precord_deferred)) {
    if (index >= XE_GPU_REG_SHADER_CONSTANT_000_X && index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
      if (frame_open_) {
        uint32_t float_constant_index = (index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
        if (float_constant_index >= 256) {
          float_constant_index -= 256;
          if (current_float_constant_map_pixel_[float_constant_index >> 6] &
              (1ull << (float_constant_index & 63))) {
            cbuffer_binding_float_pixel_.up_to_date = false;
          }
        } else {
          if (current_float_constant_map_vertex_[float_constant_index >> 6] &
              (1ull << (float_constant_index & 63))) {
            cbuffer_binding_float_vertex_.up_to_date = false;
          }
        }
      }
    } else if (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
               index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
      cbuffer_binding_bool_loop_.up_to_date = false;
    } else if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
               index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
      cbuffer_binding_fetch_.up_to_date = false;
      if (texture_cache_ != nullptr) {
        texture_cache_->TextureFetchConstantWritten(
            (index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6);
      }
      InvalidateVertexBufferResidency((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 2);
    }
  }
}

void D3D12CommandProcessor::WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                                  uint32_t num_registers) {
  if (!num_registers) {
    return;
  }
  // [NR-5C] overlap tracking: the constant-window fast paths below never
  // reach the base per-dword WriteRegister, so stamp the whole range here
  // (the base path re-stamps per dword; redundant stores are harmless).
  // The N8B dedupe early-return skips value-identical ranges only, and an
  // unchanged range needs no stamp -- but it is stamped here anyway, which
  // is merely conservative for the skip lever.
  NrbStampRange(start_index, num_registers);
  if (g_skc_on) SkcWriteBE(start_index, base, num_registers, register_file_->values);  // [SKC]
  // [GPU-PRECORD] Phase 1b-0: log the bulk write (raw guest dwords) for replay.
  // See WriteRegister; offset into the segment's frommem_data is stored, not a pointer,
  // so the side buffer may grow without invalidating earlier events. A range that
  // touches a stateful register flushes the segment so it applies once, live.
  // [GPU-PRECORD] Phase 1b-1c Inc 5: gate on precord_segment_open_ alone + track
  // precord_deferred (see WriteRegister) so overlap capture skips the per-range D3D12
  // invalidations below (worker reproduces them via PrecordApplyWriteFromMem at replay).
  bool precord_deferred = false;
  if (precord_segment_open_) {
    if (PrecordRangeMustNotDefer(start_index, start_index + num_registers - 1)) {
      PrecordFlush();
    } else {
      PrecordEvent ev;
      ev.kind = PrecordEvent::Kind::kWriteFromMem;
      ev.a = start_index;
      ev.b = num_registers;
      PrecordSegment& seg = precord_slots_[precord_capture_slot_];
      ev.c = uint32_t(seg.frommem_data.size());
      seg.frommem_data.insert(seg.frommem_data.end(), base, base + num_registers);
      seg.events.push_back(ev);
      precord_deferred = true;
    }
  }
  uint32_t end_index = start_index + num_registers - 1;
  // [GPU-PRECORD] Phase 1b-1c Inc 5: skip the D3D12 binding/texture/residency
  // invalidations during pure overlap capture (the value copy_and_swap below still
  // runs -- the next segment's snapshot must see it). False in every non-overlap mode.
  const bool precord_skip_side_effects = g_precord_overlap && precord_deferred;

  // [GPU-INST] See WriteRegister: mark the open batch un-mergeable if a
  // value-changing write touches any register outside the vertex float-constant
  // range [SHADER_CONSTANT_000_X, SHADER_CONSTANT_256_X). Bulk constant uploads
  // confined to that range (the per-instance transform) are ignored.
  if (g_instance && !g_instance_dirty) {
    constexpr uint32_t kVsFloatLo = XE_GPU_REG_SHADER_CONSTANT_000_X;
    constexpr uint32_t kVsFloatHi = XE_GPU_REG_SHADER_CONSTANT_256_X;  // exclusive
    auto first_changed_in_range = [&](uint32_t lo, uint32_t hi) -> uint32_t {
      for (uint32_t idx = lo; idx <= hi; ++idx) {
        if (memory::load_and_swap<uint32_t>(base + (idx - start_index)) !=
            register_file_->values[idx]) {
          return idx;
        }
      }
      return UINT32_MAX;
    };
    uint32_t changed = UINT32_MAX;
    if (start_index < kVsFloatLo) {
      changed = first_changed_in_range(start_index, std::min(end_index, kVsFloatLo - 1));
    }
    if (changed == UINT32_MAX && end_index >= kVsFloatHi) {
      changed = first_changed_in_range(std::max(start_index, kVsFloatHi), end_index);
    }
    if (changed != UINT32_MAX) {
      g_instance_dirty = true;
      g_instance_dirty_first_reg = changed;
    }
  }

  auto range_has_any_constant_usage = [](const uint64_t* usage_map, uint32_t first_constant,
                                         uint32_t last_constant) -> bool {
    if (first_constant > last_constant) {
      return false;
    }
    uint32_t first_word = first_constant >> 6;
    uint32_t last_word = last_constant >> 6;
    uint32_t first_bit = first_constant & 63;
    uint32_t last_bit = last_constant & 63;
    if (first_word == last_word) {
      uint32_t bit_count = last_bit - first_bit + 1;
      uint64_t mask = bit_count == 64 ? UINT64_MAX : ((UINT64_C(1) << bit_count) - 1) << first_bit;
      return (usage_map[first_word] & mask) != 0;
    }
    if (usage_map[first_word] & (UINT64_MAX << first_bit)) {
      return true;
    }
    for (uint32_t word = first_word + 1; word < last_word; ++word) {
      if (usage_map[word]) {
        return true;
      }
    }
    uint64_t last_mask = last_bit == 63 ? UINT64_MAX : ((UINT64_C(1) << (last_bit + 1)) - 1);
    return (usage_map[last_word] & last_mask) != 0;
  };

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    // [PERF/CONST-DEDUPE] If the whole incoming block matches the register file,
    // the copy is a no-op and the cbuffer is already correct -> skip both the
    // copy and the invalidation. base[] is guest-endian, so swap-compare.
    // [N8B] ALU float constants: 82.7% of every applied dword. Census every dword (mode 1+); act on a
    // wholly unchanged range only (mode 2). The early return skips the copy
    // AND the cbuffer invalidation, which is the second copy.
    if (g_n8b_mode &&
        N8bCensusRange(0, base, register_file_->values + start_index, num_registers) &&
        g_n8b_mode >= 2) {
      ++g_n8b_skip_rng;
      g_n8b_skip_dw += num_registers;
      return;
    }
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    if (frame_open_ && !precord_skip_side_effects) {
      uint32_t first_float_constant = (start_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      uint32_t last_float_constant = (end_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (first_float_constant < 256) {
        uint32_t last_vertex_constant = std::min(last_float_constant, 255u);
        if (range_has_any_constant_usage(current_float_constant_map_vertex_, first_float_constant,
                                         last_vertex_constant)) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
      if (last_float_constant >= 256) {
        uint32_t first_pixel_constant =
            first_float_constant >= 256 ? first_float_constant - 256 : 0;
        uint32_t last_pixel_constant = last_float_constant - 256;
        if (range_has_any_constant_usage(current_float_constant_map_pixel_, first_pixel_constant,
                                         last_pixel_constant)) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      }
    }
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    // [PERF/CONST-DEDUPE] See float-range note above.
    // [N8B] bool/loop constants. Census every dword (mode 1+); act on a
    // wholly unchanged range only (mode 2). The early return skips the copy
    // AND the cbuffer invalidation, which is the second copy.
    if (g_n8b_mode &&
        N8bCensusRange(1, base, register_file_->values + start_index, num_registers) &&
        g_n8b_mode >= 2) {
      ++g_n8b_skip_rng;
      g_n8b_skip_dw += num_registers;
      return;
    }
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    if (!precord_skip_side_effects) {
      cbuffer_binding_bool_loop_.up_to_date = false;
    }
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    // [N8B] fetch constants are censused but NEVER deduped: the texture-cache
    // and vertex-buffer-residency invalidations below are correctness, not
    // bookkeeping, and an unchanged fetch word does not mean an unchanged
    // resource.
    if (g_n8b_mode) {
      N8bCensusRange(2, base, register_file_->values + start_index, num_registers);
    }
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    if (!precord_skip_side_effects) {
      cbuffer_binding_fetch_.up_to_date = false;
      uint32_t first_fetch_dword = start_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
      uint32_t last_fetch_dword = end_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
      if (texture_cache_) {
        texture_cache_->TextureFetchConstantsWritten(first_fetch_dword / 6, last_fetch_dword / 6);
      }
      InvalidateVertexBufferResidencyRange(first_fetch_dword / 2, last_fetch_dword / 2);
    }
    return;
  }

  CommandProcessor::WriteRegistersFromMem(start_index, base, num_registers);
}

// [NR-PB] N-2-2 item 0: a plain state range's ONLY D3D12 tail is the
// gpu_instance dirty check. The caller proved the range wholly outside the
// vertex float-constant window (plain ranges never intersect any constant
// window), so any value change breaks the open batch -- the same rule the
// WriteRegistersFromMem override applies outside [kVsFloatLo, kVsFloatHi).
// Precord capture cannot be live here (NrSkipBackendEligible vetoes it) and
// const-dedupe is constant-window-only, so neither branch is owed.
void D3D12CommandProcessor::WriteRegisterRangePlain(uint32_t base, uint32_t* values_be,
                                                    uint32_t n) {
  if (g_instance && !g_instance_dirty) {
    for (uint32_t i = 0; i < n; ++i) {
      // VGT_EVENT_INITIATOR keeps the per-dword path's write-strobe
      // exemption: the widening replaces that path, so it is the reference.
      if (base + i == XE_GPU_REG_VGT_EVENT_INITIATOR) continue;
      if (memory::load_and_swap<uint32_t>(values_be + i) != register_file_->values[base + i]) {
        g_instance_dirty = true;
        g_instance_dirty_first_reg = base + i;
        break;
      }
    }
  }
  CommandProcessor::WriteRegisterRangePlain(base, values_be, n);
}

// [N8F] The side-effect half of WriteRegistersFromMem, re-fired once per
// MERGED span at the draw stop by the coalescer. The value copy has already
// happened at decode time, so this only evaluates the dirty tails. Kept in
// lockstep with WriteRegistersFromMem above: any new per-range side effect
// added there must be added here or the coalescer goes stale. Precord and
// gpu_instance never coexist with the coalescer (NrCoalesceEligible latching
// vetoes both), and the caller guarantees the span is wholly inside one
// window (the coalescer refuses straddlers).
void D3D12CommandProcessor::NrApplyRangeEffects(uint32_t start_index, uint32_t n) {
  if (!n) return;
  const uint32_t end_index = start_index + n - 1;
  // Duplicated from WriteRegistersFromMem's local lambda (same precedent as
  // the precord replay copy): kept local so the hot path stays untouched.
  auto range_has_any_constant_usage = [](const uint64_t* usage_map, uint32_t first_constant,
                                         uint32_t last_constant) -> bool {
    if (first_constant > last_constant) {
      return false;
    }
    uint32_t first_word = first_constant >> 6;
    uint32_t last_word = last_constant >> 6;
    uint32_t first_bit = first_constant & 63;
    uint32_t last_bit = last_constant & 63;
    if (first_word == last_word) {
      uint32_t bit_count = last_bit - first_bit + 1;
      uint64_t mask = bit_count == 64 ? UINT64_MAX : ((UINT64_C(1) << bit_count) - 1) << first_bit;
      return (usage_map[first_word] & mask) != 0;
    }
    if (usage_map[first_word] & (UINT64_MAX << first_bit)) {
      return true;
    }
    for (uint32_t word = first_word + 1; word < last_word; ++word) {
      if (usage_map[word]) {
        return true;
      }
    }
    uint64_t last_mask = last_bit == 63 ? UINT64_MAX : ((UINT64_C(1) << (last_bit + 1)) - 1);
    return (usage_map[last_word] & last_mask) != 0;
  };

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    if (frame_open_) {
      uint32_t first_float_constant = (start_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      uint32_t last_float_constant = (end_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (first_float_constant < 256) {
        uint32_t last_vertex_constant = std::min(last_float_constant, 255u);
        if (range_has_any_constant_usage(current_float_constant_map_vertex_, first_float_constant,
                                         last_vertex_constant)) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
      if (last_float_constant >= 256) {
        uint32_t first_pixel_constant =
            first_float_constant >= 256 ? first_float_constant - 256 : 0;
        uint32_t last_pixel_constant = last_float_constant - 256;
        if (range_has_any_constant_usage(current_float_constant_map_pixel_, first_pixel_constant,
                                         last_pixel_constant)) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      }
    }
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    cbuffer_binding_bool_loop_.up_to_date = false;
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    cbuffer_binding_fetch_.up_to_date = false;
    uint32_t first_fetch_dword = start_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
    uint32_t last_fetch_dword = end_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
    if (texture_cache_) {
      texture_cache_->TextureFetchConstantsWritten(first_fetch_dword / 6, last_fetch_dword / 6);
    }
    InvalidateVertexBufferResidencyRange(first_fetch_dword / 2, last_fetch_dword / 2);
    return;
  }
  // Plain state: no D3D12 tail (gpu_instance is vetoed by the latch).
}

void D3D12CommandProcessor::OnGammaRamp256EntryTableValueWritten() {
  gamma_ramp_256_entry_table_up_to_date_ = false;
}

void D3D12CommandProcessor::OnGammaRampPWLValueWritten() {
  gamma_ramp_pwl_up_to_date_ = false;
}

void D3D12CommandProcessor::IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                                      uint32_t frontbuffer_height) {
  SCOPE_profile_cpu_f("gpu");

  // [N8B] Refresh the census/dedupe mode once per frame (cmd-proc thread only).
  // `gpu_n7 3` owns it while it is cycling the three phases in place; otherwise
  // the old gpu_dedupe_constants cvar is the switch and means "census + act".
  g_dedupe_constants = REXCVAR_GET(gpu_dedupe_constants);
  g_n8b_mode = rex::graphics::g_n7_n8b_phase >= 0 ? rex::graphics::g_n7_n8b_phase
                                                  : (g_dedupe_constants ? 2 : 0);
  // The census report. Unconditional whenever the census is armed -- it must NOT
  // hang off gpu_worker_profile, whose per-packet timing inflates the thing it
  // is reporting on by ~20% ([[nested-profile-inflates-stage]]).
  // [N8C] the second copy, reported in EVERY phase (including OFF) so the
  // dedupe's effect on it is readable, and reported separately from the
  // census so it does not disappear when the census is not armed.
  if (REXCVAR_GET(gpu_n7)) {
    static uint64_t l_up[4] = {}, l_binds = 0;
    static auto l_report = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (now - l_report >= std::chrono::seconds(1)) {
      const uint64_t b = g_n8c_binds - l_binds;
      const double per = b ? 1.0 / double(b) : 0.0;
      REXGPU_INFO(
          "[n8c] cbuffer uploads/s: float_v={} float_p={} bool_loop={} fetch={} "
          "| binds={} | per bind: {:.2f} {:.2f} {:.2f} {:.2f}",
          g_n8c_up[0] - l_up[0], g_n8c_up[1] - l_up[1], g_n8c_up[2] - l_up[2],
          g_n8c_up[3] - l_up[3], b, double(g_n8c_up[0] - l_up[0]) * per,
          double(g_n8c_up[1] - l_up[1]) * per, double(g_n8c_up[2] - l_up[2]) * per,
          double(g_n8c_up[3] - l_up[3]) * per);
      for (int c = 0; c < 4; ++c) l_up[c] = g_n8c_up[c];
      l_binds = g_n8c_binds;
      l_report = now;
    }
  }
  {
    static uint64_t l_rng[3] = {}, l_rng_same[3] = {}, l_dw[3] = {}, l_dw_same[3] = {};
    static uint64_t l_pdw[3] = {}, l_pdw_same[3] = {}, l_skip_rng = 0, l_skip_dw = 0;
    static auto l_report = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (g_n8b_mode && now - l_report >= std::chrono::seconds(1)) {
      const auto pct = [](uint64_t a, uint64_t b) {
        return b ? 100.0 * double(a) / double(b) : 0.0;
      };
      static const char* const kCls[3] = {"alu", "bool/loop", "fetch"};
      static const char* const kMode[3] = {"OFF", "CENSUS", "CENSUS+DEDUPE"};
      char line[512];
      int n = snprintf(line, sizeof(line), "[n8b] mode=%d %s |", g_n8b_mode,
                       kMode[g_n8b_mode < 3 ? g_n8b_mode : 2]);
      for (int c = 0; c < 3; ++c) {
        const uint64_t r = g_n8b_rng[c] - l_rng[c], rs = g_n8b_rng_same[c] - l_rng_same[c];
        const uint64_t d = g_n8b_dw[c] - l_dw[c], ds = g_n8b_dw_same[c] - l_dw_same[c];
        n += snprintf(line + n, sizeof(line) - n,
                      " %s: rng=%llu same=%.1f%% dw=%llu same=%.1f%% |", kCls[c],
                      (unsigned long long)r, pct(rs, r), (unsigned long long)d, pct(ds, d));
        if (n >= int(sizeof(line)) - 80) break;
      }
      REXGPU_INFO("{}", line);
      const uint64_t pd = (g_n8b_pdw[0] - l_pdw[0]) + (g_n8b_pdw[1] - l_pdw[1]);
      const uint64_t pds = (g_n8b_pdw_same[0] - l_pdw_same[0]) + (g_n8b_pdw_same[1] - l_pdw_same[1]);
      REXGPU_INFO("[n8b] per-dword applies={} same={:.1f}% | SKIPPED ranges={} dwords={}",
                  pd, pct(pds, pd), g_n8b_skip_rng - l_skip_rng, g_n8b_skip_dw - l_skip_dw);
      for (int c = 0; c < 3; ++c) {
        l_rng[c] = g_n8b_rng[c]; l_rng_same[c] = g_n8b_rng_same[c];
        l_dw[c] = g_n8b_dw[c]; l_dw_same[c] = g_n8b_dw_same[c];
        l_pdw[c] = g_n8b_pdw[c]; l_pdw_same[c] = g_n8b_pdw_same[c];
      }
      l_skip_rng = g_n8b_skip_rng; l_skip_dw = g_n8b_skip_dw;
      l_report = now;
    } else if (!g_n8b_mode) {
      // Keep the deltas honest across an OFF phase: nothing accumulated, but
      // the wall clock moved, so re-base the window rather than reporting a
      // multi-second sum on the next ON phase's first line.
      l_report = now;
      for (int c = 0; c < 3; ++c) {
        l_rng[c] = g_n8b_rng[c]; l_rng_same[c] = g_n8b_rng_same[c];
        l_dw[c] = g_n8b_dw[c]; l_dw_same[c] = g_n8b_dw_same[c];
        l_pdw[c] = g_n8b_pdw[c]; l_pdw_same[c] = g_n8b_pdw_same[c];
      }
      l_skip_rng = g_n8b_skip_rng; l_skip_dw = g_n8b_skip_dw;
    }
  }

  // [PERF/DRAW-PROFILE] Refresh the per-draw timing flag once per frame; report
  // a per-second breakdown of where IssueDraw's CPU time goes.
  g_draw_prof = REXCVAR_GET(gpu_draw_profile);
  g_tile_prof = REXCVAR_GET(gpu_nr_tile_prof);
  // [GPU-DRAW-DUMP] Refresh the draw-stream dump flag once per frame too.
  g_draw_dump = REXCVAR_GET(gpu_draw_dump);
  // [GPU-PRECORD] Phase 1a correctness-probe flag (forced segment-boundary re-emit).
  g_parallel_record = REXCVAR_GET(gpu_parallel_record);
  // [GPU-PRECORD] Phase 1b-0 capture/replay correctness-probe flag.
  g_precord_capture = REXCVAR_GET(gpu_precord_capture);
  // [GPU-PRECORD] Phase 1b-1b: local-register-file replay + worker (thread ⇒ localrf).
  g_precord_localrf = REXCVAR_GET(gpu_precord_localrf);
  g_precord_thread = REXCVAR_GET(gpu_precord_thread);
  // [GPU-PRECORD] Phase 1b-1c Inc 5: overlap flip (parse runs ahead of the worker).
  g_precord_overlap = REXCVAR_GET(gpu_precord_overlap);
  // [GPU-PRECORD H3-PROBE] Phase 1b-1c Inc 6: refresh the H3 diagnostic flag and, when
  // on, report the IB-overwrite rate once/sec (checked = replayed indexed draws with a
  // captured hash; mismatch = those whose guest IB changed before replay = H3 firing).
  g_h3_probe = REXCVAR_GET(gpu_precord_h3_probe);
  if (g_h3_probe) {
    static uint64_t s_h3_last_checked = 0, s_h3_last_mismatch = 0;
    static auto s_h3_last_report = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (now - s_h3_last_report >= std::chrono::seconds(1)) {
      static uint64_t s_h3_last_vb_checked = 0, s_h3_last_vb_mismatch = 0;
      uint64_t checked = g_h3_ib_checked.load(std::memory_order_relaxed);
      uint64_t mismatch = g_h3_ib_mismatch.load(std::memory_order_relaxed);
      uint64_t vb_checked = g_h3_vb_checked.load(std::memory_order_relaxed);
      uint64_t vb_mismatch = g_h3_vb_mismatch.load(std::memory_order_relaxed);
      uint64_t dt_checked = checked - s_h3_last_checked;
      uint64_t dt_mismatch = mismatch - s_h3_last_mismatch;
      uint64_t dt_vb_checked = vb_checked - s_h3_last_vb_checked;
      uint64_t dt_vb_mismatch = vb_mismatch - s_h3_last_vb_mismatch;
      REXGPU_INFO(
          "[gpu-h3] ib_checked={} ib_overwritten={} ({:.1f}%) | vb_checked={} "
          "vb_overwritten={} ({:.1f}%)",
          dt_checked, dt_mismatch,
          dt_checked ? 100.0 * double(dt_mismatch) / double(dt_checked) : 0.0, dt_vb_checked,
          dt_vb_mismatch,
          dt_vb_checked ? 100.0 * double(dt_vb_mismatch) / double(dt_vb_checked) : 0.0);
      s_h3_last_report = now;
      s_h3_last_checked = checked;
      s_h3_last_mismatch = mismatch;
      s_h3_last_vb_checked = vb_checked;
      s_h3_last_vb_mismatch = vb_mismatch;
    }
  }
  // [NR-PSO] Phase 5-1: the state mirror's verdict, once a second. Emitted
  // from here rather than from the per-draw check so no clock is read on the
  // draw path.
  if (pipeline_cache_) {
    pipeline_cache_->NrPsoReportIfDue();
    // [NR-SC] Phase 5-2: the shader cache's verdict, from the same place.
    pipeline_cache_->NrShaderCacheReportIfDue();
    // [NR-NPSO] Phase 5-3a: the native pipeline object's verdict, likewise.
    pipeline_cache_->NrNativePsoReportIfDue();
  }
  // [NR-NPSO] Phase 5-3a: latch the two cvars once a frame, so the per-draw
  // path reads a plain bool. Binding implies building.
  g_nr_native_pso_bind = REXCVAR_GET(gpu_nr_native_pso_bind);
  g_nr_native_pso = REXCVAR_GET(gpu_nr_native_pso) || g_nr_native_pso_bind;
  // [NR-VERIFY] inc 2: master verify latch. On a rising edge the mirrors
  // whose per-draw finish/compare passes were off are re-seeded from the
  // emulated state before any gate reads them again.
  {
    const bool nr_verify_now = REXCVAR_GET(gpu_nr_verify);
    if (nr_verify_now && !g_nr_verify) {
      g_nr_sys_seeded = false;
      if (g_nr_res) {
        NrResVfetchSeedFromEmulated();
        NrResPoolReseed();
      }
    }
    g_nr_verify = nr_verify_now;
  }
  // [NR-BND] Phase 5-3b-0: the bindings mirror's verdict + latch, same shape.
  // Compare-only (lives inside the emulated UpdateBindings): forced off by
  // the perf config.
  NrBindReportIfDue();
  g_nr_bindings = REXCVAR_GET(gpu_nr_bindings) && g_nr_verify;
  // [NR-SYS] Phase 5-3b-1: verdict + latch. A rising edge re-seeds the mirror
  // from the emulated struct (it kept evolving while the gate was off, and
  // its sticky never-derived fields are only knowable by seeding).
  NrSysReportIfDue();
  {
    const bool nr_sys_now = true;
    if (nr_sys_now && !g_nr_sysconst) {
      g_nr_sys_seeded = false;
    }
    g_nr_sysconst = nr_sys_now;
  }
  // [NR-DSC] Phase 5-3b-2: verdict + latch. A rising edge re-seeds the
  // sampler-heap mirror from the emulated allocation counter (entries
  // allocated while the gate was off are then learned as 'seeded') and
  // re-runs the one-shot bit-layout self-check.
  NrDescReportIfDue();
  {
    // [NR-VERIFY] inc 2: compare-only (sampler/key/heap-mirror checks inside
    // the emulated UpdateBindings; the swap uses the derivation code and the
    // SHARED allocator directly, not this probe): forced off by perf config.
    const bool nr_desc_now = REXCVAR_GET(gpu_nr_desc) && g_nr_verify;
    if (nr_desc_now && !g_nr_desc) {
      nr::DescSamplerMapReset(&g_nr_desc_smap, sampler_bindless_heap_allocated_);
      NrDescLayoutSelfCheck();
    }
    g_nr_desc = nr_desc_now;
  }
  // [NR-RSY] Phase 5-3b-3: verdict + latch. A rising edge seeds the vfetch
  // mirror and the view-pool mirror from the emulated state, resets the
  // per-texture descriptor map (pre-existing descriptors are then learned as
  // 'seeded'), installs the primitive processor's index-request observer and
  // re-runs the one-shot layout self-check. Bindless-only, like every gate
  // it feeds.
  NrResReportIfDue();
  {
    const bool nr_res_now = bindless_resources_used_;
    if (nr_res_now && !g_nr_res) {
      NrResVfetchSeedFromEmulated();
      NrResPoolReseed();
      nr::ResTexDescMapReset(&g_nr_res_tmap);
      if (primitive_processor_) {
        primitive_processor_->SetNrIndexRequestObserver(&NrResIndexObserverThunk, nullptr);
      }
      NrResLayoutSelfCheck();
    } else if (!nr_res_now && g_nr_res) {
      if (primitive_processor_) {
        primitive_processor_->SetNrIndexRequestObserver(nullptr, nullptr);
      }
    }
    g_nr_res = nr_res_now;
  }
  // [NR-SWP] Phase 5-3b swap: verdict + latch. Eligibility is evaluated here
  // once a frame: bindless only, and the mirrors the swap assembles from
  // must be armed and warm - the 5-3b-1 system-constants mirror (seeded,
  // still byte-checked by the sysconst gate every draw) and the 5-3b-3
  // texture-descriptor map (kept warm by the FindOrCreate hook).
  NrSwapReportIfDue();
  g_nr_swap = bindless_resources_used_ &&
              g_nr_sysconst && g_nr_sys_seeded && g_nr_res;
  // [NR-LEAN] 5-4-4b inc 2b: the lean sysconst path needs the swap (mirror is
  // the upload source) and verify off (the whole-struct memcmp gate reads the
  // member the lean path stops deriving).
  g_nr_lean_sys = g_nr_swap && !g_nr_verify;
  // [NR-RUB] 5-4-5-1: verdict + latch. Needs the swap (the gate instruments
  // OUR UpdateBindings). On arm, or after a fallback made the staging
  // mirrors stale, force every compose to re-run once so the staged copies
  // are the current effective packs again.
  NrRubReportIfDue();
  NrDspReportIfDue();
  NrSprReportIfDue();
  NrSpwReportIfDue();
  NrTilReportIfDue();
  {
    // [NR-RUF] 5-4-5-2: the fast path implies the bundle machinery (its
    // captures are the restore source); the compare gate stays available
    // independently. Fast-only runs capture the SMALL restore state and
    // never stage or copy pack bytes.
    const bool nr_ruf_now = REXCVAR_GET(gpu_nr_reuse_fast);
    const bool nr_cmp_now = REXCVAR_GET(gpu_nr_ruse_bundle) && g_nr_swap;
    const bool nr_rub_now = (nr_cmp_now || nr_ruf_now) && g_nr_swap;
    g_nr_rub_fast = nr_ruf_now && nr_rub_now;
    if (nr_cmp_now && (!g_nr_rub_cmp || !g_rub_stage_ok)) {
      cbuffer_binding_system_.up_to_date = false;
      cbuffer_binding_float_vertex_.up_to_date = false;
      cbuffer_binding_float_pixel_.up_to_date = false;
      cbuffer_binding_bool_loop_.up_to_date = false;
      cbuffer_binding_fetch_.up_to_date = false;
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
      cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
      g_rub_stage_ok = true;
    }
    g_nr_rub_cmp = nr_cmp_now;
    if (!nr_rub_now && g_nr_rub) {
      NrRubRelease();
    }
    g_nr_rub = nr_rub_now;
    // [NR-RUF-V2B] the upgrade rides the reuse machinery; whole-frame refuse
    // when the CPU vertex-shader extent path could read float constants
    // outside the bitmap-packed packs.
    g_nr_ruf_v2b = REXCVAR_GET(gpu_nr_reuse_v2b) && nr_rub_now &&
                   !REXCVAR_GET(execute_unclipped_draw_vs_on_cpu);
    if (g_nr_rub && g_rub_frame != frame_current_) {
      NrRubFrameReset(frame_current_);
    }
  }
  // [NR-SPW] 5-4-7-2: the span swap consumes the bundle restore (patch
  // addresses) and the recorded store, bindless-only, and must never run
  // under verify (verify = the 5-4-7-1 compare gate; consumption would make
  // every compare vacuous). Latched once per frame like the bindings swap.
  g_nr_span_consume = REXCVAR_GET(gpu_nr_span_swap) && !g_nr_verify &&
                      g_nr_swap && g_nr_rub_fast && bindless_resources_used_;
  // [NR-SPD] 5-4-7-3: latch the dedup mode once per frame; a flip clears the
  // whole store (deduped and context-free recordings are not comparable, and
  // a stale-mode recording consumed under the other mode's rules would be
  // silently wrong).
  {
    const bool nr_spd_now = REXCVAR_GET(gpu_nr_span_dedup);
    if (nr_spd_now != g_nr_span_dedup && !g_spr_headers.empty()) {
      std::fill(g_spr_headers.begin(), g_spr_headers.end(), SprHeader{});
    }
    g_nr_span_dedup = nr_spd_now;
  }
  // [NR-TIL] N-4-1: the tile-pass replay. Bindless only (the patch sites are
  // bindless root CBVs), never under verify (the emulated compare gates read
  // members the replay stops deriving), and never alongside the 5-4-7 span
  // machinery -- both own g_spr_cap and both would record the same brackets.
  // A flip drops every recording: the store is per-frame anyway.
  {
    const int32_t nr_til_cv = REXCVAR_GET(gpu_nr_tile_replay);
    const bool nr_til_eligible = nr_til_cv != 0 && !g_nr_verify && g_nr_swap &&
                                 bindless_resources_used_ &&
                                 !REXCVAR_GET(gpu_nr_span_replay) &&
                                 !REXCVAR_GET(gpu_nr_span_swap) &&
                                 !REXCVAR_GET(gpu_nr_tile_probe);
    bool nr_til_now = nr_til_eligible;
    if (nr_til_eligible && nr_til_cv == 2) {
      // Mode 2: cycle 10 s off / 10 s on IN PLACE. One drive to the heavy
      // city then reads both conditions at matched draws/frame, which is the
      // standing rule for every fps read here and the thing two drives to
      // "the same spot" can never guarantee.
      static const auto nr_til_t0 = std::chrono::steady_clock::now();
      const auto nr_til_el = std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::steady_clock::now() - nr_til_t0)
                                 .count();
      nr_til_now = ((nr_til_el / 10) & 1) != 0;
    }
    g_tile_verify = nr_til_eligible && nr_til_cv == 3;
    if (nr_til_now != g_nr_tile_on) {
      g_tile_recs.clear();
      g_tile_segs.clear();
      g_tile_arena_used = 0;
      g_tile_seg_open = false;
      g_tile_rec_open = false;
      g_tile_sys_prev = SIZE_MAX;
      g_tile_texsnaps.clear();
      g_tile_frame = UINT32_MAX;
      REXGPU_INFO(
          "[nr-til] PHASE {}{} - repeat EDRAM tile bands {}.",
          nr_til_now ? "ON" : "OFF", g_tile_verify ? " (COMPARE GATE)" : "",
          !nr_til_now
              ? "run the full derivation (control)"
              : (g_tile_verify
                     ? "run the full derivation AND compare it against the "
                       "recording they would have replayed"
                     : "are served from the base band recorded native spans "
                       "(viewport, scissor and NDC recomputed live)"));
    }
    g_nr_tile_on = nr_til_now;
    // [NR-TILP] the outer-stamp master latch (base TU reads it per draw):
    // arm only when a replay can actually serve, or the arm is pure cost.
    g_tile_oprof = g_tile_prof && g_nr_tile_on && !g_tile_verify;
    if (!g_tile_oprof) g_tile_ostop_armed = false;
    // [NR-TILSYS] only meaningful on the consuming path, and only while the
    // mirror is the uploaded truth (that is what the snapshot restores).
    uint32_t nr_sys_cv = (g_nr_tile_on && !g_tile_verify && g_nr_lean_sys)
                             ? uint32_t(REXCVAR_GET(gpu_nr_tile_sys))
                             : 0u;
    if (nr_sys_cv == 3) {
      // Mode 3: cycle 10 s off / 10 s on IN PLACE, so the [nr-tilp] sys
      // stage is read against itself at matched load. A cross-run stage
      // comparison is worthless here - every stage drifts together with
      // cache warmth (naruto_521 vs naruto_527 moved 12% on every stage).
      static const auto nr_sys_t0 = std::chrono::steady_clock::now();
      const auto nr_sys_el = std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::steady_clock::now() - nr_sys_t0)
                                 .count();
      nr_sys_cv = ((nr_sys_el / 10) & 1) != 0 ? 1u : 0u;
    }
    if (nr_sys_cv != g_tile_sys) {
      REXGPU_INFO("[nr-tilsys] PHASE {} - tile-replay system constants {}.",
                  nr_sys_cv ? "ON" : "OFF",
                  nr_sys_cv == 0 ? "re-derived in full (control)"
                                 : (nr_sys_cv == 2
                                        ? "derived fast AND compared against "
                                          "the full derivation"
                                        : "restored from the base band "
                                          "snapshot with the 3 viewport "
                                          "fields recomputed"));
    }
    g_tile_sys = nr_sys_cv;
    uint32_t nr_tex_cv = (g_nr_tile_on && !g_tile_verify)
                             ? uint32_t(REXCVAR_GET(gpu_nr_tile_tex))
                             : 0u;
    if (nr_tex_cv == 3) {
      // Mode 3: cycle 10 s off / 10 s on IN PLACE, for the same reason
      // gpu_nr_tile_sys has one - a cross-run stage comparison here is
      // noise, every stage drifts together.
      static const auto nr_tex_t0 = std::chrono::steady_clock::now();
      const auto nr_tex_el = std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::steady_clock::now() - nr_tex_t0)
                                 .count();
      // Recording must stay ON through both halves, or the ON half spends
      // its first frames with nothing to restore. 2 = record only.
      nr_tex_cv = ((nr_tex_el / 10) & 1) != 0 ? 1u : 4u;
    }
    if (nr_tex_cv != g_tile_tex) {
      REXGPU_INFO("[nr-tiltex] PHASE {} - tile-replay texture bindings {}.",
                  nr_tex_cv == 4 ? "OFF" : (nr_tex_cv ? "ON" : "IDLE"),
                  nr_tex_cv == 4 ? "re-derived in full (control; still "
                                   "recorded)"
                                 : "restored from the base band recording");
    }
    g_tile_tex = nr_tex_cv;
  }
  // [NR-FX] Phase 5-4-0: the walk-driven side-effect counters' 1Hz line. The
  // gate itself is latched base-side (WorkerThreadMain), not here.
  NrFxReportIfDue();
  // [INST-PROBE] Refresh + reset-on-arm the instancing feasibility probe.
  // Gated on the city latch (see g_inst_city) so the one-shot capture spends
  // its cap on city draws, not the menu.
  {
    const bool inst_now = REXCVAR_GET(gpu_instance_probe);
    if (g_frame_issued >= 2500) g_inst_city = true;
    const bool inst_active = inst_now && g_inst_city;
    if (inst_active && !g_inst_probe) {
      g_inst_keys.clear();
      g_inst_draw_count = 0;
      g_inst_dumped = false;
      REXGPU_INFO("[inst-probe] city latch hit - capture armed ({} draws last frame)",
                  g_frame_issued);
    }
    g_inst_probe = inst_active;
    g_frame_issued = 0;
  }
  // [GPU-INST] Refresh the instancing enable once/frame; report the coalescing
  // ratio per second when it is on. Any open batch from the previous frame is
  // flushed below before the present.
  g_instance = REXCVAR_GET(gpu_instance);
  if (g_instance) {
    static uint64_t s_last_in = 0, s_last_out = 0;
    static auto s_last_report = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (now - s_last_report >= std::chrono::seconds(1)) {
      uint64_t din = g_instance_draws_in - s_last_in;
      uint64_t dout = g_instance_draws_out - s_last_out;
      REXGPU_INFO("[gpu-inst] coalesced {} draws -> {} instanced draws (saved {})", din, dout,
                  din >= dout ? din - dout : 0);
      // [GPU-INST] Census line (per-second deltas; dirtyreg is cumulative
      // top-4 of the register that FIRST dirtied a batch that then failed to
      // merge). fail counts are merge refusals by reason; site counts are
      // where live batches got flushed; hist is flushed batch size.
      {
        static uint64_t s_lf[5] = {}, s_ls[3] = {}, s_lh[6] = {};
        uint64_t df[5], ds[3], dh[6];
        for (int i = 0; i < 5; ++i) { df[i] = g_inst_fail[i] - s_lf[i]; s_lf[i] = g_inst_fail[i]; }
        for (int i = 0; i < 3; ++i) { ds[i] = g_inst_flush_site[i] - s_ls[i]; s_ls[i] = g_inst_flush_site[i]; }
        for (int i = 0; i < 6; ++i) { dh[i] = g_inst_hist[i] - s_lh[i]; s_lh[i] = g_inst_hist[i]; }
        uint32_t top_reg[4] = {0, 0, 0, 0};
        uint64_t top_cnt[4] = {0, 0, 0, 0};
        for (const auto& kv : g_inst_dirty_reg_census) {
          for (int i = 0; i < 4; ++i) {
            if (kv.second > top_cnt[i]) {
              for (int j = 3; j > i; --j) { top_cnt[j] = top_cnt[j - 1]; top_reg[j] = top_reg[j - 1]; }
              top_cnt[i] = kv.second;
              top_reg[i] = kv.first;
              break;
            }
          }
        }
        REXGPU_INFO(
            "[gpu-inst] fail dirty={} sh={} prim={} ib={} cap={} | site merge={} copy={} end={} | "
            "hist 1={} 2-4={} 5-16={} 17-64={} 65-256={} 257+={} | dirtyreg {:04X}:{} {:04X}:{} "
            "{:04X}:{} {:04X}:{}",
            df[0], df[1], df[2], df[3], df[4], ds[0], ds[1], ds[2], dh[0], dh[1], dh[2], dh[3],
            dh[4], dh[5], top_reg[0], top_cnt[0], top_reg[1], top_cnt[1], top_reg[2], top_cnt[2],
            top_reg[3], top_cnt[3]);
      }
      s_last_report = now;
      s_last_in = g_instance_draws_in;
      s_last_out = g_instance_draws_out;
    }
  }
  // [GPU-PRECORD] Phase 1b-0: record the frame's pending captured draws before the
  // present commands so they keep their stream position.
  PrecordFlush();
  // [GPU-INST] Emit any draw still held in the open instanced batch before the
  // frame is presented.
  if (instanced_batch_.active) {
    ++g_inst_flush_site[2];
  }
  FlushInstancedBatch();
  if (g_draw_prof) {
    static uint64_t s_last[27] = {}, s_last_cnt = 0;
    static uint64_t s_last_res[5] = {0, 0, 0, 0, 0};
    static uint64_t s_last_bnd[7] = {}, s_last_bndc[8] = {};
    static uint64_t s_last_ret[6] = {};
    static auto s_last_report = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (now - s_last_report >= std::chrono::seconds(1)) {
      double d[27];
      for (int i = 0; i < 27; ++i) d[i] = (g_draw_ns[i] - s_last[i]) / 1e6;
      double b[7];
      for (int i = 0; i < 7; ++i) b[i] = (g_bind_ns[i] - s_last_bnd[i]) / 1e6;
      uint64_t dc = g_draw_count - s_last_cnt;
      double other = d[5] - (d[0] + d[1] + d[2] + d[3] + d[4] + d[6] + d[7] + d[8]);
      double rest = other - (d[12] + d[13] + d[14] + d[15] + d[16]);
      REXGPU_INFO(
          "[gpu-draw] draws={} total={:.1f}ms | begin={:.1f} prim={:.1f} rt={:.1f} pso={:.1f} "
          "tex={:.1f} ff+sc={:.1f} bind={:.1f} tail={:.1f} other={:.1f}ms "
          "| tail: res={:.1f} topo={:.1f} emit={:.1f} | other: head={:.1f} mods={:.1f} "
          "trans={:.1f} vp={:.1f} copy={:.1f} rest={:.1f} | vp: npso={:.1f} vpk={:.1f} "
          "sci={:.1f} | resn: vis={} sync={} "
          "rearm={} req={} dup={}",
          dc, d[5], d[6], d[0], d[1], d[2], d[3], d[8], d[4], d[7], other,
          d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16],
          rest, d[17], d[18], d[19],
          g_draw_res_cnt[0] - s_last_res[0],
          g_draw_res_cnt[1] - s_last_res[1], g_draw_res_cnt[2] - s_last_res[2],
          g_draw_res_cnt[3] - s_last_res[3], g_draw_res_cnt[4] - s_last_res[4]);
      REXGPU_INFO(
          "[gpu-draw2] rest: cen={:.1f} ruse={:.1f} pret={:.1f} rsp={:.1f} rply={:.1f} "
          "tilr={:.1f} bc={:.1f} rr={:.1f}ms | ret/s: tile={} span={} novs={} noeff={} "
          "novtx={} apso={}",
          d[20], d[21], d[22], d[23], d[24], d[25], d[26],
          rest - (d[20] + d[21] + d[22] + d[23] + d[24] + d[25] + d[26]),
          g_draw_ret[0] - s_last_ret[0], g_draw_ret[1] - s_last_ret[1],
          g_draw_ret[2] - s_last_ret[2], g_draw_ret[3] - s_last_ret[3],
          g_draw_ret[4] - s_last_ret[4], g_draw_ret[5] - s_last_ret[5]);
      REXGPU_INFO(
          "[nr-bndp] cb={:.1f} smp={:.1f} key={:.1f} smpidx={:.1f} di={:.1f} root={:.1f}ms "
          "rest={:.1f} reqns={:.1f} | evals={} srvv={} req={} | fires sys={} fv={} fp={} "
          "bl={} ft={}",
          b[0], b[1], b[2], b[3], b[4], b[5],
          d[4] - (b[0] + b[1] + b[2] + b[3] + b[4] + b[5]), b[6],
          g_bind_cnt[0] - s_last_bndc[0], g_bind_cnt[1] - s_last_bndc[1],
          g_bind_cnt[2] - s_last_bndc[2], g_bind_cnt[3] - s_last_bndc[3],
          g_bind_cnt[4] - s_last_bndc[4], g_bind_cnt[5] - s_last_bndc[5],
          g_bind_cnt[6] - s_last_bndc[6], g_bind_cnt[7] - s_last_bndc[7]);
      s_last_report = now;
      for (int i = 0; i < 27; ++i) s_last[i] = g_draw_ns[i];
      for (int i = 0; i < 5; ++i) s_last_res[i] = g_draw_res_cnt[i];
      for (int i = 0; i < 7; ++i) s_last_bnd[i] = g_bind_ns[i];
      for (int i = 0; i < 8; ++i) s_last_bndc[i] = g_bind_cnt[i];
      for (int i = 0; i < 6; ++i) s_last_ret[i] = g_draw_ret[i];
      s_last_cnt = g_draw_count;
    }
  }

  // [DRAW-POOL] frame end: close the window, advance the predictor's frame,
  // report at 1 Hz, latch the cvar / cycler phase for the next frame.
  {
    // [cull] phase latch (the cycle mode is deleted: the skip is shipped).
    {
      const int32_t cv = REXCVAR_GET(gpu_cull);
      cull_phase_ = cv < 0 || cv > 2 ? 0 : uint32_t(cv);
      // [hiz] the same latch shape; a phase change closes the open window
      // (its checkpoint carried the old mode).
      const int32_t hv = hiz_available_ ? REXCVAR_GET(gpu_hiz) : 0;
      const uint32_t k1 = uint32_t(std::clamp<int32_t>(REXCVAR_GET(gpu_hiz_k), 1, kHizMaxK));
      // Drive 815: rebuild 250 / 400 / 800 = 13.9 / 14.2 / 14.8 fps at the
      // heavy city (a build is ~1.3 ms, hidden 30.8 / 29.7 / 28.4%): 800.
      const uint32_t hrb = uint32_t(std::max<int32_t>(REXCVAR_GET(gpu_hiz_rebuild), 0));
      // Drive 827 (RTX 1440p --no-vsync, GPU-bound at ~20 ms/frame): skip
      // 47.5 / plain draws issued directly 47.8 / off 49.3 fps: the per-draw
      // ExecuteIndirect costs nothing, the Hi-Z is neutral on the RTX (its
      // win is the Intel's); the decomposition cycler is deleted.
      uint32_t hp = hv < 0 || hv > 2 ? 0 : uint32_t(hv);
      {  // [hiz-cycle] alternate the latched mode with off every N seconds.
        const int32_t cyc = REXCVAR_GET(gpu_hiz_cycle);
        if (cyc > 0 && hp != 0) {
          const auto now = std::chrono::steady_clock::now();
          if (hiz_cycle_start_ == std::chrono::steady_clock::time_point{}) {
            hiz_cycle_start_ = now;
          } else if (now - hiz_cycle_start_ >= std::chrono::seconds(cyc)) {
            hiz_cycle_start_ = now;
            hiz_cycle_off_ = !hiz_cycle_off_;
          }
          if (hiz_cycle_off_) hp = 0;
        } else {
          hiz_cycle_off_ = false;
          hiz_cycle_start_ = {};
        }
      }
      if (hp != hiz_phase_ || k1 != hiz_k_ || hrb != hiz_rebuild_) {
        if (hiz_window_.open) HizWindowClose(5);
        hiz_phase_ = hp;
        hiz_k_ = k1;
        hiz_rebuild_ = hrb;
      }
      hiz_frame_slots_ = 0;
    }
    if (g_pool_on) PoolClose(kPoolCloseFrame);
    ++g_pool_frame;
    ++g_pool.frames;
    static PoolStats s_pp;
    static auto s_pp_last = std::chrono::steady_clock::now();
    const auto pp_now = std::chrono::steady_clock::now();
    if (g_pool_on && pp_now - s_pp_last >= std::chrono::seconds(1)) {
      const double secs = std::chrono::duration<double>(pp_now - s_pp_last).count();
      s_pp_last = pp_now;
      const PoolStats& c = g_pool;
      auto d = [&](uint64_t PoolStats::*f) { return c.*f - s_pp.*f; };
      {  // [cmd] the per-frame command census (state changes per draw).
        using Cmd = DeferredCommandList::Command;
        static uint64_t s_cmd_last[64] = {};
        const double frames = double(std::max<uint64_t>(d(&PoolStats::frames), 1));
        auto cd = [&](Cmd k) {
          const uint64_t now = deferred_command_list_.command_count(k);
          const double r = double(now - s_cmd_last[size_t(k)]) / frames;
          s_cmd_last[size_t(k)] = now;
          return r;
        };
        REXGPU_INFO(
            "[cmd] /fr: pso {:.0f} psoh {:.0f} rs {:.0f} cbv {:.0f} c32 {:.0f} tbl {:.0f} srv {:.0f} "
            "heaps {:.0f} ib {:.0f} topo {:.0f} vp {:.0f} sc {:.0f} rt {:.0f} blend {:.0f} sref "
            "{:.0f} bar {:.0f} ei {:.0f} drawi {:.0f} draw {:.0f} disp {:.0f} clr {:.0f} copy {:.0f} "
            "q {:.0f}",
            cd(Cmd::kD3DSetPipelineState), cd(Cmd::kSetPipelineStateHandle),
            cd(Cmd::kD3DSetGraphicsRootSignature), cd(Cmd::kD3DSetGraphicsRootConstantBufferView),
            cd(Cmd::kD3DSetGraphicsRoot32BitConstants), cd(Cmd::kD3DSetGraphicsRootDescriptorTable),
            cd(Cmd::kD3DSetGraphicsRootShaderResourceView), cd(Cmd::kSetDescriptorHeaps),
            cd(Cmd::kD3DIASetIndexBuffer), cd(Cmd::kD3DIASetPrimitiveTopology), cd(Cmd::kRSSetViewport),
            cd(Cmd::kRSSetScissorRect), cd(Cmd::kD3DOMSetRenderTargets), cd(Cmd::kD3DOMSetBlendFactor),
            cd(Cmd::kD3DOMSetStencilRef), cd(Cmd::kD3DResourceBarrier), cd(Cmd::kD3DExecuteIndirect),
            cd(Cmd::kD3DDrawIndexedInstanced), cd(Cmd::kD3DDrawInstanced), cd(Cmd::kD3DDispatch),
            cd(Cmd::kD3DClearRenderTargetView) + cd(Cmd::kD3DClearDepthStencilView),
            cd(Cmd::kD3DCopyBufferRegion) + cd(Cmd::kD3DCopyResource) + cd(Cmd::kCopyTexture) +
                cd(Cmd::kD3DCopyTextureRegion),
            cd(Cmd::kD3DBeginQuery) + cd(Cmd::kD3DEndQuery));
      }
      const uint64_t draws = d(&PoolStats::draws), hit = d(&PoolStats::hit),
                     opq = d(&PoolStats::opq), opened = d(&PoolStats::opened),
                     frames = d(&PoolStats::frames);
      uint64_t hist[5], batches = 0;
      for (int i = 0; i < 5; ++i) {
        hist[i] = c.hist[i] - s_pp.hist[i];
        batches += hist[i];
      }
      const double dd = draws ? double(draws) : 1.0;
      const double fr = frames ? double(frames) : 1.0;
      const double bb = batches ? double(batches) : 1.0;
      REXGPU_INFO(
          "[pool] in={}/s out={}/s (-{:.1f}%) dpf {:.0f} -> {:.0f} | opq={:.1f}% hit={:.1f}% "
          "batches/fr={:.1f} sizes 1/2-3/4-9/10-49/50+={:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}% "
          "| refuse pred1={} cap={} vfy regs/ps/bl={}/{}/{} elig={} tile={} alloc={} psbig={} "
          "ovf={} | close class/pass/copy/frame/submit={}/{}/{}/{}/{} | inst {:.1f} MB/s "
          "| psinst open/fr={:.1f} hit/fr={:.1f} | ei patched/unpatched={}/{}",
          draws, draws - hit, 100.0 * double(hit) / dd, double(draws) / fr,
          double(draws - hit) / fr, 100.0 * double(opq) / dd, 100.0 * double(hit) / dd,
          double(opened) / fr, 100.0 * double(hist[0]) / bb, 100.0 * double(hist[1]) / bb,
          100.0 * double(hist[2]) / bb, 100.0 * double(hist[3]) / bb,
          100.0 * double(hist[4]) / bb, d(&PoolStats::ref_pred), d(&PoolStats::ref_cap),
          d(&PoolStats::ref_vfy_regs), d(&PoolStats::ref_vfy_ps), d(&PoolStats::ref_vfy_bl),
          d(&PoolStats::ref_elig), d(&PoolStats::ref_tile), d(&PoolStats::ref_alloc),
          d(&PoolStats::ref_psbig), d(&PoolStats::ref_ovf), d(&PoolStats::close_class),
          d(&PoolStats::close_pass), d(&PoolStats::close_copy), d(&PoolStats::close_frame),
          d(&PoolStats::close_submit),
          secs > 0 ? double(d(&PoolStats::inst_bytes)) / secs / 1048576.0 : 0.0,
          double(d(&PoolStats::opened_ps)) / fr, double(d(&PoolStats::hit_ps)) / fr,
          d(&PoolStats::ei_patched), d(&PoolStats::ei_unpatched));
      {
        const uint64_t gp = d(&PoolStats::geo_idx_plain), gh = d(&PoolStats::geo_idx_hit);
        uint64_t gd[6], gi[6], gds = 0, gis = 0;
        for (int i = 0; i < 6; ++i) {
          gd[i] = c.geo_hist[i] - s_pp.geo_hist[i];
          gi[i] = c.geo_hist_idx[i] - s_pp.geo_hist_idx[i];
          gds += gd[i];
          gis += gi[i];
        }
        const double gdd = gds ? double(gds) : 1.0, gii = gis ? double(gis) : 1.0;
        REXGPU_INFO(
            "[geo] idx/fr plain={:.0f}k hit={:.0f}k total={:.0f}k | per draw plain={:.0f} "
            "hit={:.0f} | draws by idx <=16/64/256/1k/4k/4k+ = {:.0f}/{:.0f}/{:.0f}/{:.0f}/"
            "{:.0f}/{:.0f}% | idx share = {:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}%",
            double(gp) / fr / 1000.0, double(gh) / fr / 1000.0, double(gp + gh) / fr / 1000.0,
            (draws - hit) ? double(gp) / double(draws - hit) : 0.0,
            hit ? double(gh) / double(hit) : 0.0, 100.0 * double(gd[0]) / gdd,
            100.0 * double(gd[1]) / gdd, 100.0 * double(gd[2]) / gdd, 100.0 * double(gd[3]) / gdd,
            100.0 * double(gd[4]) / gdd, 100.0 * double(gd[5]) / gdd, 100.0 * double(gi[0]) / gii,
            100.0 * double(gi[1]) / gii, 100.0 * double(gi[2]) / gii, 100.0 * double(gi[3]) / gii,
            100.0 * double(gi[4]) / gii, 100.0 * double(gi[5]) / gii);
      }
      {  // [cull] the same 1 Hz window.
        const CullAcc& k = cull_acc_;
        const double tt = k.tested ? double(k.tested) : 1.0;
        const double bd = k.bounded ? double(k.bounded) : 1.0;
        const double fidx = double(d(&PoolStats::geo_idx_plain) + d(&PoolStats::geo_idx_hit) + k.skipped_idx);
        static const char* kPhase[3] = {"off", "verify", "skip"};
        REXGPU_INFO(
            "[cull] phase={} tested/fr={:.0f} refuse path/memx/prim/vte/fmt/bounds="
            "{:.1f}/{:.1f}/{:.1f}/{:.1f}/{:.1f}/{:.1f}% (idx {:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}k)"
            " | bounded/fr {:.0f} idx {:.0f}k | OUT draws/fr {:.0f} ({:.1f}% of bounded) idx "
            "{:.0f}k ({:.1f}% of frame idx) | skipped/fr {:.0f} idx {:.0f}k | bounds new={} unb={} "
            "inval={} full={} scan {:.2f} ms/s entries={}",
            kPhase[cull_phase_ < 3 ? cull_phase_ : 0], double(k.tested) / fr,
            100.0 * double(k.refuse[1]) / tt, 100.0 * double(k.refuse[2]) / tt,
            100.0 * double(k.refuse[3]) / tt, 100.0 * double(k.refuse[4]) / tt,
            100.0 * double(k.refuse[5]) / tt, 100.0 * double(k.refuse[6]) / tt,
            double(k.refuse_idx[1]) / fr / 1000.0, double(k.refuse_idx[2]) / fr / 1000.0,
            double(k.refuse_idx[3]) / fr / 1000.0, double(k.refuse_idx[4]) / fr / 1000.0,
            double(k.refuse_idx[5]) / fr / 1000.0, double(k.refuse_idx[6]) / fr / 1000.0,
            double(k.bounded) / fr, double(k.bounded_idx) / fr / 1000.0, double(k.out) / fr,
            100.0 * double(k.out) / bd, double(k.out_idx) / fr / 1000.0,
            fidx > 0 ? 100.0 * double(k.out_idx) / fidx : 0.0, double(k.skipped) / fr,
            double(k.skipped_idx) / fr / 1000.0, cull_stats_.bounds_new,
            cull_stats_.bounds_unbounded, cull_stats_.bounds_invalidated, cull_stats_.bounds_full,
            secs > 0 ? double(cull_stats_.bounds_ns) / 1e6 / secs : 0.0, nr_cull_.entries());
        cull_acc_ = CullAcc{};
        cull_stats_ = CullStats{};
        {  // [hiz]
          const HizAcc& h = hiz_acc_;
          const double el = h.elig ? double(h.elig) : 1.0;
          uint32_t worst_vs = 0;
          for (uint32_t v = 1; v < 4096; ++v) {
            if (h.wrong_vs[v] > h.wrong_vs[worst_vs]) worst_vs = v;
          }
          REXGPU_INFO(
              "[hiz] phase={} k={} rb={} chk/fr={:.1f} build/fr={:.1f} | elig/fr {:.0f} idx "
              "{:.0f}k | refuse "
              "zcls/psz/w/rect/vp/nort/ring/alloc/notplain={:.1f}/{:.1f}/{:.1f}/{:.1f}/{:.1f}/"
              "{:.1f}/{:.1f}/{:.1f}/{:.1f}% (idx {:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/"
              "{:.0f}/{:.0f}k) | tested/fr {:.0f} idx {:.0f}k ({:.1f}% of frame idx) | HIDDEN/fr "
              "{:.0f} idx {:.0f}k ({:.1f}% of frame idx) | vfy {} idx {:.0f}k WRONG {} idx {:.0f}k "
              "smp {} worst vs {:016X} x{} | close cap/pass/copy/zw/dir/sub/xfer/rt={}/{}/{}/{}/{}/"
              "{}/{}/{} | pool open/fr {:.1f} inst/fr {:.1f} tested/fr {:.1f} miss/fr {:.1f}",
              kPhase[hiz_phase_ < 3 ? hiz_phase_ : 0], hiz_k_, hiz_rebuild_,
              double(h.checkpoints) / fr, double(h.builds) / fr, double(h.elig) / fr, double(h.elig_idx) / fr / 1000.0,
              100.0 * double(h.refuse[1]) / el, 100.0 * double(h.refuse[2]) / el,
              100.0 * double(h.refuse[3]) / el, 100.0 * double(h.refuse[4]) / el,
              100.0 * double(h.refuse[5]) / el, 100.0 * double(h.refuse[6]) / el,
              100.0 * double(h.refuse[7]) / el, 100.0 * double(h.refuse[8]) / el,
              100.0 * double(h.refuse[9]) / el, double(h.refuse_idx[1]) / fr / 1000.0,
              double(h.refuse_idx[2]) / fr / 1000.0, double(h.refuse_idx[3]) / fr / 1000.0,
              double(h.refuse_idx[4]) / fr / 1000.0, double(h.refuse_idx[5]) / fr / 1000.0,
              double(h.refuse_idx[6]) / fr / 1000.0, double(h.refuse_idx[7]) / fr / 1000.0,
              double(h.refuse_idx[8]) / fr / 1000.0, double(h.refuse_idx[9]) / fr / 1000.0,
              double(h.tested) / fr, double(h.tested_idx) / fr / 1000.0,
              fidx > 0 ? 100.0 * double(h.tested_idx) / fidx : 0.0, double(h.hidden) / fr,
              double(h.hidden_idx) / fr / 1000.0,
              fidx > 0 ? 100.0 * double(h.hidden_idx) / fidx : 0.0, h.vfy,
              double(h.vfy_idx) / 1000.0, h.wrong, double(h.wrong_idx) / 1000.0, h.wrong_smp,
              worst_vs < occ_vs_hashes_.size() ? occ_vs_hashes_[worst_vs] : 0ull,
              h.wrong_vs[worst_vs], h.close[0], h.close[1], h.close[2], h.close[3], h.close[4],
              h.close[5], h.close[6], h.close[7], double(h.pool_open) / fr,
              double(h.pool_inst) / fr, double(h.pool_inst_tested) / fr, double(h.pool_miss) / fr);
          hiz_acc_ = HizAcc{};
        }
      }
      s_pp = c;
      // [SKC] the rung-2 skip census, same window.
      if (g_skc_on) {
        static SkcStats s_sk;
        const SkcStats& k = g_skc;
        uint64_t chg[8], hc[8], rd[8], rr[8];
        for (int i = 0; i < 8; ++i) {
          chg[i] = k.chg[i] - s_sk.chg[i];
          hc[i] = k.hit_cls[i] - s_sk.hit_cls[i];
          rd[i] = k.rd[i] - s_sk.rd[i];
          rr[i] = k.rd_regs[i] - s_sk.rd_regs[i];
        }
        const uint64_t hits = k.hits - s_sk.hits, plain = k.plain - s_sk.plain,
                       pt = k.plain_taint - s_sk.plain_taint;
        REXGPU_INFO(
            "[pool-skip] hits={} chg0={} ovf={} onlyalu={} | chg dw vsf/psf/fetch/bl/st/lo/oth="
            "{}/{}/{}/{}/{}/{}/{} | groups touching vsf/psf/fetch/bl/st/lo/oth={}/{}/{}/{}/{}/{}/{} "
            "| plain={} READ TAINT={} ({:.3f}%) vsf/psf/vb/tex/bl/st/lo/oth={}/{}/{}/{}/{}/{}/{}/{} "
            "regs={}/{}/{}/{}/{}/{}/{}/{} | taint pop avg={:.1f} now={}",
            hits, k.hits_chg0 - s_sk.hits_chg0, k.hits_ovf - s_sk.hits_ovf,
            k.hit_only_alu - s_sk.hit_only_alu, chg[0], chg[1], chg[2], chg[4], chg[5], chg[6],
            chg[7], hc[0], hc[1], hc[2], hc[4], hc[5], hc[6], hc[7], plain, pt,
            100.0 * double(pt) / double(plain ? plain : 1), rd[0], rd[1], rd[2], rd[3], rd[4],
            rd[5], rd[6], rd[7], rr[0], rr[1], rr[2], rr[3], rr[4], rr[5], rr[6], rr[7],
            double(k.taint_pop - s_sk.taint_pop) / double(plain ? plain : 1), g_skc_ntaint);
        s_sk = k;
        // The top tainted-and-read registers of the window.
        {
          std::string top;
          for (int n = 0; n < 12; ++n) {
            uint32_t best = 0, bi = UINT32_MAX;
            for (uint32_t r = 0; r < RegisterFile::kRegisterCount; ++r) {
              if (g_skc_rd_hist[r] > best) {
                best = g_skc_rd_hist[r];
                bi = r;
              }
            }
            if (bi == UINT32_MAX) break;
            top += fmt::format(" {:04X}:{}", bi, best);
            g_skc_rd_hist[bi] = 0;
          }
          std::memset(g_skc_rd_hist, 0, sizeof(g_skc_rd_hist));
          REXGPU_INFO("[pool-skip2] top read-tainted regs:{}", top.empty() ? " none" : top.c_str());
        }
        // [SKC] timed disarm: after N city windows the census stands down so
        // the rest of the drive prices the apply path without its hooks.
        {
          const int32_t sk_secs = REXCVAR_GET(gpu_pool_skip_census_secs);
          if (sk_secs > 0 && !g_skc_disarmed) {
            if (draws >= 100000) ++g_skc_city_windows;
            if (g_skc_city_windows >= uint32_t(sk_secs)) {
              g_skc_disarmed = true;
              REXGPU_INFO("[pool-skip] DISARMED after {} city windows "
                          "(gpu_pool_skip_census_secs); apply path clean from here",
                          g_skc_city_windows);
            }
          }
        }
      }
      // [GKEY] the rung-2 census join, same window.
      static GkStats s_gk;
      if (g_gk_census) {
        const GkStats& g = g_gk;
        auto e = [&](uint64_t GkStats::*f) { return g.*f - s_gk.*f; };
        const uint64_t gd = e(&GkStats::draws), joined = e(&GkStats::joined),
                       hit = e(&GkStats::hit);
        uint64_t ne[6], hn[6];
        for (int i = 0; i < 6; ++i) {
          ne[i] = g.mir_ne[i] - s_gk.mir_ne[i];
          hn[i] = g.hit_ne_c[i] - s_gk.hit_ne_c[i];
        }
        const nr::GkTableStats& ts = nr::GetGkTableStats();
        // The game hook's emitter-memo counters (window deltas).
        static uint64_t s_hk[8] = {};
        uint64_t hd[8] = {};
        if (const uint64_t* hk = nr::GetGkHookStats()) {
          for (int i = 0; i < 8; ++i) {
            hd[i] = hk[i] - s_hk[i];
            s_hk[i] = hk[i];
          }
        }
        REXGPU_INFO(
            "[gkey] draws={}/s join={:.1f}% (norec={} argsne={} badhdr={} nogk={}) | mirror "
            "eq={:.1f}% ne pkt/fetch/stA/stB/stC/bl={}/{}/{}/{}/{}/{} | shp new={} REHASH={} "
            "| hits={} cover eq={:.1f}% ne={} (pkt/fetch/stA/stB/stC/bl={}/{}/{}/{}/{}/{}) "
            "none={} | gkeq={} HKNE={} hit={} notopen={} "
            "ref ps/regs/bl/cap/other={}/{}/{}/{}/{} | tile={} ovf={} | mask none={} NE={} "
            "pairs set={} changed={} | tbl stored={} evict={} | hook emit={} im={} skip={} "
            "scan={} memo_ne={} inh={} inh0={}",
            gd, 100.0 * double(joined) / double(gd ? gd : 1), e(&GkStats::norec),
            e(&GkStats::argsne), e(&GkStats::badhdr), e(&GkStats::nogk),
            100.0 * double(e(&GkStats::mir_eq)) / double(joined ? joined : 1), ne[0], ne[1],
            ne[2], ne[3], ne[4], ne[5], e(&GkStats::shp_new), e(&GkStats::shp_rehash), hit,
            100.0 * double(e(&GkStats::hit_eq)) / double(hit ? hit : 1), e(&GkStats::hit_ne),
            hn[0], hn[1], hn[2], hn[3], hn[4], hn[5], e(&GkStats::hit_none), e(&GkStats::gkeq), e(&GkStats::gkeq_hkne),
            e(&GkStats::gkeq_hit), e(&GkStats::gkeq_notopen), e(&GkStats::gkeq_ref_ps),
            e(&GkStats::gkeq_ref_regs), e(&GkStats::gkeq_ref_bl), e(&GkStats::gkeq_ref_cap),
            e(&GkStats::gkeq_ref_other), e(&GkStats::tile), e(&GkStats::tab_ovf),
            e(&GkStats::nomask), e(&GkStats::mask_ne), ts.slots_set, ts.slots_changed,
            ts.stored, ts.evictions, hd[0], hd[1], hd[2], hd[3], hd[4], hd[5], hd[6]);
        s_gk = g;
      }
    }
    // [GKEY] latch the census for the next frame; tables sized on arm.
    {
      const bool gk_arm = REXCVAR_GET(gpu_gkey_census);
      if (gk_arm && g_gk_tab.empty()) {
        g_gk_tab.resize(kGkSlots);
        g_gk_shp.resize(kGkShpSlots);
      }
      g_gk_census = gk_arm;
    }
    // [SKC] latch; clear the taint state on arm.
    {
      const bool sk_arm = REXCVAR_GET(gpu_pool_skip_census) && !g_skc_disarmed;
      if (sk_arm && !g_skc_on) {
        std::memset(g_skc_taint, 0, RegisterFile::kRegisterCount);
        std::memset(g_skc_seen, 0, RegisterFile::kRegisterCount * sizeof(uint32_t));
        g_skc_ntaint = 0;
        for (int i = 0; i < 8; ++i) g_skc_ntaint_cls[i] = 0;
        SkcGroupReset();
      }
      g_skc_on = sk_arm;
    }
    // Unconditional since 2026-09-02 (naruto_764/765/766 clean; the cvar and
    // its cycler are gone): the pool is how draws are issued.
    if (!g_pool_on) {
      PoolReset();
      g_pool_on = true;
      g_pool_intel_off = GetD3D12Provider().GetAdapterVendorID() ==
                         ui::GraphicsProvider::GpuVendorID::kIntel;
      if (g_pool_intel_off) REXGPU_INFO("[pool] Intel adapter: the draw pool issues plain draws");
    }
  }

  // [BATCH-CENSUS] fold the frame that just ended (under the OLD arm state,
  // so a mid-run flag flip cannot fold a half-censused frame), then refresh
  // the arm for the next frame; reset-on-arm like the inst-probe.
  {
    if (g_batch_census) {
      BatchCensusFrameEnd();
      static uint64_t s_bc[10] = {};
      static auto s_bc_last = std::chrono::steady_clock::now();
      const auto bc_now = std::chrono::steady_clock::now();
      if (bc_now - s_bc_last >= std::chrono::seconds(1)) {
        s_bc_last = bc_now;
        const uint64_t draws = g_bc_draws - s_bc[0];
        const uint64_t adj = g_bc_adj - s_bc[1];
        const uint64_t runs = g_bc_runs - s_bc[2];
        const uint64_t rundraws = g_bc_run_draws - s_bc[3];
        const uint64_t frames = g_bc_frames - s_bc[4];
        const uint64_t keys = g_bc_keys - s_bc[5];
        const uint64_t rep = g_bc_rep_draws - s_bc[6];
        const uint64_t extra = g_bc_extra - s_bc[7];
        const uint64_t breaks = g_bc_breaks - s_bc[8];
        const uint64_t ovf = g_bc_frame_ovf - s_bc[9];
        const double dd = draws ? double(draws) : 1.0;
        REXGPU_INFO(
            "[batch-census] draws={} adj={} ({:.1f}%) runs={} rundraws={} ({:.1f}%) maxrun={} "
            "| frame: keys/fr={} rep={} ({:.1f}%) extra={} ({:.1f}%) maxkey={} ovf={} "
            "| fr={} brk={}",
            draws, adj, 100.0 * double(adj) / dd, runs, rundraws,
            100.0 * double(rundraws) / dd, g_bc_maxrun,
            frames ? keys / frames : 0, rep, 100.0 * double(rep) / dd, extra,
            100.0 * double(extra) / dd, g_bc_maxkey, ovf, frames, breaks);
        // [BATCH-CENSUS v2] the same draws through the full merge key, by
        // scope. Read `win` as what a host in-pass pooler removes today,
        // `pass`/`fr` as what order treatment could add; `blk` = windows a
        // blended/z-off draw closed (the interleave), `after` = projected
        // draws/frame once win extra is pooled.
        {
          static Bc2Counters s_bc2;
          const Bc2Counters& c = g_bc2;
          const uint64_t xf = c.extra_fr - s_bc2.extra_fr;
          const uint64_t xp = c.extra_pass - s_bc2.extra_pass;
          const uint64_t passes = c.passes - s_bc2.passes;
          const uint64_t cls[4] = {c.cls[0] - s_bc2.cls[0], c.cls[1] - s_bc2.cls[1],
                                   c.cls[2] - s_bc2.cls[2], c.cls[3] - s_bc2.cls[3]};
          const uint64_t ovf2 = c.ovf - s_bc2.ovf;
          static auto s_bc2_last = bc_now;
          const double secs = std::chrono::duration<double>(bc_now - s_bc2_last).count();
          s_bc2_last = bc_now;
          const double fr = frames ? double(frames) : 1.0;
          const uint64_t total_draws = cls[0] + cls[1] + cls[2] + cls[3];
          REXGPU_INFO(
              "[batch-census2] fullkey extra fr={} ({:.1f}%) pass={} ({:.1f}%) | "
              "passes/fr={:.1f} | class opq={:.1f}% bzw={:.1f}% zro={:.1f}% bld={:.1f}% "
              "| blend0 top {} | depthctl top {} | dpf {:.0f} | ovf={}",
              xf, 100.0 * double(xf) / dd, xp, 100.0 * double(xp) / dd, double(passes) / fr,
              100.0 * double(cls[0]) / dd, 100.0 * double(cls[1]) / dd,
              100.0 * double(cls[2]) / dd, 100.0 * double(cls[3]) / dd,
              Bc2HistTop(g_bc2_blend_hist, total_draws), Bc2HistTop(g_bc2_depth_hist, total_draws),
              double(draws) / fr, ovf2);
          // One line per window rule: win = strict (opq only), win2 = opq + bzw,
          // win3 = win2's rule with the key minus texture fetch constants.
          for (uint32_t w = 0; w < 3; ++w) {
            const uint64_t xw = c.extra_win[w] - s_bc2.extra_win[w];
            const uint64_t wins = c.wins[w] - s_bc2.wins[w];
            const uint64_t blk = c.win_blk[w] - s_bc2.win_blk[w];
            const uint64_t wdraws = c.win_draws[w] - s_bc2.win_draws[w];
            const uint64_t wkeys = c.keys_win[w] - s_bc2.keys_win[w];
            uint64_t ord[4];
            for (int i = 0; i < 4; ++i) ord[i] = c.ord[w][i] - s_bc2.ord[w][i];
            const uint64_t vec4 = c.inst_vec4[w] - s_bc2.inst_vec4[w];
            REXGPU_INFO(
                "[batch-census2] {}: extra={} ({:.1f}%) | win/fr={:.1f} blk/fr={:.1f} "
                "draws/win={:.1f} keys/win={:.1f} maxwin={} | pooled by ord "
                "2-3/4-9/10-49/50+ = {:.1f}/{:.1f}/{:.1f}/{:.1f}% | inst vec4/draw={:.1f} "
                "= {:.1f} MB/s | dpf {:.0f} -> {:.0f}",
                w == 2 ? "win3(notex)" : w ? "win2" : "win", xw, 100.0 * double(xw) / dd,
                double(wins) / fr,
                double(blk) / fr, wins ? double(wdraws) / double(wins) : 0.0,
                wins ? double(wkeys) / double(wins) : 0.0, g_bc2_maxwin[w],
                100.0 * double(ord[0]) / dd, 100.0 * double(ord[1]) / dd,
                100.0 * double(ord[2]) / dd, 100.0 * double(ord[3]) / dd,
                xw ? double(vec4) / double(xw) : 0.0,
                secs > 0 ? double(vec4) * 16.0 / secs / 1048576.0 : 0.0, double(draws) / fr,
                double(draws - xw) / fr);
            g_bc2_maxwin[w] = 0;
          }
          s_bc2 = c;
        }
        g_bc_maxrun = 0;
        g_bc_maxkey = 0;
        s_bc[0] = g_bc_draws;
        s_bc[1] = g_bc_adj;
        s_bc[2] = g_bc_runs;
        s_bc[3] = g_bc_run_draws;
        s_bc[4] = g_bc_frames;
        s_bc[5] = g_bc_keys;
        s_bc[6] = g_bc_rep_draws;
        s_bc[7] = g_bc_extra;
        s_bc[8] = g_bc_breaks;
        s_bc[9] = g_bc_frame_ovf;
      }
    }
    const bool bc_arm = REXCVAR_GET(gpu_batch_census);
    if (bc_arm && !g_batch_census) BatchCensusReset();
    g_batch_census = bc_arm;
  }

  if (g_draw_prof) g_draw_res_seen.clear();
  vertex_buffers_in_sync_[0] = 0;
  vertex_buffers_in_sync_[1] = 0;
  // [NR-RSY] Phase 5-3b-3: the swap clears the sync bits ONLY (per-slot
  // {address, size} survive) - transcribed.
  if (g_nr_res) {
    nr::ResVfetchClearSyncBits(&g_nr_res_vf);
  }

  if (!graphics_system_)
    return;
  ui::Presenter* presenter = graphics_system_->presenter();
  if (!presenter) {
    REXGPU_ERROR("IssueSwap: presenter is null");
    return;
  }

  // In case the swap command is the only one in the frame.
  if (!BeginSubmission(true)) {
    REXGPU_ERROR("IssueSwap: BeginSubmission failed");
    return;
  }
  // [gpu-census] everything from here to the submission close is swap work.
  GpuCensusSetClass(kGpuCensusSwap);

  // Obtain the actual swap source texture size (resolution-scaled if it's a
  // resolve destination, or not otherwise).
  D3D12_SHADER_RESOURCE_VIEW_DESC swap_texture_srv_desc;
  xenos::TextureFormat frontbuffer_format;
  uint32_t frontbuffer_width_unscaled = 0, frontbuffer_height_unscaled = 0;
  ID3D12Resource* swap_texture_resource =
      texture_cache_->RequestSwapTexture(swap_texture_srv_desc, frontbuffer_format,
                                         &frontbuffer_width_unscaled, &frontbuffer_height_unscaled);
  if (!swap_texture_resource) {
    // Dump texture fetch constant 0 for debugging
    const auto& regs = *register_file_;
    auto fetch = regs.GetTextureFetch(0);
    REXGPU_ERROR(
        "IssueSwap: RequestSwapTexture failed - fetch0: {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
        fetch.dword_0, fetch.dword_1, fetch.dword_2, fetch.dword_3, fetch.dword_4, fetch.dword_5);
    return;
  }
  D3D12_RESOURCE_DESC swap_texture_desc = swap_texture_resource->GetDesc();
  // The swap gamma / FXAA pass samples source texels by pixel index, but swap
  // textures may be allocation-padded. Prefer the active frontbuffer region
  // from the swap packet, scaled proportionally to the actual source texture.
  uint32_t source_width_scaled = uint32_t(swap_texture_desc.Width);
  uint32_t source_height_scaled = uint32_t(swap_texture_desc.Height);
  auto get_active_swap_dimension = [](uint32_t packet_unscaled, uint32_t source_unscaled,
                                      uint32_t source_scaled) -> uint32_t {
    if (!source_scaled) {
      return 0;
    }
    uint32_t active_unscaled = packet_unscaled ? packet_unscaled : source_unscaled;
    if (!active_unscaled) {
      return source_scaled;
    }
    if (source_unscaled) {
      active_unscaled = std::min(active_unscaled, source_unscaled);
      uint64_t active_scaled =
          (uint64_t(active_unscaled) * source_scaled + (source_unscaled >> 1)) / source_unscaled;
      return uint32_t(std::clamp<uint64_t>(active_scaled, 1, source_scaled));
    }
    return std::min(active_unscaled, source_scaled);
  };
  uint32_t guest_output_width =
      get_active_swap_dimension(frontbuffer_width, frontbuffer_width_unscaled, source_width_scaled);
  uint32_t guest_output_height = get_active_swap_dimension(
      frontbuffer_height, frontbuffer_height_unscaled, source_height_scaled);
  if (!guest_output_width) {
    guest_output_width = source_width_scaled
                             ? source_width_scaled
                             : (frontbuffer_width ? frontbuffer_width : frontbuffer_width_unscaled);
  }
  if (!guest_output_height) {
    guest_output_height = source_height_scaled ? source_height_scaled
                                               : (frontbuffer_height ? frontbuffer_height
                                                                     : frontbuffer_height_unscaled);
  }
  bool swap_source_scaled = frontbuffer_width_unscaled && frontbuffer_height_unscaled &&
                            (source_width_scaled != frontbuffer_width_unscaled ||
                             source_height_scaled != frontbuffer_height_unscaled);
  if (texture_cache_->IsDrawResolutionScaled() && !swap_source_scaled) {
    static bool draw_scale_swap_unscaled_logged = false;
    if (!draw_scale_swap_unscaled_logged) {
      draw_scale_swap_unscaled_logged = true;
      REXGPU_WARN(
          "D3D12 draw resolution scaling is enabled, but the swap source is "
          "unscaled ({}x{}). This title may be presenting from an unscaled "
          "resolve path.",
          guest_output_width, guest_output_height);
    }
  }

  system::X_VIDEO_MODE video_mode;
  kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
  uint32_t display_width = std::max(uint32_t(1), uint32_t(video_mode.display_width));
  uint32_t display_height = std::max(uint32_t(1), uint32_t(video_mode.display_height));

  REXGPU_DEBUG(
      "XELOG_GPU PRESENT: packet_size={}x{} src_unscaled={}x{} guest_output_size={}x{} "
      "display={}x{} draw_scaled={}",
      frontbuffer_width, frontbuffer_height, frontbuffer_width_unscaled, frontbuffer_height_unscaled,
      guest_output_width, guest_output_height, display_width, display_height,
      texture_cache_->IsDrawResolutionScaled());

  presenter->RefreshGuestOutput(
      guest_output_width, guest_output_height, display_width, display_height,
      [this, &swap_texture_srv_desc, frontbuffer_format, swap_texture_resource, guest_output_width,
       guest_output_height](ui::Presenter::GuestOutputRefreshContext& context) -> bool {
        const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
        ID3D12Device* device = provider.GetDevice();

        SwapPostEffect swap_post_effect = GetActualSwapPostEffect();
        bool use_fxaa = swap_post_effect == SwapPostEffect::kFxaa ||
                        swap_post_effect == SwapPostEffect::kFxaaExtreme;
        if (use_fxaa) {
          // Make sure the texture of the correct size is available for FXAA.
          if (fxaa_source_texture_) {
            D3D12_RESOURCE_DESC fxaa_source_texture_desc = fxaa_source_texture_->GetDesc();
            if (fxaa_source_texture_desc.Width != guest_output_width ||
                fxaa_source_texture_desc.Height != guest_output_height) {
              if (submission_completed_ < fxaa_source_texture_submission_) {
                fxaa_source_texture_->AddRef();
                resources_for_deletion_.emplace_back(fxaa_source_texture_submission_,
                                                     fxaa_source_texture_.Get());
              }
              fxaa_source_texture_.Reset();
              fxaa_source_texture_submission_ = 0;
            }
          }
          if (!fxaa_source_texture_) {
            D3D12_RESOURCE_DESC fxaa_source_texture_desc;
            fxaa_source_texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            fxaa_source_texture_desc.Alignment = 0;
            fxaa_source_texture_desc.Width = guest_output_width;
            fxaa_source_texture_desc.Height = guest_output_height;
            fxaa_source_texture_desc.DepthOrArraySize = 1;
            fxaa_source_texture_desc.MipLevels = 1;
            fxaa_source_texture_desc.Format = kFxaaSourceTextureFormat;
            fxaa_source_texture_desc.SampleDesc.Count = 1;
            fxaa_source_texture_desc.SampleDesc.Quality = 0;
            fxaa_source_texture_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            fxaa_source_texture_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            if (FAILED(device->CreateCommittedResource(
                    &ui::d3d12::util::kHeapPropertiesDefault, provider.GetHeapFlagCreateNotZeroed(),
                    &fxaa_source_texture_desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    nullptr, IID_PPV_ARGS(&fxaa_source_texture_)))) {
              REXGPU_ERROR("Failed to create the FXAA input texture");
              swap_post_effect = SwapPostEffect::kNone;
              use_fxaa = false;
            }
          }
        }

        // This is according to D3D::InitializePresentationParameters from a
        // game executable, which initializes the 256-entry table gamma ramp for
        // 8_8_8_8 output and the PWL gamma ramp for 2_10_10_10.
        // TODO(Triang3l): Choose between the table and PWL based on
        // DC_LUTA_CONTROL, support both for all formats (and also different
        // increments for PWL).
        bool use_pwl_gamma_ramp =
            frontbuffer_format == xenos::TextureFormat::k_2_10_10_10 ||
            frontbuffer_format == xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16;

        context.SetIs8bpc(!use_pwl_gamma_ramp && !use_fxaa);

        // Upload the new gamma ramp, using the upload buffer for the current
        // frame (will close the frame after this anyway, so can't write
        // multiple times per frame).
        if (!(use_pwl_gamma_ramp ? gamma_ramp_pwl_up_to_date_
                                 : gamma_ramp_256_entry_table_up_to_date_)) {
          uint32_t gamma_ramp_offset_bytes = use_pwl_gamma_ramp ? 256 * 4 : 0;
          uint32_t gamma_ramp_upload_offset_bytes =
              uint32_t(frame_current_ % kQueueFrames) * ((256 + 128 * 3) * 4) +
              gamma_ramp_offset_bytes;
          uint32_t gamma_ramp_size_bytes = (use_pwl_gamma_ramp ? 128 * 3 : 256) * 4;
          if (std::endian::native != std::endian::little && use_pwl_gamma_ramp) {
            // R16G16 is first R16, where the shader expects the base, and
            // second G16, where the delta should be, but gamma_ramp_pwl_rgb()
            // is an array of 32-bit DC_LUT_PWL_DATA registers - swap 16 bits in
            // each 32.
            auto gamma_ramp_pwl_upload_buffer = reinterpret_cast<reg::DC_LUT_PWL_DATA*>(
                gamma_ramp_upload_buffer_mapping_ + gamma_ramp_upload_offset_bytes);
            const reg::DC_LUT_PWL_DATA* gamma_ramp_pwl = gamma_ramp_pwl_rgb();
            for (size_t i = 0; i < 128 * 3; ++i) {
              reg::DC_LUT_PWL_DATA& gamma_ramp_pwl_upload_buffer_entry =
                  gamma_ramp_pwl_upload_buffer[i];
              reg::DC_LUT_PWL_DATA gamma_ramp_pwl_entry = gamma_ramp_pwl[i];
              gamma_ramp_pwl_upload_buffer_entry.base = gamma_ramp_pwl_entry.delta;
              gamma_ramp_pwl_upload_buffer_entry.delta = gamma_ramp_pwl_entry.base;
            }
          } else {
            std::memcpy(gamma_ramp_upload_buffer_mapping_ + gamma_ramp_upload_offset_bytes,
                        use_pwl_gamma_ramp ? static_cast<const void*>(gamma_ramp_pwl_rgb())
                                           : static_cast<const void*>(gamma_ramp_256_entry_table()),
                        gamma_ramp_size_bytes);
          }
          PushTransitionBarrier(gamma_ramp_buffer_.Get(), gamma_ramp_buffer_state_,
                                D3D12_RESOURCE_STATE_COPY_DEST);
          gamma_ramp_buffer_state_ = D3D12_RESOURCE_STATE_COPY_DEST;
          SubmitBarriers();
          deferred_command_list_.D3DCopyBufferRegion(
              gamma_ramp_buffer_.Get(), gamma_ramp_offset_bytes, gamma_ramp_upload_buffer_.Get(),
              gamma_ramp_upload_offset_bytes, gamma_ramp_size_bytes);
          (use_pwl_gamma_ramp ? gamma_ramp_pwl_up_to_date_
                              : gamma_ramp_256_entry_table_up_to_date_) = true;
        }

        // Destination, source, and if bindful, gamma ramp.
        ui::d3d12::util::DescriptorCpuGpuHandlePair apply_gamma_descriptors[3];
        ui::d3d12::util::DescriptorCpuGpuHandlePair apply_gamma_descriptor_gamma_ramp;
        if (!RequestOneUseSingleViewDescriptors(bindless_resources_used_ ? 2 : 3,
                                                apply_gamma_descriptors)) {
          return false;
        }
        // Must not call anything that can change the descriptor heap from now
        // on!
        if (bindless_resources_used_) {
          apply_gamma_descriptor_gamma_ramp = GetSystemBindlessViewHandlePair(
              use_pwl_gamma_ramp ? SystemBindlessView::kGammaRampPWLSRV
                                 : SystemBindlessView::kGammaRampTableSRV);
        } else {
          apply_gamma_descriptor_gamma_ramp = apply_gamma_descriptors[2];
          WriteGammaRampSRV(use_pwl_gamma_ramp, apply_gamma_descriptor_gamma_ramp.first);
        }

        ID3D12Resource* guest_output_resource =
            static_cast<ui::d3d12::D3D12Presenter::D3D12GuestOutputRefreshContext&>(context)
                .resource_uav_capable();

        if (use_fxaa) {
          fxaa_source_texture_submission_ = submission_current_;
        }

        ID3D12Resource* apply_gamma_dest =
            use_fxaa ? fxaa_source_texture_.Get() : guest_output_resource;
        D3D12_RESOURCE_STATES apply_gamma_dest_initial_state =
            use_fxaa ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                     : ui::d3d12::D3D12Presenter::kGuestOutputInternalState;
        static_cast<ui::d3d12::D3D12Presenter::D3D12GuestOutputRefreshContext&>(context)
            .resource_uav_capable();
        PushTransitionBarrier(apply_gamma_dest, apply_gamma_dest_initial_state,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // From now on, even in case of failure, apply_gamma_dest must be
        // transitioned back to apply_gamma_dest_initial_state!
        D3D12_UNORDERED_ACCESS_VIEW_DESC apply_gamma_dest_uav_desc;
        apply_gamma_dest_uav_desc.Format =
            use_fxaa ? kFxaaSourceTextureFormat : ui::d3d12::D3D12Presenter::kGuestOutputFormat;
        apply_gamma_dest_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        apply_gamma_dest_uav_desc.Texture2D.MipSlice = 0;
        apply_gamma_dest_uav_desc.Texture2D.PlaneSlice = 0;
        device->CreateUnorderedAccessView(apply_gamma_dest, nullptr, &apply_gamma_dest_uav_desc,
                                          apply_gamma_descriptors[0].first);

        device->CreateShaderResourceView(swap_texture_resource, &swap_texture_srv_desc,
                                         apply_gamma_descriptors[1].first);

        PushTransitionBarrier(gamma_ramp_buffer_.Get(), gamma_ramp_buffer_state_,
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gamma_ramp_buffer_state_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        deferred_command_list_.D3DSetComputeRootSignature(apply_gamma_root_signature_.Get());
        ApplyGammaConstants apply_gamma_constants;
        apply_gamma_constants.size[0] = guest_output_width;
        apply_gamma_constants.size[1] = guest_output_height;
        deferred_command_list_.D3DSetComputeRoot32BitConstants(
            UINT(ApplyGammaRootParameter::kConstants),
            sizeof(apply_gamma_constants) / sizeof(uint32_t), &apply_gamma_constants, 0);
        deferred_command_list_.D3DSetComputeRootDescriptorTable(
            UINT(ApplyGammaRootParameter::kDestination), apply_gamma_descriptors[0].second);
        deferred_command_list_.D3DSetComputeRootDescriptorTable(
            UINT(ApplyGammaRootParameter::kSource), apply_gamma_descriptors[1].second);
        deferred_command_list_.D3DSetComputeRootDescriptorTable(
            UINT(ApplyGammaRootParameter::kRamp), apply_gamma_descriptor_gamma_ramp.second);
        ID3D12PipelineState* apply_gamma_pipeline;
        if (use_pwl_gamma_ramp) {
          apply_gamma_pipeline = use_fxaa ? apply_gamma_pwl_fxaa_luma_pipeline_.Get()
                                          : apply_gamma_pwl_pipeline_.Get();
        } else {
          apply_gamma_pipeline = use_fxaa ? apply_gamma_table_fxaa_luma_pipeline_.Get()
                                          : apply_gamma_table_pipeline_.Get();
        }
        SetExternalPipeline(apply_gamma_pipeline);
        SubmitBarriers();
        uint32_t group_count_x = (guest_output_width + 15) / 16;
        uint32_t group_count_y = (guest_output_height + 7) / 8;
        deferred_command_list_.D3DDispatch(group_count_x, group_count_y, 1);

        // Apply FXAA.
        if (use_fxaa) {
          // Destination and source.
          ui::d3d12::util::DescriptorCpuGpuHandlePair fxaa_descriptors[2];
          if (!RequestOneUseSingleViewDescriptors(uint32_t(rex::countof(fxaa_descriptors)),
                                                  fxaa_descriptors)) {
            // Failed to obtain descriptors for FXAA - just copy after gamma
            // ramp application without applying FXAA.
            PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE);
            PushTransitionBarrier(guest_output_resource,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
            SubmitBarriers();
            deferred_command_list_.D3DCopyResource(guest_output_resource, apply_gamma_dest);
            PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                  apply_gamma_dest_initial_state);
            PushTransitionBarrier(guest_output_resource, D3D12_RESOURCE_STATE_COPY_DEST,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
            return false;
          } else {
            assert_true(apply_gamma_dest_initial_state ==
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  apply_gamma_dest_initial_state);
            PushTransitionBarrier(guest_output_resource,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            // From now on, even in case of failure, guest_output_resource must
            // be transitioned back to kGuestOutputInternalState!
            deferred_command_list_.D3DSetComputeRootSignature(fxaa_root_signature_.Get());
            FxaaConstants fxaa_constants;
            fxaa_constants.size[0] = guest_output_width;
            fxaa_constants.size[1] = guest_output_height;
            fxaa_constants.size_inv[0] = 1.0f / float(fxaa_constants.size[0]);
            fxaa_constants.size_inv[1] = 1.0f / float(fxaa_constants.size[1]);
            deferred_command_list_.D3DSetComputeRoot32BitConstants(
                UINT(FxaaRootParameter::kConstants), sizeof(fxaa_constants) / sizeof(uint32_t),
                &fxaa_constants, 0);
            D3D12_UNORDERED_ACCESS_VIEW_DESC fxaa_dest_uav_desc;
            fxaa_dest_uav_desc.Format = ui::d3d12::D3D12Presenter::kGuestOutputFormat;
            fxaa_dest_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            fxaa_dest_uav_desc.Texture2D.MipSlice = 0;
            fxaa_dest_uav_desc.Texture2D.PlaneSlice = 0;
            device->CreateUnorderedAccessView(guest_output_resource, nullptr, &fxaa_dest_uav_desc,
                                              fxaa_descriptors[0].first);
            deferred_command_list_.D3DSetComputeRootDescriptorTable(
                UINT(FxaaRootParameter::kDestination), fxaa_descriptors[0].second);
            D3D12_SHADER_RESOURCE_VIEW_DESC fxaa_source_srv_desc;
            fxaa_source_srv_desc.Format = kFxaaSourceTextureFormat;
            fxaa_source_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            fxaa_source_srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            fxaa_source_srv_desc.Texture2D.MostDetailedMip = 0;
            fxaa_source_srv_desc.Texture2D.MipLevels = 1;
            fxaa_source_srv_desc.Texture2D.PlaneSlice = 0;
            fxaa_source_srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
            device->CreateShaderResourceView(fxaa_source_texture_.Get(), &fxaa_source_srv_desc,
                                             fxaa_descriptors[1].first);
            deferred_command_list_.D3DSetComputeRootDescriptorTable(
                UINT(FxaaRootParameter::kSource), fxaa_descriptors[1].second);
            SetExternalPipeline(swap_post_effect == SwapPostEffect::kFxaaExtreme
                                    ? fxaa_extreme_pipeline_.Get()
                                    : fxaa_pipeline_.Get());
            SubmitBarriers();
            deferred_command_list_.D3DDispatch(group_count_x, group_count_y, 1);
            PushTransitionBarrier(guest_output_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
          }
        } else {
          assert_true(apply_gamma_dest_initial_state ==
                      ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
          PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                apply_gamma_dest_initial_state);
        }

        // Need to submit all the commands before giving the image back to the
        // presenter so it can submit its own commands for displaying it to the
        // queue.
        SubmitBarriers();
        EndSubmission(true);
        return true;
      });

  // End the frame even if did not present for any reason (the image refresher
  // was not called), to prevent leaking per-frame resources.
  EndSubmission(true);
}

void D3D12CommandProcessor::OnPrimaryBufferEnd() {
  if (REXCVAR_GET(d3d12_submit_on_primary_buffer_end) && submission_open_ &&
      CanEndSubmissionImmediately()) {
    EndSubmission(false);
  }
}

Shader* D3D12CommandProcessor::LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                                          const uint32_t* host_address, uint32_t dword_count) {
  // [GPU-PRECORD] Phase 1b-1c Inc 4 (H1): the parse thread mutates the shared shaders_
  // map here (emplace). Serialize against the replay worker's pipeline-cache access under
  // the coarse pipeline lock -- but only in worker (thread) mode, so precord-off / 1b-0 /
  // localrf-inline keep their exact single-threaded behavior with no lock at all.
  if (g_precord_thread) {
    std::lock_guard<std::mutex> pipeline_lock(precord_pipeline_mutex_);
    return pipeline_cache_->LoadShader(shader_type, host_address, dword_count);
  }
  return pipeline_cache_->LoadShader(shader_type, host_address, dword_count);
}

bool D3D12CommandProcessor::IssueDraw(xenos::PrimitiveType primitive_type, uint32_t index_count,
                                      IndexBufferInfo* index_buffer_info, bool major_mode_explicit) {
  // [NR-ISSUE] Increment 4d: the base executor armed this draw to be issued
  // from the composed shadow register file. Unsupported alongside precord
  // capture (both default off): fall through to the normal path there, but
  // count it so a silent misconfiguration shows on the [nr-issue] line.
  if (nr_issue_armed_) {
    if (!g_precord_capture) {
      return NrIssueDrawFromShadow(primitive_type, index_count, index_buffer_info,
                                   major_mode_explicit);
    }
    ++nr_issue_precord_skips_;
  }
  // [GPU-PRECORD] Phase 1b-0 capture wrapper. When capturing (and not already
  // replaying a captured segment), defer this draw into the current segment's
  // event log instead of recording it now; PrecordFlush replays the log later.
  if (!g_precord_capture || precord_replaying_) {
    return IssueDrawImpl(primitive_type, index_count, index_buffer_info, major_mode_explicit);
  }
  // A copy-mode "draw" records resolve commands (a non-draw recording op), so it
  // must order after the pending captured draws: flush them, then run it inline.
  if (register_file_->Get<reg::RB_MODECONTROL>().edram_mode == xenos::EdramMode::kCopy) {
    PrecordFlush();
    return IssueCopy();
  }
  // Open a segment on its first draw, snapshotting the register file now (it
  // already holds every write that set this draw up). Later writes are logged and
  // replayed in order so each draw sees exactly its own register state.
  if (!precord_segment_open_) {
    PrecordOpenSegment();
  }
  PrecordSegment& seg = precord_slots_[precord_capture_slot_];
  PrecordEvent ev;
  ev.kind = PrecordEvent::Kind::kDraw;
  ev.a = uint32_t(seg.draws.size());
  ev.b = 0;
  ev.c = 0;
  seg.events.push_back(ev);
  PrecordDraw draw;
  draw.primitive_type = primitive_type;
  draw.index_count = index_count;
  draw.has_index_buffer_info = index_buffer_info != nullptr;
  if (index_buffer_info) {
    draw.index_buffer_info = *index_buffer_info;
  }
  draw.major_mode_explicit = major_mode_explicit;
  // [GPU-PRECORD] capture the per-draw active shaders (parse-time CP state, not in
  // the register file) so replay can restore each draw's own shaders.
  draw.active_vs = active_vertex_shader_;
  draw.active_ps = active_pixel_shader_;
  // [GPU-PRECORD H3-PROBE] Inc 6: snapshot a hash of the guest index + vertex buffers
  // NOW (parse time). Replay re-hashes the same ranges; a change means the guest
  // recycled the source memory before the worker replayed this draw. Fields default to
  // 0 (no check) when off / when the shader has no analyzed vertex fetches yet.
  if (g_h3_probe) {
    if (index_buffer_info && index_buffer_info->count) {
      draw.h3_ib_hash =
          PrecordH3HashIndexBuffer(*index_buffer_info, index_count, &draw.h3_ib_len);
    }
    draw.h3_vb_hash = PrecordH3CaptureVertexBuffers(
        draw.active_vs, kH3MaxVbRanges, draw.h3_vb_range_addr, draw.h3_vb_range_size,
        &draw.h3_vb_range_count, &draw.h3_vb_len);
  }
  seg.draws.push_back(draw);
  if (++precord_draws_in_segment_ >= kParallelRecordSegmentDraws) {
    // [GPU-PRECORD] Phase 1b-1c Inc 5: the segment boundary -- under overlap this POSTs
    // to the worker and keeps capturing (does not block). Every other PrecordFlush caller
    // is a true flush point (default drain).
    PrecordFlush(/*from_segment_boundary=*/true);
  }
  return true;
}

bool D3D12CommandProcessor::NrIssueDrawFromShadow(xenos::PrimitiveType primitive_type,
                                                  uint32_t index_count,
                                                  IndexBufferInfo* index_buffer_info,
                                                  bool major_mode_explicit) {
  // [NR-NATIVE] Phase 5-0: build the seam record. The accessors return the
  // walk-resolved shaders here (the base's nr_issue_* gate is set for the
  // whole armed call).
  NrDrawInput input;
  input.regs = nr_issue_file_;
  input.vertex_shader = active_vertex_shader();
  input.pixel_shader = active_pixel_shader();
  input.primitive_type = primitive_type;
  input.index_count = index_count;
  input.index_buffer_info = index_buffer_info;
  input.major_mode_explicit = major_mode_explicit;
  return NrSubmitDraw(input);
}

bool D3D12CommandProcessor::NrSubmitDraw(const NrDrawInput& input) {
  // [NR-ISSUE] Increment 4d/4e: one draw, issued end to end from
  // walk-recovered state. The repoint/restore pair is PrecordReplayLocal's,
  // proven pixel-identical by the 1b-1b A/B; the file's contents are the
  // difference: the 4c shadow's decoded values on every register the stream
  // has written, live's on the rest (the four named externs, the two ports,
  // dead registers). 4e: the file is the base's PERSISTENT replay file,
  // maintained incrementally by the walk -- no copy at either end (4d's
  // per-draw 82 KB copy + full compose cost the city 4x its fps). Everything
  // runs inline on the CP thread, between two packets of the same draw, so no
  // other reader can observe the repointed holders.
  // [NR-NATIVE] Phase 5-0: this body IS the delegate path. The 5-x ladder
  // replaces it piecewise (state mirror, shader cache, native submission),
  // reading input.regs directly instead of repointing the emulated holders.
  const RegisterFile* local = input.regs;

  RegisterFile* shared = register_file_;
  active_draw_register_file_ = local;
  primitive_processor_->SetRegisterFile(local);
  render_target_cache_->SetRegisterFile(local);
  texture_cache_->SetRegisterFile(local);
  pipeline_cache_->SetRegisterFile(local);

  const bool draw_succeeded = IssueDrawImpl(input.primitive_type, input.index_count,
                                            input.index_buffer_info, input.major_mode_explicit);

  active_draw_register_file_ = nullptr;
  primitive_processor_->SetRegisterFile(shared);
  render_target_cache_->SetRegisterFile(shared);
  texture_cache_->SetRegisterFile(shared);
  pipeline_cache_->SetRegisterFile(shared);

  ++nr_issue_issued_;
  return draw_succeeded;
}

bool D3D12CommandProcessor::IssueDrawImpl(xenos::PrimitiveType primitive_type, uint32_t index_count,
                                          IndexBufferInfo* index_buffer_info,
                                          bool major_mode_explicit) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  // [PERF/DRAW-PROFILE] Time the whole IssueDraw (all return paths) + count it.
  struct DrawProfTotal {
    bool on;
    std::chrono::steady_clock::time_point t0;
    DrawProfTotal() : on(g_draw_prof) {
      if (on) t0 = std::chrono::steady_clock::now();
    }
    ~DrawProfTotal() {
      if (on) {
        g_draw_ns[5] += prof_ns_since(t0);
        ++g_draw_count;
      }
    }
  } _dp_total;

  ID3D12Device* device = GetD3D12Provider().GetDevice();
  const RegisterFile& regs = GetActiveDrawRegisterFile();

  xenos::EdramMode edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (edram_mode == xenos::EdramMode::kCopy) {
    // Special copy handling.
    // [NR-TIL] a resolve ends the tile segment: the census segments the
    // bands at exactly these points, and the base band of the next segment
    // is what the next repeat band replays.
    if (g_nr_tile_on) NrTileSegBreak();
    // [BATCH-CENSUS] a resolve breaks a mergeable run.
    if (g_batch_census) BatchCensusBreak();
    // [DRAW-POOL] a resolve reads what the pool's instances would still
    // write: close every batch.
    if (g_pool_on) PoolClose(kPoolCloseCopy);
    // [GPU-DRAW] other sub-bracket 16 (copy): the whole resolve path.
    bool _dp_copy_ok = IssueCopy();
    if (g_draw_prof) g_draw_ns[16] += prof_ns_since(_dp_total.t0);
    return _dp_copy_ok;
  }

  // [NR-TIL] N-4-1: tile-pass draw replay. BeginDraw advances the segment
  // ordinal for EVERY draw (a fall-through still holds its slot, or the
  // bands stop lining up); ReplayTry serves a repeat-band draw from the base
  // band recording. A false return is a fall-through -- every head step it
  // took is idempotent on the full path below.
  if (g_nr_tile_on) {
    // [NR-TILP] consume the outer arm: ostop = drawstop bracket entry to
    // here, obegin = NrTileBeginDraw itself. Held pending; ReplayTry's
    // success path commits them (a refusal must not charge a partial draw).
    const bool nr_otp = g_tile_oprof && g_tile_ostop_armed;
    g_tile_ostop_armed = false;
    std::chrono::steady_clock::time_point nr_ot_a;
    if (nr_otp) nr_ot_a = std::chrono::steady_clock::now();
    NrTileBeginDraw(regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET], uint32_t(primitive_type), index_count,
                    index_buffer_info ? index_buffer_info->guest_base : 0u);
    if (nr_otp) {
      const auto nr_ot_b = std::chrono::steady_clock::now();
      g_tile_pend_ostop = uint64_t(
          std::chrono::duration_cast<std::chrono::nanoseconds>(nr_ot_a - g_tile_ostop_t0)
              .count());
      g_tile_pend_obegin = uint64_t(
          std::chrono::duration_cast<std::chrono::nanoseconds>(nr_ot_b - nr_ot_a).count());
      g_tile_pend_valid = true;
    } else {
      g_tile_pend_valid = false;
    }
    if (NrTileReplayTry()) {
      // [GPU-DRAW2] rest sub-bracket 24 (rply): a replay-served draw closes
      // no phase bracket, so its whole time used to hide in rest.
      if (g_draw_prof) {
        g_draw_ns[24] += prof_ns_since(_dp_total.t0);
        ++g_draw_ret[0];
      }
      return true;
    }
  }


  // [NR-SPW] Phase 5-4-7-2: a reusable draw with a clean recording replays
  // it (live head + memcpy + patch) instead of running the derivation below.
  // A false return is a fall-through: every head step it took is idempotent
  // on this path.
  if (g_nr_span_consume && g_spr_open && NrSpanReplayTry()) {
    if (g_draw_prof) {
      g_draw_ns[24] += prof_ns_since(_dp_total.t0);
      ++g_draw_ret[1];
    }
    return true;
  }

  bool surface_pitch_is_zero = regs.Get<reg::RB_SURFACE_INFO>().surface_pitch == 0;

  // Vertex shader analysis.
  auto vertex_shader = static_cast<D3D12Shader*>(active_vertex_shader());
  if (!vertex_shader) {
    // Always need a vertex shader.
    if (g_draw_prof) ++g_draw_ret[2];
    return false;
  }

  // [GPU-INST] If an instanced batch is open, either extend it with this draw
  // (identical except the per-instance transform) or flush it before processing
  // this draw normally. The merge path skips all per-draw setup -- that skipped
  // work is the win. Safe because g_instance_dirty is false only when nothing
  // but the vertex float constants changed since the batch's last draw, and the
  // shader/primitive/index-buffer identity is checked explicitly.
  if (g_instance && instanced_batch_.active) {
    if (InstancedBatchCanMerge(primitive_type, index_count, index_buffer_info,
                               active_vertex_shader(), active_pixel_shader())) {
      InstancedBatchAppend(regs);
      ++g_instance_draws_in;
      g_instance_dirty = false;
      g_instance_dirty_first_reg = UINT32_MAX;
      return true;
    }
    ++g_inst_fail[g_inst_fail_reason];
    if (g_inst_fail_reason == 0 && g_instance_dirty_first_reg != UINT32_MAX) {
      ++g_inst_dirty_reg_census[g_instance_dirty_first_reg];
    }
    ++g_inst_flush_site[0];
    FlushInstancedBatch();
  }

  // [cull] increment 2: the frustum verdict from the draw's object-space
  // bounds and the shader's own clip transform. Refusals are counted by
  // class; a skipped draw leaves IssueDraw here (nothing below it has a side
  // effect the guest observes; memexport draws are refused).
  occ_draw_cull_ = 0;
  occ_draw_dbg_ = 0;
  // [hiz] every draw, before anything else: an open window survives only
  // while the depth it was tested against can only move nearer. A pass
  // change (another depth target) or a depth write whose function admits a
  // farther value closes it; the next eligible draw opens a new checkpoint.
  hiz_draw_.valid = false;
  hiz_draw_.dir_hint = 0;
  if (hiz_phase_ != 0) {
    ++hiz_draws_seen_;
    if (hiz_window_.open) {
      // The depth target's identity is checked at the draw site (the RT
      // cache binds there); here only the write direction can break it.
      const reg::RB_DEPTHCONTROL hdc = draw_util::GetNormalizedDepthControl(regs);
      if (hdc.z_enable && hdc.z_write_enable) {
        const xenos::CompareFunction f = hdc.zfunc;
        const bool mono =
            f == xenos::CompareFunction::kEqual || f == xenos::CompareFunction::kNever ||
            (hiz_window_.dir == 1 ? (f == xenos::CompareFunction::kGreater ||
                                     f == xenos::CompareFunction::kGreaterEqual)
                                  : (f == xenos::CompareFunction::kLess ||
                                     f == xenos::CompareFunction::kLessEqual));
        if (!mono) HizWindowClose(3);
      }
    }
  }
  if (cull_phase_ != 0 && !(g_tile_rec_open || g_tile_cmp_open || g_tile_mode != 0)) {
    ++cull_acc_.tested;
    const PosPath& cpp = vertex_shader->pos_path();
    uint32_t refuse = 0;
    if (!cpp.eligible) {
      refuse = 1;
    } else if (vertex_shader->memexport_eM_written() ||
               (active_pixel_shader() && active_pixel_shader()->memexport_eM_written())) {
      refuse = 2;
    } else if (primitive_type != xenos::PrimitiveType::kTriangleList &&
               primitive_type != xenos::PrimitiveType::kTriangleStrip &&
               primitive_type != xenos::PrimitiveType::kTriangleFan &&
               primitive_type != xenos::PrimitiveType::kLineList &&
               primitive_type != xenos::PrimitiveType::kLineStrip) {
      refuse = 3;
    } else {
      const auto ccl = regs.Get<reg::PA_CL_CLIP_CNTL>();
      const auto cvte = regs.Get<reg::PA_CL_VTE_CNTL>();
      if (ccl.clip_disable || cvte.vtx_xy_fmt || !cvte.vport_x_scale_ena ||
          !cvte.vport_y_scale_ena) {
        refuse = 4;
      } else if ((cpp.format != xenos::VertexFormat::k_32_32_32_FLOAT &&
                  cpp.format != xenos::VertexFormat::k_32_32_32_32_FLOAT) ||
                 ((cpp.used_basis_mask & 8) &&
                  cpp.format != xenos::VertexFormat::k_32_32_32_32_FLOAT)) {
        refuse = 5;
      }
    }
    if (refuse) {
      ++cull_acc_.refuse[refuse];
      cull_acc_.refuse_idx[refuse] += index_count;
    } else {
      NrCull::DrawDesc cd;
      cd.path = &cpp;
      cd.fetch = regs.GetVertexFetch(cpp.fetch_constant);
      cd.base_vertex = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
      cd.indexed = index_buffer_info != nullptr;
      cd.index_count = index_count;
      cd.ib_guest_base = index_buffer_info ? index_buffer_info->guest_base : 0u;
      cd.ib_format = index_buffer_info ? index_buffer_info->format : xenos::IndexFormat::kInt16;
      cd.ib_endian = index_buffer_info ? index_buffer_info->endianness : xenos::Endian::kNone;
      const CullBounds* cb = nr_cull_.GetBounds(cd, cull_stats_);
      if (!cb || cb->unbounded) {
        ++cull_acc_.refuse[6];
        cull_acc_.refuse_idx[6] += index_count;
      } else {
        ++cull_acc_.bounded;
        cull_acc_.bounded_idx += index_count;
        float cm[4][5];
        cpp.Evaluate(reinterpret_cast<const float*>(&regs[XE_GPU_REG_SHADER_CONSTANT_000_X]), cm);
        if (!NrCull::FrustumOutside(*cb, cm)) {
          if (hiz_phase_ != 0 && hiz_available_) HizPrepare(*cb, cm, regs, index_count);  // [hiz]
        } else {
          ++cull_acc_.out;
          cull_acc_.out_idx += index_count;
          if (cull_phase_ == 2) {
            ++cull_acc_.skipped;
            cull_acc_.skipped_idx += index_count;
            return true;
          }
          occ_draw_cull_ = 1;
          if (cull_dbg_.empty()) cull_dbg_.resize(kCullDbgRing);
          CullDbg& dg = cull_dbg_[cull_dbg_next_ % kCullDbgRing];
          dg.seq = cull_dbg_next_;
          dg.hash = vertex_shader->ucode_data_hash();
          std::memcpy(dg.mn, cb->mn, sizeof(dg.mn));
          std::memcpy(dg.mx, cb->mx, sizeof(dg.mx));
          std::memcpy(dg.m, cm, sizeof(dg.m));
          dg.index_count = index_count;
          dg.fetch0 = cd.fetch.dword_0;
          dg.fetch1 = cd.fetch.dword_1;
          dg.base_vertex = cd.base_vertex;
          dg.ib_base = cd.ib_guest_base;
          dg.stride = cpp.stride_dwords;
          dg.offset = uint32_t(cpp.offset_dwords);
          dg.indexed = cd.indexed ? 1 : 0;
          occ_draw_dbg_ = uint16_t((cull_dbg_next_ % kCullDbgRing) + 1);
          ++cull_dbg_next_;
        }
      }
    }
  }

  // [DRAW-POOL] key time. Before any derivation: the key, the window
  // bookkeeping (a non-opaque draw or an RT config change closes every
  // batch), the predictor, and the hit path.
  uint64_t pool_key = 0;
  bool pool_class_ok = false;
  uint32_t pool_pred = 0;
  if (g_pool_on) {
    ++g_pool.draws;
    g_gk_cur = 0;  // [GKEY]
    auto* pool_ps = static_cast<D3D12Shader*>(active_pixel_shader());
    uint64_t pk = 0;
    pk = inst_mix(pk, regs[XE_GPU_REG_RB_SURFACE_INFO]);
    pk = inst_mix(pk, regs[XE_GPU_REG_RB_MODECONTROL]);
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      pk = inst_mix(pk, regs[reg::RB_COLOR_INFO::rt_register_indices[i]]);
    }
    pk = inst_mix(pk, regs[XE_GPU_REG_RB_DEPTH_INFO]);
    if (!g_pool_pass_open || pk != g_pool_pass_key) {
      if (g_pool_pass_open) PoolClose(kPoolClosePass);
      g_pool_pass_key = pk;
      g_pool_pass_open = true;
    }
    pool_class_ok = PoolClassOpaque(regs);
    if (!pool_class_ok) {
      PoolClose(kPoolCloseClass);
    } else {
      ++g_pool.opq;
      pool_key = InstBatchKey(regs, vertex_shader, pool_ps, uint32_t(primitive_type), index_count,
                              0, index_buffer_info ? index_buffer_info->guest_base : 0u,
                              index_buffer_info ? int(index_buffer_info->format) : -1,
                              index_buffer_info ? index_buffer_info->count : 0u, nullptr);
      pool_key = inst_mix(pool_key, index_buffer_info ? uint32_t(index_buffer_info->endianness) + 1
                                                      : 0u);
      pool_key = inst_mix(pool_key, pk);
      uint32_t tm = vertex_shader->GetUsedTextureMaskAfterTranslation() |
                    (pool_ps ? pool_ps->GetUsedTextureMaskAfterTranslation() : 0u);
      uint32_t ti;
      while (rex::bit_scan_forward(tm, &ti)) {
        tm &= ~(uint32_t(1) << ti);
        const uint32_t base = XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 6 * ti;
        for (uint32_t j = 0; j < 6; ++j) pool_key = inst_mix(pool_key, regs[base + j]);
      }
      pool_pred = PoolPredict(pool_key);
      // [GKEY] rung-2 census: join the guest hash before the hit decision,
      // classify the outcome after it. nr_tile_draw_addr_ is the skip walk's
      // stop address of this draw's own packet (the record join key).
      uint64_t gk = 0;
      if (g_gk_census) {
        const uint32_t gk_addr = nr_tile_draw_addr_ & 0x1FFFFFFFu;
        const uint32_t* gk_pkt =
            gk_addr ? memory_->TranslatePhysical<const uint32_t*>(gk_addr) : nullptr;
        gk = GkJoin(memory_, regs, gk_pkt, gk_addr, uint32_t(primitive_type), index_count,
                    vertex_shader, pool_ps);
        g_gk_cur = gk;
      }
      if (g_tile_rec_open || g_tile_cmp_open || g_tile_mode != 0) {
        ++g_pool.ref_tile;
        if (gk) ++g_gk.tile;
      } else {
        const uint64_t gk_r[4] = {g_pool.ref_vfy_ps, g_pool.ref_vfy_regs, g_pool.ref_vfy_bl,
                                  g_pool.ref_cap};
        // [hiz-pool] drive 817 (RTX 1440p): the pool bypass of tested draws
        // dropped the hit share 29% -> 6% and the city to a 50 fps CP wall.
        // Now a tested draw joins its batch as an element of the per-instance
        // form (its own Hi-Z entry, hidden alone); a batch whose window
        // closed refuses the hit and the draw reopens a fresh one.
        PoolSlot* pool_slot = nullptr;
        PoolBatch* pool_batch =
            g_pool_intel_off ? nullptr : PoolMatch(pool_key, regs, pool_ps, &pool_slot);
        bool pool_hit = false;
        if (pool_batch) {
          if (pool_batch->hiz_base != UINT32_MAX) {
            if (HizPoolAppend(pool_batch->hiz_base, pool_batch->hiz_serial, pool_batch->count,
                              pool_batch->hiz_dir, pool_batch->idx_count)) {
              pool_hit = true;
            } else {
              ++hiz_acc_.pool_miss;
              if (pool_slot) pool_slot->batch = UINT32_MAX;
            }
          } else {
            pool_hit = true;
          }
          if (pool_hit) PoolCommit(*pool_batch, regs);
        }
        if (g_gk_census) GkAfter(gk, pool_key, pool_hit, gk_r);
        if (pool_hit) {
          g_spp_stop_kind = 1;  // [NR-SPP3]
          if (g_skc_on) {  // [SKC] the skipping host would hold the pre-group values
            SkcHit();
            SkcGroupReset();
          }
          GeoCount(index_count, true);
          ++g_frame_issued;
          if (hiz_draw_.valid) {  // [hiz] an instance of an earlier batch
            HizRefuse(9, hiz_draw_.index_count);
            hiz_draw_.valid = false;
          }
          return true;
        }
      }
    }
  }
  if (g_skc_on) {  // [SKC] a plain draw reads the register file
    SkcPlain(vertex_shader, static_cast<D3D12Shader*>(active_pixel_shader()));
    SkcGroupReset();
  }

  // [GPU-PRECORD] Phase 1b-1c Inc 4 (H1/H2): hold the coarse pipeline lock across the
  // shader-analysis / translation / ConfigurePipeline span so the replay worker's
  // pipeline-cache mutations (AnalyzeShaderUcode, GetOrCreateTranslation, ConfigurePipeline,
  // the pipeline-handle lookup) serialize against the parse thread's LoadShader. Gated on
  // worker (thread) mode via defer_lock: a no-op for precord-off / 1b-0 / localrf-inline,
  // and it is the single outer lock this thread holds -> deadlock-free. Released right after
  // the pipeline is configured; an early return in this span auto-releases it (RAII).
  std::unique_lock<std::mutex> pipeline_lock(precord_pipeline_mutex_, std::defer_lock);
  if (g_precord_thread) {
    pipeline_lock.lock();
  }

  pipeline_cache_->AnalyzeShaderUcode(*vertex_shader);
  bool memexport_used_vertex = vertex_shader->memexport_eM_written() != 0;

  // Pixel shader analysis.
  bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  bool is_rasterization_done = draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  if (surface_pitch_is_zero && is_rasterization_done) {
    // Doesn't actually draw.
    // Unlikely that zero would even really be legal though.
    if (g_draw_prof) ++g_draw_ret[3];
    return true;
  }
  D3D12Shader* pixel_shader = nullptr;
  if (is_rasterization_done) {
    // See xenos::EdramMode for explanation why the pixel shader is only used
    // when it's kColorDepth here.
    if (edram_mode == xenos::EdramMode::kColorDepth) {
      pixel_shader = static_cast<D3D12Shader*>(active_pixel_shader());
      if (pixel_shader) {
        pipeline_cache_->AnalyzeShaderUcode(*pixel_shader);
        if (!draw_util::IsPixelShaderNeededWithRasterization(*pixel_shader, regs)) {
          pixel_shader = nullptr;
        }
      }
    }
  } else {
    // Disabling pixel shader for this case is also required by the pipeline
    // cache.
    if (!memexport_used_vertex) {
      // This draw has no effect.
      if (g_draw_prof) ++g_draw_ret[3];
      return true;
    }
  }
  bool memexport_used_pixel = pixel_shader && (pixel_shader->memexport_eM_written() != 0);
  bool memexport_used = memexport_used_vertex || memexport_used_pixel;

  // [GPU-DRAW] other sub-bracket 12 (head): entry -> here. g_draw_prof is
  // launch-only, so it matched _dp_total.on when t0 was taken.
  if (g_draw_prof) g_draw_ns[12] += prof_ns_since(_dp_total.t0);
  auto _dp_bs0 = g_draw_prof ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  bool _dp_bs_ok = BeginSubmission(true);
  if (g_draw_prof) g_draw_ns[6] += prof_ns_since(_dp_bs0);
  if (!_dp_bs_ok) {
    return false;
  }
  // [GPU-DRAW2] rest sub-bracket 20 (cen): here -> primitive processing
  // (census class set, precord segment bookkeeping, rsy verify setup).
  auto _dp_cen0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  // [gpu-census] draw class (per-RT-config when known); no-op when already
  // there (one timestamp per class change, not per draw).
  GpuCensusSetClass(gpu_census_draw_class_);

  // [GPU-PRECORD] Phase 1a: at each segment boundary, force the following draws to
  // re-emit ALL command-list state (models a fresh per-segment command list). This
  // runs BEFORE any of this draw's state setup (Process/Configure/UpdateBindings),
  // so the re-emit lands in the current deferred list. Sequential single-list probe:
  // output MUST stay pixel-identical (the extra emits are redundant, not wrong).
  // counter==0 on a submission's first draw (BeginSubmission already did a full
  // reset), so the guard avoids a double reset there.
  if (g_parallel_record) {
    if (parallel_record_counter_ != 0 &&
        (parallel_record_counter_ % kParallelRecordSegmentDraws) == 0) {
      // [GPU-PRECORD] Phase 1a-ii: close the just-finished segment (move its stream
      // aside for ordered replay at EndSubmission), then force the new segment's
      // first draw to re-emit all state into the now-empty deferred list. Replaying
      // the saved segments in order, then the final list, reproduces the exact same
      // command sequence as the single-list path -> pixel-identical.
      precord_segments_.push_back(deferred_command_list_.TakeStream());
      ForceFullDrawStateReemit();
    }
    ++parallel_record_counter_;
  }

  // Process primitives.
  // [NR-RSY] Phase 5-3b-3: watch the index-buffer residency request this
  // Process call makes (the installed observer records the actual call).
  // [NR-VERIFY] inc 2: the predict/compare is verify; the observer itself
  // stays installed (map/mirror maintenance is not per-draw cost).
  const bool nr_rsy_verify = g_nr_res && g_nr_verify;
  if (nr_rsy_verify) {
    g_nr_res_ib_seen = false;
  }
  PrimitiveProcessor::ProcessingResult primitive_processing_result;
  if (g_draw_prof) g_draw_ns[20] += prof_ns_since(_dp_cen0);
  auto _dp_prim0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  bool _dp_prim_ok = primitive_processor_->Process(primitive_processing_result);
  if (g_draw_prof) g_draw_ns[0] += prof_ns_since(_dp_prim0);
  if (!_dp_prim_ok) {
    return false;
  }
  // [GPU-DRAW] other sub-bracket 13 (mods): here -> rt-update.
  auto _dp_omods0 = g_draw_prof ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
  // [NR-RSY] Phase 5-3b-3: our index-request derivation vs the observed call.
  // WHETHER the guest buffer is requested is the primitive processor's
  // convert-vs-DMA decision (declared, counted by outcome); the ARGUMENTS of
  // a request that does happen must equal ours.
  if (nr_rsy_verify) {
    ++g_nr_res_probe.ib_checks;
    nr::ResIndexPrediction nr_res_ibp;
    nr::ResIndexPredict(regs[XE_GPU_REG_VGT_DRAW_INITIATOR], regs[XE_GPU_REG_VGT_DMA_SIZE],
                        regs[XE_GPU_REG_VGT_DMA_BASE], SharedMemory::kBufferSize, &nr_res_ibp);
    const PrimitiveProcessor::ProcessedIndexBufferType nr_res_ibt =
        primitive_processing_result.index_buffer_type;
    if (nr_res_ibt == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA ||
        nr_res_ibt == PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA) {
      if (g_nr_res_ib_seen && nr_res_ibp.dma && !nr_res_ibp.out_of_bounds &&
          g_nr_res_ib_base == nr_res_ibp.base && g_nr_res_ib_len == nr_res_ibp.length) {
        ++g_nr_res_probe.ib_match;
      } else {
        ++g_nr_res_probe.ib_ne;
        if (g_nr_res_samples_this_window < kNrResMaxSamplesPerWindow) {
          ++g_nr_res_samples_this_window;
          REXGPU_WARN(
              "[nr-rsy] IB DIFF type={} seen={} ours dma={} oob={} {:08X}+{:X} theirs "
              "{:08X}+{:X}",
              uint32_t(nr_res_ibt), g_nr_res_ib_seen, nr_res_ibp.dma, nr_res_ibp.out_of_bounds,
              nr_res_ibp.base, nr_res_ibp.length, g_nr_res_ib_base, g_nr_res_ib_len);
        }
      }
    } else {
      if (nr_res_ibt == PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted) {
        ++g_nr_res_probe.ib_conv;
      } else {
        ++g_nr_res_probe.ib_auto;
      }
      if (g_nr_res_ib_seen) {
        ++g_nr_res_probe.ib_unexpected;
        if (g_nr_res_samples_this_window < kNrResMaxSamplesPerWindow) {
          ++g_nr_res_samples_this_window;
          REXGPU_WARN("[nr-rsy] IB UNEXPECTED type={} theirs {:08X}+{:X}", uint32_t(nr_res_ibt),
                      g_nr_res_ib_base, g_nr_res_ib_len);
        }
      }
    }
  }
  if (!primitive_processing_result.host_draw_vertex_count) {
    // Nothing to draw.
    if (g_draw_prof) ++g_draw_ret[4];
    return true;
  }

  reg::RB_DEPTHCONTROL normalized_depth_control = draw_util::GetNormalizedDepthControl(regs);

  // [GPU-INST] Decide whether this draw starts a new instanced batch. Only plain
  // (non-tessellated, kVertex) color/depth draws whose vertex shader uses purely
  // absolute float-constant addressing are eligible. Any open batch was already
  // flushed by the merge step above.
  bool start_instanced =
      g_instance && !memexport_used && is_rasterization_done &&
      edram_mode == xenos::EdramMode::kColorDepth &&
      !primitive_processing_result.IsTessellated() &&
      primitive_processing_result.host_vertex_shader_type ==
          Shader::HostVertexShaderType::kVertex &&
      vertex_shader->constant_register_map().float_count != 0 &&
      !vertex_shader->constant_register_map().float_dynamic_addressing;

  // [DRAW-POOL] does this first occurrence open a batch? The class and the
  // key were decided at key time; the predictor must have seen the key
  // repeat within a window last frame; the rest is the instanced VS's own
  // eligibility (the same set the old coalescer used) plus a snapshottable
  // pixel shader and no tile recording in flight.
  bool pool_open = false;
  // [hiz-pool] an eligible opener opens (the per-instance form); drive 821
  // found the bypass's other half still here: batches/frame 485 -> 58.
  if (g_pool_on && pool_class_ok && !g_instance && !g_pool_intel_off) {
    if (pool_pred < 2) {
      ++g_pool.ref_pred;
    } else if (g_tile_rec_open || g_tile_cmp_open || g_tile_mode != 0) {
      ++g_pool.ref_tile;
    } else if (memexport_used || !is_rasterization_done ||
               edram_mode != xenos::EdramMode::kColorDepth ||
               primitive_processing_result.IsTessellated() ||
               primitive_processing_result.host_vertex_shader_type !=
                   Shader::HostVertexShaderType::kVertex ||
               vertex_shader->constant_register_map().float_count == 0 ||
               vertex_shader->constant_register_map().float_dynamic_addressing ||
               !submission_open_) {
      ++g_pool.ref_elig;
    } else if (pixel_shader &&
               (pixel_shader->constant_register_map().float_dynamic_addressing ||
                pixel_shader->constant_register_map().float_count > kPoolMaxPsVec4)) {
      ++g_pool.ref_psbig;
    } else {
      pool_open = true;
    }
  }
  // Rung 1b-2: open PS-instanced when the key's pooled draws differed in PS
  // constants recently (last frame or this one). A geometry shader between
  // the stages (points, rectangles, quads) would not carry XEINSTANCEID:
  // those draws keep the plain pixel shader.
  const bool pool_ps_inst =
      pool_open && pixel_shader &&
      primitive_processing_result.host_primitive_type != xenos::PrimitiveType::kPointList &&
      primitive_processing_result.host_primitive_type != xenos::PrimitiveType::kRectangleList &&
      primitive_processing_result.host_primitive_type != xenos::PrimitiveType::kQuadList &&
      pixel_shader->constant_register_map().float_count != 0 && g_pool_last_pred &&
      g_pool_frame - g_pool_last_pred->ps_delta_frame <= 1;

  // Shader modifications.
  uint32_t ps_param_gen_pos = UINT32_MAX;
  uint32_t interpolator_mask =
      pixel_shader ? (vertex_shader->writes_interpolators() &
                      pixel_shader->GetInterpolatorInputMask(regs.Get<reg::SQ_PROGRAM_CNTL>(),
                                                             regs.Get<reg::SQ_CONTEXT_MISC>(),
                                                             ps_param_gen_pos))
                   : 0;
  DxbcShaderTranslator::Modification vertex_shader_modification =
      pipeline_cache_->GetCurrentVertexShaderModification(
          *vertex_shader, primitive_processing_result.host_vertex_shader_type, interpolator_mask,
          start_instanced || pool_open);
  DxbcShaderTranslator::Modification pixel_shader_modification =
      pixel_shader
          ? pipeline_cache_->GetCurrentPixelShaderModification(
                *pixel_shader, interpolator_mask, ps_param_gen_pos, normalized_depth_control)
          : DxbcShaderTranslator::Modification(0);
  // Rung 1b-2: the per-instance-constants pixel shader, paired with the
  // instancing vertex shader (pool_open implies it).
  if (pool_ps_inst) pixel_shader_modification.pixel.instanced = 1;

  // Set up the render targets - this may perform dispatches and draws.
  uint32_t normalized_color_mask =
      pixel_shader ? draw_util::GetNormalizedColorMask(regs, pixel_shader->writes_color_targets())
                   : 0;
  if (g_draw_prof) g_draw_ns[13] += prof_ns_since(_dp_omods0);
  auto _dp_rt0 = g_draw_prof ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  bool _dp_rt_ok = render_target_cache_->Update(is_rasterization_done, normalized_depth_control,
                                                normalized_color_mask, *vertex_shader);
  if (g_draw_prof) g_draw_ns[1] += prof_ns_since(_dp_rt0);
  if (!_dp_rt_ok) {
    return false;
  }
  // [GPU-DRAW] other sub-bracket 14 (trans): here -> ConfigurePipeline.
  auto _dp_otrans0 = g_draw_prof ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};

  // Create the pipeline (for this, need the actually used render target formats
  // from the render target cache), translating the shaders - doing this now to
  // obtain the used textures.
  D3D12Shader::D3D12Translation* vertex_shader_translation =
      static_cast<D3D12Shader::D3D12Translation*>(
          vertex_shader->GetOrCreateTranslation(vertex_shader_modification.value));
  D3D12Shader::D3D12Translation* pixel_shader_translation =
      pixel_shader ? static_cast<D3D12Shader::D3D12Translation*>(
                         pixel_shader->GetOrCreateTranslation(pixel_shader_modification.value))
                   : nullptr;
  uint32_t bound_depth_and_color_render_target_bits;
  uint32_t bound_depth_and_color_render_target_formats[1 + xenos::kMaxColorRenderTargets];
  bool host_render_targets_used =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
  if (host_render_targets_used) {
    bound_depth_and_color_render_target_bits =
        render_target_cache_->GetLastUpdateBoundRenderTargets(
            bound_depth_and_color_render_target_formats);
  } else {
    bound_depth_and_color_render_target_bits = 0;
  }
  void* pipeline_handle;
  ID3D12RootSignature* root_signature;
  if (g_draw_prof) g_draw_ns[14] += prof_ns_since(_dp_otrans0);
  // [GPU-DRAW2] rest sub-bracket 21 (ruse): here -> ConfigurePipeline (the
  // RUF-V2B upgrade verdict + the NrRubFind fast-pso probe).
  auto _dp_ruse0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  // [NR-RUF] 5-4-5-2: pipeline identity is a pure function of the draw's
  // inputs (5-1 description determinism + 5-3a description-keyed objects,
  // measured live: pso ne=0 under the bundle gate), and the objects are
  // persistent (never evicted). A v2-reusable draw therefore reuses its
  // previous execution's {handle, root signature, native object} and skips
  // ConfigurePipeline + the native lookup entirely.
  // [NR-RUF-V2B] 5-4-5-2b: compute this draw's upgrade verdict ONCE, before
  // the first consumer (the PSO bypass below; the bindings restore and both
  // bundle-compare sites read the same flag). A stale-only miss upgrades iff
  // every stale register is a float constant neither active shader reads:
  // bitmap-packed packs never carry an unread constant, so every derived and
  // uploaded artifact is byte-identical to the previous execution and the
  // 5-4-5-1 compare gate stays exact over upgraded draws. sh_eq inside the
  // stale-only flag guarantees the previous execution shared these bitmaps.
  g_ruf_v2b_up = false;
  if (g_nr_ruf_v2b) {
    uint32_t v2b_key;
    bool v2b_r2, v2b_sf, v2b_so;
    if (NrRuseCurrentDrawEx(&v2b_key, &v2b_r2, &v2b_sf, &v2b_so) && !v2b_r2 &&
        v2b_so) {
      const Shader::ConstantRegisterMap& v2b_vcm =
          vertex_shader->constant_register_map();
      const Shader::ConstantRegisterMap* v2b_pcm =
          pixel_shader ? &pixel_shader->constant_register_map() : nullptr;
      bool v2b_ok =
          !v2b_vcm.float_dynamic_addressing &&
          vertex_shader->memexport_stream_constants().empty() &&
          (!v2b_pcm ||
           (!v2b_pcm->float_dynamic_addressing &&
            pixel_shader->memexport_stream_constants().empty()));
      uint32_t v2b_regs[kNrRufV2bMaxStale];
      uint32_t v2b_n = 0;
      if (v2b_ok) {
        v2b_n = NrRuseStaleRegs(v2b_regs, kNrRufV2bMaxStale);
        v2b_ok = v2b_n > 0 && v2b_n <= kNrRufV2bMaxStale;
      }
      for (uint32_t i = 0; v2b_ok && i < v2b_n; ++i) {
        const uint32_t v2b_reg = v2b_regs[i];
        if (v2b_reg < nr::kBindFloatVertexBase ||
            v2b_reg >= nr::kBindFetchBase) {
          v2b_ok = false;
          break;
        }
        const uint32_t v2b_c = (v2b_reg - nr::kBindFloatVertexBase) >> 2;
        if (v2b_c < 256) {
          v2b_ok = (v2b_vcm.float_bitmap[v2b_c >> 6] &
                    (uint64_t(1) << (v2b_c & 63))) == 0;
        } else {
          const uint32_t v2b_pc = v2b_c - 256;
          v2b_ok = !v2b_pcm || (v2b_pcm->float_bitmap[v2b_pc >> 6] &
                                (uint64_t(1) << (v2b_pc & 63))) == 0;
        }
      }
      if (v2b_ok) {
        g_ruf_v2b_up = true;
        ++g_rub_probe.v2b_up;
      } else {
        ++g_rub_probe.v2b_ref;
      }
    }
  }
  bool nr_ruf_pso = false;
  if (g_nr_rub_fast) {
    uint32_t ruf_key;
    bool ruf_r2, ruf_sf;
    if (NrRuseCurrentDraw(&ruf_key, &ruf_r2, &ruf_sf) && (ruf_r2 || g_ruf_v2b_up)) {
      const NrRubBundle* ruf_b = NrRubFind(ruf_key);
      if (ruf_b && ruf_b->pso_valid) {
        pipeline_handle = ruf_b->pso_handle;
        root_signature = static_cast<ID3D12RootSignature*>(ruf_b->rootsig);
        nr_ruf_pso = root_signature != nullptr;
        if (nr_ruf_pso) ++g_rub_probe.fast_pso;
      }
    }
  }
  if (g_draw_prof) g_draw_ns[21] += prof_ns_since(_dp_ruse0);
  auto _dp_pso0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  bool _dp_pso_ok =
      nr_ruf_pso ||
      pipeline_cache_->ConfigurePipeline(
          vertex_shader_translation, pixel_shader_translation, primitive_processing_result,
          normalized_depth_control, normalized_color_mask, bound_depth_and_color_render_target_bits,
          bound_depth_and_color_render_target_formats, &pipeline_handle, &root_signature);
  if (g_draw_prof) g_draw_ns[2] += prof_ns_since(_dp_pso0);
  if (!_dp_pso_ok) {
    return false;
  }
  // [GPU-DRAW2] rest sub-bracket 22 (pret): here -> RequestTextures (the
  // async-compile cvar read + GetD3D12PipelineByHandle lookup + pipeline-lock
  // release + used-texture-mask derivation).
  auto _dp_pret0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  if (REXCVAR_GET(async_shader_compilation) &&
      pipeline_cache_->GetD3D12PipelineByHandle(pipeline_handle) == nullptr) {
    if (g_draw_prof) {
      g_draw_ns[22] += prof_ns_since(_dp_pret0);
      ++g_draw_ret[5];
    }
    return true;
  }

  // [GPU-PRECORD] Phase 1b-1c Inc 4: pipeline configured -> release the pipeline lock
  // before the texture-request / binding / draw-record tail (which touches no shared
  // pipeline-cache state). No-op when the lock was never taken.
  if (pipeline_lock.owns_lock()) {
    pipeline_lock.unlock();
  }

  // Update the textures - this may bind pipelines.
  uint32_t used_texture_mask =
      vertex_shader->GetUsedTextureMaskAfterTranslation() |
      (pixel_shader != nullptr ? pixel_shader->GetUsedTextureMaskAfterTranslation() : 0);
  if (g_draw_prof) g_draw_ns[22] += prof_ns_since(_dp_pret0);
  auto _dp_tex0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  texture_cache_->RequestTextures(used_texture_mask);
  // [NR-TIL] N-4-1: open the pipeline piece (texture loads and their
  // barriers are behind us; the pipeline set is the only thing ahead of the
  // viewport).
  if (g_tile_rec_open || g_tile_cmp_open) NrTileRecordAnchorA();
  if (g_draw_prof) g_draw_ns[3] += prof_ns_since(_dp_tex0);
  // [GPU-DRAW] other sub-bracket 15 (vp): here -> fixed-function state.
  auto _dp_ovp0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};

  // [NR-NPSO] Phase 5-3a: the first peel. Ask the native renderer for its own
  // pipeline object for this draw -- our description mapping over our shader
  // binaries -- and, when the bind cvar is on, record the draw with it instead
  // of the emulated cache's. Compare-only by default: the object is still
  // built, created and checked against theirs, but theirs is what draws, so a
  // mapping difference is reported before it can be seen.
  // [GPU-DRAW] vp sub-bracket 17 (npso): native pso lookup + pipeline bind.
  auto _dp_npso0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  ID3D12PipelineState* nr_native_pipeline = nullptr;
  if (g_nr_native_pso) {
    nr_native_pipeline = pipeline_cache_->NrNativePipeline(pipeline_handle, root_signature);
    if (g_nr_native_pso_bind) {
      nr::NrNpsoCountBind(nr_native_pipeline != nullptr);
    }
  }

  // Bind the pipeline after configuring it and doing everything that may bind
  // other pipelines.
  if (nr_native_pipeline && g_nr_native_pso_bind) {
    // Ours rides the same path the render target cache's own passes use.
    // current_guest_pipeline_ is cleared because the emulated pipeline is no
    // longer what is bound, so a later draw that does fall back must re-emit
    // it.
    SetExternalPipeline(nr_native_pipeline);
    current_guest_pipeline_ = nullptr;
  } else if (current_guest_pipeline_ != pipeline_handle) {
    deferred_command_list_.SetPipelineStateHandle(reinterpret_cast<void*>(pipeline_handle));
    current_guest_pipeline_ = pipeline_handle;
    current_external_pipeline_ = nullptr;
  }
  if (g_draw_prof) g_draw_ns[17] += prof_ns_since(_dp_npso0);

  // [NR-RUB] 5-4-5-1 extension: pipeline identity per draw key. Inputs
  // unchanged must mean the same pipeline handle and the same native object
  // (description-keyed, 5-1/5-3a determinism) -- evidence the ConfigurePipeline
  // bypass peel needs before it exists.
  if (g_nr_rub) {
    uint32_t rub_key;
    bool rub_r2, rub_sf;
    if (NrRuseCurrentDraw(&rub_key, &rub_r2, &rub_sf)) {
      NrRubBundle& rub_b = *NrRubGetOrCreate(rub_key);
      if ((rub_r2 || g_ruf_v2b_up) && rub_b.pso_valid) {
        if (rub_b.pso_handle != pipeline_handle ||
            rub_b.npso != static_cast<void*>(nr_native_pipeline)) {
          ++g_rub_probe.ne_pso;
        } else {
          ++g_rub_probe.pso_eq;
        }
      }
      rub_b.pso_handle = pipeline_handle;
      rub_b.npso = static_cast<void*>(nr_native_pipeline);
      rub_b.pso_valid = true;
    }
  }

  // Get dynamic rasterizer state.
  uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();

  bool convert_z_to_float24 =
      host_render_targets_used && render_target_cache_->depth_float24_convert_in_pixel_shader();
  bool ps_writes_depth = pixel_shader && pixel_shader->writes_depth();

  // Build a cache key from all viewport-affecting state to skip redundant
  // recalculation when the viewport registers haven't changed between draws.
  // [GPU-DRAW] vp sub-bracket 18 (vpk): key build/compare + viewport info.
  auto _dp_vpk0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  ViewportCacheKey viewport_key;
  viewport_key.pa_cl_clip_cntl = regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
  viewport_key.pa_cl_vte_cntl = regs[XE_GPU_REG_PA_CL_VTE_CNTL];
  viewport_key.pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
  viewport_key.pa_su_vtx_cntl = regs[XE_GPU_REG_PA_SU_VTX_CNTL];
  viewport_key.pa_sc_window_offset = regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
  viewport_key.normalized_depth_control = normalized_depth_control.value;
  std::memcpy(viewport_key.vport_regs, &regs[XE_GPU_REG_PA_CL_VPORT_XSCALE],
              sizeof(viewport_key.vport_regs));
  viewport_key.flags = (uint32_t(convert_z_to_float24) << 0) |
                       (uint32_t(host_render_targets_used) << 1) | (uint32_t(ps_writes_depth) << 2);

  draw_util::ViewportInfo viewport_info;
  if (viewport_cache_valid_ && viewport_key == previous_viewport_key_) {
    viewport_info = previous_viewport_info_;
  } else {
    draw_util::GetHostViewportInfo(regs, draw_resolution_scale_x, draw_resolution_scale_y, true,
                                   D3D12_VIEWPORT_BOUNDS_MAX, D3D12_VIEWPORT_BOUNDS_MAX, false,
                                   normalized_depth_control, convert_z_to_float24,
                                   host_render_targets_used, ps_writes_depth, viewport_info);
    previous_viewport_key_ = viewport_key;
    previous_viewport_info_ = viewport_info;
    viewport_cache_valid_ = true;
  }
  if (g_draw_prof) g_draw_ns[18] += prof_ns_since(_dp_vpk0);

  // [GPU-DRAW] vp sub-bracket 19 (sci): scissor derivation + scaling.
  auto _dp_sci0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  NrDetileWidenDrawScissor(regs, scissor);
  scissor.offset[0] *= draw_resolution_scale_x;
  scissor.offset[1] *= draw_resolution_scale_y;
  scissor.extent[0] *= draw_resolution_scale_x;
  scissor.extent[1] *= draw_resolution_scale_y;
  if (g_draw_prof) g_draw_ns[19] += prof_ns_since(_dp_sci0);

  if (g_draw_prof) g_draw_ns[15] += prof_ns_since(_dp_ovp0);
  auto _dp_ff0 = g_draw_prof ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  // Update viewport, scissor, blend factor and stencil reference.
  // [NR-TIL] N-4-1: close the pipeline piece before the bin-dependent sets.
  if (g_tile_rec_open || g_tile_cmp_open) NrTileRecordSplit();
  UpdateFixedFunctionState(viewport_info, scissor, primitive_polygonal, normalized_depth_control);

  // Update system constants before uploading them.
  // TODO(Triang3l): With ROV, pass the disabled render target mask for safety.
  UpdateSystemConstantValues(memexport_used, primitive_polygonal,
                             primitive_processing_result.line_loop_closing_index,
                             primitive_processing_result.host_shader_index_endian, viewport_info,
                             used_texture_mask, normalized_depth_control, normalized_color_mask);

  // [NR-TIL] N-4-1: the recording anchor. Everything band-dependent has been
  // emitted by now (UpdateFixedFunctionState wrote the viewport and scissor,
  // UpdateSystemConstantValues moved the NDC into the mirror), so the span
  // from here on is band-INVARIANT and can be memcpyd into the repeat bands.
  if (g_tile_rec_open || g_tile_cmp_open) NrTileRecordAnchor();
  if (g_draw_prof) g_draw_ns[8] += prof_ns_since(_dp_ff0);
  // [GPU-DRAW2] rest sub-bracket 23 (rsp): here -> UpdateBindings (the NR-BND
  // root-signature predict block).
  auto _dp_rsp0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};

  // [NR-BND] Phase 5-3b-0: the root-signature selection, ours against the one
  // the pipeline description carries. Bindless (this machine): one of two
  // static objects picked by tessellation alone. Bindful: our transcribed
  // count-packed index must resolve to the same cached object.
  if (g_nr_bindings) {
    ++g_nr_bind.rs_checks;
    const bool nr_tessellated = primitive_processing_result.IsTessellated();
    if (nr_tessellated) {
      ++g_nr_bind.rs_tess;
    }
    ID3D12RootSignature* nr_predicted = nullptr;
    if (bindless_resources_used_) {
      nr_predicted = nr_tessellated ? root_signature_bindless_ds_ : root_signature_bindless_vs_;
    } else {
      ++g_nr_bind.rs_bindful;
      const uint32_t nr_index = nr::BindRootSigBindfulIndex(
          pixel_shader ? uint32_t(pixel_shader->GetTextureBindingsAfterTranslation().size()) : 0,
          pixel_shader ? uint32_t(pixel_shader->GetSamplerBindingsAfterTranslation().size()) : 0,
          uint32_t(vertex_shader->GetTextureBindingsAfterTranslation().size()),
          uint32_t(vertex_shader->GetSamplerBindingsAfterTranslation().size()), nr_tessellated,
          D3D12Shader::kMaxTextureBindingIndexBits, D3D12Shader::kMaxSamplerBindingIndexBits);
      auto nr_it = root_signatures_bindful_.find(nr_index);
      if (nr_it != root_signatures_bindful_.end()) {
        nr_predicted = nr_it->second;
      }
    }
    if (nr_predicted != root_signature) {
      ++g_nr_bind.rs_ne;
    }
  }

  // Update constant buffers, descriptors and root parameters.
  if (g_draw_prof) g_draw_ns[23] += prof_ns_since(_dp_rsp0);
  auto _dp_bind0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  // [NR-SWP] Phase 5-3b swap: assemble the bindings with OUR UpdateBindings
  // when armed; a refusal (sampler-heap overflow) falls back to the emulated
  // one on the same coherent state machine, counted.
  bool _dp_bind_ok;
  if (g_nr_swap) {
    bool nr_swp_refused = false;
    _dp_bind_ok =
        NrUpdateBindings(vertex_shader, pixel_shader, root_signature, memexport_used,
                         &nr_swp_refused);
    if (nr_swp_refused) {
      ++g_nr_swap_probe.fallback;
      // [NR-RUB] the emulated retry recomposes into its own pools; the
      // staging mirrors no longer match the effective packs. The per-frame
      // latch re-arms by forcing one full recompose.
      g_rub_stage_ok = false;
      // [NR-LEAN] the emulated UpdateBindings uploads system_constants_; if
      // the lean path skipped the emulated derivation this draw, the mirror
      // holds the current values (it ran above) - one memcpy re-syncs.
      if (g_nr_sys_member_stale) {
        std::memcpy(&system_constants_, &g_nr_sys_state, sizeof(system_constants_));
        g_nr_sys_member_stale = false;
        ++g_nr_swap_probe.sys_lazy;
      }
      _dp_bind_ok = UpdateBindings(vertex_shader, pixel_shader, root_signature, memexport_used);
    } else {
      ++g_nr_swap_probe.swapped;
    }
  } else {
    _dp_bind_ok = UpdateBindings(vertex_shader, pixel_shader, root_signature, memexport_used);
    // [NR-RUB] a draw outside the swap invalidates the staging mirrors too.
    if (g_nr_rub) g_rub_stage_ok = false;
  }
  if (g_draw_prof) g_draw_ns[4] += prof_ns_since(_dp_bind0);
  if (!_dp_bind_ok) {
    return false;
  }
  // Must not call anything that can change the descriptor heap from now on!
  auto _dp_tail0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};

  // Ensure vertex buffers are resident.
  auto _dp_tres0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  const Shader::ConstantRegisterMap& constant_map_vertex = vertex_shader->constant_register_map();
  // [NR-RSY] Phase 5-3b-3: predict this draw's residency request list from
  // the same file the loop below reads, before the loop runs; the loop then
  // reports every actual RequestRange and its result, and the finish call at
  // each exit compares the lists, applies the walk to the mirror and checks
  // the mirror equals the emulated sync state bit for bit.
  if (nr_rsy_verify) {
    g_nr_res_vf_bitmap[0] = constant_map_vertex.vertex_fetch_bitmap[0];
    g_nr_res_vf_bitmap[1] = constant_map_vertex.vertex_fetch_bitmap[1];
    g_nr_res_vf_bitmap[2] = constant_map_vertex.vertex_fetch_bitmap[2];
    g_nr_res_vf_fetch_regs = &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0];
    g_nr_res_vf_allow = REXCVAR_GET(gpu_allow_invalid_fetch_constants);
    g_nr_res_vf_pred_status = nr::ResVfetchPredict(
        &g_nr_res_vf, g_nr_res_vf_bitmap, g_nr_res_vf_fetch_regs, g_nr_res_vf_allow,
        g_nr_res_vf_pred, kNrResMaxRequests, &g_nr_res_vf_pred_count);
    g_nr_res_vf_obs_count = 0;
    g_nr_res_vf_obs_ok = 0;
    g_nr_res_vf_active = true;
  }
  for (uint32_t i = 0; i < rex::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
    uint32_t vfetch_bits_remaining = constant_map_vertex.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(vfetch_bits_remaining, &j)) {
      vfetch_bits_remaining &= ~(uint32_t(1) << j);
      uint32_t vfetch_index = i * 32 + j;
      uint64_t vfetch_bit = uint64_t(1) << (vfetch_index & 63);
      if (g_draw_prof) ++g_draw_res_cnt[0];
      if (vertex_buffers_in_sync_[vfetch_index >> 6] & vfetch_bit) {
        if (g_draw_prof) ++g_draw_res_cnt[1];
        continue;
      }
      xenos::xe_gpu_vertex_fetch_t vfetch_constant = regs.GetVertexFetch(vfetch_index);
      switch (vfetch_constant.type) {
        case xenos::FetchConstantType::kVertex:
          break;
        case xenos::FetchConstantType::kInvalidVertex:
          if (REXCVAR_GET(gpu_allow_invalid_fetch_constants)) {
            break;
          }
          REXGPU_WARN(
              "Vertex fetch constant {} ({:08X} {:08X}) has \"invalid\" type! "
              "This is incorrect behavior, but you can try bypassing this by "
              "launching Xenia with --gpu_allow_invalid_fetch_constants=true.",
              vfetch_index, vfetch_constant.dword_0, vfetch_constant.dword_1);
          if (g_nr_res) {
            ++g_nr_res_probe.vf_abort_type;
            NrResVfetchFinishDraw(1);
          }
          return false;
        default:
          REXGPU_WARN("Vertex fetch constant {} ({:08X} {:08X}) is completely invalid!",
                      vfetch_index, vfetch_constant.dword_0, vfetch_constant.dword_1);
          if (g_nr_res) {
            ++g_nr_res_probe.vf_abort_type;
            NrResVfetchFinishDraw(1);
          }
          return false;
      }
      VertexBufferState& state = vertex_buffer_states_[vfetch_index];
      if (state.address == vfetch_constant.address && state.size == vfetch_constant.size) {
        if (g_draw_prof) ++g_draw_res_cnt[2];
        vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
        continue;
      }
      if (g_draw_prof) {
        ++g_draw_res_cnt[3];
        const uint64_t res_key =
            (uint64_t(vfetch_constant.address) << 32) | uint64_t(vfetch_constant.size);
        if (!g_draw_res_seen.insert(res_key).second) {
          ++g_draw_res_cnt[4];
        }
      }
      const bool nr_res_range_ok =
          shared_memory_->RequestRange(vfetch_constant.address << 2, vfetch_constant.size << 2);
      if (g_nr_res && g_nr_res_vf_active) {
        if (g_nr_res_vf_obs_count < kNrResMaxRequests) {
          g_nr_res_vf_obs[g_nr_res_vf_obs_count].start = vfetch_constant.address << 2;
          g_nr_res_vf_obs[g_nr_res_vf_obs_count].length = vfetch_constant.size << 2;
        }
        ++g_nr_res_vf_obs_count;
        if (nr_res_range_ok) {
          ++g_nr_res_vf_obs_ok;
        } else {
          ++g_nr_res_probe.vf_req_fail;
        }
      }
      if (!nr_res_range_ok) {
        REXGPU_ERROR(
            "Failed to request vertex buffer at 0x{:08X} (size {}) in the "
            "shared memory",
            vfetch_constant.address << 2, vfetch_constant.size << 2);
        if (g_nr_res) {
          NrResVfetchFinishDraw(2);
        }
        return false;
      }
      state.address = vfetch_constant.address;
      state.size = vfetch_constant.size;
      vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
    }
  }
  // [NR-RSY] The loop completed: close this draw's residency compare.
  if (g_nr_res) {
    NrResVfetchFinishDraw(0);
  }
  if (g_draw_prof) g_draw_ns[9] += prof_ns_since(_dp_tres0);

  // [GPU-DRAW-DUMP] Native-renderer R&D (Ch.9 path B): emit this draw's full
  // geometry/shader stream. IssueDraw is the ONE seam that sees every world draw
  // (the guest scene-graph traversal only covers the HUD), so this is the data
  // model a native renderer replays. Header line + one line per bound vertex
  // buffer (same vfetch iteration as the residency loop above). Auto-bounded.
  if (g_draw_dump && g_draw_dump_count < kDrawDumpCap) {
    const uint64_t dn = g_draw_dump_count++;
    const uint32_t ib_base = index_buffer_info ? index_buffer_info->guest_base : 0u;
    const uint32_t ib_count = index_buffer_info ? index_buffer_info->count : 0u;
    const int ib_fmt = index_buffer_info ? static_cast<int>(index_buffer_info->format) : -1;
    REXGPU_INFO(
        "[gpu-drawdump] #{} prim={} idx={} vtx={} vs={:016X}/{} ps={:016X}/{} ib={:08X}:fmt{}:cnt{}",
        dn, static_cast<uint32_t>(primitive_type), index_count,
        primitive_processing_result.host_draw_vertex_count, vertex_shader->ucode_data_hash(),
        static_cast<uint64_t>(vertex_shader->ucode_dword_count()),
        pixel_shader ? pixel_shader->ucode_data_hash() : uint64_t(0),
        pixel_shader ? static_cast<uint64_t>(pixel_shader->ucode_dword_count()) : uint64_t(0),
        ib_base, ib_fmt, ib_count);
    const Shader::ConstantRegisterMap& cm_dump = vertex_shader->constant_register_map();
    for (uint32_t i = 0; i < rex::countof(cm_dump.vertex_fetch_bitmap); ++i) {
      uint32_t bits = cm_dump.vertex_fetch_bitmap[i];
      uint32_t j;
      while (rex::bit_scan_forward(bits, &j)) {
        bits &= ~(uint32_t(1) << j);
        const uint32_t vfi = i * 32 + j;
        const xenos::xe_gpu_vertex_fetch_t vf = regs.GetVertexFetch(vfi);
        REXGPU_INFO("[gpu-drawdump-vb] #{} slot={} addr={:08X} size={} type={}", dn, vfi,
                    static_cast<uint32_t>(vf.address) << 2, static_cast<uint32_t>(vf.size) << 2,
                    static_cast<uint32_t>(vf.type));
      }
    }
  }

  // [INST-PROBE] Instancing feasibility spike: accumulate which float constants vary
  // per-instance within each batch key, then dump the report once the cap is hit.
  if (g_inst_probe && !g_inst_dumped && g_inst_draw_count < kDrawDumpCap) {
    const uint32_t ip_ib_base = index_buffer_info ? index_buffer_info->guest_base : 0u;
    const uint32_t ip_ib_count = index_buffer_info ? index_buffer_info->count : 0u;
    const int ip_ib_fmt = index_buffer_info ? static_cast<int>(index_buffer_info->format) : -1;
    InstanceProbeDraw(regs, vertex_shader, pixel_shader,
                      static_cast<uint32_t>(primitive_type), index_count,
                      primitive_processing_result.host_draw_vertex_count, ip_ib_base, ip_ib_fmt,
                      ip_ib_count);
    if (++g_inst_draw_count >= kDrawDumpCap) {
      InstanceProbeDump();
      g_inst_dumped = true;
    }
  }

  // Gather memexport ranges and ensure the heaps for them are resident, and
  // also load the data surrounding the export and to fill the regions that
  // won't be modified by the shaders.
  // [NR-RSY] Phase 5-3b-3: memexport residency is NOT transcribed (refused
  // class, like 5-3b-1's ROV) - its requests pass through uncompared and the
  // draw is counted. mx must stay 0 in this game.
  if (g_nr_res && memexport_used) {
    ++g_nr_res_probe.mx_draws;
  }
  memexport_ranges_.clear();
  if (memexport_used_vertex) {
    draw_util::AddMemExportRanges(regs, *vertex_shader, memexport_ranges_);
  }
  if (memexport_used_pixel) {
    draw_util::AddMemExportRanges(regs, *pixel_shader, memexport_ranges_);
  }
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    if (!shared_memory_->RequestRange(memexport_range.base_address_dwords << 2,
                                      memexport_range.size_bytes)) {
      REXGPU_ERROR(
          "Failed to request memexport stream at 0x{:08X} (size {}) in the "
          "shared memory",
          memexport_range.base_address_dwords << 2, memexport_range.size_bytes);
      return false;
    }
  }
  if (memexport_used && memexport_ranges_.empty()) {
    if (!shared_memory_->RequestRange(0, SharedMemory::kBufferSize)) {
      REXGPU_ERROR(
          "Failed to request full shared memory residency for unresolved "
          "memexport destinations");
      return false;
    }
  }

  // Primitive topology.
  auto _dp_tib0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  D3D_PRIMITIVE_TOPOLOGY primitive_topology;
  if (primitive_processing_result.IsTessellated()) {
    switch (primitive_processing_result.host_primitive_type) {
      // TODO(Triang3l): Support all primitive types.
      case xenos::PrimitiveType::kTriangleList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST;
        break;
      case xenos::PrimitiveType::kQuadList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST;
        break;
      case xenos::PrimitiveType::kTrianglePatch:
        primitive_topology =
            (regs.Get<reg::VGT_HOS_CNTL>().tess_mode == xenos::TessellationMode::kAdaptive)
                ? D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST
                : D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST;
        break;
      case xenos::PrimitiveType::kQuadPatch:
        primitive_topology =
            (regs.Get<reg::VGT_HOS_CNTL>().tess_mode == xenos::TessellationMode::kAdaptive)
                ? D3D_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST
                : D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST;
        break;
      default:
        REXGPU_ERROR(
            "Host tessellated primitive type {} returned by the primitive "
            "processor is not supported by the Direct3D 12 command processor",
            uint32_t(primitive_processing_result.host_primitive_type));
        assert_unhandled_case(primitive_processing_result.host_primitive_type);
        return false;
    }
  } else {
    switch (primitive_processing_result.host_primitive_type) {
      case xenos::PrimitiveType::kPointList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
        break;
      case xenos::PrimitiveType::kLineList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        break;
      case xenos::PrimitiveType::kLineStrip:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
        break;
      case xenos::PrimitiveType::kTriangleList:
      case xenos::PrimitiveType::kRectangleList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        break;
      case xenos::PrimitiveType::kTriangleStrip:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        break;
      case xenos::PrimitiveType::kQuadList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ;
        break;
      default:
        REXGPU_ERROR(
            "Host primitive type {} returned by the primitive processor is not "
            "supported by the Direct3D 12 command processor",
            uint32_t(primitive_processing_result.host_primitive_type));
        assert_unhandled_case(primitive_processing_result.host_primitive_type);
        return false;
    }
  }
  SetPrimitiveTopology(primitive_topology);
  // Must not call anything that may change the primitive topology from now on!
  // [NR-SPR] record-time capture of what a span replay's live head needs.
  // Refused classes are never replayable: memexport (barrier/UAV flow),
  // tessellated / non-kVertex host shaders, an instanced-batch start (the
  // draw is deferred out of the bracket); converted index buffers refuse in
  // the indexed branch below (their pool VA is frame-scoped).
  if (g_spr_open || g_tile_rec_open) {
    g_spr_cap.valid = true;
    g_spr_cap.refused = memexport_used || primitive_processing_result.IsTessellated() ||
                        primitive_processing_result.host_vertex_shader_type !=
                            Shader::HostVertexShaderType::kVertex ||
                        start_instanced;
    g_spr_cap.vs = vertex_shader;
    g_spr_cap.ps = pixel_shader;
    g_spr_cap.tex_mask = used_texture_mask;
    g_spr_cap.llci = primitive_processing_result.line_loop_closing_index;
    g_spr_cap.index_endian = uint8_t(primitive_processing_result.host_shader_index_endian);
    g_spr_cap.ib_dma = 0;
    g_spr_cap.ib_base = 0;
    g_spr_cap.ib_size = 0;
  }
  if (g_draw_prof) g_draw_ns[10] += prof_ns_since(_dp_tib0);
  auto _dp_temit0 = g_draw_prof ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};

  // Draw.
  if (primitive_processing_result.index_buffer_type ==
      PrimitiveProcessor::ProcessedIndexBufferType::kNone) {
    if (memexport_used) {
      shared_memory_->UseForWriting();
    } else {
      shared_memory_->UseForReading();
    }
    SubmitBarriers();
    PROFILE_DRAW_CALL();
    PROFILE_VERTICES(primitive_processing_result.host_draw_vertex_count);
    GeoCount(primitive_processing_result.host_draw_vertex_count, false);
    OccSetElig(vertex_shader, regs);  // [occ-pos]
    // [GPU-INST] Defer the draw to coalesce following identical-except-transform
    // draws; the pipeline/bindings/barriers above are recorded once for the run.
    // [DRAW-POOL] or open a batch: one indirect draw in place, instance 0 = this.
    if (inst_base_dirty_ && (pool_open || start_instanced)) HizInstanceBaseReset();  // [hiz-pool]
    if (pool_open &&
        PoolOpenAndDraw(pool_key, regs, vertex_shader, pixel_shader,
                        primitive_processing_result.host_draw_vertex_count, /*indexed=*/false,
                        std::min<uint32_t>(pool_pred + 1 + pool_pred / 4, 256), pool_ps_inst)) {
      if (hiz_draw_.valid) HizRefuse(9, hiz_draw_.index_count);  // [hiz] a batch opener
    } else if (start_instanced) {
      if (hiz_draw_.valid) HizRefuse(9, hiz_draw_.index_count);  // [hiz]
      StartInstancedBatch(regs, active_vertex_shader(), active_pixel_shader(), primitive_type,
                          index_count, index_buffer_info,
                          primitive_processing_result.host_draw_vertex_count, /*indexed=*/false);
    } else if (hiz_draw_.valid &&
               HizDraw(/*indexed=*/false, primitive_processing_result.host_draw_vertex_count)) {
      // [hiz] issued as an ExecuteIndirect whose arguments the checkpoint writes.
    } else {
      OccDrawBegin();  // [occ]
      deferred_command_list_.D3DDrawInstanced(
          primitive_processing_result.host_draw_vertex_count, 1, 0, 0);
      OccDrawEnd(primitive_processing_result.host_draw_vertex_count, 0);
    }
  } else {
    D3D12_INDEX_BUFFER_VIEW index_buffer_view;
    index_buffer_view.SizeInBytes = primitive_processing_result.host_draw_vertex_count;
    if (primitive_processing_result.host_index_format == xenos::IndexFormat::kInt16) {
      index_buffer_view.SizeInBytes *= sizeof(uint16_t);
      index_buffer_view.Format = DXGI_FORMAT_R16_UINT;
    } else {
      index_buffer_view.SizeInBytes *= sizeof(uint32_t);
      index_buffer_view.Format = DXGI_FORMAT_R32_UINT;
    }
    ID3D12Resource* scratch_index_buffer = nullptr;
    switch (primitive_processing_result.index_buffer_type) {
      case PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA: {
        if (memexport_used) {
          // If the shared memory is a UAV, it can't be used as an index buffer
          // (UAV is a read/write state, index buffer is a read-only state).
          // Need to copy the indices to a buffer in the index buffer state.
          scratch_index_buffer = RequestScratchGPUBuffer(index_buffer_view.SizeInBytes,
                                                         D3D12_RESOURCE_STATE_COPY_DEST);
          if (scratch_index_buffer == nullptr) {
            return false;
          }
          shared_memory_->UseAsCopySource();
          SubmitBarriers();
          deferred_command_list_.D3DCopyBufferRegion(
              scratch_index_buffer, 0, shared_memory_->GetBuffer(),
              primitive_processing_result.guest_index_base, index_buffer_view.SizeInBytes);
          PushTransitionBarrier(scratch_index_buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_INDEX_BUFFER);
          index_buffer_view.BufferLocation = scratch_index_buffer->GetGPUVirtualAddress();
        } else {
          index_buffer_view.BufferLocation =
              shared_memory_->GetGPUAddress() + primitive_processing_result.guest_index_base;
          // [NR-SPR] the replay must re-request this range (residency is
          // the one live dependency of a DMA index buffer).
          if (g_spr_open || g_tile_rec_open) {
            g_spr_cap.ib_dma = 1;
            g_spr_cap.ib_base = primitive_processing_result.guest_index_base;
            g_spr_cap.ib_size = index_buffer_view.SizeInBytes;
          }
        }
      } break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted:
        index_buffer_view.BufferLocation = primitive_processor_->GetConvertedIndexBufferGpuAddress(
            primitive_processing_result.host_index_buffer_handle);
        // [NR-SPR] converted indices live in a frame-scoped pool: the
        // recorded IB view would go stale. Never replayable.
        if (g_spr_open || g_tile_rec_open) {
          g_spr_cap.refused = true;
        }
        break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto:
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA:
        index_buffer_view.BufferLocation = primitive_processor_->GetBuiltinIndexBufferGpuAddress(
            primitive_processing_result.host_index_buffer_handle);
        break;
      default:
        assert_unhandled_case(primitive_processing_result.index_buffer_type);
        return false;
    }
    deferred_command_list_.D3DIASetIndexBuffer(&index_buffer_view);
    if (memexport_used) {
      shared_memory_->UseForWriting();
    } else {
      shared_memory_->UseForReading();
    }
    SubmitBarriers();
    PROFILE_DRAW_CALL();
    PROFILE_VERTICES(primitive_processing_result.host_draw_vertex_count);
    GeoCount(primitive_processing_result.host_draw_vertex_count, false);
    OccSetElig(vertex_shader, regs);  // [occ-pos]
    // [GPU-INST] Defer the draw to coalesce the run (memexport draws are never
    // instanced, so scratch_index_buffer is always null on this path).
    // [DRAW-POOL] or open a batch: one indirect draw in place, instance 0 = this.
    if (inst_base_dirty_ && (pool_open || start_instanced)) HizInstanceBaseReset();  // [hiz-pool]
    if (pool_open &&
        PoolOpenAndDraw(pool_key, regs, vertex_shader, pixel_shader,
                        primitive_processing_result.host_draw_vertex_count, /*indexed=*/true,
                        std::min<uint32_t>(pool_pred + 1 + pool_pred / 4, 256), pool_ps_inst)) {
      if (hiz_draw_.valid) HizRefuse(9, hiz_draw_.index_count);  // [hiz] a batch opener
    } else if (start_instanced) {
      if (hiz_draw_.valid) HizRefuse(9, hiz_draw_.index_count);  // [hiz]
      StartInstancedBatch(regs, active_vertex_shader(), active_pixel_shader(), primitive_type,
                          index_count, index_buffer_info,
                          primitive_processing_result.host_draw_vertex_count, /*indexed=*/true);
    } else if (hiz_draw_.valid &&
               HizDraw(/*indexed=*/true, primitive_processing_result.host_draw_vertex_count)) {
      // [hiz] issued as an ExecuteIndirect whose arguments the checkpoint writes.
    } else {
      OccDrawBegin();  // [occ]
      deferred_command_list_.D3DDrawIndexedInstanced(
          primitive_processing_result.host_draw_vertex_count, 1, 0, 0, 0);
      OccDrawEnd(primitive_processing_result.host_draw_vertex_count, 0);
    }
    if (scratch_index_buffer != nullptr) {
      ReleaseScratchGPUBuffer(scratch_index_buffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
    }
  }

  if (memexport_used) {
    // Make sure this memexporting draw is ordered with other work using shared
    // memory as a UAV.
    // TODO(Triang3l): Find some PM4 command that can be used for indication of
    // when memexports should be awaited?
    shared_memory_->MarkUAVWritesCommitNeeded();
    // Invalidate textures in memexported memory and watch for changes.
    if (!memexport_ranges_.empty()) {
      for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
        shared_memory_->RangeWrittenByGpu(memexport_range.base_address_dwords << 2,
                                          memexport_range.size_bytes);
      }
    } else {
      // Stream constants can be invalid or dynamic, so exact destinations may
      // be unknown. Keep invalidation conservative in this case.
      shared_memory_->RangeWrittenByGpu(0, SharedMemory::kBufferSize);
    }
    if (IsReadbackMemexportEnabled(REXCVAR_GET(d3d12_readback_memexport)) &&
        !memexport_ranges_.empty()) {
      uint32_t memexport_total_size = 0;
      for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
        memexport_total_size += memexport_range.size_bytes;
      }
      if (memexport_total_size != 0) {
        if (REXCVAR_GET(readback_memexport_fast)) {
          IssueDraw_MemexportReadbackFastPath(memexport_total_size);
        } else {
          IssueDraw_MemexportReadbackFullPath(memexport_total_size);
        }
      }
    }
  }

  if (g_draw_prof) {
    g_draw_ns[11] += prof_ns_since(_dp_temit0);
    g_draw_ns[7] += prof_ns_since(_dp_tail0);
  }
  // [NR-TIL] N-4-1: close the recording. Reached only by a draw that made it
  // all the way to emission -- every earlier `return false` leaves the
  // record valid=0, which is exactly right: the repeat bands must fall
  // through at that ordinal too, and the ordinal itself is already held.
  // [GPU-DRAW2] rest sub-bracket 25 (tilr): the post-tail record close.
  auto _dp_tilr0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  if (g_tile_rec_open) NrTileRecordEnd();
  if (g_tile_cmp_open) NrTileCompareEnd();
  if (g_draw_prof) g_draw_ns[25] += prof_ns_since(_dp_tilr0);
  // [INST-PROBE] city latch input: issued draws this frame.
  ++g_frame_issued;
  // [BATCH-CENSUS] the seam: an issued draw that reached emission. Outside
  // every phase bracket; its own cost is bracket 26 (bc) so it cannot muddy
  // rr when both cvars ride one drive.
  if (g_batch_census) {
    auto _dp_bc0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    const uint64_t bc_key = InstBatchKey(
        regs, vertex_shader, pixel_shader, uint32_t(primitive_type), index_count,
        primitive_processing_result.host_draw_vertex_count,
        index_buffer_info ? index_buffer_info->guest_base : 0u,
        index_buffer_info ? int(index_buffer_info->format) : -1,
        index_buffer_info ? index_buffer_info->count : 0u, nullptr);
    BatchCensusDraw(bc_key);
    // [BATCH-CENSUS v2] the pass key = the render-target config this draw
    // writes; the full key = v1 + every used texture fetch constant + the
    // blend/depth/mask/stencil/alpha state + viewport/scissor/offset + cull
    // + the pass key. Order class from the same registers: opaque = no RT
    // blends AND depth tested AND depth written.
    {
      // The pass = the BOUND render-target config (raw registers), not what
      // this draw writes: a depth-only draw into the same buffers is the
      // same pass (the mask-dependent key split 157 passes/frame, naruto_761).
      uint64_t pk = 0;
      pk = inst_mix(pk, regs[XE_GPU_REG_RB_SURFACE_INFO]);
      pk = inst_mix(pk, regs[XE_GPU_REG_RB_MODECONTROL]);
      for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
        pk = inst_mix(pk, regs[reg::RB_COLOR_INFO::rt_register_indices[i]]);
      }
      pk = inst_mix(pk, regs[XE_GPU_REG_RB_DEPTH_INFO]);
      uint64_t k2 = inst_mix(bc_key, pk);
      const uint32_t blend0_raw = regs[XE_GPU_REG_RB_BLENDCONTROL0] & 0x1FFF1FFF;
      bool blends = false;
      for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
        if (!((normalized_color_mask >> (4 * i)) & 0xF)) continue;
        const uint32_t bcv = regs[reg::RB_BLENDCONTROL::rt_register_indices[i]] & 0x1FFF1FFF;
        k2 = inst_mix(k2, bcv);
        reg::RB_BLENDCONTROL bc;
        bc.value = bcv;
        if (bc.color_srcblend != xenos::BlendFactor::kOne ||
            bc.color_destblend != xenos::BlendFactor::kZero ||
            bc.color_comb_fcn != xenos::BlendOp::kAdd ||
            bc.alpha_srcblend != xenos::BlendFactor::kOne ||
            bc.alpha_destblend != xenos::BlendFactor::kZero ||
            bc.alpha_comb_fcn != xenos::BlendOp::kAdd) {
          blends = true;
        }
      }
      k2 = inst_mix(k2, normalized_color_mask);
      k2 = inst_mix(k2, normalized_depth_control.value);
      const auto colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
      k2 = inst_mix(k2, colorcontrol.value & 0x1F);
      if (colorcontrol.alpha_test_enable) k2 = inst_mix(k2, regs[XE_GPU_REG_RB_ALPHA_REF]);
      if (normalized_depth_control.stencil_enable) {
        k2 = inst_mix(k2, regs[XE_GPU_REG_RB_STENCILREFMASK]);
        k2 = inst_mix(k2, regs[XE_GPU_REG_RB_STENCILREFMASK_BF]);
      }
      for (uint32_t j = 0; j < 6; ++j) k2 = inst_mix(k2, regs[XE_GPU_REG_PA_CL_VPORT_XSCALE + j]);
      k2 = inst_mix(k2, regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL]);
      k2 = inst_mix(k2, regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR]);
      k2 = inst_mix(k2, regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET]);
      k2 = inst_mix(k2, regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL]);
      k2 = inst_mix(k2, regs[XE_GPU_REG_PA_CL_VTE_CNTL]);
      const bool zw = normalized_depth_control.z_enable && normalized_depth_control.z_write_enable;
      const uint32_t cls = blends ? (zw ? kBc2Bzw : kBc2Bld) : (zw ? kBc2Opq : kBc2Zro);
      // The texture fetch constants last, so the key without them (win3) is
      // the prefix of the full key.
      const uint64_t k3 = k2;
      uint32_t tm = used_texture_mask, ti;
      while (rex::bit_scan_forward(tm, &ti)) {
        tm &= ~(uint32_t(1) << ti);
        const uint32_t base = XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 6 * ti;
        for (uint32_t j = 0; j < 6; ++j) k2 = inst_mix(k2, regs[base + j]);
      }
      Bc2Draw(k2, k3, pk, cls, vertex_shader->constant_register_map().float_count, blend0_raw,
              normalized_depth_control.value);
    }
    if (g_draw_prof) g_draw_ns[26] += prof_ns_since(_dp_bc0);
  }
  return true;
}

// [GPU-INST] Append the current draw's used vertex float constants as one
// instance, packed in the same order as the float-constants cbuffer so the
// instanced shader reads instance i at cb[i * float_count + packed_index].
void D3D12CommandProcessor::InstancedBatchAppend(const RegisterFile& regs) {
  InstancedBatch& b = instanced_batch_;
  if (b.count >= b.max_instances) {
    return;
  }
  size_t base = b.data.size();
  b.data.resize(base + size_t(b.float_count) * 4);
  float* dst = b.data.data() + base;
  for (uint32_t word = 0; word < 4; ++word) {
    uint64_t entry = b.float_bitmap[word];
    uint32_t fc_index;
    while (rex::bit_scan_forward(entry, &fc_index)) {
      entry &= ~(uint64_t(1) << fc_index);
      std::memcpy(dst, &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + (word << 8) + (fc_index << 2)],
                  4 * sizeof(float));
      dst += 4;
    }
  }
  ++b.count;
}

// [GPU-INST] Begin a deferred instanced batch from the current (already set up)
// draw. The pipeline/bindings/index buffer/barriers are already recorded in the
// command list; only the draw call itself is held back until the batch closes.
void D3D12CommandProcessor::StartInstancedBatch(const RegisterFile& regs, Shader* vertex_shader,
                                                Shader* pixel_shader,
                                                xenos::PrimitiveType primitive_type,
                                                uint32_t index_count,
                                                const IndexBufferInfo* index_buffer_info,
                                                uint32_t host_draw_vertex_count, bool indexed) {
  InstancedBatch& b = instanced_batch_;
  b.active = true;
  b.indexed = indexed;
  b.vs = vertex_shader;
  b.ps = pixel_shader;
  b.prim = primitive_type;
  b.index_count = index_count;
  b.has_ib = index_buffer_info != nullptr;
  if (b.has_ib) {
    b.ib_base = index_buffer_info->guest_base;
    b.ib_count = index_buffer_info->count;
    b.ib_format = index_buffer_info->format;
    b.ib_endian = index_buffer_info->endianness;
  }
  b.host_draw_vertex_count = host_draw_vertex_count;
  const Shader::ConstantRegisterMap& constant_map = vertex_shader->constant_register_map();
  b.float_count = constant_map.float_count;
  std::memcpy(b.float_bitmap, constant_map.float_bitmap, sizeof(b.float_bitmap));
  b.max_instances =
      b.float_count
          ? std::max(1u, DxbcShaderTranslator::kInstancedFloatConstantsVec4Capacity / b.float_count)
          : 1u;
  b.count = 0;
  b.data.clear();
  InstancedBatchAppend(regs);
  ++g_instance_draws_in;
  g_instance_dirty = false;
  g_instance_dirty_first_reg = UINT32_MAX;
}

// [GPU-INST] Whether the current draw can extend the open batch: nothing but the
// vertex float constants changed since the batch's last draw (g_instance_dirty),
// the same shaders/primitive/index buffer, and capacity remains.
bool D3D12CommandProcessor::InstancedBatchCanMerge(xenos::PrimitiveType primitive_type,
                                                   uint32_t index_count,
                                                   const IndexBufferInfo* index_buffer_info,
                                                   Shader* vertex_shader,
                                                   Shader* pixel_shader) const {
  const InstancedBatch& b = instanced_batch_;
  if (!b.active || g_instance_dirty) {
    g_inst_fail_reason = 0;
    return false;
  }
  if (vertex_shader != b.vs || pixel_shader != b.ps) {
    g_inst_fail_reason = 1;
    return false;
  }
  if (primitive_type != b.prim || index_count != b.index_count) {
    g_inst_fail_reason = 2;
    return false;
  }
  bool has_ib = index_buffer_info != nullptr;
  if (has_ib != b.has_ib) {
    g_inst_fail_reason = 3;
    return false;
  }
  if (has_ib &&
      (index_buffer_info->guest_base != b.ib_base || index_buffer_info->count != b.ib_count ||
       index_buffer_info->format != b.ib_format || index_buffer_info->endianness != b.ib_endian)) {
    g_inst_fail_reason = 3;
    return false;
  }
  if (b.count >= b.max_instances) {
    g_inst_fail_reason = 4;
    return false;
  }
  return true;
}

// [GPU-INST] Emit the open batch (if any) as a single DrawIndexedInstanced:
// upload the per-instance vertex float constants and point the vertex float CBV
// at them, then draw count instances. Pipeline/bindings/IB/topology/barriers
// were recorded once at the batch's start.
void D3D12CommandProcessor::FlushInstancedBatch() {
  InstancedBatch& b = instanced_batch_;
  if (!b.active) {
    return;
  }
  b.active = false;
  if (b.count) {
    ++g_inst_hist[b.count == 1 ? 0
                  : b.count <= 4 ? 1
                  : b.count <= 16 ? 2
                  : b.count <= 64 ? 3
                  : b.count <= 256 ? 4
                                   : 5];
  }
  if (b.count == 0 || b.float_count == 0 || !submission_open_) {
    b.data.clear();
    return;
  }
  const uint32_t total_floats = b.count * b.float_count * 4;
  D3D12_GPU_VIRTUAL_ADDRESS instance_cbv_address;
  uint8_t* mapping = constant_buffer_pool_->Request(
      frame_current_, sizeof(float) * total_floats, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
      nullptr, nullptr, &instance_cbv_address);
  if (mapping == nullptr) {
    REXGPU_ERROR("[gpu-inst] Failed to allocate a {}-instance constant buffer; dropping the batch",
                 b.count);
    b.data.clear();
    return;
  }
  std::memcpy(mapping, b.data.data(), sizeof(float) * total_floats);
  uint32_t root_parameter_float_constants_vertex =
      bindless_resources_used_ ? kRootParameter_Bindless_FloatConstantsVertex
                               : kRootParameter_Bindful_FloatConstantsVertex;
  deferred_command_list_.D3DSetGraphicsRootConstantBufferView(root_parameter_float_constants_vertex,
                                                              instance_cbv_address);
  if (b.indexed) {
    deferred_command_list_.D3DDrawIndexedInstanced(b.host_draw_vertex_count, b.count, 0, 0, 0);
  } else {
    deferred_command_list_.D3DDrawInstanced(b.host_draw_vertex_count, b.count, 0, 0);
  }
  // The vertex float CBV now points at the instance buffer; force the next
  // normal draw to re-upload and re-bind its single-draw constants.
  cbuffer_binding_float_vertex_.up_to_date = false;
  ++g_instance_draws_out;
  b.data.clear();
}

// [DRAW-POOL] The first occurrence: reserve [args | instance block] from the
// upload pool, write instance 0, bind the block as the vertex float
// constants, record ONE ExecuteIndirect, register the batch. False = plain
// draw instead (the instanced VS variant reads instance 0 from the normal
// cbuffer identically, so the caller's fallback draw is still correct).
bool D3D12CommandProcessor::PoolOpenAndDraw(uint64_t key, const RegisterFile& regs,
                                            const D3D12Shader* vertex_shader,
                                            const D3D12Shader* pixel_shader,
                                            uint32_t host_draw_vertex_count, bool indexed,
                                            uint32_t capacity, bool ps_inst) {
  ID3D12CommandSignature* sig = DrawCommandSignature(indexed);
  if (!sig) {
    ++g_pool.ref_alloc;
    return false;
  }
  const Shader::ConstantRegisterMap& cm = vertex_shader->constant_register_map();
  const uint32_t float_count = cm.float_count;
  // Rung 1b-2: the pixel shader's block sizes the batch too.
  const uint32_t ps_float_count =
      ps_inst && pixel_shader ? pixel_shader->constant_register_map().float_count : 0;
  ps_inst = ps_float_count != 0;
  const uint32_t cap_max =
      std::max(2u, DxbcShaderTranslator::kInstancedFloatConstantsVec4Capacity /
                       std::max(float_count, ps_float_count));
  capacity = std::max(2u, std::min(capacity, cap_max));
  if (g_pool_batch_n >= g_pool_batches.size()) {
    if (g_pool_batches.size() >= 4096) {
      ++g_pool.ref_alloc;
      return false;
    }
    g_pool_batches.resize(g_pool_batches.size() * 2);
  }
  // [args (256) | VS instance block, 256-aligned | PS instance block].
  const size_t vs_bytes = rex::align<size_t>(size_t(capacity) * float_count * 16,
                                             D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
  const size_t ps_bytes = size_t(capacity) * ps_float_count * 16;
  const size_t inst_bytes = vs_bytes + ps_bytes;
  ID3D12Resource* buffer = nullptr;
  size_t offset = 0;
  D3D12_GPU_VIRTUAL_ADDRESS gpu = 0;
  uint8_t* mapping = constant_buffer_pool_->Request(
      frame_current_, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT + inst_bytes,
      D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, &buffer, &offset, &gpu);
  if (!mapping) {
    ++g_pool.ref_alloc;
    return false;
  }
  PoolSlot* slot = PoolLookup(key);
  if (!slot) {
    ++g_pool.ref_ovf;
    return false;
  }
  PoolBatch& b = g_pool_batches[g_pool_batch_n];
  if (!PoolGather(regs, pixel_shader, b.snap)) {
    ++g_pool.ref_psbig;
    return false;
  }
  uint32_t* args = reinterpret_cast<uint32_t*>(mapping);
  args[0] = host_draw_vertex_count;
  args[1] = 1;
  args[2] = 0;
  args[3] = 0;
  if (indexed) args[4] = 0;
  b.key = key;
  b.gk = g_gk_cur;  // [GKEY]
  std::memcpy(b.gkc, g_gk_cur_c, sizeof(b.gkc));
  b.epoch = g_pool_epoch;
  b.count = 1;
  b.capacity = capacity;
  b.float_count = float_count;
  b.hiz_base = UINT32_MAX;  // [hiz-pool] decided below
  b.hiz_serial = 0;
  b.hiz_dir = 0;
  b.ei_offset = SIZE_MAX;
  g_pool_dcl = &deferred_command_list_;
  b.idx_count = host_draw_vertex_count;
  std::memcpy(b.float_bitmap, cm.float_bitmap, sizeof(b.float_bitmap));
  b.inst = mapping + D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
  b.count_ptr = args + 1;
  PoolPackInstance(regs, b.float_bitmap, b.inst);
  b.ps_inst = ps_inst;
  b.ps_float_count = ps_float_count;
  b.ps_block = nullptr;
  if (ps_inst) {
    std::memcpy(b.ps_bitmap, pixel_shader->constant_register_map().float_bitmap,
                sizeof(b.ps_bitmap));
    b.ps_block = b.inst + vs_bytes;
    PoolPackInstance(regs, b.ps_bitmap, b.ps_block, XE_GPU_REG_SHADER_CONSTANT_256_X);
  }
  slot->batch = g_pool_batch_n++;
  const uint32_t root_parameter_float_constants_vertex =
      bindless_resources_used_ ? kRootParameter_Bindless_FloatConstantsVertex
                               : kRootParameter_Bindful_FloatConstantsVertex;
  deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
      root_parameter_float_constants_vertex, gpu + D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
  if (ps_inst) {
    const uint32_t root_parameter_float_constants_pixel =
        bindless_resources_used_ ? kRootParameter_Bindless_FloatConstantsPixel
                                 : kRootParameter_Bindful_FloatConstantsPixel;
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        root_parameter_float_constants_pixel,
        gpu + D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT + vs_bytes);
    cbuffer_binding_float_pixel_.up_to_date = false;
    g_pool.inst_bytes += size_t(ps_float_count) * 16;
    ++g_pool.opened_ps;
  }
  // [hiz-pool] The per-instance form when the opener is Hi-Z eligible: a
  // capacity-long run of argument slots (one element per instance, executed
  // up to the CPU-written count at args[0]), the opener as entry 0 of the
  // window open now; hits append their own entries while that window lives.
  bool per_instance = false;
  if (hiz_available_ && hiz_phase_ != 0 && (hiz_draw_.valid || hiz_draw_.dir_hint) &&
      hiz_frame_slots_ + capacity <= kHizMaxPerFrame) {
    ID3D12CommandSignature* hsig = HizCommandSignature(indexed);
    const uint8_t open_dir = hiz_draw_.valid ? hiz_draw_.dir : hiz_draw_.dir_hint;
    if (hsig && HizWindowEnsure(open_dir) && hiz_window_.run_count < kHizMaxRuns) {
      const HizDrawPending d = hiz_draw_;
      hiz_draw_.valid = false;
      // The run must not wrap the ring (the elements are read contiguously).
      if ((hiz_slot_next_ & (kHizRing - 1)) + capacity > kHizRing) {
        for (uint32_t i = hiz_slot_next_ & (kHizRing - 1); i < kHizRing; ++i) {
          hiz_slot_idx_[i] = 0;  // skipped slots: stale verdict words, idx 0
        }
        hiz_slot_next_ += kHizRing - (hiz_slot_next_ & (kHizRing - 1));
      }
      const uint32_t base = hiz_slot_next_ & (kHizRing - 1);
      hiz_slot_next_ += capacity;
      hiz_frame_slots_ += capacity;
      for (uint32_t i = 0; i < capacity; ++i) hiz_slot_idx_[base + i] = 0;
      HizAppend(base, d.valid ? &d : nullptr, 0, host_draw_vertex_count);
      b.hiz_base = base;
      b.hiz_serial = hiz_window_.serial;
      b.hiz_dir = open_dir;
      // The run entry: the CPU keeps its count (every hit raises it, as the
      // old InstanceCount); the test CS zeroes the tail [count, capacity).
      uint32_t* run = reinterpret_cast<uint32_t*>(hiz_window_.runs +
                                                  size_t(hiz_window_.run_count) * kHizRunBytes);
      run[0] = base;
      run[1] = capacity;
      run[2] = 1;
      run[3] = 0;
      b.count_ptr = run + 2;
      ++hiz_window_.run_count;
      hiz_window_.header[1] = hiz_window_.run_count;  // read by the test CS at GPU time
      ++hiz_acc_.pool_open;
      OccDrawBegin();  // [occ] the batch = one query; its count lands at close
      occ_draw_hiz_ = base + 1;
      // Recorded with the capacity; PoolClose patches the exact count.
      b.ei_offset = deferred_command_list_.next_command_offset();
      b.ei_gen = deferred_command_list_.reset_generation();
      b.ei_sig = hsig;
      b.ei_buf = hiz_args_.Get();
      b.ei_arg_offset = uint64_t(base) * kHizArgsStride;
      deferred_command_list_.D3DExecuteIndirect(hsig, capacity, hiz_args_.Get(), b.ei_arg_offset);
      b.occ_tag = OccDrawEnd(host_draw_vertex_count, 1);
      occ_draw_hiz_ = 0;
      inst_base_dirty_ = true;
      per_instance = true;
    }
  }
  if (!per_instance) {
    OccDrawBegin();  // [occ] the batch = one query; its count lands at close
    deferred_command_list_.D3DExecuteIndirect(sig, 1, buffer, offset);
    b.occ_tag = OccDrawEnd(host_draw_vertex_count, 1);
  }
  // The vertex float CBV now points at the instance block; the next plain
  // draw must re-upload and re-bind its own constants.
  cbuffer_binding_float_vertex_.up_to_date = false;
  g_pool.inst_bytes += size_t(float_count) * 16;
  ++g_pool.opened;
  ++g_pool.first_in_batch;
  return true;
}

bool D3D12CommandProcessor::IssueDraw_MemexportReadbackFullPath(uint32_t total_size) {
  if (!total_size || memexport_ranges_.empty()) {
    return true;
  }

  ID3D12Resource* readback_buffer = RequestReadbackBuffer(total_size);
  if (!readback_buffer) {
    return true;
  }

  shared_memory_->UseAsCopySource();
  SubmitBarriers();
  ID3D12Resource* shared_memory_buffer = shared_memory_->GetBuffer();
  uint32_t readback_buffer_offset = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    deferred_command_list_.D3DCopyBufferRegion(
        readback_buffer, readback_buffer_offset, shared_memory_buffer,
        memexport_range.base_address_dwords << 2, memexport_range.size_bytes);
    readback_buffer_offset += memexport_range.size_bytes;
  }

  if (!AwaitAllQueueOperationsCompletion()) {
    return true;
  }

  D3D12_RANGE readback_range = {};
  readback_range.Begin = 0;
  readback_range.End = total_size;
  void* readback_mapping = nullptr;
  if (FAILED(readback_buffer->Map(0, &readback_range, &readback_mapping))) {
    return true;
  }

  const uint8_t* readback_bytes = reinterpret_cast<const uint8_t*>(readback_mapping);
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    std::memcpy(memory_->TranslatePhysical(memexport_range.base_address_dwords << 2),
                readback_bytes, memexport_range.size_bytes);
    readback_bytes += memexport_range.size_bytes;
  }

  D3D12_RANGE readback_write_range = {};
  readback_buffer->Unmap(0, &readback_write_range);
  return true;
}

bool D3D12CommandProcessor::IssueDraw_MemexportReadbackFastPath(uint32_t total_size) {
  if (!total_size || memexport_ranges_.empty()) {
    return true;
  }

  const uint64_t readback_key =
      MakeMemexportReadbackKey(memexport_ranges_.front().base_address_dwords, total_size);
  ReadbackBuffer& readback = memexport_readback_buffers_[readback_key];
  readback.last_used_frame = frame_current_;

  auto ensure_readback_slot = [&](uint32_t index, uint32_t size) -> bool {
    if (readback.buffers[index] && readback.mapped_data[index] && size <= readback.sizes[index]) {
      return true;
    }

    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, size, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* buffer = nullptr;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) {
      return false;
    }

    D3D12_RANGE read_range = {0, size};
    void* mapped_data = nullptr;
    if (FAILED(buffer->Map(0, &read_range, &mapped_data))) {
      buffer->Release();
      return false;
    }

    if (readback.buffers[index]) {
      if (!AwaitAllQueueOperationsCompletion()) {
        buffer->Unmap(0, nullptr);
        buffer->Release();
        return false;
      }
      if (readback.mapped_data[index]) {
        readback.buffers[index]->Unmap(0, nullptr);
      }
      readback.buffers[index]->Release();
    }

    readback.buffers[index] = buffer;
    readback.mapped_data[index] = mapped_data;
    readback.sizes[index] = size;
    readback.submission_written[index] = 0;
    readback.written_size[index] = 0;
    return true;
  };

  const uint32_t write_index = readback.current_index;
  const uint32_t read_index = 1 - write_index;
  const uint32_t readback_size = AlignReadbackBufferSize(total_size);
  if (!ensure_readback_slot(write_index, readback_size)) {
    return IssueDraw_MemexportReadbackFullPath(total_size);
  }

  shared_memory_->UseAsCopySource();
  SubmitBarriers();
  ID3D12Resource* shared_memory_buffer = shared_memory_->GetBuffer();
  uint32_t readback_offset = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    deferred_command_list_.D3DCopyBufferRegion(
        readback.buffers[write_index], readback_offset, shared_memory_buffer,
        memexport_range.base_address_dwords << 2, memexport_range.size_bytes);
    readback_offset += memexport_range.size_bytes;
  }
  readback.submission_written[write_index] = submission_current_;
  readback.written_size[write_index] = total_size;

  CheckSubmissionFence(0);
  bool previous_slot_ready = readback.buffers[read_index] && readback.mapped_data[read_index] &&
                             total_size <= readback.sizes[read_index] &&
                             total_size <= readback.written_size[read_index] &&
                             readback.submission_written[read_index] &&
                             readback.submission_written[read_index] <= submission_completed_;
  if (!previous_slot_ready) {
    IssueDraw_MemexportReadbackFullPath(total_size);
    readback.current_index = read_index;
    return true;
  }

  const uint8_t* readback_bytes = static_cast<const uint8_t*>(readback.mapped_data[read_index]);
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    std::memcpy(memory_->TranslatePhysical(memexport_range.base_address_dwords << 2),
                readback_bytes, memexport_range.size_bytes);
    readback_bytes += memexport_range.size_bytes;
  }
  readback.current_index = read_index;
  return true;
}

void D3D12CommandProcessor::InitializeTrace() {
  CommandProcessor::InitializeTrace();

  if (!BeginSubmission(false)) {
    return;
  }
  bool render_target_cache_submitted = render_target_cache_->InitializeTraceSubmitDownloads();
  bool shared_memory_submitted = shared_memory_->InitializeTraceSubmitDownloads();
  if (!render_target_cache_submitted && !shared_memory_submitted) {
    return;
  }
  AwaitAllQueueOperationsCompletion();
  if (render_target_cache_submitted) {
    render_target_cache_->InitializeTraceCompleteDownloads();
  }
  if (shared_memory_submitted) {
    shared_memory_->InitializeTraceCompleteDownloads();
  }
}

bool D3D12CommandProcessor::IssueCopy() {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES
  // [GPU-PRECORD] Phase 1b-0: a resolve/copy must observe all prior draws, so
  // record the pending captured segment before it (no-op when not capturing, and
  // already done when reached via the IssueDraw copy-mode path).
  PrecordFlush();
  // [GPU-INST] A resolve/copy must observe all prior draws, so flush the open
  // instanced batch first.
  if (instanced_batch_.active) {
    ++g_inst_flush_site[1];
  }
  FlushInstancedBatch();
  if (!BeginSubmission(true)) {
    return false;
  }
  // [gpu-census] the whole resolve (direct-resolve CS + dest copies) prices
  // as resolve work; resolve clears re-class themselves to xfer inside.
  GpuCensusScope gpu_census_scope(*this, kGpuCensusResolve);
  ReadbackResolveMode readback_mode = GetReadbackResolveMode(REXCVAR_GET(d3d12_readback_resolve));
  if (readback_mode == ReadbackResolveMode::kDisabled) {
    uint32_t written_address, written_length;
    return render_target_cache_->Resolve(*memory_, *shared_memory_, *texture_cache_,
                                         written_address, written_length);
  }
  return IssueCopy_ReadbackResolvePath();
}

bool D3D12CommandProcessor::IssueCopy_ReadbackResolvePath() {
  uint32_t written_address, written_length;
  if (!render_target_cache_->Resolve(*memory_, *shared_memory_, *texture_cache_, written_address,
                                     written_length)) {
    return false;
  }

  if (!written_length) {
    return true;
  }

  if (!memory_->TranslatePhysical(written_address)) {
    return true;
  }

  bool is_scaled = texture_cache_->IsDrawResolutionScaled();
  uint64_t resolve_key = MakeReadbackResolveKey(written_address, written_length);
  ReadbackBuffer& rb = readback_buffers_[resolve_key];
  rb.last_used_frame = frame_current_;

  uint32_t write_index = rb.current_index;
  uint32_t size = AlignReadbackBufferSize(written_length);

  if (size > rb.sizes[write_index]) {
    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, size, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* buffer = nullptr;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) {
      REXGPU_ERROR("Failed to create a {} MB readback buffer", size >> 20);
      return true;
    }
    if (rb.buffers[write_index]) {
      if (rb.mapped_data[write_index]) {
        rb.buffers[write_index]->Unmap(0, nullptr);
        rb.mapped_data[write_index] = nullptr;
      }
      rb.buffers[write_index]->Release();
    }
    rb.buffers[write_index] = buffer;
    rb.sizes[write_index] = size;
    D3D12_RANGE read_range = {0, size};
    if (FAILED(buffer->Map(0, &read_range, &rb.mapped_data[write_index]))) {
      REXGPU_ERROR("Failed to persistently map resolve readback buffer");
      rb.mapped_data[write_index] = nullptr;
    }
  }

  if (!rb.buffers[write_index]) {
    return true;
  }

  if (is_scaled) {
    if (!resolve_downscale_pipeline_ || !resolve_downscale_root_signature_) {
      return true;
    }

    // [NR-ISSUE] Increment 4f: through the active draw file, not the shared
    // one -- a shadow-issued resolve must read ITS state here too (the rest
    // of the resolve path already does: RenderTargetCache::Resolve reads the
    // repointed member).
    reg::RB_COPY_DEST_INFO copy_dest_info =
        GetActiveDrawRegisterFile().Get<reg::RB_COPY_DEST_INFO>();
    const FormatInfo* format_info = FormatInfo::Get(uint32_t(copy_dest_info.copy_dest_format));
    uint32_t bits_per_pixel = format_info->bits_per_pixel;
    if (bits_per_pixel != 8 && bits_per_pixel != 16 && bits_per_pixel != 32 &&
        bits_per_pixel != 64) {
      return true;
    }

    uint32_t pixel_size_log2;
    if (!rex::bit_scan_forward(bits_per_pixel >> 3, &pixel_size_log2)) {
      return true;
    }
    uint32_t tile_size_1x = 32 * 32 * (uint32_t(1) << pixel_size_log2);
    uint32_t tile_count = written_length / tile_size_1x;
    if (!tile_count) {
      return true;
    }

    uint32_t scaled_length = uint32_t(texture_cache_->GetCurrentScaledResolveRangeLengthScaled());
    uint64_t scaled_address = texture_cache_->GetCurrentScaledResolveRangeStartScaled();
    if (!scaled_length) {
      return true;
    }

    uint32_t downscale_buffer_size = AlignReadbackBufferSize(written_length);
    if (downscale_buffer_size > resolve_downscale_buffer_size_) {
      const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
      ID3D12Device* device = provider.GetDevice();
      D3D12_RESOURCE_DESC buffer_desc;
      ui::d3d12::util::FillBufferResourceDesc(buffer_desc, downscale_buffer_size,
                                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      ID3D12Resource* buffer = nullptr;
      if (FAILED(device->CreateCommittedResource(
              &ui::d3d12::util::kHeapPropertiesDefault, provider.GetHeapFlagCreateNotZeroed(),
              &buffer_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
              IID_PPV_ARGS(&buffer)))) {
        REXGPU_ERROR("Failed to create a {} MB resolve downscale buffer",
                     downscale_buffer_size >> 20);
        return true;
      }
      if (resolve_downscale_buffer_) {
        resources_for_deletion_.emplace_back(GetCurrentSubmission(),
                                             resolve_downscale_buffer_.Detach());
      }
      resolve_downscale_buffer_.Attach(buffer);
      resolve_downscale_buffer_size_ = downscale_buffer_size;
    }

    if (!resolve_downscale_buffer_) {
      return true;
    }

    ID3D12Resource* scaled_resolve_buffer = texture_cache_->GetCurrentScaledResolveBufferResource();
    size_t scaled_resolve_buffer_index = texture_cache_->GetCurrentScaledResolveBufferIndexPublic();
    if (!scaled_resolve_buffer) {
      return true;
    }
    uint64_t scaled_buffer_base = uint64_t(scaled_resolve_buffer_index) << 30;
    if (scaled_address < scaled_buffer_base) {
      return true;
    }
    uint64_t source_offset = scaled_address - scaled_buffer_base;

    ui::d3d12::util::DescriptorCpuGpuHandlePair downscale_descriptors[2];
    if (!RequestOneUseSingleViewDescriptors(2, downscale_descriptors)) {
      return true;
    }

    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    uint32_t aligned_scaled_length =
        rex::align(scaled_length, uint32_t(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT));
    ui::d3d12::util::CreateBufferRawSRV(device, downscale_descriptors[0].first,
                                        scaled_resolve_buffer, aligned_scaled_length,
                                        source_offset);
    uint32_t aligned_written_length =
        rex::align(written_length, uint32_t(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT));
    ui::d3d12::util::CreateBufferRawUAV(device, downscale_descriptors[1].first,
                                        resolve_downscale_buffer_.Get(), aligned_written_length, 0);

    PushUAVBarrier(scaled_resolve_buffer);
    texture_cache_->TransitionCurrentScaledResolveRange(
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    SubmitBarriers();

    SetExternalPipeline(resolve_downscale_pipeline_.Get());
    deferred_command_list_.D3DSetComputeRootSignature(resolve_downscale_root_signature_.Get());
    ResolveDownscaleConstants constants;
    constants.scale_x = texture_cache_->draw_resolution_scale_x();
    constants.scale_y = texture_cache_->draw_resolution_scale_y();
    constants.pixel_size_log2 = pixel_size_log2;
    constants.tile_count = tile_count;
    constants.half_pixel_offset = (REXCVAR_GET(readback_resolve_half_pixel_offset) &&
                                   (constants.scale_x > 1 || constants.scale_y > 1))
                                      ? 1u
                                      : 0u;
    deferred_command_list_.D3DSetComputeRoot32BitConstants(
        UINT(ResolveDownscaleRootParameter::kConstants), sizeof(constants) / sizeof(uint32_t),
        &constants, 0);
    deferred_command_list_.D3DSetComputeRootDescriptorTable(
        UINT(ResolveDownscaleRootParameter::kSource), downscale_descriptors[0].second);
    deferred_command_list_.D3DSetComputeRootDescriptorTable(
        UINT(ResolveDownscaleRootParameter::kDestination), downscale_descriptors[1].second);
    deferred_command_list_.D3DDispatch(tile_count, 1, 1);

    PushUAVBarrier(resolve_downscale_buffer_.Get());
    PushTransitionBarrier(resolve_downscale_buffer_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_COPY_SOURCE);
    SubmitBarriers();
    deferred_command_list_.D3DCopyBufferRegion(rb.buffers[write_index], 0,
                                               resolve_downscale_buffer_.Get(), 0, written_length);
    PushTransitionBarrier(resolve_downscale_buffer_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    texture_cache_->TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    SubmitBarriers();
  } else {
    shared_memory_->UseAsCopySource();
    SubmitBarriers();
    ID3D12Resource* shared_memory_buffer = shared_memory_->GetBuffer();
    deferred_command_list_.D3DCopyBufferRegion(rb.buffers[write_index], 0, shared_memory_buffer,
                                               written_address, written_length);
  }

  ReadbackResolveMode readback_mode = GetReadbackResolveMode(REXCVAR_GET(d3d12_readback_resolve));
  bool use_delayed_sync =
      readback_mode == ReadbackResolveMode::kFast || readback_mode == ReadbackResolveMode::kSome;
  uint32_t read_index = write_index;
  if (use_delayed_sync) {
    read_index = 1 - write_index;
  } else if (!AwaitAllQueueOperationsCompletion()) {
    return true;
  }

  bool is_cache_miss = false;
  if (use_delayed_sync && (!rb.buffers[read_index] || written_length > rb.sizes[read_index] ||
                           !rb.mapped_data[read_index])) {
    is_cache_miss = true;
    read_index = write_index;
    if (!AwaitAllQueueOperationsCompletion()) {
      return true;
    }
  }

  bool should_copy = (readback_mode == ReadbackResolveMode::kSome) ? is_cache_miss : true;
  if (should_copy && rb.buffers[read_index] && written_length <= rb.sizes[read_index] &&
      rb.mapped_data[read_index]) {
    uint8_t* destination = memory_->TranslatePhysical(written_address);
    if (destination) {
      std::memcpy(destination, static_cast<uint8_t*>(rb.mapped_data[read_index]), written_length);
    }
  }

  rb.current_index = 1 - rb.current_index;
  return true;
}

void D3D12CommandProcessor::CheckSubmissionFence(uint64_t await_submission) {
  if (await_submission >= submission_current_) {
    if (submission_open_) {
      EndSubmission(false);
    }
    // Ending an open submission should result in queue operations done directly
    // (like UpdateTileMappings) to be tracked within the scope of that
    // submission, but just in case of a failure, or queue operations being done
    // outside of a submission, await explicitly.
    if (queue_operations_done_since_submission_signal_) {
      UINT64 fence_value = ++queue_operations_since_submission_fence_last_;
      ID3D12CommandQueue* direct_queue = GetD3D12Provider().GetDirectQueue();
      if (SUCCEEDED(direct_queue->Signal(queue_operations_since_submission_fence_, fence_value) &&
                    SUCCEEDED(queue_operations_since_submission_fence_->SetEventOnCompletion(
                        fence_value, fence_completion_event_)))) {
        PROFILE_CMD_BUFFER_STALL();
        // [hitch] out-of-submission queue-operation fence wait on the CP.
        uint64_t hitch_wait_start = rex::chrono::Clock::QueryHostTickCount();
        WaitForSingleObject(fence_completion_event_, INFINITE);
        g_hitch_probe.qops_wait_ticks.fetch_add(
            rex::chrono::Clock::QueryHostTickCount() - hitch_wait_start,
            std::memory_order_relaxed);
        g_hitch_probe.qops_waits.fetch_add(1, std::memory_order_relaxed);
        queue_operations_done_since_submission_signal_ = false;
      } else {
        REXGPU_ERROR(
            "Failed to await an out-of-submission queue operation completion "
            "Direct3D 12 fence");
      }
    }
    // A submission won't be ended if it hasn't been started, or if ending
    // has failed - clamp the index.
    await_submission = submission_current_ - 1;
  }

  uint64_t submission_completed_before = submission_completed_;
  submission_completed_ = submission_fence_->GetCompletedValue();
  if (submission_completed_ < await_submission) {
    if (SUCCEEDED(
            submission_fence_->SetEventOnCompletion(await_submission, fence_completion_event_))) {
      PROFILE_CMD_BUFFER_STALL();
      // [hitch] submission-fence wait on the CP (frame pacing + explicit
      // awaits) - big in GPU-drowned pockets, small when CPU-bound.
      uint64_t hitch_wait_start = rex::chrono::Clock::QueryHostTickCount();
      WaitForSingleObject(fence_completion_event_, INFINITE);
      g_hitch_probe.fence_wait_ticks.fetch_add(
          rex::chrono::Clock::QueryHostTickCount() - hitch_wait_start, std::memory_order_relaxed);
      g_hitch_probe.fence_waits.fetch_add(1, std::memory_order_relaxed);
      submission_completed_ = submission_fence_->GetCompletedValue();
    }
  }
  if (submission_completed_ < await_submission) {
    REXGPU_ERROR("Failed to await a submission completion Direct3D 12 fence");
  }
  if (submission_completed_ <= submission_completed_before) {
    // Not updated - no need to reclaim or download things.
    return;
  }

  // Reclaim command allocators.
  while (command_allocator_submitted_first_) {
    if (command_allocator_submitted_first_->last_usage_submission > submission_completed_) {
      break;
    }
    if (command_allocator_writable_last_) {
      command_allocator_writable_last_->next = command_allocator_submitted_first_;
    } else {
      command_allocator_writable_first_ = command_allocator_submitted_first_;
    }
    command_allocator_writable_last_ = command_allocator_submitted_first_;
    command_allocator_submitted_first_ = command_allocator_submitted_first_->next;
    command_allocator_writable_last_->next = nullptr;
  }
  if (!command_allocator_submitted_first_) {
    command_allocator_submitted_last_ = nullptr;
  }

  // Release single-use bindless descriptors.
  while (!view_bindless_one_use_descriptors_.empty()) {
    if (view_bindless_one_use_descriptors_.front().second > submission_completed_) {
      break;
    }
    ReleaseViewBindlessDescriptorImmediately(view_bindless_one_use_descriptors_.front().first);
    view_bindless_one_use_descriptors_.pop_front();
  }

  // Delete transient resources marked for deletion.
  while (!resources_for_deletion_.empty()) {
    if (resources_for_deletion_.front().first > submission_completed_) {
      break;
    }
    resources_for_deletion_.front().second->Release();
    resources_for_deletion_.pop_front();
  }

  shared_memory_->CompletedSubmissionUpdated();

  render_target_cache_->CompletedSubmissionUpdated();

  primitive_processor_->CompletedSubmissionUpdated();

  texture_cache_->CompletedSubmissionUpdated(submission_completed_);
}

void D3D12CommandProcessor::LogDeviceRemovalDiagnostics(ID3D12Device* device, HRESULT reason) {
  const char* reason_str = "Unknown";
  switch (reason) {
    case DXGI_ERROR_DEVICE_HUNG:
      reason_str = "DEVICE_HUNG (TDR - GPU command took too long)";
      break;
    case DXGI_ERROR_DEVICE_REMOVED:
      reason_str = "DEVICE_REMOVED (driver internal error or hot-unplug)";
      break;
    case DXGI_ERROR_DEVICE_RESET:
      reason_str = "DEVICE_RESET (bad GPU command)";
      break;
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
      reason_str = "DRIVER_INTERNAL_ERROR";
      break;
    case DXGI_ERROR_INVALID_CALL:
      reason_str = "INVALID_CALL";
      break;
  }
  REXGPU_ERROR("D3D12 device removed: HRESULT 0x{:08X} - {}", static_cast<unsigned>(reason),
               reason_str);

  Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedData> dred;
  if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dred)))) {
    return;
  }

  D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT breadcrumbs = {};
  if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&breadcrumbs))) {
    for (const D3D12_AUTO_BREADCRUMB_NODE* node = breadcrumbs.pHeadAutoBreadcrumbNode; node;
         node = node->pNext) {
      if (!node->pLastBreadcrumbValue || !node->pCommandHistory ||
          *node->pLastBreadcrumbValue == 0) {
        continue;
      }
      REXGPU_ERROR("DRED breadcrumb: completed {} of {} ops", *node->pLastBreadcrumbValue,
                   node->BreadcrumbCount);
      uint32_t last = std::min(*node->pLastBreadcrumbValue, node->BreadcrumbCount);
      uint32_t start = last > 3 ? last - 3 : 0;
      uint32_t end = std::min(last + 1, node->BreadcrumbCount);
      for (uint32_t i = start; i < end; i++) {
        REXGPU_ERROR("  [{}] op type {}{}", i, static_cast<int>(node->pCommandHistory[i]),
                     i == last ? " <-- FAULT" : "");
      }
    }
  }

  D3D12_DRED_PAGE_FAULT_OUTPUT page_fault = {};
  if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&page_fault)) && page_fault.PageFaultVA != 0) {
    REXGPU_ERROR("DRED page fault at VA 0x{:016X}", page_fault.PageFaultVA);
  }
}

void D3D12CommandProcessor::ForceFullDrawStateReemit() {
  // [GPU-PRECORD] Phase 1a. Reset the SAME cached/dirty state BeginSubmission resets
  // for a fresh command list, so the next draw re-emits everything into the current
  // deferred list. Body mirrors BeginSubmission's two reset blocks verbatim; the
  // binding-state half (normally only run on frame-open) is included unconditionally
  // because a segment's fresh list has nothing bound. Emits nothing itself except the
  // bindless SetDescriptorHeaps (matching BeginSubmission).
  ff_viewport_update_needed_ = true;
  ff_scissor_update_needed_ = true;
  ff_blend_factor_update_needed_ = true;
  ff_stencil_ref_update_needed_ = true;
  viewport_cache_valid_ = false;
  current_guest_pipeline_ = nullptr;
  current_external_pipeline_ = nullptr;
  current_graphics_root_signature_ = nullptr;
  current_graphics_root_up_to_date_ = 0;
  if (bindless_resources_used_) {
    deferred_command_list_.SetDescriptorHeaps(view_bindless_heap_, sampler_bindless_heap_current_);
  } else {
    view_bindful_heap_current_ = nullptr;
    sampler_bindful_heap_current_ = nullptr;
  }
  primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;

  std::memset(current_float_constant_map_vertex_, 0, sizeof(current_float_constant_map_vertex_));
  std::memset(current_float_constant_map_pixel_, 0, sizeof(current_float_constant_map_pixel_));
  cbuffer_binding_system_.up_to_date = false;
  cbuffer_binding_float_vertex_.up_to_date = false;
  cbuffer_binding_float_pixel_.up_to_date = false;
  cbuffer_binding_bool_loop_.up_to_date = false;
  cbuffer_binding_fetch_.up_to_date = false;
  current_shared_memory_binding_is_uav_.reset();
  if (bindless_resources_used_) {
    cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
    cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
  } else {
    draw_view_bindful_heap_index_ = ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
    draw_sampler_bindful_heap_index_ = ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
    bindful_textures_written_vertex_ = false;
    bindful_textures_written_pixel_ = false;
    bindful_samplers_written_vertex_ = false;
    bindful_samplers_written_pixel_ = false;
  }
}

void D3D12CommandProcessor::PrecordOpenSegment() {
  // [GPU-PRECORD] Phase 1b-1c Inc 6 (H3 fix, Option A). Snapshot the shared register file
  // into the current capture slot (it holds every write that set up the draws so far) and
  // mark the segment open so subsequent register writes are logged + deferred. The caller
  // guarantees the capture slot is free (lazy: first draw after a drain; eager: the
  // just-reset slot at an overlap boundary). Reads register_file_ (shared, parse-owned)
  // only -- safe while the worker replays against its private local file.
  precord_slots_[precord_capture_slot_].snapshot.assign(
      register_file_->values, register_file_->values + RegisterFile::kRegisterCount);
  precord_segment_open_ = true;
  precord_draws_in_segment_ = 0;
}

void D3D12CommandProcessor::PrecordResetSlot(uint32_t slot) {
  // [GPU-PRECORD] Phase 1b-1c Inc 3: drop one slot's captured log. The snapshot is
  // overwritten (assign) at the next segment open and local_regfile is rebuilt from the
  // snapshot at replay, so both allocations are kept for reuse. The capture-side flags
  // (precord_segment_open_ / precord_draws_in_segment_) are managed by PrecordFlush's
  // slot swap, not here -- a slot has no standalone "open" state.
  PrecordSegment& seg = precord_slots_[slot];
  seg.events.clear();
  seg.draws.clear();
  seg.frommem_data.clear();
}

uint32_t D3D12CommandProcessor::PrecordH3HashIndexBuffer(const IndexBufferInfo& ibi,
                                                         uint32_t index_count,
                                                         uint32_t* out_len) {
  // [GPU-PRECORD H3-PROBE] Inc 6. Same computation at capture and replay, so any hash
  // difference isolates a change to the guest index bytes (a recycle/overwrite), not to
  // the inputs. Bounded + range-checked so it can never stall or fault the caller.
  if (out_len) {
    *out_len = 0;
  }
  const uint32_t index_size = (ibi.format == xenos::IndexFormat::kInt32) ? 4u : 2u;
  uint64_t bytes64 = uint64_t(index_count) * index_size;
  // A recycled scratch buffer changes broadly, so a bounded prefix hash still catches
  // the overwrite while keeping the parse-thread cost small for huge index buffers.
  uint32_t bytes = uint32_t(bytes64 > 65536u ? 65536u : bytes64);
  if (!bytes || ibi.guest_base == 0 || ibi.guest_base >= SharedMemory::kBufferSize ||
      SharedMemory::kBufferSize - ibi.guest_base < bytes) {
    return 0;
  }
  const uint8_t* p = memory_->TranslatePhysical<const uint8_t*>(ibi.guest_base);
  if (!p) {
    return 0;
  }
  uint32_t h = 2166136261u;  // FNV-1a over the (bounded) index bytes.
  for (uint32_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 16777619u;
  }
  if (out_len) {
    *out_len = bytes;
  }
  return h ^ bytes;
}

uint32_t D3D12CommandProcessor::PrecordH3HashCapturedVbRanges(const uint32_t* addr,
                                                              const uint32_t* size,
                                                              uint32_t count,
                                                              uint32_t* out_len) {
  // [GPU-PRECORD H3-PROBE] Inc 6. FNV-1a over the PINNED (addr,size) VB ranges in order,
  // folding in each range's position so a buffer moving between slots still differs.
  // Capture and replay both funnel through here over the same pinned ranges, so the
  // hash matches iff the guest bytes are unchanged (no register state involved).
  // Range-checked; a range that has since become unmapped contributes nothing, which
  // the len comparison catches as a change.
  if (out_len) {
    *out_len = 0;
  }
  uint32_t h = 2166136261u;
  uint32_t total = 0;
  for (uint32_t r = 0; r < count; ++r) {
    uint32_t a = addr[r];
    uint32_t s = size[r];
    if (!s || a == 0 || a >= SharedMemory::kBufferSize || SharedMemory::kBufferSize - a < s) {
      continue;
    }
    const uint8_t* p = memory_->TranslatePhysical<const uint8_t*>(a);
    if (!p) {
      continue;
    }
    h ^= r;  // fold the range position
    h *= 16777619u;
    for (uint32_t k = 0; k < s; ++k) {
      h ^= p[k];
      h *= 16777619u;
    }
    total += s;
  }
  if (out_len) {
    *out_len = total;
  }
  return h ^ total;
}

uint32_t D3D12CommandProcessor::PrecordH3CaptureVertexBuffers(Shader* vertex_shader,
                                                              uint32_t max_ranges,
                                                              uint32_t* out_addr,
                                                              uint32_t* out_size,
                                                              uint32_t* out_count,
                                                              uint32_t* out_len) {
  // [GPU-PRECORD H3-PROBE] Inc 6. Collect this draw's bound VB byte-ranges from the
  // analyzed shader (the register file is correct at parse/capture time), PIN them into
  // the out arrays, then hash their current bytes via PrecordH3HashCapturedVbRanges so
  // replay uses the identical computation over the identical ranges. The bound-slot
  // iteration mirrors the residency loop in IssueDrawImpl; bounded + range-checked so it
  // can never stall or fault. Unanalyzed shader ⇒ empty bitmap ⇒ count 0 ⇒ len 0 ⇒
  // replay skips the check.
  if (out_count) {
    *out_count = 0;
  }
  if (out_len) {
    *out_len = 0;
  }
  if (!vertex_shader) {
    return 0;
  }
  const RegisterFile& regs = GetActiveDrawRegisterFile();
  const Shader::ConstantRegisterMap& constant_map_vertex = vertex_shader->constant_register_map();
  uint32_t count = 0;
  for (uint32_t i = 0;
       i < rex::countof(constant_map_vertex.vertex_fetch_bitmap) && count < max_ranges; ++i) {
    uint32_t vfetch_bits_remaining = constant_map_vertex.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(vfetch_bits_remaining, &j)) {
      vfetch_bits_remaining &= ~(uint32_t(1) << j);
      if (count >= max_ranges) {
        break;
      }
      uint32_t vfetch_index = i * 32 + j;
      xenos::xe_gpu_vertex_fetch_t vfetch_constant = regs.GetVertexFetch(vfetch_index);
      if (vfetch_constant.type != xenos::FetchConstantType::kVertex) {
        continue;
      }
      uint32_t addr = vfetch_constant.address << 2;
      uint32_t size = vfetch_constant.size << 2;
      // Bounded prefix per slot: a recycled scratch VB changes from the start, so this
      // still catches the overwrite while keeping the parse-thread cost small at ~200k
      // draws/sec (the residency loop can request many KB per slot).
      if (size > 2048u) {
        size = 2048u;
      }
      if (!size || addr == 0 || addr >= SharedMemory::kBufferSize ||
          SharedMemory::kBufferSize - addr < size) {
        continue;
      }
      out_addr[count] = addr;
      out_size[count] = size;
      ++count;
    }
  }
  if (out_count) {
    *out_count = count;
  }
  return PrecordH3HashCapturedVbRanges(out_addr, out_size, count, out_len);
}

void D3D12CommandProcessor::PrecordReplayEvents(RegisterFile* local_target) {
  // The event/draw replay loop, shared by every replay mode. The register file is
  // already rewound/built and the draw-state cache reset; replay re-applies every
  // logged write (so the file ends at the segment-end state) interleaved with the
  // draws, each draw seeing exactly its own register state. Caller sets
  // precord_replaying_ + forces the other draw-path probes off.
  //   local_target == nullptr ⇒ 1b-0 shared-rewind mode: writes go through the normal
  //     WriteRegister/WriteRegistersFromMem against the (already rewound) shared file.
  //   local_target != nullptr ⇒ local-regfile mode: writes go to `local_target` via
  //     PrecordApplyWrite*, leaving the shared register_file_ member untouched.
  // [GPU-PRECORD] Phase 1b-1c Inc 3: the log lives in the replay slot (set by the flush
  // that handed this segment off); the capture slot is a different buffer the parse
  // thread owns.
  PrecordSegment& seg = precord_slots_[precord_replay_slot_];
  for (const PrecordEvent& ev : seg.events) {
    switch (ev.kind) {
      case PrecordEvent::Kind::kWriteSingle:
        if (local_target) {
          PrecordApplyWrite(local_target, ev.a, ev.b);
        } else {
          WriteRegister(ev.a, ev.b);
        }
        break;
      case PrecordEvent::Kind::kWriteFromMem:
        if (local_target) {
          PrecordApplyWriteFromMem(local_target, ev.a, seg.frommem_data.data() + ev.c, ev.b);
        } else {
          WriteRegistersFromMem(ev.a, seg.frommem_data.data() + ev.c, ev.b);
        }
        break;
      case PrecordEvent::Kind::kDraw: {
        const PrecordDraw& draw = seg.draws[ev.a];
        // [GPU-PRECORD] Phase 1b-1c Inc 2: this draw's own active shaders (parse-time CP
        // state not carried by the register file) go to the replay-scoped fields the
        // accessors read, NOT the shared active_*_shader_ members the parse thread owns
        // (it advances them to the latest loaded shader as it runs ahead under overlap).
        precord_replay_active_vertex_shader_ = draw.active_vs;
        precord_replay_active_pixel_shader_ = draw.active_ps;
        IndexBufferInfo ibi;
        IndexBufferInfo* ibi_ptr = nullptr;
        if (draw.has_index_buffer_info) {
          ibi = draw.index_buffer_info;
          ibi_ptr = &ibi;
        }
        // [GPU-PRECORD H3-PROBE] Inc 6: re-hash this draw's guest index buffer and
        // compare to the parse-time capture; a mismatch = the guest overwrote the IB in
        // the parse->replay lead window (the H3 flicker). Diagnostic only -- the draw
        // still issues with whatever bytes are live (matching current overlap behavior).
        if (g_h3_probe && draw.h3_ib_len && draw.has_index_buffer_info) {
          uint32_t replay_len = 0;
          uint32_t replay_hash =
              PrecordH3HashIndexBuffer(draw.index_buffer_info, draw.index_count, &replay_len);
          g_h3_ib_checked.fetch_add(1, std::memory_order_relaxed);
          if (replay_len != draw.h3_ib_len || replay_hash != draw.h3_ib_hash) {
            g_h3_ib_mismatch.fetch_add(1, std::memory_order_relaxed);
          }
        }
        // [GPU-PRECORD H3-PROBE] Inc 6: same check for the bound vertex buffers, but
        // over the ranges PINNED at capture (draw.h3_vb_range_*), so a mismatch is a
        // guest overwrite of the vertex bytes -- not the register-read instability that
        // gave the earlier ~50% false floor. The suspected H3 culprit (IB was 0%).
        if (g_h3_probe && draw.h3_vb_len) {
          uint32_t replay_len = 0;
          uint32_t replay_hash = PrecordH3HashCapturedVbRanges(
              draw.h3_vb_range_addr, draw.h3_vb_range_size, draw.h3_vb_range_count, &replay_len);
          g_h3_vb_checked.fetch_add(1, std::memory_order_relaxed);
          if (replay_len != draw.h3_vb_len || replay_hash != draw.h3_vb_hash) {
            g_h3_vb_mismatch.fetch_add(1, std::memory_order_relaxed);
          }
        }
        IssueDrawImpl(draw.primitive_type, draw.index_count, ibi_ptr, draw.major_mode_explicit);
        break;
      }
    }
  }
}

void D3D12CommandProcessor::NrWalkWriteEffects(uint32_t index) {
  // [NR-FX] Phase 5-4-0: the dirty-tracking tail of WriteRegister for the
  // three constant ranges, WITHOUT the value store (the executor still owns
  // the live file while both run) and WITHOUT the dedupe check -- by the time
  // the lockstep walk fires, the executor has already stored this value, so a
  // value compare here would be vacuously equal and skip every effect; with
  // gpu_dedupe_constants off (the shipped default) the executor fires the
  // tail for every write, which is exactly what this reproduces. Stateful
  // registers (scratch/COHER/DC_LUT) are deliberately absent: their effects
  // are not idempotent and belong to the 5-4-1 census. Keep in sync with
  // WriteRegister / PrecordApplyWrite.
  if (index >= XE_GPU_REG_SHADER_CONSTANT_000_X && index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    ++g_nr_fx_probe.fl;
    if (frame_open_) {
      uint32_t float_constant_index = (index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (float_constant_index >= 256) {
        float_constant_index -= 256;
        if (current_float_constant_map_pixel_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      } else {
        if (current_float_constant_map_vertex_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
    }
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    ++g_nr_fx_probe.bl;
    cbuffer_binding_bool_loop_.up_to_date = false;
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    ++g_nr_fx_probe.fetch;
    cbuffer_binding_fetch_.up_to_date = false;
    if (texture_cache_ != nullptr) {
      texture_cache_->TextureFetchConstantWritten((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) /
                                                  6);
    }
    InvalidateVertexBufferResidency((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 2);
  }
}

bool D3D12CommandProcessor::NrSkipBackendEligible() const {
  // [NR-SKP] Phase 5-4-2: precord capture defers draws into segments and
  // replays them against rewound state -- the skip's walk-only apply and the
  // capture's write log cannot both own WriteRegister's effects. Same
  // exclusion the gpu_nr_issue seam already enforces per draw.
  return !g_precord_capture;
}

void D3D12CommandProcessor::NrDspDrawBegin(uint32_t key, bool reusable) {
  // [NR-DSP] 5-4-7-0: latch this draw's span anchor. Storage is lazy so a
  // run without the probe never pays the ~34 MB.
  if (g_dsp_slots.empty()) g_dsp_slots.resize(kDspSlots);
  g_dsp_key = key;
  g_dsp_reusable = reusable;
  g_dsp_start = deferred_command_list_.stream_size_elements();
  g_dsp_gen = deferred_command_list_.reset_generation();
  g_dsp_open = true;
}

void D3D12CommandProcessor::NrDspDrawEnd() {
  if (!g_dsp_open) return;
  g_dsp_open = false;
  NrDspProbe& p = g_dsp_probe;
  ++p.draws;
  // A Reset inside the draw (submission boundary) invalidates the anchor --
  // same rule as the buffer census, never the submission id.
  if (deferred_command_list_.reset_generation() != g_dsp_gen) {
    ++p.reset;
    return;
  }
  DspSlot& slot = g_dsp_slots[NrDspSlotIndex(g_dsp_key)];
  const bool have_prev = slot.used && slot.key == g_dsp_key;
  if (have_prev && g_dsp_reusable) {
    DeferredCommandList::NrDspDiff d;
    deferred_command_list_.NrDspCompareSpan(slot.data, slot.len, g_dsp_start, &d);
    ++p.compared;
    p.view_sites += d.view_sites;
    p.cmds += d.cmds;
    if (d.length_differs) {
      // Direction matters for the 5-4-7 design: a span's contents depend on
      // the CP's dedupe state (a command is emitted only when it CHANGES),
      // so a length mismatch means the surrounding context differed, not
      // that this draw's inputs did. Longer-now means the recording was the
      // deduped one; shorter-now means the recording carried sets this
      // execution inherited. Either way a dedupe-free recording would cover
      // it -- these are the draws that decide record-full vs gate-on-entry.
      ++p.len_ne;
      const size_t fresh =
          deferred_command_list_.stream_size_elements() - g_dsp_start;
      if (fresh > slot.len) {
        ++p.len_longer;
        p.len_delta += fresh - slot.len;
      } else {
        ++p.len_shorter;
        p.len_delta += slot.len - fresh;
      }
      if (d.real) ++p.len_and_real;
    } else if (d.real) {
      ++p.real_ne;
      if (p.first_real == 0xFFFFFFFFu) p.first_real = d.first_real;
    } else if (d.dyn_view || d.dyn_table) {
      ++p.eq_dyn;
      p.dyn_view += d.dyn_view;
      p.dyn_table += d.dyn_table;
    } else {
      ++p.eq;
    }
  } else if (!have_prev) {
    ++p.first_seen;
    if (slot.used) ++p.collision;
  }
  // Store this execution's span as the next comparison's baseline.
  const size_t len =
      deferred_command_list_.NrDspCopySpan(g_dsp_start, slot.data, kDspSlotElements);
  p.elements += deferred_command_list_.stream_size_elements() - g_dsp_start;
  if (!len) {
    ++p.unstored;  // empty (early-out draw) or longer than a slot
    slot.used = 0;
    return;
  }
  slot.key = g_dsp_key;
  slot.len = uint32_t(len);
  slot.used = 1;
}

void D3D12CommandProcessor::NrSprCaptureCtx(void* out_ctx) const {
  // [NR-SPD] every member that gates an emission the whitelist admits:
  // pipeline sets, root signature (a switch re-emits every root parameter),
  // topology, the root-up-to-date mask, the 7 bindless cbuffer bindings
  // (address decides a set's VALUE, up_to_date decides whether a recompose
  // and hence a set happens at all), and the shared-memory table flavor.
  SprCtx& c = *static_cast<SprCtx*>(out_ctx);
  std::memset(&c, 0, sizeof(c));
  c.guest_pipeline = current_guest_pipeline_;
  c.external_pipeline = current_external_pipeline_;
  c.root_signature = current_graphics_root_signature_;
  c.topology = uint32_t(primitive_topology_);
  c.ru2d = current_graphics_root_up_to_date_;
  const ConstantBufferBinding* nr_spd_cbs[7] = {
      &cbuffer_binding_fetch_,
      &cbuffer_binding_float_vertex_,
      &cbuffer_binding_float_pixel_,
      &cbuffer_binding_system_,
      &cbuffer_binding_bool_loop_,
      &cbuffer_binding_descriptor_indices_vertex_,
      &cbuffer_binding_descriptor_indices_pixel_};
  for (int i = 0; i < 7; ++i) {
    c.cb_addr[i] = uint64_t(nr_spd_cbs[i]->address);
    c.cb_utd[i] = nr_spd_cbs[i]->up_to_date ? 1 : 0;
  }
  // [NR-SPD] the system-constants ADDRESS is excluded from the gate: sys is
  // re-uploaded to a fresh pool address nearly every draw (bin-dependent
  // NDC, the [nr-rub] sys ne=100% property), so including it reads ctxmiss
  // ~100% (measured, naruto_450). Excluding it is sound: an EMITTED sys
  // site is patched with the live address at replay, and an OMITTED one is
  // covered by the post-head coverage check (the head's sys upload clears
  // the root bit when the mirror moved). The up_to_date FLAG stays in.
  c.cb_addr[3] = 0;
  c.shm_uav = current_shared_memory_binding_is_uav_.has_value()
                  ? uint8_t(current_shared_memory_binding_is_uav_.value() ? 1 : 0)
                  : uint8_t(2);
}

void D3D12CommandProcessor::NrSprDrawBegin(uint32_t key, bool reusable) {
  // [NR-SPR] 5-4-7-1: latch the span anchor, then FORCE the tail-state
  // re-emit so this draw's span is context-free -- the emulated and native
  // binding tails both re-emit any root parameter whose up-to-date bit is
  // clear, SetPipelineStateHandle/SetExternalPipeline re-emit on a cleared
  // current pipeline, and SetPrimitiveTopology on UNDEFINED. Members only:
  // nothing is emitted here, and no upload state is touched (the cbuffer
  // up_to_date flags and float constant maps stay -- forcing those would
  // force re-PACKS, which is derivation cost, not emission state). The
  // fixed-function block deliberately stays deduped: vp/sci/blend/stencil
  // are the bin-dependent set the replay keeps LIVE (5-4-6 fixup design), so
  // a span containing one is refused by the whitelist instead of recorded.
  if (g_spr_headers.empty()) {
    g_spr_headers.resize(kSprSlots);
    g_spr_payloads.resize(kSprSlots);
  }
  g_spr_key = key;
  g_spr_reusable = reusable;
  g_spr_start = deferred_command_list_.stream_size_elements();
  g_spr_gen = deferred_command_list_.reset_generation();
  g_spr_open = true;
  g_spr_replayed = false;
  g_spr_cap = SprCapture{};
  // [NR-SPD] 5-4-7-3 dedup mode: NOTHING is ever forced -- the recording is
  // the normal deduped emission, taken for free. Store traffic is gated to
  // keys with reuse history (hot): a reusable execution always records (its
  // recording is what the next reusable execution replays), and a MISS of a
  // hot key re-records instead of invalidating -- that free re-record is
  // exactly what the period-2 alternating city population needs. Cold keys
  // (never reusable) cost zero store traffic and zero scan time. The entry
  // context is snapshotted here, before anything mutates the members.
  if (g_nr_span_dedup) {
    SprHeader& s = g_spr_headers[NrSprSlotIndex(key)];
    g_spr_forced = false;
    g_spr_record = reusable || (s.key == key && s.hot);
    if (g_spr_record) NrSprCaptureCtx(&g_spr_entry_ctx);
    return;
  }
  g_spr_record = false;
  // [NR-SPW] consume-mode forcing policy (v2, after the first city A/B read
  // net negative): force + record ONLY a REUSABLE draw without a valid
  // recording -- the one execution whose recording the next reusable
  // execution can replay. A non-reusable draw is never forced and never
  // recorded; if it holds a recording, the recording just dies (inputs
  // moved -- the staleness rule), and the key's next reusable execution
  // re-records. v1 forced ~half of all city draws (every v2-miss re-
  // recorded a ~43-element context-free span the submission thread then
  // executed for nothing) and read 2-3 fps BELOW baseline at matched load.
  // Compare mode still forces ALWAYS (both compare sides context-free).
  if (g_nr_span_consume) {
    SprHeader& s = g_spr_headers[NrSprSlotIndex(key)];
    const bool stored = s.used && s.key == key;
    if (!reusable) {
      if (stored) {
        // Inputs moved: the recording must not survive as-is. A recording
        // that has SERVED replays re-records now (the alternating
        // reusable/miss population is most of the city's coverage); one
        // that never replayed just dies (no churn on dead keys).
        if (s.replay_worthy) {
          s.used = 0;
          g_spr_forced = true;
          current_guest_pipeline_ = nullptr;
          current_external_pipeline_ = nullptr;
          current_graphics_root_signature_ = nullptr;
          current_graphics_root_up_to_date_ = 0;
          primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
          return;
        }
        s.used = 0;
      }
      g_spr_forced = false;
      return;
    }
    if (stored && s.meta_valid) {
      g_spr_forced = false;  // replay candidate: no fresh forcing
      return;
    }
    g_spr_forced = true;  // reusable, recording missing/meta-less: record now
  } else {
    g_spr_forced = true;
  }
  current_guest_pipeline_ = nullptr;
  current_external_pipeline_ = nullptr;
  current_graphics_root_signature_ = nullptr;
  current_graphics_root_up_to_date_ = 0;
  primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
}

void D3D12CommandProcessor::NrSprDrawEnd() {
  if (!g_spr_open) return;
  g_spr_open = false;
  NrSprProbe& p = g_spr_probe;
  ++p.draws;
  // [NR-SPW] a replayed draw's span IS the recording (patched): nothing to
  // compare or store.
  if (g_spr_replayed) {
    g_spr_replayed = false;
    return;
  }
  // [NR-SPD] dedup mode: only record-intent brackets (reusable or hot key)
  // have any store work; cold keys exit here with zero scan cost.
  if (g_nr_span_dedup) {
    if (!g_spr_record) {
      return;
    }
  } else if (g_nr_span_consume && !g_spr_forced) {
    // [NR-SPW] consume mode: only a forced (recording) execution has store
    // work -- v2-miss invalidation already happened at Begin, fall-through
    // candidates keep their recording. Skip the span scan for everything
    // else (it was per-draw cost serving nothing).
    return;
  }
  // A Reset inside the draw invalidates the anchor -- reset generation, never
  // the submission id (the 5-4-6 trap).
  if (deferred_command_list_.reset_generation() != g_spr_gen) {
    ++p.reset;
    return;
  }
  DeferredCommandList::NrSprScan scan;
  deferred_command_list_.NrSprScanSpan(g_spr_start, &scan);
  const size_t fresh_len =
      deferred_command_list_.stream_size_elements() - g_spr_start;
  p.elements += fresh_len;
  if (!fresh_len || !scan.draw) {
    // Early-out draw (no effect / viz-killed / deferred into an instanced
    // batch): nothing a replay would reproduce.
    ++p.empty;
    return;
  }
  const bool fresh_clean = !scan.malformed && scan.draw == 1 && !scan.ff &&
                           !scan.barrier && !scan.compute && !scan.heaps &&
                           !scan.other;
  if (scan.ff) ++p.ref_ff;
  if (scan.barrier) ++p.ref_bar;
  if (scan.compute) ++p.ref_comp;
  if (scan.heaps) ++p.ref_heaps;
  if (scan.other || scan.malformed || scan.draw != 1) ++p.ref_other;
  const uint32_t spr_idx = NrSprSlotIndex(g_spr_key);
  SprHeader& hdr = g_spr_headers[spr_idx];
  SprPayload& pay = g_spr_payloads[spr_idx];
  const bool have_prev = hdr.used && hdr.key == g_spr_key;
  // [NR-SPW] consume mode: no compares (a fall-through candidate emits a
  // DEDUPED span -- comparing it against the context-free recording would
  // read as lenne and destroy a valid recording); unforced draws already
  // returned above, so consume mode falls straight to the store.
  // [NR-SPD] dedup compare gate (the city-gate mode): a reusable execution
  // with a stored recording compares fresh-vs-stored ONLY when the entry
  // context matches the recording's snapshot -- that match is exactly the
  // production replay condition, so ne/lenne here price the gate's residual
  // (census basis says ~0). A context miss is counted, not compared. Either
  // way the code FALLS THROUGH to the record path below: dedup re-records
  // every fresh execution of a hot key, so the recording always describes
  // the PREVIOUS execution and the r2 chain stays one link long.
  if (g_nr_span_dedup && !g_nr_span_consume && have_prev && g_spr_reusable) {
    if (fresh_clean) {
      if (std::memcmp(&g_spr_entry_ctx, &pay.entry_ctx, sizeof(SprCtx)) != 0) {
        ++p.ctx_miss;
      } else {
        DeferredCommandList::NrDspDiff d;
        deferred_command_list_.NrDspCompareSpan(pay.data, pay.len, g_spr_start,
                                                &d);
        ++p.compared;
        p.view_sites += d.view_sites;
        if (d.length_differs) {
          ++p.len_ne;
        } else if (d.real) {
          ++p.real_ne;
          if (p.first_real == 0xFFFFFFFFu) p.first_real = d.first_real;
        } else if (d.dyn_view || d.dyn_table) {
          ++p.eq_dyn;
          p.dyn_view += d.dyn_view;
          p.dyn_table += d.dyn_table;
        } else {
          ++p.eq;
        }
      }
    } else {
      ++p.cmp_refused;
    }
    // no return: fall through to the upsert
  } else if (!g_nr_span_consume && have_prev && g_spr_reusable) {
    if (fresh_clean) {
      // The recording is clean by construction; the fresh span is clean too:
      // lockstep compare with the patch model (root-view addresses at the
      // same root parameter = what a replay overwrites; table bases counted
      // separately, city measured them immobile).
      DeferredCommandList::NrDspDiff d;
      deferred_command_list_.NrDspCompareSpan(pay.data, pay.len, g_spr_start,
                                              &d);
      ++p.compared;
      p.view_sites += d.view_sites;
      if (d.length_differs) {
        // Refuse-and-re-record (the 5-4-7 design's rule for the `long`
        // class): invalidate so the next execution records fresh, and ne
        // counts transition events rather than a stuck recording.
        ++p.len_ne;
        hdr.used = 0;
      } else if (d.real) {
        ++p.real_ne;
        if (p.first_real == 0xFFFFFFFFu) p.first_real = d.first_real;
        hdr.used = 0;
      } else if (d.dyn_view || d.dyn_table) {
        ++p.eq_dyn;
        p.dyn_view += d.dyn_view;
        p.dyn_table += d.dyn_table;
      } else {
        ++p.eq;
      }
    } else {
      // This execution emitted something a replay would refuse (texture
      // load, barrier, ff change riding the bracket). Production would have
      // replayed the clean recording WITHOUT that ambient work -- these are
      // the draws whose rate prices that risk, so they are counted, not
      // folded into ne.
      ++p.cmp_refused;
    }
    return;  // the recording stays fixed once taken
  }
  if (!have_prev) {
    ++p.first_seen;
    if (hdr.used) ++p.collision;
  }
  // Record (first execution of this key, a collision eviction, or a
  // NON-reusable execution -- the reuse verdict certifies identity only
  // against the PREVIOUS execution, so a v2 miss re-records; the transitive
  // chain of reusable verdicts then keeps every later compare sound against
  // this recording).
  if (!fresh_clean) {
    // Inputs moved and this execution is not recordable: a stale recording
    // must not survive to be compared (or replayed) against.
    if (have_prev) hdr.used = 0;
    return;
  }
  if (fresh_len > kSprSlotElements) {
    ++p.too_long;
    hdr.used = 0;  // same staleness rule as the unclean path
    return;
  }
  const size_t len = deferred_command_list_.NrDspCopySpan(g_spr_start, pay.data,
                                                          kSprSlotElements);
  if (!len) {
    hdr.used = 0;
    return;
  }
  if (hdr.key != g_spr_key) {  // eviction: new identity
    hdr.replay_worthy = 0;
    hdr.hot = 0;
  }
  hdr.key = g_spr_key;
  pay.len = uint32_t(len);
  pay.view_count = scan.view_offset_count;
  std::memcpy(pay.view_offsets, scan.view_offsets, sizeof(pay.view_offsets));
  hdr.used = 1;
  ++p.stored;
  // [NR-SPD] dedup recordings carry the coverage mask + both context
  // snapshots (entry from bracket Begin, exit from the members right now,
  // post-emission), and reusable executions mark the key hot.
  if (g_nr_span_dedup) {
    pay.root_mask = scan.root_mask;
    pay.entry_ctx = g_spr_entry_ctx;
    NrSprCaptureCtx(&pay.exit_ctx);
    if (g_spr_reusable) hdr.hot = 1;
  }
  // [NR-SPW] replay metadata. Eligible only when the capture reached the
  // draw tail unrefused, every view site was collected, and every patch
  // site is one of the 7 bindless root CBVs (whitelist again: an unknown
  // root refuses replay, never guesses).
  hdr.meta_valid = 0;
  if (g_spr_cap.valid && !g_spr_cap.refused &&
      scan.view_offset_count == scan.view_sites &&
      scan.view_offset_count <= DeferredCommandList::kNrSprMaxViewSites) {
    uint32_t roots[DeferredCommandList::kNrSprMaxViewSites];
    bool roots_ok = DeferredCommandList::NrSprViewSiteRoots(
        pay.data, pay.view_offsets, pay.view_count, roots);
    for (uint8_t i = 0; i < pay.view_count && roots_ok; ++i) {
      switch (roots[i]) {
        case kRootParameter_Bindless_FetchConstants:
        case kRootParameter_Bindless_FloatConstantsVertex:
        case kRootParameter_Bindless_FloatConstantsPixel:
        case kRootParameter_Bindless_SystemConstants:
        case kRootParameter_Bindless_BoolLoopConstants:
        case kRootParameter_Bindless_DescriptorIndicesVertex:
        case kRootParameter_Bindless_DescriptorIndicesPixel:
          pay.view_roots[i] = uint8_t(roots[i]);
          break;
        default:
          roots_ok = false;
          break;
      }
    }
    if (roots_ok) {
      pay.vs = g_spr_cap.vs;
      pay.ps = g_spr_cap.ps;
      pay.tex_mask = g_spr_cap.tex_mask;
      pay.llci = g_spr_cap.llci;
      pay.ib_base = g_spr_cap.ib_base;
      pay.ib_size = g_spr_cap.ib_size;
      pay.ib_dma = g_spr_cap.ib_dma;
      pay.index_endian = g_spr_cap.index_endian;
      pay.smp_heap_ptr = sampler_bindless_heap_gpu_start_.ptr;
      pay.view_heap_ptr = view_bindless_heap_gpu_start_.ptr;
      hdr.meta_valid = 1;
    }
  }
}

void D3D12CommandProcessor::NrRufRestoreFromBundle(const void* bundle) {
  // [NR-RUF] 5-4-5-2 restore body, extracted VERBATIM from NrUpdateBindings
  // so the 5-4-7-2 span replay leaves the binding state machine exactly as
  // a fast draw would: guest cbuffer + descriptor-indices addresses,
  // sampler params/indices, SRV keys and layout uids all become the
  // bundle's (= this draw's previous same-frame execution's).
  const NrRubBundle& b = *static_cast<const NrRubBundle*>(bundle);
  const auto ruf_restore_cb = [this](ConstantBufferBinding& cb,
                                     D3D12_GPU_VIRTUAL_ADDRESS a,
                                     uint32_t root_parameter) {
    if (cb.address != a) {
      cb.address = a;
      current_graphics_root_up_to_date_ &= ~(1u << root_parameter);
    }
    cb.up_to_date = true;
  };
  ruf_restore_cb(cbuffer_binding_float_vertex_, b.a_flt_v,
                 kRootParameter_Bindless_FloatConstantsVertex);
  ruf_restore_cb(cbuffer_binding_float_pixel_, b.a_flt_p,
                 kRootParameter_Bindless_FloatConstantsPixel);
  ruf_restore_cb(cbuffer_binding_bool_loop_, b.a_bl,
                 kRootParameter_Bindless_BoolLoopConstants);
  ruf_restore_cb(cbuffer_binding_fetch_, b.a_ftc,
                 kRootParameter_Bindless_FetchConstants);
  ruf_restore_cb(cbuffer_binding_descriptor_indices_vertex_, b.a_di_v,
                 kRootParameter_Bindless_DescriptorIndicesVertex);
  ruf_restore_cb(cbuffer_binding_descriptor_indices_pixel_, b.a_di_p,
                 kRootParameter_Bindless_DescriptorIndicesPixel);
  current_sampler_layout_uid_vertex_ = b.smp_uid_v;
  current_sampler_layout_uid_pixel_ = b.smp_uid_p;
  current_texture_layout_uid_vertex_ = b.tex_uid_v;
  current_texture_layout_uid_pixel_ = b.tex_uid_p;
  current_samplers_vertex_.resize(
      std::max(current_samplers_vertex_.size(), b.smp_v.size()));
  for (size_t i = 0; i < b.smp_v.size(); ++i) {
    current_samplers_vertex_[i].value =
        decltype(current_samplers_vertex_[i].value)(b.smp_v[i]);
  }
  current_samplers_pixel_.resize(
      std::max(current_samplers_pixel_.size(), b.smp_p.size()));
  for (size_t i = 0; i < b.smp_p.size(); ++i) {
    current_samplers_pixel_[i].value =
        decltype(current_samplers_pixel_[i].value)(b.smp_p[i]);
  }
  current_sampler_bindless_indices_vertex_.resize(std::max(
      current_sampler_bindless_indices_vertex_.size(), b.si_v.size()));
  for (size_t i = 0; i < b.si_v.size(); ++i) {
    current_sampler_bindless_indices_vertex_[i] = b.si_v[i];
  }
  current_sampler_bindless_indices_pixel_.resize(std::max(
      current_sampler_bindless_indices_pixel_.size(), b.si_p.size()));
  for (size_t i = 0; i < b.si_p.size(); ++i) {
    current_sampler_bindless_indices_pixel_[i] = b.si_p[i];
  }
  current_texture_srv_keys_vertex_.resize(std::max(
      current_texture_srv_keys_vertex_.size(), b.keys_v.size()));
  std::copy(b.keys_v.begin(), b.keys_v.end(),
            current_texture_srv_keys_vertex_.begin());
  current_texture_srv_keys_pixel_.resize(std::max(
      current_texture_srv_keys_pixel_.size(), b.keys_p.size()));
  std::copy(b.keys_p.begin(), b.keys_p.end(),
            current_texture_srv_keys_pixel_.begin());
}

bool D3D12CommandProcessor::NrSpanReplayTry() {
  // [NR-SPW] Phase 5-4-7-2: serve this draw from its recording. Everything
  // up to the final memcpy is a FALL-THROUGH on failure: the full path
  // redoes each head step (RT update guarded, requests dup-cheap, restore
  // idempotent, sys re-derive idempotent) with its own proven
  // logging/abort behavior, so no failure here needs one of its own.
  NrSpwProbe& w = g_spw_probe;
  if (g_spr_forced || !g_spr_reusable) return false;
  const uint32_t spr_idx = NrSprSlotIndex(g_spr_key);
  SprHeader& hdr = g_spr_headers[spr_idx];
  SprPayload& pay = g_spr_payloads[spr_idx];
  if (!hdr.used || hdr.key != g_spr_key || !hdr.meta_valid) return false;
  ++w.cand;
  // [NR-SPWP] stage stamps, successful replays only (they are ~100% of
  // candidates), accumulated once at the end.
  const bool spwp = g_draw_prof;
  std::chrono::steady_clock::time_point spwp_t[9];
  if (spwp) spwp_t[0] = std::chrono::steady_clock::now();
  // An open instanced batch owns the stream tail; the merge-or-flush
  // decision belongs to the full path.
  if (g_instance && instanced_batch_.active) {
    ++w.fb_batch;
    return false;
  }
  if (!BeginSubmission(true)) {
    ++w.fb_begin;
    return false;
  }
  // [NR-SPD] the emission-context gate, AFTER BeginSubmission on purpose: a
  // submission open resets the dedupe members, and a deduped recording's
  // omitted commands are only sound when the CURRENT members equal the
  // recording's entry snapshot (comparing before the reset would false-match
  // and replay a span that inherits from unbound state). A miss falls
  // through fresh, which re-records for free.
  if (g_nr_span_dedup) {
    SprCtx nr_spd_now;
    NrSprCaptureCtx(&nr_spd_now);
    if (std::memcmp(&nr_spd_now, &pay.entry_ctx, sizeof(SprCtx)) != 0) {
      ++w.fb_ctx;
      return false;
    }
  }
  // Heap identity: a recreated bindless heap makes the recorded root
  // descriptor tables stale (never observed live; unit-only class).
  if (pay.smp_heap_ptr != sampler_bindless_heap_gpu_start_.ptr ||
      pay.view_heap_ptr != view_bindless_heap_gpu_start_.ptr) {
    ++w.fb_heap;
    hdr.meta_valid = 0;
    return false;
  }
  // The bundle supplies the patch addresses (same-frame pool validity, the
  // fast path's own rule).
  const NrRubBundle* b = NrRubFind(g_spr_key);
  if (!b || !b->packs_valid || b->frame != frame_current_) {
    ++w.fb_bundle;
    return false;
  }
  if (spwp) spwp_t[1] = std::chrono::steady_clock::now();
  const D3D12Shader* vs = pay.vs;
  const D3D12Shader* ps = pay.ps;
  // Texture requests first (loads can move host SRV indices), then the
  // key-freshness valve over the bundle's keys -- the same live check the
  // fast path keeps; a stale key means the di pack must be recomposed,
  // which only the full path can do.
  texture_cache_->RequestTextures(pay.tex_mask);
  const std::vector<D3D12Shader::TextureBinding>& nr_tex_v =
      vs->GetTextureBindingsAfterTranslation();
  if (!nr_tex_v.empty() &&
      (b->keys_v.size() < nr_tex_v.size() ||
       !texture_cache_->AreActiveTextureSRVKeysUpToDate(
           b->keys_v.data(), nr_tex_v.data(), nr_tex_v.size()))) {
    ++w.fb_valve;
    return false;
  }
  if (ps) {
    const std::vector<D3D12Shader::TextureBinding>& nr_tex_p =
        ps->GetTextureBindingsAfterTranslation();
    if (!nr_tex_p.empty() &&
        (b->keys_p.size() < nr_tex_p.size() ||
         !texture_cache_->AreActiveTextureSRVKeysUpToDate(
             b->keys_p.data(), nr_tex_p.data(), nr_tex_p.size()))) {
      ++w.fb_valve;
      return false;
    }
  }
  if (spwp) spwp_t[2] = std::chrono::steady_clock::now();
  const RegisterFile& regs = GetActiveDrawRegisterFile();
  // The live head, in the fresh path's order, over the stored shader facts.
  const bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  const bool is_rasterization_done =
      draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  reg::RB_DEPTHCONTROL normalized_depth_control =
      draw_util::GetNormalizedDepthControl(regs);
  const uint32_t normalized_color_mask =
      ps ? draw_util::GetNormalizedColorMask(regs, ps->writes_color_targets()) : 0;
  if (!render_target_cache_->Update(is_rasterization_done, normalized_depth_control,
                                    normalized_color_mask, *vs)) {
    ++w.fb_rt;
    return false;
  }
  if (spwp) spwp_t[3] = std::chrono::steady_clock::now();
  const uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  const uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();
  const bool host_render_targets_used =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
  const bool convert_z_to_float24 =
      host_render_targets_used && render_target_cache_->depth_float24_convert_in_pixel_shader();
  const bool ps_writes_depth = ps && ps->writes_depth();
  ViewportCacheKey viewport_key;
  viewport_key.pa_cl_clip_cntl = regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
  viewport_key.pa_cl_vte_cntl = regs[XE_GPU_REG_PA_CL_VTE_CNTL];
  viewport_key.pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
  viewport_key.pa_su_vtx_cntl = regs[XE_GPU_REG_PA_SU_VTX_CNTL];
  viewport_key.pa_sc_window_offset = regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
  viewport_key.normalized_depth_control = normalized_depth_control.value;
  std::memcpy(viewport_key.vport_regs, &regs[XE_GPU_REG_PA_CL_VPORT_XSCALE],
              sizeof(viewport_key.vport_regs));
  viewport_key.flags = (uint32_t(convert_z_to_float24) << 0) |
                       (uint32_t(host_render_targets_used) << 1) |
                       (uint32_t(ps_writes_depth) << 2);
  draw_util::ViewportInfo viewport_info;
  if (viewport_cache_valid_ && viewport_key == previous_viewport_key_) {
    viewport_info = previous_viewport_info_;
  } else {
    draw_util::GetHostViewportInfo(regs, draw_resolution_scale_x, draw_resolution_scale_y, true,
                                   D3D12_VIEWPORT_BOUNDS_MAX, D3D12_VIEWPORT_BOUNDS_MAX, false,
                                   normalized_depth_control, convert_z_to_float24,
                                   host_render_targets_used, ps_writes_depth, viewport_info);
    previous_viewport_key_ = viewport_key;
    previous_viewport_info_ = viewport_info;
    viewport_cache_valid_ = true;
  }
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  NrDetileWidenDrawScissor(regs, scissor);
  scissor.offset[0] *= draw_resolution_scale_x;
  scissor.offset[1] *= draw_resolution_scale_y;
  scissor.extent[0] *= draw_resolution_scale_x;
  scissor.extent[1] *= draw_resolution_scale_y;
  UpdateFixedFunctionState(viewport_info, scissor, primitive_polygonal,
                           normalized_depth_control);
  if (spwp) spwp_t[4] = std::chrono::steady_clock::now();
  // System constants: under the consume preconditions this is the lean
  // mirror path (bin-dependent NDC lives here), which sets the sys dirty
  // flag the upload below consumes -- the swap's own upload block, minus
  // its staging probe.
  UpdateSystemConstantValues(false, primitive_polygonal, pay.llci,
                             xenos::Endian(pay.index_endian), viewport_info, pay.tex_mask,
                             normalized_depth_control, normalized_color_mask);
  if (!cbuffer_binding_system_.up_to_date) {
    uint8_t* system_constants = constant_buffer_pool_->Request(
        frame_current_, sizeof(g_nr_sys_state), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_system_.address);
    if (system_constants == nullptr) {
      ++w.fb_sys;
      return false;
    }
    std::memcpy(system_constants, &g_nr_sys_state, sizeof(g_nr_sys_state));
    cbuffer_binding_system_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_SystemConstants);
  }
  if (spwp) spwp_t[5] = std::chrono::steady_clock::now();
  // Float-map member update (the layout check the binding tail runs first),
  // then the bundle restore -- same order as NrUpdateBindings, so the
  // member machine ends exactly as a fast draw leaves it.
  {
    const Shader::ConstantRegisterMap& map_v = vs->constant_register_map();
    for (uint32_t i = 0; i < 4; ++i) {
      if (current_float_constant_map_vertex_[i] != map_v.float_bitmap[i]) {
        current_float_constant_map_vertex_[i] = map_v.float_bitmap[i];
        if (map_v.float_count) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
    }
    if (ps) {
      const Shader::ConstantRegisterMap& map_p = ps->constant_register_map();
      for (uint32_t i = 0; i < 4; ++i) {
        if (current_float_constant_map_pixel_[i] != map_p.float_bitmap[i]) {
          current_float_constant_map_pixel_[i] = map_p.float_bitmap[i];
          if (map_p.float_count) {
            cbuffer_binding_float_pixel_.up_to_date = false;
          }
        }
      }
    } else {
      std::memset(current_float_constant_map_pixel_, 0,
                  sizeof(current_float_constant_map_pixel_));
    }
  }
  NrRufRestoreFromBundle(b);
  if (spwp) spwp_t[6] = std::chrono::steady_clock::now();
  // Vertex residency: the fresh loop's happy path over the stored vertex
  // shader's fetch bitmap; any abnormal case (invalid fetch type, request
  // failure) falls through for the full path's exact behavior.
  const Shader::ConstantRegisterMap& constant_map_vertex = vs->constant_register_map();
  for (uint32_t i = 0; i < rex::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
    uint32_t vfetch_bits_remaining = constant_map_vertex.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(vfetch_bits_remaining, &j)) {
      vfetch_bits_remaining &= ~(uint32_t(1) << j);
      const uint32_t vfetch_index = i * 32 + j;
      const uint64_t vfetch_bit = uint64_t(1) << (vfetch_index & 63);
      if (vertex_buffers_in_sync_[vfetch_index >> 6] & vfetch_bit) {
        continue;
      }
      const xenos::xe_gpu_vertex_fetch_t vfetch_constant = regs.GetVertexFetch(vfetch_index);
      if (vfetch_constant.type != xenos::FetchConstantType::kVertex &&
          !(vfetch_constant.type == xenos::FetchConstantType::kInvalidVertex &&
            REXCVAR_GET(gpu_allow_invalid_fetch_constants))) {
        ++w.fb_vf;
        return false;
      }
      VertexBufferState& state = vertex_buffer_states_[vfetch_index];
      if (state.address == vfetch_constant.address && state.size == vfetch_constant.size) {
        vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
        continue;
      }
      if (!shared_memory_->RequestRange(vfetch_constant.address << 2,
                                        vfetch_constant.size << 2)) {
        ++w.fb_vf;
        return false;
      }
      state.address = vfetch_constant.address;
      state.size = vfetch_constant.size;
      vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
    }
  }
  // Index residency (guest DMA only; builtin buffers are static, converted
  // was refused at record).
  if (pay.ib_dma && !shared_memory_->RequestRange(pay.ib_base, pay.ib_size)) {
    ++w.fb_ib;
    return false;
  }
  // [NR-SPD] post-head coverage: the head (restore, sys upload) may have
  // cleared root-up-to-date bits; every root the recording's exit context
  // says must end up bound-and-current has to be either still current or
  // re-established by the span itself. A root that is neither would replay
  // into a stale binding -- refuse (fresh path re-runs everything, head
  // steps idempotent).
  if (g_nr_span_dedup) {
    const uint32_t nr_spd_missing = pay.exit_ctx.ru2d &
                                    ~current_graphics_root_up_to_date_ &
                                    ~pay.root_mask;
    if (nr_spd_missing) {
      ++w.fb_cover;
      return false;
    }
  }
  // Shared memory read state + everything the head queued.
  shared_memory_->UseForReading();
  SubmitBarriers();
  if (spwp) spwp_t[7] = std::chrono::steady_clock::now();
  // The replay: recording -> stream (one memcpy), then patch the root-CBV
  // addresses in place from the member state just established.
  uintmax_t* patched = deferred_command_list_.NrSprAppendRaw(pay.len);
  std::memcpy(patched, pay.data, pay.len * sizeof(uintmax_t));
  for (uint8_t i = 0; i < pay.view_count; ++i) {
    uint64_t nr_patch_addr;
    switch (pay.view_roots[i]) {
      case kRootParameter_Bindless_FetchConstants:
        nr_patch_addr = cbuffer_binding_fetch_.address;
        break;
      case kRootParameter_Bindless_FloatConstantsVertex:
        nr_patch_addr = cbuffer_binding_float_vertex_.address;
        break;
      case kRootParameter_Bindless_FloatConstantsPixel:
        nr_patch_addr = cbuffer_binding_float_pixel_.address;
        break;
      case kRootParameter_Bindless_SystemConstants:
        nr_patch_addr = cbuffer_binding_system_.address;
        break;
      case kRootParameter_Bindless_BoolLoopConstants:
        nr_patch_addr = cbuffer_binding_bool_loop_.address;
        break;
      case kRootParameter_Bindless_DescriptorIndicesVertex:
        nr_patch_addr = cbuffer_binding_descriptor_indices_vertex_.address;
        break;
      case kRootParameter_Bindless_DescriptorIndicesPixel:
        nr_patch_addr = cbuffer_binding_descriptor_indices_pixel_.address;
        break;
      default:
        // Cannot happen: roots validated at store time (and the span is
        // already appended -- returning here would double-draw).
        assert_unhandled_case(pay.view_roots[i]);
        continue;
    }
    DeferredCommandList::NrSprPatchViewAddress(patched, pay.view_offsets[i], nr_patch_addr);
  }
  if (g_nr_span_dedup) {
    // [NR-SPD] apply the recording's EXIT context: the members now describe
    // the stream exactly as the record-time fresh execution left them, so
    // the NEXT draw's entry context keeps matching its own recording --
    // clearing instead (the context-free rule below) would force a fresh
    // re-emit on every following candidate and reintroduce the period-2
    // tax. Every bit set in exit ru2d is genuinely bound-and-current here:
    // it was either current before the memcpy or re-established by the span
    // (the coverage check above is exactly that invariant).
    const SprCtx& nr_spd_x = pay.exit_ctx;
    current_guest_pipeline_ = const_cast<void*>(nr_spd_x.guest_pipeline);
    current_external_pipeline_ = static_cast<ID3D12PipelineState*>(
        const_cast<void*>(nr_spd_x.external_pipeline));
    current_graphics_root_signature_ = static_cast<ID3D12RootSignature*>(
        const_cast<void*>(nr_spd_x.root_signature));
    current_graphics_root_up_to_date_ = nr_spd_x.ru2d;
    primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY(nr_spd_x.topology);
    if (nr_spd_x.shm_uav == 2) {
      current_shared_memory_binding_is_uav_.reset();
    } else {
      current_shared_memory_binding_is_uav_ = nr_spd_x.shm_uav != 0;
    }
  } else {
    // The stream now holds the recording's tail state and the CP's dedupe
    // members do not know it: clear the same five the forcing clears so the
    // next fresh draw re-emits (replayed draws never read them). The span
    // bound the SRV flavor of the shared-memory table.
    current_guest_pipeline_ = nullptr;
    current_external_pipeline_ = nullptr;
    current_graphics_root_signature_ = nullptr;
    current_graphics_root_up_to_date_ = 0;
    primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
    current_shared_memory_binding_is_uav_ = false;
  }
  ++w.rep;
  w.rep_elements += pay.len;
  w.rep_patched += pay.view_count;
  hdr.replay_worthy = 1;
  g_spr_replayed = true;
  if (spwp) {
    spwp_t[8] = std::chrono::steady_clock::now();
    const auto d = [&](int a, int c) {
      return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(spwp_t[c] - spwp_t[a])
                          .count());
    };
    w.ns_pre += d(0, 1);
    w.ns_tex += d(1, 2);
    w.ns_rt += d(2, 3);
    w.ns_vp += d(3, 4);
    w.ns_sys += d(4, 5);
    w.ns_rst += d(5, 6);
    w.ns_res += d(6, 7);
    w.ns_emit += d(7, 8);
  }
  return true;
}

// [NR-TIL] N-4-1: a resolve, a band change or a frame ends the segment. The
// next draw opens a new one and decides record / replay / pass-through.
void D3D12CommandProcessor::NrTileSegBreak() { g_tile_seg_open = false; }

void D3D12CommandProcessor::NrTileBeginDraw(uint32_t win_off, uint32_t prim,
                                            uint32_t index_count, uint32_t ib_base) {
  NrTilProbe& p = g_tile_p;
  ++p.draws;
  if (g_tile_frame != frame_current_) {
    // [NR-TIL-DORMANT] decide from the frame that just ENDED, before the
    // reset: no repeat-band draw reached IssueDraw => nothing this frame's
    // recording could ever have served => the next frame records nothing.
    g_tile_dormant = !g_tile_verify && p.repeat == g_tile_frame_rep0;
    g_tile_frame_rep0 = p.repeat;
    // Every recording is same-frame BY RULE: the spans embed per-frame
    // constant-pool GPU addresses, and the three bands of one frame are the
    // only executions that ever share them.
    g_tile_frame = frame_current_;
    g_tile_sys_prev = SIZE_MAX;  // [NR-TILSYS] indices are per-frame
    g_tile_recs.clear();
    g_tile_texsnaps.clear();
    g_tile_segs.clear();
    g_tile_arena_used = 0;
    g_tile_seg_open = false;
  }
  if (g_tile_dormant) {
    // The dormant fast path: only the wake detector runs. A repeat-band
    // draw here falls through to the full derivation (mode 0) and its
    // count wakes recording at the next frame boundary.
    ++p.dormant;
    if (win_off == 0) {
      ++p.base;
    } else {
      ++p.repeat;
    }
    g_tile_mode = 0;
    g_tile_rec_open = false;
    g_tile_cmp_open = false;
    return;
  }
  if (!g_tile_seg_open || win_off != g_tile_win) {
    g_tile_win = win_off;
    g_tile_seg_open = true;
    g_tile_ord = 0;
    g_tile_mode = 0;
    if (win_off == 0) {
      if (g_tile_segs.size() < kTileMaxSegs && g_tile_recs.size() < kTileMaxRecs) {
        g_tile_segs.push_back(
            TileSeg{nr_tile_draw_addr_, uint32_t(g_tile_recs.size()), 0});
        g_tile_seg = uint32_t(g_tile_segs.size()) - 1;
        g_tile_ord_bias = 0;
        g_tile_mode = 1;
        ++p.segs_rec;
      } else {
        ++p.seg_cap;
      }
    } else {
      // Name the repeat band by its first draw stream position and find the
      // base-band segment it repeats. At most kTileMaxSegs entries, once per
      // segment: the whole lookup cost of the mechanism.
      g_tile_ord_bias = 0;
      for (uint32_t i = 0; i < g_tile_segs.size(); ++i) {
        if (g_tile_segs[i].addr0 == nr_tile_draw_addr_ && g_tile_segs[i].count) {
          g_tile_seg = i;
          g_tile_mode = 2;
          ++p.segs_rep;
          break;
        }
      }
      if (g_tile_mode != 2) {
        // The base band breaks its segments where its OWN resolves and band
        // changes fall, and a repeat band opens on the band change alone, so
        // a pass that follows another at window offset 0 with no resolve
        // between them is one base segment and two repeat ones. Name the
        // repeat segment by finding its first draw among the base RECORDS
        // and biasing the ordinal. Once per segment, and only when the
        // address is unique -- two records with it cannot be told apart.
        uint32_t hits = 0, hit_seg = 0, hit_ord = 0;
        for (uint32_t i = 0; i < g_tile_segs.size() && hits < 2; ++i) {
          const TileSeg& sg = g_tile_segs[i];
          for (uint32_t o = 0; o < sg.count; ++o) {
            if (g_tile_recs[sg.first + o].addr == nr_tile_draw_addr_) {
              ++hits;
              hit_seg = i;
              hit_ord = o;
              if (hits >= 2) break;
            }
          }
        }
        if (hits == 1) {
          g_tile_seg = hit_seg;
          g_tile_ord_bias = hit_ord;
          g_tile_mode = 2;
          ++p.segs_rep;
          ++p.nm_bias;
        } else if (hits > 1) {
          ++p.nm_ambig;
        } else {
          ++p.nm_absent;
        }
      }
      if (g_tile_mode != 2) {
        ++p.seg_nomatch;
        bool named = false;
        for (uint32_t i = 0; i < g_tile_segs.size(); ++i) {
          if (g_tile_segs[i].addr0 == nr_tile_draw_addr_) {
            named = true;
            break;
          }
        }
        if (named) {
          ++p.nm_empty;
        } else {
          ++p.nm_none;
        }
      }
    }
  }
  if (win_off == 0) {
    ++p.base;
  } else {
    ++p.repeat;
    if (g_tile_mode != 2) ++p.nm_draws;
  }
  g_tile_cur_ord = g_tile_ord++;
  g_tile_prim = prim;
  g_tile_icount = index_count;
  g_tile_ibbase = ib_base;
  g_tile_rec_open = false;
  // The tile recorder owns g_spr_cap when the span machinery is off (the
  // two are mutually exclusive at the latch); the emission tail fills it.
  g_spr_cap = SprCapture{};
  // The comparison point for both sides: state at the TOP of the draw,
  // before any of this draw own emissions. Capturing it after the pipeline
  // bind (the first version) compared the recording POST-bind context
  // against the replay PRE-bind one, and refused every draw that changed
  // pipeline -- 58% of menu candidates, all of them attributed to the
  // pipeline field by the ctxwhy split.
  g_tile_top_guest_pso = current_guest_pipeline_;
  g_tile_top_ext_pso = current_external_pipeline_;
  if (g_tile_mode == 1) {
    // Hold the ordinal for EVERY base-band draw, including one that aborts
    // or refuses: the bands only stay in lockstep if the ordinals line up,
    // and a valid=0 record is how a refusal is expressed.
    if (g_tile_recs.size() >= kTileMaxRecs) {
      ++p.rr_cap;
      g_tile_mode = 0;
      return;
    }
    NrSprCaptureCtx(&g_tile_entry_ctx);
    TileRec& r = g_tile_recs.emplace_back();
    r.addr = nr_tile_draw_addr_;
    r.prim = prim;
    r.index_count = index_count;
    r.ib_base = ib_base;
    g_tile_segs[g_tile_seg].count = g_tile_ord;
    g_tile_rec_open = true;
  }
  g_tile_cmp_open = false;
  if (g_tile_mode == 2 && g_tile_verify) {
    // Mode 3: let the full derivation run and compare its span against the
    // recording it would have replayed. Same anchors, same bracket -- only
    // the tail differs. The arming refusals reuse the consuming path
    // counters so the two modes report the same population the same way.
    const TileSeg& seg = g_tile_segs[g_tile_seg];
    if (g_tile_cur_ord + g_tile_ord_bias >= seg.count) {
      ++p.fb_ord;
    } else {
      const uint32_t idx = seg.first + g_tile_ord_bias + g_tile_cur_ord;
      const TileRec& rc = g_tile_recs[idx];
      if (!rc.valid) {
        ++p.fb_norec;
      } else if (rc.addr != nr_tile_draw_addr_ || rc.prim != prim ||
                 rc.index_count != index_count || rc.ib_base != ib_base) {
        ++p.fb_shape;
      } else {
        ++p.cand;
        g_tile_cmp_idx = idx;
        g_tile_cmp_open = true;
        NrSprCaptureCtx(&g_tile_entry_ctx);
      }
    }
  }
}

// The pipeline-bind anchor, after the texture requests (which can emit
// loads and barriers) and before the pipeline set. Everything from here to
// the fixed-function state is at most one pipeline set.
void D3D12CommandProcessor::NrTileRecordAnchorA() {
  if (!g_tile_rec_open && !g_tile_cmp_open) return;
  // The texture requests are allowed to bind pipelines (a load blit). If one
  // did, the pipeline piece below was recorded against a state the replay
  // will not reproduce -- refuse the record rather than replay a draw whose
  // pipeline set was deduped against a blit.
  if (current_guest_pipeline_ != g_tile_top_guest_pso ||
      current_external_pipeline_ != g_tile_top_ext_pso) {
    ++g_tile_p.rr_class;
    g_tile_rec_open = false;
    g_tile_cmp_open = false;
    return;
  }
  g_tile_spanA_start = deferred_command_list_.stream_size_elements();
  g_tile_spanA_end = g_tile_spanA_start;
  g_tile_span_gen = deferred_command_list_.reset_generation();
}

// Closes the pipeline piece immediately before UpdateFixedFunctionState:
// the viewport and scissor sets it is about to emit are the bin-dependent
// set, and they must never enter a recording.
void D3D12CommandProcessor::NrTileRecordSplit() {
  if (!g_tile_rec_open && !g_tile_cmp_open) return;
  g_tile_spanA_end = deferred_command_list_.stream_size_elements();
}

void D3D12CommandProcessor::NrTileRecordAnchor() {
  if (!g_tile_rec_open && !g_tile_cmp_open) return;
  g_tile_span_start = deferred_command_list_.stream_size_elements();
}

void D3D12CommandProcessor::NrTileRecordEnd() {
  if (!g_tile_rec_open) return;
  g_tile_rec_open = false;
  NrTilProbe& p = g_tile_p;
  TileRec& r = g_tile_recs[g_tile_segs[g_tile_seg].first + g_tile_cur_ord];
  // A submission ended inside the draw: the anchor points into a refilled
  // stream. (The submission id is NOT this check: the Reset happens at the
  // next BeginSubmission, so an anchor can die under an unchanged id.)
  if (deferred_command_list_.reset_generation() != g_tile_span_gen) {
    ++p.rr_gen;
    return;
  }
  // ... or a frame boundary moved the constant-buffer pool under the GPU
  // addresses the span embeds.
  if (frame_current_ != g_tile_frame) {
    ++p.rr_gen;
    return;
  }
  // The pipeline piece: empty, or exactly one pipeline set. Anything else
  // between the texture requests and the fixed-function state would be
  // silently dropped by the replay.
  {
    const size_t la = deferred_command_list_.NrTilePipelineSpan(
        g_tile_spanA_start, g_tile_spanA_end, r.dataA, rex::countof(r.dataA));
    if (la == SIZE_MAX) {
      ++p.rr_pso;
      return;
    }
    r.lenA = uint8_t(la);
  }
  if (!g_spr_cap.valid || g_spr_cap.refused || !g_spr_cap.vs) {
    ++p.rr_class;
    return;
  }
  DeferredCommandList::NrSprScan scan;
  deferred_command_list_.NrSprScanSpan(g_tile_span_start, &scan);
  if (scan.malformed) {
    ++p.rr_scan;
    return;
  }
  // Whitelist: exactly one draw, and nothing whose replay would be wrong or
  // non-idempotent. ff = a viewport / scissor / blend-factor / stencil-ref
  // set (bin-dependent, must stay live); barrier / compute / copy / clear /
  // dispatch / query = texture-load and resolve traffic that has to
  // regenerate; heaps = a descriptor-heap switch the replay does not make.
  if (scan.draw != 1 || scan.ff || scan.barrier || scan.compute || scan.heaps ||
      scan.other) {
    ++p.rr_class;
    return;
  }
  const size_t end = deferred_command_list_.stream_size_elements();
  const size_t len = end - g_tile_span_start;
  if (!len || len > kTileMaxSpanElements) {
    ++p.rr_len;
    return;
  }
  if (scan.view_sites != scan.view_offset_count) {
    ++p.rr_roots;
    return;
  }
  const size_t nkv = g_spr_cap.vs->GetTextureBindingsAfterTranslation().size();
  const size_t nkp =
      g_spr_cap.ps ? g_spr_cap.ps->GetTextureBindingsAfterTranslation().size() : 0;
  if (nkv > kTileMaxKeys || nkp > kTileMaxKeys ||
      current_texture_srv_keys_vertex_.size() < nkv ||
      current_texture_srv_keys_pixel_.size() < nkp) {
    ++p.rr_keys;
    return;
  }
  if (g_tile_arena_used + len > kTileArenaMaxElements) {
    ++p.rr_arena;
    return;
  }
  if (g_tile_arena.size() < g_tile_arena_used + len) {
    g_tile_arena.resize(std::min<size_t>(
        kTileArenaMaxElements,
        std::max<size_t>(g_tile_arena_used + len,
                         g_tile_arena.size() * 2 + (size_t(1) << 16))));
  }
  const size_t copied = deferred_command_list_.NrDspCopySpan(
      g_tile_span_start, g_tile_arena.data() + g_tile_arena_used, len);
  if (copied != len) {
    ++p.rr_len;
    return;
  }
  uint32_t roots[DeferredCommandList::kNrSprMaxViewSites];
  if (scan.view_offset_count &&
      !DeferredCommandList::NrSprViewSiteRoots(g_tile_arena.data() + g_tile_arena_used,
                                               scan.view_offsets,
                                               scan.view_offset_count, roots)) {
    ++p.rr_roots;
    return;
  }
  r.off = uint32_t(g_tile_arena_used);
  r.len = uint32_t(len);
  g_tile_arena_used += len;
  r.view_count = scan.view_offset_count;
  for (uint32_t i = 0; i < scan.view_offset_count; ++i) {
    r.view_offsets[i] = scan.view_offsets[i];
    r.view_roots[i] = uint8_t(roots[i]);
  }
  r.root_mask = scan.root_mask;
  r.vs = g_spr_cap.vs;
  r.ps = g_spr_cap.ps;
  r.tex_mask = g_spr_cap.tex_mask;
  r.llci = g_spr_cap.llci;
  r.index_endian = g_spr_cap.index_endian;
  r.ib_dma = g_spr_cap.ib_dma;
  r.ibr_base = g_spr_cap.ib_base;
  r.ibr_size = g_spr_cap.ib_size;
  r.smp_heap_ptr = sampler_bindless_heap_gpu_start_.ptr;
  r.view_heap_ptr = view_bindless_heap_gpu_start_.ptr;
  r.nkeys_v = uint8_t(nkv);
  r.nkeys_p = uint8_t(nkp);
  if (r.nkeys_v) {
    std::memcpy(r.keys_v, current_texture_srv_keys_vertex_.data(),
                r.nkeys_v * sizeof(D3D12TextureCache::TextureSRVKey));
  }
  if (r.nkeys_p) {
    std::memcpy(r.keys_p, current_texture_srv_keys_pixel_.data(),
                r.nkeys_p * sizeof(D3D12TextureCache::TextureSRVKey));
  }
  r.entry_ctx = g_tile_entry_ctx;
  NrSprCaptureCtx(&r.exit_ctx);
  // [NR-TILSYS] recorded unconditionally since N-4-2b: the consuming path
  // restores it, so it is no longer gate-only.
  r.sys_snap = g_nr_sys_state;
  r.sys_dirty = g_nr_sys_changed ? 1 : 0;
  // [NR-TILTEX] capture what this draw's texture request derived, so the
  // repeat bands can restore it instead of deriving it again.
  r.tex_count = 0;
  if (g_tile_tex && g_spr_cap.tex_mask) {
    TextureCache::NrTexSnap snaps[kTileTexMaxSlots];
    const uint32_t n =
        texture_cache_->NrTexCapture(g_spr_cap.tex_mask, snaps, kTileTexMaxSlots);
    if (n == UINT32_MAX) {
      ++p.tf_cap;
    } else if (n) {
      r.tex_off = uint32_t(g_tile_texsnaps.size());
      r.tex_count = uint8_t(n);
      g_tile_texsnaps.insert(g_tile_texsnaps.end(), snaps, snaps + n);
      p.tf_slots += n;
    }
  }
  r.valid = 1;
  ++p.recorded;
}

// [NR-TIL] N-4-1 mode 3: the compare gate. A repeat-band draw with a valid
// base-band recording runs the FULL derivation, and its freshly emitted span
// is compared against the recording it would have replayed. That turns the
// whole correctness claim -- "a repeat band draw differs from its base band
// draw only in viewport, scissor and the system-constants NDC" -- into a
// number, at menu scale and at city scale, instead of an impression.
//
// Three things are compared:
//   the pipeline piece   byte-exact,
//   the draw span        lockstep and field-aware (a root view whose only
//                        difference is its GPU address is `dyn`, which is
//                        expected: the full path allocates fresh constant
//                        buffers, and their CONTENT is identical because the
//                        same packets wrote the same registers),
//   the system constants byte-exact outside the three viewport-derived
//                        fields (ndc_scale, ndc_offset,
//                        point_screen_diameter_to_ndc_radius) -- this is the
//                        one that tests the claim directly, since the sys
//                        mirror is host memory and can be read back.
// The entry context is compared too, so the gate also reports whether the
// consuming path would have accepted this draw.
void D3D12CommandProcessor::NrTileCompareEnd() {
  if (!g_tile_cmp_open) return;
  g_tile_cmp_open = false;
  NrTilProbe& p = g_tile_p;
  if (deferred_command_list_.reset_generation() != g_tile_span_gen ||
      frame_current_ != g_tile_frame) {
    ++p.cmp_stale;
    return;
  }
  const TileRec& r = g_tile_recs[g_tile_cmp_idx];
  ++p.cmp;
  bool eq = true;
  // 1. the pipeline piece.
  {
    uintmax_t buf[8];
    const size_t la = deferred_command_list_.NrTilePipelineSpan(
        g_tile_spanA_start, g_tile_spanA_end, buf, rex::countof(buf));
    if (la == SIZE_MAX || la != r.lenA ||
        (la && std::memcmp(buf, r.dataA, la * sizeof(uintmax_t)) != 0)) {
      ++p.cmp_psone;
      eq = false;
    }
  }
  // 2. the draw span.
  {
    DeferredCommandList::NrDspDiff d;
    // The replay's head runs residency and flushes SubmitBarriers() just
    // before the span, so a barrier (`fbar`) or a shared-memory upload
    // (`fupl`) the fresh execution emitted INSIDE it is not a difference in
    // the claim -- it is counted instead of charged. Any other copy still
    // charges `real`.
    deferred_command_list_.NrDspCompareSpan(g_tile_arena.data() + r.off, r.len,
                                            g_tile_span_start, &d, true,
                                            shared_memory_->GetBuffer());
    p.cmp_dyn += d.dyn_view + d.dyn_table;
    p.cmp_fbar += d.fresh_barriers;
    p.cmp_fupl += d.fresh_uploads;
    if (d.length_differs) {
      ++p.cmp_lenne;
      eq = false;
    }
    if (d.real) {
      p.cmp_real += d.real;
      if (p.cmp_first_real == 0xFFFFFFFFu) p.cmp_first_real = d.first_real;
      eq = false;
    }
  }
  // 3. the system constants, outside the viewport-derived fields.
  {
    const uint8_t* a = reinterpret_cast<const uint8_t*>(&g_nr_sys_state);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&r.sys_snap);
    for (uint32_t i = 0; i < sizeof(nr::NrSysConstants); ++i) {
      // ndc_scale[3] at 128, ndc_offset[3] at 144,
      // point_screen_diameter_to_ndc_radius[2] at 168.
      if ((i >= 128 && i < 140) || (i >= 144 && i < 156) ||
          (i >= 168 && i < 176)) {
        continue;
      }
      if (a[i] != b[i]) {
        ++p.cmp_sysne;
        if (p.cmp_sys_off == 0xFFFFFFFFu) p.cmp_sys_off = i;
        eq = false;
        break;
      }
    }
  }
  // 4. would the consuming path have taken it?
  {
    uint32_t why = 0;
    if (!NrTileCtxEq(g_tile_entry_ctx, r.entry_ctx, &why)) ++p.cmp_ctxne;
  }
  if (eq) ++p.cmp_eq;
}

bool D3D12CommandProcessor::NrTileReplayTry() {
  NrTilProbe& p = g_tile_p;
  if (g_tile_mode != 2 || g_tile_verify) return false;
  // [NR-TILP] stage stamps, gpu_draw_profile-gated. Accumulated into locals
  // and committed only on the SUCCESS path: a refusal mid-head would
  // otherwise charge a partial draw to whichever stage it died in.
  using tilp_clock = std::chrono::steady_clock;
  const bool tp = g_tile_prof;
  tilp_clock::time_point tp0, tp_pre, tp_treq, tp_tex, tp_rt, tp_vp, tp_sys,
      tp_res;
  if (tp) tp0 = tilp_clock::now();
  const TileSeg& seg = g_tile_segs[g_tile_seg];
  if (g_tile_cur_ord + g_tile_ord_bias >= seg.count) {
    ++p.fb_ord;
    return false;
  }
  const TileRec& r = g_tile_recs[seg.first + g_tile_ord_bias + g_tile_cur_ord];
  if (!r.valid) {
    ++p.fb_norec;
    return false;
  }
  ++p.cand;
  if (r.addr != nr_tile_draw_addr_ || r.prim != g_tile_prim ||
      r.index_count != g_tile_icount || r.ib_base != g_tile_ibbase) {
    ++p.fb_shape;
    return false;
  }
  // An open instanced batch owns the stream tail; the merge-or-flush
  // decision belongs to the full path.
  if (g_instance && instanced_batch_.active) {
    ++p.fb_batch;
    return false;
  }
  if (!BeginSubmission(true)) {
    ++p.fb_begin;
    return false;
  }
  // A frame boundary inside this draw would have reset the constant-buffer
  // pool the recording GPU addresses point into: the recording is dead, and
  // the next BeginDraw clears the store.
  if (frame_current_ != g_tile_frame) {
    ++p.fb_begin;
    return false;
  }
  if (r.smp_heap_ptr != sampler_bindless_heap_gpu_start_.ptr ||
      r.view_heap_ptr != view_bindless_heap_gpu_start_.ptr) {
    ++p.fb_heap;
    return false;
  }
  // The emission-context gate, AFTER BeginSubmission: a submission open
  // resets the dedupe members, and a DEDUPED span omitted commands are only
  // sound when the current members equal the recording entry snapshot. In
  // lockstep this matches by construction (the exit context of ordinal i-1
  // is applied at the end of its replay), so a miss names a real divergence.
  SprCtx nr_til_now;
  NrSprCaptureCtx(&nr_til_now);
  uint32_t nr_til_why = 0;
  if (!NrTileCtxEq(nr_til_now, r.entry_ctx, &nr_til_why)) {
    ++p.fb_ctx;
    if (!g_tile_cur_ord) ++p.ctx_first;
    for (int i = 0; i < 12; ++i) {
      if (nr_til_why & (1u << i)) ++p.ctx_why[i];
    }
    return false;
  }
  if (tp) tp_pre = tilp_clock::now();
  const D3D12Shader* vs = r.vs;
  const D3D12Shader* ps = r.ps;
  // Textures first (a load can move host SRV indices), then the key valve:
  // the recorded span descriptor-index constant buffer holds the base band
  // indices, so a moved index means the recording is stale.
  bool tex_restored = false;
  if (g_tile_tex == 4) {
    // Control half of the cycle: recorded, deliberately not consumed.
    ++p.tf_miss;
  } else if (g_tile_tex && r.tex_count) {
    tex_restored = texture_cache_->NrTexRestore(&g_tile_texsnaps[r.tex_off], r.tex_count);
    if (tex_restored) {
      ++p.tf_hit;
      // Usage marking and the resource transitions still run: between the
      // bands a resolve or a load blit can move a texture's state.
      texture_cache_->NrTexBarriersOnly(r.tex_mask);
      if (g_tile_tex == 2) {
        // The gate: capture what the restore produced, force a re-derivation
        // from the same fetch constants, and compare. This is the claim -
        // "the two bands derive the same binding from the same fetch
        // constants" - turned into a number.
        TextureCache::NrTexSnap tf_a[kTileTexMaxSlots], tf_b[kTileTexMaxSlots];
        const uint32_t na =
            texture_cache_->NrTexCapture(r.tex_mask, tf_a, kTileTexMaxSlots);
        texture_cache_->NrTexDesync(r.tex_mask);
        texture_cache_->RequestTextures(r.tex_mask);
        const uint32_t nb =
            texture_cache_->NrTexCapture(r.tex_mask, tf_b, kTileTexMaxSlots);
        bool tf_eq = na == nb && na != UINT32_MAX;
        for (uint32_t i = 0; tf_eq && i < na; ++i) {
          tf_eq = tf_a[i].slot == tf_b[i].slot &&
                  !(tf_a[i].binding.key != tf_b[i].binding.key) &&
                  tf_a[i].binding.host_swizzle == tf_b[i].binding.host_swizzle &&
                  tf_a[i].binding.swizzled_signs == tf_b[i].binding.swizzled_signs &&
                  tf_a[i].binding.texture == tf_b[i].binding.texture &&
                  tf_a[i].binding.texture_signed == tf_b[i].binding.texture_signed &&
                  tf_a[i].descriptor_index == tf_b[i].descriptor_index &&
                  tf_a[i].descriptor_index_signed == tf_b[i].descriptor_index_signed;
        }
        if (!tf_eq) ++p.tf_ne;
      }
    } else {
      ++p.tf_miss;
    }
  } else if (g_tile_tex) {
    ++p.tf_norec;
  }
  if (!tex_restored) {
    texture_cache_->RequestTextures(r.tex_mask);
  }
  // A load blit rebinding the pipeline breaks the symmetry the recorded
  // pipeline piece was deduped under (the record side refuses the same
  // case at its own anchor).
  if (current_guest_pipeline_ != g_tile_top_guest_pso ||
      current_external_pipeline_ != g_tile_top_ext_pso) {
    ++p.fb_texpso;
    return false;
  }
  if (tp) tp_treq = tilp_clock::now();
  // [NR-TILTEX] The valve asks whether the SRV keys recorded with the base
  // band draw still describe the live bindings. Under a successful restore
  // those bindings ARE the recorded ones, so the answer is yes by
  // construction and the check is tautological - it runs only when the
  // derivation ran, and in gate mode, where a failure would be news.
  if (!tex_restored || g_tile_tex == 2) {
    const std::vector<D3D12Shader::TextureBinding>& tv =
        vs->GetTextureBindingsAfterTranslation();
    if (tv.size() != r.nkeys_v ||
        (r.nkeys_v && !texture_cache_->AreActiveTextureSRVKeysUpToDate(
                          r.keys_v, tv.data(), r.nkeys_v))) {
      ++p.fb_valve;
      return false;
    }
    if (ps) {
      const std::vector<D3D12Shader::TextureBinding>& tp =
          ps->GetTextureBindingsAfterTranslation();
      if (tp.size() != r.nkeys_p ||
          (r.nkeys_p && !texture_cache_->AreActiveTextureSRVKeysUpToDate(
                            r.keys_p, tp.data(), r.nkeys_p))) {
        ++p.fb_valve;
        return false;
      }
    } else if (r.nkeys_p) {
      ++p.fb_valve;
      return false;
    }
    if (tex_restored) ++p.tf_valve;
  }
  if (tp) tp_tex = tilp_clock::now();
  const RegisterFile& regs = GetActiveDrawRegisterFile();
  const bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  const bool is_rasterization_done =
      draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  reg::RB_DEPTHCONTROL normalized_depth_control =
      draw_util::GetNormalizedDepthControl(regs);
  const uint32_t normalized_color_mask =
      ps ? draw_util::GetNormalizedColorMask(regs, ps->writes_color_targets()) : 0;
  if (!render_target_cache_->Update(is_rasterization_done, normalized_depth_control,
                                    normalized_color_mask, *vs)) {
    ++p.fb_rt;
    return false;
  }
  if (tp) tp_rt = tilp_clock::now();
  // The bin-dependent set, recomputed from the LIVE registers: this is the
  // whole difference between the base band and this one.
  const uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  const uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();
  const bool host_render_targets_used =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
  const bool convert_z_to_float24 =
      host_render_targets_used && render_target_cache_->depth_float24_convert_in_pixel_shader();
  const bool ps_writes_depth = ps && ps->writes_depth();
  ViewportCacheKey viewport_key;
  viewport_key.pa_cl_clip_cntl = regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
  viewport_key.pa_cl_vte_cntl = regs[XE_GPU_REG_PA_CL_VTE_CNTL];
  viewport_key.pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
  viewport_key.pa_su_vtx_cntl = regs[XE_GPU_REG_PA_SU_VTX_CNTL];
  viewport_key.pa_sc_window_offset = regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
  viewport_key.normalized_depth_control = normalized_depth_control.value;
  std::memcpy(viewport_key.vport_regs, &regs[XE_GPU_REG_PA_CL_VPORT_XSCALE],
              sizeof(viewport_key.vport_regs));
  viewport_key.flags = (uint32_t(convert_z_to_float24) << 0) |
                       (uint32_t(host_render_targets_used) << 1) |
                       (uint32_t(ps_writes_depth) << 2);
  draw_util::ViewportInfo viewport_info;
  if (viewport_cache_valid_ && viewport_key == previous_viewport_key_) {
    viewport_info = previous_viewport_info_;
  } else {
    draw_util::GetHostViewportInfo(regs, draw_resolution_scale_x, draw_resolution_scale_y, true,
                                   D3D12_VIEWPORT_BOUNDS_MAX, D3D12_VIEWPORT_BOUNDS_MAX, false,
                                   normalized_depth_control, convert_z_to_float24,
                                   host_render_targets_used, ps_writes_depth, viewport_info);
    previous_viewport_key_ = viewport_key;
    previous_viewport_info_ = viewport_info;
    viewport_cache_valid_ = true;
  }
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  NrDetileWidenDrawScissor(regs, scissor);
  scissor.offset[0] *= draw_resolution_scale_x;
  scissor.offset[1] *= draw_resolution_scale_y;
  scissor.extent[0] *= draw_resolution_scale_x;
  scissor.extent[1] *= draw_resolution_scale_y;
  UpdateFixedFunctionState(viewport_info, scissor, primitive_polygonal,
                           normalized_depth_control);
  if (tp) tp_vp = tilp_clock::now();
  if (g_tile_sys) {
    // [NR-TILSYS] N-4-2b piece 1: restore the base band snapshot and patch
    // the three viewport-derived fields, instead of re-deriving 464 bytes
    // from the register file for a draw whose registers did not move.
    const bool sf_point =
        regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type == xenos::PrimitiveType::kPointList;
    if (render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock) {
      // The ROV constants are derived by the full body only; never taken in
      // this game, counted so silence is not mistaken for coverage.
      ++p.sf_rov;
      UpdateSystemConstantValues(false, primitive_polygonal, r.llci,
                                 xenos::Endian(r.index_endian), viewport_info,
                                 r.tex_mask, normalized_depth_control,
                                 normalized_color_mask);
    } else if (g_tile_sys == 2) {
      // The gate: derive fast into a temp, then run the real derivation and
      // compare every byte of it, plus the dirty verdict the two produce.
      nr::NrSysConstants sf_tmp;
      std::memcpy(&sf_tmp, &g_nr_sys_state, sizeof(sf_tmp));
      const bool sf_changed = NrTileSysFast(
          &sf_tmp, seg.first + g_tile_ord_bias + g_tile_cur_ord, r.sys_snap,
          r.sys_dirty != 0, viewport_info, sf_point, draw_resolution_scale_x,
          draw_resolution_scale_y, true);
      const bool sf_utd_before = cbuffer_binding_system_.up_to_date;
      UpdateSystemConstantValues(false, primitive_polygonal, r.llci,
                                 xenos::Endian(r.index_endian), viewport_info,
                                 r.tex_mask, normalized_depth_control,
                                 normalized_color_mask);
      ++p.sf_n;
      const uint8_t* sf_a = reinterpret_cast<const uint8_t*>(&sf_tmp);
      const uint8_t* sf_b = reinterpret_cast<const uint8_t*>(&g_nr_sys_state);
      for (uint32_t i = 0; i < sizeof(nr::NrSysConstants); ++i) {
        if (sf_a[i] != sf_b[i]) {
          ++p.sf_ne;
          if (p.sf_off == 0xFFFFFFFFu) p.sf_off = i;
          break;
        }
      }
      if (sf_utd_before &&
          sf_changed != !cbuffer_binding_system_.up_to_date) {
        ++p.sf_dne;
        if (sf_changed) {
          ++p.sf_dover;
        } else {
          ++p.sf_dunder;
        }
      }
      if (sf_changed) ++p.sf_dirty;
    } else {
      ++p.sf_n;
      if (NrTileSysFast(&g_nr_sys_state,
                        seg.first + g_tile_ord_bias + g_tile_cur_ord,
                        r.sys_snap, r.sys_dirty != 0, viewport_info, sf_point,
                        draw_resolution_scale_x, draw_resolution_scale_y,
                        true)) {
        cbuffer_binding_system_.up_to_date = false;
        ++p.sf_dirty;
      }
      // The Xenia-side member is now behind the mirror, exactly as after a
      // lean-path call; the next full body re-syncs it before reading it.
      g_nr_sys_member_stale = true;
    }
  } else {
    UpdateSystemConstantValues(false, primitive_polygonal, r.llci,
                               xenos::Endian(r.index_endian), viewport_info, r.tex_mask,
                               normalized_depth_control, normalized_color_mask);
  }
  bool sys_uploaded = false;
  if (!cbuffer_binding_system_.up_to_date) {
    uint8_t* system_constants = constant_buffer_pool_->Request(
        frame_current_, sizeof(g_nr_sys_state), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_system_.address);
    if (system_constants == nullptr) {
      ++p.fb_sys;
      return false;
    }
    std::memcpy(system_constants, &g_nr_sys_state, sizeof(g_nr_sys_state));
    cbuffer_binding_system_.up_to_date = true;
    sys_uploaded = true;
  }
  if (tp) tp_sys = tilp_clock::now();
  // Vertex residency over the recorded shader fetch bitmap; the bits are
  // already in sync from the base band unless something invalidated the
  // range, which is exactly when the request has to happen.
  const Shader::ConstantRegisterMap& constant_map_vertex = vs->constant_register_map();
  for (uint32_t i = 0; i < rex::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
    uint32_t vfetch_bits_remaining = constant_map_vertex.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(vfetch_bits_remaining, &j)) {
      vfetch_bits_remaining &= ~(uint32_t(1) << j);
      const uint32_t vfetch_index = i * 32 + j;
      const uint64_t vfetch_bit = uint64_t(1) << (vfetch_index & 63);
      if (vertex_buffers_in_sync_[vfetch_index >> 6] & vfetch_bit) {
        continue;
      }
      const xenos::xe_gpu_vertex_fetch_t vfetch_constant = regs.GetVertexFetch(vfetch_index);
      if (vfetch_constant.type != xenos::FetchConstantType::kVertex &&
          !(vfetch_constant.type == xenos::FetchConstantType::kInvalidVertex &&
            REXCVAR_GET(gpu_allow_invalid_fetch_constants))) {
        ++p.fb_vf;
        return false;
      }
      VertexBufferState& state = vertex_buffer_states_[vfetch_index];
      if (state.address == vfetch_constant.address && state.size == vfetch_constant.size) {
        vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
        continue;
      }
      if (!shared_memory_->RequestRange(vfetch_constant.address << 2,
                                        vfetch_constant.size << 2)) {
        ++p.fb_vf;
        return false;
      }
      state.address = vfetch_constant.address;
      state.size = vfetch_constant.size;
      vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
    }
  }
  if (r.ib_dma && !shared_memory_->RequestRange(r.ibr_base, r.ibr_size)) {
    ++p.fb_ib;
    return false;
  }
  shared_memory_->UseForReading();
  SubmitBarriers();
  if (tp) tp_res = tilp_clock::now();
  // Past here nothing may refuse: the span is about to enter the stream.
  //
  // The system-constants buffer moved and the recording does not re-set that
  // root (the base band found it already current): set it here, before the
  // span, or the draw would read the base band NDC.
  if (sys_uploaded &&
      !(r.root_mask & (1u << kRootParameter_Bindless_SystemConstants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_SystemConstants, cbuffer_binding_system_.address);
    ++p.sys_sets;
  }
  // The pipeline piece first (the fresh path emits it before the viewport;
  // both orders are equivalent to the command list, and this one keeps the
  // recording contiguous with the draw it belongs to).
  if (r.lenA) {
    std::memcpy(deferred_command_list_.NrSprAppendRaw(r.lenA), r.dataA,
                r.lenA * sizeof(uintmax_t));
  }
  uintmax_t* patched = deferred_command_list_.NrSprAppendRaw(r.len);
  std::memcpy(patched, g_tile_arena.data() + r.off, r.len * sizeof(uintmax_t));
  // Only the system-constants sites are patched. Every other root view in
  // the span points at a base-band pool allocation that is same-frame, still
  // live, and holds byte-identical content: the registers behind it were
  // written by the same packets.
  for (uint8_t i = 0; i < r.view_count; ++i) {
    if (r.view_roots[i] == kRootParameter_Bindless_SystemConstants) {
      DeferredCommandList::NrSprPatchViewAddress(patched, r.view_offsets[i],
                                                 cbuffer_binding_system_.address);
    }
  }
  // The stream now holds the recording tail state: adopt the recording EXIT
  // members so ordinal i+1 entry context keeps matching. The
  // system-constants ADDRESS is ours (it is excluded from the snapshot on
  // purpose); everything else is the base band, which is what the span just
  // bound.
  const SprCtx& x = r.exit_ctx;
  current_guest_pipeline_ = const_cast<void*>(x.guest_pipeline);
  current_external_pipeline_ =
      static_cast<ID3D12PipelineState*>(const_cast<void*>(x.external_pipeline));
  current_graphics_root_signature_ =
      static_cast<ID3D12RootSignature*>(const_cast<void*>(x.root_signature));
  current_graphics_root_up_to_date_ = x.ru2d;
  primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY(x.topology);
  if (x.shm_uav == 2) {
    current_shared_memory_binding_is_uav_.reset();
  } else {
    current_shared_memory_binding_is_uav_ = x.shm_uav != 0;
  }
  // Root parameter of each cbuffer_binding_ slot, in SprCtx::cb_addr order.
  static const uint32_t kTileCbRoots[7] = {
      kRootParameter_Bindless_FetchConstants,
      kRootParameter_Bindless_FloatConstantsVertex,
      kRootParameter_Bindless_FloatConstantsPixel,
      kRootParameter_Bindless_SystemConstants,
      kRootParameter_Bindless_BoolLoopConstants,
      kRootParameter_Bindless_DescriptorIndicesVertex,
      kRootParameter_Bindless_DescriptorIndicesPixel};
  ConstantBufferBinding* nr_til_cbs[7] = {
      &cbuffer_binding_fetch_,
      &cbuffer_binding_float_vertex_,
      &cbuffer_binding_float_pixel_,
      &cbuffer_binding_system_,
      &cbuffer_binding_bool_loop_,
      &cbuffer_binding_descriptor_indices_vertex_,
      &cbuffer_binding_descriptor_indices_pixel_};
  for (int i = 0; i < 7; ++i) {
    // Adopt the recorded address only for a root the SPAN actually bound --
    // that is the address now live on the command list. A root the span
    // omitted is still bound to whatever this band bound earlier, whose
    // content is identical; overwriting the member with the base band
    // address there would make the member describe a buffer the list is not
    // pointing at. The system-constants address is always ours.
    if (i != 3 && (r.root_mask & (1u << kTileCbRoots[i]))) {
      nr_til_cbs[i]->address = D3D12_GPU_VIRTUAL_ADDRESS(x.cb_addr[i]);
    }
    nr_til_cbs[i]->up_to_date = x.cb_utd[i] != 0;
  }
  // The system-constants root is bound-and-current exactly when the span set
  // it or this replay set it; otherwise the recorded mask already tells the
  // truth (the base band did not bind it either, and the previous ordinal
  // left it bound).
  if (sys_uploaded ||
      (r.root_mask & (1u << kRootParameter_Bindless_SystemConstants))) {
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SystemConstants;
  }
  if (tp) {
    const auto tp_end = tilp_clock::now();
    auto ns = [](tilp_clock::time_point a, tilp_clock::time_point b) {
      return uint64_t(
          std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };
    p.ns_pre += ns(tp0, tp_pre);
    p.ns_treq += ns(tp_pre, tp_treq);
    p.ns_tex += ns(tp_pre, tp_tex);
    p.ns_rt += ns(tp_tex, tp_rt);
    p.ns_vp += ns(tp_rt, tp_vp);
    p.ns_sys += ns(tp_vp, tp_sys);
    p.ns_res += ns(tp_sys, tp_res);
    p.ns_emit += ns(tp_res, tp_end);
    ++p.prof_rep;
  }
  ++p.rep;
  p.rep_elements += r.len;
  // [NR-TILP] commit the outer stamps and mark the return moment so the
  // drawstop bracket exit (base TU) can close opost.
  if (g_tile_pend_valid) {
    p.ns_ostop += g_tile_pend_ostop;
    p.ns_obegin += g_tile_pend_obegin;
    ++p.prof_orep;
    g_tile_pend_valid = false;
    g_tile_orep = true;
    g_tile_orep_t1 = std::chrono::steady_clock::now();
  }
  return true;
}

void D3D12CommandProcessor::NrBfcBufBegin() {
  // [NR-BFC] Phase 5-4-6-0: latch the span anchor. The submission id guards
  // against a deferred-list Reset between Begin and End (the stream dies at
  // submission boundaries -- exactly the zero-copy validity question the
  // census exists to price).
  nr_bfc_span_start_ = deferred_command_list_.stream_size_elements();
  nr_bfc_rt_runs_start_ =
      render_target_cache_ ? render_target_cache_->nr_update_body_runs() : 0;
  nr_bfc_gen_start_ = deferred_command_list_.reset_generation();
}

bool D3D12CommandProcessor::NrBfcBufEnd(NrBfcBackendSample* out) {
  // [NR-BFC] A deferred-list Reset inside the buffer means the span anchor
  // points into a refilled stream (possibly mid-command) -- skip the scan
  // rather than walking garbage. The submission id is NOT this check (it
  // increments at EndSubmission's Signal, the Reset happens at the next
  // BeginSubmission -- an anchor latched while no submission was open dies
  // under an unchanged id; first smoke AV'd exactly there). The id still
  // rides the sample for the between-replay crossing metric.
  out->submission_id = submission_current_;
  out->rt_body_runs =
      (render_target_cache_ ? render_target_cache_->nr_update_body_runs()
                            : 0) -
      nr_bfc_rt_runs_start_;
  if (deferred_command_list_.reset_generation() != nr_bfc_gen_start_) {
    out->span_elements = 0;
    return true;
  }
  const size_t end_elements = deferred_command_list_.stream_size_elements();
  out->span_elements = uint32_t(end_elements - nr_bfc_span_start_);
  DeferredCommandList::NrBfcSpanCounts c;
  deferred_command_list_.NrBfcScan(nr_bfc_span_start_, &c);
  out->cmd_draw = c.draw;
  out->cmd_pso = c.pso;
  const uint32_t sys_root = uint32_t(kRootParameter_Bindless_SystemConstants);
  out->cmd_sys_cbv = sys_root < 16 ? c.graphics_cbv_by_root[sys_root] : 0;
  out->cmd_root_cbv = c.root_cbv - out->cmd_sys_cbv;
  out->cmd_root_other = c.root_other;
  out->cmd_ia = c.ia;
  out->cmd_vp = c.vp;
  out->cmd_sci = c.sci;
  out->cmd_om_rt = c.om_rt;
  out->cmd_om_misc = c.om_misc;
  out->cmd_barrier = c.barrier;
  out->cmd_copy = c.copy;
  out->cmd_clear = c.clear;
  out->cmd_dispatch = c.dispatch;
  out->cmd_query = c.query;
  out->cmd_marker = c.marker;
  out->cmd_heaps = c.heaps;
  out->cmd_other = c.other;
  return true;
}

void D3D12CommandProcessor::PrecordApplyWrite(RegisterFile* file, uint32_t index, uint32_t value) {
  // [GPU-PRECORD] Phase 1b-1c Inc 1: the replay-time equivalent of a single
  // WriteRegister for a DEFERRABLE register, applied against `file` (the per-segment
  // local register file) instead of the shared register_file_ member. Mirrors the
  // deferrable subset of D3D12CommandProcessor::WriteRegister + its base: the constant
  // dedupe, the value store, and the D3D12 cbuffer/texture/vertex-residency
  // invalidation. The stateful registers (scratch/COHER/DC_LUT) are flush-gated
  // (PrecordRangeMustNotDefer) so they never reach replay -- intentionally omitted.
  // During replay g_instance/g_parallel_record are forced off, so their branches in
  // WriteRegister are dead here too. Keep in sync with WriteRegister (this override +
  // the CommandProcessor base).

  // [PERF/CONST-DEDUPE] mirror WriteRegister's constant dedupe, against `file`.
  if (g_n8b_mode >= 2 &&
      ((index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
        index <= XE_GPU_REG_SHADER_CONSTANT_511_W) ||
       (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
        index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31)) &&
      file->values[index] == value) {
    return;
  }

  // Value store (base CommandProcessor::WriteRegister, deferrable subset). Extended
  // (out-of-bounds) registers live in the shared side map, not the file -- delegate to
  // the base so the exact insert_or_assign + warn-once behavior is reproduced. (This
  // is a shared-member touch; extended registers are near-absent in this title and a
  // known Phase 1c overlap edge -- see the design doc's open investigations.)
  if (index >= RegisterFile::kRegisterCount) {
    CommandProcessor::WriteRegister(index, value);
    return;
  }
  file->values[index] = value;

  // D3D12 cbuffer / texture / vertex-residency invalidation (WriteRegister tail).
  if (index >= XE_GPU_REG_SHADER_CONSTANT_000_X && index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    if (frame_open_) {
      uint32_t float_constant_index = (index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (float_constant_index >= 256) {
        float_constant_index -= 256;
        if (current_float_constant_map_pixel_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      } else {
        if (current_float_constant_map_vertex_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
    }
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    cbuffer_binding_bool_loop_.up_to_date = false;
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    cbuffer_binding_fetch_.up_to_date = false;
    if (texture_cache_ != nullptr) {
      texture_cache_->TextureFetchConstantWritten((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) /
                                                  6);
    }
    InvalidateVertexBufferResidency((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 2);
  }
}

void D3D12CommandProcessor::PrecordApplyWriteFromMem(RegisterFile* file, uint32_t start_index,
                                                     uint32_t* base, uint32_t num_registers) {
  // [GPU-PRECORD] Phase 1b-1c Inc 1: replay-time equivalent of
  // D3D12CommandProcessor::WriteRegistersFromMem, applying the bulk write against
  // `file` instead of the shared register_file_ member. Mirrors the deferrable
  // constant fast-paths + the generic per-register fallback; stateful ranges are
  // flush-gated and never reach replay. Keep in sync with WriteRegistersFromMem.
  if (!num_registers) {
    return;
  }
  uint32_t end_index = start_index + num_registers - 1;

  // Duplicated from WriteRegistersFromMem's local lambda (kept local to leave the hot
  // parse path byte-for-byte untouched). Keep in sync.
  auto range_has_any_constant_usage = [](const uint64_t* usage_map, uint32_t first_constant,
                                         uint32_t last_constant) -> bool {
    if (first_constant > last_constant) {
      return false;
    }
    uint32_t first_word = first_constant >> 6;
    uint32_t last_word = last_constant >> 6;
    uint32_t first_bit = first_constant & 63;
    uint32_t last_bit = last_constant & 63;
    if (first_word == last_word) {
      uint32_t bit_count = last_bit - first_bit + 1;
      uint64_t mask = bit_count == 64 ? UINT64_MAX : ((UINT64_C(1) << bit_count) - 1) << first_bit;
      return (usage_map[first_word] & mask) != 0;
    }
    if (usage_map[first_word] & (UINT64_MAX << first_bit)) {
      return true;
    }
    for (uint32_t word = first_word + 1; word < last_word; ++word) {
      if (usage_map[word]) {
        return true;
      }
    }
    uint64_t last_mask = last_bit == 63 ? UINT64_MAX : ((UINT64_C(1) << (last_bit + 1)) - 1);
    return (usage_map[last_word] & last_mask) != 0;
  };

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    if (g_n8b_mode >= 2) {
      const uint32_t* cur = file->values + start_index;
      bool changed = false;
      for (uint32_t i = 0; i < num_registers; ++i) {
        if (memory::load_and_swap<uint32_t>(base + i) != cur[i]) {
          changed = true;
          break;
        }
      }
      if (!changed) {
        return;
      }
    }
    memory::copy_and_swap(file->values + start_index, base, num_registers);
    if (frame_open_) {
      uint32_t first_float_constant = (start_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      uint32_t last_float_constant = (end_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (first_float_constant < 256) {
        uint32_t last_vertex_constant = std::min(last_float_constant, 255u);
        if (range_has_any_constant_usage(current_float_constant_map_vertex_, first_float_constant,
                                         last_vertex_constant)) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
      if (last_float_constant >= 256) {
        uint32_t first_pixel_constant =
            first_float_constant >= 256 ? first_float_constant - 256 : 0;
        uint32_t last_pixel_constant = last_float_constant - 256;
        if (range_has_any_constant_usage(current_float_constant_map_pixel_, first_pixel_constant,
                                         last_pixel_constant)) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      }
    }
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    if (g_n8b_mode >= 2) {
      const uint32_t* cur = file->values + start_index;
      bool changed = false;
      for (uint32_t i = 0; i < num_registers; ++i) {
        if (memory::load_and_swap<uint32_t>(base + i) != cur[i]) {
          changed = true;
          break;
        }
      }
      if (!changed) {
        return;
      }
    }
    memory::copy_and_swap(file->values + start_index, base, num_registers);
    cbuffer_binding_bool_loop_.up_to_date = false;
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    memory::copy_and_swap(file->values + start_index, base, num_registers);
    cbuffer_binding_fetch_.up_to_date = false;
    uint32_t first_fetch_dword = start_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
    uint32_t last_fetch_dword = end_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
    if (texture_cache_) {
      texture_cache_->TextureFetchConstantsWritten(first_fetch_dword / 6, last_fetch_dword / 6);
    }
    InvalidateVertexBufferResidencyRange(first_fetch_dword / 2, last_fetch_dword / 2);
    return;
  }

  // Generic fallback. WriteRegistersFromMem defers to CommandProcessor::
  // WriteRegistersFromMem, which loops the virtual WriteRegister per register; during
  // replay that reduces to the deferrable subset -> PrecordApplyWrite per register
  // (which also reproduces the per-register invalidation for any constant register in
  // a mixed range).
  for (uint32_t i = 0; i < num_registers; ++i) {
    PrecordApplyWrite(file, start_index + i, memory::load_and_swap<uint32_t>(base + i));
  }
}

void D3D12CommandProcessor::PrecordReplayLocal() {
  // [GPU-PRECORD] Phase 1b-1b: replay the captured segment against a PRIVATE local
  // register file built from the segment snapshot, with every draw-path holder
  // repointed to it for the duration. This exercises the 1b-1a decoupling: the
  // draws read the segment's own state from a file other than the shared
  // parse-thread file. Equivalent to the shared-rewind path because nothing but the
  // draw path reads the register file while we replay (the parse thread is the
  // caller here for localrf, or blocked waiting on us for thread mode).
  // [GPU-PRECORD] Phase 1b-1c Inc 3: replay the REPLAY slot (handed off by PrecordFlush),
  // building its own local register file from that slot's snapshot.
  // [GPU-PRECORD] Phase 1b-1c Inc 5 (H4): assert we are the ONLY replayer -- the shared
  // subsystems repointed below stay lock-free only because capture is pure (delta 2) and
  // there is exactly one worker. A second concurrent replayer (Phase 2) trips this. The
  // fetch_add must run unconditionally (assert_true compiles out in release), so the
  // check reads its result separately.
  int precord_prev_replays = precord_replays_in_flight_.fetch_add(1, std::memory_order_acq_rel);
  assert_true(precord_prev_replays == 0);
  (void)precord_prev_replays;
  PrecordSegment& seg = precord_slots_[precord_replay_slot_];
  if (!seg.local_regfile) {
    seg.local_regfile = std::make_unique<RegisterFile>();
  }
  RegisterFile* local = seg.local_regfile.get();
  std::memcpy(local->values, seg.snapshot.data(),
              RegisterFile::kRegisterCount * sizeof(uint32_t));

  // Repoint the draw-path READ holders to the local file (the 1b-1a decoupling).
  // RenderTargetCache cascades to its DrawExtentEstimator -> ShaderInterpreter.
  // [GPU-PRECORD] Phase 1b-1c Inc 1: register_file_ (the base member) is deliberately
  // NOT repointed anymore. Replay's register WRITES go to `local` via PrecordApplyWrite*
  // (PrecordReplayEvents(local)); the shared file the parse thread owns is never mutated
  // by replay -- the write-side analogue of the read decoupling above, and the invariant
  // Phase 1c overlap depends on. IssueDrawImpl reads `local` via active_draw_register_file_
  // (GetActiveDrawRegisterFile no longer relies on the register_file_ fallback here).
  RegisterFile* shared = register_file_;
  active_draw_register_file_ = local;
  primitive_processor_->SetRegisterFile(local);
  render_target_cache_->SetRegisterFile(local);
  texture_cache_->SetRegisterFile(local);
  pipeline_cache_->SetRegisterFile(local);

  ForceFullDrawStateReemit();

  bool saved_instance = g_instance;
  bool saved_parallel = g_parallel_record;
  g_instance = false;
  g_parallel_record = false;
  precord_replaying_ = true;
  precord_replay_shaders_active_ = true;  // [1b-1c Inc 2] accessors read the replay fields

  PrecordReplayEvents(local);

  precord_replaying_ = false;
  precord_replay_shaders_active_ = false;
  g_instance = saved_instance;
  g_parallel_record = saved_parallel;

  // Restore the READ holders to the shared file. The shared file was never touched by
  // replay, so it still holds the segment-end state (the parse thread advanced it during
  // capture) -- consistent with the shared-rewind path's post-replay state.
  active_draw_register_file_ = nullptr;
  primitive_processor_->SetRegisterFile(shared);
  render_target_cache_->SetRegisterFile(shared);
  texture_cache_->SetRegisterFile(shared);
  pipeline_cache_->SetRegisterFile(shared);

  PrecordResetSlot(precord_replay_slot_);
  ForceFullDrawStateReemit();
  precord_replays_in_flight_.fetch_sub(1, std::memory_order_acq_rel);  // [Inc 5 H4]
}

void D3D12CommandProcessor::PrecordWorkerEnsureStarted() {
  if (precord_worker_started_) {
    return;
  }
  precord_worker_shutdown_ = false;
  precord_worker_job_pending_ = false;
  precord_worker_thread_ = std::thread([this] { PrecordWorkerMain(); });
  precord_worker_started_ = true;
}

void D3D12CommandProcessor::PrecordWorkerMain() {
  std::unique_lock<std::mutex> lock(precord_worker_mutex_);
  for (;;) {
    precord_worker_cv_.wait(
        lock, [this] { return precord_worker_job_pending_ || precord_worker_shutdown_; });
    if (precord_worker_shutdown_) {
      break;
    }
    // Run the replay with the worker mutex released so the parse thread can post the next
    // job / wait for us. What the worker owns while replaying:
    //   - Model C (overlap off): the parse thread is blocked in PrecordFlush, so ALL
    //     precord_* buffers + the shared D3D12 subsystems are ours exclusively.
    //   - Overlap (Inc 5): the parse thread runs concurrently but only captures into the
    //     OTHER slot (pure capture -- delta 2), so the replay slot we drain and the
    //     subsystems we repoint (H4 tripwire) are still ours exclusively; backpressure in
    //     PrecordFlush keeps the parse thread from reusing our slot until we finish.
    lock.unlock();
    PrecordReplayLocal();
    lock.lock();
    precord_worker_job_pending_ = false;
    precord_worker_cv_.notify_all();
  }
}

void D3D12CommandProcessor::PrecordWorkerShutdown() {
  if (!precord_worker_started_) {
    return;
  }
  {
    std::unique_lock<std::mutex> lock(precord_worker_mutex_);
    precord_worker_shutdown_ = true;
    precord_worker_cv_.notify_all();
  }
  if (precord_worker_thread_.joinable()) {
    precord_worker_thread_.join();
  }
  precord_worker_started_ = false;
}

void D3D12CommandProcessor::PrecordWaitWorkerIdle() {
  // [GPU-PRECORD] Phase 1b-1c Inc 5: block until the worker has finished any posted
  // segment. No-op if the worker was never started. Parse-thread-only.
  if (!precord_worker_started_) {
    return;
  }
  std::unique_lock<std::mutex> lock(precord_worker_mutex_);
  precord_worker_cv_.wait(lock, [this] { return !precord_worker_job_pending_; });
}

void D3D12CommandProcessor::PrecordFlush(bool from_segment_boundary) {
  // Reentrancy guard (thread_local): a replayed copy-mode op can re-enter via a flush on
  // the SAME replay thread. The parse thread's copy is always false in thread/overlap
  // mode, so a genuine parse-side flush proceeds even while the worker is replaying.
  if (precord_replaying_) {
    return;
  }

  // [GPU-PRECORD] Phase 1b-1c Inc 5: overlap only bites in worker mode; without it every
  // path below drains and this stays exact Model C (the pixel-identical A/B reference).
  const bool overlap = g_precord_overlap && g_precord_thread;

  // Under overlap, make the worker idle before touching the slot machinery. At a TRUE
  // flush point this DRAINS the previously-posted segment (ordering + visibility) before
  // the caller's swap/copy/present/marker runs; at a SEGMENT BOUNDARY it is BACKPRESSURE
  // -- the slot we are about to reuse (the current replay slot) must be free before we
  // swap into it, which bounds the parse lead to 1 segment (2 slots). In every non-overlap
  // mode the worker is already idle here (the previous flush waited), so this is a no-op.
  if (overlap) {
    PrecordWaitWorkerIdle();
  }

  if (!precord_segment_open_) {
    // Nothing captured; keep the capture slot clean and bail (no handoff).
    PrecordResetSlot(precord_capture_slot_);
    precord_draws_in_segment_ = 0;
    return;
  }

  // [GPU-PRECORD] Phase 1b-1c Inc 3: hand the just-filled capture slot to the replayer
  // as the replay slot, and open a fresh capture slot for what follows. In Model C the
  // replay below runs to completion before this function returns (inline, or the worker
  // is waited on), so the "other" slot is always already drained -> byte-identical with
  // the old single buffer. Under overlap (Inc 5) the parse thread keeps capturing into
  // the new slot while the worker drains this one. The write to precord_replay_slot_
  // precedes the worker-mutex acquire below, so it is visible to the worker.
  precord_replay_slot_ = precord_capture_slot_;
  precord_capture_slot_ ^= 1u;
  precord_segment_open_ = false;
  precord_draws_in_segment_ = 0;

  // [GPU-PRECORD] Phase 1b-1b: thread implies localrf. When localrf is selected,
  // replay against a private local register file (optionally on the worker);
  // otherwise take the 1b-0 shared-rewind path below.
  if (g_precord_localrf || g_precord_thread) {
    if (g_precord_thread) {
      // Hand the segment to the worker. Under overlap AT A SEGMENT BOUNDARY, post and
      // return immediately -- the worker replays this slot while the parse thread keeps
      // capturing the other one (the first parse/worker overlap). Otherwise (Model C, or
      // an overlap TRUE flush point) BLOCK until the worker has replayed it (drain).
      PrecordWorkerEnsureStarted();
      {
        std::unique_lock<std::mutex> lock(precord_worker_mutex_);
        precord_worker_job_pending_ = true;
        precord_worker_cv_.notify_all();
      }
      if (!(overlap && from_segment_boundary)) {
        PrecordWaitWorkerIdle();
      } else {
        // [GPU-PRECORD] Phase 1b-1c Inc 6 (H3 fix, Option A): we posted the segment and are
        // returning WITHOUT draining, so the parse thread is about to run the NEXT segment's
        // setup register writes. EAGER-OPEN that segment now (on the freshly-swapped capture
        // slot) so those writes DEFER -- skipping the parse-side D3D12 draw-state side
        // effects (cbuffer/texture/vertex-residency invalidations) that would otherwise RACE
        // the worker's in-flight replay on the shared draw-state members and cause the H3
        // per-mesh flicker. The new capture slot is free: the wait-idle at the TOP of this
        // flush drained the PREVIOUS posted segment, whose PrecordReplayLocal resets exactly
        // this slot (PrecordResetSlot) before finishing. With the segment always open under
        // overlap, there is no window where a parse write runs side effects live.
        PrecordOpenSegment();
      }
    } else {
      PrecordReplayLocal();
    }
    return;
  }

  // [GPU-PRECORD] Phase 1b-0 shared-rewind path. Rewind the live register file to
  // the replay slot's snapshot (taken at the segment's first draw), and reset the
  // command-list cache so the segment re-emits all state (it is being recorded as if
  // into a fresh list). All draw-path readers read this one register file, so rewinding
  // it is sufficient for single-thread replay.
  std::memcpy(register_file_->values, precord_slots_[precord_replay_slot_].snapshot.data(),
              RegisterFile::kRegisterCount * sizeof(uint32_t));
  ForceFullDrawStateReemit();

  // Replay must be deterministic and not branch on the other draw-path probes;
  // force them off for the duration (they never affect which writes apply).
  bool saved_instance = g_instance;
  bool saved_parallel = g_parallel_record;
  g_instance = false;
  g_parallel_record = false;
  precord_replaying_ = true;
  precord_replay_shaders_active_ = true;  // [1b-1c Inc 2] accessors read the replay fields

  // nullptr ⇒ shared-rewind mode: writes go through WriteRegister against the file we
  // just rewound above (this is the 1b-0 reference path, deliberately unchanged).
  PrecordReplayEvents(nullptr);

  precord_replaying_ = false;
  precord_replay_shaders_active_ = false;
  g_instance = saved_instance;
  g_parallel_record = saved_parallel;

  // The register file now again holds the segment-end state (replay re-applied
  // every logged write). Reset the replay slot and the draw caches so the following
  // op / next segment records cleanly.
  PrecordResetSlot(precord_replay_slot_);
  ForceFullDrawStateReemit();
}

bool D3D12CommandProcessor::BeginSubmission(bool is_guest_command) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  if (device_removed_) {
    return false;
  }

  bool is_opening_frame = is_guest_command && !frame_open_;
  if (submission_open_ && !is_opening_frame) {
    return true;
  }

  // Check if the device is still available.
  ID3D12Device* device = GetD3D12Provider().GetDevice();
  HRESULT device_removed_reason = device->GetDeviceRemovedReason();
  if (FAILED(device_removed_reason)) {
    device_removed_ = true;
    LogDeviceRemovalDiagnostics(device, device_removed_reason);
    if (graphics_system_) {
      graphics_system_->OnHostGpuLossFromAnyThread(device_removed_reason !=
                                                   DXGI_ERROR_DEVICE_REMOVED);
    }
    return false;
  }

  // Check the fence - needed for all kinds of submissions (to reclaim transient
  // resources early) and specifically for frames (not to queue too many), and
  // await the availability of the current frame.
  CheckSubmissionFence(is_opening_frame ? closed_frame_submissions_[frame_current_ % kQueueFrames]
                                        : 0);
  // TODO(Triang3l): If failed to await (completed submission < awaited frame
  // submission), do something like dropping the draw command that wanted to
  // open the frame.
  if (is_opening_frame) {
    // Update the completed frame index, also obtaining the actual completed
    // frame number (since the CPU may be actually less than 3 frames behind)
    // before reclaiming resources tracked with the frame number.
    frame_completed_ = std::max(frame_current_, uint64_t(kQueueFrames)) - kQueueFrames;
    for (uint64_t frame = frame_completed_ + 1; frame < frame_current_; ++frame) {
      if (closed_frame_submissions_[frame % kQueueFrames] > submission_completed_) {
        break;
      }
      frame_completed_ = frame;
    }
  }

  if (!submission_open_) {
    submission_open_ = true;

    // Start a new deferred command list - will submit it to the real one in the
    // end of the submission (when async pipeline creation requests are
    // fulfilled).
    deferred_command_list_.Reset();
    // [gpu-census] harvest whatever the fence check above completed, then
    // claim this submission's timestamp region (ts0 lands first in the list).
    if (gpu_census_available_) {
      GpuCensusConsumeCompleted();
      GpuCensusBeginSubmission();
    }
    // [occ] same fences: harvest completed draws, claim this submission's slots.
    if (occ_available_) {
      OccConsumeCompleted();
      OccBeginSubmission();
    }
    HizBeginSubmission();  // [hiz] no window may span a submission
    // [GPU-PRECORD] Phase 1a: count draws per submission (segment boundaries);
    // clear any segment streams (defensive — EndSubmission already drains them).
    parallel_record_counter_ = 0;
    precord_segments_.clear();

    // Reset cached state of the command list.
    ff_viewport_update_needed_ = true;
    ff_scissor_update_needed_ = true;
    ff_blend_factor_update_needed_ = true;
    ff_stencil_ref_update_needed_ = true;
    viewport_cache_valid_ = false;
    current_guest_pipeline_ = nullptr;
    current_external_pipeline_ = nullptr;
    current_graphics_root_signature_ = nullptr;
    current_graphics_root_up_to_date_ = 0;
    if (bindless_resources_used_) {
      deferred_command_list_.SetDescriptorHeaps(view_bindless_heap_,
                                                sampler_bindless_heap_current_);
    } else {
      view_bindful_heap_current_ = nullptr;
      sampler_bindful_heap_current_ = nullptr;
    }
    primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;

    render_target_cache_->BeginSubmission();

    primitive_processor_->BeginSubmission();

    texture_cache_->BeginSubmission(submission_current_);
  }

  if (is_opening_frame) {
    frame_open_ = true;

    // Reset bindings that depend on the data stored in the pools.
    std::memset(current_float_constant_map_vertex_, 0, sizeof(current_float_constant_map_vertex_));
    std::memset(current_float_constant_map_pixel_, 0, sizeof(current_float_constant_map_pixel_));
    cbuffer_binding_system_.up_to_date = false;
    cbuffer_binding_float_vertex_.up_to_date = false;
    cbuffer_binding_float_pixel_.up_to_date = false;
    cbuffer_binding_bool_loop_.up_to_date = false;
    cbuffer_binding_fetch_.up_to_date = false;
    current_shared_memory_binding_is_uav_.reset();
    if (bindless_resources_used_) {
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
      cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
    } else {
      draw_view_bindful_heap_index_ = ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
      draw_sampler_bindful_heap_index_ = ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
      bindful_textures_written_vertex_ = false;
      bindful_textures_written_pixel_ = false;
      bindful_samplers_written_vertex_ = false;
      bindful_samplers_written_pixel_ = false;
    }

    // Reclaim pool pages - no need to do this every small submission since some
    // may be reused.
    constant_buffer_pool_->Reclaim(frame_completed_);
    if (!bindless_resources_used_) {
      view_bindful_heap_pool_->Reclaim(frame_completed_);
      sampler_bindful_heap_pool_->Reclaim(frame_completed_);
    }
    EvictOldReadbackBuffers(readback_buffers_);
    EvictOldReadbackBuffers(memexport_readback_buffers_);

    pix_capturing_ = pix_capture_requested_.exchange(false, std::memory_order_relaxed);
    if (pix_capturing_) {
      IDXGraphicsAnalysis* graphics_analysis = GetD3D12Provider().GetGraphicsAnalysis();
      if (graphics_analysis != nullptr) {
        graphics_analysis->BeginCapture();
      }
    }

    primitive_processor_->BeginFrame();

    texture_cache_->BeginFrame();
  }

  return true;
}

bool D3D12CommandProcessor::EndSubmission(bool is_swap) {
  // [GPU-PRECORD] Phase 1b-0: record any pending captured draws before the
  // submission closes, otherwise they would be lost.
  PrecordFlush();
  // [GPU-INST] A deferred instanced draw must be emitted into this submission
  // before it closes, otherwise its recorded setup is orphaned and the draw is
  // lost. Flush while the submission is still open.
  if (instanced_batch_.active) {
    ++g_inst_flush_site[2];
  }
  FlushInstancedBatch();

  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();

  // Make sure there is a command allocator to write commands to.
  if (submission_open_ && !command_allocator_writable_first_) {
    ID3D12CommandAllocator* command_allocator;
    if (FAILED(provider.GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&command_allocator)))) {
      REXGPU_ERROR("Failed to create a command allocator");
      // Try to submit later. Completely dropping the submission is not
      // permitted because resources would be left in an undefined state.
      return false;
    }
    command_allocator_writable_first_ = new CommandAllocator;
    command_allocator_writable_first_->command_allocator = command_allocator;
    command_allocator_writable_first_->last_usage_submission = 0;
    command_allocator_writable_first_->next = nullptr;
    command_allocator_writable_last_ = command_allocator_writable_first_;
  }

  bool is_closing_frame = is_swap && frame_open_;

  if (is_closing_frame) {
    texture_cache_->EndFrame();

    primitive_processor_->EndFrame();
  }

  if (submission_open_) {
    assert_false(scratch_buffer_used_);

    if (active_occlusion_query_.valid && occlusion_query_heap_) {
      deferred_command_list_.D3DEndQuery(occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION,
                                         active_occlusion_query_.host_index);
      active_occlusion_query_ = {};
    }

    // [gpu-census] final stamp closes the last span; must precede the
    // deferred list's Execute below.
    GpuCensusCloseSubmission();

    pipeline_cache_->EndSubmission();

    // Submit barriers now because resources with the queued barriers may be
    // destroyed between frames.
    SubmitBarriers();

    ID3D12CommandQueue* direct_queue = provider.GetDirectQueue();

    // Submit the deferred command list.
    // Only one deferred command list must be executed in the same
    // ExecuteCommandLists - the boundaries of ExecuteCommandLists are a full
    // UAV and aliasing barrier, and subsystems of the emulator assume it
    // happens between Xenia submissions.
    ID3D12CommandAllocator* command_allocator =
        command_allocator_writable_first_->command_allocator;
    command_allocator->Reset();
    command_list_->Reset(command_allocator, nullptr);
    // [GPU-PRECORD] Phase 1a-ii: replay completed segment streams in submission
    // order first, then the final (still-current) deferred list. Concatenated they
    // are the exact command sequence the single-list path would have produced.
    // Empty (no-op) when gpu_parallel_record is off.
    for (auto& segment_stream : precord_segments_) {
      deferred_command_list_.ExecuteStream(segment_stream, command_list_, command_list_1_);
    }
    precord_segments_.clear();
    // [DRAW-POOL] the GPU reads the argument buffers from here on: no batch
    // recorded in this list may take another instance.
    if (g_pool_on) PoolClose(kPoolCloseSubmit);
    deferred_command_list_.Execute(command_list_, command_list_1_);
    // [gpu-census] the timestamps are replayed into the real list now, so
    // the resolve can be recorded directly on it.
    GpuCensusResolveSubmission();
    OccResolveSubmission();  // [occ] both heaps -> readback, on the real list
    HizResolveSubmission();  // [hiz] verdict slots -> readback, on the real list
    command_list_->Close();
    ID3D12CommandList* execute_command_lists[] = {command_list_};
    direct_queue->ExecuteCommandLists(1, execute_command_lists);
    command_allocator_writable_first_->last_usage_submission = submission_current_;
    if (command_allocator_submitted_last_) {
      command_allocator_submitted_last_->next = command_allocator_writable_first_;
    } else {
      command_allocator_submitted_first_ = command_allocator_writable_first_;
    }
    command_allocator_submitted_last_ = command_allocator_writable_first_;
    command_allocator_writable_first_ = command_allocator_writable_first_->next;
    command_allocator_submitted_last_->next = nullptr;
    if (!command_allocator_writable_first_) {
      command_allocator_writable_last_ = nullptr;
    }

    direct_queue->Signal(submission_fence_, submission_current_++);

    submission_open_ = false;

    // Queue operations done directly (like UpdateTileMappings) will be awaited
    // alongside the last submission if needed.
    queue_operations_done_since_submission_signal_ = false;
  }

  if (is_closing_frame) {
    if (REXCVAR_GET(clear_memory_page_state) && shared_memory_) {
      shared_memory_->SetSystemPageBlocksValidWithGpuDataWritten();
    }
    // Close the capture after submitting.
    if (pix_capturing_) {
      IDXGraphicsAnalysis* graphics_analysis = provider.GetGraphicsAnalysis();
      if (graphics_analysis != nullptr) {
        graphics_analysis->EndCapture();
      }
      pix_capturing_ = false;
    }
    frame_open_ = false;
    // Submission already closed now, so minus 1.
    closed_frame_submissions_[(frame_current_++) % kQueueFrames] = submission_current_ - 1;

    // [hitch] 1 Hz: everything that blocked the CP thread this window, plus
    // PSO creation split by thread and driver-library outcomes. Silent when
    // the window had nothing to report.
    {
      uint64_t hitch_now = rex::chrono::Clock::QueryHostTickCount();
      uint64_t hitch_freq = rex::chrono::Clock::QueryHostTickFrequency();
      static uint64_t hitch_last_report = 0;
      if (!hitch_last_report) {
        hitch_last_report = hitch_now;
      }
      if (hitch_now - hitch_last_report >= hitch_freq) {
        hitch_last_report = hitch_now;
        HitchProbe& hp = g_hitch_probe;
        uint64_t fence_t = hp.fence_wait_ticks.exchange(0, std::memory_order_relaxed);
        uint64_t fence_n = hp.fence_waits.exchange(0, std::memory_order_relaxed);
        uint64_t qops_t = hp.qops_wait_ticks.exchange(0, std::memory_order_relaxed);
        uint64_t qops_n = hp.qops_waits.exchange(0, std::memory_order_relaxed);
        uint64_t endsub_t = hp.pso_endsub_wait_ticks.exchange(0, std::memory_order_relaxed);
        uint64_t endsub_n = hp.pso_endsub_waits.exchange(0, std::memory_order_relaxed);
        uint64_t cp_t = hp.pso_cp_ticks.exchange(0, std::memory_order_relaxed);
        uint64_t cp_n = hp.pso_cp_creates.exchange(0, std::memory_order_relaxed);
        uint64_t cp_max = hp.pso_cp_max_ticks.exchange(0, std::memory_order_relaxed);
        uint64_t thr_t = hp.pso_thr_ticks.exchange(0, std::memory_order_relaxed);
        uint64_t thr_n = hp.pso_thr_creates.exchange(0, std::memory_order_relaxed);
        uint64_t lib_hit = hp.lib_hits.exchange(0, std::memory_order_relaxed);
        uint64_t lib_miss = hp.lib_misses.exchange(0, std::memory_order_relaxed);
        uint64_t lib_store = hp.lib_stores.exchange(0, std::memory_order_relaxed);
        uint64_t lib_stale = hp.lib_store_fails.exchange(0, std::memory_order_relaxed);
        if (fence_n | qops_n | endsub_n | cp_n | thr_n | lib_hit | lib_miss | lib_store |
            lib_stale) {
          double to_ms = 1000.0 / double(hitch_freq);
          REXGPU_INFO(
              "[hitch] fence {:.1f}ms/{} qops {:.1f}ms/{} psowait {:.1f}ms/{} | "
              "pso cp {:.1f}ms/{} max {:.1f}ms thr {:.1f}ms/{} | lib hit {} miss {} store {} "
              "stale {}",
              double(fence_t) * to_ms, fence_n, double(qops_t) * to_ms, qops_n,
              double(endsub_t) * to_ms, endsub_n, double(cp_t) * to_ms, cp_n,
              double(cp_max) * to_ms, double(thr_t) * to_ms, thr_n, lib_hit, lib_miss, lib_store,
              lib_stale);
        }
        // [gpu-census] same 1 Hz window: what the GPU spent the second on.
        GpuCensusReport1Hz();
        OccReport1Hz();  // [occ]
      }
    }

    if (cache_clear_requested_ && AwaitAllQueueOperationsCompletion()) {
      cache_clear_requested_ = false;

      ClearCommandAllocatorCache();

      ui::d3d12::util::ReleaseAndNull(scratch_buffer_);
      scratch_buffer_size_ = 0;

      if (bindless_resources_used_) {
        texture_cache_bindless_sampler_map_.clear();
        for (const auto& sampler_bindless_heap_overflowed : sampler_bindless_heaps_overflowed_) {
          sampler_bindless_heap_overflowed.first->Release();
        }
        sampler_bindless_heaps_overflowed_.clear();
        sampler_bindless_heap_allocated_ = 0;
      } else {
        sampler_bindful_heap_pool_->ClearCache();
        view_bindful_heap_pool_->ClearCache();
      }
      constant_buffer_pool_->ClearCache();

      texture_cache_->ClearCache();

      // Not clearing the root signatures as they're referenced by pipelines,
      // which are not destroyed.

      primitive_processor_->ClearCache();

      render_target_cache_->ClearCache();

      shared_memory_->ClearCache();
    }
  }

  return true;
}

bool D3D12CommandProcessor::CanEndSubmissionImmediately() const {
  return !submission_open_ || !pipeline_cache_->IsCreatingPipelines();
}

void D3D12CommandProcessor::ClearCommandAllocatorCache() {
  while (command_allocator_submitted_first_) {
    auto next = command_allocator_submitted_first_->next;
    command_allocator_submitted_first_->command_allocator->Release();
    delete command_allocator_submitted_first_;
    command_allocator_submitted_first_ = next;
  }
  command_allocator_submitted_last_ = nullptr;
  while (command_allocator_writable_first_) {
    auto next = command_allocator_writable_first_->next;
    command_allocator_writable_first_->command_allocator->Release();
    delete command_allocator_writable_first_;
    command_allocator_writable_first_ = next;
  }
  command_allocator_writable_last_ = nullptr;
}

void D3D12CommandProcessor::UpdateFixedFunctionState(
    const draw_util::ViewportInfo& viewport_info, const draw_util::Scissor& scissor,
    bool primitive_polygonal, reg::RB_DEPTHCONTROL normalized_depth_control) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  // Viewport.
  D3D12_VIEWPORT viewport;
  viewport.TopLeftX = float(viewport_info.xy_offset[0]);
  viewport.TopLeftY = float(viewport_info.xy_offset[1]);
  viewport.Width = float(viewport_info.xy_extent[0]);
  viewport.Height = float(viewport_info.xy_extent[1]);
  viewport.MinDepth = viewport_info.z_min;
  viewport.MaxDepth = viewport_info.z_max;
  SetViewport(viewport);

  // Scissor.
  D3D12_RECT scissor_rect;
  scissor_rect.left = LONG(scissor.offset[0]);
  scissor_rect.top = LONG(scissor.offset[1]);
  scissor_rect.right = LONG(scissor.offset[0] + scissor.extent[0]);
  scissor_rect.bottom = LONG(scissor.offset[1] + scissor.extent[1]);
  SetScissorRect(scissor_rect);

  if (render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets) {
    const RegisterFile& regs = GetActiveDrawRegisterFile();

    // Blend factor.
    float blend_factor[] = {
        regs.Get<float>(XE_GPU_REG_RB_BLEND_RED),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA),
    };
    // std::memcmp instead of != so in case of NaN, every draw won't be
    // invalidating it.
    ff_blend_factor_update_needed_ |=
        std::memcmp(ff_blend_factor_, blend_factor, sizeof(float) * 4) != 0;
    if (ff_blend_factor_update_needed_) {
      std::memcpy(ff_blend_factor_, blend_factor, sizeof(float) * 4);
      deferred_command_list_.D3DOMSetBlendFactor(ff_blend_factor_);
      ff_blend_factor_update_needed_ = false;
    }

    // Stencil reference value. Per-face reference not supported by Direct3D 12,
    // choose the back face one only if drawing only back faces.
    Register stencil_ref_mask_reg;
    auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
    if (primitive_polygonal && normalized_depth_control.backface_enable &&
        pa_su_sc_mode_cntl.cull_front && !pa_su_sc_mode_cntl.cull_back) {
      stencil_ref_mask_reg = XE_GPU_REG_RB_STENCILREFMASK_BF;
    } else {
      stencil_ref_mask_reg = XE_GPU_REG_RB_STENCILREFMASK;
    }
    uint32_t stencil_ref = regs.Get<reg::RB_STENCILREFMASK>(stencil_ref_mask_reg).stencilref;
    ff_stencil_ref_update_needed_ |= ff_stencil_ref_ != stencil_ref;
    if (ff_stencil_ref_update_needed_) {
      ff_stencil_ref_ = stencil_ref;
      deferred_command_list_.D3DOMSetStencilRef(ff_stencil_ref_);
      ff_stencil_ref_update_needed_ = false;
    }
  }
}

void D3D12CommandProcessor::UpdateSystemConstantValues(
    bool shared_memory_is_uav, bool primitive_polygonal, uint32_t line_loop_closing_index,
    xenos::Endian index_endian, const draw_util::ViewportInfo& viewport_info,
    uint32_t used_texture_mask, reg::RB_DEPTHCONTROL normalized_depth_control,
    uint32_t normalized_color_mask) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = GetActiveDrawRegisterFile();
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  auto rb_alpha_ref = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
  auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  auto rb_stencilrefmask = regs.Get<reg::RB_STENCILREFMASK>();
  auto rb_stencilrefmask_bf = regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto sq_context_misc = regs.Get<reg::SQ_CONTEXT_MISC>();
  auto sq_program_cntl = regs.Get<reg::SQ_PROGRAM_CNTL>();
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  uint32_t vgt_indx_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  uint32_t vgt_max_vtx_indx = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;
  uint32_t vgt_min_vtx_indx = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;

  bool edram_rov_used =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;
  uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();

  // [NR-SYS]/[NR-LEAN] shared input fill for the 5-3b-1 mirror derivation,
  // used by the lean early-out below and the compare-site call at the end.
  auto nr_sys_fill_inputs = [&](nr::NrSysInputs& nr_in) {
    nr_in.regs = &regs[0];
    nr_in.shared_memory_is_uav = shared_memory_is_uav;
    nr_in.primitive_polygonal = primitive_polygonal;
    nr_in.line_loop_closing_index = line_loop_closing_index;
    nr_in.index_endian = uint32_t(index_endian);
    for (uint32_t i = 0; i < 3; ++i) {
      nr_in.ndc_scale[i] = viewport_info.ndc_scale[i];
      nr_in.ndc_offset[i] = viewport_info.ndc_offset[i];
    }
    nr_in.xy_extent[0] = viewport_info.xy_extent[0];
    nr_in.xy_extent[1] = viewport_info.xy_extent[1];
    nr_in.used_texture_mask = used_texture_mask;
    nr_in.edram_rov_used = false;
    nr_in.color_exp_bias_host_remap =
        render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets &&
        !render_target_cache_->IsFixed16TruncatedToMinus1To1();
    nr_in.gamma_render_target_as_unorm16 =
        render_target_cache_->gamma_render_target_as_unorm16();
    nr_in.draw_resolution_scale_x = draw_resolution_scale_x;
    nr_in.draw_resolution_scale_y = draw_resolution_scale_y;
    nr_in.texture_signs_fn = NrSysTextureSigns;
    nr_in.texture_res_scaled_fn = NrSysTextureResScaled;
    nr_in.texture_ctx = static_cast<TextureCache*>(texture_cache_.get());
  };

  // [NR-LEAN] 5-4-4b inc 2b: the member is stale whenever the lean path below
  // skipped the emulated derivation. Re-sync it from the mirror (the byte-
  // proven equivalent of what the skipped body would have written, sticky
  // fields included) before any full-body run reads it for dirty compares.
  if (g_nr_sys_member_stale) {
    std::memcpy(&system_constants_, &g_nr_sys_state, sizeof(system_constants_));
    g_nr_sys_member_stale = false;
  }
  if (g_nr_lean_sys && !edram_rov_used) {
    // Mirror-only fast path: run the 5-3b-1 derivation the swap uploads and
    // collapse the emulated body's per-field dirty accumulation into one
    // whole-struct before/after compare (equivalent: NrSysUpdate has the
    // emulated write semantics field for field). Nothing else reads
    // system_constants_ under the swap with verify off; the rare fallback
    // re-syncs it in IssueDrawImpl.
    nr::NrSysConstants nr_prev;
    std::memcpy(&nr_prev, &g_nr_sys_state, sizeof(nr_prev));
    nr::NrSysInputs nr_in;
    nr_sys_fill_inputs(nr_in);
    if (nr::NrSysUpdate(nr_in, &g_nr_sys_state)) {
      g_nr_sys_changed =
          std::memcmp(&nr_prev, &g_nr_sys_state, sizeof(nr_prev)) != 0;
      if (g_nr_sys_changed) {
        cbuffer_binding_system_.up_to_date = false;
      }
      g_nr_sys_member_stale = true;
      ++g_nr_swap_probe.sys_lean;
      return;
    }
    // ROV refusal (never seen in this game): fall through to the full body.
  }

  // Get the color info register values for each render target. Also, for ROV,
  // exclude components that don't exist in the format from the write mask.
  // Don't exclude fully overlapping render targets, however - two render
  // targets with the same base address are used in the lighting pass of
  // 4D5307E6, for example, with the needed one picked with dynamic control
  // flow.
  reg::RB_COLOR_INFO color_infos[4];
  float rt_clamp[4][4];
  // Two UINT32_MAX if no components actually existing in the RT are written.
  uint32_t rt_keep_masks[4][2];
  for (uint32_t i = 0; i < 4; ++i) {
    auto color_info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
    color_infos[i] = color_info;
    if (edram_rov_used) {
      RenderTargetCache::GetPSIColorFormatInfo(
          color_info.color_format, (normalized_color_mask >> (i * 4)) & 0b1111, rt_clamp[i][0],
          rt_clamp[i][1], rt_clamp[i][2], rt_clamp[i][3], rt_keep_masks[i][0], rt_keep_masks[i][1]);
    }
  }

  // Disable depth and stencil if it aliases a color render target (for
  // instance, during the XBLA logo in 58410954, though depth writing is already
  // disabled there).
  bool depth_stencil_enabled =
      normalized_depth_control.stencil_enable || normalized_depth_control.z_enable;
  if (edram_rov_used && depth_stencil_enabled) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (rb_depth_info.depth_base == color_infos[i].color_base &&
          (rt_keep_masks[i][0] != UINT32_MAX || rt_keep_masks[i][1] != UINT32_MAX)) {
        depth_stencil_enabled = false;
        break;
      }
    }
  }

  bool dirty = false;

  // Flags.
  uint32_t flags = 0;
  // Whether shared memory is an SRV or a UAV. Because a resource can't be in a
  // read-write (UAV) and a read-only (SRV, IBV) state at once, if any shader in
  // the pipeline uses memexport, the shared memory buffer must be a UAV.
  if (shared_memory_is_uav) {
    flags |= DxbcShaderTranslator::kSysFlag_SharedMemoryIsUAV;
  }
  // W0 division control.
  // http://www.x.org/docs/AMD/old/evergreen_3D_registers_v2.pdf
  // 8: VTX_XY_FMT = true: the incoming XY have already been multiplied by 1/W0.
  //               = false: multiply the X, Y coordinates by 1/W0.
  // 9: VTX_Z_FMT = true: the incoming Z has already been multiplied by 1/W0.
  //              = false: multiply the Z coordinate by 1/W0.
  // 10: VTX_W0_FMT = true: the incoming W0 is not 1/W0. Perform the reciprocal
  //                        to get 1/W0.
  if (pa_cl_vte_cntl.vtx_xy_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_XYDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_z_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_ZDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_w0_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_WNotReciprocal;
  }
  // Whether the primitive is polygonal and SV_IsFrontFace matters.
  if (primitive_polygonal) {
    flags |= DxbcShaderTranslator::kSysFlag_PrimitivePolygonal;
  }
  // Primitive type.
  if (draw_util::IsPrimitiveLine(regs)) {
    flags |= DxbcShaderTranslator::kSysFlag_PrimitiveLine;
  }
  // Depth format.
  if (rb_depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
    flags |= DxbcShaderTranslator::kSysFlag_DepthFloat24;
  }
  // Alpha test.
  xenos::CompareFunction alpha_test_function = rb_colorcontrol.alpha_test_enable
                                                   ? rb_colorcontrol.alpha_func
                                                   : xenos::CompareFunction::kAlways;
  flags |= uint32_t(alpha_test_function) << DxbcShaderTranslator::kSysFlag_AlphaPassIfLess_Shift;
  // Gamma writing.
  if (!render_target_cache_->gamma_render_target_as_unorm16()) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (color_infos[i].color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
        flags |= DxbcShaderTranslator::kSysFlag_ConvertColor0ToGamma << i;
      }
    }
  }
  if (edram_rov_used && depth_stencil_enabled) {
    flags |= DxbcShaderTranslator::kSysFlag_ROVDepthStencil;
    if (normalized_depth_control.z_enable) {
      flags |= uint32_t(normalized_depth_control.zfunc)
               << DxbcShaderTranslator::kSysFlag_ROVDepthPassIfLess_Shift;
      if (normalized_depth_control.z_write_enable) {
        flags |= DxbcShaderTranslator::kSysFlag_ROVDepthWrite;
      }
    } else {
      // In case stencil is used without depth testing - always pass, and
      // don't modify the stored depth.
      flags |= DxbcShaderTranslator::kSysFlag_ROVDepthPassIfLess |
               DxbcShaderTranslator::kSysFlag_ROVDepthPassIfEqual |
               DxbcShaderTranslator::kSysFlag_ROVDepthPassIfGreater;
    }
    if (normalized_depth_control.stencil_enable) {
      flags |= DxbcShaderTranslator::kSysFlag_ROVStencilTest;
    }
    // Hint - if not applicable to the shader, will not have effect.
    if (alpha_test_function == xenos::CompareFunction::kAlways &&
        !rb_colorcontrol.alpha_to_mask_enable) {
      flags |= DxbcShaderTranslator::kSysFlag_ROVDepthStencilEarlyWrite;
    }
  }
  dirty |= system_constants_.flags != flags;
  system_constants_.flags = flags;

  // Tessellation factor range, plus 1.0 according to the images in
  // https://www.slideshare.net/blackdevilvikas/next-generation-graphics-programming-on-xbox-360
  float tessellation_factor_min = regs.Get<float>(XE_GPU_REG_VGT_HOS_MIN_TESS_LEVEL) + 1.0f;
  float tessellation_factor_max = regs.Get<float>(XE_GPU_REG_VGT_HOS_MAX_TESS_LEVEL) + 1.0f;
  dirty |= system_constants_.tessellation_factor_range_min != tessellation_factor_min;
  system_constants_.tessellation_factor_range_min = tessellation_factor_min;
  dirty |= system_constants_.tessellation_factor_range_max != tessellation_factor_max;
  system_constants_.tessellation_factor_range_max = tessellation_factor_max;

  // Line loop closing index (or 0 when drawing other primitives or using an
  // index buffer).
  dirty |= system_constants_.line_loop_closing_index != line_loop_closing_index;
  system_constants_.line_loop_closing_index = line_loop_closing_index;

  // Index or tessellation edge factor buffer endianness.
  dirty |= system_constants_.vertex_index_endian != index_endian;
  system_constants_.vertex_index_endian = index_endian;

  // Vertex index offset.
  dirty |= system_constants_.vertex_index_offset != vgt_indx_offset;
  system_constants_.vertex_index_offset = vgt_indx_offset;

  // Vertex index range.
  dirty |= system_constants_.vertex_index_min != vgt_min_vtx_indx;
  dirty |= system_constants_.vertex_index_max != vgt_max_vtx_indx;
  system_constants_.vertex_index_min = vgt_min_vtx_indx;
  system_constants_.vertex_index_max = vgt_max_vtx_indx;

  // User clip planes (UCP_ENA_#), when not CLIP_DISABLE.
  // The shader knows only the total count - tightly packing the user clip
  // planes that are actually used.
  if (!pa_cl_clip_cntl.clip_disable) {
    float* user_clip_plane_write_ptr = system_constants_.user_clip_planes[0];
    uint32_t user_clip_planes_remaining = pa_cl_clip_cntl.ucp_ena;
    uint32_t user_clip_plane_index;
    while (rex::bit_scan_forward(user_clip_planes_remaining, &user_clip_plane_index)) {
      user_clip_planes_remaining &= ~(UINT32_C(1) << user_clip_plane_index);
      const void* user_clip_plane_regs =
          &regs[XE_GPU_REG_PA_CL_UCP_0_X + user_clip_plane_index * 4];
      if (std::memcmp(user_clip_plane_write_ptr, user_clip_plane_regs, 4 * sizeof(float))) {
        dirty = true;
        std::memcpy(user_clip_plane_write_ptr, user_clip_plane_regs, 4 * sizeof(float));
      }
      user_clip_plane_write_ptr += 4;
    }
  }

  // Conversion to Direct3D 12 normalized device coordinates.
  for (uint32_t i = 0; i < 3; ++i) {
    dirty |= system_constants_.ndc_scale[i] != viewport_info.ndc_scale[i];
    dirty |= system_constants_.ndc_offset[i] != viewport_info.ndc_offset[i];
    system_constants_.ndc_scale[i] = viewport_info.ndc_scale[i];
    system_constants_.ndc_offset[i] = viewport_info.ndc_offset[i];
  }

  // Point size.
  if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    auto pa_su_point_minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    auto pa_su_point_size = regs.Get<reg::PA_SU_POINT_SIZE>();
    float point_vertex_diameter_min = float(pa_su_point_minmax.min_size) * (2.0f / 16.0f);
    float point_vertex_diameter_max = float(pa_su_point_minmax.max_size) * (2.0f / 16.0f);
    float point_constant_diameter_x = float(pa_su_point_size.width) * (2.0f / 16.0f);
    float point_constant_diameter_y = float(pa_su_point_size.height) * (2.0f / 16.0f);
    dirty |= system_constants_.point_vertex_diameter_min != point_vertex_diameter_min;
    dirty |= system_constants_.point_vertex_diameter_max != point_vertex_diameter_max;
    dirty |= system_constants_.point_constant_diameter[0] != point_constant_diameter_x;
    dirty |= system_constants_.point_constant_diameter[1] != point_constant_diameter_y;
    system_constants_.point_vertex_diameter_min = point_vertex_diameter_min;
    system_constants_.point_vertex_diameter_max = point_vertex_diameter_max;
    system_constants_.point_constant_diameter[0] = point_constant_diameter_x;
    system_constants_.point_constant_diameter[1] = point_constant_diameter_y;
    // 2 because 1 in the NDC is half of the viewport's axis, 0.5 for diameter
    // to radius conversion to avoid multiplying the per-vertex diameter by an
    // additional constant in the shader.
    float point_screen_diameter_to_ndc_radius_x =
        (/* 0.5f * 2.0f * */ float(draw_resolution_scale_x)) /
        std::max(viewport_info.xy_extent[0], uint32_t(1));
    float point_screen_diameter_to_ndc_radius_y =
        (/* 0.5f * 2.0f * */ float(draw_resolution_scale_y)) /
        std::max(viewport_info.xy_extent[1], uint32_t(1));
    dirty |= system_constants_.point_screen_diameter_to_ndc_radius[0] !=
             point_screen_diameter_to_ndc_radius_x;
    dirty |= system_constants_.point_screen_diameter_to_ndc_radius[1] !=
             point_screen_diameter_to_ndc_radius_y;
    system_constants_.point_screen_diameter_to_ndc_radius[0] =
        point_screen_diameter_to_ndc_radius_x;
    system_constants_.point_screen_diameter_to_ndc_radius[1] =
        point_screen_diameter_to_ndc_radius_y;
  }

  // Texture signedness / gamma.
  uint32_t textures_resolution_scaled = 0;
  uint32_t textures_remaining = used_texture_mask;
  uint32_t texture_index;
  while (rex::bit_scan_forward(textures_remaining, &texture_index)) {
    textures_remaining &= ~(uint32_t(1) << texture_index);
    uint32_t& texture_signs_uint = system_constants_.texture_swizzled_signs[texture_index >> 2];
    uint32_t texture_signs_shift = (texture_index & 3) * 8;
    uint8_t texture_signs = texture_cache_->GetActiveTextureSwizzledSigns(texture_index);
    uint32_t texture_signs_shifted = uint32_t(texture_signs) << texture_signs_shift;
    uint32_t texture_signs_mask = uint32_t(0b11111111) << texture_signs_shift;
    dirty |= (texture_signs_uint & texture_signs_mask) != texture_signs_shifted;
    texture_signs_uint = (texture_signs_uint & ~texture_signs_mask) | texture_signs_shifted;
    textures_resolution_scaled |=
        uint32_t(texture_cache_->IsActiveTextureResolutionScaled(texture_index)) << texture_index;
  }
  dirty |= system_constants_.textures_resolution_scaled != textures_resolution_scaled;
  system_constants_.textures_resolution_scaled = textures_resolution_scaled;

  // Log2 of sample count, for alpha to mask and with ROV, for EDRAM address
  // calculation with MSAA.
  uint32_t sample_count_log2_x = rb_surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 1 : 0;
  uint32_t sample_count_log2_y = rb_surface_info.msaa_samples >= xenos::MsaaSamples::k2X ? 1 : 0;
  dirty |= system_constants_.sample_count_log2[0] != sample_count_log2_x;
  dirty |= system_constants_.sample_count_log2[1] != sample_count_log2_y;
  system_constants_.sample_count_log2[0] = sample_count_log2_x;
  system_constants_.sample_count_log2[1] = sample_count_log2_y;

  // Alpha test and alpha to coverage.
  dirty |= system_constants_.alpha_test_reference != rb_alpha_ref;
  system_constants_.alpha_test_reference = rb_alpha_ref;
  uint32_t alpha_to_mask =
      rb_colorcontrol.alpha_to_mask_enable ? (rb_colorcontrol.value >> 24) | (1 << 8) : 0;
  dirty |= system_constants_.alpha_to_mask != alpha_to_mask;
  system_constants_.alpha_to_mask = alpha_to_mask;

  uint32_t edram_tile_dwords_scaled = xenos::kEdramTileWidthSamples *
                                      xenos::kEdramTileHeightSamples *
                                      (draw_resolution_scale_x * draw_resolution_scale_y);

  // EDRAM pitch for ROV writing.
  if (edram_rov_used) {
    // Align, then multiply by 32bpp tile size in dwords.
    uint32_t edram_32bpp_tile_pitch_dwords_scaled =
        ((rb_surface_info.surface_pitch *
          (rb_surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 2 : 1)) +
         (xenos::kEdramTileWidthSamples - 1)) /
        xenos::kEdramTileWidthSamples * edram_tile_dwords_scaled;
    dirty |= system_constants_.edram_32bpp_tile_pitch_dwords_scaled !=
             edram_32bpp_tile_pitch_dwords_scaled;
    system_constants_.edram_32bpp_tile_pitch_dwords_scaled = edram_32bpp_tile_pitch_dwords_scaled;
  }

  // Color exponent bias and ROV render target writing.
  for (uint32_t i = 0; i < 4; ++i) {
    reg::RB_COLOR_INFO color_info = color_infos[i];
    // Exponent bias is in bits 20:25 of RB_COLOR_INFO.
    int32_t color_exp_bias = color_info.color_exp_bias;
    if (color_info.color_format == xenos::ColorRenderTargetFormat::k_16_16 ||
        color_info.color_format == xenos::ColorRenderTargetFormat::k_16_16_16_16) {
      if (render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets &&
          !render_target_cache_->IsFixed16TruncatedToMinus1To1()) {
        // Remap from -32...32 to -1...1 by dividing the output values by 32,
        // losing blending correctness, but getting the full range.
        color_exp_bias -= 5;
      }
    }
    auto color_exp_bias_scale =
        rex::memory::Reinterpret<float>(int32_t(0x3F800000 + (color_exp_bias << 23)));
    dirty |= system_constants_.color_exp_bias[i] != color_exp_bias_scale;
    system_constants_.color_exp_bias[i] = color_exp_bias_scale;
    if (edram_rov_used) {
      dirty |= system_constants_.edram_rt_keep_mask[i][0] != rt_keep_masks[i][0];
      system_constants_.edram_rt_keep_mask[i][0] = rt_keep_masks[i][0];
      dirty |= system_constants_.edram_rt_keep_mask[i][1] != rt_keep_masks[i][1];
      system_constants_.edram_rt_keep_mask[i][1] = rt_keep_masks[i][1];
      if (rt_keep_masks[i][0] != UINT32_MAX || rt_keep_masks[i][1] != UINT32_MAX) {
        uint32_t rt_base_dwords_scaled = xenos::EdramGuestBaseToHost(color_info.color_base) * edram_tile_dwords_scaled;
        dirty |= system_constants_.edram_rt_base_dwords_scaled[i] != rt_base_dwords_scaled;
        system_constants_.edram_rt_base_dwords_scaled[i] = rt_base_dwords_scaled;
        uint32_t format_flags = RenderTargetCache::AddPSIColorFormatFlags(color_info.color_format);
        dirty |= system_constants_.edram_rt_format_flags[i] != format_flags;
        system_constants_.edram_rt_format_flags[i] = format_flags;
        // Can't do float comparisons here because NaNs would result in always
        // setting the dirty flag.
        dirty |=
            std::memcmp(system_constants_.edram_rt_clamp[i], rt_clamp[i], 4 * sizeof(float)) != 0;
        std::memcpy(system_constants_.edram_rt_clamp[i], rt_clamp[i], 4 * sizeof(float));
        uint32_t blend_factors_ops =
            regs[reg::RB_BLENDCONTROL::rt_register_indices[i]] & 0x1FFF1FFF;
        dirty |= system_constants_.edram_rt_blend_factors_ops[i] != blend_factors_ops;
        system_constants_.edram_rt_blend_factors_ops[i] = blend_factors_ops;
      }
    }
  }

  if (edram_rov_used) {
    uint32_t depth_base_dwords_scaled = xenos::EdramGuestBaseToHost(rb_depth_info.depth_base) * edram_tile_dwords_scaled;
    dirty |= system_constants_.edram_depth_base_dwords_scaled != depth_base_dwords_scaled;
    system_constants_.edram_depth_base_dwords_scaled = depth_base_dwords_scaled;

    // For non-polygons, front polygon offset is used, and it's enabled if
    // POLY_OFFSET_PARA_ENABLED is set, for polygons, separate front and back
    // are used.
    float poly_offset_front_scale = 0.0f, poly_offset_front_offset = 0.0f;
    float poly_offset_back_scale = 0.0f, poly_offset_back_offset = 0.0f;
    if (primitive_polygonal) {
      if (pa_su_sc_mode_cntl.poly_offset_front_enable) {
        poly_offset_front_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
        poly_offset_front_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
      }
      if (pa_su_sc_mode_cntl.poly_offset_back_enable) {
        poly_offset_back_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_SCALE);
        poly_offset_back_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_OFFSET);
      }
    } else {
      if (pa_su_sc_mode_cntl.poly_offset_para_enable) {
        poly_offset_front_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
        poly_offset_front_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
        poly_offset_back_scale = poly_offset_front_scale;
        poly_offset_back_offset = poly_offset_front_offset;
      }
    }
    // With non-square resolution scaling, make sure the worst-case impact is
    // reverted (slope only along the scaled axis), thus max. More bias is
    // better than less bias, because less bias means Z fighting with the
    // background is more likely.
    float poly_offset_scale_factor = xenos::kPolygonOffsetScaleSubpixelUnit *
                                     std::max(draw_resolution_scale_x, draw_resolution_scale_y);
    poly_offset_front_scale *= poly_offset_scale_factor;
    poly_offset_back_scale *= poly_offset_scale_factor;
    dirty |= system_constants_.edram_poly_offset_front_scale != poly_offset_front_scale;
    system_constants_.edram_poly_offset_front_scale = poly_offset_front_scale;
    dirty |= system_constants_.edram_poly_offset_front_offset != poly_offset_front_offset;
    system_constants_.edram_poly_offset_front_offset = poly_offset_front_offset;
    dirty |= system_constants_.edram_poly_offset_back_scale != poly_offset_back_scale;
    system_constants_.edram_poly_offset_back_scale = poly_offset_back_scale;
    dirty |= system_constants_.edram_poly_offset_back_offset != poly_offset_back_offset;
    system_constants_.edram_poly_offset_back_offset = poly_offset_back_offset;

    if (depth_stencil_enabled && normalized_depth_control.stencil_enable) {
      dirty |= system_constants_.edram_stencil_front_reference != rb_stencilrefmask.stencilref;
      system_constants_.edram_stencil_front_reference = rb_stencilrefmask.stencilref;
      dirty |= system_constants_.edram_stencil_front_read_mask != rb_stencilrefmask.stencilmask;
      system_constants_.edram_stencil_front_read_mask = rb_stencilrefmask.stencilmask;
      dirty |=
          system_constants_.edram_stencil_front_write_mask != rb_stencilrefmask.stencilwritemask;
      system_constants_.edram_stencil_front_write_mask = rb_stencilrefmask.stencilwritemask;
      uint32_t stencil_func_ops = (normalized_depth_control.value >> 8) & ((1 << 12) - 1);
      dirty |= system_constants_.edram_stencil_front_func_ops != stencil_func_ops;
      system_constants_.edram_stencil_front_func_ops = stencil_func_ops;

      if (primitive_polygonal && normalized_depth_control.backface_enable) {
        dirty |= system_constants_.edram_stencil_back_reference != rb_stencilrefmask_bf.stencilref;
        system_constants_.edram_stencil_back_reference = rb_stencilrefmask_bf.stencilref;
        dirty |= system_constants_.edram_stencil_back_read_mask != rb_stencilrefmask_bf.stencilmask;
        system_constants_.edram_stencil_back_read_mask = rb_stencilrefmask_bf.stencilmask;
        dirty |= system_constants_.edram_stencil_back_write_mask !=
                 rb_stencilrefmask_bf.stencilwritemask;
        system_constants_.edram_stencil_back_write_mask = rb_stencilrefmask_bf.stencilwritemask;
        uint32_t stencil_func_ops_bf = (normalized_depth_control.value >> 20) & ((1 << 12) - 1);
        dirty |= system_constants_.edram_stencil_back_func_ops != stencil_func_ops_bf;
        system_constants_.edram_stencil_back_func_ops = stencil_func_ops_bf;
      } else {
        dirty |= std::memcmp(system_constants_.edram_stencil_back,
                             system_constants_.edram_stencil_front, 4 * sizeof(uint32_t)) != 0;
        std::memcpy(system_constants_.edram_stencil_back, system_constants_.edram_stencil_front,
                    4 * sizeof(uint32_t));
      }
    }

    dirty |= system_constants_.edram_blend_constant[0] != regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    system_constants_.edram_blend_constant[0] = regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    dirty |=
        system_constants_.edram_blend_constant[1] != regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    system_constants_.edram_blend_constant[1] = regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    dirty |= system_constants_.edram_blend_constant[2] != regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    system_constants_.edram_blend_constant[2] = regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    dirty |=
        system_constants_.edram_blend_constant[3] != regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
    system_constants_.edram_blend_constant[3] = regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
  }

  cbuffer_binding_system_.up_to_date &= !dirty;

  // [NR-SYS] Phase 5-3b-1: the system-constants mirror's gate. Runs HERE, at
  // the end of the emulated derivation, because this is the only moment both
  // derivations provably see the same register file and the same subsystem
  // state ([[probe-reads-the-wrong-moment]]). Our persistent mirror gets the
  // same inputs, applies its own transcription, and the two structs are
  // byte-compared whole. The mirror's layout is pinned to the real struct
  // here, at the compare site.
  if (g_nr_sysconst) {
    using TheirSys = DxbcShaderTranslator::SystemConstants;
    static_assert(sizeof(nr::NrSysConstants) == sizeof(TheirSys),
                  "[NR-SYS] mirror size drifted from SystemConstants");
    static_assert(offsetof(TheirSys, flags) == 0);
    static_assert(offsetof(TheirSys, line_loop_closing_index) == 12);
    static_assert(offsetof(TheirSys, user_clip_planes) == 32);
    static_assert(offsetof(TheirSys, ndc_scale) == 128);
    static_assert(offsetof(TheirSys, texture_swizzled_signs) == 176);
    static_assert(offsetof(TheirSys, alpha_to_mask) == 224);
    static_assert(offsetof(TheirSys, color_exp_bias) == 240);
    static_assert(offsetof(TheirSys, edram_rt_clamp) == 336);
    static_assert(offsetof(TheirSys, edram_blend_constant) == 448);
    if (edram_rov_used) {
      ++g_nr_sys.refused_rov;
    } else {
      if (!g_nr_sys_seeded) {
        std::memcpy(&g_nr_sys_state, &system_constants_, sizeof(g_nr_sys_state));
        g_nr_sys_seeded = true;
        ++g_nr_sys.reseeds;
      }
      nr::NrSysInputs nr_in;
      nr_sys_fill_inputs(nr_in);
      // [NR-VERIFY] inc 2: the derivation ALWAYS runs -- g_nr_sys_state is
      // what the swap uploads -- but the coverage counters and the whole-
      // struct memcmp are verify.
      if (nr::NrSysUpdate(nr_in, &g_nr_sys_state) && g_nr_verify) {
        ++g_nr_sys.checks;
        // Branch coverage, from the same inputs the derivation read.
        if (!pa_cl_clip_cntl.clip_disable && pa_cl_clip_cntl.ucp_ena) {
          ++g_nr_sys.cover_clip;
        }
        if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
          ++g_nr_sys.cover_point;
        }
        if (rb_colorcontrol.alpha_test_enable) {
          ++g_nr_sys.cover_atest;
        }
        if (rb_colorcontrol.alpha_to_mask_enable) {
          ++g_nr_sys.cover_a2m;
        }
        if (g_nr_sys_state.flags & (0xFu * nr::kNrSysFlagConvertColor0ToGamma)) {
          ++g_nr_sys.cover_gamma;
        }
        g_nr_sys.cover_tex += uint64_t(rex::bit_count(used_texture_mask));
        if (std::memcmp(&g_nr_sys_state, &system_constants_,
                        sizeof(g_nr_sys_state)) != 0) {
          ++g_nr_sys.mismatch;
          if (g_nr_sys_samples_this_window < kNrSysMaxSamplesPerWindow) {
            ++g_nr_sys_samples_this_window;
            const uint32_t diff_dword = nr::NrSysFirstDifference(
                g_nr_sys_state, &system_constants_);
            char field_name[64];
            nr::NrSysDwordName(diff_dword, field_name, sizeof(field_name));
            const uint32_t* ours_dwords =
                reinterpret_cast<const uint32_t*>(&g_nr_sys_state);
            const uint32_t* theirs_dwords =
                reinterpret_cast<const uint32_t*>(&system_constants_);
            REXGPU_WARN(
                "[nr-sys] FIRST DIFFERENCE dword={} ({}) ours={:08X} "
                "theirs={:08X}",
                diff_dword, field_name,
                diff_dword < nr::kNrSysConstantsDwords ? ours_dwords[diff_dword]
                                                       : 0u,
                diff_dword < nr::kNrSysConstantsDwords
                    ? theirs_dwords[diff_dword]
                    : 0u);
          }
          // Re-sync so one divergence names itself once, not on every draw
          // until the field is next rewritten.
          std::memcpy(&g_nr_sys_state, &system_constants_,
                      sizeof(g_nr_sys_state));
        }
      }
    }
  }
}

bool D3D12CommandProcessor::UpdateBindings(const D3D12Shader* vertex_shader,
                                           const D3D12Shader* pixel_shader,
                                           ID3D12RootSignature* root_signature,
                                           bool shared_memory_is_uav) {
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  const RegisterFile& regs = GetActiveDrawRegisterFile();

  // [NR-BND] Phase 5-3b-0: our recompositions below run against this same
  // `regs` inside each upload block, right after the emulated write, on this
  // thread -- the only moment both derivations provably read the same file
  // ([[probe-reads-the-wrong-moment]]).
  const uint32_t* nr_bnd_rf = &regs[0];
  static_assert(nr::kBindFloatVertexBase == XE_GPU_REG_SHADER_CONSTANT_000_X);
  static_assert(nr::kBindFloatPixelBase == XE_GPU_REG_SHADER_CONSTANT_256_X);
  static_assert(nr::kBindBoolLoopBase == XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031);
  static_assert(nr::kBindFetchBase == XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0);
  if (g_nr_bindings) {
    ++g_nr_bind.draws;
  }
  // [NR-DSC] Phase 5-3b-2: the cvar the emulated GetSamplerParameters reads
  // per call, latched once per UpdateBindings so our per-sampler derivations
  // use the same value theirs does within this call.
  const int32_t nr_dsc_aniso =
      g_nr_desc ? REXCVAR_GET(anisotropic_override) : -1;

#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  // Set the new root signature.
  if (current_graphics_root_signature_ != root_signature) {
    current_graphics_root_signature_ = root_signature;
    if (!bindless_resources_used_) {
      GetRootBindfulExtraParameterIndices(vertex_shader, pixel_shader,
                                          current_graphics_root_bindful_extras_);
    }
    // Changing the root signature invalidates all bindings.
    current_graphics_root_up_to_date_ = 0;
    inst_base_dirty_ = true;  // [hiz-pool]
    deferred_command_list_.D3DSetGraphicsRootSignature(root_signature);
  }

  // Select the root parameter indices depending on the used binding model.
  uint32_t root_parameter_fetch_constants = bindless_resources_used_
                                                ? kRootParameter_Bindless_FetchConstants
                                                : kRootParameter_Bindful_FetchConstants;
  uint32_t root_parameter_float_constants_vertex =
      bindless_resources_used_ ? kRootParameter_Bindless_FloatConstantsVertex
                               : kRootParameter_Bindful_FloatConstantsVertex;
  uint32_t root_parameter_float_constants_pixel = bindless_resources_used_
                                                      ? kRootParameter_Bindless_FloatConstantsPixel
                                                      : kRootParameter_Bindful_FloatConstantsPixel;
  uint32_t root_parameter_system_constants = bindless_resources_used_
                                                 ? kRootParameter_Bindless_SystemConstants
                                                 : kRootParameter_Bindful_SystemConstants;
  uint32_t root_parameter_bool_loop_constants = bindless_resources_used_
                                                    ? kRootParameter_Bindless_BoolLoopConstants
                                                    : kRootParameter_Bindful_BoolLoopConstants;
  uint32_t root_parameter_shared_memory_and_bindful_edram =
      bindless_resources_used_ ? kRootParameter_Bindless_SharedMemory
                               : kRootParameter_Bindful_SharedMemoryAndEdram;

  //
  // Update root constant buffers that are common for bindful and bindless.
  //

  // These are the constant base addresses/ranges for shaders.
  // We have these hardcoded right now cause nothing seems to differ on the Xbox
  // 360 (however, OpenGL ES on Adreno 200 on Android has different ranges).
  assert_true(regs[XE_GPU_REG_SQ_VS_CONST] == 0x000FF000 ||
              regs[XE_GPU_REG_SQ_VS_CONST] == 0x00000000);
  assert_true(regs[XE_GPU_REG_SQ_PS_CONST] == 0x000FF100 ||
              regs[XE_GPU_REG_SQ_PS_CONST] == 0x00000000);
  // Check if the float constant layout is still the same and get the counts.
  const Shader::ConstantRegisterMap& float_constant_map_vertex =
      vertex_shader->constant_register_map();
  uint32_t float_constant_count_vertex = float_constant_map_vertex.float_count;
  for (uint32_t i = 0; i < 4; ++i) {
    if (current_float_constant_map_vertex_[i] != float_constant_map_vertex.float_bitmap[i]) {
      current_float_constant_map_vertex_[i] = float_constant_map_vertex.float_bitmap[i];
      // If no float constants at all, we can reuse any buffer for them, so not
      // invalidating.
      if (float_constant_count_vertex) {
        cbuffer_binding_float_vertex_.up_to_date = false;
      }
    }
  }
  uint32_t float_constant_count_pixel = 0;
  if (pixel_shader != nullptr) {
    const Shader::ConstantRegisterMap& float_constant_map_pixel =
        pixel_shader->constant_register_map();
    float_constant_count_pixel = float_constant_map_pixel.float_count;
    for (uint32_t i = 0; i < 4; ++i) {
      if (current_float_constant_map_pixel_[i] != float_constant_map_pixel.float_bitmap[i]) {
        current_float_constant_map_pixel_[i] = float_constant_map_pixel.float_bitmap[i];
        if (float_constant_count_pixel) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      }
    }
  } else {
    std::memset(current_float_constant_map_pixel_, 0, sizeof(current_float_constant_map_pixel_));
  }

  // Write the constant buffer data.
  if (!cbuffer_binding_system_.up_to_date) {
    uint8_t* system_constants = constant_buffer_pool_->Request(
        frame_current_, sizeof(system_constants_), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_system_.address);
    if (system_constants == nullptr) {
      return false;
    }
    std::memcpy(system_constants, &system_constants_, sizeof(system_constants_));
    // [NR-BND] Counted only: the system-constants DERIVATION
    // (UpdateSystemConstantValues) is its own later increment.
    if (g_nr_bindings) {
      ++g_nr_bind.sys_up;
    }
    cbuffer_binding_system_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_system_constants);
  }
  if (!cbuffer_binding_float_vertex_.up_to_date) {
    // Even if the shader doesn't need any float constants, a valid binding must
    // still be provided, so if the first draw in the frame with the current
    // root signature doesn't have float constants at all, still allocate an
    // empty buffer.
    uint8_t* float_constants = constant_buffer_pool_->Request(
        frame_current_, sizeof(float) * 4 * std::max(float_constant_count_vertex, uint32_t(1)),
        D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
        &cbuffer_binding_float_vertex_.address);
    if (float_constants == nullptr) {
      return false;
    }
    // [NR-BND] Under the gate the pack loop below writes cached staging, not
    // the upload heap: reading WC memory back for the compare cost the CP
    // thread ~2/3 of its throughput (naruto_335). Publish + compare after.
    uint8_t* nr_bnd_fv_dst = nullptr;
    if (g_nr_bindings) {
      nr_bnd_fv_dst = float_constants;
      float_constants = g_nr_bnd_staging;
    }
    const uint8_t* nr_bnd_fv_start = float_constants;
    for (uint32_t i = 0; i < 4; ++i) {
      uint64_t float_constant_map_entry = float_constant_map_vertex.float_bitmap[i];
      uint32_t float_constant_index;
      while (rex::bit_scan_forward(float_constant_map_entry, &float_constant_index)) {
        float_constant_map_entry &= ~(1ull << float_constant_index);
        std::memcpy(
            float_constants,
            &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + (i << 8) + (float_constant_index << 2)],
            4 * sizeof(float));
        float_constants += 4 * sizeof(float);
      }
    }
    // [NR-BND] Our packing of the same map vs the bytes just packed; then the
    // staged bytes become the uploaded bytes by construction.
    if (g_nr_bindings) {
      const uint32_t nr_written = uint32_t(float_constants - nr_bnd_fv_start);
      std::memcpy(nr_bnd_fv_dst, nr_bnd_fv_start, nr_written);
      ++g_nr_bind.fv_up;
      alignas(16) uint8_t nr_ours[nr::kBindFloatMaxBytes];
      const uint32_t nr_size =
          nr::BindComposeFloats(nr_bnd_rf, nr::kBindFloatVertexBase,
                                float_constant_map_vertex.float_bitmap, nr_ours, sizeof(nr_ours));
      if (nr_size != nr_written) {
        ++g_nr_bind.fv_size_ne;
      } else if (nr_size != 0 && std::memcmp(nr_ours, nr_bnd_fv_start, nr_size) != 0) {
        ++g_nr_bind.fv_ne;
      }
    }
    cbuffer_binding_float_vertex_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_float_constants_vertex);
  }
  if (!cbuffer_binding_float_pixel_.up_to_date) {
    uint8_t* float_constants = constant_buffer_pool_->Request(
        frame_current_, sizeof(float) * 4 * std::max(float_constant_count_pixel, uint32_t(1)),
        D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
        &cbuffer_binding_float_pixel_.address);
    if (float_constants == nullptr) {
      return false;
    }
    // [NR-BND] Same staging redirect as the vertex block (WC-read trap).
    uint8_t* nr_bnd_fp_dst = nullptr;
    if (g_nr_bindings) {
      nr_bnd_fp_dst = float_constants;
      float_constants = g_nr_bnd_staging;
    }
    const uint8_t* nr_bnd_fp_start = float_constants;
    if (pixel_shader != nullptr) {
      const Shader::ConstantRegisterMap& float_constant_map_pixel =
          pixel_shader->constant_register_map();
      for (uint32_t i = 0; i < 4; ++i) {
        uint64_t float_constant_map_entry = float_constant_map_pixel.float_bitmap[i];
        uint32_t float_constant_index;
        while (rex::bit_scan_forward(float_constant_map_entry, &float_constant_index)) {
          float_constant_map_entry &= ~(1ull << float_constant_index);
          std::memcpy(
              float_constants,
              &regs[XE_GPU_REG_SHADER_CONSTANT_256_X + (i << 8) + (float_constant_index << 2)],
              4 * sizeof(float));
          float_constants += 4 * sizeof(float);
        }
      }
    }
    // [NR-BND] Same gate for the pixel half; a null pixel shader must compose
    // to zero bytes on our side too. Publish staging, then compare.
    if (g_nr_bindings) {
      const uint32_t nr_written = uint32_t(float_constants - nr_bnd_fp_start);
      std::memcpy(nr_bnd_fp_dst, nr_bnd_fp_start, nr_written);
      ++g_nr_bind.fp_up;
      alignas(16) uint8_t nr_ours[nr::kBindFloatMaxBytes];
      const uint64_t nr_zero_bitmap[4] = {0, 0, 0, 0};
      const uint32_t nr_size = nr::BindComposeFloats(
          nr_bnd_rf, nr::kBindFloatPixelBase,
          pixel_shader ? pixel_shader->constant_register_map().float_bitmap : nr_zero_bitmap,
          nr_ours, sizeof(nr_ours));
      if (nr_size != nr_written) {
        ++g_nr_bind.fp_size_ne;
      } else if (nr_size != 0 && std::memcmp(nr_ours, nr_bnd_fp_start, nr_size) != 0) {
        ++g_nr_bind.fp_ne;
      }
    }
    cbuffer_binding_float_pixel_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_float_constants_pixel);
  }
  if (!cbuffer_binding_bool_loop_.up_to_date) {
    constexpr uint32_t kBoolLoopConstantsSize = (8 + 32) * sizeof(uint32_t);
    uint8_t* bool_loop_constants = constant_buffer_pool_->Request(
        frame_current_, kBoolLoopConstantsSize, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_bool_loop_.address);
    if (bool_loop_constants == nullptr) {
      return false;
    }
    std::memcpy(bool_loop_constants, &regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                kBoolLoopConstantsSize);
    // [NR-BND] Verbatim window: proves our base + size transcription against
    // the SOURCE the upload memcpy'd (the heap bytes equal it by that memcpy;
    // reading the upload heap back is the WC trap, never do it).
    if (g_nr_bindings) {
      ++g_nr_bind.bl_up;
      uint8_t nr_ours[nr::kBindBoolLoopBytes];
      static_assert(sizeof(nr_ours) == kBoolLoopConstantsSize);
      nr::BindComposeBoolLoop(nr_bnd_rf, nr_ours, sizeof(nr_ours));
      if (std::memcmp(nr_ours, &regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                      sizeof(nr_ours)) != 0) {
        ++g_nr_bind.bl_ne;
      }
    }
    cbuffer_binding_bool_loop_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_bool_loop_constants);
  }
  if (!cbuffer_binding_fetch_.up_to_date) {
    constexpr uint32_t kFetchConstantsSize = 32 * 6 * sizeof(uint32_t);
    uint8_t* fetch_constants = constant_buffer_pool_->Request(
        frame_current_, kFetchConstantsSize, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_fetch_.address);
    if (fetch_constants == nullptr) {
      return false;
    }
    std::memcpy(fetch_constants, &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], kFetchConstantsSize);
    // [NR-BND] Verbatim window, same source-compare rule as bool/loop.
    if (g_nr_bindings) {
      ++g_nr_bind.fx_up;
      uint8_t nr_ours[nr::kBindFetchBytes];
      static_assert(sizeof(nr_ours) == kFetchConstantsSize);
      nr::BindComposeFetch(nr_bnd_rf, nr_ours, sizeof(nr_ours));
      if (std::memcmp(nr_ours, &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0],
                      sizeof(nr_ours)) != 0) {
        ++g_nr_bind.fx_ne;
      }
    }
    cbuffer_binding_fetch_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_fetch_constants);
  }

  //
  // Update descriptors.
  //

  if (!current_shared_memory_binding_is_uav_.has_value() ||
      current_shared_memory_binding_is_uav_.value() != shared_memory_is_uav) {
    current_shared_memory_binding_is_uav_ = shared_memory_is_uav;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_shared_memory_and_bindful_edram);
  }

  // Get textures and samplers used by the vertex shader, check if the last used
  // samplers are compatible and update them.
  size_t texture_layout_uid_vertex = vertex_shader->GetTextureBindingLayoutUserUID();
  size_t sampler_layout_uid_vertex = vertex_shader->GetSamplerBindingLayoutUserUID();
  const std::vector<D3D12Shader::TextureBinding>& textures_vertex =
      vertex_shader->GetTextureBindingsAfterTranslation();
  const std::vector<D3D12Shader::SamplerBinding>& samplers_vertex =
      vertex_shader->GetSamplerBindingsAfterTranslation();
  size_t texture_count_vertex = textures_vertex.size();
  size_t sampler_count_vertex = samplers_vertex.size();
  if (sampler_count_vertex) {
    if (current_sampler_layout_uid_vertex_ != sampler_layout_uid_vertex) {
      current_sampler_layout_uid_vertex_ = sampler_layout_uid_vertex;
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
      bindful_samplers_written_vertex_ = false;
    }
    current_samplers_vertex_.resize(
        std::max(current_samplers_vertex_.size(), sampler_count_vertex));
    for (size_t i = 0; i < sampler_count_vertex; ++i) {
      D3D12TextureCache::SamplerParameters parameters =
          texture_cache_->GetSamplerParameters(samplers_vertex[i]);
      // [NR-DSC] Our SamplerParameters from the raw fetch dwords, against
      // theirs at the moment it is derived (same thread, same file).
      if (g_nr_desc) {
        ++g_nr_desc_probe.smp_checks;
        const uint32_t nr_dsc_ours = nr::DescSamplerParams(
            &nr_bnd_rf[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 +
                       samplers_vertex[i].fetch_constant * 6],
            uint32_t(samplers_vertex[i].mag_filter),
            uint32_t(samplers_vertex[i].min_filter),
            uint32_t(samplers_vertex[i].mip_filter),
            uint32_t(samplers_vertex[i].aniso_filter), nr_dsc_aniso);
        if (nr_dsc_ours != parameters.value) {
          ++g_nr_desc_probe.smp_ne;
          if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
            ++g_nr_desc_samples_this_window;
            REXGPU_WARN("[nr-dsc] SMP DIFF v fc={} ours={:#x} theirs={:#x}",
                        samplers_vertex[i].fetch_constant, nr_dsc_ours,
                        parameters.value);
          }
        }
      }
      if (current_samplers_vertex_[i] != parameters) {
        cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
        bindful_samplers_written_vertex_ = false;
        current_samplers_vertex_[i] = parameters;
      }
    }
  }

  // Get textures and samplers used by the pixel shader, check if the last used
  // samplers are compatible and update them.
  size_t texture_layout_uid_pixel, sampler_layout_uid_pixel;
  const std::vector<D3D12Shader::TextureBinding>* textures_pixel;
  const std::vector<D3D12Shader::SamplerBinding>* samplers_pixel;
  size_t texture_count_pixel, sampler_count_pixel;
  if (pixel_shader != nullptr) {
    texture_layout_uid_pixel = pixel_shader->GetTextureBindingLayoutUserUID();
    sampler_layout_uid_pixel = pixel_shader->GetSamplerBindingLayoutUserUID();
    textures_pixel = &pixel_shader->GetTextureBindingsAfterTranslation();
    texture_count_pixel = textures_pixel->size();
    samplers_pixel = &pixel_shader->GetSamplerBindingsAfterTranslation();
    sampler_count_pixel = samplers_pixel->size();
    if (sampler_count_pixel) {
      if (current_sampler_layout_uid_pixel_ != sampler_layout_uid_pixel) {
        current_sampler_layout_uid_pixel_ = sampler_layout_uid_pixel;
        cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
        bindful_samplers_written_pixel_ = false;
      }
      current_samplers_pixel_.resize(
          std::max(current_samplers_pixel_.size(), size_t(sampler_count_pixel)));
      for (uint32_t i = 0; i < sampler_count_pixel; ++i) {
        D3D12TextureCache::SamplerParameters parameters =
            texture_cache_->GetSamplerParameters((*samplers_pixel)[i]);
        // [NR-DSC] The pixel half of the SamplerParameters gate.
        if (g_nr_desc) {
          ++g_nr_desc_probe.smp_checks;
          const uint32_t nr_dsc_ours = nr::DescSamplerParams(
              &nr_bnd_rf[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 +
                         (*samplers_pixel)[i].fetch_constant * 6],
              uint32_t((*samplers_pixel)[i].mag_filter),
              uint32_t((*samplers_pixel)[i].min_filter),
              uint32_t((*samplers_pixel)[i].mip_filter),
              uint32_t((*samplers_pixel)[i].aniso_filter), nr_dsc_aniso);
          if (nr_dsc_ours != parameters.value) {
            ++g_nr_desc_probe.smp_ne;
            if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
              ++g_nr_desc_samples_this_window;
              REXGPU_WARN("[nr-dsc] SMP DIFF p fc={} ours={:#x} theirs={:#x}",
                          (*samplers_pixel)[i].fetch_constant, nr_dsc_ours,
                          parameters.value);
            }
          }
        }
        if (current_samplers_pixel_[i] != parameters) {
          current_samplers_pixel_[i] = parameters;
          cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
          bindful_samplers_written_pixel_ = false;
        }
      }
    }
  } else {
    texture_layout_uid_pixel = PipelineCache::kLayoutUIDEmpty;
    sampler_layout_uid_pixel = PipelineCache::kLayoutUIDEmpty;
    textures_pixel = nullptr;
    texture_count_pixel = 0;
    samplers_pixel = nullptr;
    sampler_count_pixel = 0;
  }

  // [NR-BND] Census the binding-layout tuple (the bookkeeping a native
  // UpdateBindings must own: descriptor layouts are keyed by the UIDs the
  // pipeline cache assigns per translation) and the per-draw binding volumes.
  if (g_nr_bindings) {
    nr::BindCensusAdd(&g_nr_bind.layout_census,
                      nr::BindLayoutKey(texture_layout_uid_vertex, sampler_layout_uid_vertex,
                                        texture_layout_uid_pixel, sampler_layout_uid_pixel));
    g_nr_bind.tex_v += texture_count_vertex;
    g_nr_bind.smp_v += sampler_count_vertex;
    g_nr_bind.tex_p += texture_count_pixel;
    g_nr_bind.smp_p += sampler_count_pixel;
  }

  assert_true(sampler_count_vertex + sampler_count_pixel <= kSamplerHeapSize);

  if (bindless_resources_used_) {
    //
    // Bindless descriptors path.
    //

    // Check if need to write new descriptor indices.
    // Samplers have already been checked.
    if (texture_count_vertex && cbuffer_binding_descriptor_indices_vertex_.up_to_date &&
        (current_texture_layout_uid_vertex_ != texture_layout_uid_vertex ||
         !texture_cache_->AreActiveTextureSRVKeysUpToDate(current_texture_srv_keys_vertex_.data(),
                                                          textures_vertex.data(),
                                                          texture_count_vertex))) {
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
    }
    if (texture_count_pixel && cbuffer_binding_descriptor_indices_pixel_.up_to_date &&
        (current_texture_layout_uid_pixel_ != texture_layout_uid_pixel ||
         !texture_cache_->AreActiveTextureSRVKeysUpToDate(current_texture_srv_keys_pixel_.data(),
                                                          textures_pixel->data(),
                                                          texture_count_pixel))) {
      cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
    }

    // [NR-DSC] The heap identity before the resolution block: a change across
    // it means the block overflowed + switched heaps (the emulated map was
    // cleared and every index re-allocated from zero), so the mirror resets.
    ID3D12DescriptorHeap* nr_dsc_heap_before = sampler_bindless_heap_current_;

    // Get sampler descriptor indices, write new samplers, and handle sampler
    // heap overflow if it happens.
    if ((sampler_count_vertex && !cbuffer_binding_descriptor_indices_vertex_.up_to_date) ||
        (sampler_count_pixel && !cbuffer_binding_descriptor_indices_pixel_.up_to_date)) {
      for (uint32_t i = 0; i < 2; ++i) {
        if (i) {
          // Overflow happened - invalidate sampler bindings because their
          // descriptor indices can't be used anymore (and even if heap creation
          // fails, because current_sampler_bindless_indices_#_ are in an
          // undefined state now) and switch to a new sampler heap.
          cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
          cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
          ID3D12DescriptorHeap* sampler_heap_new;
          if (!sampler_bindless_heaps_overflowed_.empty() &&
              sampler_bindless_heaps_overflowed_.front().second <= submission_completed_) {
            sampler_heap_new = sampler_bindless_heaps_overflowed_.front().first;
            sampler_bindless_heaps_overflowed_.pop_front();
          } else {
            D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_new_desc;
            sampler_heap_new_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            sampler_heap_new_desc.NumDescriptors = kSamplerHeapSize;
            sampler_heap_new_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            sampler_heap_new_desc.NodeMask = 0;
            if (FAILED(device->CreateDescriptorHeap(&sampler_heap_new_desc,
                                                    IID_PPV_ARGS(&sampler_heap_new)))) {
              REXGPU_ERROR(
                  "Failed to create a new bindless sampler descriptor heap "
                  "after an overflow of the previous one");
              return false;
            }
          }
          // Only change the heap if a new heap was created successfully, not to
          // leave the values in an undefined state in case CreateDescriptorHeap
          // has failed.
          sampler_bindless_heaps_overflowed_.push_back(
              std::make_pair(sampler_bindless_heap_current_, submission_current_));
          sampler_bindless_heap_current_ = sampler_heap_new;
          sampler_bindless_heap_cpu_start_ =
              sampler_bindless_heap_current_->GetCPUDescriptorHandleForHeapStart();
          sampler_bindless_heap_gpu_start_ =
              sampler_bindless_heap_current_->GetGPUDescriptorHandleForHeapStart();
          sampler_bindless_heap_allocated_ = 0;
          // The only thing the heap is used for now is texture cache samplers -
          // invalidate all of them.
          texture_cache_bindless_sampler_map_.clear();
          deferred_command_list_.SetDescriptorHeaps(view_bindless_heap_,
                                                    sampler_bindless_heap_current_);
          current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_SamplerHeap);
        }
        bool samplers_overflowed = false;
        if (sampler_count_vertex && !cbuffer_binding_descriptor_indices_vertex_.up_to_date) {
          current_sampler_bindless_indices_vertex_.resize(std::max(
              current_sampler_bindless_indices_vertex_.size(), size_t(sampler_count_vertex)));
          for (uint32_t j = 0; j < sampler_count_vertex; ++j) {
            D3D12TextureCache::SamplerParameters sampler_parameters = current_samplers_vertex_[j];
            uint32_t sampler_index;
            auto it = texture_cache_bindless_sampler_map_.find(sampler_parameters.value);
            if (it != texture_cache_bindless_sampler_map_.end()) {
              sampler_index = it->second;
            } else {
              if (sampler_bindless_heap_allocated_ >= kSamplerHeapSize) {
                samplers_overflowed = true;
                break;
              }
              sampler_index = sampler_bindless_heap_allocated_++;
              texture_cache_->WriteSampler(sampler_parameters,
                                           provider.OffsetSamplerDescriptor(
                                               sampler_bindless_heap_cpu_start_, sampler_index));
              texture_cache_bindless_sampler_map_.emplace(sampler_parameters.value, sampler_index);
            }
            current_sampler_bindless_indices_vertex_[j] = sampler_index;
          }
        }
        if (samplers_overflowed) {
          continue;
        }
        if (sampler_count_pixel && !cbuffer_binding_descriptor_indices_pixel_.up_to_date) {
          current_sampler_bindless_indices_pixel_.resize(std::max(
              current_sampler_bindless_indices_pixel_.size(), size_t(sampler_count_pixel)));
          for (uint32_t j = 0; j < sampler_count_pixel; ++j) {
            D3D12TextureCache::SamplerParameters sampler_parameters = current_samplers_pixel_[j];
            uint32_t sampler_index;
            auto it = texture_cache_bindless_sampler_map_.find(sampler_parameters.value);
            if (it != texture_cache_bindless_sampler_map_.end()) {
              sampler_index = it->second;
            } else {
              if (sampler_bindless_heap_allocated_ >= kSamplerHeapSize) {
                samplers_overflowed = true;
                break;
              }
              sampler_index = sampler_bindless_heap_allocated_++;
              texture_cache_->WriteSampler(sampler_parameters,
                                           provider.OffsetSamplerDescriptor(
                                               sampler_bindless_heap_cpu_start_, sampler_index));
              texture_cache_bindless_sampler_map_.emplace(sampler_parameters.value, sampler_index);
            }
            current_sampler_bindless_indices_pixel_[j] = sampler_index;
          }
        }
        if (!samplers_overflowed) {
          break;
        }
      }
    }

    // [NR-DSC] Replay the sampler-index resolutions the emulated block just
    // performed, in its own order (vertex then pixel, ascending), against the
    // mirror map: known params must agree; new params either PREDICT the
    // allocation counter (fresh) or are learned as pre-arm entries (seeded).
    // "The loop ran" == (count && !up_to_date) evaluated HERE: the flags only
    // flip true at the cbuffer builds below, and an overflow inside the block
    // clears both, so this condition survives the retry path too.
    if (g_nr_desc) {
      if (sampler_bindless_heap_current_ != nr_dsc_heap_before) {
        ++g_nr_desc_probe.heap_switches;
        nr::DescSamplerMapReset(&g_nr_desc_smap, 0);
      }
      for (uint32_t nr_dsc_pass = 0; nr_dsc_pass < 2; ++nr_dsc_pass) {
        const size_t nr_dsc_count =
            nr_dsc_pass ? sampler_count_pixel : sampler_count_vertex;
        const bool nr_dsc_ran =
            nr_dsc_count &&
            !(nr_dsc_pass ? cbuffer_binding_descriptor_indices_pixel_.up_to_date
                          : cbuffer_binding_descriptor_indices_vertex_.up_to_date);
        if (!nr_dsc_ran) {
          continue;
        }
        for (size_t j = 0; j < nr_dsc_count; ++j) {
          const uint32_t nr_dsc_params =
              (nr_dsc_pass ? current_samplers_pixel_[j]
                           : current_samplers_vertex_[j])
                  .value;
          const uint32_t nr_dsc_their_index =
              nr_dsc_pass ? current_sampler_bindless_indices_pixel_[j]
                          : current_sampler_bindless_indices_vertex_[j];
          switch (nr::DescSamplerMapObserve(&g_nr_desc_smap, nr_dsc_params,
                                            nr_dsc_their_index)) {
            case nr::kDescSamplerMatch:
              ++g_nr_desc_probe.smap_match;
              break;
            case nr::kDescSamplerFresh:
              ++g_nr_desc_probe.smap_fresh;
              break;
            case nr::kDescSamplerSeeded:
              ++g_nr_desc_probe.smap_seeded;
              break;
            case nr::kDescSamplerOverflow:
              ++g_nr_desc_probe.smap_ovf;
              break;
            default:
              ++g_nr_desc_probe.smap_ne;
              if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
                ++g_nr_desc_samples_this_window;
                REXGPU_WARN("[nr-dsc] SMAP DIFF {} j={} params={:#x} theirs={}",
                            nr_dsc_pass ? "p" : "v", j, nr_dsc_params,
                            nr_dsc_their_index);
              }
              break;
          }
        }
      }
    }

    if (!cbuffer_binding_descriptor_indices_vertex_.up_to_date) {
      // [NR-BND] Rebuild rate; [NR-DSC] (5-3b-2) checks the contents.
      if (g_nr_bindings) {
        ++g_nr_bind.div_up;
      }
      uint32_t* descriptor_indices = reinterpret_cast<uint32_t*>(constant_buffer_pool_->Request(
          frame_current_,
          std::max(texture_count_vertex + sampler_count_vertex, size_t(1)) * sizeof(uint32_t),
          D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
          &cbuffer_binding_descriptor_indices_vertex_.address));
      if (!descriptor_indices) {
        return false;
      }
      // [NR-DSC] Stage the rebuild in cached memory; published + compared
      // below (the pool is an upload heap - never read it back,
      // [[upload-heap-readback-trap]]). The SPAN is max_slot+1, NOT the
      // requested tex+smp size: the slot space is 1-BASED (the translator
      // emplaces a binding before assigning it GetBindlessResourceCount(),
      // which then includes the binding itself), so slots run 1..n and the
      // emulated slot-n write lands one dword past the request, in the
      // 256-byte pool alignment slack. The publish must carry that dword or
      // the redirect DROPS a write the shader reads.
      uint32_t nr_dsc_span_v =
          uint32_t(texture_count_vertex + sampler_count_vertex);
      uint32_t* nr_dsc_dst_v = nullptr;
      if (g_nr_desc) {
        for (size_t i = 0; i < texture_count_vertex; ++i) {
          nr_dsc_span_v = std::max(
              nr_dsc_span_v, textures_vertex[i].bindless_descriptor_index + 1);
        }
        for (size_t i = 0; i < sampler_count_vertex; ++i) {
          nr_dsc_span_v = std::max(
              nr_dsc_span_v, samplers_vertex[i].bindless_descriptor_index + 1);
        }
        if (nr_dsc_span_v <= kNrDescStagingDwords) {
          nr_dsc_dst_v = descriptor_indices;
          std::memset(g_nr_desc_staging, 0, nr_dsc_span_v * sizeof(uint32_t));
          descriptor_indices = g_nr_desc_staging;
        } else {
          ++g_nr_desc_probe.dc_refused;
        }
      }
      for (size_t i = 0; i < texture_count_vertex; ++i) {
        const D3D12Shader::TextureBinding& texture = textures_vertex[i];
        descriptor_indices[texture.bindless_descriptor_index] =
            texture_cache_->GetActiveTextureBindlessSRVIndex(texture) -
            uint32_t(SystemBindlessView::kUnboundedSRVsStart);
      }
      // [NR-RSY] Phase 5-3b-3: the texture SRV index VALUES this rebuild just
      // wrote, ours (map mirror + decision-tree transcription over declared
      // binding facts) vs theirs, per binding.
      if (g_nr_res) {
        for (size_t i = 0; i < texture_count_vertex; ++i) {
          const D3D12Shader::TextureBinding& nr_res_tb = textures_vertex[i];
          ++g_nr_res_probe.srv_checks;
          nr::ResSrvBindingFacts nr_res_facts;
          texture_cache_->NrDescribeActiveBinding(nr_res_tb.fetch_constant, &nr_res_facts);
          uint32_t nr_res_ours = 0;
          const nr::ResSrvOutcome nr_res_out = nr::ResSrvIndexForBinding(
              &g_nr_res_tmap, &nr_res_facts, uint32_t(nr_res_tb.dimension), nr_res_tb.is_signed,
              uint32_t(SystemBindlessView::kNullTexture2DArray),
              uint32_t(SystemBindlessView::kNullTexture3D),
              uint32_t(SystemBindlessView::kNullTextureCube), &nr_res_ours);
          if (nr_res_out == nr::kResSrvRefuseSpecialView) {
            ++g_nr_res_probe.srv_special;
            continue;
          }
          if (nr_res_out == nr::kResSrvRefuseUnknown) {
            ++g_nr_res_probe.srv_unknown;
            continue;
          }
          if (nr_res_out == nr::kResSrvNull) {
            ++g_nr_res_probe.srv_null;
          }
          const uint32_t nr_res_theirs =
              texture_cache_->GetActiveTextureBindlessSRVIndex(nr_res_tb);
          if (nr_res_ours != nr_res_theirs) {
            ++g_nr_res_probe.srv_ne;
            if (g_nr_res_samples_this_window < kNrResMaxSamplesPerWindow) {
              ++g_nr_res_samples_this_window;
              REXGPU_WARN("[nr-rsy] SRV DIFF v i={} fc={} outcome={} ours={} theirs={}", i,
                          nr_res_tb.fetch_constant, uint32_t(nr_res_out), nr_res_ours,
                          nr_res_theirs);
            }
          }
        }
      }
      current_texture_layout_uid_vertex_ = texture_layout_uid_vertex;
      if (texture_count_vertex) {
        current_texture_srv_keys_vertex_.resize(
            std::max(current_texture_srv_keys_vertex_.size(), size_t(texture_count_vertex)));
        texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_vertex_.data(),
                                                  textures_vertex.data(), texture_count_vertex);
      }
      // Current samplers have already been updated.
      for (size_t i = 0; i < sampler_count_vertex; ++i) {
        descriptor_indices[samplers_vertex[i].bindless_descriptor_index] =
            current_sampler_bindless_indices_vertex_[i];
      }
      // [NR-DSC] Publish the staged bytes, then compose our own buffer and
      // byte-compare. Texture SRV index VALUES are pass-through queries (warm
      // - the loop above just resolved them); the placement, the
      // unbounded-SRV base offset and the sampler indices from the mirror
      // are this project's own.
      if (nr_dsc_dst_v) {
        std::memcpy(nr_dsc_dst_v, g_nr_desc_staging,
                    nr_dsc_span_v * sizeof(uint32_t));
        descriptor_indices = nr_dsc_dst_v;
        ++g_nr_desc_probe.dc_v;
        bool nr_dsc_have_smp = true;
        for (size_t i = 0; i < texture_count_vertex; ++i) {
          const D3D12Shader::TextureBinding& texture = textures_vertex[i];
          g_nr_desc_tex_slots[i] = texture.bindless_descriptor_index;
          g_nr_desc_tex_vals[i] =
              texture_cache_->GetActiveTextureBindlessSRVIndex(texture) -
              uint32_t(SystemBindlessView::kUnboundedSRVsStart);
        }
        for (size_t i = 0; i < sampler_count_vertex; ++i) {
          g_nr_desc_smp_slots[i] = samplers_vertex[i].bindless_descriptor_index;
          if (!nr::DescSamplerMapLookup(&g_nr_desc_smap,
                                        current_samplers_vertex_[i].value,
                                        &g_nr_desc_smp_vals[i])) {
            if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
              ++g_nr_desc_samples_this_window;
              REXGPU_WARN(
                  "[nr-dsc] DC REFUSE v smp i={} params={:#x} (map count={})",
                  i, current_samplers_vertex_[i].value, g_nr_desc_smap.count);
            }
            nr_dsc_have_smp = false;
            break;
          }
        }
        if (!nr_dsc_have_smp) {
          ++g_nr_desc_probe.dc_refused;
        } else {
          const uint32_t nr_dsc_composed = nr::DescComposeIndices(
              g_nr_desc_tex_slots, g_nr_desc_tex_vals,
              uint32_t(texture_count_vertex), g_nr_desc_smp_slots,
              g_nr_desc_smp_vals, uint32_t(sampler_count_vertex),
              nr_dsc_span_v, g_nr_desc_ours, kNrDescStagingDwords);
          if (nr_dsc_composed == UINT32_MAX) {
            ++g_nr_desc_probe.dc_refused;
            if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
              ++g_nr_desc_samples_this_window;
              REXGPU_WARN(
                  "[nr-dsc] DC REFUSE v compose span={} tex={} smp={}",
                  nr_dsc_span_v, texture_count_vertex, sampler_count_vertex);
            }
          } else if (nr_dsc_span_v &&
                     std::memcmp(g_nr_desc_ours, g_nr_desc_staging,
                                 nr_dsc_span_v * sizeof(uint32_t)) != 0) {
            ++g_nr_desc_probe.dc_ne_v;
            if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
              ++g_nr_desc_samples_this_window;
              for (uint32_t d = 0; d < nr_dsc_span_v; ++d) {
                if (g_nr_desc_ours[d] != g_nr_desc_staging[d]) {
                  REXGPU_WARN(
                      "[nr-dsc] DC DIFF v dword={}/{} ours={:#x} theirs={:#x}",
                      d, nr_dsc_span_v, g_nr_desc_ours[d], g_nr_desc_staging[d]);
                  break;
                }
              }
            }
          }
        }
      }
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = true;
      current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_DescriptorIndicesVertex);
    }

    if (!cbuffer_binding_descriptor_indices_pixel_.up_to_date) {
      // [NR-BND] Rebuild rate; [NR-DSC] (5-3b-2) checks the contents.
      if (g_nr_bindings) {
        ++g_nr_bind.dip_up;
      }
      uint32_t* descriptor_indices = reinterpret_cast<uint32_t*>(constant_buffer_pool_->Request(
          frame_current_,
          std::max(texture_count_pixel + sampler_count_pixel, size_t(1)) * sizeof(uint32_t),
          D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
          &cbuffer_binding_descriptor_indices_pixel_.address));
      if (!descriptor_indices) {
        return false;
      }
      // [NR-DSC] Same staging redirect + compare as the vertex block (same
      // 1-based-slot SPAN rule; see there).
      uint32_t nr_dsc_span_p =
          uint32_t(texture_count_pixel + sampler_count_pixel);
      uint32_t* nr_dsc_dst_p = nullptr;
      if (g_nr_desc) {
        for (size_t i = 0; i < texture_count_pixel; ++i) {
          nr_dsc_span_p = std::max(
              nr_dsc_span_p, (*textures_pixel)[i].bindless_descriptor_index + 1);
        }
        for (size_t i = 0; i < sampler_count_pixel; ++i) {
          nr_dsc_span_p = std::max(
              nr_dsc_span_p, (*samplers_pixel)[i].bindless_descriptor_index + 1);
        }
        if (nr_dsc_span_p <= kNrDescStagingDwords) {
          nr_dsc_dst_p = descriptor_indices;
          std::memset(g_nr_desc_staging, 0, nr_dsc_span_p * sizeof(uint32_t));
          descriptor_indices = g_nr_desc_staging;
        } else {
          ++g_nr_desc_probe.dc_refused;
        }
      }
      for (size_t i = 0; i < texture_count_pixel; ++i) {
        const D3D12Shader::TextureBinding& texture = (*textures_pixel)[i];
        descriptor_indices[texture.bindless_descriptor_index] =
            texture_cache_->GetActiveTextureBindlessSRVIndex(texture) -
            uint32_t(SystemBindlessView::kUnboundedSRVsStart);
      }
      // [NR-RSY] The pixel half of the SRV index-value gate (see the vertex
      // block).
      if (g_nr_res) {
        for (size_t i = 0; i < texture_count_pixel; ++i) {
          const D3D12Shader::TextureBinding& nr_res_tb = (*textures_pixel)[i];
          ++g_nr_res_probe.srv_checks;
          nr::ResSrvBindingFacts nr_res_facts;
          texture_cache_->NrDescribeActiveBinding(nr_res_tb.fetch_constant, &nr_res_facts);
          uint32_t nr_res_ours = 0;
          const nr::ResSrvOutcome nr_res_out = nr::ResSrvIndexForBinding(
              &g_nr_res_tmap, &nr_res_facts, uint32_t(nr_res_tb.dimension), nr_res_tb.is_signed,
              uint32_t(SystemBindlessView::kNullTexture2DArray),
              uint32_t(SystemBindlessView::kNullTexture3D),
              uint32_t(SystemBindlessView::kNullTextureCube), &nr_res_ours);
          if (nr_res_out == nr::kResSrvRefuseSpecialView) {
            ++g_nr_res_probe.srv_special;
            continue;
          }
          if (nr_res_out == nr::kResSrvRefuseUnknown) {
            ++g_nr_res_probe.srv_unknown;
            continue;
          }
          if (nr_res_out == nr::kResSrvNull) {
            ++g_nr_res_probe.srv_null;
          }
          const uint32_t nr_res_theirs =
              texture_cache_->GetActiveTextureBindlessSRVIndex(nr_res_tb);
          if (nr_res_ours != nr_res_theirs) {
            ++g_nr_res_probe.srv_ne;
            if (g_nr_res_samples_this_window < kNrResMaxSamplesPerWindow) {
              ++g_nr_res_samples_this_window;
              REXGPU_WARN("[nr-rsy] SRV DIFF p i={} fc={} outcome={} ours={} theirs={}", i,
                          nr_res_tb.fetch_constant, uint32_t(nr_res_out), nr_res_ours,
                          nr_res_theirs);
            }
          }
        }
      }
      current_texture_layout_uid_pixel_ = texture_layout_uid_pixel;
      if (texture_count_pixel) {
        current_texture_srv_keys_pixel_.resize(
            std::max(current_texture_srv_keys_pixel_.size(), size_t(texture_count_pixel)));
        texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_pixel_.data(),
                                                  textures_pixel->data(), texture_count_pixel);
      }
      // Current samplers have already been updated.
      for (size_t i = 0; i < sampler_count_pixel; ++i) {
        descriptor_indices[(*samplers_pixel)[i].bindless_descriptor_index] =
            current_sampler_bindless_indices_pixel_[i];
      }
      if (nr_dsc_dst_p) {
        std::memcpy(nr_dsc_dst_p, g_nr_desc_staging,
                    nr_dsc_span_p * sizeof(uint32_t));
        descriptor_indices = nr_dsc_dst_p;
        ++g_nr_desc_probe.dc_p;
        bool nr_dsc_have_smp = true;
        for (size_t i = 0; i < texture_count_pixel; ++i) {
          const D3D12Shader::TextureBinding& texture = (*textures_pixel)[i];
          g_nr_desc_tex_slots[i] = texture.bindless_descriptor_index;
          g_nr_desc_tex_vals[i] =
              texture_cache_->GetActiveTextureBindlessSRVIndex(texture) -
              uint32_t(SystemBindlessView::kUnboundedSRVsStart);
        }
        for (size_t i = 0; i < sampler_count_pixel; ++i) {
          g_nr_desc_smp_slots[i] =
              (*samplers_pixel)[i].bindless_descriptor_index;
          if (!nr::DescSamplerMapLookup(&g_nr_desc_smap,
                                        current_samplers_pixel_[i].value,
                                        &g_nr_desc_smp_vals[i])) {
            if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
              ++g_nr_desc_samples_this_window;
              REXGPU_WARN(
                  "[nr-dsc] DC REFUSE p smp i={} params={:#x} (map count={})",
                  i, current_samplers_pixel_[i].value, g_nr_desc_smap.count);
            }
            nr_dsc_have_smp = false;
            break;
          }
        }
        if (!nr_dsc_have_smp) {
          ++g_nr_desc_probe.dc_refused;
        } else {
          const uint32_t nr_dsc_composed = nr::DescComposeIndices(
              g_nr_desc_tex_slots, g_nr_desc_tex_vals,
              uint32_t(texture_count_pixel), g_nr_desc_smp_slots,
              g_nr_desc_smp_vals, uint32_t(sampler_count_pixel),
              nr_dsc_span_p, g_nr_desc_ours, kNrDescStagingDwords);
          if (nr_dsc_composed == UINT32_MAX) {
            ++g_nr_desc_probe.dc_refused;
            if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
              ++g_nr_desc_samples_this_window;
              REXGPU_WARN(
                  "[nr-dsc] DC REFUSE p compose span={} tex={} smp={}",
                  nr_dsc_span_p, texture_count_pixel, sampler_count_pixel);
            }
          } else if (nr_dsc_span_p &&
                     std::memcmp(g_nr_desc_ours, g_nr_desc_staging,
                                 nr_dsc_span_p * sizeof(uint32_t)) != 0) {
            ++g_nr_desc_probe.dc_ne_p;
            if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
              ++g_nr_desc_samples_this_window;
              for (uint32_t d = 0; d < nr_dsc_span_p; ++d) {
                if (g_nr_desc_ours[d] != g_nr_desc_staging[d]) {
                  REXGPU_WARN(
                      "[nr-dsc] DC DIFF p dword={}/{} ours={:#x} theirs={:#x}",
                      d, nr_dsc_span_p, g_nr_desc_ours[d], g_nr_desc_staging[d]);
                  break;
                }
              }
            }
          }
        }
      }
      cbuffer_binding_descriptor_indices_pixel_.up_to_date = true;
      current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_DescriptorIndicesPixel);
    }

    // [NR-DSC] Our texture SRV keys from the raw fetch dwords, against the
    // stored keys - which at this point provably equal the live binding
    // state (a stale or reordered set forces the rebuild above, which
    // rewrites them). Covers the whole fetch->TextureKey decode, the
    // host-swizzle merge (host format table = declared query) and the
    // swizzled-signs derivation, per draw per binding.
    if (g_nr_desc) {
      const bool nr_dsc_allow_invalid =
          REXCVAR_GET(gpu_allow_invalid_fetch_constants);
      for (uint32_t nr_dsc_pass = 0; nr_dsc_pass < 2; ++nr_dsc_pass) {
        const size_t nr_dsc_count =
            nr_dsc_pass ? texture_count_pixel : texture_count_vertex;
        if (!nr_dsc_count) {
          continue;
        }
        const std::vector<D3D12Shader::TextureBinding>& nr_dsc_bindings =
            nr_dsc_pass ? *textures_pixel : textures_vertex;
        const auto& nr_dsc_stored = nr_dsc_pass ? current_texture_srv_keys_pixel_
                                                : current_texture_srv_keys_vertex_;
        for (size_t i = 0; i < nr_dsc_count; ++i) {
          ++g_nr_desc_probe.key_checks;
          nr::DescTexSrvKey nr_dsc_ours;
          nr::DescTextureSrvKey(
              &nr_bnd_rf[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 +
                         nr_dsc_bindings[i].fetch_constant * 6],
              nr_dsc_allow_invalid, NrDescHostFormatSwizzle, nullptr,
              &nr_dsc_ours);
          if (!(nr_dsc_ours.key[3] & (1u << 1))) {
            ++g_nr_desc_probe.key_invalid;
          }
          const auto& nr_dsc_theirs = nr_dsc_stored[i];
          static_assert(sizeof(nr_dsc_theirs.key) == sizeof(nr_dsc_ours.key));
          if (std::memcmp(&nr_dsc_theirs.key, nr_dsc_ours.key,
                          sizeof(nr_dsc_ours.key)) != 0 ||
              nr_dsc_theirs.host_swizzle != nr_dsc_ours.host_swizzle ||
              nr_dsc_theirs.swizzled_signs != nr_dsc_ours.swizzled_signs) {
            ++g_nr_desc_probe.key_ne;
            if (g_nr_desc_samples_this_window < kNrDescMaxSamplesPerWindow) {
              ++g_nr_desc_samples_this_window;
              uint32_t nr_dsc_tk[4];
              std::memcpy(nr_dsc_tk, &nr_dsc_theirs.key, sizeof(nr_dsc_tk));
              const uint32_t* nr_dsc_fd =
                  &nr_bnd_rf[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 +
                             nr_dsc_bindings[i].fetch_constant * 6];
              REXGPU_WARN(
                  "[nr-dsc] KEY DIFF {} fc={} fetch={:08x} {:08x} {:08x} "
                  "{:08x} {:08x} {:08x} ours={:08x} {:08x} {:08x} {:08x} "
                  "sw={:03x} sg={:02x} theirs={:08x} {:08x} {:08x} {:08x} "
                  "sw={:03x} sg={:02x}",
                  nr_dsc_pass ? "p" : "v", nr_dsc_bindings[i].fetch_constant,
                  nr_dsc_fd[0], nr_dsc_fd[1], nr_dsc_fd[2], nr_dsc_fd[3],
                  nr_dsc_fd[4], nr_dsc_fd[5], nr_dsc_ours.key[0],
                  nr_dsc_ours.key[1], nr_dsc_ours.key[2], nr_dsc_ours.key[3],
                  nr_dsc_ours.host_swizzle, nr_dsc_ours.swizzled_signs,
                  nr_dsc_tk[0], nr_dsc_tk[1], nr_dsc_tk[2], nr_dsc_tk[3],
                  nr_dsc_theirs.host_swizzle, nr_dsc_theirs.swizzled_signs);
            }
          }
        }
      }
    }
  } else {
    //
    // Bindful descriptors path.
    //

    // See what descriptors need to be updated.
    // Samplers have already been checked.
    bool write_textures_vertex =
        texture_count_vertex && (!bindful_textures_written_vertex_ ||
                                 current_texture_layout_uid_vertex_ != texture_layout_uid_vertex ||
                                 !texture_cache_->AreActiveTextureSRVKeysUpToDate(
                                     current_texture_srv_keys_vertex_.data(),
                                     textures_vertex.data(), texture_count_vertex));
    bool write_textures_pixel =
        texture_count_pixel &&
        (!bindful_textures_written_pixel_ ||
         current_texture_layout_uid_pixel_ != texture_layout_uid_pixel ||
         !texture_cache_->AreActiveTextureSRVKeysUpToDate(
             current_texture_srv_keys_pixel_.data(), textures_pixel->data(), texture_count_pixel));
    bool write_samplers_vertex = sampler_count_vertex && !bindful_samplers_written_vertex_;
    bool write_samplers_pixel = sampler_count_pixel && !bindful_samplers_written_pixel_;
    bool edram_rov_used =
        render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;

    // Allocate the descriptors.
    size_t view_count_partial_update = 0;
    if (write_textures_vertex) {
      view_count_partial_update += texture_count_vertex;
    }
    if (write_textures_pixel) {
      view_count_partial_update += texture_count_pixel;
    }
    // Shared memory SRV and null UAV + null SRV and shared memory UAV +
    // textures.
    size_t view_count_full_update = 4 + texture_count_vertex + texture_count_pixel;
    if (edram_rov_used) {
      // + EDRAM UAV in two tables (with the shared memory SRV and with the
      // shared memory UAV).
      view_count_full_update += 2;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE view_cpu_handle;
    D3D12_GPU_DESCRIPTOR_HANDLE view_gpu_handle;
    uint32_t descriptor_size_view = provider.GetViewDescriptorSize();
    uint64_t view_heap_index = RequestViewBindfulDescriptors(
        draw_view_bindful_heap_index_, uint32_t(view_count_partial_update),
        uint32_t(view_count_full_update), view_cpu_handle, view_gpu_handle);
    if (view_heap_index == ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
      REXGPU_ERROR("Failed to allocate view descriptors");
      return false;
    }
    size_t sampler_count_partial_update = 0;
    if (write_samplers_vertex) {
      sampler_count_partial_update += sampler_count_vertex;
    }
    if (write_samplers_pixel) {
      sampler_count_partial_update += sampler_count_pixel;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE sampler_cpu_handle = {};
    D3D12_GPU_DESCRIPTOR_HANDLE sampler_gpu_handle = {};
    uint32_t descriptor_size_sampler = provider.GetSamplerDescriptorSize();
    uint64_t sampler_heap_index = ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
    if (sampler_count_vertex != 0 || sampler_count_pixel != 0) {
      sampler_heap_index = RequestSamplerBindfulDescriptors(
          draw_sampler_bindful_heap_index_, uint32_t(sampler_count_partial_update),
          uint32_t(sampler_count_vertex + sampler_count_pixel), sampler_cpu_handle,
          sampler_gpu_handle);
      if (sampler_heap_index == ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
        REXGPU_ERROR("Failed to allocate sampler descriptors");
        return false;
      }
    }
    if (draw_view_bindful_heap_index_ != view_heap_index) {
      // Need to update all view descriptors.
      write_textures_vertex = texture_count_vertex != 0;
      write_textures_pixel = texture_count_pixel != 0;
      bindful_textures_written_vertex_ = false;
      bindful_textures_written_pixel_ = false;
      // If updating fully, write the shared memory SRV and UAV descriptors and,
      // if needed, the EDRAM descriptor.
      // SRV + null UAV + EDRAM.
      gpu_handle_shared_memory_srv_and_edram_ = view_gpu_handle;
      shared_memory_->WriteRawSRVDescriptor(view_cpu_handle);
      view_cpu_handle.ptr += descriptor_size_view;
      view_gpu_handle.ptr += descriptor_size_view;
      ui::d3d12::util::CreateBufferRawUAV(device, view_cpu_handle, nullptr, 0);
      view_cpu_handle.ptr += descriptor_size_view;
      view_gpu_handle.ptr += descriptor_size_view;
      if (edram_rov_used) {
        render_target_cache_->WriteEdramUintPow2UAVDescriptor(view_cpu_handle, 2);
        view_cpu_handle.ptr += descriptor_size_view;
        view_gpu_handle.ptr += descriptor_size_view;
      }
      // Null SRV + UAV + EDRAM.
      gpu_handle_shared_memory_uav_and_edram_ = view_gpu_handle;
      ui::d3d12::util::CreateBufferRawSRV(device, view_cpu_handle, nullptr, 0);
      view_cpu_handle.ptr += descriptor_size_view;
      view_gpu_handle.ptr += descriptor_size_view;
      shared_memory_->WriteRawUAVDescriptor(view_cpu_handle);
      view_cpu_handle.ptr += descriptor_size_view;
      view_gpu_handle.ptr += descriptor_size_view;
      if (edram_rov_used) {
        render_target_cache_->WriteEdramUintPow2UAVDescriptor(view_cpu_handle, 2);
        view_cpu_handle.ptr += descriptor_size_view;
        view_gpu_handle.ptr += descriptor_size_view;
      }
      current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindful_SharedMemoryAndEdram);
    }
    if (sampler_heap_index != ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid &&
        draw_sampler_bindful_heap_index_ != sampler_heap_index) {
      write_samplers_vertex = sampler_count_vertex != 0;
      write_samplers_pixel = sampler_count_pixel != 0;
      bindful_samplers_written_vertex_ = false;
      bindful_samplers_written_pixel_ = false;
    }

    // Write the descriptors.
    if (write_textures_vertex) {
      assert_true(current_graphics_root_bindful_extras_.textures_vertex !=
                  RootBindfulExtraParameterIndices::kUnavailable);
      gpu_handle_textures_vertex_ = view_gpu_handle;
      for (size_t i = 0; i < texture_count_vertex; ++i) {
        texture_cache_->WriteActiveTextureBindfulSRV(textures_vertex[i], view_cpu_handle);
        view_cpu_handle.ptr += descriptor_size_view;
        view_gpu_handle.ptr += descriptor_size_view;
      }
      current_texture_layout_uid_vertex_ = texture_layout_uid_vertex;
      current_texture_srv_keys_vertex_.resize(
          std::max(current_texture_srv_keys_vertex_.size(), size_t(texture_count_vertex)));
      texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_vertex_.data(),
                                                textures_vertex.data(), texture_count_vertex);
      bindful_textures_written_vertex_ = true;
      current_graphics_root_up_to_date_ &=
          ~(1u << current_graphics_root_bindful_extras_.textures_vertex);
    }
    if (write_textures_pixel) {
      assert_true(current_graphics_root_bindful_extras_.textures_pixel !=
                  RootBindfulExtraParameterIndices::kUnavailable);
      gpu_handle_textures_pixel_ = view_gpu_handle;
      for (size_t i = 0; i < texture_count_pixel; ++i) {
        texture_cache_->WriteActiveTextureBindfulSRV((*textures_pixel)[i], view_cpu_handle);
        view_cpu_handle.ptr += descriptor_size_view;
        view_gpu_handle.ptr += descriptor_size_view;
      }
      current_texture_layout_uid_pixel_ = texture_layout_uid_pixel;
      current_texture_srv_keys_pixel_.resize(
          std::max(current_texture_srv_keys_pixel_.size(), size_t(texture_count_pixel)));
      texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_pixel_.data(),
                                                textures_pixel->data(), texture_count_pixel);
      bindful_textures_written_pixel_ = true;
      current_graphics_root_up_to_date_ &=
          ~(1u << current_graphics_root_bindful_extras_.textures_pixel);
    }
    if (write_samplers_vertex) {
      assert_true(current_graphics_root_bindful_extras_.samplers_vertex !=
                  RootBindfulExtraParameterIndices::kUnavailable);
      gpu_handle_samplers_vertex_ = sampler_gpu_handle;
      for (size_t i = 0; i < sampler_count_vertex; ++i) {
        texture_cache_->WriteSampler(current_samplers_vertex_[i], sampler_cpu_handle);
        sampler_cpu_handle.ptr += descriptor_size_sampler;
        sampler_gpu_handle.ptr += descriptor_size_sampler;
      }
      // Current samplers have already been updated.
      bindful_samplers_written_vertex_ = true;
      current_graphics_root_up_to_date_ &=
          ~(1u << current_graphics_root_bindful_extras_.samplers_vertex);
    }
    if (write_samplers_pixel) {
      assert_true(current_graphics_root_bindful_extras_.samplers_pixel !=
                  RootBindfulExtraParameterIndices::kUnavailable);
      gpu_handle_samplers_pixel_ = sampler_gpu_handle;
      for (size_t i = 0; i < sampler_count_pixel; ++i) {
        texture_cache_->WriteSampler(current_samplers_pixel_[i], sampler_cpu_handle);
        sampler_cpu_handle.ptr += descriptor_size_sampler;
        sampler_gpu_handle.ptr += descriptor_size_sampler;
      }
      // Current samplers have already been updated.
      bindful_samplers_written_pixel_ = true;
      current_graphics_root_up_to_date_ &=
          ~(1u << current_graphics_root_bindful_extras_.samplers_pixel);
    }

    // Wrote new descriptors on the current page.
    draw_view_bindful_heap_index_ = view_heap_index;
    if (sampler_heap_index != ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
      draw_sampler_bindful_heap_index_ = sampler_heap_index;
    }
  }

  // Update the root parameters.
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_fetch_constants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(root_parameter_fetch_constants,
                                                                cbuffer_binding_fetch_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_fetch_constants;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_float_constants_vertex))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        root_parameter_float_constants_vertex, cbuffer_binding_float_vertex_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_float_constants_vertex;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_float_constants_pixel))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        root_parameter_float_constants_pixel, cbuffer_binding_float_pixel_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_float_constants_pixel;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_system_constants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(root_parameter_system_constants,
                                                                cbuffer_binding_system_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_system_constants;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_bool_loop_constants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(root_parameter_bool_loop_constants,
                                                                cbuffer_binding_bool_loop_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_bool_loop_constants;
  }
  if (!(current_graphics_root_up_to_date_ &
        (1u << root_parameter_shared_memory_and_bindful_edram))) {
    assert_true(current_shared_memory_binding_is_uav_.has_value());
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_shared_memory_and_bindful_edram;
    if (bindless_resources_used_) {
      gpu_handle_shared_memory_and_bindful_edram = provider.OffsetViewDescriptor(
          view_bindless_heap_gpu_start_,
          uint32_t(current_shared_memory_binding_is_uav_.value()
                       ? SystemBindlessView ::kNullRawSRVAndSharedMemoryRawUAVStart
                       : SystemBindlessView ::kSharedMemoryRawSRVAndNullRawUAVStart));
    } else {
      gpu_handle_shared_memory_and_bindful_edram = current_shared_memory_binding_is_uav_.value()
                                                       ? gpu_handle_shared_memory_uav_and_edram_
                                                       : gpu_handle_shared_memory_srv_and_edram_;
    }
    deferred_command_list_.D3DSetGraphicsRootDescriptorTable(
        root_parameter_shared_memory_and_bindful_edram, gpu_handle_shared_memory_and_bindful_edram);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_shared_memory_and_bindful_edram;
  }
  if (bindless_resources_used_) {
    if (!(current_graphics_root_up_to_date_ &
          (1u << kRootParameter_Bindless_DescriptorIndicesPixel))) {
      deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
          kRootParameter_Bindless_DescriptorIndicesPixel,
          cbuffer_binding_descriptor_indices_pixel_.address);
      current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_DescriptorIndicesPixel;
    }
    if (!(current_graphics_root_up_to_date_ &
          (1u << kRootParameter_Bindless_DescriptorIndicesVertex))) {
      deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
          kRootParameter_Bindless_DescriptorIndicesVertex,
          cbuffer_binding_descriptor_indices_vertex_.address);
      current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_DescriptorIndicesVertex;
    }
    if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_SamplerHeap))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(kRootParameter_Bindless_SamplerHeap,
                                                               sampler_bindless_heap_gpu_start_);
      current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SamplerHeap;
    }
    if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_ViewHeap))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(kRootParameter_Bindless_ViewHeap,
                                                               view_bindless_heap_gpu_start_);
      current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_ViewHeap;
    }
  } else {
    uint32_t extra_index;
    extra_index = current_graphics_root_bindful_extras_.textures_pixel;
    if (extra_index != RootBindfulExtraParameterIndices::kUnavailable &&
        !(current_graphics_root_up_to_date_ & (1u << extra_index))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(extra_index,
                                                               gpu_handle_textures_pixel_);
      current_graphics_root_up_to_date_ |= 1u << extra_index;
    }
    extra_index = current_graphics_root_bindful_extras_.samplers_pixel;
    if (extra_index != RootBindfulExtraParameterIndices::kUnavailable &&
        !(current_graphics_root_up_to_date_ & (1u << extra_index))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(extra_index,
                                                               gpu_handle_samplers_pixel_);
      current_graphics_root_up_to_date_ |= 1u << extra_index;
    }
    extra_index = current_graphics_root_bindful_extras_.textures_vertex;
    if (extra_index != RootBindfulExtraParameterIndices::kUnavailable &&
        !(current_graphics_root_up_to_date_ & (1u << extra_index))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(extra_index,
                                                               gpu_handle_textures_vertex_);
      current_graphics_root_up_to_date_ |= 1u << extra_index;
    }
    extra_index = current_graphics_root_bindful_extras_.samplers_vertex;
    if (extra_index != RootBindfulExtraParameterIndices::kUnavailable &&
        !(current_graphics_root_up_to_date_ & (1u << extra_index))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(extra_index,
                                                               gpu_handle_samplers_vertex_);
      current_graphics_root_up_to_date_ |= 1u << extra_index;
    }
  }

  return true;
}

bool D3D12CommandProcessor::NrUpdateBindings(const D3D12Shader* vertex_shader,
                                             const D3D12Shader* pixel_shader,
                                             ID3D12RootSignature* root_signature,
                                             bool shared_memory_is_uav, bool* refused_out) {
  ++g_n8c_binds;  // [N8C] the per-draw denominator for the four upload counters.
  // [NR-SWP] Phase 5-3b swap: this project's own UpdateBindings, bindless
  // path only, operating on the SAME member state machine as the emulated
  // one (dirty flags, constant pool, sampler allocator, root-parameter
  // mask), so per-draw fallback stays coherent. Every byte it publishes
  // comes from a derivation this project byte-proved at city load:
  //   system constants   the 5-3b-1 mirror (checked against the emulated
  //                      derivation at the end of every
  //                      UpdateSystemConstantValues call, which still runs)
  //   guest cbuffers     the 5-3b-0 packers (BindCompose*)
  //   sampler params     the 5-3b-2 derivation (DescSamplerParams)
  //   SRV index values   the 5-3b-3 maps + decision tree (per-value
  //                      fallback to the emulated warm query, counted)
  //   root parameters    transcribed from the emulated tail
  // Declared (cache-owned): WriteSampler's params->desc conversion, the
  // SRV-key up-to-date bookkeeping, LoadActiveTextures (runs earlier).
  // The ONLY mid-function refusal is sampler-heap overflow (the heap-switch
  // machinery stays emulated); everything mutated before a refusal is state
  // the emulated retry reads coherently.
  *refused_out = false;
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  const RegisterFile& regs = GetActiveDrawRegisterFile();
  const uint32_t* nr_rf = &regs[0];

  // Set the new root signature.
  if (current_graphics_root_signature_ != root_signature) {
    current_graphics_root_signature_ = root_signature;
    // Changing the root signature invalidates all bindings.
    current_graphics_root_up_to_date_ = 0;
    inst_base_dirty_ = true;  // [hiz-pool]
    deferred_command_list_.D3DSetGraphicsRootSignature(root_signature);
  }

  // [NR-BNDP] sub-bracket 0 (cb): float-map checks + the 5 cbuffer composes.
  auto _bp_cb0 = g_draw_prof ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  // Check if the float constant layout is still the same and get the counts.
  const Shader::ConstantRegisterMap& float_constant_map_vertex =
      vertex_shader->constant_register_map();
  uint32_t float_constant_count_vertex = float_constant_map_vertex.float_count;
  for (uint32_t i = 0; i < 4; ++i) {
    if (current_float_constant_map_vertex_[i] != float_constant_map_vertex.float_bitmap[i]) {
      current_float_constant_map_vertex_[i] = float_constant_map_vertex.float_bitmap[i];
      if (float_constant_count_vertex) {
        cbuffer_binding_float_vertex_.up_to_date = false;
      }
    }
  }
  uint32_t float_constant_count_pixel = 0;
  if (pixel_shader != nullptr) {
    const Shader::ConstantRegisterMap& float_constant_map_pixel =
        pixel_shader->constant_register_map();
    float_constant_count_pixel = float_constant_map_pixel.float_count;
    for (uint32_t i = 0; i < 4; ++i) {
      if (current_float_constant_map_pixel_[i] != float_constant_map_pixel.float_bitmap[i]) {
        current_float_constant_map_pixel_[i] = float_constant_map_pixel.float_bitmap[i];
        if (float_constant_count_pixel) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      }
    }
  } else {
    std::memset(current_float_constant_map_pixel_, 0, sizeof(current_float_constant_map_pixel_));
  }

  // [NR-RUF] 5-4-5-2: the fast path. When the v2 verdict proves this draw's
  // inputs byte-identical to its previous execution THIS frame (the pool
  // ring keeps that execution's uploads alive), restore the binding state
  // machine from the bundle instead of re-deriving: the guest cbuffers and
  // descriptor-indices keep their previous GPU addresses (values proven
  // identical by the 5-4-5-1 gate), sampler params/indices and SRV keys are
  // restored as members. System constants stay LIVE (bin-dependent, sys_ne
  // was 100%), the shared-memory flavor check stays live, the key-freshness
  // check below stays live (a mid-frame texture-cache change then clears the
  // di binding and the existing compose rebuilds it fresh -- the safety
  // valve), and the root tail runs off the restored addresses.
  bool nr_rub_fast_restore = false;
  if (g_nr_rub_fast) {
    uint32_t ruf_key;
    bool ruf_r2, ruf_sf;
    if (NrRuseCurrentDraw(&ruf_key, &ruf_r2, &ruf_sf) &&
        (ruf_r2 || g_ruf_v2b_up) && ruf_sf) {
      const NrRubBundle* ruf_b = NrRubFind(ruf_key);
      if (ruf_b && ruf_b->packs_valid && ruf_b->frame == frame_current_) {
        nr_rub_fast_restore = true;
        ++g_rub_probe.fast;
        NrRufRestoreFromBundle(ruf_b);
      } else {
        ++g_rub_probe.fast_miss;
      }
    }
  }

  // Write the constant buffer data - ours. (Direct writes into the upload
  // pointer are fine: sequential stores, never read back.)
  if (!cbuffer_binding_system_.up_to_date) {
    if (g_draw_prof) ++g_bind_cnt[3];
    if (g_draw_prof) ++g_bind_cnt[2];
    auto _bp_rq0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    uint8_t* system_constants = constant_buffer_pool_->Request(
        frame_current_, sizeof(g_nr_sys_state), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_system_.address);
    if (g_draw_prof) g_bind_ns[6] += prof_ns_since(_bp_rq0);
    if (system_constants == nullptr) {
      return false;
    }
    std::memcpy(system_constants, &g_nr_sys_state, sizeof(g_nr_sys_state));
    cbuffer_binding_system_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_SystemConstants);
  }
  if (!cbuffer_binding_float_vertex_.up_to_date) {
    ++g_n8c_up[0];  // [N8C]
    if (g_draw_prof) ++g_bind_cnt[4];
    const uint32_t float_vertex_size =
        uint32_t(sizeof(float)) * 4 * std::max(float_constant_count_vertex, uint32_t(1));
    if (g_draw_prof) ++g_bind_cnt[2];
    auto _bp_rq0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    uint8_t* float_constants = constant_buffer_pool_->Request(
        frame_current_, float_vertex_size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr,
        nullptr, &cbuffer_binding_float_vertex_.address);
    if (g_draw_prof) g_bind_ns[6] += prof_ns_since(_bp_rq0);
    if (float_constants == nullptr) {
      return false;
    }
    // [NR-RUB] stage in cached memory, publish with one memcpy (the compare
    // gate must never read the upload heap back).
    uint8_t* nr_rub_dst = float_constants;
    if (g_nr_rub_cmp && g_rub_stage_ok) {
      g_rub_stage.flt_v.resize(float_vertex_size);
      nr_rub_dst = g_rub_stage.flt_v.data();
    }
    nr::BindComposeFloats(nr_rf, nr::kBindFloatVertexBase, float_constant_map_vertex.float_bitmap,
                          nr_rub_dst, float_vertex_size);
    if (nr_rub_dst != float_constants) {
      std::memcpy(float_constants, nr_rub_dst, float_vertex_size);
      g_rub_stage.flt_v_bytes = float_vertex_size;
    }
    cbuffer_binding_float_vertex_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_FloatConstantsVertex);
  }
  if (!cbuffer_binding_float_pixel_.up_to_date) {
    ++g_n8c_up[1];  // [N8C]
    if (g_draw_prof) ++g_bind_cnt[5];
    const uint32_t float_pixel_size =
        uint32_t(sizeof(float)) * 4 * std::max(float_constant_count_pixel, uint32_t(1));
    if (g_draw_prof) ++g_bind_cnt[2];
    auto _bp_rq0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    uint8_t* float_constants = constant_buffer_pool_->Request(
        frame_current_, float_pixel_size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr,
        nullptr, &cbuffer_binding_float_pixel_.address);
    if (g_draw_prof) g_bind_ns[6] += prof_ns_since(_bp_rq0);
    if (float_constants == nullptr) {
      return false;
    }
    const uint64_t nr_zero_bitmap[4] = {0, 0, 0, 0};
    uint8_t* nr_rub_dst = float_constants;
    if (g_nr_rub_cmp && g_rub_stage_ok) {
      g_rub_stage.flt_p.resize(float_pixel_size);
      nr_rub_dst = g_rub_stage.flt_p.data();
    }
    nr::BindComposeFloats(
        nr_rf, nr::kBindFloatPixelBase,
        pixel_shader ? pixel_shader->constant_register_map().float_bitmap : nr_zero_bitmap,
        nr_rub_dst, float_pixel_size);
    if (nr_rub_dst != float_constants) {
      std::memcpy(float_constants, nr_rub_dst, float_pixel_size);
      g_rub_stage.flt_p_bytes = float_pixel_size;
    }
    cbuffer_binding_float_pixel_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_FloatConstantsPixel);
  }
  if (!cbuffer_binding_bool_loop_.up_to_date) {
    ++g_n8c_up[2];  // [N8C]
    if (g_draw_prof) ++g_bind_cnt[6];
    if (g_draw_prof) ++g_bind_cnt[2];
    auto _bp_rq0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    uint8_t* bool_loop_constants = constant_buffer_pool_->Request(
        frame_current_, nr::kBindBoolLoopBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_bool_loop_.address);
    if (g_draw_prof) g_bind_ns[6] += prof_ns_since(_bp_rq0);
    if (bool_loop_constants == nullptr) {
      return false;
    }
    uint8_t* nr_rub_dst = bool_loop_constants;
    if (g_nr_rub_cmp && g_rub_stage_ok) {
      g_rub_stage.bl.resize(nr::kBindBoolLoopBytes);
      nr_rub_dst = g_rub_stage.bl.data();
    }
    nr::BindComposeBoolLoop(nr_rf, nr_rub_dst, nr::kBindBoolLoopBytes);
    if (nr_rub_dst != bool_loop_constants) {
      std::memcpy(bool_loop_constants, nr_rub_dst, nr::kBindBoolLoopBytes);
    }
    cbuffer_binding_bool_loop_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_BoolLoopConstants);
  }
  if (!cbuffer_binding_fetch_.up_to_date) {
    ++g_n8c_up[3];  // [N8C]
    if (g_draw_prof) ++g_bind_cnt[7];
    if (g_draw_prof) ++g_bind_cnt[2];
    auto _bp_rq0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    uint8_t* fetch_constants = constant_buffer_pool_->Request(
        frame_current_, nr::kBindFetchBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_fetch_.address);
    if (g_draw_prof) g_bind_ns[6] += prof_ns_since(_bp_rq0);
    if (fetch_constants == nullptr) {
      return false;
    }
    uint8_t* nr_rub_dst = fetch_constants;
    if (g_nr_rub_cmp && g_rub_stage_ok) {
      g_rub_stage.ftc.resize(nr::kBindFetchBytes);
      nr_rub_dst = g_rub_stage.ftc.data();
    }
    nr::BindComposeFetch(nr_rf, nr_rub_dst, nr::kBindFetchBytes);
    if (nr_rub_dst != fetch_constants) {
      std::memcpy(fetch_constants, nr_rub_dst, nr::kBindFetchBytes);
    }
    cbuffer_binding_fetch_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_FetchConstants);
  }

  // The shared-memory binding flavor.
  if (!current_shared_memory_binding_is_uav_.has_value() ||
      current_shared_memory_binding_is_uav_.value() != shared_memory_is_uav) {
    current_shared_memory_binding_is_uav_ = shared_memory_is_uav;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_SharedMemory);
  }
  if (g_draw_prof) g_bind_ns[0] += prof_ns_since(_bp_cb0);

  // [NR-BNDP] sub-bracket 1 (smp): the per-draw sampler-params derivation.
  auto _bp_smp0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  // Sampler parameters - our derivation, same dirty machine.
  const int32_t nr_swp_aniso = REXCVAR_GET(anisotropic_override);
  size_t texture_layout_uid_vertex = vertex_shader->GetTextureBindingLayoutUserUID();
  size_t sampler_layout_uid_vertex = vertex_shader->GetSamplerBindingLayoutUserUID();
  const std::vector<D3D12Shader::TextureBinding>& textures_vertex =
      vertex_shader->GetTextureBindingsAfterTranslation();
  const std::vector<D3D12Shader::SamplerBinding>& samplers_vertex =
      vertex_shader->GetSamplerBindingsAfterTranslation();
  size_t texture_count_vertex = textures_vertex.size();
  size_t sampler_count_vertex = samplers_vertex.size();
  // [NR-RUF] under a fast restore the params/uids/indices are the bundle's
  // (byte-proven identical); the derivation loops are the cost being skipped.
  if (sampler_count_vertex && !nr_rub_fast_restore) {
    if (current_sampler_layout_uid_vertex_ != sampler_layout_uid_vertex) {
      current_sampler_layout_uid_vertex_ = sampler_layout_uid_vertex;
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
      bindful_samplers_written_vertex_ = false;
    }
    current_samplers_vertex_.resize(
        std::max(current_samplers_vertex_.size(), sampler_count_vertex));
    for (size_t i = 0; i < sampler_count_vertex; ++i) {
      if (g_draw_prof) ++g_bind_cnt[0];
      D3D12TextureCache::SamplerParameters parameters;
      parameters.value = nr::DescSamplerParams(
          &nr_rf[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + samplers_vertex[i].fetch_constant * 6],
          uint32_t(samplers_vertex[i].mag_filter), uint32_t(samplers_vertex[i].min_filter),
          uint32_t(samplers_vertex[i].mip_filter), uint32_t(samplers_vertex[i].aniso_filter),
          nr_swp_aniso);
      if (current_samplers_vertex_[i] != parameters) {
        cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
        bindful_samplers_written_vertex_ = false;
        current_samplers_vertex_[i] = parameters;
      }
    }
  }
  size_t texture_layout_uid_pixel, sampler_layout_uid_pixel;
  const std::vector<D3D12Shader::TextureBinding>* textures_pixel;
  const std::vector<D3D12Shader::SamplerBinding>* samplers_pixel;
  size_t texture_count_pixel, sampler_count_pixel;
  if (pixel_shader != nullptr) {
    texture_layout_uid_pixel = pixel_shader->GetTextureBindingLayoutUserUID();
    sampler_layout_uid_pixel = pixel_shader->GetSamplerBindingLayoutUserUID();
    textures_pixel = &pixel_shader->GetTextureBindingsAfterTranslation();
    texture_count_pixel = textures_pixel->size();
    samplers_pixel = &pixel_shader->GetSamplerBindingsAfterTranslation();
    sampler_count_pixel = samplers_pixel->size();
    if (sampler_count_pixel && !nr_rub_fast_restore) {
      if (current_sampler_layout_uid_pixel_ != sampler_layout_uid_pixel) {
        current_sampler_layout_uid_pixel_ = sampler_layout_uid_pixel;
        cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
        bindful_samplers_written_pixel_ = false;
      }
      current_samplers_pixel_.resize(
          std::max(current_samplers_pixel_.size(), size_t(sampler_count_pixel)));
      for (uint32_t i = 0; i < sampler_count_pixel; ++i) {
        if (g_draw_prof) ++g_bind_cnt[0];
        D3D12TextureCache::SamplerParameters parameters;
        parameters.value = nr::DescSamplerParams(
            &nr_rf[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 +
                   (*samplers_pixel)[i].fetch_constant * 6],
            uint32_t((*samplers_pixel)[i].mag_filter), uint32_t((*samplers_pixel)[i].min_filter),
            uint32_t((*samplers_pixel)[i].mip_filter), uint32_t((*samplers_pixel)[i].aniso_filter),
            nr_swp_aniso);
        if (current_samplers_pixel_[i] != parameters) {
          current_samplers_pixel_[i] = parameters;
          cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
          bindful_samplers_written_pixel_ = false;
        }
      }
    }
  } else {
    texture_layout_uid_pixel = PipelineCache::kLayoutUIDEmpty;
    sampler_layout_uid_pixel = PipelineCache::kLayoutUIDEmpty;
    textures_pixel = nullptr;
    texture_count_pixel = 0;
    samplers_pixel = nullptr;
    sampler_count_pixel = 0;
  }

  if (g_draw_prof) g_bind_ns[1] += prof_ns_since(_bp_smp0);

  // [NR-BNDP] sub-bracket 2 (key): srv-key freshness checks.
  auto _bp_key0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  // Texture-key freshness (cache bookkeeping over keys the 5-3b-2 gate
  // proved equal to our derivation).
  if (texture_count_vertex && cbuffer_binding_descriptor_indices_vertex_.up_to_date &&
      (current_texture_layout_uid_vertex_ != texture_layout_uid_vertex ||
       !texture_cache_->AreActiveTextureSRVKeysUpToDate(current_texture_srv_keys_vertex_.data(),
                                                        textures_vertex.data(),
                                                        texture_count_vertex))) {
    cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
  }
  if (texture_count_pixel && cbuffer_binding_descriptor_indices_pixel_.up_to_date &&
      (current_texture_layout_uid_pixel_ != texture_layout_uid_pixel ||
       !texture_cache_->AreActiveTextureSRVKeysUpToDate(current_texture_srv_keys_pixel_.data(),
                                                        textures_pixel->data(),
                                                        texture_count_pixel))) {
    cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
  }

  if (g_draw_prof) g_bind_ns[2] += prof_ns_since(_bp_key0);

  // [NR-BNDP] sub-bracket 3 (smpidx): sampler-heap find-or-allocate. The
  // refusal early-return leaves it open; fallback=0 at city so the loss is
  // nil.
  auto _bp_si0 = g_draw_prof ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  // Sampler heap indices: the shared find-or-allocate; overflow refuses to
  // the emulated path (which owns the heap-switch machinery). Everything
  // allocated up to a refusal stays valid for the retry.
  if ((sampler_count_vertex && !cbuffer_binding_descriptor_indices_vertex_.up_to_date) ||
      (sampler_count_pixel && !cbuffer_binding_descriptor_indices_pixel_.up_to_date)) {
    if (sampler_count_vertex && !cbuffer_binding_descriptor_indices_vertex_.up_to_date) {
      current_sampler_bindless_indices_vertex_.resize(std::max(
          current_sampler_bindless_indices_vertex_.size(), size_t(sampler_count_vertex)));
      for (uint32_t j = 0; j < sampler_count_vertex; ++j) {
        D3D12TextureCache::SamplerParameters sampler_parameters = current_samplers_vertex_[j];
        uint32_t sampler_index;
        auto it = texture_cache_bindless_sampler_map_.find(sampler_parameters.value);
        if (it != texture_cache_bindless_sampler_map_.end()) {
          sampler_index = it->second;
        } else {
          if (sampler_bindless_heap_allocated_ >= kSamplerHeapSize) {
            *refused_out = true;
            return true;
          }
          sampler_index = sampler_bindless_heap_allocated_++;
          texture_cache_->WriteSampler(sampler_parameters,
                                       provider.OffsetSamplerDescriptor(
                                           sampler_bindless_heap_cpu_start_, sampler_index));
          texture_cache_bindless_sampler_map_.emplace(sampler_parameters.value, sampler_index);
          ++g_nr_swap_probe.smp_alloc;
        }
        current_sampler_bindless_indices_vertex_[j] = sampler_index;
      }
    }
    if (sampler_count_pixel && !cbuffer_binding_descriptor_indices_pixel_.up_to_date) {
      current_sampler_bindless_indices_pixel_.resize(std::max(
          current_sampler_bindless_indices_pixel_.size(), size_t(sampler_count_pixel)));
      for (uint32_t j = 0; j < sampler_count_pixel; ++j) {
        D3D12TextureCache::SamplerParameters sampler_parameters = current_samplers_pixel_[j];
        uint32_t sampler_index;
        auto it = texture_cache_bindless_sampler_map_.find(sampler_parameters.value);
        if (it != texture_cache_bindless_sampler_map_.end()) {
          sampler_index = it->second;
        } else {
          if (sampler_bindless_heap_allocated_ >= kSamplerHeapSize) {
            *refused_out = true;
            return true;
          }
          sampler_index = sampler_bindless_heap_allocated_++;
          texture_cache_->WriteSampler(sampler_parameters,
                                       provider.OffsetSamplerDescriptor(
                                           sampler_bindless_heap_cpu_start_, sampler_index));
          texture_cache_bindless_sampler_map_.emplace(sampler_parameters.value, sampler_index);
          ++g_nr_swap_probe.smp_alloc;
        }
        current_sampler_bindless_indices_pixel_[j] = sampler_index;
      }
    }
  }

  if (g_draw_prof) g_bind_ns[3] += prof_ns_since(_bp_si0);

  // Our texture SRV index value for one binding (5-3b-3's map + decision
  // tree; the emulated warm query only as a counted per-value fallback).
  const auto nr_swp_srv_value = [this](const D3D12Shader::TextureBinding& binding) -> uint32_t {
    if (g_draw_prof) ++g_bind_cnt[1];
    nr::ResSrvBindingFacts facts;
    texture_cache_->NrDescribeActiveBinding(binding.fetch_constant, &facts);
    uint32_t ours = 0;
    const nr::ResSrvOutcome outcome = nr::ResSrvIndexForBinding(
        &g_nr_res_tmap, &facts, uint32_t(binding.dimension), binding.is_signed,
        uint32_t(SystemBindlessView::kNullTexture2DArray),
        uint32_t(SystemBindlessView::kNullTexture3D),
        uint32_t(SystemBindlessView::kNullTextureCube), &ours);
    if (outcome == nr::kResSrvValue || outcome == nr::kResSrvNull) {
      return ours;
    }
    ++g_nr_swap_probe.srv_query;
    return texture_cache_->GetActiveTextureBindlessSRVIndex(binding);
  };

  // [NR-BNDP] sub-bracket 4 (di): both descriptor-indices composes.
  auto _bp_di0 = g_draw_prof ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  // Descriptor-indices constant buffers - ours, sized by the 1-BASED slot
  // rule (max_slot + 1, never tex+smp: [[descriptor-slots-one-based]]).
  if (!cbuffer_binding_descriptor_indices_vertex_.up_to_date) {
    uint32_t span = 0;
    for (size_t i = 0; i < texture_count_vertex; ++i) {
      span = std::max(span, textures_vertex[i].bindless_descriptor_index + 1);
    }
    for (size_t i = 0; i < sampler_count_vertex; ++i) {
      span = std::max(span, samplers_vertex[i].bindless_descriptor_index + 1);
    }
    const uint32_t span_alloc = std::max(span, uint32_t(1));
    if (g_draw_prof) ++g_bind_cnt[2];
    auto _bp_rq0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    uint32_t* descriptor_indices = reinterpret_cast<uint32_t*>(constant_buffer_pool_->Request(
        frame_current_, span_alloc * sizeof(uint32_t),
        D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
        &cbuffer_binding_descriptor_indices_vertex_.address));
    if (g_draw_prof) g_bind_ns[6] += prof_ns_since(_bp_rq0);
    if (!descriptor_indices) {
      return false;
    }
    // [NR-RUB] staging redirect: full-span publish (the 1-based slot rule
    // means span_alloc already covers the last slot -- nothing is dropped).
    uint32_t* nr_rub_di = descriptor_indices;
    if (g_nr_rub_cmp && g_rub_stage_ok) {
      g_rub_stage.di_v.resize(span_alloc);
      nr_rub_di = g_rub_stage.di_v.data();
    }
    std::memset(nr_rub_di, 0, span_alloc * sizeof(uint32_t));
    for (size_t i = 0; i < texture_count_vertex; ++i) {
      const D3D12Shader::TextureBinding& texture = textures_vertex[i];
      nr_rub_di[texture.bindless_descriptor_index] =
          nr_swp_srv_value(texture) - uint32_t(SystemBindlessView::kUnboundedSRVsStart);
    }
    for (size_t i = 0; i < sampler_count_vertex; ++i) {
      nr_rub_di[samplers_vertex[i].bindless_descriptor_index] =
          current_sampler_bindless_indices_vertex_[i];
    }
    if (nr_rub_di != descriptor_indices) {
      std::memcpy(descriptor_indices, nr_rub_di, span_alloc * sizeof(uint32_t));
    }
    ++g_nr_swap_probe.di_v;
    current_texture_layout_uid_vertex_ = texture_layout_uid_vertex;
    if (texture_count_vertex) {
      current_texture_srv_keys_vertex_.resize(
          std::max(current_texture_srv_keys_vertex_.size(), size_t(texture_count_vertex)));
      texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_vertex_.data(),
                                                textures_vertex.data(), texture_count_vertex);
    }
    cbuffer_binding_descriptor_indices_vertex_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_DescriptorIndicesVertex);
  }
  if (!cbuffer_binding_descriptor_indices_pixel_.up_to_date) {
    uint32_t span = 0;
    for (size_t i = 0; i < texture_count_pixel; ++i) {
      span = std::max(span, (*textures_pixel)[i].bindless_descriptor_index + 1);
    }
    for (size_t i = 0; i < sampler_count_pixel; ++i) {
      span = std::max(span, (*samplers_pixel)[i].bindless_descriptor_index + 1);
    }
    const uint32_t span_alloc = std::max(span, uint32_t(1));
    if (g_draw_prof) ++g_bind_cnt[2];
    auto _bp_rq0 = g_draw_prof ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    uint32_t* descriptor_indices = reinterpret_cast<uint32_t*>(constant_buffer_pool_->Request(
        frame_current_, span_alloc * sizeof(uint32_t),
        D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
        &cbuffer_binding_descriptor_indices_pixel_.address));
    if (g_draw_prof) g_bind_ns[6] += prof_ns_since(_bp_rq0);
    if (!descriptor_indices) {
      return false;
    }
    uint32_t* nr_rub_di = descriptor_indices;
    if (g_nr_rub_cmp && g_rub_stage_ok) {
      g_rub_stage.di_p.resize(span_alloc);
      nr_rub_di = g_rub_stage.di_p.data();
    }
    std::memset(nr_rub_di, 0, span_alloc * sizeof(uint32_t));
    for (size_t i = 0; i < texture_count_pixel; ++i) {
      const D3D12Shader::TextureBinding& texture = (*textures_pixel)[i];
      nr_rub_di[texture.bindless_descriptor_index] =
          nr_swp_srv_value(texture) - uint32_t(SystemBindlessView::kUnboundedSRVsStart);
    }
    for (size_t i = 0; i < sampler_count_pixel; ++i) {
      nr_rub_di[(*samplers_pixel)[i].bindless_descriptor_index] =
          current_sampler_bindless_indices_pixel_[i];
    }
    if (nr_rub_di != descriptor_indices) {
      std::memcpy(descriptor_indices, nr_rub_di, span_alloc * sizeof(uint32_t));
    }
    ++g_nr_swap_probe.di_p;
    current_texture_layout_uid_pixel_ = texture_layout_uid_pixel;
    if (texture_count_pixel) {
      current_texture_srv_keys_pixel_.resize(
          std::max(current_texture_srv_keys_pixel_.size(), size_t(texture_count_pixel)));
      texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_pixel_.data(),
                                                textures_pixel->data(), texture_count_pixel);
    }
    cbuffer_binding_descriptor_indices_pixel_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_DescriptorIndicesPixel);
  }
  if (g_draw_prof) g_bind_ns[4] += prof_ns_since(_bp_di0);

  // [NR-BNDP] sub-bracket 5 (root): the root-parameter tail.
  auto _bp_root0 = g_draw_prof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  // Update the root parameters (the transcribed bindless tail).
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_FetchConstants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_FetchConstants, cbuffer_binding_fetch_.address);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_FetchConstants;
  }
  if (!(current_graphics_root_up_to_date_ &
        (1u << kRootParameter_Bindless_FloatConstantsVertex))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_FloatConstantsVertex, cbuffer_binding_float_vertex_.address);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_FloatConstantsVertex;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_FloatConstantsPixel))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_FloatConstantsPixel, cbuffer_binding_float_pixel_.address);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_FloatConstantsPixel;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_SystemConstants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_SystemConstants, cbuffer_binding_system_.address);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SystemConstants;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_BoolLoopConstants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_BoolLoopConstants, cbuffer_binding_bool_loop_.address);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_BoolLoopConstants;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_SharedMemory))) {
    assert_true(current_shared_memory_binding_is_uav_.has_value());
    deferred_command_list_.D3DSetGraphicsRootDescriptorTable(
        kRootParameter_Bindless_SharedMemory,
        provider.OffsetViewDescriptor(
            view_bindless_heap_gpu_start_,
            uint32_t(current_shared_memory_binding_is_uav_.value()
                         ? SystemBindlessView::kNullRawSRVAndSharedMemoryRawUAVStart
                         : SystemBindlessView::kSharedMemoryRawSRVAndNullRawUAVStart)));
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SharedMemory;
  }
  if (!(current_graphics_root_up_to_date_ &
        (1u << kRootParameter_Bindless_DescriptorIndicesPixel))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_DescriptorIndicesPixel,
        cbuffer_binding_descriptor_indices_pixel_.address);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_DescriptorIndicesPixel;
  }
  if (!(current_graphics_root_up_to_date_ &
        (1u << kRootParameter_Bindless_DescriptorIndicesVertex))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_DescriptorIndicesVertex,
        cbuffer_binding_descriptor_indices_vertex_.address);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_DescriptorIndicesVertex;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_SamplerHeap))) {
    deferred_command_list_.D3DSetGraphicsRootDescriptorTable(kRootParameter_Bindless_SamplerHeap,
                                                             sampler_bindless_heap_gpu_start_);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SamplerHeap;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_ViewHeap))) {
    deferred_command_list_.D3DSetGraphicsRootDescriptorTable(kRootParameter_Bindless_ViewHeap,
                                                             view_bindless_heap_gpu_start_);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_ViewHeap;
  }
  if (g_draw_prof) g_bind_ns[5] += prof_ns_since(_bp_root0);

  // [NR-RUB] 5-4-5-1: bundle compare + capture. The staging mirrors hold
  // this draw's effective packs (every compose since arm staged first);
  // sampler params / heap indices / root signature are CPU members. When the
  // base-side v2 verdict says this execution's inputs are byte-identical to
  // the draw's previous one, every derived output must byte-match the stored
  // bundle -- any ne names an input the reuse model missed. System constants
  // are counted separately (bin-dependent fields expected to differ across
  // tile repeats; whether they actually do is a fast-path design datum).
  if (g_nr_rub) {
    ++g_rub_probe.draws;
    const bool rub_cmp_live = g_nr_rub_cmp && g_rub_stage_ok;
    if (nr_rub_fast_restore) {
      // [NR-RUF] nothing was derived; the bundle already holds this state.
    } else if (g_nr_rub_cmp && !g_rub_stage_ok) {
      ++g_rub_probe.stage_off;
    } else {
      uint32_t rub_key;
      bool rub_r2, rub_sf;
      if (!NrRuseCurrentDraw(&rub_key, &rub_r2, &rub_sf)) {
        ++g_rub_probe.nostop;
      } else {
        const uint64_t rub_sys_hash =
            rub_cmp_live ? XXH3_64bits(&g_nr_sys_state, sizeof(g_nr_sys_state))
                         : 0;
        NrRubBundle* rub_found = NrRubFind(rub_key);
        if ((rub_r2 || g_ruf_v2b_up) && rub_cmp_live) {
          if (!rub_found || !rub_found->packs_have_bytes) {
            ++g_rub_probe.nobundle;
          } else {
            const NrRubBundle& b = *rub_found;
            ++g_rub_probe.checked;
            if (b.flt_v != g_rub_stage.flt_v) ++g_rub_probe.ne_fltv;
            if (b.flt_p != g_rub_stage.flt_p) ++g_rub_probe.ne_fltp;
            if (b.bl != g_rub_stage.bl) ++g_rub_probe.ne_bl;
            if (b.ftc != g_rub_stage.ftc) ++g_rub_probe.ne_ftc;
            if (b.di_v != g_rub_stage.di_v) ++g_rub_probe.ne_div;
            if (b.di_p != g_rub_stage.di_p) ++g_rub_probe.ne_dip;
            bool smp_v_eq = b.smp_v.size() == sampler_count_vertex;
            for (size_t i = 0; smp_v_eq && i < sampler_count_vertex; ++i) {
              smp_v_eq = b.smp_v[i] == current_samplers_vertex_[i].value;
            }
            if (!smp_v_eq) ++g_rub_probe.ne_smpv;
            bool smp_p_eq = b.smp_p.size() == sampler_count_pixel;
            for (size_t i = 0; smp_p_eq && i < sampler_count_pixel; ++i) {
              smp_p_eq = b.smp_p[i] == current_samplers_pixel_[i].value;
            }
            if (!smp_p_eq) ++g_rub_probe.ne_smpp;
            bool si_v_eq = b.si_v.size() == sampler_count_vertex;
            for (size_t i = 0; si_v_eq && i < sampler_count_vertex; ++i) {
              si_v_eq = b.si_v[i] == current_sampler_bindless_indices_vertex_[i];
            }
            if (!si_v_eq) ++g_rub_probe.ne_siv;
            bool si_p_eq = b.si_p.size() == sampler_count_pixel;
            for (size_t i = 0; si_p_eq && i < sampler_count_pixel; ++i) {
              si_p_eq = b.si_p[i] == current_sampler_bindless_indices_pixel_[i];
            }
            if (!si_p_eq) ++g_rub_probe.ne_sip;
            if (b.rootsig != static_cast<void*>(root_signature)) {
              ++g_rub_probe.ne_rootsig;
            }
            if (b.sys_hash == rub_sys_hash) {
              ++g_rub_probe.sys_eq;
            } else {
              ++g_rub_probe.sys_ne;
            }
          }
        }
        // Capture/refresh: the bundle always holds the LATEST execution.
        // The pack/di BYTE copies serve only the compare gate; the fast
        // path's restore state is the small tail below.
        NrRubBundle& nb = rub_found ? *rub_found : *NrRubGetOrCreate(rub_key);
        ++g_rub_probe.captured;
        if (rub_cmp_live) {
          nb.flt_v = g_rub_stage.flt_v;
          nb.flt_p = g_rub_stage.flt_p;
          nb.bl = g_rub_stage.bl;
          nb.ftc = g_rub_stage.ftc;
          nb.di_v = g_rub_stage.di_v;
          nb.di_p = g_rub_stage.di_p;
          nb.sys_hash = rub_sys_hash;
          nb.packs_have_bytes = true;
        }
        nb.smp_v.resize(sampler_count_vertex);
        for (size_t i = 0; i < sampler_count_vertex; ++i) {
          nb.smp_v[i] = current_samplers_vertex_[i].value;
        }
        nb.smp_p.resize(sampler_count_pixel);
        for (size_t i = 0; i < sampler_count_pixel; ++i) {
          nb.smp_p[i] = current_samplers_pixel_[i].value;
        }
        nb.si_v.resize(sampler_count_vertex);
        for (size_t i = 0; i < sampler_count_vertex; ++i) {
          nb.si_v[i] = current_sampler_bindless_indices_vertex_[i];
        }
        nb.si_p.resize(sampler_count_pixel);
        for (size_t i = 0; i < sampler_count_pixel; ++i) {
          nb.si_p[i] = current_sampler_bindless_indices_pixel_[i];
        }
        nb.rootsig = static_cast<void*>(root_signature);
        // [NR-RUF] the restore state: same-frame GPU addresses + the member
        // state a fast restore rebuilds.
        nb.frame = frame_current_;
        nb.a_flt_v = cbuffer_binding_float_vertex_.address;
        nb.a_flt_p = cbuffer_binding_float_pixel_.address;
        nb.a_bl = cbuffer_binding_bool_loop_.address;
        nb.a_ftc = cbuffer_binding_fetch_.address;
        nb.a_di_v = cbuffer_binding_descriptor_indices_vertex_.address;
        nb.a_di_p = cbuffer_binding_descriptor_indices_pixel_.address;
        nb.smp_uid_v = current_sampler_layout_uid_vertex_;
        nb.smp_uid_p = current_sampler_layout_uid_pixel_;
        nb.tex_uid_v = current_texture_layout_uid_vertex_;
        nb.tex_uid_p = current_texture_layout_uid_pixel_;
        if (current_texture_srv_keys_vertex_.size() >= texture_count_vertex &&
            current_texture_srv_keys_pixel_.size() >= texture_count_pixel) {
          nb.keys_v.assign(
              current_texture_srv_keys_vertex_.begin(),
              current_texture_srv_keys_vertex_.begin() + texture_count_vertex);
          nb.keys_p.assign(
              current_texture_srv_keys_pixel_.begin(),
              current_texture_srv_keys_pixel_.begin() + texture_count_pixel);
          nb.packs_valid = true;
        } else {
          nb.packs_valid = false;
        }
      }
    }
  }

  return true;
}

void D3D12CommandProcessor::EvictOldReadbackBuffers(
    std::unordered_map<uint64_t, ReadbackBuffer>& buffer_map) {
  if (buffer_map.empty()) {
    return;
  }
  const uint64_t eviction_frame_floor = (frame_current_ > kReadbackBufferEvictionAgeFrames)
                                            ? (frame_current_ - kReadbackBufferEvictionAgeFrames)
                                            : 0;
  for (auto it = buffer_map.begin(); it != buffer_map.end();) {
    ReadbackBuffer& readback = it->second;
    bool evict =
        buffer_map.size() > kMaxReadbackBuffers || readback.last_used_frame < eviction_frame_floor;
    if (!evict) {
      ++it;
      continue;
    }
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.buffers[i]) {
        if (readback.mapped_data[i]) {
          readback.buffers[i]->Unmap(0, nullptr);
        }
        readback.buffers[i]->Release();
      }
      readback.buffers[i] = nullptr;
      readback.mapped_data[i] = nullptr;
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
    it = buffer_map.erase(it);
  }
}

ID3D12Resource* D3D12CommandProcessor::RequestReadbackBuffer(uint32_t size) {
  if (size == 0) {
    return nullptr;
  }
  size = rex::align(size, kReadbackBufferSizeIncrement);
  if (size > readback_buffer_size_) {
    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, size, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* buffer;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) {
      REXGPU_ERROR("Failed to create a {} MB readback buffer", size >> 20);
      return nullptr;
    }
    if (readback_buffer_ != nullptr) {
      readback_buffer_->Release();
    }
    readback_buffer_ = buffer;
    readback_buffer_size_ = size;
  }
  return readback_buffer_;
}

// [gpu-census] N-10c: price the GPU second by work class. See the header
// for the design; report format:
//   [gpu-census] gpu <busy>ms/s | draw a/n xfer b/n resolve c/n texup d/n
//   memup e/n swap f/n other g/n | trunc T drop D
// where ms is GPU time inside recorded submissions and n is span count.
bool D3D12CommandProcessor::InitializeGpuCensusResources() {
  gpu_census_available_ = false;
  gpu_census_sub_active_ = false;
  gpu_census_pending_.clear();
  gpu_census_ring_next_ = 0;
  gpu_census_stack_depth_ = 0;
  gpu_census_trunc_ = 0;
  gpu_census_dropped_ = 0;
  std::memset(gpu_census_class_ticks_, 0, sizeof(gpu_census_class_ticks_));
  std::memset(gpu_census_class_spans_, 0, sizeof(gpu_census_class_spans_));
  gpu_census_heap_.Reset();
  gpu_census_readback_.Reset();
  gpu_census_readback_mapping_ = nullptr;
  gpu_census_draw_class_ = kGpuCensusDraw;
  gpu_census_draw_config_count_ = 0;
  std::memset(gpu_census_last_rt_keys_, 0xFF, sizeof(gpu_census_last_rt_keys_));
  if (!REXCVAR_GET(d3d12_gpu_census)) {
    return false;
  }
  ID3D12Device* device = GetD3D12Provider().GetDevice();
  ID3D12CommandQueue* direct_queue = GetD3D12Provider().GetDirectQueue();
  if (!device || !direct_queue) {
    return false;
  }
  uint64_t timestamp_frequency = 0;
  if (FAILED(direct_queue->GetTimestampFrequency(&timestamp_frequency)) || !timestamp_frequency) {
    REXGPU_WARN("[gpu-census] no timestamp frequency on the direct queue - census off");
    return false;
  }
  gpu_census_ticks_to_ms_ = 1000.0 / double(timestamp_frequency);
  D3D12_QUERY_HEAP_DESC heap_desc;
  heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
  heap_desc.Count = kGpuCensusQueryPoolSize;
  heap_desc.NodeMask = 0;
  if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&gpu_census_heap_)))) {
    REXGPU_WARN("[gpu-census] timestamp query heap creation failed - census off");
    return false;
  }
  D3D12_RESOURCE_DESC buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(buffer_desc, sizeof(uint64_t) * kGpuCensusQueryPoolSize,
                                          D3D12_RESOURCE_FLAG_NONE);
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesReadback,
                                             GetD3D12Provider().GetHeapFlagCreateNotZeroed(),
                                             &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&gpu_census_readback_)))) {
    REXGPU_WARN("[gpu-census] readback buffer allocation failed - census off");
    gpu_census_heap_.Reset();
    return false;
  }
  D3D12_RANGE read_range = {0, sizeof(uint64_t) * kGpuCensusQueryPoolSize};
  void* mapping = nullptr;
  if (FAILED(gpu_census_readback_->Map(0, &read_range, &mapping))) {
    REXGPU_WARN("[gpu-census] readback buffer map failed - census off");
    gpu_census_readback_.Reset();
    gpu_census_heap_.Reset();
    return false;
  }
  gpu_census_readback_mapping_ = reinterpret_cast<const uint64_t*>(mapping);
  gpu_census_available_ = true;
  REXGPU_INFO("[gpu-census] on: {} timestamp slots, tick {:.1f} ns", kGpuCensusQueryPoolSize,
              gpu_census_ticks_to_ms_ * 1e6);
  return true;
}

void D3D12CommandProcessor::ShutdownGpuCensusResources() {
  gpu_census_available_ = false;
  gpu_census_sub_active_ = false;
  gpu_census_pending_.clear();
  if (gpu_census_readback_ && gpu_census_readback_mapping_) {
    gpu_census_readback_->Unmap(0, nullptr);
  }
  gpu_census_readback_mapping_ = nullptr;
  gpu_census_readback_.Reset();
  gpu_census_heap_.Reset();
}

void D3D12CommandProcessor::GpuCensusEmitTimestamp(uint8_t closing_tag) {
  if (!gpu_census_sub_active_) {
    return;
  }
  if (gpu_census_sub_count_ >= kGpuCensusMaxPerSubmission) {
    ++gpu_census_trunc_;
    return;
  }
  gpu_census_sub_tags_[gpu_census_sub_count_] = closing_tag;
  deferred_command_list_.D3DEndQuery(gpu_census_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                     gpu_census_sub_start_ + gpu_census_sub_count_);
  ++gpu_census_sub_count_;
}

void D3D12CommandProcessor::GpuCensusBeginSubmission() {
  if (!gpu_census_available_) {
    return;
  }
  // Claim a contiguous ring region; wrap early rather than split. A region
  // that would overwrite a still-pending readback means the consumer is
  // very far behind - drop this submission's census and count it.
  uint32_t start = gpu_census_ring_next_;
  if (start + kGpuCensusMaxPerSubmission > kGpuCensusQueryPoolSize) {
    start = 0;
  }
  bool overlap = gpu_census_pending_.size() >= 24;
  if (!overlap) {
    uint32_t region_end = start + kGpuCensusMaxPerSubmission;
    for (const GpuCensusPending& pending : gpu_census_pending_) {
      uint32_t pending_end = pending.start + pending.count;
      if (start < pending_end && pending.start < region_end) {
        overlap = true;
        break;
      }
    }
  }
  if (overlap) {
    ++gpu_census_dropped_;
    return;
  }
  gpu_census_sub_start_ = start;
  gpu_census_sub_count_ = 0;
  gpu_census_class_ = kGpuCensusOther;
  gpu_census_stack_depth_ = 0;
  gpu_census_sub_active_ = true;
  GpuCensusEmitTimestamp(kGpuCensusOther);  // ts0; its tag is never read
}

void D3D12CommandProcessor::GpuCensusCloseSubmission() {
  if (!gpu_census_sub_active_) {
    return;
  }
  // Close the final span. Must run while commands still land in the
  // deferred list, i.e. before its Execute in EndSubmission.
  GpuCensusEmitTimestamp(gpu_census_class_);
}

void D3D12CommandProcessor::GpuCensusResolveSubmission() {
  if (!gpu_census_sub_active_) {
    return;
  }
  gpu_census_sub_active_ = false;
  if (gpu_census_sub_count_ < 2) {
    return;  // no spans recorded - leave the ring cursor alone
  }
  // Recorded directly on the real command list, after the deferred replay
  // (the timestamps are already in) and before Close.
  command_list_->ResolveQueryData(gpu_census_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                  gpu_census_sub_start_, gpu_census_sub_count_,
                                  gpu_census_readback_.Get(),
                                  uint64_t(gpu_census_sub_start_) * sizeof(uint64_t));
  GpuCensusPending& pending = gpu_census_pending_.emplace_back();
  pending.submission = submission_current_;
  pending.start = gpu_census_sub_start_;
  pending.count = gpu_census_sub_count_;
  std::memcpy(pending.tags, gpu_census_sub_tags_, gpu_census_sub_count_);
  gpu_census_ring_next_ = gpu_census_sub_start_ + gpu_census_sub_count_;
}

void D3D12CommandProcessor::GpuCensusConsumeCompleted() {
  while (!gpu_census_pending_.empty() &&
         gpu_census_pending_.front().submission <= submission_completed_) {
    const GpuCensusPending& pending = gpu_census_pending_.front();
    const uint64_t* timestamps = gpu_census_readback_mapping_ + pending.start;
    for (uint32_t i = 1; i < pending.count; ++i) {
      uint64_t dt = timestamps[i] >= timestamps[i - 1] ? timestamps[i] - timestamps[i - 1] : 0;
      uint8_t cls = pending.tags[i] < kGpuCensusTotalClasses ? pending.tags[i]
                                                             : uint8_t(kGpuCensusOther);
      gpu_census_class_ticks_[cls] += dt;
      ++gpu_census_class_spans_[cls];
    }
    gpu_census_pending_.pop_front();
  }
}

// [occ] the occlusion census. Per issued draw: one PIPELINE_STATISTICS query
// (IA primitives, VS invocations, clipper in/out, PS invocations) and one
// OCCLUSION query (samples that passed depth/stencil at draw time). The
// occlusion half is skipped while a guest occlusion query is open (two
// queries of one type may not overlap on a command list). Report format:
//   [occ] q/fr trunc drop | /fr prim vsinv clipout(%) psinv(x720p) smp |
//   hidden(smp0) draws% prim% | clipped(cprim0) draws% prim% |
//   by smp 0/<=64/<=1k/<=16k/16k+ draws% prim% ps% | main [cfg] ...
bool D3D12CommandProcessor::InitializeOccCensusResources() {
  occ_available_ = false;
  occ_sub_active_ = false;
  occ_draw_open_ = false;
  occ_pending_.clear();
  occ_ring_next_ = 0;
  occ_trunc_ = 0;
  occ_dropped_ = 0;
  occ_acc_ = OccAcc{};
  occ_stats_heap_.Reset();
  occ_smp_heap_.Reset();
  occ_stats_readback_.Reset();
  occ_smp_readback_.Reset();
  occ_stats_mapping_ = nullptr;
  occ_smp_mapping_ = nullptr;
  if (!REXCVAR_GET(gpu_occ_census)) {
    return false;
  }
  ID3D12Device* device = GetD3D12Provider().GetDevice();
  if (!device) {
    return false;
  }
  D3D12_QUERY_HEAP_DESC heap_desc;
  heap_desc.Type = D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;
  heap_desc.Count = kOccPoolSize;
  heap_desc.NodeMask = 0;
  if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&occ_stats_heap_)))) {
    REXGPU_WARN("[occ] pipeline-statistics query heap creation failed - census off");
    return false;
  }
  heap_desc.Type = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
  if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&occ_smp_heap_)))) {
    REXGPU_WARN("[occ] occlusion query heap creation failed - census off");
    occ_stats_heap_.Reset();
    return false;
  }
  auto make_readback = [&](uint64_t bytes, Microsoft::WRL::ComPtr<ID3D12Resource>& out,
                           const void** mapping_out) {
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, bytes, D3D12_RESOURCE_FLAG_NONE);
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback,
            GetD3D12Provider().GetHeapFlagCreateNotZeroed(), &buffer_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&out)))) {
      return false;
    }
    D3D12_RANGE read_range = {0, bytes};
    void* mapping = nullptr;
    if (FAILED(out->Map(0, &read_range, &mapping))) {
      out.Reset();
      return false;
    }
    *mapping_out = mapping;
    return true;
  };
  const void* stats_map = nullptr;
  const void* smp_map = nullptr;
  if (!make_readback(sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS) * kOccPoolSize,
                     occ_stats_readback_, &stats_map) ||
      !make_readback(sizeof(uint64_t) * kOccPoolSize, occ_smp_readback_, &smp_map)) {
    REXGPU_WARN("[occ] readback buffer allocation failed - census off");
    if (occ_stats_readback_ && stats_map) occ_stats_readback_->Unmap(0, nullptr);
    occ_stats_readback_.Reset();
    occ_smp_readback_.Reset();
    occ_stats_heap_.Reset();
    occ_smp_heap_.Reset();
    return false;
  }
  occ_stats_mapping_ = reinterpret_cast<const D3D12_QUERY_DATA_PIPELINE_STATISTICS*>(stats_map);
  occ_smp_mapping_ = reinterpret_cast<const uint64_t*>(smp_map);
  g_occ_sub_tags.assign(kOccMaxPerSubmission, OccCensusTag{});
  occ_available_ = true;
  REXGPU_INFO("[occ] on: {} slots, {} per submission", kOccPoolSize, kOccMaxPerSubmission);
  return true;
}

void D3D12CommandProcessor::ShutdownOccCensusResources() {
  occ_available_ = false;
  occ_sub_active_ = false;
  occ_draw_open_ = false;
  occ_pending_.clear();
  if (occ_stats_readback_ && occ_stats_mapping_) occ_stats_readback_->Unmap(0, nullptr);
  if (occ_smp_readback_ && occ_smp_mapping_) occ_smp_readback_->Unmap(0, nullptr);
  occ_stats_mapping_ = nullptr;
  occ_smp_mapping_ = nullptr;
  occ_stats_readback_.Reset();
  occ_smp_readback_.Reset();
  occ_stats_heap_.Reset();
  occ_smp_heap_.Reset();
  g_occ_sub_tags.clear();
}

// [occ-pos] the draw's cull eligibility, for the census tag (called at the
// issue sites, after the pipeline cache analyzed the shader).
void D3D12CommandProcessor::OccSetElig(const D3D12Shader* vertex_shader,
                                       const RegisterFile& regs) {
  if (!occ_available_) {
    return;
  }
  const PosPath& pp = vertex_shader->pos_path();
  {
    const uint64_t vh = vertex_shader->ucode_data_hash();
    auto sit = occ_vs_slots_.find(vh);
    if (sit != occ_vs_slots_.end()) {
      occ_draw_vs_slot_ = sit->second;
    } else if (occ_vs_hashes_.size() < 4096) {
      if (occ_vs_hashes_.empty()) occ_vs_hashes_.push_back(0);
      occ_draw_vs_slot_ = uint16_t(occ_vs_hashes_.size());
      occ_vs_hashes_.push_back(vh);
      occ_vs_slots_.emplace(vh, occ_draw_vs_slot_);
    } else {
      occ_draw_vs_slot_ = 0;
    }
  }
  const reg::RB_DEPTHCONTROL odc = draw_util::GetNormalizedDepthControl(regs);
  // Drive 801: the city's depth state is z-test GREATER_EQUAL with z-write
  // (0x00708766, 63% of draws): both directions qualify. Stencil qualifies
  // when its fail and z-fail ops keep (a hidden draw then writes nothing).
  const bool zdir = odc.zfunc == xenos::CompareFunction::kLess ||
                    odc.zfunc == xenos::CompareFunction::kLessEqual ||
                    odc.zfunc == xenos::CompareFunction::kGreater ||
                    odc.zfunc == xenos::CompareFunction::kGreaterEqual;
  const bool stok = !odc.stencil_enable ||
                    (odc.stencilfail == xenos::StencilOp::kKeep &&
                     odc.stencilzfail == xenos::StencilOp::kKeep &&
                     (!odc.backface_enable || (odc.stencilfail_bf == xenos::StencilOp::kKeep &&
                                               odc.stencilzfail_bf == xenos::StencilOp::kKeep)));
  const bool zok = odc.z_enable && zdir && stok;
  uint8_t fmt = 3;
  if (pp.format == xenos::VertexFormat::k_32_32_32_FLOAT) {
    fmt = 0;
  } else if (pp.format == xenos::VertexFormat::k_32_32_32_32_FLOAT) {
    fmt = 1;
  } else if (pp.format == xenos::VertexFormat::k_16_16_16_16) {
    fmt = 2;
  }
  occ_draw_elig_ = uint8_t((pp.eligible ? 1 : 0) | (zok ? 2 : 0) |
                           ((uint8_t(pp.reason) & 7) << 2) | (fmt << 5) |
                           (occ_draw_cull_ ? 0x80 : 0));
  static std::unordered_set<uint64_t> s_occ_logged;
  if (s_occ_logged.size() < 96 &&
      s_occ_logged.insert(vertex_shader->ucode_data_hash()).second) {
    char desc[512];
    pp.Describe(desc, sizeof(desc));
    REXGPU_INFO(
        "[occ-pos] vs {:016X} elig={} reason={} fetch={} fmt={} off={} stride={} basis={:x} "
        "idxreg={} signed={} int={} | {}",
        vertex_shader->ucode_data_hash(), pp.eligible ? 1 : 0, uint32_t(pp.reason),
        pp.fetch_constant, uint32_t(pp.format), pp.offset_dwords, pp.stride_dwords,
        pp.used_basis_mask, pp.index_register, pp.is_signed ? 1 : 0, pp.is_integer ? 1 : 0,
        desc);
  }
}

void D3D12CommandProcessor::OccBeginSubmission() {
  if (!occ_available_) {
    return;
  }
  // Mode 2: the per-draw queries only while a verify phase needs them.
  if (REXCVAR_GET(gpu_occ_census) == 2 && hiz_phase_ != 1 && cull_phase_ != 1) {
    return;
  }
  uint32_t start = occ_ring_next_;
  if (start + kOccMaxPerSubmission > kOccPoolSize) {
    start = 0;
  }
  bool overlap = occ_pending_.size() >= 24;
  if (!overlap) {
    const uint32_t region_end = start + kOccMaxPerSubmission;
    for (const OccPending& pending : occ_pending_) {
      const uint32_t pending_end = pending.start + pending.count;
      if (start < pending_end && pending.start < region_end) {
        overlap = true;
        break;
      }
    }
  }
  if (overlap) {
    ++occ_dropped_;
    return;
  }
  occ_sub_start_ = start;
  occ_sub_count_ = 0;
  occ_draw_open_ = false;
  occ_sub_active_ = true;
}

void D3D12CommandProcessor::OccDrawBegin() {
  if (!occ_sub_active_ || occ_draw_open_) {
    return;
  }
  if (occ_sub_count_ >= kOccMaxPerSubmission) {
    ++occ_trunc_;
    return;
  }
  const uint32_t slot = occ_sub_start_ + occ_sub_count_;
  deferred_command_list_.D3DBeginQuery(occ_stats_heap_.Get(),
                                       D3D12_QUERY_TYPE_PIPELINE_STATISTICS, slot);
  occ_draw_smp_ = !active_occlusion_query_.valid;
  if (occ_draw_smp_) {
    deferred_command_list_.D3DBeginQuery(occ_smp_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, slot);
  }
  occ_draw_open_ = true;
}

uint32_t D3D12CommandProcessor::OccDrawEnd(uint32_t idx, uint8_t kind) {
  if (!occ_draw_open_) {
    return UINT32_MAX;
  }
  const uint32_t slot = occ_sub_start_ + occ_sub_count_;
  deferred_command_list_.D3DEndQuery(occ_stats_heap_.Get(), D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
                                     slot);
  if (occ_draw_smp_) {
    deferred_command_list_.D3DEndQuery(occ_smp_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, slot);
  }
  OccCensusTag& t = g_occ_sub_tags[occ_sub_count_];
  t.idx = idx;
  t.inst = 1;
  t.kind = kind;
  t.cls = gpu_census_draw_class_;
  t.smp = occ_draw_smp_ ? 1 : 0;
  t.elig = occ_draw_elig_;
  t.vs_slot = occ_draw_vs_slot_;
  t.dbg = occ_draw_dbg_;
  t.hiz = occ_draw_hiz_;  // [hiz]
  t.hiz_mode = uint8_t(hiz_phase_);
  occ_draw_hiz_ = 0;
  occ_draw_open_ = false;
  return occ_sub_count_++;
}

void D3D12CommandProcessor::OccResolveSubmission() {
  if (!occ_sub_active_) {
    return;
  }
  occ_sub_active_ = false;
  occ_draw_open_ = false;
  if (!occ_sub_count_) {
    return;
  }
  command_list_->ResolveQueryData(
      occ_stats_heap_.Get(), D3D12_QUERY_TYPE_PIPELINE_STATISTICS, occ_sub_start_,
      occ_sub_count_, occ_stats_readback_.Get(),
      uint64_t(occ_sub_start_) * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS));
  command_list_->ResolveQueryData(occ_smp_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, occ_sub_start_,
                                  occ_sub_count_, occ_smp_readback_.Get(),
                                  uint64_t(occ_sub_start_) * sizeof(uint64_t));
  OccPending& pending = occ_pending_.emplace_back();
  pending.submission = submission_current_;
  pending.start = occ_sub_start_;
  pending.count = occ_sub_count_;
  pending.tags.assign(g_occ_sub_tags.begin(), g_occ_sub_tags.begin() + occ_sub_count_);
  occ_ring_next_ = occ_sub_start_ + occ_sub_count_;
}

void D3D12CommandProcessor::OccConsumeCompleted() {
  while (!occ_pending_.empty() && occ_pending_.front().submission <= submission_completed_) {
    const OccPending& pending = occ_pending_.front();
    OccAcc& a = occ_acc_;
    for (uint32_t i = 0; i < pending.count; ++i) {
      const D3D12_QUERY_DATA_PIPELINE_STATISTICS& st = occ_stats_mapping_[pending.start + i];
      const OccCensusTag& t = pending.tags[i];
      const uint64_t smp = t.smp ? occ_smp_mapping_[pending.start + i] : 0;
      const uint8_t cls = t.cls < kGpuCensusTotalClasses ? t.cls : uint8_t(kGpuCensusDraw);
      ++a.q;
      a.ia_prim += st.IAPrimitives;
      a.vs_inv += st.VSInvocations;
      a.c_inv += st.CInvocations;
      a.c_prim += st.CPrimitives;
      a.ps_inv += st.PSInvocations;
      a.smp += smp;
      ++a.cls_q[cls];
      a.cls_prim[cls] += st.IAPrimitives;
      a.cls_ps[cls] += st.PSInvocations;
      a.cls_smp[cls] += smp;
      a.cls_vs[cls] += st.VSInvocations;
      {  // [occ-pos]
        const bool pe = (t.elig & 1) != 0, pz = pe && (t.elig & 2) != 0;
        a.reason_prim[(t.elig >> 2) & 7] += st.IAPrimitives;
        if (pe) {
          ++a.pe_q;
          a.pe_prim += st.IAPrimitives;
          a.fmt_prim[(t.elig >> 5) & 3] += st.IAPrimitives;
          if (t.smp && smp == 0) a.pe_hidden_prim += st.IAPrimitives;
        }
        if (t.elig & 0x80) {  // [cull] verify
          if (t.kind == 1) {
            // A batch query covers every instance; only the opener was judged.
            ++a.cull_vfy_batch;
            if (st.CPrimitives) ++a.cull_wrong_batch;
          } else {
            ++a.cull_vfy;
            a.cull_vfy_prim += st.IAPrimitives;
            ++a.cull_vfy_vs[t.vs_slot & 4095];
            if (st.CPrimitives) {
              ++a.cull_wrong;
              a.cull_wrong_prim += st.IAPrimitives;
              ++a.cull_wrong_vs[t.vs_slot & 4095];
              if (t.smp && smp) ++a.cull_wrong_smp;
              if (t.dbg && cull_dbg_logged_ < 24 && !cull_dbg_.empty()) {
                const CullDbg& dg = cull_dbg_[(t.dbg - 1) % kCullDbgRing];
                if (dg.seq % kCullDbgRing == uint32_t(t.dbg - 1)) {
                  ++cull_dbg_logged_;
                  REXGPU_INFO(
                      "[cull-wrong] vs {:016X} idx={} indexed={} ib={:08X} base={} fetch={:08X}/{:08X} "
                      "stride={} off={} | ia={} cprim={} vsinv={} smp={} | mn=({:.3f},{:.3f},{:.3f}) "
                      "mx=({:.3f},{:.3f},{:.3f}) | m0=({:.4g},{:.4g},{:.4g},{:.4g},{:.4g}) "
                      "m1=({:.4g},{:.4g},{:.4g},{:.4g},{:.4g}) m2=({:.4g},{:.4g},{:.4g},{:.4g},{:.4g}) "
                      "m3=({:.4g},{:.4g},{:.4g},{:.4g},{:.4g})",
                      dg.hash, dg.index_count, dg.indexed, dg.ib_base, dg.base_vertex, dg.fetch0,
                      dg.fetch1, dg.stride, dg.offset, st.IAPrimitives, st.CPrimitives,
                      st.VSInvocations, smp, dg.mn[0], dg.mn[1], dg.mn[2], dg.mx[0], dg.mx[1],
                      dg.mx[2], dg.m[0][0], dg.m[0][1], dg.m[0][2], dg.m[0][3], dg.m[0][4],
                      dg.m[1][0], dg.m[1][1], dg.m[1][2], dg.m[1][3], dg.m[1][4], dg.m[2][0],
                      dg.m[2][1], dg.m[2][2], dg.m[2][3], dg.m[2][4], dg.m[3][0], dg.m[3][1],
                      dg.m[3][2], dg.m[3][3], dg.m[3][4]);
                }
              }
            }
          }
        }
        if (pz) {
          ++a.pz_q;
          a.pz_prim += st.IAPrimitives;
          if (t.smp && smp == 0) a.pz_hidden_prim += st.IAPrimitives;
          if (t.smp && st.CPrimitives == 0) a.pz_clipped_prim += st.IAPrimitives;
        }
      }
      if (t.kind == 1) {
        ++a.batches;
        a.batch_inst += t.inst;
      }
      if (t.hiz && hiz_verdict_mapping_) {  // [hiz] the GPU's verdict for this draw
        HizAcc& h = hiz_acc_;
        // [hiz-pool] a per-instance batch (kind 1) is one query over its
        // run: WRONG only when every element was hidden and samples passed.
        const uint32_t n = t.kind == 1 ? std::max<uint32_t>(t.inst, 1) : 1;
        bool verdict = true;
        for (uint32_t k = 0; k < n && verdict; ++k) {
          verdict = hiz_verdict_mapping_[(t.hiz - 1 + k) & (kHizRing - 1)] != 0;
        }
        if (t.hiz_mode == 1 && t.smp) {  // verify: the draw ran with the query
          ++h.vfy;
          h.vfy_idx += uint64_t(t.idx) * n;
          if (verdict && smp) {
            ++h.wrong;
            h.wrong_idx += uint64_t(t.idx) * n;
            h.wrong_smp += smp;
            ++h.wrong_vs[t.vs_slot & 4095];
          }
        }
      }
      if (!t.smp) {
        ++a.nosmp;
        continue;  // no visibility verdict for this draw
      }
      if (st.CPrimitives == 0) {
        ++a.clipped_n;
        a.clipped_prim += st.IAPrimitives;
      }
      if (smp == 0) {
        ++a.hidden_n;
        a.hidden_prim += st.IAPrimitives;
        a.cls_hidden_prim[cls] += st.IAPrimitives;
      }
      const int b = smp == 0 ? 0 : smp <= 64 ? 1 : smp <= 1024 ? 2 : smp <= 16384 ? 3 : 4;
      ++a.hist_n[b];
      a.hist_prim[b] += st.IAPrimitives;
      a.hist_ps[b] += st.PSInvocations;
    }
    occ_pending_.pop_front();
  }
}

// [hiz] ISSUEDRAW increment 3: the Hi-Z cull.
//
// Drive 798: 66% of the city's primitives pass zero samples at draw time;
// drive 805: the object-space bounds and the shader's clip transform are
// exact. The frustum share (12%) shipped as [cull]; this is the ~54% hidden
// INSIDE the frustum, which only the current frame's depth can decide.
//
// Shape (a PC engine's GPU-driven cull, per checkpoint, no emulation):
// - A cull-eligible plain draw (PosPath affine, bounded, in the frustum,
//   depth test in one direction with z-write or keep-on-fail stencil, the
//   pixel shader not writing depth, every bounds corner in front of the
//   camera) gets a table entry: its screen rect in depth-buffer pixels, the
//   nearest host depth of its 8 corners through the SAME viewport transform
//   the vertex shader applies (ndc_scale/offset, z_min/z_max), and the test
//   direction. It is issued as ExecuteIndirect from an argument slot the
//   checkpoint's test CS fills (InstanceCount 0 when hidden, in skip mode).
// - The checkpoint (the first eligible draw after a window closes) records,
//   inside the draw's own setup and before its ExecuteIndirect: the bound
//   depth target -> NON_PIXEL_SHADER_RESOURCE, hiz_build (every sample of
//   every pixel, 8x8 tiles -> (min, max)), hiz_test over the window's table
//   (up to K entries, the count read from the upload header at GPU time, so
//   entries appended after the dispatch was recorded are still tested), the
//   depth target back to DEPTH_WRITE, the argument buffer back to
//   INDIRECT_ARGUMENT, the guest pipeline re-bound.
// - Conservative by construction: a draw is hidden when every tile of its
//   rect is nearer than its nearest corner, judged on depth as of the
//   checkpoint; the window closes (the next eligible draw re-checks) on
//   anything that could move that depth farther: a pass/depth-target change,
//   a resolve/clear/transfer (IssueCopy, the RT cache's transfer epoch), a
//   depth write whose function admits farther values, a direction change,
//   a submission boundary. Pool hits execute at their opener's position
//   and match the opener's RB_DEPTHCONTROL, so the opener's check covers
//   them.
// - Verify: the CS keeps InstanceCount and writes the verdict word; the
//   [occ] consumer reads it back beside the draw's occlusion query. WRONG
//   (hidden verdict, samples > 0) must be 0.

ID3D12CommandSignature* D3D12CommandProcessor::DrawCommandSignature(bool indexed) {
  auto& sig = indexed ? pool_cmdsig_indexed_ : pool_cmdsig_draw_;
  if (!sig) {
    D3D12_INDIRECT_ARGUMENT_DESC arg = {};
    arg.Type = indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED
                       : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    D3D12_COMMAND_SIGNATURE_DESC desc = {};
    desc.ByteStride = indexed ? sizeof(D3D12_DRAW_INDEXED_ARGUMENTS) : sizeof(D3D12_DRAW_ARGUMENTS);
    desc.NumArgumentDescs = 1;
    desc.pArgumentDescs = &arg;
    if (FAILED(GetD3D12Provider().GetDevice()->CreateCommandSignature(&desc, nullptr,
                                                                      IID_PPV_ARGS(&sig)))) {
      return nullptr;
    }
  }
  return sig.Get();
}

bool D3D12CommandProcessor::InitializeHizResources() {
  hiz_available_ = false;
  if (!bindless_resources_used_) {
    // The checkpoint sits inside a draw's setup; a bindful one-use
    // descriptor request may switch heaps under the draw's tables.
    REXGPU_WARN("[hiz] bindful descriptors - off");
    return false;
  }
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();

  D3D12_ROOT_PARAMETER params[6] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[0].Constants.ShaderRegister = 0;
  params[0].Constants.RegisterSpace = 0;
  params[0].Constants.Num32BitValues = 8;
  params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_DESCRIPTOR_RANGE depth_range = {};
  depth_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  depth_range.NumDescriptors = 1;
  depth_range.BaseShaderRegister = 0;
  depth_range.RegisterSpace = 0;
  depth_range.OffsetInDescriptorsFromTableStart = 0;
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable.NumDescriptorRanges = 1;
  params[1].DescriptorTable.pDescriptorRanges = &depth_range;
  params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;  // u0 hiz buffer
  params[2].Descriptor.ShaderRegister = 0;
  params[2].Descriptor.RegisterSpace = 0;
  params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;  // t1 table
  params[3].Descriptor.ShaderRegister = 1;
  params[3].Descriptor.RegisterSpace = 0;
  params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;  // u1 args
  params[4].Descriptor.ShaderRegister = 1;
  params[4].Descriptor.RegisterSpace = 0;
  params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;  // u2 verdict
  params[5].Descriptor.ShaderRegister = 2;
  params[5].Descriptor.RegisterSpace = 0;
  params[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC rs_desc = {};
  rs_desc.NumParameters = 6;
  rs_desc.pParameters = params;
  rs_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(hiz_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, rs_desc);
  if (!hiz_root_signature_) {
    REXGPU_WARN("[hiz] root signature creation failed - off");
    return false;
  }
  *(hiz_build_pipelines_[0].ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::hiz_build_1x_cs, sizeof(shaders::hiz_build_1x_cs),
      hiz_root_signature_.Get());
  *(hiz_build_pipelines_[1].ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::hiz_build_2x_cs, sizeof(shaders::hiz_build_2x_cs),
      hiz_root_signature_.Get());
  *(hiz_build_pipelines_[2].ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::hiz_build_4x_cs, sizeof(shaders::hiz_build_4x_cs),
      hiz_root_signature_.Get());
  *(hiz_test_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::hiz_test_cs, sizeof(shaders::hiz_test_cs), hiz_root_signature_.Get());
  if (!hiz_build_pipelines_[0] || !hiz_build_pipelines_[1] || !hiz_build_pipelines_[2] ||
      !hiz_test_pipeline_) {
    REXGPU_WARN("[hiz] pipeline creation failed - off");
    ShutdownHizResources();
    return false;
  }
  auto make_buffer = [&](Microsoft::WRL::ComPtr<ID3D12Resource>& out, uint64_t bytes,
                         D3D12_RESOURCE_STATES state, bool readback) {
    D3D12_RESOURCE_DESC desc;
    ui::d3d12::util::FillBufferResourceDesc(
        desc, bytes,
        readback ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    return SUCCEEDED(device->CreateCommittedResource(
        readback ? &ui::d3d12::util::kHeapPropertiesReadback
                 : &ui::d3d12::util::kHeapPropertiesDefault,
        provider.GetHeapFlagCreateNotZeroed(), &desc, state, nullptr, IID_PPV_ARGS(&out)));
  };
  if (!make_buffer(hiz_buffer_, kHizBufferBytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false) ||
      !make_buffer(hiz_args_, uint64_t(kHizRing) * kHizArgsStride,
                   D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, false) ||
      !make_buffer(hiz_verdict_, uint64_t(kHizRing) * 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   false) ||
      !make_buffer(hiz_verdict_readback_, uint64_t(kHizRing) * 4, D3D12_RESOURCE_STATE_COPY_DEST,
                   true)) {
    REXGPU_WARN("[hiz] buffer allocation failed - off");
    ShutdownHizResources();
    return false;
  }
  {
    D3D12_RANGE read_range = {0, uint64_t(kHizRing) * 4};
    void* mapping = nullptr;
    if (FAILED(hiz_verdict_readback_->Map(0, &read_range, &mapping))) {
      REXGPU_WARN("[hiz] readback map failed - off");
      ShutdownHizResources();
      return false;
    }
    hiz_verdict_mapping_ = reinterpret_cast<const uint32_t*>(mapping);
  }
  hiz_slot_idx_.assign(kHizRing, 0u);
  hiz_available_ = true;
  REXGPU_INFO("[hiz] on: {} argument slots, {} per frame, k={}", kHizRing, kHizMaxPerFrame,
              REXCVAR_GET(gpu_hiz_k));
  return true;
}

void D3D12CommandProcessor::ShutdownHizResources() {
  hiz_available_ = false;
  hiz_window_ = HizWindow{};
  hiz_draw_.valid = false;
  hiz_pending_.clear();
  hiz_valid_ = false;
  if (hiz_verdict_readback_ && hiz_verdict_mapping_) hiz_verdict_readback_->Unmap(0, nullptr);
  hiz_verdict_mapping_ = nullptr;
  hiz_verdict_readback_.Reset();
  hiz_verdict_.Reset();
  hiz_args_.Reset();
  hiz_buffer_.Reset();
  hiz_test_pipeline_.Reset();
  for (auto& p : hiz_build_pipelines_) p.Reset();
  hiz_root_signature_.Reset();
  hiz_cmdsig_[0].clear();
  hiz_cmdsig_[1].clear();
}

void D3D12CommandProcessor::HizBeginSubmission() {
  if (hiz_window_.open) HizWindowClose(5);
  hiz_sub_first_slot_ = hiz_slot_next_;
  // The verdicts of completed submissions: the hidden share, without queries.
  while (!hiz_pending_.empty() && hiz_pending_.front().submission <= submission_completed_) {
    const HizPending& p = hiz_pending_.front();
    if (hiz_verdict_mapping_) {
      for (uint32_t i = 0; i < p.count; ++i) {
        const uint32_t slot = (p.first + i) & (kHizRing - 1);
        // [hiz-pool] a reserved, never appended slot has idx 0 and a stale word.
        if (hiz_verdict_mapping_[slot] && hiz_slot_idx_[slot]) {
          ++hiz_acc_.hidden;
          hiz_acc_.hidden_idx += hiz_slot_idx_[slot];
        }
      }
    }
    hiz_pending_.pop_front();
  }
}

void D3D12CommandProcessor::HizResolveSubmission() {
  if (!hiz_available_ || hiz_slot_next_ == hiz_sub_first_slot_) {
    return;
  }
  if (hiz_window_.open) HizWindowClose(5);
  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = hiz_verdict_.Get();
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  command_list_->ResourceBarrier(1, &barrier);
  const uint32_t n = hiz_slot_next_ - hiz_sub_first_slot_;
  if (n >= kHizRing) {
    command_list_->CopyBufferRegion(hiz_verdict_readback_.Get(), 0, hiz_verdict_.Get(), 0,
                                    uint64_t(kHizRing) * 4);
  } else {
    const uint32_t a = hiz_sub_first_slot_ & (kHizRing - 1), b = hiz_slot_next_ & (kHizRing - 1);
    if (a < b) {
      command_list_->CopyBufferRegion(hiz_verdict_readback_.Get(), uint64_t(a) * 4,
                                      hiz_verdict_.Get(), uint64_t(a) * 4, uint64_t(b - a) * 4);
    } else {
      command_list_->CopyBufferRegion(hiz_verdict_readback_.Get(), uint64_t(a) * 4,
                                      hiz_verdict_.Get(), uint64_t(a) * 4,
                                      uint64_t(kHizRing - a) * 4);
      if (b) {
        command_list_->CopyBufferRegion(hiz_verdict_readback_.Get(), 0, hiz_verdict_.Get(), 0,
                                        uint64_t(b) * 4);
      }
    }
  }
  std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
  command_list_->ResourceBarrier(1, &barrier);
  HizPending& p = hiz_pending_.emplace_back();
  p.submission = submission_current_;
  p.first = hiz_sub_first_slot_;
  p.count = std::min(n, kHizRing);
  hiz_sub_first_slot_ = hiz_slot_next_;
}

void D3D12CommandProcessor::HizWindowClose(uint32_t reason) {
  if (!hiz_window_.open) {
    return;
  }
  ++hiz_acc_.close[reason & 7];
  // cap (0) and submission / phase (5) leave the depth as the last build
  // saw it (or nearer); everything else may have moved it farther: the
  // Hi-Z is invalid (checkpoints hide nothing) and the rebuild countdown
  // restarts.
  if (reason != 0 && reason != 5) {
    hiz_valid_ = false;
    hiz_build_draws_ = hiz_draws_seen_;
  }
  hiz_window_.open = false;
  hiz_window_.header = nullptr;
  hiz_window_.entries = nullptr;
}

// The [cull] block found the draw bounded and inside the frustum: classify
// it for the Hi-Z test and, when eligible, compute its entry (screen rect
// and nearest host depth) from the corners through the viewport transform.
void D3D12CommandProcessor::HizPrepare(const CullBounds& b, const float m[4][5],
                                       const RegisterFile& regs, uint32_t index_count) {
  ++hiz_acc_.elig;
  hiz_acc_.elig_idx += index_count;
  const reg::RB_DEPTHCONTROL dc = draw_util::GetNormalizedDepthControl(regs);
  uint8_t dir = 0;
  if (dc.z_enable) {
    if (dc.zfunc == xenos::CompareFunction::kGreater ||
        dc.zfunc == xenos::CompareFunction::kGreaterEqual) {
      dir = 1;
    } else if (dc.zfunc == xenos::CompareFunction::kLess ||
               dc.zfunc == xenos::CompareFunction::kLessEqual) {
      dir = 2;
    }
  }
  const bool stok = !dc.stencil_enable ||
                    (dc.stencilfail == xenos::StencilOp::kKeep &&
                     dc.stencilzfail == xenos::StencilOp::kKeep &&
                     (!dc.backface_enable || (dc.stencilfail_bf == xenos::StencilOp::kKeep &&
                                              dc.stencilzfail_bf == xenos::StencilOp::kKeep)));
  if (!dir || !stok) {
    HizRefuse(1, index_count);
    return;
  }
  const Shader* ps = active_pixel_shader();
  if (ps && ps->writes_depth()) {
    HizRefuse(2, index_count);
    return;
  }
  hiz_draw_.dir_hint = dir;  // [hiz-pool] the depth class holds from here on
  draw_util::ViewportInfo vp;
  draw_util::GetHostViewportInfo(
      regs, texture_cache_->draw_resolution_scale_x(), texture_cache_->draw_resolution_scale_y(),
      true, D3D12_VIEWPORT_BOUNDS_MAX, D3D12_VIEWPORT_BOUNDS_MAX, false, dc,
      render_target_cache_->depth_float24_convert_in_pixel_shader(), true, false, vp);
  if (!vp.xy_extent[0] || !vp.xy_extent[1]) {
    HizRefuse(5, index_count);
    return;
  }
  float x0 = FLT_MAX, y0 = FLT_MAX, x1 = -FLT_MAX, y1 = -FLT_MAX;
  float zn = dir == 1 ? -FLT_MAX : FLT_MAX;
  for (uint32_t c = 0; c < 8; ++c) {
    const float p[3] = {(c & 1) ? b.mx[0] : b.mn[0], (c & 2) ? b.mx[1] : b.mn[1],
                        (c & 4) ? b.mx[2] : b.mn[2]};
    float clip[4];
    for (uint32_t i = 0; i < 4; ++i) {
      clip[i] = m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3] * b.mn[3] + m[i][4];
    }
    if (!(clip[3] > 1e-6f)) {
      HizRefuse(3, index_count);  // a corner at or behind the camera plane
      return;
    }
    const float iw = 1.0f / clip[3];
    const float nx = clip[0] * iw * vp.ndc_scale[0] + vp.ndc_offset[0];
    const float ny = clip[1] * iw * vp.ndc_scale[1] + vp.ndc_offset[1];
    float nz = clip[2] * iw * vp.ndc_scale[2] + vp.ndc_offset[2];
    if (!(nx == nx) || !(ny == ny) || !(nz == nz)) {
      HizRefuse(3, index_count);
      return;
    }
    nz = std::min(std::max(nz, 0.0f), 1.0f);
    const float px = float(vp.xy_offset[0]) + (nx * 0.5f + 0.5f) * float(vp.xy_extent[0]);
    const float py = float(vp.xy_offset[1]) + (0.5f - ny * 0.5f) * float(vp.xy_extent[1]);
    const float hz = vp.z_min + nz * (vp.z_max - vp.z_min);
    x0 = std::min(x0, px);
    x1 = std::max(x1, px);
    y0 = std::min(y0, py);
    y1 = std::max(y1, py);
    zn = dir == 1 ? std::max(zn, hz) : std::min(zn, hz);
  }
  // Clip the rect to the viewport (the clipper does), then to the tile cap.
  x0 = std::max(x0, float(vp.xy_offset[0])) - 1.0f;
  y0 = std::max(y0, float(vp.xy_offset[1])) - 1.0f;
  x1 = std::min(x1, float(vp.xy_offset[0] + vp.xy_extent[0])) + 1.0f;
  y1 = std::min(y1, float(vp.xy_offset[1] + vp.xy_extent[1])) + 1.0f;
  if (x1 - x0 > float(kHizMaxRectTiles * 8) || y1 - y0 > float(kHizMaxRectTiles * 8) ||
      x1 <= x0 || y1 <= y0) {
    HizRefuse(4, index_count);
    return;
  }
  // Precision margin toward "nearer": float24 rounding in the pixel shader
  // and the D24 quantization may store a fragment slightly nearer than the
  // interpolated value; two orders above both.
  const float eps = std::fabs(zn) * (1.0f / 262144.0f) + (1.0f / 4194304.0f);
  zn = dir == 1 ? zn + eps : zn - eps;
  hiz_draw_.valid = true;
  hiz_draw_.rect[0] = x0;
  hiz_draw_.rect[1] = y0;
  hiz_draw_.rect[2] = x1;
  hiz_draw_.rect[3] = y1;
  hiz_draw_.z_near = zn;
  hiz_draw_.dir = dir;
  hiz_draw_.index_count = index_count;
}

// Records the checkpoint for a new window: Hi-Z build from the bound depth
// target, the test dispatch over the window's (still empty) table. Called
// from the plain-draw site, after the draw's bindings and barriers.
bool D3D12CommandProcessor::HizCheckpoint(uint8_t dir) {
  uint32_t w = 0, h = 0, samples = 0;
  D3D12_CPU_DESCRIPTOR_HANDLE depth_srv = {};
  ID3D12Resource* depth = render_target_cache_->GetBoundDepthForHiz(w, h, samples, depth_srv);
  if (!depth || !w || !h || (samples != 1 && samples != 2 && samples != 4)) {
    return false;
  }
  const uint32_t tiles_x = (w + 7) / 8, tiles_y = (h + 7) / 8;
  if (uint64_t(tiles_x) * tiles_y * 8 > kHizBufferBytes) {
    return false;
  }
  const uint32_t cap = std::clamp<uint32_t>(hiz_k_, 1, kHizMaxK);
  ID3D12Resource* table_buffer = nullptr;
  size_t table_offset = 0;
  D3D12_GPU_VIRTUAL_ADDRESS table_gpu = 0;
  uint8_t* mapping = constant_buffer_pool_->Request(
      frame_current_,
      kHizHeaderBytes + size_t(cap) * kHizEntryBytes + size_t(kHizMaxRuns) * kHizRunBytes,
      D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, &table_buffer, &table_offset, &table_gpu);
  if (!mapping) {
    return false;
  }
  ui::d3d12::util::DescriptorCpuGpuHandlePair depth_descriptor;
  if (!RequestOneUseSingleViewDescriptors(1, &depth_descriptor)) {
    return false;
  }
  std::memset(mapping, 0, kHizHeaderBytes);
  GetD3D12Provider().GetDevice()->CopyDescriptorsSimple(1, depth_descriptor.first, depth_srv,
                                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  // A stale Hi-Z is conservative while the depth it saw can only have moved
  // nearer: same target, no transfer/clear since, no non-benign close. Any
  // of those invalidates it (checkpoints then hide nothing) and restarts
  // the countdown; a build happens gpu_hiz_rebuild received draws after
  // the last build or invalidation, never on an empty depth.
  const uint64_t epoch = render_target_cache_->transfer_epoch();
  if (hiz_build_resource_ != depth || hiz_build_epoch_ != epoch) {
    hiz_valid_ = false;
    hiz_build_draws_ = hiz_draws_seen_;
    hiz_build_resource_ = depth;
    hiz_build_epoch_ = epoch;
  }
  const bool rebuild = hiz_draws_seen_ - hiz_build_draws_ >= uint64_t(hiz_rebuild_);
  ++hiz_acc_.checkpoints;
  if (rebuild) ++hiz_acc_.builds;
  // Priced as three classes: the depth target's transitions (hizd: the
  // driver's decompress), the build CS (hiz), the test CS with the
  // argument-buffer barriers (hizt), so a drive names which one the ~1.3 ms
  // per build on the Intel is before a cheaper build is designed.
  GpuCensusScope census(*this, rebuild ? kGpuCensusHizDepth : kGpuCensusHizTest);
  ID3D12PipelineState* saved_external = current_external_pipeline_;
  void* saved_guest = current_guest_pipeline_;

  if (rebuild) {
    render_target_cache_->TransitionBoundDepthForHiz(
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    SubmitBarriers();
    GpuCensusSetClass(kGpuCensusHiz);
  }
  PushTransitionBarrier(hiz_args_.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  SubmitBarriers();
  struct {
    uint32_t tiles_x, tiles_y, depth_w, depth_h, mode, entry_cap, pad[2];
  } constants = {tiles_x, tiles_y, w, h, (rebuild || hiz_valid_) ? hiz_phase_ : 0u, cap, {0, 0}};
  deferred_command_list_.D3DSetComputeRootSignature(hiz_root_signature_.Get());
  deferred_command_list_.D3DSetComputeRoot32BitConstants(0, 8, &constants, 0);
  deferred_command_list_.D3DSetComputeRootDescriptorTable(1, depth_descriptor.second);
  deferred_command_list_.D3DSetComputeRootUnorderedAccessView(2,
                                                              hiz_buffer_->GetGPUVirtualAddress());
  deferred_command_list_.D3DSetComputeRootShaderResourceView(3, table_gpu);
  deferred_command_list_.D3DSetComputeRootUnorderedAccessView(4, hiz_args_->GetGPUVirtualAddress());
  deferred_command_list_.D3DSetComputeRootUnorderedAccessView(
      5, hiz_verdict_->GetGPUVirtualAddress());
  if (rebuild) {
    SetExternalPipeline(hiz_build_pipelines_[samples == 4 ? 2 : samples == 2 ? 1 : 0].Get());
    deferred_command_list_.D3DDispatch(tiles_x, tiles_y, 1);
    PushUAVBarrier(hiz_buffer_.Get());
    SubmitBarriers();
    hiz_build_draws_ = hiz_draws_seen_;
    hiz_valid_ = true;
    GpuCensusSetClass(kGpuCensusHizTest);
  }
  SetExternalPipeline(hiz_test_pipeline_.Get());
  deferred_command_list_.D3DDispatch((cap + kHizMaxRuns + 63) / 64, 1, 1);
  PushTransitionBarrier(hiz_args_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
  SubmitBarriers();
  if (rebuild) {
    GpuCensusSetClass(kGpuCensusHizDepth);
    render_target_cache_->TransitionBoundDepthForHiz(D3D12_RESOURCE_STATE_DEPTH_WRITE);
    SubmitBarriers();
  }
  // The draw's pipeline back (the compute passes replaced it).
  if (saved_external) {
    SetExternalPipeline(saved_external);
  } else if (saved_guest) {
    deferred_command_list_.SetPipelineStateHandle(saved_guest);
    current_guest_pipeline_ = saved_guest;
    current_external_pipeline_ = nullptr;
  }

  hiz_window_.open = true;
  hiz_window_.serial = ++hiz_window_serial_;  // [hiz-pool]
  hiz_window_.submission = submission_current_;
  hiz_window_.transfer_epoch = epoch;
  hiz_window_.count = 0;
  hiz_window_.cap = cap;
  hiz_window_.dir = dir;
  hiz_window_.header = reinterpret_cast<uint32_t*>(mapping);
  hiz_window_.entries = mapping + kHizHeaderBytes;
  hiz_window_.runs = hiz_window_.entries + size_t(cap) * kHizEntryBytes;  // [hiz-pool]
  hiz_window_.run_count = 0;
  hiz_window_.depth_resource = depth;
  return true;
}

// The plain-draw site: append the pending entry to the window (opening a
// checkpoint first when none is open or the open one no longer holds) and
// issue the draw as ExecuteIndirect from its argument slot.
bool D3D12CommandProcessor::HizWindowValidate(uint8_t dir) {
  if (!hiz_window_.open) {
    return false;
  }
  if (hiz_window_.submission != submission_current_) {
    HizWindowClose(5);
  } else if (hiz_window_.transfer_epoch != render_target_cache_->transfer_epoch()) {
    HizWindowClose(6);
  } else if (hiz_window_.dir != dir) {
    HizWindowClose(4);
  } else if (hiz_window_.count >= hiz_window_.cap) {
    HizWindowClose(0);
  } else {
    uint32_t w, h, samples;
    D3D12_CPU_DESCRIPTOR_HANDLE srv;
    if (render_target_cache_->GetBoundDepthForHiz(w, h, samples, srv) !=
        hiz_window_.depth_resource) {
      HizWindowClose(7);
    }
  }
  return hiz_window_.open;
}

bool D3D12CommandProcessor::HizWindowEnsure(uint8_t dir) {
  if (!hiz_available_ || hiz_phase_ == 0) {
    return false;
  }
  if (HizWindowValidate(dir)) {
    return true;
  }
  return HizCheckpoint(dir);
}

// One table entry of the open window: the slot's arguments (dword 0 = the
// instance base root constant, then the draw arguments) and, when d is set,
// the rect and depth to test; otherwise never hidden (flags bit 1).
void D3D12CommandProcessor::HizAppend(uint32_t slot, const HizDrawPending* d, uint32_t inst,
                                      uint32_t host_count, bool plain_layout) {
  hiz_slot_idx_[slot] = host_count;
  uint32_t* e = reinterpret_cast<uint32_t*>(hiz_window_.entries +
                                            size_t(hiz_window_.count) * kHizEntryBytes);
  if (d) {
    std::memcpy(e, d->rect, 4 * sizeof(float));
    std::memcpy(e + 4, &d->z_near, sizeof(float));
    e[5] = (d->dir == 1 ? 1u : 0u) | (plain_layout ? 16u : 0u) | (inst << 8);
    ++hiz_acc_.tested;
    hiz_acc_.tested_idx += host_count;
  } else {
    std::memset(e, 0, 5 * sizeof(uint32_t));
    e[5] = 2u | (plain_layout ? 16u : 0u) | (inst << 8);
  }
  e[6] = slot;
  e[7] = 0;  // argument dword 4: base vertex (indexed) / start instance
  e[8] = host_count;
  e[9] = 1;   // instances
  e[10] = 0;  // start index / start vertex
  e[11] = 0;  // base vertex (indexed) / start instance (non-indexed)
  ++hiz_window_.count;
  hiz_window_.header[0] = hiz_window_.count;  // read by the test CS at GPU time
}

// A pool hit into a per-instance batch: its element (base + inst) gets an
// entry of the batch's window. False when that window is gone (the hit is
// refused and the draw reopens a fresh batch).
bool D3D12CommandProcessor::HizPoolAppend(uint32_t base, uint32_t serial, uint32_t inst,
                                          uint8_t dir, uint32_t idx_count) {
  if (!hiz_available_ || hiz_phase_ == 0) {
    return false;
  }
  if (!HizWindowValidate(dir) || hiz_window_.serial != serial) {
    return false;
  }
  const HizDrawPending d = hiz_draw_;
  hiz_draw_.valid = false;
  const uint32_t slot = (base + inst) & (kHizRing - 1);
  const bool tested = d.valid && d.dir == dir;
  HizAppend(slot, tested ? &d : nullptr, inst, idx_count);
  ++hiz_acc_.pool_inst;
  if (tested) ++hiz_acc_.pool_inst_tested;
  return true;
}

// The command signature of the Hi-Z argument slot: the instance base root
// constant, then the draw. Created against the bound guest root signature.
ID3D12CommandSignature* D3D12CommandProcessor::HizCommandSignature(bool indexed) {
  ID3D12RootSignature* rs = current_graphics_root_signature_;
  if (!rs) {
    return nullptr;
  }
  auto& cache = hiz_cmdsig_[indexed ? 1 : 0];
  auto it = cache.find(rs);
  if (it != cache.end()) {
    return it->second.Get();
  }
  D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
  args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
  args[0].Constant.RootParameterIndex = bindless_resources_used_
                                            ? kRootParameter_Bindless_InstanceBase
                                            : kRootParameter_Bindful_InstanceBase;
  args[0].Constant.DestOffsetIn32BitValues = 0;
  args[0].Constant.Num32BitValuesToSet = 1;
  args[1].Type =
      indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
  D3D12_COMMAND_SIGNATURE_DESC desc = {};
  desc.ByteStride = kHizArgsStride;
  desc.NumArgumentDescs = 2;
  desc.pArgumentDescs = args;
  Microsoft::WRL::ComPtr<ID3D12CommandSignature> sig;
  if (FAILED(GetD3D12Provider().GetDevice()->CreateCommandSignature(&desc, rs,
                                                                    IID_PPV_ARGS(&sig)))) {
    REXGPU_WARN("[hiz] command signature with the instance base constant failed");
    cache.emplace(rs, nullptr);
    return nullptr;
  }
  cache.emplace(rs, sig);
  return sig.Get();
}

ID3D12CommandSignature* D3D12CommandProcessor::HizCommandSignaturePlain(bool indexed) {
  ID3D12RootSignature* rs = current_graphics_root_signature_;
  if (!rs) {
    return nullptr;
  }
  auto& cache = hiz_cmdsig_plain_[indexed ? 1 : 0];
  auto it = cache.find(rs);
  if (it != cache.end()) {
    return it->second.Get();
  }
  D3D12_INDIRECT_ARGUMENT_DESC arg = {};
  arg.Type = indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
  D3D12_COMMAND_SIGNATURE_DESC desc = {};
  desc.ByteStride = kHizArgsStride;
  desc.NumArgumentDescs = 1;
  desc.pArgumentDescs = &arg;
  Microsoft::WRL::ComPtr<ID3D12CommandSignature> sig;
  // A draw-only signature needs no root signature.
  if (FAILED(GetD3D12Provider().GetDevice()->CreateCommandSignature(&desc, nullptr,
                                                                    IID_PPV_ARGS(&sig)))) {
    REXGPU_WARN("[hiz] draw-only command signature failed");
    cache.emplace(rs, nullptr);
    return nullptr;
  }
  cache.emplace(rs, sig);
  return sig.Get();
}

void D3D12CommandProcessor::HizInstanceBaseReset() {
  const uint32_t zero = 0;
  deferred_command_list_.D3DSetGraphicsRoot32BitConstants(
      bindless_resources_used_ ? kRootParameter_Bindless_InstanceBase
                               : kRootParameter_Bindful_InstanceBase,
      1, &zero, 0);
  inst_base_dirty_ = false;
}

// The plain-draw site: append the pending entry to the window (opening a
// checkpoint first when none is open or the open one no longer holds) and
// issue the draw as ExecuteIndirect from its argument slot.
bool D3D12CommandProcessor::HizDraw(bool indexed, uint32_t host_draw_vertex_count) {
  const HizDrawPending d = hiz_draw_;
  hiz_draw_.valid = false;
  if (!hiz_available_ || !d.valid || hiz_phase_ == 0) {
    return false;
  }
  // Drive 838 (Intel 720p): a draw-only signature for the plain draws took
  // the draw class 81 -> 60 ms/frame (the root-constant element is the
  // Intel driver's slow path per call); the pooled batches keep [constant,
  // draw] until their instance index has another carrier.
  ID3D12CommandSignature* sig = HizCommandSignaturePlain(indexed);
  if (!sig) {
    HizRefuse(8, d.index_count);
    return false;
  }
  if (hiz_frame_slots_ >= kHizMaxPerFrame) {
    HizRefuse(7, d.index_count);
    return false;
  }
  if (!HizWindowEnsure(d.dir)) {
    HizRefuse(6, d.index_count);
    return false;
  }
  const uint32_t slot = hiz_slot_next_ & (kHizRing - 1);
  ++hiz_slot_next_;
  ++hiz_frame_slots_;
  HizAppend(slot, &d, 0, host_draw_vertex_count, true);
  OccDrawBegin();  // [occ]
  occ_draw_hiz_ = slot + 1;
  deferred_command_list_.D3DExecuteIndirect(sig, 1, hiz_args_.Get(),
                                            uint64_t(slot) * kHizArgsStride);
  OccDrawEnd(host_draw_vertex_count, 0);
  occ_draw_hiz_ = 0;
  return true;
}

void D3D12CommandProcessor::OccReport1Hz() {
  if (!occ_available_) {
    return;
  }
  static uint64_t s_frames = 0;
  const uint64_t frames_now = g_pool.frames;
  const double fr = frames_now > s_frames ? double(frames_now - s_frames) : 1.0;
  s_frames = frames_now;
  const OccAcc& a = occ_acc_;
  if (!a.q) {
    return;
  }
  const double q = double(a.q), prim = a.ia_prim ? double(a.ia_prim) : 1.0;
  const double ps = a.ps_inv ? double(a.ps_inv) : 1.0;
  const double vis = double(a.q - a.nosmp) > 0 ? double(a.q - a.nosmp) : 1.0;
  uint32_t main_cls = kGpuCensusDraw;
  for (uint32_t c = 0; c < kGpuCensusTotalClasses; ++c) {
    if (a.cls_prim[c] > a.cls_prim[main_cls]) main_cls = c;
  }
  const char* main_desc = main_cls >= kGpuCensusClassCount
                              ? gpu_census_draw_configs_[main_cls - kGpuCensusClassCount].desc
                              : "draw";
  const double mp = a.cls_prim[main_cls] ? double(a.cls_prim[main_cls]) : 1.0;
  REXGPU_INFO(
      "[occ] q/fr={:.0f} batches/fr={:.0f} inst/fr={:.0f} nosmp={} trunc={} drop={} | /fr prim "
      "{:.0f}k vsinv {:.0f}k ({:.2f}/prim) clipout {:.0f}k ({:.1f}%) psinv {:.0f}k ({:.1f}x 720p) "
      "smp {:.0f}k | hidden(smp0) draws {:.1f}% prim {:.1f}% | clipped(cprim0) draws {:.1f}% "
      "prim {:.1f}% | by smp 0/<=64/<=1k/<=16k/16k+ draws {:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}% "
      "prim {:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}% ps {:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}% | main "
      "[{}] q/fr {:.0f} prim {:.0f}k ({:.0f}%) hidden prim {:.1f}% vsinv {:.0f}k psinv {:.0f}k "
      "smp {:.0f}k",
      q / fr, double(a.batches) / fr, double(a.batch_inst) / fr, a.nosmp, occ_trunc_,
      occ_dropped_, double(a.ia_prim) / fr / 1000.0, double(a.vs_inv) / fr / 1000.0,
      double(a.vs_inv) / prim, double(a.c_prim) / fr / 1000.0, 100.0 * double(a.c_prim) / prim,
      double(a.ps_inv) / fr / 1000.0, double(a.ps_inv) / fr / 921600.0,
      double(a.smp) / fr / 1000.0, 100.0 * double(a.hidden_n) / vis,
      100.0 * double(a.hidden_prim) / prim, 100.0 * double(a.clipped_n) / vis,
      100.0 * double(a.clipped_prim) / prim, 100.0 * double(a.hist_n[0]) / vis,
      100.0 * double(a.hist_n[1]) / vis, 100.0 * double(a.hist_n[2]) / vis,
      100.0 * double(a.hist_n[3]) / vis, 100.0 * double(a.hist_n[4]) / vis,
      100.0 * double(a.hist_prim[0]) / prim, 100.0 * double(a.hist_prim[1]) / prim,
      100.0 * double(a.hist_prim[2]) / prim, 100.0 * double(a.hist_prim[3]) / prim,
      100.0 * double(a.hist_prim[4]) / prim, 100.0 * double(a.hist_ps[0]) / ps,
      100.0 * double(a.hist_ps[1]) / ps, 100.0 * double(a.hist_ps[2]) / ps,
      100.0 * double(a.hist_ps[3]) / ps, 100.0 * double(a.hist_ps[4]) / ps, main_desc,
      double(a.cls_q[main_cls]) / fr, double(a.cls_prim[main_cls]) / fr / 1000.0,
      100.0 * double(a.cls_prim[main_cls]) / prim,
      100.0 * double(a.cls_hidden_prim[main_cls]) / mp, double(a.cls_vs[main_cls]) / fr / 1000.0,
      double(a.cls_ps[main_cls]) / fr / 1000.0, double(a.cls_smp[main_cls]) / fr / 1000.0);
  {  // [occ-pos] the reachable share of the ceiling.
    const double pep = a.pe_prim ? double(a.pe_prim) : 1.0;
    const double pzp = a.pz_prim ? double(a.pz_prim) : 1.0;
    REXGPU_INFO(
        "[occ-pos] path-elig draws {:.1f}% prim {:.1f}% (hidden {:.1f}% of them) | +ztest "
        "draws {:.1f}% prim {:.1f}% (hidden {:.1f}% clipped {:.1f}% of them) = {:.1f}% of all "
        "prims hidden and cullable | ineligible prim by reason nv/nw/na/mf/nf/cf = "
        "{:.1f}/{:.1f}/{:.1f}/{:.1f}/{:.1f}/{:.1f}% | elig fmt f32x3/f32x4/s16x4/other = "
        "{:.0f}/{:.0f}/{:.0f}/{:.0f}%",
        100.0 * double(a.pe_q) / q, 100.0 * double(a.pe_prim) / prim,
        100.0 * double(a.pe_hidden_prim) / pep, 100.0 * double(a.pz_q) / q,
        100.0 * double(a.pz_prim) / prim, 100.0 * double(a.pz_hidden_prim) / pzp,
        100.0 * double(a.pz_clipped_prim) / pzp, 100.0 * double(a.pz_hidden_prim) / prim,
        100.0 * double(a.reason_prim[1]) / prim, 100.0 * double(a.reason_prim[2]) / prim,
        100.0 * double(a.reason_prim[3]) / prim, 100.0 * double(a.reason_prim[4]) / prim,
        100.0 * double(a.reason_prim[5]) / prim, 100.0 * double(a.reason_prim[6]) / prim,
        100.0 * double(a.fmt_prim[0]) / pep, 100.0 * double(a.fmt_prim[1]) / pep,
        100.0 * double(a.fmt_prim[2]) / pep, 100.0 * double(a.fmt_prim[3]) / pep);
    if (a.cull_vfy || a.cull_vfy_batch) {
      std::string top;
      for (uint32_t n = 0; n < 5; ++n) {
        uint32_t best = 0;
        for (uint32_t s = 1; s < 4096; ++s) {
          if (a.cull_wrong_vs[s] > a.cull_wrong_vs[best]) best = s;
        }
        if (!best || !a.cull_wrong_vs[best]) break;
        top += fmt::format("{:016X}:{}/{} ", best < occ_vs_hashes_.size() ? occ_vs_hashes_[best] : 0,
                           a.cull_wrong_vs[best], a.cull_vfy_vs[best]);
        const_cast<OccAcc&>(a).cull_wrong_vs[best] = 0;
      }
      REXGPU_INFO("[cull-vfy] plain out-verdict draws={} prim={} | WRONG cprim>0 draws={} "
                  "({:.3f}%) prim={} samples>0={} | batch out-verdict={} cprim>0={} | top vs "
                  "wrong/vfy: {}",
                  a.cull_vfy, a.cull_vfy_prim, a.cull_wrong,
                  100.0 * double(a.cull_wrong) / double(a.cull_vfy ? a.cull_vfy : 1),
                  a.cull_wrong_prim, a.cull_wrong_smp, a.cull_vfy_batch, a.cull_wrong_batch, top);
    }
  }
  occ_acc_ = OccAcc{};
  occ_trunc_ = 0;
  occ_dropped_ = 0;
}

// [gpu-census] draw sub-split: map the bound RT config to a stable dynamic
// class. Called by the RT cache on every SetCommandListRenderTargets; the
// same-config early-out keeps it cheap. If the current class is already a
// draw class, transition immediately so the next recorded commands (the
// first draw of the new pass) land on the right config.
void D3D12CommandProcessor::GpuCensusSetDrawConfig(const uint32_t rt_keys[5], const char* desc) {
  if (!gpu_census_available_) {
    return;
  }
  if (!std::memcmp(rt_keys, gpu_census_last_rt_keys_, sizeof(gpu_census_last_rt_keys_))) {
    return;
  }
  std::memcpy(gpu_census_last_rt_keys_, rt_keys, sizeof(gpu_census_last_rt_keys_));
  uint32_t config_index = UINT32_MAX;
  for (uint32_t i = 0; i < gpu_census_draw_config_count_; ++i) {
    if (!std::memcmp(gpu_census_draw_configs_[i].rt_keys, rt_keys, sizeof(uint32_t) * 5)) {
      config_index = i;
      break;
    }
  }
  if (config_index == UINT32_MAX) {
    if (gpu_census_draw_config_count_ < kGpuCensusDrawConfigs) {
      config_index = gpu_census_draw_config_count_++;
      GpuCensusDrawConfig& config = gpu_census_draw_configs_[config_index];
      std::memcpy(config.rt_keys, rt_keys, sizeof(uint32_t) * 5);
      snprintf(config.desc, sizeof(config.desc), "%s", desc ? desc : "?");
    } else {
      // Table full - fall back to the generic draw bucket.
      gpu_census_draw_class_ = kGpuCensusDraw;
      return;
    }
  }
  uint8_t new_class = uint8_t(kGpuCensusClassCount + config_index);
  uint8_t old_class = gpu_census_draw_class_;
  gpu_census_draw_class_ = new_class;
  if (gpu_census_sub_active_ &&
      (gpu_census_class_ == old_class || gpu_census_class_ == kGpuCensusDraw ||
       gpu_census_class_ >= kGpuCensusClassCount)) {
    GpuCensusSetClass(new_class);
  }
}

void D3D12CommandProcessor::GpuCensusReport1Hz() {
  if (!gpu_census_available_) {
    return;
  }
  GpuCensusConsumeCompleted();
  uint64_t total_ticks = 0;
  for (uint32_t i = 0; i < kGpuCensusTotalClasses; ++i) {
    total_ticks += gpu_census_class_ticks_[i];
  }
  // "draw" in the main line = generic bucket + every per-RT-config class.
  uint64_t draw_ticks = gpu_census_class_ticks_[kGpuCensusDraw];
  uint64_t draw_spans = gpu_census_class_spans_[kGpuCensusDraw];
  for (uint32_t i = kGpuCensusClassCount; i < kGpuCensusTotalClasses; ++i) {
    draw_ticks += gpu_census_class_ticks_[i];
    draw_spans += gpu_census_class_spans_[i];
  }
  const double to_ms = gpu_census_ticks_to_ms_;
  REXGPU_INFO(
      "[gpu-census] gpu {:.1f}ms/s | draw {:.1f}/{} xfer {:.1f}/{} resolve {:.1f}/{} "
      "texup {:.1f}/{} memup {:.1f}/{} swap {:.1f}/{} hiz {:.1f}/{} hizt {:.1f}/{} hizd "
      "{:.1f}/{} other {:.1f}/{} | trunc {} drop {}",
      double(total_ticks) * to_ms, double(draw_ticks) * to_ms, draw_spans,
      double(gpu_census_class_ticks_[kGpuCensusXfer]) * to_ms,
      gpu_census_class_spans_[kGpuCensusXfer],
      double(gpu_census_class_ticks_[kGpuCensusResolve]) * to_ms,
      gpu_census_class_spans_[kGpuCensusResolve],
      double(gpu_census_class_ticks_[kGpuCensusTexUp]) * to_ms,
      gpu_census_class_spans_[kGpuCensusTexUp],
      double(gpu_census_class_ticks_[kGpuCensusMemUp]) * to_ms,
      gpu_census_class_spans_[kGpuCensusMemUp],
      double(gpu_census_class_ticks_[kGpuCensusSwap]) * to_ms,
      gpu_census_class_spans_[kGpuCensusSwap],
      double(gpu_census_class_ticks_[kGpuCensusHiz]) * to_ms,
      gpu_census_class_spans_[kGpuCensusHiz],
      double(gpu_census_class_ticks_[kGpuCensusHizTest]) * to_ms,
      gpu_census_class_spans_[kGpuCensusHizTest],
      double(gpu_census_class_ticks_[kGpuCensusHizDepth]) * to_ms,
      gpu_census_class_spans_[kGpuCensusHizDepth],
      double(gpu_census_class_ticks_[kGpuCensusOther]) * to_ms,
      gpu_census_class_spans_[kGpuCensusOther], gpu_census_trunc_, gpu_census_dropped_);
  // [gpu-census-draw] sub-split: top RT configs by GPU ms this second.
  // Configs under 1 ms are noise; the generic bucket prints as "other".
  {
    uint32_t order[kGpuCensusDrawConfigs];
    for (uint32_t i = 0; i < gpu_census_draw_config_count_; ++i) {
      order[i] = kGpuCensusClassCount + i;
    }
    std::sort(order, order + gpu_census_draw_config_count_,
              [this](uint32_t a, uint32_t b) {
                return gpu_census_class_ticks_[a] > gpu_census_class_ticks_[b];
              });
    std::string draw_line;
    uint32_t printed = 0;
    for (uint32_t i = 0; i < gpu_census_draw_config_count_ && printed < 8; ++i) {
      uint32_t cls = order[i];
      double ms = double(gpu_census_class_ticks_[cls]) * to_ms;
      if (ms < 1.0) {
        break;
      }
      draw_line += fmt::format("[{}] {:.1f}/{} | ",
                               gpu_census_draw_configs_[cls - kGpuCensusClassCount].desc, ms,
                               gpu_census_class_spans_[cls]);
      ++printed;
    }
    if (gpu_census_class_ticks_[kGpuCensusDraw]) {
      draw_line += fmt::format("other {:.1f}/{}",
                               double(gpu_census_class_ticks_[kGpuCensusDraw]) * to_ms,
                               gpu_census_class_spans_[kGpuCensusDraw]);
    }
    if (!draw_line.empty()) {
      REXGPU_INFO("[gpu-census-draw] {}", draw_line);
    }
  }
  std::memset(gpu_census_class_ticks_, 0, sizeof(gpu_census_class_ticks_));
  std::memset(gpu_census_class_spans_, 0, sizeof(gpu_census_class_spans_));
  gpu_census_trunc_ = 0;
  gpu_census_dropped_ = 0;
}

bool D3D12CommandProcessor::InitializeOcclusionQueryResources() {
  active_occlusion_query_ = {};
  occlusion_query_cursor_ = 0;
  occlusion_query_resources_available_ = false;
  occlusion_query_heap_.Reset();
  occlusion_query_readback_.Reset();
  occlusion_query_readback_mapping_ = nullptr;

  ID3D12Device* device = GetD3D12Provider().GetDevice();
  if (!device) {
    return false;
  }

  D3D12_QUERY_HEAP_DESC heap_desc;
  heap_desc.Type = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
  heap_desc.Count = kMaxOcclusionQueries;
  heap_desc.NodeMask = 0;
  if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&occlusion_query_heap_)))) {
    REXGPU_WARN(
        "D3D12CommandProcessor: Failed to create occlusion query heap, using fake sample counts");
    return false;
  }

  D3D12_RESOURCE_DESC buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(buffer_desc, sizeof(uint64_t) * kMaxOcclusionQueries,
                                          D3D12_RESOURCE_FLAG_NONE);
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesReadback,
                                             GetD3D12Provider().GetHeapFlagCreateNotZeroed(),
                                             &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&occlusion_query_readback_)))) {
    REXGPU_WARN(
        "D3D12CommandProcessor: Failed to allocate occlusion query readback buffer, using fake "
        "sample counts");
    occlusion_query_heap_.Reset();
    return false;
  }

  D3D12_RANGE read_range = {0, sizeof(uint64_t) * kMaxOcclusionQueries};
  void* mapping = nullptr;
  if (FAILED(occlusion_query_readback_->Map(0, &read_range, &mapping))) {
    REXGPU_WARN(
        "D3D12CommandProcessor: Failed to map occlusion query readback buffer, using fake sample "
        "counts");
    occlusion_query_readback_.Reset();
    occlusion_query_heap_.Reset();
    return false;
  }

  occlusion_query_readback_mapping_ = reinterpret_cast<uint64_t*>(mapping);
  occlusion_query_resources_available_ = true;
  return true;
}

void D3D12CommandProcessor::ShutdownOcclusionQueryResources() {
  DisableHostOcclusionQueries();

  if (occlusion_query_readback_ && occlusion_query_readback_mapping_) {
    occlusion_query_readback_->Unmap(0, nullptr);
  }
  occlusion_query_readback_mapping_ = nullptr;
  occlusion_query_readback_.Reset();
  occlusion_query_heap_.Reset();
}

bool D3D12CommandProcessor::AcquireOcclusionQueryIndex(uint32_t& host_index_out) {
  if (occlusion_query_cursor_ >= kMaxOcclusionQueries) {
    occlusion_query_cursor_ = 0;
  }
  host_index_out = occlusion_query_cursor_++;
  return true;
}

void D3D12CommandProcessor::DisableHostOcclusionQueries() {
  if (active_occlusion_query_.valid && occlusion_query_heap_) {
    uint32_t host_index = active_occlusion_query_.host_index;
    // Clear before EndSubmission to prevent the EndSubmission safety net from issuing a second
    // EndQuery for the same index.
    active_occlusion_query_ = {};
    if (BeginSubmission(true)) {
      deferred_command_list_.D3DEndQuery(occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION,
                                         host_index);
      EndSubmission(false);
    }
  } else {
    active_occlusion_query_ = {};
  }
  occlusion_query_cursor_ = 0;
  occlusion_query_resources_available_ = false;
}

bool D3D12CommandProcessor::BeginGuestOcclusionQuery(uint32_t sample_count_address) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_) {
    return false;
  }
  if (active_occlusion_query_.valid) {
    REXGPU_WARN(
        "D3D12CommandProcessor: Occlusion query begin issued while another query is active");
    DisableHostOcclusionQueries();
    return false;
  }

  uint32_t host_index = 0;
  if (!AcquireOcclusionQueryIndex(host_index)) {
    return false;
  }
  if (!BeginSubmission(true)) {
    return false;
  }

  deferred_command_list_.D3DBeginQuery(occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION,
                                       host_index);
  active_occlusion_query_.sample_count_address = sample_count_address;
  active_occlusion_query_.host_index = host_index;
  active_occlusion_query_.valid = true;
  return true;
}

bool D3D12CommandProcessor::EndGuestOcclusionQuery(
    uint32_t sample_count_address, xenos::xe_gpu_depth_sample_counts* sample_counts) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_ ||
      !active_occlusion_query_.valid || !occlusion_query_heap_ || !occlusion_query_readback_) {
    return false;
  }

  uint32_t host_index = active_occlusion_query_.host_index;
  active_occlusion_query_ = {};

  if (!BeginSubmission(true)) {
    return false;
  }

  deferred_command_list_.D3DEndQuery(occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION,
                                     host_index);
  deferred_command_list_.D3DResolveQueryData(
      occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, host_index, 1,
      occlusion_query_readback_.Get(), sizeof(uint64_t) * host_index);

  if (!EndSubmission(false)) {
    return false;
  }

  uint64_t query_submission = submission_current_ ? submission_current_ - 1 : 0;
  CheckSubmissionFence(query_submission);
  if (submission_completed_ < query_submission) {
    return false;
  }
  if (!occlusion_query_readback_mapping_) {
    return false;
  }

  uint64_t samples = occlusion_query_readback_mapping_[host_index];
  samples = NormalizeOcclusionSamples(samples);
  WriteGuestOcclusionResult(sample_counts, samples);
  return true;
}

uint64_t D3D12CommandProcessor::NormalizeOcclusionSamples(uint64_t samples) const {
  if (samples == 0 || !texture_cache_) {
    return samples;
  }
  uint64_t scale_x = texture_cache_->draw_resolution_scale_x();
  uint64_t scale_y = texture_cache_->draw_resolution_scale_y();
  uint64_t scale = scale_x * scale_y;
  if (scale <= 1) {
    return samples;
  }
  return (samples + (scale >> 1)) / scale;
}

void D3D12CommandProcessor::WriteGuestOcclusionResult(
    xenos::xe_gpu_depth_sample_counts* sample_counts, uint64_t samples) {
  if (!sample_counts) {
    return;
  }
  uint32_t clamped = samples > uint64_t(UINT32_MAX) ? UINT32_MAX : uint32_t(samples);
  sample_counts->Total_A = clamped;
  sample_counts->Total_B = 0;
  sample_counts->ZPass_A = clamped;
  sample_counts->ZPass_B = 0;
  sample_counts->ZFail_A = 0;
  sample_counts->ZFail_B = 0;
  sample_counts->StencilFail_A = 0;
  sample_counts->StencilFail_B = 0;
}

void D3D12CommandProcessor::WriteGammaRampSRV(bool is_pwl,
                                              D3D12_CPU_DESCRIPTOR_HANDLE handle) const {
  ID3D12Device* device = GetD3D12Provider().GetDevice();
  D3D12_SHADER_RESOURCE_VIEW_DESC desc;
  desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
  desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  desc.Buffer.StructureByteStride = 0;
  desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
  if (is_pwl) {
    desc.Format = DXGI_FORMAT_R16G16_UINT;
    desc.Buffer.FirstElement = 256 * 4 / 4;
    desc.Buffer.NumElements = 128 * 3;
  } else {
    desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = 256;
  }
  device->CreateShaderResourceView(gamma_ramp_buffer_.Get(), &desc, handle);
}

}  // namespace rex::graphics::d3d12
