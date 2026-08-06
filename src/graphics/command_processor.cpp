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
#include <string_view>

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
#include <rex/graphics/nr_draw_cache.h>
#include <rex/graphics/nr_state_walk.h>
#include <rex/graphics/nr_draw_registry.h>
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
// immediate). Sizes increment 4b's live translation cache (offline corpus
// caps the honest answer at 3,320 unique).
constexpr uint32_t kCtxShaderSetSize = 4096;  // power of two
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

void CtxShaderSeen(void*, const nr::ShaderRef& ref) {
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
  const bool kNrCtx = REXCVAR_GET(gpu_nr_ctx);
  g_nr_ctx = kNrCtx;
  const bool kNrCache = REXCVAR_GET(gpu_nr_cache) || kNrBuf || kNrState || kNrCtx;
  g_nr_cache = kNrCache;
  const bool kIbLedger = REXCVAR_GET(gpu_ib_ledger) || kNrCache;  // [NR-IBL] record/replay gate
  g_ib_ledger = kIbLedger;
  g_pm4_ib_dump = REXCVAR_GET(gpu_pm4_ib_dump);  // [PM4-IB-DUMP] one-shot IB structure dump
  const bool kTimeExec = kProfile || kSplit;  // bracket ExecutePrimaryBuffer when either is on
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
      auto spin_t0 = kProfile ? prof_clock::now() : prof_clock::time_point{};
      if (kProfile) prof_starves++;
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
      if (kProfile) {
        prof_spin_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            prof_clock::now() - spin_t0).count();
      }
      if (!worker_running_ || !pending_fns_.empty()) {
        continue;
      }
    }
    assert_true(read_ptr_index_ != write_ptr_index);

    // Execute. Note that we handle wraparound transparently.
    {
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

    // [GPU-WORKER-PROFILE] / [GPU-SPLIT] / [PM4-CENSUS] ~1s wall-time report.
    if (kProfile || kSplit || kCensus || kIbLedger) {
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
            nr::ResetCacheStats();
            g_nrc_execs = g_nrc_execs_full = 0;
            g_nrc_pkts = g_nrc_hit = g_nrc_miss = 0;
            g_nrc_arg_eq = g_nrc_prim_ne = g_nrc_cnt_ne = g_nrc_rid0 = 0;
            g_nrc_samp_n = 0;
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
          // Clear for the next window. Addresses are re-walked when they
          // reappear, which is what keeps a stale draw count from surviving a
          // buffer being re-recorded at the same address with different content.
          for (uint32_t i = 0; i < kIbLedgerSize; ++i) g_ib_ledger_tab[i] = IbLedgerEntry{};
          g_ib_ledger_used = 0;
          g_ib_ledger_evictions = 0;
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

  trace_writer_.WriteIndirectBufferStart(ptr, count * sizeof(uint32_t));

  // [PM4-CENSUS] track indirect-buffer nesting + size. Depth is unconditional
  // (cheap; keeps it correct if gpu_pm4_census toggles mid-buffer); the census
  // tallies are gated on the cvar. Draw packets executed below (via ExecutePacket
  // -> ExecutePacketType3) will see g_pm4_ib_depth > 0 and count as "indirect".
  ++g_pm4_ib_depth;
  if (g_pm4_census) { ++g_pm4_ib_count; g_pm4_ib_dwords += count; }

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
      nr::CtxWalkStats cst;
      g_nr_walk_buf = ptr;
      const uint32_t nf = nr::WalkBufferContext(
          memory_->TranslatePhysical(ptr), count, ptr, &g_ctx_state,
          g_ctx_flags, kNrbMaxPkts, &cst, CtxMemRead, memory_, CtxShaderSeen,
          nullptr, NrCtxWatch, nullptr, bin_select_, bin_mask_);
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
                g_nrb_pkts[npkt++] = {ptr + j * 4, init & 0x3F, init >> 16};
              }
              j += 1 + cnt;
            } else if (ty == 0) {
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
            for (uint32_t i = 0; i < npkt; ++i) {
              const nr::PacketRef& pk = g_nrb_pkts[i];
              ++g_nrc_pkts;
              nr::DrawRecord rec;
              if (!nr::LookupDraw(pk.addr, &rec)) {
                ++g_nrc_miss;
                full = false;
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
                    if (g_nrc_samp_n < kNrcSamples) {
                      g_nrc_samp[g_nrc_samp_n++] = {
                          ptr,     pk.addr,  i,         pk.prim, pk.count,
                          rec.rid, rec.prim, rec.start, rec.count,
                          0};
                    }
                  }
                }
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

  --g_pm4_ib_depth;

  trace_writer_.WriteIndirectBufferEnd();
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
    if (!any_pass || opcode == PM4_XE_SWAP) {
      reader->AdvanceRead(count * sizeof(uint32_t));
      trace_writer_.WritePacketEnd();
      return true;
    }
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

  IssueSwap(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);

  ++counter_;
  // Pure presented-frame tally (real swaps only, no vblank pollution).
  swap_counter_.fetch_add(1, std::memory_order_relaxed);
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
                                              const char* opcode_name, uint32_t viz_query_condition,
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
      const bool kTimeDraw = g_exec_prof || g_split_prof;
      const auto draw_t0 = kTimeDraw ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      draw_succeeded = IssueDraw(vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
                                 is_indexed ? &index_buffer_info : nullptr, major_mode_explicit);
      if (kTimeDraw) {
        g_issuedraw_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - draw_t0).count();
        g_draw_cnt++;
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

  // If read the packed correctly, but merely couldn't execute it (because of,
  // for instance, features not supported by the host), don't terminate command
  // buffer processing as that would leave rendering in a way more inconsistent
  // state than just a single dropped draw command.
  return true;
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
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX", viz_query_condition,
                                count_remaining);
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX_2(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // "draw using supplied indices in packet"
  // Generally used by Xbox 360 Direct3D 9 for kAutoIndex source.
  // No viz query token.
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX_2", 0, count);
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
