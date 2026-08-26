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

#pragma once

#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/trace_writer.h>
#include <rex/graphics/xenos.h>
#include <rex/memory.h>
#include <rex/memory/ring_buffer.h>
#include <rex/system/xthread.h>
#include <rex/thread.h>
#include <rex/ui/presenter.h>

namespace rex::stream {
class ByteStream;
}  // namespace rex::stream

namespace rex::graphics {

class GraphicsSystem;
class Shader;

// [N8B] The constant-census phase, owned by the `gpu_n7 3` in-place cycler in
// command_processor.cpp and read once a frame by the D3D12 backend, which is
// where the constant ranges are actually applied. -1 = not cycling, follow the
// gpu_dedupe_constants cvar; 0 = off, 1 = census only, 2 = census + dedupe.
// Command-processor thread only, so a plain int is correct.
extern int g_n7_n8b_phase;

// [NR-5C] Register-write epoch map for the skip-unchanged-record-applies
// lever (gpu_nr_apply_skip). Every path that writes the register file at or
// above 0x2000 stamps its 32-register bucket with the current apply
// sequence; a record whose runs' buckets are all untouched since its last
// full apply still has its values in the register file and may skip
// re-applying them. The sequence is bumped once at each compose entry and
// once at each compose exit, so any write OUTSIDE a record's own apply pass
// (walk per-dword, ring inline, delegates, the executor fallback) always
// stamps strictly later than the seq the record stored -- a same-seq stamp
// can only be the record's own apply. CP thread only, plain ints.
constexpr uint32_t kNrbBucketShift = 5;  // 32 registers per bucket
constexpr uint32_t kNrbBuckets =
    (RegisterFile::kRegisterCount >> kNrbBucketShift) + 1;
extern bool g_nrb_track;
extern uint32_t g_nrb_apply_seq;
extern uint32_t g_nrb_reg_epoch[kNrbBuckets];
inline void NrbStampRange(uint32_t base, uint32_t n) {
  if (!g_nrb_track || !n) return;
  const uint32_t b1 = (base + n - 1) >> kNrbBucketShift;
  for (uint32_t b = base >> kNrbBucketShift; b <= b1 && b < kNrbBuckets; ++b) {
    g_nrb_reg_epoch[b] = g_nrb_apply_seq;
  }
}

enum class ReadbackResolveMode {
  kDisabled,
  kFast,
  kSome,
  kFull,
};

struct SwapState {
  // Lock must be held when changing data in this structure.
  std::mutex mutex;
  // Dimensions of the framebuffer textures. Should match window size.
  uint32_t width = 0;
  uint32_t height = 0;
  // Current front buffer, being drawn to the screen.
  uintptr_t front_buffer_texture = 0;
  // Current back buffer, being updated by the CP.
  uintptr_t back_buffer_texture = 0;
  // Backend data
  void* backend_data = nullptr;
  // Whether the back buffer is dirty and a swap is pending.
  bool pending = false;
};

enum class SwapMode {
  kNormal,
  kIgnored,
};

enum class GammaRampType {
  kUnknown = 0,
  kTable,
  kPWL,
};

class CommandProcessor {
 public:
  enum class SwapPostEffect {
    kNone,
    kFxaa,
    kFxaaExtreme,
  };

  CommandProcessor(GraphicsSystem* graphics_system, system::KernelState* kernel_state);
  virtual ~CommandProcessor();

  uint32_t counter() const { return counter_; }
  void increment_counter() { counter_++; }

  // Pure presented-frame counter: incremented ONLY on a real guest swap
  // (XE_SWAP packet), unlike counter() which is also bumped by every vblank in
  // GraphicsSystem::MarkVblank(). Use this to measure the true game frame rate.
  uint32_t swap_counter() const { return swap_counter_.load(std::memory_order_relaxed); }

  // [GPU-PRECORD] Phase 1b-1c Inc 2: while a captured segment is being replayed, each
  // replayed draw's own active shaders come from the per-draw replay fields (set in
  // D3D12CommandProcessor::PrecordReplayEvents) instead of the shared active_*_shader_
  // members, which the parse thread advances to the LATEST loaded shader as it runs
  // ahead under Phase 1c overlap. (nullptr is a valid shader value, so a bool gates
  // this, not a null sentinel.) Only the draw path (IssueDrawImpl) reads these
  // accessors, and it runs only during replay or a normal non-precord draw -- never on
  // the parse thread mid-capture (draws are deferred) -- so the replay fields are
  // touched by a single thread.
  Shader* active_vertex_shader() const {
    if (nr_issue_shaders_active_) return nr_issue_vertex_shader_;
    return precord_replay_shaders_active_ ? precord_replay_active_vertex_shader_
                                          : active_vertex_shader_;
  }
  Shader* active_pixel_shader() const {
    if (nr_issue_shaders_active_) return nr_issue_pixel_shader_;
    return precord_replay_shaders_active_ ? precord_replay_active_pixel_shader_
                                          : active_pixel_shader_;
  }

  virtual bool Initialize();
  virtual void Shutdown();

  void CallInThread(std::function<void()> fn);

  virtual void ClearCaches();
  virtual void InvalidateGpuMemory();

  // "Desired" is for the external thread managing the post-processing effect.
  SwapPostEffect GetDesiredSwapPostEffect() const { return swap_post_effect_desired_; }
  void SetDesiredSwapPostEffect(SwapPostEffect swap_post_effect);
  // Implementations must not make assumptions that the front buffer will
  // necessarily be a resolve destination - it may be a texture generated by any
  // means like written to by the CPU or loaded from a file (the disclaimer
  // screen right in the beginning of 4D530AA4 is not a resolved render target,
  // for instance).
  virtual void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                         uint32_t frontbuffer_height) = 0;

  // May be called not only from the command processor thread when the command
  // processor is paused, and the termination of this function may be explicitly
  // awaited.
  virtual void InitializeShaderStorage(const std::filesystem::path& cache_root, uint32_t title_id,
                                       bool blocking);

  virtual void RequestFrameTrace(const std::filesystem::path& root_path);
  virtual void BeginTracing(const std::filesystem::path& root_path);
  virtual void EndTracing();

  virtual void TracePlaybackWroteMemory(uint32_t base_ptr, uint32_t length) = 0;

  void RestoreRegisters(uint32_t first_register, const uint32_t* register_values,
                        uint32_t register_count, bool execute_callbacks);
  void RestoreGammaRamp(const reg::DC_LUT_30_COLOR* new_gamma_ramp_256_entry_table,
                        const reg::DC_LUT_PWL_DATA* new_gamma_ramp_pwl_rgb,
                        uint32_t new_gamma_ramp_rw_component);
  virtual void RestoreEdramSnapshot(const void* snapshot) = 0;

  void InitializeRingBuffer(uint32_t ptr, uint32_t size_log2);
  void EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2);

  void UpdateWritePointer(uint32_t value);

  void ExecutePacket(uint32_t ptr, uint32_t count);

  bool is_paused() const { return paused_; }
  void Pause();
  void Resume();

  bool Save(::rex::stream::ByteStream* stream);
  bool Restore(::rex::stream::ByteStream* stream);

  // [NR-FX] Phase 5-4-0: fired by the lockstep walk (NrWalkRegWrite) for every
  // decoded register write while gpu_nr_walk_effects is on. A backend override
  // runs WriteRegister's dirty-tracking tail (cbuffer/texture/vertex-residency
  // invalidation) WITHOUT the value store, so the walk can drive the draw-path
  // subsystems. Idempotent with the executor's own firing while both run.
  // Public because the walk's write hook is a file-scope function.
  virtual void NrWalkWriteEffects(uint32_t index) { (void)index; }

  // [NR-SKP] Phase 5-4-2: the skip mode's register apply -- the walk's decoded
  // write stream routed through the FULL virtual WriteRegister (value store,
  // dirty tail, stateful ports, instancing/dedupe semantics), because under
  // the skip the executor's own apply never runs. Public for the same reason
  // as NrWalkWriteEffects: the walk hook is a file-scope function.
  void NrSkipApplyRegWrite(uint32_t index, uint32_t value) {
    WriteRegister(index, value);
  }

  // [NR-SKP] Phase 5-4-3: the skip mode's RANGE apply -- a walk-decoded
  // contiguous constant range routed through the virtual WriteRegistersFromMem
  // (the executor's own bulk path for SET_CONSTANT/LOAD_ALU_CONSTANT: one
  // value copy_and_swap + ONE dirty-tail evaluation per range, per-constant
  // fetch hooks preserved inside it), then the probe mirrors fed from the
  // stored host-order values in tight loops. `values_be` = big-endian dwords
  // (inline packet data), or nullptr for a by-reference range whose values
  // live in guest memory at physical `phys`. Returns false (range not
  // applied) only when the caller should fall back to the per-dword path.
  // Public for the same reason as NrSkipApplyRegWrite; defined in
  // command_processor.cpp next to the probe state it feeds.
  // [N8C] The body of one range apply. NrSkipApplyRegRange resolves the guest
  // pointer, runs the contiguity census on the range AS THE WALK DECODED IT,
  // and then calls this once -- or k times over equal sub-ranges when the
  // synthetic-split probe is armed, which is the direct measurement of what
  // one range costs.
  bool NrSkipApplyRegRangeBody(uint32_t base, uint32_t* be, uint32_t n,
                               bool from_memory);

  // [NR-5B-3] suppressed-compose helper: translate a guest physical address
  // for a by-ref marker's per-dword apply (defined in the .cpp).
  const uint32_t* NrTranslatePhys(uint32_t phys);
  bool NrSkipApplyRegRange(uint32_t base, const uint32_t* values_be,
                           uint32_t n, uint32_t phys, bool from_memory);
  // [NR-6] Live register read for the file-local compose helpers (they are
  // free functions, so the protected ReadRegisterValue is out of reach).
  // The N-9-6 band predicate must re-read the surface signature AFTER a
  // suppressed record's mirror-class writes land, which happens inside the
  // compose (naruto_724).
  uint32_t NrPeekReg(uint32_t index) const {
    return register_file_->values[index];
  }

  // [N8F] Per-draw effect coalescing: store a walk-decoded range's VALUES
  // immediately (register file + issue mirror, plain copy_and_swap, no
  // virtual dispatch), accumulate the touched span, and fire the class side
  // effects (cbuffer dirty evaluation, fetch texture/residency
  // invalidations) once per MERGED span at the draw stop. Register-file
  // READERS between ranges always see live values; only the draw-time dirty
  // notifications are deferred, and every consumer of those is the draw.
  // Returns false for a range that straddles a constant-window boundary --
  // the caller falls through to the legacy apply.
  bool N8fApplyRange(uint32_t base, uint32_t* be, uint32_t n);
  // Fire the deferred effects for every accumulated span, in store order.
  // Called before each draw/resolve dispatch, before a delegated nested
  // indirect buffer, at an executor abort, and at buffer end.
  void N8fFlush();

  // [NR-BFC] Phase 5-4-6-0: what the backend measured across one skip-driven
  // buffer execution, for the buffer-level native-replay census. Filled by
  // NrBfcBufEnd from deltas since the matching NrBfcBufBegin. Public for the
  // same reason as NrWalkWriteEffects: the census fold is a file-scope
  // function.
  struct NrBfcBackendSample {
    uint64_t submission_id = 0;   // native stream generation (span validity)
    uint64_t rt_body_runs = 0;    // RenderTargetCache::Update body executions
    uint32_t span_elements = 0;   // deferred-list elements this buffer emitted
    // Command-class counts over the span (the would-be recorded sequence):
    uint32_t cmd_draw = 0;        // DrawIndexedInstanced / DrawInstanced
    uint32_t cmd_pso = 0;         // pipeline sets (incl. handle indirection)
    uint32_t cmd_sys_cbv = 0;     // graphics CBV sets on the sys-constants slot
    uint32_t cmd_root_cbv = 0;    // other CBV sets
    uint32_t cmd_root_other = 0;  // root sig/table/SRV/UAV/32bit sets
    uint32_t cmd_ia = 0;          // IB/VB/topology sets
    uint32_t cmd_vp = 0;          // RSSetViewport sites (bin fixup class)
    uint32_t cmd_sci = 0;         // RSSetScissorRect sites (bin fixup class)
    uint32_t cmd_om_rt = 0;       // OMSetRenderTargets sites
    uint32_t cmd_om_misc = 0;     // blend factor / stencil ref / sample pos
    uint32_t cmd_barrier = 0;     // whitelist violations from here down:
    uint32_t cmd_copy = 0;        //   non-idempotent / data-dependent
    uint32_t cmd_clear = 0;
    uint32_t cmd_dispatch = 0;
    uint32_t cmd_query = 0;
    uint32_t cmd_marker = 0;      // debug markers (whitelisted)
    uint32_t cmd_heaps = 0;       // SetDescriptorHeaps (preamble class)
    uint32_t cmd_other = 0;
  };
  // [NR-BFC] Bracket one skip-driven buffer execution. Begin latches the
  // backend's stream position / counters; End fills the sample with deltas
  // and returns true when the backend actually measured (D3D12 only).
  virtual void NrBfcBufBegin() {}
  virtual bool NrBfcBufEnd(NrBfcBackendSample* out) {
    (void)out;
    return false;
  }

  // [NR-DSP] Phase 5-4-7-0: bracket ONE draw's native command emission so the
  // backend can compare the span it emits now against the span the same draw
  // emitted at its previous execution. `reusable` is the reuse model's own
  // verdict for this draw, so the probe measures exactly the population a
  // per-draw span replay would serve.
  virtual void NrDspDrawBegin(uint32_t key, bool reusable) {
    (void)key;
    (void)reusable;
  }
  virtual void NrDspDrawEnd() {}

  // [NR-SPR] Phase 5-4-7-1: bracket ONE draw for the production span-replay
  // store. Begin FORCES the backend's tail-state re-emit (pipeline, root
  // signature, root parameters, topology) so the emitted span is
  // context-free -- recordable once, replayable in any dedupe context. End
  // scans the span against the replayable whitelist, stores the first clean
  // recording per draw key, and on later reusable executions compares the
  // fixed recording (patch model: root-view addresses) against the fresh
  // emission. Compare-only: fresh always draws.
  virtual void NrSprDrawBegin(uint32_t key, bool reusable) {
    (void)key;
    (void)reusable;
  }
  virtual void NrSprDrawEnd() {}

 protected:
  // [NR-TIL] N-4-1: the guest packet address of the draw the walk is
  // dispatching, latched at the draw stop so the tile replay can key a
  // recording by stream POSITION (bands re-execute the same packets in the
  // same order). A nested-buffer draw inherits its delegate's address, which
  // is still a deterministic function of the position - the shape guard
  // (primitive, index count, index base) is what separates two draws that
  // land on the same value.
  uint32_t nr_tile_draw_addr_ = 0;

  struct IndexBufferInfo {
    xenos::IndexFormat format = xenos::IndexFormat::kInt16;
    xenos::Endian endianness = xenos::Endian::kNone;
    uint32_t count = 0;
    uint32_t guest_base = 0;
    size_t length = 0;
  };

  void WorkerThreadMain();
  virtual bool SetupContext() = 0;
  virtual void ShutdownContext() = 0;

  virtual void WriteRegister(uint32_t index, uint32_t value);
  uint32_t ReadRegisterValue(uint32_t index) const;
  virtual void WriteRegistersFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers);
  // [NR-PB] N-2-2 item 0: bulk value store for a range of PLAIN registers --
  // no stateful port (scratch writeback, COHER, DC_LUT: all indices below
  // 0x2000), no constant-window dirty tail, no extended registers. The caller
  // (NrSkipApplyRegRange) guarantees base >= 0x2000, base + n <=
  // kRegisterCount, and no intersection with the three constant windows, so
  // the full virtual WriteRegister would have done nothing but the value
  // store for every dword. Overridden by D3D12 for the gpu_instance
  // dirty-tracking semantic only.
  virtual void WriteRegisterRangePlain(uint32_t base, uint32_t* values_be, uint32_t n);
  virtual void WriteRegisterRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                          uint32_t num_registers);
  void WriteALURangeFromRing(memory::RingBuffer* ring, uint32_t base, uint32_t num_registers);
  void WriteFetchRangeFromRing(memory::RingBuffer* ring, uint32_t base, uint32_t num_registers);
  void WriteBoolRangeFromRing(memory::RingBuffer* ring, uint32_t base, uint32_t num_registers);
  void WriteLoopRangeFromRing(memory::RingBuffer* ring, uint32_t base, uint32_t num_registers);
  void WriteREGISTERSRangeFromRing(memory::RingBuffer* ring, uint32_t base, uint32_t num_registers);
  void WriteALURangeFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers);
  void WriteFetchRangeFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers);
  void WriteBoolRangeFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers);
  void WriteLoopRangeFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers);
  void WriteREGISTERSRangeFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers);

  const reg::DC_LUT_30_COLOR* gamma_ramp_256_entry_table() const {
    return gamma_ramp_256_entry_table_;
  }
  const reg::DC_LUT_PWL_DATA* gamma_ramp_pwl_rgb() const { return gamma_ramp_pwl_rgb_[0]; }
  virtual void OnGammaRamp256EntryTableValueWritten() {}
  virtual void OnGammaRampPWLValueWritten() {}

  virtual void MakeCoherent();
  virtual void PrepareForWait();
  virtual void ReturnFromWait();

  uint32_t ExecutePrimaryBuffer(uint32_t start_index, uint32_t end_index);
  virtual void OnPrimaryBufferEnd() {}
  void ExecuteIndirectBuffer(uint32_t ptr, uint32_t length);
  // [NR-SKP] Phase 5-4-2: backend veto for the skip mode (precord capture and
  // non-D3D12 backends refuse). Base default false keeps every backend that
  // has not opted in on the executor path.
  virtual bool NrSkipBackendEligible() const { return false; }
  // [N8F] Deferred-effects support. NrApplyRangeEffects re-fires exactly the
  // side-effect half of the backend's WriteRegistersFromMem for a range whose
  // VALUES are already in the register file (the value copy has happened at
  // decode time). A backend that does not implement it must return false
  // from NrCoalesceEligible so the coalescer never arms there.
  virtual void NrApplyRangeEffects(uint32_t base, uint32_t n) {}
  virtual bool NrCoalesceEligible() const { return false; }
  // [NR-SKP] Runs one eligible depth-1 indirect buffer with the walk as the
  // ONLY decoder: native packets applied through NrSkipApplyRegWrite, draws
  // and the 5-4-1 delegate list dispatched to the executor's own handlers at
  // the walk cursor. Called from ExecuteIndirectBuffer instead of the packet
  // loop; the walker must already be begun for this buffer.
  void NrSkipExecuteBuffer(uint32_t ptr, uint32_t count);
  // [NR-SKP] Phase 5-4-4a: issue one draw stop by direct call, skipping the
  // per-draw delegated re-dispatch (span reader + ExecutePacket + packet
  // re-parse). The walk has already applied the packet's own register payload
  // through the full virtual WriteRegister before the stop returned, so the
  // only work left is the arg derivation -- reproduced from the same buffer
  // dwords ExecutePacketType3Draw reads, read for read -- and the extracted
  // tail. Returns false (nothing ran, caller delegates) for any shape but a
  // well-formed kDMA/kAutoIndex draw, so every odd case -- short packet,
  // truncated buffer, immediate/invalid source select -- keeps the proven
  // handler and its exact log/abort behavior.
  bool NrSkipDrawDirect(uint32_t opcode, uint32_t dword, const uint8_t* raw,
                        uint32_t count);
  bool ExecutePacket(memory::RingBuffer* reader);
  bool ExecutePacketType0(memory::RingBuffer* reader, uint32_t packet);
  bool ExecutePacketType1(memory::RingBuffer* reader, uint32_t packet);
  bool ExecutePacketType2(memory::RingBuffer* reader, uint32_t packet);
  bool ExecutePacketType3(memory::RingBuffer* reader, uint32_t packet);
  bool ExecutePacketType3_ME_INIT(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_NOP(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_INTERRUPT(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_XE_SWAP(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_INDIRECT_BUFFER(memory::RingBuffer* reader, uint32_t packet,
                                          uint32_t count);
  bool ExecutePacketType3_WAIT_REG_MEM(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_REG_RMW(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_REG_TO_MEM(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_MEM_WRITE(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_COND_WRITE(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_EVENT_WRITE(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_EVENT_WRITE_SHD(memory::RingBuffer* reader, uint32_t packet,
                                          uint32_t count);
  bool ExecutePacketType3_EVENT_WRITE_EXT(memory::RingBuffer* reader, uint32_t packet,
                                          uint32_t count);
  virtual bool ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count);
  // [NR-DRAW] `draw_opcode` (0x22 or 0x36) is passed rather than inferred from
  // opcode_name: increment 4c advances a walk in lockstep with execution and
  // has to agree with the executor about WHICH draw packet is running.
  bool ExecutePacketType3Draw(memory::RingBuffer* reader, uint32_t packet, const char* opcode_name,
                              uint32_t draw_opcode, uint32_t viz_query_condition,
                              uint32_t count_remaining);
  // [NR-SKP] Phase 5-4-4a: the post-parse half of ExecutePacketType3Draw,
  // extracted verbatim (viz-kill check -> lockstep arm -> IssueDraw -> disarm
  // -> coverage -> failure log) so the skip's direct draw path and the packet
  // handler run the SAME body. `index_buffer_info` is null for a non-indexed
  // draw.
  void ExecutePacketType3DrawTail(uint32_t vgt_draw_initiator_value,
                                  IndexBufferInfo* index_buffer_info,
                                  const char* opcode_name, uint32_t draw_opcode);
  bool ExecutePacketType3_DRAW_INDX(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_DRAW_INDX_2(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_SET_CONSTANT(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_SET_CONSTANT2(memory::RingBuffer* reader, uint32_t packet,
                                        uint32_t count);
  bool ExecutePacketType3_LOAD_ALU_CONSTANT(memory::RingBuffer* reader, uint32_t packet,
                                            uint32_t count);
  bool ExecutePacketType3_SET_SHADER_CONSTANTS(memory::RingBuffer* reader, uint32_t packet,
                                               uint32_t count);
  bool ExecutePacketType3_IM_LOAD(memory::RingBuffer* reader, uint32_t packet, uint32_t count);
  bool ExecutePacketType3_IM_LOAD_IMMEDIATE(memory::RingBuffer* reader,

                                            uint32_t packet, uint32_t count);
  bool ExecutePacketType3_INVALIDATE_STATE(memory::RingBuffer* reader, uint32_t packet,
                                           uint32_t count);
  bool ExecutePacketType3_VIZ_QUERY(memory::RingBuffer* reader, uint32_t packet, uint32_t count);

  virtual Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                             const uint32_t* host_address, uint32_t dword_count) = 0;

  virtual bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                         IndexBufferInfo* index_buffer_info, bool major_mode_explicit) = 0;
  virtual bool IssueCopy() = 0;

  // "Actual" is for the command processor thread, to be read by the
  // implementations.
  SwapPostEffect GetActualSwapPostEffect() const { return swap_post_effect_actual_; }

  virtual void InitializeTrace();

  // Shared readback resolve mode with backend legacy-flag alias support.
  ReadbackResolveMode GetReadbackResolveMode(bool legacy_readback_resolve_enabled) const;
  // Shared memexport readback enable state with backend legacy-flag override support.
  bool IsReadbackMemexportEnabled(bool legacy_backend_flag) const;

  memory::Memory* memory_ = nullptr;
  system::KernelState* kernel_state_ = nullptr;
  GraphicsSystem* graphics_system_ = nullptr;
  RegisterFile* register_file_ = nullptr;

  TraceWriter trace_writer_;
  enum class TraceState {
    kDisabled,
    kStreaming,
    kSingleFrame,
  };
  TraceState trace_state_ = TraceState::kDisabled;
  std::filesystem::path trace_stream_path_;
  std::filesystem::path trace_frame_path_;

  std::atomic<bool> worker_running_;
  system::object_ref<system::XHostThread> worker_thread_;

  std::queue<std::function<void()>> pending_fns_;

  // MicroEngine binary from PM4_ME_INIT
  std::vector<uint32_t> me_bin_;

  uint32_t counter_ = 0;
  std::atomic<uint32_t> swap_counter_{0};

  uint32_t primary_buffer_ptr_ = 0;
  uint32_t primary_buffer_size_ = 0;

  uint32_t read_ptr_index_ = 0;
  uint32_t read_ptr_update_freq_ = 0;
  uint32_t read_ptr_writeback_ptr_ = 0;

  std::unique_ptr<rex::thread::Event> write_ptr_index_event_;
  std::atomic<uint32_t> write_ptr_index_;

  // Some titles submit writes beyond the emulated register file range in PM4
  // packets. Preserve these values so dependent packet logic can still observe
  // them instead of dropping the write entirely.
  std::unordered_map<uint32_t, uint32_t> extended_register_values_;

  uint64_t bin_select_ = 0xFFFFFFFFull;
  uint64_t bin_mask_ = 0xFFFFFFFFull;

  Shader* active_vertex_shader_ = nullptr;
  Shader* active_pixel_shader_ = nullptr;
  // [GPU-PRECORD] Phase 1b-1c Inc 2: per-replayed-draw active shaders + the gate the
  // active_vertex_shader()/active_pixel_shader() accessors consult. Set around a
  // segment replay (alongside D3D12's precord_replaying_); the shared active_*_shader_
  // members above are left holding their capture-end value, untouched by replay.
  bool precord_replay_shaders_active_ = false;
  Shader* precord_replay_active_vertex_shader_ = nullptr;
  Shader* precord_replay_active_pixel_shader_ = nullptr;

  // [NR-ISSUE] Increment 4d/4e: the arm/disarm handshake between the base
  // executor and the backend's IssueDraw. At a lockstep draw stop the base
  // points nr_issue_file_ at ITS OWN persistent RegisterFile -- seeded once by
  // composing the 4c shadow with the live file (RegShadowCompose), then
  // maintained INCREMENTALLY by the walk's decoded writes (increment 4e; the
  // 4d per-draw recompose cost the city 4x its fps and a real replay applies
  // writes as they decode anyway) -- resolves the walk's own shader refs
  // through LoadShader, and arms. The backend repoints every draw-path holder
  // at that file (the proven precord SetRegisterFile machinery), issues, and
  // restores; no copy at either end. The base disarms unconditionally right
  // after IssueDraw returns. All on the CP thread; unsupported alongside
  // precord capture (backend falls through to the normal path and counts it).
  //
  // The shader fields have their own gate rather than riding the precord
  // replay fields: those belong to the precord worker's lifecycle, and sharing
  // them would make two default-off features corrupt each other when combined.
  bool nr_issue_armed_ = false;
  const RegisterFile* nr_issue_file_ = nullptr;
  bool nr_issue_shaders_active_ = false;
  Shader* nr_issue_vertex_shader_ = nullptr;
  Shader* nr_issue_pixel_shader_ = nullptr;
  // Written by the backend, reported (and cleared per window) by the base.
  uint64_t nr_issue_issued_ = 0;
  uint64_t nr_issue_precord_skips_ = 0;

  bool paused_ = false;

  // By default (such as for tools), post-processing is disabled.
  // "Desired" is for the external thread managing the post-processing effect.
  SwapPostEffect swap_post_effect_desired_ = SwapPostEffect::kNone;
  SwapPostEffect swap_post_effect_actual_ = SwapPostEffect::kNone;

  // Set by backend command processors to their legacy memexport readback cvar
  // name (for explicit-override compatibility).
  const char* legacy_readback_memexport_cvar_name_ = nullptr;

 private:
  reg::DC_LUT_30_COLOR gamma_ramp_256_entry_table_[256] = {};
  reg::DC_LUT_PWL_DATA gamma_ramp_pwl_rgb_[128][3] = {};
  uint32_t gamma_ramp_rw_component_ = 0;
};

// [NR-RUB] Phase 5-4-5-1: the pending draw stop's reuse verdict
// (gpu_nr_reuse_probe v2), for the backend's bundle capture/compare gate.
// Returns false when the probe is off or no stop is pending. `key` is the
// DRAW_INDX packet's physical dword address (the draw's identity across
// replays); `reusable2` the stale-set verdict; `same_frame` whether the
// previous execution was in the current frame (bin repeat).
bool NrRuseCurrentDraw(uint32_t* key, bool* reusable2, bool* same_frame);

// [NR-RUF-V2B] Phase 5-4-5-2b: NrRuseCurrentDraw plus the stale-only flag --
// true when ONLY the stale-register set blocked reuse (prev comparable,
// packet span clean, shaders equal, no delegate poison). The backend may
// upgrade such a miss to reusable when every stale register is provably
// outside the draw's read set (float constants unread by either shader's
// bitmap -- bitmap-packed packs never carry them).
bool NrRuseCurrentDrawEx(uint32_t* key, bool* reusable2, bool* same_frame,
                         bool* stale_only);

// [NR-RUF-V2B] Copy the pending stop's stale-register set (up to `max`) and
// return its TRUE size; a return > max means truncation -- refuse the
// upgrade. Only meaningful while the stop's issue runs (same thread,
// synchronous, before the walk resumes).
uint32_t NrRuseStaleRegs(uint32_t* out, uint32_t max);

}  // namespace rex::graphics
