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

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/d3d12/deferred_command_list.h>
#include <rex/graphics/d3d12/graphics_system.h>
#include <rex/graphics/d3d12/pipeline_cache.h>
#include <rex/graphics/d3d12/primitive_processor.h>
#include <rex/graphics/d3d12/render_target_cache.h>
#include <rex/graphics/d3d12/shared_memory.h>
#include <rex/graphics/d3d12/texture_cache.h>
#include <rex/graphics/pipeline/shader/dxbc.h>
#include <rex/graphics/pipeline/shader/dxbc_translator.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/system/kernel_state.h>
#include <rex/ui/d3d12/d3d12_descriptor_heap_pool.h>
#include <rex/ui/d3d12/d3d12_provider.h>
#include <rex/ui/d3d12/d3d12_upload_buffer_pool.h>
#include <rex/ui/d3d12/d3d12_util.h>

namespace rex::graphics::d3d12 {

class D3D12CommandProcessor : public CommandProcessor {
 public:
  explicit D3D12CommandProcessor(D3D12GraphicsSystem* graphics_system,
                                 system::KernelState* kernel_state);
  ~D3D12CommandProcessor();

  void ClearCaches() override;
  void InvalidateGpuMemory() override;

  void InitializeShaderStorage(const std::filesystem::path& cache_root, uint32_t title_id,
                               bool blocking) override;

  void RequestFrameTrace(const std::filesystem::path& root_path) override;

  void TracePlaybackWroteMemory(uint32_t base_ptr, uint32_t length) override;

  void RestoreEdramSnapshot(const void* snapshot) override;

  ui::d3d12::D3D12Provider& GetD3D12Provider() const {
    return *static_cast<ui::d3d12::D3D12Provider*>(graphics_system_->provider());
  }

  // Returns the deferred drawing command list for the currently open
  // submission.
  DeferredCommandList& GetDeferredCommandList() {
    assert_true(submission_open_);
    return deferred_command_list_;
  }

  // [GPU-PRECORD] Phase 1b-1: the register file the DRAW PATH should read. Normally
  // the shared parse-thread register file; a worker replaying a captured segment
  // points this at its own per-segment local register file so the draw path (and,
  // once they are decoupled, the subsystems) read that segment's state instead of
  // the parse thread's run-ahead state. nullptr (default) ⇒ the shared file, so
  // this is a true no-op until a worker sets it.
  const rex::graphics::RegisterFile& GetActiveDrawRegisterFile() const {
    return active_draw_register_file_ ? *active_draw_register_file_ : *register_file_;
  }
  // [NR-PSO] Phase 5-1: whether the draw currently being set up reads a
  // repointed file rather than the shared one -- which, with gpu_nr_issue on,
  // means it is being issued from the walk-recovered replay file. Lets a probe
  // in a subsystem report how much of its coverage is recovered state instead
  // of assuming it.
  bool is_draw_register_file_repointed() const { return active_draw_register_file_ != nullptr; }

  uint64_t GetCurrentSubmission() const { return submission_current_; }
  uint64_t GetCompletedSubmission() const { return submission_completed_; }

  // Must be called when a subsystem does something like UpdateTileMappings so
  // it can be awaited in CheckSubmissionFence(submission_current_) if it was
  // done after the latest ExecuteCommandLists + Signal.
  void NotifyQueueOperationsDoneDirectly() {
    queue_operations_done_since_submission_signal_ = true;
  }

  uint64_t GetCurrentFrame() const { return frame_current_; }
  uint64_t GetCompletedFrame() const { return frame_completed_; }

  // Returns true if the barrier has been inserted (the new state is different).
  bool PushTransitionBarrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES old_state,
                             D3D12_RESOURCE_STATES new_state,
                             UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
  void PushAliasingBarrier(ID3D12Resource* old_resource, ID3D12Resource* new_resource);
  void PushUAVBarrier(ID3D12Resource* resource);
  void SubmitBarriers();

  // Finds or creates root signature for a pipeline.
  ID3D12RootSignature* GetRootSignature(const DxbcShader* vertex_shader,
                                        const DxbcShader* pixel_shader, bool tessellated);

  ui::d3d12::D3D12UploadBufferPool& GetConstantBufferPool() const { return *constant_buffer_pool_; }

  D3D12_CPU_DESCRIPTOR_HANDLE GetViewBindlessHeapCPUStart() const {
    assert_true(bindless_resources_used_);
    return view_bindless_heap_cpu_start_;
  }
  D3D12_GPU_DESCRIPTOR_HANDLE GetViewBindlessHeapGPUStart() const {
    assert_true(bindless_resources_used_);
    return view_bindless_heap_gpu_start_;
  }
  // Returns UINT32_MAX if no free descriptors. If the unbounded SRV range for
  // bindless resources is also used in the root signature of the draw /
  // dispatch referencing this descriptor, this must only be used to allocate
  // SRVs, otherwise it won't work on Nvidia Fermi (root signature creation will
  // fail)!
  uint32_t RequestPersistentViewBindlessDescriptor();
  void ReleaseViewBindlessDescriptorImmediately(uint32_t descriptor_index);
  // Request non-contiguous CBV/SRV/UAV descriptors for use only within the next
  // draw or dispatch command done for internal purposes. May change the current
  // descriptor heap. If the unbounded SRV range for bindless resources is also
  // used in the root signature of the draw / dispatch referencing these
  // descriptors, this must only be used to allocate SRVs, otherwise it won't
  // work on Nvidia Fermi (root signature creation will fail)!
  bool RequestOneUseSingleViewDescriptors(uint32_t count,
                                          ui::d3d12::util::DescriptorCpuGpuHandlePair* handles_out);
  // [NR-RSY] Phase 5-3b-3: residency/descriptor-allocation gate observers,
  // called by the texture cache (implemented beside the gate state in
  // command_processor.cpp; all no-ops while the gate is off).
  bool NrResArmed() const;
  void NrResObserveTexDescriptor(const void* texture, uint32_t srv_key, bool hit, uint32_t index);
  void NrResObserveTexDescriptorRelease(uint32_t index);
  // These are needed often, so they are always allocated.
  enum class SystemBindlessView : uint32_t {
    // Both may be bound as one root parameter.
    kSharedMemoryRawSRVAndNullRawUAVStart,
    kSharedMemoryRawSRV = kSharedMemoryRawSRVAndNullRawUAVStart,
    kNullRawUAV,

    // Both may be bound as one root parameter.
    kNullRawSRVAndSharedMemoryRawUAVStart,
    kNullRawSRV = kNullRawSRVAndSharedMemoryRawUAVStart,
    kSharedMemoryRawUAV,

    kSharedMemoryR32UintSRV,
    kSharedMemoryR32G32UintSRV,
    kSharedMemoryR32G32B32A32UintSRV,
    kSharedMemoryR32UintUAV,
    kSharedMemoryR32G32UintUAV,
    kSharedMemoryR32G32B32A32UintUAV,

    kEdramRawSRV,
    kEdramR32UintSRV,
    kEdramR32G32UintSRV,
    kEdramR32G32B32A32UintSRV,
    kEdramRawUAV,
    kEdramR32UintUAV,
    kEdramR32G32UintUAV,
    kEdramR32G32B32A32UintUAV,

    kGammaRampTableSRV,
    kGammaRampPWLSRV,

    // Beyond this point, SRVs are accessible to shaders through an unbounded
    // range - no descriptors of other types bound to shaders alongside
    // unbounded ranges - must be located beyond this point.
    kUnboundedSRVsStart,
    kNullTexture2DArray = kUnboundedSRVsStart,
    kNullTexture3D,
    kNullTextureCube,

    kCount,
  };
  ui::d3d12::util::DescriptorCpuGpuHandlePair GetSystemBindlessViewHandlePair(
      SystemBindlessView view) const;
  ui::d3d12::util::DescriptorCpuGpuHandlePair GetSharedMemoryUintPow2BindlessSRVHandlePair(
      uint32_t element_size_bytes_pow2) const;
  ui::d3d12::util::DescriptorCpuGpuHandlePair GetSharedMemoryUintPow2BindlessUAVHandlePair(
      uint32_t element_size_bytes_pow2) const;
  ui::d3d12::util::DescriptorCpuGpuHandlePair GetEdramUintPow2BindlessSRVHandlePair(
      uint32_t element_size_bytes_pow2) const;
  ui::d3d12::util::DescriptorCpuGpuHandlePair GetEdramUintPow2BindlessUAVHandlePair(
      uint32_t element_size_bytes_pow2) const;

  // Returns a single temporary GPU-side buffer within a submission for tasks
  // like texture untiling and resolving.
  ID3D12Resource* RequestScratchGPUBuffer(uint32_t size, D3D12_RESOURCE_STATES state);
  // This must be called when done with the scratch buffer, to notify the
  // command processor about the new state in case the buffer was transitioned
  // by its user.
  void ReleaseScratchGPUBuffer(ID3D12Resource* buffer, D3D12_RESOURCE_STATES new_state);

  // Returns a pipeline with deferred creation by its handle. May return nullptr
  // if failed to create the pipeline.
  ID3D12PipelineState* GetD3D12PipelineByHandle(void* handle) const {
    return pipeline_cache_->GetD3D12PipelineByHandle(handle);
  }

  // Sets the current cached values to external ones. This is for cache
  // invalidation primarily. A submission must be open.
  void SetExternalPipeline(ID3D12PipelineState* pipeline);
  void SetExternalGraphicsRootSignature(ID3D12RootSignature* root_signature);
  void SetViewport(const D3D12_VIEWPORT& viewport);
  void SetScissorRect(const D3D12_RECT& scissor_rect);
  void SetStencilReference(uint32_t stencil_ref);
  void SetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY primitive_topology);

  // Returns the text to display in the GPU backend name in the window title.
  std::string GetWindowTitleText() const;

 protected:
  bool SetupContext() override;
  void ShutdownContext() override;

  void WriteRegister(uint32_t index, uint32_t value) override;
  void WriteRegistersFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers) override;
  // [NR-PB] N-2-2 item 0: plain bulk store + the gpu_instance dirty semantic
  // (the only D3D12 tail a plain register has).
  void WriteRegisterRangePlain(uint32_t base, uint32_t* values_be, uint32_t n) override;
  // [NR-FX] Phase 5-4-0: WriteRegister's dirty-tracking tail for the three
  // constant ranges, fired from the lockstep walk's decoded write stream. No
  // value store, no dedupe. Keep in sync with WriteRegister / PrecordApplyWrite.
  void NrWalkWriteEffects(uint32_t index) override;
  // [NR-SKP] Phase 5-4-2: this backend supports walk-only buffer execution
  // unless precord capture owns the draw path (same exclusion as the
  // gpu_nr_issue seam, counted there as precord_skip).
  bool NrSkipBackendEligible() const override;
  // [NR-BFC] Phase 5-4-6-0: buffer-replay census bracket -- Begin latches the
  // deferred-list stream position and the RT-update body-run counter, End
  // scans the emitted span (self-describing stream) and fills the sample.
  void NrBfcBufBegin() override;
  bool NrBfcBufEnd(NrBfcBackendSample* out) override;
  // [NR-DSP] Phase 5-4-7-0: per-draw native span capture + compare.
  void NrDspDrawBegin(uint32_t key, bool reusable) override;
  void NrDspDrawEnd() override;
  // [NR-SPR] Phase 5-4-7-1: context-free span record + replay-prediction
  // gate. Begin forces the tail-state re-emit (members only -- the emissions
  // happen inside the draw); End scans, stores, compares.
  void NrSprDrawBegin(uint32_t key, bool reusable) override;
  void NrSprDrawEnd() override;
  bool ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader, uint32_t packet,
                                          uint32_t count) override;

  void OnGammaRamp256EntryTableValueWritten() override;
  void OnGammaRampPWLValueWritten() override;

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                 uint32_t frontbuffer_height) override;

  void OnPrimaryBufferEnd() override;

  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                     const uint32_t* host_address, uint32_t dword_count) override;

  bool IssueDraw(xenos::PrimitiveType primitive_type, uint32_t index_count,
                 IndexBufferInfo* index_buffer_info, bool major_mode_explicit) override;
  bool IssueCopy() override;

  void InitializeTrace() override;

 private:
  static constexpr uint32_t kQueueFrames = 3;

  enum RootParameter : UINT {
    // Keep the size of the root signature at each stage 13 dwords or less
    // (better 12 or less) so it fits in user data on AMD. Descriptor tables are
    // 1 dword, root descriptors are 2 dwords (however, root descriptors require
    // less setup on the CPU - balance needs to be maintained).

    // CBVs are set in both bindful and bindless cases via root descriptors.

    // - Bindful resources - multiple root signatures depending on extra
    //   parameters.

    // These are always present.

    // Very frequently changed, especially for UI draws, and for models drawn in
    // multiple parts - contains vertex and texture fetch constants.
    kRootParameter_Bindful_FetchConstants = 0,  // +2 dwords = 2 in all.
    // Quite frequently changed (for one object drawn multiple times, for
    // instance - may contain projection matrices).
    kRootParameter_Bindful_FloatConstantsVertex,  // +2 = 4 in VS.
    // Less frequently changed (per-material).
    kRootParameter_Bindful_FloatConstantsPixel,  // +2 = 4 in PS.
    // May stay the same across many draws.
    kRootParameter_Bindful_SystemConstants,  // +2 = 6 in all.
    // Pretty rarely used and rarely changed - flow control constants.
    kRootParameter_Bindful_BoolLoopConstants,  // +2 = 8 in all.
    // Changed only when starting a new descriptor heap or when switching
    // between shared memory as SRV and UAV - shared memory byte address buffer
    // (as SRV and as UAV, either may be null if not used), and, if ROV is used
    // for EDRAM, EDRAM R32_UINT UAV.
    kRootParameter_Bindful_SharedMemoryAndEdram,  // +1 = 9 in all.

    kRootParameter_Bindful_Count_Base,

    // Extra parameter that may or may not exist:
    // - Pixel textures (+1 = 10 in PS).
    // - Pixel samplers (+1 = 11 in PS).
    // - Vertex textures (+1 = 10 in VS).
    // - Vertex samplers (+1 = 11 in VS).

    kRootParameter_Bindful_Count_Max = kRootParameter_Bindful_Count_Base + 4,

    // - Bindless resources - two global root signatures (for non-tessellated
    //   and tessellated drawing), so these are always present.

    kRootParameter_Bindless_FetchConstants = 0,    // +2 = 2 in all.
    kRootParameter_Bindless_FloatConstantsVertex,  // +2 = 4 in VS.
    kRootParameter_Bindless_FloatConstantsPixel,   // +2 = 4 in PS.
    // Changed per-material, texture and sampler descriptor indices.
    kRootParameter_Bindless_DescriptorIndicesPixel,   // +2 = 6 in PS.
    kRootParameter_Bindless_DescriptorIndicesVertex,  // +2 = 6 in VS.
    kRootParameter_Bindless_SystemConstants,          // +2 = 8 in all.
    kRootParameter_Bindless_BoolLoopConstants,        // +2 = 10 in all.
    // Changed only when switching between shared memory as SRV and UAV - shared
    // memory byte address buffer (as SRV and as UAV, either may be null if not
    // used).
    kRootParameter_Bindless_SharedMemory,  // +1 = 11 in all.
    // Unbounded sampler descriptor table - changed in case of overflow.
    kRootParameter_Bindless_SamplerHeap,  // +1 = 12 in all.
    // Unbounded SRV/UAV descriptor table - never changed.
    kRootParameter_Bindless_ViewHeap,  // +1 = 13 in all.

    kRootParameter_Bindless_Count,
  };

  struct RootBindfulExtraParameterIndices {
    uint32_t textures_pixel;
    uint32_t samplers_pixel;
    uint32_t textures_vertex;
    uint32_t samplers_vertex;
    static constexpr uint32_t kUnavailable = UINT32_MAX;
  };
  // Gets the indices of optional root parameters. Returns the total parameter
  // count.
  static uint32_t GetRootBindfulExtraParameterIndices(
      const DxbcShader* vertex_shader, const DxbcShader* pixel_shader,
      RootBindfulExtraParameterIndices& indices_out);

  // BeginSubmission and EndSubmission may be called at any time. If there's an
  // open non-frame submission, BeginSubmission(true) will promote it to a
  // frame. EndSubmission(true) will close the frame no matter whether the
  // submission has already been closed.
  // Submission (ExecuteCommandLists) boundaries are implicit full UAV and
  // aliasing barriers, and also result in common resource state promotion and
  // decay.

  // Rechecks submission number and reclaims per-submission resources. Pass 0 as
  // the submission to await to simply check status, or pass submission_current_
  // to wait for all queue operations to be completed.
  void CheckSubmissionFence(uint64_t await_submission);
  // If is_guest_command is true, a new full frame - with full cleanup of
  // resources and, if needed, starting capturing - is opened if pending (as
  // opposed to simply resuming after mid-frame synchronization). Returns
  // whether a submission is open currently and the device is not removed.
  bool BeginSubmission(bool is_guest_command);
  // If is_swap is true, a full frame is closed - with, if needed, cache
  // clearing and stopping capturing. Returns whether the submission was done
  // successfully, if it has failed, leaves it open.
  bool EndSubmission(bool is_swap);
  // [GPU-PRECORD] Phase 1a: force the next draw to re-emit ALL command-list state
  // (PSO, root sig, cbuffer bindings, descriptor heaps, topology, fixed-function)
  // by resetting the same cached/dirty state BeginSubmission resets for a fresh
  // list. Models a parallel-segment boundary where a new command list has nothing
  // bound. Body is copied verbatim from BeginSubmission's reset block.
  void ForceFullDrawStateReemit();

  // [GPU-PRECORD] Phase 1b-0: the real IssueDraw body. IssueDraw() is a thin
  // capture wrapper that, when gpu_precord_capture is on, defers a draw into the
  // current segment's event log instead of recording it; PrecordFlush() replays
  // the log (rewinding the register file to the segment snapshot) by calling this.
  bool IssueDrawImpl(xenos::PrimitiveType primitive_type, uint32_t index_count,
                     IndexBufferInfo* index_buffer_info, bool major_mode_explicit);
  // [NR-ISSUE] Increment 4d/4e: issue one draw from the replay register file
  // the base executor armed (nr_issue_file_ -- persistent, walk-maintained;
  // no copy). Builds the NrDrawInput record and hands it to NrSubmitDraw.
  bool NrIssueDrawFromShadow(xenos::PrimitiveType primitive_type, uint32_t index_count,
                             IndexBufferInfo* index_buffer_info, bool major_mode_explicit);

  // [NR-NATIVE] Phase 5-0: THE NATIVE DRAW SEAM. One record carrying
  // everything a draw is made of, all of it proven recoverable by increments
  // 4a-4f: the replay register file, the walk-resolved shaders, and the
  // executor's own draw arguments. The phase-5 ladder swaps the internals of
  // NrSubmitDraw -- state mirror (5-1), shader cache (5-2), native
  // submission subsystem by subsystem (5-3) -- without the executor or the
  // arming handshake changing shape again.
  struct NrDrawInput {
    const RegisterFile* regs;  // the replay file (never null when armed)
    Shader* vertex_shader;     // walk-resolved via LoadShader
    Shader* pixel_shader;
    xenos::PrimitiveType primitive_type;
    uint32_t index_count;
    IndexBufferInfo* index_buffer_info;  // null = auto-index draw
    bool major_mode_explicit;
  };
  // Phase 5-0 body: delegate to the emulated pipeline -- repoint every
  // draw-path holder at input.regs (the proven 1b-1a/1b-1b SetRegisterFile
  // machinery), run IssueDrawImpl, restore. The active shaders reach
  // IssueDrawImpl via the base's nr_issue_* accessor gate (set for the whole
  // armed call); the record carries them explicitly for the native path. No
  // ForceFullDrawStateReemit: this is the SAME draw at the SAME moment, so
  // the deferred list's dirty tracking stays valid whether or not the values
  // match (dirty state compares against internal shadows, not the register
  // file).
  bool NrSubmitDraw(const NrDrawInput& input);
  // Replay the pending captured segment into the current deferred command list,
  // in original write/draw order, then clear it. No-op if no segment is open or
  // already replaying (reentrancy guard). Dispatches to the shared-rewind (1b-0),
  // local-register-file (1b-1b), or worker-thread (1b-1b Model C) path per cvar.
  // [GPU-PRECORD] Phase 1b-1c Inc 5: from_segment_boundary distinguishes the 256-draw
  // boundary (the ONLY caller that passes true) from a true flush point (swap/copy/
  // end-submission/markers/stateful writes -- the default false). Under overlap the
  // boundary POSTs the segment to the worker and returns without waiting (parse keeps
  // capturing the other slot); a true flush point DRAINS the worker before returning
  // (ordering + visibility). In every non-overlap mode both fully drain -> Model C.
  void PrecordFlush(bool from_segment_boundary = false);
  // [GPU-PRECORD] Phase 1b-1c Inc 5: block until the replay worker has finished any
  // posted segment (job no longer pending). No-op if the worker was never started or is
  // idle. Used as overlap backpressure (before reusing a slot) and to drain at flush
  // points. Safe to call on the parse thread only.
  void PrecordWaitWorkerIdle();
  // [GPU-PRECORD] Phase 1b-1b/1c: the event/draw replay loop, shared by every replay
  // mode. Replays the replay slot's events in order interleaved with IssueDrawImpl. Assumes the
  // register file is already rewound/built and the draw-state cache reset.
  //   local_target == nullptr ⇒ 1b-0 shared-rewind mode: apply logged writes via the
  //     normal WriteRegister/WriteRegistersFromMem against the (rewound) shared file.
  //   local_target != nullptr ⇒ 1b-1b/1c local-regfile mode: apply logged writes via
  //     PrecordApplyWrite* against the private local file, never touching register_file_.
  void PrecordReplayEvents(RegisterFile* local_target);
  // [GPU-PRECORD] Phase 1b-1c Inc 1: replay-time equivalents of WriteRegister /
  // WriteRegistersFromMem that apply a DEFERRABLE register write against `file` (the
  // per-segment local register file) instead of the shared register_file_ member, so a
  // replaying worker never mutates the file the parse thread owns -- the write-side
  // analogue of the 1b-1a read decoupling, and the invariant Phase 1c overlap needs.
  // They mirror the deferrable-register subset of WriteRegister's effect (dedupe skip,
  // value store, D3D12 cbuffer/texture/vertex-residency invalidation); the stateful
  // registers are flush-gated (PrecordRangeMustNotDefer) so they never reach replay.
  // Keep in sync with WriteRegister / WriteRegistersFromMem.
  void PrecordApplyWrite(RegisterFile* file, uint32_t index, uint32_t value);
  void PrecordApplyWriteFromMem(RegisterFile* file, uint32_t start_index, uint32_t* base,
                                uint32_t num_registers);
  // [GPU-PRECORD] Phase 1b-1b: replay the captured segment against a PRIVATE local
  // register file (the replay slot's local_regfile) with every draw-path holder repointed to
  // it (SetRegisterFile), then restored. Exercises the 1b-1a decoupling: the draws
  // read the segment's own state from a separate file, not the shared parse-thread
  // file. Runs inline (gpu_precord_localrf) or on the worker (gpu_precord_thread).
  void PrecordReplayLocal();
  // [GPU-PRECORD] Phase 1b-1b worker (Model C): lazily start the replay worker;
  // post the open segment and BLOCK until it has replayed (no parse/worker overlap
  // yet — correctness/plumbing first); join at shutdown.
  void PrecordWorkerEnsureStarted();
  void PrecordWorkerMain();
  void PrecordWorkerShutdown();
  // [GPU-PRECORD] Phase 1b-1c Inc 3: drop a slot's captured buffers (events/draws/
  // frommem_data) without replaying, so it can capture or replay a fresh segment. Keeps
  // the snapshot/local_regfile allocations for reuse. Used after a flush replays a slot,
  // and defensively at submission boundaries.
  void PrecordResetSlot(uint32_t slot);
  // [GPU-PRECORD] Phase 1b-1c Inc 6 (H3 fix, Option A): open a capture segment on the
  // current capture slot NOW (snapshot the shared register file, mark the segment open).
  // Called lazily at the first draw, and EAGERLY at an overlap segment boundary so the
  // next segment's setup writes DEFER (skipping the parse-side D3D12 draw-state side
  // effects) instead of racing the worker's in-flight replay on the shared draw-state.
  void PrecordOpenSegment();
  // [GPU-PRECORD H3-PROBE] Inc 6: hash the guest index buffer for a draw (parse-time
  // capture and replay-time re-check use the SAME inputs, so a hash mismatch isolates a
  // guest overwrite of the source memory between capture and replay). Returns the hash;
  // *out_len receives the byte count actually hashed (0 if there is no valid IB range).
  uint32_t PrecordH3HashIndexBuffer(const IndexBufferInfo& ibi, uint32_t index_count,
                                    uint32_t* out_len);
  // Max bound vertex-fetch ranges pinned per draw for the VB probe (a draw rarely binds
  // more than a handful; extras are truncated, only under-counting).
  static constexpr uint32_t kH3MaxVbRanges = 8;
  // [GPU-PRECORD H3-PROBE] Inc 6: CAPTURE a draw's bound guest VERTEX buffer ranges +
  // a hash of their current bytes. Iterates the analyzed shader's vertex_fetch_bitmap
  // exactly like the residency loop in IssueDrawImpl, reading the register file (correct
  // at parse/capture time). Writes up to `max_ranges` (addr,size) pairs into the
  // parallel out arrays and *out_count; returns the folded hash and *out_len = bytes
  // hashed (0 if the shader is unanalyzed / binds no vertex buffers). The ranges are
  // PINNED so replay re-hashes the SAME bytes (re-deriving them from the register file
  // at replay gave a ~50% false-positive floor -- range instability, not a real
  // overwrite; the IB probe avoids this by deep-copying its range, hence its clean 0%).
  uint32_t PrecordH3CaptureVertexBuffers(Shader* vertex_shader, uint32_t max_ranges,
                                         uint32_t* out_addr, uint32_t* out_size,
                                         uint32_t* out_count, uint32_t* out_len);
  // [GPU-PRECORD H3-PROBE] Inc 6: REPLAY-side re-hash of the pinned ranges (no register
  // state involved), compared to the captured hash to detect a guest overwrite in the
  // parse->replay lead window.
  uint32_t PrecordH3HashCapturedVbRanges(const uint32_t* addr, const uint32_t* size,
                                         uint32_t count, uint32_t* out_len);
  // Checks if ending a submission right now would not cause potentially more
  // delay than it would reduce by making the GPU start working earlier - such
  // as when there are unfinished graphics pipeline creation requests that would
  // need to be fulfilled before actually submitting the command list.
  bool CanEndSubmissionImmediately() const;
  bool AwaitAllQueueOperationsCompletion() {
    CheckSubmissionFence(submission_current_);
    return submission_completed_ + 1 >= submission_current_;
  }
  void LogDeviceRemovalDiagnostics(ID3D12Device* device, HRESULT reason);

  void UpdateDebugMarkersEnabled();
  void PushDebugMarker(const char* format, ...);
  void PopDebugMarker();
  void InsertDebugMarker(const char* format, ...);
  bool debug_markers_enabled() const { return debug_markers_enabled_; }

  // Need to await submission completion before calling.
  void ClearCommandAllocatorCache();

  // Request descriptors and automatically rebind the descriptor heap on the
  // draw command list. Refer to D3D12DescriptorHeapPool::Request for partial /
  // full update explanation. Doesn't work when bindless descriptors are used.
  uint64_t RequestViewBindfulDescriptors(uint64_t previous_heap_index,
                                         uint32_t count_for_partial_update,
                                         uint32_t count_for_full_update,
                                         D3D12_CPU_DESCRIPTOR_HANDLE& cpu_handle_out,
                                         D3D12_GPU_DESCRIPTOR_HANDLE& gpu_handle_out);
  uint64_t RequestSamplerBindfulDescriptors(uint64_t previous_heap_index,
                                            uint32_t count_for_partial_update,
                                            uint32_t count_for_full_update,
                                            D3D12_CPU_DESCRIPTOR_HANDLE& cpu_handle_out,
                                            D3D12_GPU_DESCRIPTOR_HANDLE& gpu_handle_out);

  void UpdateFixedFunctionState(const draw_util::ViewportInfo& viewport_info,
                                const draw_util::Scissor& scissor, bool primitive_polygonal,
                                reg::RB_DEPTHCONTROL normalized_depth_control);
  void UpdateSystemConstantValues(bool shared_memory_is_uav, bool primitive_polygonal,
                                  uint32_t line_loop_closing_index, xenos::Endian index_endian,
                                  const draw_util::ViewportInfo& viewport_info,
                                  uint32_t used_texture_mask,
                                  reg::RB_DEPTHCONTROL normalized_depth_control,
                                  uint32_t normalized_color_mask);
  bool UpdateBindings(const D3D12Shader* vertex_shader, const D3D12Shader* pixel_shader,
                      ID3D12RootSignature* root_signature, bool shared_memory_is_uav);

  // [NR-RUF] 5-4-5-2 restore body, extracted verbatim from NrUpdateBindings
  // so the 5-4-7-2 span replay runs the exact same member restore. The
  // parameter is a cpp-local NrRubBundle (kept out of this header).
  void NrRufRestoreFromBundle(const void* bundle);
  // [NR-SPW] Phase 5-4-7-2: attempt to serve the current bracketed draw from
  // its recorded native span (live head + memcpy + address patch). Returns
  // true when the draw was fully replayed (IssueDrawImpl must return true);
  // false = fall through to the full derivation path (every head step
  // already taken is idempotent there).
  bool NrSpanReplayTry();
  // [NR-TIL] N-4-1: EDRAM tile-pass draw replay. The repeat bands re-execute
  // the base band packets in the same order with only the tile window moved,
  // so a repeat-band draw native span equals the base band one except for
  // viewport, scissor and the system-constants NDC. BeginDraw walks the
  // segment / ordinal bookkeeping and decides record vs replay; ReplayTry
  // serves the draw from the recording (true = IssueDrawImpl returns true);
  // RecordAnchor latches the span start AFTER the fixed-function and system
  // constant updates; RecordEnd scans and stores it; SegBreak ends a segment
  // at a resolve.
  void NrTileBeginDraw(uint32_t win_off, uint32_t prim, uint32_t index_count,
                       uint32_t ib_base);
  bool NrTileReplayTry();
  void NrTileRecordAnchorA();
  void NrTileRecordSplit();
  void NrTileRecordAnchor();
  void NrTileRecordEnd();
  void NrTileCompareEnd();
  void NrTileSegBreak();
  // [NR-SPD] Phase 5-4-7-3: snapshot the emission-context members (pipeline,
  // root signature, topology, root-up-to-date mask, cbuffer addresses+flags,
  // shared-memory flavor) into a cpp-local SprCtx (kept out of this header).
  // Captured at bracket Begin (entry) and store time (exit); a deduped
  // recording replays only when the current context memcmp-equals its entry
  // snapshot, and applies the exit snapshot to the members afterwards.
  void NrSprCaptureCtx(void* out_ctx) const;
  // [NR-SWP] Phase 5-3b swap: this project's own UpdateBindings (bindless
  // only, same member state machine). *refused_out = fall back to the
  // emulated function for this draw (counted by the caller).
  bool NrUpdateBindings(const D3D12Shader* vertex_shader, const D3D12Shader* pixel_shader,
                        ID3D12RootSignature* root_signature, bool shared_memory_is_uav,
                        bool* refused_out);
  bool IssueCopy_ReadbackResolvePath();
  bool IssueDraw_MemexportReadbackFullPath(uint32_t total_size);
  bool IssueDraw_MemexportReadbackFastPath(uint32_t total_size);

  // Returns a buffer for reading GPU data back to the CPU. Assuming
  // synchronizing immediately after use. Always in COPY_DEST state.
  ID3D12Resource* RequestReadbackBuffer(uint32_t size);
  struct ReadbackBuffer {
    ID3D12Resource* buffers[2] = {nullptr, nullptr};
    uint32_t sizes[2] = {0, 0};
    void* mapped_data[2] = {nullptr, nullptr};
    uint64_t submission_written[2] = {0, 0};
    uint32_t written_size[2] = {0, 0};
    uint32_t current_index = 0;
    uint64_t last_used_frame = 0;
  };
  void EvictOldReadbackBuffers(std::unordered_map<uint64_t, ReadbackBuffer>& buffer_map);
  static constexpr uint32_t kReadbackBufferSizeIncrement = 16 * 1024 * 1024;
  static constexpr size_t kMaxReadbackBuffers = 256;
  static constexpr uint64_t kReadbackBufferEvictionAgeFrames = 60;
  static inline uint32_t AlignReadbackBufferSize(uint32_t size) {
    if (size < 1 * 1024 * 1024) {
      return rex::align(size, 256u * 1024u);
    }
    if (size < 4 * 1024 * 1024) {
      return rex::align(size, 1u * 1024u * 1024u);
    }
    return rex::align(size, kReadbackBufferSizeIncrement);
  }
  static inline uint64_t MakeReadbackResolveKey(uint32_t address, uint32_t length) {
    return (uint64_t(address) << 32) | uint64_t(length);
  }
  static inline uint64_t MakeMemexportReadbackKey(uint32_t first_base_address_dwords,
                                                  uint32_t total_size) {
    return (uint64_t(first_base_address_dwords) << 32) | uint64_t(total_size);
  }

  bool InitializeOcclusionQueryResources();
  void ShutdownOcclusionQueryResources();
  bool BeginGuestOcclusionQuery(uint32_t sample_count_address);
  bool EndGuestOcclusionQuery(uint32_t sample_count_address,
                              xenos::xe_gpu_depth_sample_counts* sample_counts);
  bool AcquireOcclusionQueryIndex(uint32_t& host_index_out);
  void DisableHostOcclusionQueries();
  uint64_t NormalizeOcclusionSamples(uint64_t samples) const;
  void WriteGuestOcclusionResult(xenos::xe_gpu_depth_sample_counts* sample_counts,
                                 uint64_t samples);
  void InvalidateAllVertexBufferResidency();
  void InvalidateVertexBufferResidency(uint32_t vfetch_index);
  void InvalidateVertexBufferResidencyRange(uint32_t first_vfetch, uint32_t last_vfetch);

  // [NR-RSY] Phase 5-3b-3 internals (implemented beside the gate state):
  // reseed the pool mirror from the emulated allocator, follow an allocation
  // or a release, and close a draw's vertex-residency compare.
  void NrResPoolReseed();
  void NrResPoolObserveAlloc(uint32_t actual_index);
  void NrResPoolObserveRelease(uint32_t index);
  void NrResVfetchSeedFromEmulated();
  void NrResVfetchFinishDraw(uint32_t abort_reason);

  void WriteGammaRampSRV(bool is_pwl, D3D12_CPU_DESCRIPTOR_HANDLE handle) const;

  bool device_removed_ = false;

  bool cache_clear_requested_ = false;

  HANDLE fence_completion_event_ = nullptr;

  bool submission_open_ = false;
  // Values of submission_fence_.
  uint64_t submission_current_ = 1;
  uint64_t submission_completed_ = 0;
  ID3D12Fence* submission_fence_ = nullptr;

  // For awaiting non-submission queue operations such as UpdateTileMappings in
  // AwaitAllQueueOperationsCompletion when they're queued after the latest
  // ExecuteCommandLists + Signal, thus won't be awaited by just awaiting the
  // submission.
  ID3D12Fence* queue_operations_since_submission_fence_ = nullptr;
  uint64_t queue_operations_since_submission_fence_last_ = 0;
  bool queue_operations_done_since_submission_signal_ = false;

  bool frame_open_ = false;
  // Guest frame index, since some transient resources can be reused across
  // submissions. Values updated in the beginning of a frame.
  uint64_t frame_current_ = 1;
  uint64_t frame_completed_ = 0;
  // Submission indices of frames that have already been submitted.
  uint64_t closed_frame_submissions_[kQueueFrames] = {};

  struct CommandAllocator {
    ID3D12CommandAllocator* command_allocator;
    uint64_t last_usage_submission;
    CommandAllocator* next;
  };
  CommandAllocator* command_allocator_writable_first_ = nullptr;
  CommandAllocator* command_allocator_writable_last_ = nullptr;
  CommandAllocator* command_allocator_submitted_first_ = nullptr;
  CommandAllocator* command_allocator_submitted_last_ = nullptr;
  ID3D12GraphicsCommandList* command_list_ = nullptr;
  ID3D12GraphicsCommandList1* command_list_1_ = nullptr;
  DeferredCommandList deferred_command_list_;

  // [NR-BFC] Phase 5-4-6-0 census bracket state (valid between
  // NrBfcBufBegin and NrBfcBufEnd; the deferred list's reset generation is
  // the anchor-validity check -- the submission id lags the Reset).
  size_t nr_bfc_span_start_ = 0;
  uint64_t nr_bfc_rt_runs_start_ = 0;
  uint64_t nr_bfc_gen_start_ = 0;

  // [GPU-PRECORD] Phase 1a per-submission draw counter; every N draws a forced
  // full-state re-emit models a segment boundary (correctness probe).
  uint64_t parallel_record_counter_ = 0;
  // [GPU-PRECORD] Phase 1a-ii: completed segment command streams for this
  // submission, replayed in order before the final (current) deferred list at
  // EndSubmission. Empty when gpu_parallel_record is off (zero overhead).
  std::vector<std::vector<uintmax_t>> precord_segments_;

  // [GPU-PRECORD] Phase 1b-0: deferred draw capture/replay (single-thread
  // foundation for off-thread segment recording). When gpu_precord_capture is on,
  // each draw is deferred into a per-segment event log instead of recorded inline;
  // at a flush boundary the log is replayed against the register file rewound to a
  // snapshot taken at the segment's first draw -> byte-identical deferred stream.
  struct PrecordDraw {
    xenos::PrimitiveType primitive_type;
    uint32_t index_count;
    bool has_index_buffer_info;
    IndexBufferInfo index_buffer_info;  // deep copy (the source is transient)
    bool major_mode_explicit;
    // [GPU-PRECORD] The active vertex/pixel shaders are CP-member state set by PM4
    // shader-load packets during parse (NOT in the register file), so the parse
    // thread runs AHEAD of the deferred draws and active_*_shader_ holds the LATEST
    // loaded shader by replay time. They must be captured per draw and restored
    // before replay, or every replayed draw uses the wrong shader -> wrong
    // vertex-fetch bitmap -> spurious "invalid fetch" aborts (the black-screen bug).
    Shader* active_vs;
    Shader* active_ps;
    // [GPU-PRECORD H3-PROBE] Inc 6 (default OFF): a hash of this draw's guest index
    // buffer taken at parse/capture time, plus the byte length hashed. Re-hashed at
    // replay and compared: a mismatch means the guest overwrote the IB in the
    // parse->replay lead window (the H3 hazard that flickers foliage under overlap).
    // len == 0 ⇒ no IB captured / out of range ⇒ replay skips the check.
    uint32_t h3_ib_len = 0;
    uint32_t h3_ib_hash = 0;
    // [GPU-PRECORD H3-PROBE] Inc 6 (default OFF): same idea for the bound VERTEX
    // buffers (the IB probe measured 0% overwrites in the heavy forest, so the
    // corruption is the VB or a texture). The exact bound VB byte-ranges are PINNED at
    // capture (addr/size below) and a bounded prefix of each is folded into one hash,
    // re-hashed against the SAME ranges at replay. len == 0 ⇒ no bound VB / unanalyzed
    // shader ⇒ replay skips the check.
    uint32_t h3_vb_len = 0;
    uint32_t h3_vb_hash = 0;
    uint32_t h3_vb_range_count = 0;
    uint32_t h3_vb_range_addr[kH3MaxVbRanges] = {};
    uint32_t h3_vb_range_size[kH3MaxVbRanges] = {};
  };
  struct PrecordEvent {
    enum class Kind : uint8_t { kWriteSingle, kWriteFromMem, kDraw } kind;
    // kWriteSingle: a=reg index, b=value.
    // kWriteFromMem: a=start index, b=num registers, c=offset into frommem data.
    // kDraw: a=index into the segment's draws.
    uint32_t a;
    uint32_t b;
    uint32_t c;
  };
  // [GPU-PRECORD] Phase 1b-1c Inc 3: the per-segment capture buffers, DOUBLE-BUFFERED
  // into two slots so the parse thread can capture the next segment (into the capture
  // slot) while the replayer drains the previous one (the replay slot). In Model C the
  // replayer always drains the replay slot before the parse thread reuses it, so the two
  // slots behave exactly like the old single buffer (pixel-identical); the 2-slot
  // structure is what Phase 1c overlap (Inc 5) needs. Per slot:
  //   snapshot      - full register-file snapshot taken at the segment's first draw.
  //   events/draws/frommem_data - the ordered write/draw log (frommem_data holds the raw
  //                   guest dwords for kWriteFromMem; events store offsets into it).
  //   local_regfile - private file the local-regfile replay path builds from the snapshot
  //                   (lazily allocated ~80 KB, reused). Only the active replayer touches
  //                   a replay slot, so no per-buffer synchronization is needed (the slot
  //                   handoff rides the worker mutex/CV).
  struct PrecordSegment {
    std::vector<uint32_t> snapshot;
    std::vector<PrecordEvent> events;
    std::vector<PrecordDraw> draws;
    std::vector<uint32_t> frommem_data;
    std::unique_ptr<RegisterFile> local_regfile;
  };
  PrecordSegment precord_slots_[2];
  uint32_t precord_capture_slot_ = 0;  // parse thread captures the open segment here
  uint32_t precord_replay_slot_ = 0;   // the replayer (inline or worker) drains this
  bool precord_segment_open_ = false;  // capture-side: gates the write-log hot path
  // [GPU-PRECORD] Phase 1b-1c Inc 5: true while THIS thread is replaying a segment.
  // Made static thread_local (was a plain member) so the parse thread never observes
  // the worker's replay state under overlap: the parse-side "should I defer?" tests
  // (IssueDraw wrapper, PrecordFlush reentrancy guard) read this and must see the
  // PARSE thread's own value (always false there in thread/overlap mode), not the
  // worker's true. Inline replay (shared-rewind / localrf) runs on the parse thread,
  // so it sets/reads its own copy -- correct in every mode. Single CP instance, so a
  // static member == one value per thread. The hot WriteRegister/FromMem capture gate
  // deliberately no longer reads this (it uses precord_segment_open_ alone, which is
  // false during every inline replay) to keep TLS off that path.
  static thread_local bool precord_replaying_;  // true while a replay is in progress
  uint32_t precord_draws_in_segment_ = 0;  // capture-side draw count -> flush at 256
  // [GPU-PRECORD] Phase 1b-1: see GetActiveDrawRegisterFile(). nullptr ⇒ the shared
  // register_file_ (default; no behavior change). A worker points this at its
  // per-segment local register file while replaying that segment's draws.
  const RegisterFile* active_draw_register_file_ = nullptr;
  // [GPU-PRECORD] Phase 1b-1b worker (Model C). The parse thread posts the captured
  // segment and blocks until the worker has replayed it (no overlap yet), so while
  // the worker runs PrecordReplayLocal the parse thread is idle ⇒ the precord_*
  // buffers and the shared D3D12 subsystems are the worker's exclusively.
  std::thread precord_worker_thread_;
  std::mutex precord_worker_mutex_;
  std::condition_variable precord_worker_cv_;
  bool precord_worker_started_ = false;
  bool precord_worker_shutdown_ = false;
  bool precord_worker_job_pending_ = false;
  // [GPU-PRECORD] Phase 1b-1c Inc 5 (H4): subsystem exclusivity tripwire. Exactly ONE
  // replayer may hold the shared subsystems (PrimitiveProcessor / RenderTargetCache /
  // TextureCache / PipelineCache) repointed to a local register file at a time -- true
  // today because capture is pure (Inc 5 delta 2) and there is a single worker. Bumped
  // around PrecordReplayLocal; a value >1 means a second concurrent replayer (Phase 2)
  // has silently voided the invariant that lets those subsystems stay lock-free.
  std::atomic<int> precord_replays_in_flight_{0};

  // [GPU-PRECORD] Phase 1b-1c Inc 4: coarse lock serializing pipeline-cache access
  // between the parse thread and the replay worker (H1/H2). The PARSE thread holds it
  // around PipelineCache::LoadShader (the shaders_ map emplace) in D3D12CommandProcessor::
  // LoadShader; the WORKER holds it across the shader-analysis/translation/ConfigurePipeline
  // span of IssueDrawImpl (AnalyzeShaderUcode + GetOrCreateTranslation + ConfigurePipeline +
  // the pipeline-handle lookup), which also covers H2 (shader-pointee mutation). Both sites
  // are gated on g_precord_thread, so the lock is inert (never taken) unless the worker is
  // running -- zero overhead for precord off / 1b-0 / localrf-inline. It is the SINGLE outer
  // lock each thread takes (holding nothing else), so it cannot deadlock. Uncontended under
  // Model C (parse blocked while the worker replays); it is Inc 5 overlap that makes it bite.
  std::mutex precord_pipeline_mutex_;

  bool debug_markers_enabled_ = false;

  // Viewport info caching - avoids redundant GetHostViewportInfo recalculation
  // when viewport-affecting register state hasn't changed between draws.
  struct ViewportCacheKey {
    uint32_t pa_cl_clip_cntl;
    uint32_t pa_cl_vte_cntl;
    uint32_t pa_su_sc_mode_cntl;
    uint32_t pa_su_vtx_cntl;
    uint32_t pa_sc_window_offset;
    uint32_t normalized_depth_control;
    uint32_t vport_regs[6];  // XSCALE, XOFFSET, YSCALE, YOFFSET, ZSCALE, ZOFFSET
    uint32_t flags;          // packed: convert_z_to_float24, full_float24, ps_writes_depth
    bool operator==(const ViewportCacheKey&) const = default;
  };
  ViewportCacheKey previous_viewport_key_{};
  draw_util::ViewportInfo previous_viewport_info_{};
  bool viewport_cache_valid_ = false;

  // Should bindless textures and samplers be used - many times faster
  // UpdateBindings than bindful (that becomes a significant bottleneck with
  // bindful - mainly because of CopyDescriptorsSimple, which takes the majority
  // of UpdateBindings time, and that's outside the emulator's control even).
  bool bindless_resources_used_ = false;

  std::unique_ptr<D3D12SharedMemory> shared_memory_;

  std::unique_ptr<D3D12RenderTargetCache> render_target_cache_;

  std::unique_ptr<ui::d3d12::D3D12UploadBufferPool> constant_buffer_pool_;

  static constexpr uint32_t kViewBindfulHeapSize = 32768;
  static_assert(kViewBindfulHeapSize <= D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1);
  std::unique_ptr<ui::d3d12::D3D12DescriptorHeapPool> view_bindful_heap_pool_;
  // Currently bound descriptor heap - updated by RequestViewBindfulDescriptors.
  ID3D12DescriptorHeap* view_bindful_heap_current_;
  // Rationale: textures have 4 KB alignment in guest memory, and there can be
  // 512 MB / 4 KB in total of them at most, and multiply by 3 for different
  // swizzles, signedness, and multiple host textures for one guest texture, and
  // transient descriptors. Though in reality there will be a lot fewer of
  // course, this is just a "safe" value. The limit is 1000000 for resource
  // binding tier 2.
  static constexpr uint32_t kViewBindlessHeapSize = 262144;
  static_assert(kViewBindlessHeapSize <= D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2);
  ID3D12DescriptorHeap* view_bindless_heap_ = nullptr;
  D3D12_CPU_DESCRIPTOR_HANDLE view_bindless_heap_cpu_start_;
  D3D12_GPU_DESCRIPTOR_HANDLE view_bindless_heap_gpu_start_;
  uint32_t view_bindless_heap_allocated_ = 0;
  std::vector<uint32_t> view_bindless_heap_free_;
  // <Descriptor index, submission where requested>, sorted by the submission
  // number.
  std::deque<std::pair<uint32_t, uint64_t>> view_bindless_one_use_descriptors_;

  // Direct3D 12 only allows shader-visible heaps with no more than 2048
  // samplers (due to Nvidia addressing). However, there's also possibly a weird
  // bug in the Nvidia driver (tested on 440.97 and earlier on Windows 10 1803)
  // that caused the sampler with index 2047 not to work if a heap with 8 or
  // less samplers also exists - in case of Xenia, it's the immediate drawer's
  // sampler heap.
  // FIXME(Triang3l): Investigate the issue with the sampler 2047 on Nvidia.
  static constexpr uint32_t kSamplerHeapSize = 2000;
  static_assert(kSamplerHeapSize <= D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE);
  std::unique_ptr<ui::d3d12::D3D12DescriptorHeapPool> sampler_bindful_heap_pool_;
  ID3D12DescriptorHeap* sampler_bindful_heap_current_;
  ID3D12DescriptorHeap* sampler_bindless_heap_current_ = nullptr;
  D3D12_CPU_DESCRIPTOR_HANDLE sampler_bindless_heap_cpu_start_;
  D3D12_GPU_DESCRIPTOR_HANDLE sampler_bindless_heap_gpu_start_;
  // Currently the sampler heap is used only for texture cache samplers, so
  // individual samplers are never freed, and using a simple linear allocator
  // inside the current heap without a free list.
  uint32_t sampler_bindless_heap_allocated_ = 0;
  // <Heap, overflow submission number>, if total sampler count used so far
  // exceeds kSamplerHeapSize, and the heap has been switched (this is not a
  // totally impossible situation considering Direct3D 9 has sampler parameter
  // state instead of sampler objects, and having one "unimportant" parameter
  // changed may result in doubling of sampler count). Sorted by the submission
  // number (so checking if the first can be reused is enough).
  std::deque<std::pair<ID3D12DescriptorHeap*, uint64_t>> sampler_bindless_heaps_overflowed_;
  // D3D12TextureCache::SamplerParameters::value -> indices within the current
  // bindless sampler heap.
  std::unordered_map<uint32_t, uint32_t> texture_cache_bindless_sampler_map_;

  // Root signatures for different descriptor counts.
  std::unordered_map<uint32_t, ID3D12RootSignature*> root_signatures_bindful_;
  ID3D12RootSignature* root_signature_bindless_vs_ = nullptr;
  ID3D12RootSignature* root_signature_bindless_ds_ = nullptr;

  std::unique_ptr<D3D12PrimitiveProcessor> primitive_processor_;

  std::unique_ptr<PipelineCache> pipeline_cache_;

  std::unique_ptr<D3D12TextureCache> texture_cache_;

  // Bytes 0x0...0x3FF - 256-entry gamma ramp table with B10G10R10X2 data (read
  // as R10G10B10X2 with swizzle).
  // Bytes 0x400...0x9FF - 128-entry PWL R16G16 gamma ramp (R - base, G - delta,
  // low 6 bits of each are zero, 3 elements per entry).
  Microsoft::WRL::ComPtr<ID3D12Resource> gamma_ramp_buffer_;
  D3D12_RESOURCE_STATES gamma_ramp_buffer_state_;
  // Upload buffer for an image that is the same as gamma_ramp_, but with
  // kQueueFrames array layers.
  Microsoft::WRL::ComPtr<ID3D12Resource> gamma_ramp_upload_buffer_;
  uint8_t* gamma_ramp_upload_buffer_mapping_ = nullptr;
  bool gamma_ramp_256_entry_table_up_to_date_ = false;
  bool gamma_ramp_pwl_up_to_date_ = false;

  struct ApplyGammaConstants {
    uint32_t size[2];
  };
  enum class ApplyGammaRootParameter : UINT {
    kConstants,
    kDestination,
    kSource,
    kRamp,

    kCount,
  };
  Microsoft::WRL::ComPtr<ID3D12RootSignature> apply_gamma_root_signature_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> apply_gamma_table_pipeline_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> apply_gamma_table_fxaa_luma_pipeline_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> apply_gamma_pwl_pipeline_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> apply_gamma_pwl_fxaa_luma_pipeline_;

  struct FxaaConstants {
    uint32_t size[2];
    float size_inv[2];
  };
  enum class FxaaRootParameter : UINT {
    kConstants,
    kDestination,
    kSource,

    kCount,
  };
  Microsoft::WRL::ComPtr<ID3D12RootSignature> fxaa_root_signature_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> fxaa_pipeline_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> fxaa_extreme_pipeline_;

  struct ResolveDownscaleConstants {
    uint32_t scale_x;
    uint32_t scale_y;
    uint32_t pixel_size_log2;
    uint32_t tile_count;
    uint32_t half_pixel_offset;
  };
  enum class ResolveDownscaleRootParameter : UINT {
    kConstants,
    kSource,
    kDestination,

    kCount,
  };
  Microsoft::WRL::ComPtr<ID3D12RootSignature> resolve_downscale_root_signature_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> resolve_downscale_pipeline_;
  Microsoft::WRL::ComPtr<ID3D12Resource> resolve_downscale_buffer_;
  uint32_t resolve_downscale_buffer_size_ = 0;

  // PWL gamma ramp can result in values with more precision than 10bpc. Though
  // those sub-10bpc bits don't have any noticeable visual effect, so normally
  // R10G10B10A2_UNORM is enough. But what's the most important is that for the
  // original FXAA shader, the luma needs to be written to the alpha channel.
  // For simplicity (to avoid modifying the FXAA shader and adding more texture
  // fetches into it), and for the highest quality (preserving all 13 bits that
  // may be generated by applying the PWL gamma ramp with an increment of 2^3,
  // and also leaving some space for the result of applying fractional weights
  // to calculate the luma), using R16G16B16A16_UNORM instead of
  // R10G10B10X2_UNORM with a separate alpha texture.
  static constexpr DXGI_FORMAT kFxaaSourceTextureFormat = DXGI_FORMAT_R16G16B16A16_UNORM;
  // Kept in NON_PIXEL_SHADER_RESOURCE state.
  Microsoft::WRL::ComPtr<ID3D12Resource> fxaa_source_texture_;
  uint64_t fxaa_source_texture_submission_ = 0;

  // Unsubmitted barrier batch.
  std::vector<D3D12_RESOURCE_BARRIER> barriers_;

  // <Submission where requested, resource>, sorted by the submission number.
  std::deque<std::pair<uint64_t, ID3D12Resource*>> resources_for_deletion_;

  static constexpr uint32_t kScratchBufferSizeIncrement = 16 * 1024 * 1024;
  ID3D12Resource* scratch_buffer_ = nullptr;
  uint32_t scratch_buffer_size_ = 0;
  D3D12_RESOURCE_STATES scratch_buffer_state_;
  bool scratch_buffer_used_ = false;

  ID3D12Resource* readback_buffer_ = nullptr;
  uint32_t readback_buffer_size_ = 0;
  std::unordered_map<uint64_t, ReadbackBuffer> readback_buffers_;
  std::unordered_map<uint64_t, ReadbackBuffer> memexport_readback_buffers_;

  static constexpr uint32_t kMaxOcclusionQueries = 8192;
  Microsoft::WRL::ComPtr<ID3D12QueryHeap> occlusion_query_heap_;
  Microsoft::WRL::ComPtr<ID3D12Resource> occlusion_query_readback_;
  uint64_t* occlusion_query_readback_mapping_ = nullptr;
  uint32_t occlusion_query_cursor_ = 0;
  bool occlusion_query_resources_available_ = false;
  struct ActiveOcclusionQuery {
    uint32_t sample_count_address = 0;
    uint32_t host_index = UINT32_MAX;
    bool valid = false;
  } active_occlusion_query_;
  struct VertexBufferState {
    uint32_t address = UINT32_MAX;
    uint32_t size = UINT32_MAX;
  };
  std::array<VertexBufferState, 96> vertex_buffer_states_{};
  uint64_t vertex_buffers_in_sync_[2] = {};

  std::atomic<bool> pix_capture_requested_ = false;
  bool pix_capturing_;

  // The current fixed-function drawing state.
  D3D12_VIEWPORT ff_viewport_;
  D3D12_RECT ff_scissor_;
  float ff_blend_factor_[4];
  uint32_t ff_stencil_ref_;
  bool ff_viewport_update_needed_;
  bool ff_scissor_update_needed_;
  bool ff_blend_factor_update_needed_;
  bool ff_stencil_ref_update_needed_;

  // Currently bound pipeline, either a graphics pipeline from the pipeline
  // cache (with potentially deferred creation - current_external_pipeline_ is
  // nullptr in this case) or a non-Xenos graphics or compute pipeline
  // (current_guest_pipeline_ is nullptr in this case).
  void* current_guest_pipeline_;
  ID3D12PipelineState* current_external_pipeline_;

  // Currently bound graphics root signature.
  ID3D12RootSignature* current_graphics_root_signature_;
  // Extra parameters which may or may not be present.
  RootBindfulExtraParameterIndices current_graphics_root_bindful_extras_;
  // Whether root parameters are up to date - reset if a new signature is bound.
  uint32_t current_graphics_root_up_to_date_;

  // System shader constants.
  DxbcShaderTranslator::SystemConstants system_constants_;

  // Float constant usage masks of the last draw call.
  uint64_t current_float_constant_map_vertex_[4];
  uint64_t current_float_constant_map_pixel_[4];

  // Constant buffer bindings.
  struct ConstantBufferBinding {
    D3D12_GPU_VIRTUAL_ADDRESS address;
    bool up_to_date;
  };
  ConstantBufferBinding cbuffer_binding_system_;
  ConstantBufferBinding cbuffer_binding_float_vertex_;
  ConstantBufferBinding cbuffer_binding_float_pixel_;
  ConstantBufferBinding cbuffer_binding_bool_loop_;
  ConstantBufferBinding cbuffer_binding_fetch_;
  ConstantBufferBinding cbuffer_binding_descriptor_indices_vertex_;
  ConstantBufferBinding cbuffer_binding_descriptor_indices_pixel_;

  // [GPU-INST] Consecutive-draw instancing coalescer. A batch is "open" after a
  // draw is deferred (its pipeline/bindings/index buffer are already recorded in
  // the command list, only the DrawIndexedInstanced is held back). Subsequent
  // draws that are identical except for their vertex float constants append their
  // per-instance constant block; on the first mismatch (or swap/copy/submission
  // end) the batch is flushed as one instanced draw. Cmd-proc thread only.
  struct InstancedBatch {
    bool active = false;
    bool indexed = false;
    // Identity used to match subsequent draws (everything not covered by the
    // "only vertex float constants changed" register-dirty signal).
    Shader* vs = nullptr;
    Shader* ps = nullptr;
    xenos::PrimitiveType prim = xenos::PrimitiveType(0);
    uint32_t index_count = 0;
    bool has_ib = false;
    uint32_t ib_base = 0;
    uint32_t ib_count = 0;
    xenos::IndexFormat ib_format = xenos::IndexFormat(0);
    xenos::Endian ib_endian = xenos::Endian(0);
    // Deferred draw parameters.
    uint32_t host_draw_vertex_count = 0;
    // Per-instance vertex float constants: float_count vec4s per instance, packed
    // in the same order as the float-constants cbuffer.
    uint32_t float_count = 0;
    uint64_t float_bitmap[4] = {0, 0, 0, 0};
    uint32_t count = 0;
    uint32_t max_instances = 0;
    std::vector<float> data;
  };
  InstancedBatch instanced_batch_;

  // [GPU-INST] Append the current draw's vertex float constants as one instance.
  void InstancedBatchAppend(const RegisterFile& regs);
  // [GPU-INST] Begin a deferred instanced batch from the current draw.
  void StartInstancedBatch(const RegisterFile& regs, Shader* vertex_shader, Shader* pixel_shader,
                           xenos::PrimitiveType primitive_type, uint32_t index_count,
                           const IndexBufferInfo* index_buffer_info,
                           uint32_t host_draw_vertex_count, bool indexed);
  // [GPU-INST] Whether the current draw can extend the open batch.
  bool InstancedBatchCanMerge(xenos::PrimitiveType primitive_type, uint32_t index_count,
                              const IndexBufferInfo* index_buffer_info, Shader* vertex_shader,
                              Shader* pixel_shader) const;
  // [GPU-INST] Emit the open batch (if any) as a single instanced draw.
  void FlushInstancedBatch();

  // Whether the latest shared memory and EDRAM buffer binding contains the
  // shared memory UAV rather than the SRV.
  // Separate descriptor tables for the SRV and the UAV, even though only one is
  // accessed dynamically in the shaders, are used to prevent a validation
  // message about missing resource states in PIX.
  std::optional<bool> current_shared_memory_binding_is_uav_;

  // Pages with the descriptors currently used for handling Xenos draw calls.
  uint64_t draw_view_bindful_heap_index_;
  uint64_t draw_sampler_bindful_heap_index_;

  // Whether the last used texture sampler bindings have been written to the
  // current view descriptor heap.
  bool bindful_textures_written_vertex_;
  bool bindful_textures_written_pixel_;
  bool bindful_samplers_written_vertex_;
  bool bindful_samplers_written_pixel_;
  // Layout UIDs and last texture and sampler bindings written to the current
  // descriptor heaps (for bindful) or descriptor index constant buffer (for
  // bindless) with the last used descriptor layout. Valid only when:
  // - For bindful, when bindful_#_written_#_ is true.
  // - For bindless, when cbuffer_binding_descriptor_indices_#_.up_to_date is
  //   true.
  size_t current_texture_layout_uid_vertex_;
  size_t current_texture_layout_uid_pixel_;
  size_t current_sampler_layout_uid_vertex_;
  size_t current_sampler_layout_uid_pixel_;
  // Size of these should be ignored when checking whether these are up to date,
  // layout UID should be checked first (they will be different for different
  // binding counts).
  std::vector<D3D12TextureCache::TextureSRVKey> current_texture_srv_keys_vertex_;
  std::vector<D3D12TextureCache::TextureSRVKey> current_texture_srv_keys_pixel_;
  std::vector<D3D12TextureCache::SamplerParameters> current_samplers_vertex_;
  std::vector<D3D12TextureCache::SamplerParameters> current_samplers_pixel_;
  std::vector<uint32_t> current_sampler_bindless_indices_vertex_;
  std::vector<uint32_t> current_sampler_bindless_indices_pixel_;

  // Latest bindful descriptor handles used for handling Xenos draw calls.
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_shared_memory_srv_and_edram_;
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_shared_memory_uav_and_edram_;
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_textures_vertex_;
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_textures_pixel_;
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_samplers_vertex_;
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_samplers_pixel_;

  // Current primitive topology.
  D3D_PRIMITIVE_TOPOLOGY primitive_topology_;

  // Temporary storage for memexport stream constants used in the draw.
  std::vector<draw_util::MemExportRange> memexport_ranges_;
};

}  // namespace rex::graphics::d3d12
