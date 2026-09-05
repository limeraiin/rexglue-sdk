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
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/graphics/d3d12/render_target_cache.h>
#include <rex/graphics/d3d12/shader.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/shader/dxbc_translator.h>
#include <rex/graphics/primitive_processor.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>
#include <rex/hash.h>
#include <rex/platform.h>
#include <rex/string/buffer.h>
#include <rex/thread.h>
#include <rex/ui/d3d12/d3d12_api.h>

namespace rex::graphics::d3d12 {

class D3D12CommandProcessor;

class PipelineCache {
 public:
  static constexpr size_t kLayoutUIDEmpty = 0;

  PipelineCache(D3D12CommandProcessor& command_processor, const RegisterFile& register_file,
                const D3D12RenderTargetCache& render_target_cache, bool bindless_resources_used);
  ~PipelineCache();

  bool Initialize();
  void Shutdown();
  // No ClearCache because it's undesirable with the persistent shader storage
  // (if the storage is reloaded, effectively nothing is cleared, while the call
  // takes a long time, and if it's not, there will be heavy stuttering for the
  // rest of the execution of the guest).

  void InitializeShaderStorage(const std::filesystem::path& cache_root, uint32_t title_id,
                               bool blocking);
  void ShutdownShaderStorage();

  void EndSubmission();
  bool IsCreatingPipelines();

  // [GPU-PRECORD] Phase 1b-1: repoint the register file the draw path reads (a
  // worker points this at its per-segment local copy during segment replay).
  void SetRegisterFile(const RegisterFile* register_file) { register_file_ = register_file; }

  D3D12Shader* LoadShader(xenos::ShaderType shader_type, const uint32_t* host_address,
                          uint32_t dword_count);
  // Analyze shader microcode on the translator thread.
  void AnalyzeShaderUcode(Shader& shader) { shader.AnalyzeUcode(ucode_disasm_buffer_); }

  // Retrieves the shader modification for the current state. The shader must
  // have microcode analyzed.
  // [GPU-INST] When instanced is true, returns the GPU-instancing variant of
  // the vertex shader (reads per-instance float constants via SV_InstanceID).
  DxbcShaderTranslator::Modification GetCurrentVertexShaderModification(
      const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type,
      uint32_t interpolator_mask, bool instanced = false) const;
  DxbcShaderTranslator::Modification GetCurrentPixelShaderModification(
      const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
      reg::RB_DEPTHCONTROL normalized_depth_control) const;

  // If draw_util::IsRasterizationPotentiallyDone is false, the pixel shader
  // MUST be made nullptr BEFORE calling this!
  bool ConfigurePipeline(D3D12Shader::D3D12Translation* vertex_shader,
                         D3D12Shader::D3D12Translation* pixel_shader,
                         const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
                         reg::RB_DEPTHCONTROL normalized_depth_control,
                         uint32_t normalized_color_mask,
                         uint32_t bound_depth_and_color_render_target_bits,
                         const uint32_t* bound_depth_and_color_render_targets_formats,
                         void** pipeline_handle_out, ID3D12RootSignature** root_signature_out);

  // Returns a pipeline with deferred creation by its handle. May return nullptr
  // if failed to create the pipeline.
  ID3D12PipelineState* GetD3D12PipelineByHandle(void* handle) const {
    return reinterpret_cast<const Pipeline*>(handle)->state.load(std::memory_order_acquire);
  }

  // [NR-PSO] Phase 5-1: emit the state-mirror verdict, at most once a second.
  // Called from the frame path rather than from the per-draw check so that
  // reading a clock is not on the command-processor thread's draw path.
  void NrPsoReportIfDue();

  // [NR-SC] Phase 5-2: emit the shader-cache verdict, at most once a second,
  // from the same place and for the same reason.
  void NrShaderCacheReportIfDue();

  // [NR-NPSO] Phase 5-3a: the native renderer's own D3D12 pipeline object for
  // the pipeline the emulated cache just selected -- built from that
  // pipeline's description (5-1's gate is what makes those bytes ours) and our
  // own shader binaries (5-2's), created by us, cached by us. Returns null
  // when this draw cannot have one, always with a counted reason; the caller
  // then binds the emulated pipeline. Called from the draw path AFTER the
  // async-creation check, so the root signature handed in is final and both
  // translations exist.
  ID3D12PipelineState* NrNativePipeline(void* pipeline_handle,
                                        ID3D12RootSignature* root_signature);
  // [NR-NPSO] Phase 5-3a: emit the native-pipeline verdict once a second.
  void NrNativePsoReportIfDue();

  // [PSO-LIB] N-10b: create a graphics PSO through the persistent driver
  // pipeline-blob library (see the private section for the mechanism).
  // Public so ad-hoc CP-thread creates outside this cache (the render target
  // cache's transfer pipelines) go through the same library and the same
  // [hitch] timing. Falls back to a plain create when the library is off or
  // unavailable; returns nullptr on failure.
  ID3D12PipelineState* CreateGraphicsPipelineWithLibrary(
      const D3D12_GRAPHICS_PIPELINE_STATE_DESC& state_desc);

 private:
  REXPACKEDSTRUCT(ShaderStoredHeader, {
    uint64_t ucode_data_hash;

    uint32_t ucode_dword_count : 31;
    xenos::ShaderType type : 1;

    static constexpr uint32_t kVersion = 0x20201219;
  });

  // Update PipelineDescription::kVersion if any of the Pipeline* enums are
  // changed!

  enum class PipelineStripCutIndex : uint32_t {
    kNone,
    kFFFF,
    kFFFFFFFF,
  };

  enum class PipelineTessellationMode : uint32_t {
    kNone,
    kDiscrete,
    kContinuous,
    kAdaptive,
  };

  enum class PipelinePatchType : uint32_t {
    kNone,
    kLine,
    kTriangle,
    kQuad,
  };

  enum class PipelinePrimitiveTopologyType : uint32_t {
    kPoint,
    kLine,
    kTriangle,
  };

  enum class PipelineGeometryShader : uint32_t {
    kNone,
    kPointList,
    kRectangleList,
    kQuadList,
  };

  enum class PipelineCullMode : uint32_t {
    kNone,
    kFront,
    kBack,
    // Special case, handled via disabling the pixel shader and depth / stencil.
    kDisableRasterization,
  };

  enum class PipelineBlendFactor : uint32_t {
    kZero,
    kOne,
    kSrcColor,
    kInvSrcColor,
    kSrcAlpha,
    kInvSrcAlpha,
    kDestColor,
    kInvDestColor,
    kDestAlpha,
    kInvDestAlpha,
    kBlendFactor,
    kInvBlendFactor,
    kSrcAlphaSat,
  };

  // Update PipelineDescription::kVersion if anything is changed!
  REXPACKEDSTRUCT(PipelineRenderTarget, {
    uint32_t used : 1;                          // 1
    xenos::ColorRenderTargetFormat format : 4;  // 5
    PipelineBlendFactor src_blend : 4;          // 9
    PipelineBlendFactor dest_blend : 4;         // 13
    xenos::BlendOp blend_op : 3;                // 16
    PipelineBlendFactor src_blend_alpha : 4;    // 20
    PipelineBlendFactor dest_blend_alpha : 4;   // 24
    xenos::BlendOp blend_op_alpha : 3;          // 27
    uint32_t write_mask : 4;                    // 31
  });

  REXPACKEDSTRUCT(PipelineDescription, {
    uint64_t vertex_shader_hash;
    uint64_t vertex_shader_modification;
    // 0 if drawing without a pixel shader.
    uint64_t pixel_shader_hash;
    uint64_t pixel_shader_modification;

    int32_t depth_bias;
    float depth_bias_slope_scaled;

    PipelineStripCutIndex strip_cut_index : 2;  // 2
    // PipelinePrimitiveTopologyType for a vertex shader.
    // xenos::TessellationMode for a domain shader.
    uint32_t primitive_topology_type_or_tessellation_mode : 2;  // 4
    // Zero for non-kVertex host_vertex_shader_type.
    PipelineGeometryShader geometry_shader : 2;       // 6
    uint32_t fill_mode_wireframe : 1;                 // 7
    PipelineCullMode cull_mode : 2;                   // 9
    uint32_t front_counter_clockwise : 1;             // 10
    uint32_t depth_clip : 1;                          // 11
    xenos::MsaaSamples host_msaa_samples : 2;         // 13
    xenos::DepthRenderTargetFormat depth_format : 1;  // 14
    xenos::CompareFunction depth_func : 3;            // 17
    uint32_t depth_write : 1;                         // 18
    uint32_t stencil_enable : 1;                      // 19
    uint32_t stencil_read_mask : 8;                   // 27

    uint32_t stencil_write_mask : 8;                   // 8
    xenos::StencilOp stencil_front_fail_op : 3;        // 11
    xenos::StencilOp stencil_front_depth_fail_op : 3;  // 14
    xenos::StencilOp stencil_front_pass_op : 3;        // 17
    xenos::CompareFunction stencil_front_func : 3;     // 20
    xenos::StencilOp stencil_back_fail_op : 3;         // 23
    xenos::StencilOp stencil_back_depth_fail_op : 3;   // 26
    xenos::StencilOp stencil_back_pass_op : 3;         // 29
    xenos::CompareFunction stencil_back_func : 3;      // 32

    PipelineRenderTarget render_targets[xenos::kMaxColorRenderTargets];

    static constexpr uint32_t kVersion = 0x20210425;
  });

  REXPACKEDSTRUCT(PipelineStoredDescription, {
    uint64_t description_hash;
    PipelineDescription description;
  });

  struct PipelineRuntimeDescription {
    ID3D12RootSignature* root_signature;
    D3D12Shader::D3D12Translation* vertex_shader;
    D3D12Shader::D3D12Translation* pixel_shader;
    const std::vector<uint32_t>* geometry_shader;
    PipelineDescription description;
  };

  struct Pipeline;

  union GeometryShaderKey {
    uint32_t key;
    struct {
      PipelineGeometryShader type : 2;
      uint32_t interpolator_count : 5;
      uint32_t user_clip_plane_count : 3;
      uint32_t user_clip_plane_cull : 1;
      uint32_t has_vertex_kill_and : 1;
      uint32_t has_point_size : 1;
      uint32_t has_point_coordinates : 1;
      // PA_CL_CLIP_CNTL::ps_ucp_mode for point primitives.
      uint32_t point_ps_ucp_mode : 2;
    };

    GeometryShaderKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const GeometryShaderKey& key) const {
        return std::hash<uint32_t>{}(key.key);
      }
    };
    bool operator==(const GeometryShaderKey& other_key) const { return key == other_key.key; }
    bool operator!=(const GeometryShaderKey& other_key) const { return !(*this == other_key); }
  };

  D3D12Shader* LoadShader(xenos::ShaderType shader_type, const uint32_t* host_address,
                          uint32_t dword_count, uint64_t data_hash);

  // Can be called from multiple threads.
  bool TranslateAnalyzedShader(DxbcShaderTranslator& translator,
                               D3D12Shader::D3D12Translation& translation,
                               IDxbcConverter* dxbc_converter = nullptr,
                               IDxcUtils* dxc_utils = nullptr,
                               IDxcCompiler* dxc_compiler = nullptr);

  // If draw_util::IsRasterizationPotentiallyDone is false, the pixel shader
  // MUST be made nullptr BEFORE calling this! The shaders must be translated
  // and valid unless for_placeholder is true.
  bool GetCurrentStateDescription(
      D3D12Shader::D3D12Translation* vertex_shader, D3D12Shader::D3D12Translation* pixel_shader,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
      uint32_t bound_depth_and_color_render_target_bits,
      const uint32_t* bound_depth_and_color_render_target_formats,
      PipelineRuntimeDescription& runtime_description_out, bool for_placeholder = false);

  // [NR-PSO] Phase 5-1: THE STATE MIRROR GATE. Derives the same pipeline
  // description from the same register file with our own independent mapping
  // (nr_pipeline_state, which shares no code with this file or with
  // draw_util), and compares. `theirs` is what GetCurrentStateDescription just
  // produced. Everything the mirror cannot yet obtain from registers -- the
  // primitive processing result, the bound render targets, the shader
  // translations -- is handed over, because those are the subsystems
  // increments 5-2 and 5-3 replace; this increment measures the register
  // mapping alone. The two normalized values are compared directly as well,
  // so that a normalization error is named even on draws where it happens not
  // to reach the packed key.
  void NrPsoCheck(const D3D12Shader::D3D12Translation* vertex_shader,
                  const D3D12Shader::D3D12Translation* pixel_shader,
                  const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
                  reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
                  uint32_t bound_depth_and_color_render_target_bits,
                  const uint32_t* bound_depth_and_color_render_target_formats,
                  const PipelineDescription& theirs);

  // [NR-SC] Phase 5-2: ask the native renderer's own shader cache for this
  // draw's two shaders, and settle the byte comparison for any key it has not
  // checked yet. Runs here, beside NrPsoCheck, for the same reason: it is the
  // only point where our translation and the emulated one are provably about
  // the same shader with the same modification.
  void NrShaderCacheCheck(const D3D12Shader::D3D12Translation* vertex_shader,
                          const D3D12Shader::D3D12Translation* pixel_shader);
  void NrShaderCacheCheckOne(const D3D12Shader::D3D12Translation* translation, uint32_t stage);
  // Creates the native translator and configures the cache on first use.
  // Shared by the 5-2 probe and the 5-3 pipeline builder, which both need our
  // DXBC and must not each configure a cache.
  void NrShaderCacheEnsure();
  // Our DXBC for a translation, or nullptr if we have none for it.
  const std::vector<uint8_t>* NrShaderCacheBinary(
      const D3D12Shader::D3D12Translation* translation, uint32_t stage);
  // Translation callback handed to the cache. Builds a Shader from the ucode
  // dwords alone -- none of this class's shader bookkeeping -- which is what
  // makes byte equality with `translation.translated_binary()` mean that a
  // native renderer can translate from what the walk recovers.
  static bool NrShaderTranslate(void* ctx, uint32_t stage, uint64_t ucode_hash,
                                const uint32_t* ucode_dwords, uint32_t ucode_dword_count,
                                uint64_t modification, std::vector<uint8_t>* dxbc_out);

  static bool GetGeometryShaderKey(PipelineGeometryShader geometry_shader_type,
                                   DxbcShaderTranslator::Modification vertex_shader_modification,
                                   DxbcShaderTranslator::Modification pixel_shader_modification,
                                   GeometryShaderKey& key_out);
  static void CreateDxbcGeometryShader(GeometryShaderKey key, std::vector<uint32_t>& shader_out);
  const std::vector<uint32_t>& GetGeometryShader(GeometryShaderKey key);

  // [NR-NPSO] Phase 5-3a: extracted out of CreateD3D12Pipeline so the same
  // mapping can be run for comparison without creating anything.
  bool BuildD3D12PipelineStateDesc(const PipelineRuntimeDescription& runtime_description,
                                   D3D12_GRAPHICS_PIPELINE_STATE_DESC& state_desc);
  // [NR-NPSO] Phase 5-3a: the two callbacks the native pipeline cache reaches
  // Direct3D through, so that the module itself holds no device.
  static ID3D12PipelineState* NrNativePsoCreate(void* ctx,
                                                const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc);
  static void NrNativePsoRelease(void* ctx, ID3D12PipelineState* state);

  ID3D12PipelineState* CreateD3D12Pipeline(const PipelineRuntimeDescription& runtime_description);
  bool PrepareRuntimeDescriptionForQueuedCreation(Pipeline* pipeline,
                                                  PipelineRuntimeDescription& runtime_description);

  // [PSO-LIB] N-10b: persist DRIVER pipeline blobs between boots via
  // ID3D12PipelineLibrary. The pipeline-description storage above recreates
  // every stored pipeline each boot, which is a full driver compile of every
  // PSO (52-64 s for 4.5k pipelines on Intel UHD 630; ~2 s warm on NVIDIA
  // only because its own driver cache hits). The library keys each PSO by a
  // hash of its full creation inputs (the desc with pointers zeroed plus the
  // shader bytecode bytes), so the next boot loads driver blobs instead of
  // compiling. LoadGraphicsPipeline validates the desc against the stored
  // pipeline, so a stale or colliding name can only cost a miss (fall back
  // to a plain create), never a wrong pipeline. Both creation paths route
  // through the helper: the emulated cache's CreateD3D12Pipeline and the NR
  // path's NrNativePsoCreate (the helper itself is declared public above).
  void InitializePipelineLibrary(const std::filesystem::path& local_root, uint32_t title_id,
                                 bool edram_rov_used);
  // Writes the library to disk if any pipelines were stored since the last
  // serialize. Called after the boot-from-storage creation pass (banks the
  // first boot's compiles even if the run later crashes), from the storage
  // write thread once enough new stores accumulate, and at shutdown.
  void SerializePipelineLibrary();
  void ShutdownPipelineLibrary();

  D3D12CommandProcessor& command_processor_;
  const RegisterFile* register_file_;
  const D3D12RenderTargetCache& render_target_cache_;
  bool bindless_resources_used_;

  // Temporary storage for AnalyzeUcode calls on the processor thread.
  string::StringBuffer ucode_disasm_buffer_;
  // Reusable shader translator for the processor thread.
  std::unique_ptr<DxbcShaderTranslator> shader_translator_;
  std::mutex translation_request_lock_;

  // [NR-SC] Phase 5-2. The native renderer's own translator and disassembly
  // buffer, kept apart from the two above so that the emulated path and ours
  // never share a translator's internal state -- the comparison has to be
  // between two independent runs, not two calls on one object. Created on
  // first use, only when the probe is on.
  std::unique_ptr<DxbcShaderTranslator> nr_shader_translator_;
  string::StringBuffer nr_ucode_disasm_buffer_;

  // [NR-NPSO] Phase 5-3a. Set once the native pipeline cache has been given
  // this object's device callbacks.
  bool nr_native_pso_configured_ = false;
  // [NR-NPSO] 5-4-4b inc 3 (gpu_nr_npso_memo): pipeline handle -> our
  // verified-ok pipeline state. Safe for this object's lifetime: a handle's
  // description is immutable, the env is device-constant, our shader-cache
  // and npso-cache entries never move or evict, and handles are only freed
  // in Shutdown. A device reset builds a fresh PipelineCache = empty memo.
  std::unordered_map<void*, ID3D12PipelineState*> nr_npso_memo_;

  // Command processor thread DXIL conversion/disassembly interfaces, if DXIL
  // disassembly is enabled.
  IDxbcConverter* dxbc_converter_ = nullptr;
  IDxcUtils* dxc_utils_ = nullptr;
  IDxcCompiler* dxc_compiler_ = nullptr;

  // Ucode hash -> shader.
  std::unordered_map<uint64_t, D3D12Shader*, rex::IdentityHasher<uint64_t>> shaders_;

  struct LayoutUID {
    size_t uid;
    size_t vector_span_offset;
    size_t vector_span_length;
  };
  std::mutex layouts_mutex_;
  // Texture binding layouts of different shaders, for obtaining layout UIDs.
  std::vector<D3D12Shader::TextureBinding> texture_binding_layouts_;
  // Map of texture binding layouts used by shaders, for obtaining UIDs. Keys
  // are XXH3 hashes of layouts, values need manual collision resolution using
  // layout_vector_offset:layout_length of texture_binding_layouts_.
  std::unordered_multimap<uint64_t, LayoutUID, rex::IdentityHasher<uint64_t>>
      texture_binding_layout_map_;
  // Bindless sampler indices of different shaders, for obtaining layout UIDs.
  // For bindful, sampler count is used as the UID instead.
  std::vector<uint32_t> bindless_sampler_layouts_;
  // Keys are XXH3 hashes of used bindless sampler indices.
  std::unordered_multimap<uint64_t, LayoutUID, rex::IdentityHasher<uint64_t>>
      bindless_sampler_layout_map_;

  // Geometry shaders for Xenos primitive types not supported by Direct3D 12.
  std::unordered_map<GeometryShaderKey, std::vector<uint32_t>, GeometryShaderKey::Hasher>
      geometry_shaders_;

  // Empty depth-only pixel shader for writing to depth buffer via ROV when no
  // Xenos pixel shader provided.
  std::vector<uint8_t> depth_only_pixel_shader_;

  struct Pipeline {
    // nullptr if creation has failed.
    std::atomic<ID3D12PipelineState*> state{nullptr};
    std::atomic<ID3D12RootSignature*> root_signature{nullptr};
    PipelineRuntimeDescription description;
    D3D12Shader::D3D12Translation* pending_vertex_shader = nullptr;
    D3D12Shader::D3D12Translation* pending_pixel_shader = nullptr;
    uint8_t priority = 0;
  };
  struct PipelineCreationPriorityComparator {
    bool operator()(const Pipeline* a, const Pipeline* b) const {
      uint8_t priority_a = a ? a->priority : 0;
      uint8_t priority_b = b ? b->priority : 0;
      return priority_a < priority_b;
    }
  };
  // All previously generated pipelines identified by hash and the description.
  std::unordered_multimap<uint64_t, Pipeline*, rex::IdentityHasher<uint64_t>> pipelines_;

  // Previously used pipeline. This matches our current state settings and
  // allows us to quickly(ish) reuse the pipeline if no registers have been
  // changed.
  Pipeline* current_pipeline_ = nullptr;

  // Currently open shader storage path.
  std::filesystem::path shader_storage_cache_root_;
  uint32_t shader_storage_title_id_ = 0;

  // Shader storage output stream, for preload in the next emulator runs.
  FILE* shader_storage_file_ = nullptr;
  // For only writing shaders to the currently open storage once, incremented
  // when switching the storage.
  uint32_t shader_storage_index_ = 0;
  bool shader_storage_file_flush_needed_ = false;

  // Pipeline storage output stream, for preload in the next emulator runs.
  FILE* pipeline_storage_file_ = nullptr;
  bool pipeline_storage_file_flush_needed_ = false;

  // [PSO-LIB] Driver pipeline-blob library. pipeline_library_data_ is the
  // backing memory of the loaded file and must outlive pipeline_library_
  // (CreatePipelineLibrary keeps referencing it). The mutex guards
  // StorePipeline and Serialize against each other; loads are free-threaded.
  ID3D12PipelineLibrary* pipeline_library_ = nullptr;
  std::vector<uint8_t> pipeline_library_data_;
  std::filesystem::path pipeline_library_path_;
  std::mutex pipeline_library_mutex_;
  std::atomic<bool> pipeline_library_stale_reported_{false};
  std::atomic<uint32_t> pipeline_library_stores_unserialized_{0};
  // True while the boot-from-storage creation pass runs: the [hitch] per
  // single >5ms WARN is suppressed there (the pass has its own summary line
  // and a cold boot would spam hundreds of them). Written and read on the
  // CP thread only.
  bool pipeline_library_boot_pass_ = false;
  // [hitch] The command-processor thread's id, so the shared create helper
  // can attribute PSO creation time to the CP thread vs a creation thread.
  // Registered from InitializeShaderStorage and EndSubmission (both run on
  // the CP thread).
  std::atomic<uint32_t> cp_thread_id_{0};

  // Thread for asynchronous writing to the storage streams.
  void StorageWriteThread();
  std::mutex storage_write_request_lock_;
  std::condition_variable storage_write_request_cond_;
  // Storage thread input is protected with storage_write_request_lock_, and the
  // thread is notified about its change via storage_write_request_cond_.
  std::deque<const Shader*> storage_write_shader_queue_;
  std::deque<PipelineStoredDescription> storage_write_pipeline_queue_;
  bool storage_write_flush_shaders_ = false;
  bool storage_write_flush_pipelines_ = false;
  bool storage_write_thread_shutdown_ = false;
  std::unique_ptr<rex::thread::Thread> storage_write_thread_;

  // Pipeline creation threads.
  void CreationThread(size_t thread_index);
  void CreateQueuedPipelinesOnProcessorThread();
  std::mutex creation_request_lock_;
  std::condition_variable creation_request_cond_;
  // Protected with creation_request_lock_, notify_one creation_request_cond_
  // when set.
  std::priority_queue<Pipeline*, std::vector<Pipeline*>, PipelineCreationPriorityComparator>
      creation_queue_;
  // Number of threads that are currently creating a pipeline - incremented when
  // a pipeline is dequeued (the completion event can't be triggered before this
  // is zero). Protected with creation_request_lock_.
  size_t creation_threads_busy_ = 0;
  // Manual-reset event set when the last queued pipeline is created and there
  // are no more pipelines to create. This is triggered by the thread creating
  // the last pipeline.
  std::unique_ptr<rex::thread::Event> creation_completion_event_;
  // Whether setting the event on completion is queued. Protected with
  // creation_request_lock_, notify_one creation_request_cond_ when set.
  bool creation_completion_set_event_ = false;
  // Creation threads with this index or above need to be shut down as soon as
  // possible. Protected with creation_request_lock_, notify_all
  // creation_request_cond_ when set.
  size_t creation_threads_shutdown_from_ = SIZE_MAX;
  std::vector<std::unique_ptr<rex::thread::Thread>> creation_threads_;
};

}  // namespace rex::graphics::d3d12
