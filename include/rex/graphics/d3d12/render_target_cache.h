/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/graphics/d3d12/shared_memory.h>
#include <rex/graphics/d3d12/texture_cache.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/render_target/cache.h>
#include <rex/graphics/trace_writer.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/memory.h>
#include <rex/ui/d3d12/d3d12_cpu_descriptor_pool.h>
#include <rex/ui/d3d12/d3d12_provider.h>
#include <rex/ui/d3d12/d3d12_upload_buffer_pool.h>
#include <rex/ui/d3d12/d3d12_util.h>
namespace rex::graphics::d3d12 {

class D3D12CommandProcessor;

class D3D12RenderTargetCache final : public RenderTargetCache {
 public:
  D3D12RenderTargetCache(const RegisterFile& register_file, const memory::Memory& memory,
                         TraceWriter& trace_writer, uint32_t draw_resolution_scale_x,
                         uint32_t draw_resolution_scale_y, D3D12CommandProcessor& command_processor,
                         bool bindless_resources_used)
      : RenderTargetCache(register_file, memory, &trace_writer, draw_resolution_scale_x,
                          draw_resolution_scale_y),
        command_processor_(command_processor),
        trace_writer_(trace_writer),
        bindless_resources_used_(bindless_resources_used) {}
  ~D3D12RenderTargetCache() override;

  bool Initialize();
  void Shutdown(bool from_destructor = false);

  void CompletedSubmissionUpdated();
  void BeginSubmission();

  Path GetPath() const override { return path_; }

  bool Update(bool is_rasterization_done, reg::RB_DEPTHCONTROL normalized_depth_control,
              uint32_t normalized_color_mask, const Shader& vertex_shader) override;

  void InvalidateCommandListRenderTargets() {
    are_current_command_list_render_targets_valid_ = false;
  }

  bool msaa_2x_supported() const { return msaa_2x_supported_; }

  void WriteEdramRawSRVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle);
  void WriteEdramRawUAVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle);
  void WriteEdramUintPow2SRVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                       uint32_t element_size_bytes_pow2);
  void WriteEdramUintPow2UAVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                       uint32_t element_size_bytes_pow2);

  // Performs the resolve to a shared memory area according to the current
  // register values, and also clears the render targets if needed. Must be in a
  // frame for calling.
  bool Resolve(const memory::Memory& memory, D3D12SharedMemory& shared_memory,
               D3D12TextureCache& texture_cache, uint32_t& written_address_out,
               uint32_t& written_length_out);

  // Returns true if any downloads were submitted to the command processor.
  bool InitializeTraceSubmitDownloads();
  void InitializeTraceCompleteDownloads();
  void RestoreEdramSnapshot(const void* snapshot);

  // For host render targets.

  bool gamma_render_target_as_unorm16() const { return gamma_render_target_as_unorm16_; }

  // Using R16G16[B16A16]_SNORM, which are -1...1, not the needed -32...32.
  // Persistent data doesn't depend on this, so can be overriden by per-game
  // configuration.
  bool IsFixed16TruncatedToMinus1To1() const {
    return GetPath() == Path::kHostRenderTargets && !REXCVAR_GET(snorm16_render_target_full_range);
  }

  // [hiz] bumps whenever PerformTransfersAndResolveClears records a transfer
  // or a resolve clear (a render target written by something other than
  // rasterization): the command processor's Hi-Z window closes on a change.
  uint64_t transfer_epoch() const { return transfer_epoch_; }

  // [hiz] the depth render target the last Update bound: resource, size in
  // pixels, sample count and its non-shader-visible depth SRV. Null when no
  // depth target is bound.
  ID3D12Resource* GetBoundDepthForHiz(uint32_t& width_out, uint32_t& height_out,
                                      uint32_t& samples_out,
                                      D3D12_CPU_DESCRIPTOR_HANDLE& srv_out);
  // [hiz] pushes the bound depth target's transition through the command
  // processor's barrier list (the checkpoint reads it, then draws resume).
  void TransitionBoundDepthForHiz(D3D12_RESOURCE_STATES state);

  bool depth_float24_round() const { return depth_float24_round_; }
  bool depth_float24_convert_in_pixel_shader() const {
    return depth_float24_convert_in_pixel_shader_;
  }

  DXGI_FORMAT GetColorResourceDXGIFormat(xenos::ColorRenderTargetFormat format) const;
  DXGI_FORMAT GetColorDrawDXGIFormat(xenos::ColorRenderTargetFormat format) const;
  DXGI_FORMAT GetColorOwnershipTransferDXGIFormat(xenos::ColorRenderTargetFormat format,
                                                  bool* is_integer_out = nullptr) const;
  static DXGI_FORMAT GetDepthResourceDXGIFormat(xenos::DepthRenderTargetFormat format);
  static DXGI_FORMAT GetDepthDSVDXGIFormat(xenos::DepthRenderTargetFormat format);
  static DXGI_FORMAT GetDepthSRVDepthDXGIFormat(xenos::DepthRenderTargetFormat format);
  static DXGI_FORMAT GetDepthSRVStencilDXGIFormat(xenos::DepthRenderTargetFormat format);

 protected:
  uint32_t GetMaxRenderTargetWidth() const override { return D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION; }
  uint32_t GetMaxRenderTargetHeight() const override {
    return D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
  }

  RenderTarget* CreateRenderTarget(RenderTargetKey key) override;

  bool IsHostDepthEncodingDifferent(xenos::DepthRenderTargetFormat format) const override;

  bool IsGammaFormatHostStorageSeparate() const override;

  // [N-10b deletion c] the RequestPixelShaderInterlockBarrier override is
  // DELETED - D3D12 never runs the interlock path; the base no-op stands.

 private:
  enum class EdramBufferModificationStatus {
    // The values are ordered by how strong the barrier conditions are.
    // No uncommitted ROV/UAV writes.
    kUnmodified,
    // Need to commit before the next ROV usage with overlap.
    kAsROV,
    // Need to commit before any next ROV usage.
    kAsUAV,
  };
  void TransitionEdramBuffer(D3D12_RESOURCE_STATES new_state);
  void MarkEdramBufferModified(
      EdramBufferModificationStatus modification_status = EdramBufferModificationStatus::kAsUAV);
  void CommitEdramBufferUAVWrites(
      EdramBufferModificationStatus commit_status = EdramBufferModificationStatus::kAsROV);

  D3D12CommandProcessor& command_processor_;
  TraceWriter& trace_writer_;
  bool bindless_resources_used_;

  Path path_ = Path::kHostRenderTargets;

  // For host render targets, an EDRAM-sized scratch buffer for:
  // - Guest render target data copied from host render targets during copying
  //   in resolves and in frame trace creation.
  // - Host float32 depth in ownership transfers when the host depth texture and
  //   the destination are the same.
  // For rasterizer-ordered view, the buffer containing the EDRAM data.
  // (Note that if a hybrid RTV / DSV + ROV approach to color render targets is
  //  added, which is, however, unlikely as it would have very complicated
  //  interaction with depth / stencil testing, host depth will need to be
  //  copied to a different buffer - the same range may have ROV-owned color and
  //  host float32 depth at the same time).
  ID3D12Resource* edram_buffer_ = nullptr;
  D3D12_RESOURCE_STATES edram_buffer_state_;
  EdramBufferModificationStatus edram_buffer_modification_status_ =
      EdramBufferModificationStatus::kUnmodified;

  // Non-shader-visible descriptor heap containing pre-created SRV and UAV
  // descriptors of the EDRAM buffer, for faster binding (by copying rather
  // than creation).
  enum class EdramBufferDescriptorIndex : uint32_t {
    kRawSRV,
    kR32UintSRV,
    kR32G32UintSRV,
    kR32G32B32A32UintSRV,
    kRawUAV,
    kR32UintUAV,
    kR32G32UintUAV,
    kR32G32B32A32UintUAV,

    kCount,
  };
  ID3D12DescriptorHeap* edram_buffer_descriptor_heap_ = nullptr;
  D3D12_CPU_DESCRIPTOR_HANDLE edram_buffer_descriptor_heap_start_;

  // Resolve copying root signature and pipelines.
  // Parameter 0 - draw_util::ResolveCopyShaderConstants or its ::DestRelative.
  // Parameter 1 - destination (shared memory or a part of it).
  // Parameter 2 - source (EDRAM).
  ID3D12RootSignature* resolve_copy_root_signature_ = nullptr;
  struct ResolveCopyShaderCode {
    const void* unscaled;
    size_t unscaled_size;
    const void* scaled;
    size_t scaled_size;
  };
  static const ResolveCopyShaderCode
      kResolveCopyShaders[size_t(draw_util::ResolveCopyShaderIndex::kCount)];
  ID3D12PipelineState* resolve_copy_pipelines_[size_t(draw_util::ResolveCopyShaderIndex::kCount)] =
      {};

  // For traces.
  ID3D12Resource* edram_snapshot_download_buffer_ = nullptr;
  std::unique_ptr<ui::d3d12::D3D12UploadBufferPool> edram_snapshot_restore_pool_;

  // For host render targets.

  class D3D12RenderTarget final : public RenderTarget {
   public:
    // descriptor_load_separate is present when the DXGI formats are different
    // for drawing and bit-exact loading (for NaN pattern preservation across
    // EDRAM tile ownership transfers in floating-point formats, and to
    // distinguish between two -1 representations in snorm formats).
    D3D12RenderTarget(RenderTargetKey key, ID3D12Resource* resource,
                      ui::d3d12::D3D12CpuDescriptorPool::Descriptor&& descriptor_draw,
                      ui::d3d12::D3D12CpuDescriptorPool::Descriptor&& descriptor_load_separate,
                      ui::d3d12::D3D12CpuDescriptorPool::Descriptor&& descriptor_srv,
                      ui::d3d12::D3D12CpuDescriptorPool::Descriptor&& descriptor_srv_stencil,
                      D3D12_RESOURCE_STATES resource_state)
        : RenderTarget(key),
          resource_(resource),
          descriptor_draw_(std::move(descriptor_draw)),
          descriptor_load_separate_(std::move(descriptor_load_separate)),
          descriptor_srv_(std::move(descriptor_srv)),
          descriptor_srv_stencil_(std::move(descriptor_srv_stencil)),
          resource_state_(resource_state) {}

    ID3D12Resource* resource() const { return resource_.Get(); }
    const ui::d3d12::D3D12CpuDescriptorPool::Descriptor& descriptor_draw() const {
      return descriptor_draw_;
    }
    const ui::d3d12::D3D12CpuDescriptorPool::Descriptor& descriptor_srv() const {
      return descriptor_srv_;
    }
    const ui::d3d12::D3D12CpuDescriptorPool::Descriptor& descriptor_srv_stencil() const {
      return descriptor_srv_stencil_;
    }
    const ui::d3d12::D3D12CpuDescriptorPool::Descriptor& descriptor_load_separate() const {
      return descriptor_load_separate_;
    }

    D3D12_RESOURCE_STATES SetResourceState(D3D12_RESOURCE_STATES new_state) {
      D3D12_RESOURCE_STATES old_state = resource_state_;
      resource_state_ = new_state;
      return old_state;
    }

    uint32_t temporary_srv_descriptor_index() const { return temporary_srv_descriptor_index_; }
    void SetTemporarySRVDescriptorIndex(uint32_t index) { temporary_srv_descriptor_index_ = index; }
    uint32_t temporary_srv_descriptor_index_stencil() const {
      return temporary_srv_descriptor_index_stencil_;
    }
    void SetTemporarySRVDescriptorIndexStencil(uint32_t index) {
      temporary_srv_descriptor_index_stencil_ = index;
    }
    uint32_t temporary_sort_index() const { return temporary_sort_index_; }
    void SetTemporarySortIndex(uint32_t index) { temporary_sort_index_ = index; }

   private:
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_;
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_draw_;
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_load_separate_;
    // Texture SRV non-shader-visible descriptors, to prepare shader-visible
    // descriptors faster, by copying rather than by creating every time.
    // TODO(Triang3l): With bindless resources, persistently store them in the
    // heap.
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_srv_;
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_srv_stencil_;
    D3D12_RESOURCE_STATES resource_state_;
    // Temporary storage for indices in operations like transfers and dumps.
    uint32_t temporary_srv_descriptor_index_ = UINT32_MAX;
    uint32_t temporary_srv_descriptor_index_stencil_ = UINT32_MAX;
    uint32_t temporary_sort_index_ = 0;
  };

  enum TransferCBVRegister : uint32_t {
    kTransferCBVRegisterStencilMask,
    kTransferCBVRegisterAddress,
    kTransferCBVRegisterHostDepthAddress,
  };
  enum TransferSRVRegister : uint32_t {
    kTransferSRVRegisterColor,
    kTransferSRVRegisterDepth,
    kTransferSRVRegisterStencil,
    kTransferSRVRegisterHostDepth,
    kTransferSRVRegisterCount,
  };
  enum TransferUsedRootParameter : uint32_t {
    // Changed 8 times per transfer.
    kTransferUsedRootParameterStencilMaskConstant,
    kTransferUsedRootParameterColorSRV,
    // Mutually exclusive with ColorSRV.
    kTransferUsedRootParameterDepthSRV,
    // Mutually exclusive with ColorSRV.
    kTransferUsedRootParameterStencilSRV,
    // May happen to be the same for different sources.
    kTransferUsedRootParameterAddressConstant,
    kTransferUsedRootParameterHostDepthSRV,
    kTransferUsedRootParameterHostDepthAddressConstant,
    kTransferUsedRootParameterCount,

    kTransferUsedRootParameterStencilMaskConstantBit =
        uint32_t(1) << kTransferUsedRootParameterStencilMaskConstant,
    kTransferUsedRootParameterColorSRVBit = uint32_t(1) << kTransferUsedRootParameterColorSRV,
    kTransferUsedRootParameterDepthSRVBit = uint32_t(1) << kTransferUsedRootParameterDepthSRV,
    kTransferUsedRootParameterStencilSRVBit = uint32_t(1) << kTransferUsedRootParameterStencilSRV,
    kTransferUsedRootParameterAddressConstantBit = uint32_t(1)
                                                   << kTransferUsedRootParameterAddressConstant,
    kTransferUsedRootParameterHostDepthSRVBit = uint32_t(1)
                                                << kTransferUsedRootParameterHostDepthSRV,
    kTransferUsedRootParameterHostDepthAddressConstantBit =
        uint32_t(1) << kTransferUsedRootParameterHostDepthAddressConstant,

    kTransferUsedRootParametersDescriptorMask =
        kTransferUsedRootParameterColorSRVBit | kTransferUsedRootParameterDepthSRVBit |
        kTransferUsedRootParameterStencilSRVBit | kTransferUsedRootParameterHostDepthSRVBit,
  };
  enum class TransferRootSignatureIndex {
    kColor,
    kDepth,
    kDepthStencil,
    kColorToStencilBit,
    kStencilToStencilBit,
    kColorAndHostDepth,
    kDepthAndHostDepth,
    kDepthStencilAndHostDepth,
    kCount,
  };
  static const uint32_t kTransferUsedRootParameters[size_t(TransferRootSignatureIndex::kCount)];
  enum class TransferMode : uint32_t {
    // 1 SRV (color texture), source constant.
    kColorToDepth,
    // 1 SRV (color texture), source constant.
    kColorToColor,

    // 1 or 2 SRVs (depth texture, stencil texture if SV_StencilRef is
    // supported), source constant.
    kDepthToDepth,
    // 2 SRVs (depth texture, stencil texture), source constant.
    kDepthToColor,

    // 1 SRV (color texture), mask constant (most frequently changed, 8 times
    // per transfer), source constant.
    kColorToStencilBit,
    // 1 SRV (stencil texture), mask constant, source constant.
    kDepthToStencilBit,

    // Two-source modes, using the host depth if it, when converted to the guest
    // format, matches what's in the owner source (not modified, keep host
    // precision), or the guest data otherwise (significantly modified, possibly
    // cleared). Stencil for SV_StencilRef is always taken from the guest
    // source.

    // 2 SRVs (color texture, host depth texture or buffer), source constant,
    // host depth source constant.
    kColorAndHostDepthToDepth,
    // When using different source and destination depth formats. 2 or 3 SRVs
    // (depth texture, stencil texture if SV_StencilRef is supported, host depth
    // texture or buffer), source constant, host depth source constant.
    kDepthAndHostDepthToDepth,

    kCount,
  };
  enum class TransferOutput {
    kColor,
    kDepth,
    // With this output, kTransferCBVRegisterStencilMask is used.
    kStencilBit,
  };
  struct TransferModeInfo {
    TransferOutput output;
    TransferRootSignatureIndex root_signature_no_stencil_ref;
    TransferRootSignatureIndex root_signature_with_stencil_ref;
  };
  static const TransferModeInfo kTransferModes[size_t(TransferMode::kCount)];

  union TransferShaderKey {
    uint32_t key;
    struct {
      xenos::MsaaSamples dest_msaa_samples : xenos::kMsaaSamplesBits;
      uint32_t dest_resource_format : xenos::kRenderTargetFormatBits;
      xenos::MsaaSamples source_msaa_samples : xenos::kMsaaSamplesBits;
      // Always 1x when host_depth_source_is_copy is true not to create the same
      // pipeline for different MSAA sample counts as it doesn't matter in this
      // case.
      xenos::MsaaSamples host_depth_source_msaa_samples : xenos::kMsaaSamplesBits;
      uint32_t source_resource_format : xenos::kRenderTargetFormatBits;
      // If host depth is also fetched, whether it's pre-copied to the EDRAM
      // buffer (but since it's just a scratch buffer, with tiles laid out
      // linearly with the same pitch as in the original render target; also no
      // swapping of 40-sample columns as opposed to the host render target -
      // this is done only for the color source).
      uint32_t host_depth_source_is_copy : 1;

      // Last bits because this affects the root signature - after sorting, only
      // change it as fewer times as possible. Depth buffers have an additional
      // stencil SRV.
      static_assert(size_t(TransferMode::kCount) <= (size_t(1) << 3));
      TransferMode mode : 3;
    };

    TransferShaderKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const TransferShaderKey& key) const {
        return std::hash<uint32_t>{}(key.key);
      }
    };
    bool operator==(const TransferShaderKey& other_key) const { return key == other_key.key; }
    bool operator!=(const TransferShaderKey& other_key) const { return !(*this == other_key); }
    bool operator<(const TransferShaderKey& other_key) const { return key < other_key.key; }
  };

  union TransferAddressConstant {
    uint32_t constant;
    struct {
      // All in tiles.
      uint32_t dest_pitch : xenos::kEdramPitchTilesBits;
      uint32_t source_pitch : xenos::kEdramPitchTilesBits;
      // Destination base in tiles minus source base in tiles (not vice versa
      // because this is a transform of the coordinate system, not addresses
      // themselves).
      // + 1 bit because this is a signed difference between two EDRAM bases.
      // 0 for host_depth_source_is_copy (ignored in this case anyway as
      // destination == source anyway).
      int32_t source_to_dest : xenos::kEdramBaseTilesBits + 1;
    };
    TransferAddressConstant() : constant(0) { static_assert_size(*this, sizeof(constant)); }
    bool operator==(const TransferAddressConstant& other_constant) const {
      return constant == other_constant.constant;
    }
    bool operator!=(const TransferAddressConstant& other_constant) const {
      return !(*this == other_constant);
    }
  };

  struct TransferInvocation {
    Transfer transfer;
    TransferShaderKey shader_key;
    TransferInvocation(const Transfer& transfer, const TransferShaderKey& shader_key)
        : transfer(transfer), shader_key(shader_key) {}
    bool operator<(const TransferInvocation& other_invocation) const {
      // TODO(Triang3l): See if it may be better to sort by the source in the
      // first place, especially when reading the same data multiple times (like
      // to write the stencil bits after depth) for better read locality.
      // Sort by the shader key primarily to reduce pipeline state (context)
      // switches.
      if (shader_key != other_invocation.shader_key) {
        return shader_key < other_invocation.shader_key;
      }
      // Host depth render targets are changed rarely if they exist, won't save
      // many binding changes, ignore them for simplicity (their existence is
      // caught by the shader key change).
      assert_not_null(transfer.source);
      assert_not_null(other_invocation.transfer.source);
      uint32_t source_index =
          static_cast<const D3D12RenderTarget*>(transfer.source)->temporary_sort_index();
      uint32_t other_source_index =
          static_cast<const D3D12RenderTarget*>(other_invocation.transfer.source)
              ->temporary_sort_index();
      if (source_index != other_source_index) {
        return source_index < other_source_index;
      }
      return transfer.start_tiles < other_invocation.transfer.start_tiles;
    }
    bool CanBeMergedIntoOneDraw(const TransferInvocation& other_invocation) const {
      return shader_key == other_invocation.shader_key &&
             transfer.AreSourcesSame(other_invocation.transfer);
    }
  };

  union DumpPipelineKey {
    uint32_t key;
    struct {
      xenos::MsaaSamples msaa_samples : 2;
      uint32_t resource_format : 4;
      // Last bit because this affects the root signature - after sorting, only
      // change it at most once. Depth buffers have an additional stencil SRV.
      uint32_t is_depth : 1;
    };

    DumpPipelineKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const DumpPipelineKey& key) const { return std::hash<uint32_t>{}(key.key); }
    };
    bool operator==(const DumpPipelineKey& other_key) const { return key == other_key.key; }
    bool operator!=(const DumpPipelineKey& other_key) const { return !(*this == other_key); }
    bool operator<(const DumpPipelineKey& other_key) const { return key < other_key.key; }

    xenos::ColorRenderTargetFormat GetColorFormat() const {
      assert_false(is_depth);
      return xenos::ColorRenderTargetFormat(resource_format);
    }
    xenos::DepthRenderTargetFormat GetDepthFormat() const {
      assert_true(is_depth);
      return xenos::DepthRenderTargetFormat(resource_format);
    }
  };

  union DumpOffsets {
    uint32_t offsets;
    struct {
      // May be beyond the EDRAM tile count in case of EDRAM addressing
      // wrapping, thus + 1 bit.
      uint32_t dispatch_first_tile : xenos::kEdramBaseTilesBits + 1;
      uint32_t source_base_tiles : xenos::kEdramBaseTilesBits;
    };
    DumpOffsets() : offsets(0) { static_assert_size(*this, sizeof(offsets)); }
    bool operator==(const DumpOffsets& other_offsets) const {
      return offsets == other_offsets.offsets;
    }
    bool operator!=(const DumpOffsets& other_offsets) const { return !(*this == other_offsets); }
  };

  union DumpPitches {
    uint32_t pitches;
    struct {
      // Both in tiles.
      uint32_t dest_pitch : xenos::kEdramPitchTilesBits;
      uint32_t source_pitch : xenos::kEdramPitchTilesBits;
    };
    DumpPitches() : pitches(0) { static_assert_size(*this, sizeof(pitches)); }
    bool operator==(const DumpPitches& other_pitches) const {
      return pitches == other_pitches.pitches;
    }
    bool operator!=(const DumpPitches& other_pitches) const { return !(*this == other_pitches); }
  };

  enum DumpCbuffer : uint32_t {
    kDumpCbufferOffsets,
    kDumpCbufferPitches,
    kDumpCbufferCount,
  };

  enum DumpRootParameter : uint32_t {
    // May be changed multiple times for the same source.
    kDumpRootParameterOffsets,
    // One resolve may need multiple sources.
    kDumpRootParameterSource,

    // May be different for different sources.
    kDumpRootParameterColorPitches = kDumpRootParameterSource + 1,
    // Only changed between 32bpp and 64bpp.
    kDumpRootParameterColorEdram,

    kDumpRootParameterColorCount,

    // Same change frequency than the source (though currently the command
    // processor can't contiguously allocate multiple descriptors with bindless,
    // when such functionality is added, switch to one root signature).
    kDumpRootParameterDepthStencil = kDumpRootParameterSource + 1,
    kDumpRootParameterDepthPitches,
    kDumpRootParameterDepthEdram,

    kDumpRootParameterDepthCount,
  };

  struct DumpInvocation {
    ResolveCopyDumpRectangle rectangle;
    DumpPipelineKey pipeline_key;
    DumpInvocation(const ResolveCopyDumpRectangle& rectangle, const DumpPipelineKey& pipeline_key)
        : rectangle(rectangle), pipeline_key(pipeline_key) {}
    bool operator<(const DumpInvocation& other_invocation) const {
      // Sort by the pipeline key primarily to reduce pipeline state (context)
      // switches.
      if (pipeline_key != other_invocation.pipeline_key) {
        return pipeline_key < other_invocation.pipeline_key;
      }
      assert_not_null(rectangle.render_target);
      uint32_t render_target_index =
          static_cast<const D3D12RenderTarget*>(rectangle.render_target)->temporary_sort_index();
      const ResolveCopyDumpRectangle& other_rectangle = other_invocation.rectangle;
      uint32_t other_render_target_index =
          static_cast<const D3D12RenderTarget*>(other_rectangle.render_target)
              ->temporary_sort_index();
      if (render_target_index != other_render_target_index) {
        return render_target_index < other_render_target_index;
      }
      if (rectangle.row_first != other_rectangle.row_first) {
        return rectangle.row_first < other_rectangle.row_first;
      }
      return rectangle.row_first_start < other_rectangle.row_first_start;
    }
  };

  // [NR-DRES] N-10a direct resolve: a resolve copy that reads the host render
  // target itself and writes the tiled guest destination in ONE compute
  // dispatch. No EDRAM dump, no vendored copy shader. TryResolveCopyDirectly
  // is the PREFLIGHT only - it answers eligibility and fills
  // direct_resolve_plan_; the dispatch replaces the vendored copy dispatch at
  // its site in Resolve so shared-memory commit ordering and written-range
  // bookkeeping stay identical. Anything outside the closed set declines with
  // a counted reason ([[count-refusals-every-granularity]]) and takes the dump
  // path unchanged. Pipelines are HLSL compiled at runtime through
  // D3DCompiler_47 (already loaded by the provider for disassembly); if the
  // compiler is unavailable every resolve declines and nothing changes.
  struct DirectResolvePlan {
    ID3D12PipelineState* pipeline = nullptr;
    D3D12RenderTarget* render_target = nullptr;
    // b0 layout: dest_info raw | dest_coordinate_info raw | dest_base |
    // width|height<<16 | rt origin x|y<<16 | sample list+count |
    // folded bias factor bits | reserved.
    uint32_t constants[8] = {};
    uint32_t group_count_x = 0, group_count_y = 0;
    bool is_depth = false;
    // Resolution-scaled: the dest is the texture cache's scaled resolve
    // range (windowed UAV, dest_base 0), not shared memory.
    bool scaled = false;
    // For the verify diagnostics: the guest dest base (constants[2] is zeroed
    // when scaled) and the dest bytes-per-block log2 (2, or 0 for 8bpp).
    uint32_t dest_base_guest = 0;
    uint32_t bpp_log2 = 2;
  };
  enum class DirectResolveDecline : uint32_t {
    kScaled,         // resolution-scaled (not in increment 1)
    kShaderClass,    // not a fast-32bpp copy shader
    kDestArray,      // 3D/array destination
    kNoRect,         // no owning render target rectangle
    kMultiRect,      // span owned by more than one render target
    kRectPartial,    // single owner does not cover the whole span
    kGeometry,       // base delta / pitch / msaa mismatch vs the owning RT
    kFormat,         // format outside the implemented set
    kPipeline,       // shader compile / root signature / pso failure
    kCount,
  };

  // Returns:
  // - A pointer to 1 pipeline for writing color or depth (or stencil via
  //   SV_StencilRef).
  // - A pointer to 8 pipelines for writing stencil by discarding samples
  //   depending on whether they have one bit set, from 1 << 0 to 1 << 7.
  // - Null if failed to create.
  ID3D12PipelineState* const* GetOrCreateTransferPipelines(TransferShaderKey key);

  static TransferMode GetTransferMode(bool dest_is_stencil_bit, bool dest_is_depth,
                                      bool source_is_depth, bool source_has_host_depth) {
    assert_true(dest_is_depth || (!dest_is_stencil_bit && !source_has_host_depth));
    if (dest_is_stencil_bit) {
      return source_is_depth ? TransferMode::kDepthToStencilBit : TransferMode::kColorToStencilBit;
    }
    if (dest_is_depth) {
      if (source_is_depth) {
        return source_has_host_depth ? TransferMode::kDepthAndHostDepthToDepth
                                     : TransferMode::kDepthToDepth;
      }
      return source_has_host_depth ? TransferMode::kColorAndHostDepthToDepth
                                   : TransferMode::kColorToDepth;
    }
    return source_is_depth ? TransferMode::kDepthToColor : TransferMode::kColorToColor;
  }

  // Do ownership transfers for render targets - each render target / vector may
  // be null / empty in case there's nothing to do for them.
  // resolve_clear_rectangle is expected to be provided by
  // PrepareHostRenderTargetsResolveClear which should do all the needed size
  // bound checks.
  void PerformTransfersAndResolveClears(
      uint32_t render_target_count, RenderTarget* const* render_targets,
      const std::vector<Transfer>* render_target_transfers,
      const uint64_t* render_target_resolve_clear_values = nullptr,
      const Transfer::Rectangle* resolve_clear_rectangle = nullptr);

  // Accepts an array of (1 + xenos::kMaxColorRenderTargets) render targets,
  // first depth, then color.
  void SetCommandListRenderTargets(RenderTarget* const* depth_and_color_render_targets);

  ID3D12PipelineState* GetOrCreateDumpPipeline(DumpPipelineKey key);
  // [NR-DRES] full/pack_class/src_gamma16 extend the variant space for the
  // full-class (averaging/format-converting) shaders; the map key packs them
  // into the DumpPipelineKey's spare upper bits.
  ID3D12PipelineState* GetOrCreateDirectResolvePipeline(DumpPipelineKey key, bool full,
                                                        uint32_t pack_class, bool src_gamma16);
  bool TryResolveCopyDirectly(const draw_util::ResolveInfo& resolve_info,
                              draw_util::ResolveCopyShaderIndex copy_shader,
                              bool draw_resolution_scaled);
  // [NR-DRES] Issues the prepared direct resolve at the vendored copy dispatch
  // site. Returns false only on transient descriptor exhaustion (the same
  // failure mode the vendored path has there).
  bool DispatchDirectResolve(D3D12SharedMemory& shared_memory, D3D12TextureCache& texture_cache);
  // [NR-DRES] Verify support: snapshots the dest span into readback slot
  // `stage` (0 = legacy result, 1 = direct result) for a CPU compare a second
  // later. Rate-limited internally. Scaled dests snapshot from the scaled
  // resolve buffer chunk (single-chunk extents only).
  void DirectResolveVerifySnapshot(D3D12SharedMemory& shared_memory,
                                   D3D12TextureCache& texture_cache, uint32_t stage,
                                   uint32_t dest_start, uint32_t dest_length);
  // [NR-DRES] Verify arbiter: snapshots the source RT texture itself
  // (single-sampled color only) so a diverge report can say which side
  // matches the render target.
  void DirectResolveVerifySnapshotSource();
  void ReportDirectResolveStats();

  // Writes contents of host render targets within rectangles from
  // ResolveInfo::GetCopyEdramTileSpan to edram_buffer_.
  bool DumpRenderTargets(uint32_t dump_base, uint32_t dump_row_length_used, uint32_t dump_rows,
                         uint32_t dump_pitch);

  bool use_stencil_reference_output_ = false;

  bool gamma_render_target_as_unorm16_ = false;

  bool depth_float24_round_ = false;
  bool depth_float24_convert_in_pixel_shader_ = false;

  bool msaa_2x_supported_ = false;

  std::shared_ptr<ui::d3d12::D3D12CpuDescriptorPool> descriptor_pool_color_;
  std::shared_ptr<ui::d3d12::D3D12CpuDescriptorPool> descriptor_pool_depth_;
  std::shared_ptr<ui::d3d12::D3D12CpuDescriptorPool> descriptor_pool_srv_;
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor null_rtv_descriptor_ss_;
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor null_rtv_descriptor_ms_;

  // Possible tile ownership transfer paths:
  // - To color:
  //   - From color: 1 SRV (color).
  //   - From depth: 2 SRVs (depth, stencil).
  // - To depth / stencil (with SV_StencilRef):
  //   - From color: 1 SRV (color).
  //   - From depth: 2 SRVs (depth, stencil).
  //   - From color and float32 depth: 2 SRVs (color with stencil, depth).
  //     - Different depth buffer: depth SRV is a texture.
  //     - Same depth buffer: depth SRV is a buffer (pre-copied).
  // - To depth (no SV_StencilRef):
  //   - From color: 1 SRV (color).
  //   - From depth: 1 SRV (depth).
  //   - From color and float32 depth: 2 SRVs (color, depth).
  //     - Different depth buffer: depth SRV is a texture.
  //     - Same depth buffer: depth SRV is a buffer (pre-copied).
  // - To stencil (no SV_StencilRef):
  //   - From color: 1 SRV (color).
  //   - From depth: 1 SRV (stencil).

  const RenderTarget* const*
      current_command_list_render_targets_[1 + xenos::kMaxColorRenderTargets];
  bool are_current_command_list_render_targets_valid_ = false;

  // Temporary storage for descriptors used in PerformTransfersAndResolveClears
  // and DumpRenderTargets.
  std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> current_temporary_descriptors_cpu_;
  std::vector<ui::d3d12::util::DescriptorCpuGpuHandlePair> current_temporary_descriptors_gpu_;

  // [NR-XFER] N-10b native host-depth snapshot: when a depth transfer's host
  // depth source is the destination itself, the dest depth plane is copied
  // into this scratch texture and the transfer shader reads it as an ordinary
  // host-depth TEXTURE source with identity addressing. (The legacy compute
  // store into the EDRAM buffer was deleted after the naruto_627 gate.)
  // Keyed by the dest resource's
  // {format, sample count, width, height}; the game's closed RT config set
  // keeps this at a couple of entries, created once and cached forever.
  struct NativeHostDepthScratch {
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_srv;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
  };
  std::map<uint64_t, NativeHostDepthScratch> native_hds_scratch_;
  // Returns null on creation failure (cached; the caller then takes the
  // legacy EDRAM store).
  NativeHostDepthScratch* GetOrCreateNativeHostDepthScratch(D3D12RenderTarget& dest_rt);

  // [NR-XFER] Transfer census, printed 1 Hz from
  // PerformTransfersAndResolveClears. Answers, from any drive, which
  // ownership-transfer shapes the game actually hits - in particular whether
  // the self-referential host-depth path (the last shipping-path EDRAM
  // dependency outside the resolve fallback) ever fires.
  uint64_t transfer_epoch_ = 0;  // [hiz]
  uint64_t xfer_census_passes_ = 0;
  uint64_t xfer_census_modes_[8] = {};  // TransferMode order
  uint64_t xfer_census_stencil_bit_ = 0;
  uint64_t xfer_census_hds_fail_ = 0;
  uint64_t xfer_census_hds_native_ = 0;
  std::chrono::steady_clock::time_point xfer_census_last_report_{};

  // [xfer] The transfer READ census and the in-place skip cycler
  // (gpu_xfer_cycle). One record per ownership transfer landed (or skipped)
  // into a destination range; it is finalized by the first later event on
  // that range: the tiles taken away again (never touched), a resolve clear,
  // a draw that reads the destination (blend factor / partial mask / depth or
  // stencil test), a resolve reading it, or a plain write first and then one
  // of those. Live = read before any write; the written-then-read outcomes
  // are the coverage question a visual drive answers.
  enum XferUseOutcome : uint32_t {
    kXferUseTaken,
    kXferUseWrittenTaken,
    kXferUseCleared,
    kXferUseReadDraw,
    kXferUseWrittenReadDraw,
    kXferUseReadResolve,
    kXferUseWrittenResolve,
    kXferUseEvicted,
    kXferUseOutcomeCount,
  };
  // c2c / d2c / c2d / d2d by the (source, dest) depth bits.
  static constexpr uint32_t kXferUseClassCount = 4;
  struct XferUseRecord {
    RenderTargetKey dest;
    uint32_t start_tiles;
    uint32_t end_tiles;
    uint32_t cls;
    bool written;
    bool from_cleared;  // the source range still held a resolve clear
  };
  std::vector<XferUseRecord> xfer_use_records_;
  // Ranges a resolve clear filled and no draw has touched since, per key.
  struct XferClearedRange {
    RenderTargetKey key;
    uint32_t start_tiles;
    uint32_t end_tiles;
  };
  std::vector<XferClearedRange> xfer_cleared_ranges_;
  uint64_t xfer_use_const_[kXferUseClassCount] = {};
  uint64_t xfer_use_const_last_[kXferUseClassCount] = {};
  // Per destination key, this second: outcomes + const, printed as [xfer-key].
  std::map<uint32_t, std::array<uint64_t, kXferUseOutcomeCount + 1>> xfer_use_by_key_;
  uint64_t xfer_use_counts_[kXferUseClassCount][kXferUseOutcomeCount] = {};
  uint64_t xfer_use_counts_last_[kXferUseClassCount][kXferUseOutcomeCount] = {};
  uint64_t xfer_use_executed_ = 0, xfer_use_executed_last_ = 0;
  uint64_t xfer_use_skipped_ = 0, xfer_use_skipped_last_ = 0;
  uint64_t xfer_use_work_passes_ = 0, xfer_use_work_passes_last_ = 0;
  uint32_t xfer_cycle_phase_ = 1;  // 1 = no transfers (shipped), 0 = all performed (A/B only)
  std::chrono::steady_clock::time_point xfer_cycle_phase_start_{};
  void XferUseAdd(RenderTargetKey dest, const Transfer& transfer);
  void XferUseFinalize(RenderTargetKey key, uint32_t start_tiles, uint32_t end_tiles,
                       XferUseOutcome pending_outcome, XferUseOutcome written_outcome);
  void XferUseNoteDraw(RenderTargetKey key, bool reads_dest);
  void XferUseNoteClear(RenderTargetKey key, const Transfer::Rectangle& rectangle);
  void XferUseNoteResolveRead(RenderTargetKey key, uint32_t start_tiles, uint32_t end_tiles);
  void XferUseReport();

  std::unique_ptr<ui::d3d12::D3D12UploadBufferPool> transfer_vertex_buffer_pool_;

  ID3D12RootSignature* transfer_root_signatures_[size_t(TransferRootSignatureIndex::kCount)] = {};
  std::unordered_map<TransferShaderKey, ID3D12PipelineState*, TransferShaderKey::Hasher>
      transfer_pipelines_;
  std::unordered_map<TransferShaderKey, std::array<ID3D12PipelineState*, 8>,
                     TransferShaderKey::Hasher>
      transfer_stencil_bit_pipelines_;

  // Temporary storage for PerformTransfersAndResolveClears.
  std::vector<TransferInvocation> current_transfer_invocations_;

  // Temporary storage for DumpRenderTargets.
  std::vector<ResolveCopyDumpRectangle> dump_rectangles_;
  std::vector<DumpInvocation> dump_invocations_;

  ID3D12RootSignature* dump_root_signature_color_ = nullptr;
  ID3D12RootSignature* dump_root_signature_depth_ = nullptr;
  // Compute pipelines for copying host render target contents to the EDRAM
  // buffer. May be null if failed to create.
  std::unordered_map<DumpPipelineKey, ID3D12PipelineState*, DumpPipelineKey::Hasher>
      dump_pipelines_;
  // [NR-DRES] Direct resolve: own root signatures (color: constants + dest UAV
  // + source SRV; depth: + stencil SRV), runtime-compiled pipelines keyed by
  // the same {msaa, format, is_depth} key as the dump pipelines (null cached on
  // compile failure), the preflight-filled plan, and the decline census.
  ID3D12RootSignature* direct_resolve_root_signature_color_ = nullptr;
  ID3D12RootSignature* direct_resolve_root_signature_depth_ = nullptr;
  std::unordered_map<DumpPipelineKey, ID3D12PipelineState*, DumpPipelineKey::Hasher>
      direct_resolve_pipelines_;
  DirectResolvePlan direct_resolve_plan_;
  uint64_t direct_resolve_attempt_count_ = 0;
  uint64_t direct_resolve_success_count_ = 0;
  uint64_t direct_resolve_fallback_count_ = 0;
  uint64_t direct_resolve_declines_[size_t(DirectResolveDecline::kCount)] = {};
  // Class declines itemized by the vendored shader they fell back to.
  uint64_t direct_resolve_class_declines_[size_t(draw_util::ResolveCopyShaderIndex::kCount)] = {};
  uint64_t direct_resolve_dispatch_count_ = 0;
  // [NR-DRES] Verify ring: slot 0 = legacy result, slot 1 = direct result of
  // the same resolve; compared on the CPU one second later (the copy has long
  // since retired, same pattern as NrDetileEdramProbe). Never sampled from
  // upload/readback memory mid-flight ([[upload-heap-readback-trap]]).
  // Slot 2 holds a copy of the source RT texture itself (single-sampled
  // sources only), the arbiter when legacy and direct disagree: whichever
  // side matches the RT is right.
  ID3D12Resource* dres_verify_readback_[3] = {nullptr, nullptr, nullptr};
  uint32_t dres_verify_pending_length_ = 0;
  uint32_t dres_verify_pending_dest_ = 0;
  uint32_t dres_verify_pending_format_ = 0;
  uint32_t dres_verify_pending_src_rows_ = 0;  // 0 = no RT snapshot this pair
  uint32_t dres_verify_pending_src_pitch_ = 0;  // bytes per row in slot 2
  // When the pending pair was recorded. The compare must wait until the
  // copies have retired (>= 1 s, the NrDetileEdramProbe pattern) - comparing
  // on the next Resolve call reads the PREVIOUS pair's bytes under THIS
  // pair's metadata and misattributes every diagnostic.
  std::chrono::steady_clock::time_point dres_verify_pending_time_{};
  // The plan constants + source identity of the pending pair, so a diverge
  // report can name the exact pixel (forward-scan inverse of the tiled
  // address) and the sample mapping in effect.
  uint32_t dres_verify_pending_constants_[8] = {};
  uint32_t dres_verify_pending_rt_key_ = 0;
  uint32_t dres_verify_pending_dest_base_ = 0;
  uint32_t dres_verify_pending_bpp_log2_ = 2;
  bool dres_verify_pending_scaled_ = false;
  std::chrono::steady_clock::time_point dres_verify_last_{};
  uint64_t dres_verify_compared_ = 0;
  uint64_t dres_verify_diverged_ = 0;
  uint64_t dres_verify_diverged_dwords_ = 0;
  std::chrono::steady_clock::time_point dres_report_last_{};

  // Parameter 0 - 2 root constants (red, green).
  ID3D12RootSignature* uint32_rtv_clear_root_signature_ = nullptr;
  // [32 or 32_32][MSAA samples].
  ID3D12PipelineState* uint32_rtv_clear_pipelines_[2][size_t(xenos::MsaaSamples::k4X) + 1] = {};

  std::vector<Transfer> clear_transfers_[2];

  // Temporary storage for DXBC building.
  std::vector<uint32_t> built_shader_;

  // [N-10b deletion c] the ROV resolve-clear root signature + pipelines are
  // DELETED.
};

}  // namespace rex::graphics::d3d12
