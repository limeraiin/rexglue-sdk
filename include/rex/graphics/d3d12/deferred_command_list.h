/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2019 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <rex/assert.h>
#include <rex/literals.h>
#include <rex/math.h>
#include <rex/ui/d3d12/d3d12_api.h>

namespace rex::graphics::d3d12 {

using namespace ::rex::literals;

class D3D12CommandProcessor;

class DeferredCommandList {
 public:
  DeferredCommandList(const D3D12CommandProcessor& command_processor,
                      size_t initial_size_bytes = 1_MiB);

  void Reset();
  void Execute(ID3D12GraphicsCommandList* command_list, ID3D12GraphicsCommandList1* command_list_1);

  // [NR-BFC] Phase 5-4-6-0: current stream size in elements -- the span
  // anchor for the buffer-replay census (the stream is Reset per submission,
  // so spans are only meaningful within one submission).
  size_t stream_size_elements() const { return command_stream_.size(); }
  // [NR-BFC] Bumped on every Reset. THE span-anchor validity check: the
  // submission id alone is NOT one -- submission_current_ increments at
  // EndSubmission's Signal while the stream Reset happens at the NEXT
  // BeginSubmission, so an anchor latched while no submission was open can
  // point mid-command into a refilled stream under an unchanged id (found
  // by an AV in NrBfcScan on the first smoke).
  uint64_t reset_generation() const { return reset_generation_; }

  // [NR-BFC] Census over the stream tail from `start_elements`: per-class
  // command counts for the buffer-replay design (the stream is
  // self-describing, so the scan needs no execution). `graphics_cbv_by_root`
  // buckets kD3DSetGraphicsRootConstantBufferView by root parameter index
  // (capped at 15) so the caller can split the sys-constants slot out.
  struct NrBfcSpanCounts {
    uint32_t draw = 0, pso = 0, root_cbv = 0, root_other = 0, ia = 0, vp = 0,
             sci = 0, om_rt = 0, om_misc = 0, barrier = 0, copy = 0,
             clear = 0, dispatch = 0, query = 0, marker = 0, heaps = 0,
             other = 0;
    uint32_t graphics_cbv_by_root[16] = {};
  };
  void NrBfcScan(size_t start_elements, NrBfcSpanCounts* out) const;

  // [NR-DSP] Phase 5-4-7-0: per-draw native span capture + compare. The
  // question this answers: if a draw's recorded native commands were
  // memcpy'd back on a later execution the reuse model calls identical,
  // would they reproduce what the derivation freshly emits? Everything a
  // replay would have to PATCH is classified as dynamic rather than as a
  // difference: a root view's GPU address (the per-frame constant pool
  // hands out a new one every Request) and a descriptor-table base.
  struct NrDspDiff {
    uint32_t cmds = 0;        // commands walked
    uint32_t dyn_view = 0;    // differ only in a root view's GPU address
    uint32_t dyn_table = 0;   // differ only in a descriptor-table base
    uint32_t real = 0;        // any other difference
    uint32_t view_sites = 0;  // root-view sets = the patch sites a replay owes
    uint32_t first_real = 0xFFFFFFFFu;  // Command enum of the first real diff
    bool length_differs = false;
  };
  // Copy [start_elements, end) out; returns elements copied, 0 if it does
  // not fit in `capacity`.
  size_t NrDspCopySpan(size_t start_elements, uintmax_t* dst, size_t capacity) const;
  void NrDspCompareSpan(const uintmax_t* prev, size_t prev_len, size_t start_elements,
                        NrDspDiff* out) const;

  // [NR-SPR] Phase 5-4-7-1: scan a draw's span against the REPLAYABLE
  // WHITELIST (root signature / pipeline / graphics root parameters /
  // topology / index buffer / exactly one draw) and collect the element
  // offsets of the root-view sets -- the patch sites a replay overwrites in
  // place. Everything outside the whitelist is a refuse class: ff
  // (viewport/scissor/blend factor/stencil ref/sample positions -- the
  // bin-dependent set a replay keeps live), barriers, compute/copy/clear/
  // dispatch/query (texture-load and resolve work emitted inside the
  // bracket), descriptor heaps, and anything unknown.
  static constexpr uint32_t kNrSprMaxViewSites = 12;
  struct NrSprScan {
    uint32_t cmds = 0;
    uint32_t draw = 0;
    uint32_t view_sites = 0;   // graphics root CBV/SRV/UAV sets
    uint32_t table_sites = 0;  // graphics root descriptor tables
    uint32_t ff = 0;
    uint32_t barrier = 0;
    uint32_t compute = 0;      // compute root state / copy / clear / dispatch / query
    uint32_t heaps = 0;
    uint32_t other = 0;        // OM RT sets, markers, IA VB, anything unknown
    bool malformed = false;    // stale anchor / truncated tail
    uint8_t view_offset_count = 0;                 // capped at kNrSprMaxViewSites
    uint16_t view_offsets[kNrSprMaxViewSites] = {};  // element offset from span start
  };
  void NrSprScanSpan(size_t start_elements, NrSprScan* out) const;
  // [NR-SPW] Phase 5-4-7-2: reserve `len_elements` at the stream tail and
  // return the destination -- the caller memcpys a patched recording in.
  // The bytes are a span this list itself emitted earlier (self-describing
  // commands only, whitelist-gated at record), so Execute walks them like
  // any other range.
  uintmax_t* NrSprAppendRaw(size_t len_elements) {
    const size_t offset = command_stream_.size();
    command_stream_.resize(offset + len_elements);
    return command_stream_.data() + offset;
  }
  // [NR-SPW] patch-site plumbing over a recorded span (the command stream
  // types stay private): decode each site's root parameter index, refusing
  // any site that is not a graphics root CBV set; overwrite a site's GPU
  // address in place at replay.
  static bool NrSprViewSiteRoots(const uintmax_t* span, const uint16_t* offsets,
                                 uint32_t count, uint32_t* roots_out);
  static void NrSprPatchViewAddress(uintmax_t* span, uint32_t offset,
                                    uint64_t gpu_address);

  // [GPU-PRECORD] Phase 1a-ii: move the recorded command stream out (leaving this
  // list empty, ready to record the next segment) for later ordered replay.
  std::vector<uintmax_t> TakeStream() {
    std::vector<uintmax_t> out = std::move(command_stream_);
    command_stream_ = std::vector<uintmax_t>();
    return out;
  }
  // Replay a previously-taken segment stream into the real command list, in order,
  // without disturbing the stream currently being recorded.
  void ExecuteStream(const std::vector<uintmax_t>& stream, ID3D12GraphicsCommandList* command_list,
                     ID3D12GraphicsCommandList1* command_list_1) {
    ExecuteRange(stream.data(), stream.size(), command_list, command_list_1);
  }

  D3D12_RECT* ClearDepthStencilViewAllocatedRects(D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_view,
                                                  D3D12_CLEAR_FLAGS clear_flags, FLOAT depth,
                                                  UINT8 stencil, UINT num_rects) {
    auto args = reinterpret_cast<ClearDepthStencilViewHeader*>(
        WriteCommand(Command::kD3DClearDepthStencilView,
                     sizeof(ClearDepthStencilViewHeader) + num_rects * sizeof(D3D12_RECT)));
    args->depth_stencil_view = depth_stencil_view;
    args->clear_flags = clear_flags;
    args->depth = depth;
    args->stencil = stencil;
    args->num_rects = num_rects;
    return num_rects ? reinterpret_cast<D3D12_RECT*>(args + 1) : nullptr;
  }

  void D3DClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_view,
                                D3D12_CLEAR_FLAGS clear_flags, FLOAT depth, UINT8 stencil,
                                UINT num_rects, const D3D12_RECT* rects) {
    D3D12_RECT* allocated_rects = ClearDepthStencilViewAllocatedRects(
        depth_stencil_view, clear_flags, depth, stencil, num_rects);
    if (num_rects) {
      assert_not_null(allocated_rects);
      std::memcpy(allocated_rects, rects, num_rects * sizeof(D3D12_RECT));
    }
  }

  void D3DClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE render_target_view,
                                const FLOAT color_rgba[4], UINT num_rects,
                                const D3D12_RECT* rects) {
    auto args = reinterpret_cast<ClearRenderTargetViewHeader*>(
        WriteCommand(Command::kD3DClearRenderTargetView,
                     sizeof(ClearRenderTargetViewHeader) + num_rects * sizeof(D3D12_RECT)));
    args->render_target_view = render_target_view;
    std::memcpy(args->color_rgba, color_rgba, 4 * sizeof(FLOAT));
    args->num_rects = num_rects;
    if (num_rects != 0) {
      std::memcpy(args + 1, rects, num_rects * sizeof(D3D12_RECT));
    }
  }

  void D3DClearUnorderedAccessViewUint(D3D12_GPU_DESCRIPTOR_HANDLE view_gpu_handle_in_current_heap,
                                       D3D12_CPU_DESCRIPTOR_HANDLE view_cpu_handle,
                                       ID3D12Resource* resource, const UINT values[4],
                                       UINT num_rects, const D3D12_RECT* rects) {
    auto args = reinterpret_cast<ClearUnorderedAccessViewHeader*>(
        WriteCommand(Command::kD3DClearUnorderedAccessViewUint,
                     sizeof(ClearUnorderedAccessViewHeader) + num_rects * sizeof(D3D12_RECT)));
    args->view_gpu_handle_in_current_heap = view_gpu_handle_in_current_heap;
    args->view_cpu_handle = view_cpu_handle;
    args->resource = resource;
    std::memcpy(args->values_uint, values, 4 * sizeof(UINT));
    args->num_rects = num_rects;
    if (num_rects != 0) {
      std::memcpy(args + 1, rects, num_rects * sizeof(D3D12_RECT));
    }
  }

  void D3DCopyBufferRegion(ID3D12Resource* dst_buffer, UINT64 dst_offset,
                           ID3D12Resource* src_buffer, UINT64 src_offset, UINT64 num_bytes) {
    auto& args = *reinterpret_cast<D3DCopyBufferRegionArguments*>(
        WriteCommand(Command::kD3DCopyBufferRegion, sizeof(D3DCopyBufferRegionArguments)));
    args.dst_buffer = dst_buffer;
    args.dst_offset = dst_offset;
    args.src_buffer = src_buffer;
    args.src_offset = src_offset;
    args.num_bytes = num_bytes;
  }

  void D3DCopyResource(ID3D12Resource* dst_resource, ID3D12Resource* src_resource) {
    auto& args = *reinterpret_cast<D3DCopyResourceArguments*>(
        WriteCommand(Command::kD3DCopyResource, sizeof(D3DCopyResourceArguments)));
    args.dst_resource = dst_resource;
    args.src_resource = src_resource;
  }

  void CopyTexture(const D3D12_TEXTURE_COPY_LOCATION& dst, const D3D12_TEXTURE_COPY_LOCATION& src) {
    auto& args = *reinterpret_cast<CopyTextureArguments*>(
        WriteCommand(Command::kCopyTexture, sizeof(CopyTextureArguments)));
    std::memcpy(&args.dst, &dst, sizeof(D3D12_TEXTURE_COPY_LOCATION));
    std::memcpy(&args.src, &src, sizeof(D3D12_TEXTURE_COPY_LOCATION));
  }

  void D3DCopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION* dst, UINT dst_x, UINT dst_y,
                            UINT dst_z, const D3D12_TEXTURE_COPY_LOCATION* src,
                            const D3D12_BOX* src_box) {
    assert_not_null(dst);
    assert_not_null(src);
    auto& args = *reinterpret_cast<D3DCopyTextureRegionArguments*>(
        WriteCommand(Command::kD3DCopyTextureRegion, sizeof(D3DCopyTextureRegionArguments)));
    std::memcpy(&args.dst, dst, sizeof(D3D12_TEXTURE_COPY_LOCATION));
    args.dst_x = dst_x;
    args.dst_y = dst_y;
    args.dst_z = dst_z;
    std::memcpy(&args.src, src, sizeof(D3D12_TEXTURE_COPY_LOCATION));
    if (src_box) {
      args.has_src_box = true;
      args.src_box = *src_box;
    } else {
      args.has_src_box = false;
    }
  }

  void D3DDispatch(UINT thread_group_count_x, UINT thread_group_count_y,
                   UINT thread_group_count_z) {
    auto& args = *reinterpret_cast<D3DDispatchArguments*>(
        WriteCommand(Command::kD3DDispatch, sizeof(D3DDispatchArguments)));
    args.thread_group_count_x = thread_group_count_x;
    args.thread_group_count_y = thread_group_count_y;
    args.thread_group_count_z = thread_group_count_z;
  }

  void D3DDrawIndexedInstanced(UINT index_count_per_instance, UINT instance_count,
                               UINT start_index_location, INT base_vertex_location,
                               UINT start_instance_location) {
    auto& args = *reinterpret_cast<D3DDrawIndexedInstancedArguments*>(
        WriteCommand(Command::kD3DDrawIndexedInstanced, sizeof(D3DDrawIndexedInstancedArguments)));
    args.index_count_per_instance = index_count_per_instance;
    args.instance_count = instance_count;
    args.start_index_location = start_index_location;
    args.base_vertex_location = base_vertex_location;
    args.start_instance_location = start_instance_location;
  }

  void D3DDrawInstanced(UINT vertex_count_per_instance, UINT instance_count,
                        UINT start_vertex_location, UINT start_instance_location) {
    auto& args = *reinterpret_cast<D3DDrawInstancedArguments*>(
        WriteCommand(Command::kD3DDrawInstanced, sizeof(D3DDrawInstancedArguments)));
    args.vertex_count_per_instance = vertex_count_per_instance;
    args.instance_count = instance_count;
    args.start_vertex_location = start_vertex_location;
    args.start_instance_location = start_instance_location;
  }

  void D3DBeginQuery(ID3D12QueryHeap* query_heap, D3D12_QUERY_TYPE type, UINT index) {
    auto& args = *reinterpret_cast<D3DQueryArguments*>(
        WriteCommand(Command::kD3DBeginQuery, sizeof(D3DQueryArguments)));
    args.query_heap = query_heap;
    args.type = type;
    args.index = index;
  }

  void D3DEndQuery(ID3D12QueryHeap* query_heap, D3D12_QUERY_TYPE type, UINT index) {
    auto& args = *reinterpret_cast<D3DQueryArguments*>(
        WriteCommand(Command::kD3DEndQuery, sizeof(D3DQueryArguments)));
    args.query_heap = query_heap;
    args.type = type;
    args.index = index;
  }

  void D3DResolveQueryData(ID3D12QueryHeap* query_heap, D3D12_QUERY_TYPE type, UINT start_index,
                           UINT num_queries, ID3D12Resource* destination_buffer,
                           UINT64 aligned_destination_buffer_offset) {
    auto& args = *reinterpret_cast<D3DResolveQueryDataArguments*>(
        WriteCommand(Command::kD3DResolveQueryData, sizeof(D3DResolveQueryDataArguments)));
    args.query_heap = query_heap;
    args.type = type;
    args.start_index = start_index;
    args.num_queries = num_queries;
    args.destination_buffer = destination_buffer;
    args.aligned_destination_buffer_offset = aligned_destination_buffer_offset;
  }

  void D3DIASetIndexBuffer(const D3D12_INDEX_BUFFER_VIEW* view) {
    auto& args = *reinterpret_cast<D3D12_INDEX_BUFFER_VIEW*>(
        WriteCommand(Command::kD3DIASetIndexBuffer, sizeof(D3D12_INDEX_BUFFER_VIEW)));
    if (view != nullptr) {
      args.BufferLocation = view->BufferLocation;
      args.SizeInBytes = view->SizeInBytes;
      args.Format = view->Format;
    } else {
      args.BufferLocation = D3D12_GPU_VIRTUAL_ADDRESS(0);
      args.SizeInBytes = 0;
      args.Format = DXGI_FORMAT_UNKNOWN;
    }
  }

  void D3DIASetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY primitive_topology) {
    auto& arg = *reinterpret_cast<D3D12_PRIMITIVE_TOPOLOGY*>(
        WriteCommand(Command::kD3DIASetPrimitiveTopology, sizeof(D3D12_PRIMITIVE_TOPOLOGY)));
    arg = primitive_topology;
  }

  void D3DIASetVertexBuffers(UINT start_slot, UINT num_views,
                             const D3D12_VERTEX_BUFFER_VIEW* views) {
    if (num_views == 0) {
      return;
    }
    static_assert(alignof(D3D12_VERTEX_BUFFER_VIEW) <= alignof(uintmax_t));
    const size_t header_size =
        ::rex::align(sizeof(D3DIASetVertexBuffersHeader), alignof(D3D12_VERTEX_BUFFER_VIEW));
    auto args = reinterpret_cast<D3DIASetVertexBuffersHeader*>(
        WriteCommand(Command::kD3DIASetVertexBuffers,
                     header_size + num_views * sizeof(D3D12_VERTEX_BUFFER_VIEW)));
    args->start_slot = start_slot;
    args->num_views = num_views;
    std::memcpy(reinterpret_cast<uint8_t*>(args) + header_size, views,
                sizeof(D3D12_VERTEX_BUFFER_VIEW) * num_views);
  }

  void D3DOMSetBlendFactor(const FLOAT blend_factor[4]) {
    auto args =
        reinterpret_cast<FLOAT*>(WriteCommand(Command::kD3DOMSetBlendFactor, 4 * sizeof(FLOAT)));
    args[0] = blend_factor[0];
    args[1] = blend_factor[1];
    args[2] = blend_factor[2];
    args[3] = blend_factor[3];
  }

  void D3DOMSetRenderTargets(UINT num_render_target_descriptors,
                             const D3D12_CPU_DESCRIPTOR_HANDLE* render_target_descriptors,
                             BOOL rts_single_handle_to_descriptor_range,
                             const D3D12_CPU_DESCRIPTOR_HANDLE* depth_stencil_descriptor) {
    auto& args = *reinterpret_cast<D3DOMSetRenderTargetsArguments*>(
        WriteCommand(Command::kD3DOMSetRenderTargets, sizeof(D3DOMSetRenderTargetsArguments)));
    num_render_target_descriptors =
        std::min(num_render_target_descriptors, UINT(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT));
    args.num_render_target_descriptors = num_render_target_descriptors;
    args.rts_single_handle_to_descriptor_range = rts_single_handle_to_descriptor_range ? 1 : 0;
    if (num_render_target_descriptors != 0) {
      std::memcpy(args.render_target_descriptors, render_target_descriptors,
                  (rts_single_handle_to_descriptor_range ? 1 : num_render_target_descriptors) *
                      sizeof(D3D12_CPU_DESCRIPTOR_HANDLE));
    }
    args.depth_stencil = (depth_stencil_descriptor != nullptr) ? 1 : 0;
    if (depth_stencil_descriptor != nullptr) {
      args.depth_stencil_descriptor.ptr = depth_stencil_descriptor->ptr;
    }
  }

  void D3DOMSetStencilRef(UINT stencil_ref) {
    auto& arg = *reinterpret_cast<UINT*>(WriteCommand(Command::kD3DOMSetStencilRef, sizeof(UINT)));
    arg = stencil_ref;
  }

  void D3DResourceBarrier(UINT num_barriers, const D3D12_RESOURCE_BARRIER* barriers) {
    if (num_barriers == 0) {
      return;
    }
    static_assert(alignof(D3D12_RESOURCE_BARRIER) <= alignof(uintmax_t));
    const size_t header_size = ::rex::align(sizeof(UINT), alignof(D3D12_RESOURCE_BARRIER));
    uint8_t* args = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kD3DResourceBarrier, header_size + num_barriers * sizeof(D3D12_RESOURCE_BARRIER)));
    *reinterpret_cast<UINT*>(args) = num_barriers;
    std::memcpy(args + header_size, barriers, num_barriers * sizeof(D3D12_RESOURCE_BARRIER));
  }

  void RSSetScissorRect(const D3D12_RECT& rect) {
    auto& arg = *reinterpret_cast<D3D12_RECT*>(
        WriteCommand(Command::kRSSetScissorRect, sizeof(D3D12_RECT)));
    arg = rect;
  }

  void RSSetViewport(const D3D12_VIEWPORT& viewport) {
    auto& arg = *reinterpret_cast<D3D12_VIEWPORT*>(
        WriteCommand(Command::kRSSetViewport, sizeof(D3D12_VIEWPORT)));
    arg = viewport;
  }

  void D3DSetComputeRoot32BitConstants(UINT root_parameter_index, UINT num_32bit_values_to_set,
                                       const void* src_data, UINT dest_offset_in_32bit_values) {
    if (num_32bit_values_to_set == 0) {
      return;
    }
    auto args = reinterpret_cast<SetRoot32BitConstantsHeader*>(WriteCommand(
        Command::kD3DSetComputeRoot32BitConstants,
        sizeof(SetRoot32BitConstantsHeader) + num_32bit_values_to_set * sizeof(uint32_t)));
    args->root_parameter_index = root_parameter_index;
    args->num_32bit_values_to_set = num_32bit_values_to_set;
    args->dest_offset_in_32bit_values = dest_offset_in_32bit_values;
    std::memcpy(args + 1, src_data, num_32bit_values_to_set * sizeof(uint32_t));
  }

  void D3DSetGraphicsRoot32BitConstants(UINT root_parameter_index, UINT num_32bit_values_to_set,
                                        const void* src_data, UINT dest_offset_in_32bit_values) {
    if (num_32bit_values_to_set == 0) {
      return;
    }
    auto args = reinterpret_cast<SetRoot32BitConstantsHeader*>(WriteCommand(
        Command::kD3DSetGraphicsRoot32BitConstants,
        sizeof(SetRoot32BitConstantsHeader) + num_32bit_values_to_set * sizeof(uint32_t)));
    args->root_parameter_index = root_parameter_index;
    args->num_32bit_values_to_set = num_32bit_values_to_set;
    args->dest_offset_in_32bit_values = dest_offset_in_32bit_values;
    std::memcpy(args + 1, src_data, num_32bit_values_to_set * sizeof(uint32_t));
  }

  void D3DSetComputeRootConstantBufferView(UINT root_parameter_index,
                                           D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(WriteCommand(
        Command::kD3DSetComputeRootConstantBufferView, sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetGraphicsRootConstantBufferView(UINT root_parameter_index,
                                            D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
        WriteCommand(Command::kD3DSetGraphicsRootConstantBufferView,
                     sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetComputeRootDescriptorTable(UINT root_parameter_index,
                                        D3D12_GPU_DESCRIPTOR_HANDLE base_descriptor) {
    auto& args = *reinterpret_cast<SetRootDescriptorTableArguments*>(WriteCommand(
        Command::kD3DSetComputeRootDescriptorTable, sizeof(SetRootDescriptorTableArguments)));
    args.root_parameter_index = root_parameter_index;
    args.base_descriptor.ptr = base_descriptor.ptr;
  }

  void D3DSetGraphicsRootDescriptorTable(UINT root_parameter_index,
                                         D3D12_GPU_DESCRIPTOR_HANDLE base_descriptor) {
    auto& args = *reinterpret_cast<SetRootDescriptorTableArguments*>(WriteCommand(
        Command::kD3DSetGraphicsRootDescriptorTable, sizeof(SetRootDescriptorTableArguments)));
    args.root_parameter_index = root_parameter_index;
    args.base_descriptor.ptr = base_descriptor.ptr;
  }

  void D3DSetComputeRootShaderResourceView(UINT root_parameter_index,
                                           D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(WriteCommand(
        Command::kD3DSetComputeRootShaderResourceView, sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetGraphicsRootShaderResourceView(UINT root_parameter_index,
                                            D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
        WriteCommand(Command::kD3DSetGraphicsRootShaderResourceView,
                     sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetComputeRootUnorderedAccessView(UINT root_parameter_index,
                                            D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
        WriteCommand(Command::kD3DSetComputeRootUnorderedAccessView,
                     sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetGraphicsRootUnorderedAccessView(UINT root_parameter_index,
                                             D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
        WriteCommand(Command::kD3DSetGraphicsRootUnorderedAccessView,
                     sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetComputeRootSignature(ID3D12RootSignature* root_signature) {
    auto& arg = *reinterpret_cast<ID3D12RootSignature**>(
        WriteCommand(Command::kD3DSetComputeRootSignature, sizeof(ID3D12RootSignature*)));
    arg = root_signature;
  }

  void D3DSetGraphicsRootSignature(ID3D12RootSignature* root_signature) {
    auto& arg = *reinterpret_cast<ID3D12RootSignature**>(
        WriteCommand(Command::kD3DSetGraphicsRootSignature, sizeof(ID3D12RootSignature*)));
    arg = root_signature;
  }

  void SetDescriptorHeaps(ID3D12DescriptorHeap* cbv_srv_uav_descriptor_heap,
                          ID3D12DescriptorHeap* sampler_descriptor_heap) {
    auto& args = *reinterpret_cast<SetDescriptorHeapsArguments*>(
        WriteCommand(Command::kSetDescriptorHeaps, sizeof(SetDescriptorHeapsArguments)));
    args.cbv_srv_uav_descriptor_heap = cbv_srv_uav_descriptor_heap;
    args.sampler_descriptor_heap = sampler_descriptor_heap;
  }

  void D3DSetPipelineState(ID3D12PipelineState* pipeline_state) {
    auto& arg = *reinterpret_cast<ID3D12PipelineState**>(
        WriteCommand(Command::kD3DSetPipelineState, sizeof(ID3D12PipelineState*)));
    arg = pipeline_state;
  }

  void SetPipelineStateHandle(void* pipeline_state_handle) {
    auto& arg =
        *reinterpret_cast<void**>(WriteCommand(Command::kSetPipelineStateHandle, sizeof(void*)));
    arg = pipeline_state_handle;
  }

  void D3DSetSamplePositions(UINT num_samples_per_pixel, UINT num_pixels,
                             const D3D12_SAMPLE_POSITION* sample_positions) {
    auto& args = *reinterpret_cast<D3DSetSamplePositionsArguments*>(
        WriteCommand(Command::kD3DSetSamplePositions, sizeof(D3DSetSamplePositionsArguments)));
    args.num_samples_per_pixel = num_samples_per_pixel;
    args.num_pixels = num_pixels;
    std::memcpy(
        args.sample_positions, sample_positions,
        std::min(num_samples_per_pixel * num_pixels, UINT(16)) * sizeof(D3D12_SAMPLE_POSITION));
  }

  void BeginDebugMarker(const char* label_name) {
    size_t label_len = std::strlen(label_name);
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kBeginDebugMarker, sizeof(DebugMarkerHeader) + label_len + 1));
    auto& args = *reinterpret_cast<DebugMarkerHeader*>(args_ptr);
    args.label_length = static_cast<uint32_t>(label_len);
    std::memcpy(args_ptr + sizeof(DebugMarkerHeader), label_name, label_len + 1);
  }

  void EndDebugMarker() { WriteCommand(Command::kEndDebugMarker, 0); }

  void InsertDebugMarker(const char* label_name) {
    size_t label_len = std::strlen(label_name);
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kInsertDebugMarker, sizeof(DebugMarkerHeader) + label_len + 1));
    auto& args = *reinterpret_cast<DebugMarkerHeader*>(args_ptr);
    args.label_length = static_cast<uint32_t>(label_len);
    std::memcpy(args_ptr + sizeof(DebugMarkerHeader), label_name, label_len + 1);
  }

 private:
  enum class Command {
    kD3DClearDepthStencilView,
    kD3DClearRenderTargetView,
    kD3DClearUnorderedAccessViewUint,
    kD3DCopyBufferRegion,
    kD3DCopyResource,
    kCopyTexture,
    kD3DCopyTextureRegion,
    kD3DDispatch,
    kD3DDrawIndexedInstanced,
    kD3DDrawInstanced,
    kD3DBeginQuery,
    kD3DEndQuery,
    kD3DResolveQueryData,
    kD3DIASetIndexBuffer,
    kD3DIASetPrimitiveTopology,
    kD3DIASetVertexBuffers,
    kD3DOMSetBlendFactor,
    kD3DOMSetRenderTargets,
    kD3DOMSetStencilRef,
    kD3DResourceBarrier,
    kRSSetScissorRect,
    kRSSetViewport,
    kD3DSetComputeRoot32BitConstants,
    kD3DSetGraphicsRoot32BitConstants,
    kD3DSetComputeRootConstantBufferView,
    kD3DSetGraphicsRootConstantBufferView,
    kD3DSetComputeRootDescriptorTable,
    kD3DSetGraphicsRootDescriptorTable,
    kD3DSetComputeRootShaderResourceView,
    kD3DSetGraphicsRootShaderResourceView,
    kD3DSetComputeRootSignature,
    kD3DSetGraphicsRootSignature,
    kD3DSetComputeRootUnorderedAccessView,
    kD3DSetGraphicsRootUnorderedAccessView,
    kSetDescriptorHeaps,
    kD3DSetPipelineState,
    kSetPipelineStateHandle,
    kD3DSetSamplePositions,
    kBeginDebugMarker,
    kEndDebugMarker,
    kInsertDebugMarker,
  };

  struct CommandHeader {
    Command command;
    uint32_t arguments_size_elements;
  };
  static constexpr size_t kCommandHeaderSizeElements =
      (sizeof(CommandHeader) + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);

  struct ClearDepthStencilViewHeader {
    D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_view;
    D3D12_CLEAR_FLAGS clear_flags;
    FLOAT depth;
    UINT8 stencil;
    UINT num_rects;
  };

  struct ClearRenderTargetViewHeader {
    D3D12_CPU_DESCRIPTOR_HANDLE render_target_view;
    FLOAT color_rgba[4];
    UINT num_rects;
  };

  struct ClearUnorderedAccessViewHeader {
    D3D12_GPU_DESCRIPTOR_HANDLE view_gpu_handle_in_current_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE view_cpu_handle;
    ID3D12Resource* resource;
    union {
      FLOAT values_float[4];
      UINT values_uint[4];
    };
    UINT num_rects;
  };

  struct D3DCopyBufferRegionArguments {
    ID3D12Resource* dst_buffer;
    UINT64 dst_offset;
    ID3D12Resource* src_buffer;
    UINT64 src_offset;
    UINT64 num_bytes;
  };

  struct D3DCopyResourceArguments {
    ID3D12Resource* dst_resource;
    ID3D12Resource* src_resource;
  };

  struct CopyTextureArguments {
    D3D12_TEXTURE_COPY_LOCATION dst;
    D3D12_TEXTURE_COPY_LOCATION src;
  };

  struct D3DCopyTextureRegionArguments {
    D3D12_TEXTURE_COPY_LOCATION dst;
    UINT dst_x;
    UINT dst_y;
    UINT dst_z;
    D3D12_TEXTURE_COPY_LOCATION src;
    D3D12_BOX src_box;
    bool has_src_box;
  };

  struct D3DDispatchArguments {
    UINT thread_group_count_x;
    UINT thread_group_count_y;
    UINT thread_group_count_z;
  };

  struct D3DDrawIndexedInstancedArguments {
    UINT index_count_per_instance;
    UINT instance_count;
    UINT start_index_location;
    INT base_vertex_location;
    UINT start_instance_location;
  };

  struct D3DDrawInstancedArguments {
    UINT vertex_count_per_instance;
    UINT instance_count;
    UINT start_vertex_location;
    UINT start_instance_location;
  };

  struct D3DQueryArguments {
    ID3D12QueryHeap* query_heap;
    D3D12_QUERY_TYPE type;
    UINT index;
  };

  struct D3DResolveQueryDataArguments {
    ID3D12QueryHeap* query_heap;
    D3D12_QUERY_TYPE type;
    UINT start_index;
    UINT num_queries;
    ID3D12Resource* destination_buffer;
    UINT64 aligned_destination_buffer_offset;
  };

  struct D3DIASetVertexBuffersHeader {
    UINT start_slot;
    UINT num_views;
  };

  struct D3DOMSetRenderTargetsArguments {
    uint8_t num_render_target_descriptors;
    bool rts_single_handle_to_descriptor_range;
    bool depth_stencil;
    D3D12_CPU_DESCRIPTOR_HANDLE
    render_target_descriptors[D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT];
    D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_descriptor;
  };

  struct SetRoot32BitConstantsHeader {
    UINT root_parameter_index;
    UINT num_32bit_values_to_set;
    UINT dest_offset_in_32bit_values;
  };

  struct SetRootConstantBufferViewArguments {
    UINT root_parameter_index;
    D3D12_GPU_VIRTUAL_ADDRESS buffer_location;
  };

  struct SetRootDescriptorTableArguments {
    UINT root_parameter_index;
    D3D12_GPU_DESCRIPTOR_HANDLE base_descriptor;
  };

  struct SetDescriptorHeapsArguments {
    ID3D12DescriptorHeap* cbv_srv_uav_descriptor_heap;
    ID3D12DescriptorHeap* sampler_descriptor_heap;
  };

  struct D3DSetSamplePositionsArguments {
    UINT num_samples_per_pixel;
    UINT num_pixels;
    D3D12_SAMPLE_POSITION sample_positions[16];
  };

  struct DebugMarkerHeader {
    uint32_t label_length;
    // Followed by null-terminated label string.
  };

  void* WriteCommand(Command command, size_t arguments_size_bytes);

  // [GPU-PRECORD] Core replay over an arbitrary stream range; Execute/ExecuteStream
  // both delegate here so segment replay and full-list replay share one code path.
  void ExecuteRange(const uintmax_t* stream_data, size_t stream_size,
                    ID3D12GraphicsCommandList* command_list,
                    ID3D12GraphicsCommandList1* command_list_1);

  const D3D12CommandProcessor& command_processor_;

  // uintmax_t to ensure uint64_t and pointer alignment of all structures.
  std::vector<uintmax_t> command_stream_;
  // [NR-BFC] see reset_generation().
  uint64_t reset_generation_ = 0;
};

}  // namespace rex::graphics::d3d12
