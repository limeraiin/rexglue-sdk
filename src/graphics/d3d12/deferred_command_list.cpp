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

#include <rex/assert.h>
#include <rex/dbg.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/graphics/d3d12/deferred_command_list.h>
#include <rex/graphics/flags.h>
#include <rex/math.h>

namespace rex::graphics::d3d12 {

DeferredCommandList::DeferredCommandList(const D3D12CommandProcessor& command_processor,
                                         size_t initial_size)
    : command_processor_(command_processor) {
  command_stream_.reserve(initial_size / sizeof(uintmax_t));
}

void DeferredCommandList::Reset() {
  command_stream_.clear();
  // [NR-BFC] invalidate every outstanding span anchor.
  ++reset_generation_;
}

void DeferredCommandList::Execute(ID3D12GraphicsCommandList* command_list,
                                  ID3D12GraphicsCommandList1* command_list_1) {
  ExecuteRange(command_stream_.data(), command_stream_.size(), command_list, command_list_1);
}

void DeferredCommandList::ExecuteRange(const uintmax_t* stream_data, size_t stream_size,
                                       ID3D12GraphicsCommandList* command_list,
                                       ID3D12GraphicsCommandList1* command_list_1) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES
  const uintmax_t* stream = stream_data;
  size_t stream_remaining = stream_size;
  ID3D12PipelineState* current_pipeline_state = nullptr;
  while (stream_remaining != 0) {
    const CommandHeader& header = *reinterpret_cast<const CommandHeader*>(stream);
    stream += kCommandHeaderSizeElements;
    stream_remaining -= kCommandHeaderSizeElements;
    switch (header.command) {
      case Command::kD3DClearDepthStencilView: {
        auto& args = *reinterpret_cast<const ClearDepthStencilViewHeader*>(stream);
        command_list->ClearDepthStencilView(
            args.depth_stencil_view, args.clear_flags, args.depth, args.stencil, args.num_rects,
            args.num_rects ? reinterpret_cast<const D3D12_RECT*>(&args + 1) : nullptr);
      } break;
      case Command::kD3DClearRenderTargetView: {
        auto& args = *reinterpret_cast<const ClearRenderTargetViewHeader*>(stream);
        command_list->ClearRenderTargetView(
            args.render_target_view, args.color_rgba, args.num_rects,
            args.num_rects ? reinterpret_cast<const D3D12_RECT*>(&args + 1) : nullptr);
      } break;
      case Command::kD3DClearUnorderedAccessViewUint: {
        auto& args = *reinterpret_cast<const ClearUnorderedAccessViewHeader*>(stream);
        command_list->ClearUnorderedAccessViewUint(
            args.view_gpu_handle_in_current_heap, args.view_cpu_handle, args.resource,
            args.values_uint, args.num_rects,
            args.num_rects ? reinterpret_cast<const D3D12_RECT*>(&args + 1) : nullptr);
      } break;
      case Command::kD3DCopyBufferRegion: {
        auto& args = *reinterpret_cast<const D3DCopyBufferRegionArguments*>(stream);
        command_list->CopyBufferRegion(args.dst_buffer, args.dst_offset, args.src_buffer,
                                       args.src_offset, args.num_bytes);
      } break;
      case Command::kD3DCopyResource: {
        auto& args = *reinterpret_cast<const D3DCopyResourceArguments*>(stream);
        command_list->CopyResource(args.dst_resource, args.src_resource);
      } break;
      case Command::kCopyTexture: {
        auto& args = *reinterpret_cast<const CopyTextureArguments*>(stream);
        command_list->CopyTextureRegion(&args.dst, 0, 0, 0, &args.src, nullptr);
      } break;
      case Command::kD3DCopyTextureRegion: {
        auto& args = *reinterpret_cast<const D3DCopyTextureRegionArguments*>(stream);
        command_list->CopyTextureRegion(&args.dst, args.dst_x, args.dst_y, args.dst_z, &args.src,
                                        args.has_src_box ? &args.src_box : nullptr);
      } break;
      case Command::kD3DDispatch: {
        if (current_pipeline_state != nullptr) {
          auto& args = *reinterpret_cast<const D3DDispatchArguments*>(stream);
          command_list->Dispatch(args.thread_group_count_x, args.thread_group_count_y,
                                 args.thread_group_count_z);
        }
      } break;
      case Command::kD3DDrawIndexedInstanced: {
        if (current_pipeline_state != nullptr) {
          auto& args = *reinterpret_cast<const D3DDrawIndexedInstancedArguments*>(stream);
          command_list->DrawIndexedInstanced(args.index_count_per_instance, args.instance_count,
                                             args.start_index_location, args.base_vertex_location,
                                             args.start_instance_location);
        }
      } break;
      case Command::kD3DDrawInstanced: {
        if (current_pipeline_state != nullptr) {
          auto& args = *reinterpret_cast<const D3DDrawInstancedArguments*>(stream);
          command_list->DrawInstanced(args.vertex_count_per_instance, args.instance_count,
                                      args.start_vertex_location, args.start_instance_location);
        }
      } break;
      case Command::kD3DBeginQuery: {
        auto& args = *reinterpret_cast<const D3DQueryArguments*>(stream);
        command_list->BeginQuery(args.query_heap, args.type, args.index);
      } break;
      case Command::kD3DEndQuery: {
        auto& args = *reinterpret_cast<const D3DQueryArguments*>(stream);
        command_list->EndQuery(args.query_heap, args.type, args.index);
      } break;
      case Command::kD3DResolveQueryData: {
        auto& args = *reinterpret_cast<const D3DResolveQueryDataArguments*>(stream);
        command_list->ResolveQueryData(args.query_heap, args.type, args.start_index,
                                       args.num_queries, args.destination_buffer,
                                       args.aligned_destination_buffer_offset);
      } break;
      case Command::kD3DIASetIndexBuffer: {
        auto view = reinterpret_cast<const D3D12_INDEX_BUFFER_VIEW*>(stream);
        command_list->IASetIndexBuffer(view->Format != DXGI_FORMAT_UNKNOWN ? view : nullptr);
      } break;
      case Command::kD3DIASetPrimitiveTopology: {
        command_list->IASetPrimitiveTopology(
            *reinterpret_cast<const D3D12_PRIMITIVE_TOPOLOGY*>(stream));
      } break;
      case Command::kD3DIASetVertexBuffers: {
        static_assert(alignof(D3D12_VERTEX_BUFFER_VIEW) <= alignof(uintmax_t));
        auto& args = *reinterpret_cast<const D3DIASetVertexBuffersHeader*>(stream);
        command_list->IASetVertexBuffers(args.start_slot, args.num_views,
                                         reinterpret_cast<const D3D12_VERTEX_BUFFER_VIEW*>(
                                             reinterpret_cast<const uint8_t*>(stream) +
                                             rex::align(sizeof(D3DIASetVertexBuffersHeader),
                                                        alignof(D3D12_VERTEX_BUFFER_VIEW))));
      } break;
      case Command::kD3DOMSetBlendFactor: {
        command_list->OMSetBlendFactor(reinterpret_cast<const FLOAT*>(stream));
      } break;
      case Command::kD3DOMSetRenderTargets: {
        auto& args = *reinterpret_cast<const D3DOMSetRenderTargetsArguments*>(stream);
        command_list->OMSetRenderTargets(
            args.num_render_target_descriptors, args.render_target_descriptors,
            args.rts_single_handle_to_descriptor_range ? TRUE : FALSE,
            args.depth_stencil ? &args.depth_stencil_descriptor : nullptr);
      } break;
      case Command::kD3DOMSetStencilRef: {
        command_list->OMSetStencilRef(*reinterpret_cast<const UINT*>(stream));
      } break;
      case Command::kD3DResourceBarrier: {
        static_assert(alignof(D3D12_RESOURCE_BARRIER) <= alignof(uintmax_t));
        command_list->ResourceBarrier(
            *reinterpret_cast<const UINT*>(stream),
            reinterpret_cast<const D3D12_RESOURCE_BARRIER*>(
                reinterpret_cast<const uint8_t*>(stream) +
                rex::align(sizeof(UINT), alignof(D3D12_RESOURCE_BARRIER))));
      } break;
      case Command::kRSSetScissorRect: {
        command_list->RSSetScissorRects(1, reinterpret_cast<const D3D12_RECT*>(stream));
      } break;
      case Command::kRSSetViewport: {
        command_list->RSSetViewports(1, reinterpret_cast<const D3D12_VIEWPORT*>(stream));
      } break;
      case Command::kD3DSetComputeRoot32BitConstants: {
        auto args = reinterpret_cast<const SetRoot32BitConstantsHeader*>(stream);
        command_list->SetComputeRoot32BitConstants(args->root_parameter_index,
                                                   args->num_32bit_values_to_set, args + 1,
                                                   args->dest_offset_in_32bit_values);
      } break;
      case Command::kD3DSetGraphicsRoot32BitConstants: {
        auto args = reinterpret_cast<const SetRoot32BitConstantsHeader*>(stream);
        command_list->SetGraphicsRoot32BitConstants(args->root_parameter_index,
                                                    args->num_32bit_values_to_set, args + 1,
                                                    args->dest_offset_in_32bit_values);
      } break;
      case Command::kD3DSetComputeRootConstantBufferView: {
        auto& args = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(stream);
        command_list->SetComputeRootConstantBufferView(args.root_parameter_index,
                                                       args.buffer_location);
      } break;
      case Command::kD3DSetGraphicsRootConstantBufferView: {
        auto& args = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(stream);
        command_list->SetGraphicsRootConstantBufferView(args.root_parameter_index,
                                                        args.buffer_location);
      } break;
      case Command::kD3DSetComputeRootDescriptorTable: {
        auto& args = *reinterpret_cast<const SetRootDescriptorTableArguments*>(stream);
        command_list->SetComputeRootDescriptorTable(args.root_parameter_index,
                                                    args.base_descriptor);
      } break;
      case Command::kD3DSetGraphicsRootDescriptorTable: {
        auto& args = *reinterpret_cast<const SetRootDescriptorTableArguments*>(stream);
        command_list->SetGraphicsRootDescriptorTable(args.root_parameter_index,
                                                     args.base_descriptor);
      } break;
      case Command::kD3DSetComputeRootShaderResourceView: {
        auto& args = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(stream);
        command_list->SetComputeRootShaderResourceView(args.root_parameter_index,
                                                       args.buffer_location);
      } break;
      case Command::kD3DSetGraphicsRootShaderResourceView: {
        auto& args = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(stream);
        command_list->SetGraphicsRootShaderResourceView(args.root_parameter_index,
                                                        args.buffer_location);
      } break;
      case Command::kD3DSetComputeRootSignature: {
        command_list->SetComputeRootSignature(
            *reinterpret_cast<ID3D12RootSignature* const*>(stream));
      } break;
      case Command::kD3DSetGraphicsRootSignature: {
        command_list->SetGraphicsRootSignature(
            *reinterpret_cast<ID3D12RootSignature* const*>(stream));
      } break;
      case Command::kD3DSetComputeRootUnorderedAccessView: {
        auto& args = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(stream);
        command_list->SetComputeRootUnorderedAccessView(args.root_parameter_index,
                                                        args.buffer_location);
      } break;
      case Command::kD3DSetGraphicsRootUnorderedAccessView: {
        auto& args = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(stream);
        command_list->SetGraphicsRootUnorderedAccessView(args.root_parameter_index,
                                                         args.buffer_location);
      } break;
      case Command::kSetDescriptorHeaps: {
        auto& args = *reinterpret_cast<const SetDescriptorHeapsArguments*>(stream);
        UINT num_descriptor_heaps = 0;
        ID3D12DescriptorHeap* descriptor_heaps[2];
        if (args.cbv_srv_uav_descriptor_heap != nullptr) {
          descriptor_heaps[num_descriptor_heaps++] = args.cbv_srv_uav_descriptor_heap;
        }
        if (args.sampler_descriptor_heap != nullptr) {
          descriptor_heaps[num_descriptor_heaps++] = args.sampler_descriptor_heap;
        }
        command_list->SetDescriptorHeaps(num_descriptor_heaps, descriptor_heaps);
      } break;
      case Command::kD3DSetPipelineState: {
        current_pipeline_state = *reinterpret_cast<ID3D12PipelineState* const*>(stream);
        if (current_pipeline_state) {
          command_list->SetPipelineState(current_pipeline_state);
        }
      } break;
      case Command::kSetPipelineStateHandle: {
        current_pipeline_state =
            command_processor_.GetD3D12PipelineByHandle(*reinterpret_cast<void* const*>(stream));
        if (current_pipeline_state) {
          command_list->SetPipelineState(current_pipeline_state);
        }
      } break;
      case Command::kD3DSetSamplePositions: {
        if (command_list_1 != nullptr) {
          auto& args = *reinterpret_cast<const D3DSetSamplePositionsArguments*>(stream);
          command_list_1->SetSamplePositions(
              args.num_samples_per_pixel, args.num_pixels,
              (args.num_samples_per_pixel && args.num_pixels)
                  ? const_cast<D3D12_SAMPLE_POSITION*>(args.sample_positions)
                  : nullptr);
        }
      } break;
      case Command::kBeginDebugMarker: {
        auto& args = *reinterpret_cast<const DebugMarkerHeader*>(stream);
        const char* label_name = reinterpret_cast<const char*>(
            reinterpret_cast<const uint8_t*>(stream) + sizeof(DebugMarkerHeader));
        command_list->BeginEvent(1, label_name, static_cast<UINT>(args.label_length + 1));
      } break;
      case Command::kEndDebugMarker: {
        command_list->EndEvent();
      } break;
      case Command::kInsertDebugMarker: {
        auto& args = *reinterpret_cast<const DebugMarkerHeader*>(stream);
        const char* label_name = reinterpret_cast<const char*>(
            reinterpret_cast<const uint8_t*>(stream) + sizeof(DebugMarkerHeader));
        command_list->SetMarker(1, label_name, static_cast<UINT>(args.label_length + 1));
      } break;
      default:
        assert_unhandled_case(header.command);
        break;
    }
    stream += header.arguments_size_elements;
    stream_remaining -= header.arguments_size_elements;
  }
}

// [NR-BFC] Phase 5-4-6-0: walk the self-describing stream tail and bucket
// each command for the buffer-replay census. Mirrors ExecuteRange's header
// walk exactly; no command is executed.
void DeferredCommandList::NrBfcScan(size_t start_elements, NrBfcSpanCounts* out) const {
  const uintmax_t* stream = command_stream_.data() + start_elements;
  size_t remaining = command_stream_.size() >= start_elements
                         ? command_stream_.size() - start_elements
                         : 0;
  while (remaining >= kCommandHeaderSizeElements) {
    const CommandHeader& header = *reinterpret_cast<const CommandHeader*>(stream);
    // A malformed tail (stale anchor) must terminate the scan, never
    // underflow `remaining` -- this is a diagnostic, not an executor.
    if (header.arguments_size_elements >
        remaining - kCommandHeaderSizeElements) {
      break;
    }
    stream += kCommandHeaderSizeElements;
    remaining -= kCommandHeaderSizeElements;
    switch (header.command) {
      case Command::kD3DDrawIndexedInstanced:
      case Command::kD3DDrawInstanced:
        ++out->draw;
        break;
      case Command::kD3DSetPipelineState:
      case Command::kSetPipelineStateHandle:
        ++out->pso;
        break;
      case Command::kD3DSetGraphicsRootConstantBufferView: {
        ++out->root_cbv;
        auto& args = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(stream);
        ++out->graphics_cbv_by_root[args.root_parameter_index < 16
                                        ? args.root_parameter_index
                                        : 15];
      } break;
      case Command::kD3DSetComputeRootConstantBufferView:
        ++out->root_cbv;
        break;
      case Command::kD3DSetComputeRoot32BitConstants:
      case Command::kD3DSetGraphicsRoot32BitConstants:
      case Command::kD3DSetComputeRootDescriptorTable:
      case Command::kD3DSetGraphicsRootDescriptorTable:
      case Command::kD3DSetComputeRootShaderResourceView:
      case Command::kD3DSetGraphicsRootShaderResourceView:
      case Command::kD3DSetComputeRootUnorderedAccessView:
      case Command::kD3DSetGraphicsRootUnorderedAccessView:
      case Command::kD3DSetComputeRootSignature:
      case Command::kD3DSetGraphicsRootSignature:
        ++out->root_other;
        break;
      case Command::kD3DIASetIndexBuffer:
      case Command::kD3DIASetPrimitiveTopology:
      case Command::kD3DIASetVertexBuffers:
        ++out->ia;
        break;
      case Command::kRSSetViewport:
        ++out->vp;
        break;
      case Command::kRSSetScissorRect:
        ++out->sci;
        break;
      case Command::kD3DOMSetRenderTargets:
        ++out->om_rt;
        break;
      case Command::kD3DOMSetBlendFactor:
      case Command::kD3DOMSetStencilRef:
      case Command::kD3DSetSamplePositions:
        ++out->om_misc;
        break;
      case Command::kD3DResourceBarrier:
        ++out->barrier;
        break;
      case Command::kD3DCopyBufferRegion:
      case Command::kD3DCopyResource:
      case Command::kCopyTexture:
      case Command::kD3DCopyTextureRegion:
        ++out->copy;
        break;
      case Command::kD3DClearDepthStencilView:
      case Command::kD3DClearRenderTargetView:
      case Command::kD3DClearUnorderedAccessViewUint:
        ++out->clear;
        break;
      case Command::kD3DDispatch:
        ++out->dispatch;
        break;
      case Command::kD3DBeginQuery:
      case Command::kD3DEndQuery:
      case Command::kD3DResolveQueryData:
        ++out->query;
        break;
      case Command::kBeginDebugMarker:
      case Command::kEndDebugMarker:
      case Command::kInsertDebugMarker:
        ++out->marker;
        break;
      case Command::kSetDescriptorHeaps:
        ++out->heaps;
        break;
      default:
        ++out->other;
        break;
    }
    stream += header.arguments_size_elements;
    remaining -= header.arguments_size_elements;
  }
}

// [NR-SPR] Phase 5-4-7-1: whitelist scan + patch-site offsets. Mirrors the
// census walks' header discipline (malformed tail terminates, never
// underflows) because the caller's anchor validity is reset-generation-based
// and this is a store gate, not an executor.
void DeferredCommandList::NrSprScanSpan(size_t start_elements, NrSprScan* out) const {
  const uintmax_t* stream = command_stream_.data() + start_elements;
  size_t remaining = command_stream_.size() >= start_elements
                         ? command_stream_.size() - start_elements
                         : 0;
  size_t offset = 0;
  while (remaining >= kCommandHeaderSizeElements) {
    const CommandHeader& header = *reinterpret_cast<const CommandHeader*>(stream);
    if (header.arguments_size_elements > remaining - kCommandHeaderSizeElements) {
      out->malformed = true;
      return;
    }
    ++out->cmds;
    switch (header.command) {
      case Command::kD3DDrawIndexedInstanced:
      case Command::kD3DDrawInstanced:
        ++out->draw;
        break;
      case Command::kD3DSetPipelineState:
      case Command::kSetPipelineStateHandle:
      case Command::kD3DSetGraphicsRootSignature:
      case Command::kD3DIASetPrimitiveTopology:
      case Command::kD3DIASetIndexBuffer:
        break;
      case Command::kD3DSetGraphicsRootConstantBufferView:
      case Command::kD3DSetGraphicsRootShaderResourceView:
      case Command::kD3DSetGraphicsRootUnorderedAccessView: {
        ++out->view_sites;
        if (out->view_offset_count < kNrSprMaxViewSites && offset <= 0xFFFF) {
          out->view_offsets[out->view_offset_count++] = uint16_t(offset);
        }
        // [NR-SPD] all three share SetRootConstantBufferViewArguments; the
        // root index is the first UINT.
        const UINT nr_spd_root =
            reinterpret_cast<const SetRootConstantBufferViewArguments*>(
                stream + kCommandHeaderSizeElements)
                ->root_parameter_index;
        if (nr_spd_root < 32) {
          out->root_mask |= 1u << nr_spd_root;
        } else {
          ++out->other;
        }
        break;
      }
      case Command::kD3DSetGraphicsRootDescriptorTable: {
        ++out->table_sites;
        const UINT nr_spd_root =
            reinterpret_cast<const SetRootDescriptorTableArguments*>(
                stream + kCommandHeaderSizeElements)
                ->root_parameter_index;
        if (nr_spd_root < 32) {
          out->root_mask |= 1u << nr_spd_root;
        } else {
          ++out->other;
        }
        break;
      }
      case Command::kRSSetViewport:
      case Command::kRSSetScissorRect:
      case Command::kD3DOMSetBlendFactor:
      case Command::kD3DOMSetStencilRef:
      case Command::kD3DSetSamplePositions:
        ++out->ff;
        break;
      case Command::kD3DResourceBarrier:
        ++out->barrier;
        break;
      case Command::kD3DSetComputeRootSignature:
      case Command::kD3DSetComputeRootConstantBufferView:
      case Command::kD3DSetComputeRootShaderResourceView:
      case Command::kD3DSetComputeRootUnorderedAccessView:
      case Command::kD3DSetComputeRootDescriptorTable:
      case Command::kD3DSetComputeRoot32BitConstants:
      case Command::kD3DCopyBufferRegion:
      case Command::kD3DCopyResource:
      case Command::kCopyTexture:
      case Command::kD3DCopyTextureRegion:
      case Command::kD3DClearDepthStencilView:
      case Command::kD3DClearRenderTargetView:
      case Command::kD3DClearUnorderedAccessViewUint:
      case Command::kD3DDispatch:
      case Command::kD3DBeginQuery:
      case Command::kD3DEndQuery:
      case Command::kD3DResolveQueryData:
        ++out->compute;
        break;
      case Command::kSetDescriptorHeaps:
        ++out->heaps;
        break;
      default:
        ++out->other;
        break;
    }
    const size_t step = kCommandHeaderSizeElements + header.arguments_size_elements;
    stream += step;
    remaining -= step;
    offset += step;
  }
}

bool DeferredCommandList::NrSprViewSiteRoots(const uintmax_t* span, const uint16_t* offsets,
                                             uint32_t count, uint32_t* roots_out) {
  for (uint32_t i = 0; i < count; ++i) {
    const uintmax_t* cmd = span + offsets[i];
    const CommandHeader& hdr = *reinterpret_cast<const CommandHeader*>(cmd);
    if (hdr.command != Command::kD3DSetGraphicsRootConstantBufferView) {
      return false;
    }
    const auto& args = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(
        cmd + kCommandHeaderSizeElements);
    roots_out[i] = args.root_parameter_index;
  }
  return true;
}

void DeferredCommandList::NrSprPatchViewAddress(uintmax_t* span, uint32_t offset,
                                                uint64_t gpu_address) {
  auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
      span + offset + kCommandHeaderSizeElements);
  args.buffer_location = D3D12_GPU_VIRTUAL_ADDRESS(gpu_address);
}

size_t DeferredCommandList::NrTilePipelineSpan(size_t start_elements, size_t end_elements,
                                               uintmax_t* dst, size_t capacity) const {
  if (end_elements < start_elements || end_elements > command_stream_.size()) {
    return SIZE_MAX;
  }
  const size_t len = end_elements - start_elements;
  if (!len) return 0;
  if (len > capacity || len < kCommandHeaderSizeElements) return SIZE_MAX;
  const uintmax_t* stream = command_stream_.data() + start_elements;
  const CommandHeader& header = *reinterpret_cast<const CommandHeader*>(stream);
  if (header.command != Command::kD3DSetPipelineState &&
      header.command != Command::kSetPipelineStateHandle) {
    return SIZE_MAX;
  }
  // Exactly one command: anything trailing means something else emitted
  // between the texture requests and the fixed-function state, which the
  // replay would silently drop.
  if (kCommandHeaderSizeElements + header.arguments_size_elements != len) {
    return SIZE_MAX;
  }
  std::memcpy(dst, stream, len * sizeof(uintmax_t));
  return len;
}

size_t DeferredCommandList::NrDspCopySpan(size_t start_elements, uintmax_t* dst,
                                          size_t capacity) const {
  if (command_stream_.size() < start_elements) return 0;
  const size_t len = command_stream_.size() - start_elements;
  if (!len || len > capacity) return 0;
  std::memcpy(dst, command_stream_.data() + start_elements, len * sizeof(uintmax_t));
  return len;
}

void DeferredCommandList::NrDspCompareSpan(const uintmax_t* prev, size_t prev_len,
                                           size_t start_elements, NrDspDiff* out,
                                           bool skip_fresh_barriers,
                                           ID3D12Resource* fresh_upload_dst) const {
  // [NR-DSP] Lockstep walk of the stored span against the freshly emitted
  // one. A command whose bytes differ is charged to `real` unless the ONLY
  // differing field is one a replay would patch anyway.
  const uintmax_t* a = prev;
  size_t a_rem = prev_len;
  const uintmax_t* b = command_stream_.data() + start_elements;
  size_t b_rem = command_stream_.size() >= start_elements
                     ? command_stream_.size() - start_elements
                     : 0;
  // Without the barrier skip the two lengths must match up front; with it
  // they are allowed to differ by exactly the skipped barriers, so the test
  // moves to after the walk.
  if (!skip_fresh_barriers && a_rem != b_rem) out->length_differs = true;
  while (b_rem >= kCommandHeaderSizeElements) {
    const CommandHeader& hb = *reinterpret_cast<const CommandHeader*>(b);
    const size_t nb = hb.arguments_size_elements;
    if (nb > b_rem - kCommandHeaderSizeElements) {
      out->length_differs = true;
      return;
    }
    // [NR-TIL] N-4-1: work the fresh execution emitted inside the span that
    // the recording cannot carry (the record-time whitelist refuses both
    // classes) and that the replay emits in its HEAD, just before the span:
    // a resource barrier (order-independent against the state sets it sits
    // among, ordered only against the draw) and a shared-memory residency
    // upload (`RequestRange` + `UseForReading`, and only when the copy's
    // destination proves it is one).
    if (skip_fresh_barriers &&
        !(a_rem >= kCommandHeaderSizeElements &&
          reinterpret_cast<const CommandHeader*>(a)->command == hb.command)) {
      bool fresh_only = false;
      if (hb.command == Command::kD3DResourceBarrier) {
        fresh_only = true;
        ++out->fresh_barriers;
      } else if (fresh_upload_dst && hb.command == Command::kD3DCopyBufferRegion &&
                 nb * sizeof(uintmax_t) >= sizeof(D3DCopyBufferRegionArguments) &&
                 reinterpret_cast<const D3DCopyBufferRegionArguments*>(
                     b + kCommandHeaderSizeElements)
                         ->dst_buffer == fresh_upload_dst) {
        fresh_only = true;
        ++out->fresh_uploads;
      }
      if (fresh_only) {
        b += kCommandHeaderSizeElements + nb;
        b_rem -= kCommandHeaderSizeElements + nb;
        continue;
      }
    }
    if (a_rem < kCommandHeaderSizeElements) break;
    const CommandHeader& ha = *reinterpret_cast<const CommandHeader*>(a);
    const size_t na = ha.arguments_size_elements;
    if (na > a_rem - kCommandHeaderSizeElements) {
      out->length_differs = true;
      return;
    }
    ++out->cmds;
    const uintmax_t* pa = a + kCommandHeaderSizeElements;
    const uintmax_t* pb = b + kCommandHeaderSizeElements;
    const bool is_view = hb.command == Command::kD3DSetGraphicsRootConstantBufferView ||
                         hb.command == Command::kD3DSetComputeRootConstantBufferView ||
                         hb.command == Command::kD3DSetGraphicsRootShaderResourceView ||
                         hb.command == Command::kD3DSetComputeRootShaderResourceView ||
                         hb.command == Command::kD3DSetGraphicsRootUnorderedAccessView ||
                         hb.command == Command::kD3DSetComputeRootUnorderedAccessView;
    if (is_view) ++out->view_sites;
    if (ha.command != hb.command || na != nb) {
      ++out->real;
      if (out->first_real == 0xFFFFFFFFu) out->first_real = uint32_t(hb.command);
    } else if (na && std::memcmp(pa, pb, na * sizeof(uintmax_t)) != 0) {
      bool dynamic_only = false;
      if (is_view) {
        auto& va = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(pa);
        auto& vb = *reinterpret_cast<const SetRootConstantBufferViewArguments*>(pb);
        // Same slot, only the address moved: exactly the patch a replay does.
        dynamic_only = va.root_parameter_index == vb.root_parameter_index;
        if (dynamic_only) ++out->dyn_view;
      } else if (hb.command == Command::kD3DSetGraphicsRootDescriptorTable ||
                 hb.command == Command::kD3DSetComputeRootDescriptorTable) {
        auto& ta = *reinterpret_cast<const SetRootDescriptorTableArguments*>(pa);
        auto& tb = *reinterpret_cast<const SetRootDescriptorTableArguments*>(pb);
        dynamic_only = ta.root_parameter_index == tb.root_parameter_index;
        if (dynamic_only) ++out->dyn_table;
      }
      if (!dynamic_only) {
        ++out->real;
        if (out->first_real == 0xFFFFFFFFu) out->first_real = uint32_t(hb.command);
      }
    }
    a += kCommandHeaderSizeElements + na;
    a_rem -= kCommandHeaderSizeElements + na;
    b += kCommandHeaderSizeElements + nb;
    b_rem -= kCommandHeaderSizeElements + nb;
  }
  // Anything left on either side after the walk is a real length difference
  // (trailing fresh barriers were consumed by the skip above).
  if (skip_fresh_barriers && (a_rem || b_rem)) out->length_differs = true;
}

void* DeferredCommandList::WriteCommand(Command command, size_t arguments_size_bytes) {
  size_t arguments_size_elements =
      (arguments_size_bytes + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);
  size_t offset = command_stream_.size();
  command_stream_.resize(offset + kCommandHeaderSizeElements + arguments_size_elements);
  CommandHeader& header = *reinterpret_cast<CommandHeader*>(command_stream_.data() + offset);
  header.command = command;
  header.arguments_size_elements = uint32_t(arguments_size_elements);
  return command_stream_.data() + (offset + kCommandHeaderSizeElements);
}

}  // namespace rex::graphics::d3d12
