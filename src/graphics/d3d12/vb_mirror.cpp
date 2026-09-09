/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <rex/graphics/d3d12/vb_mirror.h>

#include <algorithm>
#include <cstring>

#include <rex/graphics/d3d12/command_processor.h>
#include <rex/graphics/d3d12/shared_memory.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/ui/d3d12/d3d12_util.h>

namespace rex::graphics::d3d12 {

namespace shaders {
// fxc, see src/graphics/shaders/vbm_fill.cs.hlsl for the build line.
#include "../shaders/bytecode/d3d12_5_1/vbm_fill_cs.h"
}  // namespace shaders

VbMirror::VbMirror(D3D12CommandProcessor& command_processor, D3D12SharedMemory& shared_memory)
    : command_processor_(command_processor), shared_memory_(shared_memory) {}

VbMirror::~VbMirror() { Shutdown(); }

bool VbMirror::Initialize(uint32_t budget_mb) {
  Shutdown();
  budget_bytes_ = uint64_t(std::max<uint32_t>(budget_mb, 64)) << 20;
  page_size_log2_ = shared_memory_.GetPageSizeLog2();
  page_count_ = SharedMemory::kBufferSize >> page_size_log2_;
  valid_words_ = (page_count_ + 63) / 64;

  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();

  // b0 = 4 constants, t0 = the shared memory (root SRV), u0 = the shadow (root UAV).
  D3D12_ROOT_PARAMETER params[3] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[0].Constants.ShaderRegister = 0;
  params[0].Constants.RegisterSpace = 0;
  params[0].Constants.Num32BitValues = 4;
  params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
  params[1].Descriptor.ShaderRegister = 0;
  params[1].Descriptor.RegisterSpace = 0;
  params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  params[2].Descriptor.ShaderRegister = 0;
  params[2].Descriptor.RegisterSpace = 0;
  params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC rs_desc = {};
  rs_desc.NumParameters = 3;
  rs_desc.pParameters = params;
  rs_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(fill_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, rs_desc);
  if (!fill_root_signature_) {
    REXGPU_WARN("[ia] shadow: fill root signature creation failed - off");
    return false;
  }
  *(fill_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::vbm_fill_cs, sizeof(shaders::vbm_fill_cs), fill_root_signature_.Get());
  if (!fill_pipeline_) {
    REXGPU_WARN("[ia] shadow: fill pipeline creation failed - off");
    fill_root_signature_.Reset();
    return false;
  }
  // Every shadow up front: the global watch (another thread) reads their
  // validity bitmaps, so none may appear later. A failure here = tiled
  // resources unavailable = the raw path.
  if (!EnsureShadow(1) || !EnsureShadow(2) || !EnsureShadow(3)) {
    for (Shadow& s : shadows_) {
      s.resource.Reset();
      s.valid.reset();
    }
    fill_pipeline_.Reset();
    fill_root_signature_.Reset();
    return false;
  }
  global_watch_ = shared_memory_.RegisterGlobalWatch(&VbMirror::GlobalWatchCallback, this);
  initialized_ = true;
  REXGPU_INFO("[ia] shadow: {} MB budget, {} KB pages, {} MB regions", budget_mb,
              1u << (page_size_log2_ - 10), 1u << (kRegionSizeLog2 - 20));
  return true;
}

void VbMirror::SetVerify(memory::Memory* memory, uint32_t pages_per_frame) {
  verify_memory_ = memory;
  verify_pages_per_frame_ = std::min<uint32_t>(pages_per_frame, kVerifyPagesMax);
  if (verify_readback_) {
    verify_readback_->Unmap(0, nullptr);
    verify_readback_.Reset();
    verify_mapping_ = nullptr;
  }
  for (VerifySlot& slot : verify_slots_) {
    slot.submission = 0;
    slot.count = 0;
  }
  if (!verify_memory_ || !verify_pages_per_frame_ || !initialized_) {
    verify_pages_per_frame_ = 0;
    return;
  }
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
  const uint64_t bytes = uint64_t(kVerifySlots) * kVerifyPagesMax * (uint64_t(1) << page_size_log2_);
  D3D12_RESOURCE_DESC desc;
  ui::d3d12::util::FillBufferResourceDesc(desc, bytes, D3D12_RESOURCE_FLAG_NONE);
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesReadback,
                                             D3D12_HEAP_FLAG_NONE, &desc,
                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&verify_readback_)))) {
    REXGPU_WARN("[ia] verify: the readback buffer failed - verify off");
    verify_pages_per_frame_ = 0;
    return;
  }
  void* mapping = nullptr;
  if (FAILED(verify_readback_->Map(0, nullptr, &mapping)) || !mapping) {
    REXGPU_WARN("[ia] verify: the readback map failed - verify off");
    verify_readback_.Reset();
    verify_pages_per_frame_ = 0;
    return;
  }
  verify_mapping_ = static_cast<const uint8_t*>(mapping);
  REXGPU_INFO("[ia] verify: {} pages per frame", verify_pages_per_frame_);
}

void VbMirror::VerifyTick() {
  verify_frame_pages_ = 0;
  ++verify_frames_;
  if (!verify_pages_per_frame_ || !verify_mapping_) {
    return;
  }
  const uint64_t completed = command_processor_.GetCompletedSubmission();
  const uint32_t page_bytes = 1u << page_size_log2_;
  const uint32_t page_dwords = page_bytes / 4;
  for (uint32_t si = 0; si < kVerifySlots; ++si) {
    VerifySlot& slot = verify_slots_[si];
    if (!slot.submission || slot.submission > completed) {
      continue;
    }
    for (uint32_t k = 0; k < slot.count; ++k) {
      const uint32_t page = slot.page[k];
      const uint32_t endian = slot.endian[k];
      Shadow& s = shadows_[endian];
      // Invalidated since the copy: the compare would be against newer bytes.
      if (!(s.valid[page >> 6].load(std::memory_order_acquire) & (uint64_t(1) << (page & 63)))) {
        ++stats_.verify_skipped;
        continue;
      }
      const uint32_t* cpu =
          verify_memory_->TranslatePhysical<const uint32_t*>(page << page_size_log2_);
      const uint32_t* gpu = reinterpret_cast<const uint32_t*>(
          verify_mapping_ + (size_t(si) * kVerifyPagesMax + k) * page_bytes);
      if (!cpu) {
        ++stats_.verify_skipped;
        continue;
      }
      uint32_t bad = 0, first_bad = UINT32_MAX;
      for (uint32_t d = 0; d < page_dwords; ++d) {
        uint32_t v = cpu[d];
        if (endian == 1 || endian == 2) {
          v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
        }
        if (endian == 2 || endian == 3) {
          v = (v << 16) | (v >> 16);
        }
        if (v != gpu[d]) {
          ++bad;
          if (first_bad == UINT32_MAX) {
            first_bad = d;
          }
        }
      }
      ++stats_.verify_pages;
      if (bad) {
        ++stats_.verify_bad_pages;
        stats_.verify_bad_dwords += bad;
        if (verify_frames_ - verify_last_warn_frame_ >= 60) {
          verify_last_warn_frame_ = verify_frames_;
          uint32_t expected = cpu[first_bad];
          if (endian == 1 || endian == 2) {
            expected = ((expected & 0x00FF00FFu) << 8) | ((expected >> 8) & 0x00FF00FFu);
          }
          if (endian == 2 || endian == 3) {
            expected = (expected << 16) | (expected >> 16);
          }
          REXGPU_WARN(
              "[ia] verify BAD page {:05X} (guest {:08X}) endian {}: {} of {} dwords differ, first "
              "at dword {}: expected {:08X} (guest {:08X}) got {:08X}",
              page, page << page_size_log2_, endian, bad, page_dwords, first_bad, expected,
              cpu[first_bad], gpu[first_bad]);
        }
      }
    }
    slot.submission = 0;
    slot.count = 0;
  }
}

void VbMirror::Shutdown() {
  if (verify_readback_) {
    verify_readback_->Unmap(0, nullptr);
    verify_readback_.Reset();
    verify_mapping_ = nullptr;
  }
  verify_pages_per_frame_ = 0;
  if (global_watch_) {
    shared_memory_.UnregisterGlobalWatch(global_watch_);
    global_watch_ = nullptr;
  }
  for (Shadow& s : shadows_) {
    s.resource.Reset();
    for (ID3D12Heap* heap : s.heaps) {
      heap->Release();
    }
    s.heaps.clear();
    std::memset(s.mapped, 0, sizeof(s.mapped));
    s.valid.reset();
    s.pending.clear();
    s.gpu_address = 0;
  }
  pending_count_ = 0;
  mapped_bytes_ = 0;
  fill_pipeline_.Reset();
  fill_root_signature_.Reset();
  initialized_ = false;
}

void VbMirror::ClearCache() {
  for (Shadow& s : shadows_) {
    if (s.valid) {
      for (uint32_t i = 0; i < valid_words_; ++i) {
        s.valid[i].store(0, std::memory_order_relaxed);
      }
    }
    s.pending.clear();
  }
  pending_count_ = 0;
}

bool VbMirror::EnsureShadow(uint32_t endian) {
  if (endian == 0 || endian >= kShadowCount) {
    return false;
  }
  Shadow& s = shadows_[endian];
  if (s.resource) {
    return true;
  }
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
  D3D12_RESOURCE_DESC desc;
  ui::d3d12::util::FillBufferResourceDesc(desc, SharedMemory::kBufferSize,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  if (FAILED(device->CreateReservedResource(&desc, kSteadyState, nullptr,
                                            IID_PPV_ARGS(&resource)))) {
    REXGPU_WARN("[ia] shadow {}: the {} MB reserved buffer failed (tiled resources?)", endian,
                SharedMemory::kBufferSize >> 20);
    return false;
  }
  s.resource = std::move(resource);
  s.gpu_address = s.resource->GetGPUVirtualAddress();
  s.valid.reset(new std::atomic<uint64_t>[valid_words_]);
  for (uint32_t i = 0; i < valid_words_; ++i) {
    s.valid[i].store(0, std::memory_order_relaxed);
  }
  std::memset(s.mapped, 0, sizeof(s.mapped));
  return true;
}

bool VbMirror::MapRegions(Shadow& s, uint32_t region_first, uint32_t region_last) {
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  ID3D12CommandQueue* direct_queue = provider.GetDirectQueue();
  for (uint32_t r = region_first; r <= region_last; ++r) {
    if (s.mapped[r >> 6] & (uint64_t(1) << (r & 63))) {
      continue;
    }
    const uint64_t region_bytes = uint64_t(1) << kRegionSizeLog2;
    if (mapped_bytes_ + region_bytes > budget_bytes_) {
      ++stats_.map_fail;
      return false;
    }
    D3D12_HEAP_DESC heap_desc = {};
    heap_desc.SizeInBytes = region_bytes;
    heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | provider.GetHeapFlagCreateNotZeroed();
    ID3D12Heap* heap;
    if (FAILED(device->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap)))) {
      REXGPU_WARN("[ia] shadow: region heap creation failed");
      ++stats_.map_fail;
      return false;
    }
    s.heaps.push_back(heap);
    D3D12_TILED_RESOURCE_COORDINATE start_coordinates;
    start_coordinates.X = UINT((uint64_t(r) << kRegionSizeLog2) / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES);
    start_coordinates.Y = 0;
    start_coordinates.Z = 0;
    start_coordinates.Subresource = 0;
    D3D12_TILE_REGION_SIZE region_size;
    region_size.NumTiles = UINT(region_bytes / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES);
    region_size.UseBox = FALSE;
    D3D12_TILE_RANGE_FLAGS range_flags = D3D12_TILE_RANGE_FLAG_NONE;
    UINT heap_range_start_offset = 0;
    direct_queue->UpdateTileMappings(s.resource.Get(), 1, &start_coordinates, &region_size, heap,
                                     1, &range_flags, &heap_range_start_offset,
                                     &region_size.NumTiles, D3D12_TILE_MAPPING_FLAG_NONE);
    command_processor_.NotifyQueueOperationsDoneDirectly();
    s.mapped[r >> 6] |= uint64_t(1) << (r & 63);
    mapped_bytes_ += region_bytes;
  }
  return true;
}

void VbMirror::GlobalWatchCallback(const std::unique_lock<std::recursive_mutex>& global_lock,
                                   void* context, uint32_t address_first, uint32_t address_last,
                                   bool invalidated_by_gpu) {
  VbMirror& self = *static_cast<VbMirror*>(context);
  if (address_last < address_first) {
    return;
  }
  const uint32_t page_first = address_first >> self.page_size_log2_;
  const uint32_t page_last =
      std::min<uint32_t>(address_last >> self.page_size_log2_, self.page_count_ - 1);
  for (Shadow& s : self.shadows_) {
    if (!s.valid) {
      continue;
    }
    for (uint32_t w = page_first >> 6; w <= (page_last >> 6); ++w) {
      const uint32_t lo = std::max(page_first, w << 6) & 63;
      const uint32_t hi = std::min(page_last, (w << 6) | 63) & 63;
      const uint64_t mask = (hi == 63 ? ~uint64_t(0) : ((uint64_t(1) << (hi + 1)) - 1)) &
                            ~((uint64_t(1) << lo) - 1);
      s.valid[w].fetch_and(~mask, std::memory_order_acq_rel);
    }
  }
  self.stats_.stale_pages += page_last - page_first + 1;
}

bool VbMirror::Acquire(uint32_t start, uint32_t length, uint32_t endian,
                       D3D12_GPU_VIRTUAL_ADDRESS* out) {
  if (!initialized_ || !length) {
    return false;
  }
  ++stats_.acquires;
  const uint64_t end = uint64_t(start) + length;
  if (end > SharedMemory::kBufferSize) {
    return false;
  }
  if (endian == 0) {
    // Already host order: the shared memory buffer itself.
    ++stats_.direct;
    *out = shared_memory_.GetGPUAddress() + start;
    return true;
  }
  if (!EnsureShadow(endian)) {
    return false;
  }
  Shadow& s = shadows_[endian];
  if (!MapRegions(s, start >> kRegionSizeLog2, uint32_t((end - 1) >> kRegionSizeLog2))) {
    return false;
  }
  *out = s.gpu_address + start;
  return true;
}

void VbMirror::QueueFills(uint32_t start, uint32_t length, uint32_t endian) {
  if (!initialized_ || !length || endian == 0 || endian >= kShadowCount) {
    return;
  }
  Shadow& s = shadows_[endian];
  if (!s.resource) {
    return;
  }
  const uint64_t end = std::min<uint64_t>(uint64_t(start) + length, SharedMemory::kBufferSize);
  const uint32_t page_first = start >> page_size_log2_;
  const uint32_t page_last = uint32_t((end - 1) >> page_size_log2_);
  uint32_t run_first = UINT32_MAX;
  for (uint32_t p = page_first; p <= page_last; ++p) {
    const uint64_t bit = uint64_t(1) << (p & 63);
    // Mark valid before the copy is recorded (see the header).
    const bool was_valid = (s.valid[p >> 6].fetch_or(bit, std::memory_order_acq_rel) & bit) != 0;
    if (!was_valid) {
      if (run_first == UINT32_MAX) {
        run_first = p;
      }
    } else if (run_first != UINT32_MAX) {
      s.pending.push_back({run_first, p - run_first});
      ++pending_count_;
      run_first = UINT32_MAX;
    }
  }
  if (run_first != UINT32_MAX) {
    s.pending.push_back({run_first, page_last + 1 - run_first});
    ++pending_count_;
  }
}

void VbMirror::EmitFills() {
  if (!pending_count_) {
    return;
  }
  DeferredCommandList& list = command_processor_.GetDeferredCommandList();
  for (uint32_t endian = 1; endian < kShadowCount; ++endian) {
    Shadow& s = shadows_[endian];
    s.touched = !s.pending.empty();
    if (s.touched) {
      command_processor_.PushTransitionBarrier(s.resource.Get(), kSteadyState,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
  }
  command_processor_.SubmitBarriers();
  list.D3DSetComputeRootSignature(fill_root_signature_.Get());
  list.D3DSetPipelineState(fill_pipeline_.Get());
  list.D3DSetComputeRootShaderResourceView(1, shared_memory_.GetGPUAddress());
  const uint32_t page_dwords = 1u << (page_size_log2_ - 2);
  // The verify: a free slot takes up to the frame's remaining page budget
  // from this emit's runs (the first pages of the runs, spread over shadows).
  VerifySlot* verify_slot = nullptr;
  if (verify_pages_per_frame_ && verify_mapping_ &&
      verify_frame_pages_ < verify_pages_per_frame_) {
    VerifySlot& candidate = verify_slots_[verify_slot_next_ % kVerifySlots];
    if (candidate.submission == 0) {
      verify_slot = &candidate;
      verify_slot->count = 0;
    }
  }
  bool verify_copies[kShadowCount] = {};
  for (uint32_t endian = 1; endian < kShadowCount; ++endian) {
    Shadow& s = shadows_[endian];
    if (!s.touched) {
      continue;
    }
    list.D3DSetComputeRootUnorderedAccessView(2, s.gpu_address);
    for (const Run& run : s.pending) {
      const uint32_t dword_first = run.page_first * page_dwords;
      const uint32_t dword_count = run.page_count * page_dwords;
      const uint32_t constants[4] = {dword_first, dword_first, dword_count, endian};
      list.D3DSetComputeRoot32BitConstants(0, 4, constants, 0);
      list.D3DDispatch((dword_count + 255) / 256, 1, 1);
      ++stats_.fills;
      stats_.fill_pages += run.page_count;
      if (verify_slot && verify_slot->count < kVerifyPagesMax &&
          verify_frame_pages_ < verify_pages_per_frame_) {
        // The first and the LAST page of the run alternately: the first
        // page alone cannot see a dispatch that stops short.
        verify_slot->page[verify_slot->count] =
            (stats_.fills & 1) ? run.page_first + run.page_count - 1 : run.page_first;
        verify_slot->endian[verify_slot->count] = endian;
        ++verify_slot->count;
        ++verify_frame_pages_;
        verify_copies[endian] = true;
      }
    }
    s.pending.clear();
  }
  pending_count_ = 0;
  ++stats_.fill_emits;
  if (verify_slot && verify_slot->count) {
    // The copies: the touched shadows with pages to read go through
    // COPY_SOURCE on their way back to the steady state.
    const uint32_t page_bytes = 1u << page_size_log2_;
    for (uint32_t endian = 1; endian < kShadowCount; ++endian) {
      if (verify_copies[endian]) {
        command_processor_.PushTransitionBarrier(shadows_[endian].resource.Get(),
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                 D3D12_RESOURCE_STATE_COPY_SOURCE);
      }
    }
    command_processor_.SubmitBarriers();
    const uint32_t si = verify_slot_next_ % kVerifySlots;
    for (uint32_t k = 0; k < verify_slot->count; ++k) {
      list.D3DCopyBufferRegion(verify_readback_.Get(),
                               (uint64_t(si) * kVerifyPagesMax + k) * page_bytes,
                               shadows_[verify_slot->endian[k]].resource.Get(),
                               uint64_t(verify_slot->page[k]) * page_bytes, page_bytes);
    }
    verify_slot->submission = command_processor_.GetCurrentSubmission();
    ++verify_slot_next_;
    for (uint32_t endian = 1; endian < kShadowCount; ++endian) {
      Shadow& s = shadows_[endian];
      if (verify_copies[endian]) {
        command_processor_.PushTransitionBarrier(s.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 kSteadyState);
        s.touched = false;
      }
    }
  }
  for (uint32_t endian = 1; endian < kShadowCount; ++endian) {
    Shadow& s = shadows_[endian];
    if (s.touched) {
      command_processor_.PushTransitionBarrier(s.resource.Get(),
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSteadyState);
      s.touched = false;
    }
  }
}

}  // namespace rex::graphics::d3d12
