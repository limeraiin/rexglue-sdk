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
#include <rex/ui/d3d12/d3d12_util.h>

namespace rex::graphics::d3d12 {

namespace shaders {
// fxc, see src/graphics/shaders/vbm_fill.cs.hlsl for the build line.
#include "../shaders/bytecode/d3d12_5_1/vbm_fill_cs.h"
}  // namespace shaders

namespace {

inline uint64_t EntryKey(uint32_t start, uint32_t length, uint32_t endian) {
  return (uint64_t(start) << 32) | uint64_t(length & 0x3FFFFFFFu) | (uint64_t(endian & 3u) << 30);
}

}  // namespace

VbMirror::VbMirror(D3D12CommandProcessor& command_processor, SharedMemory& shared_memory)
    : command_processor_(command_processor), shared_memory_(shared_memory) {}

VbMirror::~VbMirror() { Shutdown(); }

bool VbMirror::Initialize(uint32_t budget_mb, uint32_t chunk_mb) {
  Shutdown();
  chunk_mb = std::max<uint32_t>(chunk_mb, 16);
  budget_mb = std::max<uint32_t>(budget_mb, chunk_mb);
  chunk_bytes_ = chunk_mb << 20;
  max_chunks_ = std::max<uint32_t>(budget_mb / chunk_mb, 1);

  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();

  // b0 = 4 constants, t0 = the shared memory (root SRV), u0 = the chunk (root UAV).
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
    REXGPU_WARN("[ia] mirror: fill root signature creation failed - off");
    return false;
  }
  *(fill_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::vbm_fill_cs, sizeof(shaders::vbm_fill_cs), fill_root_signature_.Get());
  if (!fill_pipeline_) {
    REXGPU_WARN("[ia] mirror: fill pipeline creation failed - off");
    fill_root_signature_.Reset();
    return false;
  }
  if (!CreateChunk()) {
    fill_pipeline_.Reset();
    fill_root_signature_.Reset();
    return false;
  }
  initialized_ = true;
  REXGPU_INFO("[ia] mirror: {} MB budget in {} MB chunks", budget_mb, chunk_mb);
  return true;
}

void VbMirror::Shutdown() {
  ClearCache();
  chunks_.clear();
  fill_pipeline_.Reset();
  fill_root_signature_.Reset();
  initialized_ = false;
}

void VbMirror::ClearCache() {
  for (auto& kv : entries_) {
    Unwatch(*kv.second);
  }
  entries_.clear();
  pending_.clear();
  lru_head_ = lru_tail_ = nullptr;
  bytes_used_ = 0;
  for (Chunk& c : chunks_) {
    c.free_blocks.clear();
    c.free_blocks.emplace(0, chunk_bytes_);
  }
}

bool VbMirror::CreateChunk() {
  if (chunks_.size() >= max_chunks_) {
    return false;
  }
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
  D3D12_RESOURCE_DESC desc;
  ui::d3d12::util::FillBufferResourceDesc(desc, chunk_bytes_,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             D3D12_HEAP_FLAG_NONE, &desc, kSteadyState, nullptr,
                                             IID_PPV_ARGS(&resource)))) {
    REXGPU_WARN("[ia] mirror: chunk {} ({} MB) creation failed", chunks_.size(),
                chunk_bytes_ >> 20);
    return false;
  }
  Chunk& c = chunks_.emplace_back();
  c.resource = std::move(resource);
  c.gpu_address = c.resource->GetGPUVirtualAddress();
  c.free_blocks.emplace(0, chunk_bytes_);
  return true;
}

bool VbMirror::AllocateBlock(uint32_t length, uint32_t* chunk, uint32_t* offset) {
  for (uint32_t ci = 0; ci < uint32_t(chunks_.size()); ++ci) {
    Chunk& c = chunks_[ci];
    for (auto it = c.free_blocks.begin(); it != c.free_blocks.end(); ++it) {
      if (it->second >= length) {
        *chunk = ci;
        *offset = it->first;
        const uint32_t rest = it->second - length;
        const uint32_t next_offset = it->first + length;
        c.free_blocks.erase(it);
        if (rest) {
          c.free_blocks.emplace(next_offset, rest);
        }
        return true;
      }
    }
  }
  return false;
}

void VbMirror::FreeBlock(uint32_t chunk, uint32_t offset, uint32_t length) {
  Chunk& c = chunks_[chunk];
  auto next = c.free_blocks.lower_bound(offset);
  // Merge with the following block.
  if (next != c.free_blocks.end() && next->first == offset + length) {
    length += next->second;
    next = c.free_blocks.erase(next);
  }
  // Merge with the preceding block.
  if (next != c.free_blocks.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == offset) {
      prev->second += length;
      return;
    }
  }
  c.free_blocks.emplace(offset, length);
}

void VbMirror::WatchCallback(const std::unique_lock<std::recursive_mutex>& global_lock,
                             void* context, void* data, uint64_t argument,
                             bool invalidated_by_gpu) {
  Entry& e = *static_cast<Entry*>(data);
  e.fired.store(true, std::memory_order_release);
  e.stale.store(true, std::memory_order_release);
}

void VbMirror::Watch(Entry& e) {
  e.fired.store(false, std::memory_order_release);
  e.watch = shared_memory_.WatchMemoryRange(e.start, e.length, &VbMirror::WatchCallback, this, &e, 0);
  if (!e.watch) {
    e.fired.store(true, std::memory_order_release);
  }
}

void VbMirror::Unwatch(Entry& e) {
  if (e.watch && !e.fired.load(std::memory_order_acquire)) {
    shared_memory_.UnwatchMemoryRange(e.watch);
  }
  e.watch = nullptr;
  e.fired.store(true, std::memory_order_release);
}

void VbMirror::LruRemove(Entry& e) {
  if (e.lru_prev) {
    e.lru_prev->lru_next = e.lru_next;
  } else if (lru_head_ == &e) {
    lru_head_ = e.lru_next;
  }
  if (e.lru_next) {
    e.lru_next->lru_prev = e.lru_prev;
  } else if (lru_tail_ == &e) {
    lru_tail_ = e.lru_prev;
  }
  e.lru_prev = e.lru_next = nullptr;
}

void VbMirror::LruPushBack(Entry& e) {
  e.lru_prev = lru_tail_;
  e.lru_next = nullptr;
  if (lru_tail_) {
    lru_tail_->lru_next = &e;
  } else {
    lru_head_ = &e;
  }
  lru_tail_ = &e;
}

void VbMirror::EvictEntry(Entry& e) {
  Unwatch(e);
  LruRemove(e);
  if (e.queued) {
    pending_.erase(std::remove(pending_.begin(), pending_.end(), &e), pending_.end());
  }
  const uint32_t block = (e.length + kBlockAlign - 1) & ~(kBlockAlign - 1);
  FreeBlock(e.chunk, e.offset, block);
  bytes_used_ -= block;
  ++stats_.evicts;
  entries_.erase(EntryKey(e.start, e.length, e.endian));
}

VbMirror::Entry* VbMirror::Acquire(uint32_t start, uint32_t length, uint32_t endian) {
  if (!initialized_ || !length) {
    return nullptr;
  }
  ++stats_.acquires;
  // 4-aligned outwards: the fill copies dwords; the caller adds start & 3.
  const uint32_t aligned_start = start & ~3u;
  const uint32_t aligned_end = (start + length + 3u) & ~3u;
  const uint32_t aligned_length = aligned_end - aligned_start;
  if (aligned_length > chunk_bytes_ || aligned_end < aligned_start) {
    ++stats_.alloc_fail;
    return nullptr;
  }
  const uint64_t key = EntryKey(aligned_start, aligned_length, endian);
  auto it = entries_.find(key);
  Entry* e;
  if (it != entries_.end()) {
    e = it->second.get();
    ++stats_.hits;
    LruRemove(*e);
  } else {
    const uint32_t block = (aligned_length + kBlockAlign - 1) & ~(kBlockAlign - 1);
    uint32_t chunk, offset;
    if (!AllocateBlock(block, &chunk, &offset)) {
      // A new chunk, then the LRU entries the GPU is done with, oldest first.
      bool ok = CreateChunk() && AllocateBlock(block, &chunk, &offset);
      while (!ok) {
        Entry* victim = lru_head_;
        while (victim && victim->last_use_submission > completed_submission_) {
          victim = victim->lru_next;
        }
        if (!victim) {
          break;
        }
        EvictEntry(*victim);
        ok = AllocateBlock(block, &chunk, &offset);
      }
      if (!ok) {
        ++stats_.alloc_fail;
        return nullptr;
      }
    }
    std::unique_ptr<Entry> owned = std::make_unique<Entry>();
    e = owned.get();
    e->start = aligned_start;
    e->length = aligned_length;
    e->endian = endian;
    e->chunk = chunk;
    e->offset = offset;
    e->stale.store(true, std::memory_order_relaxed);
    e->fired.store(true, std::memory_order_relaxed);
    entries_.emplace(key, std::move(owned));
    bytes_used_ += block;
    ++stats_.allocs;
  }
  e->last_use_frame = frame_;
  e->last_use_submission = submission_;
  LruPushBack(*e);
  if (e->stale.load(std::memory_order_acquire) && !e->queued) {
    e->queued = true;
    pending_.push_back(e);
  }
  return e;
}

D3D12_GPU_VIRTUAL_ADDRESS VbMirror::GpuAddress(const Entry& e) const {
  return chunks_[e.chunk].gpu_address + e.offset;
}

void VbMirror::DropPendingFills() {
  for (Entry* e : pending_) {
    e->queued = false;
  }
  pending_.clear();
}

void VbMirror::EmitFills() {
  if (pending_.empty()) {
    return;
  }
  DeferredCommandList& list = command_processor_.GetDeferredCommandList();
  for (Chunk& c : chunks_) {
    c.touched = false;
  }
  for (Entry* e : pending_) {
    Chunk& c = chunks_[e->chunk];
    if (!c.touched) {
      c.touched = true;
      command_processor_.PushTransitionBarrier(c.resource.Get(), kSteadyState,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
  }
  command_processor_.SubmitBarriers();
  list.D3DSetComputeRootSignature(fill_root_signature_.Get());
  list.D3DSetPipelineState(fill_pipeline_.Get());
  list.D3DSetComputeRootShaderResourceView(
      1, static_cast<D3D12SharedMemory&>(shared_memory_).GetGPUAddress());
  uint32_t bound_chunk = UINT32_MAX;
  for (Entry* e : pending_) {
    // The watch is (re)armed before the copy is taken: a write racing the
    // fill leaves the entry stale again, never a stale copy marked fresh.
    if (e->fired.load(std::memory_order_acquire)) {
      Watch(*e);
    }
    e->stale.store(false, std::memory_order_release);
    e->queued = false;
    if (e->chunk != bound_chunk) {
      bound_chunk = e->chunk;
      list.D3DSetComputeRootUnorderedAccessView(2, chunks_[bound_chunk].gpu_address);
    }
    const uint32_t constants[4] = {e->start >> 2, e->offset >> 2, e->length >> 2, e->endian};
    list.D3DSetComputeRoot32BitConstants(0, 4, constants, 0);
    list.D3DDispatch((e->length / 4 + 255) / 256, 1, 1);
    ++stats_.fills;
    stats_.fill_bytes += e->length;
  }
  pending_.clear();
  ++stats_.fill_emits;
  for (Chunk& c : chunks_) {
    if (c.touched) {
      command_processor_.PushTransitionBarrier(c.resource.Get(),
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kSteadyState);
      c.touched = false;
    }
  }
}

}  // namespace rex::graphics::d3d12
