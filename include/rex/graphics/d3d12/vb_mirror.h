/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_D3D12_VB_MIRROR_H_
#define REX_GRAPHICS_D3D12_VB_MIRROR_H_

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <rex/graphics/shared_memory.h>
#include <rex/ui/d3d12/d3d12_api.h>

namespace rex::graphics::d3d12 {

class D3D12CommandProcessor;

// [ia] The host-order vertex / index buffer mirror.
//
// The input assembler reads little-endian data; the guest's vertex and index
// buffers sit in the shared memory buffer in the byte order the recompiled
// game wrote them (big-endian words, per the fetch constant's endian mode).
// This is a cache of {guest byte range, endian} -> a byte-swapped copy in a
// DEFAULT-heap arena, filled on the GPU by a small compute shader from the
// shared memory buffer (after the draw's own RequestRange upload), and
// invalidated by the shared memory's write watches (CPU writes AND GPU
// writes: resolves, memexport). Static geometry fills once and is read by
// every later draw through a vertex buffer / index buffer view; dynamic
// buffers re-fill when the guest rewrites them.
//
// Arena: chunks of `chunk_mb`, first-fit inside a chunk (256-byte blocks,
// coalescing free list), LRU eviction only of entries whose last use is in a
// completed submission (the GPU is done with them). Acquire never records
// commands: the fills are queued and emitted by EmitFills at the draw's
// barrier point.
class VbMirror {
 public:
  struct Entry {
    uint32_t start = 0;   // guest byte address, 4-aligned
    uint32_t length = 0;  // bytes, multiple of 4
    uint32_t endian = 0;  // xenos::Endian
    uint32_t chunk = 0;
    uint32_t offset = 0;  // bytes into the chunk
    uint32_t last_use_frame = 0;
    uint64_t last_use_submission = 0;
    // The watch fired (the copy no longer matches guest memory) - set from
    // the memory-write path under the shared memory's global lock; only ever
    // read and cleared by the command processor thread.
    std::atomic<bool> stale{true};
    // The watch handle is invalid once the callback ran (the shared memory
    // cancels it): never unwatch a fired one.
    std::atomic<bool> fired{true};
    SharedMemory::WatchHandle watch = nullptr;
    bool queued = false;  // in the pending fill list
    Entry* lru_prev = nullptr;
    Entry* lru_next = nullptr;
  };

  struct Stats {
    uint64_t acquires = 0, hits = 0, allocs = 0, fills = 0, fill_bytes = 0, stale = 0,
             evicts = 0, alloc_fail = 0, fill_emits = 0;
  };

  VbMirror(D3D12CommandProcessor& command_processor, SharedMemory& shared_memory);
  ~VbMirror();

  bool Initialize(uint32_t budget_mb, uint32_t chunk_mb);
  void Shutdown();
  // Drops every entry. Only when the GPU is idle (the cache clear path).
  void ClearCache();

  void BeginFrame(uint32_t frame, uint64_t submission) {
    frame_ = frame;
    submission_ = submission;
  }
  void CompletedSubmissionUpdated(uint64_t completed) { completed_submission_ = completed; }

  // The entry covering [start, start + length) (4-aligned outwards) in the
  // given endian, allocated if needed; queues a fill when the copy is stale.
  // Null when the arena cannot hold it (the caller takes the raw path).
  Entry* Acquire(uint32_t start, uint32_t length, uint32_t endian);
  // The GPU address of the entry's first byte (its `start`).
  D3D12_GPU_VIRTUAL_ADDRESS GpuAddress(const Entry& e) const;

  bool HasPendingFills() const { return !pending_.empty(); }
  // Forgets the queued fills (the entries stay stale and re-queue on their
  // next Acquire); for a draw that is abandoned after its acquires.
  void DropPendingFills();
  // Records the queued fills: the chunk transitions to UAV, the dispatches
  // (the shared memory buffer must already be readable by a compute root
  // SRV: UseForReading), and the transitions back (left for the caller's
  // next SubmitBarriers). Clears the pending list.
  void EmitFills();

  const Stats& stats() const { return stats_; }
  uint32_t entry_count() const { return uint32_t(entries_.size()); }
  uint64_t bytes_used() const { return bytes_used_; }
  uint64_t bytes_budget() const { return uint64_t(chunk_bytes_) * max_chunks_; }
  uint32_t chunk_count() const { return uint32_t(chunks_.size()); }

 private:
  static constexpr uint32_t kBlockAlign = 256;
  static constexpr D3D12_RESOURCE_STATES kSteadyState =
      D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_INDEX_BUFFER;

  struct Chunk {
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    D3D12_GPU_VIRTUAL_ADDRESS gpu_address = 0;
    // offset -> size of the free blocks, coalesced.
    std::map<uint32_t, uint32_t> free_blocks;
    bool touched = false;  // this EmitFills
  };

  static void WatchCallback(const std::unique_lock<std::recursive_mutex>& global_lock,
                            void* context, void* data, uint64_t argument,
                            bool invalidated_by_gpu);
  void Watch(Entry& e);
  void Unwatch(Entry& e);
  bool CreateChunk();
  bool AllocateBlock(uint32_t length, uint32_t* chunk, uint32_t* offset);
  void FreeBlock(uint32_t chunk, uint32_t offset, uint32_t length);
  void EvictEntry(Entry& e);
  void LruRemove(Entry& e);
  void LruPushBack(Entry& e);

  D3D12CommandProcessor& command_processor_;
  SharedMemory& shared_memory_;
  bool initialized_ = false;
  uint32_t chunk_bytes_ = 0;
  uint32_t max_chunks_ = 0;
  std::vector<Chunk> chunks_;
  std::unordered_map<uint64_t, std::unique_ptr<Entry>> entries_;
  std::vector<Entry*> pending_;
  Entry* lru_head_ = nullptr;  // the least recently used
  Entry* lru_tail_ = nullptr;
  uint64_t bytes_used_ = 0;
  uint32_t frame_ = 0;
  uint64_t submission_ = 0;
  uint64_t completed_submission_ = 0;
  Stats stats_;

  Microsoft::WRL::ComPtr<ID3D12RootSignature> fill_root_signature_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> fill_pipeline_;
};

}  // namespace rex::graphics::d3d12

#endif  // REX_GRAPHICS_D3D12_VB_MIRROR_H_
