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
#include <memory>
#include <mutex>
#include <vector>

#include <rex/graphics/shared_memory.h>
#include <rex/ui/d3d12/d3d12_api.h>

namespace rex::memory {
class Memory;
}

namespace rex::graphics::d3d12 {

class D3D12CommandProcessor;
class D3D12SharedMemory;

// [ia] The host-order shadow of guest memory for the input assembler.
//
// The IA reads little-endian data; the guest's vertex and index buffers sit
// in the shared memory buffer in the byte order the recompiled game wrote
// them (big-endian words, per the fetch constant's endian mode). A shadow is
// a byte-swapped 1:1 copy of the shared memory's address space for one
// endian mode (8in16 for 16-bit indices, 8in32 for the vertex data): a
// reserved 512 MB buffer whose 4 MB regions are mapped on first touch, like
// the shared memory buffer itself, so the physical footprint is the touched
// geometry. Validity is per page (the shared memory's page size): a page is
// copied by a compute shader (vbm_fill.cs.hlsl) from the shared memory
// buffer after the draw's own RequestRange upload, and invalidated through
// the shared memory's global watch (CPU writes AND GPU writes: resolves,
// memexport). A range's view is simply shadow_base + guest address, so no
// range is ever copied twice and nothing is allocated per range. Endian
// "none" needs no shadow: the range is bound from the shared memory buffer.
class VbMirror {
 public:
  struct Stats {
    uint64_t acquires = 0, direct = 0, fills = 0, fill_pages = 0, stale_pages = 0,
             map_fail = 0, fill_emits = 0;
    // QueueFills' scan: 64-page words examined (one load each) and the words
    // that held an invalid page (one atomic each). Drive 870: the per-page
    // atomic scan of every draw's range was a CP cost (4 MB VB = 1024 per draw).
    uint64_t queue_calls = 0, scan_words = 0, scan_rmw = 0;
    // The verify: pages read back and compared with the byte-swapped guest
    // memory; a page invalidated between the copy and the compare is skipped.
    uint64_t verify_pages = 0, verify_bad_pages = 0, verify_bad_dwords = 0,
             verify_skipped = 0;
  };

  // [ia] verify: up to `pages_per_frame` of each frame's filled pages are
  // copied to a readback buffer and, once the submission completed, compared
  // dword by dword with the guest memory swapped on the CPU. 0 = off.
  void SetVerify(memory::Memory* memory, uint32_t pages_per_frame);
  // Compares the readback slots whose submission completed; once per frame.
  void VerifyTick();

  VbMirror(D3D12CommandProcessor& command_processor, D3D12SharedMemory& shared_memory);
  ~VbMirror();

  bool Initialize(uint32_t budget_mb);
  void Shutdown();
  // Every page invalid again (mappings kept); the cache clear path.
  void ClearCache();

  // The host-order GPU address of [start, start + length) in the given
  // endian; maps the shadow regions it touches. False when the endian has no
  // shadow, the mapping failed or the budget is spent (the raw path then).
  bool Acquire(uint32_t start, uint32_t length, uint32_t endian, D3D12_GPU_VIRTUAL_ADDRESS* out);
  // Queues the copies of the range's invalid pages (call AFTER the range's
  // RequestRange, so the copy sees the upload). Pages are marked valid here,
  // before the copy: a write racing the fill leaves them invalid again.
  void QueueFills(uint32_t start, uint32_t length, uint32_t endian);
  bool HasPendingFills() const { return pending_count_ != 0; }
  // Records the queued copies: the shadow transitions to UAV, the dispatches
  // (the shared memory buffer must be readable by a compute root SRV:
  // UseForReading), the transitions back (left for the caller's next
  // SubmitBarriers).
  void EmitFills();

  const Stats& stats() const { return stats_; }
  uint64_t mapped_bytes() const { return mapped_bytes_; }
  uint64_t budget_bytes() const { return budget_bytes_; }

 private:
  static constexpr uint32_t kShadowCount = 4;  // by xenos::Endian; 0 unused
  static constexpr uint32_t kRegionSizeLog2 = 22;  // 4 MB mapping regions
  static constexpr uint32_t kRegionCount = SharedMemory::kBufferSize >> kRegionSizeLog2;
  static constexpr D3D12_RESOURCE_STATES kSteadyState =
      D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_INDEX_BUFFER;

  struct Run {
    uint32_t page_first, page_count;
  };
  struct Shadow {
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    D3D12_GPU_VIRTUAL_ADDRESS gpu_address = 0;
    std::vector<ID3D12Heap*> heaps;
    uint64_t mapped[(kRegionCount + 63) / 64] = {};
    // One bit per shared memory page: the shadow holds the page's current
    // bytes. Cleared from the global watch (another thread), set here.
    std::unique_ptr<std::atomic<uint64_t>[]> valid;
    std::vector<Run> pending;
    bool touched = false;
  };

  static void GlobalWatchCallback(const std::unique_lock<std::recursive_mutex>& global_lock,
                                  void* context, uint32_t address_first, uint32_t address_last,
                                  bool invalidated_by_gpu);
  bool EnsureShadow(uint32_t endian);
  bool MapRegions(Shadow& s, uint32_t region_first, uint32_t region_last);

  D3D12CommandProcessor& command_processor_;
  D3D12SharedMemory& shared_memory_;
  bool initialized_ = false;
  uint32_t page_size_log2_ = 12;
  uint32_t page_count_ = 0;
  uint32_t valid_words_ = 0;
  uint64_t budget_bytes_ = 0;
  uint64_t mapped_bytes_ = 0;
  Shadow shadows_[kShadowCount];
  uint32_t pending_count_ = 0;
  SharedMemory::GlobalWatchHandle global_watch_ = nullptr;
  Stats stats_;

  Microsoft::WRL::ComPtr<ID3D12RootSignature> fill_root_signature_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> fill_pipeline_;

  // The verify ring: kVerifySlots frames in flight, kVerifyPagesMax pages each.
  static constexpr uint32_t kVerifySlots = 4;
  static constexpr uint32_t kVerifyPagesMax = 16;
  struct VerifySlot {
    uint64_t submission = 0;  // 0 = free
    uint32_t count = 0;
    uint32_t page[kVerifyPagesMax];
    uint32_t endian[kVerifyPagesMax];
  };
  memory::Memory* verify_memory_ = nullptr;
  uint32_t verify_pages_per_frame_ = 0;
  Microsoft::WRL::ComPtr<ID3D12Resource> verify_readback_;
  const uint8_t* verify_mapping_ = nullptr;
  VerifySlot verify_slots_[kVerifySlots];
  uint32_t verify_slot_next_ = 0;
  uint32_t verify_frame_pages_ = 0;  // pages copied this frame (reset by VerifyTick)
  uint64_t verify_last_warn_frame_ = 0;
  uint64_t verify_frames_ = 0;
};

}  // namespace rex::graphics::d3d12

#endif  // REX_GRAPHICS_D3D12_VB_MIRROR_H_
