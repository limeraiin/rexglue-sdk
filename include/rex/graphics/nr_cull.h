// [cull] Draw culling from object-space bounds (ISSUEDRAW increment 2).
// Drive 798/801: 66% of the city's primitives pass zero samples at draw time,
// 80% of the primitives come from vertex shaders whose position path is an
// affine map of one vertex fetch (PosPath). Each draw's geometry gets an
// object-space AABB at first sight (a CPU scan of the position stream over
// the draw's index range, cached by geometry key and invalidated by the
// shared memory's write watches); the 8 corners go through the shader's own
// clip transform, evaluated from the register file, and the draw is refused
// when every corner lies outside one clip plane. Exact for the clipper
// (convexity in homogeneous space); a rasterizer with clipping disabled or a
// pre-transformed vertex format is not eligible.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include <rex/graphics/pipeline/shader/pos_path.h>
#include <rex/graphics/xenos.h>

namespace rex::memory {
class Memory;
}

namespace rex::graphics {

class SharedMemory;

struct CullBounds {
  float mn[4], mx[4];
  bool unbounded;  // could not be computed: never cull
};

struct CullStats {
  uint64_t bounds_new = 0, bounds_unbounded = 0, bounds_invalidated = 0, bounds_full = 0;
  uint64_t bounds_ns = 0;  // CPU time in the scans
};

class NrCull {
 public:
  NrCull();
  ~NrCull();
  void Initialize(memory::Memory* memory, SharedMemory* shared_memory);
  void Shutdown();

  struct DrawDesc {
    const PosPath* path;
    xenos::xe_gpu_vertex_fetch_t fetch;  // the position stream's fetch constant
    uint32_t base_vertex;                // VGT_INDX_OFFSET
    bool indexed;
    uint32_t index_count;  // indices when indexed, vertices otherwise
    uint32_t ib_guest_base;
    xenos::IndexFormat ib_format;
    xenos::Endian ib_endian;
  };
  // The draw's bounds (cached; computed at first sight). Null when the table
  // is full or the geometry cannot be bounded (counted in stats).
  const CullBounds* GetBounds(const DrawDesc& d, CullStats& stats);

  // True when all 8 corners of the bounds, carried by m (clip = m * (p, 1)),
  // lie outside one of the x/y clip planes.
  static bool FrustumOutside(const CullBounds& b, const float m[4][5]);

  size_t entries() const { return entries_.size(); }

 private:
  struct Entry;
  static void WatchCallback(const std::unique_lock<std::recursive_mutex>& global_lock,
                            void* context, void* data, uint64_t argument,
                            bool invalidated_by_gpu);
  bool Compute(const DrawDesc& d, Entry& e, CullStats& stats);
  void Watch(Entry& e);

  memory::Memory* memory_ = nullptr;
  SharedMemory* shared_memory_ = nullptr;
  std::unordered_map<uint64_t, std::unique_ptr<Entry>> entries_;
};

}  // namespace rex::graphics
