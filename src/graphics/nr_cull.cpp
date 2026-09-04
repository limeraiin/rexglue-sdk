// [cull] see nr_cull.h.
#include <rex/graphics/nr_cull.h>

#include <chrono>
#include <cmath>
#include <cstring>

#include <rex/graphics/shared_memory.h>
#include <rex/memory.h>
#include <rex/thread/mutex.h>

namespace rex::graphics {

namespace {
constexpr size_t kMaxEntries = 65536;
constexpr uint32_t kMaxIndices = 1u << 20;
constexpr uint32_t kMaxVertexSpan = 1u << 20;

inline uint64_t Mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  return h;
}

inline float LoadFloat(const uint8_t* p, xenos::Endian endian) {
  uint32_t u;
  std::memcpy(&u, p, 4);
  u = xenos::GpuSwap(u, endian);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
}  // namespace

struct NrCull::Entry {
  CullBounds bounds{};
  std::atomic<bool> valid{false};
  std::atomic<bool> fired[2]{false, false};
  SharedMemory::WatchHandle watch[2] = {nullptr, nullptr};
  uint32_t range_start[2] = {0, 0}, range_len[2] = {0, 0};
};

NrCull::NrCull() = default;
NrCull::~NrCull() { Shutdown(); }

void NrCull::Initialize(memory::Memory* memory, SharedMemory* shared_memory) {
  memory_ = memory;
  shared_memory_ = shared_memory;
}

void NrCull::Shutdown() {
  if (shared_memory_) {
    rex::thread::global_critical_region gcr;
    auto lock = gcr.Acquire();
    for (auto& kv : entries_) {
      Entry& e = *kv.second;
      for (uint32_t a = 0; a < 2; ++a) {
        if (e.watch[a] && !e.fired[a].load(std::memory_order_acquire)) {
          shared_memory_->UnwatchMemoryRange(e.watch[a]);
        }
        e.watch[a] = nullptr;
      }
    }
  }
  entries_.clear();
  shared_memory_ = nullptr;
  memory_ = nullptr;
}

void NrCull::WatchCallback(const std::unique_lock<std::recursive_mutex>& global_lock,
                           void* context, void* data, uint64_t argument, bool invalidated_by_gpu) {
  Entry& e = *static_cast<Entry*>(data);
  e.fired[argument & 1].store(true, std::memory_order_release);
  e.valid.store(false, std::memory_order_release);
}

void NrCull::Watch(Entry& e) {
  for (uint32_t a = 0; a < 2; ++a) {
    e.fired[a].store(false, std::memory_order_release);
    e.watch[a] = e.range_len[a]
                     ? shared_memory_->WatchMemoryRange(e.range_start[a], e.range_len[a],
                                                        &NrCull::WatchCallback, this, &e, a)
                     : nullptr;
  }
}

bool NrCull::Compute(const DrawDesc& d, Entry& e, CullStats& stats) {
  const auto t0 = std::chrono::steady_clock::now();
  e.bounds.unbounded = true;
  e.range_len[0] = e.range_len[1] = 0;
  const uint32_t vb_base = d.fetch.address << 2;
  const uint32_t vb_size = d.fetch.size << 2;
  const uint32_t stride = d.path->stride_dwords * 4;
  const uint32_t offset = uint32_t(d.path->offset_dwords) * 4;
  const uint32_t comps = d.path->format == xenos::VertexFormat::k_32_32_32_32_FLOAT ? 4 : 3;
  const uint32_t vsize = comps * 4;
  auto done = [&](bool ok) {
    stats.bounds_ns += uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
            .count());
    if (!ok) ++stats.bounds_unbounded;
    return ok;
  };
  if (!stride || !vb_size || !d.index_count || d.index_count > kMaxIndices ||
      (d.path->format != xenos::VertexFormat::k_32_32_32_FLOAT &&
       d.path->format != xenos::VertexFormat::k_32_32_32_32_FLOAT)) {
    return done(false);
  }
  uint32_t vmin = UINT32_MAX, vmax = 0;
  if (d.indexed) {
    const uint32_t isz = d.ib_format == xenos::IndexFormat::kInt32 ? 4 : 2;
    const uint32_t ib_phys = d.ib_guest_base & 0x1FFFFFFFu;
    const uint32_t ib_len = d.index_count * isz;
    if (!ib_phys || uint64_t(ib_phys) + ib_len > 0x20000000ull) return done(false);
    const uint8_t* ib = memory_->TranslatePhysical<const uint8_t*>(ib_phys);
    if (!ib) return done(false);
    for (uint32_t i = 0; i < d.index_count; ++i) {
      uint32_t idx;
      if (isz == 4) {
        uint32_t u;
        std::memcpy(&u, ib + i * 4, 4);
        u = xenos::GpuSwap(u, d.ib_endian);
        if (u == 0xFFFFFFFFu) continue;  // primitive reset
        idx = u & 0xFFFFFFu;
      } else {
        uint16_t u;
        std::memcpy(&u, ib + i * 2, 2);
        u = xenos::GpuSwap(u, d.ib_endian);
        if (u == 0xFFFFu) continue;
        idx = u;
      }
      vmin = std::min(vmin, idx);
      vmax = std::max(vmax, idx);
    }
    if (vmin == UINT32_MAX) return done(false);
    e.range_start[1] = ib_phys;
    e.range_len[1] = ib_len;
  } else {
    vmin = 0;
    vmax = d.index_count - 1;
  }
  vmin += d.base_vertex;
  vmax += d.base_vertex;
  if (vmax < vmin || vmax - vmin >= kMaxVertexSpan) return done(false);
  const uint64_t first = uint64_t(vmin) * stride + offset;
  const uint64_t last = uint64_t(vmax) * stride + offset + vsize;
  if (last > vb_size || uint64_t(vb_base) + last > 0x20000000ull) return done(false);
  const uint8_t* vb = memory_->TranslatePhysical<const uint8_t*>(vb_base);
  if (!vb) return done(false);
  float mn[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
  float mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  const xenos::Endian endian = d.fetch.endian;
  for (uint32_t v = vmin; v <= vmax; ++v) {
    const uint8_t* p = vb + uint64_t(v) * stride + offset;
    for (uint32_t c = 0; c < comps; ++c) {
      const float f = LoadFloat(p + c * 4, endian);
      if (!std::isfinite(f)) return done(false);
      mn[c] = std::min(mn[c], f);
      mx[c] = std::max(mx[c], f);
    }
  }
  if (comps == 3) {
    mn[3] = mx[3] = 1.0f;
  }
  std::memcpy(e.bounds.mn, mn, sizeof(mn));
  std::memcpy(e.bounds.mx, mx, sizeof(mx));
  e.bounds.unbounded = false;
  e.range_start[0] = uint32_t(vb_base + first);
  e.range_len[0] = uint32_t(last - first);
  return done(true);
}

const CullBounds* NrCull::GetBounds(const DrawDesc& d, CullStats& stats) {
  uint64_t key = 0x5EEDull;
  key = Mix(key, d.fetch.dword_0);
  key = Mix(key, d.fetch.dword_1);
  key = Mix(key, uint64_t(uint32_t(d.path->offset_dwords)) | (uint64_t(d.path->stride_dwords) << 32));
  key = Mix(key, uint64_t(uint32_t(d.path->format)) | (uint64_t(d.base_vertex) << 8));
  key = Mix(key, uint64_t(d.indexed ? 1 : 0) | (uint64_t(d.index_count) << 1));
  key = Mix(key, uint64_t(d.ib_guest_base) | (uint64_t(uint32_t(d.ib_format)) << 32) |
                     (uint64_t(uint32_t(d.ib_endian)) << 40));
  auto it = entries_.find(key);
  Entry* e;
  if (it == entries_.end()) {
    if (entries_.size() >= kMaxEntries) {
      ++stats.bounds_full;
      return nullptr;
    }
    e = entries_.emplace(key, std::make_unique<Entry>()).first->second.get();
    ++stats.bounds_new;
  } else {
    e = it->second.get();
    if (e->valid.load(std::memory_order_acquire)) {
      return &e->bounds;
    }
    ++stats.bounds_invalidated;
  }
  {
    // Cancel the watches that did not fire (a fired watch's handle is dead),
    // under the global lock so none can fire in between.
    rex::thread::global_critical_region gcr;
    auto lock = gcr.Acquire();
    for (uint32_t a = 0; a < 2; ++a) {
      if (e->watch[a] && !e->fired[a].load(std::memory_order_acquire)) {
        shared_memory_->UnwatchMemoryRange(e->watch[a]);
      }
      e->watch[a] = nullptr;
      e->fired[a].store(false, std::memory_order_release);
    }
  }
  Compute(d, *e, stats);
  if (!e->bounds.unbounded) {
    Watch(*e);
  }
  e->valid.store(true, std::memory_order_release);
  return &e->bounds;
}

bool NrCull::FrustumOutside(const CullBounds& b, const float m[4][5]) {
  if (b.unbounded) return false;
  int out[4] = {0, 0, 0, 0};
  for (uint32_t c = 0; c < 8; ++c) {
    const float p[3] = {(c & 1) ? b.mx[0] : b.mn[0], (c & 2) ? b.mx[1] : b.mn[1],
                        (c & 4) ? b.mx[2] : b.mn[2]};
    float clip[4];
    for (uint32_t i = 0; i < 4; ++i) {
      clip[i] = m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3] * b.mn[3] + m[i][4];
    }
    const float w = clip[3];
    if (clip[0] + w < 0.0f) ++out[0];
    if (clip[0] - w > 0.0f) ++out[1];
    if (clip[1] + w < 0.0f) ++out[2];
    if (clip[1] - w > 0.0f) ++out[3];
  }
  return out[0] == 8 || out[1] == 8 || out[2] == 8 || out[3] == 8;
}

}  // namespace rex::graphics
