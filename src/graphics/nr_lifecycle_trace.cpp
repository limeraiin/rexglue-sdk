#include <rex/graphics/nr_lifecycle_trace.h>
#include <atomic>

namespace {
std::atomic<NrLifecycleCallback> callback{nullptr};
std::atomic<uint64_t> next_id{0};
thread_local uint32_t depth = 0;
}
extern "C" void rex_nr_lifecycle_callback(NrLifecycleCallback value) {
  callback.store(value, std::memory_order_release);
}
namespace rex::graphics::nr {
LifecycleReplayScope::LifecycleReplayScope(uint32_t address, uint32_t dwords, uint32_t ring_index)
    : callback_(callback.load(std::memory_order_acquire)),
      address_(address), dwords_(dwords), ring_index_(ring_index) {
  if (!callback_) return;
  depth_ = depth++;
  id_ = next_id.fetch_add(1, std::memory_order_relaxed) + 1;
  callback_(true, id_, address_, dwords_, depth_, ring_index_);
}
LifecycleReplayScope::~LifecycleReplayScope() {
  if (!callback_) return;
  callback_(false, id_, address_, dwords_, depth_, ring_index_);
  --depth;
}
}
