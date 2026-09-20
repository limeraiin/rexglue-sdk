#pragma once
#include <cstdint>

// Observation only. Install before starting the guest; callbacks must not block,
// allocate or log. Removing the callback does not wait for in-flight scopes.
using NrLifecycleCallback = void (*)(bool begin, uint64_t id, uint32_t address,
                                    uint32_t dwords, uint32_t depth, uint32_t ring_index);
extern "C" void rex_nr_lifecycle_callback(NrLifecycleCallback callback);

namespace rex::graphics::nr {
class LifecycleReplayScope {
 public:
  LifecycleReplayScope(uint32_t address, uint32_t dwords, uint32_t ring_index);
  ~LifecycleReplayScope();
  LifecycleReplayScope(const LifecycleReplayScope&) = delete;
  LifecycleReplayScope& operator=(const LifecycleReplayScope&) = delete;
 private:
  NrLifecycleCallback callback_;
  uint64_t id_ = 0;
  uint32_t address_, dwords_, ring_index_, depth_ = 0;
};
}
