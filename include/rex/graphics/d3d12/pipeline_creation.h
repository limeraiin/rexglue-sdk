#pragma once
#include <atomic>
#include <cstdint>

namespace rex::graphics::d3d12 {
// Exactly one creator; completion is distinct from a null (failed) PSO.
class PipelineCreation {
 public:
  bool TryClaim() {
    uint8_t queued = 0;
    return state_.compare_exchange_strong(queued, 1, std::memory_order_acq_rel);
  }
  void Complete() {
    state_.store(2, std::memory_order_release);
    state_.notify_all();
  }
  void Wait() const {
    for (uint8_t state = state_.load(std::memory_order_acquire); state != 2;
         state = state_.load(std::memory_order_acquire)) state_.wait(state);
  }
 private:
  std::atomic<uint8_t> state_{0};
};
}  // namespace rex::graphics::d3d12
