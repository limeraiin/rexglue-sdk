#include <rex/graphics/nr_cpu_readback.h>
#include <atomic>
#include <cstring>
#include <mutex>

namespace {
std::mutex mutex;
std::atomic<bool> active{false};
rex::graphics::nr::CpuReadbackRequest current;
uint64_t serial = 0;
uint32_t copies = 0;
}

extern "C" uint64_t rex_nr_cpu_readback_begin(uint32_t physical, uint32_t size) {
  if (!physical || !size || uint64_t(physical) + size > 0x20000000ull) return 0;
  std::lock_guard lock(mutex);
  if (current.ticket) return 0;
  if (!++serial) ++serial;
  current = {serial, physical, size};
  copies = 0;
  active.store(true, std::memory_order_release);
  return serial;
}

extern "C" uint32_t rex_nr_cpu_readback_end(uint64_t ticket) {
  std::lock_guard lock(mutex);
  if (!ticket || ticket != current.ticket) return 0;
  current = {};
  active.store(false, std::memory_order_release);
  return copies;
}

namespace rex::graphics::nr {
CpuReadbackRequest GetCpuReadbackRequest() {
  if (!active.load(std::memory_order_acquire)) return {};
  std::lock_guard lock(mutex);
  return current;
}
bool CommitCpuReadback(const CpuReadbackRequest& request, uint32_t address,
                       uint32_t length, void* destination, const void* source) {
  if (!destination || !source || !request.Contains(address, length)) return false;
  std::lock_guard lock(mutex);
  if (current.ticket != request.ticket) return false;
  std::memcpy(destination, source, length);
  ++copies;
  return true;
}
}
