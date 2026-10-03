#pragma once
#include <cstdint>

// One explicitly owned CPU consumer. End cancels pending copies and waits for
// any CPU memcpy in progress, so the caller may then release the allocation.
extern "C" uint64_t rex_nr_cpu_readback_begin(uint32_t physical, uint32_t size);
extern "C" uint32_t rex_nr_cpu_readback_end(uint64_t ticket);

namespace rex::graphics::nr {
struct CpuReadbackRequest {
  uint64_t ticket = 0;
  uint32_t physical = 0;
  uint32_t size = 0;
  bool Contains(uint32_t address, uint32_t length) const {
    return ticket && length && address >= physical &&
           uint64_t(address) + length <= uint64_t(physical) + size;
  }
};
CpuReadbackRequest GetCpuReadbackRequest();
bool CommitCpuReadback(const CpuReadbackRequest& request, uint32_t address,
                       uint32_t length, void* destination, const void* source);
}
