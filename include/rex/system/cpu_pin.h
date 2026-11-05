/**
 ******************************************************************************
 * ReXGlue - [cpu-pin] CP thread and guest threads on distinct cores          *
 ******************************************************************************
 *
 * Cvar cpu_pin (src/graphics/command_processor.cpp) 1 = the command-processor
 * thread gets one physical core to itself (both SMT siblings reserved), 2 =
 * only that core's first logical processor (sibling left idle by us), and
 * every guest thread gets the other cores, so a hot guest thread can never
 * share the CP core's hyperthread. Host threads are NOT moved: run 979
 * moved all 19 of them to the guest cores and tripled the slow frames
 * (guest starvation, and an external process then landed on the CP core's
 * sibling; the [ftrace] v5 sib_busy_ms column reads it). The plan is pure
 * (tested offline in
 * tools/test-frame-trace.py); the Windows topology query, the thread calls
 * and the one [cpu-pin] log line live in src/system/xthread.cpp.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace rex::system {

class KernelState;

namespace cpu_pin {

// The cvar's help text (one place, checked by tools/test-frame-trace.py).
constexpr char kModeHelp[] =
    "0 = off, 1 = CP thread alone on one physical core, guest threads on the rest, "
    "2 = CP thread on one logical processor, its SMT sibling left idle";

struct Plan {
  uint64_t cp_mask = 0;     // both logical siblings of the CP core
  uint64_t guest_mask = 0;  // every other logical processor
  uint32_t cp_ideal = 0;    // the CP core's first logical processor
  uint32_t cores = 0;       // physical cores seen
  uint32_t logical = 0;     // logical processors seen
};

// core_masks: one logical-processor mask per physical core (SMT siblings
// share one). The CP takes the LAST core (core 0 carries the OS interrupt and
// DPC work), the guest takes everything else. False with fewer than 2 cores,
// fewer than 4 logical processors or an empty core mask: leave everything
// unpinned.
// single: the CP keeps only the core's first logical processor (mode 2), so
// even sibling hops stop; the sibling stays out of the guest mask.
inline bool PlanFromCores(const uint64_t* core_masks, size_t count, Plan* out,
                          bool single = false) {
  if (!core_masks || !out || count < 2) return false;
  uint64_t all = 0;
  uint32_t logical = 0;
  for (size_t i = 0; i < count; ++i) {
    if (!core_masks[i] || (all & core_masks[i])) return false;
    all |= core_masks[i];
    for (uint64_t m = core_masks[i]; m; m &= m - 1) ++logical;
  }
  if (logical < 4) return false;
  const uint64_t cp = core_masks[count - 1];
  uint32_t ideal = 0;
  while (!((cp >> ideal) & 1)) ++ideal;
  out->cp_mask = single ? (uint64_t(1) << ideal) : cp;
  out->guest_mask = all & ~cp;
  out->cp_ideal = ideal;
  out->cores = uint32_t(count);
  out->logical = logical;
  return true;
}

// Runtime (src/system/xthread.cpp). Windows only; elsewhere Enable returns
// false and nothing is pinned.
// Called once on the CP thread: queries the topology, pins the calling
// thread as the CP, applies the guest mask to the guest threads that
// already exist and to every guest thread created afterwards, logs one
// [cpu-pin] line. False (with a log line) when the topology is unusable.
// Mode 0 never calls this.
bool Enable(KernelState* kernel_state, int mode);
bool active();
uint64_t guest_mask();

}  // namespace cpu_pin
}  // namespace rex::system
