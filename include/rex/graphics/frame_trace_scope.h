/**
 ******************************************************************************
 * ReXGlue - [ftrace] v3 exec-split brackets (runtime side)                   *
 ******************************************************************************
 *
 * RAII brackets over the pure accounting in frame_trace.h. Kept apart from
 * that header so it stays standalone (no clock). Every scope reads
 * g_exec_split once at construction: below mode 3 a bracket is one
 * predictable branch and nothing else. CP thread only.
 */

#pragma once

#include <cstdint>

#include <rex/chrono/clock.h>
#include <rex/graphics/frame_trace.h>

namespace rex::graphics::ftrace {

// One split bucket's self time for the enclosing block, plus an optional
// count. Nesting is handled by the ledger (see SplitTotals).
class SplitScope {
 public:
  explicit SplitScope(uint64_t SplitTotals::*bucket, uint64_t SplitTotals::*count = nullptr)
      : on_(g_exec_split), bucket_(bucket), count_(count) {
    if (on_) mark_ = SplitEnter(g_split, rex::chrono::Clock::QueryHostTickCount());
  }
  ~SplitScope() {
    if (!on_) return;
    SplitLeave(g_split, mark_, rex::chrono::Clock::QueryHostTickCount(), g_split.*bucket_);
    if (count_) ++(g_split.*count_);
  }
  SplitScope(const SplitScope&) = delete;
  SplitScope& operator=(const SplitScope&) = delete;

 private:
  const bool on_;
  uint64_t SplitTotals::*bucket_;
  uint64_t SplitTotals::*count_;
  SplitMark mark_{};
};

// A HitchProbe wait that just finished: charged to the enclosing bracket as
// child time so no split bucket counts blocking as work.
inline void SplitWait(uint64_t wait_ticks) {
  if (g_exec_split) SplitChargeWait(g_split, wait_ticks);
}

}  // namespace rex::graphics::ftrace
