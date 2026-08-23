/**
 ******************************************************************************
 * ReXGlue - [hitch] N-10b hitch probe                                        *
 ******************************************************************************
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace rex::graphics::d3d12 {

// [hitch] N-10b: 1 Hz totals of everything that can block the
// command-processor thread INSIDE execution. The naruto_629 iGPU hitch
// signature was cp exec=100% with wakes collapsing 120 -> 1-33, i.e. the
// thread was blocked, not busy (exec% counts blocking - the menu-anomaly
// lesson). These counters name the blocking site instead of guessing.
// Written from the CP thread (fence waits, EndSubmission pipeline-creation
// wait, draw-time PSO creates) and from the pipeline creation threads
// (worker-side creates), reported and reset once a second from the D3D12
// command processor's frame close. All _ticks fields are host ticks from
// rex::chrono::Clock::QueryHostTickCount, converted to ms at report time.
struct HitchProbe {
  // CheckSubmissionFence: the submission-fence wait (frame-open pacing and
  // every explicit await) and the out-of-submission queue-operations wait.
  std::atomic<uint64_t> fence_wait_ticks{0};
  std::atomic<uint64_t> fence_waits{0};
  std::atomic<uint64_t> qops_wait_ticks{0};
  std::atomic<uint64_t> qops_waits{0};
  // PipelineCache::EndSubmission: the CP thread waiting for the creation
  // threads to finish every queued pipeline (a mid-drive driver compile on
  // Intel blocks the CP right here).
  std::atomic<uint64_t> pso_endsub_wait_ticks{0};
  std::atomic<uint64_t> pso_endsub_waits{0};
  // Graphics PSO creation split by thread: on the CP thread itself (NR
  // draw-path creates, queue draining, boot-from-storage) vs on the
  // creation threads. cp_max is the single largest CP-thread create in the
  // window; singles over 5 ms are also logged at the create site.
  std::atomic<uint64_t> pso_cp_ticks{0};
  std::atomic<uint64_t> pso_cp_creates{0};
  std::atomic<uint64_t> pso_cp_max_ticks{0};
  std::atomic<uint64_t> pso_thr_ticks{0};
  std::atomic<uint64_t> pso_thr_creates{0};
  // [pso-lib] driver pipeline-blob library outcomes (any thread). A healthy
  // warm boot is all hits; misses are new-to-this-machine pipelines, each
  // one a driver compile that the store then banks for the next boot.
  std::atomic<uint64_t> lib_hits{0};
  std::atomic<uint64_t> lib_misses{0};
  std::atomic<uint64_t> lib_stores{0};
};

extern HitchProbe g_hitch_probe;

}  // namespace rex::graphics::d3d12
