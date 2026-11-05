/**
 ******************************************************************************
 * ReXGlue - [ftrace] per-frame timing trace                                  *
 ******************************************************************************
 *
 * One record per presented frame (the XE_SWAP packet on the command-processor
 * thread), kept in a fixed ring, summarized once a second and dumped to a CSV
 * beside the log at shutdown. The 1 Hz [n7]/[hitch] lines average a second;
 * this names WHICH frames were slow and what they were waiting on.
 *
 * This header is pure: no D3D12, no logging, no cvars. It compiles standalone
 * (tools/test-frame-trace.py) so the ring, the classifier and the CSV layout
 * are checked offline. The runtime glue (globals, cvar, dump) lives in
 * src/graphics/frame_trace.cpp.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <ostream>
#include <vector>

namespace rex::graphics::ftrace {

// A frame slower than this is "slow" in every summary.
constexpr double kSlowFrameMs = 25.0;

// Fixed layout: plain integers and doubles only. ms fields are host-clock
// deltas between consecutive swaps; counts are deltas of cumulative counters.
struct FrameRecord {
  uint64_t frame = 0;  // swap_counter after its increment
  double t_ms = 0.0;   // ms since trace start (host ticks)
  double dt_ms = 0.0;  // interval since the previous swap (0 for the first)
  // Command-processor thread wall clock split (base CP brackets).
  double exec_ms = 0.0;  // ExecutePrimaryBuffer time
  double spin_ms = 0.0;  // starvation spin/wait time
  double idle_ms = 0.0;  // max(0, dt - exec - spin)
  uint32_t starves = 0;
  uint32_t wakes = 0;
  uint32_t draws = 0;
  // [hitch] blocking inside execution (D3D12), ms per frame.
  double fence_ms = 0.0;  // submission-fence waits
  uint32_t fence_n = 0;
  double qops_ms = 0.0;     // out-of-submission queue-operation waits
  double psowait_ms = 0.0;  // EndSubmission wait for the creation threads
  double pso_cp_ms = 0.0;   // PSO creates on the CP thread
  uint32_t pso_cp_n = 0;
  double pso_thr_ms = 0.0;  // PSO creates on the creation threads
  uint32_t pso_thr_n = 0;
  uint32_t lib_miss = 0;  // pipeline-library misses (driver compiles)
  double xlat_ms = 0.0;   // shader translation (any thread)
  uint32_t xlat_n = 0;
  double swap_ms = 0.0;  // IssueSwap time
  // Uploads recorded this frame.
  uint32_t texup_n = 0;
  uint64_t texup_bytes = 0;
  uint32_t smup_n = 0;
  uint64_t smup_bytes = 0;
  // GPU time of this frame's submissions (mode 2 only, -1 until filled).
  double gpu_busy_ms = -1.0;  // sum of (close - open) over its submissions
  double gpu_span_ms = -1.0;  // last close minus first open
  int32_t phase = -1;  // nr_native::NrNativePhase() at the swap, -1 = none
  // v2 (appended): the guest's WAIT_REG_MEM blocking inside the exec bracket
  // (vblank wait, Sleep or yield while unmatched) and the CP thread's own
  // CPU cycles between swaps (QueryThreadCycleTime, millions). On a slow
  // frame, cycles at the median means the thread slept or was descheduled;
  // cycles up means it did more work.
  double wait_ms = 0.0;
  uint32_t wait_n = 0;    // WAIT_REG_MEM packets that waited at least once
  double cpu_mcyc = 0.0;  // 0 off Windows
  // v3 (appended): the exec split, mode 3 only (0 below it). Each bucket is
  // the CP thread's SELF time in that bracket: nested brackets and the
  // HitchProbe waits (fence, qops, psowait) are charged to the enclosing one
  // as child time, so draw/endsub/texup/smup are disjoint and
  //   exec - wait - fence - qops - psowait - draw - endsub - texup - smup
  // is the walk and everything else outside those four.
  double draw_ms = 0.0;    // IssueDraw (the one base-CP dispatch site)
  uint32_t draw_n = 0;
  double endsub_ms = 0.0;  // D3D12 EndSubmission net of its waits
  uint32_t endsub_n = 0;
  double texup_ms = 0.0;   // texture-cache resident-memory loads
  double smup_ms = 0.0;    // shared-memory UploadRanges
  // Upload-pool page creations (new committed resources) by every pool, with
  // the time inside CreatePageImplementation. NOT disjoint: a page is created
  // inside whichever bracket requested it.
  uint32_t upalloc_n = 0;
  double upalloc_ms = 0.0;
  // Process page faults between swaps (any mode on Windows; 0 elsewhere).
  uint32_t pf = 0;
  // v4 (appended): where the CP thread ran. cpu_id is its logical processor
  // at the swap (-1 = never sampled); migr counts processor changes seen at
  // the sampled sites (wake, every draw, the swap) during the frame. A slow
  // frame with migr > 0 at the same work moved cores; the SMT sibling of the
  // new core may have been busy.
  int32_t cpu_id = -1;
  uint32_t migr = 0;
  // v4: the native executor's stage self times (mode 3, fast path; 0 below
  // it) and the records it executed. Inside exec, outside draw/endsub.
  double nr_lookup_ms = 0.0;
  double nr_derive_ms = 0.0;
  double nr_const_ms = 0.0;
  double nr_bind_ms = 0.0;
  double nr_emit_ms = 0.0;
  uint32_t nr_n = 0;
  // v5 (appended): busy time of the CP thread's SMT sibling logical
  // processor during the frame (kernel + user, idle excluded, from the
  // system processor times sampled at each swap) and the same for the CP's
  // own processor (sanity: near dt while the CP never sleeps). -1 = unknown
  // (first swap, no sibling on a one-way core, off Windows). The system
  // processor times advance by the 15.625 ms kernel tick, so a frame's
  // value is 0, 15.625, 31.25 or 46.875: read it as "the sibling was busy
  // at least one tick this frame", never as a duration. Run 978: slow
  // frames with no migration and twice the cycles for the same draws point
  // at the sibling; run 979 read 15.6-31.2 ms there on every such frame.
  double sib_busy_ms = -1.0;
  double own_busy_ms = -1.0;
};

// --- SMT sibling busy (v5) --------------------------------------------------
// Pure. sibling[i] = the other logical processor of i's physical core when
// the core has exactly two (SMT2), kNoSibling otherwise (one-way cores,
// SMT4, processors outside every core mask). core_masks as in cpu_pin.h.
constexpr int32_t kNoSibling = -1;

inline void SiblingTableFromCores(const uint64_t* core_masks, size_t count, int32_t* sibling,
                                  size_t logical) {
  for (size_t i = 0; i < logical; ++i) sibling[i] = kNoSibling;
  if (!core_masks || !sibling) return;
  for (size_t c = 0; c < count; ++c) {
    uint64_t m = core_masks[c];
    if (!m || (m & (m - 1)) == 0) continue;  // empty or one logical processor
    uint64_t rest = m & (m - 1);
    if (rest & (rest - 1)) continue;  // three or more: no single sibling
    uint32_t a = 0, b = 0;
    while (!((m >> a) & 1)) ++a;
    while (!((rest >> b) & 1)) ++b;
    if (a < logical && b < logical) {
      sibling[a] = int32_t(b);
      sibling[b] = int32_t(a);
    }
  }
}

// Processor times in 100 ns units (SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION:
// on Windows KernelTime includes IdleTime).
struct ProcTimes {
  uint64_t idle = 0;
  uint64_t kernel = 0;
  uint64_t user = 0;
};

// busy = (kernel - idle + user) delta in ms; a delta that went backwards
// (a counter reset) reads 0, never negative.
inline double BusyDeltaMs(const ProcTimes& prev, const ProcTimes& now) {
  const double busy_now = double(now.kernel) - double(now.idle) + double(now.user);
  const double busy_prev = double(prev.kernel) - double(prev.idle) + double(prev.user);
  const double d = busy_now - busy_prev;
  return d > 0.0 ? d / 1e4 : 0.0;
}

// --- processor sampling (v4) -----------------------------------------------
// prev_cpu starts at kNoCpu; the first sample sets it and counts nothing.
// Each later sample on a different processor counts one migration. Returns
// true when it counted. Pure; the caller supplies the processor number.
constexpr uint32_t kNoCpu = 0xFFFFFFFFu;

inline bool MigrNote(uint32_t now_cpu, uint32_t& prev_cpu, uint64_t& count) {
  if (now_cpu == prev_cpu) return false;
  const bool counted = prev_cpu != kNoCpu;
  prev_cpu = now_cpu;
  if (counted) ++count;
  return counted;
}

// --- exec split accounting (v3) ---------------------------------------------
// Cumulative host ticks and counts written by the bracket sites on the CP
// thread while g_exec_split is set; plain fields, no atomics, the swap takes
// deltas. child_ticks is the nesting ledger: a bracket that closes adds its
// WHOLE duration to it, and a bracket reads the ledger's growth while it was
// open as "time spent in nested brackets", which it subtracts from its own.
// The HitchProbe wait sites add their wait to the ledger too, so every
// bucket is net of fence/qops/psowait blocking. The ledger's absolute value
// means nothing; only differences do.
struct SplitTotals {
  uint64_t draw_ticks = 0;
  uint64_t draw_n = 0;
  uint64_t endsub_ticks = 0;
  uint64_t endsub_n = 0;
  uint64_t texup_ticks = 0;
  uint64_t smup_ticks = 0;
  // Native executor stages (fast path, outside the IssueDraw bracket):
  // object lookup, state derivation, constant packing, binding, emit.
  uint64_t nr_lookup_ticks = 0;
  uint64_t nr_derive_ticks = 0;
  uint64_t nr_const_ticks = 0;
  uint64_t nr_bind_ticks = 0;
  uint64_t nr_emit_ticks = 0;
  uint64_t nr_n = 0;
  uint64_t child_ticks = 0;
};

struct SplitMark {
  uint64_t t0 = 0;
  uint64_t child0 = 0;
};

inline SplitMark SplitEnter(const SplitTotals& s, uint64_t now_ticks) {
  SplitMark m;
  m.t0 = now_ticks;
  m.child0 = s.child_ticks;
  return m;
}

// Closes a bracket: charges its self time (duration minus the ledger growth
// while open) to `bucket`, then reports the whole duration to the enclosing
// bracket through the ledger. Returns the self ticks.
inline uint64_t SplitLeave(SplitTotals& s, const SplitMark& m, uint64_t now_ticks,
                           uint64_t& bucket) {
  const uint64_t dur = now_ticks >= m.t0 ? now_ticks - m.t0 : 0;
  const uint64_t nested = s.child_ticks - m.child0;
  const uint64_t self = dur > nested ? dur - nested : 0;
  bucket += self;
  s.child_ticks = m.child0 + dur;
  return self;
}

// A blocking wait that is already its own column: charge it to the enclosing
// bracket as child time so no bucket counts it as work.
inline void SplitChargeWait(SplitTotals& s, uint64_t wait_ticks) { s.child_ticks += wait_ticks; }

// idle = what the CP thread did outside both brackets, never negative (the
// open exec bracket is charged at the swap, so rounding can overshoot dt).
inline double IdleMs(double dt_ms, double exec_ms, double spin_ms) {
  const double idle = dt_ms - exec_ms - spin_ms;
  return idle > 0.0 ? idle : 0.0;
}

// Fixed-capacity ring, allocated once on the first Push. Frames arrive in
// order with consecutive frame numbers, which makes the FillGpu lookup O(1).
class FrameTraceRing {
 public:
  static constexpr uint32_t kCapacity = 65536;  // power of two
  static constexpr uint32_t kMask = kCapacity - 1;

  bool allocated() const { return slots_ != nullptr; }
  size_t capacity() const { return kCapacity; }
  // Records currently retained (<= kCapacity).
  size_t size() const { return size_t(std::min<uint64_t>(pushed_, kCapacity)); }
  // Records pushed in total (the next record's sequence number).
  uint64_t pushed() const { return pushed_; }
  // Sequence number of the oldest retained record.
  uint64_t oldest_seq() const { return pushed_ > kCapacity ? pushed_ - kCapacity : 0; }

  // Appends a record. Allocates the ring on first use (returns false only
  // when that allocation fails).
  bool Push(const FrameRecord& record) {
    if (!slots_) {
      slots_.reset(new (std::nothrow) FrameRecord[kCapacity]);
      if (!slots_) return false;
    }
    slots_[pushed_ & kMask] = record;
    ++pushed_;
    return true;
  }

  // Oldest first, i in [0, size()).
  const FrameRecord& At(size_t i) const { return slots_[(oldest_seq() + i) & kMask]; }
  FrameRecord& At(size_t i) { return slots_[(oldest_seq() + i) & kMask]; }

  // By sequence number; null when not retained.
  const FrameRecord* Seq(uint64_t seq) const {
    if (!slots_ || seq >= pushed_ || seq < oldest_seq()) return nullptr;
    return &slots_[seq & kMask];
  }

  // By frame number; null when the frame was never pushed, is newer than the
  // newest record or has already been overwritten.
  FrameRecord* Find(uint64_t frame) {
    if (!slots_ || pushed_ == 0) return nullptr;
    const FrameRecord& newest = slots_[(pushed_ - 1) & kMask];
    if (frame > newest.frame) return nullptr;
    const uint64_t back = newest.frame - frame;
    if (back >= size()) return nullptr;
    FrameRecord& r = slots_[(pushed_ - 1 - back) & kMask];
    return r.frame == frame ? &r : nullptr;
  }
  const FrameRecord* Find(uint64_t frame) const {
    return const_cast<FrameTraceRing*>(this)->Find(frame);
  }

  // Fills the GPU columns of a retained frame. False = dropped (the caller
  // counts it).
  bool FillGpu(uint64_t frame, double busy_ms, double span_ms) {
    FrameRecord* r = Find(frame);
    if (!r) return false;
    r->gpu_busy_ms = busy_ms;
    r->gpu_span_ms = span_ms;
    return true;
  }

 private:
  std::unique_ptr<FrameRecord[]> slots_;
  uint64_t pushed_ = 0;
};

// Dominant-component classes. Spin is folded into idle (both are the CP
// waiting on the guest).
enum class FrameClass : uint8_t {
  kExec = 0,
  kGpuFence,
  kPso,
  kXlat,
  kIdle,
  kSwap,
  kWait,
  kMixed,
  kCount
};
constexpr size_t kFrameClassCount = size_t(FrameClass::kCount);

inline const char* FrameClassName(FrameClass c) {
  switch (c) {
    case FrameClass::kExec: return "exec";
    case FrameClass::kGpuFence: return "gpu_fence";
    case FrameClass::kPso: return "pso";
    case FrameClass::kXlat: return "xlat";
    case FrameClass::kIdle: return "idle";
    case FrameClass::kSwap: return "swap";
    case FrameClass::kWait: return "wait";
    default: return "mixed";
  }
}

// Component values used by the classifier. fence/pso/xlat/swap/wait are paid
// INSIDE the exec bracket, so exec is reduced by them first: a frame whose
// exec was mostly a fence wait is a gpu_fence frame, not an exec frame, and
// one that sat in the guest's vblank WAIT_REG_MEM is a wait frame.
struct FrameComponents {
  double exec, spin, idle, fence, pso, xlat, swap, wait;
};

inline FrameComponents Components(const FrameRecord& r) {
  FrameComponents c;
  c.fence = r.fence_ms + r.qops_ms;
  c.pso = r.psowait_ms + r.pso_cp_ms;
  c.xlat = r.xlat_ms;
  c.swap = r.swap_ms;
  c.wait = r.wait_ms;
  c.exec = std::max(0.0, r.exec_ms - c.fence - c.pso - c.xlat - c.swap - c.wait);
  c.spin = r.spin_ms;
  c.idle = r.idle_ms;
  return c;
}

// The largest component names the class when it also exceeds 40% of dt;
// otherwise the frame is mixed. Ties go to the earlier component.
inline FrameClass Classify(const FrameRecord& r) {
  const FrameComponents c = Components(r);
  const double values[8] = {c.exec, c.spin, c.idle, c.fence, c.pso, c.xlat, c.swap, c.wait};
  const FrameClass classes[8] = {FrameClass::kExec, FrameClass::kIdle, FrameClass::kIdle,
                                 FrameClass::kGpuFence, FrameClass::kPso, FrameClass::kXlat,
                                 FrameClass::kSwap, FrameClass::kWait};
  int best = 0;
  for (int i = 1; i < 8; ++i) {
    if (values[i] > values[best]) best = i;
  }
  if (r.dt_ms <= 0.0 || values[best] <= 0.4 * r.dt_ms) return FrameClass::kMixed;
  return classes[best];
}

struct FrameSummary {
  uint64_t count = 0;
  double p50 = 0.0, p95 = 0.0, p99 = 0.0, max = 0.0;
  uint64_t slow = 0;  // dt_ms > kSlowFrameMs
  FrameRecord worst{};  // the record with the largest dt (valid when count > 0)
  uint64_t by_class[kFrameClassCount] = {};
};

// Nearest-rank percentile over an ascending vector (q in [0, 1]).
inline double Percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  double rank = std::ceil(q * double(sorted.size()));
  size_t index = rank < 1.0 ? 0 : size_t(rank) - 1;
  if (index >= sorted.size()) index = sorted.size() - 1;
  return sorted[index];
}

// Summary over the retained records with sequence numbers in
// [begin_seq, end_seq). The range is clamped to what the ring still holds.
inline FrameSummary Summarize(const FrameTraceRing& ring, uint64_t begin_seq, uint64_t end_seq) {
  FrameSummary s;
  begin_seq = std::max(begin_seq, ring.oldest_seq());
  end_seq = std::min(end_seq, ring.pushed());
  if (begin_seq >= end_seq) return s;
  std::vector<double> dts;
  dts.reserve(size_t(end_seq - begin_seq));
  for (uint64_t seq = begin_seq; seq < end_seq; ++seq) {
    const FrameRecord& r = *ring.Seq(seq);
    dts.push_back(r.dt_ms);
    if (r.dt_ms > kSlowFrameMs) ++s.slow;
    if (s.count == 0 || r.dt_ms > s.worst.dt_ms) s.worst = r;
    ++s.by_class[size_t(Classify(r))];
    ++s.count;
  }
  std::sort(dts.begin(), dts.end());
  s.p50 = Percentile(dts, 0.50);
  s.p95 = Percentile(dts, 0.95);
  s.p99 = Percentile(dts, 0.99);
  s.max = dts.back();
  return s;
}

// --- CSV -------------------------------------------------------------------

struct CsvInfo {
  uint64_t freq = 0;          // host ticks per second
  int64_t start_wall_ms = 0;  // Unix epoch ms (UTC) at trace start
};

// The two header lines, each newline-terminated. Returns the length. v3
// appends the exec split and pf after cpu_mcyc, v4 appends the processor
// columns and the native stages after pf, v5 appends the sibling and own
// processor busy times after nr_n; an older reader still finds every
// column it knows in place.
inline int FormatCsvHeader(char* buf, size_t n, const CsvInfo& info) {
  return std::snprintf(buf, n,
                       "#ftrace v5 freq=%llu start_wall_ms=%lld start_t_ms=0 slow_ms=25\n"
                       "frame,t_ms,wall_ms,dt_ms,exec_ms,spin_ms,idle_ms,starves,wakes,draws,"
                       "fence_ms,fence_n,qops_ms,psowait_ms,pso_cp_ms,pso_cp_n,pso_thr_ms,"
                       "pso_thr_n,lib_miss,xlat_ms,xlat_n,swap_ms,texup_n,texup_kb,smup_n,"
                       "smup_kb,gpu_busy_ms,gpu_span_ms,phase,wait_ms,wait_n,cpu_mcyc,"
                       "draw_ms,draw_n,endsub_ms,endsub_n,texup_ms,smup_ms,upalloc_n,"
                       "upalloc_ms,pf,cpu_id,migr,nr_lookup_ms,nr_derive_ms,nr_const_ms,"
                       "nr_bind_ms,nr_emit_ms,nr_n,sib_busy_ms,own_busy_ms\n",
                       static_cast<unsigned long long>(info.freq),
                       static_cast<long long>(info.start_wall_ms));
}

// One newline-terminated row. ms values carry 3 decimals, counts are
// integers, upload sizes are integer KB, unfilled GPU columns print -1.000,
// cpu_mcyc carries 3 decimals (millions of cycles), cpu_id -1 when unsampled,
// the v5 busy columns -1.000 when unknown.
inline int FormatCsvRow(char* buf, size_t n, const FrameRecord& r, int64_t start_wall_ms) {
  return std::snprintf(
      buf, n,
      "%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%u,%u,%.3f,%u,%.3f,%.3f,%.3f,%u,%.3f,%u,%u,%.3f,%u,"
      "%.3f,%u,%llu,%u,%llu,%.3f,%.3f,%d,%.3f,%u,%.3f,%.3f,%u,%.3f,%u,%.3f,%.3f,%u,%.3f,%u,"
      "%d,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%.3f,%.3f\n",
      static_cast<unsigned long long>(r.frame), r.t_ms, double(start_wall_ms) + r.t_ms, r.dt_ms,
      r.exec_ms, r.spin_ms, r.idle_ms, r.starves, r.wakes, r.draws, r.fence_ms, r.fence_n,
      r.qops_ms, r.psowait_ms, r.pso_cp_ms, r.pso_cp_n, r.pso_thr_ms, r.pso_thr_n, r.lib_miss,
      r.xlat_ms, r.xlat_n, r.swap_ms, r.texup_n,
      static_cast<unsigned long long>(r.texup_bytes / 1024), r.smup_n,
      static_cast<unsigned long long>(r.smup_bytes / 1024), r.gpu_busy_ms, r.gpu_span_ms,
      r.phase, r.wait_ms, r.wait_n, r.cpu_mcyc, r.draw_ms, r.draw_n, r.endsub_ms, r.endsub_n,
      r.texup_ms, r.smup_ms, r.upalloc_n, r.upalloc_ms, r.pf, r.cpu_id, r.migr, r.nr_lookup_ms,
      r.nr_derive_ms, r.nr_const_ms, r.nr_bind_ms, r.nr_emit_ms, r.nr_n, r.sib_busy_ms,
      r.own_busy_ms);
}

// Whole ring, oldest first, through one stream with a local buffer.
inline bool WriteCsv(std::ostream& out, const FrameTraceRing& ring, const CsvInfo& info) {
  char line[1024];
  int len = FormatCsvHeader(line, sizeof(line), info);
  if (len <= 0) return false;
  out.write(line, len);
  const size_t count = ring.size();
  for (size_t i = 0; i < count; ++i) {
    len = FormatCsvRow(line, sizeof(line), ring.At(i), info.start_wall_ms);
    if (len <= 0 || size_t(len) >= sizeof(line)) return false;
    out.write(line, len);
  }
  return bool(out);
}

// --- log lines -------------------------------------------------------------

// The 1 Hz body (the caller prefixes "[ftrace] "). The worst frame's
// processor, migrations and the v5 sibling/own busy times always follow
// (cpu_id=, so the cycles' cpu= key stays unique; sib=/own= print n/a when
// unknown); with exec_split (mode 3) its split buckets, then the native
// stages when it executed native records.
inline int FormatReport1Hz(char* buf, size_t n, const FrameSummary& s, bool exec_split = false) {
  const FrameRecord& w = s.worst;
  const FrameComponents c = Components(w);
  char gpu[32], sib[32], own[32];
  if (w.gpu_busy_ms >= 0.0) {
    std::snprintf(gpu, sizeof(gpu), "%.1f", w.gpu_busy_ms);
  } else {
    std::snprintf(gpu, sizeof(gpu), "n/a");
  }
  if (w.sib_busy_ms >= 0.0) {
    std::snprintf(sib, sizeof(sib), "%.1f", w.sib_busy_ms);
  } else {
    std::snprintf(sib, sizeof(sib), "n/a");
  }
  if (w.own_busy_ms >= 0.0) {
    std::snprintf(own, sizeof(own), "%.1f", w.own_busy_ms);
  } else {
    std::snprintf(own, sizeof(own), "n/a");
  }
  int len = std::snprintf(buf, n,
                          "n=%llu p50=%.1f p95=%.1f p99=%.1f max=%.1f slow=%llu | worst #%llu "
                          "dt=%.1f exec=%.1f spin=%.1f idle=%.1f fence=%.1f pso=%.1f xlat=%.1f "
                          "swap=%.1f gpu=%s cls=%s wait=%.1f cpu=%.3f cpu_id=%d migr=%u "
                          "sib=%s own=%s",
                          static_cast<unsigned long long>(s.count), s.p50, s.p95, s.p99, s.max,
                          static_cast<unsigned long long>(s.slow),
                          static_cast<unsigned long long>(w.frame), w.dt_ms, w.exec_ms,
                          w.spin_ms, w.idle_ms, c.fence, c.pso, c.xlat, w.swap_ms, gpu,
                          FrameClassName(Classify(w)), w.wait_ms, w.cpu_mcyc, w.cpu_id, w.migr,
                          sib, own);
  if (!exec_split || len < 0 || size_t(len) >= n) return len;
  int more = std::snprintf(buf + len, n - size_t(len),
                           " draw=%.1f endsub=%.1f texup=%.1f smup=%.1f upalloc=%u pf=%u",
                           w.draw_ms, w.endsub_ms, w.texup_ms, w.smup_ms, w.upalloc_n, w.pf);
  if (more < 0) return more;
  len += more;
  if (w.nr_n == 0 || size_t(len) >= n) return len;
  more = std::snprintf(buf + len, n - size_t(len),
                       " nr=%u lookup=%.1f derive=%.1f const=%.1f bind=%.1f emit=%.1f", w.nr_n,
                       w.nr_lookup_ms, w.nr_derive_ms, w.nr_const_ms, w.nr_bind_ms, w.nr_emit_ms);
  return more < 0 ? more : len + more;
}

// The shutdown body (the caller prefixes "[ftrace] ").
inline int FormatShutdownLine(char* buf, size_t n, const FrameSummary& s, const char* path) {
  const uint64_t* b = s.by_class;
  return std::snprintf(buf, n,
                       "dumped %llu frames to %s | p50=%.1f p95=%.1f p99=%.1f max=%.1f slow=%llu "
                       "by class: exec=%llu gpu_fence=%llu pso=%llu xlat=%llu idle=%llu "
                       "swap=%llu wait=%llu mixed=%llu",
                       static_cast<unsigned long long>(s.count), path, s.p50, s.p95, s.p99, s.max,
                       static_cast<unsigned long long>(s.slow),
                       static_cast<unsigned long long>(b[size_t(FrameClass::kExec)]),
                       static_cast<unsigned long long>(b[size_t(FrameClass::kGpuFence)]),
                       static_cast<unsigned long long>(b[size_t(FrameClass::kPso)]),
                       static_cast<unsigned long long>(b[size_t(FrameClass::kXlat)]),
                       static_cast<unsigned long long>(b[size_t(FrameClass::kIdle)]),
                       static_cast<unsigned long long>(b[size_t(FrameClass::kSwap)]),
                       static_cast<unsigned long long>(b[size_t(FrameClass::kWait)]),
                       static_cast<unsigned long long>(b[size_t(FrameClass::kMixed)]));
}

// --- runtime glue (src/graphics/frame_trace.cpp) ---------------------------
// Declarations only, so this header stays standalone.

// gpu_frame_trace mode: 0 off, 1 CPU-side records, 2 adds GPU timestamps,
// 3 adds the exec split (v3 columns). The gate rule, pure so it is tested:
inline bool ExecSplitForMode(int mode) { return mode >= 3; }
int Mode();
// Worker-loop start: stamps the start ticks and the epoch, sets the mode and
// g_exec_split.
void Begin(int mode);
// Mode >= 3. Set once in Begin before any bracket runs; every split bracket
// tests it (one predictable branch below mode 3). Written by the CP thread.
extern bool g_exec_split;
// Mode >= 1. Gates the per-draw processor sample (one
// GetCurrentProcessorNumber per draw when on). Written by the CP thread.
extern bool g_on;
// The split ledger (CP thread only).
extern SplitTotals g_split;
// The ring, or null when the trace is off.
FrameTraceRing* Ring();
// Host ticks at trace start and the tick frequency.
uint64_t StartTicks();
uint64_t TickFrequency();
// The [ftrace] 1 Hz line over the frames pushed since the previous call.
// Silent when nothing was pushed.
void Report1Hz();
// CSV beside the log plus the summary line. Idempotent.
void DumpAtShutdown();

}  // namespace rex::graphics::ftrace

// Crash-filter entry: writes the CSV (".frames.crash.csv") once if the
// shutdown dump has not happened yet. Safe to call from any thread.
extern "C" void rex_frame_trace_dump();
