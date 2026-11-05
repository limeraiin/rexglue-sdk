// R2 native submission seam, consumer side (see nr_native_submit.h).
// The ring pointer and slot count are published by the producer thread and
// read by the CP thread with acquire loads. The counters are written only on
// the CP thread (relaxed atomic stores, plain moves on x64) and snapshotted by
// rex_nr_native_counters from any thread.
#include <rex/graphics/nr_native_submit.h>
#include <rex/logging.h>

#include <atomic>
#include <cstddef>
#include <cstring>

namespace {
using namespace rex::graphics::nr_native;

std::atomic<uint8_t*> g_ring{nullptr};
std::atomic<uint32_t> g_slots{0};
std::atomic<int> g_phase{-1};
static_assert(std::atomic<uint8_t*>::is_always_lock_free);

NrNativeCounters g_counters = {};
uint64_t g_reported_markers = 0;

uint64_t Load(const uint64_t& f) {
  return std::atomic_ref<uint64_t>(const_cast<uint64_t&>(f)).load(std::memory_order_relaxed);
}
}  // namespace

extern "C" void rex_nr_native_install(void* ring, uint32_t slots) {
  g_ring.store(nullptr, std::memory_order_release);
  g_slots.store(0, std::memory_order_release);
  if (ring || slots) {
    REXGPU_WARN("[jade-native-cp] incompatible legacy ring install refused; rebuild the producer for ABI {}",
                kNrNativeAbiVersion);
  }
}

extern "C" uint32_t rex_nr_native_install_checked(void* ring, uint32_t slots, uint32_t abi,
                                                  uint32_t record_bytes, uint32_t slot_bytes) {
  g_ring.store(nullptr, std::memory_order_release);
  g_slots.store(0, std::memory_order_release);
  if (!NrNativeLayoutValid(abi, record_bytes, slot_bytes) ||
      bool(ring) != bool(slots) || slots > 65536u) {
    REXGPU_WARN("[jade-native-cp] incompatible ring install refused: abi={} record_bytes={} slot_bytes={} slots={}",
                abi, record_bytes, slot_bytes, slots);
    return 0;
  }
  // Slots first, ring last (release): a CP that sees the ring sees its size.
  g_slots.store(ring ? slots : 0, std::memory_order_release);
  g_ring.store(static_cast<uint8_t*>(ring), std::memory_order_release);
  return 1;
}

extern "C" void rex_nr_native_phase(int phase) {
  g_phase.store(phase < 0 ? -1 : (phase > 3 ? 3 : phase), std::memory_order_release);
}

extern "C" void rex_nr_native_counters(NrNativeCounters* out) {
  if (!out) return;
  constexpr size_t kFields = sizeof(NrNativeCounters) / sizeof(uint64_t);
  static_assert(sizeof(NrNativeCounters) == kFields * sizeof(uint64_t));
  const uint64_t* src = reinterpret_cast<const uint64_t*>(&g_counters);
  uint64_t* dst = reinterpret_cast<uint64_t*>(out);
  for (size_t i = 0; i < kFields; ++i) dst[i] = Load(src[i]);
}

namespace rex::graphics::nr_native {

NrNativeCounters& NrNativeCpCounters() { return g_counters; }

void NrNativeBump(uint64_t NrNativeCounters::*field, uint64_t n) {
  uint64_t& f = g_counters.*field;
  std::atomic_ref<uint64_t>(f).store(f + n, std::memory_order_relaxed);
}

int NrNativePhase() { return g_phase.load(std::memory_order_acquire); }

static NrNativeVerdict NrNativeAcquireImpl(uint32_t slot_sequence, NrNativeDrawRecord* out,
                                         bool* torn, bool filter_payload, bool band_allowed) {
  *torn = false;
  uint8_t* ring = g_ring.load(std::memory_order_acquire);
  const uint32_t slots = g_slots.load(std::memory_order_acquire);
  if (!ring) return NrNativeCheck(false, slots, slot_sequence, 0);
  if (NrNativeMarkerSlot(slot_sequence) >= slots) {
    return NrNativeCheck(true, slots, slot_sequence, 0);
  }
  auto* rec = reinterpret_cast<NrNativeDrawRecord*>(
      ring + size_t(NrNativeMarkerSlot(slot_sequence)) * kNrNativeSlotSize);
  const uint32_t seq0 =
      std::atomic_ref<uint32_t>(rec->sequence).load(std::memory_order_acquire);
  const NrNativeVerdict v = NrNativeCheck(true, slots, slot_sequence, seq0);
  if (v != NrNativeVerdict::kOk) return v;
  // Header (everything before the arrays), then only the used entries.
  constexpr size_t kHeader = offsetof(NrNativeDrawRecord, registers);
  std::memcpy(out, rec, kHeader);
  const uint32_t nr = out->register_count <= kNrNativeMaxRegisters ? out->register_count : 0;
  const uint32_t nc = out->constant_count <= kNrNativeMaxConstants ? out->constant_count : 0;
  if (!filter_payload || (band_allowed && !(out->flags & kNrNativeVerifyOnly))) {
    if (nr) std::memcpy(out->registers, rec->registers, nr * sizeof(NrNativeRegister));
    if (nc) std::memcpy(out->constants, rec->constants, nc * sizeof(NrNativeConstant));
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  const uint32_t seq1 =
      std::atomic_ref<uint32_t>(rec->sequence).load(std::memory_order_relaxed);
  if (seq1 != seq0) {
    *torn = true;
    return NrNativeVerdict::kSequence;
  }
  out->sequence = seq0;
  return NrNativeVerdict::kOk;
}

NrNativeVerdict NrNativeAcquire(uint32_t slot_sequence, NrNativeDrawRecord* out, bool* torn) {
  return NrNativeAcquireImpl(slot_sequence, out, torn, false, true);
}

NrNativeVerdict NrNativeAcquireForDraw(uint32_t slot_sequence, NrNativeDrawRecord* out,
                                     bool* torn, bool band_allowed) {
  return NrNativeAcquireImpl(slot_sequence, out, torn, true, band_allowed);
}

void NrNativeReport1Hz() {
  const NrNativeCounters& c = g_counters;
  if (c.markers == g_reported_markers) return;
  g_reported_markers = c.markers;
  // v2: CP microseconds per draw over this report window, for both paths
  // (fast_ns / fast_draws and v1_ns / v1_draws), so one run compares them.
  // Wall clocks run only with the detailed frame trace. -1 means unmeasured,
  // never a zero-cost draw. Draw and correctness counters stay unconditional.
  static uint64_t s_fast_ns = 0, s_fast_draws = 0, s_v1_ns = 0, s_v1_draws = 0;
  const uint64_t d_fast_ns = c.fast_ns - s_fast_ns, d_fast = c.fast_draws - s_fast_draws;
  const uint64_t d_v1_ns = c.v1_ns - s_v1_ns, d_v1 = c.v1_draws - s_v1_draws;
  s_fast_ns = c.fast_ns;
  s_fast_draws = c.fast_draws;
  s_v1_ns = c.v1_ns;
  s_v1_draws = c.v1_draws;
  [[maybe_unused]] const double us_fast = d_fast && d_fast_ns ? double(d_fast_ns) / 1000.0 / double(d_fast) : -1.0;
  [[maybe_unused]] const double us_v1 = d_v1 && d_v1_ns ? double(d_v1_ns) / 1000.0 / double(d_v1) : -1.0;
  REXGPU_INFO(
      "[jade-native-cp] markers={} drawn={} verified={} refused seq={} ring={} shader={} "
      "draw={} pred={} regs={} consts={} | slot={} torn={} record={} precord={} | "
      "skipped band={} viz={} reglow={} reghigh={} constrange={} | fast draws={} fallback={} "
      "entry hit/miss={}/{} pso hit/miss={}/{} shader hit/miss={}/{} tex hit/miss={}/{} "
      "rt_update_skipped={} cbv_bytes={} | us/draw fast={:.2f} ({}) v1={:.2f} ({}) | cmp "
      "markers={} eq={} ne={}",
      c.markers, c.drawn, c.verified, c.refused_sequence, c.refused_no_ring, c.refused_shader,
      c.refused_draw, c.refused_predicated, c.registers_applied, c.constants_applied,
      c.refused_slot, c.refused_torn, c.refused_record, c.refused_precord, c.skipped_band,
      c.skipped_viz, c.skipped_register_low, c.skipped_register_high, c.skipped_constant_range,
      c.fast_draws, c.fast_fallback, c.entry_hit, c.entry_miss, c.pso_hit, c.pso_miss,
      c.shader_hit, c.shader_miss, c.tex_hit, c.tex_miss, c.rt_update_skipped, c.cbv_bytes,
      us_fast, d_fast, us_v1, d_v1, c.cmp_markers, c.cmp_eq, c.cmp_ne);
}

}  // namespace rex::graphics::nr_native
