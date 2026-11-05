#include <rex/graphics/nr_draw_observer.h>
#include <atomic>

namespace {
std::atomic<NrDrawObserver> draw_observer{nullptr};
std::atomic<NrDrawReportObserver> report_observer{nullptr};
std::atomic<NrDrawSamplesObserver> samples_observer{nullptr};
static_assert(std::atomic<NrDrawObserver>::is_always_lock_free);
}

extern "C" void rex_nr_draw_observer(NrDrawObserver draw, NrDrawReportObserver report) {
  draw_observer.store(draw, std::memory_order_release);
  report_observer.store(report, std::memory_order_release);
}

extern "C" void rex_nr_draw_samples_observer(NrDrawSamplesObserver samples) {
  samples_observer.store(samples, std::memory_order_release);
}

namespace rex::graphics::nr {
void ObserveReceivedDraw(uint64_t ordinal, uint32_t packet_address,
                         uint32_t initiator, uint32_t edram_mode) {
  if (const auto callback = draw_observer.load(std::memory_order_acquire)) {
    callback(ordinal, packet_address, initiator, edram_mode);
  }
}
void ObserveDrawReport(uint64_t total_draws, uint64_t total_frames) {
  if (const auto callback = report_observer.load(std::memory_order_acquire)) {
    callback(total_draws, total_frames);
  }
}
void ObserveDrawSamples(uint64_t ordinal, uint64_t primitives, uint64_t vs_invocations,
                        uint64_t ps_invocations, uint64_t samples, uint32_t flags) {
  if (const auto callback = samples_observer.load(std::memory_order_acquire)) {
    callback(ordinal, primitives, vs_invocations, ps_invocations, samples, flags);
  }
}
}
