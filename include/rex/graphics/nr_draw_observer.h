#pragma once
#include <cstdint>

// Observation of the exact draw population counted by [n7], including copy
// draws. Install before guest startup. Callbacks must not allocate, block or
// log. Unregistering does not wait for callbacks already in flight.
using NrDrawObserver = void (*)(uint64_t ordinal, uint32_t packet_address,
                               uint32_t initiator, uint32_t edram_mode);
using NrDrawReportObserver = void (*)(uint64_t total_draws, uint64_t total_frames);
extern "C" void rex_nr_draw_observer(NrDrawObserver draw, NrDrawReportObserver report);

// Completed existing D3D12 census queries only. Flags: bit 0 samples valid,
// bit 1 pooled batch (ordinal identifies only its opener), bit 2 Hi-Z skip
// phase, bits 8..15 RT class. This population is not all received draws.
using NrDrawSamplesObserver = void (*)(uint64_t ordinal, uint64_t primitives,
    uint64_t vs_invocations, uint64_t ps_invocations, uint64_t samples, uint32_t flags);
extern "C" void rex_nr_draw_samples_observer(NrDrawSamplesObserver samples);

namespace rex::graphics::nr {
void ObserveReceivedDraw(uint64_t ordinal, uint32_t packet_address,
                         uint32_t initiator, uint32_t edram_mode);
void ObserveDrawReport(uint64_t total_draws, uint64_t total_frames);
void ObserveDrawSamples(uint64_t ordinal, uint64_t primitives, uint64_t vs_invocations,
                        uint64_t ps_invocations, uint64_t samples, uint32_t flags);
}
