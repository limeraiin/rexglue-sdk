/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_DRAW_REGISTRY_H_
#define REX_GRAPHICS_NR_DRAW_REGISTRY_H_

#include <cstdint>

// [NR-REG] Address-keyed draw registry: the first component of the native
// renderer, and the instrument that decides whether the native renderer is
// possible at all.
//
// Naruto does not draw to the ring. Its D3D9 layer records draws into
// PM4_INDIRECT_BUFFERs which the GPU then replays, sometimes for many frames
// after the recording (measured in the city: ~102 buffers at stable addresses,
// the busiest executing once per frame while being re-recorded roughly every
// third frame). A renderer that hooks the D3D9 API sees only the RECORD event,
// so it cannot draw what it sees when it sees it. It has to keep the draw list
// it captured, key it by the address it was captured at, and replay that list
// when the buffer runs. This registry is that cache, reduced to counting.
//
// The gate it answers: is every EXECUTED draw attributable to exactly one
// RECORDED draw? Correlating the two offline was tried and shown impossible --
// inside a one-second window the buffers overlap 6.4x in address space, so an
// address does not identify a buffer once ordering is discarded. In process the
// ambiguity disappears, because at the instant a buffer executes, the memory it
// points at holds exactly one recording.
//
// Design notes that are load-bearing:
//
//   * Granules, not buffers. The recorder never announces a buffer boundary,
//     and Ch.9 established that a boundary advances the write pointer
//     monotonically, making it indistinguishable from an ordinary advance. Any
//     boundary heuristic would merge buffers and fail the gate for a reason
//     that is a property of the probe rather than the engine. So the registry
//     bins by fixed 16KB granules of physical address and lets the executing
//     buffer supply its own bounds.
//
//   * Entering a granule invalidates it. The recorder walks forward through one
//     ring, so arriving in a granule always means starting to overwrite it.
//     That single rule is the whole invalidate-on-re-record policy and it costs
//     one comparison per draw.
//
//   * Physical addresses. Record sees whichever alias the guest used
//     (0xA0000000 in this title), execute sees a physical address, so both are
//     masked rather than related by a measured constant. Stats::alias is the
//     check on that assumption: if the mask is wrong, every query returns zero.

namespace rex {
namespace graphics {
namespace nr {

// 16KB granules, matching the guest-side [nrmap] histogram. A city indirect
// buffer measures ~196KB, so about twelve granules per buffer.
constexpr uint32_t kGranuleShift = 14;
constexpr uint32_t kGranuleBytes = 1u << kGranuleShift;

struct QueryResult {
  uint32_t draws;          // draws the registry holds across the range
  uint32_t granules_hit;   // granules present in the registry
  uint32_t granules_miss;  // granules never recorded, or overwritten elsewhere
  // Places where recording order runs BACKWARDS from one granule to the next.
  // A buffer written in a single forward pass has none, so any inversion means
  // the registry was caught holding part of one recording and part of another.
  // This replaced a spread-vs-threshold test, which could not tell the two
  // apart: a clean recording's spread is already about its own draw count, so a
  // tear barely moves it. Inversions separate them exactly and need no
  // threshold.
  uint32_t seq_inversions;
};

struct Stats {
  uint64_t recorded;      // draws registered
  uint64_t collisions;    // granules displaced by a direct-map conflict
  // Recorded addresses by physical alias, i.e. the top three bits. Indexing by
  // the top NIBBLE would fold bit 28 of the physical address into the answer
  // and report an alias the guest never used.
  uint64_t alias[8];
};

// Ask what was recorded into [phys_addr, phys_addr + bytes). Cost is one probe
// per granule the range covers. Safe to call while recording continues; a torn
// granule costs one sample and cannot move a rate.
QueryResult QueryRange(uint32_t phys_addr, uint32_t bytes);

const Stats& GetStats();
void ResetStats();

// Drop every recording. Only for tests -- in the running game the registry is
// meant to persist exactly as a renderer's draw-list cache would.
void Reset();

}  // namespace nr
}  // namespace graphics
}  // namespace rex

// Called once per recorded draw from the guest D3D9 recorder hook
// ([NARUTO-NRMAP], naruto-recomp/src/record_map.cpp), with the device's
// command-buffer write pointer. Runs on a guest thread at raised IRQL, so it
// must not log, allocate or lock -- and does none of those.
extern "C" void rex_nr_record_draw(uint32_t guest_addr);

#endif  // REX_GRAPHICS_NR_DRAW_REGISTRY_H_
