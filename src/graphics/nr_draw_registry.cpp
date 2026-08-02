/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_draw_registry.h"

// [NR-REG] See the header for what this is and why it is shaped this way.
//
// Deliberately unsynchronised. Recording runs on a guest thread inside the
// engine's raised-IRQL section and queries run on the command-processor thread;
// a mutex between them would both risk the lock inversion that already hung this
// game once and perturb the very timing being measured. The data is plain
// 32-bit words, torn reads cost a single sample out of ninety thousand a second,
// and every output is a rate.

namespace rex {
namespace graphics {
namespace nr {

namespace {

// Power of two. 8192 granules x 16KB = 128MB of distinct addresses, direct
// mapped; the command ring is a few MB and contiguous, so it cannot collide
// with itself. Stats::collisions is the proof rather than the assumption.
constexpr uint32_t kTableSize = 8192;
constexpr uint32_t kPhysMask = 0x1FFFFFFF;

struct Entry {
  uint32_t key;    // (physical address >> kGranuleShift) + 1; 0 means empty
  uint32_t draws;  // draws recorded since the recorder entered this granule
  uint32_t seq;    // global recording order at the last write
};

Entry g_tab[kTableSize] = {};
uint32_t g_cur_key = 0;  // granule the recorder is currently writing inside
uint32_t g_seq = 0;      // monotonic count of recorded draws
Stats g_stats = {};

}  // namespace

QueryResult QueryRange(uint32_t phys_addr, uint32_t bytes) {
  QueryResult r = {};
  if (!bytes) return r;
  const uint32_t p = phys_addr & kPhysMask;
  const uint32_t first = p >> kGranuleShift;
  const uint32_t last = (p + bytes - 1) >> kGranuleShift;
  uint32_t prev_seq = 0;
  bool have_prev = false;
  for (uint32_t g = first; g <= last; ++g) {
    const Entry& e = g_tab[g & (kTableSize - 1)];
    if (e.key == g + 1) {
      r.draws += e.draws;
      ++r.granules_hit;
      // The recorder walks a buffer front to back, so recording order must rise
      // with granule index. Where it falls, the granules on either side came
      // from different passes.
      if (have_prev && e.seq < prev_seq) ++r.seq_inversions;
      prev_seq = e.seq;
      have_prev = true;
    } else {
      ++r.granules_miss;
    }
  }
  return r;
}

const Stats& GetStats() { return g_stats; }

void ResetStats() {
  g_stats.recorded = 0;
  g_stats.collisions = 0;
  for (uint32_t i = 0; i < 8; ++i) g_stats.alias[i] = 0;
}

void Reset() {
  for (uint32_t i = 0; i < kTableSize; ++i) g_tab[i] = Entry{};
  g_cur_key = 0;
  g_seq = 0;
  ResetStats();
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex

extern "C" void rex_nr_record_draw(uint32_t guest_addr) {
  using namespace rex::graphics::nr;
  // Top three bits: the alias itself. kPhysMask keeps the low 29, so anything
  // wider would mix physical address bits into the alias tally.
  ++g_stats.alias[(guest_addr >> 29) & 0x7];
  ++g_stats.recorded;

  const uint32_t key = ((guest_addr & kPhysMask) >> kGranuleShift) + 1;
  const uint32_t seq = ++g_seq;
  Entry* e = &g_tab[(key - 1) & (kTableSize - 1)];

  if (key != g_cur_key) {
    // The recorder just arrived in this granule, so anything the slot held is a
    // previous recording now being overwritten. This is the invalidation, and
    // it is the entire reason no boundary detection is needed.
    if (e->key && e->key != key) ++g_stats.collisions;
    e->key = key;
    e->draws = 0;
    g_cur_key = key;
  } else if (e->key != key) {
    // Still inside the same granule, but another granule sharing this slot
    // displaced it. Only possible with a ring larger than the table's span.
    ++g_stats.collisions;
    e->key = key;
    e->draws = 0;
  }

  ++e->draws;
  e->seq = seq;
}
