/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_buffer_cache.h"

// [NR-BUF] See the header. Single-thread (command processor) by contract, so
// no synchronisation anywhere here; the only cross-thread inputs -- the draw
// records and the granule epochs -- are read through the draw cache's own
// race discipline.

namespace rex {
namespace graphics {
namespace nr {

namespace {

constexpr uint32_t kPhysMask = 0x1FFFFFFF;
constexpr uint32_t kSlotBits = 9;
constexpr uint32_t kSlots = 1u << kSlotBits;  // 512; the city runs ~102 live

BufSnapshot g_snaps[kSlots] = {};
BufCacheStats g_stats = {};

inline uint32_t Slot(uint32_t ptr) {
  return ((ptr >> 2) * 2654435761u) >> (32 - kSlotBits);
}

// The admission join, shared by AdmitFromPackets and VerifySnapshot: join one
// packet, applying the everywhere-rule that rid-0 records (device-state
// draws, no arguments) are hits without an argument check.
bool JoinOne(const PacketRef& p, DrawRecord* rec) {
  if (!LookupDraw(p.addr, rec)) return false;
  if (rec->rid == 0) return true;
  return (rec->prim & 0x3F) == p.prim && rec->count == p.count;
}

bool SameRecord(const DrawRecord& a, const DrawRecord& b) {
  return a.addr == b.addr && a.rid == b.rid && a.prim == b.prim &&
         a.start == b.start && a.count == b.count;
}

}  // namespace

BufQuery QuerySnapshot(uint32_t ptr, uint32_t dwords, uint64_t epoch,
                       const BufSnapshot** out) {
  const uint32_t p = ptr & kPhysMask;
  const BufSnapshot& s = g_snaps[Slot(p)];
  if (s.ptr != p) return BufQuery::kAbsent;
  if (s.dwords != dwords) return BufQuery::kResized;
  if (s.epoch != epoch) return BufQuery::kDirty;
  *out = &s;
  return BufQuery::kValid;
}

BufAdmit AdmitFromPackets(uint32_t ptr, uint32_t dwords, const PacketRef* pkts,
                          uint32_t n, uint64_t epoch) {
  if (n > kMaxSnapDraws) {
    ++g_stats.toobig;
    return BufAdmit::kTooBig;
  }
  const uint32_t p = ptr & kPhysMask;

  // Join into a scratch snapshot first: a rejected join must leave whatever
  // snapshot the slot holds untouched (it is already invalid by epoch, and
  // half-replacing it would turn a clean reject into a corrupt serve).
  static BufSnapshot s_scratch;
  for (uint32_t i = 0; i < n; ++i) {
    if (!JoinOne(pkts[i], &s_scratch.recs[i])) {
      ++g_stats.rejects;
      return BufAdmit::kRejectedJoin;
    }
  }

  BufSnapshot& s = g_snaps[Slot(p)];
  const bool same_buffer = s.ptr == p;
  if (s.ptr && !same_buffer) ++g_stats.displaced;
  if (!s.ptr) ++g_stats.live;

  // Unchanged content means the invalidation that forced this walk was false
  // (edge-granule neighbour dirt or epoch-slot aliasing) -- worth counting
  // apart, since it prices the epoch scheme's conservatism.
  bool unchanged = same_buffer && s.dwords == dwords && s.n == n;
  if (unchanged) {
    for (uint32_t i = 0; i < n; ++i) {
      if (!SameRecord(s.recs[i], s_scratch.recs[i])) {
        unchanged = false;
        break;
      }
    }
  }

  s.ptr = p;
  s.dwords = dwords;
  s.epoch = epoch;
  s.n = n;
  for (uint32_t i = 0; i < n; ++i) s.recs[i] = s_scratch.recs[i];

  if (unchanged) {
    ++g_stats.admissions_same;
    return BufAdmit::kAdmittedUnchanged;
  }
  ++g_stats.admissions;
  return BufAdmit::kAdmitted;
}

bool VerifySnapshot(const BufSnapshot* snap, const PacketRef* pkts,
                    uint32_t n) {
  if (snap->n != n) return false;
  for (uint32_t i = 0; i < n; ++i) {
    DrawRecord fresh;
    if (!JoinOne(pkts[i], &fresh)) return false;
    if (!SameRecord(snap->recs[i], fresh)) return false;
    if (snap->recs[i].addr != pkts[i].addr) return false;
  }
  return true;
}

const BufCacheStats& GetBufCacheStats() { return g_stats; }

void ResetBufCacheStats() {
  const uint64_t live = g_stats.live;
  g_stats = BufCacheStats{};
  g_stats.live = live;
}

void ResetBufCache() {
  for (uint32_t i = 0; i < kSlots; ++i) g_snaps[i].ptr = 0;
  g_stats = BufCacheStats{};
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
