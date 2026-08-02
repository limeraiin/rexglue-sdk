/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_BUFFER_CACHE_H_
#define REX_GRAPHICS_NR_BUFFER_CACHE_H_

#include <cstdint>

#include "rex/graphics/nr_draw_cache.h"

// [NR-BUF] Persistent per-buffer draw-list snapshots: native-renderer
// build-out increment 2.
//
// Increment 1 (nr_draw_cache.h) proved the per-draw join: at execution time,
// 99.8%+ of a buffer's DRAW_INDX packets find their hook context at their own
// header address, city included. This unit is the layer the renderer actually
// replays: a SNAPSHOT of a buffer's complete, joined draw list, taken the
// first time the buffer executes with a fully clean join and served on every
// later execution without re-walking -- buffers are recorded roughly once per
// three frames but execute every frame, so most executions should be served
// from a snapshot.
//
// The design answers the two questions the city runs made hard:
//
//   * WHEN is a snapshot stale? When any draw was re-recorded into the
//     buffer's range since admission. Detected by the draw cache's granule
//     dirty-epochs (SumRangeEpoch): the admitting walk reads the range's sum
//     FIRST, stores it, and the snapshot is valid only while the range still
//     sums to the same value. Patches during the walk raise the sum after the
//     read, so they invalidate the snapshot they raced -- conservative, never
//     unsafe. False invalidations (a neighbour's draws in a shared edge
//     granule, epoch-slot aliasing) cost a re-walk and are measured: a
//     re-admission whose content is unchanged is counted separately.
//
//   * WHEN may a snapshot be taken? Only from an all-clean join: every packet
//     hits, and every hit with arguments matches the packet (admission =
//     all-hit). A buffer caught mid-patch fails the join (torn window) and is
//     simply retried at its next execution.
//
// Threading: unlike the draw cache, this layer is touched by ONE thread only
// (the command processor: query, admit, verify). The guest hook never sees
// it; its writes arrive indirectly through the draw cache and the epochs.
// So there is no race discipline to state -- plain code.

namespace rex {
namespace graphics {
namespace nr {

// One executed DRAW_INDX as the consumer's packet walk found it: the packet
// header's masked physical address plus the VGT_DRAW_INITIATOR fields.
struct PacketRef {
  uint32_t addr;
  uint32_t prim;
  uint32_t count;
};

// City buffers measure ~350 draws; 1024 leaves headroom without making the
// table silly. Bigger buffers are refused (kTooBig) and measured.
constexpr uint32_t kMaxSnapDraws = 1024;

struct BufSnapshot {
  uint32_t ptr;     // masked buffer address; 0 = empty slot
  uint32_t dwords;  // buffer length at admission
  uint64_t epoch;   // SumRangeEpoch(ptr, dwords*4) read BEFORE the walk
  uint32_t n;       // draws
  DrawRecord recs[kMaxSnapDraws];  // the joined draw list, in packet order
};

enum class BufQuery {
  kValid,    // snapshot present and current: serve it
  kAbsent,   // no snapshot for this address
  kDirty,    // range epoch moved since admission (a draw was re-recorded)
  kResized,  // buffer length changed at the same address
};

// Ask for the snapshot serving an execution of [ptr, ptr+dwords*4). `epoch`
// is SumRangeEpoch over that range, read by the caller BEFORE any packet
// walk (so a patch racing the walk reads as dirty next time, never as a
// stale serve). On kValid *out points at the snapshot.
BufQuery QuerySnapshot(uint32_t ptr, uint32_t dwords, uint64_t epoch,
                       const BufSnapshot** out);

enum class BufAdmit {
  kAdmitted,           // clean join, snapshot (re)placed
  kAdmittedUnchanged,  // clean join, identical to the snapshot it replaces:
                       //   the invalidation that forced this walk was false
  kRejectedJoin,       // some packet missed or mismatched: buffer mid-patch
  kTooBig,             // more packets than kMaxSnapDraws
};

// Join pkts against the draw cache and, if every packet hits with matching
// arguments (rid-0 hits exempt from the argument check, as everywhere), store
// the result as the snapshot for ptr. `epoch` is the caller's before-walk
// read. A rejected join leaves any previous snapshot in place: it is already
// invalid by epoch, and the next clean execution will replace it.
BufAdmit AdmitFromPackets(uint32_t ptr, uint32_t dwords, const PacketRef* pkts,
                          uint32_t n, uint64_t epoch);

// The increment-2 gate, probe mode only: re-join `pkts` against the LIVE draw
// cache and compare, draw for draw, against the snapshot being served. Any
// difference is a STALE SERVE -- the invalidation rule failed to catch a
// change -- and must be ~0 for the model to hold. Returns true when the
// snapshot still matches. Comparison is on the semantic fields (addr, rid,
// prim, start, count); a re-record with identical content is not stale.
bool VerifySnapshot(const BufSnapshot* snap, const PacketRef* pkts, uint32_t n);

struct BufCacheStats {
  uint64_t admissions;        // snapshots placed (fresh content)
  uint64_t admissions_same;   // re-admissions with unchanged content
  uint64_t rejects;           // joins refused (mid-patch)
  uint64_t toobig;            // buffers over kMaxSnapDraws
  uint64_t displaced;         // snapshots evicted by a direct-map conflict
  uint64_t live;              // occupied slots right now
};

const BufCacheStats& GetBufCacheStats();
void ResetBufCacheStats();  // rates only; `live` survives

// Drop everything. Tests only.
void ResetBufCache();

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_BUFFER_CACHE_H_
