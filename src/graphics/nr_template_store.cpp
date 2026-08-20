/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_template_store.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <rex/cvar.h>

REXCVAR_DEFINE_BOOL(
    gpu_nr_tmpl, false, "GPU",
    "DEV [nr-tmpl] N-2 rung 0: build a span template (the walker's own memo "
    "op stream + a byte copy) at RECORD time for every D3D9-block call span, "
    "keyed by the span's first packet address, and COMPARE each executed "
    "depth-1 indirect buffer's walk against replaying those templates. "
    "Observation-only, no consumption. One flag arms both the game-side feed "
    "and the SDK-side compare. Off by default.");

REXCVAR_DEFINE_BOOL(
    gpu_nr_tmpl_plan, false, "GPU",
    "DEV [nr-plan] N-2-2: compile each span template into a FLAT APPLY PLAN "
    "(one op per packet, everything the walk derives resolved once, every "
    "value read live, structure guarded) instead of the walker's memo op "
    "stream, and drive the [nr-tmpl] compare from it. Requires gpu_nr_tmpl. "
    "Off by default.");

namespace rex {
namespace graphics {
namespace nr {
namespace {

constexpr uint32_t kPhysMask = 0x1FFFFFFF;
// Same bound the record-side draw locator has always used: a recorder span
// larger than this has never been observed and would not be a draw group.
constexpr uint32_t kMaxSpanBytes = 65536;
constexpr uint32_t kMaxSpanDwords = kMaxSpanBytes / 4;
// One memo op per packet, one packet is at least one dword: the op stream can
// never outgrow the span's dword count. Reserved once so the producer-thread
// push_backs never allocate (record-side rule: no host mutexes).
constexpr uint32_t kBuildOpCap = kMaxSpanDwords + 8;
// [NR-PLAN] One plan op per PACKET and at most 0.75 guards per dword (the
// densest shape is LOAD_ALU_CONSTANT: 4 dwords, 3 guards), so the span's
// dword count bounds both. Reserved statically; the producer never allocates.
constexpr uint32_t kPlanOpCap = kMaxSpanDwords + 8;
constexpr uint32_t kPlanGuardCap = kMaxSpanDwords + 8;

constexpr uint32_t kArenaBytes = 256u << 20;
constexpr uint32_t kTblBits = 18;
constexpr uint32_t kTblSize = 1u << kTblBits;
constexpr uint32_t kProbe = 8;

// ---- The store --------------------------------------------------------------
// One guest-thread writer (the span feed), one CP-thread reader (the
// compare), no locks: the nr_draw_cache discipline (invalidate the key first,
// store it last, reader re-checks after copying) plus an epoch that the
// arena wrap bumps so every published template goes stale at once instead of
// ever being overwritten in place. A torn read is a counted miss.

struct Slot {
  std::atomic<uint32_t> key{0};  // masked first-packet address; 0 = empty
  uint32_t epoch = 0;
  uint32_t off = 0;  // arena byte offset of the blob
  uint32_t ndwords = 0;
  uint32_t nops = 0;
  uint32_t nplan = 0;   // [NR-PLAN] plan ops (0 in memo mode)
  uint32_t nguard = 0;  // [NR-PLAN] structure guards
};

// Blob layout at `off`: u32 key, ndwords, nops, nplan, nguard, then the memo
// ops (nops * sizeof(CtxMemoOp)), the plan ops, the guards, and last the span
// byte copy (ndwords * 4) -- last so the byte-identity check's offset is one
// addition. The header duplicates the slot so the reader can prove the copy
// it took is the blob the slot meant. Exactly one of {nops, nplan} is
// nonzero: gpu_nr_tmpl_plan picks the shape at startup.
constexpr uint32_t kBlobHdr = 20;

inline uint32_t BlobBytesOff(uint32_t nops, uint32_t nplan, uint32_t nguard) {
  return kBlobHdr + nops * uint32_t(sizeof(CtxMemoOp)) +
         nplan * uint32_t(sizeof(CtxPlanOp)) +
         nguard * uint32_t(sizeof(CtxPlanGuard));
}

Slot g_tbl[kTblSize];
uint8_t* g_arena = nullptr;
uint32_t g_cursor = 0;  // producer-private
std::atomic<uint32_t> g_epoch{1};
bool g_active = false;
bool g_plan_mode = false;

TmplStats g_stats;

inline uint32_t SlotIndex(uint32_t key) {
  return ((key >> 2) * 2654435761u) >> (32 - kTblBits);
}

inline uint32_t LoadBE32(const uint8_t* base, uint32_t at) {
  uint32_t v;
  __builtin_memcpy(&v, base + at, 4);
  return __builtin_bswap32(v);
}

// ---- Build side (guest producer thread) -------------------------------------

std::vector<CtxMemoOp> g_build_ops;  // capacity reserved at init, never grows
std::vector<CtxMemoOp> g_build_ops2;  // ...and the scan-merged stream
// [NR-PLAN] producer-thread compile scratch (static: the feed never allocates)
CtxPlanOp g_build_plan[kPlanOpCap];
CtxPlanGuard g_build_guards[kPlanGuardCap];
StateContext g_build_ctx = {};       // scratch: ops do not depend on it
CtxWalkStats g_build_stats;

// Frame the span with the WALKER's framing rules -- zero dword = one-dword
// no-op, type-1 = 3 dwords (record_map's locator used different type-1/nop
// rules; the walker is the oracle here, so the walker's rules decide) --
// under both stwu cursor conventions, requiring an exact landing
// ([[pm4-span-parse-conventions]]). Returns the first packet header address.
bool FrameSpan(const uint8_t* base, uint32_t entry, uint32_t end,
               uint32_t* out_start) {
  for (uint32_t conv = 0; conv < 2; ++conv) {
    const uint32_t start = entry + (conv ? 0 : 4);
    const uint32_t stop = end + (conv ? 0 : 4);
    uint32_t j = start;
    uint32_t guard = 0;
    while (j < stop && ++guard <= kMaxSpanDwords) {
      const uint32_t h = LoadBE32(base, j);
      if (!h) {
        j += 4;
        continue;
      }
      const uint32_t ty = h >> 30;
      if (ty == 3 || ty == 0) {
        j += 4 * (2 + ((h >> 16) & 0x3FFF));
      } else if (ty == 1) {
        j += 12;
      } else {
        j += 4;
      }
    }
    if (j == stop) {
      *out_start = start;
      return true;
    }
  }
  return false;
}

// [NR-TMPL] N-2 rung 1 item 2 -- the FINALIZE CLASS.
// Rung 0 measured that the recorder finalizes buffers IN PLACE at kick time,
// pointer-neutrally: placeholder no-op dwords become SET_BIN / WAIT_REG_MEM /
// EVENT_WRITE / bin-window packets, real packets are nop'd back out, and
// constant payloads are patched -- all invisible to a pointer-moving record
// hook, and the dominant cause of stale templates (264k of 396k stale spans
// at menu, 1.07M of 1.30M at city). The answer is not to store those bytes
// but to record where they LIVE: every run of top-level no-op dwords becomes
// a kCtxMemoScan window in stream position, and the replay parses that window
// from the live buffer. Constant patches need nothing extra -- the replay
// reads values from the live bytes at the recorded positions.
// Framing here is the WALKER's, identical to FrameSpan's, so the merge
// positions cannot drift from the op stream's anchors.
uint32_t MergeScanOps(const uint8_t* raw, uint32_t ndwords,
                      const std::vector<CtxMemoOp>& in,
                      std::vector<CtxMemoOp>* out, uint32_t* scan_dw) {
  out->clear();
  size_t k = 0;
  uint32_t j = 0, nscan = 0;
  while (j < ndwords) {
    const uint32_t h = LoadBE32(raw, j * 4);
    if (!h || (h >> 30) == 2) {
      const uint32_t run = j;
      while (j < ndwords) {
        const uint32_t g = LoadBE32(raw, j * 4);
        if (g && (g >> 30) != 2) break;
        ++j;
      }
      while (k < in.size() && in[k].b < run) out->push_back(in[k++]);
      out->push_back({kCtxMemoScan, 0, 0, j - run, run, run});
      *scan_dw += j - run;
      ++nscan;
      continue;
    }
    j += ((h >> 30) == 1) ? 3 : (2 + ((h >> 16) & 0x3FFF));
  }
  while (k < in.size()) out->push_back(in[k++]);
  return nscan;
}

// Predicated packets must stay packet-ops (kCtxMemoPkt) so predication
// resolves against the bin state in effect at REPLAY; only unpredicated
// packets become range ops. The same rule runs in both compare passes so the
// emission shapes stay symmetric.
bool BuildRangeFn(void* user, uint32_t, const uint32_t*, uint32_t, uint32_t,
                  bool) {
  return !(static_cast<CtxWalker*>(user)->cur_hdr & 1);
}

// Enables the by-ref range OFFER (kCtxMemoRangeMem needs a non-null reader);
// only the predicated per-dword fallback ever calls it, and those writes land
// in the scratch context. By-ref VALUES are never read at build -- they are
// read at replay, per [[bindings-inline-constants-byref]].
uint32_t BuildMemRead(void*, uint32_t) { return 0; }

}  // namespace

TmplStats* TmplStatsPtr() { return &g_stats; }

}  // namespace nr
}  // namespace graphics
}  // namespace rex

extern "C" bool rex_nr_tmpl_active() {
  using namespace rex::graphics::nr;
  // Called by the game's startup latch (NarutoApp::OnPostInitLogging) AND the
  // command processor's init -- two host threads, so once_flag, not a bool.
  static std::once_flag once;
  std::call_once(once, [] {
    if (REXCVAR_GET(gpu_nr_tmpl)) {
      g_arena = static_cast<uint8_t*>(std::malloc(kArenaBytes));
      if (g_arena) {
        g_build_ops.reserve(kBuildOpCap);
        g_build_ops2.reserve(kBuildOpCap);
        g_plan_mode = REXCVAR_GET(gpu_nr_tmpl_plan);
        g_active = true;
      }
    }
  });
  return g_active;
}

extern "C" void rex_nr_tmpl_span(uint32_t entry_va, uint32_t end_va,
                                 uint8_t* base) {
  using namespace rex::graphics::nr;
  if (!g_active) return;
  TmplStats& s = g_stats;
  ++s.feed;
  if (end_va <= entry_va || end_va - entry_va > kMaxSpanBytes) {
    ++s.feed_reject;
    return;
  }
  uint32_t start;
  if (!FrameSpan(base, entry_va, end_va, &start)) {
    ++s.parse_fail;
    return;
  }
  const uint32_t ndwords = (end_va - entry_va) / 4;
  if (!ndwords) {
    ++s.feed_reject;
    return;
  }
  const uint32_t key = start & kPhysMask;
  const uint32_t epoch = g_epoch.load(std::memory_order_relaxed);

  // Probe once: find the same-key slot (and short-circuit a byte-identical
  // re-record -- the template is a pure function of the bytes, and the city
  // re-records the same packet address ~3.5k times per frame), else remember
  // an empty or stale-epoch slot to claim.
  Slot* target = nullptr;
  bool existed = false;
  {
    Slot* fallback = nullptr;
    const uint32_t h = SlotIndex(key);
    for (uint32_t p = 0; p < kProbe; ++p) {
      Slot& c = g_tbl[(h + p) & (kTblSize - 1)];
      const uint32_t k = c.key.load(std::memory_order_relaxed);
      if (k == key) {
        if (c.epoch == epoch && c.ndwords == ndwords &&
            std::memcmp(g_arena + c.off +
                            BlobBytesOff(c.nops, c.nplan, c.nguard),
                        base + start, size_t(ndwords) * 4) == 0) {
          ++s.unchanged;
          return;
        }
        target = &c;
        existed = true;
        break;
      }
      if (!fallback && (!k || c.epoch != epoch)) fallback = &c;
    }
    if (!target) target = fallback;
    if (!target) {
      target = &g_tbl[h & (kTblSize - 1)];
      ++s.slot_evict;
    }
  }

  // [NR-PLAN] N-2-2: the flat apply plan. One framing pass over the span
  // bytes -- no walker, no memo vector, no scan merge -- emitting one op per
  // packet plus the structure guards. Strictly cheaper on the producer than
  // the memo build it replaces, which matters: this runs on the guest thread.
  uint32_t nplan = 0, nguard = 0;
  if (g_plan_mode) {
    CtxPlanCompileStats cst;
    nplan = CtxPlanCompile(base + start, ndwords, g_build_plan, kPlanOpCap,
                           g_build_guards, kPlanGuardCap, &nguard, &cst);
    if (!nplan) {
      ++s.plan_fail;
      return;
    }
    s.plan_ops += nplan;
    s.plan_guards += nguard;
    s.scan_ops += cst.scans;
    ++s.plan_built;
  }
  // One parse of the just-written span through the real walker, its own memo
  // recorder attached: build-side decode semantics are the walk's by
  // construction. Bin state all-ones so every packet (all three tiles'
  // predicated blocks included) is decoded and recorded.
  g_build_ops.clear();
  if (!g_plan_mode) {
    CtxWalker w;
    CtxWalkBegin(&w, base + start, ndwords, key, &g_build_ctx, nullptr, 0,
                 &g_build_stats, BuildMemRead, nullptr, nullptr, nullptr,
                 nullptr, nullptr, ~0ull, ~0ull, nullptr, nullptr, nullptr,
                 nullptr);
    w.range_fn = BuildRangeFn;
    w.range_user = &w;
    w.rec = &g_build_ops;
    CtxDrawStop st;
    while (CtxWalkNextStop(&w, &st)) {
      if (st.delegate) CtxWalkSkipDelegated(&w);
    }
    if (w.cursor != ndwords) {
      // Framing guaranteed the landing, so this should not happen; counted so
      // a violation can never hide.
      ++s.parse_fail;
      return;
    }
    // [NR-TMPL] rung 1: fold the placeholder runs in as live-scan windows.
    // One op per no-op RUN and one per packet, so the merged stream is still
    // bounded by the span's dword count.
    uint32_t scan_dw = 0;
    const uint32_t nscan =
        MergeScanOps(base + start, ndwords, g_build_ops, &g_build_ops2,
                     &scan_dw);
    s.scan_ops += nscan;
    s.scan_dw += scan_dw;
    if (g_build_ops2.size() > kBuildOpCap) {
      ++s.ops_overflow;
      return;
    }
  } else {
    g_build_ops2.clear();
  }

  const uint32_t nops = uint32_t(g_build_ops2.size());
  const uint32_t ops_bytes = nops * uint32_t(sizeof(CtxMemoOp));
  const uint32_t plan_bytes = nplan * uint32_t(sizeof(CtxPlanOp));
  const uint32_t guard_bytes = nguard * uint32_t(sizeof(CtxPlanGuard));
  const uint32_t need =
      kBlobHdr + ops_bytes + plan_bytes + guard_bytes + ndwords * 4;
  if (g_cursor + need > kArenaBytes) {
    // Wrap: bump the epoch first (every reader-visible template goes stale
    // at once -- nothing is ever overwritten inside a live epoch), restart.
    g_epoch.fetch_add(1, std::memory_order_release);
    g_cursor = 0;
    ++s.arena_wraps;
    // The slot chosen above may now be the only fresh-epoch entry; re-stamp
    // below covers it either way.
  }
  const uint32_t pub_epoch = g_epoch.load(std::memory_order_relaxed);
  const uint32_t off = g_cursor;
  uint8_t* blob = g_arena + off;
  const uint32_t hdr5[5] = {key, ndwords, nops, nplan, nguard};
  std::memcpy(blob, hdr5, kBlobHdr);
  if (ops_bytes) std::memcpy(blob + kBlobHdr, g_build_ops2.data(), ops_bytes);
  if (plan_bytes) {
    std::memcpy(blob + kBlobHdr + ops_bytes, g_build_plan, plan_bytes);
  }
  if (guard_bytes) {
    std::memcpy(blob + kBlobHdr + ops_bytes + plan_bytes, g_build_guards,
                guard_bytes);
  }
  std::memcpy(blob + BlobBytesOff(nops, nplan, nguard), base + start,
              size_t(ndwords) * 4);
  g_cursor += (need + 15u) & ~15u;

  target->key.store(0, std::memory_order_relaxed);
  target->epoch = pub_epoch;
  target->off = off;
  target->ndwords = ndwords;
  target->nops = nops;
  target->nplan = nplan;
  target->nguard = nguard;
  target->key.store(key, std::memory_order_release);
  ++(existed ? s.rebuilt : s.built);
}

// ---- Compare side (CP thread) ------------------------------------------------

namespace rex {
namespace graphics {
namespace nr {
namespace {

// A validated copy of one template, in CP-thread scratch.
struct TmplView {
  uint32_t ndwords;
  uint32_t nops;
  uint32_t nplan;
  uint32_t nguard;
  CtxMemoOp* ops;  // in CP-thread scratch: the drift pre-pass rewrites ops
  CtxPlanOp* plan;  // ...and the guard pass rewrites plan ops
  const CtxPlanGuard* guards;
  const uint8_t* bytes;
};

// [NR-TMPL] rung 1 item 1: the buffer snapshot. Both compare passes read it,
// so a producer write landing between pass A's walk and pass B's replay can
// no longer masquerade as a decode mismatch (rung 0's city residual, 0.0054%,
// read exactly like that). The mutation itself is then measured directly, by
// re-comparing the live buffer against the snapshot when the compare is done.
constexpr size_t kMaxSnapBytes = 64u << 20;
std::vector<uint8_t> g_snap;

// [NR-PLAN] The consuming swap's own CP-thread scratch: header + plan ops +
// guards, never the span byte copy. Separate from g_view_scratch so a gate
// run and a swap run can never share a buffer.
alignas(16) uint8_t g_swap_scratch[kBlobHdr + kPlanOpCap * sizeof(CtxPlanOp) +
                                   kPlanGuardCap * sizeof(CtxPlanGuard)];
TmplSwapStats g_swap;

alignas(16) uint8_t g_view_scratch[kBlobHdr + kBuildOpCap * sizeof(CtxMemoOp) +
                       kPlanOpCap * sizeof(CtxPlanOp) +
                       kPlanGuardCap * sizeof(CtxPlanGuard) + kMaxSpanBytes];

bool Lookup(uint32_t key, TmplView* v, Slot** out_slot) {
  const uint32_t h = SlotIndex(key);
  for (uint32_t p = 0; p < kProbe; ++p) {
    Slot& c = g_tbl[(h + p) & (kTblSize - 1)];
    if (c.key.load(std::memory_order_acquire) != key) continue;
    const uint32_t e1 = g_epoch.load(std::memory_order_acquire);
    const uint32_t off = c.off;
    const uint32_t nd = c.ndwords;
    const uint32_t no = c.nops;
    const uint32_t npl = c.nplan;
    const uint32_t ngd = c.nguard;
    if (c.epoch != e1 || !nd || nd > kMaxSpanDwords || no > kBuildOpCap ||
        npl > kPlanOpCap || ngd > kPlanGuardCap) {
      continue;
    }
    const uint32_t bytes = BlobBytesOff(no, npl, ngd) + nd * 4;
    if (uint64_t(off) + bytes > kArenaBytes) continue;
    std::memcpy(g_view_scratch, g_arena + off, bytes);
    // Revalidate: the arena is append-only inside an epoch and slots publish
    // key-last, so an unchanged (epoch, key, off) proves the copy coherent.
    if (g_epoch.load(std::memory_order_acquire) != e1 ||
        c.key.load(std::memory_order_acquire) != key || c.off != off) {
      ++g_stats.lookup_stale;
      continue;
    }
    uint32_t hdr5[5];
    std::memcpy(hdr5, g_view_scratch, kBlobHdr);
    if (hdr5[0] != key || hdr5[1] != nd || hdr5[2] != no || hdr5[3] != npl ||
        hdr5[4] != ngd) {
      ++g_stats.lookup_stale;
      continue;
    }
    v->ndwords = nd;
    v->nops = no;
    v->nplan = npl;
    v->nguard = ngd;
    v->ops = reinterpret_cast<CtxMemoOp*>(g_view_scratch + kBlobHdr);
    v->plan = reinterpret_cast<CtxPlanOp*>(g_view_scratch + kBlobHdr +
                                           no * sizeof(CtxMemoOp));
    v->guards = reinterpret_cast<const CtxPlanGuard*>(
        g_view_scratch + kBlobHdr + no * sizeof(CtxMemoOp) +
        npl * sizeof(CtxPlanOp));
    v->bytes = g_view_scratch + BlobBytesOff(no, npl, ngd);
    *out_slot = &c;
    return true;
  }
  return false;
}

// One emission of either pass, in packet order. `dw` is the absolute dword
// of the emitting packet's header in the live buffer -- comparing it too
// makes the diff a framing check, not just a value check.
enum : uint8_t { kEmiReg = 1, kEmiRange, kEmiShader, kEmiStop };
struct Emi {
  uint8_t kind;
  uint8_t fm;  // from_memory (reg/range) or delegate flag (stop)
  uint16_t reg;
  uint32_t dw;
  uint32_t a;  // value (reg) / phys (range) / vs.addr (shader)
  uint32_t n;  // range length / ps.addr (shader)
  uint64_t vh; // range values FNV / shader sizes+flags
};

constexpr size_t kEmiCap = 1u << 20;
std::vector<Emi> g_a;  // pass A: the live walk's emission log
bool g_a_ovf = false;

CtxMemReadFn g_mem_read = nullptr;
void* g_mem_user = nullptr;

// Both passes read by-ref values from live guest memory moments apart; a
// difference there is a genuinely volatile source, and belongs in the count.
uint64_t FnvRange(const uint32_t* values_be, uint32_t n, uint32_t phys) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t v = values_be ? __builtin_bswap32(values_be[i])
                                 : g_mem_read(g_mem_user, phys + i * 4);
    h = (h ^ v) * 1099511628211ull;
  }
  return h;
}

// ---- Pass A: log the private live walk --------------------------------------

CtxWalker g_wa;
StateContext g_ctx_a;  // persistent across buffers, like the oracle's
CtxWalkStats g_stats_a;

void APush(const Emi& e) {
  if (g_a.size() >= kEmiCap) {
    g_a_ovf = true;
    return;
  }
  g_a.push_back(e);
}

void AReg(void*, uint32_t reg, uint32_t value, bool from_memory) {
  APush({kEmiReg, uint8_t(from_memory), uint16_t(reg), g_wa.cur_dw, value, 0,
         0});
}

bool ARange(void* user, uint32_t base, const uint32_t* values_be, uint32_t n,
            uint32_t phys, bool from_memory) {
  CtxWalker* w = static_cast<CtxWalker*>(user);
  if (w->cur_hdr & 1) return false;  // same shape rule as the build
  APush({kEmiRange, uint8_t(from_memory), uint16_t(base), w->cur_dw, phys, n,
         FnvRange(values_be, n, phys)});
  return true;
}

// The shader emission compares the LOADED ref itself (fully determined by the
// IM_LOAD packet's bytes), never the context pair: the template context only
// sees replayed spans, so a ctx compare would re-report every coverage gap as
// a shader mismatch instead of measuring the decode.
void AShader(void*, const ShaderRef& ref) {
  APush({kEmiShader, ref.immediate, 0, g_wa.cur_dw, ref.addr, ref.size_dwords,
         0});
}

// ---- Pass B: replay templates, comparing in-stream --------------------------

// This span's bytes, for the first-ne read-out only.
const uint8_t* g_span_live = nullptr;
const uint8_t* g_span_stored = nullptr;
uint32_t g_span_nd = 0;

CtxWalker g_wb;
StateContext g_ctx_b;  // persistent, fed by replayed spans only
CtxWalkStats g_stats_b;
size_t g_acur = 0;      // sweep cursor into g_a
uint32_t g_dw0 = 0;     // current span window, absolute dwords
uint32_t g_dw1 = 0;
uint32_t g_span_key = 0;
uint32_t g_span_ne = 0;

void NoteNe(uint8_t kind, uint32_t dw, uint32_t a_reg, uint32_t a_val,
            uint32_t b_reg, uint32_t b_val, uint8_t a_kind = 0,
            uint32_t a_dw = 0) {
  TmplStats& s = g_stats;
  switch (kind) {
    case kEmiReg: ++s.emi_ne_reg; break;
    case kEmiRange: ++s.emi_ne_range; break;
    case kEmiShader: ++s.emi_ne_shader; break;
    default: ++s.emi_ne_stop; break;
  }
  if (s.ne_armed) return;
  s.ne_armed = 1;
  s.ne_kind = kind;
  s.ne_dw = dw;
  s.ne_key = g_span_key;
  s.ne_a_reg = a_reg;
  s.ne_a_val = a_val;
  s.ne_b_reg = b_reg;
  s.ne_b_val = b_val;
  s.ne_a_kind = a_kind;
  s.ne_a_dw = a_dw;
  // The packet the template thinks it decoded, live and as recorded: a
  // difference here says framing, an agreement says values.
  const uint32_t idx = dw - g_dw0;
  s.ne_hdr_live =
      (g_span_live && idx < g_span_nd) ? LoadBE32(g_span_live, idx * 4) : 0;
  s.ne_hdr_stored =
      (g_span_stored && idx < g_span_nd) ? LoadBE32(g_span_stored, idx * 4) : 0;
}

void BCompare(const Emi& e) {
  TmplStats& s = g_stats;
  if (g_acur < g_a.size() && g_a[g_acur].dw >= g_dw0 &&
      g_a[g_acur].dw < g_dw1) {
    const Emi& a = g_a[g_acur];
    ++g_acur;
    if (a.kind == e.kind && a.fm == e.fm && a.reg == e.reg && a.dw == e.dw &&
        a.a == e.a && a.n == e.n && a.vh == e.vh) {
      ++s.emi_eq;
      return;
    }
    ++s.emi_ne;
    ++g_span_ne;
    // A by-ref range whose whole descriptor matches and whose VALUES differ is
    // guest memory moving between the two passes' reads, not a decode gap:
    // named so the decode verdict is never read through it.
    if (a.kind == kEmiRange && e.kind == kEmiRange && a.fm && e.fm &&
        a.reg == e.reg && a.dw == e.dw && a.a == e.a && a.n == e.n) {
      ++s.emi_ne_byref;
    }
    NoteNe(e.kind, e.dw, a.reg, a.a, e.reg, e.a, a.kind, a.dw);
    return;
  }
  // The live walk produced nothing (left) in this window for this emission.
  ++s.b_extra;
  ++s.emi_ne;
  ++g_span_ne;
  NoteNe(e.kind, e.dw, 0, 0, e.reg, e.a);
}

void BReg(void*, uint32_t reg, uint32_t value, bool from_memory) {
  BCompare({kEmiReg, uint8_t(from_memory), uint16_t(reg),
            g_wb.cur_dw + g_dw0, value, 0, 0});
}

bool BRange(void* user, uint32_t base, const uint32_t* values_be, uint32_t n,
            uint32_t phys, bool from_memory) {
  CtxWalker* w = static_cast<CtxWalker*>(user);
  // Replay-path ranges (only ever recorded from unpredicated packets) arrive
  // with a stale cur_hdr; a false decline just re-parses the packet, which
  // re-offers with the fresh header. Re-parse-path ranges carry a fresh
  // header, where this is the same shape rule as the build and pass A.
  if (w->cur_hdr & 1) return false;
  BCompare({kEmiRange, uint8_t(from_memory), uint16_t(base), w->cur_dw + g_dw0,
            phys, n, FnvRange(values_be, n, phys)});
  return true;
}

void BShader(void*, const ShaderRef& ref) {
  BCompare({kEmiShader, ref.immediate, 0, g_wb.cur_dw + g_dw0, ref.addr,
            ref.size_dwords, 0});
}

// ---- Stale-diff attribution --------------------------------------------------
// Fill cls[0..ndwords) with the TmplStaleCls of each dword's covering packet
// under the walker's framing. Exact landing required; returns false when the
// stream desyncs (caller falls back or buckets kTmplScUnframed).

uint32_t RangeBucket(uint32_t reg) {
  if (reg >= 0x2080 && reg <= 0x2082) return kTmplScWin;
  if ((reg >= 0x200E && reg <= 0x200F) || (reg >= 0x2318 && reg <= 0x2321)) {
    return kTmplScScissorCopy;
  }
  if (reg >= 0x4000 && reg < 0x4800) return kTmplScAlu;
  if (reg >= 0x4800 && reg < 0x4900) return kTmplScFetch;
  if (reg >= 0x4900 && reg < 0x4A00) return kTmplScBoolLoop;
  return kTmplScRegOther;
}

uint8_t g_cls_scratch[kMaxSpanDwords];

bool ClassifySpan(const uint8_t* raw, uint32_t ndwords, uint8_t* cls) {
  uint32_t j = 0;
  while (j < ndwords) {
    const uint32_t h = LoadBE32(raw, j * 4);
    if (!h) {
      cls[j++] = kTmplScPlaceholder;
      continue;
    }
    const uint32_t ty = h >> 30;
    if (ty == 2) {
      cls[j++] = kTmplScPlaceholder;
      continue;
    }
    uint32_t len, bucket;
    if (ty == 0) {
      const uint32_t cnt = ((h >> 16) & 0x3FFF) + 1;
      len = 1 + cnt;
      bucket = RangeBucket(h & 0x7FFF);
    } else if (ty == 1) {
      len = 3;
      bucket = RangeBucket(h & 0x7FF);
    } else {
      const uint32_t op = (h >> 8) & 0x7F;
      len = 2 + ((h >> 16) & 0x3FFF);
      if (op == 0x2D || op == 0x2F || op == 0x55 || op == 0x56) {
        // Constant writers: bucket by the target base (the payload offset
        // within the file is detail the first-diff line carries).
        uint32_t base = kCtxNoBase;
        if (op == 0x55 || op == 0x56) {
          base = LoadBE32(raw, (j + 1) * 4) & 0xFFFF;
        } else if (j + (op == 0x2F ? 2u : 1u) < ndwords) {
          base = CtxConstantBase(LoadBE32(raw, (j + (op == 0x2F ? 2 : 1)) * 4));
        }
        bucket = base == kCtxNoBase ? kTmplScRegOther : RangeBucket(base);
      } else if (op == 0x50 || op == 0x51 || (op >= 0x60 && op <= 0x63)) {
        bucket = kTmplScSetBin;
      } else if (op == 0x3C) {
        bucket = kTmplScWait;
      } else if (op == 0x46 || op == 0x5A) {
        bucket = kTmplScEvent;
      } else if (op == 0x22 || op == 0x36) {
        bucket = kTmplScDraw;
      } else {
        bucket = kTmplScOtherOp;
      }
    }
    if (j + len > ndwords) return false;
    for (uint32_t m = 0; m < len; ++m) cls[j + m] = uint8_t(bucket);
    j += len;
  }
  return j == ndwords;
}

// The caller has already run the density split (a recycled region differs
// nearly everywhere and its per-dword classes would be noise wearing names),
// so everything reaching here is a patch/finalize-shaped diff.
void AttributeStale(const uint8_t* live, const uint8_t* stored,
                    uint32_t ndwords, uint32_t span_key) {
  TmplStats& s = g_stats;
  bool framed = ClassifySpan(live, ndwords, g_cls_scratch);
  if (!framed) framed = ClassifySpan(stored, ndwords, g_cls_scratch);
  uint32_t touched = 0;  // class bitmask: one count per span per class
  for (uint32_t i = 0; i < ndwords; ++i) {
    if (std::memcmp(live + size_t(i) * 4, stored + size_t(i) * 4, 4) == 0) {
      continue;
    }
    const uint32_t c = framed ? g_cls_scratch[i] : uint32_t(kTmplScUnframed);
    touched |= 1u << c;
    const bool surprise = c == kTmplScDraw || c == kTmplScOtherOp ||
                          c == kTmplScRegOther || c == kTmplScUnframed;
    if (surprise && !s.su_armed) {
      s.su_armed = 1;
      s.su_cls = c;
      s.su_key = span_key;
      s.su_idx = i;
      s.su_stored = LoadBE32(stored, i * 4);
      s.su_live = LoadBE32(live, i * 4);
    }
  }
  for (uint32_t c = 0; c < kTmplScCount; ++c) {
    if (touched & (1u << c)) ++s.stale_cls[c];
  }
}

// Frame the live window packet-by-packet applying only bin packets, so a
// stale or missing span cannot desync predication for the spans after it.
// Returns where the live framing actually landed: the last packet can straddle
// the window's end, and resuming the sweep mid-payload would decode payload
// dwords as headers -- the live walk's framing is the only framing there is.
uint32_t FrameWindowBins(const uint8_t* raw, uint32_t dw, uint32_t dw_end,
                         uint32_t count, CtxBinState* bin) {
  while (dw < dw_end && dw < count) {
    const uint32_t h = __builtin_bswap32(
        *reinterpret_cast<const uint32_t*>(raw + size_t(dw) * 4));
    if (!h) {
      ++dw;
      continue;
    }
    const uint32_t ty = h >> 30;
    if (ty == 3) {
      const uint32_t op = (h >> 8) & 0x7F;
      if (!CtxPredicatedOut(*bin, h)) {
        const uint32_t p0 =
            (dw + 1 < count) ? __builtin_bswap32(*reinterpret_cast<const uint32_t*>(
                                   raw + size_t(dw + 1) * 4))
                             : 0;
        const uint32_t p1 =
            (dw + 2 < count) ? __builtin_bswap32(*reinterpret_cast<const uint32_t*>(
                                   raw + size_t(dw + 2) * 4))
                             : 0;
        CtxApplyBinPacket(bin, op, p0, p1);
      }
      dw += 2 + ((h >> 16) & 0x3FFF);
    } else if (ty == 0) {
      dw += 2 + ((h >> 16) & 0x3FFF);
    } else if (ty == 1) {
      dw += 3;
    } else {
      ++dw;
    }
  }
  return dw;
}

// [NR-TMPL] rung 1 item 2, apply side: the template is a DECODE PLAN over the
// buffer's live bytes, not a byte snapshot. Values (constants, draw payloads,
// by-ref addresses) are read at replay from the executing buffer, so the
// recorder's in-place patches ride along for free; the scan windows cover the
// placeholder sites; and this pre-pass names what the finalize did to the
// PACKETS. A bulk-range op is the one kind that never re-reads its header, so
// a drifted range (a nop'd-out packet, most often) is demoted to a live
// re-parse -- otherwise it would emit a range that the buffer no longer has.
void DriftPrePass(const uint8_t* live, const TmplView& v) {
  TmplStats& s = g_stats;
  for (uint32_t i = 0; i < v.nops; ++i) {
    CtxMemoOp& op = v.ops[i];
    if (op.kind == kCtxMemoScan || op.b >= v.ndwords) continue;
    const uint32_t hdr = LoadBE32(live, op.b * 4);
    bool drift = hdr != LoadBE32(v.bytes, op.b * 4);
    // A range op carries a DESCRIPTOR the walk derives from the packet, and
    // for the type-3 shapes that descriptor lives in the PAYLOAD, not the
    // header: 0x2D/0x55/0x56 take the base register from dw+1, and 0x2F
    // (LOAD_ALU_CONSTANT) takes address/type/size from dw+1..dw+3. The city
    // patches those dwords in place -- a by-ref load whose ADDRESS moved with
    // an unchanged header was the whole city residual (1,426 of 229.6M
    // emissions, every one a range). Values need no check: the range op
    // already reads them live.
    if (!drift && (hdr >> 30) == 3 &&
        (op.kind == kCtxMemoRange || op.kind == kCtxMemoRangeMem)) {
      const uint32_t nd = op.kind == kCtxMemoRangeMem ? 3u : 1u;
      for (uint32_t k = 1; k <= nd && op.b + k < v.ndwords; ++k) {
        if (LoadBE32(live, (op.b + k) * 4) !=
            LoadBE32(v.bytes, (op.b + k) * 4)) {
          drift = true;
          break;
        }
      }
    }
    if (!drift) continue;
    ++s.op_drift;
    if (op.kind == kCtxMemoRange || op.kind == kCtxMemoRangeMem) {
      // Re-parse: the walker re-derives the descriptor from the live packet,
      // so there is exactly one decoder and it cannot drift from the walk.
      ++s.op_drift_range;
      op.kind = kCtxMemoPkt;
      op.reg = 0;
      op.n = 0;
      op.a = op.b;
    }
  }
}

}  // namespace

void TmplCompareBuffer(const uint8_t* raw_live, uint32_t ptr, uint32_t count,
                       uint64_t bin_select, uint64_t bin_mask,
                       CtxMemReadFn mem_read, void* mem_user) {
  if (!g_active || !count) return;
  TmplStats& s = g_stats;
  g_mem_read = mem_read;
  g_mem_user = mem_user;
  const uint32_t pbase = ptr & kPhysMask;

  // One snapshot, both passes: see kMaxSnapBytes. `raw` is the snapshot from
  // here down -- the live buffer is read again only at the very end, to
  // measure whether the producer wrote into it while we compared.
  const size_t nbytes = size_t(count) * 4;
  if (nbytes > kMaxSnapBytes) {
    ++s.bufs_toobig;
    return;
  }
  if (g_snap.size() < nbytes) g_snap.resize(nbytes);
  std::memcpy(g_snap.data(), raw_live, nbytes);
  const uint8_t* raw = g_snap.data();

  // Pass A: the private live walk's full emission log, dword-tagged.
  if (g_a.capacity() == 0) g_a.reserve(kEmiCap);
  g_a.clear();
  g_a_ovf = false;
  CtxWalkBegin(&g_wa, raw, count, pbase, &g_ctx_a, nullptr, 0, &g_stats_a,
               mem_read, mem_user, AShader, nullptr, nullptr, nullptr,
               bin_select, bin_mask, AReg, nullptr, nullptr, nullptr);
  g_wa.range_fn = ARange;
  g_wa.range_user = &g_wa;
  CtxDrawStop st;
  while (CtxWalkNextStop(&g_wa, &st)) {
    APush({kEmiStop, st.delegate, uint16_t(st.opcode), st.dword, 0, 0, 0});
    if (st.delegate) CtxWalkSkipDelegated(&g_wa);
  }
  if (g_a_ovf) {
    ++s.bufs_aborted;
    return;
  }

  // Pass B: sweep the buffer by template spans, replaying each against the
  // log. Bin state carries across spans and gaps exactly as one walk would.
  g_acur = 0;
  CtxBinState bbin{bin_select, bin_mask};
  uint32_t dw = 0;
  while (dw < count) {
    TmplView v;
    Slot* vslot = nullptr;
    if (Lookup(pbase + dw * 4, &v, &vslot)) {
      if (v.ndwords > count - dw) {
        ++s.spans_cross;
      } else {
        ++s.spans_hit;
        s.dwords_covered += v.ndwords;
        g_dw0 = dw;
        g_dw1 = dw + v.ndwords;
        g_span_key = pbase + dw * 4;
        while (g_acur < g_a.size() && g_a[g_acur].dw < g_dw0) {
          ++s.a_uncovered;  // emissions from the gap before this span
          ++g_acur;
        }
        const uint8_t* live = raw + size_t(dw) * 4;
        bool dead = false;
        const bool stale =
            std::memcmp(live, v.bytes, size_t(v.ndwords) * 4) != 0;
        if (stale) {
          // Bytes differ from the build. Rung 0 named the two causes and rung
          // 1 treats them completely differently. (a) The recorder finalized
          // this span IN PLACE -- a few dwords, at placeholder or payload
          // positions: the template is still THIS span's plan, and the live
          // replay below is exactly the answer to it. (b) The ring recycled
          // the region and the key is dead: >50% of the dwords differ, the
          // "template" describes some other span entirely, and replaying it
          // would measure nothing. The density split is the discriminator.
          ++s.spans_stale;
          uint32_t ndiff = 0, idx = v.ndwords;
          for (uint32_t i = 0; i < v.ndwords; ++i) {
            if (std::memcmp(live + size_t(i) * 4, v.bytes + size_t(i) * 4,
                            4) != 0) {
              if (idx == v.ndwords) idx = i;
              ++ndiff;
            }
          }
          dead = ndiff * 2 > v.ndwords;
          const bool hdr_eq = std::memcmp(live, v.bytes, 4) == 0;
          ++(hdr_eq ? s.stale_hdr_eq : s.stale_hdr_ne);
          if (dead) {
            ++s.spans_dead;
            ++s.stale_cls[kTmplScDead];
          } else {
            AttributeStale(live, v.bytes, v.ndwords, g_span_key);
          }
          if (!s.st_armed && idx < v.ndwords) {
            s.st_armed = 1;
            s.st_key = g_span_key;
            s.st_idx = idx;
            uint32_t sd, ld;
            std::memcpy(&sd, v.bytes + size_t(idx) * 4, 4);
            std::memcpy(&ld, live + size_t(idx) * 4, 4);
            s.st_stored = __builtin_bswap32(sd);
            s.st_live = __builtin_bswap32(ld);
          }
        }
        if (dead) {
          // Kill the dead key so one recycled region cannot report itself
          // once per execution forever. A racing producer publish can lose a
          // fresh template here; the next feed of that span republishes it
          // (the unchanged-check needs a key match, so it misses and
          // rebuilds).
          vslot->key.store(0, std::memory_order_relaxed);
          const uint32_t landed =
              FrameWindowBins(raw, g_dw0, g_dw1, count, &bbin);
          while (g_acur < g_a.size() && g_a[g_acur].dw < landed) {
            ++s.a_uncovered;
            ++g_acur;
          }
          if (landed > g_dw1) {
            ++s.span_overrun;
            dw = landed;
            continue;
          }
        } else {
          g_span_ne = 0;
          g_span_live = live;
          g_span_stored = v.bytes;
          g_span_nd = v.ndwords;
          if (stale) {
            // [NR-PLAN] The plan's ONLY structural claim is guarded, so the
            // guard pass replaces the drift pre-pass entirely: values are
            // read live and need no check, structure is checked and demoted.
            if (g_plan_mode) {
              ++s.plan_guard_spans;
              s.plan_demoted += CtxPlanApplyGuards(live, v.ndwords, v.plan,
                                                   v.nplan, v.guards,
                                                   v.nguard);
            } else {
              DriftPrePass(live, v);
            }
          }
          // Decode bound = the rest of the BUFFER, stream bound = this span.
          // A finalize can lengthen a packet at the span's edge so its payload
          // sits in the next span's dwords; the live walk reads it, so the
          // replay must too (see CtxWalker::rep_end).
          CtxWalkBegin(&g_wb, live, count - dw, g_span_key, &g_ctx_b,
                       nullptr, 0, &g_stats_b, mem_read, mem_user, BShader,
                       nullptr, nullptr, nullptr, bbin.select, bbin.mask,
                       BReg, nullptr, nullptr, nullptr);
          g_wb.range_fn = BRange;
          g_wb.range_user = &g_wb;
          if (g_plan_mode) {
            ++s.plan_spans;
            CtxPlanBegin(&g_wb, v.plan, v.nplan, 0, v.ndwords);
          } else {
            g_wb.rep = v.ops;
            g_wb.rep_n = v.nops;
            g_wb.rep_i = 0;
            g_wb.rep_end = v.ndwords;
          }
          CtxDrawStop bst;
          while (CtxWalkNextStop(&g_wb, &bst)) {
            BCompare({kEmiStop, bst.delegate, uint16_t(bst.opcode),
                      bst.dword + g_dw0, 0, 0, 0});
            if (bst.delegate) CtxWalkSkipDelegated(&g_wb);
          }
          while (g_acur < g_a.size() && g_a[g_acur].dw < g_dw1) {
            ++s.a_extra;  // the live walk emitted, the template did not
            ++s.emi_ne;
            ++g_span_ne;
            NoteNe(g_a[g_acur].kind, g_a[g_acur].dw, g_a[g_acur].reg,
                   g_a[g_acur].a, 0, 0);
            ++g_acur;
          }
          s.scan_pkts += g_stats_b.scan_pkts;
          s.scan_over += g_stats_b.scan_over;
          s.rep_catchup += g_stats_b.rep_catchup;
          s.rep_ahead += g_stats_b.rep_ahead;
          if (stale) {
            ++(g_span_ne ? s.spans_stale_ne : s.spans_stale_eq);
          } else {
            ++(g_span_ne ? s.spans_ne : s.spans_eq);
          }
          // Name the mechanism behind every mismatching span, so a residual
          // can never be reported as an unexplained percentage.
          if (g_span_ne) {
            if (g_stats_b.scan_pkts) ++s.ne_scan;
            if (g_stats_b.rep_ahead) ++s.ne_ahead;
            if (g_stats_b.rep_catchup) ++s.ne_catchup;
            if (g_wb.cursor > v.ndwords) ++s.ne_over;
            if (!g_stats_b.scan_pkts && !g_stats_b.rep_ahead &&
                !g_stats_b.rep_catchup && g_wb.cursor <= v.ndwords) {
              ++s.ne_plain;
            }
          }
          bbin = g_wb.bin;
          // The sweep follows the LIVE framing: a straddling packet consumed
          // dwords of the next span, and re-entering the sweep mid-payload
          // would decode payload as headers.
          if (g_wb.cursor > v.ndwords) {
            ++s.span_overrun;
            dw = g_dw0 + g_wb.cursor;
            while (g_acur < g_a.size() && g_a[g_acur].dw < dw) {
              ++s.a_uncovered;  // payload dwords emit nothing; counted anyway
              ++g_acur;
            }
            continue;
          }
        }
        dw = g_dw1;
        continue;
      }
    }
    // Gap: no template starts at this packet. Frame it, apply bin packets.
    const uint32_t h = __builtin_bswap32(
        *reinterpret_cast<const uint32_t*>(raw + size_t(dw) * 4));
    ++s.gap_pkts;
    uint32_t adv;
    if (!h) {
      adv = 1;
    } else {
      const uint32_t ty = h >> 30;
      if (ty == 3) {
        if (!CtxPredicatedOut(bbin, h)) {
          const uint32_t p0 =
              (dw + 1 < count)
                  ? __builtin_bswap32(*reinterpret_cast<const uint32_t*>(
                        raw + size_t(dw + 1) * 4))
                  : 0;
          const uint32_t p1 =
              (dw + 2 < count)
                  ? __builtin_bswap32(*reinterpret_cast<const uint32_t*>(
                        raw + size_t(dw + 2) * 4))
                  : 0;
          CtxApplyBinPacket(&bbin, (h >> 8) & 0x7F, p0, p1);
        }
        adv = 2 + ((h >> 16) & 0x3FFF);
      } else if (ty == 0) {
        adv = 2 + ((h >> 16) & 0x3FFF);
      } else if (ty == 1) {
        adv = 3;
      } else {
        adv = 1;
      }
    }
    if (adv > count - dw) adv = count - dw;
    s.dwords_gap += adv;
    dw += adv;
  }
  while (g_acur < g_a.size()) {
    ++s.a_uncovered;  // emissions from the tail gap
    ++g_acur;
  }
  ++s.bufs;

  // The producer's window on us: did the buffer change under the compare?
  // Both passes read the snapshot, so this can no longer skew the decode
  // verdict -- it is now a measurement in its own right, and one every
  // consuming design has to answer (a native re-emit reads the same bytes
  // this compare just read).
  if (std::memcmp(raw_live, raw, nbytes) != 0) {
    ++s.bufs_mutated;
    for (uint32_t i = 0; i < count; ++i) {
      if (std::memcmp(raw_live + size_t(i) * 4, raw + size_t(i) * 4, 4) == 0) {
        continue;
      }
      ++s.mut_dwords;
      if (!s.mu_armed) {
        s.mu_armed = 1;
        s.mu_dw = i;
        s.mu_before = LoadBE32(raw, i * 4);
        s.mu_after = LoadBE32(raw_live, i * 4);
      }
    }
  }
}

// ---- [NR-PLAN] the consuming swap -------------------------------------------

TmplSwapStats* TmplSwapStatsPtr() { return &g_swap; }
bool TmplPlanMode() { return g_active && g_plan_mode; }

bool TmplPlanAttach(CtxWalker* w, uint32_t pbase, uint32_t dw, uint32_t count,
                    const uint8_t* raw) {
  if (!g_active || !g_plan_mode) return false;
  const uint32_t key = (pbase + dw * 4) & kPhysMask;
  const uint32_t h = SlotIndex(key);
  for (uint32_t p = 0; p < kProbe; ++p) {
    Slot& c = g_tbl[(h + p) & (kTblSize - 1)];
    if (c.key.load(std::memory_order_acquire) != key) continue;
    const uint32_t e1 = g_epoch.load(std::memory_order_acquire);
    const uint32_t off = c.off;
    const uint32_t nd = c.ndwords;
    const uint32_t no = c.nops;
    const uint32_t npl = c.nplan;
    const uint32_t ngd = c.nguard;
    if (c.epoch != e1 || !nd || !npl || nd > kMaxSpanDwords ||
        npl > kPlanOpCap || ngd > kPlanGuardCap) {
      break;
    }
    const uint32_t pl_off = kBlobHdr + no * uint32_t(sizeof(CtxMemoOp));
    const uint32_t pl_bytes = npl * uint32_t(sizeof(CtxPlanOp)) +
                              ngd * uint32_t(sizeof(CtxPlanGuard));
    if (uint64_t(off) + pl_off + pl_bytes > kArenaBytes) break;
    // Copy before use: the arena is producer-owned and its wrap would
    // overwrite a blob under our feet. Ops only -- the byte copy is dead
    // weight to this path, and it is most of the blob.
    std::memcpy(g_swap_scratch, g_arena + off, kBlobHdr);
    std::memcpy(g_swap_scratch + kBlobHdr, g_arena + off + pl_off, pl_bytes);
    // Revalidate exactly as Lookup does: an unchanged (epoch, key, off)
    // proves the copy coherent, because the arena is append-only inside an
    // epoch and slots publish key-last.
    if (g_epoch.load(std::memory_order_acquire) != e1 ||
        c.key.load(std::memory_order_acquire) != key || c.off != off) {
      ++g_swap.lk_stale;
      break;
    }
    uint32_t hdr5[5];
    std::memcpy(hdr5, g_swap_scratch, kBlobHdr);
    if (hdr5[0] != key || hdr5[1] != nd || hdr5[2] != no || hdr5[3] != npl ||
        hdr5[4] != ngd) {
      ++g_swap.lk_stale;
      break;
    }
    if (nd > count - dw) {
      ++g_swap.cross;
      break;
    }
    CtxPlanOp* plan = reinterpret_cast<CtxPlanOp*>(g_swap_scratch + kBlobHdr);
    const CtxPlanGuard* guards = reinterpret_cast<const CtxPlanGuard*>(
        g_swap_scratch + kBlobHdr + npl * sizeof(CtxPlanOp));
    const uint32_t dem =
        CtxPlanApplyGuards(raw + size_t(dw) * 4, nd, plan, npl, guards, ngd);
    g_swap.demoted += dem;
    // The ring recycles regions: a key whose plan drifted nearly everywhere
    // describes some other span, and replaying a plan of pure re-parses would
    // be correct but pointless. This is the density split, in plan form.
    if (dem * 4 > npl) {
      ++g_swap.dead;
      break;
    }
    CtxPlanBegin(w, plan, npl, dw, nd);
    ++g_swap.spans;
    g_swap.dwords += nd;
    return true;
  }
  ++g_swap.miss;
  return false;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
