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
};

// Blob layout at `off`: u32 key, u32 ndwords, u32 nops, then the ops
// (nops * sizeof(CtxMemoOp)), then the span byte copy (ndwords * 4). The
// header duplicates the slot so the reader can prove the copy it took is the
// blob the slot meant.
constexpr uint32_t kBlobHdr = 12;

Slot g_tbl[kTblSize];
uint8_t* g_arena = nullptr;
uint32_t g_cursor = 0;  // producer-private
std::atomic<uint32_t> g_epoch{1};
bool g_active = false;

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
            std::memcmp(g_arena + c.off + kBlobHdr +
                            size_t(c.nops) * sizeof(CtxMemoOp),
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

  // One parse of the just-written span through the real walker, its own memo
  // recorder attached: build-side decode semantics are the walk's by
  // construction. Bin state all-ones so every packet (all three tiles'
  // predicated blocks included) is decoded and recorded.
  g_build_ops.clear();
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
  if (w.cursor != ndwords || g_build_ops.size() > kBuildOpCap) {
    // Framing guaranteed the landing, so neither should happen; counted so a
    // violation can never hide.
    ++(w.cursor != ndwords ? s.parse_fail : s.ops_overflow);
    return;
  }

  const uint32_t nops = uint32_t(g_build_ops.size());
  const uint32_t ops_bytes = nops * uint32_t(sizeof(CtxMemoOp));
  const uint32_t need = kBlobHdr + ops_bytes + ndwords * 4;
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
  const uint32_t hdr3[3] = {key, ndwords, nops};
  std::memcpy(blob, hdr3, kBlobHdr);
  if (ops_bytes) std::memcpy(blob + kBlobHdr, g_build_ops.data(), ops_bytes);
  std::memcpy(blob + kBlobHdr + ops_bytes, base + start, size_t(ndwords) * 4);
  g_cursor += (need + 15u) & ~15u;

  target->key.store(0, std::memory_order_relaxed);
  target->epoch = pub_epoch;
  target->off = off;
  target->ndwords = ndwords;
  target->nops = nops;
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
  const CtxMemoOp* ops;
  const uint8_t* bytes;
};

uint8_t g_view_scratch[kBlobHdr + kBuildOpCap * sizeof(CtxMemoOp) +
                       kMaxSpanBytes];

bool Lookup(uint32_t key, TmplView* v, Slot** out_slot) {
  const uint32_t h = SlotIndex(key);
  for (uint32_t p = 0; p < kProbe; ++p) {
    Slot& c = g_tbl[(h + p) & (kTblSize - 1)];
    if (c.key.load(std::memory_order_acquire) != key) continue;
    const uint32_t e1 = g_epoch.load(std::memory_order_acquire);
    const uint32_t off = c.off;
    const uint32_t nd = c.ndwords;
    const uint32_t no = c.nops;
    if (c.epoch != e1 || !nd || nd > kMaxSpanDwords || no > kBuildOpCap) {
      continue;
    }
    const uint32_t bytes = kBlobHdr + no * uint32_t(sizeof(CtxMemoOp)) + nd * 4;
    if (uint64_t(off) + bytes > kArenaBytes) continue;
    std::memcpy(g_view_scratch, g_arena + off, bytes);
    // Revalidate: the arena is append-only inside an epoch and slots publish
    // key-last, so an unchanged (epoch, key, off) proves the copy coherent.
    if (g_epoch.load(std::memory_order_acquire) != e1 ||
        c.key.load(std::memory_order_acquire) != key || c.off != off) {
      ++g_stats.lookup_stale;
      continue;
    }
    uint32_t hdr3[3];
    std::memcpy(hdr3, g_view_scratch, kBlobHdr);
    if (hdr3[0] != key || hdr3[1] != nd || hdr3[2] != no) {
      ++g_stats.lookup_stale;
      continue;
    }
    v->ndwords = nd;
    v->nops = no;
    v->ops = reinterpret_cast<const CtxMemoOp*>(g_view_scratch + kBlobHdr);
    v->bytes = g_view_scratch + kBlobHdr + no * sizeof(CtxMemoOp);
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

CtxWalker g_wb;
StateContext g_ctx_b;  // persistent, fed by replayed spans only
CtxWalkStats g_stats_b;
size_t g_acur = 0;      // sweep cursor into g_a
uint32_t g_dw0 = 0;     // current span window, absolute dwords
uint32_t g_dw1 = 0;
uint32_t g_span_key = 0;
uint32_t g_span_ne = 0;

void NoteNe(uint8_t kind, uint32_t dw, uint32_t a_reg, uint32_t a_val,
            uint32_t b_reg, uint32_t b_val) {
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
    NoteNe(e.kind, e.dw, a.reg, a.a, e.reg, e.a);
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

void AttributeStale(const uint8_t* live, const uint8_t* stored,
                    uint32_t ndwords, uint32_t span_key) {
  TmplStats& s = g_stats;
  // Density split first: a finalize/patch touches a few dwords; a recycled
  // region (dead key) differs nearly everywhere, and attributing its diffs
  // to packet classes would just be noise wearing names.
  uint32_t ndiff = 0;
  for (uint32_t i = 0; i < ndwords; ++i) {
    if (std::memcmp(live + size_t(i) * 4, stored + size_t(i) * 4, 4) != 0) {
      ++ndiff;
    }
  }
  if (ndiff * 2 > ndwords) {
    ++s.stale_cls[kTmplScDead];
    return;
  }
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
void FrameWindowBins(const uint8_t* raw, uint32_t dw, uint32_t dw_end,
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
}

}  // namespace

void TmplCompareBuffer(const uint8_t* raw, uint32_t ptr, uint32_t count,
                       uint64_t bin_select, uint64_t bin_mask,
                       CtxMemReadFn mem_read, void* mem_user) {
  if (!g_active || !count) return;
  TmplStats& s = g_stats;
  g_mem_read = mem_read;
  g_mem_user = mem_user;
  const uint32_t pbase = ptr & kPhysMask;

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
        if (std::memcmp(raw + size_t(dw) * 4, v.bytes,
                        size_t(v.ndwords) * 4) != 0) {
          // Bytes differ from the build. Two very different causes hide
          // here, so discriminate: a surviving first header (layout intact,
          // payload changed) smells like a re-record the feed missed -- a
          // correctness hole; a different first header means the ring
          // recycled this region and the key is dead -- store hygiene.
          ++s.spans_stale;
          uint32_t idx = 0;
          while (idx < v.ndwords &&
                 std::memcmp(raw + size_t(dw + idx) * 4,
                             v.bytes + size_t(idx) * 4, 4) == 0) {
            ++idx;
          }
          const bool hdr_eq =
              std::memcmp(raw + size_t(dw) * 4, v.bytes, 4) == 0;
          ++(hdr_eq ? s.stale_hdr_eq : s.stale_hdr_ne);
          AttributeStale(raw + size_t(dw) * 4, v.bytes, v.ndwords,
                         g_span_key);
          if (!s.st_armed && idx < v.ndwords) {
            s.st_armed = 1;
            s.st_key = g_span_key;
            s.st_idx = idx;
            uint32_t sd, ld;
            std::memcpy(&sd, v.bytes + size_t(idx) * 4, 4);
            std::memcpy(&ld, raw + size_t(dw + idx) * 4, 4);
            s.st_stored = __builtin_bswap32(sd);
            s.st_live = __builtin_bswap32(ld);
          }
          // Kill the dead key so one recycled region cannot report itself
          // once per execution forever. A racing producer publish can lose a
          // fresh template here; the next feed of that span republishes it
          // (the unchanged-check needs a key match, so it misses and
          // rebuilds).
          vslot->key.store(0, std::memory_order_relaxed);
          FrameWindowBins(raw, g_dw0, g_dw1, count, &bbin);
          while (g_acur < g_a.size() && g_a[g_acur].dw < g_dw1) {
            ++s.a_uncovered;
            ++g_acur;
          }
        } else {
          g_span_ne = 0;
          CtxWalkBegin(&g_wb, v.bytes, v.ndwords, g_span_key, &g_ctx_b,
                       nullptr, 0, &g_stats_b, mem_read, mem_user, BShader,
                       nullptr, nullptr, nullptr, bbin.select, bbin.mask,
                       BReg, nullptr, nullptr, nullptr);
          g_wb.range_fn = BRange;
          g_wb.range_user = &g_wb;
          g_wb.rep = v.ops;
          g_wb.rep_n = v.nops;
          g_wb.rep_i = 0;
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
          ++(g_span_ne ? s.spans_ne : s.spans_eq);
          bbin = g_wb.bin;
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
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
