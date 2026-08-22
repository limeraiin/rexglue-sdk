/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * [N-5 DE-TILE] See include/rex/graphics/nr_detile.h for what and why.       *
 *                                                                            *
 * Pure logic, like every other nr_* module: no cvar and no logging live here.*
 * The command processor latches the cvar into DetileSetEnabled() once a frame*
 * and prints the [nr-detile] line beside the other native-renderer reports.  *
 ******************************************************************************
 */

#include <rex/graphics/nr_detile.h>

#include <algorithm>
#include <cstdio>

namespace rex {
namespace graphics {
namespace nr {

namespace {

// A window scissor register is {x : 14, y : 14} with flags above.
inline uint32_t ScissorX(uint32_t v) { return v & 0x3FFF; }
inline uint32_t ScissorY(uint32_t v) { return (v >> 16) & 0x3FFF; }

// PA_SC_WINDOW_OFFSET packs two signed 15-bit offsets at +0 and +16.
inline int32_t WindowOffsetY(uint32_t v) {
  int32_t y = int32_t((v >> 16) & 0x7FFF);
  return y >= 0x4000 ? y - 0x8000 : y;
}

// The RB_*_INFO signature of a pass: the fields that identify the surface set,
// never the tile window, which is the thing that varies between bands.
struct PassSig {
  uint32_t surface_info;
  uint32_t color_info;
  uint32_t depth_info;
  bool operator==(const PassSig& o) const {
    return surface_info == o.surface_info && color_info == o.color_info &&
           depth_info == o.depth_info;
  }
  bool operator!=(const PassSig& o) const { return !(*this == o); }
};

inline PassSig SigOf(const DetileRegs& r) {
  return PassSig{r.surface_info, r.color_info, r.depth_info};
}

constexpr uint32_t kMaxTaps = 8;
constexpr uint32_t kMaxOut = 12;

struct State {
  bool enabled = false;

  // --- armed from the PREVIOUS frame ---
  bool armed = false;
  PassSig sig{};
  uint32_t full_height = 0;  // union of the bands, in guest rows
  uint32_t band_rows = 0;    // one band's height, in guest rows

  // --- this frame's observation ---
  bool obs_saw_band = false;
  bool obs_consistent = true;
  PassSig obs_sig{};
  uint32_t obs_full_height = 0;
  uint32_t obs_band_rows = 0;

  // --- this frame's resolve-destination guard ---
  uint32_t band0_dest[kMaxTaps] = {};
  uint32_t band0_taps = 0;
  uint32_t band_taps = 0;       // taps seen in the CURRENT repeat band
  int32_t band_tl_y = -1;       // which repeat band we are in
  uint32_t stride_per_band = 0; // measured destination stride between bands
  int32_t tap_filter = -1;      // -1 = extend every band-1 tap
  int32_t widen_seg_limit = -1; // -1 = widen every band-1 segment
  int32_t tail_mode = -1;       // -1 = execute tail rewrites, 0 = skip them

  // Tail-rewrite record for the report.
  struct TailRec {
    uint32_t dest, y0, y1, tap, bands_in;
  };
  TailRec tail_rec{};
  bool tail_rec_valid = false;
  uint64_t tail_seen = 0;
  uint64_t tail_skipped = 0;

  // Resolve census. The tail bisect's first city run logged nothing at all,
  // and `seen=0` alone cannot tell "the frame has no tail rewrite" from "the
  // predicate is wrong" - so every resolve reaching the observe site is
  // classified, and the destinations that reach the last gate are listed.
  uint64_t res_total = 0;
  uint64_t res_band = 0;     // refused: window offset Y != 0
  uint64_t res_insig = 0;    // refused: same signature as the tiled pass
  uint64_t res_nomatch = 0;  // outside the pass, destination did not match
  uint32_t last_band0_taps = 0;  // band0_taps as of the last frame end
  struct OutRec {
    uint32_t dest, ctrl, y0, y1;
    uint64_t n;
  };
  OutRec out_rec[kMaxOut] = {};
  uint32_t out_rec_n = 0;

  // Per-tap record of what band 1 actually resolved, for the log.
  struct TapRec {
    uint32_t ctrl, dest, pitch, y0, y1;
  };
  TapRec tap_rec[kMaxTaps] = {};
  uint32_t tap_rec_n = 0;
  bool guard_tripped = false;

  DetileStats stats{};
};

State g_st;

// The whole increment rests on this: every band renders strip [tl_y, br_y) of
// the frame with its window offset putting that strip back at the EDRAM
// origin. When it holds for every band, band 1 rendering [0, full_height) at
// offset 0 covers all of them, and nothing else about the bands can differ -
// the offline census proved the rest of the state block is identical.
inline bool BandIsWellFormed(const DetileRegs& r) {
  const uint32_t tl_y = ScissorY(r.scissor_tl);
  const uint32_t br_y = ScissorY(r.scissor_br);
  return br_y > tl_y && WindowOffsetY(r.window_offset) == -int32_t(tl_y) &&
         ScissorX(r.scissor_tl) == 0;
}

}  // namespace

void DetileSetEnabled(bool enabled) {
  if (!enabled && g_st.enabled) {
    const DetileStats keep = g_st.stats;
    g_st = State{};
    g_st.stats = keep;
  }
  g_st.enabled = enabled;
}

void DetileObserve(const DetileRegs& r, bool is_resolve, uint32_t copy_dest_base) {
  if (!g_st.enabled) {
    return;
  }
  const int32_t off_y = WindowOffsetY(r.window_offset);
  const uint32_t tl_y = ScissorY(r.scissor_tl);
  const uint32_t br_y = ScissorY(r.scissor_br);

  if (off_y != 0) {
    // A repeat band. This is where the pattern is learned, and it keeps being
    // learned in the frames whose bands we skip, because observation runs
    // BEFORE the skip decision. A de-tile that stopped observing what it skips
    // would disarm itself after one frame.
    if (!g_st.obs_saw_band) {
      g_st.obs_saw_band = true;
      g_st.obs_sig = SigOf(r);
      g_st.obs_band_rows = uint32_t(-off_y);
    } else {
      if (SigOf(r) != g_st.obs_sig) {
        // Two different surface sets tiling in one frame is not a shape this
        // increment models. Refuse the frame rather than guess.
        g_st.obs_consistent = false;
      }
      g_st.obs_band_rows = std::min(g_st.obs_band_rows, uint32_t(-off_y));
    }
    if (!BandIsWellFormed(r)) {
      g_st.obs_consistent = false;
    }
    g_st.obs_full_height = std::max(g_st.obs_full_height, br_y);

    // Destination-stride guard: each repeat band's k-th resolve must land a
    // whole number of bands further into the same destination than band 1's
    // k-th, with the same step every time. If it does not, band 1's extended
    // resolve is not writing what the three of them wrote.
    if (is_resolve && g_st.armed && g_st.band_rows) {
      if (int32_t(tl_y) != g_st.band_tl_y) {
        g_st.band_tl_y = int32_t(tl_y);
        g_st.band_taps = 0;
      }
      const uint32_t k = g_st.band_taps;
      if (k < g_st.band0_taps && k < kMaxTaps) {
        const uint32_t bands_in = tl_y / g_st.band_rows;
        const uint32_t delta = copy_dest_base - g_st.band0_dest[k];
        if (!bands_in || !delta || (delta % bands_in) != 0) {
          g_st.guard_tripped = true;
        } else {
          const uint32_t per_band = delta / bands_in;
          if (!g_st.stride_per_band) {
            g_st.stride_per_band = per_band;
          } else if (g_st.stride_per_band != per_band) {
            g_st.guard_tripped = true;
          }
        }
      }
      ++g_st.band_taps;
    }
    return;
  }

  // Band 1, or an untiled pass. Record band 1's resolve destinations so the
  // repeat bands have something to be checked against.
  if (is_resolve && g_st.armed && SigOf(r) == g_st.sig && tl_y == 0 &&
      br_y < g_st.full_height) {
    if (g_st.band0_taps < kMaxTaps) {
      g_st.band0_dest[g_st.band0_taps] = copy_dest_base;
    }
    ++g_st.band0_taps;
    // Reported beside `extended`: every band-1 tap of the tiled pass MUST be
    // extended to the full frame height, or the taps that were not leave their
    // destination's lower rows unwritten. City drive 1 read extended=1 against
    // taps=3 and that one line is what named the bug.
    ++g_st.stats.taps;
  }
}

bool DetileIsRepeatBand(const DetileRegs& r) {
  if (!g_st.enabled || !g_st.armed || g_st.guard_tripped) {
    return false;
  }
  if (WindowOffsetY(r.window_offset) == 0 || SigOf(r) != g_st.sig) {
    return false;
  }
  // Only a band that band 1 provably covered: inside [0, full_height), put
  // back at the EDRAM origin by its own offset.
  return BandIsWellFormed(r) && ScissorY(r.scissor_br) <= g_st.full_height;
}

uint32_t DetileBand0FullHeight(const DetileRegs& r) {
  if (!g_st.enabled || !g_st.armed || g_st.guard_tripped) {
    return 0;
  }
  if (WindowOffsetY(r.window_offset) != 0 || SigOf(r) != g_st.sig) {
    return 0;
  }
  if (ScissorY(r.scissor_tl) != 0 || ScissorY(r.scissor_br) >= g_st.full_height) {
    return 0;
  }
  return g_st.full_height;
}

DetileFrameVerdict DetileFrameEnd() {
  DetileFrameVerdict verdict{};
  if (!g_st.enabled) {
    // Count the control half too. City drive 1's OFF phase printed nothing at
    // all, which made half of a matched A/B unreadable.
    ++g_st.stats.frames_off;
    return verdict;
  }
  verdict.was_armed = g_st.armed;
  if (g_st.guard_tripped) {
    ++g_st.stats.guard_fail;
    verdict.guard_tripped = true;
  }
  const bool ok = g_st.obs_saw_band && g_st.obs_consistent && !g_st.guard_tripped &&
                  g_st.obs_band_rows && g_st.obs_full_height > g_st.obs_band_rows;
  if (ok) {
    g_st.armed = true;
    g_st.sig = g_st.obs_sig;
    g_st.full_height = g_st.obs_full_height;
    g_st.band_rows = g_st.obs_band_rows;
    ++g_st.stats.frames;
  } else {
    g_st.armed = false;
  }
  verdict.is_armed = g_st.armed;
  verdict.full_height = g_st.full_height;
  verdict.band_rows = g_st.band_rows;

  g_st.stats.full_height = g_st.full_height;
  g_st.stats.band_rows = g_st.band_rows;
  g_st.stats.dest_stride = g_st.stride_per_band;

  g_st.obs_saw_band = false;
  g_st.obs_consistent = true;
  g_st.obs_full_height = 0;
  g_st.obs_band_rows = 0;
  g_st.last_band0_taps = g_st.band0_taps;
  g_st.band0_taps = 0;
  g_st.band_taps = 0;
  g_st.band_tl_y = -1;
  g_st.guard_tripped = false;
  return verdict;
}

void DetileCountSkippedDraw() { ++g_st.stats.draws_skipped; }
void DetileCountSkippedResolve() { ++g_st.stats.resolves_skipped; }
void DetileCountScissorWidened() { ++g_st.stats.scissor_widened; }
void DetileCountExtendedResolve() { ++g_st.stats.resolves_extended; }

void DetileSetTapFilter(int32_t tap) { g_st.tap_filter = tap; }

void DetileSetWidenSegLimit(int32_t seg) { g_st.widen_seg_limit = seg; }

void DetileSetTailMode(int32_t mode) { g_st.tail_mode = mode; }

bool DetileTailSkipActive() { return g_st.tail_mode == 0; }

namespace {
// dest == band0_dest[k] + m*stride for m in 1..2 -> returns true and fills
// tap/bands_in. Offset 0 (a tap's own base, reused as bloom scratch at the
// start of the frame) is deliberately NOT a tail rewrite.
inline bool TailMatch(uint32_t dest, uint32_t* tap, uint32_t* bands_in) {
  if (!g_st.stride_per_band) {
    return false;
  }
  const uint32_t taps = std::min(g_st.band0_taps, kMaxTaps);
  for (uint32_t k = 0; k < taps; ++k) {
    const uint32_t delta = dest - g_st.band0_dest[k];
    if (delta && delta % g_st.stride_per_band == 0) {
      const uint32_t m = delta / g_st.stride_per_band;
      if (m >= 1 && m <= 2) {
        *tap = k;
        *bands_in = m;
        return true;
      }
    }
  }
  return false;
}
}  // namespace

bool DetileIsTailRewrite(const DetileRegs& r, uint32_t copy_dest_base) {
  if (!g_st.enabled || !g_st.armed) {
    return false;
  }
  // Outside the tiled pass: window offset 0 but a different surface set.
  if (WindowOffsetY(r.window_offset) != 0 || SigOf(r) == g_st.sig) {
    return false;
  }
  uint32_t tap, bands_in;
  if (!TailMatch(copy_dest_base, &tap, &bands_in)) {
    return false;
  }
  ++g_st.tail_seen;
  if (g_st.tail_mode == 0) {
    ++g_st.tail_skipped;
    return true;
  }
  return false;
}

void DetileNoteTailResolve(const DetileRegs& r, uint32_t copy_control,
                           uint32_t dest_base, uint32_t y0, uint32_t y1) {
  if (!g_st.enabled || !g_st.armed) {
    return;
  }
  ++g_st.res_total;
  if (WindowOffsetY(r.window_offset) != 0) {
    ++g_st.res_band;
    return;
  }
  if (SigOf(r) == g_st.sig) {
    ++g_st.res_insig;
    return;
  }
  // Outside the tiled pass. Whether or not the destination matches, record it:
  // the list of destinations that get this far is the evidence that says
  // whether the capture's tail rewrite happens in the live frame.
  uint32_t slot = kMaxOut;
  for (uint32_t i = 0; i < g_st.out_rec_n; ++i) {
    if (g_st.out_rec[i].dest == dest_base) {
      slot = i;
      break;
    }
  }
  if (slot == kMaxOut && g_st.out_rec_n < kMaxOut) {
    slot = g_st.out_rec_n++;
    g_st.out_rec[slot] = State::OutRec{dest_base, copy_control, y0, y1, 0};
  }
  if (slot < kMaxOut) {
    g_st.out_rec[slot].ctrl = copy_control;
    g_st.out_rec[slot].y0 = y0;
    g_st.out_rec[slot].y1 = y1;
    ++g_st.out_rec[slot].n;
  }
  uint32_t tap, bands_in;
  if (!TailMatch(dest_base, &tap, &bands_in)) {
    ++g_st.res_nomatch;
    return;
  }
  g_st.tail_rec = State::TailRec{dest_base, y0, y1, tap, bands_in};
  g_st.tail_rec_valid = true;
}

bool DetileFormatTail(char* out, size_t out_size) {
  if (!g_st.tail_seen && !g_st.tail_rec_valid) {
    return false;
  }
  const State::TailRec& t = g_st.tail_rec;
  std::snprintf(out, out_size,
                "dst=%08X tap%u+%ubands y=%u..%u seen=%llu skipped=%llu",
                t.dest, t.tap, t.bands_in, t.y0, t.y1,
                static_cast<unsigned long long>(g_st.tail_seen),
                static_cast<unsigned long long>(g_st.tail_skipped));
  return true;
}

uint32_t DetileWidenFullHeight(const DetileRegs& r, bool count_refusal) {
  const uint32_t fh = DetileBand0FullHeight(r);
  if (!fh) {
    return 0;
  }
  // band0_taps counts the band-1 taps already resolved this frame, which is
  // exactly the draw's segment index: 0 before tap 0 (world), 1 between taps
  // 0 and 1 (refraction), 2 between taps 1 and 2 (overlay).
  if (g_st.widen_seg_limit >= 0 &&
      int32_t(g_st.band0_taps) >= g_st.widen_seg_limit) {
    if (count_refusal) {
      ++g_st.stats.widen_refused;
    }
    return 0;
  }
  return fh;
}

uint32_t DetileResolveFullHeight(const DetileRegs& r) {
  const uint32_t fh = DetileBand0FullHeight(r);
  if (!fh) {
    return 0;
  }
  if (g_st.tap_filter >= 0) {
    // DetileObserve counted this tap at the draw stop already, so the tap
    // being resolved right now is band0_taps - 1.
    const int32_t cur = int32_t(g_st.band0_taps) - 1;
    if (cur != g_st.tap_filter) {
      return 0;
    }
  }
  return fh;
}

void DetileNoteTap(const DetileRegs& r, uint32_t copy_control, uint32_t dest_base,
                   uint32_t dest_pitch, uint32_t y0, uint32_t y1) {
  // GATE THIS. Ungated, the POST pass's own resolve reached here and overwrote
  // the last slot, so the table showed the post pass's destination
  // (0x1F2F8000) where band 1's third tap (0x1DCD3000, the one carrying the
  // clear bits) should have been. A diagnostic that silently reports a
  // different packet than the one it names is worse than no diagnostic.
  if (!g_st.enabled || !g_st.armed || !DetileBand0FullHeight(r)) {
    return;
  }
  const uint32_t k = g_st.band0_taps ? g_st.band0_taps - 1 : 0;
  if (k >= kMaxTaps) {
    return;
  }
  g_st.tap_rec[k] = State::TapRec{copy_control, dest_base, dest_pitch, y0, y1};
  if (k + 1 > g_st.tap_rec_n) {
    g_st.tap_rec_n = k + 1;
  }
}

void DetileNoteHeightUsed(uint32_t height_used) {
  if (g_st.enabled) g_st.stats.height_used = height_used;
}

struct RtRec {
  uint32_t base, pitch, msaa, depth, height;
};
RtRec g_rt[16] = {};
uint32_t g_rt_n = 0;

void DetileNoteRenderTarget(uint32_t base_tiles, uint32_t pitch_tiles, uint32_t msaa,
                            bool is_depth, uint32_t height_used) {
  if (!g_st.enabled) {
    return;
  }
  for (uint32_t i = 0; i < g_rt_n; ++i) {
    if (g_rt[i].base == base_tiles && g_rt[i].pitch == pitch_tiles &&
        g_rt[i].msaa == msaa && g_rt[i].depth == uint32_t(is_depth)) {
      g_rt[i].height = height_used;
      return;
    }
  }
  if (g_rt_n < 16) {
    g_rt[g_rt_n++] = RtRec{base_tiles, pitch_tiles, msaa, uint32_t(is_depth), height_used};
  }
}

uint32_t DetileFormatRenderTargets(char* out, size_t out_size) {
  size_t used = 0;
  out[0] = ' ';
  for (uint32_t i = 0; i < g_rt_n && used + 72 < out_size; ++i) {
    const RtRec& r = g_rt[i];
    const int n = std::snprintf(out + used, out_size - used,
                                "%s%s %u..%ut pitch=%ut msaa=%u h=%u",
                                i ? " | " : "", r.depth ? "Z" : "C", r.base,
                                r.base + (r.pitch ? ((r.height + 7) / 8) * r.pitch : 0),
                                r.pitch, r.msaa, r.height);
    if (n <= 0) break;
    used += size_t(n);
  }
  const uint32_t n = g_rt_n;
  g_rt_n = 0;
  return n;
}

bool DetileFormatResolveCensus(char* out, size_t out_size) {
  if (!g_st.res_total) {
    return false;
  }
  size_t used = size_t(std::snprintf(
      out, out_size,
      "total=%llu band=%llu insig=%llu nomatch=%llu | band0_taps=%u "
      "stride=%#x | outside:",
      static_cast<unsigned long long>(g_st.res_total),
      static_cast<unsigned long long>(g_st.res_band),
      static_cast<unsigned long long>(g_st.res_insig),
      static_cast<unsigned long long>(g_st.res_nomatch), g_st.last_band0_taps,
      g_st.stride_per_band));
  for (uint32_t i = 0; i < g_st.out_rec_n && used + 64 < out_size; ++i) {
    const State::OutRec& o = g_st.out_rec[i];
    const int n = std::snprintf(out + used, out_size - used,
                                " %08X(ctrl=%08X y=%u..%u n=%llu)", o.dest,
                                o.ctrl, o.y0, o.y1,
                                static_cast<unsigned long long>(o.n));
    if (n <= 0) break;
    used += size_t(n);
  }
  return true;
}

uint32_t DetileFormatTaps(char* out, size_t out_size) {
  size_t used = 0;
  out[0] = ' ';
  for (uint32_t i = 0; i < g_st.tap_rec_n && used + 64 < out_size; ++i) {
    const State::TapRec& r = g_st.tap_rec[i];
    const int n = std::snprintf(out + used, out_size - used,
                                "%s#%u ctrl=%08X dst=%08X pitch=%08X y=%u..%u",
                                i ? " | " : "", i, r.ctrl, r.dest, r.pitch, r.y0, r.y1);
    if (n <= 0) break;
    used += size_t(n);
  }
  return g_st.tap_rec_n;
}

const DetileStats& DetileGetStats() { return g_st.stats; }

}  // namespace nr
}  // namespace graphics
}  // namespace rex
