/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * [N-5 DE-TILE] Render the frame ONCE instead of once per EDRAM tile band.   *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_DETILE_H_
#define REX_GRAPHICS_NR_DETILE_H_

#include <cstdint>

namespace rex::graphics::nr {

// This game renders the heavy scene THREE TIMES a frame, once per EDRAM tile
// band, because 1280x256 at 4xMSAA is exactly half the 10 MB EDRAM for colour
// and the other half for depth. 62.5% of all draw executions at the city exist
// only to emulate that. Offline census over the N-1 oracle capture
// (tools/n5-detile-census.py, one heavy frame, 3,015 draw packets executed 3x,
// 6,030 cross-band compares) found the bands are PURE DUPLICATES: the whole
// state block 0x2000-0x23FF is band-identical except PA_SC_WINDOW_OFFSET,
// PA_SC_WINDOW_SCISSOR_TL/BR and RB_COPY_DEST_BASE, and the ALU-float, fetch
// and bool/loop constant window matched in every single compare. The three
// resolve destinations are contiguous, +0x140000 each, which is exactly the
// tiled offset of 256 rows at pitch 1280.
//
// So: let band 1 rasterise the whole frame (widen its window scissor to the
// union of the bands), let its resolve cover the whole frame (extend the
// resolve rect to the same height, which also extends the clear that rides
// it), and do not execute bands 2 and 3 at all.
//
// The EDRAM expansion in xenos.h (kEdramExpandLog2) is what makes a 720-row
// 4xMSAA colour+depth pair addressable at once; without it this cannot work.
//
// ⚠ OBSERVE BEFORE SKIPPING. The pattern is re-learned every frame from the
// packets themselves, INCLUDING the ones whose dispatch we drop. A de-tile
// that stopped observing what it skipped would disarm itself after one frame -
// the same self-latching trap that made tile probe mode 2 skip every band.

struct DetileRegs {
  uint32_t window_offset;  // PA_SC_WINDOW_OFFSET
  uint32_t scissor_tl;     // PA_SC_WINDOW_SCISSOR_TL
  uint32_t scissor_br;     // PA_SC_WINDOW_SCISSOR_BR
  uint32_t surface_info;   // RB_SURFACE_INFO   (pitch + MSAA)
  uint32_t color_info;     // RB_COLOR_INFO     (base + format)
  uint32_t depth_info;     // RB_DEPTH_INFO     (base + format)
};

struct DetileStats {
  uint64_t frames;             // frames the pass was armed for
  uint64_t draws_skipped;      // repeat-band draws not dispatched
  uint64_t resolves_skipped;   // repeat-band resolves not dispatched
  uint64_t scissor_widened;      // band-1 draws given the full-frame scissor
  uint64_t widen_refused;      // band-1 draws NOT widened by the segment limit
  uint64_t resolves_extended;  // band-1 resolves given the full-frame height
  uint64_t guard_fail;         // destination-stride guard violations
  uint64_t taps;               // band-1 resolve taps of the tiled pass seen
  uint64_t frames_off;         // frames counted while the mode was disabled
  uint32_t full_height;        // the union height in guest rows (720)
  uint32_t band_rows;          // one band's height in guest rows (256)
  uint32_t dest_stride;        // measured destination stride between bands
  uint32_t height_used;        // guest rows the RT cache last decided to OWN
};

struct DetileFrameVerdict {
  bool was_armed;      // the frame that just ended ran de-tiled
  bool is_armed;       // the next frame will
  bool guard_tripped;  // the destination-stride guard fired this frame
  uint32_t full_height;
  uint32_t band_rows;
};

// Latched once per frame by the command processor from its cvar. Turning it
// off drops all arming so a re-enable re-learns from scratch.
void DetileSetEnabled(bool enabled);

// Called for EVERY draw stop and resolve, before any skip decision.
void DetileObserve(const DetileRegs& regs, bool is_resolve, uint32_t copy_dest_base);

// True if this packet belongs to a repeat tile band that band 1 already covers.
bool DetileIsRepeatBand(const DetileRegs& regs);

// The full frame height in guest rows if this packet is band 1 of the tiled
// pass and its window scissor is one band rather than the whole frame; 0
// otherwise. Used both to widen the draw scissor and to extend the resolve.
uint32_t DetileBand0FullHeight(const DetileRegs& regs);

// The resolve-side variant. Same answer, except that it honours the per-tap
// bisect filter: with a tap selected, only that band-1 tap is extended to the
// full frame height and the others stay at their band height. The tiled pass
// snapshots the frame three times (after the main draws, after a short depth
// group, after the transparent group); if exactly one of those extensions is
// what duplicates an overlay, this is the experiment that says which.
uint32_t DetileResolveFullHeight(const DetileRegs& regs);

// -1 = extend every tap (normal). 0..7 = extend only that tap.
void DetileSetTapFilter(int32_t tap);

// The draw-side variant. Same answer as DetileBand0FullHeight, except that it
// honours the widen SEGMENT limit. Band 1's draws are segmented by the taps:
// segment 0 runs before tap 0 (the world), segment 1 between taps 0 and 1
// (the refraction group), segment 2 between taps 1 and 2 (the overlay group).
// The offline reader census proved the visible rows below the band line come
// ONLY from tap 2's write of the post-pass input, so the duplicated overlay
// must be pixels that band 1's WIDENED draws put into EDRAM rows 256+ before
// tap 2 - and the segment limit is the bisect that says which segment draws
// them. count_refusal is false for the extent estimator so each refused draw
// is counted exactly once, by the command processor.
uint32_t DetileWidenFullHeight(const DetileRegs& regs, bool count_refusal);

// -1 = widen every band-1 segment (normal). N >= 0 = widen only segments
// below N (2 = world + refraction widened, overlay left at band height).
void DetileSetWidenSegLimit(int32_t seg);

// Called from GetResolveInfo once the final rect is known, so the log can show
// what each band-1 tap actually resolved instead of what it was assumed to.
void DetileNoteTap(const DetileRegs& regs, uint32_t copy_control, uint32_t dest_base,
                   uint32_t dest_pitch, uint32_t y0, uint32_t y1);

// Formats the per-tap table into `out` (NUL-terminated). Returns the tap count.
uint32_t DetileFormatTaps(char* out, size_t out_size);

// Promotes this frame's observation into next frame's arming. Call once per
// swap, on the command-processor thread.
DetileFrameVerdict DetileFrameEnd();

void DetileCountSkippedDraw();
void DetileCountSkippedResolve();
void DetileCountScissorWidened();
void DetileCountExtendedResolve();

// Diagnostic, poked by RenderTargetCache::Update. If de-tile is armed and this
// is not the full frame height, band 1 is rasterising into rows that the
// render target does not own and the resolve therefore cannot reach - which
// is one of the two ways the frame can come out wrong below the band line.
void DetileNoteHeightUsed(uint32_t height_used);

// Every render target the cache binds, deduped, so the EDRAM layout under the
// expansion can be read directly instead of inferred. Overlapping ranges
// between surfaces of different MSAA counts map EDRAM tiles to DIFFERENT pixel
// rows, which is one of the few mechanisms that can shift content vertically.
void DetileNoteRenderTarget(uint32_t base_tiles, uint32_t pitch_tiles, uint32_t msaa,
                            bool is_depth, uint32_t height_used);
uint32_t DetileFormatRenderTargets(char* out, size_t out_size);

const DetileStats& DetileGetStats();

}  // namespace rex::graphics::nr

#endif  // REX_GRAPHICS_NR_DETILE_H_
