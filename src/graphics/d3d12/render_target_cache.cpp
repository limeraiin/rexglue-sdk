/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include "thirdparty/dxbc/DXBCChecksum.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/graphics/d3d12/deferred_command_list.h>
#include <rex/graphics/d3d12/render_target_cache.h>
#include <rex/graphics/d3d12/texture_cache.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/format/dxbc.h>
#include <rex/graphics/pipeline/shader/dxbc_translator.h>
#include <rex/graphics/trace_writer.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/string.h>
#include <rex/ui/d3d12/d3d12_provider.h>
#include <rex/ui/d3d12/d3d12_util.h>

REXCVAR_DEFINE_BOOL(native_stencil_value_output_d3d12_intel, false, "GPU/D3D12",
                    "Native stencil value output for Intel D3D12");

// [NR-DRES] N-10a. Default ON since 2026-08-23: the verify gate passed at the
// city (naruto_614) - bit-identical to the legacy dump+vendored chain on every
// destination where that chain is trustworthy, and on high-base depth sources
// (>= 2048 host tiles) the legacy chain reads the WRONG EDRAM region (11-bit
// vendored base field, warned at boot) while the direct path returns real
// depth. The cvar survives for the A/B and per-resolve fallback stays.
// [NR-DRES] N-10a direct resolve is UNCONDITIONAL since 2026-08-24 (user
// directive: confirmed wins are code, not flags). It was `gpu_nr_direct_
// resolve`, default ON, city-gated at 100% coverage and bit-identical
// everywhere legacy is trustworthy (naruto_614..625); the eligible-set
// preflight + counted fallback below IS the safety, not a cvar.

REXCVAR_DEFINE_BOOL(gpu_nr_direct_resolve_verify, false, "GPU/D3D12",
                    "[NR-DRES] Divergence gate: run the legacy dump+vendored resolve AND the "
                    "direct resolve on the same destination, snapshot both results, compare "
                    "on the CPU and report [nr-dres] diverge counters. Slow; diagnostic only.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_nr_dres_verify_control, false, "GPU/D3D12",
                    "[NR-DRES] Verify harness control: with verify on, snapshot the legacy "
                    "result TWICE with nothing between instead of legacy-vs-direct. Any "
                    "diverge in this mode is a harness bug, not a semantic one.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(gpu_nr_dres_force_sample, -1, "GPU/D3D12",
                     "[NR-DRES] Diagnostic: force the host sample index the direct resolve "
                     "reads (-1 = the computed mapping). For isolating sample-mapping "
                     "divergence on MSAA sources.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// [N-10b deletion c] render_target_path_d3d12 is DELETED: the D3D12 path is
// host render targets, unconditionally. The ROV/pixel-shader-interlock path
// died with it (naruto_626 proved RTV on Intel; NR requires host RTs).

REXCVAR_DEFINE_BOOL(native_stencil_value_output, true, "GPU", "Enable native stencil value output");

// [NR-DETILE] N-6. The de-tile artifact is source_row = dest_row mod 512, and
// 512 guest rows is 2048 tiles, exactly the console EDRAM. The ownership maths
// says the colour RT owns all 2880 tiles, so the question is what the DUMP
// actually covers and who owns each piece. This prints the resolve's dump
// request beside the rectangles it produced, once a second.
// [NR-DETILE] N-6-2. Every writer-side register has now measured correct -
// host viewport rows 0..720 on 65k band-1 draws, scissor widened to match, RT
// 1280x768, resolve span 90 tile rows - and yet only the top 256 rows carry
// content. So stop reading registers and read the PIXELS: after the band-1
// colour tap has dumped the render targets into edram_buffer_, copy that span
// to a readback heap and report, per tile row, whether it is flat (cleared or
// never written) or varied (rasterised). That splits the last fork - "band 1
// never rasterised below row 256" against "it did, and something later loses
// it" - without needing anyone to drive.
REXCVAR_DEFINE_BOOL(gpu_nr_detile_edram_probe, false, "GPU",
                    "[nr-detile] Read the band-1 colour tap's EDRAM span back "
                    "and report per-tile-row flat/varied, once a second");

REXCVAR_DEFINE_BOOL(gpu_nr_dump_probe, false, "GPU",
                    "[nr-detile] Log each resolve's EDRAM dump request and the "
                    "rectangles it resolved to, with the owning render target "
                    "of each, once per second.");

// [xfer] The in-place A/B for the render-target-cache ownership transfers
// (the EDRAM aliasing emulation: every rebind of a host render target over
// tiles another one last wrote copies those tiles across). Seconds per phase;
// 0 = off (every transfer performed, the shipped behaviour). Phase "all" then
// phase "skip" (no transfer performed; every host render target keeps its own
// contents, ownership still tracked so resolves find their source). Read with
// the 1 Hz [xfer] line and tools/geo-sum.py. Temporary: deleted once the
// drive decides.
REXCVAR_DEFINE_INT32(gpu_xfer_cycle, 0, "GPU/D3D12",
                     "[xfer] A/B: seconds per phase, all transfers / no transfers (0 = off).");

// [NR-DETILE] N-6-5. The frozen-rows instrument's companion.
//
// The city EDRAM readback says rows 256..720 of BOTH the tiled colour and the
// tiled depth surface are byte-identical across 104 samples taken a second
// apart while driving, while rows 0..255 differ in every sample. Written once,
// never again - which is the user's "it looked fine only the first frame" in
// numbers. Nothing draws there and nothing CLEARS there.
//
// The offline city census (tools/n6-band1-clip-census.py, frame 2500) rules
// out the guest: screen scissor is 0..8192 on every draw, band 1's viewport is
// y 0..720, clip_disable is 0, vtx_window_offset_enable is 1, and the three
// bands carry exactly 3015 draws each, so the guest emits the same list to
// every band. It also names the depth convention: RB_DEPTHCONTROL zfunc is 6
// (GEQUAL) on every single draw - reverse Z, where a LARGER stored value is
// nearer. The frozen depth below row 256 reads 0xD3..0xE6 against 0x4E..0x63
// above it, so those rows hold NEAR values. If they are never cleared back to
// far, every widened draw below row 256 fails GEQUAL from frame 2 onwards and
// writes neither depth nor colour - which is exactly a freeze.
//
// So: report what each resolve clear actually covers. Rect, both render target
// keys, both tile ranges, both original bases, and the transfer counts, gated
// to the clears and deduped by signature. The clear rect is derived from the
// registers and the resolve rect, so this is STRUCTURAL and readable at the
// title screen, where the tiled pass runs with all three taps.
REXCVAR_DEFINE_BOOL(gpu_nr_detile_clear_probe, false, "GPU",
                    "[nr-detile] Log every resolve clear's rectangle, render "
                    "targets and EDRAM tile ranges, once per second per shape.");

// [NR-XFER] N-10b. Self-referential host depth (a depth RT regaining
// ownership of a range with itself as the best host depth source) is
// UNCONDITIONALLY a scratch-texture snapshot read through the ordinary
// host-depth TEXTURE shader variant with identity addressing. The legacy
// compute store into the EDRAM buffer and the is_copy buffer-read shader
// variant were deleted after the naruto_627 gate (hds native 7189 /
// legacy 0 over a full city drive, no fatals, no visual issues).

namespace rex::graphics::d3d12 {

// Generated with `xb buildshaders`.
namespace shaders {
#include "../shaders/bytecode/d3d12_5_1/clear_uint2_ps.h"
#include "../shaders/bytecode/d3d12_5_1/fullscreen_cw_vs.h"
#include "../shaders/bytecode/d3d12_5_1/passthrough_position_xy_vs.h"
// [N-10b deletion c] the 4 resolve_clear_* ROV EDRAM-clear blobs are DELETED
// (D3D12; the Vulkan SPIR-V twins stay until the Vulkan pass).
#include "../shaders/bytecode/d3d12_5_1/resolve_fast_32bpp_1x2xmsaa_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_fast_32bpp_1x2xmsaa_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_fast_32bpp_4xmsaa_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_fast_32bpp_4xmsaa_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_fast_64bpp_1x2xmsaa_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_fast_64bpp_1x2xmsaa_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_fast_64bpp_4xmsaa_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_fast_64bpp_4xmsaa_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_128bpp_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_128bpp_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_16bpp_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_16bpp_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_32bpp_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_32bpp_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_64bpp_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_64bpp_scaled_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_8bpp_cs.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_full_8bpp_scaled_cs.h"
}  // namespace shaders

namespace {

// [N-6] THE BAND ARTIFACT. The resolve shader blobs under
// src/graphics/shaders/bytecode are VENDORED PRECOMPILED DXBC - there is no
// HLSL for them anywhere in this tree - and they were compiled against the
// console's 2048-tile (10 MB) EDRAM. Every one of them ends its EDRAM address
// maths with a hardcoded wrap:
//
//   imad r0.x, r0.x, l(1280), r1.x       // tile * samples_per_tile + intra
//   udiv null, r0.x, r0.x, l(0x00280000) // r0.x %= 2048 * 1280
//
// We are not a 2048-tile EDRAM any more: kEdramExpandLog2 gives every surface
// 4x the vertical room so one pass can own all 720 rows. A de-tiled 1280x720
// 4xMSAA colour resolve spans 90 tile rows x 32 tiles = 2880 tiles, and at 4
// tiles per guest row the wrap at tile 2048 lands on GUEST ROW 512 exactly -
// so output rows 512..720 read back EDRAM rows 0..208. That is the measured
// artifact, to the row.
//
// Stock (three 256-row bands) never crosses it: 32 tile rows x 32 = 1024
// tiles, half the wrap. That is why this only shows with de-tile on.
//
// The blobs cannot be recompiled from anything we have, so rewrite the wrap
// constant in the token stream and re-sign the container. Driven off the
// xenos:: constants so it can never go stale again.
constexpr uint32_t kEdramTileSamples =
    xenos::kEdramTileWidthSamples * xenos::kEdramTileHeightSamples;
constexpr uint32_t kConsoleEdramWrapDwords = xenos::kEdramGuestTileCount * kEdramTileSamples;
constexpr uint32_t kHostEdramWrapDwords = xenos::kEdramTileCount * kEdramTileSamples;

// [N-6-6] The resolution-SCALED variants have no literal to find. Their tile
// stride is not a compile-time constant (it is scale_x * scale_y * samples),
// so they build the same modulus as a SHIFT:
//
//   imul null, r1.y, r7.z, r7.y     ; r1.y = tile samples, from the scales
//   imad r0.x, r0.x, r1.y, r2.z     ; addr = tile * tile_samples + intra
//   ishl r1.y, r1.y, l(11)          ; * 2048 = the console tile count
//   udiv null, r0.x, r0.x, r1.y     ; addr %= that
//
// 11 is log2 of the console's 2048 tiles, and it is a shift AMOUNT, which is
// why the value scan that fixed the unscaled blobs reported "0 patched, 9
// left" at resolution_scale 2 and the bottom of the frame banded again.
// tools/n6-scan-scaled-wrap.py verified the idiom is exact and unique: all 11
// scaled blobs have exactly one ishl-immediately-before-udiv, all 11 unscaled
// ones have none.
constexpr uint32_t ConstexprLog2(uint32_t value) {
  return value <= 1 ? 0 : 1 + ConstexprLog2(value >> 1);
}
constexpr uint32_t kConsoleEdramWrapShift = ConstexprLog2(xenos::kEdramGuestTileCount);
constexpr uint32_t kHostEdramWrapShift = ConstexprLog2(xenos::kEdramTileCount);

// Rewrites the console wrap constant in one vendored resolve blob. Returns
// false (and leaves `patched` empty) when the constant is not a literal in the
// shader code chunk, which is the case for the resolution-scaled variants:
// they build the modulus at runtime as `tile_samples << 11` instead.
bool PatchResolveShaderEdramWrap(const void* source, size_t source_size, const char* debug_name,
                                 std::vector<uint8_t>& patched, bool* by_shift_out = nullptr) {
  if (by_shift_out) {
    *by_shift_out = false;
  }
  patched.clear();
  if (!source || source_size < sizeof(dxbc::ContainerHeader)) {
    return false;
  }
  patched.assign(static_cast<const uint8_t*>(source),
                 static_cast<const uint8_t*>(source) + source_size);

  auto& container_header = *reinterpret_cast<dxbc::ContainerHeader*>(patched.data());
  if (container_header.fourcc != dxbc::ContainerHeader::kFourCC ||
      container_header.blob_count == 0) {
    patched.clear();
    return false;
  }
  const uint32_t* const blob_offsets = reinterpret_cast<const uint32_t*>(
      patched.data() + sizeof(dxbc::ContainerHeader));
  if (sizeof(dxbc::ContainerHeader) + sizeof(uint32_t) * container_header.blob_count >
      source_size) {
    patched.clear();
    return false;
  }

  // Only the executable code chunk - a stray match in RDEF strings or in the
  // statistics blob must not be rewritten.
  uint32_t replacements = 0;
  uint32_t shift_replacements = 0;
  for (uint32_t blob_index = 0; blob_index < container_header.blob_count; ++blob_index) {
    const uint32_t blob_offset = blob_offsets[blob_index];
    if (blob_offset + sizeof(dxbc::BlobHeader) > source_size) {
      continue;
    }
    auto& blob_header = *reinterpret_cast<dxbc::BlobHeader*>(patched.data() + blob_offset);
    if (blob_header.fourcc != dxbc::BlobHeader::FourCC::kShaderEx) {
      continue;
    }
    const size_t body_offset = blob_offset + sizeof(dxbc::BlobHeader);
    if (body_offset + blob_header.size_bytes > source_size) {
      continue;
    }
    uint32_t* const body = reinterpret_cast<uint32_t*>(patched.data() + body_offset);
    const size_t body_dwords = blob_header.size_bytes / sizeof(uint32_t);
    for (size_t i = 0; i < body_dwords; ++i) {
      if (body[i] == kConsoleEdramWrapDwords) {
        body[i] = kHostEdramWrapDwords;
        ++replacements;
      }
    }

    // [N-6-6] The scaled form. Walk the instruction stream by the length field
    // alone - no operand decoding is needed to segment instructions - and
    // rewrite the shift amount of an ISHL that is IMMEDIATELY followed by a
    // UDIV. Anchoring on the pair rather than on the bare value 11 is what
    // makes this safe: the blobs are full of other shifts by 11-ish amounts,
    // and none of them feed a modulus.
    if (body_dwords > 2) {
      size_t prev_start = 0;
      uint32_t prev_length = 0;
      uint32_t prev_opcode = UINT32_MAX;
      // The chunk opens with the version token and its own dword length.
      size_t i = 2;
      while (i < body_dwords) {
        const uint32_t token = body[i];
        const uint32_t opcode = token & 0x7FF;
        uint32_t length = (token >> 24) & 0x7F;
        if (opcode == uint32_t(dxbc::Opcode::kCustomData)) {
          length = (i + 1 < body_dwords) ? body[i + 1] : 0;
        }
        if (!length || i + length > body_dwords) {
          break;
        }
        if (opcode == uint32_t(dxbc::Opcode::kUDiv) &&
            prev_opcode == uint32_t(dxbc::Opcode::kIShL) && prev_length >= 2) {
          // Immediate operands come last, so the shift amount is the final
          // dword of the ISHL.
          uint32_t& shift = body[prev_start + prev_length - 1];
          if (shift == kConsoleEdramWrapShift) {
            shift = kHostEdramWrapShift;
            ++shift_replacements;
          }
        }
        prev_opcode = opcode;
        prev_start = i;
        prev_length = length;
        i += length;
      }
    }
  }

  if (!replacements && !shift_replacements) {
    patched.clear();
    return false;
  }
  if (by_shift_out) {
    *by_shift_out = shift_replacements != 0;
  }
  // One wrap per shader, in exactly one of the two forms. Anything else means
  // the blob changed shape under us and the assumption above needs re-checking
  // against the disassembly (tools/n6-scan-scaled-wrap.py prints it).
  if (replacements + shift_replacements != 1) {
    REXGPU_WARN(
        "D3D12RenderTargetCache: resolve shader {} had {} literal and {} shift "
        "EDRAM wraps, expected 1 in total - re-check the bytecode",
        debug_name, replacements, shift_replacements);
  }

  CalculateDXBCChecksum(reinterpret_cast<unsigned char*>(patched.data()),
                        static_cast<unsigned int>(patched.size()),
                        reinterpret_cast<unsigned int*>(&container_header.hash));
  return true;
}

// [N-6] SECOND HALF OF THE SAME STALENESS, and the horizontal one. L1.1
// widened ResolveEdramInfo (pitch 10 -> 9, base 11 -> 13) to carry the
// expanded EDRAM, but the vendored blobs above still UNPACK THE OLD LAYOUT.
// Straight out of resolve_full_32bpp_cs:
//
//   and  r0.xyzw, CB0[0][0].xzzz, l(1023, ...)             pitch = 10 bits @0
//   ubfe r4.xyzw, l(2,11,4,1), l(10,13,24,28), CB0[0][0].x msaa@10 base@13
//                                                          format@24 64bpp@28
//
// so every field above pitch is read one to two bits off. For the de-tiled
// colour tap (pitch 32, k4X, base 0, format 0) the packed word is 0x420 and
// the blob decodes msaa as k2X: it then stops doubling the sample X
// coordinate and copies the LEFT HALF of the surface stretched over the full
// width. That is the "only the top left half renders" report.
//
// Do not move the widths again - the blobs are the consumers and cannot be
// recompiled from anything in this tree. Keep the internal struct wide and
// re-pack into the layout they actually read, at the one place the value is
// handed to them.
//
// base_tiles is the one field that will not fit: the blobs give it 11 bits and
// our host bases reach 8188. Colour taps are base 0, so the de-tile path is
// exact. A depth tap at host base 4096 truncates to 0 - which is exactly what
// it already did before this function existed, so this is no regression, but
// it is counted instead of silent. The real fix is to bind the EDRAM view at a
// per-resolve tile offset and always pass base 0; that needs a per-resolve
// descriptor on both the bindless and non-bindless paths.
constexpr uint32_t kVendoredResolvePitchTilesBits = 10;
constexpr uint32_t kVendoredResolveBaseTilesBits = 11;

uint32_t PackResolveEdramInfoForVendoredShader(draw_util::ResolveEdramInfo info,
                                               bool& base_truncated_out) {
  constexpr uint32_t kMsaaShift = kVendoredResolvePitchTilesBits;
  constexpr uint32_t kIsDepthShift = kMsaaShift + xenos::kMsaaSamplesBits;
  constexpr uint32_t kBaseShift = kIsDepthShift + 1;
  constexpr uint32_t kFormatShift = kBaseShift + kVendoredResolveBaseTilesBits;
  constexpr uint32_t kFormatIs64bppShift = kFormatShift + xenos::kRenderTargetFormatBits;
  constexpr uint32_t kFillHalfPixelOffsetShift = kFormatIs64bppShift + 1;
  static_assert(kFillHalfPixelOffsetShift < 32,
                "The layout the vendored resolve blobs unpack must fit in the packed dword");

  const uint32_t base_mask = (uint32_t(1) << kVendoredResolveBaseTilesBits) - 1;
  base_truncated_out = (info.base_tiles & ~base_mask) != 0;

  return (info.pitch_tiles & ((uint32_t(1) << kVendoredResolvePitchTilesBits) - 1)) |
         (uint32_t(info.msaa_samples) << kMsaaShift) | (info.is_depth << kIsDepthShift) |
         ((info.base_tiles & base_mask) << kBaseShift) |
         ((info.format & ((uint32_t(1) << xenos::kRenderTargetFormatBits) - 1)) << kFormatShift) |
         (info.format_is_64bpp << kFormatIs64bppShift) |
         (info.fill_half_pixel_offset << kFillHalfPixelOffsetShift);
}

// [NR-DETILE] N-6-2, reworked in N-6-5. Copies the resolved EDRAM span to a
// readback heap and, one second later (so the copy has certainly retired
// without a fence and without stalling the frame), reports per tile row:
//   * flat  - every dword equal to the tile's first, i.e. cleared or never
//             written - against VARIED, i.e. rasterised, and
//   * = / * - whether the tile is BYTE-IDENTICAL to the previous sample.
//
// The second column is the one that named the defect. At the heavy city, rows
// 256..720 of both the tiled colour and the tiled depth surface read the same
// bytes in all 104 samples of a 104-second drive while rows 0..255 differ in
// every one: written once, never again. "Wrong pixels" and "pixels nobody
// writes any more" are different faults with different suspects, and only the
// change column tells them apart.
//
// N-6-5 also fixed two instrument defects the N-6-4 handoff named:
//   * ONE SLOT PER TAP. Slot 0 used to be shared by both band-1 colour taps,
//     so the once-a-second sampler reported whichever arrived first and the
//     two taps shared one rate limiter - a mixed population, which lies
//     ([[census-dedupe-hides-the-lever]]). Slots are now keyed by destination.
//   * The guest row labels assumed 4xMSAA (8 guest rows per tile row). The
//     msaa-0 control slot has 16, so its labels read half the true row. The
//     caller now passes the surface's own value.
//
// The readback buffers are deliberately never released: the probe is
// default-off, and a lifetime hook here would be more code than the probe.
constexpr uint32_t kTileDwords = xenos::kEdramTileWidthSamples * xenos::kEdramTileHeightSamples;

template <typename TransitionFn>
void NrDetileEdramProbe(D3D12CommandProcessor& command_processor, ID3D12Resource* edram_buffer,
                        uint64_t key, const char* name, uint32_t edram_base, uint32_t rows,
                        uint32_t pitch, uint32_t guest_rows_per_tile_row, uint32_t dest_base,
                        uint32_t copy_control, bool direct_resolved, TransitionFn&& transition) {
  struct Slot {
    ID3D12Resource* readback = nullptr;
    const char* name = "";
    uint32_t pending_rows = 0;
    uint32_t pending_pitch = 0;
    uint32_t pending_dest = 0;
    uint32_t pending_base = 0;
    uint32_t pending_guest_rows = 8;
    uint32_t pending_ctrl = 0;
    bool pending_direct = false;
    std::vector<uint64_t> prev_digest;
    std::chrono::steady_clock::time_point last{};
  };
  // A cap, so a shape nobody predicted cannot grow the table without bound.
  static std::map<uint64_t, Slot> s_slots;
  auto slot_it = s_slots.find(key);
  if (slot_it == s_slots.end()) {
    if (s_slots.size() >= 12) {
      return;
    }
    slot_it = s_slots.emplace(key, Slot()).first;
  }
  Slot& sl = slot_it->second;
  sl.name = name;

  const auto now = std::chrono::steady_clock::now();
  if (sl.last.time_since_epoch().count() && now - sl.last < std::chrono::seconds(1)) {
    return;
  }

  // Report the PREVIOUS capture before overwriting it. A second has passed, so
  // the copy that produced it has long since been submitted and retired.
  if (sl.readback && sl.pending_rows) {
    void* mapping = nullptr;
    if (SUCCEEDED(sl.readback->Map(0, nullptr, &mapping))) {
      const uint32_t* const dwords = static_cast<const uint32_t*>(mapping);
      std::string report;
      uint32_t sampled = 0;
      uint32_t frozen = 0;
      std::vector<uint64_t> digest;
      for (uint32_t tile_row = 0; tile_row < sl.pending_rows; tile_row += 4) {
        // Middle of the row, so a letterboxed or half-width surface still
        // lands on real content.
        const uint32_t tile = tile_row * sl.pending_pitch + sl.pending_pitch / 2;
        const uint32_t* const tile_dwords = dwords + size_t(tile) * kTileDwords;
        uint32_t differing = 0;
        uint64_t h = 1469598103934665603ull;
        for (uint32_t i = 0; i < kTileDwords; ++i) {
          differing += (i && tile_dwords[i] != tile_dwords[0]) ? 1 : 0;
          h = (h ^ tile_dwords[i]) * 1099511628211ull;
        }
        const size_t index = digest.size();
        const bool same = index < sl.prev_digest.size() && sl.prev_digest[index] == h;
        digest.push_back(h);
        ++sampled;
        frozen += same ? 1 : 0;
        report += fmt::format(" r{}{}{}{}({:08X})", tile_row * sl.pending_guest_rows,
                              sl.prev_digest.empty() ? "?" : (same ? "=" : "*"),
                              differing ? "VARIED" : "flat", differing, tile_dwords[0]);
      }
      D3D12_RANGE write_range = {};
      sl.readback->Unmap(0, &write_range);
      // A whole-span non-zero count is the honest answer to "did the copy
      // land at all": a broken readback is all zeros everywhere, a black but
      // real surface is zeros with a plausible structure.
      uint64_t nonzero = 0;
      const size_t span_dwords = size_t(sl.pending_rows) * sl.pending_pitch * kTileDwords;
      for (size_t i = 0; i < span_dwords; ++i) {
        nonzero += dwords[i] != 0 ? 1 : 0;
      }
      REXGPU_INFO(
          "[nr-detile-edram] {} dest={:08X} ctrl={:08X} direct={} base={}t span={}rows x {}t "
          "nonzero={}/{} | unchanged since last sample: {}/{} | guest row = ?|=|* then "
          "flat|VARIED differing(tile[0]):{}",
          sl.name, sl.pending_dest, sl.pending_ctrl, sl.pending_direct ? 1 : 0, sl.pending_base,
          sl.pending_rows, sl.pending_pitch, nonzero, span_dwords, frozen, sampled, report);
      sl.prev_digest = std::move(digest);
    }
    sl.pending_rows = 0;
  }

  sl.last = now;

  const uint64_t span_bytes = uint64_t(rows) * pitch * xenos::kEdramTileWidthSamples *
                              xenos::kEdramTileHeightSamples * sizeof(uint32_t);
  if (!sl.readback) {
    D3D12_RESOURCE_DESC desc;
    ui::d3d12::util::FillBufferResourceDesc(desc, xenos::kEdramSizeBytes,
                                            D3D12_RESOURCE_FLAG_NONE);
    const ui::d3d12::D3D12Provider& provider = command_processor.GetD3D12Provider();
    if (FAILED(provider.GetDevice()->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&sl.readback)))) {
      REXGPU_WARN("[nr-detile-edram] could not create the readback buffer for {}", name);
      return;
    }
  }
  if (span_bytes + uint64_t(edram_base) * kTileDwords * sizeof(uint32_t) >
      xenos::kEdramSizeBytes) {
    return;
  }

  transition(D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_processor.SubmitBarriers();
  command_processor.GetDeferredCommandList().D3DCopyBufferRegion(
      sl.readback, 0, edram_buffer, uint64_t(edram_base) * kTileDwords * sizeof(uint32_t),
      span_bytes);
  sl.pending_rows = rows;
  sl.pending_pitch = pitch;
  sl.pending_dest = dest_base;
  sl.pending_base = edram_base;
  sl.pending_guest_rows = guest_rows_per_tile_row;
  sl.pending_ctrl = copy_control;
  sl.pending_direct = direct_resolved;
}

// Picks the patched blob when the wrap constant was found, the original
// otherwise, and keeps the storage alive for the CreateComputePipeline call.
struct ResolveShaderBytecode {
  ResolveShaderBytecode(const void* source, size_t source_size, const char* debug_name)
      : data(source), size(source_size) {
    if (kHostEdramWrapDwords != kConsoleEdramWrapDwords &&
        PatchResolveShaderEdramWrap(source, source_size, debug_name, storage, &by_shift)) {
      data = storage.data();
      size = storage.size();
      patched = true;
    }
  }
  std::vector<uint8_t> storage;
  const void* data;
  size_t size;
  bool patched = false;
  // True when the wrap was a shift amount rather than a literal, i.e. this is
  // a resolution-scaled variant.
  bool by_shift = false;
};

}  // namespace

const D3D12RenderTargetCache::ResolveCopyShaderCode
    D3D12RenderTargetCache::kResolveCopyShaders[size_t(
        draw_util::ResolveCopyShaderIndex::kCount)] = {
        {shaders::resolve_fast_32bpp_1x2xmsaa_cs, sizeof(shaders::resolve_fast_32bpp_1x2xmsaa_cs),
         shaders::resolve_fast_32bpp_1x2xmsaa_scaled_cs,
         sizeof(shaders::resolve_fast_32bpp_1x2xmsaa_scaled_cs)},
        {shaders::resolve_fast_32bpp_4xmsaa_cs, sizeof(shaders::resolve_fast_32bpp_4xmsaa_cs),
         shaders::resolve_fast_32bpp_4xmsaa_scaled_cs,
         sizeof(shaders::resolve_fast_32bpp_4xmsaa_scaled_cs)},
        {shaders::resolve_fast_64bpp_1x2xmsaa_cs, sizeof(shaders::resolve_fast_64bpp_1x2xmsaa_cs),
         shaders::resolve_fast_64bpp_1x2xmsaa_scaled_cs,
         sizeof(shaders::resolve_fast_64bpp_1x2xmsaa_scaled_cs)},
        {shaders::resolve_fast_64bpp_4xmsaa_cs, sizeof(shaders::resolve_fast_64bpp_4xmsaa_cs),
         shaders::resolve_fast_64bpp_4xmsaa_scaled_cs,
         sizeof(shaders::resolve_fast_64bpp_4xmsaa_scaled_cs)},
        {shaders::resolve_full_8bpp_cs, sizeof(shaders::resolve_full_8bpp_cs),
         shaders::resolve_full_8bpp_scaled_cs, sizeof(shaders::resolve_full_8bpp_scaled_cs)},
        {shaders::resolve_full_16bpp_cs, sizeof(shaders::resolve_full_16bpp_cs),
         shaders::resolve_full_16bpp_scaled_cs, sizeof(shaders::resolve_full_16bpp_scaled_cs)},
        {shaders::resolve_full_32bpp_cs, sizeof(shaders::resolve_full_32bpp_cs),
         shaders::resolve_full_32bpp_scaled_cs, sizeof(shaders::resolve_full_32bpp_scaled_cs)},
        {shaders::resolve_full_64bpp_cs, sizeof(shaders::resolve_full_64bpp_cs),
         shaders::resolve_full_64bpp_scaled_cs, sizeof(shaders::resolve_full_64bpp_scaled_cs)},
        {shaders::resolve_full_128bpp_cs, sizeof(shaders::resolve_full_128bpp_cs),
         shaders::resolve_full_128bpp_scaled_cs, sizeof(shaders::resolve_full_128bpp_scaled_cs)},
};

const uint32_t D3D12RenderTargetCache::kTransferUsedRootParameters[size_t(
    TransferRootSignatureIndex::kCount)] = {
    // kColor
    kTransferUsedRootParameterColorSRVBit | kTransferUsedRootParameterAddressConstantBit,
    // kDepth
    kTransferUsedRootParameterDepthSRVBit | kTransferUsedRootParameterAddressConstantBit,
    // kDepthStencil
    kTransferUsedRootParameterDepthSRVBit | kTransferUsedRootParameterStencilSRVBit |
        kTransferUsedRootParameterAddressConstantBit,
    // kColorToStencilBit
    kTransferUsedRootParameterStencilMaskConstantBit | kTransferUsedRootParameterColorSRVBit |
        kTransferUsedRootParameterAddressConstantBit,
    // kStencilToStencilBit
    kTransferUsedRootParameterStencilMaskConstantBit | kTransferUsedRootParameterStencilSRVBit |
        kTransferUsedRootParameterAddressConstantBit,
    // kColorAndHostDepth
    kTransferUsedRootParameterColorSRVBit | kTransferUsedRootParameterAddressConstantBit |
        kTransferUsedRootParameterHostDepthSRVBit |
        kTransferUsedRootParameterHostDepthAddressConstantBit,
    // kDepthAndHostDepth
    kTransferUsedRootParameterDepthSRVBit | kTransferUsedRootParameterAddressConstantBit |
        kTransferUsedRootParameterHostDepthSRVBit |
        kTransferUsedRootParameterHostDepthAddressConstantBit,
    // kDepthStencilAndHostDepth
    kTransferUsedRootParameterDepthSRVBit | kTransferUsedRootParameterStencilSRVBit |
        kTransferUsedRootParameterAddressConstantBit | kTransferUsedRootParameterHostDepthSRVBit |
        kTransferUsedRootParameterHostDepthAddressConstantBit,
};

const D3D12RenderTargetCache::TransferModeInfo
    D3D12RenderTargetCache::kTransferModes[size_t(TransferMode::kCount)] = {
        // kColorToDepth
        {TransferOutput::kDepth, TransferRootSignatureIndex::kColor,
         TransferRootSignatureIndex::kColor},
        // kColorToColor
        {TransferOutput::kColor, TransferRootSignatureIndex::kColor,
         TransferRootSignatureIndex::kColor},
        // kDepthToDepth
        {TransferOutput::kDepth, TransferRootSignatureIndex::kDepth,
         TransferRootSignatureIndex::kDepthStencil},
        // kDepthToColor
        {TransferOutput::kColor, TransferRootSignatureIndex::kDepthStencil,
         TransferRootSignatureIndex::kDepthStencil},
        // kColorToStencilBit
        {TransferOutput::kStencilBit, TransferRootSignatureIndex::kColorToStencilBit,
         TransferRootSignatureIndex::kColorToStencilBit},
        // kDepthToStencilBit
        {TransferOutput::kStencilBit, TransferRootSignatureIndex::kStencilToStencilBit,
         TransferRootSignatureIndex::kStencilToStencilBit},
        // kColorAndHostDepthToDepth
        {TransferOutput::kDepth, TransferRootSignatureIndex::kColorAndHostDepth,
         TransferRootSignatureIndex::kColorAndHostDepth},
        // kDepthAndHostDepthToDepth
        {TransferOutput::kDepth, TransferRootSignatureIndex::kDepthAndHostDepth,
         TransferRootSignatureIndex::kDepthStencilAndHostDepth},
};

D3D12RenderTargetCache::~D3D12RenderTargetCache() {
  Shutdown(true);
}

bool D3D12RenderTargetCache::Initialize() {
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();

  // [N-10b deletion c] Host render targets, unconditionally. The ROV /
  // pixel-shader-interlock path is DELETED on D3D12 (the old Intel ROV
  // forcing dated to a 2021 driver stencil bug; naruto_626 proved RTV on
  // UHD 630, and the NR path requires host RTs). Vulkan keeps its own
  // path selection until the Vulkan pass.
  path_ = Path::kHostRenderTargets;

  // Create the buffer for reinterpreting EDRAM contents.
  uint32_t edram_buffer_size =
      xenos::kEdramSizeBytes * (draw_resolution_scale_x() * draw_resolution_scale_y());
  D3D12_RESOURCE_DESC edram_buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(edram_buffer_desc, edram_buffer_size,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  // The first operation will likely be depth self-comparison with host render
  // targets or drawing with ROV.
  edram_buffer_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  // Creating zeroed for stable initial value with ROV (though on a real
  // console it has to be cleared anyway probably) and not to leak irrelevant
  // data to trace dumps when not covered by host render targets entirely.
  if (FAILED(device->CreateCommittedResource(
          &ui::d3d12::util::kHeapPropertiesDefault, D3D12_HEAP_FLAG_NONE, &edram_buffer_desc,
          edram_buffer_state_, nullptr, IID_PPV_ARGS(&edram_buffer_)))) {
    REXGPU_ERROR("D3D12RenderTargetCache: Failed to create the EDRAM buffer");
    Shutdown();
    return false;
  }
  edram_buffer_->SetName(L"EDRAM Buffer");
  edram_buffer_modification_status_ = EdramBufferModificationStatus::kUnmodified;

  // Create non-shader-visible descriptors of the EDRAM buffer for copying.
  D3D12_DESCRIPTOR_HEAP_DESC edram_buffer_descriptor_heap_desc;
  edram_buffer_descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  edram_buffer_descriptor_heap_desc.NumDescriptors = uint32_t(EdramBufferDescriptorIndex::kCount);
  edram_buffer_descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  edram_buffer_descriptor_heap_desc.NodeMask = 0;
  if (FAILED(device->CreateDescriptorHeap(&edram_buffer_descriptor_heap_desc,
                                          IID_PPV_ARGS(&edram_buffer_descriptor_heap_)))) {
    REXGPU_ERROR(
        "D3D12RenderTargetCache: Failed to create the descriptor heap for "
        "EDRAM buffer views");
    Shutdown();
    return false;
  }
  edram_buffer_descriptor_heap_start_ =
      edram_buffer_descriptor_heap_->GetCPUDescriptorHandleForHeapStart();
  ui::d3d12::util::CreateBufferRawSRV(
      device,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kRawSRV)),
      edram_buffer_, edram_buffer_size);
  ui::d3d12::util::CreateBufferTypedSRV(
      device,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kR32UintSRV)),
      edram_buffer_, DXGI_FORMAT_R32_UINT, edram_buffer_size >> 2);
  ui::d3d12::util::CreateBufferTypedSRV(
      device,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kR32G32UintSRV)),
      edram_buffer_, DXGI_FORMAT_R32G32_UINT, edram_buffer_size >> 3);
  ui::d3d12::util::CreateBufferTypedSRV(
      device,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kR32G32B32A32UintSRV)),
      edram_buffer_, DXGI_FORMAT_R32G32B32A32_UINT, edram_buffer_size >> 4);
  ui::d3d12::util::CreateBufferRawUAV(
      device,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kRawUAV)),
      edram_buffer_, edram_buffer_size);
  ui::d3d12::util::CreateBufferTypedUAV(
      device,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kR32UintUAV)),
      edram_buffer_, DXGI_FORMAT_R32_UINT, edram_buffer_size >> 2);
  ui::d3d12::util::CreateBufferTypedUAV(
      device,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kR32G32UintUAV)),
      edram_buffer_, DXGI_FORMAT_R32G32_UINT, edram_buffer_size >> 3);
  ui::d3d12::util::CreateBufferTypedUAV(
      device,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kR32G32B32A32UintUAV)),
      edram_buffer_, DXGI_FORMAT_R32G32B32A32_UINT, edram_buffer_size >> 4);

  bool draw_resolution_scaled = IsDrawResolutionScaled();

  // Create the resolve copying root signature.
  std::array<D3D12_ROOT_PARAMETER, 3> resolve_copy_root_parameters;
  // Parameter 0 is constants.
  resolve_copy_root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  resolve_copy_root_parameters[0].Constants.ShaderRegister = 0;
  resolve_copy_root_parameters[0].Constants.RegisterSpace = 0;
  // Binding all of the shared memory at 1x resolution, portions with scaled
  // resolution.
  resolve_copy_root_parameters[0].Constants.Num32BitValues =
      (draw_resolution_scaled ? sizeof(draw_util::ResolveCopyShaderConstants::DestRelative)
                              : sizeof(draw_util::ResolveCopyShaderConstants)) /
      sizeof(uint32_t);
  resolve_copy_root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  // Parameter 1 is the destination (shared memory).
  D3D12_DESCRIPTOR_RANGE resolve_copy_dest_range;
  resolve_copy_dest_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  resolve_copy_dest_range.NumDescriptors = 1;
  resolve_copy_dest_range.BaseShaderRegister = 0;
  resolve_copy_dest_range.RegisterSpace = 0;
  resolve_copy_dest_range.OffsetInDescriptorsFromTableStart = 0;
  resolve_copy_root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  resolve_copy_root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
  resolve_copy_root_parameters[1].DescriptorTable.pDescriptorRanges = &resolve_copy_dest_range;
  resolve_copy_root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  // Parameter 2 is the source (EDRAM).
  D3D12_DESCRIPTOR_RANGE resolve_copy_source_range;
  resolve_copy_source_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  resolve_copy_source_range.NumDescriptors = 1;
  resolve_copy_source_range.BaseShaderRegister = 0;
  resolve_copy_source_range.RegisterSpace = 0;
  resolve_copy_source_range.OffsetInDescriptorsFromTableStart = 0;
  resolve_copy_root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  resolve_copy_root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
  resolve_copy_root_parameters[2].DescriptorTable.pDescriptorRanges = &resolve_copy_source_range;
  resolve_copy_root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC resolve_copy_root_signature_desc;
  resolve_copy_root_signature_desc.NumParameters = UINT(resolve_copy_root_parameters.size());
  resolve_copy_root_signature_desc.pParameters = resolve_copy_root_parameters.data();
  resolve_copy_root_signature_desc.NumStaticSamplers = 0;
  resolve_copy_root_signature_desc.pStaticSamplers = nullptr;
  resolve_copy_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  resolve_copy_root_signature_ =
      ui::d3d12::util::CreateRootSignature(provider, resolve_copy_root_signature_desc);
  if (resolve_copy_root_signature_ == nullptr) {
    REXGPU_ERROR(
        "D3D12RenderTargetCache: Failed to create the resolve copy root "
        "signature");
    Shutdown();
    return false;
  }
  // [NR-DRES] Direct resolve root signatures. Color: 6 root constants (b0) +
  // dest UAV table (u0, shared memory R32_UINT) + source SRV table (t0, the
  // host RT texture). Depth adds the stencil plane SRV table (t1).
  {
    D3D12_ROOT_PARAMETER dres_root_parameters[4];
    dres_root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    dres_root_parameters[0].Constants.ShaderRegister = 0;
    dres_root_parameters[0].Constants.RegisterSpace = 0;
    dres_root_parameters[0].Constants.Num32BitValues = 8;
    dres_root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_DESCRIPTOR_RANGE dres_dest_range;
    dres_dest_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    dres_dest_range.NumDescriptors = 1;
    dres_dest_range.BaseShaderRegister = 0;
    dres_dest_range.RegisterSpace = 0;
    dres_dest_range.OffsetInDescriptorsFromTableStart = 0;
    dres_root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    dres_root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    dres_root_parameters[1].DescriptorTable.pDescriptorRanges = &dres_dest_range;
    dres_root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_DESCRIPTOR_RANGE dres_source_range;
    dres_source_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    dres_source_range.NumDescriptors = 1;
    dres_source_range.BaseShaderRegister = 0;
    dres_source_range.RegisterSpace = 0;
    dres_source_range.OffsetInDescriptorsFromTableStart = 0;
    dres_root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    dres_root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
    dres_root_parameters[2].DescriptorTable.pDescriptorRanges = &dres_source_range;
    dres_root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_DESCRIPTOR_RANGE dres_stencil_range;
    dres_stencil_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    dres_stencil_range.NumDescriptors = 1;
    dres_stencil_range.BaseShaderRegister = 1;
    dres_stencil_range.RegisterSpace = 0;
    dres_stencil_range.OffsetInDescriptorsFromTableStart = 0;
    dres_root_parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    dres_root_parameters[3].DescriptorTable.NumDescriptorRanges = 1;
    dres_root_parameters[3].DescriptorTable.pDescriptorRanges = &dres_stencil_range;
    dres_root_parameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC dres_root_signature_desc;
    dres_root_signature_desc.NumParameters = 3;
    dres_root_signature_desc.pParameters = dres_root_parameters;
    dres_root_signature_desc.NumStaticSamplers = 0;
    dres_root_signature_desc.pStaticSamplers = nullptr;
    dres_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    direct_resolve_root_signature_color_ =
        ui::d3d12::util::CreateRootSignature(provider, dres_root_signature_desc);
    dres_root_signature_desc.NumParameters = 4;
    direct_resolve_root_signature_depth_ =
        ui::d3d12::util::CreateRootSignature(provider, dres_root_signature_desc);
    if (!direct_resolve_root_signature_color_ || !direct_resolve_root_signature_depth_) {
      // Not fatal: every direct resolve declines and the dump path serves.
      REXGPU_WARN(
          "[nr-dres] Failed to create the direct resolve root signatures; all "
          "resolves will take the dump path");
    }
  }

  // Create the resolve copying pipelines.
  uint32_t resolve_shaders_edram_wrap_patched = 0;
  uint32_t resolve_shaders_edram_wrap_unpatched = 0;
  // Broken out because the two forms fail independently: the literal one is
  // what the unscaled blobs carry, the shift one what the resolution-scaled
  // blobs build at runtime. Reporting only the total let "0 patched, 9 left"
  // at resolution_scale 2 look like the same fix simply not applying.
  uint32_t resolve_shaders_edram_wrap_by_shift = 0;
  for (size_t i = 0; i < size_t(draw_util::ResolveCopyShaderIndex::kCount); ++i) {
    const draw_util::ResolveCopyShaderInfo& resolve_copy_shader_info =
        draw_util::resolve_copy_shader_info[i];
    const ResolveCopyShaderCode& resolve_copy_shader_code = kResolveCopyShaders[i];
    // Somewhat verification whether resolve_copy_shaders_ is up to date.
    assert_true(resolve_copy_shader_code.unscaled && resolve_copy_shader_code.unscaled_size &&
                resolve_copy_shader_code.scaled && resolve_copy_shader_code.scaled_size);
    ResolveShaderBytecode resolve_copy_bytecode(
        draw_resolution_scaled ? resolve_copy_shader_code.scaled
                               : resolve_copy_shader_code.unscaled,
        draw_resolution_scaled ? resolve_copy_shader_code.scaled_size
                               : resolve_copy_shader_code.unscaled_size,
        resolve_copy_shader_info.debug_name);
    resolve_shaders_edram_wrap_patched += resolve_copy_bytecode.patched ? 1 : 0;
    resolve_shaders_edram_wrap_unpatched += resolve_copy_bytecode.patched ? 0 : 1;
    resolve_shaders_edram_wrap_by_shift += resolve_copy_bytecode.by_shift ? 1 : 0;
    ID3D12PipelineState* resolve_copy_pipeline = ui::d3d12::util::CreateComputePipeline(
        device, resolve_copy_bytecode.data, resolve_copy_bytecode.size,
        resolve_copy_root_signature_);
    if (resolve_copy_pipeline == nullptr) {
      REXGPU_ERROR("D3D12RenderTargetCache: Failed to create {} resolve copy pipeline",
                   resolve_copy_shader_info.debug_name);
      Shutdown();
      return false;
    }
    std::u16string resolve_copy_pipeline_name =
        rex::string::to_utf16(resolve_copy_shader_info.debug_name);
    resolve_copy_pipeline->SetName(reinterpret_cast<LPCWSTR>(resolve_copy_pipeline_name.c_str()));
    resolve_copy_pipelines_[i] = resolve_copy_pipeline;
  }

  // Using the cvar on emulator initialization so used pipelines are consistent
  // across different titles launched in one emulator instance.
  use_stencil_reference_output_ =
      REXCVAR_GET(native_stencil_value_output) &&
      provider.IsPSSpecifiedStencilReferenceSupported() &&
      (REXCVAR_GET(native_stencil_value_output_d3d12_intel) ||
       provider.GetAdapterVendorID() != ui::GraphicsProvider::GpuVendorID::kIntel);

  if (path_ == Path::kHostRenderTargets) {
    // Host render targets.

    gamma_render_target_as_unorm16_ = REXCVAR_GET(gamma_render_target_as_unorm16);

    depth_float24_round_ = REXCVAR_GET(depth_float24_round);
    depth_float24_convert_in_pixel_shader_ = REXCVAR_GET(depth_float24_convert_in_pixel_shader);

    // Check if 2x MSAA is supported or needs to be emulated with 4x MSAA
    // instead.
    if (REXCVAR_GET(native_2x_msaa)) {
      msaa_2x_supported_ = true;
      static const DXGI_FORMAT kRenderTargetDXGIFormats[] = {
          DXGI_FORMAT_R16G16B16A16_FLOAT,
          DXGI_FORMAT_R16G16B16A16_SNORM,
          DXGI_FORMAT_R32G32_FLOAT,
          DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
          DXGI_FORMAT_R10G10B10A2_UNORM,
          DXGI_FORMAT_R8G8B8A8_UNORM,
          DXGI_FORMAT_R16G16_FLOAT,
          DXGI_FORMAT_R16G16_SNORM,
          DXGI_FORMAT_R32_FLOAT,
          DXGI_FORMAT_D24_UNORM_S8_UINT,
          // For ownership transfer.
          DXGI_FORMAT_R16G16B16A16_UINT,
          DXGI_FORMAT_R32G32_UINT,
          DXGI_FORMAT_R16G16_UINT,
          DXGI_FORMAT_R32_UINT,
      };
      D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS multisample_quality_levels;
      multisample_quality_levels.SampleCount = 2;
      multisample_quality_levels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
      for (size_t i = 0; i < rex::countof(kRenderTargetDXGIFormats); ++i) {
        multisample_quality_levels.Format = kRenderTargetDXGIFormats[i];
        multisample_quality_levels.NumQualityLevels = 0;
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                               &multisample_quality_levels,
                                               sizeof(multisample_quality_levels))) ||
            !multisample_quality_levels.NumQualityLevels) {
          msaa_2x_supported_ = false;
          break;
        }
      }
    } else {
      msaa_2x_supported_ = false;
    }
    if (msaa_2x_supported_ && gamma_render_target_as_unorm16_) {
      D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS multisample_quality_levels;
      multisample_quality_levels.SampleCount = 2;
      multisample_quality_levels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
      multisample_quality_levels.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
      multisample_quality_levels.NumQualityLevels = 0;
      if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                             &multisample_quality_levels,
                                             sizeof(multisample_quality_levels))) ||
          !multisample_quality_levels.NumQualityLevels) {
        msaa_2x_supported_ = false;
      }
    }
    if (!msaa_2x_supported_) {
      REXGPU_WARN(
          "2x MSAA is not supported, emulated via top-left and bottom-right "
          "samples of 4x MSAA");
    }

    descriptor_pool_color_ = std::make_unique<ui::d3d12::D3D12CpuDescriptorPool>(
        provider, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 11);
    descriptor_pool_depth_ = std::make_unique<ui::d3d12::D3D12CpuDescriptorPool>(
        provider, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 11);
    descriptor_pool_srv_ = std::make_unique<ui::d3d12::D3D12CpuDescriptorPool>(
        provider, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 11);

    // Create null render target descriptors for gaps, must be fully typed
    // (though in pipeline states, DXGI_FORMAT_UNKNOWN must be used instead -
    // this would also cause a mismatching format error in the debug layer, but
    // it's a bug in the debug layer itself - needs to be suppressed, and
    // already fixed in some version of Windows).
    null_rtv_descriptor_ss_ = descriptor_pool_color_->AllocateDescriptor();
    null_rtv_descriptor_ms_ = descriptor_pool_color_->AllocateDescriptor();
    if (!null_rtv_descriptor_ss_ || !null_rtv_descriptor_ms_) {
      Shutdown();
      return false;
    }
    D3D12_RENDER_TARGET_VIEW_DESC null_rtv_desc;
    // The format doesn't matter, but it must be bindable as a render target,
    // not DXGI_FORMAT_UNKNOWN.
    null_rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    null_rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    null_rtv_desc.Texture2D.MipSlice = 0;
    null_rtv_desc.Texture2D.PlaneSlice = 0;
    device->CreateRenderTargetView(nullptr, &null_rtv_desc, null_rtv_descriptor_ss_.GetHandle());
    null_rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
    device->CreateRenderTargetView(nullptr, &null_rtv_desc, null_rtv_descriptor_ms_.GetHandle());

    // [NR-XFER] N-10b: the host depth storing root signature and the three
    // host_depth_store compute pipelines are DELETED - the native scratch
    // texture snapshot replaced the EDRAM round trip.

    // Transfer and clear vertex buffer, for quads of up to tile granularity.
    transfer_vertex_buffer_pool_ = std::make_unique<ui::d3d12::D3D12UploadBufferPool>(
        provider, std::max(ui::d3d12::D3D12UploadBufferPool::kDefaultPageSize,
                           sizeof(float) * 2 * 6 * Transfer::kMaxCutoutBorderRectangles *
                               xenos::kEdramTileCount));

    // Transfer root signatures.
    D3D12_DESCRIPTOR_RANGE transfer_root_color_srv_range;
    transfer_root_color_srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    transfer_root_color_srv_range.NumDescriptors = 1;
    transfer_root_color_srv_range.BaseShaderRegister = kTransferSRVRegisterColor;
    transfer_root_color_srv_range.RegisterSpace = 0;
    transfer_root_color_srv_range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_DESCRIPTOR_RANGE transfer_root_depth_srv_range;
    transfer_root_depth_srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    transfer_root_depth_srv_range.NumDescriptors = 1;
    transfer_root_depth_srv_range.BaseShaderRegister = kTransferSRVRegisterDepth;
    transfer_root_depth_srv_range.RegisterSpace = 0;
    transfer_root_depth_srv_range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_DESCRIPTOR_RANGE transfer_root_stencil_srv_range;
    transfer_root_stencil_srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    transfer_root_stencil_srv_range.NumDescriptors = 1;
    transfer_root_stencil_srv_range.BaseShaderRegister = kTransferSRVRegisterStencil;
    transfer_root_stencil_srv_range.RegisterSpace = 0;
    transfer_root_stencil_srv_range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_DESCRIPTOR_RANGE transfer_root_host_depth_srv_range;
    transfer_root_host_depth_srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    transfer_root_host_depth_srv_range.NumDescriptors = 1;
    transfer_root_host_depth_srv_range.BaseShaderRegister = kTransferSRVRegisterHostDepth;
    transfer_root_host_depth_srv_range.RegisterSpace = 0;
    transfer_root_host_depth_srv_range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_ROOT_PARAMETER
    transfer_root_parameters[kTransferUsedRootParameterCount];
    D3D12_ROOT_SIGNATURE_DESC transfer_root_desc;
    transfer_root_desc.pParameters = transfer_root_parameters;
    transfer_root_desc.NumStaticSamplers = 0;
    transfer_root_desc.pStaticSamplers = nullptr;
    transfer_root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    for (size_t i = 0; i < size_t(TransferRootSignatureIndex::kCount); ++i) {
      uint32_t transfer_root_mask = kTransferUsedRootParameters[i];
      // Stencil mask constant.
      if (transfer_root_mask & kTransferUsedRootParameterStencilMaskConstantBit) {
        D3D12_ROOT_PARAMETER& transfer_root_stencil_mask_constant =
            transfer_root_parameters[rex::bit_count(
                transfer_root_mask & (kTransferUsedRootParameterStencilMaskConstantBit - 1))];
        transfer_root_stencil_mask_constant.ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        transfer_root_stencil_mask_constant.Constants.ShaderRegister =
            kTransferCBVRegisterStencilMask;
        transfer_root_stencil_mask_constant.Constants.RegisterSpace = 0;
        transfer_root_stencil_mask_constant.Constants.Num32BitValues = 1;
        transfer_root_stencil_mask_constant.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      }
      // Color SRV.
      if (transfer_root_mask & kTransferUsedRootParameterColorSRVBit) {
        D3D12_ROOT_PARAMETER& transfer_root_color_srv = transfer_root_parameters[rex::bit_count(
            transfer_root_mask & (kTransferUsedRootParameterColorSRVBit - 1))];
        transfer_root_color_srv.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        transfer_root_color_srv.DescriptorTable.NumDescriptorRanges = 1;
        transfer_root_color_srv.DescriptorTable.pDescriptorRanges = &transfer_root_color_srv_range;
        transfer_root_color_srv.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      }
      // Depth SRV.
      if (transfer_root_mask & kTransferUsedRootParameterDepthSRVBit) {
        D3D12_ROOT_PARAMETER& transfer_root_depth_srv = transfer_root_parameters[rex::bit_count(
            transfer_root_mask & (kTransferUsedRootParameterDepthSRVBit - 1))];
        transfer_root_depth_srv.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        transfer_root_depth_srv.DescriptorTable.NumDescriptorRanges = 1;
        transfer_root_depth_srv.DescriptorTable.pDescriptorRanges = &transfer_root_depth_srv_range;
        transfer_root_depth_srv.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      }
      // Stencil SRV.
      if (transfer_root_mask & kTransferUsedRootParameterStencilSRVBit) {
        D3D12_ROOT_PARAMETER& transfer_root_stencil_srv = transfer_root_parameters[rex::bit_count(
            transfer_root_mask & (kTransferUsedRootParameterStencilSRVBit - 1))];
        transfer_root_stencil_srv.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        transfer_root_stencil_srv.DescriptorTable.NumDescriptorRanges = 1;
        transfer_root_stencil_srv.DescriptorTable.pDescriptorRanges =
            &transfer_root_stencil_srv_range;
        transfer_root_stencil_srv.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      }
      // Address constant.
      if (transfer_root_mask & kTransferUsedRootParameterAddressConstantBit) {
        D3D12_ROOT_PARAMETER& transfer_root_address_constant =
            transfer_root_parameters[rex::bit_count(
                transfer_root_mask & (kTransferUsedRootParameterAddressConstantBit - 1))];
        transfer_root_address_constant.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        transfer_root_address_constant.Constants.ShaderRegister = kTransferCBVRegisterAddress;
        transfer_root_address_constant.Constants.RegisterSpace = 0;
        transfer_root_address_constant.Constants.Num32BitValues =
            sizeof(TransferAddressConstant) / sizeof(uint32_t);
        transfer_root_address_constant.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      }
      // Host depth SRV.
      if (transfer_root_mask & kTransferUsedRootParameterHostDepthSRVBit) {
        D3D12_ROOT_PARAMETER& transfer_root_host_depth_srv =
            transfer_root_parameters[rex::bit_count(
                transfer_root_mask & (kTransferUsedRootParameterHostDepthSRVBit - 1))];
        transfer_root_host_depth_srv.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        transfer_root_host_depth_srv.DescriptorTable.NumDescriptorRanges = 1;
        transfer_root_host_depth_srv.DescriptorTable.pDescriptorRanges =
            &transfer_root_host_depth_srv_range;
        transfer_root_host_depth_srv.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      }
      // Host depth address constant.
      if (transfer_root_mask & kTransferUsedRootParameterHostDepthAddressConstantBit) {
        D3D12_ROOT_PARAMETER& transfer_root_host_address_constant =
            transfer_root_parameters[rex::bit_count(
                transfer_root_mask & (kTransferUsedRootParameterHostDepthAddressConstantBit - 1))];
        transfer_root_host_address_constant.ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        transfer_root_host_address_constant.Constants.ShaderRegister =
            kTransferCBVRegisterHostDepthAddress;
        transfer_root_host_address_constant.Constants.RegisterSpace = 0;
        transfer_root_host_address_constant.Constants.Num32BitValues =
            sizeof(TransferAddressConstant) / sizeof(uint32_t);
        transfer_root_host_address_constant.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      }
      transfer_root_desc.NumParameters = rex::bit_count(transfer_root_mask);
      assert_true(transfer_root_desc.NumParameters <= kTransferUsedRootParameterCount);
      transfer_root_signatures_[i] =
          ui::d3d12::util::CreateRootSignature(provider, transfer_root_desc);
      if (!transfer_root_signatures_[i]) {
        REXGPU_ERROR(
            "D3D12RenderTargetCache: Failed to create the render target "
            "ownership transfer root signature {:X}",
            transfer_root_mask);
        Shutdown();
        return false;
      }
    }

    // Dumping root signatures.
    D3D12_DESCRIPTOR_RANGE dump_root_source_range;
    dump_root_source_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    dump_root_source_range.NumDescriptors = 1;
    dump_root_source_range.BaseShaderRegister = 0;
    dump_root_source_range.RegisterSpace = 0;
    dump_root_source_range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_DESCRIPTOR_RANGE dump_root_stencil_range;
    dump_root_stencil_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    dump_root_stencil_range.NumDescriptors = 1;
    dump_root_stencil_range.BaseShaderRegister = 1;
    dump_root_stencil_range.RegisterSpace = 0;
    dump_root_stencil_range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_DESCRIPTOR_RANGE dump_root_edram_range;
    dump_root_edram_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    dump_root_edram_range.NumDescriptors = 1;
    dump_root_edram_range.BaseShaderRegister = 0;
    dump_root_edram_range.RegisterSpace = 0;
    dump_root_edram_range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_ROOT_PARAMETER
    dump_root_color_parameters[kDumpRootParameterColorCount];
    D3D12_ROOT_PARAMETER
    dump_root_depth_parameters[kDumpRootParameterDepthCount];
    for (uint32_t i = 0; i < 2; ++i) {
      // Offsets.
      D3D12_ROOT_PARAMETER& dump_root_offsets =
          i ? dump_root_depth_parameters[kDumpRootParameterOffsets]
            : dump_root_color_parameters[kDumpRootParameterOffsets];
      dump_root_offsets.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      dump_root_offsets.Constants.ShaderRegister = kDumpCbufferOffsets;
      dump_root_offsets.Constants.RegisterSpace = 0;
      dump_root_offsets.Constants.Num32BitValues = sizeof(DumpOffsets) / sizeof(uint32_t);
      dump_root_offsets.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      // Source.
      D3D12_ROOT_PARAMETER& dump_root_source =
          i ? dump_root_depth_parameters[kDumpRootParameterSource]
            : dump_root_color_parameters[kDumpRootParameterSource];
      dump_root_source.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      dump_root_source.DescriptorTable.NumDescriptorRanges = 1;
      dump_root_source.DescriptorTable.pDescriptorRanges = &dump_root_source_range;
      dump_root_source.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      // Stencil.
      if (i) {
        D3D12_ROOT_PARAMETER& dump_root_stencil =
            dump_root_depth_parameters[kDumpRootParameterDepthStencil];
        dump_root_stencil.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        dump_root_stencil.DescriptorTable.NumDescriptorRanges = 1;
        dump_root_stencil.DescriptorTable.pDescriptorRanges = &dump_root_stencil_range;
        dump_root_stencil.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      }
      // Pitches.
      D3D12_ROOT_PARAMETER& dump_root_pitches =
          i ? dump_root_depth_parameters[kDumpRootParameterDepthPitches]
            : dump_root_color_parameters[kDumpRootParameterColorPitches];
      dump_root_pitches.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      dump_root_pitches.Constants.ShaderRegister = kDumpCbufferPitches;
      dump_root_pitches.Constants.RegisterSpace = 0;
      dump_root_pitches.Constants.Num32BitValues = sizeof(DumpPitches) / sizeof(uint32_t);
      dump_root_pitches.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      // EDRAM.
      D3D12_ROOT_PARAMETER& dump_root_edram =
          i ? dump_root_depth_parameters[kDumpRootParameterDepthEdram]
            : dump_root_color_parameters[kDumpRootParameterColorEdram];
      dump_root_edram.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      dump_root_edram.DescriptorTable.NumDescriptorRanges = 1;
      dump_root_edram.DescriptorTable.pDescriptorRanges = &dump_root_edram_range;
      dump_root_edram.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC dump_root_desc;
    dump_root_desc.NumParameters = UINT(rex::countof(dump_root_color_parameters));
    dump_root_desc.pParameters = dump_root_color_parameters;
    dump_root_desc.NumStaticSamplers = 0;
    dump_root_desc.pStaticSamplers = nullptr;
    dump_root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    dump_root_signature_color_ = ui::d3d12::util::CreateRootSignature(provider, dump_root_desc);
    if (!dump_root_signature_color_) {
      REXGPU_ERROR(
          "D3D12RenderTargetCache: Failed to create the color render target "
          "dumping root signature");
      Shutdown();
      return false;
    }
    dump_root_desc.NumParameters = UINT(rex::countof(dump_root_depth_parameters));
    dump_root_desc.pParameters = dump_root_depth_parameters;
    dump_root_signature_depth_ = ui::d3d12::util::CreateRootSignature(provider, dump_root_desc);
    if (!dump_root_signature_depth_) {
      REXGPU_ERROR(
          "D3D12RenderTargetCache: Failed to create the depth render target "
          "dumping root signature");
      Shutdown();
      return false;
    }

    // k_32_FLOAT and k_32_32_FLOAT clear root signature and pipelines.
    D3D12_ROOT_PARAMETER uint32_rtv_clear_root_constants;
    uint32_rtv_clear_root_constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    uint32_rtv_clear_root_constants.Constants.ShaderRegister = 0;
    uint32_rtv_clear_root_constants.Constants.RegisterSpace = 0;
    uint32_rtv_clear_root_constants.Constants.Num32BitValues = 2;
    uint32_rtv_clear_root_constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC uint32_rtv_clear_root_desc;
    uint32_rtv_clear_root_desc.NumParameters = 1;
    uint32_rtv_clear_root_desc.pParameters = &uint32_rtv_clear_root_constants;
    uint32_rtv_clear_root_desc.NumStaticSamplers = 0;
    uint32_rtv_clear_root_desc.pStaticSamplers = nullptr;
    uint32_rtv_clear_root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    uint32_rtv_clear_root_signature_ =
        ui::d3d12::util::CreateRootSignature(provider, uint32_rtv_clear_root_desc);
    if (!uint32_rtv_clear_root_signature_) {
      REXGPU_ERROR(
          "D3D12RenderTargetCache: Failed to create the k_32_FLOAT / "
          "k_32_32_FLOAT render target clearing root signature");
      Shutdown();
      return false;
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC uint32_rtv_clear_pipeline_desc = {};
    uint32_rtv_clear_pipeline_desc.pRootSignature = uint32_rtv_clear_root_signature_;
    uint32_rtv_clear_pipeline_desc.VS.pShaderBytecode = shaders::fullscreen_cw_vs;
    uint32_rtv_clear_pipeline_desc.VS.BytecodeLength = sizeof(shaders::fullscreen_cw_vs);
    uint32_rtv_clear_pipeline_desc.PS.pShaderBytecode = shaders::clear_uint2_ps;
    uint32_rtv_clear_pipeline_desc.PS.BytecodeLength = sizeof(shaders::clear_uint2_ps);
    uint32_rtv_clear_pipeline_desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
        D3D12_COLOR_WRITE_ENABLE_ALL;
    uint32_rtv_clear_pipeline_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    uint32_rtv_clear_pipeline_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    uint32_rtv_clear_pipeline_desc.RasterizerState.DepthClipEnable = TRUE;
    uint32_rtv_clear_pipeline_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    uint32_rtv_clear_pipeline_desc.NumRenderTargets = 1;
    for (size_t i = 0; i < 2; ++i) {
      uint32_rtv_clear_pipeline_desc.RTVFormats[0] =
          GetColorOwnershipTransferDXGIFormat(i ? xenos::ColorRenderTargetFormat::k_32_32_FLOAT
                                                : xenos::ColorRenderTargetFormat::k_32_FLOAT);
      for (size_t j = size_t(xenos::MsaaSamples::k1X); j <= size_t(xenos::MsaaSamples::k4X); ++j) {
        if (xenos::MsaaSamples(j) == xenos::MsaaSamples::k2X && !msaa_2x_supported_) {
          // Using sample 0 as 0 and 3 as 1 for 2x instead.
          uint32_rtv_clear_pipeline_desc.SampleMask = 0b1001;
          uint32_rtv_clear_pipeline_desc.SampleDesc.Count = 4;
        } else {
          uint32_rtv_clear_pipeline_desc.SampleMask = UINT_MAX;
          uint32_rtv_clear_pipeline_desc.SampleDesc.Count = 1 << j;
        }
        ID3D12PipelineState* uint32_rtv_clear_pipeline;
        if (FAILED(device->CreateGraphicsPipelineState(&uint32_rtv_clear_pipeline_desc,
                                                       IID_PPV_ARGS(&uint32_rtv_clear_pipeline)))) {
          REXGPU_ERROR(
              "D3D12RenderTargetCache: Failed to create the {} {}-sample "
              "render target clearing pipeline",
              i ? "k_32_32_FLOAT" : "k_32_FLOAT", uint32_t(1) << j);
          Shutdown();
          return false;
        }
        uint32_rtv_clear_pipelines_[i][j] = uint32_rtv_clear_pipeline;
        auto uint32_rtv_clear_pipeline_name = rex::string::to_utf16(fmt::format(
            "Resolve Clear {} {}xMSAA", i ? "k_32_32_FLOAT" : "k_32_FLOAT", uint32_t(1) << j));
        uint32_rtv_clear_pipeline->SetName(
            reinterpret_cast<LPCWSTR>(uint32_rtv_clear_pipeline_name.c_str()));
      }
    }

    // FXC-compiled depth / stencil dumping shader is ~2 KB, reserve 4 KB for
    // some additional space.
    built_shader_.reserve(1024);
  } else {
    // [N-10b deletion c] the ROV / pixel-shader-interlock init branch (the
    // resolve EDRAM clear root signature + 2 compute pipelines) is DELETED.
    assert_unhandled_case(path_);
    Shutdown();
    return false;
  }

  // [N-6] Say out loud whether the console EDRAM wrap was rewritten. An
  // unpatched resolve shader still folds any address past tile
  // kEdramGuestTileCount back to zero, which is guest row 512 at the de-tiled
  // pitch - the band artifact.
  //
  // [N-6-6] Both forms are now rewritten, and the counts are broken out
  // because they fail independently. The unscaled blobs carry the wrap as a
  // literal; the resolution-scaled ones build it as `tile_samples << 11`,
  // where 11 is log2 of the console tile count. At resolution_scale 2 only
  // the scaled variants are ever created, so before this the line read
  // "0 patched, 9 left" and 1440p banded while 720p did not. `left` must be 0
  // at every resolution scale.
  REXGPU_INFO(
      "D3D12RenderTargetCache: resolve EDRAM wrap {} -> {} dwords (shift {} -> {}): "
      "{} shader(s) patched, {} of them resolution-scaled, {} left on the console wrap",
      kConsoleEdramWrapDwords, kHostEdramWrapDwords, kConsoleEdramWrapShift,
      kHostEdramWrapShift, resolve_shaders_edram_wrap_patched,
      resolve_shaders_edram_wrap_by_shift, resolve_shaders_edram_wrap_unpatched);

  InitializeCommon();

  return true;
}

void D3D12RenderTargetCache::Shutdown(bool from_destructor) {

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = size_t(xenos::MsaaSamples::k1X); j <= size_t(xenos::MsaaSamples::k4X); ++j) {
      ui::d3d12::util::ReleaseAndNull(uint32_rtv_clear_pipelines_[i][j]);
    }
  }
  ui::d3d12::util::ReleaseAndNull(uint32_rtv_clear_root_signature_);

  for (const auto& dump_pipeline_pair : dump_pipelines_) {
    if (dump_pipeline_pair.second) {
      dump_pipeline_pair.second->Release();
    }
  }
  dump_pipelines_.clear();
  for (const auto& direct_resolve_pipeline_pair : direct_resolve_pipelines_) {
    if (direct_resolve_pipeline_pair.second) {
      direct_resolve_pipeline_pair.second->Release();
    }
  }
  direct_resolve_pipelines_.clear();
  direct_resolve_plan_ = DirectResolvePlan();
  ui::d3d12::util::ReleaseAndNull(direct_resolve_root_signature_depth_);
  ui::d3d12::util::ReleaseAndNull(direct_resolve_root_signature_color_);
  ui::d3d12::util::ReleaseAndNull(dres_verify_readback_[0]);
  ui::d3d12::util::ReleaseAndNull(dres_verify_readback_[1]);
  ui::d3d12::util::ReleaseAndNull(dres_verify_readback_[2]);
  dres_verify_pending_length_ = 0;
  ui::d3d12::util::ReleaseAndNull(dump_root_signature_depth_);
  ui::d3d12::util::ReleaseAndNull(dump_root_signature_color_);

  // [NR-XFER] ComPtr + descriptor RAII; cleared before the SRV descriptor
  // pool member is destroyed.
  native_hds_scratch_.clear();

  for (const auto& transfer_pipeline_array_pair : transfer_stencil_bit_pipelines_) {
    for (ID3D12PipelineState* transfer_pipeline : transfer_pipeline_array_pair.second) {
      if (transfer_pipeline) {
        transfer_pipeline->Release();
      }
    }
  }
  transfer_stencil_bit_pipelines_.clear();
  for (const auto& transfer_pipeline_pair : transfer_pipelines_) {
    if (transfer_pipeline_pair.second) {
      transfer_pipeline_pair.second->Release();
    }
  }
  transfer_pipelines_.clear();
  for (size_t i = 0; i < rex::countof(transfer_root_signatures_); ++i) {
    ui::d3d12::util::ReleaseAndNull(transfer_root_signatures_[i]);
  }

  transfer_vertex_buffer_pool_.reset();


  null_rtv_descriptor_ms_.Free();
  null_rtv_descriptor_ss_.Free();
  descriptor_pool_srv_.reset();
  descriptor_pool_depth_.reset();
  descriptor_pool_color_.reset();

  for (size_t i = 0; i < rex::countof(resolve_copy_pipelines_); ++i) {
    ui::d3d12::util::ReleaseAndNull(resolve_copy_pipelines_[i]);
  }
  ui::d3d12::util::ReleaseAndNull(resolve_copy_root_signature_);

  edram_snapshot_restore_pool_.reset();
  ui::d3d12::util::ReleaseAndNull(edram_snapshot_download_buffer_);

  ui::d3d12::util::ReleaseAndNull(edram_buffer_descriptor_heap_);
  ui::d3d12::util::ReleaseAndNull(edram_buffer_);

  if (!from_destructor) {
    ShutdownCommon();
  }
}

void D3D12RenderTargetCache::CompletedSubmissionUpdated() {
  if (edram_snapshot_restore_pool_) {
    edram_snapshot_restore_pool_->Reclaim(command_processor_.GetCompletedSubmission());
  }
  if (transfer_vertex_buffer_pool_) {
    transfer_vertex_buffer_pool_->Reclaim(command_processor_.GetCompletedSubmission());
  }
}

void D3D12RenderTargetCache::BeginSubmission() {
  // New command list - render targets not bound.
  InvalidateCommandListRenderTargets();
  // ExecuteCommandLists is a full UAV barrier.
  if (edram_buffer_modification_status_ != EdramBufferModificationStatus::kUnmodified) {
    assert_true(edram_buffer_state_ == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    edram_buffer_modification_status_ = EdramBufferModificationStatus::kUnmodified;
    PixelShaderInterlockFullEdramBarrierPlaced();
  }
}

bool D3D12RenderTargetCache::Update(bool is_rasterization_done,
                                    reg::RB_DEPTHCONTROL normalized_depth_control,
                                    uint32_t normalized_color_mask, const Shader& vertex_shader) {
  if (!RenderTargetCache::Update(is_rasterization_done, normalized_depth_control,
                                 normalized_color_mask, vertex_shader)) {
    return false;
  }
  switch (GetPath()) {
    case Path::kHostRenderTargets: {
      RenderTarget* const* depth_and_color_render_targets =
          last_update_accumulated_render_targets();
      PerformTransfersAndResolveClears(1 + xenos::kMaxColorRenderTargets,
                                       depth_and_color_render_targets, last_update_transfers());
      // [xfer] The read census: does this draw read what the transfers landed?
      // Depth: a depth or stencil test that can reject. Color: a blend factor
      // that samples the destination, a min/max blend op, or a partial write
      // mask. Everything else is a plain write (coverage unknown).
      {
        const RegisterFile& regs = register_file();
        RenderTarget* const* used = last_update_used_render_targets();
        if (used[0]) {
          auto op_writes = [](xenos::StencilOp op) {
            return op == xenos::StencilOp::kReplace || op == xenos::StencilOp::kZero;
          };
          const auto& dc = normalized_depth_control;
          const bool z_reads = dc.z_enable && dc.zfunc != xenos::CompareFunction::kAlways;
          // A stencil pass is a pure write only when nothing can depend on the
          // old value: func ALWAYS and every reachable op replaces.
          bool stencil_reads = false;
          if (dc.stencil_enable) {
            const bool front_writes = dc.stencilfunc == xenos::CompareFunction::kAlways &&
                                      op_writes(dc.stencilzpass) &&
                                      (!z_reads || op_writes(dc.stencilzfail));
            const bool back_writes = !dc.backface_enable ||
                                     (dc.stencilfunc_bf == xenos::CompareFunction::kAlways &&
                                      op_writes(dc.stencilzpass_bf) &&
                                      (!z_reads || op_writes(dc.stencilzfail_bf)));
            stencil_reads = !(front_writes && back_writes);
          }
          const bool depth_reads = z_reads || stencil_reads;
          XferUseNoteDraw(used[0]->key(), depth_reads);
        }
        auto factor_reads_dest = [](xenos::BlendFactor f) {
          return f == xenos::BlendFactor::kDstColor || f == xenos::BlendFactor::kOneMinusDstColor ||
                 f == xenos::BlendFactor::kDstAlpha || f == xenos::BlendFactor::kOneMinusDstAlpha ||
                 f == xenos::BlendFactor::kSrcAlphaSaturate;
        };
        for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
          if (!used[1 + i]) {
            continue;
          }
          auto blend = regs.Get<reg::RB_BLENDCONTROL>(reg::RB_BLENDCONTROL::rt_register_indices[i]);
          const bool color_reads =
              factor_reads_dest(blend.color_srcblend) ||
              blend.color_destblend != xenos::BlendFactor::kZero ||
              blend.color_comb_fcn == xenos::BlendOp::kMin ||
              blend.color_comb_fcn == xenos::BlendOp::kMax ||
              factor_reads_dest(blend.alpha_srcblend) ||
              blend.alpha_destblend != xenos::BlendFactor::kZero ||
              blend.alpha_comb_fcn == xenos::BlendOp::kMin ||
              blend.alpha_comb_fcn == xenos::BlendOp::kMax ||
              ((normalized_color_mask >> (i * 4)) & 0xF) != 0xF;
          XferUseNoteDraw(used[1 + i]->key(), color_reads);
        }
      }
      SetCommandListRenderTargets(depth_and_color_render_targets);
    } break;
    // [N-10b deletion c] the kPixelShaderInterlock (ROV) case is DELETED.
    default:
      assert_unhandled_case(GetPath());
      return false;
  }
  return true;
}

void D3D12RenderTargetCache::WriteEdramRawSRVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle) {
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  device->CopyDescriptorsSimple(
      1, handle,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kRawSRV)),
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void D3D12RenderTargetCache::WriteEdramRawUAVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle) {
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  device->CopyDescriptorsSimple(
      1, handle,
      provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                    uint32_t(EdramBufferDescriptorIndex::kRawUAV)),
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void D3D12RenderTargetCache::WriteEdramUintPow2SRVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                                             uint32_t element_size_bytes_pow2) {
  EdramBufferDescriptorIndex descriptor_index;
  switch (element_size_bytes_pow2) {
    case 2:
      descriptor_index = EdramBufferDescriptorIndex::kR32UintSRV;
      break;
    case 3:
      descriptor_index = EdramBufferDescriptorIndex::kR32G32UintSRV;
      break;
    case 4:
      descriptor_index = EdramBufferDescriptorIndex::kR32G32B32A32UintSRV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      return;
  }
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  device->CopyDescriptorsSimple(1, handle,
                                provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                                              uint32_t(descriptor_index)),
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void D3D12RenderTargetCache::WriteEdramUintPow2UAVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                                             uint32_t element_size_bytes_pow2) {
  EdramBufferDescriptorIndex descriptor_index;
  switch (element_size_bytes_pow2) {
    case 2:
      descriptor_index = EdramBufferDescriptorIndex::kR32UintUAV;
      break;
    case 3:
      descriptor_index = EdramBufferDescriptorIndex::kR32G32UintUAV;
      break;
    case 4:
      descriptor_index = EdramBufferDescriptorIndex::kR32G32B32A32UintUAV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      return;
  }
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  device->CopyDescriptorsSimple(1, handle,
                                provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                                              uint32_t(descriptor_index)),
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

bool D3D12RenderTargetCache::Resolve(const memory::Memory& memory, D3D12SharedMemory& shared_memory,
                                     D3D12TextureCache& texture_cache,
                                     uint32_t& written_address_out, uint32_t& written_length_out) {
  written_address_out = 0;
  written_length_out = 0;

  bool draw_resolution_scaled = IsDrawResolutionScaled();

  draw_util::ResolveInfo resolve_info;
  bool fixed_16_truncated_to_minus_1_to_1 = IsFixed16TruncatedToMinus1To1();
  if (!draw_util::GetResolveInfo(register_file(), memory, trace_writer_, draw_resolution_scale_x(),
                                 draw_resolution_scale_y(), fixed_16_truncated_to_minus_1_to_1,
                                 fixed_16_truncated_to_minus_1_to_1, resolve_info)) {
    return false;
  }

  // Nothing to copy/clear.
  if (!resolve_info.coordinate_info.width_div_8 || !resolve_info.height_div_8) {
    return true;
  }

  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();

  // Copying.
  bool copied = false;
  if (resolve_info.copy_dest_extent_length) {
    draw_util::ResolveCopyShaderConstants copy_shader_constants;
    uint32_t copy_group_count_x, copy_group_count_y;
    draw_util::ResolveCopyShaderIndex copy_shader =
        resolve_info.GetCopyShader(draw_resolution_scale_x(), draw_resolution_scale_y(),
                                   copy_shader_constants, copy_group_count_x, copy_group_count_y);
    assert_true(copy_group_count_x && copy_group_count_y);
    if (copy_shader != draw_util::ResolveCopyShaderIndex::kUnknown) {
      const draw_util::ResolveCopyShaderInfo& copy_shader_info =
          draw_util::resolve_copy_shader_info[size_t(copy_shader)];
      bool direct_resolved = false;
      bool dres_verify_this = false;
      if (GetPath() == Path::kHostRenderTargets) {
        // [NR-DRES] Direct resolve is unconditional (N-10a shipped);
        // direct_host_resolve stays the shared outer switch.
        if (REXCVAR_GET(direct_host_resolve)) {
          direct_resolved =
              TryResolveCopyDirectly(resolve_info, copy_shader, draw_resolution_scaled);
          if (direct_resolved) {
            ++direct_resolve_success_count_;
            if (REXCVAR_GET(gpu_nr_direct_resolve_verify) && !dres_verify_pending_length_) {
              const auto dres_now = std::chrono::steady_clock::now();
              if (dres_now - dres_verify_last_ >= std::chrono::seconds(1)) {
                dres_verify_last_ = dres_now;
                dres_verify_this = true;
              }
            }
          } else {
            ++direct_resolve_fallback_count_;
          }
        }
        // [NR-DETILE] N-6 probe, resolve level. The dump probe alone can only
        // speak when a dump HAPPENS; a resolve that never dumps leaves it
        // silent, and silence has already been mistaken for absence twice in
        // this hunt. Report every resolve's EDRAM span and the path it took,
        // so the colour tap cannot go missing without saying so.
        if (REXCVAR_GET(gpu_nr_dump_probe)) {
          uint32_t rp_base, rp_row_len, rp_rows, rp_pitch;
          resolve_info.GetCopyEdramTileSpan(rp_base, rp_row_len, rp_rows, rp_pitch);
          const uint64_t rp_sig = (uint64_t(rp_base) << 40) ^ (uint64_t(rp_pitch) << 24) ^
                                  (uint64_t(rp_rows) << 8) ^ uint64_t(rp_row_len);
          static std::map<uint64_t, std::chrono::steady_clock::time_point> rp_seen;
          const auto rp_now = std::chrono::steady_clock::now();
          auto rp_it = rp_seen.find(rp_sig);
          if (rp_it == rp_seen.end() || rp_now - rp_it->second >= std::chrono::seconds(1)) {
            if (rp_seen.size() > 64) {
              rp_seen.clear();
            }
            rp_seen[rp_sig] = rp_now;
            REXGPU_INFO(
                "[nr-detile] resolve: edram base={}t row_len={}t rows={} pitch={}t "
                "| shader={} direct={} dest={:08X} extent={:08X}+{:X} ({} rows of 1280x4)",
                rp_base, rp_row_len, rp_rows, rp_pitch, uint32_t(copy_shader),
                direct_resolved ? 1 : 0, resolve_info.copy_dest_base,
                resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length,
                resolve_info.copy_dest_extent_length / (1280 * 4));
          }
        }
        if (!direct_resolved || dres_verify_this) {
          // Dump the current contents of the render targets owning the affected
          // range to edram_buffer_. Under verify the dump runs even when the
          // direct path is taken, so the legacy result exists to compare with.
          uint32_t dump_base;
          uint32_t dump_row_length_used;
          uint32_t dump_rows;
          uint32_t dump_pitch;
          resolve_info.GetCopyEdramTileSpan(dump_base, dump_row_length_used, dump_rows, dump_pitch);
          if (!DumpRenderTargets(dump_base, dump_row_length_used, dump_rows, dump_pitch)) {
            REXGPU_ERROR("D3D12RenderTargetCache: Failed to dump host render targets for resolve");
            return false;
          }
        }

        // [NR-DETILE] N-6-2, re-keyed in N-6-5. The pixels themselves.
        if (REXCVAR_GET(gpu_nr_detile_edram_probe)) {
          uint32_t ep_base, ep_row_len, ep_rows, ep_pitch;
          resolve_info.GetCopyEdramTileSpan(ep_base, ep_row_len, ep_rows, ep_pitch);
          const bool ep_depth = resolve_info.IsCopyingDepth();
          const draw_util::ResolveEdramInfo& ep_info =
              ep_depth ? resolve_info.depth_edram_info : resolve_info.color_edram_info;
          // A tile is 16 samples tall; at 2x/4x MSAA two of them are one guest
          // row, so a tile row is 8 guest rows there and 16 at 1x. The old
          // probe assumed 8 unconditionally and mislabelled the msaa-0 control
          // slot by a factor of two.
          const uint32_t ep_guest_rows =
              xenos::kEdramTileHeightSamples >>
              uint32_t(xenos::MsaaSamples(ep_info.msaa_samples) >= xenos::MsaaSamples::k2X);
          // Gate to the packets this names: the de-tiled full-frame taps are
          // pitch 32, 90 tile rows (720 guest rows at 4xMSAA). The pitch-16
          // msaa-0 post resolve carries the image that actually reaches the
          // screen and is the instrument's own CONTROL - if it reads flat too,
          // the probe is broken, not the render target.
          //
          // ONE SLOT PER TAP, keyed by destination: the two band-1 colour taps
          // share every field this used to key on, and sharing a slot made
          // them share a rate limiter too.
          const char* ep_name = nullptr;
          if (ep_pitch == 32 && ep_rows >= 90) {
            ep_name = ep_depth ? "tiled depth" : "tiled color";
          } else if (ep_base == 0 && ep_pitch == 16 && ep_rows >= 45 && !ep_depth) {
            ep_name = "post color CONTROL";
          }
          if (ep_name) {
            const uint64_t ep_key = (uint64_t(resolve_info.copy_dest_base) << 2) |
                                    (ep_depth ? 1u : 0u) |
                                    (ep_pitch == 16 ? 2u : 0u);
            NrDetileEdramProbe(
                command_processor_, edram_buffer_, ep_key, ep_name, ep_base, ep_rows, ep_pitch,
                ep_guest_rows, resolve_info.copy_dest_base, resolve_info.rb_copy_control.value,
                direct_resolved,
                [this](D3D12_RESOURCE_STATES state) { TransitionEdramBuffer(state); });
          }
        }
      }

      // Make sure there is memory to write to.
      bool copy_dest_committed;
      if (draw_resolution_scaled) {
        // Committing starting with the beginning of the potentially written
        // extent, but making the buffer containing the base current as the
        // beginning of the bound buffer is the base.
        copy_dest_committed =
            texture_cache.EnsureScaledResolveMemoryCommitted(
                resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length) &&
            texture_cache.MakeScaledResolveRangeCurrent(resolve_info.copy_dest_base,
                                                        resolve_info.copy_dest_extent_start -
                                                            resolve_info.copy_dest_base +
                                                            resolve_info.copy_dest_extent_length);
      } else {
        copy_dest_committed = shared_memory.RequestRange(resolve_info.copy_dest_extent_start,
                                                         resolve_info.copy_dest_extent_length);
      }
      if (copy_dest_committed) {
        // [NR-DRES] The vendored EDRAM copy runs when the direct path declined
        // (the normal fallback) or when verify wants the legacy result to
        // compare against; the direct dispatch then overwrites the same
        // destination so what ships is always the direct result.
        bool resolve_written = false;
        if (!direct_resolved || dres_verify_this) {
        // Write the descriptors and transition the resources.
        // Full shared memory without resolution scaling, range of the scaled
        // resolve buffer with scaling because only at least 128 * 2^20 R32
        // elements must be addressable
        // (D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP).
        ui::d3d12::util::DescriptorCpuGpuHandlePair descriptor_dest;
        ui::d3d12::util::DescriptorCpuGpuHandlePair descriptor_source;
        ui::d3d12::util::DescriptorCpuGpuHandlePair descriptors[2];
        if (command_processor_.RequestOneUseSingleViewDescriptors(
                bindless_resources_used_ ? uint32_t(draw_resolution_scaled) : 2, descriptors)) {
          if (bindless_resources_used_) {
            if (draw_resolution_scaled) {
              descriptor_dest = descriptors[0];
            } else {
              descriptor_dest = command_processor_.GetSharedMemoryUintPow2BindlessUAVHandlePair(
                  copy_shader_info.dest_bpe_log2);
            }
            if (copy_shader_info.source_is_raw) {
              descriptor_source = command_processor_.GetSystemBindlessViewHandlePair(
                  D3D12CommandProcessor::SystemBindlessView::kEdramRawSRV);
            } else {
              descriptor_source = command_processor_.GetEdramUintPow2BindlessSRVHandlePair(
                  copy_shader_info.source_bpe_log2);
            }
          } else {
            descriptor_dest = descriptors[0];
            if (!draw_resolution_scaled) {
              shared_memory.WriteUintPow2UAVDescriptor(descriptor_dest.first,
                                                       copy_shader_info.dest_bpe_log2);
            }
            descriptor_source = descriptors[1];
            if (copy_shader_info.source_is_raw) {
              WriteEdramRawSRVDescriptor(descriptor_source.first);
            } else {
              WriteEdramUintPow2SRVDescriptor(descriptor_source.first,
                                              copy_shader_info.source_bpe_log2);
            }
          }
          if (draw_resolution_scaled) {
            texture_cache.CreateCurrentScaledResolveRangeUintPow2UAV(
                descriptor_dest.first, copy_shader_info.dest_bpe_log2);
            texture_cache.TransitionCurrentScaledResolveRange(
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
          } else {
            shared_memory.UseForWriting();
          }
          TransitionEdramBuffer(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

          // Submit the resolve.
          command_list.D3DSetComputeRootSignature(resolve_copy_root_signature_);
          command_list.D3DSetComputeRootDescriptorTable(2, descriptor_source.second);
          command_list.D3DSetComputeRootDescriptorTable(1, descriptor_dest.second);
          // [N-6] The blobs unpack the pre-L1.1 field layout, so hand them a
          // word in that layout. Local copy: copy_shader_constants stays in
          // the internal layout for anything else that reads it.
          draw_util::ResolveCopyShaderConstants vendored_constants = copy_shader_constants;
          bool vendored_base_truncated = false;
          vendored_constants.dest_relative.edram_info.packed =
              PackResolveEdramInfoForVendoredShader(copy_shader_constants.dest_relative.edram_info,
                                                    vendored_base_truncated);
          if (vendored_base_truncated) {
            // Counted, and named the first time. Every resolve that lands here
            // reads the wrong EDRAM surface - the same way it did before the
            // repack existed, so this is a pre-existing break made visible, not
            // a new one.
            static uint32_t base_truncated_count = 0;
            const uint32_t base_tiles =
                uint32_t(copy_shader_constants.dest_relative.edram_info.base_tiles);
            if (!base_truncated_count++) {
              REXGPU_WARN(
                  "D3D12RenderTargetCache: resolve source base {} tiles does not fit the {} "
                  "bits the vendored resolve shaders give it - reading tile {} instead. Bind "
                  "the EDRAM view at a tile offset to fix this.",
                  base_tiles, kVendoredResolveBaseTilesBits,
                  base_tiles & ((uint32_t(1) << kVendoredResolveBaseTilesBits) - 1));
            }
          }
          if (draw_resolution_scaled) {
            command_list.D3DSetComputeRoot32BitConstants(
                0, sizeof(vendored_constants.dest_relative) / sizeof(uint32_t),
                &vendored_constants.dest_relative, 0);
          } else {
            command_list.D3DSetComputeRoot32BitConstants(
                0, sizeof(vendored_constants) / sizeof(uint32_t), &vendored_constants, 0);
          }
          command_processor_.SetExternalPipeline(resolve_copy_pipelines_[size_t(copy_shader)]);
          command_processor_.SubmitBarriers();
          command_list.D3DDispatch(copy_group_count_x, copy_group_count_y, 1);

          resolve_written = true;
        }
        }  // !direct_resolved || dres_verify_this
        const bool dres_control = REXCVAR_GET(gpu_nr_dres_verify_control);
        if (resolve_written && dres_verify_this) {
          // Snapshot the legacy result before the direct dispatch overwrites
          // the destination. In control mode take BOTH snapshots here with
          // nothing between: any diverge then is the harness's own.
          DirectResolveVerifySnapshot(shared_memory, texture_cache, 0,
                                      resolve_info.copy_dest_extent_start,
                                      resolve_info.copy_dest_extent_length);
          if (dres_control) {
            DirectResolveVerifySnapshot(shared_memory, texture_cache, 1,
                                        resolve_info.copy_dest_extent_start,
                                        resolve_info.copy_dest_extent_length);
            DirectResolveVerifySnapshotSource();
          }
        }
        if (direct_resolved) {
          if (DispatchDirectResolve(shared_memory, texture_cache)) {
            resolve_written = true;
            if (dres_verify_this && !dres_control) {
              DirectResolveVerifySnapshot(shared_memory, texture_cache, 1,
                                          resolve_info.copy_dest_extent_start,
                                          resolve_info.copy_dest_extent_length);
              DirectResolveVerifySnapshotSource();
            }
          }
        }
        if (resolve_written) {
          // Order the resolve with other work using the destination as a UAV.
          if (draw_resolution_scaled) {
            texture_cache.MarkCurrentScaledResolveRangeUAVWritesCommitNeeded();
          } else {
            shared_memory.MarkUAVWritesCommitNeeded();
          }

          // Invalidate textures and mark the range as scaled if needed.
          texture_cache.MarkRangeAsResolved(resolve_info.copy_dest_extent_start,
                                            resolve_info.copy_dest_extent_length);
          written_address_out = resolve_info.copy_dest_extent_start;
          written_length_out = resolve_info.copy_dest_extent_length;
          copied = true;
        }
      } else {
        REXGPU_ERROR(
            "D3D12RenderTargetCache: Failed to obtain the resolve destination "
            "memory region");
      }
    }
  } else {
    copied = true;
  }

  // [NR-DRES] Flush the verify compare and print the 1 Hz census.
  ReportDirectResolveStats();

  // Clearing.
  bool cleared = false;
  bool clear_depth = resolve_info.IsClearingDepth();
  bool clear_color = resolve_info.IsClearingColor();
  if (clear_depth || clear_color) {
    switch (GetPath()) {
      case Path::kHostRenderTargets: {
        Transfer::Rectangle clear_rectangle = {};
        RenderTarget* clear_render_targets[2] = {};
        // If PrepareHostRenderTargetsResolveClear returns false, may be just an
        // empty region (success) or an error - don't care.
        const bool clear_prepared = PrepareHostRenderTargetsResolveClear(
            resolve_info, clear_rectangle, clear_render_targets[0], clear_transfers_[0],
            clear_render_targets[1], clear_transfers_[1]);
        // [NR-DETILE] N-6-5. What the clear that rides this resolve actually
        // covers. Deduped by shape so every DISTINCT clear reports once a
        // second and none can hide behind a busier one.
        if (REXCVAR_GET(gpu_nr_detile_clear_probe)) {
          auto rt_desc = [](const RenderTarget* rt) {
            if (!rt) {
              return std::string("none");
            }
            RenderTargetKey k = rt->key();
            return fmt::format("base={}t pitch={}t msaa={} depth={} fmt={}",
                               uint32_t(k.base_tiles), uint32_t(k.pitch_tiles_at_32bpp),
                               uint32_t(k.msaa_samples), uint32_t(k.is_depth),
                               uint32_t(k.resource_format));
          };
          const uint64_t cp_sig =
              (uint64_t(clear_rectangle.y_pixels) << 44) ^
              (uint64_t(clear_rectangle.height_pixels) << 30) ^
              (uint64_t(clear_rectangle.x_pixels) << 16) ^
              uint64_t(clear_rectangle.width_pixels) ^
              (uint64_t(resolve_info.rb_copy_control.value) << 52) ^
              (uint64_t(clear_prepared) << 63);
          static std::map<uint64_t, std::chrono::steady_clock::time_point> cp_seen;
          const auto cp_now = std::chrono::steady_clock::now();
          auto cp_it = cp_seen.find(cp_sig);
          if (cp_it == cp_seen.end() || cp_now - cp_it->second >= std::chrono::seconds(1)) {
            if (cp_seen.size() > 64) {
              cp_seen.clear();
            }
            cp_seen[cp_sig] = cp_now;
            REXGPU_INFO(
                "[nr-detile] clear: prepared={} ctrl={:08X} depth={} color={} | rect x={} y={} "
                "w={} h={} | resolve y={}..{} | depth RT {} (orig_base={}t edram_base={}t "
                "pitch={}t) xfers={} | color RT {} (orig_base={}t edram_base={}t pitch={}t) "
                "xfers={}",
                clear_prepared ? 1 : 0, resolve_info.rb_copy_control.value, clear_depth ? 1 : 0,
                clear_color ? 1 : 0, clear_rectangle.x_pixels, clear_rectangle.y_pixels,
                clear_rectangle.width_pixels, clear_rectangle.height_pixels,
                uint32_t(resolve_info.coordinate_info.edram_offset_y_div_8) << 3,
                (uint32_t(resolve_info.coordinate_info.edram_offset_y_div_8) << 3) +
                    (resolve_info.height_div_8 << 3),
                rt_desc(clear_render_targets[0]), resolve_info.depth_original_base,
                uint32_t(resolve_info.depth_edram_info.base_tiles),
                uint32_t(resolve_info.depth_edram_info.pitch_tiles), clear_transfers_[0].size(),
                rt_desc(clear_render_targets[1]), resolve_info.color_original_base,
                uint32_t(resolve_info.color_edram_info.base_tiles),
                uint32_t(resolve_info.color_edram_info.pitch_tiles), clear_transfers_[1].size());
          }
        }
        if (clear_prepared) {
          uint64_t clear_values[2];
          clear_values[0] = resolve_info.rb_depth_clear;
          clear_values[1] =
              resolve_info.rb_color_clear | (uint64_t(resolve_info.rb_color_clear_lo) << 32);
          PerformTransfersAndResolveClears(2, clear_render_targets, clear_transfers_, clear_values,
                                           &clear_rectangle);
        }
        cleared = true;
      } break;
      // [N-10b deletion c] the kPixelShaderInterlock (ROV) resolve-clear
      // case - the ONE EDRAM-buffer clear path in the game - is DELETED.
      default:
        assert_unhandled_case(GetPath());
    }
  } else {
    cleared = true;
  }

  return copied && cleared;
}

bool D3D12RenderTargetCache::InitializeTraceSubmitDownloads() {
  if (IsDrawResolutionScaled()) {
    // No 1:1 mapping.
    return false;
  }
  if (!edram_snapshot_download_buffer_) {
    D3D12_RESOURCE_DESC edram_snapshot_download_buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(edram_snapshot_download_buffer_desc,
                                            xenos::kEdramSizeBytes, D3D12_RESOURCE_FLAG_NONE);
    const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &edram_snapshot_download_buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&edram_snapshot_download_buffer_)))) {
      REXGPU_ERROR(
          "D3D12RenderTargetCache: Failed to create a EDRAM snapshot download "
          "buffer");
      return false;
    }
  }
  if (GetPath() == Path::kHostRenderTargets) {
    // Dump all host render targets to edram_buffer_.
    if (!DumpRenderTargets(0, xenos::kEdramTileCount, 1, xenos::kEdramTileCount)) {
      REXGPU_ERROR("D3D12RenderTargetCache: Failed to dump host render targets for trace");
      return false;
    }
  }
  TransitionEdramBuffer(D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_processor_.SubmitBarriers();
  command_processor_.GetDeferredCommandList().D3DCopyBufferRegion(
      edram_snapshot_download_buffer_, 0, edram_buffer_, 0, xenos::kEdramSizeBytes);
  return true;
}

void D3D12RenderTargetCache::InitializeTraceCompleteDownloads() {
  if (!edram_snapshot_download_buffer_) {
    return;
  }
  void* download_mapping;
  if (SUCCEEDED(edram_snapshot_download_buffer_->Map(0, nullptr, &download_mapping))) {
    trace_writer_.WriteEdramSnapshot(download_mapping);
    D3D12_RANGE download_write_range = {};
    edram_snapshot_download_buffer_->Unmap(0, &download_write_range);
  } else {
    REXGPU_ERROR(
        "D3D12RenderTargetCache: Failed to map the EDRAM snapshot download "
        "buffer");
  }
  edram_snapshot_download_buffer_->Release();
  edram_snapshot_download_buffer_ = nullptr;
}

void D3D12RenderTargetCache::RestoreEdramSnapshot(const void* snapshot) {
  if (IsDrawResolutionScaled()) {
    // No 1:1 mapping.
    return;
  }

  // Create the buffer - will be used for copying to either a 32-bit 1280x2048
  // render target or the EDRAM buffer.
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  if (!edram_snapshot_restore_pool_) {
    edram_snapshot_restore_pool_ =
        std::make_unique<ui::d3d12::D3D12UploadBufferPool>(provider, xenos::kEdramSizeBytes);
  }
  ID3D12Resource* upload_buffer;
  size_t upload_buffer_offset;
  void* upload_buffer_mapping = edram_snapshot_restore_pool_->Request(
      command_processor_.GetCurrentSubmission(), xenos::kEdramSizeBytes, 1, &upload_buffer,
      &upload_buffer_offset, nullptr);
  if (!upload_buffer_mapping) {
    REXGPU_ERROR(
        "D3D12RenderTargetCache: Failed to get a buffer for restoring a EDRAM "
        "snapshot");
    return;
  }

  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();

  switch (GetPath()) {
    case Path::kHostRenderTargets: {
      // k_32_FLOAT because it's unambiguous (not effected by something like
      // DXGI_FORMAT_R8G8B8A8 vs. DXGI_FORMAT_B8G8R8A8).
      D3D12RenderTarget* full_edram_render_target =
          static_cast<D3D12RenderTarget*>(PrepareFullEdram1280xRenderTargetForSnapshotRestoration(
              xenos::ColorRenderTargetFormat::k_32_FLOAT));
      if (!full_edram_render_target) {
        return;
      }
      D3D12_TEXTURE_COPY_LOCATION copy_source_location;
      copy_source_location.pResource = upload_buffer;
      copy_source_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      UINT64 copy_total_bytes;
      D3D12_RESOURCE_DESC full_edram_render_target_desc =
          full_edram_render_target->resource()->GetDesc();
      provider.GetDevice()->GetCopyableFootprints(&full_edram_render_target_desc, 0, 1, 0,
                                                  &copy_source_location.PlacedFootprint, nullptr,
                                                  nullptr, &copy_total_bytes);
      // 1280 width * sizeof(uint32_t) is aligned to
      // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256).
      assert_true(copy_total_bytes <= xenos::kEdramSizeBytes);
      assert_false(full_edram_render_target->key().Is64bpp());
      uint32_t pitch_tiles = full_edram_render_target->key().pitch_tiles_at_32bpp;
      uint32_t tile_rows = xenos::kEdramTileCount / pitch_tiles;
      assert_true(pitch_tiles * tile_rows == xenos::kEdramTileCount);
      const uint8_t* snapshot_sample_row = reinterpret_cast<const uint8_t*>(snapshot);
      for (uint32_t y_tile = 0; y_tile < tile_rows; ++y_tile) {
        uint8_t* upload_buffer_tile_row_origin =
            reinterpret_cast<uint8_t*>(upload_buffer_mapping) +
            copy_source_location.PlacedFootprint.Offset +
            xenos::kEdramTileHeightSamples * y_tile *
                copy_source_location.PlacedFootprint.Footprint.RowPitch;
        for (uint32_t x_tile = 0; x_tile < pitch_tiles; ++x_tile) {
          uint8_t* upload_buffer_sample_row =
              upload_buffer_tile_row_origin +
              sizeof(uint32_t) * xenos::kEdramTileWidthSamples * x_tile;
          for (uint32_t sample_row = 0; sample_row < xenos::kEdramTileHeightSamples; ++sample_row) {
            std::memcpy(upload_buffer_sample_row, snapshot_sample_row,
                        sizeof(uint32_t) * xenos::kEdramTileWidthSamples);
            snapshot_sample_row += sizeof(uint32_t) * xenos::kEdramTileWidthSamples;
            upload_buffer_sample_row += copy_source_location.PlacedFootprint.Footprint.RowPitch;
          }
        }
      }
      command_processor_.PushTransitionBarrier(
          full_edram_render_target->resource(),
          full_edram_render_target->SetResourceState(D3D12_RESOURCE_STATE_COPY_DEST),
          D3D12_RESOURCE_STATE_COPY_DEST);
      command_processor_.SubmitBarriers();
      D3D12_TEXTURE_COPY_LOCATION copy_dest_location;
      copy_dest_location.pResource = full_edram_render_target->resource();
      copy_dest_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      copy_dest_location.SubresourceIndex = 0;
      command_list.D3DCopyTextureRegion(&copy_dest_location, 0, 0, 0, &copy_source_location,
                                        nullptr);
    } break;

    // [N-10b deletion c] the kPixelShaderInterlock (ROV) snapshot-restore
    // case is DELETED.
    default:
      assert_unhandled_case(GetPath());
  }
}

DXGI_FORMAT D3D12RenderTargetCache::GetColorResourceDXGIFormat(
    xenos::ColorRenderTargetFormat format) const {
  // Typed should be preferred over typeless so there are more opportunities for
  // compression.
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
      return gamma_render_target_as_unorm16_ ? DXGI_FORMAT_R16G16B16A16_UNORM
                                             : DXGI_FORMAT_R8G8B8A8_UNORM;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
      return DXGI_FORMAT_R10G10B10A2_UNORM;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16:
      return DXGI_FORMAT_R16G16B16A16_FLOAT;
    // SNORM has two representations of -1.
    case xenos::ColorRenderTargetFormat::k_16_16:
      return DXGI_FORMAT_R16G16_TYPELESS;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16:
      return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    // Floating-point - ensure NaN propagation during ownership transfer for
    // unmodified data.
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
      return DXGI_FORMAT_R16G16_TYPELESS;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
      return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    // TODO(Triang3l): Check if NaN propagation defined in the D3D11.3
    // specification can be relied on for 32-bit float render targets.
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
      return DXGI_FORMAT_R32_TYPELESS;
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      return DXGI_FORMAT_R32G32_TYPELESS;
    default:
      assert_unhandled_case(format);
      return DXGI_FORMAT_UNKNOWN;
  }
}

DXGI_FORMAT D3D12RenderTargetCache::GetColorDrawDXGIFormat(
    xenos::ColorRenderTargetFormat format) const {
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_16_16:
      return DXGI_FORMAT_R16G16_SNORM;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16:
      return DXGI_FORMAT_R16G16B16A16_SNORM;
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
      return DXGI_FORMAT_R16G16_FLOAT;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
      return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      return DXGI_FORMAT_R32G32_FLOAT;
    default:
      return GetColorResourceDXGIFormat(format);
  }
}

DXGI_FORMAT D3D12RenderTargetCache::GetColorOwnershipTransferDXGIFormat(
    xenos::ColorRenderTargetFormat format, bool* is_integer_out) const {
  if (is_integer_out) {
    *is_integer_out = true;
  }
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
      return DXGI_FORMAT_R16G16_UINT;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
      return DXGI_FORMAT_R16G16B16A16_UINT;
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
      return DXGI_FORMAT_R32_UINT;
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      return DXGI_FORMAT_R32G32_UINT;
    default:
      if (is_integer_out) {
        *is_integer_out = false;
      }
      return GetColorDrawDXGIFormat(format);
  }
}

DXGI_FORMAT D3D12RenderTargetCache::GetDepthResourceDXGIFormat(
    xenos::DepthRenderTargetFormat format) {
  switch (format) {
    case xenos::DepthRenderTargetFormat::kD24S8:
      return DXGI_FORMAT_R24G8_TYPELESS;
    case xenos::DepthRenderTargetFormat::kD24FS8:
      return DXGI_FORMAT_R32G8X24_TYPELESS;
    default:
      assert_unhandled_case(format);
      return DXGI_FORMAT_UNKNOWN;
  }
}

DXGI_FORMAT D3D12RenderTargetCache::GetDepthDSVDXGIFormat(xenos::DepthRenderTargetFormat format) {
  switch (format) {
    case xenos::DepthRenderTargetFormat::kD24S8:
      return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case xenos::DepthRenderTargetFormat::kD24FS8:
      return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    default:
      assert_unhandled_case(format);
      return DXGI_FORMAT_UNKNOWN;
  }
}

DXGI_FORMAT D3D12RenderTargetCache::GetDepthSRVDepthDXGIFormat(
    xenos::DepthRenderTargetFormat format) {
  switch (format) {
    case xenos::DepthRenderTargetFormat::kD24S8:
      return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case xenos::DepthRenderTargetFormat::kD24FS8:
      return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    default:
      assert_unhandled_case(format);
      return DXGI_FORMAT_UNKNOWN;
  }
}

DXGI_FORMAT D3D12RenderTargetCache::GetDepthSRVStencilDXGIFormat(
    xenos::DepthRenderTargetFormat format) {
  switch (format) {
    case xenos::DepthRenderTargetFormat::kD24S8:
      return DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    case xenos::DepthRenderTargetFormat::kD24FS8:
      return DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    default:
      assert_unhandled_case(format);
      return DXGI_FORMAT_UNKNOWN;
  }
}

ID3D12Resource* D3D12RenderTargetCache::GetBoundDepthForHiz(uint32_t& width_out,
                                                            uint32_t& height_out,
                                                            uint32_t& samples_out,
                                                            D3D12_CPU_DESCRIPTOR_HANDLE& srv_out) {
  if (GetPath() != Path::kHostRenderTargets) {
    return nullptr;
  }
  RenderTarget* rt = last_update_accumulated_render_targets()[0];
  if (!rt) {
    return nullptr;
  }
  auto& d3d12_rt = *static_cast<D3D12RenderTarget*>(rt);
  if (!d3d12_rt.resource() || !d3d12_rt.descriptor_srv().IsValid()) {
    return nullptr;
  }
  const D3D12_RESOURCE_DESC desc = d3d12_rt.resource()->GetDesc();
  width_out = uint32_t(desc.Width);
  height_out = desc.Height;
  samples_out = desc.SampleDesc.Count;
  srv_out = d3d12_rt.descriptor_srv().GetHandle();
  return d3d12_rt.resource();
}

void D3D12RenderTargetCache::TransitionBoundDepthForHiz(D3D12_RESOURCE_STATES state) {
  RenderTarget* rt = last_update_accumulated_render_targets()[0];
  if (!rt) {
    return;
  }
  auto& d3d12_rt = *static_cast<D3D12RenderTarget*>(rt);
  command_processor_.PushTransitionBarrier(d3d12_rt.resource(), d3d12_rt.SetResourceState(state),
                                           state);
}

RenderTargetCache::RenderTarget* D3D12RenderTargetCache::CreateRenderTarget(RenderTargetKey key) {
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();

  D3D12_RESOURCE_DESC resource_desc;
  resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  resource_desc.Alignment = 0;
  resource_desc.Width = key.GetWidth() * draw_resolution_scale_x();
  resource_desc.Height =
      GetRenderTargetHeight(key.pitch_tiles_at_32bpp, key.msaa_samples) * draw_resolution_scale_y();
  resource_desc.DepthOrArraySize = 1;
  resource_desc.MipLevels = 1;
  if (key.is_depth) {
    resource_desc.Format = GetDepthResourceDXGIFormat(key.GetDepthFormat());
  } else {
    resource_desc.Format = GetColorResourceDXGIFormat(key.GetColorFormat());
  }
  assert_true(resource_desc.Format != DXGI_FORMAT_UNKNOWN);
  if (resource_desc.Format == DXGI_FORMAT_UNKNOWN) {
    REXGPU_ERROR("D3D12RenderTargetCache: Unknown {} render target format {}",
                 key.is_depth ? "depth" : "color", uint32_t(key.resource_format));
    return nullptr;
  }
  if (key.msaa_samples == xenos::MsaaSamples::k2X && !msaa_2x_supported()) {
    resource_desc.SampleDesc.Count = 4;
  } else {
    resource_desc.SampleDesc.Count = UINT(1) << UINT(key.msaa_samples);
  }
  resource_desc.SampleDesc.Quality = 0;
  resource_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  resource_desc.Flags = key.is_depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                                     : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
  // The first access will be ownership transfer into this render target or
  // starting to draw directly.
  D3D12_RESOURCE_STATES resource_state =
      key.is_depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_RENDER_TARGET;
  D3D12_CLEAR_VALUE optimized_clear_value;
  if (key.is_depth) {
    optimized_clear_value.Format = GetDepthDSVDXGIFormat(key.GetDepthFormat());
    // Fixed-point depth is generally direct (1 being the farthest),
    // floating-point is used for more uniform precision across the range (0
    // being the farthest).
    optimized_clear_value.DepthStencil.Depth =
        key.GetDepthFormat() == xenos::DepthRenderTargetFormat::kD24S8 ? 1.0f : 0.0f;
    optimized_clear_value.DepthStencil.Stencil = 0;
  } else {
    optimized_clear_value.Format = GetColorDrawDXGIFormat(key.GetColorFormat());
    optimized_clear_value.Color[0] = 0.0f;
    optimized_clear_value.Color[1] = 0.0f;
    optimized_clear_value.Color[2] = 0.0f;
    optimized_clear_value.Color[3] = 0.0f;
  }
  // Create zeroed for more determinism, primarily with respect to compression
  // and depth float24 / float32 mirroring.
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             D3D12_HEAP_FLAG_NONE, &resource_desc, resource_state,
                                             &optimized_clear_value, IID_PPV_ARGS(&resource)))) {
    return nullptr;
  }
  {
    std::u16string resource_name = rex::string::to_utf16(key.GetDebugName());
    resource->SetName(reinterpret_cast<LPCWSTR>(resource_name.c_str()));
  }

  ui::d3d12::D3D12CpuDescriptorPool& descriptor_pool =
      key.is_depth ? *descriptor_pool_depth_ : *descriptor_pool_color_;
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_draw =
      descriptor_pool.AllocateDescriptor();
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_srv =
      descriptor_pool_srv_->AllocateDescriptor();
  if (!descriptor_draw.IsValid() || !descriptor_srv.IsValid()) {
    return nullptr;
  }
  D3D12_CPU_DESCRIPTOR_HANDLE descriptor_draw_handle = descriptor_draw.GetHandle();
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_load_separate;
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_srv_stencil;
  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
  srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  if (resource_desc.SampleDesc.Count > 1) {
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
  } else {
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.PlaneSlice = 0;
    srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
  }
  if (key.is_depth) {
    // DSV and stencil SRV.
    descriptor_srv_stencil = descriptor_pool_srv_->AllocateDescriptor();
    if (!descriptor_srv_stencil.IsValid()) {
      return nullptr;
    }
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc;
    dsv_desc.Format = optimized_clear_value.Format;
    dsv_desc.Flags = D3D12_DSV_FLAG_NONE;
    D3D12_SHADER_RESOURCE_VIEW_DESC stencil_srv_desc;
    stencil_srv_desc.Format = GetDepthSRVStencilDXGIFormat(key.GetDepthFormat());
    stencil_srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (resource_desc.SampleDesc.Count > 1) {
      dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
      stencil_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
    } else {
      dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
      dsv_desc.Texture2D.MipSlice = 0;
      stencil_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      stencil_srv_desc.Texture2D.MostDetailedMip = 0;
      stencil_srv_desc.Texture2D.MipLevels = 1;
      stencil_srv_desc.Texture2D.PlaneSlice = 1;
      stencil_srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
    }
    device->CreateDepthStencilView(resource.Get(), &dsv_desc, descriptor_draw_handle);
    device->CreateShaderResourceView(resource.Get(), &stencil_srv_desc,
                                     descriptor_srv_stencil.GetHandle());
    // Depth SRV.
    srv_desc.Format = GetDepthSRVDepthDXGIFormat(key.GetDepthFormat());
  } else {
    // Drawing RTV.
    D3D12_RENDER_TARGET_VIEW_DESC rtv_desc;
    rtv_desc.Format = optimized_clear_value.Format;
    if (resource_desc.SampleDesc.Count > 1) {
      rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
    } else {
      rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
      rtv_desc.Texture2D.MipSlice = 0;
      rtv_desc.Texture2D.PlaneSlice = 0;
    }
    device->CreateRenderTargetView(resource.Get(), &rtv_desc, descriptor_draw_handle);
    // Ownership transfer RTV.
    DXGI_FORMAT load_format = GetColorOwnershipTransferDXGIFormat(key.GetColorFormat());
    if (rtv_desc.Format != load_format) {
      descriptor_load_separate = descriptor_pool.AllocateDescriptor();
      if (!descriptor_load_separate.IsValid()) {
        return nullptr;
      }
      rtv_desc.Format = load_format;
      device->CreateRenderTargetView(resource.Get(), &rtv_desc,
                                     descriptor_load_separate.GetHandle());
    }
    // SRV for ownership transfer and dumping.
    srv_desc.Format = load_format;
  }
  device->CreateShaderResourceView(resource.Get(), &srv_desc, descriptor_srv.GetHandle());

  return new D3D12RenderTarget(key, resource.Get(), std::move(descriptor_draw),
                               std::move(descriptor_load_separate), std::move(descriptor_srv),
                               std::move(descriptor_srv_stencil), resource_state);
}

bool D3D12RenderTargetCache::IsHostDepthEncodingDifferent(
    xenos::DepthRenderTargetFormat format) const {
  if (format == xenos::DepthRenderTargetFormat::kD24FS8) {
    return !depth_float24_convert_in_pixel_shader_;
  }
  return false;
}

bool D3D12RenderTargetCache::IsGammaFormatHostStorageSeparate() const {
  return gamma_render_target_as_unorm16_;
}

void D3D12RenderTargetCache::TransitionEdramBuffer(D3D12_RESOURCE_STATES new_state) {
  if (command_processor_.PushTransitionBarrier(edram_buffer_, edram_buffer_state_, new_state)) {
    // Resetting edram_buffer_modification_status_ only if the barrier has been
    // truly inserted - in particular, not resetting it for UAV > UAV as
    // barriers are dropped if the state hasn't been changed.
    edram_buffer_modification_status_ = EdramBufferModificationStatus::kUnmodified;
  }
  edram_buffer_state_ = new_state;
}

void D3D12RenderTargetCache::MarkEdramBufferModified(
    EdramBufferModificationStatus modification_status) {
  assert_true(modification_status != EdramBufferModificationStatus::kUnmodified);
  assert_true(edram_buffer_state_ == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  if (edram_buffer_state_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
    return;
  }
  // max because being modified as a UAV requires stricter synchronization than
  // as ROV.
  edram_buffer_modification_status_ =
      std::max(edram_buffer_modification_status_, modification_status);
}

void D3D12RenderTargetCache::CommitEdramBufferUAVWrites(
    EdramBufferModificationStatus commit_status) {
  assert_true(commit_status != EdramBufferModificationStatus::kUnmodified);
  if (edram_buffer_modification_status_ < commit_status) {
    return;
  }
  assert_true(edram_buffer_state_ == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  if (edram_buffer_state_ == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
    command_processor_.PushUAVBarrier(edram_buffer_);
  }
  edram_buffer_modification_status_ = EdramBufferModificationStatus::kUnmodified;
  PixelShaderInterlockFullEdramBarrierPlaced();
}

ID3D12PipelineState* const* D3D12RenderTargetCache::GetOrCreateTransferPipelines(
    TransferShaderKey key) {
  const TransferModeInfo& mode = kTransferModes[size_t(key.mode)];
  bool dest_is_stencil_bit = (mode.output == TransferOutput::kStencilBit);

  if (dest_is_stencil_bit) {
    auto pipelines_it = transfer_stencil_bit_pipelines_.find(key);
    if (pipelines_it != transfer_stencil_bit_pipelines_.end()) {
      return pipelines_it->second[0] ? pipelines_it->second.data() : nullptr;
    }
  } else {
    auto pipeline_it = transfer_pipelines_.find(key);
    if (pipeline_it != transfer_pipelines_.end()) {
      return pipeline_it->second ? &pipeline_it->second : nullptr;
    }
  }

  uint32_t rs = kTransferUsedRootParameters[size_t(use_stencil_reference_output_
                                                       ? mode.root_signature_with_stencil_ref
                                                       : mode.root_signature_no_stencil_ref)];

  // If not dest_is_color, it's depth, or stencil bit - 40-sample columns are
  // swapped as opposed to color source.
  bool dest_is_color = (mode.output == TransferOutput::kColor);

  xenos::ColorRenderTargetFormat dest_color_format =
      xenos::ColorRenderTargetFormat(key.dest_resource_format);
  xenos::DepthRenderTargetFormat dest_depth_format =
      xenos::DepthRenderTargetFormat(key.dest_resource_format);
  bool dest_is_64bpp = dest_is_color && xenos::IsColorRenderTargetFormat64bpp(dest_color_format);

  xenos::ColorRenderTargetFormat source_color_format =
      xenos::ColorRenderTargetFormat(key.source_resource_format);
  xenos::DepthRenderTargetFormat source_depth_format =
      xenos::DepthRenderTargetFormat(key.source_resource_format);
  // If not source_is_color, it's depth / stencil - 40-sample columns are
  // swapped as opposed to color destination.
  bool source_is_color = (rs & kTransferUsedRootParameterColorSRVBit) != 0;
  bool source_is_64bpp;
  uint32_t source_color_format_component_count;
  uint32_t source_color_srv_component_mask;
  bool source_color_is_uint;
  if (source_is_color) {
    assert_zero(rs & kTransferUsedRootParameterDepthSRVBit);
    assert_zero(rs & kTransferUsedRootParameterStencilSRVBit);
    source_is_64bpp = xenos::IsColorRenderTargetFormat64bpp(source_color_format);
    source_color_format_component_count =
        xenos::GetColorRenderTargetFormatComponentCount(source_color_format);
    if (dest_is_stencil_bit) {
      if (source_is_64bpp && !dest_is_64bpp) {
        // Need one component, but choosing from the two 32bpp halves of the
        // 64bpp sample.
        source_color_srv_component_mask = 0b1 | (0b1 << (source_color_format_component_count >> 1));
      } else {
        // Red is at least 8 bits per component in all formats.
        source_color_srv_component_mask = 0b1;
      }
    } else {
      source_color_srv_component_mask = (uint32_t(1) << source_color_format_component_count) - 1;
    }
    GetColorOwnershipTransferDXGIFormat(source_color_format, &source_color_is_uint);
  } else {
    source_is_64bpp = false;
    source_color_format_component_count = 0;
    source_color_srv_component_mask = 0;
    source_color_is_uint = false;
  }

  bool shader_uses_stencil_reference_output =
      mode.output == TransferOutput::kDepth && use_stencil_reference_output_;

  // Because of built_shader_.resize(), pointers can't be kept persistently
  // here! Resizing also zeroes the memory.

  built_shader_.clear();

  // RDEF, ISGN, OSGN, SHEX, optionally SFI0, STAT.
  uint32_t blob_count = 5 + uint32_t(shader_uses_stencil_reference_output);

  // Allocate space for the container header and the blob offsets.
  built_shader_.resize(sizeof(dxbc::ContainerHeader) / sizeof(uint32_t) + blob_count);
  uint32_t blob_offset_position_dwords = sizeof(dxbc::ContainerHeader) / sizeof(uint32_t);
  uint32_t blob_position_dwords = uint32_t(built_shader_.size());
  constexpr uint32_t kBlobHeaderSizeDwords = sizeof(dxbc::BlobHeader) / sizeof(uint32_t);

  uint32_t name_ptr;

  // ***************************************************************************
  // Resource definition
  // ***************************************************************************

  built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
  uint32_t rdef_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
  // Not needed, as the next operation done is resize, to allocate the space for
  // both the blob header and the resource definition header.
  // built_shader_.resize(rdef_position_dwords);

  // Allocate space for the RDEF header.
  built_shader_.resize(rdef_position_dwords + sizeof(dxbc::RdefHeader) / sizeof(uint32_t));
  // Generator name.
  dxbc::AppendAlignedString(built_shader_, "Xenia");

  // Constant types - uint (aka "dword" when it's scalar) only.
  // Names.
  name_ptr = uint32_t((built_shader_.size() - rdef_position_dwords) * sizeof(uint32_t));
  uint32_t rdef_dword_name_ptr = name_ptr;
  name_ptr += dxbc::AppendAlignedString(built_shader_, "dword");
  // Types.
  uint32_t rdef_type_uint_position_dwords = uint32_t(built_shader_.size());
  uint32_t rdef_type_uint_ptr =
      uint32_t((rdef_type_uint_position_dwords - rdef_position_dwords) * sizeof(uint32_t));
  built_shader_.resize(rdef_type_uint_position_dwords + sizeof(dxbc::RdefType) / sizeof(uint32_t));
  {
    auto& rdef_type_uint =
        *reinterpret_cast<dxbc::RdefType*>(built_shader_.data() + rdef_type_uint_position_dwords);
    rdef_type_uint.variable_class = dxbc::RdefVariableClass::kScalar;
    rdef_type_uint.variable_type = dxbc::RdefVariableType::kUInt;
    rdef_type_uint.row_count = 1;
    rdef_type_uint.column_count = 1;
    rdef_type_uint.name_ptr = rdef_dword_name_ptr;
  }

  // Constants, if needed:
  // - uint xe_transfer_stencil_mask
  // - uint xe_transfer_address
  // - uint xe_transfer_host_depth_address
  uint32_t rdef_constant_count = 0;
  uint32_t rdef_constant_index_stencil_mask =
      (rs & kTransferUsedRootParameterStencilMaskConstantBit) ? rdef_constant_count++ : UINT32_MAX;
  assert_false(dest_is_stencil_bit && rdef_constant_index_stencil_mask == UINT32_MAX);
  uint32_t rdef_constant_index_address =
      (rs & kTransferUsedRootParameterAddressConstantBit) ? rdef_constant_count++ : UINT32_MAX;
  assert_true(rdef_constant_index_address != UINT32_MAX);
  uint32_t rdef_constant_index_host_depth_address =
      (rs & kTransferUsedRootParameterHostDepthAddressConstantBit) ? rdef_constant_count++
                                                                   : UINT32_MAX;
  // Names.
  name_ptr = uint32_t((built_shader_.size() - rdef_position_dwords) * sizeof(uint32_t));
  uint32_t rdef_xe_transfer_stencil_mask_name_ptr = name_ptr;
  if (rdef_constant_index_stencil_mask != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_transfer_stencil_mask");
  }
  uint32_t rdef_xe_transfer_address_name_ptr = name_ptr;
  if (rdef_constant_index_address != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_transfer_address");
  }
  uint32_t rdef_xe_transfer_host_depth_address_name_ptr = name_ptr;
  if (rdef_constant_index_host_depth_address != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_transfer_host_depth_address");
  }
  // Constants.
  uint32_t rdef_constants_position_dwords = uint32_t(built_shader_.size());
  uint32_t rdef_constants_ptr =
      uint32_t((rdef_constants_position_dwords - rdef_position_dwords) * sizeof(uint32_t));
  built_shader_.resize(rdef_constants_position_dwords +
                       sizeof(dxbc::RdefVariable) / sizeof(uint32_t) * rdef_constant_count);
  {
    auto rdef_constants = reinterpret_cast<dxbc::RdefVariable*>(built_shader_.data() +
                                                                rdef_constants_position_dwords);
    // uint xe_transfer_stencil_mask
    if (rdef_constant_index_stencil_mask != UINT32_MAX) {
      dxbc::RdefVariable& rdef_constant_stencil_mask =
          rdef_constants[rdef_constant_index_stencil_mask];
      rdef_constant_stencil_mask.name_ptr = rdef_xe_transfer_stencil_mask_name_ptr;
      rdef_constant_stencil_mask.size_bytes = sizeof(uint32_t);
      rdef_constant_stencil_mask.flags = dxbc::kRdefVariableFlagUsed;
      rdef_constant_stencil_mask.type_ptr = rdef_type_uint_ptr;
      rdef_constant_stencil_mask.start_texture = UINT32_MAX;
      rdef_constant_stencil_mask.start_sampler = UINT32_MAX;
    }
    // uint xe_transfer_address
    if (rdef_constant_index_address != UINT32_MAX) {
      dxbc::RdefVariable& rdef_constant_address = rdef_constants[rdef_constant_index_address];
      rdef_constant_address.name_ptr = rdef_xe_transfer_address_name_ptr;
      rdef_constant_address.size_bytes = sizeof(uint32_t);
      rdef_constant_address.flags = dxbc::kRdefVariableFlagUsed;
      rdef_constant_address.type_ptr = rdef_type_uint_ptr;
      rdef_constant_address.start_texture = UINT32_MAX;
      rdef_constant_address.start_sampler = UINT32_MAX;
    }
    // uint xe_transfer_host_depth_address
    if (rdef_constant_index_host_depth_address != UINT32_MAX) {
      dxbc::RdefVariable& rdef_constant_host_depth_address =
          rdef_constants[rdef_constant_index_host_depth_address];
      rdef_constant_host_depth_address.name_ptr = rdef_xe_transfer_host_depth_address_name_ptr;
      rdef_constant_host_depth_address.size_bytes = sizeof(uint32_t);
      rdef_constant_host_depth_address.flags = dxbc::kRdefVariableFlagUsed;
      rdef_constant_host_depth_address.type_ptr = rdef_type_uint_ptr;
      rdef_constant_host_depth_address.start_texture = UINT32_MAX;
      rdef_constant_host_depth_address.start_sampler = UINT32_MAX;
    }
  }

  // Constant buffers, if needed:
  // - xe_transfer_stencil_mask { uint xe_transfer_stencil_mask; }
  // - xe_transfer_address { uint xe_transfer_address; }
  // - xe_transfer_host_depth_address { uint xe_transfer_host_depth_address; }
  // Reusing the constant names for constant buffers.
  uint32_t rdef_cbuffer_count = 0;
  uint32_t cbuffer_index_stencil_mask =
      rdef_constant_index_stencil_mask != UINT32_MAX ? rdef_cbuffer_count++ : UINT32_MAX;
  uint32_t cbuffer_index_address =
      rdef_constant_index_address != UINT32_MAX ? rdef_cbuffer_count++ : UINT32_MAX;
  uint32_t cbuffer_index_host_depth_address =
      rdef_constant_index_host_depth_address != UINT32_MAX ? rdef_cbuffer_count++ : UINT32_MAX;
  uint32_t rdef_cbuffer_position_dwords = uint32_t(built_shader_.size());
  built_shader_.resize(rdef_cbuffer_position_dwords +
                       sizeof(dxbc::RdefCbuffer) / sizeof(uint32_t) * rdef_cbuffer_count);
  {
    auto rdef_cbuffers =
        reinterpret_cast<dxbc::RdefCbuffer*>(built_shader_.data() + rdef_cbuffer_position_dwords);
    // xe_transfer_stencil_mask
    if (cbuffer_index_stencil_mask != UINT32_MAX) {
      dxbc::RdefCbuffer& rdef_cbuffer_stencil_mask = rdef_cbuffers[cbuffer_index_stencil_mask];
      rdef_cbuffer_stencil_mask.name_ptr = rdef_xe_transfer_stencil_mask_name_ptr;
      rdef_cbuffer_stencil_mask.variable_count = 1;
      rdef_cbuffer_stencil_mask.variables_ptr = uint32_t(
          rdef_constants_ptr + sizeof(dxbc::RdefVariable) * rdef_constant_index_stencil_mask);
      rdef_cbuffer_stencil_mask.size_vector_aligned_bytes = sizeof(uint32_t) * 4;
    }
    // xe_transfer_address
    if (cbuffer_index_address != UINT32_MAX) {
      dxbc::RdefCbuffer& rdef_cbuffer_address = rdef_cbuffers[cbuffer_index_address];
      rdef_cbuffer_address.name_ptr = rdef_xe_transfer_address_name_ptr;
      rdef_cbuffer_address.variable_count = 1;
      rdef_cbuffer_address.variables_ptr =
          uint32_t(rdef_constants_ptr + sizeof(dxbc::RdefVariable) * rdef_constant_index_address);
      rdef_cbuffer_address.size_vector_aligned_bytes = sizeof(uint32_t) * 4;
    }
    // xe_transfer_host_depth_address
    if (cbuffer_index_host_depth_address != UINT32_MAX) {
      dxbc::RdefCbuffer& rdef_cbuffer_host_depth_address =
          rdef_cbuffers[cbuffer_index_host_depth_address];
      rdef_cbuffer_host_depth_address.name_ptr = rdef_xe_transfer_host_depth_address_name_ptr;
      rdef_cbuffer_host_depth_address.variable_count = 1;
      rdef_cbuffer_host_depth_address.variables_ptr = uint32_t(
          rdef_constants_ptr + sizeof(dxbc::RdefVariable) * rdef_constant_index_host_depth_address);
      rdef_cbuffer_host_depth_address.size_vector_aligned_bytes = sizeof(uint32_t) * 4;
    }
  }

  // Bindings.
  // - Texture2D/Texture2DMS<floatN/uintN> xe_transfer_color
  // - Texture2D/Texture2DMS<float> xe_transfer_depth
  // - Texture2D/Texture2DMS<uint2> xe_transfer_stencil
  // - Texture2D<float>/Texture2DMS<float>/Buffer<uint> xe_transfer_host_depth
  // - Constant buffers
  uint32_t rdef_srv_count = 0;
  uint32_t srv_index_color =
      (rs & kTransferUsedRootParameterColorSRVBit) ? rdef_srv_count++ : UINT32_MAX;
  uint32_t srv_index_depth =
      (rs & kTransferUsedRootParameterDepthSRVBit) ? rdef_srv_count++ : UINT32_MAX;
  uint32_t srv_index_stencil =
      (rs & kTransferUsedRootParameterStencilSRVBit) ? rdef_srv_count++ : UINT32_MAX;
  uint32_t srv_index_host_depth =
      (rs & kTransferUsedRootParameterHostDepthSRVBit) ? rdef_srv_count++ : UINT32_MAX;
  uint32_t rdef_binding_count = rdef_srv_count + rdef_cbuffer_count;
  // Names.
  name_ptr = uint32_t((built_shader_.size() - rdef_position_dwords) * sizeof(uint32_t));
  uint32_t rdef_xe_transfer_color_name_ptr = name_ptr;
  if (srv_index_color != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_transfer_color");
  }
  uint32_t rdef_xe_transfer_depth_name_ptr = name_ptr;
  if (srv_index_depth != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_transfer_depth");
  }
  uint32_t rdef_xe_transfer_stencil_name_ptr = name_ptr;
  if (srv_index_stencil != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_transfer_stencil");
  }
  uint32_t rdef_xe_transfer_host_depth_name_ptr = name_ptr;
  if (srv_index_host_depth != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_transfer_host_depth");
  }
  // Bindings.
  uint32_t rdef_binding_position_dwords = uint32_t(built_shader_.size());
  built_shader_.resize(rdef_binding_position_dwords +
                       sizeof(dxbc::RdefInputBind) / sizeof(uint32_t) * rdef_binding_count);
  {
    auto rdef_bindings =
        reinterpret_cast<dxbc::RdefInputBind*>(built_shader_.data() + rdef_binding_position_dwords);
    uint32_t rdef_binding_index = 0;
    // xe_transfer_color
    if (srv_index_color != UINT32_MAX) {
      dxbc::RdefInputBind& rdef_binding_color = rdef_bindings[rdef_binding_index++];
      rdef_binding_color.name_ptr = rdef_xe_transfer_color_name_ptr;
      rdef_binding_color.type = dxbc::RdefInputType::kTexture;
      rdef_binding_color.return_type =
          source_color_is_uint ? dxbc::ResourceReturnType::kUInt : dxbc::ResourceReturnType::kFloat;
      if (key.source_msaa_samples != xenos::MsaaSamples::k1X) {
        rdef_binding_color.dimension = dxbc::RdefDimension::kSRVTexture2DMS;
      } else {
        rdef_binding_color.dimension = dxbc::RdefDimension::kSRVTexture2D;
        rdef_binding_color.sample_count = UINT32_MAX;
      }
      rdef_binding_color.bind_point = kTransferSRVRegisterColor;
      rdef_binding_color.bind_count = 1;
      assert_not_zero(source_color_srv_component_mask);
      rdef_binding_color.flags = (32 - rex::lzcnt(source_color_srv_component_mask) - 1)
                                 << dxbc::kRdefInputFlagsComponentsShift;
      rdef_binding_color.id = srv_index_color;
    }
    // xe_transfer_depth
    if (srv_index_depth != UINT32_MAX) {
      dxbc::RdefInputBind& rdef_binding_depth = rdef_bindings[rdef_binding_index++];
      rdef_binding_depth.name_ptr = rdef_xe_transfer_depth_name_ptr;
      rdef_binding_depth.type = dxbc::RdefInputType::kTexture;
      rdef_binding_depth.return_type = dxbc::ResourceReturnType::kFloat;
      if (key.source_msaa_samples != xenos::MsaaSamples::k1X) {
        rdef_binding_depth.dimension = dxbc::RdefDimension::kSRVTexture2DMS;
      } else {
        rdef_binding_depth.dimension = dxbc::RdefDimension::kSRVTexture2D;
        rdef_binding_depth.sample_count = UINT32_MAX;
      }
      rdef_binding_depth.bind_point = kTransferSRVRegisterDepth;
      rdef_binding_depth.bind_count = 1;
      rdef_binding_depth.id = srv_index_depth;
    }
    // xe_transfer_stencil
    if (srv_index_stencil != UINT32_MAX) {
      dxbc::RdefInputBind& rdef_binding_stencil = rdef_bindings[rdef_binding_index++];
      rdef_binding_stencil.name_ptr = rdef_xe_transfer_stencil_name_ptr;
      rdef_binding_stencil.type = dxbc::RdefInputType::kTexture;
      rdef_binding_stencil.return_type = dxbc::ResourceReturnType::kUInt;
      if (key.source_msaa_samples != xenos::MsaaSamples::k1X) {
        rdef_binding_stencil.dimension = dxbc::RdefDimension::kSRVTexture2DMS;
      } else {
        rdef_binding_stencil.dimension = dxbc::RdefDimension::kSRVTexture2D;
        rdef_binding_stencil.sample_count = UINT32_MAX;
      }
      rdef_binding_stencil.bind_point = kTransferSRVRegisterStencil;
      rdef_binding_stencil.bind_count = 1;
      rdef_binding_stencil.flags = dxbc::kRdefInputFlags2Component;
      rdef_binding_stencil.id = srv_index_stencil;
    }
    // xe_transfer_host_depth
    if (srv_index_host_depth != UINT32_MAX) {
      dxbc::RdefInputBind& rdef_binding_host_depth = rdef_bindings[rdef_binding_index++];
      rdef_binding_host_depth.name_ptr = rdef_xe_transfer_host_depth_name_ptr;
      rdef_binding_host_depth.type = dxbc::RdefInputType::kTexture;
      if (key.host_depth_source_is_copy) {
        // Float as uint.
        rdef_binding_host_depth.return_type = dxbc::ResourceReturnType::kUInt;
        rdef_binding_host_depth.dimension = dxbc::RdefDimension::kSRVBuffer;
        rdef_binding_host_depth.sample_count = UINT32_MAX;
      } else {
        rdef_binding_host_depth.return_type = dxbc::ResourceReturnType::kFloat;
        if (key.host_depth_source_msaa_samples != xenos::MsaaSamples::k1X) {
          rdef_binding_host_depth.dimension = dxbc::RdefDimension::kSRVTexture2DMS;
        } else {
          rdef_binding_host_depth.dimension = dxbc::RdefDimension::kSRVTexture2D;
          rdef_binding_host_depth.sample_count = UINT32_MAX;
        }
      }
      rdef_binding_host_depth.bind_point = kTransferSRVRegisterHostDepth;
      rdef_binding_host_depth.bind_count = 1;
      rdef_binding_host_depth.id = srv_index_host_depth;
    }
    // xe_transfer_stencil_mask
    if (cbuffer_index_stencil_mask != UINT32_MAX) {
      dxbc::RdefInputBind& rdef_binding_stencil_mask = rdef_bindings[rdef_binding_index++];
      rdef_binding_stencil_mask.name_ptr = rdef_xe_transfer_stencil_mask_name_ptr;
      rdef_binding_stencil_mask.type = dxbc::RdefInputType::kCbuffer;
      rdef_binding_stencil_mask.bind_point = kTransferCBVRegisterStencilMask;
      rdef_binding_stencil_mask.bind_count = 1;
      rdef_binding_stencil_mask.flags = dxbc::kRdefInputFlagUserPacked;
      rdef_binding_stencil_mask.id = cbuffer_index_stencil_mask;
    }
    // xe_transfer_address
    if (cbuffer_index_address != UINT32_MAX) {
      dxbc::RdefInputBind& rdef_binding_address = rdef_bindings[rdef_binding_index++];
      rdef_binding_address.name_ptr = rdef_xe_transfer_address_name_ptr;
      rdef_binding_address.type = dxbc::RdefInputType::kCbuffer;
      rdef_binding_address.bind_point = kTransferCBVRegisterAddress;
      rdef_binding_address.bind_count = 1;
      rdef_binding_address.flags = dxbc::kRdefInputFlagUserPacked;
      rdef_binding_address.id = cbuffer_index_address;
    }
    // xe_transfer_host_depth_address
    if (cbuffer_index_host_depth_address != UINT32_MAX) {
      dxbc::RdefInputBind& rdef_binding_host_depth_address = rdef_bindings[rdef_binding_index++];
      rdef_binding_host_depth_address.name_ptr = rdef_xe_transfer_host_depth_address_name_ptr;
      rdef_binding_host_depth_address.type = dxbc::RdefInputType::kCbuffer;
      rdef_binding_host_depth_address.bind_point = kTransferCBVRegisterHostDepthAddress;
      rdef_binding_host_depth_address.bind_count = 1;
      rdef_binding_host_depth_address.flags = dxbc::kRdefInputFlagUserPacked;
      rdef_binding_host_depth_address.id = cbuffer_index_host_depth_address;
    }
  }

  // Header.
  {
    auto& rdef_header =
        *reinterpret_cast<dxbc::RdefHeader*>(built_shader_.data() + rdef_position_dwords);
    rdef_header.cbuffer_count = rdef_cbuffer_count;
    rdef_header.cbuffers_ptr =
        uint32_t((rdef_cbuffer_position_dwords - rdef_position_dwords) * sizeof(uint32_t));
    rdef_header.input_bind_count = rdef_binding_count;
    rdef_header.input_binds_ptr =
        uint32_t((rdef_binding_position_dwords - rdef_position_dwords) * sizeof(uint32_t));
    rdef_header.shader_model = dxbc::RdefShaderModel::kPixelShader5_1;
    rdef_header.compile_flags =
        dxbc::kCompileFlagNoPreshader | dxbc::kCompileFlagPreferFlowControl |
        dxbc::kCompileFlagIeeeStrictness | dxbc::kCompileFlagAllResourcesBound;
    // Generator name is right after the header.
    rdef_header.generator_name_ptr = sizeof(dxbc::RdefHeader);
    rdef_header.fourcc = dxbc::RdefHeader::FourCC::k5_1;
    rdef_header.InitializeSizes();
  }

  {
    auto& blob_header =
        *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
    blob_header.fourcc = dxbc::BlobHeader::FourCC::kResourceDefinition;
    blob_position_dwords = uint32_t(built_shader_.size());
    blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                             built_shader_[blob_offset_position_dwords++];
  }

  // ***************************************************************************
  // Input signature
  // ***************************************************************************

  // Registers for accessing in the shader code - multiple inputs may be packed
  // into the same register.
  enum InputRegister : uint32_t {
    kInputRegisterPosition,
    kInputRegisterSampleIndex,
    kInputRegisterCount,
  };

  // Position, and for multisampled, sample index.
  uint32_t isgn_parameter_count = 1 + uint32_t(key.dest_msaa_samples != xenos::MsaaSamples::k1X);

  // Reserve space for the header and the parameters.
  built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
  uint32_t isgn_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
  built_shader_.resize(isgn_position_dwords + sizeof(dxbc::Signature) / sizeof(uint32_t) +
                       sizeof(dxbc::SignatureParameter) / sizeof(uint32_t) * isgn_parameter_count);

  // Names (after the parameters).
  name_ptr = uint32_t((built_shader_.size() - isgn_position_dwords) * sizeof(uint32_t));
  uint32_t isgn_sv_position_name_ptr = name_ptr;
  name_ptr += dxbc::AppendAlignedString(built_shader_, "SV_Position");
  uint32_t isgn_sv_sample_index_name_ptr = name_ptr;
  if (key.dest_msaa_samples != xenos::MsaaSamples::k1X) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "SV_SampleIndex");
  }

  // Header and parameters.
  {
    // Header.
    auto& isgn_header =
        *reinterpret_cast<dxbc::Signature*>(built_shader_.data() + isgn_position_dwords);
    isgn_header.parameter_count = isgn_parameter_count;
    isgn_header.parameter_info_ptr = sizeof(dxbc::Signature);
    // Parameters.
    auto isgn_parameters = reinterpret_cast<dxbc::SignatureParameter*>(
        built_shader_.data() + isgn_position_dwords + sizeof(dxbc::Signature) / sizeof(uint32_t));
    uint32_t isgn_parameter_index = 0;
    // SV_Position.xy
    dxbc::SignatureParameter& isgn_sv_position = isgn_parameters[isgn_parameter_index++];
    isgn_sv_position.semantic_name_ptr = isgn_sv_position_name_ptr;
    isgn_sv_position.system_value = dxbc::Name::kPosition;
    isgn_sv_position.component_type = dxbc::SignatureRegisterComponentType::kFloat32;
    isgn_sv_position.register_index = kInputRegisterPosition;
    isgn_sv_position.mask = 0b1111;
    isgn_sv_position.always_reads_mask = 0b0011;
    // SV_SampleIndex
    if (key.dest_msaa_samples != xenos::MsaaSamples::k1X) {
      dxbc::SignatureParameter& isgn_sv_sample_index = isgn_parameters[isgn_parameter_index++];
      isgn_sv_sample_index.semantic_name_ptr = isgn_sv_sample_index_name_ptr;
      isgn_sv_sample_index.system_value = dxbc::Name::kSampleIndex;
      isgn_sv_sample_index.component_type = dxbc::SignatureRegisterComponentType::kUInt32;
      isgn_sv_sample_index.register_index = kInputRegisterSampleIndex;
      isgn_sv_sample_index.mask = 0b0001;
      isgn_sv_sample_index.always_reads_mask = 0b0001;
    }
  }

  {
    auto& blob_header =
        *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
    blob_header.fourcc = dxbc::BlobHeader::FourCC::kInputSignature;
    blob_position_dwords = uint32_t(built_shader_.size());
    blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                             built_shader_[blob_offset_position_dwords++];
  }

  // ***************************************************************************
  // Output signature
  // ***************************************************************************

  // Color or depth.
  uint32_t osgn_parameter_count = 0;
  uint32_t osgn_parameter_index_sv_target =
      mode.output == TransferOutput::kColor ? osgn_parameter_count++ : UINT32_MAX;
  uint32_t osgn_parameter_index_sv_depth =
      mode.output == TransferOutput::kDepth ? osgn_parameter_count++ : UINT32_MAX;
  uint32_t osgn_parameter_index_sv_stencil_ref =
      shader_uses_stencil_reference_output ? osgn_parameter_count++ : UINT32_MAX;

  // Reserve space for the header and the parameters.
  built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
  uint32_t osgn_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
  built_shader_.resize(osgn_position_dwords + sizeof(dxbc::Signature) / sizeof(uint32_t) +
                       sizeof(dxbc::SignatureParameter) / sizeof(uint32_t) * osgn_parameter_count);

  // Names (after the parameters).
  name_ptr = uint32_t((built_shader_.size() - osgn_position_dwords) * sizeof(uint32_t));
  uint32_t osgn_sv_target_name_ptr = name_ptr;
  if (osgn_parameter_index_sv_target != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "SV_Target");
  }
  uint32_t osgn_sv_depth_name_ptr = name_ptr;
  if (osgn_parameter_index_sv_depth != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "SV_Depth");
  }
  uint32_t osgn_sv_stencil_ref_name_ptr = name_ptr;
  if (osgn_parameter_index_sv_stencil_ref != UINT32_MAX) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "SV_StencilRef");
  }

  bool dest_color_is_uint;
  if (mode.output == TransferOutput::kColor) {
    GetColorOwnershipTransferDXGIFormat(dest_color_format, &dest_color_is_uint);
  } else {
    dest_color_is_uint = false;
  }

  // Header and parameters.
  {
    // Header.
    auto& osgn_header =
        *reinterpret_cast<dxbc::Signature*>(built_shader_.data() + osgn_position_dwords);
    osgn_header.parameter_count = osgn_parameter_count;
    osgn_header.parameter_info_ptr = sizeof(dxbc::Signature);
    // Parameters.
    auto osgn_parameters = reinterpret_cast<dxbc::SignatureParameter*>(
        built_shader_.data() + osgn_position_dwords + sizeof(dxbc::Signature) / sizeof(uint32_t));
    // SV_Target
    if (osgn_parameter_index_sv_target != UINT32_MAX) {
      dxbc::SignatureParameter& osgn_sv_target = osgn_parameters[osgn_parameter_index_sv_target];
      osgn_sv_target.semantic_name_ptr = osgn_sv_target_name_ptr;
      osgn_sv_target.component_type = dest_color_is_uint
                                          ? dxbc::SignatureRegisterComponentType::kUInt32
                                          : dxbc::SignatureRegisterComponentType::kFloat32;
      osgn_sv_target.register_index = 0;
      osgn_sv_target.mask = 0b1111;
    }
    // SV_Depth
    if (osgn_parameter_index_sv_depth != UINT32_MAX) {
      dxbc::SignatureParameter& osgn_sv_depth = osgn_parameters[osgn_parameter_index_sv_depth];
      osgn_sv_depth.semantic_name_ptr = osgn_sv_depth_name_ptr;
      osgn_sv_depth.component_type = dxbc::SignatureRegisterComponentType::kFloat32;
      osgn_sv_depth.register_index = UINT32_MAX;
      osgn_sv_depth.mask = 0b0001;
      osgn_sv_depth.never_writes_mask = 0b1110;
    }
    // SV_StencilRef
    if (osgn_parameter_index_sv_stencil_ref != UINT32_MAX) {
      dxbc::SignatureParameter& osgn_sv_stencil_ref =
          osgn_parameters[osgn_parameter_index_sv_stencil_ref];
      osgn_sv_stencil_ref.semantic_name_ptr = osgn_sv_stencil_ref_name_ptr;
      // Older versions of FXC incorrectly expect SV_StencilRef to be float,
      // it's always uint in DXC and also in the latest versions of FXC.
      osgn_sv_stencil_ref.component_type = dxbc::SignatureRegisterComponentType::kUInt32;
      osgn_sv_stencil_ref.register_index = UINT32_MAX;
      osgn_sv_stencil_ref.mask = 0b0001;
      osgn_sv_stencil_ref.never_writes_mask = 0b1110;
    }
  }

  {
    auto& blob_header =
        *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
    blob_header.fourcc = dxbc::BlobHeader::FourCC::kOutputSignature;
    blob_position_dwords = uint32_t(built_shader_.size());
    blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                             built_shader_[blob_offset_position_dwords++];
  }

  // ***************************************************************************
  // Shader program
  // ***************************************************************************

  built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
  uint32_t shex_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
  built_shader_.resize(shex_position_dwords);

  built_shader_.push_back(dxbc::VersionToken(dxbc::ProgramType::kPixelShader, 5, 1));
  // Reserve space for the length token.
  built_shader_.push_back(0);

  dxbc::Statistics stat;
  std::memset(&stat, 0, sizeof(dxbc::Statistics));
  dxbc::Assembler a(built_shader_, stat);

  a.OpDclGlobalFlags(dxbc::kGlobalFlagAllResourcesBound);
  if (cbuffer_index_stencil_mask != UINT32_MAX) {
    a.OpDclConstantBuffer(
        dxbc::Src::CB(dxbc::Src::Dcl, cbuffer_index_stencil_mask, kTransferCBVRegisterStencilMask,
                      kTransferCBVRegisterStencilMask),
        1);
  }
  if (cbuffer_index_address != UINT32_MAX) {
    a.OpDclConstantBuffer(dxbc::Src::CB(dxbc::Src::Dcl, cbuffer_index_address,
                                        kTransferCBVRegisterAddress, kTransferCBVRegisterAddress),
                          1);
  }
  if (cbuffer_index_host_depth_address != UINT32_MAX) {
    a.OpDclConstantBuffer(
        dxbc::Src::CB(dxbc::Src::Dcl, cbuffer_index_host_depth_address,
                      kTransferCBVRegisterHostDepthAddress, kTransferCBVRegisterHostDepthAddress),
        1);
  }
  if (srv_index_color != UINT32_MAX) {
    a.OpDclResource(
        key.source_msaa_samples != xenos::MsaaSamples::k1X ? dxbc::ResourceDimension::kTexture2DMS
                                                           : dxbc::ResourceDimension::kTexture2D,
        dxbc::ResourceReturnTypeX4Token(source_color_is_uint ? dxbc::ResourceReturnType::kUInt
                                                             : dxbc::ResourceReturnType::kFloat),
        dxbc::Src::T(dxbc::Src::Dcl, srv_index_color, kTransferSRVRegisterColor,
                     kTransferSRVRegisterColor));
  }
  if (srv_index_depth != UINT32_MAX) {
    a.OpDclResource(key.source_msaa_samples != xenos::MsaaSamples::k1X
                        ? dxbc::ResourceDimension::kTexture2DMS
                        : dxbc::ResourceDimension::kTexture2D,
                    dxbc::ResourceReturnTypeX4Token(dxbc::ResourceReturnType::kFloat),
                    dxbc::Src::T(dxbc::Src::Dcl, srv_index_depth, kTransferSRVRegisterDepth,
                                 kTransferSRVRegisterDepth));
  }
  if (srv_index_stencil != UINT32_MAX) {
    a.OpDclResource(key.source_msaa_samples != xenos::MsaaSamples::k1X
                        ? dxbc::ResourceDimension::kTexture2DMS
                        : dxbc::ResourceDimension::kTexture2D,
                    dxbc::ResourceReturnTypeX4Token(dxbc::ResourceReturnType::kUInt),
                    dxbc::Src::T(dxbc::Src::Dcl, srv_index_stencil, kTransferSRVRegisterStencil,
                                 kTransferSRVRegisterStencil));
  }
  if (srv_index_host_depth != UINT32_MAX) {
    a.OpDclResource(key.host_depth_source_is_copy
                        ? dxbc::ResourceDimension::kBuffer
                        : (key.host_depth_source_msaa_samples != xenos::MsaaSamples::k1X
                               ? dxbc::ResourceDimension::kTexture2DMS
                               : dxbc::ResourceDimension::kTexture2D),
                    dxbc::ResourceReturnTypeX4Token(dxbc::ResourceReturnType::kFloat),
                    dxbc::Src::T(dxbc::Src::Dcl, srv_index_host_depth,
                                 kTransferSRVRegisterHostDepth, kTransferSRVRegisterHostDepth));
  }
  a.OpDclInputPSSIV(dxbc::InterpolationMode::kLinearNoPerspective,
                    dxbc::Dest::V1D(kInputRegisterPosition, 0b0011), dxbc::Name::kPosition);
  if (key.dest_msaa_samples != xenos::MsaaSamples::k1X) {
    a.OpDclInputPSSGV(dxbc::Dest::V1D(kInputRegisterSampleIndex, 0b0001), dxbc::Name::kSampleIndex);
  }
  if (osgn_parameter_index_sv_target != UINT32_MAX) {
    a.OpDclOutput(dxbc::Dest::O(0));
  }
  if (osgn_parameter_index_sv_depth != UINT32_MAX) {
    a.OpDclOutput(dxbc::Dest::ODepth());
  }
  if (osgn_parameter_index_sv_stencil_ref != UINT32_MAX) {
    a.OpDclOutput(dxbc::Dest::OStencilRef());
  }
  // r0:r2 are involved at least in common addressing code. Texture loads
  // usually can overwrite some of the addressing temps as they are only needed
  // for the coordinates for that load. Currently 3 temps are enough.
  a.OpDclTemps(3);

  uint32_t draw_resolution_scale_x = this->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = this->draw_resolution_scale_y();

  uint32_t tile_width_samples = xenos::kEdramTileWidthSamples * draw_resolution_scale_x;
  uint32_t tile_height_samples = xenos::kEdramTileHeightSamples * draw_resolution_scale_y;

  // Split the destination pixel index into 32bpp tile in r0.zw and
  // 32bpp-tile-relative pixel index in r0.xy.
  // r0.xy = pixel XY as uint
  a.OpFToU(dxbc::Dest::R(0, 0b0011), dxbc::Src::V1D(kInputRegisterPosition));
  uint32_t dest_tile_width_pixels =
      tile_width_samples >>
      (uint32_t(dest_is_64bpp) + uint32_t(key.dest_msaa_samples >= xenos::MsaaSamples::k4X));
  uint32_t dest_tile_height_pixels =
      tile_height_samples >> uint32_t(key.dest_msaa_samples >= xenos::MsaaSamples::k2X);
  // r0.xy = destination pixel XY index within the 32bpp tile
  // r0.zw = 32bpp tile XY index
  a.OpUDiv(dxbc::Dest::R(0, 0b1100), dxbc::Dest::R(0, 0b0011), dxbc::Src::R(0, 0b01000100),
           dxbc::Src::LU(dest_tile_width_pixels, dest_tile_height_pixels, dest_tile_width_pixels,
                         dest_tile_height_pixels));

  // r1.x = destination pitch in 32bpp tiles
  a.OpUBFE(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(xenos::kEdramPitchTilesBits), dxbc::Src::LU(0),
           dxbc::Src::CB(cbuffer_index_address, kTransferCBVRegisterAddress, 0, dxbc::Src::kXXXX));
  // r0.z = 32bpp tile index relative to the destination base
  // r0.w = free
  // r1.x = free
  a.OpUMAd(dxbc::Dest::R(0, 0b0100), dxbc::Src::R(1, dxbc::Src::kXXXX),
           dxbc::Src::R(0, dxbc::Src::kWWWW), dxbc::Src::R(0, dxbc::Src::kZZZZ));

  // Now the tile index doesn't have any dependencies on the destination. The
  // dword index within the source tile, however, is calculated from both the
  // source and the destination pixel size, sample count and color vs. depth.

  // Source can be 64bpp or 32bpp - depth if only depth is available, color in
  // all other cases.

  // Load the source to r1 (or low to r0, high to r1 if need 64bpp color as the
  // result, as the address is loaded to r1).

  // Source pixel and sample index within the 32bpp tile.
  // X to r1.x (or keep r0.x if not modifying).
  // Y to r1.y (or keep r0.y if not modifying).
  // Sample index to r1.z (or use v# if not modifying); r1.z will also be set
  // to 0 before sampling for the LOD of the single-sampled source (needs to
  // be in the register).
  // If 64bpp -> 32bpp, also the needed half in r0.w.

  dxbc::Src dest_sample(dxbc::Src::V1D(kInputRegisterSampleIndex, dxbc::Src::kXXXX));
  dxbc::Src source_sample(dest_sample);
  uint32_t source_tile_pixel_x_reg = 0;
  uint32_t source_tile_pixel_y_reg = 0;

  // First sample bit at 4x in Direct3D 10.1+ - horizontal sample.
  // Second sample bit at 4x in Direct3D 10.1+ - vertical sample.
  // At 2x:
  // - Native 2x: top is 1 in Direct3D 10.1+, bottom is 0.
  // - 2x as 4x: top is 0, bottom is 3.

  if (!source_is_64bpp && dest_is_64bpp) {
    // 32bpp -> 64bpp, need two samples of the source.
    if (key.source_msaa_samples >= xenos::MsaaSamples::k4X) {
      // 32bpp -> 64bpp, 4x ->.
      // Source has 32bpp halves in two adjacent samples.
      if (key.dest_msaa_samples >= xenos::MsaaSamples::k4X) {
        // 32bpp -> 64bpp, 4x -> 4x.
        // 1 destination horizontal sample = 2 source horizontal samples.
        // D p0,0 s0,0 = S p0,0 s0,0 | S p0,0 s1,0
        // D p0,0 s1,0 = S p1,0 s0,0 | S p1,0 s1,0
        // D p0,0 s0,1 = S p0,0 s0,1 | S p0,0 s1,1
        // D p0,0 s1,1 = S p1,0 s0,1 | S p1,0 s1,1
        // Thus destination horizontal sample -> source horizontal pixel,
        // vertical samples are 1:1.
        a.OpAnd(dxbc::Dest::R(1, 0b0100), dest_sample, dxbc::Src::LU(0b10));
        source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
        a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(31), dxbc::Src::LU(1),
                dxbc::Src::R(0, dxbc::Src::kXXXX),
                dxbc::Src::V1D(kInputRegisterSampleIndex, dxbc::Src::kXXXX));
        source_tile_pixel_x_reg = 1;
      } else if (key.dest_msaa_samples == xenos::MsaaSamples::k2X) {
        // 32bpp -> 64bpp, 4x -> 2x.
        // 1 destination horizontal pixel = 2 source horizontal samples.
        // D p0,0 s0 = S p0,0 s0,0 | S p0,0 s1,0
        // D p0,0 s1 = S p0,0 s0,1 | S p0,0 s1,1
        // D p1,0 s0 = S p1,0 s0,0 | S p1,0 s1,0
        // D p1,0 s1 = S p1,0 s0,1 | S p1,0 s1,1
        // Pixel index can be reused. Sample 1 (for native 2x) or 0 (for 2x as
        // 4x) should become samples 01, sample 0 or 3 should become samples 23.
        source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
        if (msaa_2x_supported_) {
          a.OpXOr(dxbc::Dest::R(1, 0b0100), dest_sample, dxbc::Src::LU(1));
          a.OpIShL(dxbc::Dest::R(1, 0b0100), source_sample, dxbc::Src::LU(1));
        } else {
          a.OpAnd(dxbc::Dest::R(1, 0b0100), dest_sample, dxbc::Src::LU(0b10));
        }
      } else {
        // 32bpp -> 64bpp, 4x -> 1x.
        // 1 destination horizontal pixel = 2 source horizontal samples.
        // D p0,0 = S p0,0 s0,0 | S p0,0 s1,0
        // D p0,1 = S p0,0 s0,1 | S p0,0 s1,1
        // Horizontal pixel index can be reused. Vertical pixel 1 should
        // become sample 2.
        a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(1),
                dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::LU(0));
        source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
        a.OpUShR(dxbc::Dest::R(1, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::LU(1));
        source_tile_pixel_y_reg = 1;
      }
    } else {
      // 32bpp -> 64bpp, 1x/2x ->.
      // Source has 32bpp halves in two adjacent pixels.
      if (key.dest_msaa_samples >= xenos::MsaaSamples::k4X) {
        // 32bpp -> 64bpp, 1x/2x -> 4x.
        // The X part.
        // 1 destination horizontal sample = 2 source horizontal pixels.
        a.OpIShL(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(2));
        a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(1), dxbc::Src::LU(1),
                dxbc::Src::V1D(kInputRegisterSampleIndex, dxbc::Src::kXXXX),
                dxbc::Src::R(1, dxbc::Src::kXXXX));
        source_tile_pixel_x_reg = 1;
        // Y is handled by common code.
      } else {
        // 32bpp -> 64bpp, 1x/2x -> 1x/2x.
        // The X part.
        // 1 destination horizontal pixel = 2 source horizontal pixels.
        a.OpIShL(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(1));
        source_tile_pixel_x_reg = 1;
        // Y is handled by common code.
      }
    }
  } else if (source_is_64bpp && !dest_is_64bpp) {
    // 64bpp -> 32bpp, also the half to r0.w.
    if (key.dest_msaa_samples >= xenos::MsaaSamples::k4X) {
      // 64bpp -> 32bpp, -> 4x.
      // The needed half is in the destination horizontal sample index.
      if (key.source_msaa_samples >= xenos::MsaaSamples::k4X) {
        // 64bpp -> 32bpp, 4x -> 4x.
        // D p0,0 s0,0 = S s0,0 low
        // D p0,0 s1,0 = S s0,0 high
        // D p1,0 s0,0 = S s1,0 low
        // D p1,0 s1,0 = S s1,0 high
        // Vertical pixel and sample (second bit) addressing is the same.
        // However, 1 horizontal destination pixel = 1 horizontal source sample.
        a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(0),
                dxbc::Src::R(0, dxbc::Src::kXXXX), dest_sample);
        source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
        // 2 destination horizontal samples = 1 source horizontal sample, thus
        // 2 destination horizontal pixels = 1 source horizontal pixel.
        a.OpUShR(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(1));
        source_tile_pixel_x_reg = 1;
      } else {
        // 64bpp -> 32bpp, 1x/2x -> 4x.
        // 2 destination horizontal samples = 1 source horizontal pixel, thus
        // 1 destination horizontal pixel = 1 source horizontal pixel. Can reuse
        // horizontal pixel index.
        // Y is handled by common code.
      }
      // Half in r0.w from the destination horizontal sample index.
      a.OpAnd(dxbc::Dest::R(0, 0b1000), dest_sample, dxbc::Src::LU(1));
    } else {
      // 64bpp -> 32bpp, -> 1x/2x.
      // The needed half is in the destination horizontal pixel index.
      if (key.source_msaa_samples >= xenos::MsaaSamples::k4X) {
        // 64bpp -> 32bpp, 4x -> 1x/2x.
        // (Destination horizontal pixel >> 1) & 1 = source horizontal sample
        // (first bit).
        a.OpUBFE(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(1),
                 dxbc::Src::R(0, dxbc::Src::kXXXX));
        source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
        if (key.dest_msaa_samples == xenos::MsaaSamples::k2X) {
          // 64bpp -> 32bpp, 4x -> 2x.
          // Destination vertical samples (1/0 in the first bit for native 2x or
          // 0/1 in the second bit for 2x as 4x) = source vertical samples
          // (second bit).
          if (msaa_2x_supported_) {
            a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(1), dest_sample,
                    source_sample);
            a.OpXOr(dxbc::Dest::R(1, 0b0100), source_sample, dxbc::Src::LU(1 << 1));
          } else {
            a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(0), source_sample,
                    dest_sample);
          }
        } else {
          // 64bpp -> 32bpp, 4x -> 1x.
          // 1 destination vertical pixel = 1 source vertical sample.
          a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(1),
                  dxbc::Src::R(0, dxbc::Src::kYYYY), source_sample);
          a.OpUShR(dxbc::Dest::R(1, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::LU(1));
          source_tile_pixel_y_reg = 1;
        }
        // 2 destination horizontal pixels = 1 source horizontal sample.
        // 4 destination horizontal pixels = 1 source horizontal pixel.
        a.OpUShR(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(2));
        source_tile_pixel_x_reg = 1;
      } else {
        // 64bpp -> 32bpp, 1x/2x -> 1x/2x.
        // The X part.
        // 2 destination horizontal pixels = 1 destination source pixel.
        a.OpUShR(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(1));
        source_tile_pixel_x_reg = 1;
        // Y is handled by common code.
      }
      // Half in r0.w from the destination horizontal pixel index.
      a.OpAnd(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(1));
    }
  } else {
    // Same bit count.
    if (key.source_msaa_samples != key.dest_msaa_samples) {
      if (key.source_msaa_samples >= xenos::MsaaSamples::k4X) {
        // Same BPP, 4x -> 1x/2x.
        if (key.dest_msaa_samples == xenos::MsaaSamples::k2X) {
          // Same BPP, 4x -> 2x.
          // Horizontal pixels to samples. Vertical sample (1/0 in the first bit
          // for native 2x or 0/1 in the second bit for 2x as 4x) to second
          // sample bit.
          source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
          if (msaa_2x_supported_) {
            a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(31), dxbc::Src::LU(1), dest_sample,
                    dxbc::Src::R(0, dxbc::Src::kXXXX));
            a.OpXOr(dxbc::Dest::R(1, 0b0100), source_sample, dxbc::Src::LU(1 << 1));
          } else {
            a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(0),
                    dxbc::Src::R(0, dxbc::Src::kXXXX), dest_sample);
          }
          a.OpUShR(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(1));
          source_tile_pixel_x_reg = 1;
        } else {
          // Same BPP, 4x -> 1x.
          // Pixels to samples.
          a.OpAnd(dxbc::Dest::R(1, 0b0100), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(1));
          source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
          a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(1),
                  dxbc::Src::R(0, dxbc::Src::kYYYY), source_sample);
          a.OpUShR(dxbc::Dest::R(1, 0b0011), dxbc::Src::R(0), dxbc::Src::LU(1));
          source_tile_pixel_x_reg = 1;
          source_tile_pixel_y_reg = 1;
        }
      } else {
        // Same BPP, 1x/2x -> 1x/2x/4x (as long as they're different).
        // Only the X part - Y is handled by common code.
        if (key.dest_msaa_samples >= xenos::MsaaSamples::k4X) {
          // Horizontal samples to pixels.
          a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(31), dxbc::Src::LU(1),
                  dxbc::Src::R(0, dxbc::Src::kXXXX), dest_sample);
          source_tile_pixel_x_reg = 1;
        }
      }
    }
  }
  // Common source Y and sample index for 1x/2x AA sources, independent of bits
  // per sample.
  if (key.source_msaa_samples < xenos::MsaaSamples::k4X &&
      key.source_msaa_samples != key.dest_msaa_samples) {
    if (key.dest_msaa_samples >= xenos::MsaaSamples::k4X) {
      // 1x/2x -> 4x.
      if (key.source_msaa_samples == xenos::MsaaSamples::k2X) {
        // 2x -> 4x.
        // Vertical samples (second bit) of 4x destination to vertical sample
        // (1, 0 for native 2x, or 0, 3 for 2x as 4x) of 2x source.
        a.OpUShR(dxbc::Dest::R(1, 0b0100), dest_sample, dxbc::Src::LU(1));
        source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
        if (msaa_2x_supported_) {
          a.OpXOr(dxbc::Dest::R(1, 0b0100), source_sample, dxbc::Src::LU(1));
        } else {
          a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(1), source_sample,
                  source_sample);
        }
      } else {
        // 1x -> 4x.
        // Vertical samples (second bit) to Y pixels.
        a.OpUShR(dxbc::Dest::R(1, 0b0010), dest_sample, dxbc::Src::LU(1));
        a.OpBFI(dxbc::Dest::R(1, 0b0010), dxbc::Src::LU(31), dxbc::Src::LU(1),
                dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::R(1, dxbc::Src::kYYYY));
        source_tile_pixel_y_reg = 1;
      }
    } else {
      // 1x/2x -> different 1x/2x.
      if (key.source_msaa_samples == xenos::MsaaSamples::k2X) {
        // 2x -> 1x.
        // Vertical pixels of 2x destination to vertical samples (1, 0 for
        // native 2x, or 0, 3 for 2x as 4x) of 1x source.
        a.OpAnd(dxbc::Dest::R(1, 0b0100), dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::LU(1));
        source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
        if (msaa_2x_supported_) {
          a.OpXOr(dxbc::Dest::R(1, 0b0100), source_sample, dxbc::Src::LU(1));
        } else {
          a.OpBFI(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(1), dxbc::Src::LU(1), source_sample,
                  source_sample);
        }
        a.OpUShR(dxbc::Dest::R(1, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::LU(1));
        source_tile_pixel_y_reg = 1;
      } else {
        // 1x -> 2x.
        // Vertical samples (1/0 in the first bit for native 2x or 0/1 in the
        // second bit for 2x as 4x) of 2x destination to vertical pixels of 1x
        // source.
        if (msaa_2x_supported_) {
          a.OpBFI(dxbc::Dest::R(1, 0b0010), dxbc::Src::LU(31), dxbc::Src::LU(1),
                  dxbc::Src::R(0, dxbc::Src::kYYYY), dest_sample);
          a.OpXOr(dxbc::Dest::R(1, 0b0010), dxbc::Src::R(1, dxbc::Src::kYYYY), dxbc::Src::LU(1));
        } else {
          a.OpUShR(dxbc::Dest::R(1, 0b0010), dest_sample, dxbc::Src::LU(1));
          a.OpBFI(dxbc::Dest::R(1, 0b0010), dxbc::Src::LU(31), dxbc::Src::LU(1),
                  dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::R(1, dxbc::Src::kYYYY));
        }
        source_tile_pixel_y_reg = 1;
      }
    }
  }

  uint32_t source_pixel_width_dwords_log2 =
      uint32_t(key.source_msaa_samples >= xenos::MsaaSamples::k4X) + uint32_t(source_is_64bpp);

  if (source_is_color != dest_is_color) {
    // Copying between color and depth / stencil - swap 40-32bpp-sample columns
    // in the pixel index within the source 32bpp tile using r1.w as temporary.
    uint32_t source_32bpp_tile_half_pixels =
        tile_width_samples >> (1 + source_pixel_width_dwords_log2);
    a.OpULT(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(source_tile_pixel_x_reg, dxbc::Src::kXXXX),
            dxbc::Src::LU(source_32bpp_tile_half_pixels));
    a.OpMovC(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW),
             dxbc::Src::LI(int32_t(source_32bpp_tile_half_pixels)),
             dxbc::Src::LI(-int32_t(source_32bpp_tile_half_pixels)));
    a.OpIAdd(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(source_tile_pixel_x_reg, dxbc::Src::kXXXX),
             dxbc::Src::R(1, dxbc::Src::kWWWW));
    source_tile_pixel_x_reg = 1;
    // r1.w = free
  }

  // Current register allocation:
  // r0.xy = pixel index within the destination 32bpp tile
  // r0.z = 32bpp tile index relative to the destination base
  // r0.w for 64bpp -> 32bpp - needed 32bpp half index of 64bpp data
  // r1.xy = pixel index within the source 32bpp tile
  // r1.z for 2x/4x -> = sample index within the source pixel

  // Apply the source 32bpp tile index.
  // r1.w = destination to source EDRAM tile adjustment
  a.OpIBFE(dxbc::Dest::R(1, 0b1000), dxbc::Src::LU(xenos::kEdramBaseTilesBits + 1),
           dxbc::Src::LU(xenos::kEdramPitchTilesBits * 2),
           dxbc::Src::CB(cbuffer_index_address, kTransferCBVRegisterAddress, 0, dxbc::Src::kXXXX));
  // r1.w = 32bpp tile index within the source, or the tile index within the
  //        source minus the EDRAM tile count if transferring across addressing
  //        wrapping (if negative)
  a.OpIAdd(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(0, dxbc::Src::kZZZZ),
           dxbc::Src::R(1, dxbc::Src::kWWWW));
  // r1.w = 32bpp tile index within the source
  a.OpAnd(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW),
          dxbc::Src::LU(xenos::kEdramTileCount - 1));
  // r2.x = source pitch in 32bpp tiles
  a.OpUBFE(dxbc::Dest::R(2, 0b0001), dxbc::Src::LU(xenos::kEdramPitchTilesBits),
           dxbc::Src::LU(xenos::kEdramPitchTilesBits),
           dxbc::Src::CB(cbuffer_index_address, kTransferCBVRegisterAddress, 0, dxbc::Src::kXXXX));
  // r1.w = source tile row
  // r2.x = source 32bpp tile within the row
  a.OpUDiv(dxbc::Dest::R(1, 0b1000), dxbc::Dest::R(2, 0b0001), dxbc::Src::R(1, dxbc::Src::kWWWW),
           dxbc::Src::R(2, dxbc::Src::kXXXX));
  // r1.x = pixel X within the source texture
  // r2.x = free
  a.OpUMAd(
      dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(tile_width_samples >> source_pixel_width_dwords_log2),
      dxbc::Src::R(2, dxbc::Src::kXXXX), dxbc::Src::R(source_tile_pixel_x_reg, dxbc::Src::kXXXX));
  // r1.y = pixel Y within the source texture
  // r1.w = free
  a.OpUMAd(dxbc::Dest::R(1, 0b0010),
           dxbc::Src::LU(tile_height_samples >>
                         uint32_t(key.source_msaa_samples >= xenos::MsaaSamples::k2X)),
           dxbc::Src::R(1, dxbc::Src::kWWWW),
           dxbc::Src::R(source_tile_pixel_y_reg, dxbc::Src::kYYYY));

  // Load the source to r1, or, for 32bpp | 32bpp -> 64bpp, the first dword to
  // r0 since addressing will not be needed anymore for color, and the second
  // dword to r1.
  // Depth will be loaded to w before loading stencil (so it doesn't overwrite
  // the coordinates needed for stencil loading).
  // Stencil will be loaded to x.
  // Color will be loaded to x...w.
  bool source_load_is_two_dwords = !source_is_64bpp && dest_is_64bpp;
  if (key.source_msaa_samples != xenos::MsaaSamples::k1X) {
    for (uint32_t i = 0; i <= uint32_t(source_load_is_two_dwords); ++i) {
      uint32_t source_load_register = source_load_is_two_dwords ? i : 1;
      if (srv_index_depth != UINT32_MAX) {
        a.OpLdMS(dxbc::Dest::R(source_load_register, 0b1000), dxbc::Src::R(1), 0b0011,
                 dxbc::Src::T(srv_index_depth, kTransferSRVRegisterDepth, dxbc::Src::kXXXX),
                 source_sample);
      }
      if (srv_index_stencil != UINT32_MAX) {
        a.OpLdMS(dxbc::Dest::R(source_load_register, 0b0001), dxbc::Src::R(1), 0b0011,
                 dxbc::Src::T(srv_index_stencil, kTransferSRVRegisterStencil, dxbc::Src::kYYYY),
                 source_sample);
      } else if (srv_index_color != UINT32_MAX) {
        a.OpLdMS(dxbc::Dest::R(source_load_register, source_color_srv_component_mask),
                 dxbc::Src::R(1), 0b0011, dxbc::Src::T(srv_index_color, kTransferSRVRegisterColor),
                 source_sample);
      }
      if (source_load_is_two_dwords && !i) {
        // Go to the next sample or pixel along X if need to load two dwords.
        if (key.source_msaa_samples >= xenos::MsaaSamples::k4X) {
          a.OpOr(dxbc::Dest::R(1, 0b0100), source_sample, dxbc::Src::LU(1));
          source_sample = dxbc::Src::R(1, dxbc::Src::kZZZZ);
        } else {
          a.OpOr(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX), dxbc::Src::LU(1));
        }
      }
    }
  } else {
    // Write zero to the LOD index in r1.z.
    a.OpMov(dxbc::Dest::R(1, 0b0100), dxbc::Src::LU(0));
    dxbc::Src source_coordinates(dxbc::Src::R(1, 0b10000100));
    for (uint32_t i = 0; i <= uint32_t(source_load_is_two_dwords); ++i) {
      uint32_t source_load_register = source_load_is_two_dwords ? i : 1;
      if (srv_index_depth != UINT32_MAX) {
        a.OpLd(dxbc::Dest::R(source_load_register, 0b1000), source_coordinates, 0b1011,
               dxbc::Src::T(srv_index_depth, kTransferSRVRegisterDepth, dxbc::Src::kXXXX));
      }
      if (srv_index_stencil != UINT32_MAX) {
        a.OpLd(dxbc::Dest::R(source_load_register, 0b0001), source_coordinates, 0b1011,
               dxbc::Src::T(srv_index_stencil, kTransferSRVRegisterStencil, dxbc::Src::kYYYY));
      } else if (srv_index_color != UINT32_MAX) {
        a.OpLd(dxbc::Dest::R(source_load_register, source_color_srv_component_mask),
               source_coordinates, 0b1011,
               dxbc::Src::T(srv_index_color, kTransferSRVRegisterColor));
      }
      if (source_load_is_two_dwords && !i) {
        // Go to the next pixel along X if need to load two dwords.
        a.OpOr(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX), dxbc::Src::LU(1));
      }
    }
  }
  // Pick the needed 32bpp half of the 64bpp color based on r0.w.
  if (source_is_64bpp && !dest_is_64bpp) {
    uint32_t source_color_half_component_count = source_color_format_component_count >> 1;
    if (dest_is_stencil_bit) {
      a.OpMovC(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(0, dxbc::Src::kWWWW),
               dxbc::Src::R(1).Select(source_color_half_component_count),
               dxbc::Src::R(1, dxbc::Src::kXXXX));
    } else {
      uint32_t color_high_dword_swizzle =
          (source_color_half_component_count * 0b01010101) &
          ~((uint32_t(1) << (source_color_half_component_count * 2)) - 1);
      for (uint32_t i = 0; i < source_color_half_component_count; ++i) {
        color_high_dword_swizzle |= (source_color_half_component_count + i) << (i * 2);
      }
      a.OpMovC(dxbc::Dest::R(1, (1 << source_color_half_component_count) - 1),
               dxbc::Src::R(0, dxbc::Src::kWWWW), dxbc::Src::R(1, color_high_dword_swizzle),
               dxbc::Src::R(1));
    }
  }

  if (osgn_parameter_index_sv_stencil_ref != UINT32_MAX && srv_index_stencil != UINT32_MAX) {
    // For the depth -> depth case, write the stencil loaded to r1.x directly to
    // the output.
    assert_true(mode.output == TransferOutput::kDepth);
    a.OpMov(dxbc::Dest::OStencilRef(), dxbc::Src::R(1, dxbc::Src::kXXXX));
  }

  if (dest_is_64bpp) {
    // Handle construction of 64bpp color, either from two 32-bit samples in r0
    // and r1, or from one 64bpp sample in r1. Using r2.x as temporary when
    // needed.
    // If color_packed_in_r0x_and_r1x, use the generic path for combining two
    // 32-bit samples - as raw in r0.x and r1.x - into the destination.
    bool color_packed_in_r0x_and_r1x = false;
    if (source_is_color) {
      switch (source_color_format) {
        case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA: {
          // 8_8_8_8_GAMMA is represented by linear stored in
          // R16G16B16A16_UNORM.
          for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 3; ++j) {
              DxbcShaderTranslator::PreSaturatedLinearToPWLGamma(a, i, j, i, j, 2, 0, 2, 1);
            }
          }
        }
          [[fallthrough]];
        case xenos::ColorRenderTargetFormat::k_8_8_8_8: {
          color_packed_in_r0x_and_r1x = true;
          for (uint32_t i = 0; i < 2; ++i) {
            a.OpMAd(dxbc::Dest::R(i), dxbc::Src::R(i), dxbc::Src::LF(255.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(i), dxbc::Src::R(i));
            for (uint32_t j = 1; j < 4; ++j) {
              a.OpBFI(dxbc::Dest::R(i, 0b0001), dxbc::Src::LU(8), dxbc::Src::LU(j * 8),
                      dxbc::Src::R(i).Select(j), dxbc::Src::R(i, dxbc::Src::kXXXX));
            }
          }
        } break;
        case xenos::ColorRenderTargetFormat::k_2_10_10_10:
        case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10: {
          color_packed_in_r0x_and_r1x = true;
          for (uint32_t i = 0; i < 2; ++i) {
            a.OpMAd(dxbc::Dest::R(i), dxbc::Src::R(i),
                    dxbc::Src::LF(1023.0f, 1023.0f, 1023.0f, 3.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(i), dxbc::Src::R(i));
            for (uint32_t j = 1; j < 4; ++j) {
              a.OpBFI(dxbc::Dest::R(i, 0b0001), dxbc::Src::LU(j == 3 ? 2 : 10),
                      dxbc::Src::LU(j * 10), dxbc::Src::R(i).Select(j),
                      dxbc::Src::R(i, dxbc::Src::kXXXX));
            }
          }
        } break;
        case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
        case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16: {
          color_packed_in_r0x_and_r1x = true;
          for (uint32_t i = 0; i < 2; ++i) {
            // Float16 has a wider range for both color and alpha, also NaNs -
            // clamp and convert.
            for (uint32_t j = 0; j < 3; ++j) {
              DxbcShaderTranslator::UnclampedFloat32To7e3(a, i, j, i, j, 2, 0);
              if (j) {
                a.OpBFI(dxbc::Dest::R(i, 0b0001), dxbc::Src::LU(10), dxbc::Src::LU(j * 10),
                        dxbc::Src::R(i).Select(j), dxbc::Src::R(i, dxbc::Src::kXXXX));
              }
            }
            // Saturate and convert the alpha.
            a.OpMov(dxbc::Dest::R(i, 0b1000), dxbc::Src::R(i, dxbc::Src::kWWWW), true);
            a.OpMAd(dxbc::Dest::R(i, 0b1000), dxbc::Src::R(i, dxbc::Src::kWWWW),
                    dxbc::Src::LF(3.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(i, 0b1000), dxbc::Src::R(i, dxbc::Src::kWWWW));
            a.OpBFI(dxbc::Dest::R(i, 0b0001), dxbc::Src::LU(2), dxbc::Src::LU(30),
                    dxbc::Src::R(i, dxbc::Src::kWWWW), dxbc::Src::R(i, dxbc::Src::kXXXX));
          }
        } break;
        // All 64bpp formats, and all 16 bits per component formats, are
        // represented as integers in ownership transfer for safe handling of
        // NaNs and -32768 / -32767.
        case xenos::ColorRenderTargetFormat::k_16_16:
        case xenos::ColorRenderTargetFormat::k_16_16_FLOAT: {
          if (dest_color_format == xenos::ColorRenderTargetFormat::k_32_32_FLOAT) {
            for (uint32_t i = 0; i < 2; ++i) {
              a.OpBFI(dxbc::Dest::O(0, 1 << i), dxbc::Src::LU(16), dxbc::Src::LU(16),
                      dxbc::Src::R(i, dxbc::Src::kYYYY), dxbc::Src::R(i, dxbc::Src::kXXXX));
            }
          } else {
            a.OpMov(dxbc::Dest::O(0, 0b0011), dxbc::Src::R(0));
            a.OpMov(dxbc::Dest::O(0, 0b1100), dxbc::Src::R(1, 0b0100 << 4));
          }
        } break;
        case xenos::ColorRenderTargetFormat::k_16_16_16_16:
        case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT: {
          if (dest_color_format == xenos::ColorRenderTargetFormat::k_32_32_FLOAT) {
            a.OpBFI(dxbc::Dest::O(0, 0b0011), dxbc::Src::LU(16), dxbc::Src::LU(16),
                    dxbc::Src::R(1, 0b1101), dxbc::Src::R(1, 0b1000));
          } else {
            a.OpMov(dxbc::Dest::O(0), dxbc::Src::R(1));
          }
        } break;
        case xenos::ColorRenderTargetFormat::k_32_FLOAT: {
          color_packed_in_r0x_and_r1x = true;
        } break;
        case xenos::ColorRenderTargetFormat::k_32_32_FLOAT: {
          if (dest_color_format == xenos::ColorRenderTargetFormat::k_32_32_FLOAT) {
            a.OpMov(dxbc::Dest::O(0, 0b0011), dxbc::Src::R(1));
          } else {
            a.OpUBFE(dxbc::Dest::O(0), dxbc::Src::LU(16), dxbc::Src::LU(0, 16, 0, 16),
                     dxbc::Src::R(1, 0b01010000));
          }
        } break;
      }
    } else {
      assert_not_zero(rs & kTransferUsedRootParameterDepthSRVBit);
      color_packed_in_r0x_and_r1x = true;
      for (uint32_t i = 0; i < 2; ++i) {
        switch (source_depth_format) {
          case xenos::DepthRenderTargetFormat::kD24S8: {
            // Round to the nearest even integer. This seems to be the correct
            // conversion, adding +0.5 and rounding towards zero results in red
            // instead of black in the 4D5307E6 clear shader.
            a.OpMul(dxbc::Dest::R(i, 0b1000), dxbc::Src::R(i, dxbc::Src::kWWWW),
                    dxbc::Src::LF(float(0xFFFFFF)));
            a.OpRoundNE(dxbc::Dest::R(i, 0b1000), dxbc::Src::R(i, dxbc::Src::kWWWW));
            a.OpFToU(dxbc::Dest::R(i, 0b1000), dxbc::Src::R(i, dxbc::Src::kWWWW));
          } break;
          case xenos::DepthRenderTargetFormat::kD24FS8: {
            // Convert using r1.y as temporary.
            // When converting the depth in pixel shaders, it's always exact,
            // truncating not to insert additional rounding instructions.
            DxbcShaderTranslator::PreClampedDepthTo20e4(
                a, i, 3, i, 3, 1, 1,
                !depth_float24_convert_in_pixel_shader() && depth_float24_round(), true);
          } break;
        }
        // Merge depth and stencil into r0/r1.x.
        a.OpBFI(dxbc::Dest::R(i, 0b0001), dxbc::Src::LU(24), dxbc::Src::LU(8),
                dxbc::Src::R(i, dxbc::Src::kWWWW), dxbc::Src::R(i, dxbc::Src::kXXXX));
      }
    }
    if (color_packed_in_r0x_and_r1x) {
      if (dest_color_format == xenos::ColorRenderTargetFormat::k_32_32_FLOAT) {
        a.OpMov(dxbc::Dest::O(0, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX));
        a.OpMov(dxbc::Dest::O(0, 0b0010), dxbc::Src::R(1, dxbc::Src::kXXXX));
      } else {
        for (uint32_t i = 0; i < 2; ++i) {
          a.OpUBFE(dxbc::Dest::O(0, 0b11 << (i * 2)), dxbc::Src::LU(16),
                   dxbc::Src::LU(0, 16, 0, 16), dxbc::Src::R(i, dxbc::Src::kXXXX));
        }
      }
    }
  } else {
    // Handle a 32bpp destination (32bpp color, or depth / stencil). If
    // color_packed_in_r1x is true, a raw 32bpp color value was written, and
    // common handling will be done.
    bool color_packed_in_r1x = false;
    bool depth_loaded_in_guest_format = false;
    if (source_is_color) {
      switch (source_color_format) {
        case xenos::ColorRenderTargetFormat::k_8_8_8_8:
        case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA: {
          if (dest_is_stencil_bit) {
            if (source_color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
              DxbcShaderTranslator::PreSaturatedLinearToPWLGamma(a, 1, 0, 1, 0, 2, 0, 2, 1);
            }
            a.OpMAd(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX),
                    dxbc::Src::LF(255.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX));
          } else if (dest_is_color &&
                     (dest_color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8 ||
                      dest_color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA)) {
            if (source_color_format != dest_color_format) {
              // Color space conversion between k_8_8_8_8 and
              // k_8_8_8_8_GAMMA.
              if (dest_color_format != xenos::ColorRenderTargetFormat::k_8_8_8_8) {
                for (uint32_t i = 0; i < 3; ++i) {
                  DxbcShaderTranslator::PreSaturatedLinearToPWLGamma(a, 1, i, 1, i, 2, 0, 2, 1);
                }
              } else {
                for (uint32_t i = 0; i < 3; ++i) {
                  DxbcShaderTranslator::PWLGammaToLinear(a, 1, i, 1, i, true, 2, 0, 2, 1);
                }
              }
            }
            // Same or converted format - passthrough.
            a.OpMov(dxbc::Dest::O(0), dxbc::Src::R(1));
          } else if (mode.output == TransferOutput::kDepth) {
            if (source_color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
              for (uint32_t i = osgn_parameter_index_sv_stencil_ref != UINT32_MAX ? 0 : 1; i < 3;
                   ++i) {
                DxbcShaderTranslator::PreSaturatedLinearToPWLGamma(a, 1, i, 1, i, 2, 0, 2, 1);
              }
            }
            // When need only depth, not stencil, skip the red component.
            a.OpMAd(dxbc::Dest::R(
                        1, osgn_parameter_index_sv_stencil_ref != UINT32_MAX ? 0b1111 : 0b1110),
                    dxbc::Src::R(1), dxbc::Src::LF(255.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(1, 0b1110), dxbc::Src::R(1));
            if (osgn_parameter_index_sv_stencil_ref != UINT32_MAX) {
              // Write the red component to the stencil reference.
              a.OpFToU(dxbc::Dest::OStencilRef(), dxbc::Src::R(1, dxbc::Src::kXXXX));
            }
            // Put depth in 0:23 of r1.w.
            // r1.y = 0xGGBB0000.
            a.OpBFI(dxbc::Dest::R(1, 0b0010), dxbc::Src::LU(8), dxbc::Src::LU(8),
                    dxbc::Src::R(1, dxbc::Src::kZZZZ), dxbc::Src::R(1, dxbc::Src::kYYYY));
            // r1.w = 0xGGBBAA00.
            a.OpBFI(dxbc::Dest::R(1, 0b1000), dxbc::Src::LU(8), dxbc::Src::LU(16),
                    dxbc::Src::R(1, dxbc::Src::kWWWW), dxbc::Src::R(1, dxbc::Src::kYYYY));
          } else {
            if (source_color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
              for (uint32_t i = 0; i < 3; ++i) {
                DxbcShaderTranslator::PreSaturatedLinearToPWLGamma(a, 1, i, 1, i, 2, 0, 2, 1);
              }
            }
            color_packed_in_r1x = true;
            a.OpMAd(dxbc::Dest::R(1), dxbc::Src::R(1), dxbc::Src::LF(255.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(1), dxbc::Src::R(1));
            for (uint32_t i = 1; i < 4; ++i) {
              a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(8), dxbc::Src::LU(i * 8),
                      dxbc::Src::R(1).Select(i), dxbc::Src::R(1, dxbc::Src::kXXXX));
            }
          }
        } break;
        case xenos::ColorRenderTargetFormat::k_2_10_10_10:
        case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10: {
          if (dest_is_stencil_bit) {
            a.OpMAd(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX),
                    dxbc::Src::LF(1023.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX));
          } else if (dest_is_color &&
                     (dest_color_format == xenos::ColorRenderTargetFormat::k_2_10_10_10 ||
                      dest_color_format ==
                          xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10)) {
            a.OpMov(dxbc::Dest::O(0), dxbc::Src::R(1));
          } else {
            color_packed_in_r1x = true;
            a.OpMAd(dxbc::Dest::R(1), dxbc::Src::R(1),
                    dxbc::Src::LF(1023.0f, 1023.0f, 1023.0f, 3.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(1), dxbc::Src::R(1));
            for (uint32_t i = 1; i < 4; ++i) {
              a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(i == 3 ? 2 : 10),
                      dxbc::Src::LU(i * 10), dxbc::Src::R(1).Select(i),
                      dxbc::Src::R(1, dxbc::Src::kXXXX));
            }
          }
        } break;
        case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
        case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16: {
          if (dest_is_stencil_bit) {
            DxbcShaderTranslator::UnclampedFloat32To7e3(a, 1, 0, 1, 0, 2, 0);
          } else if (dest_is_color &&
                     (dest_color_format == xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT ||
                      dest_color_format ==
                          xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16)) {
            a.OpMov(dxbc::Dest::O(0), dxbc::Src::R(1));
          } else {
            color_packed_in_r1x = true;
            // Float16 has a wider range for both color and alpha, also NaNs -
            // clamp and convert.
            for (uint32_t i = 0; i < 3; ++i) {
              DxbcShaderTranslator::UnclampedFloat32To7e3(a, 1, i, 1, i, 2, 0);
              if (i) {
                a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(10), dxbc::Src::LU(i * 10),
                        dxbc::Src::R(1).Select(i), dxbc::Src::R(1, dxbc::Src::kXXXX));
              }
            }
            // Saturate and convert the alpha.
            a.OpMov(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW), true);
            a.OpMAd(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW),
                    dxbc::Src::LF(3.0f), dxbc::Src::LF(0.5f));
            a.OpFToU(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW));
            a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(2), dxbc::Src::LU(30),
                    dxbc::Src::R(1, dxbc::Src::kWWWW), dxbc::Src::R(1, dxbc::Src::kXXXX));
          }
        } break;
        case xenos::ColorRenderTargetFormat::k_16_16:
        case xenos::ColorRenderTargetFormat::k_16_16_16_16:
        case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
        case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT: {
          // All 16 bits per component formats are represented as integers in
          // ownership transfer for safe handling of NaNs and -32768 / -32767.
          if (dest_is_stencil_bit) {
            // High bits are not important for discarding, as only one bit is
            // checked - already loaded to red.
          } else if (dest_is_color &&
                     (dest_color_format == xenos::ColorRenderTargetFormat::k_16_16 ||
                      dest_color_format == xenos::ColorRenderTargetFormat::k_16_16_FLOAT)) {
            a.OpMov(dxbc::Dest::O(0, 0b0011), dxbc::Src::R(1));
          } else {
            color_packed_in_r1x = true;
            a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(16), dxbc::Src::LU(16),
                    dxbc::Src::R(1, dxbc::Src::kYYYY), dxbc::Src::R(1, dxbc::Src::kXXXX));
          }
        } break;
        case xenos::ColorRenderTargetFormat::k_32_FLOAT:
        case xenos::ColorRenderTargetFormat::k_32_32_FLOAT: {
          color_packed_in_r1x = true;
        } break;
      }
    } else if (rs & kTransferUsedRootParameterDepthSRVBit) {
      if (dest_is_color || dest_depth_format != source_depth_format) {
        // Need to reinterpret the depth value as color or as a different depth
        // format. Convert the depth within r1.w.
        depth_loaded_in_guest_format = true;
        switch (source_depth_format) {
          case xenos::DepthRenderTargetFormat::kD24S8: {
            // Round to the nearest even integer. This seems to be the correct
            // conversion, adding +0.5 and rounding towards zero results in red
            // instead of black in the 4D5307E6 clear shader.
            a.OpMul(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW),
                    dxbc::Src::LF(float(0xFFFFFF)));
            a.OpRoundNE(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW));
            a.OpFToU(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW));
          } break;
          case xenos::DepthRenderTargetFormat::kD24FS8: {
            // Convert using r1.y as temporary.
            // When converting the depth in pixel shaders, it's always exact,
            // truncating not to insert additional rounding instructions.
            DxbcShaderTranslator::PreClampedDepthTo20e4(
                a, 1, 3, 1, 3, 1, 1,
                !depth_float24_convert_in_pixel_shader() && depth_float24_round(), true);
          } break;
        }
        if (dest_is_color) {
          // Merge depth and stencil into r1.x for reinterpretation as color.
          color_packed_in_r1x = true;
          a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(24), dxbc::Src::LU(8),
                  dxbc::Src::R(1, dxbc::Src::kWWWW), dxbc::Src::R(1, dxbc::Src::kXXXX));
        }
      }
    }
    switch (mode.output) {
      case TransferOutput::kColor:
        // Unless a special path was taken, unpack the raw 32bpp value into the
        // 32bpp color output. Any register can be used as temporary if needed -
        // this is the end of the shader.
        if (color_packed_in_r1x) {
          switch (dest_color_format) {
            case xenos::ColorRenderTargetFormat::k_8_8_8_8: {
              a.OpUBFE(dxbc::Dest::R(1), dxbc::Src::LU(8), dxbc::Src::LU(0, 8, 16, 24),
                       dxbc::Src::R(1, dxbc::Src::kXXXX));
              a.OpUToF(dxbc::Dest::R(1), dxbc::Src::R(1));
              a.OpMul(dxbc::Dest::O(0), dxbc::Src::R(1), dxbc::Src::LF(1.0f / 255.0f));
            } break;
            case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA: {
              // 8_8_8_8_GAMMA is represented by linear stored in
              // R16G16B16A16_UNORM.
              a.OpUBFE(dxbc::Dest::R(1), dxbc::Src::LU(8), dxbc::Src::LU(0, 8, 16, 24),
                       dxbc::Src::R(1, dxbc::Src::kXXXX));
              a.OpUToF(dxbc::Dest::R(1), dxbc::Src::R(1));
              a.OpMul(dxbc::Dest::R(1, 0b0111), dxbc::Src::R(1), dxbc::Src::LF(1.0f / 255.0f));
              a.OpMul(dxbc::Dest::O(0, 0b1000), dxbc::Src::R(1), dxbc::Src::LF(1.0f / 255.0f));
              for (uint32_t i = 0; i < 3; ++i) {
                DxbcShaderTranslator::PWLGammaToLinear(a, 1, i, 1, i, true, 0, 0, 0, 1);
              }
              a.OpMov(dxbc::Dest::O(0, 0b0111), dxbc::Src::R(1));
            } break;
            case xenos::ColorRenderTargetFormat::k_2_10_10_10:
            case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10: {
              a.OpUBFE(dxbc::Dest::R(1), dxbc::Src::LU(10, 10, 10, 2), dxbc::Src::LU(0, 10, 20, 30),
                       dxbc::Src::R(1, dxbc::Src::kXXXX));
              a.OpUToF(dxbc::Dest::R(1), dxbc::Src::R(1));
              a.OpMul(dxbc::Dest::O(0), dxbc::Src::R(1),
                      dxbc::Src::LF(1.0f / 1023.0f, 1.0f / 1023.0f, 1.0f / 1023.0f, 1.0f / 3.0f));
            } break;
            case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
            case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16: {
              // Color using r1.yz as temporary.
              for (uint32_t i = 0; i < 3; ++i) {
                DxbcShaderTranslator::Float7e3To32(a, dxbc::Dest::O(0, 1 << i), 1, 0, i * 10, 1, 1,
                                                   1, 2);
              }
              // Alpha.
              a.OpUBFE(dxbc::Dest::R(1, 0b1000), dxbc::Src::LU(2), dxbc::Src::LU(30),
                       dxbc::Src::R(1, dxbc::Src::kXXXX));
              a.OpUToF(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW));
              a.OpMul(dxbc::Dest::O(0, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW),
                      dxbc::Src::LF(1.0f / 3.0f));
            } break;
            case xenos::ColorRenderTargetFormat::k_16_16:
            case xenos::ColorRenderTargetFormat::k_16_16_FLOAT: {
              // All 16 bits per component formats are represented as integers
              // in ownership transfer for safe handling of NaNs and
              // -32768 / -32767.
              a.OpUBFE(dxbc::Dest::O(0, 0b0011), dxbc::Src::LU(16), dxbc::Src::LU(0, 16, 0, 0),
                       dxbc::Src::R(1, dxbc::Src::kXXXX));
            } break;
            case xenos::ColorRenderTargetFormat::k_32_FLOAT: {
              // Already as a 32-bit value.
              a.OpMov(dxbc::Dest::O(0, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX));
            } break;
            default:
              // A 64bpp format (handled separately) or an invalid one.
              assert_unhandled_case(dest_color_format);
          }
        }
        break;
      case TransferOutput::kDepth:
        if (source_is_color || depth_loaded_in_guest_format) {
          if (color_packed_in_r1x) {
            // Extract the depth bits to r1.w.
            a.OpUBFE(dxbc::Dest::R(1, 0b1000), dxbc::Src::LU(24), dxbc::Src::LU(8),
                     dxbc::Src::R(1, dxbc::Src::kXXXX));
            if (osgn_parameter_index_sv_stencil_ref != UINT32_MAX) {
              // Extract the stencil bits to the stencil reference.
              // The depth -> depth case is handled earlier, not long after
              // loading the stencil, for simplicity.
              a.OpUBFE(dxbc::Dest::OStencilRef(), dxbc::Src::LU(8), dxbc::Src::LU(0),
                       dxbc::Src::R(1, dxbc::Src::kXXXX));
            }
          }
          // r1.w contains the depth in the guest format. If a host depth source
          // is available, need to check if it's up to date - if it is, the host
          // precision value needs to be written. Otherwise, the new guest value
          // needs to be converted to the host format. Using `if` here because
          // it's likely that the values will either be the same - if not
          // modified - or different - if cleared or totally overwritten - in
          // large amounts of samples, usually whole waves, at once.
          if (rs & kTransferUsedRootParameterHostDepthSRVBit) {
            // Load the host float32 depth to r0.x, check if, when converted to
            // the guest format, it's the same as the guest source, thus up to
            // date, and if it is, write host float32 depth to r1.w, otherwise
            // do the guest -> host conversion on the `else` path.

            // Current register allocation:
            // r0.xy = pixel index within the destination 32bpp tile
            // r0.z = 32bpp tile index relative to the destination base
            // r1.w = depth in guest format

            if (key.host_depth_source_is_copy) {
              // Get the address in the EDRAM scratch buffer and load from
              // there.
              // The beginning of the buffer is (0, 0) of the destination.
              // 40-sample columns are not swapped for addressing simplicity
              // (because this is used for depth -> depth transfers, where
              // swapping isn't needed).
              // Convert samples to pixels.
              assert_true(key.host_depth_source_msaa_samples == xenos::MsaaSamples::k1X);
              if (key.dest_msaa_samples >= xenos::MsaaSamples::k2X) {
                if (key.dest_msaa_samples >= xenos::MsaaSamples::k4X) {
                  // Horizontal sample index in bit 0.
                  a.OpBFI(dxbc::Dest::R(0, 0b0001), dxbc::Src::LU(31), dxbc::Src::LU(1),
                          dxbc::Src::R(0, dxbc::Src::kXXXX), dest_sample);
                }
                // Vertical sample index as 1 or 0 in bit 0 for true 2x or as 0
                // or 1 in bit 1 for 4x or for 2x emulated as 4x.
                if (key.dest_msaa_samples == xenos::MsaaSamples::k2X && msaa_2x_supported_) {
                  a.OpBFI(dxbc::Dest::R(0, 0b0010), dxbc::Src::LU(31), dxbc::Src::LU(1),
                          dxbc::Src::R(0, dxbc::Src::kYYYY), dest_sample);
                  a.OpXOr(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY),
                          dxbc::Src::LU(1));
                } else {
                  // Using r0.w as a temporary.
                  a.OpUShR(dxbc::Dest::R(0, 0b1000), dest_sample, dxbc::Src::LU(1));
                  a.OpBFI(dxbc::Dest::R(0, 0b0010), dxbc::Src::LU(31), dxbc::Src::LU(1),
                          dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::R(0, dxbc::Src::kWWWW));
                }
              }
              // Combine the tile sample index and the tile index into buffer
              // address to r0.x.
              // The tile index doesn't need to be wrapped, as the host depth is
              // written to the beginning of the buffer, without the base
              // offset.
              a.OpUMAd(dxbc::Dest::R(0, 0b0001), dxbc::Src::LU(tile_width_samples),
                       dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::R(0, dxbc::Src::kXXXX));
              a.OpUMAd(dxbc::Dest::R(0, 0b0001),
                       dxbc::Src::LU(tile_width_samples * tile_height_samples),
                       dxbc::Src::R(0, dxbc::Src::kZZZZ), dxbc::Src::R(0, dxbc::Src::kXXXX));
              // Load from the buffer.
              a.OpLd(dxbc::Dest::R(0, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX), 0b0001,
                     dxbc::Src::T(srv_index_host_depth, kTransferSRVRegisterHostDepth,
                                  dxbc::Src::kXXXX));
            } else {
              // Adjust the tile index from the destination to the host depth
              // source.
              // r0.w = destination to host depth source EDRAM tile adjustment
              a.OpIBFE(dxbc::Dest::R(0, 0b1000), dxbc::Src::LU(xenos::kEdramBaseTilesBits + 1),
                       dxbc::Src::LU(xenos::kEdramPitchTilesBits * 2),
                       dxbc::Src::CB(cbuffer_index_host_depth_address,
                                     kTransferCBVRegisterHostDepthAddress, 0, dxbc::Src::kXXXX));
              // r0.z = tile index relative to the host depth source base, or
              //        the tile index within the host depth source minus the
              //        EDRAM tile count if transferring across addressing
              //        wrapping (if negative)
              // r0.w = free
              a.OpIAdd(dxbc::Dest::R(0, 0b0100), dxbc::Src::R(0, dxbc::Src::kZZZZ),
                       dxbc::Src::R(0, dxbc::Src::kWWWW));
              // r0.z = tile index relative to the host depth source base
              a.OpAnd(dxbc::Dest::R(0, 0b0100), dxbc::Src::R(0, dxbc::Src::kZZZZ),
                      dxbc::Src::LU(xenos::kEdramTileCount - 1));
              // Convert position and sample index from within the destination
              // tile to within the host depth source tile, like for the guest
              // render target, but for 32bpp -> 32bpp only.
              dxbc::Src host_depth_source_sample(dest_sample);
              if (key.host_depth_source_msaa_samples != key.dest_msaa_samples) {
                if (key.host_depth_source_msaa_samples >= xenos::MsaaSamples::k4X) {
                  // 4x -> 1x/2x.
                  if (key.dest_msaa_samples == xenos::MsaaSamples::k2X) {
                    // 4x -> 2x.
                    // Horizontal pixels to samples. Vertical sample (1, 0 in
                    // the first bit for native 2x or 0, 1 in the second bit for
                    // 2x as 4x) to second sample bit.
                    host_depth_source_sample = dxbc::Src::R(0, dxbc::Src::kWWWW);
                    if (msaa_2x_supported_) {
                      a.OpBFI(dxbc::Dest::R(0, 0b1000), dxbc::Src::LU(31), dxbc::Src::LU(1),
                              dest_sample, dxbc::Src::R(0, dxbc::Src::kXXXX));
                      a.OpXOr(dxbc::Dest::R(0, 0b1000), host_depth_source_sample,
                              dxbc::Src::LU(1 << 1));
                    } else {
                      a.OpBFI(dxbc::Dest::R(0, 0b1000), dxbc::Src::LU(1), dxbc::Src::LU(0),
                              dxbc::Src::R(0, dxbc::Src::kXXXX), dest_sample);
                    }
                    a.OpUShR(dxbc::Dest::R(0, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX),
                             dxbc::Src::LU(1));
                  } else {
                    // 4x -> 1x.
                    // Pixels to samples.
                    a.OpAnd(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kXXXX),
                            dxbc::Src::LU(1));
                    host_depth_source_sample = dxbc::Src::R(0, dxbc::Src::kWWWW);
                    a.OpBFI(dxbc::Dest::R(0, 0b1000), dxbc::Src::LU(1), dxbc::Src::LU(1),
                            dxbc::Src::R(0, dxbc::Src::kYYYY), host_depth_source_sample);
                    a.OpUShR(dxbc::Dest::R(0, 0b0011), dxbc::Src::R(0), dxbc::Src::LU(1));
                  }
                } else {
                  // 1x/2x -> 1x/2x/4x (as long as they're different).
                  // Only the X part - Y is handled by common code.
                  if (key.dest_msaa_samples >= xenos::MsaaSamples::k4X) {
                    // Horizontal samples to pixels.
                    a.OpBFI(dxbc::Dest::R(0, 0b0001), dxbc::Src::LU(31), dxbc::Src::LU(1),
                            dxbc::Src::R(0, dxbc::Src::kXXXX), dest_sample);
                  }
                }
                // Host depth source Y and sample index for 1x/2x AA sources.
                if (key.host_depth_source_msaa_samples < xenos::MsaaSamples::k4X) {
                  if (key.dest_msaa_samples >= xenos::MsaaSamples::k4X) {
                    // 1x/2x -> 4x.
                    if (key.host_depth_source_msaa_samples == xenos::MsaaSamples::k2X) {
                      // 2x -> 4x.
                      // Vertical samples (second bit) of 4x destination to
                      // vertical sample (1, 0 for native 2x, or 0, 3 for 2x as
                      // 4x) of 2x source.
                      a.OpUShR(dxbc::Dest::R(0, 0b1000), dest_sample, dxbc::Src::LU(1));
                      host_depth_source_sample = dxbc::Src::R(0, dxbc::Src::kWWWW);
                      if (msaa_2x_supported_) {
                        a.OpXOr(dxbc::Dest::R(0, 0b1000), host_depth_source_sample,
                                dxbc::Src::LU(1));
                      } else {
                        a.OpBFI(dxbc::Dest::R(0, 0b1000), dxbc::Src::LU(1), dxbc::Src::LU(1),
                                host_depth_source_sample, host_depth_source_sample);
                      }
                    } else {
                      // 1x -> 4x.
                      // Vertical samples (second bit) to Y pixels, using r0.w
                      // (not needed without source MSAA) as a temporary.
                      a.OpUShR(dxbc::Dest::R(0, 0b1000), dest_sample, dxbc::Src::LU(1));
                      a.OpBFI(dxbc::Dest::R(0, 0b0010), dxbc::Src::LU(31), dxbc::Src::LU(1),
                              dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::R(0, dxbc::Src::kWWWW));
                    }
                  } else {
                    // 1x/2x -> different 1x/2x.
                    if (key.host_depth_source_msaa_samples == xenos::MsaaSamples::k2X) {
                      // 2x -> 1x.
                      // Vertical pixels of 2x destination to vertical samples
                      // (1, 0 for native 2x, or 0, 3 for 2x as 4x) of 1x
                      // source.
                      a.OpAnd(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kYYYY),
                              dxbc::Src::LU(1));
                      host_depth_source_sample = dxbc::Src::R(0, dxbc::Src::kWWWW);
                      if (msaa_2x_supported_) {
                        a.OpXOr(dxbc::Dest::R(0, 0b1000), host_depth_source_sample,
                                dxbc::Src::LU(1));
                      } else {
                        a.OpBFI(dxbc::Dest::R(0, 0b1000), dxbc::Src::LU(1), dxbc::Src::LU(1),
                                host_depth_source_sample, host_depth_source_sample);
                      }
                      a.OpUShR(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY),
                               dxbc::Src::LU(1));
                    } else {
                      // 1x -> 2x.
                      // Vertical samples (1, 0 in the first bit for native 2x
                      // or 0, 1 in the second bit for 2x as 4x) of 2x
                      // destination to vertical pixels of 1x source.
                      // Using r0.w (not needed without source MSAA) as a
                      // temporary.
                      if (msaa_2x_supported_) {
                        a.OpBFI(dxbc::Dest::R(0, 0b0010), dxbc::Src::LU(31), dxbc::Src::LU(1),
                                dxbc::Src::R(0, dxbc::Src::kYYYY), dest_sample);
                        a.OpXOr(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY),
                                dxbc::Src::LU(1));
                      } else {
                        a.OpUShR(dxbc::Dest::R(0, 0b1000), dest_sample, dxbc::Src::LU(1));
                        a.OpBFI(dxbc::Dest::R(0, 0b0010), dxbc::Src::LU(31), dxbc::Src::LU(1),
                                dxbc::Src::R(0, dxbc::Src::kYYYY),
                                dxbc::Src::R(0, dxbc::Src::kWWWW));
                      }
                    }
                  }
                }
              }
              // r1.x = host depth source pitch in tiles
              a.OpUBFE(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(xenos::kEdramPitchTilesBits),
                       dxbc::Src::LU(xenos::kEdramPitchTilesBits),
                       dxbc::Src::CB(cbuffer_index_host_depth_address,
                                     kTransferCBVRegisterHostDepthAddress, 0, dxbc::Src::kXXXX));
              // r0.z = host depth source tile row
              // r1.x = host depth source tile within the row
              a.OpUDiv(dxbc::Dest::R(0, 0b0100), dxbc::Dest::R(1, 0b0001),
                       dxbc::Src::R(0, dxbc::Src::kZZZZ), dxbc::Src::R(1, dxbc::Src::kXXXX));
              // r0.x = pixel X within the host depth source texture
              // r1.x = free
              a.OpUMAd(
                  dxbc::Dest::R(0, 0b0001),
                  dxbc::Src::LU(tile_width_samples >> uint32_t(key.host_depth_source_msaa_samples >=
                                                               xenos::MsaaSamples::k4X)),
                  dxbc::Src::R(1, dxbc::Src::kXXXX), dxbc::Src::R(0, dxbc::Src::kXXXX));
              // r0.y = pixel Y within the host depth source texture
              // r0.z = free
              a.OpUMAd(dxbc::Dest::R(0, 0b0010),
                       dxbc::Src::LU(
                           tile_height_samples >>
                           uint32_t(key.host_depth_source_msaa_samples >= xenos::MsaaSamples::k2X)),
                       dxbc::Src::R(0, dxbc::Src::kZZZZ), dxbc::Src::R(0, dxbc::Src::kYYYY));
              // Load from the host depth texture.
              if (key.host_depth_source_msaa_samples != xenos::MsaaSamples::k1X) {
                a.OpLdMS(dxbc::Dest::R(0, 0b0001), dxbc::Src::R(0), 0b0011,
                         dxbc::Src::T(srv_index_host_depth, kTransferSRVRegisterHostDepth,
                                      dxbc::Src::kXXXX),
                         host_depth_source_sample);
              } else {
                // Write zero to the LOD index in r0.z.
                a.OpMov(dxbc::Dest::R(0, 0b0100), dxbc::Src::LU(0));
                a.OpLd(dxbc::Dest::R(0, 0b0001), dxbc::Src::R(0, 0b10000100), 0b1011,
                       dxbc::Src::T(srv_index_host_depth, kTransferSRVRegisterHostDepth,
                                    dxbc::Src::kXXXX));
              }
            }
            // Convert the host depth value in r0.x to the guest format in r0.y
            // using r0.z as a temporary and check if it matches the value in
            // the currently owning guest render target.
            switch (dest_depth_format) {
              case xenos::DepthRenderTargetFormat::kD24S8: {
                // Round to the nearest even integer. This seems to be the
                // correct, adding +0.5 and rounding towards zero results in red
                // instead of black in the 4D5307E6 clear shader.
                a.OpMul(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(0, dxbc::Src::kXXXX),
                        dxbc::Src::LF(float(0xFFFFFF)));
                a.OpRoundNE(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY));
                a.OpFToU(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY));
              } break;
              case xenos::DepthRenderTargetFormat::kD24FS8: {
                // When converting the depth in pixel shaders, it's always
                // exact, truncating not to insert additional rounding
                // instructions.
                DxbcShaderTranslator::PreClampedDepthTo20e4(
                    a, 0, 1, 0, 0, 0, 2,
                    !depth_float24_convert_in_pixel_shader() && depth_float24_round(), true);
              } break;
            }
            a.OpIEq(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY),
                    dxbc::Src::R(1, dxbc::Src::kWWWW));
            a.OpIf(true, dxbc::Src::R(0, dxbc::Src::kYYYY));
            // If the host depth is up to date, write it to oDepth at the host
            // precision instead of converting the guest depth.
            a.OpMov(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(0, dxbc::Src::kXXXX));
            a.OpElse();
          }
          // Convert using r0.x as a temporary.
          switch (dest_depth_format) {
            case xenos::DepthRenderTargetFormat::kD24S8: {
              // Multiplying by 1.0 / 0xFFFFFF produces an incorrect result (for
              // 0xC00000, for instance - which is 2_10_10_10 clear to 0001) -
              // rescale from 0...0xFFFFFF to 0...0x1000000 doing what true
              // float division followed by multiplication does (on x86-64 MSVC
              // with default SSE rounding) - values starting from 0x800000
              // become bigger by 1; then accurately bias the result's exponent.
              a.OpUShR(dxbc::Dest::R(0, 0b0001), dxbc::Src::R(1, dxbc::Src::kWWWW),
                       dxbc::Src::LU(23));
              a.OpIAdd(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW),
                       dxbc::Src::R(0, dxbc::Src::kXXXX));
              a.OpUToF(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW));
              a.OpMul(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW),
                      dxbc::Src::LF(1.0f / float(1 << 24)));
            } break;
            case xenos::DepthRenderTargetFormat::kD24FS8: {
              DxbcShaderTranslator::Depth20e4To32(a, dxbc::Dest::R(1, 0b1000), 1, 3, 0, 1, 3, 0, 0,
                                                  true);
            } break;
          }
          // Host depth is different, or not available - convert the guest depth
          // to the destination format.
          if (rs & kTransferUsedRootParameterHostDepthSRVBit) {
            // Close the conditional for the host / guest depth.
            a.OpEndIf();
          }
        }
        a.OpMov(dxbc::Dest::ODepth(), dxbc::Src::R(1, dxbc::Src::kWWWW));
        break;
      case TransferOutput::kStencilBit:
        // Discard the sample if the needed stencil bit is not set.
        assert_true(cbuffer_index_stencil_mask != UINT32_MAX);
        a.OpAnd(dxbc::Dest::R(0, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX),
                dxbc::Src::CB(cbuffer_index_stencil_mask, kTransferCBVRegisterStencilMask, 0,
                              dxbc::Src::kXXXX));
        a.OpDiscard(false, dxbc::Src::R(0, dxbc::Src::kXXXX));
        break;
    }
  }

  if (dest_is_color) {
    // Fill the unused components of the color result.
    uint32_t dest_color_component_count =
        xenos::GetColorRenderTargetFormatComponentCount(dest_color_format);
    uint32_t dest_color_unwritten_mask = 0b1111 & ~uint32_t((1 << dest_color_component_count) - 1);
    if (dest_color_component_count < 4) {
      a.OpMov(dxbc::Dest::O(0, dest_color_unwritten_mask), dxbc::Src::LU(0));
    }
  }

  a.OpRet();

  // Write the shader program length in dwords.
  built_shader_[shex_position_dwords + 1] = uint32_t(built_shader_.size()) - shex_position_dwords;

  {
    auto& blob_header =
        *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
    blob_header.fourcc = dxbc::BlobHeader::FourCC::kShaderEx;
    blob_position_dwords = uint32_t(built_shader_.size());
    blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                             built_shader_[blob_offset_position_dwords++];
  }

  // ***************************************************************************
  // Shader feature info
  // ***************************************************************************

  if (shader_uses_stencil_reference_output) {
    built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
    uint32_t sfi0_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
    built_shader_.resize(sfi0_position_dwords + sizeof(dxbc::ShaderFeatureInfo) / sizeof(uint32_t));
    auto& shader_feature_info =
        *reinterpret_cast<dxbc::ShaderFeatureInfo*>(built_shader_.data() + sfi0_position_dwords);
    shader_feature_info.feature_flags[0] |= dxbc::kShaderFeature0_StencilRef;
    {
      auto& blob_header =
          *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
      blob_header.fourcc = dxbc::BlobHeader::FourCC::kShaderFeatureInfo;
      blob_position_dwords = uint32_t(built_shader_.size());
      blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                               built_shader_[blob_offset_position_dwords++];
    }
  }

  // ***************************************************************************
  // Statistics
  // ***************************************************************************

  built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
  uint32_t stat_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
  built_shader_.resize(stat_position_dwords + sizeof(dxbc::Statistics) / sizeof(uint32_t));
  std::memcpy(built_shader_.data() + stat_position_dwords, &stat, sizeof(dxbc::Statistics));
  {
    auto& blob_header =
        *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
    blob_header.fourcc = dxbc::BlobHeader::FourCC::kStatistics;
    blob_position_dwords = uint32_t(built_shader_.size());
    blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                             built_shader_[blob_offset_position_dwords++];
  }

  // ***************************************************************************
  // Container header
  // ***************************************************************************

  uint32_t built_shader_size_bytes = uint32_t(built_shader_.size() * sizeof(uint32_t));
  {
    auto& container_header = *reinterpret_cast<dxbc::ContainerHeader*>(built_shader_.data());
    container_header.InitializeIdentification();
    container_header.size_bytes = built_shader_size_bytes;
    container_header.blob_count = blob_count;
    CalculateDXBCChecksum(reinterpret_cast<unsigned char*>(built_shader_.data()),
                          static_cast<unsigned int>(built_shader_size_bytes),
                          reinterpret_cast<unsigned int*>(&container_header.hash));
  }

  // ***************************************************************************
  // Pipeline
  // ***************************************************************************

  ID3D12PipelineState* const* pipelines;
  D3D12_INPUT_ELEMENT_DESC pipeline_input_element_desc;
  pipeline_input_element_desc.SemanticName = "POSITION";
  pipeline_input_element_desc.SemanticIndex = 0;
  pipeline_input_element_desc.Format = DXGI_FORMAT_R32G32_FLOAT;
  pipeline_input_element_desc.InputSlot = 0;
  pipeline_input_element_desc.AlignedByteOffset = 0;
  pipeline_input_element_desc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
  pipeline_input_element_desc.InstanceDataStepRate = 0;
  D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline_desc = {};
  pipeline_desc.pRootSignature = transfer_root_signatures_[size_t(
      use_stencil_reference_output_ ? mode.root_signature_with_stencil_ref
                                    : mode.root_signature_no_stencil_ref)];
  pipeline_desc.VS.pShaderBytecode = shaders::passthrough_position_xy_vs;
  pipeline_desc.VS.BytecodeLength = sizeof(shaders::passthrough_position_xy_vs);
  pipeline_desc.PS.pShaderBytecode = built_shader_.data();
  pipeline_desc.PS.BytecodeLength = built_shader_size_bytes;
  if (key.dest_msaa_samples == xenos::MsaaSamples::k2X && !msaa_2x_supported_) {
    // Using sample 0 as 0 and 3 as 1 for 2x instead.
    pipeline_desc.SampleMask = 0b1001;
    pipeline_desc.SampleDesc.Count = 4;
  } else {
    pipeline_desc.SampleMask = UINT_MAX;
    pipeline_desc.SampleDesc.Count = UINT(1) << UINT(key.dest_msaa_samples);
  }
  pipeline_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
  pipeline_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
  pipeline_desc.RasterizerState.DepthClipEnable = TRUE;
  pipeline_desc.InputLayout.pInputElementDescs = &pipeline_input_element_desc;
  pipeline_desc.InputLayout.NumElements = 1;
  pipeline_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  if (dest_is_stencil_bit) {
    pipeline_desc.DepthStencilState.StencilEnable = TRUE;
    pipeline_desc.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    pipeline_desc.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    pipeline_desc.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_REPLACE;
    pipeline_desc.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    pipeline_desc.DepthStencilState.BackFace = pipeline_desc.DepthStencilState.FrontFace;
    pipeline_desc.DSVFormat = GetDepthDSVDXGIFormat(dest_depth_format);
    // Even if creation fails, still store the null pointers not to try to
    // create again.
    std::array<ID3D12PipelineState*, 8>& stencil_bit_pipelines =
        transfer_stencil_bit_pipelines_
            .emplace(std::piecewise_construct, std::make_tuple(key), std::make_tuple())
            .first->second;
    bool stencil_pipelines_created = true;
    for (uint32_t i = 0; i < 8; ++i) {
      pipeline_desc.DepthStencilState.StencilWriteMask = UINT8(1) << i;
      // [PSO-LIB] Draw-time CP-thread create; route through the driver-blob
      // library (also counts into the [hitch] probe).
      stencil_bit_pipelines[i] =
          command_processor_.CreateGraphicsPipelineWithLibrary(pipeline_desc);
      if (stencil_bit_pipelines[i]) {
        continue;
      }
      stencil_pipelines_created = false;
      for (uint32_t j = 0; j < i; ++j) {
        stencil_bit_pipelines[j]->Release();
        stencil_bit_pipelines[j] = nullptr;
      }
      break;
    }
    pipelines = stencil_pipelines_created ? stencil_bit_pipelines.data() : nullptr;
  } else {
    if (dest_is_color) {
      pipeline_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
      pipeline_desc.NumRenderTargets = 1;
      pipeline_desc.RTVFormats[0] = GetColorOwnershipTransferDXGIFormat(dest_color_format);
    } else {
      pipeline_desc.DepthStencilState.DepthEnable = TRUE;
      pipeline_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
      pipeline_desc.DepthStencilState.DepthFunc = REXCVAR_GET(depth_transfer_not_equal_test)
                                                      ? D3D12_COMPARISON_FUNC_NOT_EQUAL
                                                      : D3D12_COMPARISON_FUNC_ALWAYS;
      if (use_stencil_reference_output_) {
        pipeline_desc.DepthStencilState.StencilEnable = TRUE;
        pipeline_desc.DepthStencilState.StencilWriteMask = UINT8_MAX;
        pipeline_desc.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
        pipeline_desc.DepthStencilState.FrontFace.StencilDepthFailOp =
            REXCVAR_GET(depth_transfer_not_equal_test) ? D3D12_STENCIL_OP_REPLACE
                                                       : D3D12_STENCIL_OP_KEEP;
        pipeline_desc.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_REPLACE;
        // Using ALWAYS, not NOT_EQUAL, so depth writing is unaffected by
        // stencil being different.
        pipeline_desc.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        pipeline_desc.DepthStencilState.BackFace = pipeline_desc.DepthStencilState.FrontFace;
      }
      pipeline_desc.DSVFormat = GetDepthDSVDXGIFormat(dest_depth_format);
    }
    // [PSO-LIB] Draw-time CP-thread create; route through the driver-blob
    // library (also counts into the [hitch] probe). Null on failure.
    ID3D12PipelineState* pipeline = command_processor_.CreateGraphicsPipelineWithLibrary(pipeline_desc);
    // Even if creation fails, still store the null pointer not to try to create
    // again.
    // Return a pointer to the persistent location.
    ID3D12PipelineState*& inserted_pipeline =
        transfer_pipelines_.emplace(key, pipeline).first->second;
    pipelines = inserted_pipeline ? &inserted_pipeline : nullptr;
  }
  // TODO(Triang3l): Pipeline state name debug names (lots of variables - but
  // not very important since everything can be derived from the bindings and
  // outputs in a debugger).

  if (!pipelines) {
    // Stencil bit copying uses only the stencil SRV for depth / stencil source,
    // can't use srv_index_depth for checking.
    const char* source_format_name =
        (rs & kTransferUsedRootParameterColorSRVBit)
            ? xenos::GetColorRenderTargetFormatName(source_color_format)
            : xenos::GetDepthRenderTargetFormatName(source_depth_format);
    const char* dest_format_name = mode.output == TransferOutput::kColor
                                       ? xenos::GetColorRenderTargetFormatName(dest_color_format)
                                       : xenos::GetDepthRenderTargetFormatName(dest_depth_format);
    if (srv_index_host_depth != UINT32_MAX) {
      REXGPU_ERROR(
          "D3D12RenderTargetCache: Failed to create a render target ownership "
          "transfer pipeline for {}-sample {} + {}-sample host depth{} -> "
          "{}-sample {} for mode {}",
          uint32_t(1) << uint32_t(key.source_msaa_samples), source_format_name,
          uint32_t(1) << uint32_t(key.host_depth_source_msaa_samples),
          key.host_depth_source_is_copy ? " copy" : "",
          uint32_t(1) << uint32_t(key.dest_msaa_samples), dest_format_name, uint32_t(key.mode));
    } else {
      REXGPU_ERROR(
          "D3D12RenderTargetCache: Failed to create a render target ownership "
          "transfer pipeline for {}-sample {} -> {}-sample {} for mode {}",
          uint32_t(1) << uint32_t(key.source_msaa_samples), source_format_name,
          uint32_t(1) << uint32_t(key.dest_msaa_samples), dest_format_name, uint32_t(key.mode));
    }
  }
  return pipelines;
}

D3D12RenderTargetCache::NativeHostDepthScratch*
D3D12RenderTargetCache::GetOrCreateNativeHostDepthScratch(D3D12RenderTarget& dest_rt) {
  D3D12_RESOURCE_DESC desc = dest_rt.resource()->GetDesc();
  const uint64_t key = (uint64_t(desc.Format) << 48) ^ (uint64_t(desc.SampleDesc.Count) << 40) ^
                       (uint64_t(desc.Width) << 16) ^ uint64_t(desc.Height);
  auto it = native_hds_scratch_.find(key);
  if (it != native_hds_scratch_.end()) {
    // Null resource = a cached creation failure; the caller falls back.
    return it->second.resource ? &it->second : nullptr;
  }
  NativeHostDepthScratch& scratch = native_hds_scratch_[key];
  // Reusing the live resource's own desc keeps the CopyResource pair matched
  // exactly. The scratch is never a DSV, but MSAA textures must carry one of
  // the RTV/DSV flags, so ALLOW_DEPTH_STENCIL stays.
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             D3D12_HEAP_FLAG_NONE, &desc,
                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&resource)))) {
    REXGPU_ERROR("[nr-xfer] Failed to create the native host-depth scratch texture");
    return nullptr;
  }
  resource->SetName(L"NR Native Host Depth Scratch");
  ui::d3d12::D3D12CpuDescriptorPool::Descriptor descriptor_srv =
      descriptor_pool_srv_->AllocateDescriptor();
  if (!descriptor_srv.IsValid()) {
    return nullptr;
  }
  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
  srv_desc.Format = GetDepthSRVDepthDXGIFormat(dest_rt.key().GetDepthFormat());
  srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  if (desc.SampleDesc.Count > 1) {
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
  } else {
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.PlaneSlice = 0;
    srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
  }
  device->CreateShaderResourceView(resource.Get(), &srv_desc, descriptor_srv.GetHandle());
  scratch.resource = std::move(resource);
  scratch.descriptor_srv = std::move(descriptor_srv);
  scratch.state = D3D12_RESOURCE_STATE_COPY_DEST;
  return &scratch;
}

// [xfer] The transfer read census. Records are few (tens live at once: one
// per transfer landed and not yet finalized), so linear scans are fine.
static constexpr size_t kXferUseRecordCap = 1024;

void D3D12RenderTargetCache::XferUseFinalize(RenderTargetKey key, uint32_t start_tiles,
                                             uint32_t end_tiles, XferUseOutcome pending_outcome,
                                             XferUseOutcome written_outcome) {
  for (size_t i = 0; i < xfer_use_records_.size();) {
    XferUseRecord& r = xfer_use_records_[i];
    if (r.dest == key && r.start_tiles < end_tiles && start_tiles < r.end_tiles) {
      const XferUseOutcome outcome = r.written ? written_outcome : pending_outcome;
      ++xfer_use_counts_[r.cls][outcome];
      auto& by_key = xfer_use_by_key_[r.dest.key];
      ++by_key[outcome];
      if (r.from_cleared) {
        ++by_key[kXferUseOutcomeCount];
      }
      r = xfer_use_records_.back();
      xfer_use_records_.pop_back();
      continue;
    }
    ++i;
  }
}

void D3D12RenderTargetCache::XferUseAdd(RenderTargetKey dest, const Transfer& transfer) {
  const RenderTargetKey source = transfer.source ? transfer.source->key() : RenderTargetKey();
  // The source loses these tiles: whatever landed in it there was never read.
  XferUseFinalize(source, transfer.start_tiles, transfer.end_tiles, kXferUseTaken,
                  kXferUseWrittenTaken);
  XferUseFinalize(dest, transfer.start_tiles, transfer.end_tiles, kXferUseTaken,
                  kXferUseWrittenTaken);
  if (xfer_use_records_.size() >= kXferUseRecordCap) {
    ++xfer_use_counts_[xfer_use_records_.front().cls][kXferUseEvicted];
    xfer_use_records_.front() = xfer_use_records_.back();
    xfer_use_records_.pop_back();
  }
  XferUseRecord r;
  r.dest = dest;
  r.start_tiles = transfer.start_tiles;
  r.end_tiles = transfer.end_tiles;
  // c2c 0, d2c 1, c2d 2, d2d 3.
  r.cls = (dest.is_depth ? 2u : 0u) | (source.is_depth ? 1u : 0u);
  r.written = false;
  r.from_cleared = false;
  for (const XferClearedRange& c : xfer_cleared_ranges_) {
    if (c.key == source && c.start_tiles <= transfer.start_tiles &&
        transfer.end_tiles <= c.end_tiles) {
      r.from_cleared = true;
      ++xfer_use_const_[r.cls];
      break;
    }
  }
  xfer_use_records_.push_back(r);
}

void D3D12RenderTargetCache::XferUseNoteDraw(RenderTargetKey key, bool reads_dest) {
  for (size_t i = 0; i < xfer_cleared_ranges_.size();) {
    if (xfer_cleared_ranges_[i].key == key) {
      xfer_cleared_ranges_[i] = xfer_cleared_ranges_.back();
      xfer_cleared_ranges_.pop_back();
      continue;
    }
    ++i;
  }
  if (reads_dest) {
    XferUseFinalize(key, 0, xenos::kEdramTileCount, kXferUseReadDraw, kXferUseWrittenReadDraw);
    return;
  }
  for (XferUseRecord& r : xfer_use_records_) {
    if (r.dest == key) {
      r.written = true;
    }
  }
}

void D3D12RenderTargetCache::XferUseNoteClear(RenderTargetKey key,
                                              const Transfer::Rectangle& rectangle) {
  const uint32_t rows_per_tile_row =
      xenos::kEdramTileHeightSamples >> uint32_t(key.msaa_samples >= xenos::MsaaSamples::k2X);
  const uint32_t pitch_tiles = key.GetPitchTiles();
  const uint32_t start_tiles =
      key.base_tiles + (rectangle.y_pixels / rows_per_tile_row) * pitch_tiles;
  const uint32_t end_tiles =
      key.base_tiles +
      ((rectangle.y_pixels + rectangle.height_pixels + rows_per_tile_row - 1) / rows_per_tile_row) *
          pitch_tiles;
  XferUseFinalize(key, start_tiles, end_tiles, kXferUseCleared, kXferUseCleared);
  if (xfer_cleared_ranges_.size() < 64) {
    xfer_cleared_ranges_.push_back({key, start_tiles, end_tiles});
  }
}

void D3D12RenderTargetCache::XferUseNoteResolveRead(RenderTargetKey key, uint32_t start_tiles,
                                                    uint32_t end_tiles) {
  XferUseFinalize(key, start_tiles, end_tiles, kXferUseReadResolve, kXferUseWrittenResolve);
}

void D3D12RenderTargetCache::XferUseReport() {
  static const char* const kClassNames[kXferUseClassCount] = {"c2c", "d2c", "c2d", "d2d"};
  std::string per_class;
  for (uint32_t c = 0; c < kXferUseClassCount; ++c) {
    uint64_t d[kXferUseOutcomeCount];
    uint64_t total = 0;
    for (uint32_t o = 0; o < kXferUseOutcomeCount; ++o) {
      d[o] = xfer_use_counts_[c][o] - xfer_use_counts_last_[c][o];
      xfer_use_counts_last_[c][o] = xfer_use_counts_[c][o];
      total += d[o];
    }
    const uint64_t k = xfer_use_const_[c] - xfer_use_const_last_[c];
    xfer_use_const_last_[c] = xfer_use_const_[c];
    per_class += fmt::format(
        " | {} {} taken/wrtaken/clr/rd/wrrd/rres/wrres/ev={}/{}/{}/{}/{}/{}/{}/{} const={}",
        kClassNames[c], total, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], k);
  }
  if (!xfer_use_by_key_.empty()) {
    std::string per_key;
    for (const auto& kv : xfer_use_by_key_) {
      RenderTargetKey key;
      key.key = kv.first;
      const auto& v = kv.second;
      per_key += fmt::format(" | {}{}t/p{}/m{}: {}/{}/{}/{}/{}/{}/{}/{} const={}",
                             key.is_depth ? "Z" : "C", uint32_t(key.base_tiles),
                             uint32_t(key.pitch_tiles_at_32bpp), uint32_t(key.msaa_samples), v[0],
                             v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
    }
    REXGPU_INFO("[xfer-key] taken/wrtaken/clr/rd/wrrd/rres/wrres/ev{}", per_key);
    xfer_use_by_key_.clear();
  }
  REXGPU_INFO("[xfer] phase={} passes/s={} exec/s={} skip/s={} live={}{}",
              xfer_cycle_phase_ ? "skip" : "all",
              xfer_use_work_passes_ - xfer_use_work_passes_last_,
              xfer_use_executed_ - xfer_use_executed_last_,
              xfer_use_skipped_ - xfer_use_skipped_last_, xfer_use_records_.size(), per_class);
  xfer_use_work_passes_last_ = xfer_use_work_passes_;
  xfer_use_executed_last_ = xfer_use_executed_;
  xfer_use_skipped_last_ = xfer_use_skipped_;
}

void D3D12RenderTargetCache::PerformTransfersAndResolveClears(
    uint32_t render_target_count, RenderTarget* const* render_targets,
    const std::vector<Transfer>* render_target_transfers,
    const uint64_t* render_target_resolve_clear_values,
    const Transfer::Rectangle* resolve_clear_rectangle) {
  assert_true(GetPath() == Path::kHostRenderTargets);

  // [xfer] The cycler, the read census records, and the skip itself. The skip
  // substitutes empty transfer lists: nothing below copies, the resolve clear
  // (if any) still runs, ownership was already claimed by the caller.
  {
    const int32_t cycle_seconds = REXCVAR_GET(gpu_xfer_cycle);
    const auto now = std::chrono::steady_clock::now();
    if (cycle_seconds > 0) {
      if (xfer_cycle_phase_start_ == std::chrono::steady_clock::time_point{}) {
        xfer_cycle_phase_start_ = now;
      } else if (now - xfer_cycle_phase_start_ >= std::chrono::seconds(cycle_seconds)) {
        xfer_cycle_phase_start_ = now;
        xfer_cycle_phase_ ^= 1;
        REXGPU_INFO("[xfer-cyc] phase={}", xfer_cycle_phase_ ? "skip" : "all");
      }
    } else if (xfer_cycle_phase_) {
      xfer_cycle_phase_ = 0;
      REXGPU_INFO("[xfer-cyc] phase=all (cycler off)");
    }
    if (render_target_transfers) {
      for (uint32_t i = 0; i < render_target_count; ++i) {
        if (!render_targets[i]) {
          continue;
        }
        const RenderTargetKey dest_key = render_targets[i]->key();
        for (const Transfer& transfer : render_target_transfers[i]) {
          XferUseAdd(dest_key, transfer);
          if (xfer_cycle_phase_) {
            ++xfer_use_skipped_;
          } else {
            ++xfer_use_executed_;
          }
        }
      }
      if (xfer_cycle_phase_) {
        static const std::vector<Transfer> kNoTransfers[1 + xenos::kMaxColorRenderTargets];
        assert_true(render_target_count <= 1 + xenos::kMaxColorRenderTargets);
        render_target_transfers = kNoTransfers;
      }
    }
    if (render_target_resolve_clear_values && resolve_clear_rectangle) {
      for (uint32_t i = 0; i < render_target_count; ++i) {
        if (render_targets[i]) {
          XferUseNoteClear(render_targets[i]->key(), *resolve_clear_rectangle);
        }
      }
    }
  }

  // [NR-DETILE] N-6. The resolve path measured correct end to end, so the
  // mod-512 fold must already be in the render target when the resolve reads
  // it. Ownership transfers are what write a render target other than
  // rasterisation, so name every one: which tiles, into which target, out of
  // which source. A transfer landing on dest rows 512..720 from source rows
  // 0..208 is the bug. Deduped by signature so no shape hides behind a busier
  // one, and reported even when the list is EMPTY for a target that has one -
  // silence has been mistaken for absence twice in this hunt already.
  if (REXCVAR_GET(gpu_nr_dump_probe) && render_target_transfers) {
    for (uint32_t ti = 0; ti < render_target_count; ++ti) {
      const RenderTarget* dest_rt = render_targets[ti];
      if (!dest_rt || render_target_transfers[ti].empty()) {
        continue;
      }
      RenderTargetKey dk = dest_rt->key();
      // Only the tiled pass shape is interesting here: 4xMSAA, 32-tile pitch.
      const uint64_t tf_sig = (uint64_t(dk.base_tiles) << 40) ^
                              (uint64_t(dk.pitch_tiles_at_32bpp) << 24) ^
                              (uint64_t(dk.msaa_samples) << 16) ^
                              (uint64_t(dk.is_depth) << 8) ^
                              uint64_t(render_target_transfers[ti].size());
      static std::map<uint64_t, std::chrono::steady_clock::time_point> tf_seen;
      const auto tf_now = std::chrono::steady_clock::now();
      auto tf_it = tf_seen.find(tf_sig);
      if (tf_it != tf_seen.end() && tf_now - tf_it->second < std::chrono::seconds(1)) {
        continue;
      }
      if (tf_seen.size() > 64) {
        tf_seen.clear();
      }
      tf_seen[tf_sig] = tf_now;
      const uint32_t dest_rows_per_tile_row =
          xenos::kEdramTileHeightSamples >>
          uint32_t(dk.msaa_samples >= xenos::MsaaSamples::k2X);
      std::string tf;
      for (const Transfer& t : render_target_transfers[ti]) {
        RenderTargetKey sk = t.source ? t.source->key() : RenderTargetKey();
        // Destination rows this tile range lands on, in the dest RT's own
        // layout - the number that has to be compared against 512..720.
        const uint32_t dr0 = dk.pitch_tiles_at_32bpp
                                 ? (t.start_tiles - std::min(t.start_tiles, uint32_t(dk.base_tiles))) /
                                       dk.pitch_tiles_at_32bpp * dest_rows_per_tile_row
                                 : 0;
        const uint32_t dr1 = dk.pitch_tiles_at_32bpp
                                 ? (t.end_tiles - std::min(t.end_tiles, uint32_t(dk.base_tiles))) /
                                       dk.pitch_tiles_at_32bpp * dest_rows_per_tile_row
                                 : 0;
        tf += fmt::format(" [{}..{}t -> destrows {}..{} <- src base={}t pitch={}t msaa={} depth={}]",
                          t.start_tiles, t.end_tiles, dr0, dr1, uint32_t(sk.base_tiles),
                          uint32_t(sk.pitch_tiles_at_32bpp), uint32_t(sk.msaa_samples),
                          uint32_t(sk.is_depth));
      }
      REXGPU_INFO("[nr-detile] xfer: dest base={}t pitch={}t msaa={} depth={} | {} transfer(s):{}",
                  uint32_t(dk.base_tiles), uint32_t(dk.pitch_tiles_at_32bpp),
                  uint32_t(dk.msaa_samples), uint32_t(dk.is_depth),
                  render_target_transfers[ti].size(), tf);
    }
  }

  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  uint64_t current_submission = command_processor_.GetCurrentSubmission();
  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();

  bool resolve_clear_needed = render_target_resolve_clear_values && resolve_clear_rectangle;

  // [gpu-census] this function runs ~once per draw and almost always records
  // nothing (the 52k/s "passes" are calls, not work) - only bracket the GPU
  // class when there IS work, or two timestamps per draw would be emitted.
  bool gpu_census_work = resolve_clear_needed;
  if (!gpu_census_work && render_target_transfers) {
    for (uint32_t i = 0; i < render_target_count; ++i) {
      if (render_targets[i] && !render_target_transfers[i].empty()) {
        gpu_census_work = true;
        break;
      }
    }
  }
  // [hiz] anything that writes a render target other than rasterization
  // invalidates an open Hi-Z window (the checkpoint's depth may have moved).
  if (gpu_census_work) {
    ++transfer_epoch_;
    ++xfer_use_work_passes_;  // [xfer]
  }
  std::optional<D3D12CommandProcessor::GpuCensusScope> gpu_census_scope;
  if (gpu_census_work) {
    gpu_census_scope.emplace(command_processor_, D3D12CommandProcessor::kGpuCensusXfer);
  }

  D3D12_RECT clear_rect;
  if (resolve_clear_needed) {
    // Assuming the rectangle is already clamped by the setup function from the
    // common render target cache.
    clear_rect.left = LONG(resolve_clear_rectangle->x_pixels * draw_resolution_scale_x());
    clear_rect.top = LONG(resolve_clear_rectangle->y_pixels * draw_resolution_scale_y());
    clear_rect.right =
        LONG((resolve_clear_rectangle->x_pixels + resolve_clear_rectangle->width_pixels) *
             draw_resolution_scale_x());
    clear_rect.bottom =
        LONG((resolve_clear_rectangle->y_pixels + resolve_clear_rectangle->height_pixels) *
             draw_resolution_scale_y());
  }

  // [NR-XFER] N-10b, legacy store DELETED (gated naruto_627: hds native
  // 7189 / legacy 0 over a full city drive, no fatals, no visual issues).
  // When the depth destination is its own host depth source, snapshot its
  // depth plane into a scratch TEXTURE with one CopyResource; the transfer
  // shader takes the ordinary host-depth texture variant with identity
  // addressing (the source key IS the dest key). No EDRAM buffer, no
  // compute store. If scratch creation ever fails (cached, effectively
  // never), the key builder below degrades those transfers to guest-depth
  // only and counts them in the census.
  bool native_hds_used = false;
  D3D12_CPU_DESCRIPTOR_HANDLE native_hds_srv_handle = {};
  for (uint32_t i = 0; i < render_target_count; ++i) {
    RenderTarget* dest_rt = render_targets[i];
    if (!dest_rt) {
      continue;
    }
    auto& dest_d3d12_rt = *static_cast<D3D12RenderTarget*>(dest_rt);
    if (!dest_d3d12_rt.key().is_depth) {
      continue;
    }
    const std::vector<Transfer>& depth_transfers = render_target_transfers[i];
    for (const Transfer& transfer : depth_transfers) {
      if (transfer.host_depth_source != dest_rt) {
        continue;
      }
      NativeHostDepthScratch* scratch = GetOrCreateNativeHostDepthScratch(dest_d3d12_rt);
      if (scratch) {
        command_processor_.PushTransitionBarrier(
            dest_d3d12_rt.resource(),
            dest_d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_COPY_SOURCE),
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        if (scratch->state != D3D12_RESOURCE_STATE_COPY_DEST) {
          command_processor_.PushTransitionBarrier(scratch->resource.Get(), scratch->state,
                                                   D3D12_RESOURCE_STATE_COPY_DEST);
          scratch->state = D3D12_RESOURCE_STATE_COPY_DEST;
        }
        command_processor_.SubmitBarriers();
        command_list.D3DCopyResource(scratch->resource.Get(), dest_d3d12_rt.resource());
        command_processor_.PushTransitionBarrier(scratch->resource.Get(), scratch->state,
                                                 D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        scratch->state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        native_hds_srv_handle = scratch->descriptor_srv.GetHandle();
        native_hds_used = true;
        ++xfer_census_hds_native_;
      } else {
        ++xfer_census_hds_fail_;
      }
      // One whole-resource snapshot covers every transfer rectangle.
      break;
    }
    break;
  }

  // Try to insert as many barriers as possible in one place, hoping that in the
  // best case (no cross-copying between current render targets), barriers will
  // need to be only inserted here, not between transfers. In case of
  // cross-copying, if the destination use is going to happen before the source
  // use, choose the destination state, otherwise the source state - to match
  // the order in which transfers will actually happen (otherwise there will be
  // just a useless switch back and forth).
  for (uint32_t i = 0; i < render_target_count; ++i) {
    RenderTarget* dest_rt = render_targets[i];
    if (!dest_rt) {
      continue;
    }
    auto& dest_d3d12_rt = *static_cast<D3D12RenderTarget*>(dest_rt);
    const std::vector<Transfer>& dest_transfers = render_target_transfers[i];
    if (!resolve_clear_needed && dest_transfers.empty()) {
      continue;
    }
    // Transition the sources, only if not going to be used as destinations
    // earlier.
    for (const Transfer& transfer : render_target_transfers[i]) {
      bool source_previously_used_as_dest = false;
      bool host_depth_source_previously_used_as_dest = false;
      for (uint32_t j = 0; j < i; ++j) {
        if (render_target_transfers[j].empty()) {
          continue;
        }
        const RenderTarget* previous_rt = render_targets[j];
        if (transfer.source == previous_rt) {
          source_previously_used_as_dest = true;
        }
        if (transfer.host_depth_source == previous_rt) {
          host_depth_source_previously_used_as_dest = true;
        }
      }
      if (!source_previously_used_as_dest) {
        auto& source_d3d12_rt = *static_cast<D3D12RenderTarget*>(transfer.source);
        command_processor_.PushTransitionBarrier(
            source_d3d12_rt.resource(),
            source_d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      }
      // transfer.host_depth_source == dest_rt means the EDRAM buffer will be
      // used instead, no need to transition.
      if (transfer.host_depth_source && transfer.host_depth_source != dest_rt &&
          !host_depth_source_previously_used_as_dest) {
        auto& host_depth_source_d3d12_rt =
            *static_cast<D3D12RenderTarget*>(transfer.host_depth_source);
        command_processor_.PushTransitionBarrier(
            host_depth_source_d3d12_rt.resource(),
            host_depth_source_d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      }
    }
    // Transition the destination, only if not going to be used as a source
    // earlier.
    bool dest_used_previously_as_source = false;
    for (uint32_t j = 0; j < i; ++j) {
      for (const Transfer& previous_transfer : render_target_transfers[j]) {
        if (previous_transfer.source == dest_rt || previous_transfer.host_depth_source == dest_rt) {
          dest_used_previously_as_source = true;
          break;
        }
      }
    }
    if (!dest_used_previously_as_source) {
      D3D12_RESOURCE_STATES dest_state = dest_d3d12_rt.key().is_depth
                                             ? D3D12_RESOURCE_STATE_DEPTH_WRITE
                                             : D3D12_RESOURCE_STATE_RENDER_TARGET;
      command_processor_.PushTransitionBarrier(
          dest_d3d12_rt.resource(), dest_d3d12_rt.SetResourceState(dest_state), dest_state);
    }
  }
  // Copy source descriptors to the shader-visible heap.
  // Clear previously set shader-visible descriptor indices.
  for (uint32_t i = 0; i < render_target_count; ++i) {
    RenderTarget* dest_rt = render_targets[i];
    if (!dest_rt) {
      continue;
    }
    for (const Transfer& transfer : render_target_transfers[i]) {
      assert_not_null(transfer.source);
      auto& source_d3d12_rt = *static_cast<D3D12RenderTarget*>(transfer.source);
      source_d3d12_rt.SetTemporarySRVDescriptorIndex(UINT32_MAX);
      source_d3d12_rt.SetTemporarySRVDescriptorIndexStencil(UINT32_MAX);
      auto* host_depth_source_d3d12_rt =
          static_cast<D3D12RenderTarget*>(transfer.host_depth_source);
      if (host_depth_source_d3d12_rt) {
        host_depth_source_d3d12_rt->SetTemporarySRVDescriptorIndex(UINT32_MAX);
        host_depth_source_d3d12_rt->SetTemporarySRVDescriptorIndexStencil(UINT32_MAX);
      }
    }
  }
  current_temporary_descriptors_cpu_.clear();
  // [NR-XFER] The native host-depth scratch SRV rides the same one-use heap
  // as the render target SRVs.
  uint32_t native_hds_srv_index = UINT32_MAX;
  if (native_hds_used) {
    native_hds_srv_index = uint32_t(current_temporary_descriptors_cpu_.size());
    current_temporary_descriptors_cpu_.push_back(native_hds_srv_handle);
  }
  for (uint32_t i = 0; i < render_target_count; ++i) {
    RenderTarget* dest_rt = render_targets[i];
    if (!dest_rt) {
      continue;
    }
    bool dest_is_depth = dest_rt->key().is_depth;
    for (const Transfer& transfer : render_target_transfers[i]) {
      assert_not_null(transfer.source);
      auto& source_d3d12_rt = *static_cast<D3D12RenderTarget*>(transfer.source);
      if (source_d3d12_rt.temporary_srv_descriptor_index() == UINT32_MAX) {
        source_d3d12_rt.SetTemporarySRVDescriptorIndex(
            uint32_t(current_temporary_descriptors_cpu_.size()));
        current_temporary_descriptors_cpu_.push_back(source_d3d12_rt.descriptor_srv().GetHandle());
      }
      if (source_d3d12_rt.key().is_depth &&
          source_d3d12_rt.temporary_srv_descriptor_index_stencil() == UINT32_MAX) {
        source_d3d12_rt.SetTemporarySRVDescriptorIndexStencil(
            uint32_t(current_temporary_descriptors_cpu_.size()));
        current_temporary_descriptors_cpu_.push_back(
            source_d3d12_rt.descriptor_srv_stencil().GetHandle());
      }
      bool source_is_depth = source_d3d12_rt.key().is_depth;
      auto* host_depth_source_d3d12_rt =
          static_cast<D3D12RenderTarget*>(transfer.host_depth_source);
      // The host_depth_source_d3d12_rt == dest_rt case would use the EDRAM
      // buffer instead.
      if (host_depth_source_d3d12_rt && host_depth_source_d3d12_rt != dest_rt &&
          host_depth_source_d3d12_rt->temporary_srv_descriptor_index() == UINT32_MAX) {
        host_depth_source_d3d12_rt->SetTemporarySRVDescriptorIndex(
            uint32_t(current_temporary_descriptors_cpu_.size()));
        current_temporary_descriptors_cpu_.push_back(
            host_depth_source_d3d12_rt->descriptor_srv().GetHandle());
      }
    }
  }
  uint32_t descriptor_count = uint32_t(current_temporary_descriptors_cpu_.size());
  current_temporary_descriptors_gpu_.resize(descriptor_count);
  if (!command_processor_.RequestOneUseSingleViewDescriptors(
          descriptor_count, current_temporary_descriptors_gpu_.data())) {
    return;
  }
  for (uint32_t i = 0; i < descriptor_count; ++i) {
    device->CopyDescriptorsSimple(1, current_temporary_descriptors_gpu_[i].first,
                                  current_temporary_descriptors_cpu_[i],
                                  D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  }

  // Perform the transfers and clears.

  bool transfer_viewport_set = false;
  float pixels_to_ndc_unscaled = 2.0f / float(D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION);
  float pixels_to_ndc_x = pixels_to_ndc_unscaled * draw_resolution_scale_x();
  float pixels_to_ndc_y = pixels_to_ndc_unscaled * draw_resolution_scale_y();

  TransferRootSignatureIndex last_transfer_root_signature_index =
      TransferRootSignatureIndex::kCount;
  uint32_t transfer_root_parameters_set = 0;
  uint32_t last_descriptor_index_color = UINT32_MAX;
  uint32_t last_descriptor_index_depth = UINT32_MAX;
  uint32_t last_descriptor_index_stencil = UINT32_MAX;
  uint32_t last_descriptor_index_host_depth = UINT32_MAX;
  TransferAddressConstant last_address_constant;
  TransferAddressConstant last_host_depth_address_constant;

  for (uint32_t i = 0; i < render_target_count; ++i) {
    RenderTarget* dest_rt = render_targets[i];
    if (!dest_rt) {
      continue;
    }

    const std::vector<Transfer>& current_transfers = render_target_transfers[i];
    if (current_transfers.empty() && !resolve_clear_needed) {
      continue;
    }

    auto& dest_d3d12_rt = *static_cast<D3D12RenderTarget*>(dest_rt);
    RenderTargetKey dest_rt_key = dest_d3d12_rt.key();

    // Late barrier in case there was cross-copying that prevented merging of
    // barriers.
    D3D12_RESOURCE_STATES dest_state = dest_rt_key.is_depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE
                                                            : D3D12_RESOURCE_STATE_RENDER_TARGET;
    command_processor_.PushTransitionBarrier(
        dest_d3d12_rt.resource(), dest_d3d12_rt.SetResourceState(dest_state), dest_state);

    if (!current_transfers.empty()) {
      are_current_command_list_render_targets_valid_ = false;
      if (dest_rt_key.is_depth) {
        D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle = dest_d3d12_rt.descriptor_draw().GetHandle();
        command_list.D3DOMSetRenderTargets(0, nullptr, FALSE, &dsv_handle);
        if (!use_stencil_reference_output_) {
          command_processor_.SetStencilReference(UINT8_MAX);
        }
      } else {
        D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle =
            dest_d3d12_rt.descriptor_load_separate().IsValid()
                ? dest_d3d12_rt.descriptor_load_separate().GetHandle()
                : dest_d3d12_rt.descriptor_draw().GetHandle();
        command_list.D3DOMSetRenderTargets(1, &rtv_handle, FALSE, nullptr);
      }

      uint32_t dest_pitch_tiles = dest_rt_key.GetPitchTiles();
      bool dest_is_64bpp = dest_rt_key.Is64bpp();

      // Gather shader keys and sort to reduce pipeline state and binding
      // switches. Also gather stencil rectangles to clear if needed.
      bool need_stencil_bit_draws = dest_rt_key.is_depth && !use_stencil_reference_output_;
      current_transfer_invocations_.clear();
      current_transfer_invocations_.reserve(current_transfers.size()
                                            << uint32_t(need_stencil_bit_draws));
      uint32_t rt_sort_index = 0;
      TransferShaderKey new_transfer_shader_key;
      new_transfer_shader_key.dest_msaa_samples = dest_rt_key.msaa_samples;
      new_transfer_shader_key.dest_resource_format = dest_rt_key.resource_format;
      uint32_t stencil_clear_rectangle_count = 0;
      for (uint32_t j = 0; j <= uint32_t(need_stencil_bit_draws); ++j) {
        // j == 0 - color or depth.
        // j == 1 - stencil bits.
        // Stencil bit writing always requires a different root signature,
        // handle these separately. Stencil never has a host depth source.
        // Clear previously set sort indices.
        for (const Transfer& transfer : current_transfers) {
          auto* host_depth_source_d3d12_rt =
              static_cast<D3D12RenderTarget*>(transfer.host_depth_source);
          if (host_depth_source_d3d12_rt) {
            host_depth_source_d3d12_rt->SetTemporarySortIndex(UINT32_MAX);
          }
          assert_not_null(transfer.source);
          auto& source_d3d12_rt = *static_cast<D3D12RenderTarget*>(transfer.source);
          source_d3d12_rt.SetTemporarySortIndex(UINT32_MAX);
        }
        for (const Transfer& transfer : current_transfers) {
          assert_not_null(transfer.source);
          auto& source_d3d12_rt = *static_cast<D3D12RenderTarget*>(transfer.source);
          D3D12RenderTarget* host_depth_source_d3d12_rt =
              j ? nullptr : static_cast<D3D12RenderTarget*>(transfer.host_depth_source);
          if (host_depth_source_d3d12_rt &&
              host_depth_source_d3d12_rt->temporary_sort_index() == UINT32_MAX) {
            host_depth_source_d3d12_rt->SetTemporarySortIndex(rt_sort_index++);
          }
          if (source_d3d12_rt.temporary_sort_index() == UINT32_MAX) {
            source_d3d12_rt.SetTemporarySortIndex(rt_sort_index++);
          }
          RenderTargetKey source_rt_key = source_d3d12_rt.key();
          new_transfer_shader_key.source_msaa_samples = source_rt_key.msaa_samples;
          new_transfer_shader_key.source_resource_format = source_rt_key.resource_format;
          // [NR-XFER] The self case reads the native scratch snapshot with
          // the dest's own key, so every host depth source takes the
          // ordinary TEXTURE shader variant now; the EDRAM buffer variant
          // (is_copy) is dead. If the scratch failed (cached, effectively
          // never), degrade to a guest-depth-only transfer.
          bool host_depth_degraded = false;
          if (host_depth_source_d3d12_rt == &dest_d3d12_rt && !native_hds_used) {
            host_depth_source_d3d12_rt = nullptr;
            host_depth_degraded = true;
          }
          new_transfer_shader_key.host_depth_source_is_copy = false;
          new_transfer_shader_key.host_depth_source_msaa_samples =
              host_depth_source_d3d12_rt ? host_depth_source_d3d12_rt->key().msaa_samples
                                         : xenos::MsaaSamples::k1X;
          if (j) {
            new_transfer_shader_key.mode = source_rt_key.is_depth
                                               ? TransferMode::kDepthToStencilBit
                                               : TransferMode::kColorToStencilBit;
            stencil_clear_rectangle_count += transfer.GetRectangles(
                dest_rt_key.base_tiles, dest_pitch_tiles, dest_rt_key.msaa_samples, dest_is_64bpp,
                nullptr, resolve_clear_rectangle);
          } else {
            if (dest_rt_key.is_depth) {
              if (host_depth_source_d3d12_rt) {
                new_transfer_shader_key.mode = source_rt_key.is_depth
                                                   ? TransferMode::kDepthAndHostDepthToDepth
                                                   : TransferMode::kColorAndHostDepthToDepth;
              } else {
                new_transfer_shader_key.mode = source_rt_key.is_depth ? TransferMode::kDepthToDepth
                                                                      : TransferMode::kColorToDepth;
              }
            } else {
              new_transfer_shader_key.mode = source_rt_key.is_depth ? TransferMode::kDepthToColor
                                                                    : TransferMode::kColorToColor;
            }
          }
          current_transfer_invocations_.emplace_back(transfer, new_transfer_shader_key);
          // [NR-XFER] Census: every transfer invocation by mode, stencil-bit
          // passes separately.
          if (j) {
            ++xfer_census_stencil_bit_;
            current_transfer_invocations_.back().transfer.host_depth_source = nullptr;
          } else {
            if (host_depth_degraded) {
              current_transfer_invocations_.back().transfer.host_depth_source = nullptr;
            }
            if (size_t(new_transfer_shader_key.mode) < rex::countof(xfer_census_modes_)) {
              ++xfer_census_modes_[size_t(new_transfer_shader_key.mode)];
            }
          }
        }
      }
      std::sort(current_transfer_invocations_.begin(), current_transfer_invocations_.end());

      // Clear the stencil to 0 where it will be loaded - will be setting the
      // bits that need to be 1 by discarding samples. Clearing everything here
      // to reduce context switches internally in the driver if clear causes
      // them.
      if (stencil_clear_rectangle_count) {
        command_processor_.SubmitBarriers();
        D3D12_RECT* stencil_clear_rect_write_ptr = command_list.ClearDepthStencilViewAllocatedRects(
            dest_d3d12_rt.descriptor_draw().GetHandle(), D3D12_CLEAR_FLAG_STENCIL, 0.0f, 0,
            stencil_clear_rectangle_count);
        assert_not_null(stencil_clear_rect_write_ptr);
        for (const Transfer& transfer : current_transfers) {
          Transfer::Rectangle transfer_stencil_clear_rectangles[Transfer::kMaxRectanglesWithCutout];
          uint32_t transfer_stencil_clear_rectangle_count = transfer.GetRectangles(
              dest_rt_key.base_tiles, dest_pitch_tiles, dest_rt_key.msaa_samples, dest_is_64bpp,
              transfer_stencil_clear_rectangles, resolve_clear_rectangle);
          for (uint32_t j = 0; j < transfer_stencil_clear_rectangle_count; ++j) {
            const Transfer::Rectangle& stencil_clear_rectangle =
                transfer_stencil_clear_rectangles[j];
            stencil_clear_rect_write_ptr->left =
                LONG(stencil_clear_rectangle.x_pixels * draw_resolution_scale_x());
            stencil_clear_rect_write_ptr->top =
                LONG(stencil_clear_rectangle.y_pixels * draw_resolution_scale_y());
            stencil_clear_rect_write_ptr->right =
                LONG((stencil_clear_rectangle.x_pixels + stencil_clear_rectangle.width_pixels) *
                     draw_resolution_scale_x());
            stencil_clear_rect_write_ptr->bottom =
                LONG((stencil_clear_rectangle.y_pixels + stencil_clear_rectangle.height_pixels) *
                     draw_resolution_scale_y());
            ++stencil_clear_rect_write_ptr;
          }
        }
      }

      // Perform the transfers for the render target.

      if (!transfer_viewport_set) {
        transfer_viewport_set = true;
        // Will be passing NDC directly, set the viewport to the maximum host
        // render target size for simplicity. Using a power-of-two scale for
        // exact pixel coordinates.
        D3D12_VIEWPORT transfer_viewport;
        transfer_viewport.TopLeftX = 0.0f;
        transfer_viewport.TopLeftY = 0.0f;
        transfer_viewport.Width = float(D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION);
        transfer_viewport.Height = float(D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION);
        transfer_viewport.MinDepth = 0.0f;
        transfer_viewport.MaxDepth = 1.0f;
        command_processor_.SetViewport(transfer_viewport);
        // TODO(Triang3l): Reduce scissor to the smallest transfer region for
        // more tiling friendliness.
        D3D12_RECT transfer_scissor;
        transfer_scissor.left = 0;
        transfer_scissor.top = 0;
        transfer_scissor.right = LONG(D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION);
        transfer_scissor.bottom = LONG(D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION);
        command_processor_.SetScissorRect(transfer_scissor);
      }

      for (auto it = current_transfer_invocations_.cbegin();
           it != current_transfer_invocations_.cend(); ++it) {
        const TransferInvocation& transfer_invocation_first = *it;
        // Will be merging transfers from the same source into one mesh.
        auto it_merged_first = it, it_merged_last = it;
        uint32_t transfer_rectangle_count = transfer_invocation_first.transfer.GetRectangles(
            dest_rt_key.base_tiles, dest_pitch_tiles, dest_rt_key.msaa_samples, dest_is_64bpp,
            nullptr, resolve_clear_rectangle);
        for (auto it_merge = std::next(it_merged_first);
             it_merge != current_transfer_invocations_.cend(); ++it_merge) {
          if (!transfer_invocation_first.CanBeMergedIntoOneDraw(*it_merge)) {
            break;
          }
          transfer_rectangle_count += it_merge->transfer.GetRectangles(
              dest_rt_key.base_tiles, dest_pitch_tiles, dest_rt_key.msaa_samples, dest_is_64bpp,
              nullptr, resolve_clear_rectangle);
          it_merged_last = it_merge;
        }
        assert_not_zero(transfer_rectangle_count);
        // Skip the merged transfers in the subsequent iterations.
        it = it_merged_last;

        assert_not_null(it->transfer.source);
        auto& source_d3d12_rt = *static_cast<D3D12RenderTarget*>(it->transfer.source);
        auto* host_depth_source_d3d12_rt =
            static_cast<D3D12RenderTarget*>(it->transfer.host_depth_source);
        TransferShaderKey transfer_shader_key = it->shader_key;
        const TransferModeInfo& transfer_mode_info =
            kTransferModes[size_t(transfer_shader_key.mode)];
        TransferRootSignatureIndex transfer_root_signature_index =
            use_stencil_reference_output_ ? transfer_mode_info.root_signature_with_stencil_ref
                                          : transfer_mode_info.root_signature_no_stencil_ref;
        uint32_t transfer_root_parameters_used =
            kTransferUsedRootParameters[size_t(transfer_root_signature_index)];
        bool is_stencil_bit =
            (transfer_root_parameters_used & kTransferUsedRootParameterStencilMaskConstantBit) != 0;

        // Late barriers in case there was cross-copying that prevented merging
        // of barriers.
        command_processor_.PushTransitionBarrier(
            source_d3d12_rt.resource(),
            source_d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        if (host_depth_source_d3d12_rt) {
          if (host_depth_source_d3d12_rt == &dest_d3d12_rt) {
            // [NR-XFER] Native self snapshot: the scratch texture was left in
            // PIXEL_SHADER_RESOURCE by the setup copy; the destination itself
            // must stay bound as the depth target.
          } else {
            // Reading host depth from the texture.
            command_processor_.PushTransitionBarrier(
                host_depth_source_d3d12_rt->resource(),
                host_depth_source_d3d12_rt->SetResourceState(
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
          }
        }

        uint32_t transfer_vertex_count = 6 * transfer_rectangle_count;
        D3D12_VERTEX_BUFFER_VIEW transfer_rectangle_buffer_view;
        transfer_rectangle_buffer_view.StrideInBytes = sizeof(float) * 2;
        transfer_rectangle_buffer_view.SizeInBytes =
            transfer_rectangle_buffer_view.StrideInBytes * transfer_vertex_count;
        float* transfer_rectangle_write_ptr =
            reinterpret_cast<float*>(transfer_vertex_buffer_pool_->Request(
                current_submission, transfer_rectangle_buffer_view.SizeInBytes, sizeof(float),
                nullptr, nullptr, &transfer_rectangle_buffer_view.BufferLocation));
        if (!transfer_rectangle_write_ptr) {
          continue;
        }
        for (auto it_merged = it_merged_first; it_merged <= it_merged_last; ++it_merged) {
          Transfer::Rectangle transfer_invocation_rectangles[Transfer::kMaxRectanglesWithCutout];
          uint32_t transfer_invocation_rectangle_count = it_merged->transfer.GetRectangles(
              dest_rt_key.base_tiles, dest_pitch_tiles, dest_rt_key.msaa_samples, dest_is_64bpp,
              transfer_invocation_rectangles, resolve_clear_rectangle);
          assert_not_zero(transfer_invocation_rectangle_count);
          for (uint32_t j = 0; j < transfer_invocation_rectangle_count; ++j) {
            const Transfer::Rectangle& transfer_rectangle = transfer_invocation_rectangles[j];
            float transfer_rectangle_x0 = -1.0f + transfer_rectangle.x_pixels * pixels_to_ndc_x;
            float transfer_rectangle_y0 = 1.0f - transfer_rectangle.y_pixels * pixels_to_ndc_y;
            float transfer_rectangle_x1 =
                transfer_rectangle_x0 + transfer_rectangle.width_pixels * pixels_to_ndc_x;
            float transfer_rectangle_y1 =
                transfer_rectangle_y0 - transfer_rectangle.height_pixels * pixels_to_ndc_y;
            // O-*
            // |/
            // *
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_x0;
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_y0;
            // *-O
            // |/
            // *
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_x1;
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_y0;
            // *-*
            // |/
            // O
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_x0;
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_y1;
            //   *
            //  /|
            // O-*
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_x0;
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_y1;
            //   O
            //  /|
            // *-*
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_x1;
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_y0;
            //   *
            //  /|
            // *-O
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_x1;
            *(transfer_rectangle_write_ptr++) = transfer_rectangle_y1;
          }
        }
        command_processor_.SetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        command_list.D3DIASetVertexBuffers(0, 1, &transfer_rectangle_buffer_view);

        ID3D12PipelineState* const* transfer_pipelines =
            GetOrCreateTransferPipelines(transfer_shader_key);
        if (!transfer_pipelines) {
          continue;
        }
        if (last_transfer_root_signature_index != transfer_root_signature_index) {
          last_transfer_root_signature_index = transfer_root_signature_index;
          command_processor_.SetExternalGraphicsRootSignature(
              transfer_root_signatures_[size_t(transfer_root_signature_index)]);
          transfer_root_parameters_set = 0;
        }

        // Invalidate outdated bindings.
        if (transfer_root_parameters_used & kTransferUsedRootParameterColorSRVBit) {
          uint32_t descriptor_index_color = source_d3d12_rt.temporary_srv_descriptor_index();
          assert_true(descriptor_index_color != UINT32_MAX);
          if (last_descriptor_index_color != descriptor_index_color) {
            last_descriptor_index_color = descriptor_index_color;
            transfer_root_parameters_set &= ~kTransferUsedRootParameterColorSRVBit;
          }
        }
        if (transfer_root_parameters_used & kTransferUsedRootParameterDepthSRVBit) {
          uint32_t descriptor_index_depth = source_d3d12_rt.temporary_srv_descriptor_index();
          assert_true(descriptor_index_depth != UINT32_MAX);
          if (last_descriptor_index_depth != descriptor_index_depth) {
            last_descriptor_index_depth = descriptor_index_depth;
            transfer_root_parameters_set &= ~kTransferUsedRootParameterDepthSRVBit;
          }
        }
        if (transfer_root_parameters_used & kTransferUsedRootParameterStencilSRVBit) {
          uint32_t descriptor_index_stencil =
              source_d3d12_rt.temporary_srv_descriptor_index_stencil();
          assert_true(descriptor_index_stencil != UINT32_MAX);
          if (last_descriptor_index_stencil != descriptor_index_stencil) {
            last_descriptor_index_stencil = descriptor_index_stencil;
            transfer_root_parameters_set &= ~kTransferUsedRootParameterStencilSRVBit;
          }
        }
        if (transfer_root_parameters_used & kTransferUsedRootParameterHostDepthSRVBit) {
          assert_not_null(host_depth_source_d3d12_rt);
          // [NR-XFER] The native self snapshot binds the scratch texture's
          // one-use SRV; the destination has no temporary SRV of its own.
          uint32_t descriptor_index_host_depth =
              host_depth_source_d3d12_rt == &dest_d3d12_rt
                  ? native_hds_srv_index
                  : host_depth_source_d3d12_rt->temporary_srv_descriptor_index();
          assert_true(descriptor_index_host_depth != UINT32_MAX);
          if (last_descriptor_index_host_depth != descriptor_index_host_depth) {
            transfer_root_parameters_set &= ~kTransferUsedRootParameterHostDepthSRVBit;
          }
          last_descriptor_index_host_depth = descriptor_index_host_depth;
        }
        if (transfer_root_parameters_used & kTransferUsedRootParameterAddressConstantBit) {
          RenderTargetKey source_rt_key = source_d3d12_rt.key();
          TransferAddressConstant address_constant;
          address_constant.dest_pitch = dest_pitch_tiles;
          address_constant.source_pitch = source_rt_key.GetPitchTiles();
          address_constant.source_to_dest =
              int32_t(dest_rt_key.base_tiles) - int32_t(source_rt_key.base_tiles);
          if (last_address_constant != address_constant) {
            last_address_constant = address_constant;
            transfer_root_parameters_set &= ~kTransferUsedRootParameterAddressConstantBit;
          }
        }
        if (transfer_root_parameters_used & kTransferUsedRootParameterHostDepthAddressConstantBit) {
          assert_not_null(host_depth_source_d3d12_rt);
          RenderTargetKey host_depth_source_rt_key = host_depth_source_d3d12_rt->key();
          TransferAddressConstant host_depth_address_constant;
          host_depth_address_constant.dest_pitch = dest_pitch_tiles;
          host_depth_address_constant.source_pitch = host_depth_source_rt_key.GetPitchTiles();
          host_depth_address_constant.source_to_dest =
              int32_t(dest_rt_key.base_tiles) - int32_t(host_depth_source_rt_key.base_tiles);
          if (last_host_depth_address_constant != host_depth_address_constant) {
            last_host_depth_address_constant = host_depth_address_constant;
            transfer_root_parameters_set &= ~kTransferUsedRootParameterHostDepthAddressConstantBit;
          }
        }

        // Apply the new bindings.
        uint32_t transfer_root_parameters_unset =
            transfer_root_parameters_used & ~transfer_root_parameters_set;
        if (transfer_root_parameters_unset &
            kTransferUsedRootParameterHostDepthAddressConstantBit) {
          command_list.D3DSetGraphicsRoot32BitConstants(
              rex::bit_count(transfer_root_parameters_used &
                             (kTransferUsedRootParameterHostDepthAddressConstantBit - 1)),
              sizeof(last_host_depth_address_constant) / sizeof(uint32_t),
              &last_host_depth_address_constant, 0);
          transfer_root_parameters_set |= kTransferUsedRootParameterHostDepthAddressConstantBit;
        }
        if (transfer_root_parameters_unset & kTransferUsedRootParameterHostDepthSRVBit) {
          assert_true(last_descriptor_index_host_depth != UINT32_MAX);
          command_list.D3DSetGraphicsRootDescriptorTable(
              rex::bit_count(transfer_root_parameters_used &
                             (kTransferUsedRootParameterHostDepthSRVBit - 1)),
              current_temporary_descriptors_gpu_[last_descriptor_index_host_depth].second);
          transfer_root_parameters_set |= kTransferUsedRootParameterHostDepthSRVBit;
        }
        if (transfer_root_parameters_unset & kTransferUsedRootParameterAddressConstantBit) {
          command_list.D3DSetGraphicsRoot32BitConstants(
              rex::bit_count(transfer_root_parameters_used &
                             (kTransferUsedRootParameterAddressConstantBit - 1)),
              sizeof(last_address_constant) / sizeof(uint32_t), &last_address_constant, 0);
          transfer_root_parameters_set |= kTransferUsedRootParameterAddressConstantBit;
        }
        if (transfer_root_parameters_unset & kTransferUsedRootParameterStencilSRVBit) {
          assert_true(last_descriptor_index_stencil != UINT32_MAX);
          command_list.D3DSetGraphicsRootDescriptorTable(
              rex::bit_count(transfer_root_parameters_used &
                             (kTransferUsedRootParameterStencilSRVBit - 1)),
              current_temporary_descriptors_gpu_[last_descriptor_index_stencil].second);
          transfer_root_parameters_set |= kTransferUsedRootParameterStencilSRVBit;
        }
        if (transfer_root_parameters_unset & kTransferUsedRootParameterDepthSRVBit) {
          assert_true(last_descriptor_index_depth != UINT32_MAX);
          command_list.D3DSetGraphicsRootDescriptorTable(
              rex::bit_count(transfer_root_parameters_used &
                             (kTransferUsedRootParameterDepthSRVBit - 1)),
              current_temporary_descriptors_gpu_[last_descriptor_index_depth].second);
          transfer_root_parameters_set |= kTransferUsedRootParameterDepthSRVBit;
        }
        if (transfer_root_parameters_unset & kTransferUsedRootParameterColorSRVBit) {
          assert_true(last_descriptor_index_color != UINT32_MAX);
          command_list.D3DSetGraphicsRootDescriptorTable(
              rex::bit_count(transfer_root_parameters_used &
                             (kTransferUsedRootParameterColorSRVBit - 1)),
              current_temporary_descriptors_gpu_[last_descriptor_index_color].second);
          transfer_root_parameters_set |= kTransferUsedRootParameterColorSRVBit;
        }

        // Draw the transfer rectangles.
        command_processor_.SubmitBarriers();
        for (uint32_t j = 0; j <= uint32_t(is_stencil_bit) * 7; ++j) {
          if (is_stencil_bit) {
            uint32_t transfer_stencil_bit = uint32_t(1) << j;
            command_list.D3DSetGraphicsRoot32BitConstants(
                rex::bit_count(transfer_root_parameters_used &
                               (kTransferUsedRootParameterStencilMaskConstantBit - 1)),
                sizeof(transfer_stencil_bit) / sizeof(uint32_t), &transfer_stencil_bit, 0);
          }
          command_processor_.SetExternalPipeline(transfer_pipelines[j]);
          command_list.D3DDrawInstanced(transfer_vertex_count, 1, 0, 0);
        }
      }
    }

    // Perform the clear.
    if (resolve_clear_needed) {
      uint64_t clear_value = render_target_resolve_clear_values[i];
      if (dest_rt_key.is_depth) {
        uint32_t depth_guest_clear_value = (uint32_t(clear_value) >> 8) & 0xFFFFFF;
        float depth_host_clear_value = 0.0f;
        switch (dest_rt_key.GetDepthFormat()) {
          case xenos::DepthRenderTargetFormat::kD24S8:
            depth_host_clear_value = xenos::UNorm24To32(depth_guest_clear_value);
            break;
          case xenos::DepthRenderTargetFormat::kD24FS8:
            // Taking [0, 2) -> [0, 1) remapping into account.
            depth_host_clear_value = xenos::Float20e4To32(depth_guest_clear_value) * 0.5f;
            break;
        }
        command_processor_.PushTransitionBarrier(
            dest_d3d12_rt.resource(),
            dest_d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_DEPTH_WRITE),
            D3D12_RESOURCE_STATE_DEPTH_WRITE);
        command_processor_.SubmitBarriers();
        command_list.D3DClearDepthStencilView(dest_d3d12_rt.descriptor_draw().GetHandle(),
                                              D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
                                              depth_host_clear_value, UINT(clear_value) & 0xFF, 1,
                                              &clear_rect);
      } else {
        float color_clear_value[4] = {};
        bool clear_via_drawing = false;
        switch (dest_rt_key.GetColorFormat()) {
          case xenos::ColorRenderTargetFormat::k_8_8_8_8: {
            for (uint32_t j = 0; j < 4; ++j) {
              color_clear_value[j] = ((clear_value >> (j * 8)) & 0xFF) * (1.0f / 0xFF);
            }
          } break;
          case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA: {
            // 8_8_8_8_GAMMA is represented by linear stored in
            // R16G16B16A16_UNORM.
            for (uint32_t j = 0; j < 4; ++j) {
              color_clear_value[j] = ((clear_value >> (j * 8)) & 0xFF) * (1.0f / 0xFF);
            }
            for (uint32_t j = 0; j < 3; ++j) {
              color_clear_value[j] = xenos::PWLGammaToLinear(color_clear_value[j]);
            }
          } break;
          case xenos::ColorRenderTargetFormat::k_2_10_10_10:
          case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10: {
            for (uint32_t j = 0; j < 3; ++j) {
              color_clear_value[j] = ((clear_value >> (j * 10)) & 0x3FF) * (1.0f / 0x3FF);
            }
            color_clear_value[3] = ((clear_value >> 30) & 0x3) * (1.0f / 0x3);
          } break;
          case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
          case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16: {
            for (uint32_t j = 0; j < 3; ++j) {
              color_clear_value[j] = xenos::Float7e3To32((clear_value >> (j * 10)) & 0x3FF);
            }
            color_clear_value[3] = ((clear_value >> 30) & 0x3) * (1.0f / 0x3);
          } break;
          case xenos::ColorRenderTargetFormat::k_16_16:
          case xenos::ColorRenderTargetFormat::k_16_16_FLOAT: {
            // Using uint for loading both. Disregarding the current -32...32
            // vs. -1...1 settings for consistency with color clear via depth
            // aliasing.
            for (uint32_t j = 0; j < 2; ++j) {
              color_clear_value[j] = float((clear_value >> (j * 16)) & 0xFFFF);
            }
          } break;
          case xenos::ColorRenderTargetFormat::k_16_16_16_16:
          case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT: {
            // Using uint for loading both. Disregarding the current -32...32
            // vs. -1...1 settings for consistency with color clear via depth
            // aliasing.
            for (uint32_t j = 0; j < 4; ++j) {
              color_clear_value[j] = float((clear_value >> (j * 16)) & 0xFFFF);
            }
          } break;
          case xenos::ColorRenderTargetFormat::k_32_FLOAT: {
            // Using uint for proper denormal and NaN handling.
            color_clear_value[0] = float(uint32_t(clear_value));
            // Numbers > 2^24 can't be represented with a step of 1 as floats,
            // need to clear by drawing a uint rectangle.
            if (uint64_t(color_clear_value[0]) != uint32_t(clear_value)) {
              clear_via_drawing = true;
            }
          } break;
          case xenos::ColorRenderTargetFormat::k_32_32_FLOAT: {
            // Using uint for proper denormal and NaN handling.
            color_clear_value[0] = float(uint32_t(clear_value));
            color_clear_value[1] = float(uint32_t(clear_value >> 32));
            // Numbers > 2^24 can't be represented with a step of 1 as floats,
            // need to clear by drawing a uint rectangle.
            if (uint64_t(color_clear_value[0]) != uint32_t(clear_value) ||
                uint64_t(color_clear_value[1]) != uint32_t(clear_value >> 32)) {
              clear_via_drawing = true;
            }
          } break;
        }
        command_processor_.PushTransitionBarrier(
            dest_d3d12_rt.resource(),
            dest_d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_RENDER_TARGET),
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        if (clear_via_drawing) {
          D3D12_CPU_DESCRIPTOR_HANDLE clear_rtv_handle =
              dest_d3d12_rt.descriptor_load_separate().IsValid()
                  ? dest_d3d12_rt.descriptor_load_separate().GetHandle()
                  : dest_d3d12_rt.descriptor_draw().GetHandle();
          command_list.D3DOMSetRenderTargets(1, &clear_rtv_handle, FALSE, nullptr);
          are_current_command_list_render_targets_valid_ = true;
          D3D12_VIEWPORT clear_viewport;
          clear_viewport.TopLeftX = float(clear_rect.left);
          clear_viewport.TopLeftY = float(clear_rect.top);
          clear_viewport.Width = float(clear_rect.right - clear_rect.left);
          clear_viewport.Height = float(clear_rect.bottom - clear_rect.top);
          clear_viewport.MinDepth = 0.0f;
          clear_viewport.MaxDepth = 1.0f;
          command_processor_.SetViewport(clear_viewport);
          command_processor_.SetScissorRect(clear_rect);
          command_processor_.SetExternalGraphicsRootSignature(uint32_rtv_clear_root_signature_);
          uint32_t clear_via_drawing_value[2] = {uint32_t(clear_value),
                                                 uint32_t(clear_value >> 32)};
          command_list.D3DSetGraphicsRoot32BitConstants(0, 2, clear_via_drawing_value, 0);
          command_processor_.SetExternalPipeline(uint32_rtv_clear_pipelines_[size_t(
              dest_rt_key.GetColorFormat() ==
              xenos::ColorRenderTargetFormat::k_32_32_FLOAT)][size_t(dest_rt_key.msaa_samples)]);
          command_processor_.SetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
          command_list.D3DDrawInstanced(3, 1, 0, 0);
        } else {
          command_processor_.SubmitBarriers();
          command_list.D3DClearRenderTargetView(
              dest_d3d12_rt.descriptor_load_separate().IsValid()
                  ? dest_d3d12_rt.descriptor_load_separate().GetHandle()
                  : dest_d3d12_rt.descriptor_draw().GetHandle(),
              color_clear_value, 1, &clear_rect);
        }
      }
    }
  }

  // [NR-XFER] N-10b census, cumulative, 1 Hz. c2d..dhd follow the TransferMode
  // order; "dhd"/"chd" self hits are what decide whether the EDRAM host-depth
  // round trip is reachable at all in this game.
  ++xfer_census_passes_;
  const auto xfer_census_now = std::chrono::steady_clock::now();
  if (xfer_census_now - xfer_census_last_report_ >= std::chrono::seconds(1)) {
    xfer_census_last_report_ = xfer_census_now;
    REXGPU_INFO(
        "[nr-xfer] cum: passes {} | c2d {} c2c {} d2d {} d2c {} c2sb {} d2sb {} chd {} dhd {} | "
        "sb-draws {} | hds fail {} native {}",
        xfer_census_passes_, xfer_census_modes_[0], xfer_census_modes_[1], xfer_census_modes_[2],
        xfer_census_modes_[3], xfer_census_modes_[4], xfer_census_modes_[5], xfer_census_modes_[6],
        xfer_census_modes_[7], xfer_census_stencil_bit_, xfer_census_hds_fail_,
        xfer_census_hds_native_);
    XferUseReport();  // [xfer]
  }
}

void D3D12RenderTargetCache::SetCommandListRenderTargets(
    RenderTarget* const* depth_and_color_render_targets) {
  assert_true(GetPath() == Path::kHostRenderTargets);

  // Ensure the render targets are in the needed resource state.
  if (depth_and_color_render_targets[0]) {
    auto& d3d12_rt = *static_cast<D3D12RenderTarget*>(depth_and_color_render_targets[0]);
    command_processor_.PushTransitionBarrier(
        d3d12_rt.resource(), d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_DEPTH_WRITE),
        D3D12_RESOURCE_STATE_DEPTH_WRITE);
  }
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    RenderTarget* render_target = depth_and_color_render_targets[1 + i];
    if (!render_target) {
      continue;
    }
    auto& d3d12_rt = *static_cast<D3D12RenderTarget*>(render_target);
    command_processor_.PushTransitionBarrier(
        d3d12_rt.resource(), d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_RENDER_TARGET),
        D3D12_RESOURCE_STATE_RENDER_TARGET);
  }

  // Bind the render targets.
  if (are_current_command_list_render_targets_valid_ &&
      std::memcmp(current_command_list_render_targets_, depth_and_color_render_targets,
                  sizeof(current_command_list_render_targets_))) {
    are_current_command_list_render_targets_valid_ = false;
  }
  if (!are_current_command_list_render_targets_valid_) {
    std::memcpy(current_command_list_render_targets_, depth_and_color_render_targets,
                sizeof(current_command_list_render_targets_));
    // [gpu-census] draw sub-split: report the freshly bound RT config so
    // draws are priced per pass class. Formatted only when first seen.
    {
      uint32_t census_rt_keys[5];
      RenderTargetKey census_first_key;
      char census_desc[64];
      size_t census_len = 0;
      for (uint32_t i = 0; i < 1 + xenos::kMaxColorRenderTargets; ++i) {
        const RenderTarget* census_rt = depth_and_color_render_targets[i];
        RenderTargetKey census_key = census_rt ? census_rt->key() : RenderTargetKey();
        census_rt_keys[i] = census_key.key;
        if (census_key.IsEmpty() || census_len >= sizeof(census_desc)) {
          continue;
        }
        if (census_first_key.IsEmpty()) {
          census_first_key = census_key;
        }
        int census_n =
            i == 0 ? snprintf(census_desc + census_len, sizeof(census_desc) - census_len,
                              "d%ub%u ", uint32_t(census_key.resource_format),
                              uint32_t(census_key.base_tiles))
                   : snprintf(census_desc + census_len, sizeof(census_desc) - census_len,
                              "c%u:%ub%u ", i - 1, uint32_t(census_key.resource_format),
                              uint32_t(census_key.base_tiles));
        census_len = std::min(census_len + size_t(std::max(census_n, 0)), sizeof(census_desc));
      }
      if (census_len < sizeof(census_desc)) {
        snprintf(census_desc + census_len, sizeof(census_desc) - census_len, "p%um%u",
                 uint32_t(census_first_key.pitch_tiles_at_32bpp),
                 uint32_t(census_first_key.msaa_samples));
      }
      command_processor_.GpuCensusSetDrawConfig(census_rt_keys, census_desc);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle;
    if (depth_and_color_render_targets[0]) {
      dsv_handle = static_cast<const D3D12RenderTarget*>(depth_and_color_render_targets[0])
                       ->descriptor_draw()
                       .GetHandle();
    }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_handles[xenos::kMaxColorRenderTargets];
    uint32_t rtv_count = 0;
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      const RenderTarget* render_target = depth_and_color_render_targets[1 + i];
      if (!render_target) {
        continue;
      }
      // Fill the gaps with a null descriptor.
      while (rtv_count < i) {
        rtv_handles[rtv_count++] = render_target->key().msaa_samples != xenos::MsaaSamples::k1X
                                       ? null_rtv_descriptor_ms_.GetHandle()
                                       : null_rtv_descriptor_ss_.GetHandle();
      }
      auto& d3d12_rt = *static_cast<const D3D12RenderTarget*>(render_target);
      rtv_handles[rtv_count++] = d3d12_rt.descriptor_draw().GetHandle();
    }
    command_processor_.GetDeferredCommandList().D3DOMSetRenderTargets(
        rtv_count, rtv_handles, FALSE, depth_and_color_render_targets[0] ? &dsv_handle : nullptr);
    are_current_command_list_render_targets_valid_ = true;
  }
}

ID3D12PipelineState* D3D12RenderTargetCache::GetOrCreateDumpPipeline(DumpPipelineKey key) {
  auto pipeline_it = dump_pipelines_.find(key);
  if (pipeline_it != dump_pipelines_.end()) {
    return pipeline_it->second;
  }

  // Because of built_shader_.resize(), pointers can't be kept persistently
  // here! Resizing also zeroes the memory.

  built_shader_.clear();

  // RDEF, ISGN, OSGN, SHEX, STAT.
  constexpr uint32_t kBlobCount = 5;

  // Allocate space for the container header and the blob offsets.
  built_shader_.resize(sizeof(dxbc::ContainerHeader) / sizeof(uint32_t) + kBlobCount);
  uint32_t blob_offset_position_dwords = sizeof(dxbc::ContainerHeader) / sizeof(uint32_t);
  uint32_t blob_position_dwords = uint32_t(built_shader_.size());
  constexpr uint32_t kBlobHeaderSizeDwords = sizeof(dxbc::BlobHeader) / sizeof(uint32_t);

  uint32_t name_ptr;

  // ***************************************************************************
  // Resource definition
  // ***************************************************************************

  built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
  uint32_t rdef_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
  // Not needed, as the next operation done is resize, to allocate the space for
  // both the blob header and the resource definition header.
  // built_shader_.resize(rdef_position_dwords);

  // Allocate space for the RDEF header.
  built_shader_.resize(rdef_position_dwords + sizeof(dxbc::RdefHeader) / sizeof(uint32_t));
  // Generator name.
  dxbc::AppendAlignedString(built_shader_, "Xenia");

  // Constant types - uint (aka "dword" when it's scalar) only.
  // Names.
  name_ptr = uint32_t((built_shader_.size() - rdef_position_dwords) * sizeof(uint32_t));
  uint32_t rdef_dword_name_ptr = name_ptr;
  name_ptr += dxbc::AppendAlignedString(built_shader_, "dword");
  // Types.
  uint32_t rdef_type_uint_position_dwords = uint32_t(built_shader_.size());
  uint32_t rdef_type_uint_ptr =
      uint32_t((rdef_type_uint_position_dwords - rdef_position_dwords) * sizeof(uint32_t));
  built_shader_.resize(rdef_type_uint_position_dwords + sizeof(dxbc::RdefType) / sizeof(uint32_t));
  {
    auto& rdef_type_uint =
        *reinterpret_cast<dxbc::RdefType*>(built_shader_.data() + rdef_type_uint_position_dwords);
    rdef_type_uint.variable_class = dxbc::RdefVariableClass::kScalar;
    rdef_type_uint.variable_type = dxbc::RdefVariableType::kUInt;
    rdef_type_uint.row_count = 1;
    rdef_type_uint.column_count = 1;
    rdef_type_uint.name_ptr = rdef_dword_name_ptr;
  }

  // Constants:
  // - uint xe_edram_dump_offsets
  // - uint xe_edram_dump_pitches
  enum Constant : uint32_t {
    kConstantOffsets,
    kConstantPitches,
    kConstantCount,
  };
  // Names.
  name_ptr = uint32_t((built_shader_.size() - rdef_position_dwords) * sizeof(uint32_t));
  uint32_t rdef_xe_edram_dump_offsets_name_ptr = name_ptr;
  name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_edram_dump_offsets");
  uint32_t rdef_xe_edram_dump_pitches_name_ptr = name_ptr;
  name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_edram_dump_pitches");
  // Constants.
  uint32_t rdef_constants_position_dwords = uint32_t(built_shader_.size());
  uint32_t rdef_constants_ptr =
      uint32_t((rdef_constants_position_dwords - rdef_position_dwords) * sizeof(uint32_t));
  built_shader_.resize(rdef_constants_position_dwords +
                       sizeof(dxbc::RdefVariable) / sizeof(uint32_t) * kConstantCount);
  {
    auto rdef_constants = reinterpret_cast<dxbc::RdefVariable*>(built_shader_.data() +
                                                                rdef_constants_position_dwords);
    // uint xe_edram_dump_offsets
    dxbc::RdefVariable& rdef_constant_offsets = rdef_constants[kConstantOffsets];
    rdef_constant_offsets.name_ptr = rdef_xe_edram_dump_offsets_name_ptr;
    rdef_constant_offsets.size_bytes = sizeof(uint32_t);
    rdef_constant_offsets.flags = dxbc::kRdefVariableFlagUsed;
    rdef_constant_offsets.type_ptr = rdef_type_uint_ptr;
    rdef_constant_offsets.start_texture = UINT32_MAX;
    rdef_constant_offsets.start_sampler = UINT32_MAX;
    // uint xe_edram_dump_pitches
    dxbc::RdefVariable& rdef_constant_pitches = rdef_constants[kConstantPitches];
    rdef_constant_pitches.name_ptr = rdef_xe_edram_dump_pitches_name_ptr;
    rdef_constant_pitches.size_bytes = sizeof(uint32_t);
    rdef_constant_pitches.flags = dxbc::kRdefVariableFlagUsed;
    rdef_constant_pitches.type_ptr = rdef_type_uint_ptr;
    rdef_constant_pitches.start_texture = UINT32_MAX;
    rdef_constant_pitches.start_sampler = UINT32_MAX;
  }

  // Constant buffers:
  // - xe_edram_dump_offsets : b0 { uint xe_edram_dump_offsets; }
  // - xe_edram_dump_pitches : b1 { uint xe_edram_dump_pitches; }
  // Reusing the constant names for constant buffers.
  uint32_t rdef_cbuffer_position_dwords = uint32_t(built_shader_.size());
  built_shader_.resize(rdef_cbuffer_position_dwords +
                       sizeof(dxbc::RdefCbuffer) / sizeof(uint32_t) * kDumpCbufferCount);
  {
    auto rdef_cbuffers =
        reinterpret_cast<dxbc::RdefCbuffer*>(built_shader_.data() + rdef_cbuffer_position_dwords);
    // xe_edram_dump_offsets
    dxbc::RdefCbuffer& rdef_cbuffer_offsets = rdef_cbuffers[kDumpCbufferOffsets];
    rdef_cbuffer_offsets.name_ptr = rdef_xe_edram_dump_offsets_name_ptr;
    rdef_cbuffer_offsets.variable_count = 1;
    rdef_cbuffer_offsets.variables_ptr =
        uint32_t(rdef_constants_ptr + sizeof(dxbc::RdefVariable) * kConstantOffsets);
    rdef_cbuffer_offsets.size_vector_aligned_bytes = sizeof(uint32_t) * 4;
    // xe_edram_dump_pitches
    dxbc::RdefCbuffer& rdef_cbuffer_pitches = rdef_cbuffers[kDumpCbufferPitches];
    rdef_cbuffer_pitches.name_ptr = rdef_xe_edram_dump_pitches_name_ptr;
    rdef_cbuffer_pitches.variable_count = 1;
    rdef_cbuffer_pitches.variables_ptr =
        uint32_t(rdef_constants_ptr + sizeof(dxbc::RdefVariable) * kConstantPitches);
    rdef_cbuffer_pitches.size_vector_aligned_bytes = sizeof(uint32_t) * 4;
  }

  // Bindings.
  // - Texture2D/Texture2DMS<float4/uint4> xe_edram_dump_source : t0
  // - Optionally, Texture2D/Texture2DMS<uint2> xe_edram_dump_stencil : t1
  // - RWBuffer<uint/uint2> xe_edram : u0
  // - Constant buffers
  uint32_t rdef_binding_count = 1 + key.is_depth + 1 + kDumpCbufferCount;
  // Names.
  name_ptr = uint32_t((built_shader_.size() - rdef_position_dwords) * sizeof(uint32_t));
  uint32_t rdef_xe_edram_dump_source_name_ptr = name_ptr;
  name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_edram_dump_source");
  uint32_t rdef_xe_edram_dump_stencil_name_ptr = name_ptr;
  if (key.is_depth) {
    name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_edram_dump_stencil");
  }
  uint32_t rdef_xe_edram_name_ptr = name_ptr;
  name_ptr += dxbc::AppendAlignedString(built_shader_, "xe_edram");
  // Bindings.
  uint32_t rdef_binding_position_dwords = uint32_t(built_shader_.size());
  built_shader_.resize(rdef_binding_position_dwords +
                       sizeof(dxbc::RdefInputBind) / sizeof(uint32_t) * rdef_binding_count);
  bool source_is_uint;
  if (key.is_depth) {
    source_is_uint = false;
  } else {
    GetColorOwnershipTransferDXGIFormat(key.GetColorFormat(), &source_is_uint);
  }
  dxbc::ResourceReturnType source_return_type =
      source_is_uint ? dxbc::ResourceReturnType::kUInt : dxbc::ResourceReturnType::kFloat;
  uint32_t source_component_count =
      key.is_depth ? 1 : xenos::GetColorRenderTargetFormatComponentCount(key.GetColorFormat());
  bool format_is_64bpp =
      !key.is_depth && xenos::IsColorRenderTargetFormat64bpp(key.GetColorFormat());
  {
    auto rdef_bindings =
        reinterpret_cast<dxbc::RdefInputBind*>(built_shader_.data() + rdef_binding_position_dwords);
    uint32_t rdef_binding_index = 0;
    // xe_edram_dump_source
    dxbc::RdefInputBind& rdef_binding_source = rdef_bindings[rdef_binding_index++];
    rdef_binding_source.name_ptr = rdef_xe_edram_dump_source_name_ptr;
    rdef_binding_source.type = dxbc::RdefInputType::kTexture;
    rdef_binding_source.return_type = source_return_type;
    if (key.msaa_samples != xenos::MsaaSamples::k1X) {
      rdef_binding_source.dimension = dxbc::RdefDimension::kSRVTexture2DMS;
      // Sample count is dynamic on Shader Model 5.
    } else {
      rdef_binding_source.dimension = dxbc::RdefDimension::kSRVTexture2D;
      rdef_binding_source.sample_count = UINT32_MAX;
    }
    rdef_binding_source.bind_count = 1;
    rdef_binding_source.flags = (source_component_count - 1)
                                << dxbc::kRdefInputFlagsComponentsShift;
    // xe_edram_dump_stencil
    if (key.is_depth) {
      dxbc::RdefInputBind& rdef_binding_stencil = rdef_bindings[rdef_binding_index++];
      rdef_binding_stencil.name_ptr = rdef_xe_edram_dump_stencil_name_ptr;
      rdef_binding_stencil.type = dxbc::RdefInputType::kTexture;
      rdef_binding_stencil.return_type = dxbc::ResourceReturnType::kUInt;
      rdef_binding_stencil.dimension = rdef_binding_source.dimension;
      rdef_binding_stencil.sample_count = rdef_binding_source.sample_count;
      rdef_binding_stencil.bind_point = 1;
      rdef_binding_stencil.bind_count = 1;
      rdef_binding_stencil.flags = dxbc::kRdefInputFlags2Component;
      rdef_binding_stencil.id = 1;
    }
    // xe_edram
    dxbc::RdefInputBind& rdef_binding_edram = rdef_bindings[rdef_binding_index++];
    rdef_binding_edram.name_ptr = rdef_xe_edram_name_ptr;
    rdef_binding_edram.type = dxbc::RdefInputType::kUAVRWTyped;
    rdef_binding_edram.return_type = dxbc::ResourceReturnType::kUInt;
    rdef_binding_edram.dimension = dxbc::RdefDimension::kUAVBuffer;
    rdef_binding_edram.sample_count = UINT32_MAX;
    rdef_binding_edram.bind_count = 1;
    rdef_binding_edram.flags = format_is_64bpp ? dxbc::kRdefInputFlags2Component : 0;
    // xe_edram_dump_offsets
    dxbc::RdefInputBind& rdef_binding_offsets = rdef_bindings[rdef_binding_index++];
    rdef_binding_offsets.name_ptr = rdef_xe_edram_dump_offsets_name_ptr;
    rdef_binding_offsets.type = dxbc::RdefInputType::kCbuffer;
    rdef_binding_offsets.bind_point = kDumpCbufferOffsets;
    rdef_binding_offsets.bind_count = 1;
    rdef_binding_offsets.flags = dxbc::kRdefInputFlagUserPacked;
    rdef_binding_offsets.id = kDumpCbufferOffsets;
    // xe_edram_dump_pitches
    dxbc::RdefInputBind& rdef_binding_pitches = rdef_bindings[rdef_binding_index++];
    rdef_binding_pitches.name_ptr = rdef_xe_edram_dump_pitches_name_ptr;
    rdef_binding_pitches.type = dxbc::RdefInputType::kCbuffer;
    rdef_binding_pitches.bind_point = kDumpCbufferPitches;
    rdef_binding_pitches.bind_count = 1;
    rdef_binding_pitches.flags = dxbc::kRdefInputFlagUserPacked;
    rdef_binding_pitches.id = kDumpCbufferPitches;
  }

  // Header.
  {
    auto& rdef_header =
        *reinterpret_cast<dxbc::RdefHeader*>(built_shader_.data() + rdef_position_dwords);
    rdef_header.cbuffer_count = kDumpCbufferCount;
    rdef_header.cbuffers_ptr =
        uint32_t((rdef_cbuffer_position_dwords - rdef_position_dwords) * sizeof(uint32_t));
    rdef_header.input_bind_count = rdef_binding_count;
    rdef_header.input_binds_ptr =
        uint32_t((rdef_binding_position_dwords - rdef_position_dwords) * sizeof(uint32_t));
    rdef_header.shader_model = dxbc::RdefShaderModel::kComputeShader5_1;
    rdef_header.compile_flags =
        dxbc::kCompileFlagNoPreshader | dxbc::kCompileFlagPreferFlowControl |
        dxbc::kCompileFlagIeeeStrictness | dxbc::kCompileFlagAllResourcesBound;
    // Generator name is right after the header.
    rdef_header.generator_name_ptr = sizeof(dxbc::RdefHeader);
    rdef_header.fourcc = dxbc::RdefHeader::FourCC::k5_1;
    rdef_header.InitializeSizes();
  }

  {
    auto& blob_header =
        *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
    blob_header.fourcc = dxbc::BlobHeader::FourCC::kResourceDefinition;
    blob_position_dwords = uint32_t(built_shader_.size());
    blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                             built_shader_[blob_offset_position_dwords++];
  }

  // ***************************************************************************
  // Input and output signatures (empty)
  // ***************************************************************************

  for (uint32_t i = 0; i < 2; ++i) {
    built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
    uint32_t signature_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
    built_shader_.resize(signature_position_dwords + sizeof(dxbc::Signature) / sizeof(uint32_t));
    {
      auto& signature =
          *reinterpret_cast<dxbc::Signature*>(built_shader_.data() + signature_position_dwords);
      // Empty - just set parameter pointer to the end.
      signature.parameter_info_ptr = sizeof(dxbc::Signature);
    }
    {
      auto& blob_header =
          *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
      blob_header.fourcc = i ? dxbc::BlobHeader::FourCC::kOutputSignature
                             : dxbc::BlobHeader::FourCC::kInputSignature;
      blob_position_dwords = uint32_t(built_shader_.size());
      blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                               built_shader_[blob_offset_position_dwords++];
    }
  }

  // ***************************************************************************
  // Shader program
  // ***************************************************************************

  built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
  uint32_t shex_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
  built_shader_.resize(shex_position_dwords);

  built_shader_.push_back(dxbc::VersionToken(dxbc::ProgramType::kComputeShader, 5, 1));
  // Reserve space for the length token.
  built_shader_.push_back(0);

  dxbc::Statistics stat;
  std::memset(&stat, 0, sizeof(dxbc::Statistics));
  dxbc::Assembler a(built_shader_, stat);

  a.OpDclGlobalFlags(dxbc::kGlobalFlagAllResourcesBound);
  a.OpDclConstantBuffer(
      dxbc::Src::CB(dxbc::Src::Dcl, kDumpCbufferOffsets, kDumpCbufferOffsets, kDumpCbufferOffsets),
      1);
  a.OpDclConstantBuffer(
      dxbc::Src::CB(dxbc::Src::Dcl, kDumpCbufferPitches, kDumpCbufferPitches, kDumpCbufferPitches),
      1);
  // Source texture.
  dxbc::ResourceDimension source_dimension = key.msaa_samples != xenos::MsaaSamples::k1X
                                                 ? dxbc::ResourceDimension::kTexture2DMS
                                                 : dxbc::ResourceDimension::kTexture2D;
  a.OpDclResource(
      source_dimension,
      dxbc::ResourceReturnTypeX4Token(source_is_uint ? dxbc::ResourceReturnType::kUInt
                                                     : dxbc::ResourceReturnType::kFloat),
      dxbc::Src::T(dxbc::Src::Dcl, 0, 0, 0));
  // Source stencil texture.
  if (key.is_depth) {
    a.OpDclResource(source_dimension,
                    dxbc::ResourceReturnTypeX4Token(dxbc::ResourceReturnType::kUInt),
                    dxbc::Src::T(dxbc::Src::Dcl, 1, 1, 1));
  }
  // EDRAM buffer.
  a.OpDclUnorderedAccessViewTyped(dxbc::ResourceDimension::kBuffer, 0,
                                  dxbc::ResourceReturnTypeX4Token(dxbc::ResourceReturnType::kUInt),
                                  dxbc::Src::U(dxbc::Src::Dcl, 0, 0, 0));
  a.OpDclInput(dxbc::Dest::VThreadID(0b0011));
  // r0 - addressing before the load, then addressing and conversion scratch
  // r1 - addressing scratch before the load, then data
  stat.temp_register_count = 2;
  a.OpDclTemps(stat.temp_register_count);
  // There's no strict dependency on the group size here, for simplicity of
  // calculations especially with resolution scaling, dividing manually (as the
  // group size is not unlimited). The only restriction is that an integer
  // multiple of it must be 80x16 samples (and no larger than that) for 32bpp,
  // or 40x16 samples for 64bpp (because only a half of the pair of tiles may
  // need to be dumped). The group size limit in Direct3D 11 is 1024, and 40x16
  // fits in it, while 80x16 doesn't.
  a.OpDclThreadGroup(40, 16, 1);

  uint32_t draw_resolution_scale_x = this->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = this->draw_resolution_scale_y();

  // For now, as the exact addressing in 64bpp render targets relatively to
  // 32bpp is unknown, treating 64bpp tiles as storing 40x16 samples rather than
  // 80x16 for simplicity of addressing into the texture.

  uint32_t tile_width =
      (xenos::kEdramTileWidthSamples * draw_resolution_scale_x) >> uint32_t(format_is_64bpp);
  uint32_t tile_height = xenos::kEdramTileHeightSamples * draw_resolution_scale_y;

  // Get the parts of the address - tile row index within the dispatch to r0.zw,
  // sample Y within the tile to r0.xy.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = X tile position
  // r0.w = Y tile position
  a.OpUDiv(dxbc::Dest::R(0, 0b1100), dxbc::Dest::R(0, 0b0011), dxbc::Src::VThreadID(0b01000100),
           dxbc::Src::LU(tile_width, tile_height, tile_width, tile_height));

  // Extract the dump rectangle tile row pitch to r1.x.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = X tile position
  // r0.w = Y tile position
  // r1.x = dump rectangle pitch in tiles
  a.OpUBFE(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(xenos::kEdramPitchTilesBits), dxbc::Src::LU(0),
           dxbc::Src::CB(kDumpCbufferPitches, kDumpCbufferPitches, 0, dxbc::Src::kXXXX));
  // Get the tile index in the EDRAM relative to the dump rectangle base tile to
  // r0.w.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = free
  // r0.w = tile index relative to the dump rectangle base
  // r1.x = free
  a.OpUMAd(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kWWWW),
           dxbc::Src::R(1, dxbc::Src::kXXXX), dxbc::Src::R(0, dxbc::Src::kZZZZ));

  // Extract the index of the first tile (taking EDRAM addressing wrapping into
  // account) of the dispatch in the EDRAM to r0.z.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = first EDRAM tile index in the dispatch
  // r0.w = tile index relative to the dump rectangle base
  a.OpUBFE(dxbc::Dest::R(0, 0b0100), dxbc::Src::LU(xenos::kEdramBaseTilesBits + 1),
           dxbc::Src::LU(0),
           dxbc::Src::CB(kDumpCbufferOffsets, kDumpCbufferOffsets, 0, dxbc::Src::kXXXX));
  // Add the base tile in the dispatch to the dispatch-local tile index to r0.w,
  // not wrapping yet so in case of a wraparound, the address relative to the
  // base in the texture after subtraction of the base won't be negative.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = free
  // r0.w = non-wrapped tile index in the EDRAM
  a.OpIAdd(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kWWWW),
           dxbc::Src::R(0, dxbc::Src::kZZZZ));
  // Wrap the address of the tile in the EDRAM to r0.z.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = wrapped tile index in the EDRAM
  // r0.w = non-wrapped tile index in the EDRAM
  a.OpAnd(dxbc::Dest::R(0, 0b0100), dxbc::Src::R(0, dxbc::Src::kWWWW),
          dxbc::Src::LU(xenos::kEdramTileCount - 1));
  // Convert the tile index to samples and add the X sample index to it to r0.z.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = tile sample offset in the EDRAM plus X sample offset
  // r0.w = non-wrapped tile index in the EDRAM
  a.OpUMAd(dxbc::Dest::R(0, 0b0100), dxbc::Src::R(0, dxbc::Src::kZZZZ),
           dxbc::Src::LU(draw_resolution_scale_x * draw_resolution_scale_y *
                         (xenos::kEdramTileWidthSamples >> uint32_t(format_is_64bpp)) *
                         xenos::kEdramTileHeightSamples),
           dxbc::Src::R(0, dxbc::Src::kXXXX));
  // Add the contribution of the Y sample position within the tile to the sample
  // address in the EDRAM to r0.z.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = sample offset in the EDRAM without the depth column swapping
  // r0.w = non-wrapped tile index in the EDRAM
  a.OpUMAd(dxbc::Dest::R(0, 0b0100), dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::LU(tile_width),
           dxbc::Src::R(0, dxbc::Src::kZZZZ));
  if (key.is_depth) {
    uint32_t tile_width_half = tile_width >> 1;
    // Get which 40-sample half within the tile is being processed to r1.x.
    // r0.x = X sample position within the tile
    // r0.y = Y sample position within the tile
    // r0.z = sample offset in the EDRAM without the depth column swapping
    // r0.w = non-wrapped tile index in the EDRAM
    // r1.x = 0xFFFFFFFF if in the right 40-sample half, 0 otherwise
    a.OpUGE(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(0, dxbc::Src::kXXXX),
            dxbc::Src::LU(tile_width_half));
    // Get the offset needed to swap 40-sample halves for depth.
    // r0.x = X sample position within the tile
    // r0.y = Y sample position within the tile
    // r0.z = sample offset in the EDRAM without the depth column swapping
    // r0.w = non-wrapped tile index in the EDRAM
    // r1.x = depth half-tile flipping offset
    a.OpMovC(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX),
             dxbc::Src::LI(-int32_t(tile_width_half)), dxbc::Src::LI(int32_t(tile_width_half)));
    // Swap 40-sample columns in the depth buffer in the destination address in
    // r0.w to get the final address of the sample in EDRAM.
    // r0.x = X sample position within the tile
    // r0.y = Y sample position within the tile
    // r0.z = sample offset in the EDRAM
    // r0.w = non-wrapped tile index in the EDRAM
    // r1.x = free
    a.OpIAdd(dxbc::Dest::R(0, 0b0100), dxbc::Src::R(0, dxbc::Src::kZZZZ),
             dxbc::Src::R(1, dxbc::Src::kXXXX));
  }

  // Extract the source texture base tile index to r1.x.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = sample offset in the EDRAM
  // r0.w = non-wrapped tile index in the EDRAM
  // r1.x = source texture base tile index
  a.OpUBFE(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(xenos::kEdramBaseTilesBits),
           dxbc::Src::LU(xenos::kEdramBaseTilesBits + 1),
           dxbc::Src::CB(kDumpCbufferOffsets, kDumpCbufferOffsets, 0, dxbc::Src::kXXXX));
  // Get the linear tile index within the source texture to r0.w.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = sample offset in the EDRAM
  // r0.w = linear tile index in the source texture
  // r1.x = free
  a.OpIAdd(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kWWWW),
           -dxbc::Src::R(1, dxbc::Src::kXXXX));
  // Get the source texture pitch in tiles to r1.x.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = sample offset in the EDRAM
  // r0.w = linear tile index in the source texture
  // r1.x = source texture pitch in tiles
  a.OpUBFE(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(xenos::kEdramPitchTilesBits),
           dxbc::Src::LU(xenos::kEdramPitchTilesBits),
           dxbc::Src::CB(kDumpCbufferPitches, kDumpCbufferPitches, 0, dxbc::Src::kXXXX));
  // Split the linear tile index in the source texture into X and Y in tiles.
  // r0.x = X sample position within the tile
  // r0.y = Y sample position within the tile
  // r0.z = sample offset in the EDRAM
  // r0.w = X tile index within the tile row in the source texture
  // r1.x = Y tile row index within the source texture
  a.OpUDiv(dxbc::Dest::R(1, 0b0001), dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kWWWW),
           dxbc::Src::R(1, dxbc::Src::kXXXX));
  // Add the source texture tile X offset to the source texture sample X
  // coordinate.
  // r0.x = X sample position within the source texture
  // r0.y = Y sample position within the tile
  // r0.z = sample offset in the EDRAM
  // r0.w = free
  // r1.x = Y tile row index within the source texture
  a.OpUMAd(dxbc::Dest::R(0, 0b0001), dxbc::Src::R(0, dxbc::Src::kWWWW), dxbc::Src::LU(tile_width),
           dxbc::Src::R(0, dxbc::Src::kXXXX));
  // Add the source texture tile Y offset to the source texture sample Y
  // coordinate.
  // r0.x = X sample position within the source texture
  // r0.y = Y sample position within the source texture
  // r0.z = sample offset in the EDRAM
  // r1.x = free
  a.OpUMAd(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(1, dxbc::Src::kXXXX),
           dxbc::Src::LU(xenos::kEdramTileHeightSamples * draw_resolution_scale_y),
           dxbc::Src::R(0, dxbc::Src::kYYYY));
  // Will be using the source texture coordinates from r0.xy, and for
  // single-sampled source, LOD from r0.w.
  dxbc::Src source_address_src(dxbc::Src::R(0, 0b11000100));
  if (key.msaa_samples >= xenos::MsaaSamples::k2X) {
    if (key.msaa_samples >= xenos::MsaaSamples::k4X) {
      // 4x MSAA source texture sample index - bit 0 for horizontal, bit 1 for
      // vertical.
      // Extract the horizontal sample index to r0.w.
      // r0.x = X sample position within the source texture
      // r0.y = Y sample position within the source texture
      // r0.z = sample offset in the EDRAM
      // r0.w = horizontal sample index within the source pixel
      a.OpAnd(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::LU(1));
      // Insert the vertical sample index to r0.w.
      // r0.x = X sample position within the source texture
      // r0.y = Y sample position within the source texture
      // r0.z = sample offset in the EDRAM
      // r0.w = sample index within the source pixel
      a.OpBFI(dxbc::Dest::R(0, 0b1000), dxbc::Src::LU(1), dxbc::Src::LU(1),
              dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::R(0, dxbc::Src::kWWWW));
      // Convert sample to pixel coordinates in the source texture to r0.xy.
      // r0.x = X pixel position within the source texture
      // r0.y = Y pixel position within the source texture
      // r0.z = sample offset in the EDRAM
      // r0.w = sample index within the source pixel
      a.OpUShR(dxbc::Dest::R(0, 0b0011), dxbc::Src::R(0), dxbc::Src::LU(1));
    } else {
      // 2x MSAA source texture sample index.
      // Extract the vertical sample index to r0.w.
      // r0.x = X pixel position within the source texture
      // r0.y = Y sample position within the source texture
      // r0.z = sample offset in the EDRAM
      // r0.w = vertical sample index within the destination pixel
      a.OpAnd(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::LU(1));
      // Convert the 2x MSAA sample index from the guest to Direct3D 10.1+.
      // r0.x = X pixel position within the source texture
      // r0.y = Y sample position within the source texture
      // r0.z = sample offset in the EDRAM
      // r0.w = sample index within the source pixel
      a.OpMovC(dxbc::Dest::R(0, 0b1000), dxbc::Src::R(0, dxbc::Src::kWWWW),
               dxbc::Src::LU(draw_util::GetD3D10SampleIndexForGuest2xMSAA(1, msaa_2x_supported_)),
               dxbc::Src::LU(draw_util::GetD3D10SampleIndexForGuest2xMSAA(0, msaa_2x_supported_)));
      // Convert sample Y to pixel Y in the source texture to r0.y.
      // r0.x = X pixel position within the source texture
      // r0.y = Y pixel position within the source texture
      // r0.z = sample offset in the EDRAM
      // r0.w = sample index within the source pixel
      a.OpUShR(dxbc::Dest::R(0, 0b0010), dxbc::Src::R(0, dxbc::Src::kYYYY), dxbc::Src::LU(1));
    }
    // Load the source to r1.
    // r0.x = X pixel position within the source texture if stencil is needed
    // r0.y = Y pixel position within the source texture if stencil is needed
    // r0.z = sample offset in the EDRAM
    // r0.w = sample index within the source pixel if stencil is needed
    // r1 = source texel value
    a.OpLdMS(dxbc::Dest::R(1, (1 << source_component_count) - 1), source_address_src, 0b0011,
             dxbc::Src::T(0, 0), dxbc::Src::R(0, dxbc::Src::kWWWW));
    if (key.is_depth) {
      // Load the source stencil to r1.y.
      // r0.x = free
      // r0.y = free
      // r0.z = sample offset in the EDRAM
      // r0.w = free
      // r1.x = source depth value
      // r1.y = source stencil value
      a.OpLdMS(dxbc::Dest::R(1, 0b0010), source_address_src, 0b0011, dxbc::Src::T(1, 1),
               dxbc::Src::R(0, dxbc::Src::kWWWW));
    }
  } else {
    // Write the LOD index (0) to the register with texture coordinates for
    // loading from the single-sampled source texture.
    // r0.x = X pixel position within the source texture
    // r0.y = Y pixel position within the source texture
    // r0.z = sample offset in the EDRAM
    // r0.w = LOD for the texture load (zero)
    a.OpMov(dxbc::Dest::R(0, 0b1000), dxbc::Src::LF(0.0f));
    // Load the source to r1.
    // r0.x = X pixel position within the source texture if stencil is needed
    // r0.y = Y pixel position within the source texture if stencil is needed
    // r0.z = sample offset in the EDRAM
    // r0.w = LOD for the texture load (zero)
    // r1 = source texel value
    a.OpLd(dxbc::Dest::R(1, (1 << source_component_count) - 1), source_address_src, 0b1011,
           dxbc::Src::T(0, 0));
    if (key.is_depth) {
      // Load the source stencil to r1.y.
      // r0.x = free
      // r0.y = free
      // r0.z = sample offset in the EDRAM
      // r0.w = free
      // r1.x = source depth value
      // r1.y = source stencil value
      a.OpLd(dxbc::Dest::R(1, 0b0010), source_address_src, 0b1011, dxbc::Src::T(1, 1));
    }
  }

  // Pack in the needed format, writing the result to r1.x for 32bpp or r1.xy
  // for 64bpp.
  // r0.xyw are usable as temporary storage.
  if (key.is_depth) {
    switch (key.GetDepthFormat()) {
      case xenos::DepthRenderTargetFormat::kD24S8:
        // Round to the nearest even integer. This seems to be the correct
        // conversion, adding +0.5 and rounding towards zero results in red
        // instead of black in the 4D5307E6 clear shader.
        a.OpMul(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX),
                dxbc::Src::LF(float(0xFFFFFF)));
        a.OpRoundNE(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX));
        a.OpFToU(dxbc::Dest::R(1, 0b0001), dxbc::Src::R(1, dxbc::Src::kXXXX));
        break;
      case xenos::DepthRenderTargetFormat::kD24FS8:
        // Convert to [0, 2) float24 from [0, 1) float32, using r0.x as
        // temporary.
        // When converting the depth in pixel shaders, it's always exact,
        // truncating not to insert additional rounding instructions.
        DxbcShaderTranslator::PreClampedDepthTo20e4(
            a, 1, 0, 1, 0, 0, 0, !depth_float24_convert_in_pixel_shader() && depth_float24_round(),
            true);
        break;
    }
    // Combine 24-bit depth and stencil into r1.x.
    a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(24), dxbc::Src::LU(8),
            dxbc::Src::R(1, dxbc::Src::kXXXX), dxbc::Src::R(1, dxbc::Src::kYYYY));
  } else {
    switch (key.GetColorFormat()) {
      case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
        // 8_8_8_8_GAMMA is represented by linear stored in
        // R16G16B16A16_UNORM.
        assert_false(source_is_uint);
        for (uint32_t i = 0; i < 3; ++i) {
          DxbcShaderTranslator::PreSaturatedLinearToPWLGamma(a, 1, i, 1, i, 0, 0, 0, 1);
        }
        a.OpMAd(dxbc::Dest::R(1), dxbc::Src::R(1), dxbc::Src::LF(255.0f), dxbc::Src::LF(0.5f));
        a.OpFToU(dxbc::Dest::R(1), dxbc::Src::R(1));
        for (uint32_t i = 1; i < 4; ++i) {
          a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(8), dxbc::Src::LU(i * 8),
                  dxbc::Src::R(1).Select(i), dxbc::Src::R(1, dxbc::Src::kXXXX));
        }
        break;
      case xenos::ColorRenderTargetFormat::k_8_8_8_8:
        if (!source_is_uint) {
          a.OpMAd(dxbc::Dest::R(1), dxbc::Src::R(1), dxbc::Src::LF(255.0f), dxbc::Src::LF(0.5f));
          a.OpFToU(dxbc::Dest::R(1), dxbc::Src::R(1));
        }
        for (uint32_t i = 1; i < 4; ++i) {
          a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(8), dxbc::Src::LU(i * 8),
                  dxbc::Src::R(1).Select(i), dxbc::Src::R(1, dxbc::Src::kXXXX));
        }
        break;
      case xenos::ColorRenderTargetFormat::k_2_10_10_10:
      case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
        if (!source_is_uint) {
          a.OpMAd(dxbc::Dest::R(1), dxbc::Src::R(1), dxbc::Src::LF(1023.0f, 1023.0f, 1023.0f, 3.0f),
                  dxbc::Src::LF(0.5f));
          a.OpFToU(dxbc::Dest::R(1), dxbc::Src::R(1));
        }
        for (uint32_t i = 1; i < 4; ++i) {
          a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(i == 3 ? 2 : 10), dxbc::Src::LU(i * 10),
                  dxbc::Src::R(1).Select(i), dxbc::Src::R(1, dxbc::Src::kXXXX));
        }
        break;
      case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
      case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16:
        // Float16 has a wider range for both color and alpha, also NaNs.
        // Color - clamp and convert.
        // Convert red in r1.x to the result register r1.x - the same, but
        // UnclampedFloat32To7e3 allows that - using r0.x as a temporary.
        DxbcShaderTranslator::UnclampedFloat32To7e3(a, 1, 0, 1, 0, 0, 0);
        for (uint32_t i = 1; i < 3; ++i) {
          // Convert green and blue to a temporary register r0.x using r0.y
          // as an internal temporary, then insert them into the result in
          // r1.x.
          DxbcShaderTranslator::UnclampedFloat32To7e3(a, 0, 0, 1, i, 0, 1);
          a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(10), dxbc::Src::LU(i * 10),
                  dxbc::Src::R(0, dxbc::Src::kXXXX), dxbc::Src::R(1, dxbc::Src::kXXXX));
        }
        // Alpha - saturate and convert.
        a.OpMov(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW), true);
        a.OpMAd(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW), dxbc::Src::LF(3.0f),
                dxbc::Src::LF(0.5f));
        a.OpFToU(dxbc::Dest::R(1, 0b1000), dxbc::Src::R(1, dxbc::Src::kWWWW));
        a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(2), dxbc::Src::LU(30),
                dxbc::Src::R(1, dxbc::Src::kWWWW), dxbc::Src::R(1, dxbc::Src::kXXXX));
        break;
      case xenos::ColorRenderTargetFormat::k_16_16:
      case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
        assert_true(source_is_uint);
        a.OpBFI(dxbc::Dest::R(1, 0b0001), dxbc::Src::LU(16), dxbc::Src::LU(16),
                dxbc::Src::R(1, dxbc::Src::kYYYY), dxbc::Src::R(1, dxbc::Src::kXXXX));
        break;
      case xenos::ColorRenderTargetFormat::k_16_16_16_16:
      case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
        assert_true(source_is_uint);
        a.OpBFI(dxbc::Dest::R(1, 0b0011), dxbc::Src::LU(16), dxbc::Src::LU(16),
                dxbc::Src::R(1, 0b1101), dxbc::Src::R(1, 0b1000));
        break;
      case xenos::ColorRenderTargetFormat::k_32_FLOAT:
      case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
        assert_true(source_is_uint);
        // Already has the needed representation.
        break;
    }
  }

  // Write the sample to the destination address stored in r0.z.
  a.OpStoreUAVTyped(dxbc::Dest::U(0, 0), dxbc::Src::R(0, dxbc::Src::kZZZZ), 1,
                    dxbc::Src::R(1, format_is_64bpp ? 0b0100 : dxbc::Src::kXXXX));

  a.OpRet();

  // Write the shader program length in dwords.
  built_shader_[shex_position_dwords + 1] = uint32_t(built_shader_.size()) - shex_position_dwords;

  {
    auto& blob_header =
        *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
    blob_header.fourcc = dxbc::BlobHeader::FourCC::kShaderEx;
    blob_position_dwords = uint32_t(built_shader_.size());
    blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                             built_shader_[blob_offset_position_dwords++];
  }

  // ***************************************************************************
  // Statistics
  // ***************************************************************************

  built_shader_[blob_offset_position_dwords] = uint32_t(blob_position_dwords * sizeof(uint32_t));
  uint32_t stat_position_dwords = blob_position_dwords + kBlobHeaderSizeDwords;
  built_shader_.resize(stat_position_dwords + sizeof(dxbc::Statistics) / sizeof(uint32_t));
  std::memcpy(built_shader_.data() + stat_position_dwords, &stat, sizeof(dxbc::Statistics));
  {
    auto& blob_header =
        *reinterpret_cast<dxbc::BlobHeader*>(built_shader_.data() + blob_position_dwords);
    blob_header.fourcc = dxbc::BlobHeader::FourCC::kStatistics;
    blob_position_dwords = uint32_t(built_shader_.size());
    blob_header.size_bytes = (blob_position_dwords - kBlobHeaderSizeDwords) * sizeof(uint32_t) -
                             built_shader_[blob_offset_position_dwords++];
  }

  // ***************************************************************************
  // Container header
  // ***************************************************************************

  uint32_t built_shader_size_bytes = uint32_t(built_shader_.size() * sizeof(uint32_t));
  {
    auto& container_header = *reinterpret_cast<dxbc::ContainerHeader*>(built_shader_.data());
    container_header.InitializeIdentification();
    container_header.size_bytes = built_shader_size_bytes;
    container_header.blob_count = kBlobCount;
    CalculateDXBCChecksum(reinterpret_cast<unsigned char*>(built_shader_.data()),
                          static_cast<unsigned int>(built_shader_size_bytes),
                          reinterpret_cast<unsigned int*>(&container_header.hash));
  }

  // ***************************************************************************
  // Pipeline
  // ***************************************************************************
  ID3D12PipelineState* pipeline = ui::d3d12::util::CreateComputePipeline(
      command_processor_.GetD3D12Provider().GetDevice(), built_shader_.data(),
      built_shader_size_bytes,
      key.is_depth ? dump_root_signature_depth_ : dump_root_signature_color_);
  const char* format_name = key.is_depth
                                ? xenos::GetDepthRenderTargetFormatName(key.GetDepthFormat())
                                : xenos::GetColorRenderTargetFormatName(key.GetColorFormat());
  if (pipeline) {
    std::u16string pipeline_name = rex::string::to_utf16(
        fmt::format("RT Dump {} {}xMSAA", format_name, uint32_t(1) << uint32_t(key.msaa_samples)));
    pipeline->SetName(reinterpret_cast<LPCWSTR>(pipeline_name.c_str()));
  } else {
    REXGPU_ERROR(
        "D3D12RenderTargetCache: Failed to create a render target dumping "
        "pipeline for {}-sample render targets with format {}",
        uint32_t(1) << uint32_t(key.msaa_samples), format_name);
  }
  // Even if creation fails, still store the null pointer not to try to create
  // again.
  dump_pipelines_.emplace(key, pipeline);
  return pipeline;
}

// [NR-DRES] The direct resolve compute shader, compiled at runtime per
// {msaa, owner format, is_depth} variant. One thread = one guest dest pixel:
// load the owning host render target's texel (the sample the resolve
// selects), pack it to the guest 32bpp value EXACTLY the way the dump shader
// does (same constants, same rounding - the vendored fast copy then moves
// that value raw, so dump conversion + tiled store == the whole legacy
// pipeline), compute the tiled guest address the way resolve.xesli does, and
// write it endian-swapped to the shared memory UAV.
static const char kNrDirectResolveShaderSource[] = R"nrdres(
cbuffer XeDrConstants : register(b0) {
  uint xe_dr_dest_info;    // RB_COPY_DEST_INFO raw: endian 2:0, swap bit 24
  uint xe_dr_dest_coord;   // pitch macro tiles 9:0, offx_div8 23:20, offy_div8 27:24
  uint xe_dr_dest_base;    // guest byte address the dest offsets are relative to
  uint xe_dr_size;         // resolve width | height<<16 (pixels)
  uint xe_dr_src_origin;   // host RT pixel coords of the resolve origin: x | y<<16
  uint xe_dr_samples;      // host sample list s0|s1<<4|s2<<8|s3<<12, count<<16
  uint xe_dr_bias_bits;    // full only: float bits of exp_bias * (1/count)
  uint xe_dr_fill;         // half-pixel-offset fill: x | y<<8 (0 unscaled)
};
RWBuffer<uint> xe_dr_dest : register(u0);
#if XE_DR_SRC_MS
Texture2DMS<float4> xe_dr_source : register(t0);
#else
Texture2D<float4> xe_dr_source : register(t0);
#endif
#if XE_DR_DEPTH
#if XE_DR_SRC_MS
Texture2DMS<uint2> xe_dr_stencil : register(t1);
#else
Texture2D<uint2> xe_dr_stencil : register(t1);
#endif
#endif

// texture_address.xesli XenosTextureTiledAddress2D.
uint XeDrTiledAddress2DBpp(uint2 p, uint pitch_mt, uint bpp_log2) {
  uint outer = ((p.y >> 5u) * pitch_mt + (p.x >> 5u)) << 6u;
  uint inner = (((p.y >> 1u) & 7u) << 3u) | (p.x & 7u);
  uint oib = (outer | inner) << bpp_log2;
  uint bank = (p.y >> 4u) & 1u;
  uint pipe = ((p.x >> 3u) & 3u) ^ (((p.y >> 3u) & 1u) << 1u);
  return ((p.y & 1u) << 4u) | (pipe << 6u) | (bank << 11u) | (oib & 0xFu) |
         (((oib >> 4u) & 1u) << 5u) | (((oib >> 5u) & 7u) << 8u) |
         ((oib >> 8u) << 12u);
}

uint XeDrTiledAddress2D(uint2 p, uint pitch_mt) {
  return XeDrTiledAddress2DBpp(p, pitch_mt, 2u);
}

// Dest byte address, resolution-scale aware. Unscaled: the guest tiled
// address. Scaled: THE VENDORED BLOBS' scheme, decoded from the embedded
// disassembly of resolve_fast_32bpp_1x2xmsaa_scaled_cs.h - NOT current
// upstream's 8-row guest groups ([[vendored-blobs-freeze-layouts]]): the
// granule is a 16-byte one-row run; final = guest run address * (sx*sy) +
// ((sub_x*sy + sub_y) << 4) + byte_in_run, with sub_x = (host_x/run)%sx,
// sub_y = host_y%sy. The vendored texture-load blobs read this same layout,
// so it is the law regardless of what upstream does today.
uint XeDrDestAddress(uint2 dp, uint pitch_mt, uint bpp_log2) {
#if XE_DR_SCALE_X == 1 && XE_DR_SCALE_Y == 1
  return XeDrTiledAddress2DBpp(dp, pitch_mt, bpp_log2);
#else
  // Run size = the linear span the guest tiling keeps along x: 8 bytes at
  // 1bpb (resolve_full_8bpp_scaled blob: >>3, sub<<3), 16 bytes at 4bpb
  // (resolve_fast_32bpp scaled blob: >>2 on dwords, sub<<4).
  uint run_bytes_log2 = bpp_log2 == 0u ? 3u : 4u;
  uint blocks_per_run = (1u << run_bytes_log2) >> bpp_log2;
  uint host_unit_x = dp.x / blocks_per_run;
  uint guest_unit_x = host_unit_x / uint(XE_DR_SCALE_X);
  uint guest_y = dp.y / uint(XE_DR_SCALE_Y);
  uint sub_x = host_unit_x - guest_unit_x * uint(XE_DR_SCALE_X);
  uint sub_y = dp.y - guest_y * uint(XE_DR_SCALE_Y);
  uint byte_in_run = (dp.x - host_unit_x * blocks_per_run) << bpp_log2;
  uint guest_addr = XeDrTiledAddress2DBpp(uint2(guest_unit_x * blocks_per_run, guest_y),
                                          pitch_mt, bpp_log2);
  return guest_addr * uint(XE_DR_SCALE_X * XE_DR_SCALE_Y) +
         ((sub_x * uint(XE_DR_SCALE_Y) + sub_y) << run_bytes_log2) + byte_in_run;
#endif
}

// endian.xesli XeEndianSwap32 (Endian128 low bits; 8in64/8in128 do not apply
// to a single dword and pass through, same as the vendored fast shader).
uint XeDrEndianSwap32(uint v, uint endian) {
  if (endian == 1u || endian == 2u) {
    v = ((v & 0x00FF00FFu) << 8u) | ((v & 0xFF00FF00u) >> 8u);
  }
  if (endian == 2u || endian == 3u) {
    v = (v << 16u) | (v >> 16u);
  }
  return v;
}

#if XE_DR_DEPTH && XE_DR_DEPTH_FLOAT
// pixel_formats.xesli XeFloat32To20e4; the host D32F holds the guest value
// remapped to [0, 0.5], so the caller multiplies by 2 first.
uint XeDrFloat32To20e4(uint f32u32) {
  f32u32 = min((f32u32 <= 0x7FFFFFFFu) ? f32u32 : 0u, 0x3FFFFFF8u);
  uint denorm = ((f32u32 & 0x7FFFFFu) | 0x800000u) >> min(113u - (f32u32 >> 23u), 24u);
  uint f24 = (f32u32 < 0x38800000u) ? denorm : (f32u32 + 0xC8000000u);
#if XE_DR_F24_ROUND
  f24 += 3u + ((f24 >> 3u) & 1u);
#endif
  return (f24 >> 3u) & 0xFFFFFFu;
}
#endif

#if !XE_DR_DEPTH && (XE_DR_COLOR_FMT == 1 || XE_DR_SRC_GAMMA16)
// The 360 piecewise-linear gamma ramp (DxbcShaderTranslator
// PreSaturatedLinearToPWLGamma): host stores linear in R16G16B16A16_UNORM.
float XeDrLinearToPWLGamma(float lin) {
  float scale, offset;
  if (lin >= 128.0f / 1023.0f) {
    bool hi = lin >= 512.0f / 1023.0f;
    scale = hi ? (1023.0f / 8.0f) : (1023.0f / 4.0f);
    offset = hi ? (128.0f / 255.0f) : (64.0f / 255.0f);
  } else {
    bool mid = lin >= 64.0f / 1023.0f;
    scale = mid ? (1023.0f / 2.0f) : 1023.0f;
    offset = mid ? (32.0f / 255.0f) : 0.0f;
  }
  return trunc(lin * scale) * (1.0f / 255.0f) + offset;
}
#endif

[numthreads(8, 8, 1)]
void main(uint3 xe_dr_thread : SV_DispatchThreadID) {
  uint2 size = uint2(xe_dr_size & 0xFFFFu, xe_dr_size >> 16u);
#if XE_DR_FULL && XE_DR_BPP8
  // 8bpp dest (kFull8bpp): one thread packs 4 horizontally adjacent dest
  // pixels into one dword. resolve_full_8bpp.xesli semantics: the red
  // channel (blue under dest_swap), same averaging/bias as full 32bpp, byte
  // = saturate * 255 + 0.5, NO endian swap. At 1bpp tiling, 8 consecutive x
  // share consecutive byte addresses, so a 4-aligned group is one dword.
  uint2 p4 = uint2(xe_dr_thread.x << 2u, xe_dr_thread.y);
  if (p4.x >= size.x || p4.y >= size.y) {
    return;
  }
  uint2 xe_dr_fill8 = uint2(xe_dr_fill & 0xFFu, (xe_dr_fill >> 8u) & 0xFFu);
  uint xe_dr_count8 = xe_dr_samples >> 16u;
  uint dest_bytes = 0u;
  for (uint pi = 0u; pi < 4u; ++pi) {
    uint2 pe8 = max(uint2(p4.x + pi, p4.y), xe_dr_fill8);
    int2 src8 = int2(pe8) +
                int2(int(xe_dr_src_origin & 0xFFFFu), int(xe_dr_src_origin >> 16u));
    float4 acc8 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint si = 0u; si < xe_dr_count8; ++si) {
      uint s8 = (xe_dr_samples >> (si * 4u)) & 0xFu;
#if XE_DR_SRC_MS
      float4 v8 = xe_dr_source.Load(src8, int(s8));
#else
      float4 v8 = xe_dr_source.Load(int3(src8, 0));
#endif
#if XE_DR_SRC_GAMMA16
      v8.r = floor(XeDrLinearToPWLGamma(v8.r) * 255.0f + 0.5f) * (1.0f / 255.0f);
      v8.g = floor(XeDrLinearToPWLGamma(v8.g) * 255.0f + 0.5f) * (1.0f / 255.0f);
      v8.b = floor(XeDrLinearToPWLGamma(v8.b) * 255.0f + 0.5f) * (1.0f / 255.0f);
#endif
      acc8 += v8;
    }
    float4 c8 = acc8 * asfloat(xe_dr_bias_bits);
    float red8 = ((xe_dr_dest_info & (1u << 24u)) != 0u) ? c8.b : c8.r;
    dest_bytes |= uint(saturate(red8) * 255.0f + 0.5f) << (pi * 8u);
  }
  uint2 dp4 = p4 + (((uint2(xe_dr_dest_coord >> 20u, xe_dr_dest_coord >> 24u) & 0xFu) << 3u) *
                    uint2(XE_DR_SCALE_X, XE_DR_SCALE_Y));
  uint addr8 = xe_dr_dest_base + XeDrDestAddress(dp4, xe_dr_dest_coord & 0x3FFu, 0u);
  xe_dr_dest[addr8 >> 2u] = dest_bytes;
#else
  if (xe_dr_thread.x >= size.x || xe_dr_thread.y >= size.y) {
    return;
  }
  uint2 p = xe_dr_thread.xy;
  uint2 pe = max(p, uint2(xe_dr_fill & 0xFFu, (xe_dr_fill >> 8u) & 0xFFu));
  int2 src = int2(pe) + int2(int(xe_dr_src_origin & 0xFFFFu), int(xe_dr_src_origin >> 16u));
  uint packed;
#if XE_DR_DEPTH
  uint xe_dr_smp0 = xe_dr_samples & 0xFu;
#if XE_DR_SRC_MS
  float depth = xe_dr_source.Load(src, int(xe_dr_smp0)).x;
  uint stencil = xe_dr_stencil.Load(src, int(xe_dr_smp0)).g & 0xFFu;
#else
  float depth = xe_dr_source.Load(int3(src, 0)).x;
  uint stencil = xe_dr_stencil.Load(int3(src, 0)).g & 0xFFu;
#endif
#if XE_DR_DEPTH_FLOAT
  uint depth24 = XeDrFloat32To20e4(asuint(depth * 2.0f));
#else
  // The dump shader rounds to nearest even (mul + round_ne + ftou).
  uint depth24 = uint(round(depth * 16777215.0f));
#endif
  packed = (depth24 << 8u) | stencil;
#elif XE_DR_FULL
  // Full-class color: N-sample average in the legacy shader's exact
  // summation order, exp bias and 1/count folded into one factor on the
  // CPU (both powers of two, so bit-identical to the vendored chain),
  // then swap and dest-format pack per pixel_formats.xesli.
  uint xe_dr_count = xe_dr_samples >> 16u;
  float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint xe_dr_i = 0u; xe_dr_i < xe_dr_count; ++xe_dr_i) {
    uint s = (xe_dr_samples >> (xe_dr_i * 4u)) & 0xFu;
#if XE_DR_SRC_MS
    float4 v = xe_dr_source.Load(src, int(s));
#else
    float4 v = xe_dr_source.Load(int3(src, 0));
#endif
#if XE_DR_SRC_GAMMA16
    // The legacy chain averages the DUMPED gamma bytes; reproduce the dump's
    // quantized PWL output per sample before summing.
    v.r = floor(XeDrLinearToPWLGamma(v.r) * 255.0f + 0.5f) * (1.0f / 255.0f);
    v.g = floor(XeDrLinearToPWLGamma(v.g) * 255.0f + 0.5f) * (1.0f / 255.0f);
    v.b = floor(XeDrLinearToPWLGamma(v.b) * 255.0f + 0.5f) * (1.0f / 255.0f);
#endif
    acc += v;
  }
  float4 c = acc * asfloat(xe_dr_bias_bits);
  if ((xe_dr_dest_info & (1u << 24u)) != 0u) {
    c = c.bgra;
  }
#if XE_DR_PACK == 1
  uint4 q = uint4(saturate(c) * float4(1023.0f, 1023.0f, 1023.0f, 3.0f) + 0.5f);
  packed = q.x | (q.y << 10u) | (q.z << 20u) | (q.w << 30u);
#elif XE_DR_PACK == 2
  uint3 q = uint3(saturate(c.rgb) * float3(2047.0f, 2047.0f, 1023.0f) + 0.5f);
  packed = q.x | (q.y << 11u) | (q.z << 22u);
#elif XE_DR_PACK == 3
  uint3 q = uint3(saturate(c.rgb) * float3(1023.0f, 2047.0f, 2047.0f) + 0.5f);
  packed = q.x | (q.y << 10u) | (q.z << 21u);
#elif XE_DR_PACK == 4
  uint2 q = uint2(saturate(c.rg) * 65535.0f + 0.5f);
  packed = q.x | (q.y << 16u);
#elif XE_DR_PACK == 5
  packed = f32tof16(c.r) | (f32tof16(c.g) << 16u);
#elif XE_DR_PACK == 6
  packed = asuint(c.r);
#else
  uint4 q = uint4(saturate(c) * 255.0f + 0.5f);
  packed = q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
#endif
#else
  uint xe_dr_smp0 = xe_dr_samples & 0xFu;
#if XE_DR_SRC_MS
  float4 c = xe_dr_source.Load(src, int(xe_dr_smp0));
#else
  float4 c = xe_dr_source.Load(int3(src, 0));
#endif
#if XE_DR_COLOR_FMT == 1
  c.r = XeDrLinearToPWLGamma(c.r);
  c.g = XeDrLinearToPWLGamma(c.g);
  c.b = XeDrLinearToPWLGamma(c.b);
#endif
  if ((xe_dr_dest_info & (1u << 24u)) != 0u) {
    c = c.bgra;
  }
#if XE_DR_COLOR_FMT == 2
  uint4 q = uint4(c * float4(1023.0f, 1023.0f, 1023.0f, 3.0f) + 0.5f);
  packed = q.x | (q.y << 10u) | (q.z << 20u) | (q.w << 30u);
#else
  uint4 q = uint4(c * 255.0f + 0.5f);
  packed = q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
#endif
#endif
  uint2 dp = p + (((uint2(xe_dr_dest_coord >> 20u, xe_dr_dest_coord >> 24u) & 0xFu) << 3u) *
                  uint2(XE_DR_SCALE_X, XE_DR_SCALE_Y));
  uint addr = xe_dr_dest_base + XeDrDestAddress(dp, xe_dr_dest_coord & 0x3FFu, 2u);
  xe_dr_dest[addr >> 2u] = XeDrEndianSwap32(packed, xe_dr_dest_info & 7u);
#endif
}
)nrdres";

typedef HRESULT(WINAPI* PFN_NrD3DCompile)(LPCVOID src_data, SIZE_T src_size,
                                          LPCSTR source_name, const D3D_SHADER_MACRO* defines,
                                          ID3DInclude* include, LPCSTR entry, LPCSTR target,
                                          UINT flags1, UINT flags2, ID3DBlob** code,
                                          ID3DBlob** errors);

static PFN_NrD3DCompile NrGetD3DCompile() {
  static PFN_NrD3DCompile compile = [] {
    HMODULE library = LoadLibraryW(L"D3DCompiler_47.dll");
    return library ? reinterpret_cast<PFN_NrD3DCompile>(GetProcAddress(library, "D3DCompile"))
                   : nullptr;
  }();
  return compile;
}

ID3D12PipelineState* D3D12RenderTargetCache::GetOrCreateDirectResolvePipeline(
    DumpPipelineKey key, bool full, uint32_t pack_class, bool src_gamma16) {
  DumpPipelineKey map_key = key;
  map_key.key |= (full ? 1u : 0u) << 16 | (pack_class & 7u) << 17 | (src_gamma16 ? 1u : 0u) << 20;
  auto pipeline_it = direct_resolve_pipelines_.find(map_key);
  if (pipeline_it != direct_resolve_pipelines_.end()) {
    return pipeline_it->second;
  }
  ID3D12PipelineState* pipeline = nullptr;
  PFN_NrD3DCompile compile = NrGetD3DCompile();
  ID3D12RootSignature* root_signature =
      key.is_depth ? direct_resolve_root_signature_depth_ : direct_resolve_root_signature_color_;
  if (compile && root_signature) {
    char define_ms[2], define_depth[2], define_depth_float[2], define_round[2], define_fmt[2];
    char define_full[2], define_pack[2], define_gamma16[2];
    bool is_msaa = key.msaa_samples != xenos::MsaaSamples::k1X;
    bool depth_float = key.is_depth && key.GetDepthFormat() == xenos::DepthRenderTargetFormat::kD24FS8;
    bool f24_round = !depth_float24_convert_in_pixel_shader() && depth_float24_round();
    uint32_t color_fmt = 0;
    if (!key.is_depth) {
      switch (key.GetColorFormat()) {
        case xenos::ColorRenderTargetFormat::k_8_8_8_8:
          color_fmt = 0;
          break;
        case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
          color_fmt = gamma_render_target_as_unorm16_ ? 1 : 0;
          break;
        case xenos::ColorRenderTargetFormat::k_2_10_10_10:
        case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
          color_fmt = 2;
          break;
        default:
          // The preflight only asks for the implemented set; anything else is
          // a bug there, refuse the pipeline.
          compile = nullptr;
          break;
      }
    }
    if (compile) {
      std::snprintf(define_ms, sizeof(define_ms), "%u", is_msaa ? 1u : 0u);
      std::snprintf(define_depth, sizeof(define_depth), "%u", key.is_depth ? 1u : 0u);
      std::snprintf(define_depth_float, sizeof(define_depth_float), "%u", depth_float ? 1u : 0u);
      std::snprintf(define_round, sizeof(define_round), "%u", f24_round ? 1u : 0u);
      std::snprintf(define_fmt, sizeof(define_fmt), "%u", color_fmt);
      std::snprintf(define_full, sizeof(define_full), "%u", full ? 1u : 0u);
      std::snprintf(define_pack, sizeof(define_pack), "%u", pack_class & 7u);
      std::snprintf(define_gamma16, sizeof(define_gamma16), "%u", src_gamma16 ? 1u : 0u);
      // pack_class 7 = the 8bpp dest shader (kFull8bpp).
      char define_bpp8[2];
      std::snprintf(define_bpp8, sizeof(define_bpp8), "%u",
                    (full && (pack_class & 7u) == 7u) ? 1u : 0u);
      // The resolution scale is fixed for the session, so it is a plain
      // compile-time constant, not a variant key bit.
      char define_scale_x[2], define_scale_y[2];
      std::snprintf(define_scale_x, sizeof(define_scale_x), "%u", draw_resolution_scale_x());
      std::snprintf(define_scale_y, sizeof(define_scale_y), "%u", draw_resolution_scale_y());
      const D3D_SHADER_MACRO defines[] = {
          {"XE_DR_SRC_MS", define_ms},         {"XE_DR_DEPTH", define_depth},
          {"XE_DR_DEPTH_FLOAT", define_depth_float}, {"XE_DR_F24_ROUND", define_round},
          {"XE_DR_COLOR_FMT", define_fmt},     {"XE_DR_FULL", define_full},
          {"XE_DR_PACK", define_pack},         {"XE_DR_SRC_GAMMA16", define_gamma16},
          {"XE_DR_BPP8", define_bpp8},         {"XE_DR_SCALE_X", define_scale_x},
          {"XE_DR_SCALE_Y", define_scale_y},
          {nullptr, nullptr},
      };
      ID3DBlob* code = nullptr;
      ID3DBlob* errors = nullptr;
      HRESULT compile_result =
          compile(kNrDirectResolveShaderSource, sizeof(kNrDirectResolveShaderSource) - 1,
                  "nr_direct_resolve", defines, nullptr, "main", "cs_5_0", 0, 0, &code, &errors);
      if (SUCCEEDED(compile_result) && code) {
        pipeline = ui::d3d12::util::CreateComputePipeline(
            command_processor_.GetD3D12Provider().GetDevice(), code->GetBufferPointer(),
            code->GetBufferSize(), root_signature);
        if (pipeline) {
          pipeline->SetName(key.is_depth ? L"NR Direct Resolve Depth" : L"NR Direct Resolve Color");
        }
      } else {
        REXGPU_WARN("[nr-dres] shader compile failed for key {:08X}: {}", key.key,
                    errors ? static_cast<const char*>(errors->GetBufferPointer()) : "no output");
      }
      if (code) {
        code->Release();
      }
      if (errors) {
        errors->Release();
      }
    }
  }
  // Cache even a null result so a failing variant is not recompiled per
  // resolve; the preflight then declines with kPipeline and the dump serves.
  direct_resolve_pipelines_.emplace(map_key, pipeline);
  return pipeline;
}

bool D3D12RenderTargetCache::TryResolveCopyDirectly(const draw_util::ResolveInfo& resolve_info,
                                                    draw_util::ResolveCopyShaderIndex copy_shader,
                                                    bool draw_resolution_scaled) {
  ++direct_resolve_attempt_count_;
  direct_resolve_plan_ = DirectResolvePlan();
  auto decline = [this](DirectResolveDecline reason) {
    ++direct_resolve_declines_[size_t(reason)];
    return false;
  };
  if (!direct_resolve_root_signature_color_ || !direct_resolve_root_signature_depth_) {
    return decline(DirectResolveDecline::kPipeline);
  }
  // 32bpp classes only: fast (raw single-sample copy) and full (averaging /
  // format conversion / exp bias). 8/16/64/128bpp stay on the dump path.
  const bool class_fast = copy_shader == draw_util::ResolveCopyShaderIndex::kFast32bpp1x2xMSAA ||
                          copy_shader == draw_util::ResolveCopyShaderIndex::kFast32bpp4xMSAA;
  const bool class_full8 = copy_shader == draw_util::ResolveCopyShaderIndex::kFull8bpp;
  const bool class_full =
      copy_shader == draw_util::ResolveCopyShaderIndex::kFull32bpp || class_full8;
  if (!class_fast && !class_full) {
    if (size_t(copy_shader) < size_t(draw_util::ResolveCopyShaderIndex::kCount)) {
      ++direct_resolve_class_declines_[size_t(copy_shader)];
    }
    return decline(DirectResolveDecline::kShaderClass);
  }
  if (resolve_info.copy_dest_info.copy_dest_array) {
    return decline(DirectResolveDecline::kDestArray);
  }

  uint32_t dump_base, dump_row_length_used, dump_rows, dump_pitch;
  resolve_info.GetCopyEdramTileSpan(dump_base, dump_row_length_used, dump_rows, dump_pitch);
  GetResolveCopyRectanglesToDump(dump_base, dump_row_length_used, dump_rows, dump_pitch,
                                 dump_rectangles_);
  if (dump_rectangles_.empty()) {
    return decline(DirectResolveDecline::kNoRect);
  }
  if (dump_rectangles_.size() > 1) {
    return decline(DirectResolveDecline::kMultiRect);
  }
  const ResolveCopyDumpRectangle& rectangle = dump_rectangles_.front();
  auto* render_target = static_cast<D3D12RenderTarget*>(rectangle.render_target);
  if (!render_target) {
    return decline(DirectResolveDecline::kNoRect);
  }
  if (rectangle.row_first != 0 || rectangle.rows != dump_rows || rectangle.row_first_start != 0 ||
      rectangle.row_last_end != dump_row_length_used) {
    return decline(DirectResolveDecline::kRectPartial);
  }

  RenderTargetKey rt_key = render_target->key();
  // [xfer] The resolve reads this owner's tiles.
  XferUseNoteResolveRead(rt_key, dump_base, dump_base + dump_rows * dump_pitch);
  const draw_util::ResolveEdramInfo& edram_info =
      resolve_info.IsCopyingDepth() ? resolve_info.depth_edram_info : resolve_info.color_edram_info;
  // The owner's layout must be the resolve's layout, or the legacy path would
  // be doing a raw cross-layout reinterpretation this shader does not model.
  if (rt_key.GetPitchTiles() != uint32_t(edram_info.pitch_tiles) ||
      uint32_t(rt_key.msaa_samples) != uint32_t(edram_info.msaa_samples) ||
      edram_info.format_is_64bpp || rt_key.Is64bpp() ||
      uint32_t(edram_info.base_tiles) < rt_key.base_tiles || !edram_info.pitch_tiles) {
    return decline(DirectResolveDecline::kGeometry);
  }
  if (!rt_key.is_depth) {
    switch (rt_key.GetColorFormat()) {
      case xenos::ColorRenderTargetFormat::k_8_8_8_8:
      case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
      case xenos::ColorRenderTargetFormat::k_2_10_10_10:
      case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
        break;
      default:
        // Float and 16-bit sources would need the EDRAM quantization round
        // trip (7e3 etc.) reproduced to match the legacy chain bit-exactly.
        return decline(DirectResolveDecline::kFormat);
    }
  }
  if (class_full && rt_key.is_depth) {
    // GetCopyShader never routes depth to full; if it ever did, decline.
    return decline(DirectResolveDecline::kGeometry);
  }

  // Dest pack class for the full shader, mirroring XePack32bpp4Pixels
  // (unknown formats fall through to raw 32-float bits there too, so nothing
  // declines here).
  uint32_t pack_class = 0;
  if (class_full8) {
    switch (xenos::ColorFormat(uint32_t(resolve_info.copy_dest_info.copy_dest_format))) {
      case xenos::ColorFormat::k_8:
      case xenos::ColorFormat::k_8_A:
      case xenos::ColorFormat::k_8_B:
        pack_class = 7;
        break;
      default:
        return decline(DirectResolveDecline::kFormat);
    }
  } else if (class_full) {
    switch (xenos::ColorFormat(uint32_t(resolve_info.copy_dest_info.copy_dest_format))) {
      case xenos::ColorFormat::k_8_8_8_8:
      case xenos::ColorFormat::k_8_8_8_8_A:
      case xenos::ColorFormat::k_8_8_8_8_AS_16_16_16_16:
        pack_class = 0;
        break;
      case xenos::ColorFormat::k_2_10_10_10:
      case xenos::ColorFormat::k_2_10_10_10_AS_16_16_16_16:
        pack_class = 1;
        break;
      case xenos::ColorFormat::k_10_11_11:
      case xenos::ColorFormat::k_10_11_11_AS_16_16_16_16:
        pack_class = 2;
        break;
      case xenos::ColorFormat::k_11_11_10:
      case xenos::ColorFormat::k_11_11_10_AS_16_16_16_16:
        pack_class = 3;
        break;
      case xenos::ColorFormat::k_16_16:
        pack_class = 4;
        break;
      case xenos::ColorFormat::k_16_16_FLOAT:
        pack_class = 5;
        break;
      default:
        pack_class = 6;
        break;
    }
  }
  bool src_gamma16 = class_full && !rt_key.is_depth &&
                     rt_key.GetColorFormat() == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA &&
                     gamma_render_target_as_unorm16_;

  DumpPipelineKey pipeline_key;
  pipeline_key.msaa_samples = rt_key.msaa_samples;
  pipeline_key.resource_format = rt_key.resource_format;
  pipeline_key.is_depth = rt_key.is_depth;
  ID3D12PipelineState* pipeline =
      GetOrCreateDirectResolvePipeline(pipeline_key, class_full, pack_class, src_gamma16);
  if (!pipeline) {
    return decline(DirectResolveDecline::kPipeline);
  }

  // Source geometry: where the resolve origin lands in the owning render
  // target, in host pixels. An EDRAM tile is 80x16 samples = 80x16 pixels at
  // 1x, 80x8 at 2x (vertical samples), 40x8 at 4x.
  xenos::MsaaSamples msaa = xenos::MsaaSamples(uint32_t(edram_info.msaa_samples));
  uint32_t tile_width_pixels =
      xenos::kEdramTileWidthSamples >> uint32_t(msaa >= xenos::MsaaSamples::k4X);
  uint32_t tile_height_pixels =
      xenos::kEdramTileHeightSamples >> uint32_t(msaa >= xenos::MsaaSamples::k2X);
  uint32_t base_delta_tiles = uint32_t(edram_info.base_tiles) - rt_key.base_tiles;
  uint32_t origin_x = (base_delta_tiles % edram_info.pitch_tiles) * tile_width_pixels +
                      (uint32_t(resolve_info.coordinate_info.edram_offset_x_div_8) << 3);
  uint32_t origin_y = (base_delta_tiles / edram_info.pitch_tiles) * tile_height_pixels +
                      (uint32_t(resolve_info.coordinate_info.edram_offset_y_div_8) << 3);
  uint32_t width_pixels = uint32_t(resolve_info.coordinate_info.width_div_8) << 3;
  uint32_t height_pixels = resolve_info.height_div_8 << 3;
  // Resolution scaling: everything the shader sees is in HOST pixels; the
  // half-pixel-offset gap on the left/top is filled from the first covered
  // column/row like the vendored scaled shaders do.
  uint32_t fill_x = 0, fill_y = 0;
  if (draw_resolution_scaled) {
    uint32_t scale_x = draw_resolution_scale_x(), scale_y = draw_resolution_scale_y();
    origin_x *= scale_x;
    origin_y *= scale_y;
    width_pixels *= scale_x;
    height_pixels *= scale_y;
    if (edram_info.fill_half_pixel_offset) {
      fill_x = scale_x >> 1;
      fill_y = scale_y >> 1;
    }
  }
  if (!width_pixels || !height_pixels || width_pixels > 0xFFFF || height_pixels > 0xFFFF ||
      origin_x > 0xFFFF || origin_y > 0xFFFF) {
    return decline(DirectResolveDecline::kGeometry);
  }

  // The host samples to read. Single-sample (fast, or full without
  // averaging): XeResolveFirstSampleIndex of the sanitized select, mapped to
  // the host index. Averaged full: the sample list in the LEGACY shader's
  // summation order (base, +vertical, +horizontal, +both), host-mapped.
  uint32_t sample_select = uint32_t(resolve_info.copy_dest_coordinate_info.copy_sample_select);
  uint32_t samples[4] = {0, 0, 0, 0};
  uint32_t sample_count = 1;
  bool averaged = class_full && sample_select >= 4;
  if (!averaged) {
    uint32_t first_sample = sample_select <= 3 ? sample_select : (sample_select == 5 ? 2 : 0);
    if (msaa == xenos::MsaaSamples::k2X) {
      samples[0] =
          draw_util::GetD3D10SampleIndexForGuest2xMSAA(first_sample & 1, msaa_2x_supported_);
    } else if (msaa == xenos::MsaaSamples::k4X) {
      samples[0] = first_sample & 3;
    }
    if (msaa != xenos::MsaaSamples::k1X && REXCVAR_GET(gpu_nr_dres_force_sample) >= 0) {
      samples[0] = uint32_t(REXCVAR_GET(gpu_nr_dres_force_sample)) & 3;
    }
  } else if (msaa == xenos::MsaaSamples::k1X) {
    // Sanitization should have reduced the select; a single sample matches
    // whatever the legacy shader would do with no extra rows to read.
    sample_count = 1;
  } else if (msaa == xenos::MsaaSamples::k2X) {
    if (sample_select == 5) {
      // select 23 on a 2-sample surface would make the legacy shader read the
      // horizontal neighbour pixel; not reproduced.
      return decline(DirectResolveDecline::kGeometry);
    }
    sample_count = 2;
    samples[0] = draw_util::GetD3D10SampleIndexForGuest2xMSAA(0, msaa_2x_supported_);
    samples[1] = draw_util::GetD3D10SampleIndexForGuest2xMSAA(1, msaa_2x_supported_);
    if (sample_select >= 6) {
      // 0123 on 2x: the legacy shader would also add the horizontal
      // neighbour; not reproduced.
      return decline(DirectResolveDecline::kGeometry);
    }
  } else {
    // 4x. EDRAM sample position (h, v) maps to host sample h | (v << 1)
    // (the dump shader's composition); legacy sums base, +vertical,
    // +horizontal, +both.
    if (sample_select == 4) {
      sample_count = 2;
      samples[0] = 0;
      samples[1] = 2;
    } else if (sample_select == 5) {
      sample_count = 2;
      samples[0] = 1;
      samples[1] = 3;
    } else {
      sample_count = 4;
      samples[0] = 0;
      samples[1] = 2;
      samples[2] = 1;
      samples[3] = 3;
    }
  }

  // The folded factor: 2^exp_bias times 1/count, built the way the legacy
  // shader builds it (successive *0.5f) so the float bits match.
  float bias_factor = 1.0f;
  if (class_full) {
    bias_factor = std::ldexp(1.0f, int32_t(resolve_info.copy_dest_info.copy_dest_exp_bias));
    if (sample_count >= 2) {
      bias_factor *= 0.5f;
    }
    if (sample_count >= 4) {
      bias_factor *= 0.5f;
    }
  }
  uint32_t bias_bits;
  std::memcpy(&bias_bits, &bias_factor, sizeof(bias_bits));

  direct_resolve_plan_.pipeline = pipeline;
  direct_resolve_plan_.render_target = render_target;
  direct_resolve_plan_.is_depth = rt_key.is_depth;
  direct_resolve_plan_.scaled = draw_resolution_scaled;
  direct_resolve_plan_.dest_base_guest = resolve_info.copy_dest_base;
  direct_resolve_plan_.bpp_log2 = class_full8 ? 0 : 2;
  direct_resolve_plan_.constants[0] = resolve_info.copy_dest_info.value;
  direct_resolve_plan_.constants[1] = resolve_info.copy_dest_coordinate_info.packed;
  // Scaled dests are windowed at the range made current, so addresses are
  // texture-relative with no base.
  direct_resolve_plan_.constants[2] = draw_resolution_scaled ? 0 : resolve_info.copy_dest_base;
  direct_resolve_plan_.constants[3] = width_pixels | (height_pixels << 16);
  direct_resolve_plan_.constants[4] = origin_x | (origin_y << 16);
  direct_resolve_plan_.constants[5] = samples[0] | (samples[1] << 4) | (samples[2] << 8) |
                                      (samples[3] << 12) | (sample_count << 16);
  direct_resolve_plan_.constants[6] = bias_bits;
  direct_resolve_plan_.constants[7] = fill_x | (fill_y << 8);
  // 8bpp: one thread covers 4 dest pixels along x (32 per 8-thread group).
  direct_resolve_plan_.group_count_x =
      class_full8 ? (width_pixels + 31) >> 5 : (width_pixels + 7) >> 3;
  direct_resolve_plan_.group_count_y = (height_pixels + 7) >> 3;
  return true;
}

bool D3D12RenderTargetCache::DispatchDirectResolve(D3D12SharedMemory& shared_memory,
                                                   D3D12TextureCache& texture_cache) {
  const DirectResolvePlan& plan = direct_resolve_plan_;
  if (!plan.pipeline || !plan.render_target) {
    return false;
  }
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();

  // Descriptors: dest UAV (bindless has a persistent one for shared memory;
  // scaled dests always use a one-use view of the current scaled resolve
  // range), source SRV, and the stencil plane SRV for depth - the latter two
  // are one-use copies of the render target's own descriptors, the same way
  // DumpRenderTargets binds them.
  uint32_t source_descriptor_count = plan.is_depth ? 2 : 1;
  bool dest_one_use = plan.scaled || !bindless_resources_used_;
  uint32_t one_use_count = source_descriptor_count + (dest_one_use ? 1 : 0);
  ui::d3d12::util::DescriptorCpuGpuHandlePair descriptors[3];
  if (!command_processor_.RequestOneUseSingleViewDescriptors(one_use_count, descriptors)) {
    return false;
  }
  ui::d3d12::util::DescriptorCpuGpuHandlePair descriptor_dest;
  uint32_t source_descriptor_base = 0;
  if (plan.scaled) {
    descriptor_dest = descriptors[0];
    texture_cache.CreateCurrentScaledResolveRangeUintPow2UAV(descriptor_dest.first, 2);
    source_descriptor_base = 1;
  } else if (bindless_resources_used_) {
    descriptor_dest = command_processor_.GetSharedMemoryUintPow2BindlessUAVHandlePair(2);
  } else {
    descriptor_dest = descriptors[0];
    shared_memory.WriteUintPow2UAVDescriptor(descriptor_dest.first, 2);
    source_descriptor_base = 1;
  }
  auto& d3d12_rt = *plan.render_target;
  device->CopyDescriptorsSimple(1, descriptors[source_descriptor_base].first,
                                d3d12_rt.descriptor_srv().GetHandle(),
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  if (plan.is_depth) {
    device->CopyDescriptorsSimple(1, descriptors[source_descriptor_base + 1].first,
                                  d3d12_rt.descriptor_srv_stencil().GetHandle(),
                                  D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  }

  command_processor_.PushTransitionBarrier(
      d3d12_rt.resource(),
      d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  if (plan.scaled) {
    texture_cache.TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  } else {
    shared_memory.UseForWriting();
  }

  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
  ID3D12RootSignature* root_signature = plan.is_depth ? direct_resolve_root_signature_depth_
                                                      : direct_resolve_root_signature_color_;
  command_list.D3DSetComputeRootSignature(root_signature);
  command_list.D3DSetComputeRoot32BitConstants(0, 8, plan.constants, 0);
  command_list.D3DSetComputeRootDescriptorTable(1, descriptor_dest.second);
  command_list.D3DSetComputeRootDescriptorTable(2, descriptors[source_descriptor_base].second);
  if (plan.is_depth) {
    command_list.D3DSetComputeRootDescriptorTable(3,
                                                  descriptors[source_descriptor_base + 1].second);
  }
  command_processor_.SetExternalPipeline(plan.pipeline);
  command_processor_.SubmitBarriers();
  command_list.D3DDispatch(plan.group_count_x, plan.group_count_y, 1);
  ++direct_resolve_dispatch_count_;
  return true;
}

void D3D12RenderTargetCache::DirectResolveVerifySnapshot(D3D12SharedMemory& shared_memory,
                                                         D3D12TextureCache& texture_cache,
                                                         uint32_t stage, uint32_t dest_start,
                                                         uint32_t dest_length) {
  constexpr uint32_t kVerifyCapBytes = 16 * 1024 * 1024;
  bool scaled = direct_resolve_plan_.scaled;
  uint64_t copy_offset;
  uint64_t copy_length = dest_length;
  ID3D12Resource* source_resource;
  if (scaled) {
    uint32_t scale_sq = draw_resolution_scale_x() * draw_resolution_scale_y();
    uint64_t start_scaled = uint64_t(dest_start) * scale_sq;
    copy_length = uint64_t(dest_length) * scale_sq;
    constexpr uint64_t kChunkMask = (UINT64_C(1) << 31) - 1;
    if ((start_scaled >> 31) != ((start_scaled + copy_length - 1) >> 31) ||
        size_t(start_scaled >> 31) != texture_cache.GetCurrentScaledResolveBufferIndexPublic()) {
      return;
    }
    copy_offset = start_scaled & kChunkMask;
    source_resource = texture_cache.GetCurrentScaledResolveBufferResource();
  } else {
    copy_offset = dest_start;
    source_resource = shared_memory.GetBuffer();
  }
  if (copy_length > kVerifyCapBytes || !copy_length) {
    return;
  }
  if (!dres_verify_readback_[stage]) {
    D3D12_RESOURCE_DESC desc;
    ui::d3d12::util::FillBufferResourceDesc(desc, kVerifyCapBytes, D3D12_RESOURCE_FLAG_NONE);
    const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
    if (FAILED(provider.GetDevice()->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&dres_verify_readback_[stage])))) {
      dres_verify_readback_[stage] = nullptr;
      return;
    }
  }
  if (scaled) {
    texture_cache.TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATE_COPY_SOURCE);
  } else {
    shared_memory.UseAsCopySource();
  }
  command_processor_.SubmitBarriers();
  command_processor_.GetDeferredCommandList().D3DCopyBufferRegion(
      dres_verify_readback_[stage], 0, source_resource, copy_offset, copy_length);
  if (stage == 1) {
    dres_verify_pending_length_ = uint32_t(copy_length);
    dres_verify_pending_dest_ = dest_start;
    dres_verify_pending_time_ = std::chrono::steady_clock::now();
    std::memcpy(dres_verify_pending_constants_, direct_resolve_plan_.constants,
                sizeof(dres_verify_pending_constants_));
    dres_verify_pending_rt_key_ =
        direct_resolve_plan_.render_target ? direct_resolve_plan_.render_target->key().key : 0;
    dres_verify_pending_dest_base_ = direct_resolve_plan_.dest_base_guest;
    dres_verify_pending_bpp_log2_ = direct_resolve_plan_.bpp_log2;
    dres_verify_pending_scaled_ = scaled;
  }
}

void D3D12RenderTargetCache::DirectResolveVerifySnapshotSource() {
  dres_verify_pending_src_rows_ = 0;
  D3D12RenderTarget* rt = direct_resolve_plan_.render_target;
  if (!rt) {
    return;
  }
  RenderTargetKey rt_key = rt->key();
  if (rt_key.msaa_samples != xenos::MsaaSamples::k1X || rt_key.is_depth) {
    return;
  }
  DXGI_FORMAT format = GetColorResourceDXGIFormat(rt_key.GetColorFormat());
  if (format != DXGI_FORMAT_R8G8B8A8_UNORM && format != DXGI_FORMAT_R10G10B10A2_UNORM) {
    return;
  }
  uint32_t rect_h = direct_resolve_plan_.constants[3] >> 16;
  uint32_t origin_y = direct_resolve_plan_.constants[4] >> 16;
  uint32_t rows = origin_y + rect_h;
  uint32_t width = rt_key.GetWidth() *
                   (direct_resolve_plan_.scaled ? draw_resolution_scale_x() : 1);
  uint32_t row_pitch = (width * 4 + 255) & ~uint32_t(255);
  constexpr uint32_t kVerifyCapBytes = 16 * 1024 * 1024;
  if (uint64_t(row_pitch) * rows > kVerifyCapBytes) {
    return;
  }
  if (!dres_verify_readback_[2]) {
    D3D12_RESOURCE_DESC desc;
    ui::d3d12::util::FillBufferResourceDesc(desc, kVerifyCapBytes, D3D12_RESOURCE_FLAG_NONE);
    const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
    if (FAILED(provider.GetDevice()->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&dres_verify_readback_[2])))) {
      dres_verify_readback_[2] = nullptr;
      return;
    }
  }
  command_processor_.PushTransitionBarrier(
      rt->resource(), rt->SetResourceState(D3D12_RESOURCE_STATE_COPY_SOURCE),
      D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_processor_.SubmitBarriers();
  D3D12_TEXTURE_COPY_LOCATION src_location;
  src_location.pResource = rt->resource();
  src_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src_location.SubresourceIndex = 0;
  D3D12_TEXTURE_COPY_LOCATION dst_location;
  dst_location.pResource = dres_verify_readback_[2];
  dst_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst_location.PlacedFootprint.Offset = 0;
  dst_location.PlacedFootprint.Footprint.Format = format;
  dst_location.PlacedFootprint.Footprint.Width = width;
  dst_location.PlacedFootprint.Footprint.Height = rows;
  dst_location.PlacedFootprint.Footprint.Depth = 1;
  dst_location.PlacedFootprint.Footprint.RowPitch = row_pitch;
  D3D12_BOX src_box;
  src_box.left = 0;
  src_box.top = 0;
  src_box.front = 0;
  src_box.right = width;
  src_box.bottom = rows;
  src_box.back = 1;
  command_processor_.GetDeferredCommandList().D3DCopyTextureRegion(&dst_location, 0, 0, 0,
                                                                   &src_location, &src_box);
  dres_verify_pending_src_rows_ = rows;
  dres_verify_pending_src_pitch_ = row_pitch;
}

// CPU mirror of the shader's XeDrTiledAddress2DBpp / XeDrDestAddress, for
// naming a diverging pixel (scale-aware).
static uint32_t NrDresTiledAddress2DBpp(uint32_t x, uint32_t y, uint32_t pitch_mt,
                                        uint32_t bpp_log2) {
  uint32_t outer = ((y >> 5) * pitch_mt + (x >> 5)) << 6;
  uint32_t inner = (((y >> 1) & 7) << 3) | (x & 7);
  uint32_t oib = (outer | inner) << bpp_log2;
  uint32_t bank = (y >> 4) & 1;
  uint32_t pipe = ((x >> 3) & 3) ^ (((y >> 3) & 1) << 1);
  return ((y & 1) << 4) | (pipe << 6) | (bank << 11) | (oib & 0xF) | (((oib >> 4) & 1) << 5) |
         (((oib >> 5) & 7) << 8) | ((oib >> 8) << 12);
}

static uint32_t NrDresTiledAddress2D(uint32_t x, uint32_t y, uint32_t pitch_mt) {
  return NrDresTiledAddress2DBpp(x, y, pitch_mt, 2);
}

static uint32_t NrDresDestAddress(uint32_t x, uint32_t y, uint32_t pitch_mt, uint32_t bpp_log2,
                                  uint32_t scale_x, uint32_t scale_y) {
  if (scale_x == 1 && scale_y == 1) {
    return NrDresTiledAddress2DBpp(x, y, pitch_mt, bpp_log2);
  }
  // The vendored blobs' run-granule scaled layout (see XeDrDestAddress):
  // 8-byte runs at 1bpb, 16-byte runs at 4bpb.
  uint32_t run_bytes_log2 = bpp_log2 == 0 ? 3 : 4;
  uint32_t blocks_per_run = (1u << run_bytes_log2) >> bpp_log2;
  uint32_t host_unit_x = x / blocks_per_run;
  uint32_t guest_unit_x = host_unit_x / scale_x;
  uint32_t guest_y = y / scale_y;
  uint32_t sub_x = host_unit_x - guest_unit_x * scale_x;
  uint32_t sub_y = y - guest_y * scale_y;
  uint32_t byte_in_run = (x - host_unit_x * blocks_per_run) << bpp_log2;
  uint32_t guest_addr =
      NrDresTiledAddress2DBpp(guest_unit_x * blocks_per_run, guest_y, pitch_mt, bpp_log2);
  return guest_addr * (scale_x * scale_y) + ((sub_x * scale_y + sub_y) << run_bytes_log2) +
         byte_in_run;
}

void D3D12RenderTargetCache::ReportDirectResolveStats() {
  // Compare the previous verify pair first: a second has passed, the copies
  // have retired ([[upload-heap-readback-trap]] does not apply to readback
  // heaps read after retirement, which is exactly this pattern).
  if (dres_verify_pending_length_ && dres_verify_readback_[0] && dres_verify_readback_[1] &&
      std::chrono::steady_clock::now() - dres_verify_pending_time_ >= std::chrono::seconds(1)) {
    void* mapping_legacy = nullptr;
    void* mapping_direct = nullptr;
    if (SUCCEEDED(dres_verify_readback_[0]->Map(0, nullptr, &mapping_legacy)) ) {
      if (SUCCEEDED(dres_verify_readback_[1]->Map(0, nullptr, &mapping_direct))) {
        const uint32_t* legacy = static_cast<const uint32_t*>(mapping_legacy);
        const uint32_t* direct = static_cast<const uint32_t*>(mapping_direct);
        uint32_t dword_count = dres_verify_pending_length_ >> 2;
        uint32_t diverged = 0;
        uint32_t first_offset = UINT32_MAX;
        uint32_t first_legacy = 0, first_direct = 0;
        for (uint32_t i = 0; i < dword_count; ++i) {
          if (legacy[i] != direct[i]) {
            if (!diverged) {
              first_offset = i << 2;
              first_legacy = legacy[i];
              first_direct = direct[i];
            }
            ++diverged;
          }
        }
        ++dres_verify_compared_;
        if (!diverged) {
          REXGPU_INFO("[nr-dres] verify clean dest={:08X} len={:X} rt_key={:08X}",
                      dres_verify_pending_dest_, dres_verify_pending_length_,
                      dres_verify_pending_rt_key_);
        }
        if (diverged) {
          ++dres_verify_diverged_;
          dres_verify_diverged_dwords_ += diverged;
          // Name the diverging pixel: forward-scan the resolve rect for the
          // dest dword the shader maps to the first differing offset. Runs
          // only on a diverge, which the snapshot already rate-limits to 1/s.
          const uint32_t* pc = dres_verify_pending_constants_;
          uint32_t rect_w = pc[3] & 0xFFFF, rect_h = pc[3] >> 16;
          uint32_t pitch_mt = pc[1] & 0x3FF;
          uint32_t scan_scale_x = dres_verify_pending_scaled_ ? draw_resolution_scale_x() : 1;
          uint32_t scan_scale_y = dres_verify_pending_scaled_ ? draw_resolution_scale_y() : 1;
          uint32_t bpp_log2 = dres_verify_pending_bpp_log2_;
          uint32_t off_x = (((pc[1] >> 20) & 0xF) << 3) * scan_scale_x;
          uint32_t off_y = (((pc[1] >> 24) & 0xF) << 3) * scan_scale_y;
          // The shader address of the first differing dword: snapshot offsets
          // are relative to the extent start (scaled), shader addresses to
          // the dest base (0-based window when scaled).
          uint32_t rel = first_offset + (dres_verify_pending_dest_ - dres_verify_pending_dest_base_) *
                                            scan_scale_x * scan_scale_y;
          auto scan_addr = [&](uint32_t x, uint32_t y) {
            return NrDresDestAddress(x + off_x, y + off_y, pitch_mt, bpp_log2, scan_scale_x,
                                     scan_scale_y);
          };
          int32_t px = -1, py = -1;
          for (uint32_t y = 0; y < rect_h && px < 0; ++y) {
            for (uint32_t x = 0; x < rect_w; ++x) {
              if ((scan_addr(x, y) & ~3u) == (rel & ~3u)) {
                px = int32_t(x);
                py = int32_t(y);
                break;
              }
            }
          }
          // Shift hypothesis: does legacy's value equal direct's value at a
          // neighbouring pixel? Names an off-by-one source mapping in one run.
          std::string shift_report;
          if (px >= 0 && bpp_log2 == 2) {
            for (int32_t dy = -2; dy <= 2; ++dy) {
              for (int32_t dx = -2; dx <= 2; ++dx) {
                int64_t nx = px + dx, ny = py + dy;
                if (nx < 0 || ny < 0 || nx >= rect_w || ny >= rect_h) {
                  continue;
                }
                uint32_t addr2 = scan_addr(uint32_t(nx), uint32_t(ny));
                int64_t snap2 = int64_t(addr2) - int64_t(rel) + int64_t(first_offset);
                if (snap2 < 0 || (snap2 >> 2) >= dword_count) {
                  continue;
                }
                if (direct[snap2 >> 2] == first_legacy && (dx | dy)) {
                  shift_report += fmt::format(" legacy==direct@({},{})", dx, dy);
                }
              }
            }
          }
          // The arbiter: what the source RT itself held at that pixel, plus a
          // 5x5 neighborhood so a shifted mapping names its own shift.
          uint32_t rt_texel = 0;
          bool rt_texel_valid = false;
          std::string rt_grid;
          if (px >= 0 && dres_verify_pending_src_rows_ && dres_verify_readback_[2]) {
            uint32_t src_x = (pc[4] & 0xFFFF) + uint32_t(px);
            uint32_t src_y = (pc[4] >> 16) + uint32_t(py);
            if (src_y < dres_verify_pending_src_rows_) {
              void* mapping_src = nullptr;
              if (SUCCEEDED(dres_verify_readback_[2]->Map(0, nullptr, &mapping_src))) {
                const uint32_t* texels = static_cast<const uint32_t*>(mapping_src);
                uint32_t row_pitch_dwords = dres_verify_pending_src_pitch_ >> 2;
                rt_texel = texels[size_t(src_y) * row_pitch_dwords + src_x];
                rt_texel_valid = true;
                for (int32_t dy = -2; dy <= 2; ++dy) {
                  rt_grid += " |";
                  for (int32_t dx = -2; dx <= 2; ++dx) {
                    int64_t gx = int64_t(src_x) + dx, gy = int64_t(src_y) + dy;
                    if (gx < 0 || gy < 0 || gy >= dres_verify_pending_src_rows_ ||
                        gx >= row_pitch_dwords) {
                      rt_grid += " --------";
                    } else {
                      rt_grid +=
                          fmt::format(" {:08X}", texels[size_t(gy) * row_pitch_dwords + gx]);
                    }
                  }
                }
                D3D12_RANGE write_range = {};
                dres_verify_readback_[2]->Unmap(0, &write_range);
              }
            }
          }
          REXGPU_WARN(
              "[nr-dres] VERIFY DIVERGED dest={:08X} len={:X}: {} of {} dwords, first at "
              "+{:X} legacy={:08X} direct={:08X} | pixel=({},{}) of {}x{} rt_key={:08X} "
              "origin=({},{}) hostsmp={} 2xnative={} | dest_info={:08X} dest_coord={:08X} "
              "rt_texel={} {:08X}{} grid(y-2..y+2 x-2..x+2):{}",
              dres_verify_pending_dest_, dres_verify_pending_length_, diverged, dword_count,
              first_offset, first_legacy, first_direct, px, py, rect_w, rect_h,
              dres_verify_pending_rt_key_, pc[4] & 0xFFFF, pc[4] >> 16, pc[5],
              msaa_2x_supported_ ? 1 : 0, pc[0], pc[1], rt_texel_valid ? "raw" : "n/a", rt_texel,
              shift_report, rt_grid);
        }
        D3D12_RANGE write_range = {};
        dres_verify_readback_[1]->Unmap(0, &write_range);
      }
      D3D12_RANGE write_range = {};
      dres_verify_readback_[0]->Unmap(0, &write_range);
    }
    dres_verify_pending_length_ = 0;
    dres_verify_pending_src_rows_ = 0;
  }

  const auto now = std::chrono::steady_clock::now();
  if (now - dres_report_last_ < std::chrono::seconds(1)) {
    return;
  }
  dres_report_last_ = now;
  REXGPU_INFO(
      "[nr-dres] attempts={} direct={} fb={} disp={} | declines: scaled={} class={} array={} "
      "norect={} multi={} partial={} geom={} fmt={} pso={} | verify cmp={} div={} divdw={}",
      direct_resolve_attempt_count_, direct_resolve_success_count_, direct_resolve_fallback_count_,
      direct_resolve_dispatch_count_,
      direct_resolve_declines_[size_t(DirectResolveDecline::kScaled)],
      direct_resolve_declines_[size_t(DirectResolveDecline::kShaderClass)],
      direct_resolve_declines_[size_t(DirectResolveDecline::kDestArray)],
      direct_resolve_declines_[size_t(DirectResolveDecline::kNoRect)],
      direct_resolve_declines_[size_t(DirectResolveDecline::kMultiRect)],
      direct_resolve_declines_[size_t(DirectResolveDecline::kRectPartial)],
      direct_resolve_declines_[size_t(DirectResolveDecline::kGeometry)],
      direct_resolve_declines_[size_t(DirectResolveDecline::kFormat)],
      direct_resolve_declines_[size_t(DirectResolveDecline::kPipeline)],
      dres_verify_compared_, dres_verify_diverged_, dres_verify_diverged_dwords_);
  REXGPU_INFO(
      "[nr-dres] class declines by shader: f32_12x={} f32_4x={} f64_12x={} f64_4x={} "
      "full8={} full16={} full32={} full64={} full128={}",
      direct_resolve_class_declines_[0], direct_resolve_class_declines_[1],
      direct_resolve_class_declines_[2], direct_resolve_class_declines_[3],
      direct_resolve_class_declines_[4], direct_resolve_class_declines_[5],
      direct_resolve_class_declines_[6], direct_resolve_class_declines_[7],
      direct_resolve_class_declines_[8]);
}

bool D3D12RenderTargetCache::DumpRenderTargets(uint32_t dump_base, uint32_t dump_row_length_used,
                                               uint32_t dump_rows, uint32_t dump_pitch) {
  assert_true(GetPath() == Path::kHostRenderTargets);

  GetResolveCopyRectanglesToDump(dump_base, dump_row_length_used, dump_rows, dump_pitch,
                                 dump_rectangles_);
  // [xfer] The dump path (the direct resolve declined) reads every owner.
  for (const ResolveCopyDumpRectangle& r : dump_rectangles_) {
    if (r.render_target) {
      XferUseNoteResolveRead(r.render_target->key(), dump_base + r.row_first * dump_pitch,
                             dump_base + (r.row_first + r.rows) * dump_pitch);
    }
  }
  // [NR-DETILE] N-6 probe: which tiles this dump actually covers, and who owns
  // them. A dump that stops short of the requested span leaves those EDRAM
  // tiles holding whatever was there before, which is exactly the shape of the
  // mod-512 duplicate.
  if (REXCVAR_GET(gpu_nr_dump_probe)) {
    // Rate-limiting blindly samples ONE resolve a second out of ~64 a frame,
    // and the first run of this probe never once caught the colour tap - the
    // only request the question is about. Dedupe by request signature instead,
    // so every DISTINCT dump shape is reported once a second and none can hide.
    const uint64_t dp_sig = (uint64_t(dump_base) << 40) ^ (uint64_t(dump_pitch) << 24) ^
                            (uint64_t(dump_rows) << 8) ^ uint64_t(dump_row_length_used);
    static std::map<uint64_t, std::chrono::steady_clock::time_point> dp_seen;
    const auto dp_now = std::chrono::steady_clock::now();
    auto dp_it = dp_seen.find(dp_sig);
    if (dp_it == dp_seen.end() || dp_now - dp_it->second >= std::chrono::seconds(1)) {
      if (dp_seen.size() > 64) {
        dp_seen.clear();
      }
      dp_seen[dp_sig] = dp_now;
      std::string dp;
      for (const ResolveCopyDumpRectangle& r : dump_rectangles_) {
        RenderTargetKey k = r.render_target ? r.render_target->key() : RenderTargetKey();
        // The HOST texture height is the open question: the dump reads RT rows
        // 0..720, and if the texture is shorter the read can fold back. Note
        // that the edram: line's h= is height_used (ownership), NOT this.
        const uint32_t rt_h = GetRenderTargetHeight(k.pitch_tiles_at_32bpp, k.msaa_samples);
        dp += fmt::format(
            " [rows {}..{} start={} end={} <- RT base={}t pitch={}t msaa={} depth={} "
            "hostH={} scaleY={}]",
            r.row_first, r.row_first + r.rows, r.row_first_start, r.row_last_end,
            uint32_t(k.base_tiles), uint32_t(k.pitch_tiles_at_32bpp),
            uint32_t(k.msaa_samples), uint32_t(k.is_depth), rt_h, draw_resolution_scale_y());
      }
      REXGPU_INFO(
          "[nr-detile] dump: base={}t row_len={}t rows={} pitch={}t -> {} rect(s):{}",
          dump_base, dump_row_length_used, dump_rows, dump_pitch,
          dump_rectangles_.size(), dp);
    }
  }
  if (dump_rectangles_.empty()) {
    return true;
  }

  // Clear previously set temporary indices.
  for (const ResolveCopyDumpRectangle& rectangle : dump_rectangles_) {
    auto& d3d12_rt = *static_cast<D3D12RenderTarget*>(rectangle.render_target);
    d3d12_rt.SetTemporarySortIndex(UINT32_MAX);
    d3d12_rt.SetTemporarySRVDescriptorIndex(UINT32_MAX);
    d3d12_rt.SetTemporarySRVDescriptorIndexStencil(UINT32_MAX);
  }
  // Gather all needed barriers and info needed to create descriptors and to
  // sort the invocations.
  TransitionEdramBuffer(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  dump_invocations_.clear();
  dump_invocations_.reserve(dump_rectangles_.size());
  current_temporary_descriptors_cpu_.clear();
  bool any_sources_32bpp_64bpp[2] = {};
  uint32_t rt_sort_index = 0;
  for (const ResolveCopyDumpRectangle& rectangle : dump_rectangles_) {
    auto& d3d12_rt = *static_cast<D3D12RenderTarget*>(rectangle.render_target);
    command_processor_.PushTransitionBarrier(
        d3d12_rt.resource(),
        d3d12_rt.SetResourceState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (d3d12_rt.temporary_sort_index() == UINT32_MAX) {
      d3d12_rt.SetTemporarySortIndex(rt_sort_index++);
    }
    if (d3d12_rt.temporary_srv_descriptor_index() == UINT32_MAX) {
      d3d12_rt.SetTemporarySRVDescriptorIndex(uint32_t(current_temporary_descriptors_cpu_.size()));
      current_temporary_descriptors_cpu_.push_back(d3d12_rt.descriptor_srv().GetHandle());
    }
    RenderTargetKey rt_key = d3d12_rt.key();
    if (rt_key.is_depth && d3d12_rt.temporary_srv_descriptor_index_stencil() == UINT32_MAX) {
      d3d12_rt.SetTemporarySRVDescriptorIndexStencil(
          uint32_t(current_temporary_descriptors_cpu_.size()));
      current_temporary_descriptors_cpu_.push_back(d3d12_rt.descriptor_srv_stencil().GetHandle());
    }
    any_sources_32bpp_64bpp[size_t(rt_key.Is64bpp())] = true;
    DumpPipelineKey pipeline_key;
    pipeline_key.msaa_samples = rt_key.msaa_samples;
    pipeline_key.resource_format = rt_key.resource_format;
    pipeline_key.is_depth = rt_key.is_depth;
    dump_invocations_.emplace_back(rectangle, pipeline_key);
  }
  // 32bpp and 64bpp.
  size_t edram_uav_indices[2] = {SIZE_MAX, SIZE_MAX};
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  if (!bindless_resources_used_) {
    if (any_sources_32bpp_64bpp[0]) {
      edram_uav_indices[0] = current_temporary_descriptors_cpu_.size();
      current_temporary_descriptors_cpu_.push_back(provider.OffsetViewDescriptor(
          edram_buffer_descriptor_heap_start_, uint32_t(EdramBufferDescriptorIndex::kR32UintUAV)));
    }
    if (any_sources_32bpp_64bpp[1]) {
      edram_uav_indices[1] = current_temporary_descriptors_cpu_.size();
      current_temporary_descriptors_cpu_.push_back(
          provider.OffsetViewDescriptor(edram_buffer_descriptor_heap_start_,
                                        uint32_t(EdramBufferDescriptorIndex::kR32G32UintUAV)));
    }
  }

  // Copy source descriptors to a shader-visible heap.
  ID3D12Device* device = provider.GetDevice();
  uint32_t descriptor_count = uint32_t(current_temporary_descriptors_cpu_.size());
  current_temporary_descriptors_gpu_.resize(descriptor_count);
  if (!command_processor_.RequestOneUseSingleViewDescriptors(
          descriptor_count, current_temporary_descriptors_gpu_.data())) {
    return false;
  }
  for (uint32_t i = 0; i < descriptor_count; ++i) {
    device->CopyDescriptorsSimple(1, current_temporary_descriptors_gpu_[i].first,
                                  current_temporary_descriptors_cpu_[i],
                                  D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  }

  // Sort the invocations to reduce context and binding switches.
  std::sort(dump_invocations_.begin(), dump_invocations_.end());

  // Dump the render targets.
  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
  ID3D12RootSignature* last_root_signature = nullptr;
  uint32_t root_parameters_set = 0;
  uint32_t last_descriptor_index_source = UINT32_MAX;
  uint32_t last_descriptor_index_stencil = UINT32_MAX;
  bool last_edram_uav_is_64bpp = false;
  DumpOffsets last_offsets;
  DumpPitches last_pitches;
  bool all_pipelines_available = true;
  for (const DumpInvocation& invocation : dump_invocations_) {
    const ResolveCopyDumpRectangle& rectangle = invocation.rectangle;
    auto& d3d12_rt = *static_cast<D3D12RenderTarget*>(rectangle.render_target);
    RenderTargetKey rt_key = d3d12_rt.key();
    DumpPipelineKey pipeline_key = invocation.pipeline_key;
    ID3D12PipelineState* pipeline = GetOrCreateDumpPipeline(pipeline_key);
    if (!pipeline) {
      all_pipelines_available = false;
      continue;
    }
    command_processor_.SetExternalPipeline(pipeline);

    ID3D12RootSignature* root_signature =
        pipeline_key.is_depth ? dump_root_signature_depth_ : dump_root_signature_color_;
    if (last_root_signature != root_signature) {
      last_root_signature = root_signature;
      command_list.D3DSetComputeRootSignature(root_signature);
      root_parameters_set = 0;
    }

    DumpRootParameter root_parameter_edram =
        pipeline_key.is_depth ? kDumpRootParameterDepthEdram : kDumpRootParameterColorEdram;
    uint32_t root_parameter_edram_bit = uint32_t(1) << root_parameter_edram;
    bool format_is_64bpp = rt_key.Is64bpp();
    if (last_edram_uav_is_64bpp != format_is_64bpp) {
      last_edram_uav_is_64bpp = format_is_64bpp;
      root_parameters_set &= ~root_parameter_edram_bit;
    }
    if (!(root_parameters_set & root_parameter_edram_bit)) {
      D3D12_GPU_DESCRIPTOR_HANDLE descriptor_handle_edram;
      if (bindless_resources_used_) {
        descriptor_handle_edram =
            command_processor_
                .GetEdramUintPow2BindlessUAVHandlePair(2 + uint32_t(last_edram_uav_is_64bpp))
                .second;
      } else {
        assert_true(edram_uav_indices[size_t(last_edram_uav_is_64bpp)] != SIZE_MAX);
        descriptor_handle_edram =
            current_temporary_descriptors_gpu_[edram_uav_indices[size_t(last_edram_uav_is_64bpp)]]
                .second;
      }
      command_list.D3DSetComputeRootDescriptorTable(root_parameter_edram, descriptor_handle_edram);
      root_parameters_set |= root_parameter_edram_bit;
    }

    DumpRootParameter root_parameter_pitches =
        pipeline_key.is_depth ? kDumpRootParameterDepthPitches : kDumpRootParameterColorPitches;
    uint32_t root_parameter_pitches_bit = uint32_t(1) << root_parameter_pitches;
    DumpPitches pitches;
    pitches.dest_pitch = dump_pitch;
    pitches.source_pitch = rt_key.GetPitchTiles();
    if (last_pitches != pitches) {
      last_pitches = pitches;
      root_parameters_set &= ~root_parameter_pitches_bit;
    }
    if (!(root_parameters_set & root_parameter_pitches_bit)) {
      command_list.D3DSetComputeRoot32BitConstants(
          root_parameter_pitches, sizeof(last_pitches) / sizeof(uint32_t), &last_pitches, 0);
      root_parameters_set |= root_parameter_pitches_bit;
    }

    if (pipeline_key.is_depth) {
      constexpr uint32_t kDumpRootParameterDepthStencilBit = uint32_t(1)
                                                             << kDumpRootParameterDepthStencil;
      uint32_t descriptor_index_stencil = d3d12_rt.temporary_srv_descriptor_index_stencil();
      assert_true(descriptor_index_stencil != UINT32_MAX);
      if (last_descriptor_index_stencil != descriptor_index_stencil) {
        last_descriptor_index_stencil = descriptor_index_stencil;
        root_parameters_set &= ~kDumpRootParameterDepthStencilBit;
      }
      if (!(root_parameters_set & kDumpRootParameterDepthStencilBit)) {
        command_list.D3DSetComputeRootDescriptorTable(
            kDumpRootParameterDepthStencil,
            current_temporary_descriptors_gpu_[last_descriptor_index_stencil].second);
        root_parameters_set |= kDumpRootParameterDepthStencilBit;
      }
    }

    constexpr uint32_t kDumpRootParameterSourceBit = uint32_t(1) << kDumpRootParameterSource;
    uint32_t descriptor_index_source = d3d12_rt.temporary_srv_descriptor_index();
    assert_true(descriptor_index_source != UINT32_MAX);
    if (last_descriptor_index_source != descriptor_index_source) {
      last_descriptor_index_source = descriptor_index_source;
      root_parameters_set &= ~kDumpRootParameterSourceBit;
    }
    if (!(root_parameters_set & kDumpRootParameterSourceBit)) {
      command_list.D3DSetComputeRootDescriptorTable(
          kDumpRootParameterSource,
          current_temporary_descriptors_gpu_[last_descriptor_index_source].second);
      root_parameters_set |= kDumpRootParameterSourceBit;
    }

    constexpr uint32_t kDumpRootParameterOffsetsBit = uint32_t(1) << kDumpRootParameterOffsets;
    DumpOffsets offsets;
    offsets.source_base_tiles = rt_key.base_tiles;
    ResolveCopyDumpRectangle::Dispatch dispatches[ResolveCopyDumpRectangle::kMaxDispatches];
    uint32_t dispatch_count = rectangle.GetDispatches(dump_pitch, dump_row_length_used, dispatches);
    for (uint32_t i = 0; i < dispatch_count; ++i) {
      const ResolveCopyDumpRectangle::Dispatch& dispatch = dispatches[i];
      offsets.dispatch_first_tile = dump_base + dispatch.offset;
      if (last_offsets != offsets) {
        last_offsets = offsets;
        root_parameters_set &= ~kDumpRootParameterOffsetsBit;
      }
      if (!(root_parameters_set & kDumpRootParameterOffsetsBit)) {
        command_list.D3DSetComputeRoot32BitConstants(
            kDumpRootParameterOffsets, sizeof(last_offsets) / sizeof(uint32_t), &last_offsets, 0);
        root_parameters_set |= kDumpRootParameterOffsetsBit;
      }
      command_processor_.SubmitBarriers();
      // Processing 40 x 16 x scale samples per dispatch (a 32bpp tile in two
      // dispatches at 1x1 scale, 64bpp in one dispatch).
      command_list.D3DDispatch((dispatch.width_tiles * draw_resolution_scale_x())
                                   << uint32_t(!format_is_64bpp),
                               dispatch.height_tiles * draw_resolution_scale_y(), 1);
    }
    MarkEdramBufferModified();
  }
  return all_pipelines_available;
}

}  // namespace rex::graphics::d3d12
