/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_SYS_CONSTANTS_H_
#define REX_GRAPHICS_NR_SYS_CONSTANTS_H_

#include <cstdint>

#include <rex/graphics/nr_pipeline_state.h>

// [NR-SYS] The system-constants mirror: native-renderer phase 5, increment
// 5-3b-1.
//
// 5-3b-0 proved the four guest constant buffers are ours byte for byte. The
// fifth buffer a draw binds is the system-constants cbuffer: not guest data
// but the runtime's own derivation (UpdateSystemConstantValues), a 5-1-shaped
// register->struct mapping that feeds the translated shaders their flags, NDC
// transform, point sprite sizes, texture signs and colour exponent biases.
// This unit is that derivation, transcribed independently: every register
// index, bit offset and mapping rule is taken from the register definitions,
// NOT from the emulated function, which is the ground truth it is checked
// against ([NR-PSO] gave the pattern).
//
// The gate: the consumer keeps ONE persistent NrSysConstants beside the
// emulated system_constants_, updates it through NrSysUpdate at the end of
// every emulated UpdateSystemConstantValues call (same thread, same register
// file, same instant), and memcmp's the two whole structs. mismatch must be 0
// at city load. The struct is STICKY by design - the emulated function writes
// many fields conditionally (clip planes only when enabled, point sizes only
// for point lists, byte-granular texture signs) and never initializes its
// member, so the consumer must SEED the mirror from the emulated struct once
// when the gate arms; every field the function writes after that moment is
// ours.
//
// What is taken as INPUT rather than derived, and why:
//   - the NDC scale/offset and viewport extent (draw_util::GetHostViewportInfo)
//     are the host viewport derivation - a separate 5-1-sized transcription,
//     its own later peel. Both sides consume the same ViewportInfo here.
//   - primitive_polygonal, line_loop_closing_index, the index endian and
//     shared_memory_is_uav come from the primitive processor / memexport
//     analysis, same as 5-1 took them.
//   - texture swizzled signs and resolution-scaledness are texture-cache
//     queries (guest texture headers, not registers) - reached through
//     callbacks so this file stays SDK-free.
//   - the render-target-cache configuration bools (gamma-as-unorm16, the
//     fixed16 -32..32 remap).
//
// NOT COVERED, by explicit scope: the ROV (pixel-shader-interlock) half of
// the derivation - every edram_* field, the ROV flag bits, and
// GetPSIColorFormatInfo's format tables. This game runs the host-render-target
// path in every measured run (5-1 onward), so those branches cannot be
// exercised; NrSysUpdate REFUSES (returns false, touches nothing) when
// edram_rov_used is set, and the consumer counts the refusal instead of
// comparing. If a refusal ever shows up outside a unit test, that is itself a
// finding.
//
// Pure functions over caller-owned state, no globals, no SDK dependencies:
// tools/nr-sys-constants-test.cpp builds this file bare.

namespace rex {
namespace graphics {
namespace nr {

// ---------------------------------------------------------------------------
// Guest register indices this increment adds (transcribed from
// register_table.inc; the shared ones come from nr_pipeline_state.h).
// ---------------------------------------------------------------------------
enum : uint32_t {
  kNrRegVgtMaxVtxIndx = 0x2100,
  kNrRegVgtMinVtxIndx = 0x2101,
  kNrRegVgtIndxOffset = 0x2102,
  kNrRegRbBlendRed = 0x2105,    // float; +1 green, +2 blue, +3 alpha
  kNrRegRbAlphaRef = 0x210E,    // float
  kNrRegRbColorControl = 0x2202,
  kNrRegPaClVteCntl = 0x2206,
  kNrRegPaSuPointSize = 0x2280,
  kNrRegPaSuPointMinmax = 0x2281,
  kNrRegVgtHosMaxTessLevel = 0x2286,  // float
  kNrRegVgtHosMinTessLevel = 0x2287,  // float
  kNrRegPaClUcp0X = 0x2388,           // 6 planes x 4 floats, consecutive
};

// xenos::PrimitiveType values the derivation names beyond 5-1's set.
enum : uint32_t {
  kNrPrimLineLoop = 0x0C,
  kNrPrimLinePatch = 0x10,
};

// xenos::ColorRenderTargetFormat values the derivation names.
enum : uint32_t {
  kNrColorFmt8888Gamma = 1,
  kNrColorFmt1616 = 4,
  kNrColorFmt16161616 = 5,
};

// DxbcShaderTranslator system-constant flag bits (transcribed; the ROV bits
// beyond kNrSysFlagConvertColor0ToGamma+3 are out of scope with the ROV path).
enum : uint32_t {
  kNrSysFlagSharedMemoryIsUAV = 1u << 0,
  kNrSysFlagXYDividedByW = 1u << 1,
  kNrSysFlagZDividedByW = 1u << 2,
  kNrSysFlagWNotReciprocal = 1u << 3,
  kNrSysFlagPrimitivePolygonal = 1u << 4,
  kNrSysFlagPrimitiveLine = 1u << 5,
  kNrSysFlagDepthFloat24 = 1u << 6,
  kNrSysFlagAlphaPassIfLessShift = 7,  // 3 bits: xenos::CompareFunction
  kNrSysFlagConvertColor0ToGamma = 1u << 10,  // << rt index for 1..3
};

// ---------------------------------------------------------------------------
// The mirror struct. BYTE-LAYOUT MIRROR of
// DxbcShaderTranslator::SystemConstants: field order, widths and packing are
// identical so a memcmp of the two is the whole gate (the consumer
// static_asserts the sizes and spot offsets against the real one). The
// original's unions are flattened; enum-typed fields are plain uint32_t.
// ---------------------------------------------------------------------------
struct NrSysConstants {
  uint32_t flags;                              // 0
  float tessellation_factor_range_min;         // 4
  float tessellation_factor_range_max;         // 8
  uint32_t line_loop_closing_index;            // 12

  uint32_t vertex_index_endian;                // 16  xenos::Endian
  uint32_t vertex_index_offset;                // 20
  uint32_t vertex_index_min;                   // 24
  uint32_t vertex_index_max;                   // 28

  float user_clip_planes[6][4];                // 32

  float ndc_scale[3];                          // 128
  float point_vertex_diameter_min;             // 140
  float ndc_offset[3];                         // 144
  float point_vertex_diameter_max;             // 156
  float point_constant_diameter[2];            // 160
  float point_screen_diameter_to_ndc_radius[2];  // 168

  uint32_t texture_swizzled_signs[8];          // 176
  uint32_t textures_resolution_scaled;         // 208
  uint32_t sample_count_log2[2];               // 212
  float alpha_test_reference;                  // 220
  uint32_t alpha_to_mask;                      // 224

  // ROV-only from here on (sticky: seeded once, never derived while the ROV
  // path is refused), except color_exp_bias which is written on every call.
  uint32_t edram_32bpp_tile_pitch_dwords_scaled;        // 228
  uint32_t edram_depth_base_dwords_scaled;              // 232
  uint32_t padding_edram_depth_base_dwords_scaled;      // 236
  float color_exp_bias[4];                              // 240
  float edram_poly_offset_front[2];                     // 256
  float edram_poly_offset_back[2];                      // 264
  uint32_t edram_stencil_front[4];                      // 272
  uint32_t edram_stencil_back[4];                       // 288
  uint32_t edram_rt_base_dwords_scaled[4];              // 304
  uint32_t edram_rt_format_flags[4];                    // 320
  float edram_rt_clamp[4][4];                           // 336
  uint32_t edram_rt_keep_mask[4][2];                    // 400
  uint32_t edram_rt_blend_factors_ops[4];               // 432
  float edram_blend_constant[4];                        // 448
};
static_assert(sizeof(NrSysConstants) == 464, "layout drifted");
constexpr uint32_t kNrSysConstantsDwords = sizeof(NrSysConstants) / 4;

// ---------------------------------------------------------------------------
// Inputs the derivation cannot obtain from the register file (see the header
// comment for why each is an input).
// ---------------------------------------------------------------------------
typedef uint8_t (*NrSysTextureSignsFn)(void* ctx, uint32_t fetch_constant_index);
typedef bool (*NrSysTextureResScaledFn)(void* ctx, uint32_t fetch_constant_index);

struct NrSysInputs {
  // The draw register file's flat value array (RegisterFile::values).
  const uint32_t* regs;

  // Draw-call inputs (primitive processor / memexport analysis).
  bool shared_memory_is_uav;
  bool primitive_polygonal;
  uint32_t line_loop_closing_index;
  uint32_t index_endian;  // xenos::Endian

  // Host viewport derivation (draw_util::GetHostViewportInfo) - a later peel.
  float ndc_scale[3];
  float ndc_offset[3];
  uint32_t xy_extent[2];

  // Shaders' used-texture mask after translation.
  uint32_t used_texture_mask;

  // Render-target path configuration. edram_rov_used true => REFUSE.
  bool edram_rov_used;
  // GetPath() == kHostRenderTargets && !IsFixed16TruncatedToMinus1To1():
  // the -32..32 fixed16 formats get their exponent bias remapped by -5.
  bool color_exp_bias_host_remap;
  // gamma_render_target_as_unorm16(): suppresses the ConvertColorToGamma bits.
  bool gamma_render_target_as_unorm16;

  // Resolution scale (texture cache; 1 unless resolution_scale is set).
  uint32_t draw_resolution_scale_x;
  uint32_t draw_resolution_scale_y;

  // Texture-cache queries, only consulted for bits set in used_texture_mask.
  NrSysTextureSignsFn texture_signs_fn;
  NrSysTextureResScaledFn texture_res_scaled_fn;
  void* texture_ctx;
};

// Applies one UpdateSystemConstantValues-equivalent derivation to `state`,
// with the same conditional/sticky write semantics as the emulated function.
// Returns false WITHOUT touching state when in.edram_rov_used is set (the ROV
// derivation is out of scope - the consumer counts the refusal).
bool NrSysUpdate(const NrSysInputs& in, NrSysConstants* state);

// Returns kNrSysConstantsDwords when equal, otherwise the first differing
// dword index (compare as dwords: float fields may legitimately hold NaNs and
// must still compare bit-exactly).
uint32_t NrSysFirstDifference(const NrSysConstants& ours, const void* theirs);

// Names a dword for the FIRST DIFFERENCE report, "user_clip_planes[2][1]"
// style. Writes into a caller buffer.
void NrSysDwordName(uint32_t dword_index, char* out, uint32_t out_size);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_SYS_CONSTANTS_H_
