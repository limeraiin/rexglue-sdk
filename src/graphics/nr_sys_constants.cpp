/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <rex/graphics/nr_sys_constants.h>

#include <cstdio>
#include <cstring>

// [NR-SYS] Phase 5-3b-1: the system-constants derivation, transcribed from
// the register definitions. See the header for scope and the refusal rule.

namespace rex {
namespace graphics {
namespace nr {

namespace {

inline float SysBitcastFloat(uint32_t v) {
  float f;
  std::memcpy(&f, &v, sizeof(f));
  return f;
}

// Lowest set bit scan; the derivation only walks small masks (6 clip planes,
// 32 textures), so a loop is fine and keeps this file intrinsic-free.
inline bool SysBitScan(uint32_t mask, uint32_t* index) {
  if (!mask) {
    return false;
  }
  uint32_t i = 0;
  while (!(mask & (1u << i))) {
    ++i;
  }
  *index = i;
  return true;
}

// Whether the primitive is a line: patch types take their major mode from
// VGT_OUTPUT_PATH_CNTL::path_select (2 bits at +0), explicit line types are
// line list / strip / loop and the 2D line strip.
inline bool SysIsPrimitiveLine(const uint32_t* regs) {
  const uint32_t prim = regs[kNrRegVgtDrawInitiator] & 0x3F;
  if ((regs[kNrRegVgtOutputPathCntl] & 0x3) == kNrVgtOutputTessellationEnable &&
      prim == kNrPrimLinePatch) {
    return true;
  }
  return prim == kNrPrimLineList || prim == kNrPrimLineStrip ||
         prim == kNrPrimLineLoop || prim == kNrPrim2DLineStrip;
}

}  // namespace

bool NrSysUpdate(const NrSysInputs& in, NrSysConstants* state) {
  if (in.edram_rov_used) {
    // The ROV half of the derivation (every edram_* field, the ROV flag bits,
    // the PSI format tables) is out of scope; refusing loudly beats deriving
    // half of it. This game never takes the path (5-1 onward).
    return false;
  }
  const uint32_t* regs = in.regs;

  // --- Flags -------------------------------------------------------------
  uint32_t flags = 0;
  if (in.shared_memory_is_uav) {
    flags |= kNrSysFlagSharedMemoryIsUAV;
  }
  // PA_CL_VTE_CNTL: vtx_xy_fmt +8, vtx_z_fmt +9, vtx_w0_fmt +10.
  const uint32_t vte_cntl = regs[kNrRegPaClVteCntl];
  if (vte_cntl & (1u << 8)) {
    flags |= kNrSysFlagXYDividedByW;
  }
  if (vte_cntl & (1u << 9)) {
    flags |= kNrSysFlagZDividedByW;
  }
  if (vte_cntl & (1u << 10)) {
    flags |= kNrSysFlagWNotReciprocal;
  }
  if (in.primitive_polygonal) {
    flags |= kNrSysFlagPrimitivePolygonal;
  }
  if (SysIsPrimitiveLine(regs)) {
    flags |= kNrSysFlagPrimitiveLine;
  }
  // RB_DEPTH_INFO: depth_format 1 bit at +16 (1 = kD24FS8).
  if ((regs[kNrRegRbDepthInfo] >> 16) & 1u) {
    flags |= kNrSysFlagDepthFloat24;
  }
  // RB_COLORCONTROL: alpha_func 3 bits at +0, alpha_test_enable +3,
  // alpha_to_mask_enable +4, alpha-to-mask offsets at +24..31. Disabled test
  // encodes as kAlways (7).
  const uint32_t colorcontrol = regs[kNrRegRbColorControl];
  const uint32_t alpha_test_function =
      (colorcontrol & (1u << 3)) ? (colorcontrol & 0x7u) : 7u;
  flags |= alpha_test_function << kNrSysFlagAlphaPassIfLessShift;
  // Gamma conversion per colour target, unless gamma render targets are host
  // unorm16 (then the conversion is done by the render target itself).
  if (!in.gamma_render_target_as_unorm16) {
    for (uint32_t i = 0; i < 4; ++i) {
      // RB_COLORx_INFO: color_format 4 bits at +16.
      if (((regs[kNrRegRbColorInfo[i]] >> 16) & 0xFu) == kNrColorFmt8888Gamma) {
        flags |= kNrSysFlagConvertColor0ToGamma << i;
      }
    }
  }
  state->flags = flags;

  // --- Tessellation factor range (register float + 1.0) -------------------
  state->tessellation_factor_range_min =
      SysBitcastFloat(regs[kNrRegVgtHosMinTessLevel]) + 1.0f;
  state->tessellation_factor_range_max =
      SysBitcastFloat(regs[kNrRegVgtHosMaxTessLevel]) + 1.0f;

  // --- Index plumbing ------------------------------------------------------
  state->line_loop_closing_index = in.line_loop_closing_index;
  state->vertex_index_endian = in.index_endian;
  // VGT_INDX_OFFSET / VGT_MIN_VTX_INDX / VGT_MAX_VTX_INDX: 24-bit fields.
  state->vertex_index_offset = regs[kNrRegVgtIndxOffset] & 0xFFFFFFu;
  state->vertex_index_min = regs[kNrRegVgtMinVtxIndx] & 0xFFFFFFu;
  state->vertex_index_max = regs[kNrRegVgtMaxVtxIndx] & 0xFFFFFFu;

  // --- User clip planes (tightly packed; only when not CLIP_DISABLE) ------
  // PA_CL_CLIP_CNTL: ucp_ena 6 bits at +0, clip_disable +16.
  const uint32_t clip_cntl = regs[kNrRegPaClClipCntl];
  if (!(clip_cntl & (1u << 16))) {
    float* write_ptr = state->user_clip_planes[0];
    uint32_t remaining = clip_cntl & 0x3Fu;
    uint32_t plane;
    while (SysBitScan(remaining, &plane)) {
      remaining &= ~(1u << plane);
      std::memcpy(write_ptr, &regs[kNrRegPaClUcp0X + plane * 4],
                  4 * sizeof(float));
      write_ptr += 4;
    }
  }

  // --- NDC transform (input: host viewport derivation) --------------------
  for (uint32_t i = 0; i < 3; ++i) {
    state->ndc_scale[i] = in.ndc_scale[i];
    state->ndc_offset[i] = in.ndc_offset[i];
  }

  // --- Point sprites (only for point lists; sticky otherwise) -------------
  if ((regs[kNrRegVgtDrawInitiator] & 0x3F) == kNrPrimPointList) {
    // PA_SU_POINT_MINMAX / PA_SU_POINT_SIZE: 12.4 fixed half-sizes, height /
    // min in the LOW 16, width / max in the HIGH 16.
    const uint32_t point_minmax = regs[kNrRegPaSuPointMinmax];
    const uint32_t point_size = regs[kNrRegPaSuPointSize];
    state->point_vertex_diameter_min =
        float(point_minmax & 0xFFFFu) * (2.0f / 16.0f);
    state->point_vertex_diameter_max =
        float(point_minmax >> 16) * (2.0f / 16.0f);
    state->point_constant_diameter[0] = float(point_size >> 16) * (2.0f / 16.0f);
    state->point_constant_diameter[1] =
        float(point_size & 0xFFFFu) * (2.0f / 16.0f);
    // Guest screen diameter -> host NDC radius: the 0.5 (radius) and the 2
    // (NDC spans 2 per axis) cancel, leaving resolution scale over the
    // viewport extent.
    state->point_screen_diameter_to_ndc_radius[0] =
        float(in.draw_resolution_scale_x) /
        float(in.xy_extent[0] ? in.xy_extent[0] : 1u);
    state->point_screen_diameter_to_ndc_radius[1] =
        float(in.draw_resolution_scale_y) /
        float(in.xy_extent[1] ? in.xy_extent[1] : 1u);
  }

  // --- Texture signs / resolution-scaledness ------------------------------
  // Signs: byte-granular read-modify-write, only for used textures.
  // Resolution-scaled: rebuilt whole from the used mask (NOT sticky).
  uint32_t textures_resolution_scaled = 0;
  uint32_t textures_remaining = in.used_texture_mask;
  uint32_t texture_index;
  while (SysBitScan(textures_remaining, &texture_index)) {
    textures_remaining &= ~(1u << texture_index);
    uint32_t& signs_uint = state->texture_swizzled_signs[texture_index >> 2];
    const uint32_t shift = (texture_index & 3) * 8;
    const uint32_t signs =
        uint32_t(in.texture_signs_fn(in.texture_ctx, texture_index)) << shift;
    signs_uint = (signs_uint & ~(0xFFu << shift)) | signs;
    textures_resolution_scaled |=
        uint32_t(in.texture_res_scaled_fn(in.texture_ctx, texture_index))
        << texture_index;
  }
  state->textures_resolution_scaled = textures_resolution_scaled;

  // --- Sample count log2 --------------------------------------------------
  // RB_SURFACE_INFO: msaa_samples 2 bits at +16 (0=1x, 1=2x, 2=4x). X doubles
  // only at 4x, Y at 2x and above.
  const uint32_t msaa_samples = (regs[kNrRegRbSurfaceInfo] >> 16) & 0x3u;
  state->sample_count_log2[0] = msaa_samples >= 2 ? 1 : 0;
  state->sample_count_log2[1] = msaa_samples >= 1 ? 1 : 0;

  // --- Alpha test / alpha to mask -----------------------------------------
  state->alpha_test_reference = SysBitcastFloat(regs[kNrRegRbAlphaRef]);
  state->alpha_to_mask = (colorcontrol & (1u << 4))
                             ? ((colorcontrol >> 24) | (1u << 8))
                             : 0u;

  // --- Colour exponent bias (written on every call, ROV or not) -----------
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t color_info = regs[kNrRegRbColorInfo[i]];
    // color_exp_bias: SIGNED 6 bits at +20.
    int32_t exp_bias = int32_t(color_info << 6) >> 26;
    const uint32_t color_format = (color_info >> 16) & 0xFu;
    if (color_format == kNrColorFmt1616 || color_format == kNrColorFmt16161616) {
      if (in.color_exp_bias_host_remap) {
        // Host render targets remap the -32..32 fixed16 range to -1..1 by
        // baking a /32 into the bias.
        exp_bias -= 5;
      }
    }
    // 2^bias as a float scale: add the bias straight into the exponent field.
    state->color_exp_bias[i] =
        SysBitcastFloat(0x3F800000u + (uint32_t(exp_bias) << 23));
  }

  return true;
}

uint32_t NrSysFirstDifference(const NrSysConstants& ours, const void* theirs) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&ours);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(theirs);
  for (uint32_t i = 0; i < kNrSysConstantsDwords; ++i) {
    if (a[i] != b[i]) {
      return i;
    }
  }
  return kNrSysConstantsDwords;
}

namespace {

struct SysNameRange {
  uint32_t first_dword;
  uint32_t count;
  const char* name;
  // 0 = scalar/flat array, 1 = [n], 2 = [n][m] with m columns.
  uint32_t dims;
  uint32_t columns;
};

const SysNameRange kSysNameRanges[] = {
    {0, 1, "flags", 0, 0},
    {1, 1, "tessellation_factor_range_min", 0, 0},
    {2, 1, "tessellation_factor_range_max", 0, 0},
    {3, 1, "line_loop_closing_index", 0, 0},
    {4, 1, "vertex_index_endian", 0, 0},
    {5, 1, "vertex_index_offset", 0, 0},
    {6, 1, "vertex_index_min", 0, 0},
    {7, 1, "vertex_index_max", 0, 0},
    {8, 24, "user_clip_planes", 2, 4},
    {32, 3, "ndc_scale", 1, 0},
    {35, 1, "point_vertex_diameter_min", 0, 0},
    {36, 3, "ndc_offset", 1, 0},
    {39, 1, "point_vertex_diameter_max", 0, 0},
    {40, 2, "point_constant_diameter", 1, 0},
    {42, 2, "point_screen_diameter_to_ndc_radius", 1, 0},
    {44, 8, "texture_swizzled_signs", 1, 0},
    {52, 1, "textures_resolution_scaled", 0, 0},
    {53, 2, "sample_count_log2", 1, 0},
    {55, 1, "alpha_test_reference", 0, 0},
    {56, 1, "alpha_to_mask", 0, 0},
    {57, 1, "edram_32bpp_tile_pitch_dwords_scaled", 0, 0},
    {58, 1, "edram_depth_base_dwords_scaled", 0, 0},
    {59, 1, "padding_edram_depth_base_dwords_scaled", 0, 0},
    {60, 4, "color_exp_bias", 1, 0},
    {64, 2, "edram_poly_offset_front", 1, 0},
    {66, 2, "edram_poly_offset_back", 1, 0},
    {68, 4, "edram_stencil_front", 1, 0},
    {72, 4, "edram_stencil_back", 1, 0},
    {76, 4, "edram_rt_base_dwords_scaled", 1, 0},
    {80, 4, "edram_rt_format_flags", 1, 0},
    {84, 16, "edram_rt_clamp", 2, 4},
    {100, 8, "edram_rt_keep_mask", 2, 2},
    {108, 4, "edram_rt_blend_factors_ops", 1, 0},
    {112, 4, "edram_blend_constant", 1, 0},
};

}  // namespace

void NrSysDwordName(uint32_t dword_index, char* out, uint32_t out_size) {
  if (!out || !out_size) {
    return;
  }
  for (const SysNameRange& r : kSysNameRanges) {
    if (dword_index < r.first_dword || dword_index >= r.first_dword + r.count) {
      continue;
    }
    const uint32_t rel = dword_index - r.first_dword;
    if (r.dims == 0) {
      std::snprintf(out, out_size, "%s", r.name);
    } else if (r.dims == 1) {
      std::snprintf(out, out_size, "%s[%u]", r.name, rel);
    } else {
      std::snprintf(out, out_size, "%s[%u][%u]", r.name, rel / r.columns,
                    rel % r.columns);
    }
    return;
  }
  std::snprintf(out, out_size, "dword_%u", dword_index);
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
