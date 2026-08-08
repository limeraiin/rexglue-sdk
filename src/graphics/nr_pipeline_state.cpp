/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_pipeline_state.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// [NR-PSO] See the header. Every bit offset below is transcribed from the
// register definitions rather than reused from them, and every mapping rule
// from the back end's derivation rather than called into it, because the whole
// value of this unit is that the two are INDEPENDENT and then compared.

namespace rex {
namespace graphics {
namespace nr {

const uint32_t kNrRegRbColorInfo[4] = {0x2001, 0x2003, 0x2004, 0x2005};
const uint32_t kNrRegRbBlendControl[4] = {0x2201, 0x2209, 0x220A, 0x220B};

namespace {

// --- register field accessors -------------------------------------------
// Named after the fields they read so a wrong offset is a wrong name, not an
// anonymous shift buried in an expression.

inline uint32_t Field(uint32_t value, uint32_t shift, uint32_t bits) {
  return (value >> shift) & ((uint32_t(1) << bits) - 1);
}

inline float AsFloat(uint32_t value) {
  float f;
  std::memcpy(&f, &value, sizeof(f));
  return f;
}

// RB_MODECONTROL
inline uint32_t EdramMode(const uint32_t* regs) { return Field(regs[kNrRegRbModeControl], 0, 3); }
inline bool EdramModeUsesDepth(uint32_t edram_mode) {
  return edram_mode == kNrEdramColorDepth || edram_mode == kNrEdramDepthOnly;
}

// RB_SURFACE_INFO
inline uint32_t SurfacePitch(const uint32_t* regs) { return Field(regs[kNrRegRbSurfaceInfo], 0, 14); }
inline uint32_t MsaaSamples(const uint32_t* regs) { return Field(regs[kNrRegRbSurfaceInfo], 16, 2); }

// SQ_PROGRAM_CNTL
inline uint32_t VsExportMode(const uint32_t* regs) { return Field(regs[kNrRegSqProgramCntl], 24, 3); }

// VGT_DRAW_INITIATOR / VGT_OUTPUT_PATH_CNTL
inline uint32_t PrimType(const uint32_t* regs) { return Field(regs[kNrRegVgtDrawInitiator], 0, 6); }
inline uint32_t VgtOutputPath(const uint32_t* regs) {
  return Field(regs[kNrRegVgtOutputPathCntl], 0, 2);
}

// PA_SU_SC_MODE_CNTL
inline uint32_t CullFront(const uint32_t* regs) { return Field(regs[kNrRegPaSuScModeCntl], 0, 1); }
inline uint32_t CullBack(const uint32_t* regs) { return Field(regs[kNrRegPaSuScModeCntl], 1, 1); }
inline uint32_t Face(const uint32_t* regs) { return Field(regs[kNrRegPaSuScModeCntl], 2, 1); }
inline uint32_t PolyMode(const uint32_t* regs) { return Field(regs[kNrRegPaSuScModeCntl], 3, 2); }
inline uint32_t PolymodeFrontPtype(const uint32_t* regs) {
  return Field(regs[kNrRegPaSuScModeCntl], 5, 3);
}
inline uint32_t PolymodeBackPtype(const uint32_t* regs) {
  return Field(regs[kNrRegPaSuScModeCntl], 8, 3);
}
inline uint32_t PolyOffsetFrontEnable(const uint32_t* regs) {
  return Field(regs[kNrRegPaSuScModeCntl], 11, 1);
}
inline uint32_t PolyOffsetBackEnable(const uint32_t* regs) {
  return Field(regs[kNrRegPaSuScModeCntl], 12, 1);
}
inline uint32_t PolyOffsetParaEnable(const uint32_t* regs) {
  return Field(regs[kNrRegPaSuScModeCntl], 13, 1);
}

// PA_CL_CLIP_CNTL
inline uint32_t ClipDisable(const uint32_t* regs) { return Field(regs[kNrRegPaClClipCntl], 16, 1); }

// RB_DEPTH_INFO
inline uint32_t DepthFormat(const uint32_t* regs) { return Field(regs[kNrRegRbDepthInfo], 16, 1); }

// RB_DEPTHCONTROL, taken on an already-normalized value.
inline uint32_t DcStencilEnable(uint32_t dc) { return Field(dc, 0, 1); }
inline uint32_t DcZEnable(uint32_t dc) { return Field(dc, 1, 1); }
inline uint32_t DcZWriteEnable(uint32_t dc) { return Field(dc, 2, 1); }
inline uint32_t DcZFunc(uint32_t dc) { return Field(dc, 4, 3); }
inline uint32_t DcBackfaceEnable(uint32_t dc) { return Field(dc, 7, 1); }
inline uint32_t DcStencilFunc(uint32_t dc) { return Field(dc, 8, 3); }
inline uint32_t DcStencilFail(uint32_t dc) { return Field(dc, 11, 3); }
inline uint32_t DcStencilZPass(uint32_t dc) { return Field(dc, 14, 3); }
inline uint32_t DcStencilZFail(uint32_t dc) { return Field(dc, 17, 3); }
inline uint32_t DcStencilFuncBf(uint32_t dc) { return Field(dc, 20, 3); }
inline uint32_t DcStencilFailBf(uint32_t dc) { return Field(dc, 23, 3); }
inline uint32_t DcStencilZPassBf(uint32_t dc) { return Field(dc, 26, 3); }
inline uint32_t DcStencilZFailBf(uint32_t dc) { return Field(dc, 29, 3); }

// RB_STENCILREFMASK
inline uint32_t StencilMask(uint32_t v) { return Field(v, 8, 8); }
inline uint32_t StencilWriteMask(uint32_t v) { return Field(v, 16, 8); }

// RB_COLOR_INFO
inline uint32_t ColorFormat(uint32_t v) { return Field(v, 16, 4); }

// RB_BLENDCONTROL
inline uint32_t ColorSrcBlend(uint32_t v) { return Field(v, 0, 5); }
inline uint32_t ColorCombFcn(uint32_t v) { return Field(v, 5, 3); }
inline uint32_t ColorDestBlend(uint32_t v) { return Field(v, 8, 5); }
inline uint32_t AlphaSrcBlend(uint32_t v) { return Field(v, 16, 5); }
inline uint32_t AlphaCombFcn(uint32_t v) { return Field(v, 21, 3); }
inline uint32_t AlphaDestBlend(uint32_t v) { return Field(v, 24, 5); }

// --- mapping tables ------------------------------------------------------

// Guest blend factor (5 bits, so 32 rows for safety) -> host factor. The two
// unassigned guest codes 2 and 3 map to zero, as do all codes above 16.
const uint32_t kBlendFactorMap[32] = {
    kNrPsoBlendZero,          //  0
    kNrPsoBlendOne,           //  1
    kNrPsoBlendZero,          //  2 unassigned
    kNrPsoBlendZero,          //  3 unassigned
    kNrPsoBlendSrcColor,      //  4
    kNrPsoBlendInvSrcColor,   //  5
    kNrPsoBlendSrcAlpha,      //  6
    kNrPsoBlendInvSrcAlpha,   //  7
    kNrPsoBlendDestColor,     //  8
    kNrPsoBlendInvDestColor,  //  9
    kNrPsoBlendDestAlpha,     // 10
    kNrPsoBlendInvDestAlpha,  // 11
    kNrPsoBlendBlendFactor,      // 12 CONSTANT_COLOR
    kNrPsoBlendInvBlendFactor,   // 13 ONE_MINUS_CONSTANT_COLOR
    kNrPsoBlendBlendFactor,      // 14 CONSTANT_ALPHA
    kNrPsoBlendInvBlendFactor,   // 15 ONE_MINUS_CONSTANT_ALPHA
    kNrPsoBlendSrcAlphaSat,      // 16
};

// The same, with the colour modes replaced by their alpha equivalents: the
// alpha channel of a colour factor IS the alpha factor, and collapsing them
// keeps titles that use a colour mode for alpha from keying extra pipelines.
const uint32_t kBlendFactorAlphaMap[32] = {
    kNrPsoBlendZero,          //  0
    kNrPsoBlendOne,           //  1
    kNrPsoBlendZero,          //  2 unassigned
    kNrPsoBlendZero,          //  3 unassigned
    kNrPsoBlendSrcAlpha,      //  4 SRC_COLOR
    kNrPsoBlendInvSrcAlpha,   //  5 ONE_MINUS_SRC_COLOR
    kNrPsoBlendSrcAlpha,      //  6
    kNrPsoBlendInvSrcAlpha,   //  7
    kNrPsoBlendDestAlpha,     //  8 DST_COLOR
    kNrPsoBlendInvDestAlpha,  //  9 ONE_MINUS_DST_COLOR
    kNrPsoBlendDestAlpha,     // 10
    kNrPsoBlendInvDestAlpha,  // 11
    kNrPsoBlendBlendFactor,     // 12
    kNrPsoBlendInvBlendFactor,  // 13
    kNrPsoBlendBlendFactor,     // 14
    kNrPsoBlendInvBlendFactor,  // 15
    kNrPsoBlendSrcAlphaSat,     // 16
};

// xenos::GetColorRenderTargetFormatComponentCount. Unknown formats yield 0,
// matching the back end's unhandled-case fallthrough (its assert is compiled
// out of release builds, which is where this runs).
uint32_t ColorFormatComponentCount(uint32_t format) {
  switch (format) {
    case 0:   // k_8_8_8_8
    case 1:   // k_8_8_8_8_GAMMA
    case 2:   // k_2_10_10_10
    case 3:   // k_2_10_10_10_FLOAT
    case 5:   // k_16_16_16_16
    case 7:   // k_16_16_16_16_FLOAT
    case 10:  // k_2_10_10_10_AS_10_10_10_10
    case 12:  // k_2_10_10_10_FLOAT_AS_16_16_16_16
      return 4;
    case 4:   // k_16_16
    case 6:   // k_16_16_FLOAT
    case 15:  // k_32_32_FLOAT
      return 2;
    case 14:  // k_32_FLOAT
      return 1;
    default:
      return 0;
  }
}

// The polygon offset unit conversions, transcribed from draw_util. Named
// constants rather than literals so the two factors cannot be swapped
// silently.
constexpr float kPolygonOffsetFactorUnorm24 = float((uint32_t(1) << 24) - 1);
constexpr float kPolygonOffsetFactorFloat24 = float(uint32_t(1) << (21 + 3));
constexpr float kPolygonOffsetScaleSubpixelUnit = 1.0f / 16.0f;
constexpr uint32_t kDepthFormatD24FS8 = 1;

const char* const kFieldNames[] = {
    "vertex_shader_hash",
    "vertex_shader_modification",
    "pixel_shader_hash",
    "pixel_shader_modification",
    "depth_bias",
    "depth_bias_slope_scaled",
    "strip_cut_index",
    "primitive_topology_type",
    "geometry_shader",
    "fill_mode_wireframe",
    "cull_mode",
    "front_counter_clockwise",
    "depth_clip",
    "host_msaa_samples",
    "depth_format",
    "depth_func",
    "depth_write",
    "stencil_enable",
    "stencil_read_mask",
    "stencil_write_mask",
    "stencil_front_fail_op",
    "stencil_front_depth_fail_op",
    "stencil_front_pass_op",
    "stencil_front_func",
    "stencil_back_fail_op",
    "stencil_back_depth_fail_op",
    "stencil_back_pass_op",
    "stencil_back_func",
};
static_assert(sizeof(kFieldNames) / sizeof(kFieldNames[0]) == kNrPsoFieldRenderTarget0,
              "scalar field names must cover every scalar field");

const char* const kRenderTargetFieldNames[kNrPsoRenderTargetFieldCount] = {
    "used",     "format",          "src_blend",       "dest_blend",     "blend_op",
    "src_blend_alpha", "dest_blend_alpha", "blend_op_alpha", "write_mask",
};

// One render target's field, widened. Kept beside the name table so the two
// orders cannot drift apart.
uint64_t RenderTargetField(const NrPsoRenderTarget& rt, uint32_t sub) {
  switch (sub) {
    case 0: return rt.used;
    case 1: return rt.format;
    case 2: return rt.src_blend;
    case 3: return rt.dest_blend;
    case 4: return rt.blend_op;
    case 5: return rt.src_blend_alpha;
    case 6: return rt.dest_blend_alpha;
    case 7: return rt.blend_op_alpha;
    default: return rt.write_mask;
  }
}

uint64_t ScalarField(const NrPsoDesc& d, uint32_t field) {
  switch (field) {
    case kNrPsoFieldVertexShaderHash: return d.vertex_shader_hash;
    case kNrPsoFieldVertexShaderModification: return d.vertex_shader_modification;
    case kNrPsoFieldPixelShaderHash: return d.pixel_shader_hash;
    case kNrPsoFieldPixelShaderModification: return d.pixel_shader_modification;
    case kNrPsoFieldDepthBias: return uint32_t(d.depth_bias);
    case kNrPsoFieldDepthBiasSlopeScaled: {
      // By bit pattern: a difference that compares equal as a float (a signed
      // zero, a differently-encoded NaN) still changes the pipeline key,
      // because the back end hashes and memcmps these bytes.
      uint32_t bits;
      std::memcpy(&bits, &d.depth_bias_slope_scaled, sizeof(bits));
      return bits;
    }
    case kNrPsoFieldStripCutIndex: return d.strip_cut_index;
    case kNrPsoFieldPrimitiveTopologyType:
      return d.primitive_topology_type_or_tessellation_mode;
    case kNrPsoFieldGeometryShader: return d.geometry_shader;
    case kNrPsoFieldFillModeWireframe: return d.fill_mode_wireframe;
    case kNrPsoFieldCullMode: return d.cull_mode;
    case kNrPsoFieldFrontCounterClockwise: return d.front_counter_clockwise;
    case kNrPsoFieldDepthClip: return d.depth_clip;
    case kNrPsoFieldHostMsaaSamples: return d.host_msaa_samples;
    case kNrPsoFieldDepthFormat: return d.depth_format;
    case kNrPsoFieldDepthFunc: return d.depth_func;
    case kNrPsoFieldDepthWrite: return d.depth_write;
    case kNrPsoFieldStencilEnable: return d.stencil_enable;
    case kNrPsoFieldStencilReadMask: return d.stencil_read_mask;
    case kNrPsoFieldStencilWriteMask: return d.stencil_write_mask;
    case kNrPsoFieldStencilFrontFailOp: return d.stencil_front_fail_op;
    case kNrPsoFieldStencilFrontDepthFailOp: return d.stencil_front_depth_fail_op;
    case kNrPsoFieldStencilFrontPassOp: return d.stencil_front_pass_op;
    case kNrPsoFieldStencilFrontFunc: return d.stencil_front_func;
    case kNrPsoFieldStencilBackFailOp: return d.stencil_back_fail_op;
    case kNrPsoFieldStencilBackDepthFailOp: return d.stencil_back_depth_fail_op;
    case kNrPsoFieldStencilBackPassOp: return d.stencil_back_pass_op;
    default: return d.stencil_back_func;
  }
}

}  // namespace

bool NrPsoIsPrimitivePolygonal(const uint32_t* regs) {
  const uint32_t type = PrimType(regs);
  if (VgtOutputPath(regs) == kNrVgtOutputTessellationEnable &&
      (type == kNrPrimTrianglePatch || type == kNrPrimQuadPatch)) {
    // For patch primitive types the major mode is always explicit, so the path
    // select alone decides.
    return true;
  }
  switch (type) {
    case kNrPrimTriangleList:
    case kNrPrimTriangleFan:
    case kNrPrimTriangleStrip:
    case kNrPrimTriangleWithWFlags:
    case kNrPrimQuadList:
    case kNrPrimQuadStrip:
    case kNrPrimPolygon:
      return true;
    default:
      // Rectangle lists deliberately included here: the geometry shader that
      // emulates them ignores winding, so treating them as polygonal would
      // enable backface culling that the guest does not do.
      return false;
  }
}

bool NrPsoIsRasterizationPotentiallyDone(const uint32_t* regs, bool primitive_polygonal) {
  if (!EdramModeUsesDepth(EdramMode(regs))) {
    return false;
  }
  if (VsExportMode(regs) == kNrVsExportMultipass || !SurfacePitch(regs)) {
    return false;
  }
  if (primitive_polygonal && CullFront(regs) && CullBack(regs)) {
    return false;
  }
  return true;
}

uint32_t NrPsoNormalizedDepthControl(const uint32_t* regs) {
  if (!EdramModeUsesDepth(EdramMode(regs))) {
    // Depth and stencil are both off, whatever the register holds.
    return 0;
  }
  uint32_t dc = regs[kNrRegRbDepthControl];
  // A depth test that always passes and never writes cannot affect anything,
  // so fold it away and let such draws share a pipeline (and skip depth render
  // target management entirely). Stencil is left alone: it is complex, and
  // titles enable it explicitly when they mean it.
  if (DcZEnable(dc) && !DcZWriteEnable(dc) && DcZFunc(dc) == kNrCompareAlways) {
    dc &= ~(uint32_t(1) << 1);
  }
  return dc;
}

uint32_t NrPsoNormalizedColorMask(const uint32_t* regs,
                                  uint32_t pixel_shader_writes_color_targets) {
  if (EdramMode(regs) != kNrEdramColorDepth) {
    return 0;
  }
  const uint32_t rb_color_mask = regs[kNrRegRbColorMask];
  uint32_t normalized = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    // A target the shader does not statically write must not be written, and
    // must not take part in host render target ownership either.
    if (!(pixel_shader_writes_color_targets & (uint32_t(1) << i))) {
      continue;
    }
    const uint32_t component_mask =
        (uint32_t(1) << ColorFormatComponentCount(ColorFormat(regs[kNrRegRbColorInfo[i]]))) - 1;
    uint32_t rt_write_mask = ((rb_color_mask >> (4 * i)) & 0xF) & component_mask;
    if (!rt_write_mask) {
      continue;
    }
    // Force the components the format does not have to "written", so a driver
    // that does not check for itself cannot take a read-modify-write path over
    // components that do not exist.
    rt_write_mask |= 0xF & ~component_mask;
    normalized |= rt_write_mask << (4 * i);
  }
  return normalized;
}

void NrPsoPreferredFacePolygonOffset(const uint32_t* regs, bool primitive_polygonal,
                                     float* scale_out, float* offset_out) {
  float scale = 0.0f, offset = 0.0f;
  if (primitive_polygonal) {
    // Prefer the front face: it is the one usually rendered (shadow volumes
    // being the exception).
    if (PolyOffsetFrontEnable(regs) && !CullFront(regs)) {
      scale = AsFloat(regs[kNrRegPaSuPolyOffsetFrontScale]);
      offset = AsFloat(regs[kNrRegPaSuPolyOffsetFrontOffset]);
    }
    if (PolyOffsetBackEnable(regs) && !CullBack(regs) && !scale && !offset) {
      scale = AsFloat(regs[kNrRegPaSuPolyOffsetBackScale]);
      offset = AsFloat(regs[kNrRegPaSuPolyOffsetBackOffset]);
    }
  } else {
    // Non-polygonal primitives use the front registers, gated by the "para"
    // enable rather than the front one.
    if (PolyOffsetParaEnable(regs)) {
      scale = AsFloat(regs[kNrRegPaSuPolyOffsetFrontScale]);
      offset = AsFloat(regs[kNrRegPaSuPolyOffsetFrontOffset]);
    }
  }
  *scale_out = scale;
  *offset_out = offset;
}

int32_t NrPsoD3D10IntegerPolygonOffset(uint32_t depth_format, float polygon_offset) {
  const bool is_float24 = depth_format == kDepthFormatD24FS8;
  // Ceil, not floor: if an offset is specified at all the primitives are meant
  // to be separated, and flooring can round that intent to nothing.
  int32_t polygon_offset_int = int32_t(
      std::ceil(std::abs(polygon_offset) * (is_float24 ? kPolygonOffsetFactorFloat24 * (1.0f / 8.0f)
                                                       : kPolygonOffsetFactorUnorm24)));
  // float24 may be produced by truncation in a translated pixel shader, so the
  // bias is kept in multiples of 8 float32 ULPs (the mantissa bit difference),
  // with the ceil done before the unit change.
  if (is_float24) {
    polygon_offset_int <<= 3;
  }
  return polygon_offset < 0 ? -polygon_offset_int : polygon_offset_int;
}

bool NrPsoDerive(const NrPsoInputs& in, NrPsoDesc* out) {
  std::memset(out, 0, sizeof(*out));

  const uint32_t* regs = in.regs;
  const bool primitive_polygonal = NrPsoIsPrimitivePolygonal(regs);
  const bool rasterization_enabled =
      NrPsoIsRasterizationPotentiallyDone(regs, primitive_polygonal);
  if (!rasterization_enabled && in.has_pixel_shader) {
    // Rasterization is disabled by disabling the pixel shader and depth /
    // stencil, so the caller must already have dropped the pixel shader --
    // otherwise the texture binding layout would belong to a shader that is
    // not the one being used.
    return false;
  }

  out->vertex_shader_hash = in.vertex_shader_hash;
  out->vertex_shader_modification = in.vertex_shader_modification;

  out->strip_cut_index = in.host_primitive_reset_enabled
                             ? (in.host_index_format_32 ? kNrPsoStripCutFFFFFFFF
                                                        : kNrPsoStripCutFFFF)
                             : kNrPsoStripCutNone;

  if (in.tessellated) {
    out->primitive_topology_type_or_tessellation_mode = in.tessellation_mode;
  } else {
    switch (in.host_primitive_type) {
      case kNrPrimPointList:
        out->primitive_topology_type_or_tessellation_mode = kNrPsoTopologyPoint;
        break;
      case kNrPrimLineList:
      case kNrPrimLineStrip:
      // Quads are emulated as line lists with adjacency.
      case kNrPrimQuadList:
      case kNrPrim2DLineStrip:
        out->primitive_topology_type_or_tessellation_mode = kNrPsoTopologyLine;
        break;
      default:
        out->primitive_topology_type_or_tessellation_mode = kNrPsoTopologyTriangle;
        break;
    }
    switch (in.host_primitive_type) {
      case kNrPrimPointList:
        out->geometry_shader = kNrPsoGsPointList;
        break;
      case kNrPrimRectangleList:
        out->geometry_shader = kNrPsoGsRectangleList;
        break;
      case kNrPrimQuadList:
        out->geometry_shader = kNrPsoGsQuadList;
        break;
      default:
        out->geometry_shader = kNrPsoGsNone;
        break;
    }
  }

  // With rasterization off nothing downstream can write anywhere or count a
  // sample, so the rest of the description is deliberately left zeroed.
  if (!rasterization_enabled) {
    out->cull_mode = kNrPsoCullDisableRasterization;
    return true;
  }

  if (in.has_pixel_shader) {
    out->pixel_shader_hash = in.pixel_shader_hash;
    out->pixel_shader_modification = in.pixel_shader_modification;
  }

  // Rasterizer state. Direct3D 12 has neither per-side fill mode nor per-side
  // depth bias, so which side is culled decides which side's values are used.
  // With nothing culled, a wireframe or bias on either side is taken as
  // intentional (the front one for the bias -- SetRenderState writes both).
  bool cull_front = false, cull_back = false;
  if (primitive_polygonal) {
    out->front_counter_clockwise = Face(regs) == 0;
    cull_front = CullFront(regs) != 0;
    cull_back = CullBack(regs) != 0;
    if (cull_front) {
      // Both sides culled is handled by disabling rasterization above.
      out->cull_mode = kNrPsoCullFront;
    } else if (cull_back) {
      out->cull_mode = kNrPsoCullBack;
    } else {
      out->cull_mode = kNrPsoCullNone;
    }
    // Direct3D 12 has no point fill mode either; assume a title asking for
    // points did not want the primitive filled, and use wireframe.
    if (!cull_front && PolymodeFrontPtype(regs) != kNrPolygonTypeTriangles) {
      out->fill_mode_wireframe = 1;
    }
    if (!cull_back && PolymodeBackPtype(regs) != kNrPolygonTypeTriangles) {
      out->fill_mode_wireframe = 1;
    }
    if (PolyMode(regs) != kNrPolygonModeDual) {
      out->fill_mode_wireframe = 0;
    }
  }

  if (!in.edram_rov_used) {
    // With the ROV path the bias is applied in the pixel shader instead,
    // because MSAA needs it per sample.
    float polygon_offset = 0.0f, polygon_offset_scale = 0.0f;
    NrPsoPreferredFacePolygonOffset(regs, primitive_polygonal, &polygon_offset_scale,
                                    &polygon_offset);
    out->depth_bias = NrPsoD3D10IntegerPolygonOffset(DepthFormat(regs), polygon_offset);
    out->depth_bias_slope_scaled = polygon_offset_scale * kPolygonOffsetScaleSubpixelUnit;
  }
  if (in.tessellated && in.tessellation_wireframe) {
    out->fill_mode_wireframe = 1;
  }
  out->depth_clip = !ClipDisable(regs);

  bool depth_stencil_bound_and_used = false;
  if (!in.edram_rov_used) {
    const uint32_t dc = NrPsoNormalizedDepthControl(regs);
    if (in.bound_rt_bits & 1) {
      if (DcZEnable(dc)) {
        out->depth_func = DcZFunc(dc);
        out->depth_write = DcZWriteEnable(dc);
      } else {
        out->depth_func = kNrCompareAlways;
      }
      if (DcStencilEnable(dc)) {
        out->stencil_enable = 1;
        const bool stencil_backface_enable = primitive_polygonal && DcBackfaceEnable(dc);
        // Direct3D 12 has no per-face masks, so take the back face ones only
        // when the front faces are culled and the back ones are all that draw.
        const uint32_t stencil_ref_mask =
            regs[(stencil_backface_enable && cull_front) ? kNrRegRbStencilRefMaskBf
                                                         : kNrRegRbStencilRefMask];
        out->stencil_read_mask = StencilMask(stencil_ref_mask);
        out->stencil_write_mask = StencilWriteMask(stencil_ref_mask);
        out->stencil_front_fail_op = DcStencilFail(dc);
        out->stencil_front_depth_fail_op = DcStencilZFail(dc);
        out->stencil_front_pass_op = DcStencilZPass(dc);
        out->stencil_front_func = DcStencilFunc(dc);
        if (stencil_backface_enable) {
          out->stencil_back_fail_op = DcStencilFailBf(dc);
          out->stencil_back_depth_fail_op = DcStencilZFailBf(dc);
          out->stencil_back_pass_op = DcStencilZPassBf(dc);
          out->stencil_back_func = DcStencilFuncBf(dc);
        } else {
          out->stencil_back_fail_op = out->stencil_front_fail_op;
          out->stencil_back_depth_fail_op = out->stencil_front_depth_fail_op;
          out->stencil_back_pass_op = out->stencil_front_pass_op;
          out->stencil_back_func = out->stencil_front_func;
        }
      }
      // If the depth / stencil view will not be bound, its format must stay
      // out of the key.
      if (out->depth_func != kNrCompareAlways || out->depth_write || out->stencil_enable) {
        out->depth_format = in.bound_rt_formats[0];
        depth_stencil_bound_and_used = true;
      }
    } else {
      out->depth_func = kNrCompareAlways;
    }

    const uint32_t normalized_color_mask =
        NrPsoNormalizedColorMask(regs, in.pixel_shader_writes_color_targets);
    for (uint32_t i = 0; i < 4; ++i) {
      if (!(in.bound_rt_bits & (uint32_t(1) << (1 + i)))) {
        continue;
      }
      NrPsoRenderTarget& rt = out->render_targets[i];
      rt.used = 1;
      rt.format = in.bound_rt_formats[1 + i];
      rt.write_mask = (normalized_color_mask >> (i * 4)) & 0xF;
      if (rt.write_mask) {
        const uint32_t blendcontrol = regs[kNrRegRbBlendControl[i]];
        rt.src_blend = kBlendFactorMap[ColorSrcBlend(blendcontrol)];
        rt.dest_blend = kBlendFactorMap[ColorDestBlend(blendcontrol)];
        rt.blend_op = ColorCombFcn(blendcontrol);
        rt.src_blend_alpha = kBlendFactorAlphaMap[AlphaSrcBlend(blendcontrol)];
        rt.dest_blend_alpha = kBlendFactorAlphaMap[AlphaDestBlend(blendcontrol)];
        rt.blend_op_alpha = AlphaCombFcn(blendcontrol);
      } else {
        // A target written by nothing still needs a defined blend state, and
        // src=one dest=zero add is the one that keys the fewest pipelines.
        rt.src_blend = kNrPsoBlendOne;
        rt.dest_blend = kNrPsoBlendZero;
        rt.blend_op = kNrBlendOpAdd;
        rt.src_blend_alpha = kNrPsoBlendOne;
        rt.dest_blend_alpha = kNrPsoBlendZero;
        rt.blend_op_alpha = kNrBlendOpAdd;
      }
    }
  }

  uint32_t host_msaa_samples = MsaaSamples(regs);
  if (in.edram_rov_used) {
    if (host_msaa_samples == kNrMsaa2X) {
      // 2x is not a valid ForcedSampleCount on Nvidia.
      host_msaa_samples = kNrMsaa4X;
    }
  } else if (!(in.bound_rt_bits & ~uint32_t(1)) && !depth_stencil_bound_and_used) {
    // Direct3D 12 requires a sample count of 1 when nothing is bound.
    host_msaa_samples = kNrMsaa1X;
  }
  out->host_msaa_samples = host_msaa_samples;

  return true;
}

void NrPsoFieldName(uint32_t field, char* out, uint32_t out_size) {
  if (!out || !out_size) {
    return;
  }
  if (field < kNrPsoFieldRenderTarget0) {
    std::snprintf(out, out_size, "%s", kFieldNames[field]);
    return;
  }
  if (field >= kNrPsoFieldCount) {
    std::snprintf(out, out_size, "?%u", field);
    return;
  }
  const uint32_t rel = field - kNrPsoFieldRenderTarget0;
  std::snprintf(out, out_size, "rt%u.%s", rel / kNrPsoRenderTargetFieldCount,
                kRenderTargetFieldNames[rel % kNrPsoRenderTargetFieldCount]);
}

uint32_t NrPsoFirstDifference(const NrPsoDesc& ours, const NrPsoDesc& theirs, uint64_t* ours_out,
                              uint64_t* theirs_out) {
  for (uint32_t field = 0; field < kNrPsoFieldRenderTarget0; ++field) {
    const uint64_t a = ScalarField(ours, field), b = ScalarField(theirs, field);
    if (a != b) {
      if (ours_out) *ours_out = a;
      if (theirs_out) *theirs_out = b;
      return field;
    }
  }
  for (uint32_t rt = 0; rt < 4; ++rt) {
    for (uint32_t sub = 0; sub < kNrPsoRenderTargetFieldCount; ++sub) {
      const uint64_t a = RenderTargetField(ours.render_targets[rt], sub);
      const uint64_t b = RenderTargetField(theirs.render_targets[rt], sub);
      if (a != b) {
        if (ours_out) *ours_out = a;
        if (theirs_out) *theirs_out = b;
        return kNrPsoFieldRenderTarget0 + rt * kNrPsoRenderTargetFieldCount + sub;
      }
    }
  }
  return kNrPsoFieldCount;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
