/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_PIPELINE_STATE_H_
#define REX_GRAPHICS_NR_PIPELINE_STATE_H_

#include <cstdint>

// [NR-PSO] THE STATE MIRROR: native-renderer phase 5, increment 5-1.
//
// Phase 4 closed the INPUT question: every register a draw reads is
// recoverable from the depth-1 buffer stream (diverge=0 over the whole file at
// full city load), and 4d-4f proved it by consumption -- every draw and
// resolve in the game now runs from the walk-maintained replay file, pixel
// identical, at baseline speed. What phase 5 replaces is what happens BEHIND
// that seam: today the recovered registers are handed to Xenia's emulated
// pipeline, which re-derives D3D12 state from them. The native renderer has to
// do that derivation itself.
//
// This unit is that derivation, and nothing else. It reads the replay register
// file and produces the packed pipeline description a D3D12 PSO is keyed on --
// topology, cull, fill, depth bias, depth/stencil, per-render-target blend and
// write mask, MSAA. It deliberately does NOT reuse Xenia's draw_util or its
// reg:: structs: those are the ground truth this is checked AGAINST, so
// calling them would make the check a tautology. Every field offset and every
// mapping rule here is transcribed independently and then measured.
//
// The gate is exact and mechanical: per draw, byte-compare our description
// against the one the emulated PipelineCache derived from the same register
// file at the same moment. mismatch must be 0 at city load. A transcription
// slip in a bit offset, a blend-factor row, or a corner of the cull/fill logic
// cannot survive that -- which is the point of doing it this way rather than
// reading the two implementations side by side and believing they agree.
//
// What is NOT derived here, and why:
//
//   - host primitive type / tessellation / index reset come from the primitive
//     processor, render target formats and bound bits from the render target
//     cache, shader hashes and modifications from the translations. Those are
//     separate subsystems that increments 5-2 and 5-3 peel; they are passed in
//     so this increment measures exactly one thing.
//   - the root signature and the geometry shader are host objects, not part of
//     the packed key, so they are not mirrored.
//
// Pure functions over caller-owned structs: no globals, no threads, no SDK
// dependencies, so tools/nr-pipeline-state-test.cpp builds this file bare.

namespace rex {
namespace graphics {
namespace nr {

// ---------------------------------------------------------------------------
// Guest register indices, transcribed from register_table.inc.
// ---------------------------------------------------------------------------
enum : uint32_t {
  kNrRegRbSurfaceInfo = 0x2000,
  kNrRegRbDepthInfo = 0x2002,
  kNrRegRbColorMask = 0x2104,
  kNrRegRbStencilRefMaskBf = 0x210C,
  kNrRegRbStencilRefMask = 0x210D,
  kNrRegSqProgramCntl = 0x2180,
  kNrRegVgtDrawInitiator = 0x21FC,
  kNrRegRbDepthControl = 0x2200,
  kNrRegPaClClipCntl = 0x2204,
  kNrRegPaSuScModeCntl = 0x2205,
  kNrRegRbModeControl = 0x2208,
  kNrRegVgtOutputPathCntl = 0x2284,
  kNrRegPaSuPolyOffsetFrontScale = 0x2380,
  kNrRegPaSuPolyOffsetFrontOffset = 0x2381,
  kNrRegPaSuPolyOffsetBackScale = 0x2382,
  kNrRegPaSuPolyOffsetBackOffset = 0x2383,
};
// RB_COLOR[0-3]_INFO and RB_BLENDCONTROL[0-3] are not contiguous.
extern const uint32_t kNrRegRbColorInfo[4];
extern const uint32_t kNrRegRbBlendControl[4];

// ---------------------------------------------------------------------------
// The packed description. BYTE-LAYOUT MIRROR of the D3D12 back end's private
// PipelineCache::PipelineDescription: field order, widths and packing are
// identical so a memcmp is the whole gate. The consumer static_asserts the two
// sizes against each other; this header stays free of SDK includes so the unit
// test can build bare.
//
// Enum-valued fields are plain uint32_t here with the value sets below, rather
// than typed enums, so that this header depends on nothing. The values are
// transcribed from the back end's enums and from xenos.h.
// ---------------------------------------------------------------------------

// PipelineStripCutIndex.
enum : uint32_t {
  kNrPsoStripCutNone = 0,
  kNrPsoStripCutFFFF = 1,
  kNrPsoStripCutFFFFFFFF = 2,
};
// PipelinePrimitiveTopologyType (occupies the same field as
// xenos::TessellationMode for a tessellated draw).
enum : uint32_t {
  kNrPsoTopologyPoint = 0,
  kNrPsoTopologyLine = 1,
  kNrPsoTopologyTriangle = 2,
};
// PipelineGeometryShader.
enum : uint32_t {
  kNrPsoGsNone = 0,
  kNrPsoGsPointList = 1,
  kNrPsoGsRectangleList = 2,
  kNrPsoGsQuadList = 3,
};
// PipelineCullMode.
enum : uint32_t {
  kNrPsoCullNone = 0,
  kNrPsoCullFront = 1,
  kNrPsoCullBack = 2,
  kNrPsoCullDisableRasterization = 3,
};
// PipelineBlendFactor. Note this is the HOST-side condensed set, not
// xenos::BlendFactor -- the guest's 0..16 sparse encoding is mapped onto it.
enum : uint32_t {
  kNrPsoBlendZero = 0,
  kNrPsoBlendOne = 1,
  kNrPsoBlendSrcColor = 2,
  kNrPsoBlendInvSrcColor = 3,
  kNrPsoBlendSrcAlpha = 4,
  kNrPsoBlendInvSrcAlpha = 5,
  kNrPsoBlendDestColor = 6,
  kNrPsoBlendInvDestColor = 7,
  kNrPsoBlendDestAlpha = 8,
  kNrPsoBlendInvDestAlpha = 9,
  kNrPsoBlendBlendFactor = 10,
  kNrPsoBlendInvBlendFactor = 11,
  kNrPsoBlendSrcAlphaSat = 12,
};
// xenos::CompareFunction.
enum : uint32_t { kNrCompareNever = 0, kNrCompareAlways = 7 };
// xenos::BlendOp.
enum : uint32_t { kNrBlendOpAdd = 0 };
// xenos::MsaaSamples.
enum : uint32_t { kNrMsaa1X = 0, kNrMsaa2X = 1, kNrMsaa4X = 2 };
// xenos::EdramMode.
enum : uint32_t {
  kNrEdramNoOperation = 0,
  kNrEdramColorDepth = 4,
  kNrEdramDepthOnly = 5,
  kNrEdramCopy = 6,
};
// xenos::PolygonType.
enum : uint32_t { kNrPolygonTypeTriangles = 2 };
// xenos::PolygonModeEnable.
enum : uint32_t { kNrPolygonModeDual = 1 };
// xenos::VGTOutputPath.
enum : uint32_t { kNrVgtOutputTessellationEnable = 1 };
// xenos::VertexShaderExportMode.
enum : uint32_t { kNrVsExportMultipass = 7 };
// xenos::PrimitiveType (only the ones the topology mapping names).
enum : uint32_t {
  kNrPrimPointList = 0x01,
  kNrPrimLineList = 0x02,
  kNrPrimLineStrip = 0x03,
  kNrPrimTriangleList = 0x04,
  kNrPrimTriangleFan = 0x05,
  kNrPrimTriangleStrip = 0x06,
  kNrPrimTriangleWithWFlags = 0x07,
  kNrPrimRectangleList = 0x08,
  kNrPrimQuadList = 0x0D,
  kNrPrimQuadStrip = 0x0E,
  kNrPrimPolygon = 0x0F,
  kNrPrimTrianglePatch = 0x11,
  kNrPrimQuadPatch = 0x12,
  kNrPrim2DLineStrip = 0x15,
};

#pragma pack(push, 1)
struct NrPsoRenderTarget {
  uint32_t used : 1;               // 1
  uint32_t format : 4;             // 5   xenos::ColorRenderTargetFormat
  uint32_t src_blend : 4;          // 9
  uint32_t dest_blend : 4;         // 13
  uint32_t blend_op : 3;           // 16  xenos::BlendOp
  uint32_t src_blend_alpha : 4;    // 20
  uint32_t dest_blend_alpha : 4;   // 24
  uint32_t blend_op_alpha : 3;     // 27
  uint32_t write_mask : 4;         // 31
};

struct NrPsoDesc {
  uint64_t vertex_shader_hash;
  uint64_t vertex_shader_modification;
  // 0 when drawing without a pixel shader.
  uint64_t pixel_shader_hash;
  uint64_t pixel_shader_modification;

  int32_t depth_bias;
  float depth_bias_slope_scaled;

  uint32_t strip_cut_index : 2;                                // 2
  uint32_t primitive_topology_type_or_tessellation_mode : 2;   // 4
  uint32_t geometry_shader : 2;                                // 6
  uint32_t fill_mode_wireframe : 1;                            // 7
  uint32_t cull_mode : 2;                                      // 9
  uint32_t front_counter_clockwise : 1;                        // 10
  uint32_t depth_clip : 1;                                     // 11
  uint32_t host_msaa_samples : 2;                              // 13
  uint32_t depth_format : 1;                                   // 14
  uint32_t depth_func : 3;                                     // 17
  uint32_t depth_write : 1;                                    // 18
  uint32_t stencil_enable : 1;                                 // 19
  uint32_t stencil_read_mask : 8;                              // 27

  uint32_t stencil_write_mask : 8;                             // 8
  uint32_t stencil_front_fail_op : 3;                          // 11
  uint32_t stencil_front_depth_fail_op : 3;                    // 14
  uint32_t stencil_front_pass_op : 3;                          // 17
  uint32_t stencil_front_func : 3;                             // 20
  uint32_t stencil_back_fail_op : 3;                           // 23
  uint32_t stencil_back_depth_fail_op : 3;                     // 26
  uint32_t stencil_back_pass_op : 3;                           // 29
  uint32_t stencil_back_func : 3;                              // 32

  NrPsoRenderTarget render_targets[4];
};
#pragma pack(pop)

static_assert(sizeof(NrPsoRenderTarget) == 4, "render target entry must be one dword");
static_assert(sizeof(NrPsoDesc) == 64, "description layout drifted from the back end's");

// ---------------------------------------------------------------------------
// Inputs the derivation cannot obtain from the register file yet. Each one is
// a subsystem a later increment peels; passing them in keeps 5-1 measuring the
// register mapping alone.
// ---------------------------------------------------------------------------
struct NrPsoInputs {
  // The replay register file's flat value array (RegisterFile::values). Read
  // as a flat array rather than through an accessor for the same reason the
  // 4c shadow does: this runs per draw on the command-processor thread, which
  // Ch.9 measured as the frame's bottleneck.
  const uint32_t* regs;

  // --- primitive processor (increment 5-3) ---
  bool tessellated;
  uint32_t tessellation_mode;  // xenos::TessellationMode
  uint32_t host_primitive_type;
  bool host_primitive_reset_enabled;
  bool host_index_format_32;  // xenos::IndexFormat::kInt32

  // --- render target cache (increment 5-3) ---
  // Bit 0 = depth, bits 1..4 = colour 0..3.
  uint32_t bound_rt_bits;
  // [0] = xenos::DepthRenderTargetFormat, [1..4] = ColorRenderTargetFormat.
  const uint32_t* bound_rt_formats;
  // Pixel-shader-interlock (ROV) path: depth/stencil, blending and the depth
  // bias are all done in the shader, so none of them enter the key.
  bool edram_rov_used;

  // --- shader translations (increment 5-2) ---
  uint64_t vertex_shader_hash;
  uint64_t vertex_shader_modification;
  bool has_pixel_shader;
  uint64_t pixel_shader_hash;
  uint64_t pixel_shader_modification;
  // Shader::writes_color_targets(), needed by the colour mask normalization.
  uint32_t pixel_shader_writes_color_targets;

  // d3d12_tessellation_wireframe.
  bool tessellation_wireframe;
};

// ---------------------------------------------------------------------------
// The derivation, and the register-level rules it is built from. The helpers
// are exposed because they are individually testable and individually wrong-
// able: each one is a place a transcription slip can hide.
// ---------------------------------------------------------------------------

// Whether the primitive has faces, which is what makes culling, per-face fill
// mode and per-face polygon offset meaningful.
bool NrPsoIsPrimitivePolygonal(const uint32_t* regs);

// Whether any sample can be rasterized at all. When false the description
// collapses to "rasterization disabled" and nothing after it matters.
bool NrPsoIsRasterizationPotentiallyDone(const uint32_t* regs, bool primitive_polygonal);

// RB_DEPTHCONTROL with the modes that cannot affect the result folded away, so
// that draws not actually using depth do not key distinct pipelines.
uint32_t NrPsoNormalizedDepthControl(const uint32_t* regs);

// RB_COLOR_MASK restricted to targets the pixel shader writes and to
// components the format has, with non-existent components forced on.
uint32_t NrPsoNormalizedColorMask(const uint32_t* regs,
                                  uint32_t pixel_shader_writes_color_targets);

// The polygon offset of the face that will actually be drawn (Direct3D 12 has
// no per-side depth bias).
void NrPsoPreferredFacePolygonOffset(const uint32_t* regs, bool primitive_polygonal,
                                     float* scale_out, float* offset_out);

// Guest absolute depth bias -> Direct3D 10+ relative integer depth bias.
int32_t NrPsoD3D10IntegerPolygonOffset(uint32_t depth_format, float polygon_offset);

// Builds the description. Returns false exactly where the emulated derivation
// does: rasterization disabled while a pixel shader is still bound, which the
// caller is required to have resolved. `out` is fully zeroed first, including
// the padding, so the result is memcmp-able.
bool NrPsoDerive(const NrPsoInputs& in, NrPsoDesc* out);

// ---------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------
enum NrPsoField : uint32_t {
  kNrPsoFieldVertexShaderHash = 0,
  kNrPsoFieldVertexShaderModification,
  kNrPsoFieldPixelShaderHash,
  kNrPsoFieldPixelShaderModification,
  kNrPsoFieldDepthBias,
  kNrPsoFieldDepthBiasSlopeScaled,
  kNrPsoFieldStripCutIndex,
  kNrPsoFieldPrimitiveTopologyType,
  kNrPsoFieldGeometryShader,
  kNrPsoFieldFillModeWireframe,
  kNrPsoFieldCullMode,
  kNrPsoFieldFrontCounterClockwise,
  kNrPsoFieldDepthClip,
  kNrPsoFieldHostMsaaSamples,
  kNrPsoFieldDepthFormat,
  kNrPsoFieldDepthFunc,
  kNrPsoFieldDepthWrite,
  kNrPsoFieldStencilEnable,
  kNrPsoFieldStencilReadMask,
  kNrPsoFieldStencilWriteMask,
  kNrPsoFieldStencilFrontFailOp,
  kNrPsoFieldStencilFrontDepthFailOp,
  kNrPsoFieldStencilFrontPassOp,
  kNrPsoFieldStencilFrontFunc,
  kNrPsoFieldStencilBackFailOp,
  kNrPsoFieldStencilBackDepthFailOp,
  kNrPsoFieldStencilBackPassOp,
  kNrPsoFieldStencilBackFunc,
  // Render target fields follow, 9 per target, in declaration order.
  kNrPsoFieldRenderTarget0,
  kNrPsoFieldCount = kNrPsoFieldRenderTarget0 + 4 * 9,
};
constexpr uint32_t kNrPsoRenderTargetFieldCount = 9;

// Names the field, including "rt2.dest_blend" style names for the render
// target entries. Writes into a caller buffer rather than returning a pointer
// to shared storage, so a sample can be held and formatted later.
void NrPsoFieldName(uint32_t field, char* out, uint32_t out_size);

// Returns kNrPsoFieldCount when the two descriptions are equal, otherwise the
// FIRST differing field, with both values widened to uint64 (the float field
// is returned by its bit pattern, so a signalling difference is still visible).
//
// This is field-wise rather than a memcmp so that the padding bits the two
// layouts may or may not share can never be reported as a state difference,
// and so that a difference is named. The consumer memcmps as well: the two
// disagreeing is itself a finding (it would mean the layouts differ, not the
// state).
uint32_t NrPsoFirstDifference(const NrPsoDesc& ours, const NrPsoDesc& theirs,
                              uint64_t* ours_out, uint64_t* theirs_out);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_PIPELINE_STATE_H_
