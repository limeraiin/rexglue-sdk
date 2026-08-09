/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <rex/graphics/nr_native_pso.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rex {
namespace graphics {
namespace nr {

// ---------------------------------------------------------------------------
// Guest enum values this mapping names. Transcribed, like the rest of the
// native-renderer units, rather than included from the back end.
// ---------------------------------------------------------------------------
enum : uint32_t {
  kColor_8_8_8_8 = 0,
  kColor_8_8_8_8_GAMMA = 1,
  kColor_2_10_10_10 = 2,
  kColor_2_10_10_10_FLOAT = 3,
  kColor_16_16 = 4,
  kColor_16_16_16_16 = 5,
  kColor_16_16_FLOAT = 6,
  kColor_16_16_16_16_FLOAT = 7,
  kColor_2_10_10_10_AS_10_10_10_10 = 10,
  kColor_2_10_10_10_FLOAT_AS_16_16_16_16 = 12,
  kColor_32_FLOAT = 14,
  kColor_32_32_FLOAT = 15,
};
enum : uint32_t { kDepthD24S8 = 0, kDepthD24FS8 = 1 };

const char* NrNpsoStatusName(uint32_t status) {
  switch (status) {
    case kNrNpsoOk:
      return "ok";
    case kNrNpsoTessellation:
      return "tessellation";
    case kNrNpsoRov:
      return "rov";
    case kNrNpsoBadTopology:
      return "topology";
    case kNrNpsoBadColorFormat:
      return "color_format";
    case kNrNpsoBadDepthFormat:
      return "depth_format";
    case kNrNpsoBadMsaa:
      return "msaa";
    case kNrNpsoNoShaderBytes:
      return "no_shader_bytes";
    case kNrNpsoCacheFull:
      return "cache_full";
    case kNrNpsoCreateFailed:
      return "create_failed";
    default:
      return "?";
  }
}

// ---------------------------------------------------------------------------
// Formats.
// ---------------------------------------------------------------------------
DXGI_FORMAT NrNpsoColorDrawFormat(uint32_t color_format, bool gamma_as_unorm16) {
  // The drawing view is TYPED wherever a typed format can represent the guest
  // format exactly. The typeless resource formats the ownership-transfer path
  // uses are deliberately not reachable from here: binding a typeless format
  // as a render target is invalid.
  switch (color_format) {
    case kColor_8_8_8_8:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
    case kColor_8_8_8_8_GAMMA:
      return gamma_as_unorm16 ? DXGI_FORMAT_R16G16B16A16_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
    case kColor_2_10_10_10:
    case kColor_2_10_10_10_AS_10_10_10_10:
      return DXGI_FORMAT_R10G10B10A2_UNORM;
    case kColor_2_10_10_10_FLOAT:
    case kColor_2_10_10_10_FLOAT_AS_16_16_16_16:
      return DXGI_FORMAT_R16G16B16A16_FLOAT;
    // The four below are stored typeless (SNORM has two encodings of -1, and
    // float needs NaN propagation through ownership transfers), so the draw
    // view names the type the shader writes.
    case kColor_16_16:
      return DXGI_FORMAT_R16G16_SNORM;
    case kColor_16_16_16_16:
      return DXGI_FORMAT_R16G16B16A16_SNORM;
    case kColor_16_16_FLOAT:
      return DXGI_FORMAT_R16G16_FLOAT;
    case kColor_16_16_16_16_FLOAT:
      return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case kColor_32_FLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case kColor_32_32_FLOAT:
      return DXGI_FORMAT_R32G32_FLOAT;
    default:
      return DXGI_FORMAT_UNKNOWN;
  }
}

DXGI_FORMAT NrNpsoDepthDsvFormat(uint32_t depth_format) {
  switch (depth_format) {
    case kDepthD24S8:
      return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case kDepthD24FS8:
      // Guest 24-bit float depth is kept as host 32-bit float, converted where
      // the guest would see it, so this is NOT a 24-bit unorm format.
      return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    default:
      return DXGI_FORMAT_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// The description -> D3D12_GRAPHICS_PIPELINE_STATE_DESC mapping.
// ---------------------------------------------------------------------------
uint32_t NrNpsoBuildStateDesc(const NrPsoDesc& desc, const NrNpsoEnv& env,
                              const NrNpsoBlobs& blobs,
                              D3D12_GRAPHICS_PIPELINE_STATE_DESC* out) {
  std::memset(out, 0, sizeof(*out));

  if (env.edram_rov_used) {
    // Rasterization under pixel-shader interlock puts depth, stencil and
    // blending inside the pixel shader and drives a forced sample count. A
    // different model; not built here.
    return kNrNpsoRov;
  }

  out->pRootSignature = blobs.root_signature;

  // Index buffer strip cut.
  switch (desc.strip_cut_index) {
    case kNrPsoStripCutFFFF:
      out->IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
      break;
    case kNrPsoStripCutFFFFFFFF:
      out->IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
      break;
    default:
      out->IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
      break;
  }

  // Vertex shader and topology type. A tessellated draw puts the guest vertex
  // shader in the DOMAIN stage behind one of the prebuilt hull shaders; that
  // pairing is host-authored, so it is out of scope here.
  if (!blobs.vs || !blobs.vs_size) {
    std::memset(out, 0, sizeof(*out));
    return kNrNpsoNoShaderBytes;
  }
  out->VS.pShaderBytecode = blobs.vs;
  out->VS.BytecodeLength = blobs.vs_size;
  switch (desc.primitive_topology_type_or_tessellation_mode) {
    case kNrPsoTopologyPoint:
      out->PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
      break;
    case kNrPsoTopologyLine:
      out->PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
      break;
    case kNrPsoTopologyTriangle:
      out->PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      break;
    default:
      std::memset(out, 0, sizeof(*out));
      return kNrNpsoBadTopology;
  }

  // Pixel shader. With no guest pixel shader there is still one case that
  // needs host code: writing guest 24-bit float depth from a depth-only pass,
  // where the host depth buffer holds a value the guest would have rounded.
  if (blobs.ps && blobs.ps_size) {
    out->PS.pShaderBytecode = blobs.ps;
    out->PS.BytecodeLength = blobs.ps_size;
  } else if (env.depth_float24_convert_in_pixel_shader &&
             (desc.depth_func != kNrCompareAlways || desc.depth_write) &&
             desc.depth_format == kDepthD24FS8) {
    if (env.depth_float24_round) {
      out->PS.pShaderBytecode = blobs.float24_round_ps;
      out->PS.BytecodeLength = blobs.float24_round_ps_size;
    } else {
      out->PS.pShaderBytecode = blobs.float24_truncate_ps;
      out->PS.BytecodeLength = blobs.float24_truncate_ps_size;
    }
  }

  // Geometry shader, for the guest primitive types Direct3D has no equivalent
  // of (point sprites, rectangle and quad lists).
  if (blobs.gs && blobs.gs_size) {
    out->GS.pShaderBytecode = blobs.gs;
    out->GS.BytecodeLength = blobs.gs_size;
  }

  // Rasterizer.
  out->RasterizerState.FillMode =
      desc.fill_mode_wireframe ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
  switch (desc.cull_mode) {
    case kNrPsoCullFront:
      out->RasterizerState.CullMode = D3D12_CULL_MODE_FRONT;
      break;
    case kNrPsoCullBack:
      out->RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
      break;
    default:
      // kNone, and kDisableRasterization -- which is expressed further down by
      // removing the pixel shader and depth/stencil, not by a cull mode.
      out->RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
      break;
  }
  out->RasterizerState.FrontCounterClockwise = desc.front_counter_clockwise ? TRUE : FALSE;
  out->RasterizerState.DepthBias = desc.depth_bias;
  out->RasterizerState.DepthBiasClamp = 0.0f;
  // Slope-scaled bias is in host pixels, and a scaled render target has more
  // of them per guest pixel. With a non-square scale the larger axis is used:
  // over-biasing costs precision, under-biasing costs Z-fighting.
  out->RasterizerState.SlopeScaledDepthBias =
      desc.depth_bias_slope_scaled *
      float(std::max(env.draw_resolution_scale_x, env.draw_resolution_scale_y));
  out->RasterizerState.DepthClipEnable = desc.depth_clip ? TRUE : FALSE;

  // Sample mask and count.
  const uint32_t msaa_sample_count = uint32_t(1) << desc.host_msaa_samples;
  out->SampleMask = UINT_MAX;
  if (msaa_sample_count > 4) {
    std::memset(out, 0, sizeof(*out));
    return kNrNpsoBadMsaa;
  }
  if (msaa_sample_count == 2 && !env.msaa_2x_supported) {
    // 2x is emulated on 4x hardware by keeping samples 0 and 3, which are the
    // top-left and bottom-right of the 4x pattern -- not the guest's two
    // positions, but the same coverage split.
    out->SampleMask = 0b1001;
    out->SampleDesc.Count = 4;
  } else {
    out->SampleDesc.Count = msaa_sample_count;
  }

  // Depth and stencil. A pipeline that neither tests nor writes depth leaves
  // the whole depth stage off, which is also what keeps such draws from
  // keying distinct pipelines.
  if (desc.depth_func != kNrCompareAlways || desc.depth_write) {
    out->DepthStencilState.DepthEnable = TRUE;
    out->DepthStencilState.DepthWriteMask =
        desc.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    // The guest comparison function encoding is Direct3D's minus one (bit 0
    // less, bit 1 equal, bit 2 greater; 0 never, 7 always).
    out->DepthStencilState.DepthFunc =
        D3D12_COMPARISON_FUNC(uint32_t(D3D12_COMPARISON_FUNC_NEVER) + desc.depth_func);
  }
  if (desc.stencil_enable) {
    out->DepthStencilState.StencilEnable = TRUE;
    out->DepthStencilState.StencilReadMask = UINT8(desc.stencil_read_mask);
    out->DepthStencilState.StencilWriteMask = UINT8(desc.stencil_write_mask);
    // Stencil operations are Direct3D's minus one as well.
    out->DepthStencilState.FrontFace.StencilFailOp =
        D3D12_STENCIL_OP(uint32_t(D3D12_STENCIL_OP_KEEP) + desc.stencil_front_fail_op);
    out->DepthStencilState.FrontFace.StencilDepthFailOp =
        D3D12_STENCIL_OP(uint32_t(D3D12_STENCIL_OP_KEEP) + desc.stencil_front_depth_fail_op);
    out->DepthStencilState.FrontFace.StencilPassOp =
        D3D12_STENCIL_OP(uint32_t(D3D12_STENCIL_OP_KEEP) + desc.stencil_front_pass_op);
    out->DepthStencilState.FrontFace.StencilFunc =
        D3D12_COMPARISON_FUNC(uint32_t(D3D12_COMPARISON_FUNC_NEVER) + desc.stencil_front_func);
    out->DepthStencilState.BackFace.StencilFailOp =
        D3D12_STENCIL_OP(uint32_t(D3D12_STENCIL_OP_KEEP) + desc.stencil_back_fail_op);
    out->DepthStencilState.BackFace.StencilDepthFailOp =
        D3D12_STENCIL_OP(uint32_t(D3D12_STENCIL_OP_KEEP) + desc.stencil_back_depth_fail_op);
    out->DepthStencilState.BackFace.StencilPassOp =
        D3D12_STENCIL_OP(uint32_t(D3D12_STENCIL_OP_KEEP) + desc.stencil_back_pass_op);
    out->DepthStencilState.BackFace.StencilFunc =
        D3D12_COMPARISON_FUNC(uint32_t(D3D12_COMPARISON_FUNC_NEVER) + desc.stencil_back_func);
  }
  if (out->DepthStencilState.DepthEnable || out->DepthStencilState.StencilEnable) {
    out->DSVFormat = NrNpsoDepthDsvFormat(desc.depth_format);
    if (out->DSVFormat == DXGI_FORMAT_UNKNOWN) {
      std::memset(out, 0, sizeof(*out));
      return kNrNpsoBadDepthFormat;
    }
  }

  // Render targets and blending. Independent blending is always on: the guest
  // has per-target blend state and the description carries it per target.
  static const D3D12_BLEND kBlendFactorMap[] = {
      D3D12_BLEND_ZERO,           D3D12_BLEND_ONE,
      D3D12_BLEND_SRC_COLOR,      D3D12_BLEND_INV_SRC_COLOR,
      D3D12_BLEND_SRC_ALPHA,      D3D12_BLEND_INV_SRC_ALPHA,
      D3D12_BLEND_DEST_COLOR,     D3D12_BLEND_INV_DEST_COLOR,
      D3D12_BLEND_DEST_ALPHA,     D3D12_BLEND_INV_DEST_ALPHA,
      D3D12_BLEND_BLEND_FACTOR,   D3D12_BLEND_INV_BLEND_FACTOR,
      D3D12_BLEND_SRC_ALPHA_SAT,
  };
  // Three bits reach this from the guest, so all eight values must map.
  static const D3D12_BLEND_OP kBlendOpMap[] = {
      D3D12_BLEND_OP_ADD, D3D12_BLEND_OP_SUBTRACT,     D3D12_BLEND_OP_MIN,
      D3D12_BLEND_OP_MAX, D3D12_BLEND_OP_REV_SUBTRACT, D3D12_BLEND_OP_ADD,
      D3D12_BLEND_OP_ADD, D3D12_BLEND_OP_ADD};
  out->BlendState.IndependentBlendEnable = TRUE;
  for (uint32_t i = 0; i < 4; ++i) {
    const NrPsoRenderTarget& rt = desc.render_targets[i];
    if (!rt.used) {
      // An unused slot is described as UNKNOWN, which is what lets a null
      // render target view be bound there.
      out->RTVFormats[i] = DXGI_FORMAT_UNKNOWN;
      continue;
    }
    out->NumRenderTargets = i + 1;
    out->RTVFormats[i] = NrNpsoColorDrawFormat(rt.format, env.gamma_render_target_as_unorm16);
    if (out->RTVFormats[i] == DXGI_FORMAT_UNKNOWN) {
      std::memset(out, 0, sizeof(*out));
      return kNrNpsoBadColorFormat;
    }
    D3D12_RENDER_TARGET_BLEND_DESC& blend = out->BlendState.RenderTarget[i];
    // Blending is only enabled where it would do something: src ONE, dest
    // ZERO, ADD on both colour and alpha is exactly a copy.
    if (rt.src_blend != kNrPsoBlendOne || rt.dest_blend != kNrPsoBlendZero ||
        rt.blend_op != kNrBlendOpAdd || rt.src_blend_alpha != kNrPsoBlendOne ||
        rt.dest_blend_alpha != kNrPsoBlendZero || rt.blend_op_alpha != kNrBlendOpAdd) {
      blend.BlendEnable = TRUE;
      blend.SrcBlend = kBlendFactorMap[rt.src_blend];
      blend.DestBlend = kBlendFactorMap[rt.dest_blend];
      blend.BlendOp = kBlendOpMap[rt.blend_op];
      blend.SrcBlendAlpha = kBlendFactorMap[rt.src_blend_alpha];
      blend.DestBlendAlpha = kBlendFactorMap[rt.dest_blend_alpha];
      blend.BlendOpAlpha = kBlendOpMap[rt.blend_op_alpha];
    }
    blend.RenderTargetWriteMask = UINT8(rt.write_mask);
  }

  // Rasterization disabled. Direct3D has no switch for it, so it is expressed
  // the way the API intends: no pixel shader and no depth or stencil. Every
  // other field that would have been made irrelevant by this was already
  // folded away when the description was derived.
  if (desc.cull_mode == kNrPsoCullDisableRasterization) {
    out->PS.pShaderBytecode = nullptr;
    out->PS.BytecodeLength = 0;
    out->DepthStencilState.DepthEnable = FALSE;
    out->DepthStencilState.StencilEnable = FALSE;
  }

  return kNrNpsoOk;
}

// ---------------------------------------------------------------------------
// Comparison.
// ---------------------------------------------------------------------------
namespace {

uint32_t CompareBytecode(const D3D12_SHADER_BYTECODE& ours, const D3D12_SHADER_BYTECODE& theirs,
                         uint32_t* offset_out) {
  const bool ours_present = ours.pShaderBytecode != nullptr && ours.BytecodeLength != 0;
  const bool theirs_present = theirs.pShaderBytecode != nullptr && theirs.BytecodeLength != 0;
  if (ours_present != theirs_present) {
    return kNrNpsoCmpBytecodePresence;
  }
  if (!ours_present) {
    return kNrNpsoCmpEqual;
  }
  if (ours.BytecodeLength != theirs.BytecodeLength) {
    return kNrNpsoCmpBytecodeSize;
  }
  const uint8_t* a = reinterpret_cast<const uint8_t*>(ours.pShaderBytecode);
  const uint8_t* b = reinterpret_cast<const uint8_t*>(theirs.pShaderBytecode);
  for (size_t i = 0; i < ours.BytecodeLength; ++i) {
    if (a[i] != b[i]) {
      *offset_out = uint32_t(i);
      return kNrNpsoCmpBytecodeBytes;
    }
  }
  return kNrNpsoCmpEqual;
}

}  // namespace

uint32_t NrNpsoCompareStateDesc(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& ours,
                                const D3D12_GRAPHICS_PIPELINE_STATE_DESC& theirs,
                                uint32_t* stage_out, uint32_t* offset_out) {
  *stage_out = 0;
  *offset_out = 0;

  // The five bytecode members hold pointers into different buffers by
  // construction, so they are compared by content and then excluded from the
  // structural comparison below.
  D3D12_SHADER_BYTECODE D3D12_GRAPHICS_PIPELINE_STATE_DESC::*const kStages[5] = {
      &D3D12_GRAPHICS_PIPELINE_STATE_DESC::VS, &D3D12_GRAPHICS_PIPELINE_STATE_DESC::PS,
      &D3D12_GRAPHICS_PIPELINE_STATE_DESC::DS, &D3D12_GRAPHICS_PIPELINE_STATE_DESC::HS,
      &D3D12_GRAPHICS_PIPELINE_STATE_DESC::GS};
  for (uint32_t stage = 0; stage < 5; ++stage) {
    uint32_t offset = 0;
    const uint32_t verdict = CompareBytecode(ours.*kStages[stage], theirs.*kStages[stage], &offset);
    if (verdict != kNrNpsoCmpEqual) {
      *stage_out = stage;
      *offset_out = offset;
      return verdict;
    }
  }

  // Everything else, byte for byte. Both sides were zeroed before filling, so
  // padding compares equal and a byte difference is a real state difference.
  D3D12_GRAPHICS_PIPELINE_STATE_DESC a = ours;
  D3D12_GRAPHICS_PIPELINE_STATE_DESC b = theirs;
  for (uint32_t stage = 0; stage < 5; ++stage) {
    std::memset(&(a.*kStages[stage]), 0, sizeof(D3D12_SHADER_BYTECODE));
    std::memset(&(b.*kStages[stage]), 0, sizeof(D3D12_SHADER_BYTECODE));
  }
  const uint8_t* pa = reinterpret_cast<const uint8_t*>(&a);
  const uint8_t* pb = reinterpret_cast<const uint8_t*>(&b);
  for (size_t i = 0; i < sizeof(a); ++i) {
    if (pa[i] != pb[i]) {
      *offset_out = uint32_t(i);
      return kNrNpsoCmpField;
    }
  }
  return kNrNpsoCmpEqual;
}

void NrNpsoFieldName(uint32_t byte_offset, char* out, uint32_t out_size) {
  if (!out || !out_size) {
    return;
  }
  using Desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC;
  const uint32_t o = byte_offset;

  auto in_range = [o](size_t begin, size_t size) { return o >= begin && o < begin + size; };
  auto put = [out, out_size](const char* name) { std::snprintf(out, out_size, "%s", name); };

  if (in_range(offsetof(Desc, pRootSignature), sizeof(void*))) {
    put("pRootSignature");
    return;
  }
  if (in_range(offsetof(Desc, StreamOutput), sizeof(D3D12_STREAM_OUTPUT_DESC))) {
    put("StreamOutput");
    return;
  }
  if (in_range(offsetof(Desc, BlendState), sizeof(D3D12_BLEND_DESC))) {
    const uint32_t in_blend = o - uint32_t(offsetof(Desc, BlendState));
    if (in_blend < offsetof(D3D12_BLEND_DESC, RenderTarget)) {
      put(in_blend < sizeof(BOOL) ? "BlendState.AlphaToCoverageEnable"
                                  : "BlendState.IndependentBlendEnable");
      return;
    }
    const uint32_t in_rts = in_blend - uint32_t(offsetof(D3D12_BLEND_DESC, RenderTarget));
    const uint32_t index = in_rts / uint32_t(sizeof(D3D12_RENDER_TARGET_BLEND_DESC));
    const uint32_t in_rt = in_rts % uint32_t(sizeof(D3D12_RENDER_TARGET_BLEND_DESC));
    using Rt = D3D12_RENDER_TARGET_BLEND_DESC;
    const char* member = "?";
    if (in_rt < offsetof(Rt, LogicOpEnable)) {
      member = "BlendEnable";
    } else if (in_rt < offsetof(Rt, SrcBlend)) {
      member = "LogicOpEnable";
    } else if (in_rt < offsetof(Rt, DestBlend)) {
      member = "SrcBlend";
    } else if (in_rt < offsetof(Rt, BlendOp)) {
      member = "DestBlend";
    } else if (in_rt < offsetof(Rt, SrcBlendAlpha)) {
      member = "BlendOp";
    } else if (in_rt < offsetof(Rt, DestBlendAlpha)) {
      member = "SrcBlendAlpha";
    } else if (in_rt < offsetof(Rt, BlendOpAlpha)) {
      member = "DestBlendAlpha";
    } else if (in_rt < offsetof(Rt, LogicOp)) {
      member = "BlendOpAlpha";
    } else if (in_rt < offsetof(Rt, RenderTargetWriteMask)) {
      member = "LogicOp";
    } else {
      member = "RenderTargetWriteMask";
    }
    std::snprintf(out, out_size, "BlendState.RenderTarget[%u].%s", index, member);
    return;
  }
  if (in_range(offsetof(Desc, SampleMask), sizeof(UINT))) {
    put("SampleMask");
    return;
  }
  if (in_range(offsetof(Desc, RasterizerState), sizeof(D3D12_RASTERIZER_DESC))) {
    using Rs = D3D12_RASTERIZER_DESC;
    const uint32_t r = o - uint32_t(offsetof(Desc, RasterizerState));
    const char* member = "RasterizerState.?";
    if (r < offsetof(Rs, CullMode)) {
      member = "RasterizerState.FillMode";
    } else if (r < offsetof(Rs, FrontCounterClockwise)) {
      member = "RasterizerState.CullMode";
    } else if (r < offsetof(Rs, DepthBias)) {
      member = "RasterizerState.FrontCounterClockwise";
    } else if (r < offsetof(Rs, DepthBiasClamp)) {
      member = "RasterizerState.DepthBias";
    } else if (r < offsetof(Rs, SlopeScaledDepthBias)) {
      member = "RasterizerState.DepthBiasClamp";
    } else if (r < offsetof(Rs, DepthClipEnable)) {
      member = "RasterizerState.SlopeScaledDepthBias";
    } else if (r < offsetof(Rs, MultisampleEnable)) {
      member = "RasterizerState.DepthClipEnable";
    } else if (r < offsetof(Rs, AntialiasedLineEnable)) {
      member = "RasterizerState.MultisampleEnable";
    } else if (r < offsetof(Rs, ForcedSampleCount)) {
      member = "RasterizerState.AntialiasedLineEnable";
    } else if (r < offsetof(Rs, ConservativeRaster)) {
      member = "RasterizerState.ForcedSampleCount";
    } else {
      member = "RasterizerState.ConservativeRaster";
    }
    put(member);
    return;
  }
  if (in_range(offsetof(Desc, DepthStencilState), sizeof(D3D12_DEPTH_STENCIL_DESC))) {
    using Ds = D3D12_DEPTH_STENCIL_DESC;
    const uint32_t d = o - uint32_t(offsetof(Desc, DepthStencilState));
    if (d < offsetof(Ds, DepthWriteMask)) {
      put("DepthStencilState.DepthEnable");
    } else if (d < offsetof(Ds, DepthFunc)) {
      put("DepthStencilState.DepthWriteMask");
    } else if (d < offsetof(Ds, StencilEnable)) {
      put("DepthStencilState.DepthFunc");
    } else if (d < offsetof(Ds, StencilReadMask)) {
      put("DepthStencilState.StencilEnable");
    } else if (d < offsetof(Ds, StencilWriteMask)) {
      put("DepthStencilState.StencilReadMask");
    } else if (d < offsetof(Ds, FrontFace)) {
      put("DepthStencilState.StencilWriteMask");
    } else {
      const bool back = d >= offsetof(Ds, BackFace);
      const uint32_t f =
          d - uint32_t(back ? offsetof(Ds, BackFace) : offsetof(Ds, FrontFace));
      using Op = D3D12_DEPTH_STENCILOP_DESC;
      const char* member = "?";
      if (f < offsetof(Op, StencilDepthFailOp)) {
        member = "StencilFailOp";
      } else if (f < offsetof(Op, StencilPassOp)) {
        member = "StencilDepthFailOp";
      } else if (f < offsetof(Op, StencilFunc)) {
        member = "StencilPassOp";
      } else {
        member = "StencilFunc";
      }
      std::snprintf(out, out_size, "DepthStencilState.%s.%s", back ? "BackFace" : "FrontFace",
                    member);
    }
    return;
  }
  if (in_range(offsetof(Desc, InputLayout), sizeof(D3D12_INPUT_LAYOUT_DESC))) {
    put("InputLayout");
    return;
  }
  if (in_range(offsetof(Desc, IBStripCutValue), sizeof(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE))) {
    put("IBStripCutValue");
    return;
  }
  if (in_range(offsetof(Desc, PrimitiveTopologyType), sizeof(D3D12_PRIMITIVE_TOPOLOGY_TYPE))) {
    put("PrimitiveTopologyType");
    return;
  }
  if (in_range(offsetof(Desc, NumRenderTargets), sizeof(UINT))) {
    put("NumRenderTargets");
    return;
  }
  if (in_range(offsetof(Desc, RTVFormats), sizeof(DXGI_FORMAT) * 8)) {
    const uint32_t index =
        (o - uint32_t(offsetof(Desc, RTVFormats))) / uint32_t(sizeof(DXGI_FORMAT));
    std::snprintf(out, out_size, "RTVFormats[%u]", index);
    return;
  }
  if (in_range(offsetof(Desc, DSVFormat), sizeof(DXGI_FORMAT))) {
    put("DSVFormat");
    return;
  }
  if (in_range(offsetof(Desc, SampleDesc), sizeof(DXGI_SAMPLE_DESC))) {
    put(o < offsetof(Desc, SampleDesc) + sizeof(UINT) ? "SampleDesc.Count"
                                                      : "SampleDesc.Quality");
    return;
  }
  if (in_range(offsetof(Desc, NodeMask), sizeof(UINT))) {
    put("NodeMask");
    return;
  }
  if (in_range(offsetof(Desc, CachedPSO), sizeof(D3D12_CACHED_PIPELINE_STATE))) {
    put("CachedPSO");
    return;
  }
  if (in_range(offsetof(Desc, Flags), sizeof(D3D12_PIPELINE_STATE_FLAGS))) {
    put("Flags");
    return;
  }
  std::snprintf(out, out_size, "byte %u", o);
}

// ---------------------------------------------------------------------------
// The cache.
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kProbeLimit = 32;

uint64_t Mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t DescHash(const NrPsoDesc& desc) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(&desc);
  uint64_t h = 0xCBF29CE484222325ull;
  for (size_t i = 0; i < sizeof(desc); ++i) {
    h = (h ^ p[i]) * 0x100000001B3ull;
  }
  return Mix64(h);
}

struct Cache {
  NrNpsoCreateFn create_fn = nullptr;
  NrNpsoReleaseFn release_fn = nullptr;
  void* ctx = nullptr;
  uint32_t max_entries = 0;

  std::vector<NrNpsoEntry> entries;
  // Open addressing over indices into `entries`; kEmpty means free.
  std::vector<uint32_t> slots;
  uint32_t slot_mask = 0;

  NrNpsoStats stats{};
};

constexpr uint32_t kEmpty = 0xFFFFFFFFu;

Cache& Get() {
  static Cache cache;
  return cache;
}

uint32_t RoundUpPow2(uint32_t v) {
  uint32_t p = 1;
  while (p < v) {
    p <<= 1;
  }
  return p;
}

}  // namespace

void NrNpsoCacheConfigure(NrNpsoCreateFn create_fn, NrNpsoReleaseFn release_fn, void* ctx,
                          uint32_t max_entries) {
  Cache& c = Get();
  NrNpsoCacheReset();
  c.create_fn = create_fn;
  c.release_fn = release_fn;
  c.ctx = ctx;
  c.max_entries = max_entries ? max_entries : 1;
  // Reserved so an entry pointer handed to a draw never moves.
  c.entries.clear();
  c.entries.reserve(c.max_entries);
  // Half load at most, which keeps the probe chain short enough that the
  // overflow counter should stay at zero.
  const uint32_t slot_count = RoundUpPow2(std::max<uint32_t>(c.max_entries * 2, 16));
  c.slots.assign(slot_count, kEmpty);
  c.slot_mask = slot_count - 1;
}

bool NrNpsoCacheConfigured() { return Get().create_fn != nullptr; }

void NrNpsoCacheReset() {
  Cache& c = Get();
  if (c.release_fn) {
    for (NrNpsoEntry& e : c.entries) {
      if (e.state) {
        c.release_fn(c.ctx, e.state);
        e.state = nullptr;
      }
    }
  }
  c.entries.clear();
  std::fill(c.slots.begin(), c.slots.end(), kEmpty);
  c.stats = NrNpsoStats{};
}

NrNpsoEntry* NrNpsoCacheLookup(const NrPsoDesc& desc, const NrNpsoEnv& env,
                               const NrNpsoBlobs& blobs) {
  Cache& c = Get();
  ++c.stats.requests;
  if (!c.create_fn) {
    return nullptr;
  }

  const uint64_t hash = DescHash(desc);
  uint32_t slot = uint32_t(hash) & c.slot_mask;
  uint32_t free_slot = kEmpty;
  for (uint32_t probe = 0; probe < kProbeLimit; ++probe) {
    const uint32_t index = c.slots[slot];
    if (index == kEmpty) {
      free_slot = slot;
      break;
    }
    NrNpsoEntry& e = c.entries[index];
    if (!std::memcmp(&e.desc, &desc, sizeof(desc))) {
      ++c.stats.hits;
      return &e;
    }
    slot = (slot + 1) & c.slot_mask;
  }
  if (free_slot == kEmpty) {
    ++c.stats.probe_ovf;
    ++c.stats.refused;
    ++c.stats.fallbacks;
    ++c.stats.fallback_reason[kNrNpsoCacheFull];
    return nullptr;
  }
  if (c.entries.size() >= c.max_entries) {
    // Refuse rather than evict: a released pipeline could still be bound, and
    // a moved entry would invalidate a pointer a draw is holding.
    ++c.stats.refused;
    ++c.stats.fallbacks;
    ++c.stats.fallback_reason[kNrNpsoCacheFull];
    return nullptr;
  }

  ++c.stats.misses;
  const uint32_t index = uint32_t(c.entries.size());
  c.entries.emplace_back();
  NrNpsoEntry& e = c.entries.back();
  e.desc = desc;
  e.state = nullptr;
  e.verified = false;
  e.agreed = false;
  e.status = NrNpsoBuildStateDesc(desc, env, blobs, &e.state_desc);
  if (e.status == kNrNpsoOk) {
    const auto t0 = std::chrono::steady_clock::now();
    e.state = c.create_fn(c.ctx, e.state_desc);
    const uint64_t ns =
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0)
                     .count());
    c.stats.create_ns_total += ns;
    c.stats.create_ns_max = std::max(c.stats.create_ns_max, ns);
    if (e.state) {
      ++c.stats.created;
    } else {
      ++c.stats.create_fail;
      e.status = kNrNpsoCreateFailed;
    }
  }
  if (e.status != kNrNpsoOk) {
    ++c.stats.fallbacks;
    ++c.stats.fallback_reason[e.status];
  }
  c.slots[free_slot] = index;
  c.stats.entries = uint32_t(c.entries.size());
  if (e.status == kNrNpsoOk) {
    ++c.stats.entries_ok;
  }
  return &e;
}

uint32_t NrNpsoVerify(NrNpsoEntry* entry, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& theirs) {
  Cache& c = Get();
  if (!entry || entry->verified) {
    return kNrNpsoCmpEqual;
  }
  entry->verified = true;
  ++c.stats.verified;
  uint32_t stage = 0;
  uint32_t offset = 0;
  const uint32_t verdict = NrNpsoCompareStateDesc(entry->state_desc, theirs, &stage, &offset);
  if (verdict == kNrNpsoCmpEqual) {
    entry->agreed = true;
    ++c.stats.agreed;
    return verdict;
  }
  ++c.stats.desc_ne;
  if (!c.stats.have_first_ne) {
    c.stats.have_first_ne = true;
    c.stats.first_ne_verdict = verdict;
    c.stats.first_ne_stage = stage;
    c.stats.first_ne_offset = offset;
    c.stats.first_ne_vs_hash = entry->desc.vertex_shader_hash;
    c.stats.first_ne_ps_hash = entry->desc.pixel_shader_hash;
  }
  return verdict;
}

namespace {
// Kept apart from the cache counters: a draw that reused a cached entry and a
// draw that actually bound our pipeline are different questions, and in
// compare-only mode the second one is zero while the first is not.
uint64_t g_bound_ours = 0;
uint64_t g_bound_theirs = 0;
}  // namespace

void NrNpsoCountBind(bool ours) {
  if (ours) {
    ++g_bound_ours;
  } else {
    ++g_bound_theirs;
  }
}

uint64_t NrNpsoBoundOurs() { return g_bound_ours; }
uint64_t NrNpsoBoundTheirs() { return g_bound_theirs; }

const NrNpsoStats& NrNpsoGetStats() { return Get().stats; }

void NrNpsoEndWindow() {
  Cache& c = Get();
  c.stats.requests = 0;
  c.stats.hits = 0;
  c.stats.misses = 0;
  c.stats.fallbacks = 0;
  std::fill(std::begin(c.stats.fallback_reason), std::end(c.stats.fallback_reason), uint64_t(0));
  g_bound_ours = 0;
  g_bound_theirs = 0;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
