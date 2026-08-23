/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <cstdint>

#include <rex/assert.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/render_target/cache.h>
#include <rex/graphics/pipeline/shader/dxbc_translator.h>
#include <rex/graphics/pipeline/texture/cache.h>
#include <rex/graphics/util/draw.h>
#include <rex/math.h>

namespace rex::graphics {
using namespace ucode;

void DxbcShaderTranslator::CompletePixelShader_WriteToRTVs() {
  uint32_t shader_writes_color_targets = current_shader().writes_color_targets();
  if (!shader_writes_color_targets) {
    return;
  }

  uint32_t gamma_temp = PushSystemTemp();
  for (uint32_t i = 0; i < 4; ++i) {
    if (!(shader_writes_color_targets & (1 << i))) {
      continue;
    }
    uint32_t system_temp_color = system_temps_color_[i];
    // Apply the exponent bias after alpha to coverage because it needs the
    // unbiased alpha from the shader.
    a_.OpMul(dxbc::Dest::R(system_temp_color), dxbc::Src::R(system_temp_color),
             LoadSystemConstant(SystemConstants::Index::kColorExpBias,
                                offsetof(SystemConstants, color_exp_bias) + sizeof(float) * i,
                                dxbc::Src::kXXXX));
    if (gamma_render_target_as_unorm8_) {
      // Convert to gamma space - this is incorrect, since it must be done after
      // blending on the Xbox 360, but this is just one of many blending issues
      // in the RTV path.
      a_.OpAnd(dxbc::Dest::R(gamma_temp, 0b0001), LoadFlagsSystemConstant(),
               dxbc::Src::LU(kSysFlag_ConvertColor0ToGamma << i));
      a_.OpIf(true, dxbc::Src::R(gamma_temp, dxbc::Src::kXXXX));
      // Saturate before the gamma conversion.
      a_.OpMov(dxbc::Dest::R(system_temp_color, 0b0111), dxbc::Src::R(system_temp_color), true);
      for (uint32_t j = 0; j < 3; ++j) {
        PreSaturatedLinearToPWLGamma(a_, system_temp_color, j, system_temp_color, j, gamma_temp, 0,
                                     gamma_temp, 1);
      }
      a_.OpEndIf();
    }
    // Copy the color from a readable temp register to an output register.
    a_.OpMov(dxbc::Dest::O(i), dxbc::Src::R(system_temp_color));
  }
  // Release gamma_temp.
  PopSystemTemp();
}

void DxbcShaderTranslator::CompletePixelShader_DSV_DepthTo24Bit() {
  bool shader_writes_depth = current_shader().writes_depth();

  if (!DSV_IsWritingFloat24Depth()) {
    if (shader_writes_depth) {
      // If not converting, but the shader writes depth explicitly, for float24,
      // need to scale it from guest 0...1 to host 0...0.5 to support
      // reinterpretation round trips as viewport scaling doesn't apply to
      // oDepth.
      a_.OpAnd(dxbc::Dest::R(system_temp_depth_stencil_, 0b0010), LoadFlagsSystemConstant(),
               dxbc::Src::LU(kSysFlag_DepthFloat24));
      a_.OpIf(true, dxbc::Src::R(system_temp_depth_stencil_, dxbc::Src::kYYYY));
      a_.OpMul(dxbc::Dest::R(system_temp_depth_stencil_, 0b0001),
               dxbc::Src::R(system_temp_depth_stencil_, dxbc::Src::kXXXX), dxbc::Src::LF(0.5f));
      a_.OpEndIf();
      // Write the depth from the temporary to the system depth output.
      a_.OpMov(dxbc::Dest::ODepth(), dxbc::Src::R(system_temp_depth_stencil_, dxbc::Src::kXXXX));
    }
    return;
  }

  uint32_t temp;
  if (shader_writes_depth) {
    // The depth is already written to system_temp_depth_stencil_.x and clamped
    // to 0...1 with NaNs dropped (saturating in StoreResult); yzw are free.
    temp = system_temp_depth_stencil_;
  } else {
    // Need a temporary variable; remap the sample's depth input from host
    // 0...0.5 back to guest 0...1 for conversion purposes to it and saturate it
    // (in Direct3D 11, depth is clamped to the viewport bounds after the pixel
    // shader, and SV_Position.z contains the unclamped depth, which may be
    // outside the viewport's depth range if it's biased); though it will be
    // clamped to the viewport bounds anyway, but to be able to make the
    // assumption of it being clamped while working with the bit representation.
    temp = PushSystemTemp();
    in_position_used_ |= 0b0100;
    a_.OpMul(dxbc::Dest::R(temp, 0b0001), dxbc::Src::V1D(in_reg_ps_position_, dxbc::Src::kZZZZ),
             dxbc::Src::LF(2.0f), true);
  }

  dxbc::Dest temp_x_dest(dxbc::Dest::R(temp, 0b0001));
  dxbc::Src temp_x_src(dxbc::Src::R(temp, dxbc::Src::kXXXX));
  dxbc::Dest temp_y_dest(dxbc::Dest::R(temp, 0b0010));
  dxbc::Src temp_y_src(dxbc::Src::R(temp, dxbc::Src::kYYYY));

  if (GetDxbcShaderModification().pixel.depth_stencil_mode ==
      Modification::DepthStencilMode::kFloat24Truncating) {
    // Simplified conversion, always less than or equal to the original value -
    // just drop the lower bits.
    // The float32 exponent bias is 127.
    // After saturating, the exponent range is -127...0.
    // The smallest normalized 20e4 exponent is -14 - should drop 3 mantissa
    // bits at -14 or above.
    // The smallest denormalized 20e4 number is -34 - should drop 23 mantissa
    // bits at -34.
    // Anything smaller than 2^-34 becomes 0.
    dxbc::Dest truncate_dest(shader_writes_depth ? dxbc::Dest::ODepth() : dxbc::Dest::ODepthLE());
    // Check if the number is representable as a float24 after truncation - the
    // exponent is at least -34.
    a_.OpUGE(temp_y_dest, temp_x_src, dxbc::Src::LU(0x2E800000));
    a_.OpIf(true, temp_y_src);
    {
      // Extract the biased float32 exponent to temp.y.
      // temp.y = 113+ at exponent -14+.
      // temp.y = 93 at exponent -34.
      a_.OpUBFE(temp_y_dest, dxbc::Src::LU(8), dxbc::Src::LU(23), temp_x_src);
      // Convert exponent to the unclamped number of bits to truncate.
      // 116 - 113 = 3.
      // 116 - 93 = 23.
      // temp.y = 3+ at exponent -14+.
      // temp.y = 23 at exponent -34.
      a_.OpIAdd(temp_y_dest, dxbc::Src::LI(116), -temp_y_src);
      // Clamp the truncated bit count to drop 3 bits of any normal number.
      // Exponents below -34 are handled separately.
      // temp.y = 3 at exponent -14.
      // temp.y = 23 at exponent -34.
      a_.OpIMax(temp_y_dest, temp_y_src, dxbc::Src::LI(3));
      // Truncate the mantissa - fill the low bits with zeros.
      // temp.x = result in 0...1 range
      a_.OpBFI(temp_x_dest, temp_y_src, dxbc::Src::LU(0), dxbc::Src::LU(0), temp_x_src);
      // Remap from guest 0...1 to host 0...0.5.
      a_.OpMul(truncate_dest, temp_x_src, dxbc::Src::LF(0.5f));
    }
    // The number is not representable as float24 after truncation - zero.
    a_.OpElse();
    a_.OpMov(truncate_dest, dxbc::Src::LF(0.0f));
    // Close the non-zero result check.
    a_.OpEndIf();
  } else {
    // Properly convert to 20e4, with rounding to the nearest even (the bias was
    // pre-applied by multiplying by 2), then convert back restoring the bias.
    PreClampedDepthTo20e4(a_, temp, 0, temp, 0, temp, 1, true, false);
    Depth20e4To32(a_, dxbc::Dest::ODepth(), temp, 0, 0, temp, 0, temp, 1, true);
  }

  if (!shader_writes_depth) {
    // Release temp.
    PopSystemTemp();
  }
}

void DxbcShaderTranslator::CompletePixelShader_AlphaToMaskSample(
    bool initialize, uint32_t sample_index, float threshold_base, dxbc::Src threshold_offset,
    float threshold_offset_scale, uint32_t coverage_temp, uint32_t coverage_temp_component,
    uint32_t temp, uint32_t temp_component) {
  dxbc::Dest temp_dest(dxbc::Dest::R(temp, 1 << temp_component));
  dxbc::Src temp_src(dxbc::Src::R(temp).Select(temp_component));
  // Calculate the threshold.
  a_.OpMAd(temp_dest, threshold_offset, dxbc::Src::LF(-threshold_offset_scale),
           dxbc::Src::LF(threshold_base));
  // Check if alpha of oC0 is at or greater than the threshold (handling NaN
  // according to the Direct3D 11.3 functional specification, as not covered).
  a_.OpGE(temp_dest, dxbc::Src::R(system_temps_color_[0], dxbc::Src::kWWWW), temp_src);
  dxbc::Dest coverage_dest(dxbc::Dest::R(coverage_temp, 1 << coverage_temp_component));
  dxbc::Src coverage_src(dxbc::Src::R(coverage_temp).Select(coverage_temp_component));
  // [N-10b deletion c] the ROV coverage-masking branch is DELETED.
  if (initialize) {
    // First sample tested - initialize.
    assert_true(coverage_temp != temp || coverage_temp_component != temp_component);
    a_.OpAnd(coverage_dest, temp_src, dxbc::Src::LU(uint32_t(1) << sample_index));
  } else {
    // Not first sample tested - add.
    a_.OpAnd(temp_dest, temp_src, dxbc::Src::LU(uint32_t(1) << sample_index));
    a_.OpOr(coverage_dest, coverage_src, temp_src);
  }
}

void DxbcShaderTranslator::CompletePixelShader_AlphaToMask() {
  // Check if alpha to coverage can be done at all in this shader.
  if (!current_shader().writes_color_target(0) || IsForceEarlyDepthStencilGlobalFlagEnabled()) {
    return;
  }

  // Initialize the output coverage for the case if alpha to mask is not
  // enabled - it needs to be written on every execution path.
  a_.OpMov(dxbc::Dest::OMask(), dxbc::Src::LU(UINT32_MAX));

  // Check if alpha to coverage is enabled.
  dxbc::Src alpha_to_mask_constant_src(LoadSystemConstant(SystemConstants::Index::kAlphaToMask,
                                                          offsetof(SystemConstants, alpha_to_mask),
                                                          dxbc::Src::kXXXX));
  a_.OpIf(true, alpha_to_mask_constant_src);

  uint32_t temp = PushSystemTemp();
  dxbc::Dest temp_x_dest(dxbc::Dest::R(temp, 0b0001));
  dxbc::Src temp_x_src(dxbc::Src::R(temp, dxbc::Src::kXXXX));

  // Get the dithering threshold offset index for the pixel, Y - low bit of
  // offset index, X - high bit, and extract the offset and convert it to
  // floating-point. With resolution scaling, still using host pixels, to
  // preserve the idea of dithering.
  // temp.x = alpha to coverage offset as float 0.0...3.0.
  in_position_used_ |= 0b0011;
  a_.OpFToU(dxbc::Dest::R(temp, 0b0011), dxbc::Src::V1D(in_reg_ps_position_));
  a_.OpAnd(dxbc::Dest::R(temp, 0b0010), dxbc::Src::R(temp, dxbc::Src::kYYYY), dxbc::Src::LU(1));
  a_.OpBFI(temp_x_dest, dxbc::Src::LU(1), dxbc::Src::LU(1), temp_x_src,
           dxbc::Src::R(temp, dxbc::Src::kYYYY));
  a_.OpIShL(temp_x_dest, temp_x_src, dxbc::Src::LU(1));
  a_.OpUBFE(temp_x_dest, dxbc::Src::LU(2), temp_x_src, alpha_to_mask_constant_src);
  a_.OpUToF(temp_x_dest, temp_x_src);

  // Write the result to temp.z.
  // temp.x = alpha to coverage offset as float 0.0...3.0.
  // temp.z = accumulated coverage.
  uint32_t coverage_temp = temp;
  uint32_t coverage_temp_component = 2;

  // Check if MSAA is enabled.
  a_.OpIf(true, LoadSystemConstant(SystemConstants::Index::kSampleCountLog2,
                                   offsetof(SystemConstants, sample_count_log2), dxbc::Src::kYYYY));
  {
    // Check if MSAA is 4x or 2x.
    a_.OpIf(true,
            LoadSystemConstant(SystemConstants::Index::kSampleCountLog2,
                               offsetof(SystemConstants, sample_count_log2), dxbc::Src::kXXXX));
    // 4x MSAA.
    // Sample 0 must be checked first - CompletePixelShader_AlphaToMaskSample
    // initializes the result for sample index 0.
    CompletePixelShader_AlphaToMaskSample(true, 0, 0.75f, temp_x_src, 1.0f / 16.0f, coverage_temp,
                                          coverage_temp_component, temp, 1);
    CompletePixelShader_AlphaToMaskSample(false, 1, 0.25f, temp_x_src, 1.0f / 16.0f, coverage_temp,
                                          coverage_temp_component, temp, 1);
    CompletePixelShader_AlphaToMaskSample(false, 2, 0.5f, temp_x_src, 1.0f / 16.0f, coverage_temp,
                                          coverage_temp_component, temp, 1);
    CompletePixelShader_AlphaToMaskSample(false, 3, 1.0f, temp_x_src, 1.0f / 16.0f, coverage_temp,
                                          coverage_temp_component, temp, 1);
    // 2x MSAA.
    // - Native 2x: top (0 in Xenia) is 1 in D3D10.1+, bottom (1 in Xenia) is 0.
    // - 2x as 4x: top is 0, bottom is 3.
    a_.OpElse();
    CompletePixelShader_AlphaToMaskSample(true, msaa_2x_supported_ ? 1 : 0, 0.5f, temp_x_src,
                                          1.0f / 8.0f, coverage_temp, coverage_temp_component, temp,
                                          1);
    CompletePixelShader_AlphaToMaskSample(false, msaa_2x_supported_ ? 0 : 3, 1.0f, temp_x_src,
                                          1.0f / 8.0f, coverage_temp, coverage_temp_component, temp,
                                          1);
    // Close the 4x check.
    a_.OpEndIf();
  }
  // MSAA is disabled.
  a_.OpElse();
  CompletePixelShader_AlphaToMaskSample(true, 0, 1.0f, temp_x_src, 1.0f / 4.0f, coverage_temp,
                                        coverage_temp_component, temp, 1);
  // Close the 2x/4x check.
  a_.OpEndIf();

  // Check if any sample is still covered and return to avoid unneeded work (the
  // driver's shader compiler may place return after a discard, but it will
  // likely not place one during SV_Coverage assignment - that's what the AMD
  // compiler does, at least). Then, if needed, write the coverage value.
  {
    dxbc::Src coverage_src(dxbc::Src::R(coverage_temp, coverage_temp_component));
    a_.OpDiscard(false, coverage_src);
    a_.OpMov(dxbc::Dest::OMask(), coverage_src);
  }

  // Release temp.
  PopSystemTemp();

  // Close the alpha to coverage check.
  a_.OpEndIf();
}

void DxbcShaderTranslator::CompletePixelShader() {
  // [N-10b deletion c] the ROV epilogue (rt0-written check, WriteToROV) is
  // DELETED throughout this function.
  if (is_depth_only_pixel_shader_) {
    return;
  }

  if (current_shader().writes_color_target(0) && !IsForceEarlyDepthStencilGlobalFlagEnabled()) {
    // Alpha test.
    // X - mask, then masked result (SGPR for loading, VGPR for masking).
    // Y - operation result (SGPR for mask operations, VGPR for alpha
    //     operations).
    // Z - fuzzy diff (alpha - reference), used when fuzzy epsilon is enabled.
    uint32_t alpha_test_temp = PushSystemTemp();
    dxbc::Dest alpha_test_mask_dest(dxbc::Dest::R(alpha_test_temp, 0b0001));
    dxbc::Src alpha_test_mask_src(dxbc::Src::R(alpha_test_temp, dxbc::Src::kXXXX));
    dxbc::Dest alpha_test_op_dest(dxbc::Dest::R(alpha_test_temp, 0b0010));
    dxbc::Src alpha_test_op_src(dxbc::Src::R(alpha_test_temp, dxbc::Src::kYYYY));
    dxbc::Dest alpha_test_fuzzy_diff_dest(dxbc::Dest::R(alpha_test_temp, 0b0100));
    dxbc::Src alpha_test_fuzzy_diff_src(dxbc::Src::R(alpha_test_temp, dxbc::Src::kZZZZ));
    // Extract the comparison mask to check if the test needs to be done at all.
    // Don't care about flow control being somewhat dynamic - early Z is forced
    // using a special version of the shader anyway.
    a_.OpUBFE(alpha_test_mask_dest, dxbc::Src::LU(3), dxbc::Src::LU(kSysFlag_AlphaPassIfLess_Shift),
              LoadFlagsSystemConstant());
    // Compare the mask to ALWAYS to check if the test shouldn't be done (will
    // pass even for NaNs, though the expected behavior in this case hasn't been
    // checked, but let's assume this means "always", not "less, equal or
    // greater".
    // TODO(Triang3l): Check how alpha test works with NaN on Direct3D 9.
    a_.OpINE(alpha_test_op_dest, alpha_test_mask_src,
             dxbc::Src::LU(uint32_t(xenos::CompareFunction::kAlways)));
    // Don't do the test if the mode is "always".
    a_.OpIf(true, alpha_test_op_src);
    {
      // Do the test.
      dxbc::Src alpha_src(dxbc::Src::R(system_temps_color_[0], dxbc::Src::kWWWW));
      dxbc::Src alpha_test_reference_src(
          LoadSystemConstant(SystemConstants::Index::kAlphaTestReference,
                             offsetof(SystemConstants, alpha_test_reference), dxbc::Src::kXXXX));
      // Epsilon for fuzzy alpha checks (prevents flickering on NVIDIA GPUs).
      dxbc::Src fuzzy_epsilon = dxbc::Src::LF(1e-3f);
      // Handle "not equal" specially (specifically as "not equal" so it's true
      // for NaN, not "less or greater" which is false for NaN).
      a_.OpIEq(alpha_test_op_dest, alpha_test_mask_src,
               dxbc::Src::LU(uint32_t(xenos::CompareFunction::kNotEqual)));
      a_.OpIf(true, alpha_test_op_src);
      {
        if (REXCVAR_GET(use_fuzzy_alpha_epsilon)) {
          a_.OpAdd(alpha_test_fuzzy_diff_dest, alpha_src, -alpha_test_reference_src);
          // Check if distance to desired value is less than epsilon (false for
          // NaN) and write the negated result.
          a_.OpLT(alpha_test_mask_dest, alpha_test_fuzzy_diff_src.Abs(), fuzzy_epsilon);
          a_.OpNot(alpha_test_mask_dest, alpha_test_mask_src);
        } else {
          a_.OpNE(alpha_test_mask_dest, alpha_src, alpha_test_reference_src);
        }
      }
      a_.OpElse();
      {
        if (REXCVAR_GET(use_fuzzy_alpha_epsilon)) {
          // Less than.
          a_.OpAdd(alpha_test_fuzzy_diff_dest, alpha_src, -fuzzy_epsilon);
          a_.OpLT(alpha_test_op_dest, alpha_test_fuzzy_diff_src, alpha_test_reference_src);
          a_.OpOr(alpha_test_op_dest, alpha_test_op_src, dxbc::Src::LU(~uint32_t(1 << 0)));
          a_.OpAnd(alpha_test_mask_dest, alpha_test_mask_src, alpha_test_op_src);
          // "Equals" to.
          a_.OpAdd(alpha_test_fuzzy_diff_dest, alpha_src, -alpha_test_reference_src);
          a_.OpLT(alpha_test_op_dest, alpha_test_fuzzy_diff_src.Abs(), fuzzy_epsilon);
          a_.OpOr(alpha_test_op_dest, alpha_test_op_src, dxbc::Src::LU(~uint32_t(1 << 1)));
          a_.OpAnd(alpha_test_mask_dest, alpha_test_mask_src, alpha_test_op_src);
          // Greater than.
          a_.OpAdd(alpha_test_fuzzy_diff_dest, alpha_src, fuzzy_epsilon);
          a_.OpLT(alpha_test_op_dest, alpha_test_reference_src, alpha_test_fuzzy_diff_src);
          a_.OpOr(alpha_test_op_dest, alpha_test_op_src, dxbc::Src::LU(~uint32_t(1 << 2)));
          a_.OpAnd(alpha_test_mask_dest, alpha_test_mask_src, alpha_test_op_src);
        } else {
          // Less than.
          a_.OpLT(alpha_test_op_dest, alpha_src, alpha_test_reference_src);
          a_.OpOr(alpha_test_op_dest, alpha_test_op_src, dxbc::Src::LU(~uint32_t(1 << 0)));
          a_.OpAnd(alpha_test_mask_dest, alpha_test_mask_src, alpha_test_op_src);
          // Equals to.
          a_.OpEq(alpha_test_op_dest, alpha_src, alpha_test_reference_src);
          a_.OpOr(alpha_test_op_dest, alpha_test_op_src, dxbc::Src::LU(~uint32_t(1 << 1)));
          a_.OpAnd(alpha_test_mask_dest, alpha_test_mask_src, alpha_test_op_src);
          // Greater than.
          a_.OpLT(alpha_test_op_dest, alpha_test_reference_src, alpha_src);
          a_.OpOr(alpha_test_op_dest, alpha_test_op_src, dxbc::Src::LU(~uint32_t(1 << 2)));
          a_.OpAnd(alpha_test_mask_dest, alpha_test_mask_src, alpha_test_op_src);
        }
      }
      // Close the "not equal" check.
      a_.OpEndIf();
      // Discard the pixel if it has failed the test.
      a_.OpDiscard(false, alpha_test_mask_src);
    }
    // Close the "not always" check.
    a_.OpEndIf();
    // Release alpha_test_temp.
    PopSystemTemp();

    // Discard samples with alpha to coverage.
    CompletePixelShader_AlphaToMask();
  }

  // Write the values to the render targets. Not applying the exponent bias yet
  // because the original 0 to 1 alpha value is needed for alpha to coverage.
  CompletePixelShader_WriteToRTVs();
  CompletePixelShader_DSV_DepthTo24Bit();
}

void DxbcShaderTranslator::PreClampedFloat32To7e3(dxbc::Assembler& a, uint32_t f10_temp,
                                                  uint32_t f10_temp_component, uint32_t f32_temp,
                                                  uint32_t f32_temp_component, uint32_t temp_temp,
                                                  uint32_t temp_temp_component) {
  assert_true(temp_temp != f10_temp || temp_temp_component != f10_temp_component);
  assert_true(temp_temp != f32_temp || temp_temp_component != f32_temp_component);
  // Source and destination may be the same.
  dxbc::Dest f10_dest(dxbc::Dest::R(f10_temp, 1 << f10_temp_component));
  dxbc::Src f10_src(dxbc::Src::R(f10_temp).Select(f10_temp_component));
  dxbc::Src f32_src(dxbc::Src::R(f32_temp).Select(f32_temp_component));
  dxbc::Dest temp_dest(dxbc::Dest::R(temp_temp, 1 << temp_temp_component));
  dxbc::Src temp_src(dxbc::Src::R(temp_temp).Select(temp_temp_component));

  // https://github.com/Microsoft/DirectXTex/blob/master/DirectXTex/DirectXTexConvert.cpp
  // Assuming the color is already clamped to [0, 31.875].

  // Check if the number is too small to be represented as normalized 7e3.
  // temp = f32 < 2^-2
  a.OpULT(temp_dest, f32_src, dxbc::Src::LU(0x3E800000));
  // Handle denormalized numbers separately.
  a.OpIf(true, temp_src);
  {
    // temp = f32 >> 23
    a.OpUShR(temp_dest, f32_src, dxbc::Src::LU(23));
    // temp = 125 - (f32 >> 23)
    a.OpIAdd(temp_dest, dxbc::Src::LI(125), -temp_src);
    // Don't allow the shift to overflow, since in DXBC the lower 5 bits of the
    // shift amount are used.
    // temp = min(125 - (f32 >> 23), 24)
    a.OpUMin(temp_dest, temp_src, dxbc::Src::LU(24));
    // biased_f32 = (f32 & 0x7FFFFF) | 0x800000
    a.OpBFI(f10_dest, dxbc::Src::LU(9), dxbc::Src::LU(23), dxbc::Src::LU(1), f32_src);
    // biased_f32 = ((f32 & 0x7FFFFF) | 0x800000) >> min(125 - (f32 >> 23), 24)
    a.OpUShR(f10_dest, f10_src, temp_src);
  }
  // Not denormalized?
  a.OpElse();
  {
    // Bias the exponent.
    // biased_f32 = f32 + (-124 << 23)
    // (left shift of a negative value is undefined behavior)
    a.OpIAdd(f10_dest, f32_src, dxbc::Src::LU(0xC2000000u));
  }
  // Close the denormal check.
  a.OpEndIf();
  // Build the 7e3 number.
  // temp = (biased_f32 >> 16) & 1
  a.OpUBFE(temp_dest, dxbc::Src::LU(1), dxbc::Src::LU(16), f10_src);
  // f10 = biased_f32 + 0x7FFF
  a.OpIAdd(f10_dest, f10_src, dxbc::Src::LU(0x7FFF));
  // f10 = biased_f32 + 0x7FFF + ((biased_f32 >> 16) & 1)
  a.OpIAdd(f10_dest, f10_src, temp_src);
  // f24 = ((biased_f32 + 0x7FFF + ((biased_f32 >> 16) & 1)) >> 16) & 0x3FF
  a.OpUBFE(f10_dest, dxbc::Src::LU(10), dxbc::Src::LU(16), f10_src);
}

void DxbcShaderTranslator::UnclampedFloat32To7e3(dxbc::Assembler& a, uint32_t f10_temp,
                                                 uint32_t f10_temp_component, uint32_t f32_temp,
                                                 uint32_t f32_temp_component, uint32_t temp_temp,
                                                 uint32_t temp_temp_component) {
  // Source and destination might be the same or different, just like in
  // PreClampedFloat32To7e3 - clamp to the destination and use it as source.
  a.OpMax(dxbc::Dest::R(f10_temp, 1 << f10_temp_component),
          dxbc::Src::R(f32_temp).Select(f32_temp_component), dxbc::Src::LF(0.0f));
  a.OpMin(dxbc::Dest::R(f10_temp, 1 << f10_temp_component),
          dxbc::Src::R(f10_temp).Select(f10_temp_component), dxbc::Src::LF(31.875f));
  PreClampedFloat32To7e3(a, f10_temp, f10_temp_component, f10_temp, f10_temp_component, temp_temp,
                         temp_temp_component);
}

void DxbcShaderTranslator::Float7e3To32(dxbc::Assembler& a, const dxbc::Dest& f32,
                                        uint32_t f10_temp, uint32_t f10_temp_component,
                                        uint32_t f10_shift, uint32_t temp1_temp,
                                        uint32_t temp1_temp_component, uint32_t temp2_temp,
                                        uint32_t temp2_temp_component) {
  assert_true(f10_shift <= (32 - 10));
  assert_true(temp1_temp != temp2_temp || temp1_temp_component != temp2_temp_component);
  // Source may be the same as temp1 or temp2.
  dxbc::Dest exponent_dest(dxbc::Dest::R(temp1_temp, 1 << temp1_temp_component));
  dxbc::Src exponent_src(dxbc::Src::R(temp1_temp).Select(temp1_temp_component));
  dxbc::Dest mantissa_dest(dxbc::Dest::R(temp2_temp, 1 << temp2_temp_component));
  dxbc::Src mantissa_src(dxbc::Src::R(temp2_temp).Select(temp2_temp_component));

  // https://github.com/Microsoft/DirectXTex/blob/master/DirectXTex/DirectXTexConvert.cpp

  if (!(f10_temp == temp1_temp && f10_temp_component == temp1_temp_component)) {
    // Unpack the exponent before the mantissa if that doesn't overwrite the
    // source.
    a.OpUBFE(exponent_dest, dxbc::Src::LU(3), dxbc::Src::LU(f10_shift + 7),
             dxbc::Src::R(f10_temp).Select(f10_temp_component));
  }
  // Unpack the mantissa.
  a.OpUBFE(mantissa_dest, dxbc::Src::LU(7), dxbc::Src::LU(f10_shift),
           dxbc::Src::R(f10_temp).Select(f10_temp_component));
  if (f10_temp == temp1_temp && f10_temp_component == temp1_temp_component) {
    // Unpack the exponent after the mantissa if doing that before the mantissa
    // would overwrite the source.
    a.OpUBFE(exponent_dest, dxbc::Src::LU(3), dxbc::Src::LU(f10_shift + 7),
             dxbc::Src::R(f10_temp).Select(f10_temp_component));
  }
  // Check if the number is denormalized.
  a.OpIf(false, exponent_src);
  {
    // Check if the number is non-zero (if the mantissa isn't zero - the
    // exponent is known to be zero at this point).
    a.OpIf(true, mantissa_src);
    {
      // Normalize the mantissa.
      // Note that HLSL firstbithigh(x) is compiled to DXBC like:
      // `x ? 31 - firstbit_hi(x) : -1`
      // (returns the index from the LSB, not the MSB, but -1 for zero too).
      // exponent = firstbit_hi(mantissa)
      a.OpFirstBitHi(exponent_dest, mantissa_src);
      // exponent = 7 - firstbithigh(mantissa)
      // Or:
      // exponent = 7 - (31 - firstbit_hi(mantissa))
      a.OpIAdd(exponent_dest, exponent_src, dxbc::Src::LI(7 - 31));
      // mantissa = mantissa << (7 - firstbithigh(mantissa))
      // AND 0x7F not needed after this - BFI will do it.
      a.OpIShL(mantissa_dest, mantissa_src, exponent_src);
      // Get the normalized exponent.
      // exponent = 1 - (7 - firstbithigh(mantissa))
      a.OpIAdd(exponent_dest, dxbc::Src::LI(1), -exponent_src);
    }
    // The number is zero.
    a.OpElse();
    {
      // Set the unbiased exponent to -124 for zero - 124 will be added later,
      // resulting in zero float32.
      a.OpMov(exponent_dest, dxbc::Src::LI(-124));
    }
    // Close the non-zero check.
    a.OpEndIf();
  }
  // Close the denormal check.
  a.OpEndIf();
  // Bias the exponent and move it to the correct location in f32.
  a.OpIMAd(exponent_dest, exponent_src, dxbc::Src::LI(1 << 23), dxbc::Src::LI(124 << 23));
  // Combine the mantissa and the exponent.
  a.OpBFI(f32, dxbc::Src::LU(7), dxbc::Src::LU(23 - 7), mantissa_src, exponent_src);
}

void DxbcShaderTranslator::PreClampedDepthTo20e4(dxbc::Assembler& a, uint32_t f24_temp,
                                                 uint32_t f24_temp_component, uint32_t f32_temp,
                                                 uint32_t f32_temp_component, uint32_t temp_temp,
                                                 uint32_t temp_temp_component,
                                                 bool round_to_nearest_even,
                                                 bool remap_from_0_to_0_5) {
  assert_true(temp_temp != f24_temp || temp_temp_component != f24_temp_component);
  assert_true(temp_temp != f32_temp || temp_temp_component != f32_temp_component);
  // Source and destination may be the same.
  dxbc::Dest f24_dest(dxbc::Dest::R(f24_temp, 1 << f24_temp_component));
  dxbc::Src f24_src(dxbc::Src::R(f24_temp).Select(f24_temp_component));
  dxbc::Src f32_src(dxbc::Src::R(f32_temp).Select(f32_temp_component));
  dxbc::Dest temp_dest(dxbc::Dest::R(temp_temp, 1 << temp_temp_component));
  dxbc::Src temp_src(dxbc::Src::R(temp_temp).Select(temp_temp_component));

  // CFloat24 from d3dref9.dll +
  // https://github.com/Microsoft/DirectXTex/blob/master/DirectXTex/DirectXTexConvert.cpp
  // Assuming the depth is already clamped to [0, 2) (in all places, the depth
  // is written with the saturate flag set).

  uint32_t remap_bias = uint32_t(remap_from_0_to_0_5);

  // Check if the number is too small to be represented as normalized 20e4.
  // temp = f32 < 2^-14
  a.OpULT(temp_dest, f32_src, dxbc::Src::LU(0x38800000 - (remap_bias << 23)));
  // Handle denormalized numbers separately.
  a.OpIf(true, temp_src);
  {
    // temp = f32 >> 23
    a.OpUShR(temp_dest, f32_src, dxbc::Src::LU(23));
    // temp = 113 - (f32 >> 23)
    a.OpIAdd(temp_dest, dxbc::Src::LI(113 - remap_bias), -temp_src);
    // Don't allow the shift to overflow, since in DXBC the lower 5 bits of the
    // shift amount are used (otherwise 0 becomes 8).
    // temp = min(113 - (f32 >> 23), 24)
    a.OpUMin(temp_dest, temp_src, dxbc::Src::LU(24));
    // biased_f32 = (f32 & 0x7FFFFF) | 0x800000
    a.OpBFI(f24_dest, dxbc::Src::LU(9), dxbc::Src::LU(23), dxbc::Src::LU(1), f32_src);
    // biased_f32 = ((f32 & 0x7FFFFF) | 0x800000) >> min(113 - (f32 >> 23), 24)
    a.OpUShR(f24_dest, f24_src, temp_src);
  }
  // Not denormalized?
  a.OpElse();
  {
    // Bias the exponent.
    // biased_f32 = f32 + (-112 << 23)
    // (left shift of a negative value is undefined behavior)
    a.OpIAdd(f24_dest, f32_src, dxbc::Src::LU(0xC8000000u + (remap_bias << 23)));
  }
  // Close the denormal check.
  a.OpEndIf();
  // Build the 20e4 number.
  if (round_to_nearest_even) {
    // temp = (biased_f32 >> 3) & 1
    a.OpUBFE(temp_dest, dxbc::Src::LU(1), dxbc::Src::LU(3), f24_src);
    // f24 = biased_f32 + 3
    a.OpIAdd(f24_dest, f24_src, dxbc::Src::LU(3));
    // f24 = biased_f32 + 3 + ((biased_f32 >> 3) & 1)
    a.OpIAdd(f24_dest, f24_src, temp_src);
  }
  // For rounding to the nearest even:
  // f24 = ((biased_f32 + 3 + ((biased_f32 >> 3) & 1)) >> 3) & 0xFFFFFF
  // For rounding towards zero:
  // f24 = (biased_f32 >> 3) & 0xFFFFFF
  a.OpUBFE(f24_dest, dxbc::Src::LU(24), dxbc::Src::LU(3), f24_src);
}

void DxbcShaderTranslator::Depth20e4To32(dxbc::Assembler& a, const dxbc::Dest& f32,
                                         uint32_t f24_temp, uint32_t f24_temp_component,
                                         uint32_t f24_shift, uint32_t temp1_temp,
                                         uint32_t temp1_temp_component, uint32_t temp2_temp,
                                         uint32_t temp2_temp_component, bool remap_to_0_to_0_5) {
  assert_true(f24_shift <= (32 - 24));
  assert_true(temp1_temp != temp2_temp || temp1_temp_component != temp2_temp_component);
  // Source may be the same as temp1 or temp2.
  dxbc::Dest exponent_dest(dxbc::Dest::R(temp1_temp, 1 << temp1_temp_component));
  dxbc::Src exponent_src(dxbc::Src::R(temp1_temp).Select(temp1_temp_component));
  dxbc::Dest mantissa_dest(dxbc::Dest::R(temp2_temp, 1 << temp2_temp_component));
  dxbc::Src mantissa_src(dxbc::Src::R(temp2_temp).Select(temp2_temp_component));

  // CFloat24 from d3dref9.dll +
  // https://github.com/Microsoft/DirectXTex/blob/master/DirectXTex/DirectXTexConvert.cpp

  uint32_t remap_bias = uint32_t(remap_to_0_to_0_5);

  if (!(f24_temp == temp1_temp && f24_temp_component == temp1_temp_component)) {
    // Unpack the exponent before the mantissa if that doesn't overwrite the
    // source.
    a.OpUBFE(exponent_dest, dxbc::Src::LU(4), dxbc::Src::LU(f24_shift + 20),
             dxbc::Src::R(f24_temp).Select(f24_temp_component));
  }
  // Unpack the mantissa.
  a.OpUBFE(mantissa_dest, dxbc::Src::LU(20), dxbc::Src::LU(f24_shift),
           dxbc::Src::R(f24_temp).Select(f24_temp_component));
  if (f24_temp == temp1_temp && f24_temp_component == temp1_temp_component) {
    // Unpack the exponent after the mantissa if doing that before the mantissa
    // would overwrite the source.
    a.OpUBFE(exponent_dest, dxbc::Src::LU(4), dxbc::Src::LU(f24_shift + 20),
             dxbc::Src::R(f24_temp).Select(f24_temp_component));
  }
  // Check if the number is denormalized.
  a.OpIf(false, exponent_src);
  {
    // Check if the number is non-zero (if the mantissa isn't zero - the
    // exponent is known to be zero at this point).
    a.OpIf(true, mantissa_src);
    {
      // Normalize the mantissa.
      // Note that HLSL firstbithigh(x) is compiled to DXBC like:
      // `x ? 31 - firstbit_hi(x) : -1`
      // (returns the index from the LSB, not the MSB, but -1 for zero too).
      // exponent = firstbit_hi(mantissa)
      a.OpFirstBitHi(exponent_dest, mantissa_src);
      // exponent = 20 - firstbithigh(mantissa)
      // Or:
      // exponent = 20 - (31 - firstbit_hi(mantissa))
      a.OpIAdd(exponent_dest, exponent_src, dxbc::Src::LI(20 - 31));
      // mantissa = mantissa << (20 - firstbithigh(mantissa))
      // AND 0xFFFFF not needed after this - BFI will do it.
      a.OpIShL(mantissa_dest, mantissa_src, exponent_src);
      // Get the normalized exponent.
      // exponent = 1 - (20 - firstbithigh(mantissa))
      a.OpIAdd(exponent_dest, dxbc::Src::LI(1), -exponent_src);
    }
    // The number is zero.
    a.OpElse();
    {
      // Set the unbiased exponent to -112 for zero - 112 will be added later
      // (taking the range remap bias into account), resulting in zero float32.
      a.OpMov(exponent_dest, dxbc::Src::LI(-int32_t(112 - remap_bias)));
    }
    // Close the non-zero check.
    a.OpEndIf();
  }
  // Close the denormal check.
  a.OpEndIf();
  // Bias the exponent and move it to the correct location in f32, and also
  // remap from guest 0...1 to host 0...0.5 if needed.
  a.OpIMAd(exponent_dest, exponent_src, dxbc::Src::LI(1 << 23),
           dxbc::Src::LI((112 - remap_bias) << 23));
  // Combine the mantissa and the exponent.
  a.OpBFI(f32, dxbc::Src::LU(20), dxbc::Src::LU(23 - 20), mantissa_src, exponent_src);
}

}  // namespace rex::graphics
