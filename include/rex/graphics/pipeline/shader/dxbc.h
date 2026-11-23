/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <atomic>
#include <vector>

#include <rex/graphics/pipeline/shader/dxbc_translator.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/xenos.h>

namespace rex::graphics {

class DxbcShader : public Shader {
 public:
  class DxbcTranslation : public Translation {
   public:
    DxbcTranslation(DxbcShader& shader, uint64_t modification)
        : Translation(shader, modification) {}

    // [ia] The input elements this translation's vertex shader declares (the
    // input-assembler variant only; empty otherwise). Slot = the shader's
    // vertex_bindings() index; stride = that binding's stride_words * 4.
    const std::vector<DxbcShaderTranslator::VertexInputElement>& GetVertexInputElements() const {
      return vertex_input_elements_;
    }

   private:
    friend class DxbcShaderTranslator;
    std::vector<DxbcShaderTranslator::VertexInputElement> vertex_input_elements_;
  };

  DxbcShader(xenos::ShaderType shader_type, uint64_t ucode_data_hash, const uint32_t* ucode_dwords,
             size_t ucode_dword_count, std::endian ucode_source_endian = std::endian::big);

  // Resource bindings are gathered after the successful translation of any
  // modification for simplicity of translation (and they don't depend on
  // modification bits).

  static constexpr uint32_t kMaxTextureBindingIndexBits =
      DxbcShaderTranslator::kMaxTextureBindingIndexBits;
  static constexpr uint32_t kMaxTextureBindings = DxbcShaderTranslator::kMaxTextureBindings;
  struct TextureBinding {
    uint32_t bindless_descriptor_index;
    uint32_t fetch_constant;
    // Stacked and 3D are separate TextureBindings, even for bindless for null
    // descriptor handling simplicity.
    xenos::FetchOpDimension dimension;
    bool is_signed;
  };
  // Safe to hash and compare with memcmp for layout hashing.
  const std::vector<TextureBinding>& GetTextureBindingsAfterTranslation() const {
    return texture_bindings_;
  }
  uint32_t GetUsedTextureMaskAfterTranslation() const { return used_texture_mask_; }

  static constexpr uint32_t kMaxSamplerBindingIndexBits =
      DxbcShaderTranslator::kMaxSamplerBindingIndexBits;
  static constexpr uint32_t kMaxSamplerBindings = DxbcShaderTranslator::kMaxSamplerBindings;
  struct SamplerBinding {
    uint32_t bindless_descriptor_index;
    uint32_t fetch_constant;
    xenos::TextureFilter mag_filter;
    xenos::TextureFilter min_filter;
    xenos::TextureFilter mip_filter;
    xenos::AnisoFilter aniso_filter;
  };
  const std::vector<SamplerBinding>& GetSamplerBindingsAfterTranslation() const {
    return sampler_bindings_;
  }

  // [fetch-need] What the translations of this shader read from the fetch
  // constants buffer, over every modification translated so far. A bit per
  // two-dword pair of the 96 (texture fetch constant * 3 + pair) that a
  // texture fetch reads; vertex fetch constants are not in it (a raw vertex
  // fetch reads the ones in constant_register_map().vertex_fetch_bitmap, an
  // input-assembler variant none unless IaFetchReadsConstant()).
  void GetTextureFetchPairsRead(uint32_t pairs_out[3]) const {
    for (uint32_t i = 0; i < 3; ++i) {
      pairs_out[i] = texture_fetch_pairs_read_[i].load(std::memory_order_relaxed);
    }
  }
  bool IaFetchReadsConstant() const {
    return ia_fetch_reads_constant_.load(std::memory_order_relaxed);
  }

 protected:
  Translation* CreateTranslationInstance(uint64_t modification) override;

 private:
  friend class DxbcShaderTranslator;

  std::atomic_flag bindings_setup_entered_ = ATOMIC_FLAG_INIT;
  std::vector<TextureBinding> texture_bindings_;
  std::vector<SamplerBinding> sampler_bindings_;
  uint32_t used_texture_mask_ = 0;
  // [fetch-need] Written by PostTranslation of any modification, on any thread.
  std::atomic<uint32_t> texture_fetch_pairs_read_[3]{};
  std::atomic<bool> ia_fetch_reads_constant_{false};
};

}  // namespace rex::graphics
