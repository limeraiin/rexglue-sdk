/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <rex/graphics/nr_descriptors.h>

#include <cstring>

namespace rex {
namespace graphics {
namespace nr {

namespace {

// Raw bitfield reads from the six fetch-constant dwords (the
// xe_gpu_texture_fetch_t layout, transcribed - the register file holds these
// in host order).
inline uint32_t Bits(uint32_t dword, uint32_t shift, uint32_t width) {
  return (dword >> shift) & ((1u << width) - 1u);
}

// GetClampModesForDimension: cube ignores the fetch clamps entirely; lower
// dimensions take exactly as many axes as they have, the rest default to
// clamp-to-edge.
void FetchClampModes(const uint32_t dw[6], uint32_t* cx, uint32_t* cy,
                     uint32_t* cz) {
  *cx = kDescClampToEdge;
  *cy = kDescClampToEdge;
  *cz = kDescClampToEdge;
  const uint32_t dimension = Bits(dw[5], 9, 2);
  switch (dimension) {
    case kDescDim3D:
      *cz = Bits(dw[0], 16, 3);
      [[fallthrough]];
    case kDescDim2DOrStacked:
      *cy = Bits(dw[0], 13, 3);
      [[fallthrough]];
    case kDescDim1D:
      *cx = Bits(dw[0], 10, 3);
      break;
    default:
      break;
  }
}

// The D3D12 backend's clamp normalization (modes with no D3D12 equivalent
// collapse onto their nearest supported neighbour).
uint32_t NormalizeClamp(uint32_t clamp) {
  if (clamp == kDescClampToHalfway) {
    return kDescClampToEdge;
  }
  if (clamp == kDescMirrorClampToHalfway || clamp == kDescMirrorClampToBorder) {
    return kDescMirrorClampToEdge;
  }
  return clamp;
}

inline bool ClampUsesBorder(uint32_t clamp) {
  return clamp == kDescClampToBorder || clamp == kDescMirrorClampToBorder;
}

inline uint32_t Log2Floor(uint32_t v) {
  uint32_t r = 0;
  while (v >>= 1) {
    ++r;
  }
  return r;
}

// GetSubresourcesFromFetchConstant, whole: sizes by dimension, then the
// base/mip page + mip level interlock (a missing mip chain forces levels to
// 0; a mip chain starting past level 0 drops the base page).
struct FetchSubresources {
  uint32_t width_minus_1;
  uint32_t height_minus_1;
  uint32_t depth_or_array_size_minus_1;
  uint32_t base_page;
  uint32_t mip_page;
  uint32_t mip_min_level;
  uint32_t mip_max_level;
};

void FetchGetSubresources(const uint32_t dw[6], FetchSubresources* out) {
  uint32_t width_minus_1 = 0;
  uint32_t height_minus_1 = 0;
  uint32_t depth_or_array_size_minus_1 = 0;
  const uint32_t dimension = Bits(dw[5], 9, 2);
  const uint32_t stacked = Bits(dw[1], 10, 1);
  switch (dimension) {
    case kDescDim1D:
      width_minus_1 = Bits(dw[2], 0, 24);
      break;
    case kDescDim2DOrStacked:
      width_minus_1 = Bits(dw[2], 0, 13);
      height_minus_1 = Bits(dw[2], 13, 13);
      depth_or_array_size_minus_1 = stacked ? Bits(dw[2], 26, 6) : 0;
      break;
    case kDescDim3D:
      width_minus_1 = Bits(dw[2], 0, 11);
      height_minus_1 = Bits(dw[2], 11, 11);
      depth_or_array_size_minus_1 = Bits(dw[2], 22, 10);
      break;
    case kDescDimCube:
      width_minus_1 = Bits(dw[2], 0, 13);
      height_minus_1 = Bits(dw[2], 13, 13);
      depth_or_array_size_minus_1 = 5;
      break;
  }
  out->width_minus_1 = width_minus_1;
  out->height_minus_1 = height_minus_1;
  out->depth_or_array_size_minus_1 = depth_or_array_size_minus_1;

  uint32_t longest_axis_minus_1 =
      width_minus_1 > height_minus_1 ? width_minus_1 : height_minus_1;
  if (dimension == kDescDim3D &&
      depth_or_array_size_minus_1 > longest_axis_minus_1) {
    longest_axis_minus_1 = depth_or_array_size_minus_1;
  }
  const uint32_t size_mip_max_level = Log2Floor(longest_axis_minus_1 + 1u);

  uint32_t base_page = Bits(dw[1], 12, 20) & 0x1FFFF;
  uint32_t mip_page = Bits(dw[5], 12, 20) & 0x1FFFF;

  uint32_t mip_min_level, mip_max_level;
  if (mip_page == 0) {
    mip_min_level = 0;
    mip_max_level = 0;
  } else {
    mip_min_level = Bits(dw[4], 2, 4);
    if (mip_min_level > size_mip_max_level) {
      mip_min_level = size_mip_max_level;
    }
    mip_max_level = Bits(dw[4], 6, 4);
    if (mip_max_level > size_mip_max_level) {
      mip_max_level = size_mip_max_level;
    }
    if (mip_max_level < mip_min_level) {
      mip_max_level = mip_min_level;
    }
  }
  if (mip_max_level != 0) {
    if (base_page == 0 && mip_min_level < 1) {
      mip_min_level = 1;
    }
    if (mip_min_level != 0) {
      base_page = 0;
    }
  } else {
    mip_page = 0;
  }

  out->base_page = base_page;
  out->mip_page = mip_page;
  out->mip_min_level = mip_min_level;
  out->mip_max_level = mip_max_level;
}

// GetBaseFormat: the resampling/gamma alias formats fold onto their base.
uint32_t BaseFormat(uint32_t format) {
  switch (format) {
    case 27:  // k_16_EXPAND
      return 30;  // k_16_FLOAT
    case 28:  // k_16_16_EXPAND
      return 31;  // k_16_16_FLOAT
    case 29:  // k_16_16_16_16_EXPAND
      return 32;  // k_16_16_16_16_FLOAT
    case 50:  // k_8_8_8_8_AS_16_16_16_16
      return 6;  // k_8_8_8_8
    case 51:  // k_DXT1_AS_16_16_16_16
      return 18;  // k_DXT1
    case 52:  // k_DXT2_3_AS_16_16_16_16
      return 19;  // k_DXT2_3
    case 53:  // k_DXT4_5_AS_16_16_16_16
      return 20;  // k_DXT4_5
    case 54:  // k_2_10_10_10_AS_16_16_16_16
      return 7;  // k_2_10_10_10
    case 55:  // k_10_11_11_AS_16_16_16_16
      return 16;  // k_10_11_11
    case 56:  // k_11_11_10_AS_16_16_16_16
      return 17;  // k_11_11_10
    case 62:  // k_8_8_8_8_GAMMA_EDRAM
      return 6;  // k_8_8_8_8
    default:
      return format;
  }
}

// SwizzleSigns: per destination component, a constant selector (4/5) defers
// to the constants' collective sign; a data selector reads that DATA
// component's sign bits from dword 0.
uint8_t FetchSwizzledSigns(const uint32_t dw[6]) {
  const uint32_t swizzle = Bits(dw[3], 1, 12);
  uint8_t signs = 0;
  bool any_not_signed = false, any_signed = false;
  uint8_t constant_mask = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t component = (swizzle >> (i * 3)) & 0b111;
    if (component & 0b100) {
      constant_mask |= uint8_t(1) << (i * 2);
    } else {
      const uint32_t sign = (dw[0] >> (2 + component * 2)) & 0b11;
      signs |= uint8_t(sign) << (i * 2);
      if (sign == 1) {  // kSigned
        any_signed = true;
      } else {
        any_not_signed = true;
      }
    }
  }
  uint32_t constants_sign = 0;  // kUnsigned
  if (constant_mask == 0b01010101) {
    if (((dw[0] >> 2) & 0xFF) == 1u * 0b01010101) {  // all four kSigned
      constants_sign = 1;
    }
  } else {
    if (any_signed && !any_not_signed) {
      constants_sign = 1;
    }
  }
  signs |= uint8_t(constants_sign * constant_mask);
  return signs;
}

// GuestToHostSwizzle: destination selection through the host format's own
// component mapping; constant selectors pass through (with the undefined 6/7
// encodings folded onto 0/1).
uint32_t MergeHostSwizzle(uint32_t guest_swizzle, uint32_t host_format_swizzle) {
  uint32_t host_swizzle = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t guest_component = (guest_swizzle >> (3 * i)) & 0b111;
    uint32_t host_component;
    if (guest_component >= 4) {  // XE_GPU_TEXTURE_SWIZZLE_0
      host_component = guest_component & 0b101;
    } else {
      host_component = (host_format_swizzle >> (3 * guest_component)) & 0b111;
    }
    host_swizzle |= host_component << (3 * i);
  }
  return host_swizzle;
}

// The open-addressed probe hash (splitmix64 finalizer, as nr_bindings).
inline uint32_t MapSlot(uint32_t key) {
  uint64_t h = uint64_t(key) + 0x9E3779B97F4A7C15ull;
  h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
  h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
  h ^= h >> 31;
  return uint32_t(h) & (DescSamplerMap::kSize - 1);
}

}  // namespace

uint32_t DescSamplerParams(const uint32_t fetch_dwords[6],
                           uint32_t binding_mag_filter,
                           uint32_t binding_min_filter,
                           uint32_t binding_mip_filter,
                           uint32_t binding_aniso_filter,
                           int32_t anisotropic_override) {
  const uint32_t* dw = fetch_dwords;

  uint32_t fetch_clamp_x, fetch_clamp_y, fetch_clamp_z;
  FetchClampModes(dw, &fetch_clamp_x, &fetch_clamp_y, &fetch_clamp_z);
  const uint32_t clamp_x = NormalizeClamp(fetch_clamp_x);
  const uint32_t clamp_y = NormalizeClamp(fetch_clamp_y);
  const uint32_t clamp_z = NormalizeClamp(fetch_clamp_z);
  uint32_t border_color = 0;  // k_ABGR_Black
  if (ClampUsesBorder(clamp_x) || ClampUsesBorder(clamp_y) ||
      ClampUsesBorder(clamp_z)) {
    border_color = Bits(dw[5], 0, 2);
  }

  FetchSubresources sub;
  FetchGetSubresources(dw, &sub);
  const uint32_t mip_min_level = sub.mip_min_level;
  const bool has_mips = sub.mip_max_level > sub.mip_min_level;

  const uint32_t mag_filter = binding_mag_filter == kDescFilterUseFetchConst
                                  ? Bits(dw[3], 19, 2)
                                  : binding_mag_filter;
  const uint32_t min_filter = binding_min_filter == kDescFilterUseFetchConst
                                  ? Bits(dw[3], 21, 2)
                                  : binding_min_filter;
  const uint32_t mip_filter = binding_mip_filter == kDescFilterUseFetchConst
                                  ? Bits(dw[3], 23, 2)
                                  : binding_mip_filter;
  const bool min_mag_linear = mag_filter == kDescFilterLinear &&
                              min_filter == kDescFilterLinear;
  const bool mip_bilinear_or_trilinear = mip_filter == kDescFilterPoint ||
                                         mip_filter == kDescFilterLinear;
  const bool mip_base_map = mip_filter == kDescFilterBaseMap;

  uint32_t aniso_filter = binding_aniso_filter == kDescAnisoUseFetchConst
                              ? Bits(dw[3], 25, 3)
                              : binding_aniso_filter;
  if (anisotropic_override > -1 && anisotropic_override < 6 && has_mips &&
      !mip_base_map && min_mag_linear && mip_bilinear_or_trilinear) {
    aniso_filter = uint32_t(anisotropic_override);
  }
  if (aniso_filter > kDescAnisoMax16) {
    aniso_filter = kDescAnisoMax16;
  }
  uint32_t mag_linear, min_linear, mip_linear;
  if (aniso_filter != kDescAnisoDisabled) {
    mag_linear = 1;
    min_linear = 1;
    mip_linear = 1;
  } else {
    mag_linear = mag_filter == kDescFilterLinear;
    min_linear = min_filter == kDescFilterLinear;
    mip_linear = mip_filter == kDescFilterLinear;
  }

  // The SamplerParameters bit layout: clamp_x:3 clamp_y:3 clamp_z:3
  // border_color:2 mag_linear:1 min_linear:1 mip_linear:1 aniso_filter:3
  // mip_min_level:4 mip_base_map:1.
  return clamp_x | clamp_y << 3 | clamp_z << 6 | border_color << 9 |
         mag_linear << 11 | min_linear << 12 | mip_linear << 13 |
         aniso_filter << 14 | mip_min_level << 17 |
         uint32_t(mip_base_map) << 21;
}

void DescTextureSrvKey(const uint32_t fetch_dwords[6],
                       bool allow_invalid_fetch_constants,
                       DescHostFormatSwizzleFn host_swizzle_fn, void* fn_ctx,
                       DescTexSrvKey* out) {
  const uint32_t* dw = fetch_dwords;

  // The null-binding triple - every refusal path below leaves this.
  std::memset(out->key, 0, sizeof(out->key));
  out->host_swizzle = kDescHostSwizzle0000;
  out->swizzled_signs = kDescSwizzledSignsUnsigned;

  const uint32_t type = Bits(dw[0], 0, 2);
  if (type != kDescFetchTypeTexture &&
      !(type == kDescFetchTypeInvalidTexture && allow_invalid_fetch_constants)) {
    return;
  }

  FetchSubresources sub;
  FetchGetSubresources(dw, &sub);
  if (sub.base_page == 0 && sub.mip_page == 0) {
    return;
  }
  const uint32_t dimension = Bits(dw[5], 9, 2);
  const uint32_t tiled = Bits(dw[0], 31, 1);
  const uint32_t packed_mips = Bits(dw[5], 11, 1);
  if (dimension == kDescDim1D &&
      (sub.width_minus_1 >= kDescMaxWidthHeight2D || tiled || packed_mips)) {
    return;
  }

  const uint32_t format = BaseFormat(Bits(dw[1], 0, 6));
  const uint32_t endianness = Bits(dw[1], 6, 2);
  const uint32_t pitch = Bits(dw[0], 22, 9);

  // The TextureKey bit layout (98 bits over four dwords, padding zero):
  //   dw0: base_page:17 dimension:2 width_minus_1:13
  //   dw1: height_minus_1:13 tiled:1 packed_mips:1 mip_page:17
  //   dw2: depth_or_array_size_minus_1:10 pitch:9 mip_max_level:4 format:6
  //        endianness:2 signed_separate:1
  //   dw3: scaled_resolve:1 is_valid:1
  out->key[0] = sub.base_page | dimension << 17 | sub.width_minus_1 << 19;
  out->key[1] = sub.height_minus_1 | tiled << 13 | packed_mips << 14 |
                sub.mip_page << 15;
  out->key[2] = sub.depth_or_array_size_minus_1 | pitch << 10 |
                sub.mip_max_level << 19 | format << 23 | endianness << 29;
  out->key[3] = 1u << 1;  // is_valid (signed_separate/scaled_resolve stay 0)

  out->swizzled_signs = FetchSwizzledSigns(dw);
  out->host_swizzle = MergeHostSwizzle(Bits(dw[3], 1, 12),
                                       host_swizzle_fn(fn_ctx, format));
}

void DescSamplerMapReset(DescSamplerMap* map, uint32_t allocated_seed) {
  std::memset(map, 0, sizeof(*map));
  map->allocated = allocated_seed;
}

DescSamplerVerdict DescSamplerMapObserve(DescSamplerMap* map,
                                         uint32_t params_value,
                                         uint32_t their_index) {
  const uint32_t start = MapSlot(params_value);
  const uint32_t mask = DescSamplerMap::kSize - 1;
  uint32_t free_at = UINT32_MAX;
  for (uint32_t probe = 0; probe < DescSamplerMap::kProbes; ++probe) {
    const uint32_t at = (start + probe) & mask;
    if (!map->used[at]) {
      free_at = at;
      break;
    }
    if (map->keys[at] == params_value) {
      if (map->index[at] == their_index) {
        return kDescSamplerMatch;
      }
      map->index[at] = their_index;  // re-sync so one slip names itself once
      return kDescSamplerMismatch;
    }
  }
  if (free_at == UINT32_MAX) {
    ++map->ovf;
    return kDescSamplerOverflow;
  }
  map->used[free_at] = 1;
  map->keys[free_at] = params_value;
  map->index[free_at] = their_index;
  ++map->count;
  if (their_index == map->allocated) {
    // A fresh allocation, predicted: the emulated counter advanced with ours.
    ++map->allocated;
    return kDescSamplerFresh;
  }
  if (their_index < map->allocated) {
    // Allocated before this mirror was (re)seeded - learn it.
    return kDescSamplerSeeded;
  }
  // An index past the mirrored counter cannot happen if the transcription is
  // right; re-sync the counter so later observations stay meaningful.
  map->allocated = their_index + 1;
  return kDescSamplerMismatch;
}

bool DescSamplerMapLookup(const DescSamplerMap* map, uint32_t params_value,
                          uint32_t* index_out) {
  const uint32_t start = MapSlot(params_value);
  const uint32_t mask = DescSamplerMap::kSize - 1;
  for (uint32_t probe = 0; probe < DescSamplerMap::kProbes; ++probe) {
    const uint32_t at = (start + probe) & mask;
    if (!map->used[at]) {
      return false;
    }
    if (map->keys[at] == params_value) {
      *index_out = map->index[at];
      return true;
    }
  }
  return false;
}

uint32_t DescComposeIndices(const uint32_t* tex_slots,
                            const uint32_t* tex_values, uint32_t tex_count,
                            const uint32_t* smp_slots,
                            const uint32_t* smp_values, uint32_t smp_count,
                            uint32_t span_dwords, uint32_t* out_dwords,
                            uint32_t out_cap_dwords) {
  if (span_dwords > out_cap_dwords) {
    return UINT32_MAX;
  }
  for (uint32_t i = 0; i < tex_count; ++i) {
    if (tex_slots[i] >= span_dwords) {
      return UINT32_MAX;
    }
  }
  for (uint32_t i = 0; i < smp_count; ++i) {
    if (smp_slots[i] >= span_dwords) {
      return UINT32_MAX;
    }
  }
  std::memset(out_dwords, 0, span_dwords * sizeof(uint32_t));
  for (uint32_t i = 0; i < tex_count; ++i) {
    out_dwords[tex_slots[i]] = tex_values[i];
  }
  for (uint32_t i = 0; i < smp_count; ++i) {
    out_dwords[smp_slots[i]] = smp_values[i];
  }
  return span_dwords;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
