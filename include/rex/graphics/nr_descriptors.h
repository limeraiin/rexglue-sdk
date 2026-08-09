/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_DESCRIPTORS_H_
#define REX_GRAPHICS_NR_DESCRIPTORS_H_

#include <cstdint>

// [NR-DSC] The descriptor/sampler mirror: native-renderer phase 5,
// increment 5-3b-2.
//
// 5-3b-0 proved our packing of the four guest constant buffers; 5-3b-1 proved
// the system constants. What UpdateBindings still owns exclusively is the
// DESCRIPTOR side: which sampler each draw gets (SamplerParameters, derived
// from the fetch constants), which texture identity each SRV slot refers to
// (the TextureSRVKey the validity checks compare), the bindless sampler-heap
// index bookkeeping, and the bytes of the two descriptor-indices constant
// buffers. This module owns our transcription of each:
//
//   DescSamplerParams   GetSamplerParameters from the RAW fetch dwords:
//                       clamp normalization (D3D12 rules), border-color
//                       gating, mip min/max subresource clamping, the
//                       binding/fetch filter override lattice, the
//                       anisotropic_override cvar rule, the packed 22-bit
//                       SamplerParameters value.
//   DescTextureSrvKey   BindingInfoFromFetchConstant + GuestToHostSwizzle +
//                       SwizzleSigns from the RAW fetch dwords: the 16-byte
//                       TextureKey bit layout, invalid-fetch semantics, the
//                       1D refusal rules, base/mip page interlock, base
//                       format folding. The HOST FORMAT swizzle table is a
//                       declared query (callback): it is static config of the
//                       D3D12 backend and belongs to the later format peel.
//   DescSamplerMap      A mirror of the texture cache's bindless sampler-heap
//                       map {params -> heap index} + its allocation counter.
//                       Entries allocated before the gate armed are SEEDED
//                       from the emulated side (counted, like 5-3b-1's seed);
//                       allocations while armed are PREDICTED and compared.
//   DescComposeIndices  The bytes of a descriptor-indices cbuffer from
//                       (slot, value) pairs. Texture SRV index VALUES are
//                       pass-through queries of the texture cache (their
//                       derivation is the texture/residency peel); what this
//                       increment owns is the placement, the unbounded-SRV
//                       base offset, and the sampler indices from OUR map.
//
// The gates run inside UpdateBindings on the CP thread, from the same
// repointed register file, at the moment the emulated side derives the same
// thing ([[probe-reads-the-wrong-moment]]). The descriptor-indices compare
// stages the emulated write in cached memory first - never read an upload
// heap back ([[upload-heap-readback-trap]]).
//
// Pure functions, no globals, no SDK dependencies: raw xenos enum values are
// transcribed as plain constants below so tools/nr-descriptors-test.cpp
// builds this file bare.

namespace rex {
namespace graphics {
namespace nr {

// Raw xenos enum values used by the transcriptions (asserted against the SDK
// enums at the gate site).
constexpr uint32_t kDescFetchTypeInvalidTexture = 0;  // FetchConstantType
constexpr uint32_t kDescFetchTypeTexture = 2;
constexpr uint32_t kDescDim1D = 0;  // DataDimension
constexpr uint32_t kDescDim2DOrStacked = 1;
constexpr uint32_t kDescDim3D = 2;
constexpr uint32_t kDescDimCube = 3;
constexpr uint32_t kDescClampToEdge = 2;  // ClampMode
constexpr uint32_t kDescClampToHalfway = 4;
constexpr uint32_t kDescMirrorClampToEdge = 3;
constexpr uint32_t kDescMirrorClampToHalfway = 5;
constexpr uint32_t kDescClampToBorder = 6;
constexpr uint32_t kDescMirrorClampToBorder = 7;
constexpr uint32_t kDescFilterPoint = 0;  // TextureFilter
constexpr uint32_t kDescFilterLinear = 1;
constexpr uint32_t kDescFilterBaseMap = 2;
constexpr uint32_t kDescFilterUseFetchConst = 3;
constexpr uint32_t kDescAnisoDisabled = 0;  // AnisoFilter
constexpr uint32_t kDescAnisoMax16 = 5;
constexpr uint32_t kDescAnisoUseFetchConst = 7;
// The null-binding constants WriteActiveTextureSRVKeys emits.
constexpr uint32_t kDescHostSwizzle0000 = 0x924;  // XE_GPU_TEXTURE_SWIZZLE_0000
constexpr uint8_t kDescSwizzledSignsUnsigned = 0;
// 2D/cube max width/height (the 1D width refusal bound).
constexpr uint32_t kDescMaxWidthHeight2D = 8192;

// Our packing of D3D12TextureCache::SamplerParameters (bit-layout mirror,
// low 22 bits of one dword; compared against theirs' .value whole).
// binding_*_filter are the shader binding's overrides (TextureFilter raw,
// kDescFilterUseFetchConst = defer to the fetch constant; AnisoFilter raw,
// kDescAnisoUseFetchConst likewise); anisotropic_override is the cvar,
// read by the caller at the same moment the emulated side reads it.
uint32_t DescSamplerParams(const uint32_t fetch_dwords[6],
                           uint32_t binding_mag_filter,
                           uint32_t binding_min_filter,
                           uint32_t binding_mip_filter,
                           uint32_t binding_aniso_filter,
                           int32_t anisotropic_override);

// Our mirror of D3D12TextureCache::TextureSRVKey: the 16-byte TextureKey bit
// layout (padding zeroed, like MakeInvalid), the merged host swizzle and the
// swizzled signs. An invalid fetch produces the exact null-binding triple
// (zeroed key, swizzle 0000, unsigned signs).
struct DescTexSrvKey {
  uint32_t key[4];
  uint32_t host_swizzle;
  uint8_t swizzled_signs;
};

// Host-format swizzle query: 12-bit swizzle for a BASE TextureFormat value
// (static config of the D3D12 backend - a declared input of this increment).
typedef uint32_t (*DescHostFormatSwizzleFn)(void* ctx, uint32_t base_format);

void DescTextureSrvKey(const uint32_t fetch_dwords[6],
                       bool allow_invalid_fetch_constants,
                       DescHostFormatSwizzleFn host_swizzle_fn, void* fn_ctx,
                       DescTexSrvKey* out);

// The bindless sampler-heap mirror: {SamplerParameters value -> heap index}
// plus the allocation counter. Open-addressed, bounded probes, overflow
// counted (an overflowed mirror refuses, never lies).
struct DescSamplerMap {
  static constexpr uint32_t kSize = 512;  // power of two
  static constexpr uint32_t kProbes = 16;
  uint32_t keys[kSize];
  uint32_t index[kSize];
  uint8_t used[kSize];
  uint32_t count;
  uint32_t allocated;  // mirror of sampler_bindless_heap_allocated_
  uint32_t ovf;
};

// allocated_seed = the emulated allocation counter at the moment of the reset
// (gate arm or heap switch); entries are then learned/predicted from there.
void DescSamplerMapReset(DescSamplerMap* map, uint32_t allocated_seed);

enum DescSamplerVerdict : uint32_t {
  kDescSamplerMatch = 0,     // known params, index agrees
  kDescSamplerFresh = 1,     // new params, allocation index PREDICTED right
  kDescSamplerSeeded = 2,    // new params with a pre-arm index - learned
  kDescSamplerMismatch = 3,  // disagreement (mirror re-synced from theirs)
  kDescSamplerOverflow = 4,  // mirror full - observation refused
};

// Replays one resolved sampler in the emulated resolution order.
DescSamplerVerdict DescSamplerMapObserve(DescSamplerMap* map,
                                         uint32_t params_value,
                                         uint32_t their_index);

// Compose-time lookup; false = params unknown (the caller refuses + counts).
bool DescSamplerMapLookup(const DescSamplerMap* map, uint32_t params_value,
                          uint32_t* index_out);

// Builds the descriptor-indices cbuffer contents over span_dwords dwords:
// zero-filled, then each (slot, value) placed at out[slot]. Returns
// span_dwords on success; UINT32_MAX when it must refuse (capacity or a slot
// outside [0, span) - impossible if the caller derived the span right,
// counted either way).
//
// ⚠ The slot space is 1-BASED: the translator emplaces a binding BEFORE
// assigning bindless_descriptor_index = GetBindlessResourceCount(), so the
// count includes the new binding itself and a shader with n bindings uses
// slots 1..n (slot 0 is never written). The emulated rebuild requests only
// n dwords and its slot-n write lands in the 256-byte pool alignment slack;
// the real extent - and the span a native UpdateBindings must allocate - is
// max_slot + 1 dwords.
uint32_t DescComposeIndices(const uint32_t* tex_slots,
                            const uint32_t* tex_values, uint32_t tex_count,
                            const uint32_t* smp_slots,
                            const uint32_t* smp_values, uint32_t smp_count,
                            uint32_t span_dwords, uint32_t* out_dwords,
                            uint32_t out_cap_dwords);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_DESCRIPTORS_H_
