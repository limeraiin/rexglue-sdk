/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_BINDINGS_H_
#define REX_GRAPHICS_NR_BINDINGS_H_

#include <cstdint>

// [NR-BND] The bindings mirror: native-renderer phase 5, increment 5-3b-0.
//
// 5-3a gave the draw its own pipeline object; what still belongs entirely to
// the emulated path is UpdateBindings — the root signature, the guest
// constant buffers, the descriptors, the samplers and vertex/index residency.
// 5-3b peels that, and this first increment owns the part that is a pure
// function of the register file and the shaders' constant maps: the BYTES of
// the four guest constant buffers a draw binds
//
//   float VS   packed float4s the vertex shader's constant map references,
//              gathered from ALU file regs 0x4000+ (constants 0-255)
//   float PS   the same for the pixel shader, from 0x4400+ (constants 256-511)
//   bool/loop  40 dwords verbatim from 0x4900
//   fetch      192 dwords verbatim from 0x4800
//
// plus the root-signature SELECTION (bindless: one of two static objects by
// tessellation; bindful: an index packed from the four binding counts).
//
// The gate: our compose runs inside UpdateBindings right after the emulated
// upload, on the same thread, from the same repointed register file, and the
// uploaded bytes are memcmp'd against ours ([[probe-reads-the-wrong-moment]]:
// this is the only spot where both sides provably read the same file). What
// is NOT yet covered, by design, and stays delegated for later peels: the
// system-constants derivation (UpdateSystemConstantValues — its own
// increment, it is 5-1-sized), the bindless descriptor-indices cbuffers,
// sampler parameters, texture SRV keys, and residency requests.
//
// Pure functions, no globals, no SDK dependencies: tools/nr-bindings-test.cpp
// builds this file bare.

namespace rex {
namespace graphics {
namespace nr {

// First dword index of each composed window in the flat register file.
// 0x4000/0x4800/0x4900 agree with nr_resource's file map; the pixel float
// window is the upper half of the ALU file (constant 256 = dword 1024).
constexpr uint32_t kBindFloatVertexBase = 0x4000;
constexpr uint32_t kBindFloatPixelBase = 0x4400;
constexpr uint32_t kBindFetchBase = 0x4800;
constexpr uint32_t kBindBoolLoopBase = 0x4900;

constexpr uint32_t kBindFloatMaxBytes = 256 * 4 * sizeof(uint32_t);  // 4096
constexpr uint32_t kBindBoolLoopBytes = (8 + 32) * sizeof(uint32_t);  // 160
constexpr uint32_t kBindFetchBytes = 192 * sizeof(uint32_t);          // 768

// Packs the float constants a shader's map references, in the runtime's own
// order: bitmap words ascending, bits ascending within each word, 16 bytes
// per set bit, word w bit b -> dwords [base + (w << 8) + (b << 2), +4).
// Returns the byte size required; writes only when out_cap holds it (the
// caller compares byte-for-byte against the emulated upload, so a truncated
// compose must refuse rather than truncate).
uint32_t BindComposeFloats(const uint32_t* regs, uint32_t base,
                           const uint64_t bitmap[4], void* out,
                           uint32_t out_cap);

// The two verbatim windows, same contract.
uint32_t BindComposeBoolLoop(const uint32_t* regs, void* out, uint32_t out_cap);
uint32_t BindComposeFetch(const uint32_t* regs, void* out, uint32_t out_cap);

// The bindful root-signature key, transcribed from the runtime's packing
// (pixel counts in the low bits, then vertex, then the tessellation bit).
// Bit widths are passed in so this file needs no D3D12 headers; the caller
// hands it D3D12Shader::kMaxTextureBindingIndexBits / kMaxSamplerBindingIndexBits.
uint32_t BindRootSigBindfulIndex(uint32_t texture_count_pixel,
                                 uint32_t sampler_count_pixel,
                                 uint32_t texture_count_vertex,
                                 uint32_t sampler_count_vertex,
                                 bool tessellated, uint32_t texture_index_bits,
                                 uint32_t sampler_index_bits);

// Distinct-key census for the binding-layout UIDs (the bookkeeping a native
// UpdateBindings must own: descriptor layouts are keyed by the UIDs the
// pipeline cache assigns per translation). Open-addressed, bounded probes,
// overflow counted — an ovf'd census reads as a lower bound, never silently.
struct BindCensus {
  static constexpr uint32_t kSize = 4096;  // power of two
  static constexpr uint32_t kProbes = 16;
  uint64_t keys[kSize];
  uint8_t used[kSize];
  uint32_t count;
  uint32_t ovf;
};

void BindCensusReset(BindCensus* census);
// Adds a key (0 is a valid key); returns true when newly counted.
bool BindCensusAdd(BindCensus* census, uint64_t key);

// Mixes the four layout UIDs of a draw into one census key.
uint64_t BindLayoutKey(uint64_t texture_uid_vertex, uint64_t sampler_uid_vertex,
                       uint64_t texture_uid_pixel, uint64_t sampler_uid_pixel);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_BINDINGS_H_
