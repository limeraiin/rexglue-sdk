/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_RESIDENCY_H_
#define REX_GRAPHICS_NR_RESIDENCY_H_

#include <cstdint>

// [NR-RSY] Vertex/index residency + the live texture SRV descriptor
// allocation: native-renderer phase 5, increment 5-3b-3.
//
// 5-3b-2 closed the descriptor CONTENTS (sampler parameters, texture SRV
// keys, sampler-heap indices, descriptor-indices cbuffer bytes) but left the
// texture SRV index VALUES as pass-through queries and never touched the
// draw's shared-memory residency requests. This module owns our transcription
// of each remaining piece:
//
//   ResVfetchMirror   The command processor's vertex-buffer residency state:
//                     the 96-slot sync bitmap + per-slot {address, size}
//                     cache, its invalidation semantics (per-fetch-constant
//                     write, whole clears, the swap bits-only clear), and a
//                     per-draw walk of the residency loop that predicts the
//                     exact ordered RequestRange argument list from the draw
//                     register file's raw fetch dwords + the vertex shader's
//                     fetch bitmap.
//   ResIndexPredict   The primitive processor's index-buffer request:
//                     base = VGT_DMA_BASE masked to the index size,
//                     length = min(num_indices, DMA words) << size_log2,
//                     gated on source_select == kDMA with the physical-bounds
//                     refusal. WHETHER the request fires also depends on the
//                     primitive processor's conversion decisions (reset-index
//                     scan, primitive-type conversion) - that decision is a
//                     declared input (the result's index_buffer_type); what
//                     is ours is the argument derivation.
//   ResViewPool       A mirror of the bindless view-descriptor heap
//                     allocator: allocation counter + free stack (LIFO),
//                     seeded from the emulated state at gate arm, predicting
//                     the index of EVERY allocation (persistent texture
//                     descriptors and one-use per-submission views share this
//                     pool) and following releases.
//   ResTexDescMap     A mirror of the per-texture SRV descriptor maps:
//                     {texture identity, SRVDescriptorKey} -> heap index,
//                     find-or-create semantics (fresh allocations predicted
//                     through the pool mirror, pre-arm entries learned as
//                     seeded, entries evicted on descriptor release).
//   ResSrvIndexForBinding
//                     The GetActiveTextureBindlessSRVIndex decision tree:
//                     signedness selection (with the separate-signed-version
//                     rule), dimension compatibility, the 3D-sampled-as-2D
//                     special view (refused + counted - it allocates at query
//                     time), and the null-view fallbacks per fetch dimension.
//                     Which texture OBJECT a binding resolves to is the
//                     texture cache's own (a declared input); the index a
//                     binding's cbuffer value carries is ours.
//
// Pure functions + POD state, no globals, no SDK dependencies: raw xenos
// values are transcribed as plain constants below so
// tools/nr-residency-test.cpp builds this file bare. Every enum value and
// bit layout is pinned at the gate site (static_asserts + a one-shot
// bit-layout self-check at every gate arm).

namespace rex {
namespace graphics {
namespace nr {

// Raw xenos enum values used by the transcriptions (asserted at the gate).
constexpr uint32_t kResFetchTypeInvalidVertex = 1;  // FetchConstantType
constexpr uint32_t kResFetchTypeVertex = 3;
constexpr uint32_t kResSourceSelectDma = 0;  // SourceSelect
constexpr uint32_t kResIndexFormatInt16 = 0;  // IndexFormat
constexpr uint32_t kResIndexFormatInt32 = 1;
constexpr uint32_t kResFetchDim1D = 0;  // FetchOpDimension
constexpr uint32_t kResFetchDim2D = 1;
constexpr uint32_t kResFetchDim3DOrStacked = 2;
constexpr uint32_t kResFetchDimCube = 3;
constexpr uint32_t kResDataDim1D = 0;  // DataDimension
constexpr uint32_t kResDataDim2DOrStacked = 1;
constexpr uint32_t kResDataDim3D = 2;
constexpr uint32_t kResDataDimCube = 3;

constexpr uint32_t kResVfetchSlots = 96;
constexpr uint32_t kResInvalidIndex = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Vertex-buffer residency.

struct ResRange {
  uint32_t start;
  uint32_t length;
};

// Byte-for-byte the command processor's state: in_sync bit set = this slot's
// last-checked fetch constant is known resident; state = the last requested
// {address dwords, size words} per slot (kResInvalidIndex = never requested).
struct ResVfetchMirror {
  uint64_t in_sync[2];
  uint32_t state_address[kResVfetchSlots];
  uint32_t state_size[kResVfetchSlots];
};

// InvalidateAllVertexBufferResidency: bits + states.
void ResVfetchInvalidateAll(ResVfetchMirror* m);
// InvalidateVertexBufferResidency: one bit, out-of-range is a no-op.
void ResVfetchInvalidateOne(ResVfetchMirror* m, uint32_t vfetch_index);
// InvalidateVertexBufferResidencyRange: swapped bounds normalized, clamped.
void ResVfetchInvalidateRange(ResVfetchMirror* m, uint32_t first_vfetch,
                              uint32_t last_vfetch);
// The IssueSwap clear: sync bits only, per-slot states retained.
void ResVfetchClearSyncBits(ResVfetchMirror* m);

enum ResVfetchStatus : uint32_t {
  kResVfetchOk = 0,
  // A used slot's fetch type fails the draw (completely invalid, or
  // kInvalidVertex without the allow cvar). Requests up to that slot are
  // still emitted, then the emulated loop returns false.
  kResVfetchAbortInvalidType,
};

// Walks the residency loop WITHOUT mutating the mirror. bitmap = the vertex
// shader's vertex_fetch_bitmap (3 dwords, 96 slots ascending); fetch_regs =
// the 192 raw vertex-fetch dwords of the draw register file; allow_invalid
// mirrors gpu_allow_invalid_fetch_constants. Fills the RequestRange argument
// list (byte units, <<2 applied) in loop order.
ResVfetchStatus ResVfetchPredict(const ResVfetchMirror* m,
                                 const uint32_t bitmap[3],
                                 const uint32_t* fetch_regs, bool allow_invalid,
                                 ResRange* out_requests, uint32_t out_cap,
                                 uint32_t* out_count);

// Applies the loop's mutations to the mirror: skip/state-match slots update
// their bits exactly as the loop does; the first `requests_succeeded` emitted
// requests commit {state, bit}; the next emitted request (if any) is the
// failed one - the emulated loop returns there, so the walk stops.
void ResVfetchApply(ResVfetchMirror* m, const uint32_t bitmap[3],
                    const uint32_t* fetch_regs, bool allow_invalid,
                    uint32_t requests_succeeded);

// ---------------------------------------------------------------------------
// Index-buffer residency.

struct ResIndexPrediction {
  // source_select == kDMA: the draw names a guest index buffer. Whether the
  // primitive processor actually REQUESTS it (vs converting to a host copy)
  // is its own decision - the caller cross-checks against the result type.
  bool dma;
  // The primitive processor refuses the draw (physical bounds).
  bool out_of_bounds;
  uint32_t base;
  uint32_t length;
};

// vgt_draw_initiator/vgt_dma_size/vgt_dma_base = the raw register dwords;
// buffer_size = SharedMemory::kBufferSize.
void ResIndexPredict(uint32_t vgt_draw_initiator, uint32_t vgt_dma_size,
                     uint32_t vgt_dma_base, uint32_t buffer_size,
                     ResIndexPrediction* out);

// ---------------------------------------------------------------------------
// The bindless view-descriptor pool mirror.

struct ResViewPool {
  static constexpr uint32_t kFreeCap = 8192;
  uint32_t free_stack[kFreeCap];
  uint32_t free_count;
  uint32_t allocated;
  uint32_t heap_size;
  // The free stack overflowed since the seed: the mirror refuses predictions
  // (never lies) until the next reseed.
  uint8_t refused;
};

void ResViewPoolReset(ResViewPool* p, uint32_t allocated_seed,
                      const uint32_t* free_seed, uint32_t free_seed_count,
                      uint32_t heap_size);
// The index the NEXT allocation must return. false = the mirror refuses
// (overflowed) or predicts exhaustion (out_index = kResInvalidIndex then).
bool ResViewPoolPredictAlloc(const ResViewPool* p, uint32_t* out_index);
// true = actual matched the prediction and the mirror advanced; false = a
// divergence (the caller counts it and reseeds from the emulated state).
bool ResViewPoolObserveAlloc(ResViewPool* p, uint32_t actual_index);
void ResViewPoolObserveRelease(ResViewPool* p, uint32_t index);

// ---------------------------------------------------------------------------
// The per-texture SRV descriptor map mirror.

// Mirror of D3D12Texture::SRVDescriptorKey's bit layout (pinned at the gate):
// is_signed:1 | host_swizzle:12 | dimension:2.
inline uint32_t ResSrvDescriptorKey(bool is_signed, uint32_t host_swizzle,
                                    uint32_t dimension) {
  return (is_signed ? 1u : 0u) | (host_swizzle & 0xFFFu) << 1 |
         (dimension & 3u) << 13;
}

struct ResTexDescMap {
  static constexpr uint32_t kSize = 4096;  // power of two
  static constexpr uint32_t kProbes = 24;
  uint64_t keys[kSize];
  uint32_t index[kSize];
  uint8_t state[kSize];  // 0 empty, 1 used, 2 tombstone
  uint32_t count;
  uint32_t ovf;
};

void ResTexDescMapReset(ResTexDescMap* m);

enum ResTexDescVerdict : uint32_t {
  kResTexDescMatch = 0,   // known entry, index agrees
  kResTexDescFresh = 1,   // new entry created while armed (pool-predicted)
  kResTexDescSeeded = 2,  // existing emulated entry first seen - learned
  kResTexDescMismatch = 3,  // disagreement (entry re-synced from theirs)
  kResTexDescOverflow = 4,  // map full - observation refused
};

// Replays one FindOrCreateTextureDescriptor call: their_hit = the emulated
// per-texture map already had the key (no allocation happened). A
// their_index of kResInvalidIndex (creation refused: unsupported format,
// exhaustion) is LEARNED so later lookups resolve it to the null fallback.
ResTexDescVerdict ResTexDescObserve(ResTexDescMap* m, uint64_t texture_handle,
                                    uint32_t srv_key, bool their_hit,
                                    uint32_t their_index);
// Compose-time lookup; false = the pair was never observed.
bool ResTexDescLookup(const ResTexDescMap* m, uint64_t texture_handle,
                      uint32_t srv_key, uint32_t* out_index);
// A released descriptor index invalidates every entry holding it (the
// texture is being destroyed; its handle may be reused).
void ResTexDescEvictIndex(ResTexDescMap* m, uint32_t index);

// ---------------------------------------------------------------------------
// The GetActiveTextureBindlessSRVIndex decision tree.

// AreDimensionsCompatible, transcribed (16-combo self-checked at gate arm).
bool ResDimensionsCompatible(uint32_t fetch_op_dimension,
                             uint32_t data_dimension);

// Declared inputs, extracted from the texture cache's resolved binding state
// (which texture object a fetch constant resolves to is the texture cache's
// own; host_swizzle and the signs are byte-proven ours since 5-3b-2).
struct ResSrvBindingFacts {
  uint8_t has_binding;        // GetValidTextureBinding != null
  uint8_t binding_dimension;  // DataDimension of the bound TextureKey
  uint8_t signed_separate;    // IsSignedVersionSeparateForFormat(key)
  uint8_t any_sign_signed;
  uint8_t any_sign_not_signed;
  uint32_t host_swizzle;
  uint64_t texture_handle;         // 0 = none
  uint64_t texture_signed_handle;  // 0 = none
};

enum ResSrvOutcome : uint32_t {
  kResSrvValue = 0,  // *out_index = a live descriptor index from the map
  kResSrvNull,       // *out_index = the null view for the fetch dimension
  // 3D resource sampled as 1D/2D: the emulated query builds a special view
  // (allocating at query time) - refused + counted unless already mapped.
  kResSrvRefuseSpecialView,
  // A live texture whose descriptor the map never observed (pre-seed state).
  kResSrvRefuseUnknown,
};

ResSrvOutcome ResSrvIndexForBinding(const ResTexDescMap* map,
                                    const ResSrvBindingFacts* facts,
                                    uint32_t fetch_op_dimension,
                                    bool host_shader_is_signed,
                                    uint32_t null_2darray, uint32_t null_3d,
                                    uint32_t null_cube, uint32_t* out_index);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_RESIDENCY_H_
