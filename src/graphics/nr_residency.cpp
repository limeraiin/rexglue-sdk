/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <rex/graphics/nr_residency.h>

#include <cstring>

namespace rex {
namespace graphics {
namespace nr {

namespace {

inline bool BitScanForward32(uint32_t v, uint32_t* out) {
  if (!v) {
    return false;
  }
  uint32_t i = 0;
  while (!(v & 1u)) {
    v >>= 1;
    ++i;
  }
  *out = i;
  return true;
}

// xe_gpu_vertex_fetch_t, raw: dword0 = type:2 | address:30 (dwords);
// dword1 = endian:2 | size:24 (words) | pad:6.
inline uint32_t VfetchType(uint32_t dword_0) { return dword_0 & 3u; }
inline uint32_t VfetchAddress(uint32_t dword_0) { return dword_0 >> 2; }
inline uint32_t VfetchSize(uint32_t dword_1) { return (dword_1 >> 2) & 0xFFFFFFu; }

}  // namespace

// ---------------------------------------------------------------------------
// Vertex-buffer residency.

void ResVfetchInvalidateAll(ResVfetchMirror* m) {
  m->in_sync[0] = 0;
  m->in_sync[1] = 0;
  for (uint32_t i = 0; i < kResVfetchSlots; ++i) {
    m->state_address[i] = kResInvalidIndex;
    m->state_size[i] = kResInvalidIndex;
  }
}

void ResVfetchInvalidateOne(ResVfetchMirror* m, uint32_t vfetch_index) {
  if (vfetch_index >= kResVfetchSlots) {
    return;
  }
  m->in_sync[vfetch_index >> 6] &= ~(uint64_t(1) << (vfetch_index & 63));
}

void ResVfetchInvalidateRange(ResVfetchMirror* m, uint32_t first_vfetch,
                              uint32_t last_vfetch) {
  if (first_vfetch > last_vfetch) {
    uint32_t t = first_vfetch;
    first_vfetch = last_vfetch;
    last_vfetch = t;
  }
  if (first_vfetch >= kResVfetchSlots) {
    return;
  }
  if (last_vfetch > kResVfetchSlots - 1) {
    last_vfetch = kResVfetchSlots - 1;
  }
  for (uint32_t i = first_vfetch; i <= last_vfetch; ++i) {
    ResVfetchInvalidateOne(m, i);
  }
}

void ResVfetchClearSyncBits(ResVfetchMirror* m) {
  m->in_sync[0] = 0;
  m->in_sync[1] = 0;
}

ResVfetchStatus ResVfetchPredict(const ResVfetchMirror* m,
                                 const uint32_t bitmap[3],
                                 const uint32_t* fetch_regs, bool allow_invalid,
                                 ResRange* out_requests, uint32_t out_cap,
                                 uint32_t* out_count) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    uint32_t bits = bitmap[i];
    uint32_t j;
    while (BitScanForward32(bits, &j)) {
      bits &= ~(uint32_t(1) << j);
      const uint32_t vfetch_index = i * 32 + j;
      const uint64_t bit = uint64_t(1) << (vfetch_index & 63);
      if (m->in_sync[vfetch_index >> 6] & bit) {
        continue;
      }
      const uint32_t dword_0 = fetch_regs[vfetch_index * 2];
      const uint32_t dword_1 = fetch_regs[vfetch_index * 2 + 1];
      const uint32_t type = VfetchType(dword_0);
      if (type != kResFetchTypeVertex &&
          !(type == kResFetchTypeInvalidVertex && allow_invalid)) {
        *out_count = count;
        return kResVfetchAbortInvalidType;
      }
      const uint32_t address = VfetchAddress(dword_0);
      const uint32_t size = VfetchSize(dword_1);
      if (m->state_address[vfetch_index] == address &&
          m->state_size[vfetch_index] == size) {
        continue;
      }
      if (count < out_cap) {
        out_requests[count].start = address << 2;
        out_requests[count].length = size << 2;
      }
      ++count;
    }
  }
  *out_count = count;
  return kResVfetchOk;
}

void ResVfetchApply(ResVfetchMirror* m, const uint32_t bitmap[3],
                    const uint32_t* fetch_regs, bool allow_invalid,
                    uint32_t requests_succeeded) {
  uint32_t consumed = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    uint32_t bits = bitmap[i];
    uint32_t j;
    while (BitScanForward32(bits, &j)) {
      bits &= ~(uint32_t(1) << j);
      const uint32_t vfetch_index = i * 32 + j;
      const uint64_t bit = uint64_t(1) << (vfetch_index & 63);
      if (m->in_sync[vfetch_index >> 6] & bit) {
        continue;
      }
      const uint32_t dword_0 = fetch_regs[vfetch_index * 2];
      const uint32_t dword_1 = fetch_regs[vfetch_index * 2 + 1];
      const uint32_t type = VfetchType(dword_0);
      if (type != kResFetchTypeVertex &&
          !(type == kResFetchTypeInvalidVertex && allow_invalid)) {
        return;  // the emulated loop returned false here
      }
      const uint32_t address = VfetchAddress(dword_0);
      const uint32_t size = VfetchSize(dword_1);
      if (m->state_address[vfetch_index] == address &&
          m->state_size[vfetch_index] == size) {
        m->in_sync[vfetch_index >> 6] |= bit;
        continue;
      }
      if (consumed >= requests_succeeded) {
        return;  // this request failed - the emulated loop returned false
      }
      ++consumed;
      m->state_address[vfetch_index] = address;
      m->state_size[vfetch_index] = size;
      m->in_sync[vfetch_index >> 6] |= bit;
    }
  }
}

// ---------------------------------------------------------------------------
// Index-buffer residency.

void ResIndexPredict(uint32_t vgt_draw_initiator, uint32_t vgt_dma_size,
                     uint32_t vgt_dma_base, uint32_t buffer_size,
                     ResIndexPrediction* out) {
  out->dma = false;
  out->out_of_bounds = false;
  out->base = 0;
  out->length = 0;
  // VGT_DRAW_INITIATOR: prim_type:6 | source_select:2 | major_mode:2 | pad:1 |
  // index_size:1 | not_eop:1 | pad:3 | num_indices:16.
  const uint32_t source_select = (vgt_draw_initiator >> 6) & 3u;
  if (source_select != kResSourceSelectDma) {
    return;
  }
  out->dma = true;
  uint32_t num_indices = (vgt_draw_initiator >> 16) & 0xFFFFu;
  // VGT_DMA_SIZE: num_words:24 | pad:6 | swap_mode:2. The processor clamps
  // the vertex count to the DMA words before sizing the request.
  const uint32_t num_words = vgt_dma_size & 0xFFFFFFu;
  if (num_indices > num_words) {
    num_indices = num_words;
  }
  const uint32_t index_size_log2 =
      ((vgt_draw_initiator >> 11) & 1u) == kResIndexFormatInt16 ? 1 : 2;
  out->base = vgt_dma_base & ~uint32_t((1u << index_size_log2) - 1);
  out->length = num_indices << index_size_log2;
  if (out->base > buffer_size || buffer_size - out->base < out->length) {
    out->out_of_bounds = true;
  }
}

// ---------------------------------------------------------------------------
// The bindless view-descriptor pool mirror.

void ResViewPoolReset(ResViewPool* p, uint32_t allocated_seed,
                      const uint32_t* free_seed, uint32_t free_seed_count,
                      uint32_t heap_size) {
  p->free_count = 0;
  p->allocated = allocated_seed;
  p->heap_size = heap_size;
  p->refused = 0;
  if (free_seed_count > ResViewPool::kFreeCap) {
    // Cannot represent the emulated free list - refuse rather than lie.
    p->refused = 1;
    return;
  }
  for (uint32_t i = 0; i < free_seed_count; ++i) {
    p->free_stack[i] = free_seed[i];
  }
  p->free_count = free_seed_count;
}

bool ResViewPoolPredictAlloc(const ResViewPool* p, uint32_t* out_index) {
  *out_index = kResInvalidIndex;
  if (p->refused) {
    return false;
  }
  if (p->free_count) {
    *out_index = p->free_stack[p->free_count - 1];
    return true;
  }
  if (p->allocated >= p->heap_size) {
    // Predicted exhaustion (the emulated allocator returns UINT32_MAX).
    return true;
  }
  *out_index = p->allocated;
  return true;
}

bool ResViewPoolObserveAlloc(ResViewPool* p, uint32_t actual_index) {
  uint32_t predicted;
  if (!ResViewPoolPredictAlloc(p, &predicted)) {
    return false;
  }
  if (predicted != actual_index) {
    return false;
  }
  if (predicted == kResInvalidIndex) {
    return true;  // exhaustion predicted right; nothing advanced
  }
  if (p->free_count) {
    --p->free_count;
  } else {
    ++p->allocated;
  }
  return true;
}

void ResViewPoolObserveRelease(ResViewPool* p, uint32_t index) {
  if (p->refused) {
    return;
  }
  if (p->free_count >= ResViewPool::kFreeCap) {
    p->refused = 1;
    return;
  }
  p->free_stack[p->free_count++] = index;
}

// ---------------------------------------------------------------------------
// The per-texture SRV descriptor map mirror.

namespace {

inline uint64_t TexDescMix(uint64_t texture_handle, uint32_t srv_key) {
  uint64_t h = texture_handle ^ (uint64_t(srv_key) << 48) ^
               (uint64_t(srv_key) * 0x9E3779B97F4A7C15ull);
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 32;
  // 0 marks an empty slot key; keep real keys nonzero.
  return h ? h : 1u;
}

}  // namespace

void ResTexDescMapReset(ResTexDescMap* m) {
  std::memset(m->state, 0, sizeof(m->state));
  m->count = 0;
  m->ovf = 0;
}

namespace {

// Returns the slot holding the key, or the first insertable slot (empty or
// tombstone), or kResInvalidIndex when the probe budget is exhausted.
uint32_t TexDescFindSlot(const ResTexDescMap* m, uint64_t key, bool* found) {
  *found = false;
  uint32_t slot = uint32_t(key) & (ResTexDescMap::kSize - 1);
  uint32_t insert = kResInvalidIndex;
  for (uint32_t probe = 0; probe < ResTexDescMap::kProbes; ++probe) {
    const uint32_t s = (slot + probe) & (ResTexDescMap::kSize - 1);
    if (m->state[s] == 0) {
      return insert != kResInvalidIndex ? insert : s;
    }
    if (m->state[s] == 2) {
      if (insert == kResInvalidIndex) {
        insert = s;
      }
      continue;
    }
    if (m->keys[s] == key) {
      *found = true;
      return s;
    }
  }
  return insert;
}

}  // namespace

ResTexDescVerdict ResTexDescObserve(ResTexDescMap* m, uint64_t texture_handle,
                                    uint32_t srv_key, bool their_hit,
                                    uint32_t their_index) {
  const uint64_t key = TexDescMix(texture_handle, srv_key);
  bool found;
  const uint32_t slot = TexDescFindSlot(m, key, &found);
  if (found) {
    if (m->index[slot] == their_index && their_hit) {
      return kResTexDescMatch;
    }
    // Either the value disagrees, or the emulated side allocated anew where
    // we believed an entry lived (a release we failed to track). Re-sync.
    m->index[slot] = their_index;
    return kResTexDescMismatch;
  }
  if (slot == kResInvalidIndex) {
    ++m->ovf;
    return kResTexDescOverflow;
  }
  m->keys[slot] = key;
  m->index[slot] = their_index;
  m->state[slot] = 1;
  ++m->count;
  return their_hit ? kResTexDescSeeded : kResTexDescFresh;
}

bool ResTexDescLookup(const ResTexDescMap* m, uint64_t texture_handle,
                      uint32_t srv_key, uint32_t* out_index) {
  const uint64_t key = TexDescMix(texture_handle, srv_key);
  bool found;
  const uint32_t slot = TexDescFindSlot(m, key, &found);
  if (!found) {
    return false;
  }
  *out_index = m->index[slot];
  return true;
}

void ResTexDescEvictIndex(ResTexDescMap* m, uint32_t index) {
  for (uint32_t s = 0; s < ResTexDescMap::kSize; ++s) {
    if (m->state[s] == 1 && m->index[s] == index) {
      m->state[s] = 2;  // tombstone
      if (m->count) {
        --m->count;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The GetActiveTextureBindlessSRVIndex decision tree.

bool ResDimensionsCompatible(uint32_t fetch_op_dimension,
                             uint32_t data_dimension) {
  switch (fetch_op_dimension) {
    case kResFetchDim1D:
    case kResFetchDim2D:
      return data_dimension == kResDataDim1D ||
             data_dimension == kResDataDim2DOrStacked ||
             data_dimension == kResDataDim3D;
    case kResFetchDim3DOrStacked:
      return data_dimension == kResDataDim3D;
    case kResFetchDimCube:
      return data_dimension == kResDataDimCube;
    default:
      return false;
  }
}

ResSrvOutcome ResSrvIndexForBinding(const ResTexDescMap* map,
                                    const ResSrvBindingFacts* facts,
                                    uint32_t fetch_op_dimension,
                                    bool host_shader_is_signed,
                                    uint32_t null_2darray, uint32_t null_3d,
                                    uint32_t null_cube, uint32_t* out_index) {
  uint32_t descriptor_index = kResInvalidIndex;
  bool refused_special = false;
  bool refused_unknown = false;
  if (facts->has_binding &&
      ResDimensionsCompatible(fetch_op_dimension, facts->binding_dimension)) {
    const bool force_special_view =
        facts->binding_dimension == kResDataDim3D &&
        (fetch_op_dimension == kResFetchDim1D ||
         fetch_op_dimension == kResFetchDim2D);
    if (force_special_view) {
      // The emulated query resolves this with a find-or-create at QUERY time
      // (a 2D-array view of the 3D resource). If the map has already observed
      // that creation, predict it; otherwise refuse.
      const bool use_signed = host_shader_is_signed && facts->any_sign_signed;
      const uint64_t handle =
          use_signed ? (facts->signed_separate ? facts->texture_signed_handle
                                               : facts->texture_handle)
                     : facts->texture_handle;
      if (handle) {
        const uint32_t key = ResSrvDescriptorKey(use_signed, facts->host_swizzle,
                                                 kResDataDim2DOrStacked);
        if (!ResTexDescLookup(map, handle, key, &descriptor_index)) {
          refused_special = true;
        }
      }
    } else {
      // The emulated query reads the per-binding descriptor index that
      // UpdateTextureBindingsImpl resolved with these exact gates.
      uint64_t handle = 0;
      if (host_shader_is_signed) {
        if (facts->any_sign_signed) {
          handle = facts->signed_separate ? facts->texture_signed_handle
                                          : facts->texture_handle;
        }
      } else {
        if (facts->any_sign_not_signed) {
          handle = facts->texture_handle;
        }
      }
      if (handle) {
        const uint32_t key =
            ResSrvDescriptorKey(host_shader_is_signed, facts->host_swizzle,
                                facts->binding_dimension);
        if (!ResTexDescLookup(map, handle, key, &descriptor_index)) {
          refused_unknown = true;
        }
      }
    }
  }
  if (refused_special) {
    return kResSrvRefuseSpecialView;
  }
  if (refused_unknown) {
    return kResSrvRefuseUnknown;
  }
  if (descriptor_index == kResInvalidIndex) {
    switch (fetch_op_dimension) {
      case kResFetchDim3DOrStacked:
        *out_index = null_3d;
        break;
      case kResFetchDimCube:
        *out_index = null_cube;
        break;
      default:
        *out_index = null_2darray;
        break;
    }
    return kResSrvNull;
  }
  *out_index = descriptor_index;
  return kResSrvValue;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
