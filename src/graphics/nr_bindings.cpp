/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <rex/graphics/nr_bindings.h>

#include <cstring>

namespace rex {
namespace graphics {
namespace nr {

uint32_t BindComposeFloats(const uint32_t* regs, uint32_t base,
                           const uint64_t bitmap[4], void* out,
                           uint32_t out_cap) {
  uint32_t needed = 0;
  for (uint32_t w = 0; w < 4; ++w) {
    uint64_t word = bitmap[w];
    // No <bit>, no intrinsics: this file is built bare by the unit test.
    word = word - ((word >> 1) & 0x5555555555555555ull);
    word = (word & 0x3333333333333333ull) + ((word >> 2) & 0x3333333333333333ull);
    word = (word + (word >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    needed += uint32_t((word * 0x0101010101010101ull) >> 56);
  }
  needed *= 4 * sizeof(uint32_t);
  if (needed > out_cap) {
    return needed;
  }
  uint8_t* dst = static_cast<uint8_t*>(out);
  for (uint32_t w = 0; w < 4; ++w) {
    const uint64_t word = bitmap[w];
    if (!word) {
      continue;
    }
    // Ascending bit order — the runtime's bit_scan_forward loop.
    for (uint32_t b = 0; b < 64; ++b) {
      if (!(word >> b & 1)) {
        continue;
      }
      std::memcpy(dst, &regs[base + (w << 8) + (b << 2)], 4 * sizeof(uint32_t));
      dst += 4 * sizeof(uint32_t);
    }
  }
  return needed;
}

uint32_t BindComposeBoolLoop(const uint32_t* regs, void* out,
                             uint32_t out_cap) {
  if (kBindBoolLoopBytes > out_cap) {
    return kBindBoolLoopBytes;
  }
  std::memcpy(out, &regs[kBindBoolLoopBase], kBindBoolLoopBytes);
  return kBindBoolLoopBytes;
}

uint32_t BindComposeFetch(const uint32_t* regs, void* out, uint32_t out_cap) {
  if (kBindFetchBytes > out_cap) {
    return kBindFetchBytes;
  }
  std::memcpy(out, &regs[kBindFetchBase], kBindFetchBytes);
  return kBindFetchBytes;
}

uint32_t BindRootSigBindfulIndex(uint32_t texture_count_pixel,
                                 uint32_t sampler_count_pixel,
                                 uint32_t texture_count_vertex,
                                 uint32_t sampler_count_vertex,
                                 bool tessellated, uint32_t texture_index_bits,
                                 uint32_t sampler_index_bits) {
  uint32_t index = 0;
  uint32_t index_offset = 0;
  index |= texture_count_pixel << index_offset;
  index_offset += texture_index_bits;
  index |= sampler_count_pixel << index_offset;
  index_offset += sampler_index_bits;
  index |= texture_count_vertex << index_offset;
  index_offset += texture_index_bits;
  index |= sampler_count_vertex << index_offset;
  index_offset += sampler_index_bits;
  index |= uint32_t(tessellated) << index_offset;
  return index;
}

void BindCensusReset(BindCensus* census) {
  std::memset(census, 0, sizeof(*census));
}

bool BindCensusAdd(BindCensus* census, uint64_t key) {
  // splitmix64 finalizer as the probe hash so clustered keys spread.
  uint64_t h = key + 0x9E3779B97F4A7C15ull;
  h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
  h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
  h ^= h >> 31;
  const uint32_t mask = BindCensus::kSize - 1;
  uint32_t slot = uint32_t(h) & mask;
  for (uint32_t probe = 0; probe < BindCensus::kProbes; ++probe) {
    const uint32_t at = (slot + probe) & mask;
    if (!census->used[at]) {
      census->used[at] = 1;
      census->keys[at] = key;
      ++census->count;
      return true;
    }
    if (census->keys[at] == key) {
      return false;
    }
  }
  ++census->ovf;
  return false;
}

uint64_t BindLayoutKey(uint64_t texture_uid_vertex, uint64_t sampler_uid_vertex,
                       uint64_t texture_uid_pixel,
                       uint64_t sampler_uid_pixel) {
  // Sequential splitmix-style combine: order-sensitive, 64-bit, no truncation
  // of any input (the UIDs are size_t counters but nothing bounds them small).
  uint64_t h = 0x243F6A8885A308D3ull;  // pi, nothing-up-my-sleeve
  const uint64_t parts[4] = {texture_uid_vertex, sampler_uid_vertex,
                             texture_uid_pixel, sampler_uid_pixel};
  for (uint64_t part : parts) {
    h ^= part + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
  }
  return h;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
