/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <rex/graphics/nr_resource.h>

namespace rex {
namespace graphics {
namespace nr {

namespace {

struct FileRange {
  uint32_t first_reg;
  uint32_t count;
  uint32_t first_slot;
  ResFile file;
};

// Dense slot space, in file order. Declared once so ResSlot and its inverses
// cannot disagree.
constexpr FileRange kFiles[] = {
    {kResAluFirst, kResAluCount, 0, kResFileAlu},
    {kResFetchFirst, kResFetchCount, kResAluCount, kResFileFetch},
    {kResBoolFirst, kResBoolCount, kResAluCount + kResFetchCount,
     kResFileBool},
    {kResLoopFirst, kResLoopCount,
     kResAluCount + kResFetchCount + kResBoolCount, kResFileLoop},
};

constexpr uint32_t kFetchSlot0 = kResAluCount;

// A constant of `dwords` dwords starting at fetch-file offset `first` is live
// when every dword is defined AND its type field claims `want`. The type is
// the low 2 bits of the constant's first dword.
bool ConstantLive(const ResourceContext* ctx, uint32_t first, uint32_t dwords,
                  ResFetchType want) {
  const uint32_t base = kFetchSlot0 + first;
  for (uint32_t d = 0; d < dwords; ++d) {
    if (!ctx->defined[base + d]) return false;
  }
  return ResFetchType(ctx->values[base] & 3u) == want;
}

void SetBit(uint32_t* mask, uint32_t bit, bool value) {
  if (value) {
    mask[bit >> 5] |= 1u << (bit & 31);
  } else {
    mask[bit >> 5] &= ~(1u << (bit & 31));
  }
}

uint32_t PopCount(uint32_t x) {
  // No <bit> and no intrinsics: this file is built bare by the unit test.
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  return (x * 0x01010101u) >> 24;
}

}  // namespace

int32_t ResSlot(uint32_t reg) {
  for (const auto& f : kFiles) {
    if (reg >= f.first_reg && reg < f.first_reg + f.count) {
      return int32_t(f.first_slot + (reg - f.first_reg));
    }
  }
  return -1;
}

uint32_t ResSlotReg(uint32_t slot) {
  for (const auto& f : kFiles) {
    if (slot >= f.first_slot && slot < f.first_slot + f.count) {
      return f.first_reg + (slot - f.first_slot);
    }
  }
  return 0;
}

ResFile ResSlotFile(uint32_t slot) {
  for (const auto& f : kFiles) {
    if (slot >= f.first_slot && slot < f.first_slot + f.count) {
      return f.file;
    }
  }
  return kResFileCount;
}

const char* ResFileName(ResFile file) {
  switch (file) {
    case kResFileAlu:
      return "alu";
    case kResFileFetch:
      return "fetch";
    case kResFileBool:
      return "bool";
    case kResFileLoop:
      return "loop";
    default:
      return "?";
  }
}

void ResBeginBuffer(ResourceContext* ctx) {
  for (uint32_t s = 0; s < kResRegCount; ++s) ctx->in_buffer[s] = 0;
  ctx->alu_any_in_buffer = 0;
  ctx->vfetch_in_buffer_mask[0] = 0;
  ctx->vfetch_in_buffer_mask[1] = 0;
  ctx->vfetch_in_buffer_mask[2] = 0;
  ctx->tfetch_in_buffer_mask = 0;
}

void ResApplyWrite(ResourceContext* ctx, ResStats* stats, uint32_t reg,
                   uint32_t value, bool from_memory) {
  const int32_t s = ResSlot(reg);
  if (s < 0) return;
  const uint32_t slot = uint32_t(s);
  ctx->values[slot] = value;
  const uint8_t was_defined = ctx->defined[slot];
  ctx->defined[slot] = 1;
  ctx->in_buffer[slot] = 1;
  ctx->from_memory[slot] = from_memory ? 1 : 0;

  const ResFile file = ResSlotFile(slot);
  if (file == kResFileAlu) {
    if (!was_defined) ++ctx->alu_defined_count;
    ctx->alu_any_in_buffer = 1;
  } else if (file == kResFileFetch) {
    const uint32_t d = slot - kFetchSlot0;
    // Both aliased views are updated: a write lands in exactly one vertex
    // constant and one texture constant, and which of the two is real is
    // decided by the type field, not by us.
    const uint32_t v = d / kResFetchVertexDwords;
    const uint32_t t = d / kResFetchTextureDwords;
    ctx->vfetch_in_buffer_mask[v >> 5] |= 1u << (v & 31);
    ctx->tfetch_in_buffer_mask |= 1u << t;
    // The type field can change with any write to a constant's first dword,
    // so liveness is recomputed for the touched constants rather than latched.
    SetBit(ctx->vfetch_live_mask, v, ConstantLive(ctx, v * kResFetchVertexDwords,
                                                  kResFetchVertexDwords,
                                                  kResFetchVertex));
    const bool tlive = ConstantLive(ctx, t * kResFetchTextureDwords,
                                    kResFetchTextureDwords, kResFetchTexture);
    if (tlive) {
      ctx->tfetch_live_mask |= 1u << t;
    } else {
      ctx->tfetch_live_mask &= ~(1u << t);
    }
  }

  if (!stats) return;
  ++stats->writes;
  ++stats->writes_by_file[file];
  if (from_memory) {
    ++stats->writes_from_memory;
    ++stats->writes_from_memory_by_file[file];
  }
}

void ResApplyRange(ResourceContext* ctx, ResStats* stats, uint32_t base,
                   const uint32_t* values, uint32_t n, bool from_memory) {
  if (!n) return;
  const int32_t s0 = ResSlot(base);
  const int32_t s1 = ResSlot(base + n - 1);
  // Fast path only for a range wholly inside one file (slots contiguous by
  // construction when both ends land in the same file). Everything else --
  // spanning a file boundary or reaching outside the files -- replays the
  // per-dword apply so the two paths cannot disagree.
  if (s0 < 0 || s1 < 0 || uint32_t(s1) != uint32_t(s0) + (n - 1) ||
      ResSlotFile(uint32_t(s1)) != ResSlotFile(uint32_t(s0))) {
    for (uint32_t i = 0; i < n; ++i) {
      ResApplyWrite(ctx, stats, base + i, values[i], from_memory);
    }
    return;
  }
  const uint32_t slot0 = uint32_t(s0);
  const ResFile file = ResSlotFile(slot0);
  const uint8_t fm = from_memory ? 1 : 0;
  if (file == kResFileAlu) {
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t slot = slot0 + i;
      ctx->values[slot] = values[i];
      if (!ctx->defined[slot]) {
        ctx->defined[slot] = 1;
        ++ctx->alu_defined_count;
      }
      ctx->in_buffer[slot] = 1;
      ctx->from_memory[slot] = fm;
    }
    ctx->alu_any_in_buffer = 1;
  } else if (file == kResFileFetch) {
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t slot = slot0 + i;
      ctx->values[slot] = values[i];
      ctx->defined[slot] = 1;
      ctx->in_buffer[slot] = 1;
      ctx->from_memory[slot] = fm;
    }
    // Liveness over the FINAL values, once per touched constant in each
    // aliased view -- the state the per-dword recompute ends on.
    const uint32_t d0 = slot0 - kFetchSlot0;
    const uint32_t d1 = d0 + n - 1;
    for (uint32_t v = d0 / kResFetchVertexDwords;
         v <= d1 / kResFetchVertexDwords; ++v) {
      ctx->vfetch_in_buffer_mask[v >> 5] |= 1u << (v & 31);
      SetBit(ctx->vfetch_live_mask, v,
             ConstantLive(ctx, v * kResFetchVertexDwords, kResFetchVertexDwords,
                          kResFetchVertex));
    }
    for (uint32_t t = d0 / kResFetchTextureDwords;
         t <= d1 / kResFetchTextureDwords; ++t) {
      ctx->tfetch_in_buffer_mask |= 1u << t;
      if (ConstantLive(ctx, t * kResFetchTextureDwords, kResFetchTextureDwords,
                       kResFetchTexture)) {
        ctx->tfetch_live_mask |= 1u << t;
      } else {
        ctx->tfetch_live_mask &= ~(1u << t);
      }
    }
  } else {
    // Bool / loop: value + flag stores only.
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t slot = slot0 + i;
      ctx->values[slot] = values[i];
      ctx->defined[slot] = 1;
      ctx->in_buffer[slot] = 1;
      ctx->from_memory[slot] = fm;
    }
  }
  if (!stats) return;
  stats->writes += n;
  stats->writes_by_file[file] += n;
  if (from_memory) {
    stats->writes_from_memory += n;
    stats->writes_from_memory_by_file[file] += n;
  }
}

ResFetchType ResFetchTypeAt(const ResourceContext* ctx, uint32_t slot) {
  if (slot >= kResFetchVertexSlots) return kResFetchInvalidTexture;
  return ResFetchType(
      ctx->values[kFetchSlot0 + slot * kResFetchVertexDwords] & 3u);
}

ResVertexFetch ResDecodeVertexFetch(const ResourceContext* ctx,
                                    uint32_t slot) {
  ResVertexFetch out{false, 0, 0};
  if (slot >= kResFetchVertexSlots) return out;
  if (!(ctx->vfetch_live_mask[slot >> 5] & (1u << (slot & 31)))) return out;
  const uint32_t first = kFetchSlot0 + slot * kResFetchVertexDwords;
  // xe_gpu_vertex_fetch_t: type:2 then address:30, and the address COUNTS
  // DWORDS, so the byte address is it shifted back up by two -- which is the
  // stored dword with its type bits cleared.
  out.valid = true;
  out.base = ctx->values[first] & ~0x3u;
  out.words = (ctx->values[first + 1] >> 2) & 0xFFFFFF;
  return out;
}

namespace {

ResCoverage CoverageOf(const ResourceContext* ctx, uint32_t first_dword,
                       uint32_t dwords, ResFetchType want) {
  const uint32_t base = kFetchSlot0 + first_dword;
  for (uint32_t d = 0; d < dwords; ++d) {
    if (!ctx->defined[base + d]) return kResCoverUndefined;
  }
  return ResFetchType(ctx->values[base] & 3u) == want ? kResCoverLive
                                                      : kResCoverWrongType;
}

}  // namespace

uint32_t ResFetchDword0(const ResourceContext* ctx, uint32_t slot,
                        bool texture_view) {
  const uint32_t stride =
      texture_view ? kResFetchTextureDwords : kResFetchVertexDwords;
  const uint32_t limit =
      texture_view ? kResFetchTextureSlots : kResFetchVertexSlots;
  if (slot >= limit) return 0;
  return ctx->values[kFetchSlot0 + slot * stride];
}

ResCoverage ResVertexFetchCoverage(const ResourceContext* ctx,
                                   uint32_t slot) {
  if (slot >= kResFetchVertexSlots) return kResCoverUndefined;
  return CoverageOf(ctx, slot * kResFetchVertexDwords, kResFetchVertexDwords,
                    kResFetchVertex);
}

ResCoverage ResTextureFetchCoverage(const ResourceContext* ctx,
                                    uint32_t slot) {
  if (slot >= kResFetchTextureSlots) return kResCoverUndefined;
  return CoverageOf(ctx, slot * kResFetchTextureDwords, kResFetchTextureDwords,
                    kResFetchTexture);
}

void ResObserveDraw(const ResourceContext* ctx, ResStats* stats) {
  if (!stats) return;
  ++stats->draws;
  // O(1): every quantity here is maintained by ResApplyWrite.
  for (uint32_t w = 0; w < 3; ++w) {
    const uint32_t live = ctx->vfetch_live_mask[w];
    stats->vfetch_live += PopCount(live);
    // Carried means this buffer did not (re)establish the constant before the
    // draw, so the draw depends on state from an earlier buffer.
    stats->vfetch_carried += PopCount(live & ~ctx->vfetch_in_buffer_mask[w]);
  }
  stats->tfetch_live += PopCount(ctx->tfetch_live_mask);
  stats->tfetch_carried +=
      PopCount(ctx->tfetch_live_mask & ~ctx->tfetch_in_buffer_mask);
  stats->alu_defined_at_draw += ctx->alu_defined_count;
  if (!ctx->alu_any_in_buffer) ++stats->draws_all_alu_carried;
}

uint32_t ResCompareLive(const ResourceContext* ctx, ResStats* stats,
                        ResLiveReadFn read, void* user,
                        ResDivergence* samples, uint32_t max_samples) {
  if (!read) return 0;
  uint32_t nsamples = 0;
  for (uint32_t s = 0; s < kResRegCount; ++s) {
    if (!ctx->defined[s]) continue;
    const uint32_t reg = ResSlotReg(s);
    const uint32_t live = read(user, reg);
    if (stats) ++stats->checks;
    if (live == ctx->values[s]) continue;
    if (stats) {
      ++stats->diverge;
      ++stats->diverge_by_file[ResSlotFile(s)];
    }
    if (samples && nsamples < max_samples) {
      samples[nsamples++] = {reg, ctx->values[s], live};
    }
  }
  return nsamples;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
