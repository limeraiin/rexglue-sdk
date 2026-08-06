/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_context.h"

// [NR-CTX] See the header. Decode rules mirror the executor exactly
// (ExecutePacketType0/1/3 in command_processor.cpp), and the walk structure
// mirrors nr_state_walk.cpp -- same packet subset, same skip rules, so the
// per-draw flag list stays index-aligned with the [nr-cache] join list.
// PM4 in guest memory is big-endian.
//
// IM_LOAD (0x27): payload dw0 = ucode physical address | shader type in the
// low 2 bits (0 vertex, 1 pixel), dw1 = start<<16 | size_dwords.
// IM_LOAD_IMMEDIATE (0x2B): dw0 = shader type, dw1 = start<<16 | size_dwords,
// ucode inline from the third payload dword (its physical address is
// buffer_phys + that offset -- a real address a translator can read, valid
// until the buffer is re-recorded, which is exactly the lifetime the draw
// cache already models).
// ★ A ZERO DWORD IS A ONE-DWORD NO-OP, not a type-0 packet. ExecutePacket
// tests `packet == 0` before dispatching on the type field and consumes just
// that dword; decoding it as type-0 (count 1, base 0) consumes TWO and puts
// the rest of the walk one dword out of phase with the executor, after which
// payload data decodes as headers and writes garbage into whatever registers
// it names. That desync -- not any out-of-stream write path -- was the
// increment-4a divergence (see the 4b-0 verdict: the ring observer captured
// zero writes while the context held impossible window-offset values).
//
// SET_CONSTANT2 (0x55) / SET_SHADER_CONSTANTS (0x56): dw0 = a RAW register
// index (& 0xFFFF, no typed base added), values inline after it. Both write
// registers through the same path as everything else, so a walk that skips
// them silently misses state; they are the other half of the 4b-0 blind spot
// (REG_RMW / COND_WRITE remain value-dependent and stay tapped, not decoded).
//
// LOAD_ALU_CONSTANT (0x2F): dw0 = memory address of the values, dw1 =
// offset|type, dw2 = size_dwords. Type 4 targets registers base 0x2000; the
// values are NOT in the packet, so mirrored slots covered by one either read
// through the caller's memory callback or become UNDEFINED (poisoned) --
// carrying a stale value as "defined" would be a silent lie the ground-truth
// compare exists to catch.

namespace rex {
namespace graphics {
namespace nr {

namespace {

inline uint32_t BE32(const uint8_t* raw, uint32_t dw) {
  return __builtin_bswap32(*(const uint32_t*)(raw + dw * 4));
}

// Slot layout (dense, grouped):
//   0-5   0x2000-0x2005  RT (required: 0-2)
//   6-11  0x210F-0x2114  viewport (required: all)
//   12-13 0x200E-0x200F  scissor
//   14-16 0x2080-0x2082  scissor
//   17    0x2208         RB_MODECONTROL
//   18-21 0x2318-0x231B  copy
//   22-23 0x2320-0x2321  copy
//   24-26 0x231D-0x231F  clear
constexpr uint32_t kSlotMode = 17;

constexpr uint32_t kRtRequired0 = 0, kRtRequiredN = 3;
constexpr uint32_t kVportFirst = 6, kVportN = 6;

struct SlotRange {
  uint32_t first_reg, count, first_slot;
  CtxGroup group;
};
constexpr SlotRange kRanges[] = {
    {0x2000, 6, 0, kCtxGroupRt},        {0x210F, 6, 6, kCtxGroupViewport},
    {0x200E, 2, 12, kCtxGroupScissor},  {0x2080, 3, 14, kCtxGroupScissor},
    {0x2208, 1, 17, kCtxGroupMode},     {0x2318, 4, 18, kCtxGroupCopy},
    {0x2320, 2, 22, kCtxGroupCopy},     {0x231D, 3, 24, kCtxGroupClear},
};

}  // namespace

bool CtxApplyBinPacket(CtxBinState* bin, uint32_t op, uint32_t p0,
                       uint32_t p1) {
  switch (op) {
    case 0x60:  // SET_BIN_MASK_LO
      bin->mask = (bin->mask & 0xFFFFFFFF00000000ull) | p0;
      return true;
    case 0x61:  // SET_BIN_MASK_HI
      bin->mask = (bin->mask & 0xFFFFFFFFull) | (uint64_t(p0) << 32);
      return true;
    case 0x62:  // SET_BIN_SELECT_LO
      bin->select = (bin->select & 0xFFFFFFFF00000000ull) | p0;
      return true;
    case 0x63:  // SET_BIN_SELECT_HI
      bin->select = (bin->select & 0xFFFFFFFFull) | (uint64_t(p0) << 32);
      return true;
    case 0x50:  // SET_BIN_MASK: hi dword first, then lo
      bin->mask = (uint64_t(p0) << 32) | p1;
      return true;
    case 0x51:  // SET_BIN_SELECT: same layout
      bin->select = (uint64_t(p0) << 32) | p1;
      return true;
    default:
      return false;
  }
}

int32_t CtxSlot(uint32_t reg) {
  for (const SlotRange& r : kRanges) {
    if (reg >= r.first_reg && reg < r.first_reg + r.count) {
      return int32_t(r.first_slot + (reg - r.first_reg));
    }
  }
  return -1;
}

uint32_t CtxSlotReg(uint32_t slot) {
  for (const SlotRange& r : kRanges) {
    if (slot >= r.first_slot && slot < r.first_slot + r.count) {
      return r.first_reg + (slot - r.first_slot);
    }
  }
  return 0;
}

CtxGroup CtxSlotGroup(uint32_t slot) {
  for (const SlotRange& r : kRanges) {
    if (slot >= r.first_slot && slot < r.first_slot + r.count) {
      return r.group;
    }
  }
  return kCtxGroupCount;
}

int32_t CtxApplyExternalWrite(StateContext* ctx, uint32_t reg,
                              uint32_t value) {
  const int32_t s = CtxSlot(reg);
  if (s < 0) return -1;
  ctx->values[s] = value;
  ctx->defined[s] = 1;
  // Deliberately NOT in_buffer: the value did not come from the buffer
  // stream, so carry attribution must keep counting draws that depend on it.
  return s;
}

uint32_t WalkBufferContext(const uint8_t* raw, uint32_t dwords,
                           uint32_t buffer_phys, StateContext* ctx,
                           uint16_t* draw_flags, uint32_t max_draws,
                           CtxWalkStats* stats, CtxMemReadFn mem_read,
                           void* mem_user, CtxShaderFn shader_fn,
                           void* shader_user, CtxWatchFn watch_fn,
                           void* watch_user, uint64_t bin_select,
                           uint64_t bin_mask) {
  *stats = CtxWalkStats{};
  uint32_t nflags = 0;

  // Packet currently being decoded, for the write sampler.
  uint32_t cur_dw = 0, cur_hdr = 0, cur_arg = 0;
  // Bin state in effect for this pass, advanced by in-buffer SET_BIN_*.
  CtxBinState bin{bin_select, bin_mask};

  // "Written by THIS buffer" resets at entry; definedness and values persist.
  for (uint32_t s = 0; s < kCtxRegCount; ++s) ctx->in_buffer[s] = 0;
  ctx->vs_in_buffer = 0;
  ctx->ps_in_buffer = 0;

  auto write_reg = [&](uint32_t reg, uint32_t value) {
    const int32_t s = CtxSlot(reg);
    if (s < 0) return;
    ctx->values[s] = value;
    ctx->defined[s] = 1;
    ctx->in_buffer[s] = 1;
    if (watch_fn) watch_fn(watch_user, reg, value, cur_dw, cur_hdr, cur_arg);
  };

  auto group_bits = [&](uint32_t first, uint32_t n, uint16_t def_bit,
                        uint16_t carry_bit) -> uint16_t {
    bool def = true, all_local = true;
    for (uint32_t s = first; s < first + n; ++s) {
      if (!ctx->defined[s]) def = false;
      if (!ctx->in_buffer[s]) all_local = false;
    }
    if (!def) return 0;
    return uint16_t(def_bit | (all_local ? 0 : carry_bit));
  };

  auto flag_draw = [&]() {
    uint16_t f = 0;
    f |= group_bits(kRtRequired0, kRtRequiredN, kCtxDrawRtDef, kCtxDrawRtCarried);
    f |= group_bits(kVportFirst, kVportN, kCtxDrawVportDef, kCtxDrawVportCarried);
    f |= group_bits(kSlotMode, 1, kCtxDrawModeDef, kCtxDrawModeCarried);
    if (ctx->vs.valid && ctx->ps.valid) {
      f |= kCtxDrawShadersDef;
      if (!ctx->vs_in_buffer || !ctx->ps_in_buffer) f |= kCtxDrawShadersCarried;
    }
    if (ctx->defined[kSlotMode] && (ctx->values[kSlotMode] & 0x7) == 6) {
      f |= kCtxDrawCopy;
    }
    if (nflags < max_draws) draw_flags[nflags++] = f;
  };

  for (uint32_t j = 0; j < dwords;) {
    const uint32_t hdr = BE32(raw, j);
    // ExecutePacket's first test: a zero dword is a one-dword no-op.
    if (!hdr) {
      ++stats->nop0;
      ++j;
      continue;
    }
    const uint32_t ty = hdr >> 30;
    cur_dw = j;
    cur_hdr = hdr;
    cur_arg = (j + 1 < dwords) ? BE32(raw, j + 1) : 0;
    if (ty == 0) {
      const uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;
      const uint32_t base = hdr & 0x7FFF;
      const bool one_reg = (hdr >> 15) & 1;
      for (uint32_t m = 0; m < cnt && j + 1 + m < dwords; ++m) {
        write_reg(one_reg ? base : base + m, BE32(raw, j + 1 + m));
      }
      j += 1 + cnt;
    } else if (ty == 1) {
      if (j + 2 < dwords) {
        write_reg(hdr & 0x7FF, BE32(raw, j + 1));
        write_reg((hdr >> 11) & 0x7FF, BE32(raw, j + 2));
      }
      j += 3;
    } else if (ty == 2) {
      ++j;
    } else {
      const uint32_t op = (hdr >> 8) & 0x7F;
      const uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;
      // Predicate bit: the command processor skips the whole packet when no
      // selected bin passes the mask. Mirror it before decoding anything --
      // a predicated-out packet writes nothing and a predicated-out draw does
      // not execute, so it gets no flags word either.
      if (CtxPredicatedOut(bin, hdr)) {
        ++stats->pred_skipped;
        if (op == 0x22) ++stats->pred_draws;
        j += 1 + cnt;
        continue;
      }
      if (hdr & 1) {
        if (op == 0x22) ++stats->pred_draws_run;
      }
      if (op == 0x22) {
        ++stats->draws22;
        flag_draw();
      } else if (op == 0x27 && cnt >= 2 && j + 2 < dwords) {
        // IM_LOAD: pointer-based shader load.
        const uint32_t addr_type = BE32(raw, j + 1);
        const uint32_t start_size = BE32(raw, j + 2);
        ShaderRef ref;
        ref.addr = (addr_type & ~3u) & 0x1FFFFFFF;
        ref.size_dwords = start_size & 0xFFFF;
        ref.immediate = 0;
        ref.valid = 1;
        if ((addr_type & 0x3) == 0) {
          ctx->vs = ref;
          ctx->vs_in_buffer = 1;
        } else {
          ctx->ps = ref;
          ctx->ps_in_buffer = 1;
        }
        ++stats->im_loads;
        if (shader_fn) shader_fn(shader_user, ref);
      } else if (op == 0x2B && cnt >= 3 && j + 2 < dwords) {
        // IM_LOAD_IMMEDIATE: ucode inline after the two header dwords.
        const uint32_t type = BE32(raw, j + 1);
        const uint32_t start_size = BE32(raw, j + 2);
        ShaderRef ref;
        ref.addr = ((buffer_phys & 0x1FFFFFFF) + (j + 3) * 4);
        ref.size_dwords = start_size & 0xFFFF;
        ref.immediate = 1;
        ref.valid = 1;
        if (type == 0) {
          ctx->vs = ref;
          ctx->vs_in_buffer = 1;
        } else {
          ctx->ps = ref;
          ctx->ps_in_buffer = 1;
        }
        ++stats->im_load_imms;
        if (shader_fn) shader_fn(shader_user, ref);
      } else if (op == 0x2D && cnt >= 2 && j + 1 < dwords) {
        // SET_CONSTANT: typed offset, values inline. Type 4 = registers.
        const uint32_t offset_type = BE32(raw, j + 1);
        if (((offset_type >> 16) & 0xFF) == 4) {
          const uint32_t base = 0x2000 + (offset_type & 0x7FF);
          for (uint32_t m = 0; m + 1 < cnt && j + 2 + m < dwords; ++m) {
            write_reg(base + m, BE32(raw, j + 2 + m));
          }
        }
      } else if (CtxApplyBinPacket(
                     &bin, op, (j + 1 < dwords) ? BE32(raw, j + 1) : 0,
                     (j + 2 < dwords) ? BE32(raw, j + 2) : 0)) {
        // SET_BIN_MASK / SET_BIN_SELECT (and the LO/HI halves): which of the
        // following predicated blocks this pass executes.
        ++stats->bin_pkts;
      } else if ((op == 0x55 || op == 0x56) && cnt >= 2 && j + 1 < dwords) {
        // SET_CONSTANT2 / SET_SHADER_CONSTANTS: RAW register index, values
        // inline (ExecutePacketType3_SET_CONSTANT2 / _SET_SHADER_CONSTANTS ->
        // WriteRegisterRangeFromRing with no typed base).
        const uint32_t base = BE32(raw, j + 1) & 0xFFFF;
        ++stats->set_const2;
        for (uint32_t m = 0; m + 1 < cnt && j + 2 + m < dwords; ++m) {
          write_reg(base + m, BE32(raw, j + 2 + m));
        }
      } else if (op == 0x2F && cnt >= 3 && j + 3 < dwords) {
        // LOAD_ALU_CONSTANT: typed offset, values in guest memory.
        const uint32_t address = BE32(raw, j + 1) & 0x1FFFFFFF;
        const uint32_t offset_type = BE32(raw, j + 2);
        const uint32_t size_dwords = BE32(raw, j + 3) & 0xFFF;
        if (((offset_type >> 16) & 0xFF) == 4) {
          const uint32_t base = 0x2000 + (offset_type & 0x7FF);
          for (uint32_t m = 0; m < size_dwords; ++m) {
            const int32_t s = CtxSlot(base + m);
            if (s < 0) continue;
            if (mem_read) {
              write_reg(base + m, mem_read(mem_user, address + m * 4));
              ++stats->mem_loads;
            } else {
              ctx->defined[s] = 0;
              ctx->in_buffer[s] = 0;
              ++stats->mem_poisoned;
            }
          }
        }
      }
      j += 1 + cnt;
    }
  }
  return nflags;
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
