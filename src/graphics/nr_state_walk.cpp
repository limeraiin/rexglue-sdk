/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_state_walk.h"

// [NR-STATE] See the header. Decode rules mirror the executor exactly
// (ExecutePacketType0/1/3 in command_processor.cpp): type-0 base = bits 0-14,
// one-reg = bit 15, count = ((hdr >> 16) & 0x3FFF) + 1; type-1 = two 11-bit
// register indices; type-3 opcode = (hdr >> 8) & 0x7F. PM4 in guest memory is
// big-endian.

namespace rex {
namespace graphics {
namespace nr {

namespace {

inline uint32_t BE32(const uint8_t* raw, uint32_t dw) {
  return __builtin_bswap32(*(const uint32_t*)(raw + dw * 4));
}

void TallyOther(StateWalkResult* out, uint32_t reg) {
  for (auto& slot : out->other_top) {
    if (slot.count && slot.reg == reg) {
      ++slot.count;
      return;
    }
    if (!slot.count) {
      slot.reg = reg;
      slot.count = 1;
      return;
    }
  }
  // Table full: drop. Eight distinct unknown registers per buffer is already
  // a finding; exact tail counts are not what this instrument is for.
}

}  // namespace

RegClass ClassifyReg(uint32_t reg) {
  if (reg >= 0x4000 && reg <= 0x47FF) return kRegShaderConst;
  if (reg >= 0x4800 && reg <= 0x48FF) return kRegFetchConst;
  // Bool 0x4900-0x4907, loop 0x4908-0x4927 (the first census read loop
  // constants 08-10 as unclassified with the old 0x490F cutoff).
  if (reg >= 0x4900 && reg <= 0x49FF) return kRegBoolLoop;
  // SHADER_CONSTANT_FLUSH_FETCH_*: the recorder pairs them with the fetch
  // constants (measured ~1 per draw in the menu census).
  if (reg >= 0x5000 && reg <= 0x5003) return kRegFetchConst;
  if (reg >= 0x2000 && reg <= 0x2005) return kRegRenderTarget;
  if (reg >= 0x210F && reg <= 0x2114) return kRegViewport;
  if ((reg >= 0x200E && reg <= 0x200F) || (reg >= 0x2080 && reg <= 0x2082)) {
    return kRegScissor;
  }
  if ((reg >= 0x2318 && reg <= 0x231B) || reg == 0x2320 || reg == 0x2321) {
    return kRegCopy;
  }
  if (reg >= 0x231D && reg <= 0x231F) return kRegClearValue;
  if (reg == 0x2208) return kRegModeControl;
  if (reg >= 0x2000 && reg <= 0x23FF) return kRegDeviceState;
  // 0x0F01 RB_BC_CONTROL and neighbours: backend control, device family.
  if (reg >= 0x0F00 && reg <= 0x0F1F) return kRegDeviceState;
  // COHER plus the 0x04xx-0x05xx callback/scratch/WAIT_UNTIL sync block and
  // the 0x14xx-0x19xx display-controller block (primary surface address,
  // gamma LUT -- the once-per-frame swap buffer): all system.
  if (reg >= 0x0A00 && reg <= 0x0AFF) return kRegCoher;
  if (reg >= 0x0400 && reg <= 0x05FF) return kRegCoher;
  if (reg >= 0x1400 && reg <= 0x19FF) return kRegCoher;
  return kRegOther;
}

uint32_t WalkBufferState(const uint8_t* raw, uint32_t dwords,
                         StateWalkResult* out, uint8_t* draw_flags,
                         uint32_t max_draws) {
  *out = StateWalkResult{};
  uint32_t nflags = 0;

  // In-buffer context. mode < 0 means no RB_MODECONTROL written yet -- the
  // draw runs under whatever the ring left in the register file.
  int32_t mode = -1;
  uint8_t seen = 0;  // kDrawFlagRtSet | kDrawFlagVportSet | kDrawFlagScissorSet

  // One register write, with its value, updating the context.
  auto write_reg = [&](uint32_t reg, uint32_t value) {
    const RegClass c = ClassifyReg(reg);
    ++out->reg_writes[c];
    switch (c) {
      case kRegRenderTarget:
        seen |= kDrawFlagRtSet;
        break;
      case kRegViewport:
        seen |= kDrawFlagVportSet;
        break;
      case kRegScissor:
        seen |= kDrawFlagScissorSet;
        break;
      case kRegModeControl:
        mode = int32_t(value & 0x7);  // xenos::EdramMode, bits 0-2
        break;
      case kRegCopy:
        if (reg == 0x2318) out->last_copy_control = value;
        if (reg == 0x2319) out->last_copy_dest_base = value;
        if (reg == 0x231B) out->last_copy_dest_info = value;
        break;
      case kRegOther:
        TallyOther(out, reg);
        break;
      default:
        break;
    }
  };

  auto flag_draw = [&](bool is_22) {
    uint8_t f = seen;
    if (mode < 0) {
      f |= kDrawFlagModeInherited;
      ++out->inherited_draws;
    } else if (mode == 6) {  // xenos::EdramMode::kCopy
      f |= kDrawFlagCopyMode;
      ++out->copy_draws;
    }
    if (is_22 && nflags < max_draws) draw_flags[nflags++] = f;
  };

  for (uint32_t j = 0; j < dwords;) {
    const uint32_t hdr = BE32(raw, j);
    // A zero dword is a one-dword no-op (ExecutePacket tests `packet == 0`
    // before the type field). Decoding it as type-0 consumes two dwords and
    // desyncs the rest of the walk -- see nr_context.cpp's header.
    if (!hdr) {
      ++out->type2_pkts;
      ++j;
      continue;
    }
    const uint32_t ty = hdr >> 30;
    if (ty == 0) {
      ++out->type0_pkts;
      const uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;
      const uint32_t base = hdr & 0x7FFF;
      const bool one_reg = (hdr >> 15) & 1;
      for (uint32_t m = 0; m < cnt && j + 1 + m < dwords; ++m) {
        write_reg(one_reg ? base : base + m, BE32(raw, j + 1 + m));
      }
      j += 1 + cnt;
    } else if (ty == 1) {
      ++out->type1_pkts;
      if (j + 2 < dwords) {
        write_reg(hdr & 0x7FF, BE32(raw, j + 1));
        write_reg((hdr >> 11) & 0x7FF, BE32(raw, j + 2));
      }
      j += 3;
    } else if (ty == 2) {
      ++out->type2_pkts;
      ++j;
    } else {
      ++out->type3_pkts;
      const uint32_t op = (hdr >> 8) & 0x7F;
      const uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;
      ++out->type3_ops[op];
      if (op == 0x22) {
        ++out->draws22;
        flag_draw(true);
      } else if (op == 0x36) {
        ++out->draws36;
        flag_draw(false);
      } else if (op == 0x2D && cnt >= 2 && j + 1 < dwords) {
        // SET_CONSTANT: register writes by typed offset, values inline.
        // Bases mirror Write{ALU,Fetch,Bool,Loop,REGISTERS}RangeFromRing.
        static constexpr uint32_t kBases[5] = {0x4000, 0x4800, 0x4900, 0x4908,
                                               0x2000};
        const uint32_t offset_type = BE32(raw, j + 1);
        const uint32_t type = (offset_type >> 16) & 0xFF;
        if (type < 5) {
          const uint32_t base = kBases[type] + (offset_type & 0x7FF);
          for (uint32_t m = 0; m + 1 < cnt && j + 2 + m < dwords; ++m) {
            write_reg(base + m, BE32(raw, j + 2 + m));
          }
        }
      } else if (op == 0x2F && cnt >= 3 && j + 3 < dwords) {
        // LOAD_ALU_CONSTANT: same typed offset, but the VALUES live in
        // guest memory this pure walk cannot read. Classify the register
        // range and update the seen-state flags; a value-dependent context
        // change through this path (RB_MODECONTROL via a type-4 REGISTERS
        // load) would be invisible, so count it if the range covers it.
        static constexpr uint32_t kBases[5] = {0x4000, 0x4800, 0x4900, 0x4908,
                                               0x2000};
        const uint32_t offset_type = BE32(raw, j + 2);
        const uint32_t size_dwords = BE32(raw, j + 3) & 0xFFF;
        const uint32_t type = (offset_type >> 16) & 0xFF;
        if (type < 5) {
          const uint32_t base = kBases[type] + (offset_type & 0x7FF);
          for (uint32_t m = 0; m < size_dwords; ++m) {
            const uint32_t reg = base + m;
            ++out->reg_writes[ClassifyReg(reg)];
            switch (ClassifyReg(reg)) {
              case kRegRenderTarget: seen |= kDrawFlagRtSet; break;
              case kRegViewport: seen |= kDrawFlagVportSet; break;
              case kRegScissor: seen |= kDrawFlagScissorSet; break;
              case kRegModeControl: ++out->mode_loads_from_mem; break;
              case kRegOther: TallyOther(out, reg); break;
              default: break;
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
