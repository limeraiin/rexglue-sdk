/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_context.h"

#include <cstring>
#include <unordered_map>

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

// [NR-DRAW] The registers a draw packet's own payload writes. The executor
// writes these from ExecutePacketType3Draw before issuing (VGT_DRAW_INITIATOR
// always; the two DMA registers only for an indexed draw), so a walk that
// skips them leaves the index-buffer binding -- which is the draw's geometry
// source -- missing from the mirror while the live file has it. Values from
// register_table.inc; spelled numerically because this file builds bare.
constexpr uint32_t kRegVgtDmaBase = 0x21FA;
constexpr uint32_t kRegVgtDmaSize = 0x21FB;
constexpr uint32_t kRegVgtDrawInitiator = 0x21FC;
// VGT_DRAW_INITIATOR::source_select, bits 6-7. kDMA(0) is the indexed draw
// that carries VGT_DMA_BASE + VGT_DMA_SIZE after the initiator.
constexpr uint32_t kSourceSelectDma = 0;

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

uint32_t CtxConstantBase(uint32_t offset_type) {
  // The index field is 11 bits for the constant files and the register file
  // alike (ExecutePacketType3_SET_CONSTANT masks 0x7FF before dispatching).
  const uint32_t index = offset_type & 0x7FF;
  switch ((offset_type >> 16) & 0xFF) {
    case 0:
      return 0x4000 + index;  // ALU
    case 1:
      return 0x4800 + index;  // FETCH
    case 2:
      return 0x4900 + index;  // BOOL
    case 3:
      return 0x4908 + index;  // LOOP
    case 4:
      return 0x2000 + index;  // REGISTERS
    default:
      return kCtxNoBase;
  }
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

namespace {

// Every decoded write reaches reg_fn, mirrored or not: the resource census is
// interested in precisely the registers the mirror ignores.
void CtxWriteReg(CtxWalker* w, uint32_t reg, uint32_t value,
                 bool from_memory = false) {
  if (w->reg_fn) w->reg_fn(w->reg_user, reg, value, from_memory);
  const int32_t s = CtxSlot(reg);
  if (s < 0) return;
  w->ctx->values[s] = value;
  w->ctx->defined[s] = 1;
  w->ctx->in_buffer[s] = 1;
  if (w->watch_fn) {
    w->watch_fn(w->watch_user, reg, value, w->cur_dw, w->cur_hdr, w->cur_arg);
  }
}

// [NR-SKP] 5-4-3: a range may be offered in bulk only when it cannot touch
// the 27-reg recovery mirror. For such a range, per-dword CtxWriteReg would
// have done nothing but the reg_fn call (mirror store and watch_fn both fire
// only for mirrored slots), so a consumed range skips exactly the reg_fn
// calls and nothing else.
// [NR-PB] N-2-2 item 0: the check is per-slot, not the whole [0x2000, 0x2322)
// window -- a state range that misses all 27 mirrored slots (and so all 4
// watched registers, each of which IS a mirrored slot) is offerable. Whether
// the consumer takes a state range is its own gate (gpu_nr_plain_bulk in
// NrWalkRegRange); a declined offer falls to the per-dword path unchanged.
inline bool CtxRangeTouchesMirror(uint32_t base, uint32_t n) {
  for (const SlotRange& r : kRanges) {
    if (base < r.first_reg + r.count && base + n > r.first_reg) return true;
  }
  return false;
}
inline bool CtxRangeOfferable(const CtxWalker* w, uint32_t base, uint32_t n) {
  if (!w->range_fn || n == 0) return false;
  if (base >= 0x2322u || base + n <= 0x2000u) return true;
  return !CtxRangeTouchesMirror(base, n);
}

uint16_t CtxGroupBits(const StateContext* ctx, uint32_t first, uint32_t n,
                      uint16_t def_bit, uint16_t carry_bit) {
  bool def = true, all_local = true;
  for (uint32_t s = first; s < first + n; ++s) {
    if (!ctx->defined[s]) def = false;
    if (!ctx->in_buffer[s]) all_local = false;
  }
  if (!def) return 0;
  return uint16_t(def_bit | (all_local ? 0 : carry_bit));
}

// Computes this draw's flags word and stores it if there is room. Returns it
// either way, so a stop past max_draws still reports the draw truthfully.
uint16_t CtxFlagDraw(CtxWalker* w) {
  const StateContext* ctx = w->ctx;
  uint16_t f = 0;
  f |= CtxGroupBits(ctx, kRtRequired0, kRtRequiredN, kCtxDrawRtDef,
                    kCtxDrawRtCarried);
  f |= CtxGroupBits(ctx, kVportFirst, kVportN, kCtxDrawVportDef,
                    kCtxDrawVportCarried);
  f |= CtxGroupBits(ctx, kSlotMode, 1, kCtxDrawModeDef, kCtxDrawModeCarried);
  if (ctx->vs.valid && ctx->ps.valid) {
    f |= kCtxDrawShadersDef;
    if (!ctx->vs_in_buffer || !ctx->ps_in_buffer) f |= kCtxDrawShadersCarried;
  }
  if (ctx->defined[kSlotMode] && (ctx->values[kSlotMode] & 0x7) == 6) {
    f |= kCtxDrawCopy;
  }
  if (w->nflags < w->max_draws) w->draw_flags[w->nflags++] = f;
  return f;
}

// [NR-DRAW] The draw packet's own payload, applied exactly as
// ExecutePacketType3Draw applies it: VGT_DRAW_INITIATOR always, and for an
// indexed (kDMA) draw the two registers that carry the index buffer.
// DRAW_INDX leads with a viz-query token; DRAW_INDX_2 does not.
void CtxApplyDrawPayload(CtxWalker* w, uint32_t op, uint32_t j, uint32_t cnt) {
  const uint32_t end = j + 1 + cnt;
  uint32_t p = j + 1 + (op == 0x22 ? 1u : 0u);
  if (p >= end || p >= w->dwords) return;
  const uint32_t initiator = BE32(w->raw, p);
  CtxWriteReg(w, kRegVgtDrawInitiator, initiator);
  if (((initiator >> 6) & 3u) != kSourceSelectDma) return;
  if (++p < end && p < w->dwords) CtxWriteReg(w, kRegVgtDmaBase, BE32(w->raw, p));
  if (++p < end && p < w->dwords) CtxWriteReg(w, kRegVgtDmaSize, BE32(w->raw, p));
}

// [NR-SKP] The packet classes this walk decodes (or skips as pure no-ops)
// itself. Everything OUTSIDE this set is a delegate stop when the caller
// walks via CtxWalkNextStop: the executor's own handler runs the packet.
// INVALIDATE_STATE (0x3B) is native because the executor's implementation is
// read-and-ignore; PM4_NOP (0x10) skips by count exactly as the executor
// does. Keep in sync with the decode chain in CtxWalkStep below and with
// NrPktStreamOp (command_processor.cpp) minus the draws, which stop anyway.
bool CtxNativeOp(uint32_t op) {
  switch (op) {
    case 0x10:  // NOP
    case 0x22:  // DRAW_INDX        (draw stop)
    case 0x36:  // DRAW_INDX_2      (draw stop)
    case 0x27:  // IM_LOAD
    case 0x2B:  // IM_LOAD_IMMEDIATE
    case 0x2D:  // SET_CONSTANT
    case 0x2F:  // LOAD_ALU_CONSTANT
    case 0x3B:  // INVALIDATE_STATE (executor reads and ignores)
    case 0x50:  // SET_BIN_MASK
    case 0x51:  // SET_BIN_SELECT
    case 0x55:  // SET_CONSTANT2
    case 0x56:  // SET_SHADER_CONSTANTS
    case 0x60:  // SET_BIN_MASK_LO
    case 0x61:  // SET_BIN_MASK_HI
    case 0x62:  // SET_BIN_SELECT_LO
    case 0x63:  // SET_BIN_SELECT_HI
      return true;
    default:
      return false;
  }
}

// Decodes exactly one packet at the cursor, advancing it past that packet.
// Returns true when the packet was an EXECUTED draw, filling `stop`.
// [NR-SKP] With `delegate_stops` set (CtxWalkNextStop), also returns true --
// WITHOUT advancing the cursor -- for an executed type-3 packet outside the
// native set, so the caller can run the executor's handler on it.
bool CtxWalkStep(CtxWalker* w, CtxDrawStop* stop, bool delegate_stops = false) {
  const uint8_t* raw = w->raw;
  const uint32_t dwords = w->dwords;
  const uint32_t j = w->cursor;
  const uint32_t hdr = BE32(raw, j);
  // ExecutePacket's first test: a zero dword is a one-dword no-op.
  if (!hdr) {
    ++w->stats->nop0;
    w->cursor = j + 1;
    return false;
  }
  const uint32_t ty = hdr >> 30;
  w->cur_dw = j;
  w->cur_hdr = hdr;
  w->cur_arg = (j + 1 < dwords) ? BE32(raw, j + 1) : 0;
  if (ty == 0) {
    const uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;
    const uint32_t base = hdr & 0x7FFF;
    const bool one_reg = (hdr >> 15) & 1;
    // [NR-SKP] 5-4-3: a full-fit multi-register packet goes out as ONE range.
    // A one-reg packet stores cnt times to a single register (multiplicity a
    // bulk store cannot express) and a truncated packet keeps the per-dword
    // path's exact partial-apply behavior, so neither is offered.
    if (!one_reg && j + 1 + cnt <= dwords && CtxRangeOfferable(w, base, cnt) &&
        w->range_fn(w->range_user, base,
                    (const uint32_t*)(raw + (j + 1) * 4), cnt,
                    w->buffer_phys + (j + 1) * 4, /*from_memory=*/false)) {
      if (w->rec) w->rec->push_back({kCtxMemoRange, 0, uint16_t(base), cnt, j + 1, j});
      w->cursor = j + 1 + cnt;
      return false;
    }
    if (w->rec) w->rec->push_back({kCtxMemoPkt, 0, 0, 0, j, j});
    for (uint32_t m = 0; m < cnt && j + 1 + m < dwords; ++m) {
      CtxWriteReg(w, one_reg ? base : base + m, BE32(raw, j + 1 + m));
    }
    w->cursor = j + 1 + cnt;
    return false;
  }
  if (ty == 1) {
    if (j + 2 < dwords) {
      if (w->rec) w->rec->push_back({kCtxMemoPkt, 0, 0, 0, j, j});
      CtxWriteReg(w, hdr & 0x7FF, BE32(raw, j + 1));
      CtxWriteReg(w, (hdr >> 11) & 0x7FF, BE32(raw, j + 2));
    }
    w->cursor = j + 3;
    return false;
  }
  if (ty == 2) {
    w->cursor = j + 1;
    return false;
  }

  const uint32_t op = (hdr >> 8) & 0x7F;
  const uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;
  w->cursor = j + 1 + cnt;
  // Predicate bit: the command processor skips the whole packet when no
  // selected bin passes the mask. Mirror it before decoding anything --
  // a predicated-out packet writes nothing and a predicated-out draw does
  // not execute, so it gets no flags word either.
  if (CtxPredicatedOut(w->bin, hdr)) {
    ++w->stats->pred_skipped;
    if (op == 0x22) ++w->stats->pred_draws;
    return false;
  }
  if (hdr & 1) {
    if (op == 0x22) ++w->stats->pred_draws_run;
  }
  // [NR-SKP] Delegate stop: an executed type-3 packet this walk does not
  // decode. After the predicate check (a predicated-out packet never runs and
  // is skipped natively above), before any decode. The cursor stays AT the
  // header for the executor's dispatch; CtxWalkSkipDelegated resumes past it.
  if (delegate_stops && stop && !CtxNativeOp(op)) {
    if (w->rec) w->rec->push_back({kCtxMemoDeleg, 0, uint16_t(op), 0, j, j});
    w->cursor = j;
    ++w->stats->delegate_stops;
    stop->opcode = op;
    stop->dword = j;
    stop->flags = 0;
    stop->index = 0;
    stop->delegate = 1;
    return true;
  }
  if (op == 0x22 || op == 0x36) {
    if (w->rec) w->rec->push_back({kCtxMemoDraw, 0, uint16_t(op), 0, j, j});
    uint16_t f = 0;
    uint32_t index = 0;
    if (op == 0x22) {
      ++w->stats->draws22;
      f = CtxFlagDraw(w);
      // Ordinal among executed 0x22 draws, taken from the stat rather than the
      // flags array so it stays right past max_draws.
      index = w->stats->draws22 - 1;
      // After the flags, so both see the same moment: everything written by
      // this buffer up to and including the packets before this draw.
      if (w->draw_fn) w->draw_fn(w->draw_user);
    } else {
      ++w->stats->draws36;
    }
    CtxApplyDrawPayload(w, op, j, cnt);
    if (stop) {
      stop->opcode = op;
      stop->dword = j;
      stop->flags = f;
      stop->index = index;
      stop->delegate = 0;
    }
    return true;
  }
  StateContext* ctx = w->ctx;
  if (op == 0x27 && cnt >= 2 && j + 2 < dwords) {
    if (w->rec) w->rec->push_back({kCtxMemoPkt, 0, 0, 0, j, j});
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
    ++w->stats->im_loads;
    if (w->shader_fn) w->shader_fn(w->shader_user, ref);
  } else if (op == 0x2B && cnt >= 3 && j + 2 < dwords) {
    if (w->rec) w->rec->push_back({kCtxMemoPkt, 0, 0, 0, j, j});
    // IM_LOAD_IMMEDIATE: ucode inline after the two header dwords.
    const uint32_t type = BE32(raw, j + 1);
    const uint32_t start_size = BE32(raw, j + 2);
    ShaderRef ref;
    ref.addr = ((w->buffer_phys & 0x1FFFFFFF) + (j + 3) * 4);
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
    ++w->stats->im_load_imms;
    if (w->shader_fn) w->shader_fn(w->shader_user, ref);
  } else if (op == 0x2D && cnt >= 2 && j + 1 < dwords) {
    // SET_CONSTANT: typed offset, values inline. All five constant files,
    // not just type 4: the recovery mirror only ever needed the register
    // file, but the resource census consumes the ALU/fetch/bool/loop
    // writes through reg_fn and a walk that dropped them would report
    // them as never written.
    const uint32_t offset_type = BE32(raw, j + 1);
    const uint32_t base = CtxConstantBase(offset_type);
    if (base != kCtxNoBase) {
      // [NR-SKP] 5-4-3: full-fit constant payload as one range.
      if (j + 1 + cnt <= dwords && CtxRangeOfferable(w, base, cnt - 1) &&
          w->range_fn(w->range_user, base,
                      (const uint32_t*)(raw + (j + 2) * 4), cnt - 1,
                      w->buffer_phys + (j + 2) * 4, /*from_memory=*/false)) {
        if (w->rec) w->rec->push_back({kCtxMemoRange, 0, uint16_t(base), cnt - 1, j + 2, j});
        return false;
      }
      if (w->rec) w->rec->push_back({kCtxMemoPkt, 0, 0, 0, j, j});
      for (uint32_t m = 0; m + 1 < cnt && j + 2 + m < dwords; ++m) {
        CtxWriteReg(w, base + m, BE32(raw, j + 2 + m));
      }
    }
  } else if (CtxApplyBinPacket(&w->bin, op,
                               (j + 1 < dwords) ? BE32(raw, j + 1) : 0,
                               (j + 2 < dwords) ? BE32(raw, j + 2) : 0)) {
    // SET_BIN_MASK / SET_BIN_SELECT (and the LO/HI halves): which of the
    // following predicated blocks this pass executes.
    if (w->rec) w->rec->push_back({kCtxMemoPkt, 0, 0, 0, j, j});
    ++w->stats->bin_pkts;
  } else if ((op == 0x55 || op == 0x56) && cnt >= 2 && j + 1 < dwords) {
    // SET_CONSTANT2 / SET_SHADER_CONSTANTS: RAW register index, values
    // inline (ExecutePacketType3_SET_CONSTANT2 / _SET_SHADER_CONSTANTS ->
    // WriteRegisterRangeFromRing with no typed base).
    const uint32_t base = BE32(raw, j + 1) & 0xFFFF;
    ++w->stats->set_const2;
    // [NR-SKP] 5-4-3: same range shape as SET_CONSTANT, raw register base.
    if (j + 1 + cnt <= dwords && CtxRangeOfferable(w, base, cnt - 1) &&
        w->range_fn(w->range_user, base,
                    (const uint32_t*)(raw + (j + 2) * 4), cnt - 1,
                    w->buffer_phys + (j + 2) * 4, /*from_memory=*/false)) {
      if (w->rec) w->rec->push_back({kCtxMemoRange, 0, uint16_t(base), cnt - 1, j + 2, j});
      return false;
    }
    if (w->rec) w->rec->push_back({kCtxMemoPkt, 0, 0, 0, j, j});
    for (uint32_t m = 0; m + 1 < cnt && j + 2 + m < dwords; ++m) {
      CtxWriteReg(w, base + m, BE32(raw, j + 2 + m));
    }
  } else if (op == 0x2F && cnt >= 3 && j + 3 < dwords) {
    // LOAD_ALU_CONSTANT: typed offset, values in guest memory. Same five
    // files as SET_CONSTANT.
    const uint32_t address = BE32(raw, j + 1) & 0x1FFFFFFF;
    const uint32_t offset_type = BE32(raw, j + 2);
    const uint32_t size_dwords = BE32(raw, j + 3) & 0xFFF;
    const uint32_t base = CtxConstantBase(offset_type);
    if (base != kCtxNoBase) {
      // [NR-SKP] 5-4-3: by-reference range -- no inline values, the consumer
      // reads guest memory at `address` itself. Only offered when a memory
      // reader exists, so the no-reader poison semantics stay untouched.
      if (w->mem_read && CtxRangeOfferable(w, base, size_dwords) &&
          w->range_fn(w->range_user, base, nullptr, size_dwords, address,
                      /*from_memory=*/true)) {
        if (w->rec) w->rec->push_back({kCtxMemoRangeMem, 0, uint16_t(base), size_dwords, address, j});
        w->stats->mem_loads += size_dwords;
        return false;
      }
      if (w->rec) w->rec->push_back({kCtxMemoPkt, 0, 0, 0, j, j});
      for (uint32_t m = 0; m < size_dwords; ++m) {
        const int32_t s = CtxSlot(base + m);
        // A non-mirrored register still has to reach reg_fn: the resource
        // census cares about exactly the registers the mirror does not.
        if (!w->mem_read) {
          if (s >= 0) {
            ctx->defined[s] = 0;
            ctx->in_buffer[s] = 0;
          }
          ++w->stats->mem_poisoned;
          continue;
        }
        const uint32_t value = w->mem_read(w->mem_user, address + m * 4);
        CtxWriteReg(w, base + m, value, /*from_memory=*/true);
        ++w->stats->mem_loads;
      }
    }
  }
  return false;
}

// ---- [NR-WM] 5-4-8: memo store + replay drive ------------------------------
// Single-threaded on the CP thread, like the walker. Streams are owned here;
// the walker only ever holds borrowed pointers between Begin and End.

// 64MB thrashed at city under verify (naruto_446: ~52MB held, 5 clear-alls
// in a short run); 256MB gives the city working set room. Clear-all stays as
// the backstop only.
constexpr size_t kCtxMemoByteCap = 256u << 20;
constexpr size_t kCtxMemoStreamsPerBuf = 8;    // bin regimes per buffer

struct CtxMemoBufEntry {
  uint8_t refused = 0;
  std::vector<CtxMemoStream> streams;
};

std::unordered_map<uint32_t, CtxMemoBufEntry> g_memo_store;
CtxMemoStats g_memo_stats = {};
std::vector<CtxMemoOp> g_memo_rec;  // one recording at a time (CP thread)
bool g_memo_rec_busy = false;

// Replay one op stream through the walker's own callbacks. Contract: the
// caller proved the buffer bytes identical to the recorded execution and the
// entry bin state matches the stream key, so every re-parse (kCtxMemoPkt /
// kCtxMemoDraw / declined ranges) decodes the same bytes the recording did.
// `stop == nullptr` is the finish-drain: draw ops apply without surfacing and
// delegate ops are skipped, exactly as the parsed CtxWalkFinish behaves past
// the last stop. A cursor already at the end means the caller aborted the
// buffer (delegated ExecutePacket failure): nothing further applies, matching
// the parsed path's abort semantics.
bool CtxMemoNext(CtxWalker* w, CtxDrawStop* stop) {
  const bool deleg = w->rep_stops != 0;
  const uint32_t stream_end = w->rep_end ? w->rep_end : w->dwords;
  while (w->rep_i < w->rep_n) {
    if (w->cursor >= stream_end) return false;
    const CtxMemoOp& op = w->rep[w->rep_i];
    const bool scan = op.kind == kCtxMemoScan;
    // Where this op expects the cursor to be. In sync, every op leaves the
    // cursor exactly on the next op's anchor -- that invariant IS the framing
    // check, and it costs one compare per op.
    const uint32_t anchor = scan ? op.a : op.b;
    const uint32_t wend =
        scan ? ((op.a + op.n < w->dwords) ? op.a + op.n : w->dwords) : anchor;
    // [NR-TMPL] N-2 rung 1: FRAMING RESYNC. The in-place finalize can change
    // a span's framing, not just its values: a packet nop'd out leaves the
    // live parse SHORT of the next op, and a finalize longer than the
    // placeholder run it replaced leaves it PAST one. Neither may silently
    // skip or double-decode dwords -- the walk is the oracle. Behind: parse
    // live until the cursor reaches the anchor (the walker's own decode, so
    // the catch-up is bit-identical to a full parse). Ahead: drop the ops the
    // live framing already covered. Both resync at the very next op, so one
    // drifted packet costs its own neighbourhood, never the rest of the span.
    if (w->cursor > anchor && !(scan && w->cursor < wend)) {
      ++w->rep_i;
      ++w->stats->rep_ahead;
      continue;
    }
    if (w->cursor < anchor) {
      while (w->cursor < anchor) {
        ++w->stats->rep_catchup;
        if (CtxWalkStep(w, stop, deleg) && stop) return true;
      }
      continue;  // re-evaluate this op against the resynced cursor
    }
    // A live-scan window is RE-ENTRANT -- peeked, not consumed, until its
    // dwords are exhausted, so a stop inside the window resumes in the window
    // instead of skipping the rest of it. The window is the finalize class's
    // home: dwords that were placeholders when the template was built and may
    // be real packets now. Still-placeholder dwords are skipped by the same
    // rules CtxWalkStep applies, without the call.
    if (scan) {
      while (w->cursor < wend) {
        const uint32_t h = BE32(w->raw, w->cursor);
        if (!h || (h >> 30) == 2) {
          ++w->cursor;
          continue;
        }
        ++w->stats->scan_pkts;
        if (CtxWalkStep(w, stop, deleg) && stop) return true;
      }
      if (w->cursor > wend) ++w->stats->scan_over;
      ++w->rep_i;
      continue;
    }
    ++w->rep_i;
    // [NR-TMPL] Refresh the packet context for the range kinds (the re-parse
    // kinds set it through CtxWalkStep): range_fn consumers that read
    // cur_dw/cur_hdr (the N-2 template compare tags emissions by packet)
    // must see this op's packet, not whichever packet re-parsed last.
    if (op.kind == kCtxMemoRange || op.kind == kCtxMemoRangeMem) {
      w->cur_dw = op.b;
      w->cur_hdr = BE32(w->raw, op.b);
      w->cur_arg = (op.b + 1 < w->dwords) ? BE32(w->raw, op.b + 1) : 0;
    }
    // [NR-TMPL] Both range kinds consume their whole packet without a
    // re-parse, so they must advance the cursor themselves: the framing
    // resync above reads the cursor as the walk's position, and until rung 1
    // nothing did (the ops carry absolute positions, so a stale cursor was
    // invisible). Type-0 and type-3 share the count encoding, and the header
    // is read LIVE, so the length is the live framing's, not the record's.
    const uint32_t pkt_end = op.b + 2 + ((BE32(w->raw, op.b) >> 16) & 0x3FFF);
    switch (op.kind) {
      case kCtxMemoRange:
        if (w->range_fn &&
            w->range_fn(w->range_user, op.reg,
                        (const uint32_t*)(w->raw + size_t(op.a) * 4), op.n,
                        w->buffer_phys + op.a * 4, /*from_memory=*/false)) {
          w->cursor = pkt_end;
          break;
        }
        // Consumer declined what it consumed at record time (config drift):
        // re-parse the packet itself, bit-identically to the parsed walk.
        ++g_memo_stats.fallbacks;
        w->cursor = op.b;
        if (CtxWalkStep(w, stop, deleg) && stop) return true;
        break;
      case kCtxMemoRangeMem:
        // By-ref values are re-read from guest memory at replay, exactly as
        // the parsed walk re-reads them ([[bindings-inline-constants-byref]]).
        if (w->mem_read && w->range_fn &&
            w->range_fn(w->range_user, op.reg, nullptr, op.n, op.a,
                        /*from_memory=*/true)) {
          w->stats->mem_loads += op.n;
          w->cursor = pkt_end;
          break;
        }
        ++g_memo_stats.fallbacks;
        w->cursor = op.b;
        if (CtxWalkStep(w, stop, deleg) && stop) return true;
        break;
      case kCtxMemoPkt:
        w->cursor = op.a;
        // [NR-TMPL] Every re-parse gets the caller's stop and delegate rules,
        // not a bare apply: under the in-place finalize the packet recorded
        // here can BE a draw or a delegate today (a register write patched
        // into a DRAW_INDX is a measured class), and applying it silently
        // would swallow the stop the live walk surfaces. Byte-identical
        // replay never reaches this -- a pkt op decoded neither at record.
        if (CtxWalkStep(w, stop, deleg) && stop) return true;
        break;
      case kCtxMemoDraw: {
        w->cursor = op.a;
        const bool got = CtxWalkStep(w, stop, deleg);
        // [NR-TMPL] Under the memo's bin-identical contract a recorded draw
        // always executes at replay, so `got` used to be returned directly.
        // The N-2 template replay drives these ops under a DIFFERENT bin
        // state than the record (bin-agnostic streams, predication resolved
        // live), where a recorded draw can be predicated out: CtxWalkStep
        // then returns false with the cursor already past the packet, and
        // the stream must continue rather than report end-of-buffer.
        if (got && stop) return true;
        break;  // finish-drain or predicated-out: applied/skipped, not surfaced
      }
      case kCtxMemoDeleg:
        w->cursor = op.a;
        if (stop) {
          // [NR-TMPL] Same rule: a predicated-out delegate at replay is
          // skipped natively by CtxWalkStep (false, cursor advanced) and the
          // stream continues.
          if (CtxWalkStep(w, stop, /*delegate_stops=*/true)) return true;
          break;
        }
        CtxWalkSkipDelegated(w);  // finish-drain skips delegates, as parsed
        break;
      default:
        break;
    }
  }
  // [NR-TMPL] Ops exhausted with buffer left: in sync that is only the
  // trailing no-ops (every packet has an op, every no-op run has a window),
  // but under drift it can be real packets. Parse them rather than jumping
  // the cursor to the end -- dropping live packets is exactly the silent
  // divergence this whole gate exists to catch.
  while (w->cursor < stream_end) {
    ++w->stats->rep_catchup;
    if (CtxWalkStep(w, stop, deleg) && stop) return true;
  }
  return false;
}


// ---- [NR-PLAN] N-2-2: the flat plan interpreter -----------------------------
// One op per packet, no header decode, no type dispatch, no offerable test --
// everything the walk DERIVES from a packet was resolved once at record time
// and is guarded (nr_context.h [NR-PLAN]). Everything the walk READS is read
// here, from the live buffer, at replay.
//
// The drift machinery is CtxMemoNext's, unchanged and for the same reasons:
// in sync every op leaves the cursor on the next op's anchor, and that
// invariant IS the framing check. A scan window can still lengthen or shorten
// the live framing, so behind => parse live to the anchor, ahead => drop the
// ops the live framing already covered.
bool CtxPlanNext(CtxWalker* w, CtxDrawStop* stop) {
  const bool deleg = w->rep_stops != 0;
  const uint8_t* raw = w->raw;
  const uint32_t dwords = w->dwords;
  const uint32_t pbase = w->plan_base;
  const uint32_t stream_end =
      w->plan_end ? (pbase + w->plan_end < dwords ? pbase + w->plan_end : dwords)
                  : dwords;
  StateContext* ctx = w->ctx;
  while (w->plan_i < w->plan_n) {
    if (w->cursor >= stream_end) return false;
    const CtxPlanOp& op = w->plan[w->plan_i];
    const uint32_t at = pbase + op.hdr;
    const bool scan = op.kind == kPlanScan;
    const uint32_t wend =
        scan ? ((at + op.n < dwords) ? at + op.n : dwords) : at;
    if (w->cursor > at && !(scan && w->cursor < wend)) {
      ++w->plan_i;
      ++w->stats->rep_ahead;
      continue;
    }
    if (w->cursor < at) {
      while (w->cursor < at) {
        ++w->stats->rep_catchup;
        if (CtxWalkStep(w, stop, deleg) && stop) return true;
      }
      continue;  // re-evaluate this op against the resynced cursor
    }
    if (scan) {
      // Re-entrant, exactly as the memo's window: peeked until exhausted, so
      // a stop inside the window resumes inside the window.
      while (w->cursor < wend) {
        const uint32_t h = BE32(raw, w->cursor);
        if (!h || (h >> 30) == 2) {
          ++w->cursor;
          continue;
        }
        ++w->stats->scan_pkts;
        if (CtxWalkStep(w, stop, deleg) && stop) return true;
      }
      if (w->cursor > wend) ++w->stats->scan_over;
      ++w->plan_i;
      continue;
    }
    ++w->plan_i;
    if (op.kind == kPlanPkt) {
      // Guard-demoted or an odd shape: one full parse, so there is exactly
      // one decoder and it cannot drift from the walk.
      w->cursor = at;
      if (CtxWalkStep(w, stop, deleg) && stop) return true;
      continue;
    }
    const uint32_t hdr = BE32(raw, at);
    w->cur_dw = at;
    w->cur_hdr = hdr;
    w->cur_arg = (at + 1 < dwords) ? BE32(raw, at + 1) : 0;
    w->cursor = at + op.len;
    if (op.flags & kPlanFlagPred) {
      // Bin-agnostic: predication resolves against the state in effect NOW.
      if (!(w->bin.select & w->bin.mask)) {
        ++w->stats->pred_skipped;
        if (op.kind == kPlanDraw && op.reg == 0x22) ++w->stats->pred_draws;
        continue;
      }
      if (op.kind == kPlanDraw && op.reg == 0x22) ++w->stats->pred_draws_run;
    }
    const uint32_t payload = at + op.poff;
    if (op.flags & kPlanFlagSetConst2) ++w->stats->set_const2;
    switch (op.kind) {
      case kPlanRange:
        // Compiled as bulk-offerable: the run touches no mirrored slot, so a
        // declined offer costs exactly the reg_fn calls and nothing else --
        // the same equivalence CtxRangeOfferable rests on.
        if (w->range_fn && op.n &&
            w->range_fn(w->range_user, op.reg,
                        (const uint32_t*)(raw + size_t(payload) * 4), op.n,
                        w->buffer_phys + payload * 4, /*from_memory=*/false)) {
          break;
        }
        if (w->reg_fn) {
          for (uint32_t m = 0; m < op.n && payload + m < dwords; ++m) {
            w->reg_fn(w->reg_user, op.reg + m, BE32(raw, payload + m), false);
          }
        }
        break;
      case kPlanRegs:
        for (uint32_t m = 0; m < op.n && payload + m < dwords; ++m) {
          CtxWriteReg(w, op.reg + m, BE32(raw, payload + m));
        }
        break;
      case kPlanRegRep:
        for (uint32_t m = 0; m < op.n && payload + m < dwords; ++m) {
          CtxWriteReg(w, op.reg, BE32(raw, payload + m));
        }
        break;
      case kPlanReg: {
        const uint32_t v = BE32(raw, payload);
        if (w->reg_fn) w->reg_fn(w->reg_user, op.reg, v, false);
        if (op.slot >= 0) {
          ctx->values[op.slot] = v;
          ctx->defined[op.slot] = 1;
          ctx->in_buffer[op.slot] = 1;
          if (w->watch_fn) {
            w->watch_fn(w->watch_user, op.reg, v, at, hdr, w->cur_arg);
          }
        }
      } break;
      case kPlanRangeMem: {
        // The ADDRESS is patched in place by the recorder (the whole N-2-1
        // city residual was exactly this), so it is read live and never
        // guarded; type and size are guarded and compiled.
        const uint32_t address = BE32(raw, at + 1) & 0x1FFFFFFF;
        if (w->mem_read && w->range_fn && op.n &&
            w->range_fn(w->range_user, op.reg, nullptr, op.n, address,
                        /*from_memory=*/true)) {
          w->stats->mem_loads += op.n;
          break;
        }
        for (uint32_t m = 0; m < op.n; ++m) {
          if (!w->mem_read) {
            const int32_t sl = CtxSlot(op.reg + m);
            if (sl >= 0) {
              ctx->defined[sl] = 0;
              ctx->in_buffer[sl] = 0;
            }
            ++w->stats->mem_poisoned;
            continue;
          }
          CtxWriteReg(w, op.reg + m, w->mem_read(w->mem_user, address + m * 4),
                      /*from_memory=*/true);
          ++w->stats->mem_loads;
        }
      } break;
      case kPlanShader: {
        const uint32_t a1 = BE32(raw, at + 1);
        const uint32_t a2 = BE32(raw, at + 2);
        ShaderRef ref;
        ref.size_dwords = a2 & 0xFFFF;
        ref.valid = 1;
        bool is_ps;
        if (op.flags & kPlanFlagImm) {
          ref.addr = (w->buffer_phys & 0x1FFFFFFF) + (at + 3) * 4;
          ref.immediate = 1;
          is_ps = a1 != 0;
          ++w->stats->im_load_imms;
        } else {
          ref.addr = (a1 & ~3u) & 0x1FFFFFFF;
          ref.immediate = 0;
          is_ps = (a1 & 3u) != 0;
          ++w->stats->im_loads;
        }
        if (is_ps) {
          ctx->ps = ref;
          ctx->ps_in_buffer = 1;
        } else {
          ctx->vs = ref;
          ctx->vs_in_buffer = 1;
        }
        if (w->shader_fn) w->shader_fn(w->shader_user, ref);
      } break;
      case kPlanBin: {
        const uint32_t p0 = (at + 1 < dwords) ? BE32(raw, at + 1) : 0;
        const uint32_t p1 = (at + 2 < dwords) ? BE32(raw, at + 2) : 0;
        CtxApplyBinPacket(&w->bin, op.reg, p0, p1);
        ++w->stats->bin_pkts;
      } break;
      case kPlanDraw: {
        uint16_t f = 0;
        uint32_t index = 0;
        if (op.reg == 0x22) {
          ++w->stats->draws22;
          f = CtxFlagDraw(w);
          index = w->stats->draws22 - 1;
          if (w->draw_fn) w->draw_fn(w->draw_user);
        } else {
          ++w->stats->draws36;
        }
        CtxApplyDrawPayload(w, op.reg, at, uint32_t(op.len) - 1);
        if (stop) {
          stop->opcode = op.reg;
          stop->dword = at;
          stop->flags = f;
          stop->index = index;
          stop->delegate = 0;
          return true;
        }
      } break;
      case kPlanDeleg:
        if (stop) {
          // Cursor stays AT the header for the caller's ExecutePacket
          // dispatch; CtxWalkSkipDelegated resumes past it.
          w->cursor = at;
          ++w->stats->delegate_stops;
          stop->opcode = op.reg;
          stop->dword = at;
          stop->flags = 0;
          stop->index = 0;
          stop->delegate = 1;
          return true;
        }
        break;  // finish-drain skips delegates, as the parsed walk does
      case kPlanSkip:
      default:
        break;
    }
  }
  // Ops exhausted with stream left: in sync only trailing no-ops remain, but
  // under drift these can be real packets. Parse them; dropping live packets
  // is exactly the silent divergence the gates exist to catch.
  while (w->cursor < stream_end) {
    ++w->stats->rep_catchup;
    if (CtxWalkStep(w, stop, deleg) && stop) return true;
  }
  return false;
}

}  // namespace

void CtxWalkBegin(CtxWalker* w, const uint8_t* raw, uint32_t dwords,
                  uint32_t buffer_phys, StateContext* ctx, uint16_t* draw_flags,
                  uint32_t max_draws, CtxWalkStats* stats, CtxMemReadFn mem_read,
                  void* mem_user, CtxShaderFn shader_fn, void* shader_user,
                  CtxWatchFn watch_fn, void* watch_user, uint64_t bin_select,
                  uint64_t bin_mask, CtxRegWriteFn reg_fn, void* reg_user,
                  CtxDrawFn draw_fn, void* draw_user) {
  *w = CtxWalker{};
  w->raw = raw;
  w->dwords = dwords;
  w->buffer_phys = buffer_phys;
  w->ctx = ctx;
  w->draw_flags = draw_flags;
  w->max_draws = max_draws;
  w->stats = stats;
  w->mem_read = mem_read;
  w->mem_user = mem_user;
  w->shader_fn = shader_fn;
  w->shader_user = shader_user;
  w->watch_fn = watch_fn;
  w->watch_user = watch_user;
  w->reg_fn = reg_fn;
  w->reg_user = reg_user;
  w->draw_fn = draw_fn;
  w->draw_user = draw_user;
  // Bin state in effect for this pass, advanced by in-buffer SET_BIN_*.
  w->bin = CtxBinState{bin_select, bin_mask};

  *stats = CtxWalkStats{};
  // "Written by THIS buffer" resets at entry; definedness and values persist.
  for (uint32_t s = 0; s < kCtxRegCount; ++s) ctx->in_buffer[s] = 0;
  ctx->vs_in_buffer = 0;
  ctx->ps_in_buffer = 0;
}

bool CtxWalkNextDraw(CtxWalker* w, CtxDrawStop* stop) {
  if (w->plan) {
    w->rep_stops = 0;
    return CtxPlanNext(w, stop);
  }
  if (w->rep) {
    w->rep_stops = 0;
    return CtxMemoNext(w, stop);
  }
  while (w->cursor < w->dwords) {
    if (CtxWalkStep(w, stop)) return true;
  }
  return false;
}

bool CtxWalkNextStop(CtxWalker* w, CtxDrawStop* stop) {
  // [NR-PLAN] a compiled plan drives first; [NR-WM] a memo stream second.
  // The callbacks fire identically from all three paths.
  if (w->plan) {
    w->rep_stops = 1;
    return CtxPlanNext(w, stop);
  }
  if (w->rep) {
    w->rep_stops = 1;
    return CtxMemoNext(w, stop);
  }
  while (w->cursor < w->dwords) {
    if (CtxWalkStep(w, stop, /*delegate_stops=*/true)) return true;
  }
  return false;
}

bool CtxWalkStepOne(CtxWalker* w, CtxDrawStop* stop) {
  if (w->cursor >= w->dwords) return false;
  return CtxWalkStep(w, stop, /*delegate_stops=*/true);
}

void CtxWalkSkipDelegated(CtxWalker* w) {
  if (w->cursor >= w->dwords) return;
  const uint32_t hdr = BE32(w->raw, w->cursor);
  // Delegate stops are always type-3 packets (header + count payload dwords).
  w->cursor += 1 + (((hdr >> 16) & 0x3FFF) + 1);
}

uint32_t CtxWalkFinish(CtxWalker* w) {
  if (w->plan) {
    while (CtxPlanNext(w, nullptr)) {
    }
    return w->nflags;
  }
  if (w->rep) {
    while (CtxMemoNext(w, nullptr)) {
    }
    return w->nflags;
  }
  while (w->cursor < w->dwords) CtxWalkStep(w, nullptr);
  return w->nflags;
}

// ---- [NR-WM] store API ------------------------------------------------------

const CtxMemoStream* CtxMemoFind(uint32_t ptr, uint32_t dwords,
                                 uint64_t select, uint64_t mask) {
  auto it = g_memo_store.find(ptr);
  if (it == g_memo_store.end()) return nullptr;
  for (const CtxMemoStream& s : it->second.streams) {
    if (s.dwords == dwords && s.select == select && s.mask == mask) return &s;
  }
  return nullptr;
}

bool CtxMemoRefused(uint32_t ptr) {
  auto it = g_memo_store.find(ptr);
  return it != g_memo_store.end() && it->second.refused;
}

void CtxMemoRefuse(uint32_t ptr) {
  CtxMemoBufEntry& e = g_memo_store[ptr];
  if (!e.refused) ++g_memo_stats.refused;
  e.refused = 1;
  for (CtxMemoStream& s : e.streams) {
    g_memo_stats.bytes -= s.ops.capacity() * sizeof(CtxMemoOp);
    --g_memo_stats.streams;
  }
  if (!e.streams.empty()) {
    --g_memo_stats.bufs;
    e.streams.clear();
  }
}

void CtxMemoRecordBegin(CtxWalker* w) {
  if (g_memo_rec_busy) return;  // one at a time; caller nests never
  g_memo_rec_busy = true;
  g_memo_rec.clear();
  w->rec = &g_memo_rec;
}

void CtxMemoRecordAbandon(CtxWalker* w) {
  if (w->rec == &g_memo_rec) w->rec = nullptr;
  g_memo_rec.clear();
  g_memo_rec_busy = false;
}

bool CtxMemoRecordCommit(CtxWalker* w, uint32_t ptr, uint32_t dwords,
                         uint64_t select, uint64_t mask) {
  if (w->rec != &g_memo_rec) return true;
  w->rec = nullptr;
  g_memo_rec_busy = false;
  bool clean = true;
  const size_t add = g_memo_rec.size() * sizeof(CtxMemoOp);
  if (g_memo_stats.bytes + add > kCtxMemoByteCap) {
    CtxMemoClear();
    ++g_memo_stats.evicts;
    clean = false;
  }
  CtxMemoBufEntry& e = g_memo_store[ptr];
  if (e.refused) {
    g_memo_rec.clear();
    return clean;
  }
  CtxMemoStream* slot = nullptr;
  for (CtxMemoStream& s : e.streams) {
    if (s.dwords == dwords && s.select == select && s.mask == mask) {
      slot = &s;
      break;
    }
  }
  if (!slot) {
    if (e.streams.size() >= kCtxMemoStreamsPerBuf) {
      // Bin regime churn beyond any real tiling: start this buffer over.
      for (CtxMemoStream& s : e.streams) {
        g_memo_stats.bytes -= s.ops.capacity() * sizeof(CtxMemoOp);
        --g_memo_stats.streams;
      }
      e.streams.clear();
    }
    if (e.streams.empty()) ++g_memo_stats.bufs;
    e.streams.emplace_back();
    slot = &e.streams.back();
    ++g_memo_stats.streams;
  } else {
    g_memo_stats.bytes -= slot->ops.capacity() * sizeof(CtxMemoOp);
  }
  slot->select = select;
  slot->mask = mask;
  slot->dwords = dwords;
  slot->ops = std::move(g_memo_rec);
  g_memo_stats.bytes += slot->ops.capacity() * sizeof(CtxMemoOp);
  ++g_memo_stats.commits;
  g_memo_rec = std::vector<CtxMemoOp>();
  return clean;
}

bool CtxMemoRecordMatches(const CtxWalker* w, const CtxMemoStream* s,
                          uint32_t* first_ne) {
  if (w->rec != &g_memo_rec || !s) return false;
  const size_t n = g_memo_rec.size() < s->ops.size() ? g_memo_rec.size()
                                                     : s->ops.size();
  for (size_t i = 0; i < n; ++i) {
    if (std::memcmp(&g_memo_rec[i], &s->ops[i], sizeof(CtxMemoOp)) != 0) {
      if (first_ne) *first_ne = uint32_t(i);
      return false;
    }
  }
  if (g_memo_rec.size() != s->ops.size()) {
    if (first_ne) *first_ne = uint32_t(n);
    return false;
  }
  return true;
}

void CtxMemoReplayBegin(CtxWalker* w, const CtxMemoStream* s) {
  w->rep = s->ops.data();
  w->rep_n = uint32_t(s->ops.size());
  w->rep_i = 0;
}

void CtxMemoReplayEnd(CtxWalker* w) {
  w->rep = nullptr;
  w->rep_n = w->rep_i = 0;
}

uint32_t CtxMemoInvalidate(uint32_t ptr) {
  auto it = g_memo_store.find(ptr);
  if (it == g_memo_store.end()) return 0;
  uint32_t n = uint32_t(it->second.streams.size());
  for (CtxMemoStream& s : it->second.streams) {
    g_memo_stats.bytes -= s.ops.capacity() * sizeof(CtxMemoOp);
    --g_memo_stats.streams;
  }
  if (n) {
    --g_memo_stats.bufs;
    ++g_memo_stats.invals;
  }
  // Keep a refused mark; drop plain entries entirely.
  if (it->second.refused) {
    it->second.streams.clear();
  } else {
    g_memo_store.erase(it);
  }
  return n;
}

void CtxMemoClear() {
  g_memo_store.clear();
  g_memo_stats.bytes = 0;
  g_memo_stats.bufs = 0;
  g_memo_stats.streams = 0;
}

CtxMemoStats* CtxMemoStatsPtr() { return &g_memo_stats; }

// ---- [NR-PLAN] compile + guard ---------------------------------------------

void CtxPlanBegin(CtxWalker* w, const CtxPlanOp* ops, uint32_t nops,
                  uint32_t base_dw, uint32_t end_dw) {
  w->plan = ops;
  w->plan_n = nops;
  w->plan_i = 0;
  w->plan_base = base_dw;
  w->plan_end = end_dw;
}

void CtxPlanEnd(CtxWalker* w) {
  w->plan = nullptr;
  w->plan_n = w->plan_i = w->plan_base = w->plan_end = 0;
}

uint32_t CtxPlanCompile(const uint8_t* raw, uint32_t ndwords, CtxPlanOp* out,
                        uint32_t out_cap, CtxPlanGuard* guards,
                        uint32_t guard_cap, uint32_t* nguard,
                        CtxPlanCompileStats* st) {
  uint32_t np = 0, ng = 0, j = 0;
  if (st) *st = CtxPlanCompileStats{};
  if (nguard) *nguard = 0;
  bool full = false;
  auto emit = [&](uint8_t kind, int32_t slot, uint8_t poff, uint8_t flags,
                  uint32_t reg, uint32_t n, uint32_t len, uint32_t hdr) {
    if (np >= out_cap) {
      full = true;
      return;
    }
    CtxPlanOp& o = out[np++];
    o.kind = kind;
    o.slot = int8_t(slot);
    o.poff = poff;
    o.flags = flags;
    o.reg = uint16_t(reg);
    o.n = uint16_t(n);
    o.len = uint16_t(len);
    o.pad = 0;
    o.hdr = hdr;
  };
  // Guards attach to the op just emitted; a header two ops share (the type-1
  // pair) gets one guard each, so a demotion covers both halves.
  auto guard = [&](uint32_t pos, uint32_t val) {
    if (ng >= guard_cap || !np) {
      full = true;
      return;
    }
    guards[ng].pos = pos;
    guards[ng].val = val;
    guards[ng].op = np - 1;
    ++ng;
  };
  while (j < ndwords && !full) {
    const uint32_t h = BE32(raw, j);
    if (!h || (h >> 30) == 2) {
      // A placeholder run: the finalize's home. Stored as a window, never as
      // bytes -- the replay parses whatever stands there (see kCtxMemoScan).
      const uint32_t run = j;
      while (j < ndwords) {
        const uint32_t g = BE32(raw, j);
        if (g && (g >> 30) != 2) break;
        ++j;
      }
      emit(kPlanScan, -1, 0, 0, 0, j - run, j - run, run);
      if (st) ++st->scans;
      continue;
    }
    const uint32_t ty = h >> 30;
    if (ty == 1) {
      if (j + 3 > ndwords) return 0;
      const uint32_t r0 = h & 0x7FF, r1 = (h >> 11) & 0x7FF;
      emit(kPlanReg, CtxSlot(r0), 1, 0, r0, 1, 3, j);
      guard(j, h);
      emit(kPlanReg, CtxSlot(r1), 2, 0, r1, 1, 3, j);
      guard(j, h);
      if (st) st->regs += 2;
      j += 3;
      continue;
    }
    const uint32_t cnt = ((h >> 16) & 0x3FFF) + 1;
    const uint32_t len = 1 + cnt;
    if (j + len > ndwords) return 0;
    if (ty == 0) {
      const uint32_t base = h & 0x7FFF;
      const bool one_reg = (h >> 15) & 1;
      if (one_reg) {
        // n stores to ONE register: a multiplicity a bulk store cannot
        // express, exactly as CtxWalkStep refuses to offer it.
        emit(kPlanRegRep, -1, 1, 0, base, cnt, len, j);
        if (st) st->regs += cnt;
      } else if (CtxRangeTouchesMirror(base, cnt)) {
        emit(kPlanRegs, -1, 1, 0, base, cnt, len, j);
        if (st) st->regs += cnt;
      } else {
        emit(kPlanRange, -1, 1, 0, base, cnt, len, j);
        if (st) ++st->ranges;
      }
      guard(j, h);
      j += len;
      continue;
    }
    const uint32_t op = (h >> 8) & 0x7F;
    const uint8_t pf = (h & 1) ? kPlanFlagPred : 0;
    switch (op) {
      case 0x22:
      case 0x36:
        emit(kPlanDraw, -1, 0, pf, op, 0, len, j);
        guard(j, h);
        if (st) ++st->draws;
        break;
      case 0x27:
        if (cnt >= 2) {
          emit(kPlanShader, -1, 0, pf, op, 0, len, j);
        } else {
          emit(kPlanSkip, -1, 0, pf, op, 0, len, j);
        }
        guard(j, h);
        break;
      case 0x2B:
        if (cnt >= 3) {
          emit(kPlanShader, -1, 0, uint8_t(pf | kPlanFlagImm), op, 0, len, j);
        } else {
          emit(kPlanSkip, -1, 0, pf, op, 0, len, j);
        }
        guard(j, h);
        break;
      case 0x2D: {
        if (cnt < 2) {
          emit(kPlanSkip, -1, 0, pf, op, 0, len, j);
          guard(j, h);
          break;
        }
        const uint32_t ot = BE32(raw, j + 1);
        const uint32_t b = CtxConstantBase(ot);
        const uint32_t n = cnt - 1;
        if (b == kCtxNoBase) {
          emit(kPlanSkip, -1, 0, pf, op, 0, len, j);
        } else if (CtxRangeTouchesMirror(b, n)) {
          emit(kPlanRegs, -1, 2, pf, b, n, len, j);
          if (st) st->regs += n;
        } else {
          emit(kPlanRange, -1, 2, pf, b, n, len, j);
          if (st) ++st->ranges;
        }
        guard(j, h);
        guard(j + 1, ot);  // the typed offset the base is derived from
        break;
      }
      case 0x55:
      case 0x56: {
        if (cnt < 2) {
          emit(kPlanSkip, -1, 0, pf, op, 0, len, j);
          guard(j, h);
          break;
        }
        const uint32_t b1 = BE32(raw, j + 1);
        const uint32_t b = b1 & 0xFFFF;
        const uint32_t n = cnt - 1;
        const uint8_t fl = uint8_t(pf | kPlanFlagSetConst2);
        if (CtxRangeTouchesMirror(b, n)) {
          emit(kPlanRegs, -1, 2, fl, b, n, len, j);
          if (st) st->regs += n;
        } else {
          emit(kPlanRange, -1, 2, fl, b, n, len, j);
          if (st) ++st->ranges;
        }
        guard(j, h);
        guard(j + 1, b1);  // the RAW register base
        break;
      }
      case 0x2F: {
        if (cnt < 3 || j + 3 >= ndwords) {
          emit(kPlanSkip, -1, 0, pf, op, 0, len, j);
          guard(j, h);
          break;
        }
        const uint32_t ot = BE32(raw, j + 2);
        const uint32_t szd = BE32(raw, j + 3);
        const uint32_t b = CtxConstantBase(ot);
        const uint32_t n = szd & 0xFFF;
        if (b == kCtxNoBase) {
          emit(kPlanSkip, -1, 0, pf, op, 0, len, j);
        } else if (!n || CtxRangeTouchesMirror(b, n)) {
          // The walker's per-dword by-ref path poisons/writes slot by slot;
          // one re-parse is cheaper than reproducing it, and it is rare.
          emit(kPlanPkt, -1, 0, pf, 0, 0, len, j);
          if (st) ++st->pkts;
        } else {
          emit(kPlanRangeMem, -1, 0, pf, b, n, len, j);
          if (st) ++st->ranges;
        }
        guard(j, h);
        guard(j + 2, ot);   // offset|type
        guard(j + 3, szd);  // size_dwords  (dw+1, the ADDRESS, is read live)
        break;
      }
      case 0x50:
      case 0x51:
      case 0x60:
      case 0x61:
      case 0x62:
      case 0x63:
        emit(kPlanBin, -1, 0, pf, op, 0, len, j);
        guard(j, h);
        break;
      case 0x10:  // NOP: the executor skips by count
      case 0x3B:  // INVALIDATE_STATE: the executor reads and ignores
        emit(kPlanSkip, -1, 0, pf, op, 0, len, j);
        guard(j, h);
        if (st) ++st->skips;
        break;
      default:
        emit(kPlanDeleg, -1, 0, pf, op, 0, len, j);
        guard(j, h);
        if (st) ++st->delegs;
        break;
    }
    j += len;
  }
  if (full || j != ndwords) return 0;
  if (nguard) *nguard = ng;
  if (st) {
    st->ops = np;
    st->guards = ng;
  }
  return np;
}

uint32_t CtxPlanApplyGuards(const uint8_t* live, uint32_t live_dwords,
                            CtxPlanOp* ops, uint32_t nops,
                            const CtxPlanGuard* guards, uint32_t nguard) {
  uint32_t demoted = 0;
  for (uint32_t k = 0; k < nguard; ++k) {
    const CtxPlanGuard& g = guards[k];
    if (g.pos < live_dwords && BE32(live, g.pos) == g.val) continue;
    if (g.op >= nops) continue;
    CtxPlanOp& o = ops[g.op];
    if (o.kind == kPlanPkt) continue;  // already demoted by a sibling guard
    o.kind = kPlanPkt;
    o.slot = -1;
    o.poff = 0;
    o.flags = 0;
    o.reg = 0;
    o.n = 0;
    ++demoted;
  }
  return demoted;
}

uint32_t WalkBufferContext(const uint8_t* raw, uint32_t dwords,
                           uint32_t buffer_phys, StateContext* ctx,
                           uint16_t* draw_flags, uint32_t max_draws,
                           CtxWalkStats* stats, CtxMemReadFn mem_read,
                           void* mem_user, CtxShaderFn shader_fn,
                           void* shader_user, CtxWatchFn watch_fn,
                           void* watch_user, uint64_t bin_select,
                           uint64_t bin_mask, CtxRegWriteFn reg_fn,
                           void* reg_user, CtxDrawFn draw_fn,
                           void* draw_user) {
  CtxWalker w;
  CtxWalkBegin(&w, raw, dwords, buffer_phys, ctx, draw_flags, max_draws, stats,
               mem_read, mem_user, shader_fn, shader_user, watch_fn, watch_user,
               bin_select, bin_mask, reg_fn, reg_user, draw_fn, draw_user);
  return CtxWalkFinish(&w);
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
