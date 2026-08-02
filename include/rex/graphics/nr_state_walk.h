/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_STATE_WALK_H_
#define REX_GRAPHICS_NR_STATE_WALK_H_

#include <cstdint>

// [NR-STATE] State recovery from the packet walk: native-renderer build-out
// increment 3.
//
// The D3D9 map closed most of Risk 2: blend/depth/stencil/colour-write/
// alpha-test and all sampler state are hookable device methods with known
// shadows (the two bctr dispatch tables). What is NOT hookable and must be
// recovered from the recorded command buffers themselves: render targets,
// clears, resolves and viewport. Increments 1-2 settled that the renderer
// walks every executed buffer's packets and joins draws per-draw; this unit
// rides that same walk and answers, with measurements instead of hope:
//
//   1. CENSUS -- what register writes and type-3 opcodes actually appear
//      inside Naruto's indirect buffers, classified into: shader constants,
//      fetch constants, bool/loop, render-target (RB_SURFACE_INFO/
//      RB_COLOR*_INFO/RB_DEPTH_INFO), viewport (PA_CL_VPORT_*), scissor/
//      window, copy/resolve (RB_COPY_*), clear values (RB_*_CLEAR),
//      RB_MODECONTROL, the hookable device-state family, COHER, other
//      (top offenders named). Anything big in 'other' is a coverage hole
//      with a name.
//
//   2. RESOLVES/CLEARS -- on Xenos a resolve IS a draw executed with
//      RB_MODECONTROL.edram_mode == kCopy (6), clears ride the same path
//      via RB_COPY_CONTROL; there is no separate resolve packet. So some
//      DRAW_INDX packets in the buffers are not geometry at all. The walk
//      flags each draw with the mode in effect, which also tests the
//      standing hypothesis that the [nr-cache] join misses are precisely
//      the copy-mode draws (the D3D9 Resolve() path is not one of the
//      three hooked draw recorders).
//
//   3. SELF-CONTAINMENT -- per draw, was RT/viewport/mode state written
//      earlier IN THIS BUFFER, or inherited from whatever executed before
//      it? Buffers are recorded once and replayed for many frames, so
//      inherited state means the renderer must carry a state context
//      across buffer boundaries in ring order; in-buffer state means each
//      buffer re-establishes its own. The split decides how much context
//      the replay needs.
//
// Pure functions over a big-endian PM4 buffer; no globals, no threads. The
// consumer (command_processor.cpp, cvar gpu_nr_state) owns the counters and
// the 1Hz report; tools/nr-state-walk-test.cpp owns the fixtures.

namespace rex {
namespace graphics {
namespace nr {

// Per-DRAW_INDX (0x22) context flags, in packet order -- index-aligned with
// the consumer's join list (same walk rules, same packet subset).
constexpr uint8_t kDrawFlagCopyMode = 0x01;   // edram_mode == kCopy in effect
constexpr uint8_t kDrawFlagModeInherited = 0x02;  // no RB_MODECONTROL yet
constexpr uint8_t kDrawFlagRtSet = 0x04;      // an RT reg written earlier in-buffer
constexpr uint8_t kDrawFlagVportSet = 0x08;   // a viewport reg written earlier
constexpr uint8_t kDrawFlagScissorSet = 0x10; // a scissor/window reg written earlier

enum RegClass : uint32_t {
  kRegShaderConst,  // 0x4000-0x47FF float constants (the transform firehose)
  kRegFetchConst,   // 0x4800-0x48FF vertex/texture fetch constants
  kRegBoolLoop,     // 0x4900-0x490F
  kRegRenderTarget, // 0x2000-0x2005 SURFACE/COLOR0-3/DEPTH INFO
  kRegViewport,     // 0x210F-0x2114 PA_CL_VPORT_*
  kRegScissor,      // 0x200E-0x200F, 0x2080-0x2082 screen/window scissor
  kRegCopy,         // 0x2318-0x231B, 0x2320-0x2321 RB_COPY_*
  kRegClearValue,   // 0x231D-0x231F RB_DEPTH_CLEAR/RB_COLOR_CLEAR(_LO)
  kRegModeControl,  // 0x2208 RB_MODECONTROL
  kRegDeviceState,  // the rest of 0x2000-0x23FF: the hookable family
  kRegCoher,        // 0x0A00-0x0AFF coherency/system
  kRegOther,        // anything else (tallied by index below)
  kRegClassCount
};

RegClass ClassifyReg(uint32_t reg);

struct StateWalkResult {
  uint64_t reg_writes[kRegClassCount];  // per class, in register-dwords
  uint32_t type3_ops[128];              // count per type-3 opcode
  uint32_t type0_pkts, type1_pkts, type2_pkts, type3_pkts;
  uint32_t draws22, draws36;
  uint32_t copy_draws;       // draws (either opcode) under kCopy
  uint32_t inherited_draws;  // draws before any in-buffer RB_MODECONTROL
  uint32_t mode_loads_from_mem;  // RB_MODECONTROL covered by a memory load
                                 // (value invisible to a pure walk)
  // Last copy setup seen, so a report sample can name a resolve target.
  uint32_t last_copy_control, last_copy_dest_base, last_copy_dest_info;
  // Top unclassified registers, so a coverage hole names itself.
  struct OtherReg {
    uint32_t reg;
    uint32_t count;
  } other_top[8];
};

// Walk one indirect buffer (big-endian PM4, `dwords` long), classify every
// packet into `out` (zeroed first), and emit one flags byte per DRAW_INDX
// (0x22) packet into draw_flags, packet order, up to max_draws (excess draws
// still counted in the census, flags dropped). Returns the number of flag
// bytes written.
uint32_t WalkBufferState(const uint8_t* raw, uint32_t dwords,
                         StateWalkResult* out, uint8_t* draw_flags,
                         uint32_t max_draws);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_STATE_WALK_H_
