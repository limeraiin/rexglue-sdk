/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_CONTEXT_H_
#define REX_GRAPHICS_NR_CONTEXT_H_

#include <cstdint>

// [NR-CTX] The RUNNING state context: native-renderer build-out increment 4a.
//
// Increment 3's city census killed context-free translation: 81% of city
// draws execute with an RT never written inside their own buffer, 40% before
// any in-buffer RB_MODECONTROL. The replay therefore carries a running state
// context across buffers in EXECUTION order -- which is exactly what the real
// register file is. This unit is that context, built the way the native
// renderer will build it: a persistent mirror of the recovery register set
// (render target, viewport, scissor, RB_MODECONTROL, copy/resolve, clear
// values) plus the active vertex/pixel shader from IM_LOAD /
// IM_LOAD_IMMEDIATE, updated by walking each executed buffer's packets in
// order.
//
// What increment 4a measures (the consumer, cvar gpu_nr_ctx):
//
//   1. COMPLETENESS -- with carry, what fraction of draws have a fully
//      defined RT + viewport + mode + shader context? (In-buffer alone the
//      city reads 19%/15%/60%; carried it should approach 100% after the
//      first frames. If it does not, the context has a source this walk
//      cannot see.)
//
//   2. GROUND TRUTH -- at buffer entry the consumer compares every defined
//      mirrored register against the live RegisterFile, which at that moment
//      holds exactly the carried state a ring-order replay would have.
//      Divergence means recovery state arrives OUTSIDE depth-1 indirect
//      buffers (primary-ring writes, nested buffers) and names the register.
//      This is the gate: diverge ~0 means the IB stream alone determines the
//      recovery state and the native replay needs nothing else.
//
//   3. CARRY ATTRIBUTION -- per draw, is each context group in effect from
//      THIS buffer or carried from a previous one? Prices what cross-buffer
//      carry actually does (the menu should read ~0% carry, the city high).
//
//   4. SHADER TRAFFIC -- IM_LOAD/IM_LOAD_IMMEDIATE rates and per-window
//      distinct shader count: the working-set size increment 4b's live
//      translation cache must absorb (the offline corpus is 3,320 unique).
//
// Pure functions + a caller-owned context struct; no globals, no threads, no
// SDK dependencies (the unit test builds this file bare). The consumer owns
// counters and the 1Hz report; tools/nr-context-test.cpp owns the fixtures.

namespace rex {
namespace graphics {
namespace nr {

// ---- The mirrored recovery register set ------------------------------------
// Slots are dense indices over the exact registers the replay must recover
// from buffers (the increment-3 classes, minus the hookable device family).

enum CtxGroup : uint8_t {
  kCtxGroupRt,        // 0x2000-0x2005 RB_SURFACE_INFO/COLOR*_INFO/DEPTH_INFO
  kCtxGroupViewport,  // 0x210F-0x2114 PA_CL_VPORT_*
  kCtxGroupScissor,   // 0x200E-0x200F, 0x2080-0x2082
  kCtxGroupMode,      // 0x2208 RB_MODECONTROL
  kCtxGroupCopy,      // 0x2318-0x231B, 0x2320-0x2321 RB_COPY_*
  kCtxGroupClear,     // 0x231D-0x231F RB_*_CLEAR
  kCtxGroupCount
};

constexpr uint32_t kCtxRegCount = 27;

// reg -> slot (-1 if not mirrored), and the inverses for reporting.
int32_t CtxSlot(uint32_t reg);
uint32_t CtxSlotReg(uint32_t slot);
CtxGroup CtxSlotGroup(uint32_t slot);

struct ShaderRef {
  uint32_t addr;         // physical address of the ucode, masked 0x1FFFFFFF
  uint32_t size_dwords;  // ucode length
  uint8_t immediate;     // 1 = ucode lives inside the buffer (IM_LOAD_IMMEDIATE)
  uint8_t valid;
};

// The persistent context. Zero-initialize once; WalkBufferContext updates it
// per executed buffer, in execution order. in_buffer[] and the *_in_buffer
// shader bits are reset at each walk's entry -- they mean "written by the
// buffer currently being walked", which is what carry attribution needs.
struct StateContext {
  uint32_t values[kCtxRegCount];
  uint8_t defined[kCtxRegCount];
  uint8_t in_buffer[kCtxRegCount];
  ShaderRef vs, ps;
  uint8_t vs_in_buffer, ps_in_buffer;
};

// ---- Per-draw effective-context flags --------------------------------------
// One uint16 per DRAW_INDX (0x22), packet order, index-aligned with the
// consumer's join list (same walk rules, same packet subset).
//
// "Defined" bits ask: is the group fully established (every required register
// written at least once since boot, by any buffer)? RT requires
// SURFACE_INFO + COLOR_INFO + DEPTH_INFO (0x2000-0x2002; COLOR1-3 are
// mirrored but not required -- single-RT title). Viewport requires all six
// PA_CL_VPORT_*. Shaders require a valid vs AND ps.
//
// "Carried" bits ask: is the group defined but NOT fully re-established by
// THIS buffer before this draw -- i.e. does this draw depend on cross-buffer
// carry for at least one required register?
constexpr uint16_t kCtxDrawRtDef = 0x0001;
constexpr uint16_t kCtxDrawVportDef = 0x0002;
constexpr uint16_t kCtxDrawModeDef = 0x0004;
constexpr uint16_t kCtxDrawShadersDef = 0x0008;
constexpr uint16_t kCtxDrawCopy = 0x0010;  // effective EDRAM mode == kCopy(6)
constexpr uint16_t kCtxDrawRtCarried = 0x0020;
constexpr uint16_t kCtxDrawVportCarried = 0x0040;
constexpr uint16_t kCtxDrawModeCarried = 0x0080;
constexpr uint16_t kCtxDrawShadersCarried = 0x0100;

struct CtxWalkStats {
  uint32_t draws22;         // 0x22 packets seen (flags may be capped by max_draws)
  uint32_t im_loads;        // IM_LOAD (0x27) packets applied
  uint32_t im_load_imms;    // IM_LOAD_IMMEDIATE (0x2B) packets applied
  uint32_t mem_loads;       // LOAD_ALU_CONSTANT dwords applied to mirrored regs
                            // through the memory reader
  uint32_t mem_poisoned;    // same, but no reader supplied: the slot is set
                            // UNDEFINED (its value is unknowable to this walk)
};

// Optional reader for LOAD_ALU_CONSTANT values (they live in guest memory,
// not the packet). Must return the register value in HOST byte order for the
// big-endian dword at physical address `phys`. Pass nullptr to poison
// covered mirrored slots instead (counted in mem_poisoned).
using CtxMemReadFn = uint32_t (*)(void* user, uint32_t phys);

// Optional per-shader-load callback, invoked for every applied IM_LOAD /
// IM_LOAD_IMMEDIATE in packet order -- the consumer's distinct-shader census
// (sizing increment 4b's live translation cache) without the walker owning
// any storage.
using CtxShaderFn = void (*)(void* user, const ShaderRef& ref);

// Walk one executed indirect buffer (big-endian PM4, `dwords` long) at
// physical address `buffer_phys`, updating `ctx` and emitting one flags word
// per DRAW_INDX (0x22) into draw_flags, packet order, up to max_draws
// (excess draws still update the context and stats; flags dropped). `stats`
// is zeroed first. Returns the number of flag words written.
uint32_t WalkBufferContext(const uint8_t* raw, uint32_t dwords,
                           uint32_t buffer_phys, StateContext* ctx,
                           uint16_t* draw_flags, uint32_t max_draws,
                           CtxWalkStats* stats, CtxMemReadFn mem_read,
                           void* mem_user, CtxShaderFn shader_fn = nullptr,
                           void* shader_user = nullptr);

// [NR-RING] Increment 4b-0: the ring-side observer's apply. The 4a city
// verdict proved exactly 4 recovery registers arrive OUTSIDE the depth-1 IB
// stream (per-frame swap state: PA_SC_WINDOW_OFFSET/SCISSOR_TL/BR +
// RB_COPY_DEST_BASE); the consumer taps those writes at the command
// processor's WriteRegister and applies them here. Sets value + defined for a
// mirrored register but NOT in_buffer -- an out-of-stream value is by
// definition not established by any buffer, so draws depending on it
// truthfully read as carried. Returns the slot, or -1 if `reg` is not
// mirrored (nothing written).
int32_t CtxApplyExternalWrite(StateContext* ctx, uint32_t reg, uint32_t value);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_CONTEXT_H_
