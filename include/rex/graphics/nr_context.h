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
#include <vector>

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

// ---- Predicated tiling -----------------------------------------------------
// Shared by every walker that must agree with the executor about which
// packets actually run (this walk, the per-draw join list, the ledger's draw
// count): one rule, one place, unit-tested once.

struct CtxBinState {
  uint64_t select = 0xFFFFFFFFull;  // CommandProcessor::bin_select_ at entry
  uint64_t mask = 0xFFFFFFFFull;    // CommandProcessor::bin_mask_ at entry
};

// True when a type-3 header is predicated (bit 0) and no selected bin passes
// the mask: the command processor skips the whole packet.
inline bool CtxPredicatedOut(const CtxBinState& bin, uint32_t hdr) {
  return (hdr & 1) && !(bin.select & bin.mask);
}

// Apply a SET_BIN_MASK/SELECT packet (ops 0x50, 0x51, 0x60-0x63) to the bin
// state; any other opcode is ignored. `p0`/`p1` are the two dwords after the
// header (pass 0 for dwords past the end of the buffer). Returns true if the
// packet was a bin packet.
bool CtxApplyBinPacket(CtxBinState* bin, uint32_t op, uint32_t p0, uint32_t p1);

// ---- Constant-file addressing ----------------------------------------------
// SET_CONSTANT (0x2D) and LOAD_ALU_CONSTANT (0x2F) both carry a typed offset
// whose type byte selects one of five register files. This is the executor's
// own map (CommandProcessor::Write{ALU,Fetch,Bool,Loop,REGISTERS}RangeFromRing):
//   0 ALU 0x4000 · 1 FETCH 0x4800 · 2 BOOL 0x4900 · 3 LOOP 0x4908 ·
//   4 REGISTERS 0x2000
// Kept here so every walker uses one map. The recovery mirror only ever needed
// type 4, which is why the 4a/4b walk decoded that alone.
constexpr uint32_t kCtxNoBase = 0xFFFFFFFFu;

// Absolute base register for a packet's offset_type dword, or kCtxNoBase for
// an unknown type. The index field is already added in.
uint32_t CtxConstantBase(uint32_t offset_type);

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
  uint32_t draws36;         // executed DRAW_INDX_2 packets (no flags, no draw_fn)
  uint32_t im_loads;        // IM_LOAD (0x27) packets applied
  uint32_t im_load_imms;    // IM_LOAD_IMMEDIATE (0x2B) packets applied
  uint32_t mem_loads;       // LOAD_ALU_CONSTANT dwords applied to mirrored regs
                            // through the memory reader
  uint32_t mem_poisoned;    // same, but no reader supplied: the slot is set
                            // UNDEFINED (its value is unknowable to this walk)
  uint32_t nop0;            // zero dwords consumed as one-dword no-ops
  uint32_t set_const2;      // SET_CONSTANT2 / SET_SHADER_CONSTANTS packets
  uint32_t pred_skipped;    // predicated type-3 packets the bin check rejected
  uint32_t pred_draws;      // of those, DRAW_INDX (0x22) -- draws that do NOT
                            // execute in this bin and get no flags word
  uint32_t pred_draws_run;  // predicated draws that DID pass the bin check
                            // (bin-dependent draws: the replay cannot cache a
                            // buffer's draw list across bins if this is > 0)
  uint32_t bin_pkts;        // SET_BIN_MASK/SELECT packets seen inside the
                            // buffer (0 => bin state only ever arrives from
                            // outside, i.e. per-tile replay of one buffer)
  uint32_t delegate_stops;  // [NR-SKP] non-native type-3 packets surfaced as
                            // delegate stops by CtxWalkNextStop
  uint32_t scan_pkts;       // [NR-TMPL] packets decoded inside live-scan
                            // windows (the finalize class, seen at replay)
  uint32_t scan_over;       // scan windows whose live framing ran PAST the
                            // window end: the finalize changed the length,
                            // and the ops after it re-anchor over decoded
                            // dwords. Never silent, never zero-checked away.
  uint32_t rep_catchup;     // [NR-TMPL] packets parsed live to resync the
                            // replay after the live framing landed SHORT of
                            // the next op (a packet nop'd out in place)
  uint32_t rep_ahead;       // ops dropped because the live framing had
                            // already passed them (a finalize longer than
                            // the placeholder run it replaced)
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

// [NR-WSAMP] Increment 4b-1: the walker-side write sampler. Invoked for every
// write this walk applies to a mirrored register, with the packet it came
// from: `dw` is the header's dword index in the buffer, `hdr` the header
// itself and `arg` the dword after it (the typed offset for SET_CONSTANT, the
// first value for type-0, 0 when the buffer ends). Paired with an
// executor-side twin on the same registers, a one-frame diff names the exact
// packet bytes behind any divergence instead of inferring it.
using CtxWatchFn = void (*)(void* user, uint32_t reg, uint32_t value,
                            uint32_t dw, uint32_t hdr, uint32_t arg);

// [NR-RES] Increment 4b-3: EVERY register write this walk decodes, mirrored
// or not, with where it came from. The recovery mirror covers 27 registers;
// the resource half of a draw (vertex/texture fetch constants, the ALU
// constant file) is thousands more, and collecting those with a second walk
// would mean a second decoder to keep in step -- exactly the drift 4b-1 had
// to repair across four walkers. So the resource census consumes this stream
// instead of re-deriving it.
//
// `from_memory` marks a value the walk obtained through CtxMemReadFn rather
// than from the packet (LOAD_ALU_CONSTANT). That distinction is a design
// input, not bookkeeping: a buffer is replayed for many frames after it was
// recorded, so state that arrives by reference to guest memory must be read
// at REPLAY time, while state carried inline in the packet is fixed when the
// buffer is recorded.
using CtxRegWriteFn = void (*)(void* user, uint32_t reg, uint32_t value,
                               bool from_memory);

// [NR-RES] Invoked at each EXECUTED draw, in packet order, at the moment its
// state is in effect. The resource census needs this rather than a tally
// after the walk: carry attribution asks what the buffer had written BEFORE
// this draw, which end-of-buffer state cannot answer.
using CtxDrawFn = void (*)(void* user);

// [NR-SKP] Phase 5-4-3: bulk range delivery. When set (a plain field write
// after CtxWalkBegin -- the begin signature stays put), a contiguous decoded
// register range that touches NO mirrored slot is offered here INSTEAD of one
// reg_fn call per dword; the only walker work a consumed range skips is
// exactly those reg_fn calls (mirror and watch_fn updates never applied to
// such a range in the first place). `values_be` points at the n big-endian
// dwords inside the buffer for inline packets (type-0, SET_CONSTANT,
// SET_CONSTANT2/SET_SHADER_CONSTANTS); for LOAD_ALU_CONSTANT it is nullptr
// and the values live in guest memory at physical address `phys`
// (from_memory=true, same meaning as CtxRegWriteFn's flag). Return true =
// consumed in bulk; false = the walker falls back to the per-dword path,
// bit-identically to a walk with no range_fn at all.
using CtxRegRangeFn = bool (*)(void* user, uint32_t base_reg,
                               const uint32_t* values_be, uint32_t n,
                               uint32_t phys, bool from_memory);

// Walk one executed indirect buffer (big-endian PM4, `dwords` long) at
// physical address `buffer_phys`, updating `ctx` and emitting one flags word
// per EXECUTED DRAW_INDX (0x22) into draw_flags, packet order, up to
// max_draws (excess draws still update the context and stats; flags dropped).
// `stats` is zeroed first. Returns the number of flag words written.
//
// ★ PREDICATED TILING. A type-3 header with bit 0 set is predicated: the
// command processor executes it only while `(bin_select & bin_mask) != 0` and
// otherwise skips the whole packet (ExecutePacketType3). Naruto renders 720p
// as three EDRAM tiles and emits one predicated block per tile -- same
// buffer, replayed once per bin, with that bin's PA_SC_WINDOW_OFFSET /
// WINDOW_SCISSOR_TL/BR and RB_COPY_DEST_BASE. A walk that ignores the
// predicate applies all three blocks and ends on the LAST tile's values,
// which is exactly the increment-4a divergence. Pass the bin state in effect
// at buffer entry (the consumer's own bin_select_/bin_mask_, which the ring
// sets before executing the buffer); the walk tracks SET_BIN_* packets found
// inside the buffer itself from there.
uint32_t WalkBufferContext(const uint8_t* raw, uint32_t dwords,
                           uint32_t buffer_phys, StateContext* ctx,
                           uint16_t* draw_flags, uint32_t max_draws,
                           CtxWalkStats* stats, CtxMemReadFn mem_read,
                           void* mem_user, CtxShaderFn shader_fn = nullptr,
                           void* shader_user = nullptr,
                           CtxWatchFn watch_fn = nullptr,
                           void* watch_user = nullptr,
                           uint64_t bin_select = 0xFFFFFFFFull,
                           uint64_t bin_mask = 0xFFFFFFFFull,
                           CtxRegWriteFn reg_fn = nullptr,
                           void* reg_user = nullptr,
                           CtxDrawFn draw_fn = nullptr,
                           void* draw_user = nullptr);

// ---- Resumable walking: the per-draw moment ---------------------------------
// [NR-DRAW] Increment 4c.
//
// WalkBufferContext consumes a whole buffer in one call. That is what a census
// wants and what a REPLAY cannot use: the consumer walks at buffer entry and
// the draws execute afterwards, so at every draw the mirror holds the
// buffer's END state -- including writes from packets AFTER that draw. Any
// per-draw VALUE question asked from outside the walk therefore reads a
// lookahead across the whole buffer (~151 draws in the city).
//
// It does not affect the results that were read at buffer ENTRY: the 4a/4b-1
// `diverge` check and 4b-3's `ResCompareLive` both run before the walk, where
// no lookahead exists. It does affect 4b-3's per-draw COVERAGE check, which
// runs at execution time: `undef` survives (definedness is sticky and
// cross-buffer, so a later write in the same buffer can only matter for a
// slot's first write ever) but `wrongtype` does not, because it reads the
// fetch slot's CURRENT type bits, which by then are whatever the buffer's
// LAST binding wrote.
//
// So the same decoder is now resumable, and the consumer advances it in
// LOCKSTEP with execution: one stop per executed draw packet, with the mirror
// holding precisely the state that draw runs under. That is both the honest
// way to ask a per-draw question and the shape the native replay needs, since
// a replay walks and issues in one order.

// Where a CtxWalkNextDraw / CtxWalkNextStop stop landed.
struct CtxDrawStop {
  uint32_t opcode;  // 0x22 DRAW_INDX or 0x36 DRAW_INDX_2 (draw stops); the
                    // packet's type-3 opcode for a delegate stop
  uint32_t dword;   // the header's dword index within the buffer
  uint16_t flags;   // the draw's flags word (0x22 only; 0 for 0x36/delegate)
  uint32_t index;   // ordinal among EXECUTED 0x22 draws, 0-based (0x22 only),
                    // so it is index-aligned with the join list exactly as the
                    // whole-buffer flags array is
  uint8_t delegate; // [NR-SKP] 1 = a type-3 packet the walk does not decode
                    // (CtxWalkNextStop only). The cursor is left AT the
                    // packet header so the caller can hand it to the
                    // executor's own dispatch, then CtxWalkSkipDelegated.
};

// ---- [NR-WM] 5-4-8: the walk memo ------------------------------------------
// A parsed walk records its emission stream; a later execution of the SAME
// buffer bytes under the SAME entry bin state replays that stream through the
// same callbacks, skipping the PM4 parse. Soundness: the decode is a pure
// function of (buffer bytes, entry bin state); byte identity is the caller's
// gate (the ruse shadow compare runs anyway under the skip), bin identity is
// the store key (predicated tiles resolve per entry state, so one buffer gets
// one stream per bin regime it executes under). Recording free-rides on a
// walk that happens regardless -- the record-once/replay-many economics that
// the 5-4-7-2 span swap lacked hold here by construction.
//
// Op inventory (deliberately small; anything per-dword or stateful re-parses
// its ONE packet through CtxWalkStep so semantics can never drift):
//   kCtxMemoRange    consumed inline bulk range: range_fn(base, raw+a*4, n)
//   kCtxMemoRangeMem consumed by-ref bulk range: range_fn(base, null, n, a)
//                    (a = phys addr; values re-read at replay, per
//                    [[bindings-inline-constants-byref]])
//   kCtxMemoPkt      re-parse the single packet at header dword a (per-dword
//                    type-0/1, truncated/one-reg shapes, SET_CONSTANT
//                    fallbacks, IM_LOAD*, SET_BIN*)
//   kCtxMemoDraw     draw stop: re-parse at a (flags/draw_fn/payload/stop
//                    exactly as the parsed walk)
//   kCtxMemoDeleg    delegate stop at a (cursor positioned for the caller's
//                    ExecutePacket dispatch)
//   kCtxMemoScan     [NR-TMPL] N-2 rung 1: LIVE-SCAN WINDOW -- parse the
//                    dwords [a, a+n) as they stand at replay, packet by
//                    packet, instead of assuming the record-time decode.
//                    Recorded over every run of top-level no-op dwords,
//                    because the recorder FINALIZES buffers in place: at kick
//                    time placeholder nops become SET_BIN / WAIT_REG_MEM /
//                    EVENT_WRITE / bin-window packets (and real packets are
//                    nop'd back out), pointer-neutrally, so no record-side
//                    hook ever sees it. Those sites are known by POSITION at
//                    record time and only by VALUE at execute time: the
//                    template stores the window, never the bytes.
// `b` always holds the packet header dword, so any declined range replays by
// re-parsing its own packet (counted, never silent).

enum CtxMemoKind : uint8_t {
  kCtxMemoRange = 1,
  kCtxMemoRangeMem,
  kCtxMemoPkt,
  kCtxMemoDraw,
  kCtxMemoDeleg,
  kCtxMemoScan,
};

struct CtxMemoOp {
  uint8_t kind;
  uint8_t pad;
  uint16_t reg;  // range base register (range kinds); 0 otherwise
  uint32_t n;    // range dword count (range kinds); 0 otherwise
  uint32_t a;    // payload dword index / phys addr / header dword index
  uint32_t b;    // header dword index (fallback re-parse anchor)
};

struct CtxMemoStream {
  uint64_t select;  // entry bin state this stream was recorded under
  uint64_t mask;
  uint32_t dwords;  // buffer length the stream was recorded from
  std::vector<CtxMemoOp> ops;
};

// ---- [NR-PLAN] N-2-2: the FLAT APPLY PLAN -----------------------------------
// The memo stream (above) is a list of "re-parse this packet" instructions:
// CtxMemoNext skips the PM4 parse for exactly the two range kinds and calls
// CtxWalkStep for everything else, so a memo replay still re-parses ~2/3 of
// city packets (naruto_483 op mix). That is why 5-4-8 measured fps NEUTRAL,
// and no amount of coverage changes the mechanism.
//
// The plan is the answer: one op per PACKET, compiled ONCE at record time on
// the guest producer thread (70% idle at city per N-0), carrying everything
// the walk DERIVES from the packet -- kind, base register, run length, packet
// length, mirrored slot, predicate bit -- so replay does ZERO header decoding
// and zero type dispatch for the common kinds. What it deliberately does NOT
// carry is VALUES: every value (constants, draw payloads, by-ref addresses,
// shader addr/size, bin masks) is read from the LIVE buffer at replay, which
// is what makes the plan survive the recorder's in-place finalize exactly as
// the N-2-1 decode plan does.
//
// Soundness rests on one claim per op -- "the packet at `hdr` still has the
// header it had at compile" -- and that claim is CHECKED, not assumed: the
// compiler emits a guard {position, expected dword} for the header and for
// every payload dword a descriptor is derived from (SET_CONSTANT's typed
// offset, SET_CONSTANT2's raw base, LOAD_ALU_CONSTANT's type and size). A
// failing guard demotes that ONE op to kPlanPkt (a live re-parse), exactly as
// N-2-1's DriftPrePass demotes a drifted range. The values a guard does not
// cover are precisely the ones read live. See the [[GENERAL LESSON]] in
// CLAUDE.md: anything the walk derives from a packet is re-read or checked.
//
// Placeholder runs stay live-scan windows (kPlanScan), same as the memo.

enum CtxPlanKind : uint8_t {
  kPlanSkip = 1,   // native packet the walk decodes to nothing (NOP, 0x3B,
                   // a constant writer whose typed base is unknown)
  kPlanRange,      // register run offerable in bulk (touches no mirror slot)
  kPlanRegs,       // register run applied per dword (touches the mirror)
  kPlanRegRep,     // type-0 one_reg: n stores to the SAME register
  kPlanReg,        // one register write (the two halves of a type-1 packet)
  kPlanRangeMem,   // by-ref range: LOAD_ALU_CONSTANT, address read live
  kPlanShader,     // IM_LOAD (0x27) / IM_LOAD_IMMEDIATE (0x2B)
  kPlanBin,        // SET_BIN_MASK/SELECT family
  kPlanDraw,       // draw stop (0x22 / 0x36)
  kPlanDeleg,      // delegate stop (the packet the executor must run)
  kPlanScan,       // live-scan window over a placeholder run
  kPlanPkt,        // re-parse this one packet through CtxWalkStep
};

constexpr uint8_t kPlanFlagPred = 1;       // type-3 predicate bit was set
constexpr uint8_t kPlanFlagSetConst2 = 2;  // 0x55/0x56: count set_const2
constexpr uint8_t kPlanFlagImm = 4;        // kPlanShader: IM_LOAD_IMMEDIATE

// 16 bytes. `hdr` is the packet header's dword index relative to the SPAN the
// plan was compiled from; the replay adds CtxWalker::plan_base to reach the
// executing buffer's coordinates (the memo's positions were absolute, which
// is why a span-based replay needed a base bias here).
struct CtxPlanOp {
  uint8_t kind;
  int8_t slot;    // kPlanReg: mirrored slot, or -1
  uint8_t poff;   // payload start, in dwords from `hdr`
  uint8_t flags;
  uint16_t reg;   // base register, or the type-3 opcode for stops/bin/shader
  uint16_t n;     // run dwords (register kinds) / window dwords (kPlanScan)
  uint16_t len;   // packet dwords: the cursor advance
  uint16_t pad;
  uint32_t hdr;
};

struct CtxPlanGuard {
  uint32_t pos;  // dword index, span-relative
  uint32_t val;  // the dword as it stood when the plan was compiled
  uint32_t op;   // plan op to demote if it no longer matches
};

struct CtxPlanCompileStats {
  uint32_t ops, guards, scans, ranges, regs, draws, delegs, pkts, skips;
};

// Compile one span's bytes into a flat plan. Frames the span with the
// walker's own rules and fails (returns 0) unless the framing lands exactly
// on `ndwords` -- the same exactness [[pm4-span-parse-conventions]] demands
// of the feed. `*nguard` receives the guard count. Pure function of the
// bytes: safe on the producer thread, no allocation, no globals.
uint32_t CtxPlanCompile(const uint8_t* raw, uint32_t ndwords, CtxPlanOp* out,
                        uint32_t out_cap, CtxPlanGuard* guards,
                        uint32_t guard_cap, uint32_t* nguard,
                        CtxPlanCompileStats* st);

// Check every guard against the live span and demote each op whose structure
// moved to kPlanPkt. Returns the number demoted (0 = the plan is exact for
// these bytes). `ops` must be the caller's own copy: this rewrites it.
uint32_t CtxPlanApplyGuards(const uint8_t* live, uint32_t live_dwords,
                            CtxPlanOp* ops, uint32_t nops,
                            const CtxPlanGuard* guards, uint32_t nguard);

// [N9] Non-mutating guard check: 0 if every guard matches the live bytes,
// else 1 + the index of the first mismatch. The whole-buffer program replay
// falls back to a fresh compile on ANY mismatch instead of demoting per op,
// so it never needs the caller-copy ApplyGuards requires -- the shared plan
// is replayed in place, zero copies on the hot path.
uint32_t CtxPlanCheckGuards(const uint8_t* live, uint32_t live_dwords,
                            const CtxPlanGuard* guards, uint32_t nguard);

// [N9] Whole-buffer plan store. One plan per (buffer address, dword count),
// compiled from the buffer's own bytes at its FIRST skip-path execution and
// replayed on every later one. This is the same CtxPlanOp interpreter the
// 5-4-8/N-2-2 template swap proved decode-exact at city scale; what changed
// is the store shape, aimed at that rung's three measured killers:
//   - keyed per BUFFER (2-3k lookups/s), not per span (the template store
//     paid a ~138 ns probe+copy per DRAW, more than the parse it replaced);
//   - validity = the STRUCTURAL guard set, not byte identity (the memo's
//     byte gate served only ~17% of city executions because the recorder
//     patches VALUES in place; guards ignore values -- they are read live);
//   - CP-thread-owned, so no arena, no epochs, no coherency copies.
// A guard mismatch recompiles from the live bytes and replays the fresh plan
// in the same execution; framing failures and compile thrash mark the buffer
// refused (counted, self-healing retry every 64th execution).
struct CtxProgStats {
  uint64_t serves;         // buffer executions replayed from a plan
  uint64_t compiles;       // plans compiled (first sight or guard-fail)
  uint64_t gfails;         // executions whose guard check failed
  uint64_t refused_execs;  // executions of a refused buffer (live walk)
  uint64_t refuse_marks;   // buffers marked refused (framing / thrash)
  uint64_t retries;        // refused buffers re-tried (self-heal probe)
  uint64_t evicts;         // whole-store clears at the byte cap
  uint64_t guards_checked; // total guard compares run
  size_t bytes;            // ops + guards held (capacity)
  uint32_t bufs;           // entries in the store
};
// Attach a stored (or freshly compiled) plan covering [0, count) to the
// walker. Returns true when attached (caller counts a replay); false =
// refused or compile-failed, walk live (caller counts a fallback). CP thread
// only, like every other walker call.
struct CtxWalker;
bool CtxProgAttach(CtxWalker* w, uint32_t bufkey, uint32_t count,
                   const uint8_t* raw);
void CtxProgClear();
CtxProgStats* CtxProgStatsPtr();

struct CtxMemoStats {
  uint64_t commits;    // streams committed
  uint64_t replaced;   // verify-mode re-records over a mismatching stream
  uint64_t fallbacks;  // replayed ranges the consumer declined (re-parsed)
  uint64_t evicts;     // whole-store clears at the byte cap
  uint64_t invals;     // buffers dropped because their bytes changed
  uint64_t refused;    // buffers marked never-memo (nested indirect buffer)
  size_t bytes;        // op storage held
  uint32_t bufs;       // buffers with at least one stream
  uint32_t streams;    // total streams held
};

// Decoder state. Zero-initialize through CtxWalkBegin, never by hand.
struct CtxWalker {
  const uint8_t* raw;
  uint32_t dwords;
  uint32_t buffer_phys;
  StateContext* ctx;
  uint16_t* draw_flags;
  uint32_t max_draws;
  CtxWalkStats* stats;
  CtxMemReadFn mem_read;
  void* mem_user;
  CtxShaderFn shader_fn;
  void* shader_user;
  CtxWatchFn watch_fn;
  void* watch_user;
  CtxRegWriteFn reg_fn;
  void* reg_user;
  CtxDrawFn draw_fn;
  void* draw_user;
  // [NR-SKP] 5-4-3: optional bulk range consumer. Zeroed by CtxWalkBegin;
  // the caller sets both fields afterwards when it wants ranges as ranges.
  CtxRegRangeFn range_fn;
  void* range_user;

  CtxBinState bin;
  uint32_t cursor;   // next dword to decode
  uint32_t nflags;   // flags words written so far
  uint32_t cur_dw, cur_hdr, cur_arg;  // packet being decoded, for watch_fn

  // [NR-WM] 5-4-8 attachments. Zeroed by CtxWalkBegin; the caller attaches
  // ONE of them afterwards (record via CtxMemoRecordBegin, replay via
  // CtxMemoReplayBegin). Never both.
  std::vector<CtxMemoOp>* rec;  // parse emissions append here
  const CtxMemoOp* rep;         // replay stream (owned by the memo store)
  uint32_t rep_n, rep_i;
  // [NR-TMPL] The op stream may describe a WINDOW (one record-time span) of a
  // longer buffer: `dwords` then bounds the DECODE (a packet whose payload
  // runs past the window must still read its real bytes -- a SET_BIN read as
  // mask 0 because the payload fell outside would diverge the bin state for
  // everything after it), while rep_end bounds the STREAM. Zero = the stream
  // covers the whole buffer, which is what the 5-4-8 memo always does.
  uint32_t rep_end;
  uint8_t rep_stops;  // [NR-TMPL] the replay's catch-up parse surfaces
                      // delegate stops iff the caller asked for them; set by
                      // CtxWalkNextStop / CtxWalkNextDraw, never by hand.

  // [NR-PLAN] N-2-2. Attached by CtxPlanBegin, never by hand; exclusive with
  // `rep`. Positions in the plan are SPAN-relative, so plan_base carries the
  // span's dword offset inside the executing buffer (0 when the walker was
  // begun at the span itself, as the N-2 compare gate does). plan_end bounds
  // the STREAM exactly as rep_end does; the DECODE stays bounded by `dwords`,
  // because a packet finalized longer at the span edge must still read its
  // real bytes.
  const CtxPlanOp* plan;
  uint32_t plan_n, plan_i, plan_base, plan_end;
};

// Same arguments as WalkBufferContext, which is now a wrapper over these three.
// `stats` is zeroed and the context's in-buffer marks are cleared here.
void CtxWalkBegin(CtxWalker* w, const uint8_t* raw, uint32_t dwords,
                  uint32_t buffer_phys, StateContext* ctx, uint16_t* draw_flags,
                  uint32_t max_draws, CtxWalkStats* stats, CtxMemReadFn mem_read,
                  void* mem_user, CtxShaderFn shader_fn = nullptr,
                  void* shader_user = nullptr, CtxWatchFn watch_fn = nullptr,
                  void* watch_user = nullptr,
                  uint64_t bin_select = 0xFFFFFFFFull,
                  uint64_t bin_mask = 0xFFFFFFFFull, CtxRegWriteFn reg_fn = nullptr,
                  void* reg_user = nullptr, CtxDrawFn draw_fn = nullptr,
                  void* draw_user = nullptr);

// Applies every packet up to and including the next EXECUTED draw packet, then
// returns true with `stop` describing it. False at the end of the buffer, with
// all remaining state applied. Stops at BOTH draw opcodes: 0x36 executes and
// consumes state exactly as 0x22 does (increment 4a found the resolves live
// there), so stopping only at 0x22 would leave the walk behind the executor.
// Only 0x22 emits a flags word and fires draw_fn, so every existing consumer
// sees what it saw before.
bool CtxWalkNextDraw(CtxWalker* w, CtxDrawStop* stop);

// [NR-SKP] Phase 5-4-2: like CtxWalkNextDraw, but ALSO stops at every
// executed type-3 packet the walk does not natively decode (the 5-4-1
// delegate list: WAIT_REG_MEM, the EVENT_WRITE family, REG_RMW, MEM_WRITE,
// COND_WRITE, INTERRUPT, XE_SWAP, nested INDIRECT_BUFFER, and anything
// unknown -- the set is defined by inversion, so a new opcode can never be
// silently mis-walked). For a delegate stop (stop->delegate == 1) the cursor
// is left AT the packet header: the caller runs the executor's own dispatch
// on that packet, then calls CtxWalkSkipDelegated to resume the walk past
// it. Predicated-out packets of every kind are still skipped natively, per
// the executor's own rule. Draw stops behave exactly as in CtxWalkNextDraw
// (payload applied, flags word emitted, cursor past the packet).
bool CtxWalkNextStop(CtxWalker* w, CtxDrawStop* stop);

// [NR-PLAN] Decode exactly ONE packet at the cursor, surfacing draw and
// delegate stops exactly as CtxWalkNextStop does. The consuming swap needs
// this: it must re-offer the plan store at EVERY packet boundary, so it
// cannot let the walker run free to the end of the buffer the way every
// earlier consumer could. False = the packet applied and nothing surfaced.
bool CtxWalkStepOne(CtxWalker* w, CtxDrawStop* stop);

// Advances the cursor past the delegated packet CtxWalkNextStop stopped at.
// The caller may have let the executor change the bin members meanwhile
// (a nested indirect buffer can contain SET_BIN packets); it is the caller's
// job to write those back into w->bin before the next CtxWalkNextStop.
void CtxWalkSkipDelegated(CtxWalker* w);

// Applies the rest of the buffer. Returns the total flags words written.
uint32_t CtxWalkFinish(CtxWalker* w);

// ---- [NR-WM] memo store + record/replay attachment -------------------------
// All single-threaded on the CP thread, like the walker itself.

// The stream recorded for {ptr, dwords, entry bin}, or nullptr. Valid only
// while the caller has independently proven the buffer bytes identical to the
// execution the stream was recorded from (the ruse shadow-compare chain).
const CtxMemoStream* CtxMemoFind(uint32_t ptr, uint32_t dwords,
                                 uint64_t select, uint64_t mask);
// True when this buffer is marked never-memo (a nested indirect buffer was
// seen during a record: its content is outside the byte-identity gate and can
// steer bin state, so no stream from it can be trusted).
bool CtxMemoRefused(uint32_t ptr);
void CtxMemoRefuse(uint32_t ptr);
// Attach the recording scratch to the walker (one recording at a time).
void CtxMemoRecordBegin(CtxWalker* w);
// Detach and discard the scratch.
void CtxMemoRecordAbandon(CtxWalker* w);
// Detach and store the scratch under {ptr, dwords, select, mask}, replacing
// any prior stream for that key. False when the byte cap forced a whole-store
// clear first (the stream is still stored after the clear).
bool CtxMemoRecordCommit(CtxWalker* w, uint32_t ptr, uint32_t dwords,
                         uint64_t select, uint64_t mask);
// Verify mode: does the just-recorded scratch equal the stored stream?
// Sets *first_ne to the first differing op index (or the shorter length).
bool CtxMemoRecordMatches(const CtxWalker* w, const CtxMemoStream* s,
                          uint32_t* first_ne);
// Attach a stream for replay. CtxWalkNextStop / CtxWalkFinish then drive from
// it instead of parsing.
void CtxMemoReplayBegin(CtxWalker* w, const CtxMemoStream* s);
void CtxMemoReplayEnd(CtxWalker* w);
// [NR-PLAN] Attach a compiled plan covering [base_dw, base_dw + end_dw) of
// the walker's buffer. CtxWalkNextStop / CtxWalkFinish then drive from it.
void CtxPlanBegin(CtxWalker* w, const CtxPlanOp* ops, uint32_t nops,
                  uint32_t base_dw, uint32_t end_dw);
void CtxPlanEnd(CtxWalker* w);
// Drop every stream for this buffer (its bytes changed). Returns how many.
uint32_t CtxMemoInvalidate(uint32_t ptr);
void CtxMemoClear();
CtxMemoStats* CtxMemoStatsPtr();

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

// [NR-4B-EXT] Does [base, base+n) intersect the 27-reg recovery mirror? The
// walker never offers such a range in bulk (CtxRangeOfferable), so any
// consumer predicate that models "which ranges arrive as ranges" must apply
// the same test. Exported so the defer/compose filter in the command
// processor and the walker can never drift apart.
bool CtxRangeTouchesMirrorRegs(uint32_t base, uint32_t n);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_CONTEXT_H_
