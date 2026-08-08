/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_REGFILE_H_
#define REX_GRAPHICS_NR_REGFILE_H_

#include <cstdint>

// [NR-DRAW] The SHADOW REGISTER FILE: native-renderer build-out increment 4c.
//
// Increments 4a through 4b-3 each mirrored a HAND-PICKED slice of GPU state
// and gated it: 27 recovery registers, then the four constant files, then the
// shaders. Every one passed (diverge 0 at full city load). But a draw does not
// read a hand-picked slice, it reads whatever the pipeline reads -- blend,
// depth, stencil, colour mask, cull, the SQ program registers, the VGT setup.
// None of that has been mirrored, so the honest state of the input model is
// "the parts we chose to look at are recoverable".
//
// This unit removes the choosing. It mirrors the WHOLE register file, from the
// same walk, and asks the same question over all of it. That is worth doing
// for two reasons beyond completeness:
//
//   1. It is the last input question. A draw issued from a complete shadow
//      that equals the live file is identical to the emulated draw BY
//      CONSTRUCTION, whatever it happens to read. No enumeration of "which
//      registers matter" is needed, and getting that enumeration wrong is
//      exactly the kind of silent hole the earlier increments kept finding.
//
//   2. Its failures are the port scope. A register the live file has and the
//      stream never wrote is an input the native replay cannot obtain by
//      walking, so it must come from the D3D9 hook layer instead. The address
//      map says blend/depth/stencil/sampler state IS set through hookable
//      device methods, so a NAMED list of such registers is directly the list
//      of hooks that must exist. That list has never been measured; it has
//      only been reasoned about.
//
// Two findings are therefore counted apart, because they mean opposite things:
//
//   DIVERGE  the shadow wrote this register and holds a DIFFERENT value than
//            the live file. The walk decoded something wrong. Must be 0; this
//            is the same gate 4a/4b-1/4b-3 passed, now over everything.
//   EXTERN   the shadow never wrote this register and the live file holds a
//            nonzero value. The stream does not carry it. Not a bug -- a
//            boundary. Every one of these has to be named.
//
// The comparison runs AT EACH DRAW, against a walk advanced in lockstep with
// execution (CtxWalkNextDraw), so both sides are read at the same moment. It
// sweeps a slice of the file per draw rather than all of it: at ~200k draws a
// second a full 20,483-register compare per draw would cost more than the
// thing being measured, and a rolling sweep covers the whole file thousands of
// times a second anyway. A divergence persists until the register is
// rewritten, so a sweep catches it within a fraction of a millisecond.
//
// Pure functions over a caller-owned struct: no globals, no threads, no SDK
// dependencies, so tools/nr-regfile-test.cpp builds this file bare.

namespace rex {
namespace graphics {
namespace nr {

// Must equal RegisterFile::kRegisterCount. The consumer static_asserts it;
// this header stays free of SDK includes so the unit test can build bare.
// Sized to match so the shadow can be memcpy'd straight into a RegisterFile,
// which is how increment 4c issues a draw from it.
constexpr uint32_t kRegShadowCount = 0x5003;

struct RegShadow {
  uint32_t values[kRegShadowCount];
  uint8_t defined[kRegShadowCount];
  uint32_t sweep;  // rolling compare cursor
};

struct RegShadowStats {
  uint64_t writes;          // decoded writes accepted into the shadow
  uint64_t writes_ignored;  // decoded writes naming a register out of range
  uint64_t defined_count;   // distinct registers the stream has ever written
  uint64_t checks;          // registers compared
  uint64_t diverge;         // ... defined, and holding a different value
  uint64_t externs;         // ... never written, and live is nonzero
  uint64_t sweeps;          // completed passes over the whole file
  uint64_t compares;        // draws at which a sweep ran
};

// A write the walk decoded. Registers past the end of the file are counted and
// dropped rather than clamped: a clamp would corrupt a real register with a
// bogus one's value and the compare would then blame the walk for it.
void RegShadowApplyWrite(RegShadow* s, RegShadowStats* stats, uint32_t reg,
                         uint32_t value);

enum RegShadowFinding : uint8_t {
  kRegShadowDiverge,
  kRegShadowExtern,
};

// Reported per finding, in sweep order. `ours` is meaningless for an extern
// (the shadow never wrote it) and is passed as 0.
using RegShadowFindFn = void (*)(void* user, RegShadowFinding what,
                                 uint32_t reg, uint32_t ours, uint32_t live);

// Compares `count` registers from the rolling cursor and advances it, wrapping
// at the end of the file (each wrap counts a sweep). `find` may be null.
//
// `live` is the register file's own value array, read directly rather than
// through a callback. That is not a micro-optimization: this runs on the
// command-processor thread, which Ch.9 measured as the frame's bottleneck, and
// an indirect call per register would put ~50M of them a second on it. A probe
// that changes the frame it is measuring produces numbers about itself.
void RegShadowSweep(RegShadow* s, RegShadowStats* stats, uint32_t count,
                    const uint32_t* live, RegShadowFindFn find, void* find_user);

// Compares one register on demand, without touching the cursor: for a caller
// that wants a specific register checked at a specific moment rather than
// whenever the sweep reaches it. Tallies into `stats` exactly as a swept
// register would.
void RegShadowCheckOne(const RegShadow* s, RegShadowStats* stats, uint32_t reg,
                       const uint32_t* live, RegShadowFindFn find,
                       void* find_user);

// [NR-ISSUE] Increment 4d: the register file a draw is ISSUED from.
//
// out[reg] = the shadow's value where the stream has ever written the
// register, the live file's value everywhere else. The defined side is the
// recovered state under test (the walk's own decoded values, ~2,466
// registers); the undefined side is by definition what the stream cannot
// provide -- the four named externs (SQ_GPR_MANAGEMENT et al.), the two
// side-effect ports, and dead registers -- which a native renderer obtains
// once at init or from hooks, and which this composition takes from live
// exactly as "read it once" prescribes. Composing rather than memcpy'ing the
// shadow alone is what keeps the first issued draw from failing on a register
// no buffer ever carries.
void RegShadowCompose(const RegShadow* s, const uint32_t* live, uint32_t* out);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_REGFILE_H_
