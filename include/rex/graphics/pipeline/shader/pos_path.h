// [occ-pos] The vertex shader's position path, recovered from the ucode once
// at analysis: clip = M * (fetched position, 1) where every entry of M is a
// small polynomial in the shader's float constants. Drive 798 read 66% of the
// city's primitives passing zero samples at draw time; culling them needs each
// draw's object-space bounds carried to clip space by the SAME arithmetic the
// shader uses, whatever form the Xenon compiler emitted (dp4 rows, mad
// chains, a packed-position scale and offset before the matrix). A shader
// whose position is not an affine function of one vertex fetch (skinning,
// per-vertex branches, nonlinear math) is simply not eligible.
#pragma once

#include <cstdint>
#include <memory>

#include <rex/graphics/xenos.h>

namespace rex::graphics {

struct ParsedVertexFetchInstruction;
struct ParsedAluInstruction;

struct PosPath {
  // One float constant component: c[reg].comp, vertex shader base.
  struct Sym {
    uint8_t reg, comp;
  };
  // sign * product of up to 3 symbols (n == 0 is the literal 1).
  struct Mono {
    int8_t sign;
    uint8_t n;
    Sym s[3];
  };
  // Sum of up to 8 monomials (n == 0 is zero).
  struct Poly {
    uint8_t n;
    Mono m[8];
  };
  static constexpr uint32_t kBasisOne = 4;
  enum Reason : uint8_t {
    kEligible = 0,
    kNotVertex,      // not a vertex shader / not analyzed
    kPosNotWritten,  // some clip component never written
    kNonAffine,      // a clip component is not affine in one fetch (skinning, nonlinear)
    kMultiFetch,     // clip depends on more than one vertex fetch
    kNoFetch,        // clip is a constant (no vertex dependence)
    kControlFlow,    // loops / calls / jumps in the shader
    kIndexReg,       // the position fetch is not indexed by an untouched r0.x
    kReasonCount
  };

  bool eligible = false;
  Reason reason = kNotVertex;
  // clip[i] = sum_j M[i][j] * p_j, p = (fetched x, y, z, w, 1).
  Poly M[4][5] = {};
  uint8_t used_basis_mask = 0;  // bit j set = some M[i][j] nonzero, j < 4
  // The vertex fetch the position comes from.
  uint32_t fetch_constant = 0;  // vertex fetch constant slot [0-95]
  uint32_t index_register = 0;  // the fetch's index operand temp register
  xenos::VertexFormat format = xenos::VertexFormat::kUndefined;
  int32_t offset_dwords = 0;
  uint32_t stride_dwords = 0;
  int32_t exp_adjust = 0;
  bool is_signed = false;
  bool is_integer = false;
  bool is_index_rounded = false;
  xenos::SignedRepeatingFractionMode signed_rf_mode =
      xenos::SignedRepeatingFractionMode::kZeroClampMinusOne;

  // m[i][j] from the vertex shader's float constants (256 float4).
  void Evaluate(const float* consts, float m[4][5]) const;
  // A compact human-readable form of M, for the per-shader log line.
  void Describe(char* out, size_t out_size) const;
};

// Fed by Shader::AnalyzeUcode in program order; Finish() writes the PosPath.
class PosPathTracker {
 public:
  PosPathTracker();
  ~PosPathTracker();
  void OnVertexFetch(const ParsedVertexFetchInstruction& instr, bool conditional);
  void OnAlu(const ParsedAluInstruction& instr, bool conditional);
  void MarkControlFlowUnsafe();
  void Finish(PosPath& out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rex::graphics
