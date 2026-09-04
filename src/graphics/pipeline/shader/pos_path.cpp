// [occ-pos] see pos_path.h. A symbolic pass over the vertex shader's ALU
// stream: every temp register component holds a Form = sum over basis
// elements (a vertex fetch's x/y/z/w, or the literal 1) of a polynomial in
// float constants. Affine ops (add, mul/mad by a constant expression, dp4 /
// dp3 / dp2add against constants, max(a,a) = mov) carry the forms through;
// anything else marks the written components unknown. At the end the four
// position exports must be forms over ONE fetch plus the literal.
#include <rex/graphics/pipeline/shader/pos_path.h>

#include <cstdio>
#include <cstring>

#include <rex/graphics/pipeline/shader/shader.h>

namespace rex::graphics {

namespace {

using Poly = PosPath::Poly;
using Mono = PosPath::Mono;

constexpr uint32_t kMaxTerms = 5;
constexpr uint32_t kMaxFetches = 32;

struct Term {
  uint8_t basis;  // 0..3 = the fetch's component, PosPath::kBasisOne
  uint8_t fetch;  // fetch id (unused for the literal basis)
  Poly p;
};

struct Form {
  bool unknown = true;
  uint8_t n = 0;
  Term t[kMaxTerms];
};

Form Unknown() { return Form{}; }

Form Zero() {
  Form f;
  f.unknown = false;
  f.n = 0;
  return f;
}

Poly PolyOne() {
  Poly p{};
  p.n = 1;
  p.m[0].sign = 1;
  p.m[0].n = 0;
  return p;
}

Poly PolySym(uint32_t reg, uint32_t comp) {
  Poly p{};
  p.n = 1;
  p.m[0].sign = 1;
  p.m[0].n = 1;
  p.m[0].s[0].reg = uint8_t(reg);
  p.m[0].s[0].comp = uint8_t(comp);
  return p;
}

Form Literal(uint32_t basis, uint32_t fetch, const Poly& p) {
  Form f;
  f.unknown = false;
  f.n = 1;
  f.t[0].basis = uint8_t(basis);
  f.t[0].fetch = uint8_t(fetch);
  f.t[0].p = p;
  return f;
}

Form One() { return Literal(PosPath::kBasisOne, 0, PolyOne()); }

Form Const(uint32_t reg, uint32_t comp) {
  return Literal(PosPath::kBasisOne, 0, PolySym(reg, comp));
}

bool PolyAdd(Poly& a, const Poly& b) {
  if (a.n + b.n > 8) return false;
  for (uint32_t i = 0; i < b.n; ++i) a.m[a.n++] = b.m[i];
  return true;
}

bool PolyMul(const Poly& a, const Poly& b, Poly& out) {
  out = Poly{};
  for (uint32_t i = 0; i < a.n; ++i) {
    for (uint32_t j = 0; j < b.n; ++j) {
      if (a.m[i].n + b.m[j].n > 3 || out.n >= 8) return false;
      Mono& m = out.m[out.n++];
      m.sign = int8_t(a.m[i].sign * b.m[j].sign);
      m.n = 0;
      for (uint32_t k = 0; k < a.m[i].n; ++k) m.s[m.n++] = a.m[i].s[k];
      for (uint32_t k = 0; k < b.m[j].n; ++k) m.s[m.n++] = b.m[j].s[k];
    }
  }
  return true;
}

Form Neg(Form f) {
  if (f.unknown) return f;
  for (uint32_t i = 0; i < f.n; ++i) {
    for (uint32_t j = 0; j < f.t[i].p.n; ++j) f.t[i].p.m[j].sign = int8_t(-f.t[i].p.m[j].sign);
  }
  return f;
}

Form Add(const Form& a, const Form& b) {
  if (a.unknown || b.unknown) return Unknown();
  Form r = a;
  for (uint32_t i = 0; i < b.n; ++i) {
    const Term& bt = b.t[i];
    bool merged = false;
    for (uint32_t j = 0; j < r.n; ++j) {
      Term& rt = r.t[j];
      if (rt.basis == bt.basis && (rt.basis == PosPath::kBasisOne || rt.fetch == bt.fetch)) {
        if (!PolyAdd(rt.p, bt.p)) return Unknown();
        merged = true;
        break;
      }
    }
    if (!merged) {
      if (r.n >= kMaxTerms) return Unknown();
      r.t[r.n++] = bt;
    }
  }
  return r;
}

// True when the form has no vertex dependence (zero or a constant expression).
bool IsConstant(const Form& f) {
  if (f.unknown) return false;
  for (uint32_t i = 0; i < f.n; ++i) {
    if (f.t[i].basis != PosPath::kBasisOne) return false;
  }
  return true;
}

Form Mul(const Form& a, const Form& b) {
  if (a.unknown || b.unknown) return Unknown();
  if (a.n == 0 || b.n == 0) return Zero();
  const Form* k = nullptr;
  const Form* v = nullptr;
  if (IsConstant(a)) {
    k = &a;
    v = &b;
  } else if (IsConstant(b)) {
    k = &b;
    v = &a;
  } else {
    return Unknown();  // quadratic in the vertex
  }
  // k has exactly one term (the literal basis) after Add merged it.
  Form r = *v;
  for (uint32_t i = 0; i < r.n; ++i) {
    Poly p;
    if (!PolyMul(r.t[i].p, k->t[0].p, p)) return Unknown();
    r.t[i].p = p;
  }
  return r;
}

bool SameOperand(const InstructionOperand& a, const InstructionOperand& b) {
  if (a.storage_source != b.storage_source || a.storage_index != b.storage_index ||
      a.storage_addressing_mode != b.storage_addressing_mode || a.is_negated != b.is_negated ||
      a.is_absolute_value != b.is_absolute_value || a.component_count != b.component_count) {
    return false;
  }
  for (uint32_t i = 0; i < a.component_count; ++i) {
    if (a.components[i] != b.components[i]) return false;
  }
  return true;
}

}  // namespace

struct PosPathTracker::Impl {
  Form reg[64][4];
  Form pos[4];
  bool pos_written[4] = {false, false, false, false};
  struct Fetch {
    uint32_t fetch_constant, index_register;
    bool index_ok;  // indexed by r0.x, untouched so far
    ParsedVertexFetchInstruction::Attributes attributes;
  };
  Fetch fetches[kMaxFetches];
  uint32_t nfetch = 0;
  bool have_full = false;
  Fetch last_full{};
  bool cf_unsafe = false;
  bool r0x_written = false;

  Impl() {
    for (auto& r : reg) {
      for (auto& c : r) c = Unknown();
    }
    for (auto& p : pos) p = Unknown();
  }

  Form Operand(const InstructionOperand& op, uint32_t comp) const {
    const SwizzleSource sc = op.GetComponent(comp);
    Form f;
    if (sc == SwizzleSource::k0) {
      f = Zero();
    } else if (sc == SwizzleSource::k1) {
      f = One();
    } else {
      const uint32_t c = uint32_t(sc) - uint32_t(SwizzleSource::kX);
      if (op.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute) {
        return Unknown();
      }
      switch (op.storage_source) {
        case InstructionStorageSource::kRegister:
          if (op.storage_index >= 64) return Unknown();
          f = reg[op.storage_index][c];
          break;
        case InstructionStorageSource::kConstantFloat:
          if (op.storage_index >= 256) return Unknown();
          f = Const(op.storage_index, c);
          break;
        default:
          return Unknown();
      }
    }
    if (op.is_absolute_value) return Unknown();
    if (op.is_negated) f = Neg(f);
    return f;
  }

  void Store(const InstructionResult& res, const Form* value, bool force_unknown) {
    const uint32_t mask = res.GetUsedWriteMask();
    if (!mask) return;
    if (res.storage_target == InstructionStorageTarget::kRegister) {
      if (res.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute) {
        for (auto& r : reg) {
          for (auto& c : r) c = Unknown();
        }
        return;
      }
      if (res.storage_index >= 64) return;
    } else if (res.storage_target != InstructionStorageTarget::kPosition) {
      return;
    }
    for (uint32_t i = 0; i < 4; ++i) {
      if (!(mask & (1u << i))) continue;
      Form f;
      if (force_unknown || value == nullptr || res.is_clamped) {
        f = Unknown();
      } else {
        const SwizzleSource sc = res.components[i];
        if (sc == SwizzleSource::k0) {
          f = Zero();
        } else if (sc == SwizzleSource::k1) {
          f = One();
        } else {
          f = value[uint32_t(sc) - uint32_t(SwizzleSource::kX)];
        }
      }
      if (res.storage_target == InstructionStorageTarget::kRegister) {
        if (res.storage_index == 0 && i == 0) r0x_written = true;
        reg[res.storage_index][i] = f;
      } else {
        pos[i] = f;
        pos_written[i] = true;
      }
    }
  }
};

PosPathTracker::PosPathTracker() : impl_(std::make_unique<Impl>()) {}
PosPathTracker::~PosPathTracker() = default;

void PosPathTracker::MarkControlFlowUnsafe() { impl_->cf_unsafe = true; }

void PosPathTracker::OnVertexFetch(const ParsedVertexFetchInstruction& instr, bool conditional) {
  Impl& im = *impl_;
  Impl::Fetch fe{};
  if (!instr.is_mini_fetch && instr.operand_count >= 2) {
    fe.index_register = instr.operands[0].storage_source == InstructionStorageSource::kRegister
                            ? instr.operands[0].storage_index
                            : UINT32_MAX;
    fe.fetch_constant = instr.operands[1].storage_index;
    const InstructionOperand& io = instr.operands[0];
    fe.index_ok = io.storage_source == InstructionStorageSource::kRegister &&
                  io.storage_index == 0 &&
                  io.storage_addressing_mode == InstructionStorageAddressingMode::kAbsolute &&
                  io.GetComponent(0) == SwizzleSource::kX && !io.is_negated &&
                  !io.is_absolute_value && !im.r0x_written;
    fe.attributes = instr.attributes;
    im.last_full = fe;
    im.have_full = true;
  } else {
    fe = im.last_full;
    fe.attributes = instr.attributes;
    if (!im.have_full) {
      im.Store(instr.result, nullptr, true);
      return;
    }
  }
  if (im.nfetch >= kMaxFetches) {
    im.Store(instr.result, nullptr, true);
    return;
  }
  const uint32_t id = im.nfetch++;
  im.fetches[id] = fe;
  Form v[4];
  for (uint32_t c = 0; c < 4; ++c) v[c] = Literal(c, id, PolyOne());
  im.Store(instr.result, v, conditional || instr.is_predicated);
}

void PosPathTracker::OnAlu(const ParsedAluInstruction& instr, bool conditional) {
  Impl& im = *impl_;
  const bool force = conditional || instr.is_predicated;
  // The vector op.
  if (!instr.IsVectorOpDefaultNop() && instr.vector_and_constant_result.GetUsedWriteMask()) {
    Form A[4], B[4], C[4], res[4];
    const uint32_t nop = instr.vector_operand_count;
    for (uint32_t c = 0; c < 4; ++c) {
      A[c] = nop > 0 ? im.Operand(instr.vector_operands[0], c) : Unknown();
      B[c] = nop > 1 ? im.Operand(instr.vector_operands[1], c) : Unknown();
      C[c] = nop > 2 ? im.Operand(instr.vector_operands[2], c) : Unknown();
      res[c] = Unknown();
    }
    switch (instr.vector_opcode) {
      case ucode::AluVectorOpcode::kAdd:
        for (uint32_t c = 0; c < 4; ++c) res[c] = Add(A[c], B[c]);
        break;
      case ucode::AluVectorOpcode::kMul:
        for (uint32_t c = 0; c < 4; ++c) res[c] = Mul(A[c], B[c]);
        break;
      case ucode::AluVectorOpcode::kMad:
        for (uint32_t c = 0; c < 4; ++c) res[c] = Add(Mul(A[c], B[c]), C[c]);
        break;
      case ucode::AluVectorOpcode::kMax:
        if (nop >= 2 && SameOperand(instr.vector_operands[0], instr.vector_operands[1])) {
          for (uint32_t c = 0; c < 4; ++c) res[c] = A[c];
        }
        break;
      case ucode::AluVectorOpcode::kDp4:
      case ucode::AluVectorOpcode::kDp3: {
        const uint32_t n = instr.vector_opcode == ucode::AluVectorOpcode::kDp4 ? 4 : 3;
        Form s = Zero();
        for (uint32_t m = 0; m < n; ++m) s = Add(s, Mul(A[m], B[m]));
        for (uint32_t c = 0; c < 4; ++c) res[c] = s;
      } break;
      case ucode::AluVectorOpcode::kDp2Add: {
        Form s = Add(Add(Mul(A[0], B[0]), Mul(A[1], B[1])), C[0]);
        for (uint32_t c = 0; c < 4; ++c) res[c] = s;
      } break;
      default:
        break;
    }
    im.Store(instr.vector_and_constant_result, res, force);
  }
  // Any scalar result on a tracked target is unknown.
  if (!instr.IsScalarOpDefaultNop()) {
    im.Store(instr.scalar_result, nullptr, true);
  }
}

void PosPathTracker::Finish(PosPath& out) {
  Impl& im = *impl_;
  out = PosPath{};
  out.eligible = false;
  if (im.cf_unsafe) {
    out.reason = PosPath::kControlFlow;
    return;
  }
  for (uint32_t i = 0; i < 4; ++i) {
    if (!im.pos_written[i]) {
      out.reason = PosPath::kPosNotWritten;
      return;
    }
    if (im.pos[i].unknown) {
      out.reason = PosPath::kNonAffine;
      return;
    }
  }
  int fetch = -1;
  for (uint32_t i = 0; i < 4; ++i) {
    for (uint32_t t = 0; t < im.pos[i].n; ++t) {
      const Term& term = im.pos[i].t[t];
      if (term.basis == PosPath::kBasisOne) continue;
      if (fetch < 0) {
        fetch = term.fetch;
      } else if (fetch != int(term.fetch)) {
        out.reason = PosPath::kMultiFetch;
        return;
      }
    }
  }
  if (fetch < 0) {
    out.reason = PosPath::kNoFetch;
    return;
  }
  if (!im.fetches[fetch].index_ok) {
    out.reason = PosPath::kIndexReg;
    return;
  }
  for (uint32_t i = 0; i < 4; ++i) {
    for (uint32_t t = 0; t < im.pos[i].n; ++t) {
      const Term& term = im.pos[i].t[t];
      out.M[i][term.basis] = term.p;
      if (term.basis < 4 && term.p.n) out.used_basis_mask |= uint8_t(1u << term.basis);
    }
  }
  const Impl::Fetch& fe = im.fetches[fetch];
  out.fetch_constant = fe.fetch_constant;
  out.index_register = fe.index_register;
  out.format = fe.attributes.data_format;
  out.offset_dwords = fe.attributes.offset;
  out.stride_dwords = fe.attributes.stride;
  out.exp_adjust = fe.attributes.exp_adjust;
  out.is_signed = fe.attributes.is_signed;
  out.is_integer = fe.attributes.is_integer;
  out.is_index_rounded = fe.attributes.is_index_rounded;
  out.signed_rf_mode = fe.attributes.signed_rf_mode;
  out.reason = PosPath::kEligible;
  out.eligible = true;
}

void PosPath::Evaluate(const float* consts, float m[4][5]) const {
  for (uint32_t i = 0; i < 4; ++i) {
    for (uint32_t j = 0; j < 5; ++j) {
      const Poly& p = M[i][j];
      float v = 0.0f;
      for (uint32_t k = 0; k < p.n; ++k) {
        float t = float(p.m[k].sign);
        for (uint32_t s = 0; s < p.m[k].n; ++s) {
          t *= consts[p.m[k].s[s].reg * 4 + p.m[k].s[s].comp];
        }
        v += t;
      }
      m[i][j] = v;
    }
  }
}

void PosPath::Describe(char* out, size_t out_size) const {
  static const char kComp[] = "xyzw";
  static const char* kBasis[] = {"x", "y", "z", "w", "1"};
  size_t pos = 0;
  auto put = [&](const char* s) {
    const size_t n = std::strlen(s);
    if (pos + n + 1 >= out_size) return;
    std::memcpy(out + pos, s, n);
    pos += n;
    out[pos] = 0;
  };
  char buf[48];
  for (uint32_t i = 0; i < 4; ++i) {
    std::snprintf(buf, sizeof(buf), "%s%c=", i ? " " : "", kComp[i]);
    put(buf);
    bool any = false;
    for (uint32_t j = 0; j < 5; ++j) {
      const Poly& p = M[i][j];
      for (uint32_t k = 0; k < p.n; ++k) {
        const Mono& mo = p.m[k];
        put(any ? (mo.sign < 0 ? "-" : "+") : (mo.sign < 0 ? "-" : ""));
        any = true;
        if (mo.n == 0) put("1");
        for (uint32_t s = 0; s < mo.n; ++s) {
          std::snprintf(buf, sizeof(buf), "%sc%u.%c", s ? "*" : "", mo.s[s].reg,
                        kComp[mo.s[s].comp]);
          put(buf);
        }
        if (j < 4) {
          std::snprintf(buf, sizeof(buf), "*%s", kBasis[j]);
          put(buf);
        }
      }
    }
    if (!any) put("0");
  }
}

}  // namespace rex::graphics
