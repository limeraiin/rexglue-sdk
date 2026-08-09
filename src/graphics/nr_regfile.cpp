/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "rex/graphics/nr_regfile.h"

// [NR-DRAW] See the header. Deliberately trivial: the value of this unit is
// in WHERE it is fed and WHEN it is compared, not in anything it computes.

namespace rex {
namespace graphics {
namespace nr {

namespace {

// One register's verdict, shared by the sweep and the single-register check so
// the two can never drift into disagreeing about what a finding is.
void CompareOne(const RegShadow* s, RegShadowStats* stats, uint32_t reg,
                uint32_t live, RegShadowFindFn find, void* find_user) {
  ++stats->checks;
  if (s->defined[reg]) {
    if (s->values[reg] == live) return;
    ++stats->diverge;
    if (find) find(find_user, kRegShadowDiverge, reg, s->values[reg], live);
    return;
  }
  // Never written by the stream. Only a NONZERO live value is a finding: the
  // register file starts zeroed, so a zero on both sides says nothing about
  // whether anything writes it.
  if (!live) return;
  ++stats->externs;
  if (find) find(find_user, kRegShadowExtern, reg, 0, live);
}

}  // namespace

void RegShadowApplyWrite(RegShadow* s, RegShadowStats* stats, uint32_t reg,
                         uint32_t value) {
  if (reg >= kRegShadowCount) {
    ++stats->writes_ignored;
    return;
  }
  ++stats->writes;
  if (!s->defined[reg]) {
    s->defined[reg] = 1;
    ++stats->defined_count;
  }
  s->values[reg] = value;
}

void RegShadowApplyRange(RegShadow* s, RegShadowStats* stats, uint32_t base,
                         const uint32_t* values, uint32_t n) {
  if (base >= kRegShadowCount) {
    stats->writes_ignored += n;
    return;
  }
  uint32_t in = n;
  if (base + n > kRegShadowCount) {
    in = kRegShadowCount - base;
    stats->writes_ignored += n - in;
  }
  stats->writes += in;
  for (uint32_t i = 0; i < in; ++i) {
    if (!s->defined[base + i]) {
      s->defined[base + i] = 1;
      ++stats->defined_count;
    }
    s->values[base + i] = values[i];
  }
}

void RegShadowSweep(RegShadow* s, RegShadowStats* stats, uint32_t count,
                    const uint32_t* live, RegShadowFindFn find,
                    void* find_user) {
  if (!live) return;
  ++stats->compares;
  uint32_t pos = s->sweep;
  for (uint32_t i = 0; i < count; ++i) {
    if (pos >= kRegShadowCount) {
      pos = 0;
      ++stats->sweeps;
    }
    CompareOne(s, stats, pos, live[pos], find, find_user);
    ++pos;
  }
  s->sweep = pos;
}

void RegShadowCheckOne(const RegShadow* s, RegShadowStats* stats, uint32_t reg,
                       const uint32_t* live, RegShadowFindFn find,
                       void* find_user) {
  if (!live || reg >= kRegShadowCount) return;
  CompareOne(s, stats, reg, live[reg], find, find_user);
}

void RegShadowCompose(const RegShadow* s, const uint32_t* live, uint32_t* out) {
  for (uint32_t reg = 0; reg < kRegShadowCount; ++reg) {
    out[reg] = s->defined[reg] ? s->values[reg] : live[reg];
  }
}

}  // namespace nr
}  // namespace graphics
}  // namespace rex
