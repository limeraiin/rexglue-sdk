#pragma once

#include <cstdint>

namespace rex::graphics::nr_native {

// Counts words, not draws. Buckets cover finite, unequal, register-matched
// words except opposite signed zeros. Nonfinite words never enter ULP buckets.
struct FloatDifferenceStats {
  uint64_t words = 0, different_words = 0;
  uint64_t mapping_different_words = 0, mapping_unknown_words = 0;
  uint64_t signed_zero_words = 0, sign_different_words = 0;
  uint64_t nonfinite_words = 0, nan_words = 0, infinity_words = 0;
  uint64_t ulp1_words = 0, ulp2_words = 0, ulp3_4_words = 0, ulp_gt4_words = 0;
  uint32_t max_ulp = 0, max_register = UINT32_MAX, max_lane = 0;
  uint32_t max_native_bits = 0, max_retail_bits = 0;
};

inline void AccumulateFloatWord(FloatDifferenceStats& s, uint32_t native_bits,
                                uint32_t retail_bits, uint32_t native_register,
                                uint32_t retail_register, uint32_t lane) {
  ++s.words;
  if (native_bits != retail_bits) ++s.different_words;
  if (native_register == UINT32_MAX || retail_register == UINT32_MAX) {
    ++s.mapping_unknown_words;
    return;
  }
  if (native_register != retail_register) {
    ++s.mapping_different_words;
    return;
  }
  if (native_bits == retail_bits) return;
  const uint32_t a = native_bits & 0x7FFFFFFFu;
  const uint32_t b = retail_bits & 0x7FFFFFFFu;
  if ((native_bits ^ retail_bits) & 0x80000000u) ++s.sign_different_words;
  if (!a && !b) {
    ++s.signed_zero_words;
    return;
  }
  if (a >= 0x7F800000u || b >= 0x7F800000u) {
    ++s.nonfinite_words;
    if (a > 0x7F800000u || b > 0x7F800000u) ++s.nan_words;
    if (a == 0x7F800000u || b == 0x7F800000u) ++s.infinity_words;
    return;
  }
  // Monotonic float order, with +0 and -0 sharing one position.
  const uint32_t ao = (native_bits & 0x80000000u) ? 0x80000000u - a : 0x80000000u + a;
  const uint32_t bo = (retail_bits & 0x80000000u) ? 0x80000000u - b : 0x80000000u + b;
  const uint32_t ulp = ao > bo ? ao - bo : bo - ao;
  if (ulp == 1) ++s.ulp1_words;
  else if (ulp == 2) ++s.ulp2_words;
  else if (ulp <= 4) ++s.ulp3_4_words;
  else ++s.ulp_gt4_words;
  if (ulp > s.max_ulp) {
    s.max_ulp = ulp;
    s.max_register = native_register;
    s.max_lane = lane;
    s.max_native_bits = native_bits;
    s.max_retail_bits = retail_bits;
  }
}

}  // namespace rex::graphics::nr_native
