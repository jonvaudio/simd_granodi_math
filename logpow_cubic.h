#pragma once

#include "frexp_ldexp_vec.h"

using namespace simd_granodi;

// Polynomial coefficients
// Calculate log2(x) for x in [1, 2]
static constexpr double log2_coeff_[] = { 1.6404256133344508e-1,
    -1.0988652862227437, 3.1482979293341158, -2.2134752044448169 };
static constexpr float log2_coeff_f_[] = { static_cast<float>(log2_coeff_[0]),
    static_cast<float>(log2_coeff_[1]), static_cast<float>(log2_coeff_[2]),
    static_cast<float>(log2_coeff_[3]) };

// Calculate exp2(x) for x in [0, 1]
static constexpr double exp2_coeff_[] = { 7.944154167983597e-2,
    2.2741127776021886e-1, 6.931471805599453e-1, 1.0 };
static constexpr float exp2_coeff_f_[] = { static_cast<float>(exp2_coeff_[0]),
    static_cast<float>(exp2_coeff_[1]), static_cast<float>(exp2_coeff_[2]),
    static_cast<float>(exp2_coeff_[3]) };

inline Vec_pd log2_p3_pd(const Vec_pd& x) {
    // .shuffle<3, 2, 2, 0>().convert_to_pd() is pi64 -> pi32 -> pd, but
    // avoiding the unneeded step of zero-ing the upper elements when calling
    // convert_to_pi32(). We do this to be consistent across platforms,
    // as pi64 -> pd is slow on SSE2
    Vec_pd exponent = ((x.bitcast_to_pi64().shift_rl_imm<52>() & 0x7ff) - 1023)
        .bitcast_to_pi32().shuffle<3, 2, 2, 0>().convert_to_pd(),
    mantissa = ((x.bitcast_to_pi64() & 0x800fffffffffffff)
        | 0x3ff0000000000000).bitcast_to_pd();

    mantissa = ((log2_coeff_[0]*mantissa + log2_coeff_[1])*mantissa +
        log2_coeff_[2])*mantissa + log2_coeff_[3];

    // Return zero for invalid input instead of -inf, for practical DSP reasons
    return (x > 0.0).choose_else_zero(exponent + mantissa);
}

inline Vec_ps log2_p3_ps(const Vec_ps& x) {
    Vec_ps exponent = ((x.bitcast_to_pi32().shift_rl_imm<23>() & 0xff) - 127)
        .convert_to_ps(),
    mantissa = ((x.bitcast_to_pi32() & 0x807fffff) | 0x3f800000)
        .bitcast_to_ps();

    mantissa = ((log2_coeff_f_[0]*mantissa + log2_coeff_f_[1])*mantissa +
        log2_coeff_f_[2])*mantissa + log2_coeff_f_[3];

    return (x > 0.0f).choose_else_zero(exponent + mantissa);
}

inline Vec_pd exp2_p3_pd(const Vec_pd& x) {
    const Vec_pi32 floor_pi32 = x.floor_to_pi32();
    const Vec_pd floor_pd = floor_pi32.convert_to_pd();
    Vec_pd frac = x - floor_pd;
    frac = ((exp2_coeff_[0]*frac + exp2_coeff_[1])*frac + exp2_coeff_[2])
        *frac + exp2_coeff_[3];
    return ldexp_pd(frac, floor_pi32);
}

inline Vec_ps exp2_p3_ps(const Vec_ps& x) {
    const Vec_pi32 floor_pi32 = x.floor_to_pi32();
    const Vec_ps floor_ps = floor_pi32.convert_to_ps();
    Vec_ps frac = x - floor_ps;
    frac = ((exp2_coeff_f_[0]*frac + exp2_coeff_f_[1])*frac + exp2_coeff_f_[2])
        *frac + exp2_coeff_f_[3];
    return ldexp_ps(frac, floor_pi32);
}
