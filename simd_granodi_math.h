#pragma once

#include "simd_granodi.h"

namespace simd_granodi {

//
//
// CUBIC APPROXIMATIONS

// Polynomial coefficients
// Calculate log2(x) for x in [1, 2]
static constexpr double log2_coeff_[] = { 1.6404256133344508e-1,
    -1.0988652862227437, 3.1482979293341158, -2.2134752044448169 };

// Calculate exp2(x) for x in [0, 1]
static constexpr double exp2_coeff_[] = { 7.944154167983597e-2,
    2.2741127776021886e-1, 6.931471805599453e-1, 1.0 };

template <typename VecType>
inline VecType log2_p3(const VecType& x) {
    VecType exponent = VecType::from(x.exponent_s32()),
        mantissa = x.mantissa();
    mantissa = mantissa.mul_add(VecType::elem_t(log2_coeff_[0]),
            VecType::elem_t(log2_coeff_[1]))
        .mul_add(mantissa, VecType::elem_t(log2_coeff_[2]))
        .mul_add(mantissa, VecType::elem_t(log2_coeff_[3]));
    return (x > 0.0).choose(exponent + mantissa, VecType::minus_infinity());
}

template <typename VecType>
inline VecType exp2_p3(const VecType& x) {
    const auto floor_s32 = x.floor_to_s32();
    const VecType floor_f = VecType::from(floor_s32);
    VecType frac = x - floor_f;
    frac = frac.mul_add(VecType::elem_t(exp2_coeff_[0]),
            VecType::elem_t(exp2_coeff_[1]))
        .mul_add(frac, VecType::elem_t(exp2_coeff_[2]))
        .mul_add(frac, VecType::elem_t(exp2_coeff_[3]));
    return frac.ldexp(floor_s32);
}

template <typename VecType>
inline VecType exp_p3(const VecType& x) {
    return exp2_p3(VecType{x * VecType::elem_t(6.931471805599453e-1)});
}

//
//
// CEPHES MATH LIBRARY 32-BIT FLOAT IMPLEMENTATIONS

static constexpr double SQRTH = 7.07106781186547524401e-1;

// log constants
static constexpr double logf_coeff_[] = { 7.0376836292e-2, -1.1514610310e-1,
    1.1676998740e-1, -1.2420140846e-1, 1.4249322787e-1, -1.6668057665e-1,
    2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1, 0.0 };
static constexpr double log_q1_ = -2.12194440e-4;
static constexpr double log_q2_ = 6.93359375e-1;

// exp constants
static constexpr double log2e_ = 1.44269504088896341; // log2(e)
static constexpr double expf_coeff_[] = { 1.9875691500e-4, 1.3981999507e-3,
    8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1 };

// sincos constants
static constexpr double FOPI = 1.27323954473516; // 4/pi
static constexpr double dp1_ = 0.78515625, dp2_ = 2.4187564849853515625e-4,
    dp3_ = 3.77489497744594108e-8;
static constexpr double sincof_[] = { -1.9515295891e-4, 8.3321608736e-3,
    -1.6666654611e-1 };
static constexpr double coscof_[] = { 2.443315711809948e-5,
    -1.388731625493765e-3, 4.166664568298827e-2 };

template <typename VecType>
inline VecType logf_cm(const VecType& x) {
    auto x_lt_sqrth = x < VecType::elem_t(SQRTH);
    VecType e = VecType::from(x.exponent_frexp_s32())
        - x_lt_sqrth.choose_else_zero(1.0);
    VecType mantissa = x.mantissa_frexp();
    mantissa = (mantissa + x_lt_sqrth.choose_else_zero(mantissa)) - 1.0;

    VecType z = mantissa * mantissa;

    VecType y = mantissa.mul_add(VecType::elem_t(logf_coeff_[0]),
            VecType::elem_t(logf_coeff_[1]))
        .mul_add(mantissa, VecType::elem_t(logf_coeff_[2]))
        .mul_add(mantissa, VecType::elem_t(logf_coeff_[3]))
        .mul_add(mantissa, VecType::elem_t(logf_coeff_[4]))
        .mul_add(mantissa, VecType::elem_t(logf_coeff_[5]))
        .mul_add(mantissa, VecType::elem_t(logf_coeff_[6]))
        .mul_add(mantissa, VecType::elem_t(logf_coeff_[7]))
        .mul_add(mantissa, VecType::elem_t(logf_coeff_[8]))
        .mul_add(mantissa, VecType::elem_t(logf_coeff_[9]));

    y *= z;
    y += e*VecType::elem_t(log_q1_) - 0.5*z;

    z = mantissa + y + e*VecType::elem_t(log_q2_);

    return (x > 0.0).choose(z, VecType::minus_infinity());
}

template <typename VecType>
inline VecType expf_cm(const VecType& x) {
    VecType xx = x;
    VecType z = xx * VecType::elem_t(log2e_);

    auto n = z.convert_to_nearest_s32();
    z = VecType::from(n);

    xx -= z*VecType::elem_t(log_q2_) + z*VecType::elem_t(log_q1_);
    z = xx * xx;
    VecType tmp_z = z;

    z = xx.mul_add(VecType::elem_t(expf_coeff_[0]),
            VecType::elem_t(expf_coeff_[1]))
        .mul_add(xx, VecType::elem_t(expf_coeff_[2]))
        .mul_add(xx, VecType::elem_t(expf_coeff_[3]))
        .mul_add(xx, VecType::elem_t(expf_coeff_[4]))
        .mul_add(xx, VecType::elem_t(expf_coeff_[5]));
    z *= tmp_z;
    z += xx + 1.0;
    return z.ldexp(n);
}

template <typename VecType>
struct sincosf_result { VecType sin_result, cos_result; };

// Breaks for x > 8192
template <typename VecType>
inline sincosf_result<VecType> sincosf_cm(const VecType& x) {
    VecType sin_signbit = x & -0.0,
        xx = x.abs();
    auto floor_s32 = (xx * VecType::elem_t(FOPI)).floor_to_s32();
    VecType floor_f = VecType::from(floor_s32);
    const auto floor_odd = (floor_s32 & 1) == 1;
    floor_s32 += floor_odd.choose_else_zero(1);
    floor_f += VecType::compare_t(floor_odd).choose_else_zero(1.0);

    floor_s32 &= 7;

    const auto floor_gt3 = floor_s32 > 3;
    const auto floor_gt3_f = VecType::compare_t(floor_gt3);
    floor_s32 -= floor_gt3.choose_else_zero(4);
    sin_signbit ^= floor_gt3_f.choose_else_zero(-0.0);
    VecType cos_signbit = floor_gt3_f.choose_else_zero(-0.0);

    const auto floor_gt1 = VecType::compare_t(floor_s32 > 1);
    cos_signbit ^= floor_gt1.choose_else_zero(-0.0);

    xx -= floor_f*VecType::elem_t(dp1_)
        + floor_f*VecType::elem_t(dp2_)
        + floor_f*VecType::elem_t(dp3_);
    VecType z = xx * xx;

    // Calculate cos
    VecType cos_y = z.mul_add(VecType::elem_t(coscof_[0]),
        VecType::elem_t(coscof_[1])).mul_add(z, VecType::elem_t(coscof_[2]));
    cos_y = (cos_y*z*z - z*0.5) + 1.0;

    // Calculate sin
    VecType sin_y = z.mul_add(VecType::elem_t(sincof_[0]),
        VecType::elem_t(sincof_[1])).mul_add(z, sincof_[2]);
    sin_y = sin_y*z*xx + xx;

    // Choose results
    const auto swap_results = VecType::compare_t(
        (floor_s32 == 1) || (floor_s32 == 2));
    sincosf_result<VecType> result;
    result.sin_result = swap_results.choose(cos_y, sin_y);
    result.cos_result = swap_results.choose(sin_y, cos_y);

    // Appy signs
    result.sin_result ^= sin_signbit;
    result.cos_result ^= cos_signbit;

    return result;
}

template <typename VecType>
inline VecType sinf_cm(const VecType& x) { return sincosf_cm(x).sin_result; }

template <typename VecType>
inline VecType cosf_cm(const VecType& x) { return sincosf_cm(x).cos_result; }

} // namespace simd_granodi
