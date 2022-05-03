#pragma once

#include "../simd_granodi/simd_granodi.h"

namespace simd_granodi {

//
//
// CUBIC APPROXIMATIONS

template <typename VecType>
inline VecType log2_p3(const VecType& x) {
    VecType exponent = VecType::from(x.exponent_s32()),
        mantissa = x.mantissa();
    using elem = typename VecType::elem_t;
    // Calculate log2(x) for x in [1, 2) using a cubic approximation.
    // Gradient matches gradient of log2() at either end
    mantissa = mantissa.mul_add(elem{1.6404256133344508e-1},
            elem{-1.0988652862227437})
        .mul_add(mantissa, elem{3.1482979293341158})
        .mul_add(mantissa, elem{-2.2134752044448169});
    return (x > 0.0).choose(exponent + mantissa, VecType::minus_infinity());
}

template <typename VecType>
inline VecType exp2_p3(const VecType& x) {
    const auto floor_s32 = x.floor_to_s32();
    const VecType floor_f = VecType::from(floor_s32);
    VecType frac = x - floor_f;
    using elem = typename VecType::elem_t;
    // Calculate exp2(x) for x in [0, 1]
    frac = frac.mul_add(elem{7.944154167983597e-2},
            elem{2.2741127776021886e-1})
        .mul_add(frac, elem{6.931471805599453e-1})
        .mul_add(frac, elem{1.0});
    return frac.ldexp(floor_s32);
}

template <typename VecType>
inline VecType exp_p3(const VecType& x) {
    using elem = typename VecType::elem_t;
    return exp2_p3(VecType{x * elem{6.931471805599453e-1}});
}

//
//
// CEPHES MATH LIBRARY 32-BIT FLOAT IMPLEMENTATIONS

static constexpr double SQRTH = 7.07106781186547524401e-1;

// logf and expf constants
static constexpr double log_q1_ = -2.12194440e-4,
    log_q2_ = 6.93359375e-1,
    log2e_ = 1.44269504088896341; // log2(e)

template <typename VecType>
inline VecType logf_cm(const VecType& x) {
    using elem = typename VecType::elem_t;
    auto x_lt_sqrth = x < elem{SQRTH};
    VecType e = VecType::from(x.exponent_frexp_s32())
        - x_lt_sqrth.choose_else_zero(1.0);
    VecType mantissa = x.mantissa_frexp();
    mantissa = (mantissa + x_lt_sqrth.choose_else_zero(mantissa)) - 1.0;

    VecType z = mantissa * mantissa;

    VecType y = mantissa.mul_add(elem{7.0376836292e-2}, elem{-1.1514610310e-1})
        .mul_add(mantissa, elem{1.1676998740e-1})
        .mul_add(mantissa, elem{-1.2420140846e-1})
        .mul_add(mantissa, elem{1.4249322787e-1})
        .mul_add(mantissa, elem{-1.6668057665e-1})
        .mul_add(mantissa, elem{2.0000714765e-1})
        .mul_add(mantissa, elem{-2.4999993993e-1})
        .mul_add(mantissa, elem{3.3333331174e-1}) * mantissa;

    y *= z;
    y += e*elem{log_q1_} - 0.5*z;

    z = mantissa + y + e*elem{log_q2_};

    return (x > 0.0).choose(z, VecType::minus_infinity());
}

template <typename VecType>
inline VecType expf_cm(const VecType& x) {
    using elem = typename VecType::elem_t;
    VecType xx = x;
    VecType z = xx * elem{log2e_};

    auto n = z.convert_to_nearest_s32();
    z = VecType::from(n);

    xx -= z*elem{log_q2_} + z*elem{log_q1_};
    z = xx * xx;
    VecType tmp_z = z;

    z = xx.mul_add(elem{1.9875691500e-4}, elem{1.3981999507e-3})
        .mul_add(xx, elem{8.3334519073e-3})
        .mul_add(xx, elem{4.1665795894e-2})
        .mul_add(xx, elem{1.6666665459e-1})
        .mul_add(xx, elem{5.0000001201e-1});
    z *= tmp_z;
    z += xx + 1.0;
    return z.ldexp(n);
}

// sincos constants
static constexpr double four_over_pi_ = 1.27323954473516;

template <typename VecType>
struct sincosf_result { VecType sin_result, cos_result; };

// Breaks for x > 8192
template <typename VecType>
inline sincosf_result<VecType> sincosf_cm(const VecType& x) {
    VecType sin_signbit = x & -0.0,
        xx = x.abs();
    using elem = typename VecType::elem_t;
    using compare = typename VecType::compare_t;
    auto floor_s32 = (xx * elem{four_over_pi_}).floor_to_s32();
    VecType floor_f = VecType::from(floor_s32);
    const auto floor_odd = (floor_s32 & 1) == 1;
    floor_s32 += floor_odd.choose_else_zero(1);
    floor_f += compare::from(floor_odd).choose_else_zero(1.0);

    floor_s32 &= 7;

    const auto floor_gt3 = floor_s32 > 3;
    const auto floor_gt3_f = compare::from(floor_gt3);
    floor_s32 -= floor_gt3.choose_else_zero(4);
    sin_signbit ^= floor_gt3_f.choose_else_zero(-0.0);
    VecType cos_signbit = floor_gt3_f.choose_else_zero(-0.0);

    const auto floor_gt1 = compare::from(floor_s32 > 1);
    cos_signbit ^= floor_gt1.choose_else_zero(-0.0);

    xx -= floor_f.mul_add(elem{7.8515625e-1},
        floor_f.mul_add(elem{2.4187564849853515625e-4},
        floor_f * elem{3.77489497744594108e-8}));
    VecType z = xx * xx;

    // Calculate cos
    VecType cos_y = z.mul_add(elem{2.443315711809948e-5},
            elem{-1.388731625493765e-3})
        .mul_add(z, elem{4.166664568298827e-2});
    cos_y = (cos_y*z*z - z*0.5) + 1.0;

    // Calculate sin
    VecType sin_y = z.mul_add(elem{-1.9515295891e-4},
            elem{8.3321608736e-3})
        .mul_add(z, elem{-1.6666654611e-1});
    sin_y = sin_y*z*xx + xx;

    // Choose results
    const auto swap_results = compare::from(
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

//
//
// CEPHES 64-BIT IMPLEMENTATIONS

static constexpr double P[] = {
 1.01875663804580931796e-4,
 4.97494994976747001425e-1,
 4.70579119878881725854e0,
 1.44989225341610930846e1,
 1.79368678507819816313e1,
 7.70838733755885391666e0,
};
static constexpr double Q[] = {
/* 1.00000000000000000000, */
 1.12873587189167450590e1,
 4.52279145837532221105e1,
 8.29875266912776603211e1,
 7.11544750618563894466e1,
 2.31251620126765340583e1,
};

} // namespace simd_granodi
