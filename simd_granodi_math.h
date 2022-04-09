#pragma once

#include "simd_granodi.h"

namespace simd_granodi {

//
//
// FREXP

struct frexp_result_pd {
    Vec_pd mantissa;
    Vec_pi32 exponent;
};

struct frexp_result_ps {
    Vec_ps mantissa;
    Vec_pi32 exponent;
};

/* To match C standard library convention, we actually find half the mantissa,
and the exponent + 1 */
inline frexp_result_pd frexp_pd(const Vec_pd& x) {
    frexp_result_pd result;
    result.exponent = ((x.bitcast_to_pi64().shift_rl_imm<52>() & 0x7ff) - 1022)
        .convert_to_pi32();
    result.mantissa = ((x.bitcast_to_pi64() & 0x800fffffffffffff)
        | 0x3fe0000000000000).bitcast_to_pd();
    return result;
}

inline frexp_result_ps frexp_ps(const Vec_ps& x) {
    frexp_result_ps result;
    result.exponent = (x.bitcast_to_pi32().shift_rl_imm<23>() & 0xff) - 126;
    result.mantissa = ((x.bitcast_to_pi32() & 0x807fffff) | 0x3f000000)
        .bitcast_to_ps();
    return result;
}

inline Vec_pd ldexp_pd(const Vec_pd& x, const Vec_pi32& e) {
    return (x.bitcast_to_pi64() + e.convert_to_pi64().shift_l_imm<52>())
        .bitcast_to_pd();
}

inline Vec_ps ldexp_ps(const Vec_ps& x, const Vec_pi32& e) {
    return (x.bitcast_to_pi32() + e.shift_l_imm<23>()).bitcast_to_ps();
}

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

inline Vec_pd log2_p3(const Vec_pd& x) {
    Vec_pd exponent = ((x.bitcast_to_pi64().shift_rl_imm<52>() & 0x7ff) - 1023)
        .convert_to_pi32().convert_to_pd(),
    mantissa = ((x.bitcast_to_pi64() & 0x800fffffffffffff)
        | 0x3ff0000000000000).bitcast_to_pd();

    mantissa = mantissa.mul_add(log2_coeff_[0], log2_coeff_[1])
        .mul_add(mantissa, log2_coeff_[2]).mul_add(mantissa, log2_coeff_[3]);

    // Return zero for invalid input instead of -inf, for practical DSP reasons
    return (x > 0.0).choose_else_zero(exponent + mantissa);
}

inline Vec_ps log2_p3(const Vec_ps& x) {
    Vec_ps exponent = ((x.bitcast_to_pi32().shift_rl_imm<23>() & 0xff) - 127)
        .convert_to_ps(),
    mantissa = ((x.bitcast_to_pi32() & 0x807fffff) | 0x3f800000)
        .bitcast_to_ps();

    mantissa = mantissa.mul_add(Vec_ps{log2_coeff_[0]}, Vec_ps{log2_coeff_[1]})
        .mul_add(mantissa, Vec_ps{log2_coeff_[2]})
        .mul_add(mantissa, Vec_ps{log2_coeff_[3]});

    return (x > 0.0f).choose_else_zero(exponent + mantissa);
}

inline Vec_pd exp2_p3(const Vec_pd& x) {
    const Vec_pi32 floor_pi32 = x.floor_to_pi32();
    const Vec_pd floor_pd = floor_pi32.convert_to_pd();
    Vec_pd frac = x - floor_pd;
    frac = frac.mul_add(exp2_coeff_[0], exp2_coeff_[1])
        .mul_add(frac, exp2_coeff_[2]).mul_add(frac, exp2_coeff_[3]);
    return ldexp_pd(frac, floor_pi32);
}

inline Vec_ps exp2_p3(const Vec_ps& x) {
    const Vec_pi32 floor_pi32 = x.floor_to_pi32();
    const Vec_ps floor_ps = floor_pi32.convert_to_ps();
    Vec_ps frac = x - floor_ps;
    frac = frac.mul_add(static_cast<float>(exp2_coeff_[0]),
            static_cast<float>(exp2_coeff_[1]))
        .mul_add(frac, static_cast<float>(exp2_coeff_[2]))
        .mul_add(frac, static_cast<float>(exp2_coeff_[3]));
    return ldexp_ps(frac, floor_pi32);
}

inline float log2_p3(const float x) { return log2_p3(Vec_ps{x}).f0(); }
inline double log2_p3(const double x) { return log2_p3(Vec_pd{x}).d0(); }
inline float exp2_p3(const float x) { return exp2_p3(Vec_ps{x}).f0(); }
inline double exp2_p3(const double x) { return exp2_p3(Vec_pd{x}).d0(); }

static constexpr double exp2_to_exp_scale_pre_ = 6.931471805599453e-1;

inline float exp_p3(const float x) {
    return exp2_p3(Vec_ps{x * static_cast<float>(exp2_to_exp_scale_pre_)}).f0();
}
inline double exp_p3(const double x) {
    return exp2_p3(Vec_pd{x * exp2_to_exp_scale_pre_}).d0();
}
inline Vec_ps exp_p3(const Vec_ps& x) {
    return exp2_p3(x * static_cast<float>(exp2_to_exp_scale_pre_));
}
inline Vec_pd exp_p3(const Vec_pd& x) {
    return exp2_p3(x * exp2_to_exp_scale_pre_);
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

inline Vec_ps logf_cm(const Vec_ps& x) {
    frexp_result_ps fr = frexp_ps(x);
    Compare_ps x_lt_sqrth { x < static_cast<float>(SQRTH) };
    Vec_ps e = fr.exponent.convert_to_ps() - x_lt_sqrth.choose_else_zero(1.0f);
    fr.mantissa = (fr.mantissa + x_lt_sqrth.choose_else_zero(fr.mantissa))
        - 1.0f;

    Vec_ps z = fr.mantissa * fr.mantissa;

    Vec_ps y = fr.mantissa.mul_add(static_cast<float>(logf_coeff_[0]),
            static_cast<float>(logf_coeff_[1]))
        .mul_add(fr.mantissa, static_cast<float>(logf_coeff_[2]))
        .mul_add(fr.mantissa, static_cast<float>(logf_coeff_[3]))
        .mul_add(fr.mantissa, static_cast<float>(logf_coeff_[4]))
        .mul_add(fr.mantissa, static_cast<float>(logf_coeff_[5]))
        .mul_add(fr.mantissa, static_cast<float>(logf_coeff_[6]))
        .mul_add(fr.mantissa, static_cast<float>(logf_coeff_[7]))
        .mul_add(fr.mantissa, static_cast<float>(logf_coeff_[8]))
        .mul_add(fr.mantissa, static_cast<float>(logf_coeff_[9]));
    y *= z;
    y += e*static_cast<float>(log_q1_) - 0.5f*z;

    z = fr.mantissa + y + e*static_cast<float>(log_q2_);

    return (x > 0.0f).choose_else_zero(z);
}

// This is the exact same float algorithm, but using doubles. Does NOT
// have the accuracy of doubles, but is smoother than wrapping the float alg
inline Vec_pd logf_cm(const Vec_pd& x) {
    frexp_result_pd fr = frexp_pd(x);
    Compare_pd x_lt_sqrth { x < SQRTH };
    Vec_pd e = fr.exponent.convert_to_pd() - x_lt_sqrth.choose_else_zero(1.0);
    fr.mantissa = (fr.mantissa + x_lt_sqrth.choose_else_zero(fr.mantissa))
        - 1.0;

    Vec_pd z = fr.mantissa * fr.mantissa;

    Vec_pd y = fr.mantissa.mul_add(logf_coeff_[0], logf_coeff_[1])
        .mul_add(fr.mantissa, logf_coeff_[2])
        .mul_add(fr.mantissa, logf_coeff_[3])
        .mul_add(fr.mantissa, logf_coeff_[4])
        .mul_add(fr.mantissa, logf_coeff_[5])
        .mul_add(fr.mantissa, logf_coeff_[6])
        .mul_add(fr.mantissa, logf_coeff_[7])
        .mul_add(fr.mantissa, logf_coeff_[8])
        .mul_add(fr.mantissa, logf_coeff_[9]);
    y *= z;
    y += e*log_q1_ - 0.5*z;

    z = fr.mantissa + y + e*log_q2_;

    return (x > 0.0).choose_else_zero(z);
}

inline Vec_ps expf_cm(const Vec_ps& x) {
    Vec_ps xx = x;
    Vec_ps z = xx * static_cast<float>(log2e_);

    Vec_pi32 n = z.convert_to_nearest_pi32();
    z = n.convert_to_ps();

    xx -= z*static_cast<float>(log_q2_) + z*static_cast<float>(log_q1_);
    z = xx * xx;
    Vec_ps tmp_z = z;

    z = xx.mul_add(static_cast<float>(expf_coeff_[0]),
            static_cast<float>(expf_coeff_[1]))
        .mul_add(xx, static_cast<float>(expf_coeff_[2]))
        .mul_add(xx, static_cast<float>(expf_coeff_[3]))
        .mul_add(xx, static_cast<float>(expf_coeff_[4]))
        .mul_add(xx, static_cast<float>(expf_coeff_[5]));
    z *= tmp_z;
    z += xx + 1.0f;

    return ldexp_ps(z, n);
}

inline Vec_pd expf_cm(const Vec_pd& x) {
    Vec_pd xx = x;
    Vec_pd z = xx * log2e_;

    Vec_pi32 n = z.convert_to_nearest_pi32();
    z = n.convert_to_pd();

    xx -= z*log_q2_ + z*log_q1_;
    z = xx * xx;
    Vec_pd tmp_z = z;

    z = xx.mul_add(expf_coeff_[0], expf_coeff_[1])
        .mul_add(xx, expf_coeff_[2]).mul_add(xx, expf_coeff_[3])
        .mul_add(xx, expf_coeff_[4]).mul_add(xx, expf_coeff_[5]);

    z *= tmp_z;
    z += xx + 1.0;

    return ldexp_pd(z, n);
}

inline float logf_cm(const float x) { return logf_cm(Vec_ps{x}).f0(); }
inline double logf_cm(const double x) { return logf_cm(Vec_pd{x}).d0(); }
inline float expf_cm(const float x) { return expf_cm(Vec_ps{x}).f0(); }
inline double expf_cm(const double x) { return expf_cm(Vec_pd{x}).d0(); }

struct sincosf_result_ps { Vec_ps sin_result, cos_result; };
struct sincosf_result_pd { Vec_pd sin_result, cos_result; };

// Breaks for x > 8192
inline sincosf_result_ps sincosf_cm(const Vec_ps& x) {
    Vec_ps sin_signbit = x & Vec_ps::bitcast_from_u32(0x80000000),
        xx = x.abs();
    Vec_pi32 floor_pi32 = (xx * static_cast<float>(FOPI)).floor_to_pi32();
    Vec_ps floor_ps = floor_pi32.convert_to_ps();
    const Compare_pi32 floor_odd { (floor_pi32 & 1) == 1 };
    floor_pi32 += floor_odd.choose_else_zero(1);
    floor_ps += floor_odd.bitcast_to_cmp_ps().choose_else_zero(1.0f);

    floor_pi32 &= 7;

    const Compare_pi32 floor_gt3 { floor_pi32 > 3 };
    floor_pi32 -= floor_gt3.choose_else_zero(4);
    sin_signbit ^= floor_gt3.bitcast_to_cmp_ps()
        .choose_else_zero(Vec_ps::bitcast_from_u32(0x80000000));
    Vec_ps cos_signbit = floor_gt3.bitcast_to_cmp_ps()
        .choose_else_zero(Vec_ps::bitcast_from_u32(0x80000000));

    const Compare_ps floor_gt1 = (floor_pi32 > 1).bitcast_to_cmp_ps();
    cos_signbit ^= floor_gt1.choose_else_zero(
        Vec_ps::bitcast_from_u32(0x80000000));

    xx -= floor_ps*static_cast<float>(dp1_)
        + floor_ps*static_cast<float>(dp2_)
        + floor_ps*static_cast<float>(dp3_);
    Vec_ps z = xx * xx;

    // Calculate cos
    Vec_ps cos_y = z.mul_add(static_cast<float>(coscof_[0]),
        static_cast<float>(coscof_[1])).mul_add(z, coscof_[2]);
    cos_y = (cos_y*z*z - z*0.5f) + 1.0f;

    // Calculate sin
    Vec_ps sin_y = z.mul_add(static_cast<float>(sincof_[0]),
        static_cast<float>(sincof_[1])).mul_add(z, sincof_[2]);
    sin_y = sin_y*z*xx + xx;

    // Choose results
    Compare_ps swap_results = ((floor_pi32 == 1) || (floor_pi32 == 2))
        .bitcast_to_cmp_ps();
    sincosf_result_ps result;
    result.sin_result = swap_results.choose(cos_y, sin_y);
    result.cos_result = swap_results.choose(sin_y, cos_y);

    // Appy signs
    result.sin_result ^= sin_signbit;
    result.cos_result ^= cos_signbit;

    return result;
}

inline sincosf_result_pd sincosf_cm(const Vec_pd& x) {
    Vec_pd sin_signbit = x & Vec_pd::bitcast_from_u64(0x8000000000000000),
        xx = x.abs();
    Vec_pi32 floor_pi32 = (xx * FOPI).floor_to_pi32();
    Vec_pd floor_pd = floor_pi32.convert_to_pd();
    Compare_pi32 floor_odd { (floor_pi32 & 1) == 1 };
    floor_pi32 += floor_odd.choose_else_zero(1);
    floor_pd += floor_odd.convert_to_cmp_pd().choose_else_zero(1.0);

    floor_pi32 &= 7;

    const Compare_pi32 floor_gt3 { floor_pi32 > 3 };
    floor_pi32 -= floor_gt3.choose_else_zero(4);
    const Compare_pd floor_gt3_pd = floor_gt3.convert_to_cmp_pd();
    sin_signbit ^= floor_gt3_pd.choose_else_zero(
        Vec_pd::bitcast_from_u64(0x8000000000000000));
    Vec_pd cos_signbit = floor_gt3_pd.choose_else_zero(
        Vec_pd::bitcast_from_u64(0x8000000000000000));

    const Compare_pd floor_gt1 = (floor_pi32 > 1).convert_to_cmp_pd();
    cos_signbit ^= floor_gt1.choose_else_zero(
        Vec_pd::bitcast_from_u64(0x8000000000000000));

    xx -= floor_pd*dp1_ + floor_pd*dp2_ + floor_pd*dp3_;
    Vec_pd z = xx * xx;

    Vec_pd cos_y = z.mul_add(coscof_[0], coscof_[1]).mul_add(z, coscof_[2]);
    cos_y = (cos_y*z*z - z*0.5f) + 1.0;

    Vec_pd sin_y = z.mul_add(sincof_[0], sincof_[1]).mul_add(z, sincof_[2]);
    sin_y = sin_y*z*xx + xx;

    Compare_pd swap_results = ((floor_pi32 == 1) || (floor_pi32 == 2))
        .convert_to_cmp_pd();
    sincosf_result_pd result;
    result.sin_result = swap_results.choose(cos_y, sin_y);
    result.cos_result = swap_results.choose(sin_y, cos_y);

    result.sin_result ^= sin_signbit;
    result.cos_result ^= cos_signbit;

    return result;
}

inline Vec_ps sinf_cm(const Vec_ps& x) { return sincosf_cm(x).sin_result; }
inline Vec_ps cosf_cm(const Vec_ps& x) { return sincosf_cm(x).cos_result; }
inline Vec_pd sinf_cm(const Vec_pd& x) { return sincosf_cm(x).sin_result; }
inline Vec_pd cosf_cm(const Vec_pd& x) { return sincosf_cm(x).cos_result; }

inline float sinf_cm(const float x) {
    return sincosf_cm(Vec_ps{x}).sin_result.f0();
}
inline float cosf_cm(const float x) {
    return sincosf_cm(Vec_ps{x}).cos_result.f0();
}
inline double sinf_cm(const double x) {
    return sincosf_cm(Vec_pd{x}).sin_result.d0();
}
inline double cosf_cm(const double x) {
    return sincosf_cm(Vec_pd{x}).cos_result.d0();
}

struct sincosf_result_f { float sin_result, cos_result; };
struct sincosf_result_d { double sin_result, cos_result; };

inline sincosf_result_f sincosf_cm(const float x) {
    sincosf_result_ps r = sincosf_cm(Vec_ps{x});
    sincosf_result_f result;
    result.sin_result = r.sin_result.f0();
    result.cos_result = r.cos_result.f0();
    return result;
}

inline sincosf_result_d sincosf_cm(const double x) {
    sincosf_result_pd r = sincosf_cm(Vec_pd{x});
    sincosf_result_d result;
    result.sin_result = r.sin_result.d0();
    result.cos_result = r.sin_result.d0();
    return result;
}

} // namespace simd_granodi
