#pragma once

#include "frexp_ldexp_vec.h"

static constexpr double SQRTH = 7.07106781186547524401e-1;
static constexpr float SQRTH_f = static_cast<float>(SQRTH);

// log constants
static constexpr double log_f_coeff_[] = { 7.0376836292e-2, -1.1514610310e-1,
    1.1676998740e-1, -1.2420140846e-1, 1.4249322787e-1, -1.6668057665e-1,
    2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1, 0.0 };
static constexpr float log_f_coeff_f_[] ={ static_cast<float>(log_f_coeff_[0]),
    static_cast<float>(log_f_coeff_[1]), static_cast<float>(log_f_coeff_[2]),
    static_cast<float>(log_f_coeff_[3]), static_cast<float>(log_f_coeff_[4]),
    static_cast<float>(log_f_coeff_[5]), static_cast<float>(log_f_coeff_[6]),
    static_cast<float>(log_f_coeff_[7]), static_cast<float>(log_f_coeff_[8]),
    static_cast<float>(log_f_coeff_[9]) };
static constexpr double log_q1_ = -2.12194440e-4;
static constexpr float log_q1_f_ = static_cast<float>(log_q1_);
static constexpr double log_q2_ = 6.93359375e-1;
static constexpr float log_q2_f_ = static_cast<float>(log_q2_);

// exp constants
static constexpr double log2e_ = 1.44269504088896341; // log2(e)
static constexpr float log2e_f_ = static_cast<float>(log2e_);
static constexpr double exp_f_coeff_[] = { 1.9875691500e-4, 1.3981999507e-3,
    8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1 };
static constexpr float exp_f_coeff_f_[] = {
    static_cast<float>(exp_f_coeff_[0]), static_cast<float>(exp_f_coeff_[1]),
    static_cast<float>(exp_f_coeff_[2]), static_cast<float>(exp_f_coeff_[3]),
    static_cast<float>(exp_f_coeff_[4]), static_cast<float>(exp_f_coeff_[5]) };

// sincos constants
static constexpr double FOPI = 1.27323954473516; // 4/pi
static constexpr float FOPI_f = static_cast<float>(FOPI);
static constexpr double dp1_ = 0.78515625, dp2_ = 2.4187564849853515625e-4,
    dp3_ = 3.77489497744594108e-8;
static constexpr float dp1_f_ = static_cast<float>(dp1_),
    dp2_f_ = static_cast<float>(dp2_), dp3_f_ = static_cast<float>(dp3_);
static constexpr double sincof_[] = { -1.9515295891e-4, 8.3321608736e-3,
    -1.6666654611e-1 };
static constexpr float sincof_f_[] = { static_cast<float>(sincof_[0]),
    static_cast<float>(sincof_[1]), static_cast<float>(sincof_[2])};
static constexpr double coscof_[] = { 2.443315711809948e-5,
    -1.388731625493765e-3, 4.166664568298827e-2 };
static constexpr float coscof_f_[] = { static_cast<float>(coscof_[0]),
    static_cast<float>(coscof_[1]), static_cast<float>(coscof_[2]) };

inline Vec_ps log_cm(const Vec_ps& x) {
    frexp_result_ps fr = frexp_ps(x);
    Compare_ps x_lt_sqrth { x < SQRTH_f };
    Vec_ps e = fr.exponent.convert_to_ps() - x_lt_sqrth.choose_else_zero(1.0f);
    fr.mantissa = (fr.mantissa + x_lt_sqrth.choose_else_zero(fr.mantissa))
        - 1.0f;

    Vec_ps z = fr.mantissa * fr.mantissa;

    Vec_ps y = ((((((((log_f_coeff_f_[0]*fr.mantissa + log_f_coeff_f_[1])
        * fr.mantissa + log_f_coeff_f_[2])*fr.mantissa + log_f_coeff_f_[3])
        * fr.mantissa + log_f_coeff_f_[4])*fr.mantissa + log_f_coeff_f_[5])
        * fr.mantissa + log_f_coeff_f_[6])*fr.mantissa + log_f_coeff_f_[7])
        * fr.mantissa + log_f_coeff_f_[8])*fr.mantissa + log_f_coeff_f_[9];
    y *= z;
    y += e*log_q1_f_ - 0.5f*z;

    z = fr.mantissa + y + e*log_q2_f_;

    return (x > 0.0f).choose_else_zero(z);
}

inline Vec_ps exp_cm(const Vec_ps& x) {
    Vec_ps xx = x;
    Vec_ps z = xx * log2e_f_;

    Vec_pi32 n = z.convert_to_nearest_pi32();
    z = n.convert_to_ps();

    xx -= z*log_q2_f_ + z*log_q1_f_;
    z = xx * xx;
    Vec_ps tmp_z = z;

    z = ((((exp_f_coeff_f_[0]*xx + exp_f_coeff_f_[1])*xx
        + exp_f_coeff_f_[2])*xx + exp_f_coeff_f_[3])*xx
        + exp_f_coeff_f_[4])*xx + exp_f_coeff_f_[5];
    z *= tmp_z;
    z += xx + 1.0f;

    return ldexp_ps(z, n);
}

struct sincos_result {
    Vec_ps sin_result, cos_result;
};

// Breaks for x > 8192
inline sincos_result sincos_cm(const Vec_ps& x) {
    Vec_ps sin_signbit = x & Vec_ps::bitcast_from_u32(0x80000000),
        xx = x.abs();
    floor_result_ps fr = floor_vec(xx * FOPI_f);
    Compare_pi32 floor_odd { (fr.floor_pi32 & 1) == 1 };
    fr.floor_pi32 += floor_odd.choose_else_zero(1);
    fr.floor_ps += floor_odd.bitcast_to_cmp_ps().choose_else_zero(1.0f);

    fr.floor_pi32 &= 7;

    Compare_pi32 floor_gt3 { fr.floor_pi32 > 3 };
    fr.floor_pi32 -= floor_gt3.choose_else_zero(4);
    sin_signbit ^= floor_gt3.bitcast_to_cmp_ps()
        .choose_else_zero(Vec_ps::bitcast_from_u32(0x80000000));
    Vec_ps cos_signbit = floor_gt3.bitcast_to_cmp_ps()
        .choose_else_zero(Vec_ps::bitcast_from_u32(0x80000000));

    Compare_pi32 floor_gt1 { fr.floor_pi32 > 1 };
    cos_signbit ^= floor_gt1.bitcast_to_cmp_ps()
        .choose_else_zero(Vec_ps::bitcast_from_u32(0x80000000));

    xx -= fr.floor_ps*dp1_f_ + fr.floor_ps*dp2_f_ + fr.floor_ps*dp3_f_;
    Vec_ps z = xx * xx;

    // Calculate cos
    Vec_ps cos_y = (coscof_f_[0]*z + coscof_f_[1])*z + coscof_f_[2];
    cos_y = (cos_y*z*z - z*0.5f) + 1.0f;

    // Calculate sin
    Vec_ps sin_y = (sincof_f_[0]*z + sincof_f_[1])*z + sincof_f_[2];
    sin_y = sin_y*z*xx + xx;

    // Choose results
    Compare_ps swap_results = ((fr.floor_pi32 == 1) || (fr.floor_pi32 == 2))
        .bitcast_to_cmp_ps();
    sincos_result result;
    result.sin_result = swap_results.choose(cos_y, sin_y);
    result.cos_result = swap_results.choose(sin_y, cos_y);

    // Appy signs
    result.sin_result ^= sin_signbit;
    result.cos_result ^= cos_signbit;

    return result;
}
