#pragma once

#include <cassert>
#include <initializer_list>

#include "../simd_granodi/simd_granodi.h"

// WARNING: These functions use a faster frexp() implementation that has not
// been tested on denormal numbers.
// They are, in general, written to be fast and used in audio DSP.
// Also, they do not return correct error values (eg log(-1) returns minus
// infinity)

namespace simd_granodi {

template <typename CoeffType, int32_t N>
class Poly {
    CoeffType coeff_[N] {};
public:
    Poly(const std::initializer_list<CoeffType>& coeff) {
        static_assert(N > 0, "must have at least one coefficient");
        assert(coeff.size() == N);
        std::size_t i = 0;
        for (const CoeffType& c : coeff) {
            if (i < N) coeff_[i++] = c;
        }
    }

    // To evaluate two polynomials at the same time, where ArgType is
    // Vec_ps with all elements giving equal value, and the results stored in
    // 0 and 1
    Poly(const Poly<Vec_ss, N>& poly1, const Poly<Vec_ss, N>& poly0) {
        for (int32_t i = 0; i < N; ++i) {
            coeff_[i] = Vec_ps{0.0f, 0.0f, poly1[i].data(), poly0[i].data()};
        }
    }

    Poly(const Poly<Vec_sd, N>& poly1, const Poly<Vec_sd, N>& poly0) {
        for (int32_t i = 0; i < N; ++i) {
            coeff_[i] = Vec_pd{poly1[i].data(), poly0[i].data()};
        }
    }

    template <typename ArgType>
    ArgType eval(const ArgType& x) const {
        ArgType result;
        if (N == 1) {
            result = x * ArgType::from(coeff_[0]);
        } else if (N > 1) {
            result = x.mul_add(ArgType::from(coeff_[0]),
                ArgType::from(coeff_[1]));
            for (int32_t i = 2; i < N; ++i) {
                result = result.mul_add(x, ArgType::from(coeff_[i]));
            }
        } else {
            result = x;
        }
        return result;
    }

    // evaluates ((insert + coeff[0])*x + coeff[1])*x ...
    // use case: first coefficient is 1.0, or 0.0 (for parallel eval)
    template <typename ArgType>
    ArgType eval_insert(const ArgType& x, const ArgType& insert) const {
        ArgType result;
        if (N > 0) {
            result = insert + ArgType::from(coeff_[0]);
            for (int32_t i = 1; i < N; ++i) {
                result = result.mul_add(x, ArgType::from(coeff_[i]));
            }
        } else {
            result = x;
        }
        return result;
    }

    const CoeffType& operator[](std::size_t i) const {
        assert(0 <= i && i < N);
        return coeff_[i];
    }
};

//
//
// CUBIC APPROXIMATIONS: very fast, very smooth, but not accurate

// Calculate log2(x) for x in [1, 2) using a cubic approximation.
// Gradient matches gradient of log2() at either end
static const Poly<Vec_sd, 4> log2_p3_poly_ {
 1.6404256133344508e-1,
-1.0988652862227437,
 3.1482979293341158,
-2.2134752044448169 };

template <typename VecType>
inline VecType log2_p3(const VecType& x) {
    VecType exponent = VecType::from(x.exponent_s32()),
        mantissa = x.mantissa();
    mantissa = log2_p3_poly_.eval(mantissa);
    return (x > 0.0).choose(exponent + mantissa, VecType::minus_infinity());
}

// Calculate exp2(x) for x in [0, 1]. Gradient matches exp2() at both ends
static const Poly<Vec_sd, 4> exp2_p3_poly_ {
7.944154167983597e-2,
2.2741127776021886e-1,
6.931471805599453e-1,
1.0 };

template <typename VecType>
inline VecType exp2_p3(const VecType& x) {
    const auto floor_s32 = x.floor_to_s32();
    const VecType floor_f = VecType::from(floor_s32);
    VecType frac = x - floor_f;
    frac = exp2_p3_poly_.eval(frac);
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

static const Poly<Vec_ss, 9> logf_poly_ {
 7.0376836292e-2f,
-1.1514610310e-1f,
 1.1676998740e-1f,
-1.2420140846e-1f,
 1.4249322787e-1f,
-1.6668057665e-1f,
 2.0000714765e-1f,
-2.4999993993e-1f,
 3.3333331174e-1f };

template <typename VecType>
inline VecType logf_cm(const VecType& a) {
    using elem = typename VecType::elem_t;
    static_assert(sizeof(elem) == 4, "logf_cm() is for f32 types");

    VecType x = a.mantissa_frexp();
    VecType e = VecType::from(a.exponent_frexp_s32());
    auto x_lt_sqrth = x < elem{SQRTH};
    e -= x_lt_sqrth.choose_else_zero(1.0);
    x += x_lt_sqrth.choose_else_zero(x);
    x -= 1.0;

    VecType z = x * x;

    VecType y = logf_poly_.eval(x) * x * z;

    y += e*elem{log_q1_};
    y += -0.5 * z;
    z = x + y;
    z += e*elem{log_q2_};

    return (a > 0.0).choose(z, VecType::minus_infinity());
}

static const Poly<Vec_ss, 6> expf_poly_ {
1.9875691500e-4f,
1.3981999507e-3f,
8.3334519073e-3f,
4.1665795894e-2f,
1.6666665459e-1f,
5.0000001201e-1f };

template <typename VecType>
inline VecType expf_cm(const VecType& a) {
    using elem = typename VecType::elem_t;
    static_assert(sizeof(elem) == 4, "expf_cm() is for f32 types");

    VecType x = a;
    VecType z = x * elem{log2e_};
    auto n = z.convert_to_nearest_s32();
    z = VecType::from(n);

    x -= z*elem{log_q2_} + z*elem{log_q1_};
    z = x * x;
    z *= expf_poly_.eval(x);
    z += x + 1.0;
    return z.ldexp(n);
}

// sincos constants
static constexpr double four_over_pi_ = 1.27323954473516;
static constexpr float DP1_ = 0.78515625f;
static constexpr float DP2_ = 2.4187564849853515625e-4f;
static constexpr float DP3_ = 3.77489497744594108e-8f;

static const Poly<Vec_ss, 3> sinf_coeff_ {
-1.9515295891e-4f,
 8.3321608736e-3f,
-1.6666654611e-1f };
static const Poly<Vec_ss, 3> cosf_coeff_ {
 2.443315711809948e-5f,
-1.388731625493765e-3f,
 4.166664568298827e-2f };

// {cos, sin}
static const Poly<Vec_ps, 3> sincosf_coeff_ { cosf_coeff_, sinf_coeff_ };

template <typename VecType>
struct sincos_result { VecType sin_result, cos_result; };

// sin and cos for f32 break when x >= 8192
inline sincos_result<Vec_ss> sincosf_cm(const Vec_ss& xx) {
    // {cos sign bit, sin sign bit}
    Vec_ps signbits {0.0f, 0.0f, 0.0f, (xx & -0.0f).data()};
    float x = xx.abs().data();
    int32_t j = static_cast<int32_t>(x * static_cast<float>(four_over_pi_));
    float y = static_cast<float>(j);
    if (j & 1) {
        ++j;
        y += 1.0f;
    }
    j &= 7;
    if (j > 3) {
        signbits ^= Vec_ps{0.0f, 0.0f, -0.0f, -0.0f};
        j -= 4;
    }
    if (j > 1) signbits ^= Vec_ps{0.0f, 0.0f, -0.0f, 0.0f};
    x = ((x - y * DP1_) - y * DP2_) - y * DP3_;
    const float z = x * x;
    // From here, calculate both {cos, sin} results in parallel
    Vec_ps result = sincosf_coeff_.eval(Vec_ps{0.0f, 0.0f, z, z}) * z;
    result *= Vec_ps{0.0f, 0.0f, z, x};
    result += Vec_ps{0.0f, 0.0f, 1.0f - 0.5f*z, x};
    if ((j == 1) || (j == 2)) result = result.shuffle<3, 2, 0, 1>();
    result ^= signbits;
    sincos_result<Vec_ss> r;
    r.sin_result = result.f0();
    r.cos_result = result.f1();
    return r;
}

inline Vec_ss sinf_cm(const Vec_ss& x) { return sincosf_cm(x).sin_result; }
inline Vec_ss cosf_cm(const Vec_ss& x) { return sincosf_cm(x).cos_result; }

inline sincos_result<Vec_ps> sincosf_cm(const Vec_ps& xx) {
    Vec_ps cos_signbit = 0.0f, sin_signbit = xx & -0.0f;
    Vec_ps x = xx.abs();
    Vec_pi32 j = (x * static_cast<float>(four_over_pi_)).truncate_to_s32();
    Vec_ps y = j.convert_to_f32();
    const Compare_pi32 j_odd {(j & 1) != 0};
    j += j_odd.choose_else_zero(1);
    y += j_odd.convert_to_cmp_f32().choose_else_zero(1.0f);
    j &= 7;
    const Compare_pi32 j_gt_3 { j > 3 };
    j -= j_gt_3.choose_else_zero(4);
    cos_signbit ^= j_gt_3.convert_to_cmp_f32().choose_else_zero(-0.0f);
    sin_signbit ^= j_gt_3.convert_to_cmp_f32().choose_else_zero(-0.0f);
    const Compare_ps j_gt_1 = (j > 1).convert_to_cmp_f32();
    cos_signbit ^= j_gt_1.choose_else_zero(-0.0f);
    x = ((x - y * DP1_) - y * DP2_) - y * DP3_;
    const Vec_ps z = x * x;
    // Brackets on following line needed for identical scalar / vec behaviour
    const Vec_ps cos_result = cosf_coeff_.eval(z) * z * z + (1.0f - 0.5f*z),
        sin_result = sinf_coeff_.eval(z) * z * x + x;
    const Compare_ps swap = ((j == 1) || (j == 2)).convert_to_cmp_f32();
    sincos_result<Vec_ps> result;
    result.cos_result = swap.choose(sin_result, cos_result) ^ cos_signbit;
    result.sin_result = swap.choose(cos_result, sin_result) ^ sin_signbit;
    return result;
}

inline Vec_ps sinf_cm(const Vec_ps& x) { return sincosf_cm(x).sin_result; }
inline Vec_ps cosf_cm(const Vec_ps& x) { return sincosf_cm(x).cos_result; }

//
//
// CEPHES 64-BIT IMPLEMENTATIONS

template <typename VecType>
struct log_cm_setup_result_ {
    VecType x, y, z, e;
};

template <typename VecType>
inline VecType log_cm_setup_(const VecType& a) {
    //
}

static const Poly<Vec_sd, 4> log_coeff_R_ {
-7.89580278884799154124e-1,
 1.63866645699558079767e1
 -6.41409952958715622951e1 };
static const Poly<Vec_sd, 4> log_coeff_S_ {
// 1.0,
-3.56722798256324312549e1,
 3.12093766372244180303e2,
-7.69691943550460008604e2 };

static const Poly<Vec_pd, 4> log_coeff_R_S_ { log_coeff_R_, log_coeff_S_ };

inline Vec_sd log_cm(const Vec_sd& a) {
    if (a.data() <= 0.0) return sg_minus_infinity_f64x1;
    double x = a.mantissa_frexp().data();
    int32_t e = a.exponent_frexp().data();
    if ((e < -2) || (e > 2)) {
        double y, z;
        if (x < SQRTH) {
            e -= 1;
            z = x - 0.5;
            y = 0.5*z + 0.5;
        } else {
            z = x - 1.0;
            y = 0.5*x + 0.5;
        }
        x = z / y;
        z = x * x;

        // {R, S}
        Vec_pd z_ratio {z};
        z_ratio = z_ratio.mul_add(Vec_pd{0.0, 1.0},
                Vec_pd{-7.89580278884799154124e-1, -3.56722798256324312549e1})
            .mul_add(z_ratio, Vec_pd{1.63866645699558079767e1,
                3.12093766372244180303e2})
            .mul_add(z_ratio, Vec_pd{-6.41409952958715622951e1,
                -7.69691943550460008604e2});

        z = x * ((z * z_ratio.d1()) / z_ratio.d0());
        const double e_double = static_cast<double>(e);
        y = e_double;
        z -= y * 2.121944400546905827679e-4;
        z += x;
        z += e_double * 6.93359375e-1;
        return z;
    } else {
        if (x < SQRTH) {
            e -= 1;
            x = 2.0*x - 1.0;
        } else {
            x -= 1.0;
        }
        double y, z;
        z = x * x;

        // {P, Q}
        Vec_pd x_ratio{x};
        x_ratio = x_ratio.mul_add(Vec_pd{1.01875663804580931796e-4, 1.0},
                Vec_pd{4.97494994976747001425e-1, 1.12873587189167450590e1})
            .mul_add(x_ratio, Vec_pd{4.70579119878881725854,
                4.52279145837532221105e1})
            .mul_add(x_ratio, Vec_pd{1.44989225341610930846e1,
                8.29875266912776603211e1})
            .mul_add(x_ratio, Vec_pd{1.79368678507819816313e1,
                7.11544750618563894466e1})
            .mul_add(x_ratio, Vec_pd{7.70838733755885391666e0,
                2.31251620126765340583e1});

        y = x * ((z * x_ratio.d1()) / x_ratio.d0());
        const double e_double = static_cast<double>(e);
        if (e) y -= e_double * 2.121944400546905827679e-4;
        y -= 0.5 * z;
        z = x + y;
        if (e) z += e_double * 6.93359375e-1;
        return z;
    }
}

inline Vec_pd log_cm(const Vec_pd& x) {
    return Vec_pd{log_cm(Vec_sd{x.d1()}).data(),
        log_cm(Vec_sd{x.d0()}).data()};
}

template <typename VecType>
inline VecType exp_cm(const VecType& a) {
    VecType x = a;
    auto n = (x * log2e_).convert_to_nearest_s32();
    VecType px = VecType::from(n);

    x -= px * 6.93145751953125E-1 + px * 1.42860682030941723212E-6;

    VecType xx = x * x;
    VecType P = x * xx.mul_add(1.26177193074810590878e-4,
            3.02994407707441961300e-2)
        .mul_add(xx, 9.99999999999999999910e-1);
    VecType Q = xx.mul_add(3.00198505138664455042e-6,
            2.52448340349684104192e-3)
        .mul_add(xx, 2.27265548208155028766e-1)
        .mul_add(xx, 2.00000000000000000009e0);
    x = P / (Q - P);
    x = 2.0*x + 1.0;
    return x.ldexp(n);
}

static constexpr double DP1 = 7.85398125648498535156e-1;
static constexpr double DP2 = 3.77489470793079817668e-8;
static constexpr double DP3 = 2.69515142907905952645e-15;

inline Vec_sd sin_cm(const Vec_sd& a) {
    double x = a.data();
    int32_t sign = x < 0.0 ? -1 : 1;
    x = std::abs(x);

    double y = std::floor(x * four_over_pi_);

    double z = std::floor(y * 0.0625);
    z = y - 16.0*z;

    int32_t j = static_cast<int32_t>(z);
    if (j & 1) {
        ++j;
        y += 1.0;
    }
    j &= 7;
    if (j > 3) {
        sign = -sign;
        j -= 4;
    }

    z = ((x - DP1*y) - DP2*y) - DP3*y;

    double zz = z * z;

    Vec_sd poly{zz};
    if ((j == 1) || (j == 2)) {
        poly = poly.mul_add(-1.13585365213876817300e-11,
                2.08757008419747316778e-9)
            .mul_add(poly, -2.75573141792967388112e-7)
            .mul_add(poly, 2.48015872888517045348e-5)
            .mul_add(poly, -1.38888888888730564116e-3)
            .mul_add(poly, 4.16666666666665929218e-2);
        y = 1.0 - 0.5*zz + zz * zz * poly.data();
    } else {
        poly = poly.mul_add(1.58962301576546568060e-10,
                -2.50507477628578072866e-8)
            .mul_add(poly, 2.75573136213857245213e-6)
            .mul_add(poly, -1.98412698295895385996e-4)
            .mul_add(poly, 8.33333333332211858878e-3)
            .mul_add(poly, -1.66666666666666307295e-1);
        y = z + z * (zz * poly.data());
    }

    y = sign < 0 ? -y : y;

    return y;
}

inline Vec_pd sin_cm(const Vec_pd& a) {
    return Vec_pd{
        sin_cm(Vec_sd{a.d1()}).data(),
        sin_cm(Vec_sd{a.d0()}).data()
    };
}

} // namespace simd_granodi
