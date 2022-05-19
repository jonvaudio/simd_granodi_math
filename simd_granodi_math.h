#pragma once

#include <cassert>
#include <initializer_list>

#include "../simd_granodi/simd_granodi.h"

/*

WARNINGS:

- These functions use faster frexp()/ldexp() implementations that do
  NOT work for denormal (subnormal) numbers.

- They do NOT return correct error values (eg log(-1) returns minus
  infinity).

- Assertions are used, so you MUST define NDEBUG as a compilier argument for
  optimized builds.

- They aim to be fast and suitable for use in audio DSP.

SINGLE PRECISION FUNCTIONS:

log2_p3():
Smooth, but very inaccurate, log2() approximation for singles or doubles
Returns -inf for x <= 0.0

exp2_p3():
Smooth, but very inaccurate, exp2() approximation for singles or doubles

exp_p3():
Simple wrapper that changes the base of exp2_p3() to e.

logf_cm():
Accurate logf() for singles.
Returns -inf for x <= 0.0

expf_cm():
Accuare expf() for singles

sincosf_cm():
Accurate sinf() and cosf() for singles where x < 8192
Scalar optimization: calculates sinf() and cosf() in parallel, whereas the
vector version calculates both sinf() and cosf() (but many of the computations
are still shared).
sinf_cm() and cosf_cm() wrap sincosf_cm() and discard unneeded results.

sqrtf_cm():
Accurate sqrtf() for singles.
Scalar optimization: uses branching to select the correct polynomial, whereas
the vector version selects different polynomial coefficients via masking.

DOUBLE PRECISION FUNCTIONS:

Note: all double precision functions are overloaded such that when called with
a single precision type (ie Vec_ps or Vec_ss), they call their equivalent
single precision version.

log_cm():
Accurate log() for doubles
Scalar optimization: computes numerator and denominator polynomials in parallel.

exp_cm():
Accurate exp() for doubles
Scalar optimization: computes numerator and denominator polynomials in parallel.

sincos_cm():
Accurate sin() and cos() for x < 1.07e9
Scalar optimization: calculates sin and cos in parallel, whereas the vector
version calculates both sin() and cos() (but many of the computations
are still shared).
sin_cm() and cos_cm() wrap sincos_cm() and discard uneeded results.

sqrt_cm():
Accurate sqrt for doubles.

*/

namespace simd_granodi {

namespace sg_math_impl {

template <typename CoeffType, int32_t N>
struct Poly {
    // Needed public because of .prepend() method
    CoeffType coeff_[N] {};

    Poly() {}
    Poly(const std::initializer_list<CoeffType>& coeff_arg) {
        static_assert(N > 1, "");
        assert(coeff_arg.size() == N);
        std::size_t i = 0;
        for (const CoeffType& c : coeff_arg) {
            //if (size_mismatch) printf("size mismatch: %.4e\n", c.data());
            if (i < static_cast<std::size_t>(N)) coeff_[i++] = c;
        }
    }

    // To evaluate two polynomials at the same time, where ArgType is
    // Vec_ps with all elements giving equal value, and the results stored in
    // 0 and 1
    Poly(const Poly<Vec_ss, N>& poly1, const Poly<Vec_ss, N>& poly0) {
        for (int32_t i = 0; i < N; ++i) {
            coeff_[i] = Vec_ps{0.0f, 0.0f, poly1.coeff_[i].data(),
                poly0.coeff_[i].data()};
        }
    }

    Poly(const Poly<Vec_sd, N>& poly1, const Poly<Vec_sd, N>& poly0) {
        for (int32_t i = 0; i < N; ++i) {
            coeff_[i] = Vec_pd{poly1.coeff_[i].data(), poly0.coeff_[i].data()};
        }
    }

    Poly<CoeffType, N+1> prepend(const CoeffType& new_coeff0) const {
        Poly<CoeffType, N+1> result;
        result.coeff_[0] = new_coeff0;
        for (int32_t i = 0; i < N; ++i) result.coeff_[i+1] = coeff_[i];
        return result;
    }

    template <typename ArgType>
    ArgType eval(const ArgType& x) const {
        ArgType result = x.mul_add(coeff_[0].template to<ArgType>(),
            coeff_[1].template to<ArgType>());
        for (int32_t i = 2; i < N; ++i) {
            result = result.mul_add(x, coeff_[i].template to<ArgType>());
        }
        return result;
    }

    template <typename ArgType>
    ArgType eval1(const ArgType& x) const {
        ArgType result {x + coeff_[0].template to<ArgType>()};
        for (int32_t i = 1; i < N; ++i) {
            result = result.mul_add(x, coeff_[i].template to<ArgType>());
        }
        return result;
    }
};

template <typename CmpType>
inline void make_if_elseif_else(const CmpType& cmp1, CmpType& cmp2,
    CmpType& cmp3)
{
    cmp2 = cmp2 && !cmp1;
    cmp3 = !(cmp1 || cmp2);
}

template <typename ArgType>
inline ArgType choose3(const typename ArgType::compare_t& cmp1,
    const typename ArgType::compare_t& cmp2,
    const typename ArgType::compare_t& cmp3,
    const ArgType& x1, const ArgType& x2, const ArgType& x3)
{
    assert((cmp1 != (cmp2 || cmp3)).debug_valid_eq(true));
    assert((cmp2 != (cmp1 || cmp3)).debug_valid_eq(true));
    assert((cmp3 != (cmp1 || cmp2)).debug_valid_eq(true));
    return cmp1.choose_else_zero(x1) |
        cmp2.choose_else_zero(x2) |
        cmp3.choose_else_zero(x3);
}

template <typename CoeffType, int32_t N>
struct Poly_choose3 {
    CoeffType coeff_[N*3] {};

    Poly_choose3(const Poly<CoeffType, N>& p1,
        const Poly<CoeffType, N>& p2,
        const Poly<CoeffType, N>& p3)
    {
        static_assert(N > 1, "");
        for (int32_t i = 0; i < N; ++i) {
            coeff_[3*i] = p1.coeff_[i];
            coeff_[3*i+1] = p2.coeff_[i];
            coeff_[3*i+2] = p3.coeff_[i];
        }
    }

    template <typename ArgType>
    ArgType eval(const typename ArgType::compare_t& cmp1,
        const typename ArgType::compare_t& cmp2,
        const typename ArgType::compare_t& cmp3,
        const ArgType& x) const
    {
        ArgType result = x.mul_add(choose3(cmp1, cmp2, cmp3,
                coeff_[0].template to<ArgType>(),
                coeff_[1].template to<ArgType>(),
                coeff_[2].template to<ArgType>()),
            choose3(cmp1, cmp2, cmp3,
                coeff_[3].template to<ArgType>(),
                coeff_[4].template to<ArgType>(),
                coeff_[5].template to<ArgType>()));
        for (int32_t i = 6; i < N*3; i += 3) {
            result = result.mul_add(x, choose3(cmp1, cmp2, cmp3,
                coeff_[i].template to<ArgType>(),
                coeff_[i+1].template to<ArgType>(),
                coeff_[i+2].template to<ArgType>()));
        }
        return result;
    }

};

//
//
// CUBIC APPROXIMATIONS: very fast, very smooth, but not accurate at all.
// Suitable for musical use in envelope generators etc

template <typename VecType> struct FloatBits {};

template <> struct FloatBits<Vec_ps> {
    static constexpr int32_t exp_shift = 23, exp_mask = 0xff, exp_bias = 127,
        mant_mask = 0x807fffff, exp1 = 0x3f800000, exph = 0x3f000000;
};
template <> struct FloatBits<Vec_ss> : public FloatBits<Vec_ps> {};

template <> struct FloatBits<Vec_pd> {
    static constexpr int32_t exp_shift = 52, exp_bias = 1023;
    static constexpr int64_t exp_mask = 0x7ff, mant_mask = 0x800fffffffffffff,
        exp1 = 0x3ff0000000000000, exph = 0x3fe0000000000000;
};
template <> struct FloatBits<Vec_sd> : public FloatBits<Vec_pd> {};

// These implementations of ldexp and frexp only work for finite, non-denormal
// inputs! Not comparable to standard library versions!
template <typename VecType>
inline VecType sg_ldexp(const VecType& x, const typename VecType::fast_int_t& e) {
    static_assert(VecType::is_float_t, "");
    using equiv_int = typename VecType::equiv_int_t;
    using fb = FloatBits<VecType>;
    return (x.template bitcast<equiv_int>() +
            e.template to<equiv_int>().template shift_l_imm<fb::exp_shift>())
        .template bitcast<VecType>();
}

template <typename VecType>
inline typename VecType::fast_int_t exponent(const VecType& x) {
    static_assert(VecType::is_float_t, "");
    using equiv_int = typename VecType::equiv_int_t;
    using fast_int = typename VecType::fast_int_t;
    using fb = FloatBits<VecType>;
    return ((x.template bitcast<equiv_int>()
            .template shift_rl_imm<fb::exp_shift>()
        & fb::exp_mask) - fb::exp_bias).template to<fast_int>();
}

template <typename VecType>
inline typename VecType::fast_int_t exponent_frexp(const VecType& x) {
    static_assert(VecType::is_float_t, "");
    using equiv_int = typename VecType::equiv_int_t;
    using fast_int = typename VecType::fast_int_t;
    using fb = FloatBits<VecType>;
    return ((x.template bitcast<equiv_int>()
            .template shift_rl_imm<fb::exp_shift>()
        & fb::exp_mask) - (fb::exp_bias-1)).template to<fast_int>();
}

template <typename VecType>
inline VecType mantissa(const VecType& x) {
    static_assert(VecType::is_float_t, "");
    using equiv_int = typename VecType::equiv_int_t;
    using fb = FloatBits<VecType>;
    return ((x.template bitcast<equiv_int>() & fb::mant_mask) | fb::exp1)
        .template bitcast<VecType>();
}

template <typename VecType>
inline VecType mantissa_frexp(const VecType& x) {
    static_assert(VecType::is_float_t, "");
    using equiv_int = typename VecType::equiv_int_t;
    using fb = FloatBits<VecType>;
    return ((x.template bitcast<equiv_int>() & fb::mant_mask) | fb::exph)
        .template bitcast<VecType>();
}

// Calculate log2(x) for x in [1, 2) using a cubic approximation.
// Gradient matches gradient of log2() at either end
static const Poly<Vec_sd, 4> log2_p3_poly {
 1.6404256133344508e-1,
-1.0988652862227437,
 3.1482979293341158,
-2.2134752044448169 };

// Calculate exp2(x) for x in [0, 1]. Gradient matches exp2() at both ends
static const Poly<Vec_sd, 4> exp2_p3_poly {
7.944154167983597e-2,
2.2741127776021886e-1,
6.931471805599453e-1,
1.0 };

} // namespace sg_math_impl

template <typename VecType>
inline VecType log2_p3(const VecType& x) {
    VecType exponent = sg_math_impl::exponent(x).template to<VecType>(),
        mantissa = sg_math_impl::mantissa(x);
    mantissa = sg_math_impl::log2_p3_poly.eval(mantissa);
    return (x > 0.0).choose(exponent + mantissa, VecType::minus_infinity());
}

template <typename VecType>
inline VecType exp2_p3(const VecType& x) {
    const auto floor = x.template floor<typename VecType::fast_int_t>();
    const VecType floor_f = floor.template to<VecType>();
    VecType frac = x - floor_f;
    frac = sg_math_impl::exp2_p3_poly.eval(frac);
    return sg_math_impl::sg_ldexp(frac, floor);
}

template <typename VecType>
inline VecType exp_p3(const VecType& x) {
    using elem = typename VecType::elem_t;
    return exp2_p3(VecType{x * elem{6.931471805599453e-1}});
}

//
//
// CEPHES MATH LIBRARY 32-BIT FLOAT IMPLEMENTATIONS

namespace sg_math_impl {

static constexpr double sqrt_half = 7.07106781186547524401e-1;

// logf and expf constants
static constexpr double log_q1 = -2.12194440e-4,
    log_q2 = 6.93359375e-1,
    log2e = 1.44269504088896341; // log2(e)

static const Poly<Vec_ss, 9> logf_poly {
 7.0376836292e-2f,
-1.1514610310e-1f,
 1.1676998740e-1f,
-1.2420140846e-1f,
 1.4249322787e-1f,
-1.6668057665e-1f,
 2.0000714765e-1f,
-2.4999993993e-1f,
 3.3333331174e-1f };
static const Poly<Vec_ss, 6> expf_poly {
1.9875691500e-4f,
1.3981999507e-3f,
8.3334519073e-3f,
4.1665795894e-2f,
1.6666665459e-1f,
5.0000001201e-1f };

template <typename VecType>
inline VecType logf_impl(const VecType& a) {
    VecType x = mantissa_frexp(a);
    VecType e = exponent_frexp(a).template to<VecType>();
    auto x_lt_sqrth = x < static_cast<float>(sqrt_half);
    e -= x_lt_sqrth.choose_else_zero(1.0f);
    x += x_lt_sqrth.choose_else_zero(x);
    x -= 1.0f;

    VecType z = x * x;

    VecType y = logf_poly.eval(x) * x * z;

    y += e*static_cast<float>(log_q1);
    y += -0.5f * z;
    z = x + y;
    z += e*static_cast<float>(log_q2);

    return (a > 0.0f).choose(z, VecType::minus_infinity());
}

} // namespace sg_math_impl

inline Vec_ss logf_cm(const Vec_ss& a) { return sg_math_impl::logf_impl(a); }
inline Vec_ps logf_cm(const Vec_ps& a) { return sg_math_impl::logf_impl(a); }
inline Vec_ss log_cm(const Vec_ss& a) { return sg_math_impl::logf_impl(a); }
inline Vec_ps log_cm(const Vec_ps& a) { return sg_math_impl::logf_impl(a); }

namespace sg_math_impl {

template <typename VecType>
inline VecType expf_impl(const VecType& a) {
    VecType x = a;
    VecType z = x * static_cast<float>(log2e);
    auto n = z.template nearest<typename VecType::equiv_int_t>();
    z = n.template to<VecType>();

    x -= z*static_cast<float>(log_q2) + z*static_cast<float>(log_q1);
    z = x * x;
    z *= expf_poly.eval(x);
    z += x + 1.0;
    return sg_ldexp(z, n);
}

} // namespace sg_math_impl

inline Vec_ss expf_cm(const Vec_ss& a) { return sg_math_impl::expf_impl(a); }
inline Vec_ps expf_cm(const Vec_ps& a) { return sg_math_impl::expf_impl(a); }
inline Vec_ss exp_cm(const Vec_ss& a) { return sg_math_impl::expf_impl(a); }
inline Vec_ps exp_cm(const Vec_ps& a) { return sg_math_impl::expf_impl(a); }

namespace sg_math_impl {

// sincos constants
static constexpr double four_over_pi = 1.2732395447351628;
static constexpr float dp1_f = 7.8515625e-1f;
static constexpr float dp2_f = 2.4187564849853515625e-4f;
static constexpr float dp3_f = 3.77489497744594108e-8f;

static const Poly<Vec_ss, 3> sinf_poly {
-1.9515295891e-4f,
 8.3321608736e-3f,
-1.6666654611e-1f };
static const Poly<Vec_ss, 3> cosf_poly {
 2.443315711809948e-5f,
-1.388731625493765e-3f,
 4.166664568298827e-2f };

// {cos, sin}
static const Poly<Vec_ps, 3> sincosf_poly { cosf_poly, sinf_poly };

} // namespace sg_math_impl

template <typename VecType>
struct sincos_result { VecType sin_result, cos_result; };

// sin and cos for f32 break when x >= 8192
inline sincos_result<Vec_ss> sincosf_cm(const Vec_ss& xx) {
    // {cos sign bit, sin sign bit}
    Vec_ps signbits {0.0f, 0.0f, 0.0f, (xx & -0.0f).data()};
    float x = xx.abs().data();
    int32_t j = static_cast<int32_t>(x *
        static_cast<float>(sg_math_impl::four_over_pi));
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
    x = ((x - y * sg_math_impl::dp1_f) - y * sg_math_impl::dp2_f) -
        y * sg_math_impl::dp3_f;
    const float z = x * x;
    // From here, calculate both {cos, sin} results in parallel
    Vec_ps result = sg_math_impl::sincosf_poly.eval(Vec_ps{0.0f, 0.0f, z, z})
        * z;
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
    Vec_pi32 j = (x * static_cast<float>(sg_math_impl::four_over_pi))
        .truncate<Vec_pi32>();
    Vec_ps y = j.to<Vec_ps>();
    const Compare_pi32 j_odd {(j & 1) != 0};
    j += j_odd.choose_else_zero(1);
    y += j_odd.to<Compare_ps>().choose_else_zero(1.0f);
    j &= 7;
    const Compare_pi32 j_gt_3 { j > 3 };
    j -= j_gt_3.choose_else_zero(4);
    cos_signbit ^= j_gt_3.to<Compare_ps>().choose_else_zero(-0.0f);
    sin_signbit ^= j_gt_3.to<Compare_ps>().choose_else_zero(-0.0f);
    const Compare_ps j_gt_1 = (j > 1).to<Compare_ps>();
    cos_signbit ^= j_gt_1.choose_else_zero(-0.0f);
    x = ((x - y * sg_math_impl::dp1_f) - y * sg_math_impl::dp2_f) -
        y * sg_math_impl::dp3_f;
    const Vec_ps z = x * x;
    // Brackets on following line needed for identical scalar / vec behaviour
    const Vec_ps cos_result = sg_math_impl::cosf_poly.eval(z) * z * z +
            (1.0f - 0.5f*z),
        sin_result = sg_math_impl::sinf_poly.eval(z) * z * x + x;
    const Compare_ps swap = ((j == 1) || (j == 2)).to<Compare_ps>();
    sincos_result<Vec_ps> result;
    result.cos_result = swap.choose(sin_result, cos_result) ^ cos_signbit;
    result.sin_result = swap.choose(cos_result, sin_result) ^ sin_signbit;
    return result;
}

inline Vec_ps sinf_cm(const Vec_ps& x) { return sincosf_cm(x).sin_result; }
inline Vec_ps cosf_cm(const Vec_ps& x) { return sincosf_cm(x).cos_result; }

inline sincos_result<Vec_ss> sincos_cm(const Vec_ss& xx) {
    return sincosf_cm(xx);
}
inline sincos_result<Vec_ps> sincos_cm(const Vec_ps& xx) {
    return sincosf_cm(xx);
}
inline Vec_ss sin_cm(const Vec_ss& x) { return sincosf_cm(x).sin_result; }
inline Vec_ps sin_cm(const Vec_ps& x) { return sincosf_cm(x).sin_result; }
inline Vec_ss cos_cm(const Vec_ss& x) { return sincosf_cm(x).cos_result; }
inline Vec_ps cos_cm(const Vec_ps& x) { return sincosf_cm(x).cos_result; }

namespace sg_math_impl {

static constexpr double sqrt_2 = 1.4142135623730951;

static const Poly<Vec_ss, 7> sqrtf_poly1 {
-9.8843065718e-4f,
 7.9479950957e-4f,
-3.5890535377e-3f,
 1.1028809744e-2f,
-4.4195203560e-2f,
 3.5355338194e-1f,
 1.41421356237f };
static const Poly<Vec_ss, 6> sqrtf_poly2 {
 1.35199291026e-2f,
-2.26657767832e-2f,
 2.78720776889e-2f,
-3.89582788321e-2f,
 6.24811144548e-2f,
-1.25001503933e-1f };
static const Poly<Vec_ss, 7> sqrtf_poly3 {
-3.9495006054e-1f,
 5.1743034569e-1f,
-4.3214437330e-1f,
 3.5310730460e-1f,
-3.5354581892e-1f,
 7.0710676017e-1f,
 7.07106781187e-1f };
static const Poly_choose3<Vec_ss, 7> sqrtf_choose3{ sqrtf_poly1,
    sqrtf_poly2.prepend(0.0f), sqrtf_poly3 };

} // namespace sg_math_impl

inline Vec_ss sqrtf_cm(const Vec_ss& a) {
    if (a.data() <= 0.0f) return 0.0f;
    auto e = sg_math_impl::exponent_frexp(a);
    Vec_ss x = sg_math_impl::mantissa_frexp(a);
    if (e.data() & 1) {
        --e;
        x += x;
    }
    e = e.shift_ra_imm<1>();
    Vec_ss y;
    if (x.data() > static_cast<float>(sg_math_impl::sqrt_2)) {
        x -= 2.0f;
        y = sg_math_impl::sqrtf_poly1.eval(x);
    } else if (x.data() > static_cast<float>(sg_math_impl::sqrt_half)) {
        x -= 1.0f;
        // Brackets needed to match output of vector version
        y = sg_math_impl::sqrtf_poly2.eval(x) * (x * x) + (0.5f*x + 1.0f);
    } else {
        x -= 0.5f;
        y = sg_math_impl::sqrtf_poly3.eval(x);
    }
    return sg_math_impl::sg_ldexp(y, e);
}

inline Vec_ps sqrtf_cm(const Vec_ps& a) {
    Vec_pi32 e = sg_math_impl::exponent_frexp(a);
    Vec_ps x = sg_math_impl::mantissa_frexp(a);
    const Compare_pi32 e_odd { (e & 1) != 0 };
    e -= e_odd.choose_else_zero(1);
    x += e_odd.to<Compare_ps>().choose_else_zero(x);
    e = e.shift_ra_imm<1>();
    Vec_ps y;
    const Compare_ps x_gt_sqrt2 {x > static_cast<float>(sg_math_impl::sqrt_2)};
    Compare_ps x_gt_sqrth { x > static_cast<float>(sg_math_impl::sqrt_half) },
        x_else;
    sg_math_impl::make_if_elseif_else(x_gt_sqrt2, x_gt_sqrth, x_else);
    x -= sg_math_impl::choose3(x_gt_sqrt2, x_gt_sqrth, x_else,
        Vec_ps{2.0f}, Vec_ps{1.0f}, Vec_ps{0.5f});
    y = sg_math_impl::sqrtf_choose3.eval(x_gt_sqrt2, x_gt_sqrth, x_else, x);
    y *= x_gt_sqrth.choose(x * x, 1.0f);
    y += x_gt_sqrth.choose_else_zero(0.5f*x + 1.0f);
    return (a > 0.0f).choose_else_zero(sg_math_impl::sg_ldexp(y, e));
}

inline Vec_ss sqrt_cm(const Vec_ss& a) { return sqrtf_cm(a); }
inline Vec_ps sqrt_cm(const Vec_ps& a) { return sqrtf_cm(a); }

//
//
// CEPHES 64-BIT IMPLEMENTATIONS

namespace sg_math_impl {

static constexpr double log_c1 = 2.121944400546905827679e-4,
    log_c2 = 6.93359375e-1;

static const Poly<Vec_sd, 3> log_poly_R {
-7.89580278884799154124e-1,
 1.63866645699558079767e1,
 -6.41409952958715622951e1 };
static const Poly<Vec_sd, 3> log_poly_S {
// 1.0,
-3.56722798256324312549e1,
 3.12093766372244180303e2,
-7.69691943550460008604e2 };
static const Poly<Vec_pd, 4> log_poly_R_S {
    log_poly_R.prepend(0.0), log_poly_S.prepend(1.0)
};

} // namespace sg_math_impl

inline Vec_sd log_cm(const Vec_sd& a) {
    if (a.data() <= 0.0) return sg_minus_infinity_f64x1;
    double x = sg_math_impl::mantissa_frexp(a).data();
    int32_t e = sg_math_impl::exponent_frexp(a).data();
    double y, z;
    if (x < sg_math_impl::sqrt_half) {
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
    Vec_pd  poly_eval = sg_math_impl::log_poly_R_S.eval(Vec_pd{z});
    z = x * ((z * poly_eval.d1()) / poly_eval.d0());
    const double e_double = static_cast<double>(e);
    y = e_double;
    z -= y * sg_math_impl::log_c1;
    z += x;
    z += e_double * sg_math_impl::log_c2;
    return z;
}

inline Vec_pd log_cm(const Vec_pd& a) {
    Vec_pd x = sg_math_impl::mantissa_frexp(a);
    auto e = sg_math_impl::exponent_frexp(a);
    Vec_pd y, z;
    Compare_pd x_lt_sqrth {x < sg_math_impl::sqrt_half};
    e -= x_lt_sqrth.to<Vec_pd::fast_int_t::compare_t>().choose_else_zero(1);
    z = x - x_lt_sqrth.choose(0.5, 1.0);
    y = 0.5 * x_lt_sqrth.choose(z, x) + 0.5;
    x = z / y;
    z = x * x;
    Vec_pd R = sg_math_impl::log_poly_R.eval(Vec_pd{z}),
        S = sg_math_impl::log_poly_S.eval1(Vec_pd{z});
    z = x * ((z * R) / S);
    const Vec_pd e_pd {e.to<Vec_pd>()};
    y = e_pd;
    z -= y * sg_math_impl::log_c1;
    z += x;
    z += e_pd * sg_math_impl::log_c2;
    return (a > 0.0).choose(z, Vec_pd::minus_infinity());
}

namespace sg_math_impl {

static const Poly<Vec_sd, 3>exp_poly_P {
1.26177193074810590878e-4,
3.02994407707441961300e-2,
9.99999999999999999910e-1 };
static const Poly<Vec_sd, 4>exp_poly_Q {
3.00198505138664455042e-6,
2.52448340349684104192e-3,
2.27265548208155028766e-1,
2.00000000000000000009e0 };
static const Poly<Vec_pd, 4>exp_poly_P_Q {
    exp_poly_P.prepend(0.0), exp_poly_Q
};

static constexpr double exp_c1 = 6.93145751953125e-1,
    exp_c2 = 1.42860682030941723212e-6;

} // namespace sg_math_impl

inline Vec_sd exp_cm(const Vec_sd& a) {
    Vec_sd x = a.data();
    auto n = (x * sg_math_impl::log2e).nearest<Vec_sd::fast_int_t>();
    Vec_sd px = n.to<Vec_f64x1>();
    x -= px*sg_math_impl::exp_c1 + px*sg_math_impl::exp_c2;
    Vec_sd xx = x * x;
    // {P, Q}
    const Vec_pd poly = Vec_pd{x.data(), 1.0} *
        sg_math_impl::exp_poly_P_Q.eval(xx.to<Vec_pd>());
    const Vec_sd P = poly.d1(), Q = poly.d0();
    x = P / (Q - P);
    x = 2.0*x + 1.0;
    return sg_math_impl::sg_ldexp(x, n);
}

inline Vec_pd exp_cm(const Vec_pd& a) {
    Vec_pd x = a;
    auto n = (x * sg_math_impl::log2e).nearest<Vec_pd::fast_int_t>();
    Vec_pd px = n.to<Vec_pd>();
    x -= px*sg_math_impl::exp_c1 + px*sg_math_impl::exp_c2;
    Vec_pd xx = x * x;
    Vec_pd P = x * sg_math_impl::exp_poly_P.eval(xx),
        Q = sg_math_impl::exp_poly_Q.eval(xx);
    x = P / (Q - P);
    x = 2.0*x + 1.0;
    return sg_math_impl::sg_ldexp(x, n);
}

namespace sg_math_impl {

static const Poly<Vec_sd, 6> sin_poly {
 1.58962301576546568060e-10,
-2.50507477628578072866e-8,
 2.75573136213857245213e-6,
-1.98412698295895385996e-4,
 8.33333333332211858878e-3,
-1.66666666666666307295e-1 };
static const Poly<Vec_sd, 6> cos_poly {
-1.13585365213876817300e-11,
 2.08757008419747316778e-9,
-2.75573141792967388112e-7,
 2.48015872888517045348e-5,
-1.38888888888730564116e-3,
 4.16666666666665929218e-2 };
static const Poly<Vec_pd, 6> cossin_poly { cos_poly, sin_poly };

static constexpr double dp1 = 7.85398125648498535156e-1;
static constexpr double dp2 = 3.77489470793079817668e-8;
static constexpr double dp3 = 2.69515142907905952645e-15;

} // namespace sg_math_impl

inline sincos_result<Vec_sd> sincos_cm(const Vec_sd& a) {
    // { cos sign bit, sin sign bit }
    Vec_pd signbits { 0.0, (a & -0.0).data() };
    double x = a.abs().data();
    double y = Vec_sd{x * sg_math_impl::four_over_pi}
        .floor<Vec_sd::fast_int_t>().to<Vec_f64x1>().data();
    double z = Vec_sd{y * 0.0625}.floor<Vec_sd::fast_int_t>()
        .to<Vec_f64x1>().data();
    z = y - 16.0*z;
    int32_t j = static_cast<int32_t>(z);
    if (j & 1) {
        ++j;
        y += 1.0;
    }
    j &= 7;
    if (j > 3) {
        signbits ^= Vec_pd{-0.0};
        j -= 4;
    }
    if (j > 1) signbits ^= Vec_pd{-0.0, 0.0};
    z = ((x - sg_math_impl::dp1*y) - sg_math_impl::dp2*y)
        - sg_math_impl::dp3*y;
    const double zz = z * z;

    Vec_pd cossin_poly = Vec_pd{1.0 - 0.5*zz, z} + Vec_pd{zz, z} *
        zz * sg_math_impl::cossin_poly.eval(Vec_pd{zz});
    if ((j == 1) || (j == 2)) cossin_poly = cossin_poly.shuffle<0, 1>();
    cossin_poly ^= signbits;

    sincos_result<Vec_sd> result;
    result.sin_result = cossin_poly.d0();
    result.cos_result = cossin_poly.d1();
    return result;
}

inline Vec_sd sin_cm(const Vec_sd& a) { return sincos_cm(a).sin_result; }
inline Vec_sd cos_cm(const Vec_sd& a) { return sincos_cm(a).cos_result; }

inline sincos_result<Vec_pd> sincos_cm(const Vec_pd& a) {
    Vec_pd cos_signbits {0.0}, sin_signbits{a & -0.0};
    Vec_pd x = a.abs();
    Vec_pd y = (x * sg_math_impl::four_over_pi)
        .floor<Vec_pd::fast_int_t>().to<Vec_pd>();
    Vec_pd z = (y * 0.0625).floor<Vec_pd::fast_int_t>().to<Vec_pd>();
    z = y - 16.0*z;
    auto j = z.truncate<Vec_pd::fast_int_t>();
    const auto j_odd {(j & 1) != 0};
    j += j_odd.choose_else_zero(1);
    y += j_odd.to<Compare_pd>().choose_else_zero(1.0);
    j &= 7;
    const auto j_gt_3 {j > 3};
    j -= j_gt_3.choose_else_zero(4);
    const Compare_pd j_gt_3_pd = j_gt_3.to<Compare_pd>();
    cos_signbits ^= j_gt_3_pd.choose_else_zero(-0.0);
    sin_signbits ^= j_gt_3_pd.choose_else_zero(-0.0);
    const auto j_gt_1 {j > 1};
    cos_signbits ^= j_gt_1.to<Compare_pd>().choose_else_zero(-0.0);
    z = ((x - sg_math_impl::dp1*y) - sg_math_impl::dp2*y)
        - sg_math_impl::dp3*y;
    const Vec_pd zz = z * z;
    const Vec_pd sin_result = z + z*zz*sg_math_impl::sin_poly.eval(zz);
    const Vec_pd cos_result = 1.0 - 0.5*zz +
        zz*zz*sg_math_impl::cos_poly.eval(zz);
    const Compare_pd swap_results = ((j == 1) || (j == 2)).to<Compare_pd>();
    sincos_result<Vec_pd> result;
    result.sin_result = swap_results.choose(cos_result, sin_result)
        ^ sin_signbits;
    result.cos_result = swap_results.choose(sin_result, cos_result)
        ^ cos_signbits;

    return result;
}

inline Vec_pd sin_cm(const Vec_pd& a) { return sincos_cm(a).sin_result; }
inline Vec_pd cos_cm(const Vec_pd& a) { return sincos_cm(a).cos_result; }

namespace sg_math_impl {

template <typename VecType>
VecType sqrt_impl(const VecType& a) {
    VecType x = a;
    const VecType w = x;
    const VecType z = mantissa_frexp(a);
    const auto e = exponent_frexp(a);
    x = 4.173075996388649989089e-1 + 5.9016206709064458299663e-1*z;
    x *= ((e & 1) != 0)
        .template to<typename VecType::compare_t>()
        .choose(sqrt_2, 1.0);
    x = sg_ldexp(x, e.template shift_ra_imm<1>());
    x = 0.5*(x + w/x);
    x = 0.5*(x + w/x);
    x = 0.5*(x + w/x);
    return x;
}

} // namespace sg_math_impl

inline Vec_pd sqrt_cm(const Vec_pd& a) { return sg_math_impl::sqrt_impl(a); }
inline Vec_sd sqrt_cm(const Vec_sd& a) { return sg_math_impl::sqrt_impl(a); }

} // namespace simd_granodi
