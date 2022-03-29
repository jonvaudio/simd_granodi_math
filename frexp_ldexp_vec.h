#pragma once

#include "simd_granodi.h"

using namespace simd_granodi;

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
