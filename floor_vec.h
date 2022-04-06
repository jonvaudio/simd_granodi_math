#pragma once

#include "simd_granodi.h"

using namespace simd_granodi;

struct floor_result_pd {
    Vec_pd floor_pd;
    Vec_pi32 floor_pi32;
};

struct floor_result_ps {
    Vec_ps floor_ps;
    Vec_pi32 floor_pi32;
};

// Round towards minus infinity
inline floor_result_pd floor_pd(const Vec_pd& x) {
    // We use pi32 here as truncating to pi64 is slow on SSE2 and we want cross
    // platform consistency too
    floor_result_pd result;
    result.floor_pi32 = x.truncate_to_pi32();
    result.floor_pd = result.floor_pi32.convert_to_pd();
    Compare_pd sub_one { result.floor_pd > x };
    result.floor_pd -= sub_one.choose_else_zero(1.0);
    result.floor_pi32 -= sub_one.convert_to_cmp_pi32().choose_else_zero(1);
    return result;
}

inline floor_result_ps floor_ps(const Vec_ps& x) {
    floor_result_ps result;
    result.floor_pi32 = x.truncate_to_pi32();
    result.floor_ps = result.floor_pi32.convert_to_ps();
    Compare_ps sub_one { result.floor_ps > x };
    result.floor_ps -= sub_one.choose_else_zero(1.0f);
    result.floor_pi32 -= sub_one.bitcast_to_cmp_pi32().choose_else_zero(1);
    return result;
}
