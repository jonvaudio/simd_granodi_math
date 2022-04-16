#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include "simd_granodi_math.h"

using namespace simd_granodi;

void test_func(const int start, const int stop, const double scale,
    const std::string filename,
    const std::function<Vec_ps(const Vec_ps&)> func_ps,
    const std::function<Vec_f32x1(const Vec_f32x1&)> func_ss,
    const std::function<Vec_pd(const Vec_pd&)> func_pd,
    const std::function<Vec_f64x1(const Vec_f64x1&)> func_sd)
{
    FILE *output = fopen((filename + ".csv").data(), "w");
    for (int i = start; i < stop; ++i) {
        const double xd = static_cast<double>(i) * scale;
        const float xf = static_cast<float>(xd);
        fprintf(output, "%.9f,", func_ps(Vec_ps{xf}).f0());
        fprintf(output, "%.9f,", float{func_ss(xf)});
        fprintf(output, "%.9f,", func_pd(Vec_pd{xd}).d0());
        fprintf(output, "%.9f\n", double{func_sd(xd)});
    }
    fclose(output);
}

// These are defined to get past problems with ambiguous overloads in the test
// code
inline Vec_ps log2_p3_ps(const Vec_ps& x) { return log2_p3(x); }
inline Vec_f32x1 log2_p3_ss(const Vec_f32x1& x) { return log2_p3(x); }
inline Vec_pd log2_p3_pd(const Vec_pd& x) { return log2_p3(x); }
inline Vec_f64x1 log2_p3_sd(const Vec_f64x1& x) { return log2_p3(x); }

inline Vec_ps exp2_p3_ps(const Vec_ps& x) { return exp2_p3(x); }
inline Vec_f32x1 exp2_p3_ss(const Vec_f32x1& x) { return exp2_p3(x); }
inline Vec_pd exp2_p3_pd(const Vec_pd& x) { return exp2_p3(x); }
inline Vec_f64x1 exp2_p3_sd(const Vec_f64x1& x) { return exp2_p3(x); }

inline Vec_ps logf_cm_ps(const Vec_ps& x) { return logf_cm(x); }
inline Vec_f32x1 logf_cm_ss(const Vec_f32x1& x) { return logf_cm(x); }
inline Vec_pd logf_cm_pd(const Vec_pd& x) { return logf_cm(x); }
inline Vec_f64x1 logf_cm_sd(const Vec_f64x1& x) { return logf_cm(x); }

inline Vec_ps expf_cm_ps(const Vec_ps& x) { return expf_cm(x); }
inline Vec_f32x1 expf_cm_ss(const Vec_f32x1& x) { return expf_cm(x); }
inline Vec_pd expf_cm_pd(const Vec_pd& x) { return expf_cm(x); }
inline Vec_f64x1 expf_cm_sd(const Vec_f64x1& x) { return expf_cm(x); }

inline Vec_ps sinf_cm_ps(const Vec_ps& x) { return sinf_cm(x); }
inline Vec_f32x1 sinf_cm_ss(const Vec_f32x1& x) { return sinf_cm(x); }
inline Vec_pd sinf_cm_pd(const Vec_pd& x) { return sinf_cm(x); }
inline Vec_f64x1 sinf_cm_sd(const Vec_f64x1& x) { return sinf_cm(x); }

inline Vec_ps cosf_cm_ps(const Vec_ps& x) { return cosf_cm(x); }
inline Vec_f32x1 cosf_cm_ss(const Vec_f32x1& x) { return cosf_cm(x); }
inline Vec_pd cosf_cm_pd(const Vec_pd& x) { return cosf_cm(x); }
inline Vec_f64x1 cosf_cm_sd(const Vec_f64x1& x) { return cosf_cm(x); }

int main() {
    std::string file_prefix;
    #ifdef SIMD_GRANODI_SSE2
    file_prefix += "sse2_tests";
    #elif defined SIMD_GRANODI_NEON
    file_prefix += "neon_tests";
    #endif

    #ifdef TEST_OPT
    file_prefix += "_opt/";
    #else
    file_prefix += "_dbg/";
    #endif

    test_func(0, 20000, 0.01, file_prefix + "log2_p3_test",
        log2_p3_ps, log2_p3_ss, log2_p3_pd, log2_p3_sd);

    test_func(-2000, 2000, 0.01, file_prefix + "exp2_p3_test",
        exp2_p3_ps, exp2_p3_ss, exp2_p3_pd, exp2_p3_sd);

    test_func(0, 20000, 0.01, file_prefix + "logf_cm_test",
        logf_cm_ps, logf_cm_ss, logf_cm_pd, logf_cm_sd);

    test_func(-2000, 2000, 0.01, file_prefix + "expf_cm_test",
        expf_cm_ps, expf_cm_ss, expf_cm_pd, expf_cm_sd);

    test_func(-800, 800, 0.01, file_prefix + "sinf_cm_test",
        sinf_cm_ps, sinf_cm_ss, sinf_cm_pd, sinf_cm_sd);

    test_func(-800, 800, 0.01, file_prefix + "cosf_cm_test",
        cosf_cm_ps, cosf_cm_ss, cosf_cm_pd, cosf_cm_sd);

    return 0;
}
