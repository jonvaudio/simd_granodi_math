#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include "../simd_granodi_math.h"

using namespace simd_granodi;

void test_func(const int start, const int stop, const double scale,
    const std::string filename,
    const std::function<Vec_ps(const Vec_ps&)> func_ps,
    const std::function<Vec_f32x1(const Vec_f32x1&)> func_ss,
    const std::function<Vec_pd(const Vec_pd&)> func_pd,
    const std::function<Vec_f64x1(const Vec_f64x1&)> func_sd)
{
    FILE *output = fopen((filename + ".csv").data(), "w");
    for (int32_t i = start; i < stop; ++i) {
        const double xd = static_cast<double>(i) * scale;
        const float xf = static_cast<float>(xd);
        fprintf(output, "%.9f,", func_ps(xf).f0());
        fprintf(output, "%.9f,", func_ss(xf).f0());
        fprintf(output, "%.9f,", func_pd(xd).d0());
        fprintf(output, "%.9f\n", func_sd(xd).d0());
    }
    fclose(output);
}

void test_func(const int start, const int stop, const double scale,
    const std::string filename,
    const std::function<double(const double)> func_d)
{
    FILE *output = fopen((filename + ".csv").data(), "w");
    for (int32_t i = start; i < stop; ++i) {
        const double xd = static_cast<double>(i) * scale;
        fprintf(output, "%.15f\n", func_d(xd));
    }
    fclose(output);
}

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
        log2_p3<Vec_ps>, log2_p3<Vec_ss>, log2_p3<Vec_pd>, log2_p3<Vec_sd>);

    test_func(-2000, 2000, 0.01, file_prefix + "exp2_p3_test",
        exp2_p3<Vec_ps>, exp2_p3<Vec_ss>, exp2_p3<Vec_pd>, exp2_p3<Vec_sd>);

    test_func(0, 20000, 0.01, file_prefix + "logf_cm_test",
        logf_cm<Vec_ps>, logf_cm<Vec_ss>, logf_cm<Vec_pd>, logf_cm<Vec_sd>);

    test_func(-2000, 2000, 0.01, file_prefix + "expf_cm_test",
        expf_cm<Vec_ps>, expf_cm<Vec_ss>, expf_cm<Vec_pd>, expf_cm<Vec_sd>);

    test_func(-800, 800, 0.01, file_prefix + "sinf_cm_test",
        sinf_cm<Vec_ps>, sinf_cm<Vec_ss>, sinf_cm<Vec_pd>, sinf_cm<Vec_sd>);

    test_func(-800, 800, 0.01, file_prefix + "cosf_cm_test",
        cosf_cm<Vec_ps>, cosf_cm<Vec_ss>, cosf_cm<Vec_pd>, cosf_cm<Vec_sd>);

    test_func(0, 20000, 0.01, file_prefix + "logd_cm_test", log_cm);

    return 0;
}
