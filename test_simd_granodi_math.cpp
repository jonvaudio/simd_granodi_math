#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include "simd_granodi_math.h"

using namespace simd_granodi;

void test_func(const int start, const int stop, const double scale,
    const std::string filename_pd, const std::string filename_ps,
    const std::function<Vec_pd(const Vec_pd&)> func_pd,
    const std::function<Vec_ps(const Vec_ps&)> func_ps)
{
    FILE *output_ps = fopen(filename_ps.data(), "w"),
        *output_pd = fopen(filename_pd.data(), "w");
    for (int i = start; i < stop; ++i) {
        const double xd = static_cast<double>(i) * scale;
        const float xf = static_cast<float>(xd);
        fprintf(output_pd, "%.7f\n", func_pd(Vec_pd{xd}).d0());
        fprintf(output_ps, "%.7f\n", func_ps(Vec_ps{xf}).f0());
    }
    fclose(output_ps); fclose(output_pd);
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

    test_func(0, 20000, 0.01, file_prefix + "log2_p3_pd_test.txt",
        file_prefix + "log2_p3_ps_test.txt", log2_p3_pd, log2_p3_ps);

    test_func(-2000, 2000, 0.01, file_prefix + "exp2_p3_pd_test.txt",
        file_prefix + "exp2_p3_ps_test.txt", exp2_p3_pd, exp2_p3_ps);

    test_func(0, 20000, 0.01, file_prefix + "logf_cm_pd_test.txt",
        file_prefix + "logf_cm_ps_test.txt", logf_cm_pd, logf_cm_ps);

    test_func(-2000, 2000, 0.01, file_prefix + "expf_cm_pd_test.txt",
        file_prefix + "expf_cm_ps_test.txt", expf_cm_pd, expf_cm_ps);

    test_func(-800, 800, 0.01, file_prefix + "sinf_cm_pd_test.txt",
        file_prefix + "sinf_cm_ps_test.txt", sinf_cm_pd, sinf_cm_ps);

    test_func(-800, 800, 0.01, file_prefix + "cosf_cm_pd_test.txt",
        file_prefix + "cosf_cm_ps_test.txt", cosf_cm_pd, cosf_cm_ps);

    return 0;
}
