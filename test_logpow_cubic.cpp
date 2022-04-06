#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include "logpow_cubic.h"
#include "math_cmf.h"

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
    std::string test_dir {"tests/"};

    test_func(0, 20000, 0.01, test_dir + "log2_p3_pd_test.txt",
        test_dir + "log2_p3_ps_test.txt", log2_p3_pd, log2_p3_ps);

    test_func(-20000, 20000, 0.01, test_dir + "exp2_p3_pd_test.txt",
        test_dir + "exp2_p3_ps_test.txt", exp2_p3_pd, exp2_p3_ps);

    test_func(0, 20000, 0.01, test_dir + "logf_cm_pd_test.txt",
        test_dir + "logf_cm_ps_test.txt", logf_cm_pd, logf_cm_ps);

    test_func(0, 20000, 0.01, test_dir + "expf_cm_pd_test.txt",
        test_dir + "expf_cm_ps_test.txt", expf_cm_pd, expf_cm_ps);

    return 0;
}
