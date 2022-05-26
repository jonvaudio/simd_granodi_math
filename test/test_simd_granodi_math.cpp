#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include "../../jon_dsp/jon_dsp.h"
#include "../simd_granodi_math.h"

using namespace simd_granodi;

inline float std_log2f(const float x) { return std::log2(x); }
inline float std_exp2f(const float x) { return std::exp2(x); }

inline float std_logf(const float x) { return std::log(x); }
inline Vec_ps logf_cm_ps(const Vec_ps& x) { return logf_cm(x); }
inline Vec_ss logf_cm_ss(const Vec_ss& x) { return logf_cm(x); }

inline float std_expf(const float x) { return std::exp(x); }
inline Vec_ps expf_cm_ps(const Vec_ps& x) { return expf_cm(x); }
inline Vec_ss expf_cm_ss(const Vec_ss& x) { return expf_cm(x); }

inline float std_sinf(const float x) { return std::sin(x); }
inline float std_cosf(const float x) { return std::cos(x); }
inline Vec_ps sinf_cm_ps(const Vec_ps& x) { return sinf_cm(x); }
inline Vec_ss sinf_cm_ss(const Vec_ss& x) { return sinf_cm(x); }
inline Vec_ps cosf_cm_ps(const Vec_ps& x) { return cosf_cm(x); }
inline Vec_ss cosf_cm_ss(const Vec_ss& x) { return cosf_cm(x); }

inline float std_sqrtf(const float x) { return std::sqrt(x); }
inline Vec_ps sqrtf_cm_ps(const Vec_ps& x) { return sqrtf_cm(x); }
inline Vec_ss sqrtf_cm_ss(const Vec_ss& x) { return sqrtf_cm(x); }

inline double std_log(const double x) { return std::log(x); }
inline Vec_pd log_cm_pd(const Vec_pd& x) { return log_cm(x); }
inline Vec_sd log_cm_sd(const Vec_sd& x) { return log_cm(x); }

inline double std_exp(const double x) { return std::exp(x); }
inline Vec_pd exp_cm_pd(const Vec_pd& x) { return exp_cm(x); }
inline Vec_sd exp_cm_sd(const Vec_sd& x) { return exp_cm(x); }

inline double std_sin(const double x) { return std::sin(x); }
inline Vec_pd sin_cm_pd(const Vec_pd& x) { return sin_cm(x); }
inline Vec_sd sin_cm_sd(const Vec_sd& x) { return sin_cm(x); }

inline double std_cos(const double x) { return std::cos(x); }
inline Vec_pd cos_cm_pd(const Vec_pd& x) { return cos_cm(x); }
inline Vec_sd cos_cm_sd(const Vec_sd& x) { return cos_cm(x); }

inline double std_sqrt(const double x) { return std::sqrt(x); }
inline Vec_pd sqrt_cm_pd(const Vec_pd& x) { return sqrt_cm(x); }
inline Vec_sd sqrt_cm_sd(const Vec_sd& x) { return sqrt_cm(x); }

inline Vec_sd relative_error(const Vec_sd& reference, const Vec_sd& test) {
    //printf("ref: %.15f, test: %.15f\n", reference.data(), test.data());
    Vec_sd abs_error = test - reference;
    //printf("absolute error: %.15f\n", abs_error.data());
    Vec_sd relative_error;
    if (reference.data() != 0.0) {
        relative_error = abs_error / reference;
    } else {
        relative_error = 0.0;
    }
    //printf("relative error: %.15f\n\n", relative_error.data());
    return relative_error;
}

inline double get_arg(const double start, const double end,
    const int64_t num_trials, const int64_t trial)
{
    const double x = start + (end-start) *
        (static_cast<double>(trial) / static_cast<double>(num_trials));
    //printf("start: %.1f, end: %.1f, num_trials: %d, current_trial: %d, x: %.1f\n",
        //start, end, static_cast<int32_t>(num_trials), static_cast<int32_t>(trial), x);
    return x;
}

template <typename VecType>
void func_error(const double start, const double stop, const int64_t trials,
    const std::string filename,
    const std::function<typename VecType::elem_t(typename VecType::elem_t)> func_ref,
    const std::function<VecType(const VecType&)> func_vec,
    const std::function<typename VecType::scalar_t(const typename VecType::scalar_t&)> func_scalar)
{
    assert(start < stop);
    assert(trials > 0);
    double error_max = 0.0, error_total = 0.0, discrep_max = 0.0;
    for (int64_t i = 0; i <= trials; ++i) {
        const auto x = static_cast<typename VecType::elem_t>(get_arg(start, stop, trials, i)),
            ref_result = func_ref(x),
            scalar_result = func_scalar(typename VecType::scalar_t{x}).data();
        const VecType vec_result = func_vec(VecType{x});
        // Assert all elements of the vector are the same
        assert(vec_result.debug_eq(vec_result.template get<0>()));
        // Check errors are the same
        /*if (std::isfinite(ref_result) != std::isfinite(scalar_result) ||
            std::isfinite(ref_result) != std::isfinite(vec_result.template get<0>()) ||
            std::isfinite(scalar_result) != std::isfinite(vec_result.template get<0>())) {
            printf("Finite disagreement (ref, scalar, vec): %.4f, %.4f, %.4f\n",
                ref_result, scalar_result, vec_result.template get<0>());
            return;
        }*/
        if (std::isfinite(ref_result) && std::isfinite(scalar_result)
            && std::isfinite(vec_result.template get<0>())) {
            // Calculate any scalar / vector discrepancy
            discrep_max = std::max(discrep_max, static_cast<double>(
                std::abs(scalar_result - vec_result.template get<0>())));
            const double re = relative_error(ref_result, scalar_result).data(),
                re_discrep = relative_error(scalar_result, vec_result.template get<0>()).data();
            error_total += re * re;
            if (std::abs(re) > std::abs(error_max)) error_max = re;
            if (std::abs(re_discrep) > std::abs(discrep_max)) discrep_max = re_discrep;

            if (std::abs(re) > 0.9) printf("Large error(x, ref, scalar): %.1e, %.1e, %.1e\n", x, ref_result, scalar_result);
        }
    }
    const double error_avg = std::sqrt(error_total /
        static_cast<double>(trials));
    printf("%s in [%.1e, %.1e] \trms: %.1e\tmax: %.1e\tdiscrep: %.1e\n", filename.data(),
        start, stop, error_avg, error_max, discrep_max);
}

template <typename VecType>
void func_csv(const double start, const double stop, const int64_t trials,
    const std::string filename,
    const std::function<VecType(const VecType&)> func)
{
    (void) start; (void) stop; (void) trials; (void) filename; (void) func;
    #ifdef NDEBUG
    FILE *output = fopen((filename + ".csv").data(), "w");
    for (int64_t i = 0; i <= trials; ++i) {
        const double x = get_arg(start, stop, trials, i);
        const auto result = func(x).template get<0>();
        if (VecType::elem_size == 4) {
            fprintf(output, "%.9f\n", result);
        }
        else {
            fprintf(output, "%.15f\n", result);
        }
    }
    fclose(output);
    #endif
}

int main() {
    std::string file_prefix;
    #ifdef SIMD_GRANODI_SSE2
    file_prefix += "sse2_tests";
    #elif defined SIMD_GRANODI_NEON
    file_prefix += "neon_tests";
    #endif

    #ifdef NDEBUG
    file_prefix += "_opt/";
    printf("OPTIMIZED RESULTS\n");
    #else
    file_prefix += "_dbg/";
    printf("DEBUG RESULTS\n");
    #endif

    jon_dsp::ScopedDenormalDisable sdd;

    static constexpr int64_t num_trials_base = 100000,
        #ifdef NDEBUG
        num_trials = num_trials_base * 100,
        #else
        num_trials = num_trials_base,
        #endif
        num_trials_csv = 1000;

    static constexpr double log_start_1 = 0.0, log_end_1 = 2.0,
        log_start_2 = 1.0, log_end_2 = 1e9,
        log_start_csv = 0.0, log_end_csv = 20.0;

    func_error<Vec_ps>(log_start_1, log_end_1, num_trials,
        "log2_p3", std_log2f, log2_p3<Vec_ps>, log2_p3<Vec_ss>);
    func_error<Vec_ps>(log_start_2, log_end_2, num_trials,
        "log2_p3", std_log2f, log2_p3<Vec_ps>, log2_p3<Vec_ss>);
    func_csv<Vec_ss>(log_start_csv, log_end_csv, num_trials_csv,
        file_prefix + "log2_p3", log2_p3<Vec_ss>);

    printf("\n");

    func_error<Vec_ps>(log_start_1, log_end_1, num_trials,
        "logf_cm", std_logf, logf_cm_ps, logf_cm_ss);
    func_error<Vec_ps>(log_start_2, log_end_2, num_trials,
        "logf_cm", std_logf, logf_cm_ps, logf_cm_ss);
    func_csv<Vec_ss>(log_start_csv, log_end_csv, num_trials_csv,
        file_prefix + "logf_cm", logf_cm_ss);

    printf("\n");

    func_error<Vec_pd>(log_start_1, log_end_1, num_trials,
        "log_cm", std_log, log_cm_pd, log_cm_sd);
    func_error<Vec_pd>(log_start_2, log_end_2, num_trials,
        "log_cm", std_log, log_cm_pd, log_cm_sd);
    func_csv<Vec_sd>(log_start_csv, log_end_csv, num_trials_csv,
        file_prefix + "log_cm", log_cm_sd);

    printf("\n");

    static constexpr double exp_start = -708.0, exp_end = 708.0,
        exp_start_csv = -20.0, exp_end_csv = 20.0;

    func_error<Vec_ps>(exp_start, exp_end, num_trials,
        "exp2_p3", std_exp2f, exp2_p3<Vec_ps>, exp2_p3<Vec_ss>);
    func_csv<Vec_ss>(exp_start_csv, exp_end_csv, num_trials_csv,
        file_prefix + "exp2_p3", exp2_p3<Vec_ss>);

    func_error<Vec_ps>(exp_start, exp_end, num_trials,
        "expf_cm", std_expf, expf_cm_ps, expf_cm_ss);
    func_csv<Vec_ss>(exp_start_csv, exp_end_csv, num_trials_csv,
        file_prefix + "expf_cm", expf_cm_ss);

    func_error<Vec_pd>(exp_start, exp_end, num_trials,
        "exp_cm", std_exp, exp_cm_pd, exp_cm_sd);
    func_csv<Vec_sd>(exp_start_csv, exp_end_csv, num_trials_csv,
        file_prefix + "exp_cm", exp_cm_sd);

    printf("\n");

    static constexpr double sincos_start = -4096.0, sincos_end = 4096.0,
        sincos_start_csv = -7.0, sincos_end_csv = 7.0;

    func_error<Vec_ps>(sincos_start, sincos_end, num_trials,
        "sinf_cm", std_sinf, sinf_cm_ps, sinf_cm_ss);
    func_csv<Vec_ss>(sincos_start_csv, sincos_end_csv, num_trials_csv,
        file_prefix + "sinf_cm", sinf_cm_ss);

    func_error<Vec_pd>(sincos_start, sincos_end, num_trials,
        "sin_cm", std_sin, sin_cm_pd, sin_cm_sd);
    func_csv<Vec_sd>(sincos_start_csv, sincos_end_csv, num_trials_csv,
        file_prefix + "sin_cm", sin_cm_sd);

    printf("\n");

    func_error<Vec_ps>(sincos_start, sincos_end, num_trials,
        "cosf_cm", std_cosf, cosf_cm_ps, cosf_cm_ss);
    func_csv<Vec_ss>(sincos_start_csv, sincos_end_csv, num_trials_csv,
        file_prefix + "cosf_cm", cosf_cm_ss);

    func_error<Vec_pd>(sincos_start, sincos_end, num_trials,
        "cos_cm", std_cos, cos_cm_pd, cos_cm_sd);
    func_csv<Vec_sd>(sincos_start_csv, sincos_end_csv, num_trials_csv,
        file_prefix + "cos_cm", cos_cm_sd);

    printf("\n");

    static constexpr double sqrt_start = 0.0, sqrt_end = 1.0e38,
        sqrt_start_csv = 0.0, sqrt_end_csv = 20.0;

    func_error<Vec_ps>(sqrt_start, sqrt_end, num_trials,
        "sqrtf_cm", std_sqrtf, sqrtf_cm_ps, sqrtf_cm_ss);
    func_csv<Vec_ss>(sqrt_start_csv, sqrt_end_csv, num_trials_csv,
        file_prefix + "sqrtf_cm", sqrtf_cm_ss);

    func_error<Vec_pd>(sqrt_start, sqrt_end, num_trials,
        "sqrt_cm", std_sqrt, sqrt_cm_pd, sqrt_cm_sd);
    func_csv<Vec_sd>(sqrt_start_csv, sqrt_end_csv, num_trials_csv,
        file_prefix + "sqrt_cm", sqrt_cm_sd);

    printf("\n");

    return 0;
}
