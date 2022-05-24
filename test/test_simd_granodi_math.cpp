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
        const auto x = static_cast<typename VecType::elem_t>(
                start + (static_cast<double>(i) / (stop-start))),
            ref_result = func_ref(x),
            scalar_result = func_scalar(typename VecType::scalar_t{x}).data();
        const VecType vec_result = func_vec(VecType{x});
        // Assert all elements of the vector are the same
        assert(vec_result.debug_eq(vec_result.template get<0>()));
        // Check errors are the same
        if (std::isfinite(ref_result) != std::isfinite(scalar_result) ||
            std::isfinite(ref_result) != std::isfinite(vec_result.template get<0>()) ||
            std::isfinite(scalar_result) != std::isfinite(vec_result.template get<0>())) {
            printf("Big problem, ref, scalar, vec: %.4f, %.4f, %.4f\n",
                ref_result, scalar_result, vec_result.template get<0>());
            return;
        }
        if (std::isfinite(ref_result)) {
            // Calculate any scalar / vector discrepancy
            discrep_max = std::max(discrep_max, static_cast<double>(
                std::abs(scalar_result - vec_result.template get<0>())));
            const double re = relative_error(ref_result, scalar_result).data(),
                re_discrep = relative_error(scalar_result, vec_result.template get<0>()).data();
            error_total += re * re;
            if (std::abs(re) > std::abs(error_max)) error_max = re;
            if (std::abs(re_discrep) > std::abs(discrep_max)) discrep_max = re_discrep;
        }
    }
    const double error_avg = std::sqrt(error_total /
        static_cast<double>(trials));
    printf("%s in [%.1e, %.1e] \trms: %.1e\tmax: %.1e\tdiscrep: %.1e\n", filename.data(),
        start, stop, error_avg, error_max, discrep_max);
}

template <typename VecType, typename ScalarType, typename FloatType>
void func_csv(const double start, const double stop, const double interval,
    const std::string filename,
    const std::function<FloatType(FloatType)> func_ref,
    const std::function<VecType(const VecType&)> func_vec,
    const std::function<ScalarType(const ScalarType&)> func_scalar)
{
    static_assert(VecType::elem_count == 4 || VecType::elem_count == 2,
        "VecType must have 2 or 4 elements");
    static_assert(ScalarType::elem_count == 1, "Scalar type must be scalar");
    static_assert(sizeof(FloatType) == ScalarType::elem_size,
        "Wrong float type");
    #ifdef NDEBUG
    FILE *output = fopen((filename + ".csv").data(), "w");
    fprintf(output, "reference,result,relative error\n,,\n");
    #endif
    double diff_max = 0.0, diff_total = 0.0, result_count = 0.0;
    for (double i = start; i <= stop; i += interval, result_count += 1.0) {
        const double xd = static_cast<double>(i);
        const FloatType x = static_cast<FloatType>(xd),
            ref_result = func_ref(x),
            scalar_result = func_scalar(ScalarType{x}).data();
        const VecType vec_result = func_vec(VecType{x});
        (void) vec_result; // NDEBUG builds warn vec_result not used

        if (!vec_result.debug_eq(scalar_result)) {
            printf("%s scalar, vec discrepancy: %.20f, %.20f\n",
                filename.data(), scalar_result, vec_result.template get<0>());
        }

        #ifdef NDEBUG
        if (VecType::elem_size == 4) {
            fprintf(output, "%.9f,%.9f,", ref_result, scalar_result);
        }
        else fprintf(output, "%.15f,%.15f,", ref_result, scalar_result);
        #endif
        if (std::isfinite(ref_result) != std::isfinite(scalar_result)) {
            printf("Big problem: %.4f, %.4f\n", ref_result, scalar_result);
        }
        if (std::isfinite(ref_result) && std::isfinite(scalar_result)) {
            const double re = relative_error(ref_result, scalar_result)
                .data();
            diff_total += re * re;
            if (std::abs(re) > std::abs(diff_max)) diff_max = re;
            #ifdef NDEBUG
            if (VecType::elem_size == 4) fprintf(output, "%.9e", re);
            else fprintf(output, "%.15e", re);
            #endif
        }
        #ifdef NDEBUG
        fprintf(output, "\n");
        #endif
    }
    #ifdef NDEBUG
    fclose(output);
    #endif
    // Check accuracy on optimized builds as results may be different due to
    // FMA etc
    //#ifdef NDEBUG
    const double diff_avg = std::sqrt(diff_total / result_count);
    printf("%s\trms error: %.1e\tmax error: %.1e\n", filename.data(),
        diff_avg, diff_max);
    //#endif
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
    #else
    file_prefix += "_dbg/";
    #endif

    static constexpr double scale = 0.001, log_begin = 0.0, log_end = 20.0,
        exp_begin = -20.0, exp_end = 20.0,
        sincos_begin = -16.0, sincos_end = 16.0,
        sqrt_begin = 0.0, sqrt_end = 20.0;

    static constexpr int64_t num_trials_base = 100000,
        #ifdef NDEBUG
        num_trials = num_trials_base * 100;
        #else
        num_trials = num_trials_base;
        #endif

    static constexpr double log_start_1 = 0.0, log_end_1 = 2.0,
        log_start_2 = 1.0, log_end_2 = 1e9;

    func_error<Vec_ps>(log_start_1, log_end_1, num_trials,
        "log2_p3", std_log2f, log2_p3<Vec_ps>, log2_p3<Vec_ss>);
    func_error<Vec_ps>(log_start_2, log_end_2, num_trials,
        "log2_p3", std_log2f, log2_p3<Vec_ps>, log2_p3<Vec_ss>);

    printf("\n");

    func_error<Vec_ps>(log_start_1, log_end_1, num_trials,
        "logf_cm", std_logf, logf_cm_ps, logf_cm_ss);
    func_error<Vec_ps>(log_start_2, log_end_2, num_trials,
        "logf_cm", std_logf, logf_cm_ps, logf_cm_ss);

    printf("\n");

    /*func_csv<Vec_ps, Vec_ss, float>(log_begin, log_end, scale,
        file_prefix + "log2_p3",
        std_log2f, log2_p3<Vec_ps>, log2_p3<Vec_ss>);

    func_csv<Vec_ps, Vec_ss, float>(exp_begin, exp_end, scale,
        file_prefix + "exp2_p3",
        std_exp2f, exp2_p3<Vec_ps>, exp2_p3<Vec_ss>);

    func_csv<Vec_ps, Vec_ss, float>(log_begin, log_end, scale,
        file_prefix + "logf_cm",
        std_logf, logf_cm_ps, logf_cm_ss);

    func_csv<Vec_ps, Vec_ss, float>(exp_begin, exp_end, scale,
        file_prefix + "expf_cm",
        std_expf, expf_cm_ps, expf_cm_ss);

    func_csv<Vec_ps, Vec_ss, float>(sincos_begin, sincos_end, scale,
        file_prefix + "sinf_cm",
        std_sinf, sinf_cm_ps, sinf_cm_ss);

    func_csv<Vec_ps, Vec_ss, float>(sincos_begin, sincos_end, scale,
        file_prefix + "cosf_cm",
        std_cosf, cosf_cm_ps, cosf_cm_ss);

    func_csv<Vec_ps, Vec_ss, float>(sqrt_begin, sqrt_end, scale,
        file_prefix + "sqrtf_cm",
        std_sqrtf, sqrtf_cm_ps, sqrtf_cm_ss);

    func_csv<Vec_pd, Vec_sd, double>(log_begin, log_end, scale,
        file_prefix + "log_cm",
        std_log, log_cm_pd, log_cm_sd);

    func_csv<Vec_pd, Vec_sd, double>(exp_begin, exp_end, scale,
        file_prefix + "exp_cm",
        std_exp, exp_cm_pd, exp_cm_sd);

    func_csv<Vec_pd, Vec_sd, double>(sincos_begin, sincos_end, scale,
        file_prefix + "sin_cm",
        std_sin, sin_cm_pd, sin_cm_sd);

    func_csv<Vec_pd, Vec_sd, double>(sincos_begin, sincos_end, scale,
        file_prefix + "cos_cm",
        std_cos, cos_cm_pd, cos_cm_sd);

    func_csv<Vec_pd, Vec_sd, double>(sqrt_begin, sqrt_end, scale,
        file_prefix + "sqrt_cm",
        std_sqrt, sqrt_cm_pd, sqrt_cm_sd);*/

    return 0;
}
