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
inline float std_expf(const float x) { return std::exp(x); }

inline float std_sinf(const float x) { return std::sin(x); }
inline float std_cosf(const float x) { return std::cos(x); }
inline Vec_ps sinf_cm_ps(const Vec_ps& x) { return sinf_cm(x); }
inline Vec_ss sinf_cm_ss(const Vec_ss& x) { return sinf_cm(x); }
inline Vec_ps cosf_cm_ps(const Vec_ps& x) { return cosf_cm(x); }
inline Vec_ss cosf_cm_ss(const Vec_ss& x) { return cosf_cm(x); }

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
            printf("Big problem\n");
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

    func_csv<Vec_ps, Vec_ss, float>(0.0, 20.0, 0.001,
        file_prefix + "log2_p3",
        std_log2f, log2_p3<Vec_ps>, log2_p3<Vec_ss>);

    func_csv<Vec_ps, Vec_ss, float>(-20.0, 20.0, 0.001,
        file_prefix + "exp2_p3",
        std_exp2f, exp2_p3<Vec_ps>, exp2_p3<Vec_ss>);

    func_csv<Vec_ps, Vec_ss, float>(0.0, 20.0, 0.0001,
        file_prefix + "logf_cm",
        std_logf, logf_cm<Vec_ps>, logf_cm<Vec_ss>);

    func_csv<Vec_ps, Vec_ss, float>(-20.0, 20.0, 0.001,
        file_prefix + "expf_cm",
        std_expf, expf_cm<Vec_ps>, expf_cm<Vec_ss>);

    func_csv<Vec_ps, Vec_ss, float>(-8.0, 8.0, 0.001,
        file_prefix + "sinf_cm",
        std_sinf, sinf_cm_ps, sinf_cm_ss);

    func_csv<Vec_ps, Vec_ss, float>(-8.0, 8.0, 0.001,
        file_prefix + "cosf_cm",
        std_cosf, cosf_cm_ps, cosf_cm_ss);

    func_csv<Vec_pd, Vec_sd, double>(0.0, 20.0, 0.001,
        file_prefix + "log_cm",
        std_log, log_cm_pd, log_cm_sd);

    func_csv<Vec_pd, Vec_sd, double>(-20.0, 20.0, 0.001,
        file_prefix + "exp_cm",
        std_exp, exp_cm_pd, exp_cm_sd);

    func_csv<Vec_pd, Vec_sd, double>(-8.0, 8.0, 0.001,
        file_prefix + "sin_cm",
        std_sin, sin_cm_pd, sin_cm_sd);

    func_csv<Vec_pd, Vec_sd, double>(-8.0, 8.0, 0.001,
        file_prefix + "cos_cm",
        std_cos, cos_cm_pd, cos_cm_sd);

    return 0;
}
