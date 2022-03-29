#include <cmath>
#include <cstdio>
#include "logpow_cubic.h"

int main() {
    for (int i = 0; i < 10; ++i) {
        const double xd = static_cast<double>(i) * 2.3;
        const float xf = static_cast<float>(xd);
        printf("%f\t%f\n", std::log2(xd), log2_p3(Vec_pd{xd}).d0());
        printf("%f\t%f\n", std::log2(xf), log2_p3(Vec_ps{xf}).f0());
    }

    for (int i = -10; i < 10; ++i) {
        const double xd = static_cast<double>(i) * 2.3;
        const float xf = static_cast<float>(xd);
        printf("%f\t%f\n", std::exp2(xd), exp2_p3(Vec_pd{xd}).d0());
        printf("%f\t%f\n", std::exp2(xf), exp2_p3(Vec_ps{xf}).f0());
    }

    return 0;
}
