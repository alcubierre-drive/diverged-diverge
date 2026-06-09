#define CT_SUITE_NAME rng
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_common.h"
#include "../../src/misc/mt19937.h"

CT_TEST(histogram) {
    const index_t n_num = 1024*1024*4,
                  n_bin = 32;
    mt19937_64_t rng = mt64_init(0);
    index_t* cnt = calloc(n_bin, sizeof*cnt);
    for (index_t i=0; i<n_num; ++i)
        cnt[mt64_next(&rng)%n_bin]++;

    index_t n_per_bin = n_num/n_bin;

    for (index_t b=0; b<n_bin; ++b) {
        double sigma = fabs(1. - (double)cnt[b]/(double)n_per_bin);
        CT_CHECK( sigma < 1.e-2 );
    }
    free(cnt);
}

CT_TEST(histogram_dbl) {
    const index_t n_num = 1024*1024*4,
                  n_bin = 32;
    mt19937_64_t rng = mt64_init(1);
    index_t* cnt = calloc(n_bin, sizeof*cnt);
    for (index_t i=0; i<n_num; ++i) {
        double dval = (double)n_bin * mt64_next_dbl(&rng);
        cnt[(index_t)dval]++;
    }

    index_t n_per_bin = n_num/n_bin;

    for (index_t b=0; b<n_bin; ++b) {
        double sigma = fabs(1. - (double)cnt[b]/(double)n_per_bin);
        CT_CHECK( sigma < 1.e-2 );
    }
    free(cnt);
}

#endif // DIVERGE_SKIP_TESTS
