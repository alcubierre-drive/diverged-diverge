#define CT_SUITE_NAME common
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"
#include "../../src/misc/mt19937.h"

typedef struct {
    int i1, i2, i3;
    double d1, d2, d3;
    float f1, f2, f3;
    void *ptr;
} swap_t;

CT_TEST(swap) {
    mt19937_64_t rng = mt64_init(12345);
    swap_t A = {0}, B = {0};
    int ndbl = sizeof(swap_t) / sizeof(double) + 1;
    int64_t rng_ary[ndbl];
    for (int i=0; i<ndbl; ++i) rng_ary[i] = mt64_next(&rng);
    memcpy(&A, rng_ary, sizeof(A));
    for (int i=0; i<ndbl; ++i) rng_ary[i] = mt64_next(&rng);
    memcpy(&B, rng_ary, sizeof(B));

    swap_t A_copy = {0};
    memcpy(&A_copy, &A, sizeof(A));
    swap_t B_copy = {0};
    memcpy(&B_copy, &B, sizeof(B));

    MEMSWAP(A, B);
    swap_t A_copy_2 = MEMSWAP_RET(A, B);

    int A_eq = memcmp(&A, &A_copy, sizeof(A));
    int B_eq = memcmp(&B, &B_copy, sizeof(B));
    int A_eq_2 = memcmp(&A, &A_copy_2, sizeof(A));

    CT_CHECK(A_eq == 0);
    CT_CHECK(B_eq == 0);
    CT_CHECK(A_eq_2 == 0);

    int i1 = A.i1, i2 = A.i2;
    int i2_cpy = XORSWAP(A.i1, A.i2);

    CT_CHECK(i1 == A.i2);
    CT_CHECK(i2 == A.i1);
    CT_CHECK(i2 == i2_cpy);
}

#endif
