#define CT_SUITE_NAME bitary
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/misc/mpi_functions.h"
#include "../../src/misc/bitary.h"
#include "../../src/misc/mt19937.h"

CT_TEST(bitary) {
    index_t size = 1024*8;
    cbytes_t* bits = bitary_alloc(size);
    int* check = calloc(size, sizeof*check);

    mt19937_64_t rng = mt64_init(12345);
    for (index_t i=0; i<1024; ++i) {
        index_t idx = mt64_next(&rng)%size;
        bitary_set(bits, idx);
        check[idx] = 1;
    }
    for (index_t i=0; i<1024; ++i) {
        index_t idx = mt64_next(&rng)%size;
        bitary_unset(bits, idx);
        check[idx] = 0;
    }
    for (index_t i=0; i<1024; ++i) {
        index_t idx = mt64_next(&rng)%size;
        bitary_toggle(bits, idx);
        check[idx] = !check[idx];
    }
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<1024*8; ++i) {
        index_t idx = -1;
        #pragma omp critical
        { idx = mt64_next(&rng)%size; }

        #pragma omp atomic
        BITARY_SET(bits, idx);
        #pragma omp atomic write
        check[idx] = 1;
    }

    index_t nset = 0;
    for (index_t i=0; i<size; ++i) if (bitary_get(bits, i)) nset++;
    mpi_log_printf( "#bits set = %li/%li\n", nset, size );

    for (index_t i=0; i<size; ++i)
        CT_CHECK( !!bitary_get(bits, i) == !!check[i] );

    free(bits);
    free(check);
}

#endif
