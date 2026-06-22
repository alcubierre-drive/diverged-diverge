#define CT_SUITE_NAME bit_arrays
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/misc/mpi_functions.h"
#include "../../src/misc/bitary.h"
#include "../../src/misc/mt19937.h"
#include <omp.h>

CT_TEST(bitary) {
    size_t size = 1024*8 + 1751*7;
    cbytes_t* bits = bitary_alloc(size);
    int* check = calloc(size, sizeof*check);

    mpi_log_printf("random set…\n");
    mt19937_64_t rng = mt64_init(12345);
    for (index_t i=0; i<1024; ++i) {
        index_t idx = mt64_next(&rng)%size;
        bitary_set(bits, idx);
        check[idx] = 1;
    }
    mpi_log_printf("random unset…\n");
    for (index_t i=0; i<1024; ++i) {
        index_t idx = mt64_next(&rng)%size;
        bitary_unset(bits, idx);
        check[idx] = 0;
    }
    mpi_log_printf("random toggle…\n");
    for (index_t i=0; i<1024; ++i) {
        index_t idx = mt64_next(&rng)%size;
        bitary_toggle(bits, idx);
        check[idx] = !check[idx];
    }
    mpi_log_printf("parallel random set…\n");
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

    mpi_log_printf("popcount trivial\n");
    size_t nset = 0;
    for (size_t i=0; i<size; ++i) if (bitary_get(bits, i)) nset++;
    mpi_log_printf("#bits set = %li/%li\n", nset, size);

    mpi_log_printf("popcount func\n");
    size_t nset_cnt = bitary_count(bits, size),
           nset_cnt_par = bitary_count_parallel(bits, size, diverge_omp_num_threads());

    CT_CHECK( nset == nset_cnt );
    CT_CHECK( nset == nset_cnt_par );

    mpi_log_printf("element-wise check\n");
    for (size_t i=0; i<size; ++i)
        CT_CHECK( !!bitary_get(bits, i) == !!check[i] );

    mpi_log_printf("toggle_all\n");
    bitary_toggle_all(bits, size);
    nset_cnt = bitary_count(bits, size);
    nset_cnt_par = bitary_count_parallel(bits, size, diverge_omp_num_threads());
    CT_CHECK( nset_cnt == nset_cnt_par );
    CT_CHECK( nset_cnt == size-nset );

    mpi_log_printf("unset_all\n");
    bitary_unset_all(bits, size);
    nset = bitary_count(bits, size);
    CT_CHECK( nset == 0 );

    mpi_log_printf("set_all\n");
    bitary_set_all(bits, size);
    nset = bitary_count(bits, size);
    CT_CHECK( nset == size );

    bitary_toggle_all(bits, size);
    nset = bitary_count(bits, size);
    CT_CHECK( nset == 0 );

    free(bits);
    free(check);
}

CT_TEST_HIDDEN(speed) {
    index_t size = 1024*1024*512;
    cbytes_t* bits = bitary_alloc(size);
    bool* ibits = calloc(size, sizeof*ibits);

    index_t nset = size/8;
    index_t* indices = calloc(nset, sizeof*indices);
    const int nthr = diverge_omp_num_threads();
    mt19937_64_t rng[nthr];
    for (int t=0; t<nthr; ++t) rng[t] = mt64_init(t);
    #pragma omp parallel for num_threads(nthr)
    for (index_t i=0; i<nset; ++i)
        indices[i] = mt64_next(&rng[omp_get_thread_num()])%size;

    double tick, tock;

    tick = diverge_mpi_wtime();
    for (index_t i=0; i<nset; ++i) {
        bitary_set(bits, indices[i]);
    }
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bitary/serial  : %.3fs\n", tock-tick );

    tick = diverge_mpi_wtime();
    for (index_t i=0; i<nset; ++i) {
        ibits[indices[i]] = true;
    }
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bool  /serial  : %.3fs\n", tock-tick );

    ZERO_OUT(bits, bitary_nwords(size));
    tick = diverge_mpi_wtime();
    #pragma omp parallel for num_threads(nthr)
    for (index_t i=0; i<nset; ++i) {
        #pragma omp atomic update
        BITARY_SET(bits, indices[i]);
    }
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bitary/parallel: %.3fs\n", tock-tick );

    ZERO_OUT(ibits, size);
    tick = diverge_mpi_wtime();
    #pragma omp parallel for num_threads(nthr)
    for (index_t i=0; i<nset; ++i) {
        #pragma omp atomic write
        ibits[indices[i]] = true;
    }
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bool  /parallel: %.3fs\n", tock-tick );

    tick = diverge_mpi_wtime();
    index_t bool_sum = 0;
    for (index_t i=0; i<size; ++i) if (ibits[i]) bool_sum++;
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bool-sum/serial: %.3fs\n", tock-tick );

    tick = diverge_mpi_wtime();
    index_t bit_sum = 0;
    for (index_t i=0; i<size; ++i) if (bitary_get(bits,i)) bit_sum++;
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bit-sum /serial: %.3fs\n", tock-tick );

    tick = diverge_mpi_wtime();
    index_t bit_sum_popcount = bitary_count(bits, size);
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bit-sum /popcnt: %.3fs\n", tock-tick );

    tick = diverge_mpi_wtime();
    index_t bit_sum_popcnt_p = bitary_count_parallel(bits, size, diverge_omp_num_threads());
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bit-sum /pcnpar: %.3fs\n", tock-tick );

    tick = diverge_mpi_wtime();
    index_t bit_sum_popcnt_g = 0;
    for (size_t w=0; w<bitary_nwords(size); ++w) bit_sum_popcnt_g += generic_popcount64(bits[w]);
    tock = diverge_mpi_wtime();
    mpi_tim_printf( "bit-sum /pcngen: %.3fs\n", tock-tick );

    CT_CHECK( bit_sum==bool_sum );
    CT_CHECK( bit_sum==bit_sum_popcount );
    CT_CHECK( bit_sum==bit_sum_popcnt_p );
    CT_CHECK( bit_sum==bit_sum_popcnt_g );

    for (index_t i=0; i<size; ++i) ibits[i] = !ibits[i];
    bitary_toggle_all(bits, size);
    bool_sum = 0;
    for (index_t i=0; i<size; ++i) if (ibits[i]) bool_sum++;
    bit_sum_popcnt_p = bitary_count_parallel(bits, size, diverge_omp_num_threads());
    CT_CHECK( bool_sum==bit_sum_popcnt_p );

    index_t nerr = 0;
    for (index_t i=0; i<size; ++i) {
        bool bb = !!bitary_get(bits, i);
        bool ii = !!ibits[i];
        if (bb != ii) nerr++;
    }
    CT_CHECK(nerr == 0);

    free(indices);
    free(bits);
    free(ibits);
}

#endif
