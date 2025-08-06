#ifndef DIVERGE_SKIP_TESTS

// #include "../catch.hpp"
#include "../../src/diverge_common.h"
#include "../../src/misc/mpi_functions.h"

#include "../../src/misc/mergesort.h"
#include <algorithm>

#define CT_SUITE_NAME parsort
#include "../ctest.h"

typedef struct {
    index_t idx;
    double val;
} testdata_t;

static inline bool operator==( const testdata_t& A, const testdata_t& B ) {
    return A.idx == B.idx && A.val == B.val;
}

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) (x.idx < y.idx ? -1 : \
                                x.idx > y.idx ? 1 : 0)
MERGESORT_IMPL( sort_idx, testdata_t )

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) (x.val < y.val ? -1 : \
                                x.val > y.val ? 1 : 0)
MERGESORT_IMPL( sort_val, testdata_t )

static testdata_t* gen_random_data( index_t size, index_t nidx=20 ) {
    testdata_t* data = (testdata_t*)calloc( size, sizeof*data );
    for (index_t i=0; i<size; ++i)
        data[i].idx = rand()%nidx,
        data[i].val = (double)rand()/(double)RAND_MAX;
    return data;
}

static testdata_t* copy_random_data( const testdata_t* data, index_t size ) {
    testdata_t* copy = (testdata_t*)calloc( size, sizeof*copy );
    memcpy( copy, data, sizeof*copy * size );
    return copy;
}

static index_t* gen_indices( index_t N ) {
    index_t* ii = (index_t*)calloc(N, sizeof*ii);
    for (index_t i=0; i<N; ++i) ii[i] = i;
    return ii;
}

CT_TEST(direct) {
    index_t N = 1024*128;
    testdata_t* d = gen_random_data( N );
    testdata_t *c = copy_random_data( d, N ), *s = copy_random_data( d, N );
    double tick, tock;

    tick = diverge_mpi_wtime();
    sort_val(c, N, diverge_omp_num_threads(), 128);
    sort_idx(c, N, diverge_omp_num_threads(), 128);
    tock = diverge_mpi_wtime();
    double tim_mergesort = tock-tick;

    tick = diverge_mpi_wtime();
    std::stable_sort(s, s+N, [](const testdata_t&A, const testdata_t&B)->bool{ return A.val < B.val; });
    std::stable_sort(s, s+N, [](const testdata_t&A, const testdata_t&B)->bool{ return A.idx < B.idx; });
    tock = diverge_mpi_wtime();
    double tim_stdsort = tock-tick;

    mpi_tim_printf( "mergesort: %.3fs, std::sort: %.3fs\n", tim_mergesort, tim_stdsort );
    index_t equiv = 0; for (index_t i=0; i<N; ++i) equiv += (s[i] == c[i]);
    CT_CHECK( equiv == N );

    free(c);
    free(s);
    free(d);
}

#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R( x, y, ctx ) (((const testdata_t*)ctx)[x].idx < ((const testdata_t*)ctx)[y].idx ? -1 : \
                                       ((const testdata_t*)ctx)[x].idx > ((const testdata_t*)ctx)[y].idx ? 1 : 0)
MERGESORT_IMPL_R( sort_idx, index_t )

#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R( x, y, ctx ) (((const testdata_t*)ctx)[x].val < ((const testdata_t*)ctx)[y].val ? -1 : \
                                       ((const testdata_t*)ctx)[x].val > ((const testdata_t*)ctx)[y].val ? 1 : 0)
MERGESORT_IMPL_R( sort_val, index_t )

CT_TEST(indirect) {
    index_t N = 1024*128;
    testdata_t* d = gen_random_data( N );
    index_t *c = gen_indices(N), *s = gen_indices(N);
    double tick, tock;

    tick = diverge_mpi_wtime();
    sort_val_r(c, N, d, diverge_omp_num_threads(), 128);
    sort_idx_r(c, N, d, diverge_omp_num_threads(), 128);
    tock = diverge_mpi_wtime();
    double tim_mergesort = tock-tick;

    tick = diverge_mpi_wtime();
    std::stable_sort(s, s+N, [d](const index_t&A, const index_t&B)->bool{ return d[A].val < d[B].val; });
    std::stable_sort(s, s+N, [d](const index_t&A, const index_t&B)->bool{ return d[A].idx < d[B].idx; });
    tock = diverge_mpi_wtime();
    double tim_stdsort = tock-tick;

    mpi_tim_printf( "mergesort: %.3fs, std::sort: %.3fs\n", tim_mergesort, tim_stdsort );
    index_t equiv = 0; for (index_t i=0; i<N; ++i) equiv += (s[i] == c[i]);
    CT_CHECK( equiv == N );

    free(c);
    free(s);
    free(d);
}

#endif
