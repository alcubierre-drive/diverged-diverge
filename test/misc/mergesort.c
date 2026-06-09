#define CT_SUITE_NAME parsort
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_common.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/misc/mergesort.h"
#include "../helper_functions.h"

typedef struct {
    index_t idx;
    double val;
    index_t ary_pos; // needed for qsort
} testdata_t;

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) (x.idx < y.idx ? -1 : \
                                x.idx > y.idx ? 1 : 0)
MERGESORT_IMPL( sort_idx, testdata_t )

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) (x.val < y.val ? -1 : \
                                x.val > y.val ? 1 : 0)
MERGESORT_IMPL( sort_val, testdata_t )

static testdata_t* gen_random_data( index_t size/*, index_t nidx=20*/ ) {
    index_t nidx = 10;
    testdata_t* data = calloc( size, sizeof*data );
    for (index_t i=0; i<size; ++i)
        data[i].idx = random_num()%nidx,
        data[i].val = random_dbl(),
        data[i].ary_pos = i;
    return data;
}

static testdata_t* copy_random_data( const testdata_t* data, index_t size ) {
    testdata_t* copy = calloc( size, sizeof*copy );
    memcpy( copy, data, sizeof*copy * size );
    return copy;
}

static index_t* gen_indices( index_t N ) {
    index_t* ii = calloc(N, sizeof*ii);
    for (index_t i=0; i<N; ++i) ii[i] = i;
    return ii;
}

static inline int testdata_val_cmp( const void* vA, const void* vB ) {
    const testdata_t *A = vA, *B = vB;
    if (A->val < B->val) return -1;
    if (A->val > B->val) return 1;
    return A->ary_pos - B->ary_pos;
}
static inline int testdata_idx_cmp( const void* vA, const void* vB ) {
    const testdata_t *A = vA, *B = vB;
    if (A->idx == B->idx) return A->ary_pos - B->ary_pos;
    else return A->idx - B->idx;
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
    for (index_t i=0; i<N; ++i) s[i].ary_pos = i; // set array positions to simulate stable sort
    qsort( s, N, sizeof*s, &testdata_val_cmp );
    for (index_t i=0; i<N; ++i) s[i].ary_pos = i; // set array positions to simulate stable sort
    qsort( s, N, sizeof*s, &testdata_idx_cmp );
    tock = diverge_mpi_wtime();
    double tim_stdsort = tock-tick;

    mpi_tim_printf( "mergesort: %.3fs, qsort: %.3fs\n", tim_mergesort, tim_stdsort );
    index_t equiv = 0; for (index_t i=0; i<N; ++i) equiv += (s[i].val == c[i].val && s[i].idx == c[i].idx);
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

typedef struct { index_t i; index_t ary_pos; const testdata_t* t; } qsort_compat_layer_t;
static inline int qsort_compat_layer_val_cmp( const void* vA, const void* vB ) {
    const qsort_compat_layer_t *A = vA, *B = vB;
    if (A->t[A->i].val < B->t[B->i].val) return -1;
    if (A->t[A->i].val > B->t[B->i].val) return 1;
    return A->ary_pos - B->ary_pos;
}
static inline int qsort_compat_layer_idx_cmp( const void* vA, const void* vB ) {
    const qsort_compat_layer_t *A = vA, *B = vB;
    if (A->t[A->i].idx == B->t[B->i].idx) return A->ary_pos - B->ary_pos;
    else return A->t[A->i].idx - B->t[B->i].idx;
}

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

    qsort_compat_layer_t* qs = calloc(N, sizeof*qs);
    for (index_t i=0; i<N; ++i) qs[i] = (qsort_compat_layer_t){ i, i, d };
    qsort( qs, N, sizeof*qs, &qsort_compat_layer_val_cmp );
    for (index_t i=0; i<N; ++i) qs[i].ary_pos = i; // set array positions to simulate stable sort
    qsort( qs, N, sizeof*qs, &qsort_compat_layer_idx_cmp );
    tock = diverge_mpi_wtime();
    double tim_stdsort = tock-tick;

    mpi_tim_printf( "mergesort: %.3fs, qsort: %.3fs\n", tim_mergesort, tim_stdsort );
    index_t equiv = 0; for (index_t i=0; i<N; ++i) equiv += (qs[i].i == c[i]);
    CT_CHECK( equiv == N );

    free(c);
    free(s);
    free(d);
}

#endif
