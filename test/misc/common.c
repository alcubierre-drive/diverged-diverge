#define CT_SUITE_NAME common
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"
#include "../../src/misc/mt19937.h"
#include "../../src/misc/bitary.h"
#include "../../src/diverge_model.h"
#include "../../src/misc/qnan.h"
#include "../../src/misc/hashtable.h"
#include "../../src/misc/dv_vector.h"
#include <time.h>
#include <limits.h>
#include <omp.h>

typedef struct {
    int i1, i2, i3;
    double d1, d2, d3;
    float f1, f2, f3;
    void *ptr;
} swap_t;

CT_TEST_NAME(swap, "xorswap,memswap") {
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

CT_TEST(modelswap) {
    diverge_model_t *m1 = diverge_model_init(),
                    *m2 = diverge_model_init();

    void *p1 = m1->internals,
         *p2 = m2->internals;

    MEMSWAP(*m1, *m2);
    CT_CHECK( m1->internals == p2 );
    CT_CHECK( m2->internals == p1 );

    MEMSWAP(m1, m2);
    CT_CHECK( m1->internals == p1 );
    CT_CHECK( m2->internals == p2 );

    XORSWAP_PTR(m1, m2);
    CT_CHECK( m1->internals == p2 );
    CT_CHECK( m2->internals == p1 );

    diverge_model_t m3 = MEMSWAP_RET(*m1, *m2);
    CT_CHECK( m1->internals == p1 );
    CT_CHECK( m2->internals == p2 );
    CT_CHECK( m1->internals == m3.internals );

    diverge_model_free(m1);
    diverge_model_free(m2);
}

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

CT_TEST_NAME_HIDDEN(bitary_speed, "bitary,speed") {
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

static void print_bits_64( const char* prefix, uint64_t x ) {
    char output[64+32] = {0};
    int output_idx = 0;
    for (int i = sizeof(x) * CHAR_BIT - 1; i >= 0; i--) {
        output[output_idx++] = (x & (1ULL << i)) ? '1' : '0';
        if (i % CHAR_BIT == 0) output[output_idx++] = ' ';
    }
    mpi_log_printf( "%s64bit pattern:\n%s\n", prefix, output );
}

static void print_bits_32( const char* prefix, uint32_t x ) {
    char output[32+32] = {0};
    int output_idx = 0;
    for (int i = sizeof(x) * CHAR_BIT - 1; i >= 0; i--) {
        output[output_idx++] = (x & (1U << i)) ? '1' : '0';
        if (i % CHAR_BIT == 0) output[output_idx++] = ' ';
    }
    mpi_log_printf( "%s32bit pattern:\n%s\n", prefix, output );
}

CT_TEST_NAME( qnan_int, "qnan,int" ) {
    uint32_t payload = clock();
    double qnan_with_payload = qnan_gen( payload );
    double regular_number = payload;

    uint64_t qnan_bits = 0; memcpy( &qnan_bits, &qnan_with_payload, sizeof qnan_bits );
    print_bits_64( "qnan ", qnan_bits );

    mpi_log_printf( "quiet NaN (%.1f) unsigned payload=%lu\n", qnan_with_payload, payload );
    CT_CHECK( qnan_isnan( qnan_with_payload ) );
    CT_CHECK( !qnan_isnan( regular_number ) );
    CT_CHECK( qnan_get( qnan_with_payload ) == payload );
}

CT_TEST_NAME( qnan_float, "qnan,float" ) {
    double payload = -(double)clock() / 1234.;
    double fnan_with_payload = fnan_gen( payload );
    double regular_number = payload;

    uint64_t fnan_bits = 0; memcpy( &fnan_bits, &fnan_with_payload, sizeof fnan_bits );
    print_bits_64( "fnan ", fnan_bits );

    mpi_log_printf( "quiet NaN (%.1f) floating point payload=%.5e\n", fnan_with_payload, payload );
    CT_CHECK( fnan_isnan( fnan_with_payload ) );
    CT_CHECK( !fnan_isnan( regular_number ) );

    double payload_trunc = fnan_get(fnan_with_payload);
    mpi_log_printf( "truncated vs full payload: %.18e %.18e\n", payload, payload_trunc );
    double denom = fabs(payload);
    if (denom < 1.e-2) denom = 1.;
    CT_CHECK( fabs(payload_trunc-payload)/denom < 1.e-1 );
}

CT_TEST_NAME( qnan_int32, "qnan,int32" ) {
    uint16_t payload = clock();
    float qnan_with_payload = qnanf_gen( payload );
    float regular_number = payload;

    uint32_t qnan_bits = 0; memcpy( &qnan_bits, &qnan_with_payload, sizeof qnan_bits );
    print_bits_32( "qnan ", qnan_bits );

    mpi_log_printf( "quiet NaNf (%.1f) unsigned payload=%u\n", qnan_with_payload, payload );
    CT_CHECK( qnanf_isnan( qnan_with_payload ) );
    CT_CHECK( !qnanf_isnan( regular_number ) );
    CT_CHECK( qnanf_get( qnan_with_payload ) == payload );
}

CT_TEST_NAME( qnan_float32, "qnan,float32" ) {
    float payload = -(float)clock() / 1234.;
    float fnan_with_payload = fnanf_gen( payload );
    float regular_number = payload;

    uint32_t fnan_bits = 0; memcpy( &fnan_bits, &fnan_with_payload, sizeof fnan_bits );
    print_bits_32( "fnan ", fnan_bits );

    mpi_log_printf( "quiet NaNf (%.1f) floating point payload=%.5e\n", fnan_with_payload, payload );
    CT_CHECK( fnanf_isnan( fnan_with_payload ) );
    CT_CHECK( !fnanf_isnan( regular_number ) );

    float payload_trunc = fnanf_get(fnan_with_payload);
    mpi_log_printf( "truncated vs full payload: %.18e %.18e\n", payload, payload_trunc );
    float denom = fabsf(payload);
    if (denom < 1.e-2) denom = 1.;
    CT_CHECK( fabsf(payload_trunc-payload)/denom < 1.e-1 );
}

CT_TEST(rng_histogram) {
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

CT_TEST(rng_histogram_dbl) {
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

typedef void* void_ptr;
#define vtree_key_t index_t
#define vtree_val_t void_ptr
#include "../../src/misc/vtree.h"

CT_TEST_NAME( dict_vtree_hashtable, "dict,vtree,hashtable" ) {
    xvtree_t* xv = xvtree_init( 16 );
    void_ptr ht_empty = (void_ptr)-1;
    hashtable_t* ht = hashtable_init_fg( 16, 2, (index_t)ht_empty );

    index_t n_insert = 1024;
    index_t max_val = 64;
    for (index_t i=0; i<n_insert; ++i) {
        index_t key = rand() % max_val;
        void_ptr val = (void_ptr)i;
        xvtree_find_or_insert( xv, key )->val = val;
        hashtable_insert_safe( ht, key, val );
    }

    // iterate, but without function
    for (vtree_node_t* xn=vtree_node_minimum(xv->head); xn!=NULL; xn=vtree_node_next(xn)) {
        void* ht_val = hashtable_find(ht, xn->key);
        CT_CHECK( ht_val != ht_empty );
        CT_CHECK( ht_val == xn->val );
    }

    xvtree_free(xv);
    hashtable_free(ht);
}

CT_TEST(dv_vector_init_and_basic_properties) {
    int *v = NULL;

    CT_CHECK(dv_vec_sz(v) == 0);
    CT_CHECK(dv_vec_cap(v) == 0);

    dv_vec_init(v);

    CT_CHECK(dv_vec_cap(v) > 0);
    CT_CHECK(dv_vec_sz(v) == 0);

    dv_vec_free(v);
}

CT_TEST(dv_vector_push_elements) {
    int *v = NULL;

    for (int i=0; i<100; ++i) dv_vec_push(v, i);

    CT_CHECK(dv_vec_sz(v) == 100);
    for (int i=0; i<100; ++i) CT_CHECK(v[i] == i);

    dv_vec_free(v);
}

CT_TEST(dv_vector_automatic_growth) {
    int *v = NULL;

    int initial_cap = DV_VEC_INITIAL_CAP;

    for (int i=0; i<initial_cap+10; ++i) dv_vec_push(v, i);

    CT_CHECK(dv_vec_cap(v) >= (initial_cap * 2));
    CT_CHECK(dv_vec_sz(v) == (initial_cap + 10));

    dv_vec_free(v);
}

CT_TEST(dv_vector_reserve_and_resize) {
    int *v = NULL;

    dv_vec_reserve(v, 50);

    CT_CHECK(dv_vec_cap(v) == 50);

    dv_vec_resize(v, 20);

    CT_CHECK(dv_vec_sz(v) == 20);
    CT_CHECK(dv_vec_cap(v) == 20);

    dv_vec_free(v);
}

CT_TEST(dv_vector_resize_lazy) {
    int *v = NULL;

    dv_vec_reserve(v, 100);
    dv_vec_resize_lazy(v, 10);

    CT_CHECK(dv_vec_sz(v) == 10);
    CT_CHECK(dv_vec_cap(v) == 100);

    dv_vec_free(v);
}

CT_TEST(dv_vector_shrink) {
    int *v = NULL;

    for (int i = 0; i < 50; ++i) {
        dv_vec_push(v, i);
    }

    dv_vec_resize_lazy(v, 10);
    dv_vec_shrink(v);

    CT_CHECK(dv_vec_sz(v) == 10);
    CT_CHECK(dv_vec_cap(v) == 10);

    dv_vec_free(v);
}

CT_TEST(dv_vector_reserve_zero_frees) {
    int *v = NULL;

    dv_vec_push(v, 1);
    CT_CHECK(v != NULL);

    dv_vec_reserve(v, 0);
    CT_CHECK(v == NULL);
}

CT_TEST(dv_vector_multiple_types) {
    double *vd = NULL;

    for (int i=0; i<10; ++i) dv_vec_push(vd, (double)i * 0.5);

    CT_CHECK(dv_vec_sz(vd) == 10);
    for (int i=0; i<10; ++i) CT_CHECK(vd[i] == (double)i * 0.5);

    dv_vec_free(vd);
}

CT_TEST(dv_vector_pointer_stability_after_realloc) {
    int *v = NULL;

    for (int i=0; i<1000; ++i) dv_vec_push(v, i);

    CT_CHECK(dv_vec_sz(v) == 1000);
    for (int i=0; i<1000; ++i) CT_CHECK(v[i] == i);

    dv_vec_free(v);
}

/*
// TODO
VECTOR_DECL
VECTOR_DECL_INIT_CAP
VECTOR_INIT_CAP
VECTOR_DECL_INIT
VECTOR_INIT
VECTOR_PUSH_BACK
VECTOR_RESERVE
VECTOR_RESIZE
VECTOR_SHRINK_TO_FIT
VECTOR_FREE
VECTOR_COPY
*/


#endif
