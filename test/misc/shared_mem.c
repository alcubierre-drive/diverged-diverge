#define CT_SUITE_NAME shared_mem
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/misc/shared_mem.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"
#include "../../src/diverge_model.h"
#include "../../src/diverge_hack.h"
#include "../../src/diverge_model_internals.h"

#include "../../src/diverge_internals_struct.h" // kdimtot
#include "../models/all.h"

#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/propagator_details.h"
#include "../../src/misc/compilation_status.h"

CT_TEST(exclusive) {
    mpi_log_printf_all( "shared malloc: overall rank %i, shared rank %i, shared sz %i\n",
            diverge_mpi_comm_rank(), shared_malloc_rank(), shared_malloc_size() );

    index_t N = 1024;
    double* local_data = calloc( N, sizeof(double) );
    double* data = shared_malloc( N*sizeof(double) );

    if (shared_exclusive_enter( data ) == 1) {
        for (index_t i=0; i<N; ++i) data[i] = pow(i, 0.5);
    }
    shared_exclusive_wait( data );
    memcpy( local_data, data, sizeof(double)*N );

    double error = 0.0;
    for (index_t i=0; i<N; ++i)
        error += fabs(local_data[i] - pow(i,0.5));

    double error_sum = 0.0;
    diverge_mpi_allreduce_double_sum( &error, &error_sum, 1 );

    CT_CHECK(error_sum < 1.e-11);

    shared_free( data );
    free( local_data );
}

CT_TEST(barrier) {
    index_t N = 1024 * 1024;
    double* local_data = calloc( N, sizeof(double) );
    double* data = shared_malloc( N*sizeof(double) );

    int rank = shared_malloc_rank();
    int size = shared_malloc_size();
    index_t* counts = calloc(size, 2*sizeof(index_t));
    index_t* displs = counts + size;

    for (index_t i=0; i<N; ++i) counts[i%size]++;
    for (int r=1; r<size; ++r) displs[r] = displs[r-1] + counts[r-1];
    index_t count = counts[rank],
            displ = displs[rank];
    free( counts );

    shared_malloc_barrier();
    for (index_t i=0; i<count; ++i)
        data[displ+i] = pow(displ+i, 0.5);
    shared_malloc_barrier();

    memcpy( local_data, data, sizeof(double)*N );

    double error = 0.0;
    for (index_t i=0; i<N; ++i)
        error += fabs(local_data[i] - pow(i,0.5));

    double error_sum = 0.0;
    diverge_mpi_allreduce_double_sum( &error, &error_sum, 1 );

    CT_CHECK(error_sum < 1.e-11);

    shared_free( data );
    free( local_data );
}

CT_TEST(shared_greens) {
    int nk = 128;
    diverge_model_t* m_loc = gen_random( nk, nk );
    diverge_model_t* m_shm = gen_random( nk, nk );
    diverge_model_hack( m_shm, "model_shared_gf", "1" );

    gf_complex_t *g_loc = diverge_model_internals_get_greens( m_loc ),
                 *g_shm = diverge_model_internals_get_greens( m_shm );

    m_loc->gfill( m_loc, 1+0.1*I, g_loc );
    m_shm->gfill( m_shm, 1+0.1*I, g_shm );

    double error = 0.0;
    for (index_t i=0; i<kdimtot(m_loc->nk,m_loc->nkf)*POW2(m_loc->n_orb*m_loc->n_spin)*2; ++i)
        error += cabs( g_loc[i] - g_shm[i] );
    double error_sum = 0.0;
    diverge_mpi_allreduce_double_sum( &error, &error_sum, 1 );

    CT_CHECK(error_sum < 1.e-11);

    diverge_model_free( m_loc );
    diverge_model_free( m_shm );
}

typedef struct {
    diverge_model_t* _mod;
    tu_prop_t* _loop;
} tu_loop_test_t;
static inline tu_loop_test_t tu_loop_test_init( diverge_model_t* mod ) {
    return (tu_loop_test_t){ ._mod=mod, ._loop=tu_prop_init( mod ) };
}
static inline void tu_loop_test_free( tu_loop_test_t t ) { tu_prop_free(t._loop); }
static inline index_t tu_loop_test_size( tu_loop_test_t t ) {
    return POW2(t._loop->n_orbff*POW2(t._mod->n_spin)) * t._loop->nk;
}
static inline void tu_loop_test_calculate( tu_loop_test_t t, complex128_t* buf, complex128_t L ) {
    tu_prop_GF_zero_T( t._loop, L, NULL );
    tu_prop_ph_loop( t._loop, buf, NULL );
}

CT_TEST(shared_greens_loop) {
    // only execute without cuda
    if (diverge_compilation_status_cuda(NULL)) return;
    int nk = 12;
    diverge_model_t *m_loc = gen_random( nk, nk ),
                    *m_shm = gen_random( nk, nk );
    diverge_model_hack( m_shm, "model_shared_gf", "1" );

    diverge_model_internals_tu( m_loc, 1.01 );
    diverge_model_internals_tu( m_shm, 1.01 );

    tu_loop_test_t l_loc = tu_loop_test_init( m_loc );
    tu_loop_test_t l_shm = tu_loop_test_init( m_shm );

    complex128_t *buf_loc = calloc( tu_loop_test_size(l_loc), sizeof(complex128_t) ),
                 *buf_shm = calloc( tu_loop_test_size(l_shm), sizeof(complex128_t) );

    tu_loop_test_calculate( l_loc, buf_loc, I128*0.1 );
    tu_loop_test_calculate( l_shm, buf_shm, I128*0.1 );

    double error = 0.0;
    for (index_t i=0; i<tu_loop_test_size(l_loc); ++i)
        error += cabs( buf_loc[i] - buf_shm[i] );
    double error_sum = 0.0;
    diverge_mpi_allreduce_double_sum( &error, &error_sum, 1 );

    CT_CHECK(error_sum < 1.e-11);

    tu_loop_test_free( l_loc );
    tu_loop_test_free( l_shm );

    diverge_model_free( m_loc );
    diverge_model_free( m_shm );

    free( buf_loc );
    free( buf_shm );
}

#endif // DIVERGE_SKIP_TESTS
