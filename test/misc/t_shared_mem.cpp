#ifndef DIVERGE_SKIP_TESTS

#include <iostream>

// #include "../catch.hpp"
#define CT_SUITE_NAME shared_mem
#include "../ctest.h"

#include "../../src/misc/shared_mem.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"
#include "../../src/diverge_model.h"
#include "../../src/diverge_hack.h"
#include "../../src/diverge_model_internals.h"

#include "../../src/diverge_internals_struct.h" // kdimtot
#include "../model_examples/model_constructor.hpp"

#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/propagator_details.h"

CT_TEST(exclusive) {
    mpi_log_printf_all( "shared malloc: overall rank %i, shared rank %i, shared sz %i\n",
            diverge_mpi_comm_rank(), shared_malloc_rank(), shared_malloc_size() );

    index_t N = 1024;
    double* local_data = (double*)calloc( N, sizeof(double) );
    double* data = (double*)shared_malloc( N*sizeof(double) );

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
    double* local_data = (double*)calloc( N, sizeof(double) );
    double* data = (double*)shared_malloc( N*sizeof(double) );

    int rank = shared_malloc_rank();
    int size = shared_malloc_size();
    index_t* counts = (index_t*)calloc(size, 2*sizeof(index_t));
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

CT_TEST(greens) {
    int nk = 128;
    diverge_model_t* m_loc = gen_random( nk, nk );
    diverge_model_t* m_shm = gen_random( nk, nk );
    diverge_model_hack( m_shm, "model_shared_gf", "1" );

    gf_complex_t *g_loc = diverge_model_internals_get_greens( m_loc ),
                 *g_shm = diverge_model_internals_get_greens( m_shm );

    m_loc->gfill( m_loc, complex128_t(1,0.1), g_loc );
    m_shm->gfill( m_shm, complex128_t(1,0.1), g_shm );

    double error = 0.0;
    for (index_t i=0; i<kdimtot(m_loc->nk,m_loc->nkf)*POW2(m_loc->n_orb*m_loc->n_spin)*2; ++i)
        error += std::abs( g_loc[i] - g_shm[i] );
    double error_sum = 0.0;
    diverge_mpi_allreduce_double_sum( &error, &error_sum, 1 );

    CT_CHECK(error_sum < 1.e-11);

    diverge_model_free( m_loc );
    diverge_model_free( m_shm );
}

#ifndef USE_CUDA
class tu_loop_test_t {
    public:
        tu_loop_test_t( diverge_model_t* mod ) {
            _mod = mod;
            _loop = tu_prop_init( mod );
        }
        ~tu_loop_test_t() {
            tu_prop_free( _loop );
        }
        index_t size( void ) {
            return POW2(_loop->n_orbff*POW2(_mod->n_spin)) * _loop->nk;
        }
        void calculate( complex128_t* buf, complex128_t L ) {
            tu_prop_GF_zero_T( _loop, L, NULL );
            tu_prop_ph_loop( _loop, buf, NULL );
        }
    private:
        diverge_model_t* _mod;
        tu_prop_t* _loop;
};

CT_TEST(loop) {
    int nk = 12;
    diverge_model_t *m_loc = gen_random( nk, nk ),
                    *m_shm = gen_random( nk, nk );
    diverge_model_hack( m_shm, "model_shared_gf", "1" );

    diverge_model_internals_tu( m_loc, 1.01 );
    diverge_model_internals_tu( m_shm, 1.01 );

    tu_loop_test_t l_loc( m_loc );
    tu_loop_test_t l_shm( m_shm );

    complex128_t *buf_loc = (complex128_t*)calloc( l_loc.size(), sizeof(complex128_t) ),
                 *buf_shm = (complex128_t*)calloc( l_shm.size(), sizeof(complex128_t) );

    l_loc.calculate( buf_loc, I128*0.1 );
    l_shm.calculate( buf_shm, I128*0.1 );

    double error = 0.0;
    for (index_t i=0; i<l_loc.size(); ++i)
        error += std::abs( buf_loc[i] - buf_shm[i] );
    double error_sum = 0.0;
    diverge_mpi_allreduce_double_sum( &error, &error_sum, 1 );

    CT_CHECK(error_sum < 1.e-11);

    diverge_model_free( m_loc );
    diverge_model_free( m_shm );

    free( buf_loc );
    free( buf_shm );
}
#endif // USE_CUDA

#endif // DIVERGE_SKIP_TESTS
