#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../../src/misc/shared_mem.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"

TEST_CASE("MPI shared memory exclusive", "[shared_mem][exclusive]") {
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

    CHECK(error_sum < 1.e-11);

    shared_free( data );
    free( local_data );
}

TEST_CASE("MPI shared memory barrier", "[shared_mem][barrier]") {
    mpi_log_printf_all( "shared malloc: overall rank %i, shared rank %i, shared sz %i\n",
            diverge_mpi_comm_rank(), shared_malloc_rank(), shared_malloc_size() );

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

    CHECK(error_sum < 1.e-11);

    shared_free( data );
    free( local_data );
}

#endif // DIVERGE_SKIP_TESTS
