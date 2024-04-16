/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef USE_MPI
#include <mpi.h>
#else // USE_MPI
/**
 * MPI Communicator, might be typedef'ed to ``void*`` in case MPI is not
 * compiled in
 */
typedef void* MPI_Comm;
#define MPI_IN_PLACE NULL
#endif // USE_MPI

#ifdef __cplusplus
extern "C" {
#endif

/**
 * useful if divERGe is the only MPI component, calls ``MPI_Init();`` and other
 * setup tasks; sets the internal communicator automatically.
 */
void diverge_init( int* pargc, char*** pargv );
/** call ``MPI_Finalize();`` and other cleanup functions */
void diverge_finalize( void );

/** useful if divERGe should be called as a library with an existing MPI context */
void diverge_embed( MPI_Comm comm );
/** reset the internal context, but do not clean up the MPI communicator */
void diverge_reset( void );

/** call ``MPI_Finalize();`` and ``exit(status);`` */
void diverge_mpi_exit( int status );

/** get current walltime in seconds */
double diverge_mpi_wtime( void );
/** get global MPI communicator */
MPI_Comm diverge_mpi_get_comm( void );

/**
 * returns array of length two times :c:func:`diverge_mpi_comm_size` that must
 * be freed manually. First half is the count, second the displ. second half can
 * be safely ignored.
 */
index_t* diverge_mpi_distribute( index_t sz );

/**
 * Actual MPI Functions
 * ====================
 *
 * divERGe wraps several MPI functions, for a full list see the source. They all
 * act on the MPI Communicator set on initialization through
 * :c:func:`diverge_init` or :c:func:`diverge_embed`.
 */

/** wrapped ``MPI_Barrier();`` */
void diverge_mpi_barrier( void );
/** return the number of ranks in the communicator */
int diverge_mpi_comm_size( void );
/** return the rank */
int diverge_mpi_comm_rank( void );

void diverge_mpi_allreduce_int_max_inplace( void* buf, int num );

void diverge_mpi_allreduce_double_max( const void* sendbuf, void* recvbuf, int num );
void diverge_mpi_allreduce_double_sum( const void* sendbuf, void* recvbuf, int num );

void diverge_mpi_allreduce_complex_sum( const void* sendbuf, void* recvbuf, int num );
void diverge_mpi_allreduce_complex_sum_inplace( void* sendrecvbuf, int num );
void diverge_mpi_allreduce_complex_bor_inplace( void* sendrecvbuf, int num );

void diverge_mpi_allgather_int( const int* sendbuf, int* recvbuf, int num );
void diverge_mpi_allgather_index( const void* sendbuf, void* recvbuf, int num );
void diverge_mpi_allgather_double( const void* sendbuf, void* recvbuf, int num );
void diverge_mpi_allgather_index_inplace( void* sendrecvbuf, int num );

void diverge_mpi_send_double( void* data, int num, int destination, int tag );
void diverge_mpi_recv_double( void* data, int num, int source, int tag );

void diverge_mpi_gatherv_index( const void* send, int count, void* recv, int* counts, int* displs, int root );
void diverge_mpi_gatherv_bytes( const void* send, int count, void* recv, int* counts, int* displs, int root );
void diverge_mpi_gatherv_cdoub( const void* send, int count, void* recv, int* counts, int* displs, int root );
void diverge_mpi_write_cdoub_to_file( const char* fname, const void* data, int displ, int count );

index_t diverge_mpi_write_byte_to_file( void* file, const void* data, index_t displ, index_t count);
index_t diverge_mpi_write_byte_to_file_master( void* file, const void* data, index_t displ, index_t count);
void* diverge_mpi_open_file(const char* fname);
void diverge_mpi_close_file(void* file);

void diverge_mpi_alltoall_index( const void* send, void* recv, index_t count );
void diverge_mpi_alltoallv_bytes( const void* send, const index_t* sendcounts, const index_t* senddispls,
                                        void* recv, const index_t* recvcounts, const index_t* recvdispls,
                                        index_t numbytes );
/**
 * this version of alltoallv requires some explanation: since MPI typically
 * only knows 32bit integers for any count/displ variables, we can trick it into
 * operation on larger arrays by making packed datatypes. This is exactly what
 * this function is here for. Note that the other version of alltoallv
 * (undocumented) is restricted to 32bit bytes of communication.
 */
void diverge_mpi_alltoallv_bytes_packed( const void* s, const int* sc, const int* sd,
                                               void* r, const int* rc, const int* rd,
                                               index_t numbytes );
void diverge_mpi_alltoallv_complex( const complex128_t* send, const int* sendcounts, const int* senddispls,
                                          complex128_t* recv, const int* recvcounts, const int* recvdispls );

void diverge_mpi_allgatherv( complex128_t* recv, const int* rcnt, const int* rdsp );
void diverge_mpi_allgatherv_index( index_t* recv, const int* rcnt, const int* rdsp );
void diverge_mpi_allgatherv_bytes( void* recv, int elementsz, const int* rcnt, const int* rdsp );
double diverge_mpi_max( double* send );
void diverge_mpi_bcast_bytes( void* inout, int size, int root );
void diverge_mpi_gather_double( const double* send, int scnt, double* recv, int rcnt, int root );

/**
 * Other miscellaneous (parallel) functions
 * ========================================
 */

/** get the number of threads to use in diverge's OMP loops */
int diverge_omp_num_threads( void );

/**
 * limit the number of threads for *every* OMP/FFTW parallel region; except for
 * cases where the number is specifically set higher.
 */
void diverge_force_thread_limit( int nthr );

#ifdef __cplusplus
}
#endif
