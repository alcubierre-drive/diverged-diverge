/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef USE_MPI
#include <mpi.h>
#else // USE_MPI
typedef void* MPI_Comm;
#define MPI_IN_PLACE NULL
#endif // USE_MPI

#ifdef __cplusplus
extern "C" {
#endif

// Function: diverge_init
// useful if divERGe is the only MPI component
// call MPI_Init() and set the internal communicator automatically.
void diverge_init( int* pargc, char*** pargv );
// Function: diverge_finalize
// call MPI_Finalize().
void diverge_finalize( void );

// Function: diverge_embed
// useful if divERGe should be called as a library with an existing MPI context
void diverge_embed( MPI_Comm comm );
// Function: diverge_reset
// reset the internal context
void diverge_reset( void );

// Function: diverge_mpi_exit
// call MPI_Finalize() and exit(status);
void diverge_mpi_exit( int status );

// Function: diverge_mpi_wtime
double diverge_mpi_wtime( void );
// Function: diverge_mpi_get_comm
MPI_Comm diverge_mpi_get_comm( void );

// returns array of length diverge_mpi_comm_size() that must be freed manually
index_t* diverge_mpi_distribute( index_t sz );

// actual MPI functions

// Function: diverge_mpi_barrier
void diverge_mpi_barrier( void );
// Function: diverge_mpi_comm_size
int diverge_mpi_comm_size( void );
// Function: diverge_mpi_comm_rank
int diverge_mpi_comm_rank( void );

void diverge_mpi_allreduce_double_max( const void* sendbuf, void* recvbuf, int num );
void diverge_mpi_allreduce_double_sum( const void* sendbuf, void* recvbuf, int num );

void diverge_mpi_allreduce_complex_sum( const void* sendbuf, void* recvbuf, int num );
void diverge_mpi_allreduce_complex_sum_inplace( void* sendrecvbuf, int num );

void diverge_mpi_allgather_int( const int* sendbuf, int* recvbuf, int num );
void diverge_mpi_allgather_index( const void* sendbuf, void* recvbuf, int num );
void diverge_mpi_allgather_double( const void* sendbuf, void* recvbuf, int num );

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

void diverge_mpi_alltoallv_bytes( const void* send, const index_t* sendcounts, const index_t* senddispls,
                                        void* recv, const index_t* recvcounts, const index_t* recvdispls,
                                        index_t numbytes );
void diverge_mpi_alltoallv_complex( const complex128_t* send, const int* sendcounts, const int* senddispls,
                                          complex128_t* recv, const int* recvcounts, const int* recvdispls );

void diverge_mpi_allgatherv( complex128_t* recv, const int* rcnt, const int* rdsp );
void diverge_mpi_allgatherv_index( index_t* recv, const int* rcnt, const int* rdsp );
void diverge_mpi_allgatherv_bytes( void* recv, int elementsz, const int* rcnt, const int* rdsp );
double diverge_mpi_max( double* send );
void diverge_mpi_bcast_bytes( void* inout, int size, int root );
void diverge_mpi_gather_double( const double* send, int scnt, double* recv, int rcnt, int root );

// Function: diverge_omp_num_threads
// get the number of threads to use in diverge's OMP loops
int diverge_omp_num_threads( void );

// Function: diverge_force_thread_limit
// limit the number of threads for *every* OMP/FFTW parallel region; except for
// cases where the number is specifically set higher.
void diverge_force_thread_limit( int nthr );

#ifdef __cplusplus
}
#endif
