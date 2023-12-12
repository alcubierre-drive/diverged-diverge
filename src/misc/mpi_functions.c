/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "mpi_functions.h"
#include "init_internal_libs.h"

#include <assert.h>

static MPI_Comm diverge_mpi_comm = NULL;

index_t* diverge_mpi_distribute( index_t sz ) {
    int nranks = diverge_mpi_comm_size();
    index_t* howmany = (index_t*)calloc( nranks, sizeof(index_t) );
    for (index_t s=0; s<sz; ++s) {
        howmany[s%nranks]++;
    }
    return howmany;
}

#ifdef USE_MPI

static int diverge_owns_mpi_context = 0;

#define MPICHK() { \
    if ( diverge_mpi_comm == NULL ) { \
        mpi_err_printf( "MPI context is not initialized. either call diverge_init() or diverge_embed()\n" ); \
        assert( diverge_mpi_comm != NULL ); \
    } \
}

void diverge_init( int* pargc, char*** pargv ) {
    int argc = 1;
    char arg_[] = "diverge";
    char* arg = arg_;
    char** argv = &arg;
    if (pargc == NULL || pargv == NULL) {
        pargc = &argc;
        pargv = &argv;
    }
    int provided = 0;
    MPI_Init_thread( pargc, pargv, MPI_THREAD_SERIALIZED, &provided );
    diverge_mpi_comm = MPI_COMM_WORLD;
    diverge_owns_mpi_context = 1;

    if (provided < MPI_THREAD_SERIALIZED)
        mpi_wrn_printf( "MPI thread implementation is non optimal\n" );

    init_internal_libs();
}

void diverge_finalize( void ) {
    finalize_internal_libs();
    if (!diverge_owns_mpi_context) {
        mpi_err_printf("cannot call MPI_Finalize when divERGe doesnt own the context\n");
        return;
    }
    diverge_owns_mpi_context = 0;
    diverge_mpi_comm = NULL;
    MPI_Finalize();
}

void diverge_embed( MPI_Comm comm ) {
    diverge_mpi_comm = comm;
    init_internal_libs();
}

void diverge_reset( void ) {
    finalize_internal_libs();
    if (diverge_owns_mpi_context) {
        mpi_err_printf("cannot reset owned MPI context\n");
        return;
    }
    diverge_owns_mpi_context = 0;
    diverge_mpi_comm = NULL;
}

double diverge_mpi_wtime() {
    MPICHK();
    return MPI_Wtime();
}

MPI_Comm diverge_mpi_get_comm( void ) {
    MPICHK();
    return diverge_mpi_comm;
}

void diverge_mpi_barrier( void ) {
    MPICHK();
    MPI_Barrier( diverge_mpi_comm );
}

int diverge_mpi_comm_size( void ) {
    MPICHK();
    int size;
    MPI_Comm_size( diverge_mpi_comm, &size );
    return size;
}

int diverge_mpi_comm_rank( void ) {
    MPICHK();
    int rank;
    MPI_Comm_rank( diverge_mpi_comm, &rank );
    return rank;
}

void diverge_mpi_allreduce_double_max( const void* sendbuf, void* recvbuf, int num ) {
    MPICHK();
    MPI_Allreduce( sendbuf, recvbuf, num, MPI_DOUBLE, MPI_MAX, diverge_mpi_comm );
}

void diverge_mpi_allreduce_double_sum( const void* sendbuf, void* recvbuf, int num ) {
    MPICHK();
    MPI_Allreduce( sendbuf, recvbuf, num, MPI_DOUBLE, MPI_SUM, diverge_mpi_comm );
}

void diverge_mpi_allreduce_complex_sum( const void* sendbuf, void* recvbuf, int num ){
    MPICHK();
    MPI_Allreduce( sendbuf, recvbuf, num, MPI_C_DOUBLE_COMPLEX, MPI_SUM, diverge_mpi_comm );
}

void diverge_mpi_allreduce_complex_sum_inplace( void* sendrecvbuf, int num ) {
    MPICHK();
    MPI_Allreduce( MPI_IN_PLACE, sendrecvbuf, num, MPI_C_DOUBLE_COMPLEX, MPI_SUM, diverge_mpi_comm);
}

void diverge_mpi_allgather_int( const int* sendbuf, int* recvbuf, int num ) {
    MPICHK();
    MPI_Allgather( sendbuf, num, MPI_INT, recvbuf, num, MPI_INT, diverge_mpi_comm );
}

void diverge_mpi_allgather_index( const void* sendbuf, void* recvbuf, int num ) {
    MPICHK();
    MPI_Allgather( sendbuf, num, MPI_LONG_LONG, recvbuf, num, MPI_LONG_LONG, diverge_mpi_comm );
}

void diverge_mpi_allgather_double( const void* sendbuf, void* recvbuf, int num ) {
    MPICHK();
    MPI_Allgather( sendbuf, num, MPI_DOUBLE, recvbuf, num, MPI_DOUBLE, diverge_mpi_comm );
}

void diverge_mpi_send_double( void* data, int num, int destination, int tag ) {
    MPICHK();
    MPI_Send( data, num, MPI_DOUBLE, destination, tag, diverge_mpi_comm );
}

void diverge_mpi_recv_double( void* data, int num, int source, int tag ) {
    MPICHK();
    MPI_Status result;
    MPI_Recv( data, num, MPI_DOUBLE, source, tag, diverge_mpi_comm, &result );
}

void diverge_mpi_gatherv_index( const void* send, int count, void* recv, int* counts, int* displs, int root ) {
    MPICHK();
    MPI_Gatherv( send, count, MPI_LONG_LONG, recv, counts, displs, MPI_LONG_LONG, root, diverge_mpi_comm );
}

void diverge_mpi_gatherv_bytes( const void* send, int count, void* recv, int* counts, int* displs, int root ) {
    MPICHK();
    MPI_Gatherv( send, count, MPI_BYTE, recv, counts, displs, MPI_BYTE, root, diverge_mpi_comm );
}

void diverge_mpi_gatherv_cdoub( const void* send, int count, void* recv, int* counts, int* displs, int root ) {
    MPICHK();
    MPI_Gatherv( send, count, MPI_DOUBLE_COMPLEX, recv, counts, displs, MPI_DOUBLE_COMPLEX, root, diverge_mpi_comm );
}

void diverge_mpi_write_cdoub_to_file( const char* fname, const void* data, int displ, int count ) {
    MPICHK();
    MPI_File fh;
    if (MPI_File_open( diverge_mpi_comm, fname, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh ) != MPI_SUCCESS) {
        mpi_err_printf("could not open file %s. aborting.\n", fname );
        return;
    }
    MPI_File_set_view( fh, displ*sizeof(complex128_t), MPI_DOUBLE_COMPLEX, MPI_DOUBLE_COMPLEX, "native", MPI_INFO_NULL );
    MPI_File_write_at( fh, 0, data, count, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE );
    MPI_File_close( &fh );
}

index_t diverge_mpi_write_byte_to_file( void* file, const void* data, index_t displ, index_t count ) {
    if (file == NULL) {
        mpi_err_printf("got NULL as file argument. returning 0.\n");
        return 0;
    }
    MPICHK();
    MPI_File* fh = (MPI_File*)file;
    MPI_File_set_view( *fh, displ, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL );
    MPI_File_write_at( *fh, 0, data, count, MPI_BYTE, MPI_STATUS_IGNORE );
    index_t end_data = displ + count;
    MPI_Allreduce( MPI_IN_PLACE, &end_data, 1, MPI_INT64_T, MPI_MAX, diverge_mpi_comm );
    return end_data;
}


index_t diverge_mpi_write_byte_to_file_master( void* file, const void* data, index_t displ, index_t count) {
    if (file == NULL) {
        mpi_err_printf("got NULL as file argument. returning 0.\n");
        return 0;
    }
    MPICHK();
    index_t end_data = -1;
    if(diverge_mpi_comm_rank() == 0) {
        MPI_File* fh = (MPI_File*)file;
        MPI_File_set_view( *fh, displ, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL );
        MPI_File_write_at( *fh, 0, data, count, MPI_BYTE, MPI_STATUS_IGNORE );
        end_data = displ + count;
        MPI_Bcast( &end_data, sizeof(index_t), MPI_BYTE, 0, diverge_mpi_comm );
    }else{
        MPI_Bcast( &end_data, sizeof(index_t), MPI_BYTE, 0, diverge_mpi_comm );
    }
    return end_data;
}

void* diverge_mpi_open_file(const char* fname) {
    MPICHK();
    MPI_File* fh = malloc(sizeof(MPI_File));
    if (MPI_File_open( diverge_mpi_comm, fname, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, fh ) != MPI_SUCCESS) {
        mpi_err_printf("could not open file %s. returning NULL.\n", fname );
        return NULL;
    }
    return (void*)fh;
}
void diverge_mpi_close_file(void* file) {
    MPICHK();
    if (file == NULL) {
        mpi_err_printf("got NULL as file argument. returning.\n");
        return;
    }
    MPI_File_close( (MPI_File*)file );
    free( file );
}

void diverge_mpi_alltoallv_bytes( const void* send, const index_t* sendcounts_, const index_t* senddispls_,
                                        void* recv, const index_t* recvcounts_, const index_t* recvdispls_,
                                        index_t numbytes ) {
    MPICHK();
    int nranks = diverge_mpi_comm_size();
    int* sendcounts = (int*)malloc( sizeof(int)*nranks );
    int* senddispls = (int*)malloc( sizeof(int)*nranks );
    int* recvcounts = (int*)malloc( sizeof(int)*nranks );
    int* recvdispls = (int*)malloc( sizeof(int)*nranks );
    for (int i=0; i<nranks; ++i) {
        sendcounts[i] = sendcounts_[i] * numbytes;
        senddispls[i] = senddispls_[i] * numbytes;
        recvcounts[i] = recvcounts_[i] * numbytes;
        recvdispls[i] = recvdispls_[i] * numbytes;
    }
    MPI_Alltoallv( send, sendcounts, senddispls, MPI_BYTE,
                   recv, recvcounts, recvdispls, MPI_BYTE, diverge_mpi_comm );
    free( sendcounts );
    free( senddispls );
    free( recvcounts );
    free( recvdispls );
}

void diverge_mpi_alltoallv_complex( const complex128_t* send, const int* sendcounts, const int* senddispls,
                                          complex128_t* recv, const int* recvcounts, const int* recvdispls ) {
    MPICHK();
    MPI_Alltoallv( send, sendcounts, senddispls, MPI_C_DOUBLE_COMPLEX,
                   recv, recvcounts, recvdispls, MPI_C_DOUBLE_COMPLEX, diverge_mpi_comm );
}

void diverge_mpi_allgatherv( complex128_t* recv, const int* rcnt, const int* rdsp ) {
    MPICHK();
    MPI_Allgatherv( MPI_IN_PLACE, 0, MPI_DOUBLE_COMPLEX, recv, rcnt, rdsp, MPI_DOUBLE_COMPLEX, diverge_mpi_comm );
}

void diverge_mpi_allgatherv_index( index_t* recv, const int* rcnt, const int* rdsp ) {
    MPICHK();
    MPI_Allgatherv( MPI_IN_PLACE, 0, MPI_LONG_LONG, recv, rcnt, rdsp, MPI_LONG_LONG, diverge_mpi_comm );
}

double diverge_mpi_max( double* send ) {
    MPICHK();
    double recv;
    MPI_Allreduce( send, &recv, 1, MPI_DOUBLE, MPI_MAX, diverge_mpi_comm );
    return recv;
}

void diverge_mpi_bcast_bytes( void* inout, int size, int root ) {
    MPICHK();
    MPI_Bcast( inout, size, MPI_BYTE, root, diverge_mpi_comm );
}

void diverge_mpi_gather_double( const double* send, int scnt, double* recv, int rcnt, int root ) {
    MPICHK();
    MPI_Gather( send, scnt, MPI_DOUBLE, recv, rcnt, MPI_DOUBLE, root, diverge_mpi_comm );
}

#else // USE_MPI

#include <time.h>

void diverge_init( int* pargc, char*** pargv ) {
    (void)pargc;
    (void)pargv;
    init_internal_libs();
}
void diverge_finalize( void ) {
    finalize_internal_libs();
}
void diverge_embed( MPI_Comm comm ) {
    (void)comm;
    init_internal_libs();
}
void diverge_reset( void ) {
    finalize_internal_libs();
}

double diverge_mpi_wtime( void ) {
#ifdef USE_ALTERNATIVE_TIME
    return (double)clock()/(double)CLOCKS_PER_SEC;
#else // USE_ALTERNATIVE_TIME
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.e+9;
#endif // USE_ALTERNATIVE_TIME
}

MPI_Comm diverge_mpi_get_comm( void ) { return diverge_mpi_comm; }
void diverge_mpi_barrier( void ) {}
int diverge_mpi_comm_size( void ) { return 1; }
int diverge_mpi_comm_rank( void ) { return 0; }

void diverge_mpi_allreduce_double_max( const void* sendbuf, void* recvbuf, int num ) {
    (void)sendbuf; (void)recvbuf; (void)num;
}
void diverge_mpi_allreduce_double_sum( const void* sendbuf, void* recvbuf, int num ) {
    (void)sendbuf; (void)recvbuf; (void)num;
}
void diverge_mpi_allreduce_complex_sum( const void* sendbuf, void* recvbuf, int num ){
    (void)sendbuf; (void)recvbuf; (void)num;
}

void diverge_mpi_allreduce_complex_sum_inplace( void* sendrecvbuf, int num ) {
    (void)sendrecvbuf; (void) num;}

void diverge_mpi_allgather_int( const int* sendbuf, int* recvbuf, int num ) {
    memcpy( recvbuf, sendbuf, sizeof(int)*num );
}
void diverge_mpi_allgather_index( const void* sendbuf, void* recvbuf, int num ) {
    memcpy( recvbuf, sendbuf, sizeof(index_t)*num );
}
void diverge_mpi_allgather_double( const void* sendbuf, void* recvbuf, int num ) {
    memcpy( recvbuf, sendbuf, sizeof(double)*num );
}
void diverge_mpi_send_double( void* data, int num, int destination, int tag ) {
    (void)data; (void)num; (void)destination; (void)tag;
}
void diverge_mpi_recv_double( void* data, int num, int source, int tag ) {
    (void)data; (void)num; (void)source; (void)tag;
}

void diverge_mpi_send_index_t( void* data, int num, int destination, int tag ) {
    (void)data; (void)num; (void)destination; (void)tag;
}
void diverge_mpi_recv_index_t( void* data, int num, int source, int tag ) {
    (void)data; (void)num; (void)source; (void)tag;
}

void diverge_mpi_gatherv_index( const void* send, int count, void* recv, int* counts, int* displs, int root ) {
    (void)counts;
    (void)displs;
    (void)root;
    if (send != recv) memcpy( recv, send, sizeof(index_t)*count );
}

void diverge_mpi_gatherv_bytes( const void* send, int count, void* recv, int* counts, int* displs, int root ) {
    (void)counts;
    (void)displs;
    (void)root;
    if (send != recv) memcpy( recv, send, count );
}

void diverge_mpi_gatherv_cdoub( const void* send, int count, void* recv, int* counts, int* displs, int root ) {
    (void)counts;
    (void)displs;
    (void)root;
    if (send != recv) memcpy( recv, send, sizeof(complex128_t)*count );
}

void diverge_mpi_write_cdoub_to_file( const char* fname, const void* data, int displ, int count ) {
    FILE* f = fopen(fname,"w");
    if (!f) {
        mpi_err_printf("could not open file %s. aborting\n", fname );
        return;
    }
    fwrite( (const char*)data+sizeof(complex128_t)*displ, sizeof(complex128_t), count, f );
    fclose(f);
}


index_t diverge_mpi_write_byte_to_file( void* file, const void* data, index_t displ, index_t count){
    if (file == NULL) {
        mpi_err_printf("got NULL as file argument. returning 0.\n");
        return 0;
    }
    fwrite( (const char*)data, 1, count, (FILE*)file );
    return displ+count;
}
index_t diverge_mpi_write_byte_to_file_master( void* file, const void* data, index_t displ, index_t count) {
    return diverge_mpi_write_byte_to_file(file, data, displ, count);
}

void* diverge_mpi_open_file(const char* fname) {
    FILE* f = fopen(fname,"w");
    if (!f) {
        mpi_err_printf("could not open file %s. returning NULL.\n", fname );
        return NULL;
    }
    return (void*)f;
}
void diverge_mpi_close_file(void* file) {
    if (file == NULL) {
        mpi_err_printf("got NULL as file argument. returning.\n");
        return;
    }
    fclose((FILE*)file);
}

void diverge_mpi_alltoallv_bytes( const void* send, const index_t* sendcounts, const index_t* senddispls,
                                        void* recv, const index_t* recvcounts, const index_t* recvdispls,
                                        index_t numbytes ) {
    assert( sendcounts[0] == recvcounts[0] );
    (void)recvcounts;
    memcpy( (char*)recv + recvdispls[0] * numbytes,
            (char*)send + senddispls[0] * numbytes,
            sendcounts[0] * numbytes );
}

void diverge_mpi_alltoallv_complex( const complex128_t* send, const int* sendcounts, const int* senddispls,
                                          complex128_t* recv, const int* recvcounts, const int* recvdispls ) {
    assert( sendcounts[0] == recvcounts[0] );
    (void)recvcounts;
    memcpy( recv + recvdispls[0], send + senddispls[0], sendcounts[0] * sizeof(complex128_t) );
}

void diverge_mpi_allgatherv( complex128_t* recv, const int* rcnt, const int* rdsp ) {
    (void)recv; (void)rcnt; (void)rdsp;
}
void diverge_mpi_allgatherv_index( index_t* recv, const int* rcnt, const int* rdsp ) {
    (void)recv; (void)rcnt; (void)rdsp;
}
double diverge_mpi_max( double* send ) { return *send; }
void diverge_mpi_bcast_bytes( void* inout, int size, int root ) {
    (void)root; (void)size; (void)inout;
}
void diverge_mpi_gather_double( const double* send, int scnt, double* recv, int rcnt, int root ) {
    (void)root;
    (void)rcnt;
    memcpy(recv, send, sizeof(double)*scnt);
}

#endif // USE_MPI

void diverge_mpi_exit( int status ) {
    diverge_finalize();
    exit(status);
}

