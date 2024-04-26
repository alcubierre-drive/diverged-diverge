/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "propagator_gpu_mpi.h"
#include "../misc/mpi_functions.h"
#include <assert.h>

static inline void atomic_set_complex128_t( complex128_t* dest_, const complex128_t* src_ ) {
    double* dest = (double*)dest_;
    const double* src = (const double*)src_;
    #pragma omp atomic write
    dest[0] = src[0];
    #pragma omp atomic write
    dest[1] = src[1];
}

#ifndef PACK_MPI_MULTIPLIER
#define PACK_MPI_MULTIPLIER 128
#endif // PACK_MPI_MULTIPLIER

static inline index_t align_to( index_t number, index_t n ) {
    if (number == 0) return 0;
    while (number % n != 0) number++;
    return number;
}

static inline void align_pack_sizes( index_t* send_counts, index_t* recv_counts, int size,
        index_t* send_counts_packed, index_t* send_displs_packed,
        index_t* recv_counts_packed, index_t* recv_displs_packed ) {
    send_displs_packed[0] = recv_displs_packed[0] = 0;
    for (int r=0; r<size; ++r) {
        send_counts[r] = align_to( send_counts[r], PACK_MPI_MULTIPLIER );
        recv_counts[r] = align_to( recv_counts[r], PACK_MPI_MULTIPLIER );
        send_counts_packed[r] = send_counts[r] / PACK_MPI_MULTIPLIER;
        recv_counts_packed[r] = recv_counts[r] / PACK_MPI_MULTIPLIER;
        if (r > 0) {
            send_displs_packed[r] = send_displs_packed[r-1] + send_counts_packed[r-1];
            recv_displs_packed[r] = recv_displs_packed[r-1] + recv_counts_packed[r-1];
        }
    }
}

struct tu_prop_gpu_mpi_dist_t {

    cbyte_t* touched;

    index_t n_touched;
    index_t* touched_to_idx;

    index_t my_nk, my_nk_off, nk;
    index_t L_orbff_spin, L_tot;

    int rank;
    int size;

    index_t* k_counts;
    index_t* k_displs;

    /* index_t n_touched; */
    pack_t* send_pack;
    index_t* send_counts;
    index_t *send_displs;

    index_t* recv_counts;
    int owns_recv_pack;

    pack_t** p_recv_pack;
    index_t* p_n_recv_pack;

    index_t *send_counts_packed, *send_displs_packed,
            *recv_counts_packed, *recv_displs_packed;
    int *scpi, *sdpi, *rcpi, *rdpi; // the same as above but as plain old integer pointers for MPI.
    index_t *send_counts_cpy, *send_displs_cpy,
            *recv_counts_cpy, *recv_displs_cpy;


    union {
        struct {
            double t_prepare,
                   t_sendpack_gen,
                   t_comm_alltoallv,
                   t_writeout_full,
                   t_writeout_L,
                   t_total,
                   _t_last;
        };
        struct {
            double _t_vector[7];
        };
    };

    int first_pass; // document whether this is the first pass of calling the
                    // communicator

    index_t n_send_pack;
};

static const char _t_descr[][256] = {
    "t_prepare",
    "t_sendpack_gen",
    "t_comm_alltoallv",
    "t_writeout_full",
    "t_writeout_L",
    "t_total",
};

void tu_prop_gpu_mpi_dist_free( tu_prop_gpu_mpi_dist_t* t ) {
    free( t->send_pack );
    if (t->owns_recv_pack)
        free( *(t->p_recv_pack) );
    free( t->recv_counts );

    free( t->recv_displs_cpy );

    free( t->scpi );
    free( t->sdpi );
    free( t->rcpi );
    free( t->rdpi );

    free( t );
}

#ifdef TU_GPU_MPI_ALLOC_PRINT
#ifndef NDEBUG
#define EXPORT_TU_GPU_MPI_ALLOC_PRINT
#else
#define EXPORT_TU_GPU_MPI_ALLOC_PRINT static inline
#endif
EXPORT_TU_GPU_MPI_ALLOC_PRINT void* calloc_print_( size_t cnt, size_t sz, int line ) {
    void* ptr = calloc( cnt, sz );
    int scc = ptr != NULL;
    mpi_vrb_printf_all( "rank %i: calloc@%03i %.2fGB (%s)\n", diverge_mpi_comm_rank(), line,
            (double)sz*cnt / (1024.*1024.*1024), scc ? "success!!!" : "no success" );
    return ptr;
}

EXPORT_TU_GPU_MPI_ALLOC_PRINT void* realloc_print_( void* ptr, size_t sz, int line ) {
    ptr = realloc( ptr, sz );
    int scc = ptr != NULL;
    mpi_vrb_printf_all( "rank %i:realloc@%03i %.2fGB (%s)\n", diverge_mpi_comm_rank(), line,
            (double)sz / (1024.*1024.*1024), scc ? "success!!!" : "no success" );
    return ptr;
}
#define calloc_print( cnt, sz ) calloc_print_( cnt, sz, __LINE__ )
#define realloc_print( ptr, sz ) realloc_print_( ptr, sz, __LINE__ )
#else
#define calloc_print calloc
#define realloc_print realloc
#endif

cbyte_t* tu_prop_gpu_mpi_dist_cuda_kernel_bitary( tu_prop_gpu_mpi_dist_t* t ) {
    return t->first_pass ? t->touched : NULL;
}

tu_prop_gpu_mpi_dist_t* tu_prop_gpu_mpi_dist_init( index_t my_nk,
        index_t my_nk_off, index_t nk, index_t L_orbff_spin,
        pack_t** p_recv_pack, index_t* p_n_recv_pack ) {
    tu_prop_gpu_mpi_dist_t* t = calloc( 1, sizeof(tu_prop_gpu_mpi_dist_t) );
    t->my_nk = my_nk;
    t->my_nk_off = my_nk_off;
    t->nk = nk;
    t->L_orbff_spin = L_orbff_spin;
    t->L_tot = t->L_orbff_spin * t->nk;
    t->touched = bitary_alloc( t->L_tot );

    t->rank = diverge_mpi_comm_rank();
    t->size = diverge_mpi_comm_size();

    t->k_counts = calloc( t->size, sizeof(index_t) );
    t->k_displs = calloc( t->size, sizeof(index_t) );

    t->k_counts[t->rank] = t->my_nk;
    t->k_displs[t->rank] = t->my_nk_off;
    diverge_mpi_allgather_index( MPI_IN_PLACE, t->k_counts, 1 );
    diverge_mpi_allgather_index( MPI_IN_PLACE, t->k_displs, 1 );

    t->send_counts = calloc( t->size, sizeof(index_t) );
    t->send_displs = calloc( t->size, sizeof(index_t) );
    t->recv_counts = calloc( t->size, sizeof(index_t) );

    t->send_counts_cpy = calloc( t->size, sizeof(index_t) );
    t->recv_counts_cpy = calloc( t->size, sizeof(index_t) );
    t->send_displs_cpy = calloc( t->size, sizeof(index_t) );
    t->recv_displs_cpy = calloc( t->size, sizeof(index_t) );

    t->send_counts_packed = calloc( t->size, sizeof(index_t) );
    t->recv_counts_packed = calloc( t->size, sizeof(index_t) );
    t->send_displs_packed = calloc( t->size, sizeof(index_t) );
    t->recv_displs_packed = calloc( t->size, sizeof(index_t) );

    t->scpi = calloc( t->size, sizeof(int) );
    t->sdpi = calloc( t->size, sizeof(int) );
    t->rcpi = calloc( t->size, sizeof(int) );
    t->rdpi = calloc( t->size, sizeof(int) );

    t->first_pass = 1;

    t->p_recv_pack = p_recv_pack;
    t->p_n_recv_pack = p_n_recv_pack;
    return t;
}

static inline int find_rank( index_t i, index_t* displs, index_t* counts, int size ) {
    int rank;
    for (rank=0; rank<size; ++rank)
        if ((i >= displs[rank]) && (i < (displs[rank] + counts[rank]))) break;
    return rank;
}

static void tu_prop_gpu_mpi_dist_comm_communicate( tu_prop_gpu_mpi_dist_t* t ) {
    double tick, tock;
    tick = diverge_mpi_wtime();
    diverge_mpi_alltoallv_bytes_packed( t->send_pack, t->scpi, t->sdpi,
                                        *(t->p_recv_pack), t->rcpi, t->rdpi,
                                        sizeof(pack_t) * PACK_MPI_MULTIPLIER );
    tock = diverge_mpi_wtime();
    t->t_comm_alltoallv += tock - tick;
}

static void tu_prop_gpu_mpi_dist_comm_writeout( tu_prop_gpu_mpi_dist_t* t, complex128_t* L ) {
    double tick, tock;

    tick = diverge_mpi_wtime();
    pack_t* recv_pack = *(t->p_recv_pack);
    memset( L, 0, sizeof(complex128_t) * t->nk * t->L_orbff_spin );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (int r=0; r<t->size; ++r) {
        for (index_t c=0; c<t->recv_counts[r]; ++c) {
            index_t idx = t->recv_displs_cpy[r] + c;
            assert( recv_pack[idx].dest_rank == t->rank );
            index_t bq_idx = recv_pack[idx].dest;
            index_t b = bq_idx / t->nk;
            index_t qrel = (bq_idx % t->nk) - t->my_nk_off;
            atomic_set_complex128_t( L + IDX2(b, qrel, t->my_nk), &recv_pack[idx].val );
        }
    }
    tock = diverge_mpi_wtime();
    t->t_writeout_L += tock - tick;

    /*
    double tick, tock;
    tick = diverge_mpi_wtime();
    // write into L_full_q, but only those elements that we found
    memset( L_full_q, 0, sizeof(complex128_t) * t->nk * t->L_orbff_spin );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (int r=0; r<t->size; ++r) {
        for (index_t c=0; c<t->recv_counts[r]; ++c) {
            index_t idx = t->recv_displs_cpy[r] + c;
            assert( t->recv_pack[idx].dest_rank == t->rank );
            atomic_set_complex128_t( L_full_q+t->recv_pack[idx].dest, &t->recv_pack[idx].val );
        }
    }
    tock = diverge_mpi_wtime();
    t->t_writeout_full += tock - tick;

    tick = diverge_mpi_wtime();
    // write out into L
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(2)
    for (index_t b=0; b<t->L_orbff_spin; ++b)
    for (index_t qr=0; qr<t->my_nk; ++qr)
        L[IDX2(b,qr,t->my_nk)] = L_full_q[IDX2(b,qr+t->my_nk_off,t->nk)];
    tock = diverge_mpi_wtime();
    t->t_writeout_L += tock - tick;
    */
}

static void tu_prop_gpu_mpi_dist_comm_make_send_pack( tu_prop_gpu_mpi_dist_t* t, complex128_t* L ) {
    double tick, tock;
    tick = diverge_mpi_wtime();
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t s=0; s<t->n_send_pack; ++s) {
        if (t->send_pack[s].dest != -1) {
            assert( t->send_pack[s].orig_rank == t->rank );
            t->send_pack[s].val = L[t->send_pack[s].dest];
        }
    }
    tock = diverge_mpi_wtime();
    t->t_sendpack_gen += tock-tick;
}

#define free_null( ptr ) { \
    free( ptr ); \
    ptr = NULL; \
}

static void tu_prop_gpu_mpi_dist_comm_first_pass( tu_prop_gpu_mpi_dist_t* t, complex128_t* L );

void tu_prop_gpu_mpi_dist_comm( tu_prop_gpu_mpi_dist_t* t, complex128_t* L ) {
    if (t->first_pass) {
        tu_prop_gpu_mpi_dist_comm_first_pass( t, L );
        t->first_pass = 0;

        free_null( t->touched );
        free_null( t->touched_to_idx );

        free_null( t->k_counts );
        free_null( t->k_displs );

        free_null( t->send_counts );
        free_null( t->send_displs );

        free_null( t->send_counts_cpy );
        free_null( t->send_displs_cpy );
        free_null( t->send_counts_packed );
        free_null( t->send_displs_packed );

        free_null( t->recv_counts_cpy );
        free_null( t->recv_counts_packed );
        free_null( t->recv_displs_packed );
    } else {
        double tick, tock;
        tick = diverge_mpi_wtime();
        // L -> send_pack, requires *only* send_pack
        tu_prop_gpu_mpi_dist_comm_make_send_pack( t, L );
        // send_pack -> recv_pack, requires send_pack, recv_pack, scpi, sdpi, rcpi, rdpi
        tu_prop_gpu_mpi_dist_comm_communicate( t );
        // recv_pack -> L, requires *only* recv_pack
        tu_prop_gpu_mpi_dist_comm_writeout( t, L );
        tock = diverge_mpi_wtime();
        t->t_total += tock - tick;
    }
}

static void tu_prop_gpu_mpi_dist_comm_first_pass( tu_prop_gpu_mpi_dist_t* t, complex128_t* L ) {

    double tick, tock;

    tick = diverge_mpi_wtime();
    // find the touched indices
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<t->L_tot; ++i) {
        uint64_t* L_uint = (uint64_t*)(L + i);
        if (L_uint[0] | L_uint[1]) {
            bitary_set(t->touched, i);
        }
    }

    // find the touched indices
    // accumulate the number of touched indices
    {
        index_t t_n_touched = 0;
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(+:t_n_touched)
        for (index_t i=0; i<t->L_tot; ++i) {
            if (bitary_get(t->touched, i)) {
                t_n_touched++;
            }
        }
        t->n_touched = t_n_touched;
    }

    // realloc the array of touched indices. This should be a no-op after first
    // iteration.
    t->touched_to_idx = calloc_print( t->n_touched, sizeof(index_t) );

    // save the touched indices
    t->n_touched = 0;
    for (index_t i=0; i<t->L_tot; ++i) {
        if (bitary_get(t->touched, i)) {
            t->touched_to_idx[t->n_touched++] = i;
        }
    }

    // make some space. no-op after first iteration
    t->send_pack = calloc_print( t->n_touched, sizeof(pack_t) );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t s=0; s<t->n_touched; ++s) {
        index_t i = t->touched_to_idx[s];
        t->send_pack[s] = (pack_t){
            .dest = i,
            .orig_rank = t->rank,
            .dest_rank = find_rank( i % t->nk, t->k_displs, t->k_counts, t->size ),
            .val = L[i],
        };
    }

    // sort the send pack
    sort_pack( t->send_pack, t->send_pack + t->n_touched );

    // set the send indices and communicate
    for (index_t s=0; s<t->n_touched; ++s)
        t->send_counts[t->send_pack[s].dest_rank]++;
    diverge_mpi_alltoall_index( t->send_counts, t->recv_counts, 1 );

    // now we have the recv and send counts in units of sizeof(pack_t). But we
    // must convert to sizeof(pack_t) * PACK_MPI_MULTIPLIER...
    memcpy( t->send_counts_cpy, t->send_counts, sizeof(index_t)*t->size );
    memcpy( t->recv_counts_cpy, t->recv_counts, sizeof(index_t)*t->size );
    align_pack_sizes( t->send_counts_cpy, t->recv_counts_cpy, t->size,
            t->send_counts_packed, t->send_displs_packed,
            t->recv_counts_packed, t->recv_displs_packed );

    // align the sizes and realloc memory accordingly
    t->send_displs_cpy[0] = t->recv_displs_cpy[0] = 0;
    for (int r=1; r<t->size; ++r) {
        t->send_displs_cpy[r] = t->send_displs_cpy[r-1] + t->send_counts_cpy[r-1];
        t->recv_displs_cpy[r] = t->recv_displs_cpy[r-1] + t->recv_counts_cpy[r-1];
    }
    t->n_send_pack = (t->send_displs_cpy[t->size-1]+t->send_counts_cpy[t->size-1]);
    t->send_pack = realloc_print( t->send_pack, sizeof(pack_t) * t->n_send_pack );
    index_t n_recv_pack = (t->recv_displs_cpy[t->size-1]+t->recv_counts_cpy[t->size-1]);

    if (*(t->p_recv_pack) == NULL) {
        *(t->p_n_recv_pack) = n_recv_pack;
        *(t->p_recv_pack) = calloc_print( *(t->p_n_recv_pack), sizeof(pack_t) );
        t->owns_recv_pack = 1;
    } else {
        *(t->p_n_recv_pack) = MAX( *(t->p_n_recv_pack), n_recv_pack );
        *(t->p_recv_pack) = realloc_print( *(t->p_recv_pack), sizeof(pack_t) * (*(t->p_n_recv_pack)) );
        t->owns_recv_pack = 0;
    }

    // move the memory in send_pack to the right in order to pad accordingly. We
    // iterate in reverse and therefore don't need a copy. First need to
    // generade send_displs, though
    t->send_displs[0] = 0;
    for (int r=1; r<t->size; ++r)
        t->send_displs[r] = t->send_displs[r-1] + t->send_counts[r-1];
    // and the actual padding
    for (int r=t->size-1; r>=0; --r) {
        for (index_t c=t->send_counts_cpy[r]-1; c>=t->send_counts[r]; --c) {
            // set all the invalid ones to -1!
            t->send_pack[t->send_displs_cpy[r] + c] = (pack_t){
                .dest = -1, .orig_rank = -1, .dest_rank = -1, .val = 0
            };
        }
        for (index_t c=t->send_counts[r]-1; c>=0; --c) {
            // and copy the other ones over
            t->send_pack[t->send_displs_cpy[r] + c] = t->send_pack[t->send_displs[r] + c];
        }
    }

    // send the pack in larger chunks, copy over the counts/displs to integer
    // arrays for MPI
    for (int r=0; r<t->size; ++r) {
        t->scpi[r] = t->send_counts_packed[r];
        t->sdpi[r] = t->send_displs_packed[r];
        t->rcpi[r] = t->recv_counts_packed[r];
        t->rdpi[r] = t->recv_displs_packed[r];
    }
    tu_prop_gpu_mpi_dist_comm_communicate( t );

    tu_prop_gpu_mpi_dist_comm_writeout( t, L );

    tock = diverge_mpi_wtime();
    t->t_prepare += tock - tick;
}

index_t tu_prop_gpu_mpi_dist_timing_num( const tu_prop_gpu_mpi_dist_t* t ) {
    (void)t;
    return (offsetof(tu_prop_gpu_mpi_dist_t, _t_last) -
            offsetof(tu_prop_gpu_mpi_dist_t, t_prepare)) / sizeof(double);
}
double tu_prop_gpu_mpi_dist_timing( const tu_prop_gpu_mpi_dist_t* t, index_t i ) {
    return t->_t_vector[i];
}
const char* tu_prop_gpu_mpi_dist_timing_name( const tu_prop_gpu_mpi_dist_t* t, index_t i ) {
    (void)t;
    return _t_descr[i];
}
