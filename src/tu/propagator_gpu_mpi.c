#include "propagator_gpu_mpi.h"
#include "../misc/bitary.h"
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
#define PACK_MPI_MULTIPLIER 1
#endif // PACK_MPI_MULTIPLIER

static inline void align_pack_sizes( index_t* send_counts, index_t* recv_counts, int size,
        index_t* send_counts_packed, index_t* send_displs_packed,
        index_t* recv_counts_packed, index_t* recv_displs_packed ) {
    for (int r=0; r<size; ++r) {
        send_counts_packed[r] = ((send_counts[r]/PACK_MPI_MULTIPLIER)+
                                 (send_counts[r]%PACK_MPI_MULTIPLIER!=0));
        recv_counts_packed[r] = ((recv_counts[r]/PACK_MPI_MULTIPLIER)+
                                 (recv_counts[r]%PACK_MPI_MULTIPLIER!=0));
        send_counts[r] = send_counts_packed[r] * PACK_MPI_MULTIPLIER;
        recv_counts[r] = recv_counts_packed[r] * PACK_MPI_MULTIPLIER;
        if (r > 0) {
            send_displs_packed[r] = send_displs_packed[r-1] + send_counts_packed[r-1];
            recv_displs_packed[r] = recv_displs_packed[r-1] + recv_counts_packed[r-1];
        }
    }
}

struct tu_prop_gpu_mpi_dist_t {
    cbyte_t* touched;
    index_t touched_bytes;

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

    index_t n_recv_pack;
    pack_t* recv_pack;
    index_t* recv_counts;

    index_t *send_counts_packed, *send_displs_packed,
            *recv_counts_packed, *recv_displs_packed;
    int *scpi, *sdpi, *rcpi, *rdpi; // the same as above but as plain old integer pointers for MPI.
    index_t *send_counts_cpy, *send_displs_cpy,
            *recv_counts_cpy, *recv_displs_cpy;


    union {
        struct {
            double t_prepare,
                   t_touched_find,
                   t_touched_acc,
                   t_touched_realloc,
                   t_touched_save,
                   t_sendpack_realloc,
                   t_sendpack_gen,
                   t_sendpack_sort,
                   t_comm_counts,
                   t_counts_mult,
                   t_sendpack_align,
                   t_sendpack_pad,
                   t_comm_alltoallv,
                   t_writeout_full,
                   t_writeout_L,
                   _t_last;
        };
        struct {
            double _t_vector[15];
        };
    };
};

static const char _t_descr[][256] = {
    "t_prepare",
    "t_touched_find",
    "t_touched_acc",
    "t_touched_realloc",
    "t_touched_save",
    "t_sendpack_realloc",
    "t_sendpack_gen",
    "t_sendpack_sort",
    "t_comm_counts",
    "t_counts_mult",
    "t_sendpack_align",
    "t_sendpack_pad",
    "t_comm_alltoallv",
    "t_writeout_full",
    "t_writeout_L",
};

void tu_prop_gpu_mpi_dist_free( tu_prop_gpu_mpi_dist_t* t ) {
    free( t->touched );
    free( t->touched_to_idx );

    free( t->k_counts );
    free( t->k_displs );

    free( t->send_pack );
    free( t->send_counts );
    free( t->send_displs );

    free( t->recv_pack );
    free( t->recv_counts );

    free( t->send_counts_cpy );
    free( t->send_displs_cpy );
    free( t->send_counts_packed );
    free( t->send_displs_packed );

    free( t->recv_counts_cpy );
    free( t->recv_displs_cpy );
    free( t->recv_counts_packed );
    free( t->recv_displs_packed );

    free( t->scpi );
    free( t->sdpi );
    free( t->rcpi );
    free( t->rdpi );

    free( t );
}

#ifdef TU_GPU_MPI_ALLOC_PRINT
#define calloc_print( cnt, sz ) \
    calloc( cnt, sz ); \
    mpi_vrb_printf( "calloc %.2fGB\n", (double)sz*cnt / (1024.*1024.*1024) )

#define realloc_print( ptr, cnt_sz ) \
    realloc( ptr, cnt_sz ); \
    mpi_vrb_printf( "realloc %.2fGB\n", (double)cnt_sz / (1024.*1024.*1024) )
#else
#define calloc_print calloc
#define realloc_print realloc
#endif

tu_prop_gpu_mpi_dist_t* tu_prop_gpu_mpi_dist_init( index_t my_nk,
        index_t my_nk_off, index_t nk, index_t L_orbff_spin ) {
    tu_prop_gpu_mpi_dist_t* t = calloc_print( 1, sizeof(tu_prop_gpu_mpi_dist_t) );
    t->my_nk = my_nk;
    t->my_nk_off = my_nk_off;
    t->nk = nk;
    t->L_orbff_spin = L_orbff_spin;
    t->L_tot = t->L_orbff_spin * t->nk;
    t->touched = bitary_alloc( t->L_tot );
    t->touched_bytes = CHAR_DIV(t->L_tot)+(CHAR_MOD(t->L_tot) != 0);

    t->rank = diverge_mpi_comm_rank();
    t->size = diverge_mpi_comm_size();

    t->k_counts = calloc_print( t->size, sizeof(index_t) );
    t->k_displs = calloc_print( t->size, sizeof(index_t) );

    t->k_counts[t->rank] = t->my_nk;
    t->k_displs[t->rank] = t->my_nk_off;
    diverge_mpi_allgather_index( MPI_IN_PLACE, t->k_counts, 1 );
    diverge_mpi_allgather_index( MPI_IN_PLACE, t->k_displs, 1 );

    t->send_counts = calloc_print( t->size, sizeof(index_t) );
    t->send_displs = calloc_print( t->size, sizeof(index_t) );
    t->recv_counts = calloc_print( t->size, sizeof(index_t) );

    t->send_counts_cpy = calloc_print( t->size, sizeof(index_t) );
    t->recv_counts_cpy = calloc_print( t->size, sizeof(index_t) );
    t->send_displs_cpy = calloc_print( t->size, sizeof(index_t) );
    t->recv_displs_cpy = calloc_print( t->size, sizeof(index_t) );

    t->send_counts_packed = calloc_print( t->size, sizeof(index_t) );
    t->recv_counts_packed = calloc_print( t->size, sizeof(index_t) );
    t->send_displs_packed = calloc_print( t->size, sizeof(index_t) );
    t->recv_displs_packed = calloc_print( t->size, sizeof(index_t) );

    t->scpi = calloc_print( t->size, sizeof(int) );
    t->sdpi = calloc_print( t->size, sizeof(int) );
    t->rcpi = calloc_print( t->size, sizeof(int) );
    t->rdpi = calloc_print( t->size, sizeof(int) );
    return t;
}

static inline int find_rank( index_t i, index_t* displs, index_t* counts, int size ) {
    int rank;
    for (rank=0; rank<size; ++rank)
        if ((i >= displs[rank]) && (i < (displs[rank] + counts[rank]))) break;
    return rank;
}

void tu_prop_gpu_mpi_dist_comm( tu_prop_gpu_mpi_dist_t* t, complex128_t* L_full_q, complex128_t* L ) {

    double tick, tock;

    tick = diverge_mpi_wtime();
    // reset the touched index array
    if (t->n_touched) {
        memset( t->touched_to_idx, 0, sizeof(index_t)*t->n_touched );
        memset( t->send_pack, 0, sizeof(pack_t)*t->n_touched );
    }
    if (t->n_recv_pack) {
        memset( t->recv_pack, 0, sizeof(pack_t)*t->n_recv_pack );
    }
    // clear send counts to count again
    memset( t->recv_counts, 0, sizeof(index_t) * t->size );
    memset( t->send_counts, 0, sizeof(index_t) * t->size );
    // clear touched bytes
    memset( t->touched, 0, t->touched_bytes );
    tock = diverge_mpi_wtime();
    t->t_prepare += tock-tick;

    tick = diverge_mpi_wtime();
    // find the touched indices
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<t->L_tot; ++i) {
        uint64_t* L_full_q_uint = (uint64_t*)(L_full_q + i);
        if (L_full_q_uint[0] | L_full_q_uint[1]) {
            bitary_set(t->touched, i);
        }
    }
    tock = diverge_mpi_wtime();
    t->t_touched_find += tock-tick;

    tick = diverge_mpi_wtime();
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
    tock = diverge_mpi_wtime();
    t->t_touched_acc += tock-tick;

    tick = diverge_mpi_wtime();
    // realloc the array of touched indices. This should be a no-op after first
    // iteration.
    t->touched_to_idx = realloc_print( t->touched_to_idx, sizeof(index_t)*t->n_touched );
    tock = diverge_mpi_wtime();
    t->t_touched_realloc += tock-tick;

    tick = diverge_mpi_wtime();
    // save the touched indices
    t->n_touched = 0;
    for (index_t i=0; i<t->L_tot; ++i) {
        if (bitary_get(t->touched, i)) {
            t->touched_to_idx[t->n_touched++] = i;
        }
    }
    tock = diverge_mpi_wtime();
    t->t_touched_save += tock-tick;

    tick = diverge_mpi_wtime();
    // make some space. no-op after first iteration
    t->send_pack = realloc_print( t->send_pack, sizeof(pack_t)*t->n_touched );
    tock = diverge_mpi_wtime();
    t->t_sendpack_realloc += tock-tick;

    tick = diverge_mpi_wtime();
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t s=0; s<t->n_touched; ++s) {
        index_t i = t->touched_to_idx[s];
        t->send_pack[s] = (pack_t){
            .dest = i,
            .orig_rank = t->rank,
            .dest_rank = find_rank( i % t->nk, t->k_displs, t->k_counts, t->size ),
            .val = L_full_q[i],
        };
    }
    tock = diverge_mpi_wtime();
    t->t_sendpack_gen += tock-tick;

    tick = diverge_mpi_wtime();
    // sort the send pack
    sort_pack( t->send_pack, t->send_pack + t->n_touched );
    tock = diverge_mpi_wtime();
    t->t_sendpack_sort += tock-tick;

    tick = diverge_mpi_wtime();
    // set the send indices and communicate
    for (index_t s=0; s<t->n_touched; ++s)
        t->send_counts[t->send_pack[s].dest_rank]++;
    diverge_mpi_alltoall_index( t->send_counts, t->recv_counts, 1 );
    tock = diverge_mpi_wtime();
    t->t_comm_counts += tock-tick;

    tick = diverge_mpi_wtime();
    // now we have the recv and send counts in units of sizeof(pack_t). But we
    // must convert to sizeof(pack_t) * PACK_MPI_MULTIPLIER...
    memcpy( t->send_counts_cpy, t->send_counts, sizeof(index_t)*t->size );
    memcpy( t->recv_counts_cpy, t->recv_counts, sizeof(index_t)*t->size );
    align_pack_sizes( t->send_counts_cpy, t->recv_counts_cpy, t->size,
            t->send_counts_packed, t->send_displs_packed,
            t->recv_counts_packed, t->recv_displs_packed );
    tock = diverge_mpi_wtime();
    t->t_counts_mult += tock - tick;

    tick = diverge_mpi_wtime();
    // align the sizes and realloc memory accordingly
    index_t aligned_send_pack_size = 0, aligned_recv_pack_size = 0;
    t->send_displs_cpy[0] = t->recv_displs_cpy[0] = 0;
    for (int r=0; r<t->size; ++r) {
        aligned_send_pack_size += t->send_counts_cpy[r];
        aligned_recv_pack_size += t->recv_counts_cpy[r];
        if (r > 0) {
            t->send_displs_cpy[r] = t->send_displs_cpy[r-1] + t->send_counts_cpy[r-1];
            t->recv_displs_cpy[r] = t->recv_displs_cpy[r-1] + t->recv_counts_cpy[r-1];
        }
    }
    t->send_pack = realloc_print( t->send_pack, sizeof(pack_t) * aligned_send_pack_size );
    t->recv_pack = realloc_print( t->recv_pack, sizeof(pack_t) * aligned_recv_pack_size );
    tock = diverge_mpi_wtime();
    t->t_sendpack_align += tock - tick;

    tick = diverge_mpi_wtime();
    // move the memory in send_pack to the right in order to pad accordingly. We
    // iterate in reverse and therefore don't need a copy. First need to
    // generade send_displs, though
    t->send_displs[0] = 0;
    for (int r=1; r<t->size; ++r)
        t->send_displs[r] = t->send_displs[r-1] + t->send_counts[r-1];
    // and the actual padding
    for (int r=t->size-1; r>=0; --r) {
        for (index_t c=t->send_counts[r]-1; c>=0; --c) {
            t->send_pack[t->send_displs_cpy[r] + c] = t->send_pack[t->send_displs[r] + c];
        }
    }
    tock = diverge_mpi_wtime();
    t->t_sendpack_pad += tock - tick;

    tick = diverge_mpi_wtime();
    // send the pack in larger chunks, copy over the counts/displs to integer
    // arrays for MPI
    for (int r=0; r<t->size; ++r) {
        t->scpi[r] = t->send_counts_packed[r];
        t->sdpi[r] = t->send_displs_packed[r];
        t->rcpi[r] = t->recv_counts_packed[r];
        t->rdpi[r] = t->recv_displs_packed[r];
    }
    diverge_mpi_alltoallv_bytes_packed( t->send_pack, t->scpi, t->sdpi,
                                        t->recv_pack, t->rcpi, t->rdpi,
                                        sizeof(pack_t) * PACK_MPI_MULTIPLIER );
    tock = diverge_mpi_wtime();
    t->t_comm_alltoallv += tock - tick;

    tick = diverge_mpi_wtime();
    // write into L_full_q, but only those elements that we found
    memset( L_full_q, 0, sizeof(complex128_t) * t->nk * t->L_orbff_spin );
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
