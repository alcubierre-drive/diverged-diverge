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

    pack_t* send_pack;
    index_t* send_counts;
    index_t* send_displs;

    index_t n_recv_pack;
    pack_t* recv_pack;
    index_t* recv_counts;
    index_t* recv_displs;
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
    free( t->recv_displs );

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
    t->recv_displs = calloc_print( t->size, sizeof(index_t) );
    return t;
}

static inline int find_rank( index_t i, index_t* displs, index_t* counts, int size ) {
    int rank;
    for (rank=0; rank<size; ++rank)
        if ((i >= displs[rank]) && (i < (displs[rank] + counts[rank]))) break;
    return rank;
}

void tu_prop_gpu_mpi_dist_comm( tu_prop_gpu_mpi_dist_t* t, complex128_t* L_full_q, complex128_t* L ) {
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

    memset( t->touched, 0, t->touched_bytes );

    // find the touched indices
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<t->L_tot; ++i) {
        uint64_t* L_full_q_uint = (uint64_t*)(L_full_q + i);
        if (L_full_q_uint[0] | L_full_q_uint[1]) {
            bitary_set(t->touched, i);
        }
    }

    // accumulate the number of touched indices
    t->n_touched = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<t->L_tot; ++i) {
        if (bitary_get(t->touched, i))
            #pragma omp atomic
            t->n_touched++;
    }

    // realloc the array of touched indices. This should be a no-op after first
    // iteration.
    t->touched_to_idx = realloc_print( t->touched_to_idx, sizeof(index_t)*t->n_touched );

    // save the touched indices
    t->n_touched = 0;
    for (index_t i=0; i<t->L_tot; ++i) {
        if (bitary_get(t->touched, i)) {
            t->touched_to_idx[t->n_touched++] = i;
        }
    }

    // make some space. no-op after first iteration
    t->send_pack = realloc_print( t->send_pack, sizeof(pack_t)*t->n_touched );

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

    // sort the send pack
    sort_pack( t->send_pack, t->send_pack + t->n_touched );

    // set the send indices and communicate
    for (index_t s=0; s<t->n_touched; ++s)
        t->send_counts[t->send_pack[s].dest_rank]++;
    diverge_mpi_alltoall_index( t->send_counts, t->recv_counts, 1 );

    // set the displs
    t->send_displs[0] = t->recv_displs[0] = 0;
    for (int r=1; r<t->size; ++r) {
        t->send_displs[r] = t->send_displs[r-1] + t->send_counts[r-1];
        t->recv_displs[r] = t->recv_displs[r-1] + t->recv_counts[r-1];
    }

    // alloc recv pack, no-op after first iter
    t->n_recv_pack = t->recv_counts[t->size-1] + t->recv_displs[t->size-1];
    t->recv_pack = realloc_print( t->recv_pack, sizeof(pack_t)*t->n_recv_pack );

    // send the pack!
    diverge_mpi_alltoallv_bytes( t->send_pack, t->send_counts, t->send_displs,
                                 t->recv_pack, t->recv_counts, t->recv_displs, sizeof(pack_t) );

    // write into L_full_q, but only those elements that we found
    memset( L_full_q, 0, sizeof(complex128_t) * t->nk * t->L_orbff_spin );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t s=0; s<t->n_recv_pack; ++s) {
        assert( t->recv_pack[s].dest_rank == t->rank );
        atomic_set_complex128_t( L_full_q+t->recv_pack[s].dest, &t->recv_pack[s].val );
    }

    // write out into L
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(2)
    for (index_t b=0; b<t->L_orbff_spin; ++b)
    for (index_t qr=0; qr<t->my_nk; ++qr)
        L[IDX2(b,qr,t->my_nk)] = L_full_q[IDX2(b,qr+t->my_nk_off,t->nk)];
}
