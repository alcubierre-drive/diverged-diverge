#include "propagator_gpu_mpi.h"
#include "../misc/bitary.h"
#include "../misc/mpi_functions.h"

typedef struct {
    index_t dest;
    int dest_rank, orig_rank;
    complex128_t val;
} pack_t;

struct tu_prop_gpu_mpi_dist_t {

    cbyte_t* touched;
    index_t touched_n_bytes;

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

    free( t->send_pack );
    free( t->send_counts );
    free( t->send_displs );

    free( t->recv_pack );
    free( t->recv_counts );
    free( t->recv_displs );

    free( t );
}

tu_prop_gpu_mpi_dist_t* tu_prop_gpu_mpi_dist_init( index_t my_nk,
        index_t my_nk_off, index_t nk, index_t L_orbff_spin ) {
    tu_prop_gpu_mpi_dist_t* t = calloc( 1, sizeof(tu_prop_gpu_mpi_dist_t) );
    t->my_nk = my_nk;
    t->my_nk_off = my_nk_off;
    t->nk = nk;
    t->L_orbff_spin = L_orbff_spin;
    t->L_tot = t->L_orbff_spin * t->nk;
    t->touched = bitary_alloc( t->L_tot );
    t->touched_n_bytes = CHAR_DIV(t->L_tot) + (CHAR_MOD(t->L_tot) != 0);

    t->rank = diverge_mpi_comm_rank();
    t->size = diverge_mpi_comm_size();

    t->k_counts = diverge_mpi_distribute( t->nk );
    t->k_displs = t->k_counts + t->size;

    t->send_counts = calloc( t->size, sizeof(index_t) );
    t->send_displs = calloc( t->size, sizeof(index_t) );
    t->recv_counts = calloc( t->size, sizeof(index_t) );
    t->recv_displs = calloc( t->size, sizeof(index_t) );
    return t;
}

static inline int find_rank( index_t ii, index_t* displs, int size ) {
    int rank;
    for (rank=0; rank<size; ++rank)
        if (ii >= displs[rank]) break;
    return rank;
}

static inline int pack_compar( const void* p1_, const void* p2_ ) {
    const pack_t *p1 = (const pack_t*)p1_,
                 *p2 = (const pack_t*)p2_;
    return p1->dest_rank - p2->dest_rank;
}

void tu_prop_gpu_mpi_dist_comm( tu_prop_gpu_mpi_dist_t* t, complex128_t* L_full_q, complex128_t* L ) {
    // reset the touched bit array
    memset( t->touched, 0, t->touched_n_bytes );
    index_t n_touched_prev = t->n_touched;
    index_t n_recv_pack_prev = t->n_recv_pack;
    // reset the touched index array
    if (n_touched_prev) {
        memset( t->touched_to_idx, 0, sizeof(index_t)*n_touched_prev );
    }
    if (n_touched_prev) {
        memset( t->recv_pack, 0, sizeof(pack_t)*n_touched_prev );
    }

    // find the number of touched indices
    t->n_touched = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<t->L_tot; ++i) {
        uint64_t* L_full_q_uint = (uint64_t*)(L_full_q + i);
        if (L_full_q_uint[0] | L_full_q_uint[1]) {
            bitary_set(t->touched, i);
            #pragma omp atomic
            t->n_touched++;
        }
    }

    // realloc the array of touched indices. This should be a no-op after first
    // iteration.
    t->touched_to_idx = realloc( t->touched_to_idx, sizeof(index_t) * MAX(n_touched_prev, t->n_touched) );

    // save the touched indices
    t->n_touched = 0;
    for (index_t i=0; i<t->L_tot; ++i) {
        if (bitary_get(t->touched, i)) {
            t->touched_to_idx[t->n_touched++] = i;
        }
    }

    // make some space. no-op after first iteration
    t->send_pack = realloc( t->send_pack, sizeof(pack_t) * MAX(n_touched_prev, t->n_touched) );

    // clear send counts to count again
    memset( t->send_counts, 0, sizeof(index_t) * t->size );
    memset( t->recv_counts, 0, sizeof(index_t) * t->size );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t tt=0; tt<t->n_touched; ++tt) {
        index_t ii = t->touched_to_idx[tt];
        t->send_pack[tt] = (pack_t){
            .dest = ii,
            .orig_rank = t->rank,
            .dest_rank = find_rank( ii % t->nk, t->k_displs, t->size ),
            .val = L_full_q[ii],
        };
        #pragma omp atomic
        t->send_counts[t->send_pack[tt].dest_rank]++;
    }
    // set the send displs
    memset( t->send_displs, 0, sizeof(index_t) * t->size );
    for (int r=1; r<t->size; ++r)
        t->send_displs[r] = t->send_displs[r-1] + t->send_counts[r-1];

    // gimme those indices
    diverge_mpi_alltoall_index( t->send_counts, t->recv_counts, 1 );

    // prepare recv displs
    memset( t->recv_displs, 0, sizeof(index_t) * t->size );
    for (int r=1; r<t->size; ++r)
        t->recv_displs[r] = t->recv_displs[r-1] + t->recv_counts[r-1];

    // alloc recv pack, no-op after first iter
    t->n_recv_pack = t->recv_counts[t->size-1] + t->recv_displs[t->size-1];
    t->recv_pack = realloc( t->recv_pack, sizeof(pack_t) * MAX(n_recv_pack_prev, t->n_recv_pack) );

    // sort the send pack
    qsort( t->send_pack, t->n_touched, sizeof(pack_t), &pack_compar );

    // send the pack!
    diverge_mpi_alltoallv_bytes( t->send_pack, t->send_counts, t->send_displs,
                                 t->recv_pack, t->recv_counts, t->recv_displs, sizeof(pack_t) );

    // save the output
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t tt=0; tt<t->n_recv_pack; ++tt)
        L_full_q[t->recv_pack[tt].dest] = t->recv_pack[tt].val;

    // and save *that* result back into L
    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t b=0; b<t->L_orbff_spin; ++b)
    for (index_t qrel=0; qrel<t->my_nk; ++qrel)
        L[IDX2(b, qrel, t->my_nk)] = L_full_q[IDX2(b, t->my_nk_off+qrel, t->nk)];
}
