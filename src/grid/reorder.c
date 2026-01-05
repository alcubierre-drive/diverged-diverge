/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "reorder.h"
#include "../misc/mergesort.h"

dist_info_t* dist_info_init( index_t nk, index_t nb ) {
    dist_info_t* d = calloc(1, sizeof*d);
    d->myrank = diverge_mpi_comm_rank();
    d->nranks = diverge_mpi_comm_size();
    d->qcounts = diverge_mpi_distribute(nk);
    d->qdispls = d->qcounts + d->nranks;
    d->counts = calloc(d->nranks, sizeof*d->qcounts * 2);
    for (int i=0; i<d->nranks*2; ++i) d->counts[i] = d->qcounts[i] * POW2(nk*POW2(nb));
    d->displs = d->counts + d->nranks;
    return d;
}

void dist_info_free( dist_info_t* d ) {
    free( d->qcounts );
    free( d->counts );
    free( d );
}

static void gen_send_pack( const complex128_t* vertex, const index_t* send_map,
        const dist_info_t* dist, complex128_t* pack );
static void communicate_packs( const complex128_t* send_pack, const int* send_displs,
        const int* send_counts, complex128_t* recv_pack, const int* recv_displs,
        const int* recv_counts );
static void write_recv_pack( complex128_t* vertex_T, const index_t* recv_map,
        const complex128_t* pack, const dist_info_t* dist, bool add, double scale );
static inline int rank_of_index( index_t index, const dist_info_t* dist );

typedef struct {
    index_t local_index;
    index_t remote_index;
    int remote_rank;
} trafo_index;

static void gen_send_pack( const complex128_t* vertex, const index_t* send_map,
        const dist_info_t* dist, complex128_t* pack ) {
    const index_t displ = dist->displs[dist->myrank], count = dist->counts[dist->myrank];
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t loc=0; loc<count; ++loc)
        pack[loc] = vertex[send_map[loc] - displ];
}

static void communicate_packs( const complex128_t* send_pack, const int* send_displs, const int* send_counts,
                        complex128_t* recv_pack, const int* recv_displs, const int* recv_counts ) {
    diverge_mpi_alltoallv_complex( send_pack, send_counts, send_displs, recv_pack, recv_counts, recv_displs );
}

static void write_recv_pack( complex128_t* vertex_T, const index_t* recv_map,
        const complex128_t* pack, const dist_info_t* dist, bool add, double scale ) {
    const index_t displ = dist->displs[dist->myrank], count = dist->counts[dist->myrank];
    if (add) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t loc=0; loc<count; ++loc)
            vertex_T[recv_map[loc] - displ] += scale*pack[loc];
    } else {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t loc=0; loc<count; ++loc)
            vertex_T[recv_map[loc] - displ] = scale*pack[loc];
    }
}

void full_pack_exchange( const complex128_t* vertex, complex128_t* vertex_T,
        const index_t* send_map, complex128_t* send_pack, const int* send_displs, const int* send_counts,
        const index_t* recv_map, complex128_t* recv_pack, const int* recv_displs, const int* recv_counts,
        const dist_info_t* dist, bool add, double scale ) {
    gen_send_pack( vertex, send_map, dist, send_pack );
    communicate_packs( send_pack, send_displs, send_counts,
                       recv_pack, recv_displs, recv_counts );
    write_recv_pack( vertex_T, recv_map, recv_pack, dist, add, scale );
}

static inline int rank_of_index( index_t index, const dist_info_t* dist ) {
    for (int i=0; i<dist->nranks; ++i)
        if ((dist->displs[i] <= index) && ((dist->counts[i] + dist->displs[i]) > index))
            return i;
    return -1;
}

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( \
    ((x).remote_index - (y).remote_index) < 0 ? -1 : \
    ((x).remote_index - (y).remote_index) > 0 ? 1 : 0 \
)
MERGESORT_IMPL( mergesort_remote, trafo_index )
#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( (x).remote_rank - (y).remote_rank )
MERGESORT_IMPL( mergesort_rank, trafo_index )

void gen_maps( index_t* send_map, int* send_displs, int* send_counts,
        index_t* recv_map, int* recv_displs, int* recv_counts, const dist_info_t* dist,
        index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2,
        index_gymnastics_t newindex_tFun, index_gymnastics_t oldindex_tFun ) {
    const index_t displ = dist->displs[dist->myrank],
                  count = dist->counts[dist->myrank];

    trafo_index* send_map_ = calloc(count,sizeof(trafo_index));
    trafo_index* recv_map_ = calloc(count,sizeof(trafo_index));

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t loc=0; loc<count; ++loc) {
        send_map_[loc].local_index = loc + displ;
        send_map_[loc].remote_index = (newindex_tFun)( loc + displ, nk, nb, k1m2, k1p2 );
        send_map_[loc].remote_rank = rank_of_index( send_map_[loc].remote_index, dist );

        recv_map_[loc].local_index = loc + displ;
        recv_map_[loc].remote_index = (oldindex_tFun)( loc + displ, nk, nb, k1m2, k1p2 );
        recv_map_[loc].remote_rank = rank_of_index( recv_map_[loc].remote_index, dist );
    }

    mergesort_remote( send_map_, count, diverge_omp_num_threads(), 0 );
    mergesort_rank( recv_map_, count, diverge_omp_num_threads(), 0 );

    for (int r=0; r<dist->nranks; ++r) {
        send_counts[r] = 0;
        recv_counts[r] = 0;
    }
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t j=0; j<dist->counts[dist->myrank]; ++j) {
        #pragma omp atomic
        send_counts[send_map_[j].remote_rank]++;
        #pragma omp atomic
        recv_counts[recv_map_[j].remote_rank]++;
    }
    send_displs[0] = 0;
    recv_displs[0] = 0;
    for (int r=1; r<dist->nranks; ++r) {
        send_displs[r] = send_displs[r-1] + send_counts[r-1];
        recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
    }

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t j=0; j<dist->counts[dist->myrank]; ++j) {
        send_map[j] = send_map_[j].local_index;
        recv_map[j] = recv_map_[j].local_index;
    }
    free(send_map_);
    free(recv_map_);

    #ifndef NDEBUG
    index_t dbg = 0;
    index_t size_full = nk*nk*nk*nb*nb*nb*nb;
    for (index_t i=0; i<size_full; ++i) {
        index_t ii = (newindex_tFun)( (oldindex_tFun)( i, nk, nb, k1m2, k1p2 ),
                nk, nb, k1m2, k1p2  );
        if (ii != i) { dbg = i+1; break; }
    }
    if (dbg != 0) mpi_err_printf( "not a unique map @ idx %li\n", dbg-1 );
    #endif
}
