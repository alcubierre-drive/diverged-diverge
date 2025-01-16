/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "reorder.h"
#include <assert.h>
#include "../misc/mergesort.h"

static void gen_send_pack( const complex128_t* vertex, const index_t* send_map,
        const dist_info* dist, complex128_t* pack );
static void communicate_packs( const complex128_t* send_pack, const int* send_displs,
        const int* send_counts, complex128_t* recv_pack, const int* recv_displs,
        const int* recv_counts );
static void write_recv_pack( complex128_t* vertex_T, const index_t* recv_map,
        const complex128_t* pack, const dist_info* dist, bool add, double scale );
static inline int rank_of_index( index_t index, const dist_info* dist );

typedef struct {
    index_t local_index;
    index_t remote_index;
    int remote_rank;
} trafo_index;

static void gen_send_pack( const complex128_t* vertex, const index_t* send_map,
        const dist_info* dist, complex128_t* pack ) {
    const index_t start = dist->starts[dist->myrank];
    const index_t stop = dist->stops[dist->myrank];
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=start; i<stop; ++i) {
        const index_t loc = i-start;
        pack[loc] = vertex[send_map[loc] - start];
    }
}

static void communicate_packs( const complex128_t* send_pack, const int* send_displs, const int* send_counts,
                        complex128_t* recv_pack, const int* recv_displs, const int* recv_counts ) {
    diverge_mpi_alltoallv_complex( send_pack, send_counts, send_displs, recv_pack, recv_counts, recv_displs );
}

static void write_recv_pack( complex128_t* vertex_T, const index_t* recv_map,
        const complex128_t* pack, const dist_info* dist, bool add, double scale ) {
    const index_t start = dist->starts[dist->myrank];
    const index_t stop = dist->stops[dist->myrank];
    if (add) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=start; i<stop; ++i) {
            const index_t loc = i - start;
            vertex_T[recv_map[loc] - start] += scale*pack[loc];
        }
    } else {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=start; i<stop; ++i) {
            const index_t loc = i - start;
            vertex_T[recv_map[loc] - start] = scale*pack[loc];
        }
    }
}

void gen_dist_info( dist_info* dist, index_t* displs, index_t* counts, MPI_Comm comm ) {
    dist->nranks = diverge_mpi_comm_size();
    dist->myrank = diverge_mpi_comm_rank();
    dist->starts = calloc(3*dist->nranks, sizeof*dist->starts);
    dist->stops = dist->starts + dist->nranks;
    dist->sizes = dist->stops + dist->nranks;
    memcpy( dist->starts, displs, sizeof*dist->starts * dist->nranks );
    memcpy( dist->sizes, counts, sizeof*dist->sizes * dist->nranks );
    for (int i=0; i<dist->nranks; ++i) dist->stops[i] = dist->starts[i] + dist->sizes[i];
    dist->comm = comm;
}

void free_dist_info( const dist_info* dist ) {
    free(dist->starts);
}

void full_pack_exchange( const complex128_t* vertex, complex128_t* vertex_T,
        const index_t* send_map, complex128_t* send_pack, const int* send_displs, const int* send_counts,
        const index_t* recv_map, complex128_t* recv_pack, const int* recv_displs, const int* recv_counts,
        const dist_info* dist, bool add, double scale ) {
    gen_send_pack( vertex, send_map, dist, send_pack );
    communicate_packs( send_pack, send_displs, send_counts,
                       recv_pack, recv_displs, recv_counts );
    write_recv_pack( vertex_T, recv_map, recv_pack, dist, add, scale );
}

static inline int rank_of_index( index_t index, const dist_info* dist ) {
    for (int i=0; i<dist->nranks; ++i) {
        if ((dist->starts[i] <= index) && (dist->stops[i] > index)) {
            return i;
        }
    }
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
        index_t* recv_map, int* recv_displs, int* recv_counts, const dist_info* dist,
        index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2, const index_t* k3map,
        index_gymnastics_t newindex_tFun, index_gymnastics_t oldindex_tFun ) {
    index_t start = dist->starts[dist->myrank];
    index_t stop = dist->stops[dist->myrank];

    trafo_index* send_map_ = (trafo_index*)calloc((stop-start),sizeof(trafo_index));
    trafo_index* recv_map_ = (trafo_index*)calloc((stop-start),sizeof(trafo_index));

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=start; i<stop; ++i) {
        index_t loc = i - start;
        send_map_[loc].local_index = i;
        send_map_[loc].remote_index = (newindex_tFun)( i, nk, nb, k1m2, k1p2, k3map );
        send_map_[loc].remote_rank = rank_of_index( send_map_[loc].remote_index, dist );

        recv_map_[loc].local_index = i;
        recv_map_[loc].remote_index = (oldindex_tFun)( i, nk, nb, k1m2, k1p2, k3map );
        recv_map_[loc].remote_rank = rank_of_index( recv_map_[loc].remote_index, dist );
    }

    mergesort_remote( send_map_, stop-start, diverge_omp_num_threads(), 0 );
    mergesort_rank( recv_map_, stop-start, diverge_omp_num_threads(), 0 );

    for (int r=0; r<dist->nranks; ++r) {
        send_counts[r] = 0;
        recv_counts[r] = 0;
    }
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t j=0; j<dist->sizes[dist->myrank]; ++j) {
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
    for (index_t j=0; j<dist->sizes[dist->myrank]; ++j) {
        send_map[j] = send_map_[j].local_index;
        recv_map[j] = recv_map_[j].local_index;
    }
    free(send_map_);
    free(recv_map_);

    #ifndef NDEBUG
    index_t dbg = 0;
    index_t size_full = nk*nk*nk*nb*nb*nb*nb;
    for (index_t i=0; i<size_full; ++i) {
        index_t ii = (newindex_tFun)( (oldindex_tFun)( i, nk, nb, k1m2, k1p2, k3map ),
                nk, nb, k1m2, k1p2, k3map  );
        if (ii != i) dbg = i+1; break;
    }
    if (dbg != 0) mpi_err_printf( "not a unique map @ idx %li\n", dbg-1 );
    #endif
}
