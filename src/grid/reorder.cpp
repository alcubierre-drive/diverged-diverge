/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "reorder.hpp"
#include <vector>
#include <cassert>
#include <cstring>
#include "../misc/mergesort.h"

namespace grid {

static void gen_send_pack( const complex128_t* vertex, const index_t* send_map,
        const dist_info* dist, complex128_t* pack );
static void communicate_packs( const complex128_t* send_pack, const int* send_displs,
        const int* send_counts, complex128_t* recv_pack, const int* recv_displs,
        const int* recv_counts, const dist_info* dist, MPI_Comm comm );
static void write_recv_pack( complex128_t* vertex_T, const index_t* recv_map,
        const complex128_t* pack, const dist_info* dist, bool add, double scale );

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
                        complex128_t* recv_pack, const int* recv_displs, const int* recv_counts,
                        const dist_info*, MPI_Comm comm ) {
    (void)comm;
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

void gen_dist_info( dist_info* dist, index_t* starts, index_t* stops, MPI_Comm comm ) {
    dist->nranks = diverge_mpi_comm_size();
    dist->myrank = diverge_mpi_comm_rank();
    dist->starts = new index_t[dist->nranks];
    dist->stops = new index_t[dist->nranks];
    dist->sizes = new index_t[dist->nranks];
    for (int i=0; i<dist->nranks; ++i) {
        dist->starts[i] = starts[i];
        dist->stops[i] = stops[i];
        dist->sizes[i] = stops[i] - starts[i];
    }
    dist->comm = comm;
}

void free_dist_info( const dist_info* dist ) {
    delete[] dist->starts;
    delete[] dist->stops;
    delete[] dist->sizes;
}

void full_pack_exchange( const complex128_t* vertex, complex128_t* vertex_T,
        const index_t* send_map, complex128_t* send_pack, const int* send_displs, const int* send_counts,
        const index_t* recv_map, complex128_t* recv_pack, const int* recv_displs, const int* recv_counts,
        const dist_info* dist, bool add, double scale, MPI_Comm comm ) {
    gen_send_pack( vertex, send_map, dist, send_pack );
    communicate_packs( send_pack, send_displs, send_counts,
                       recv_pack, recv_displs, recv_counts,
                       dist, comm );
    write_recv_pack( vertex_T, recv_map, recv_pack, dist, add, scale );
}

int rank_of_index( index_t index, const dist_info* dist ) {
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

void sort_trafo_index_remote( trafo_index* start, index_t size ) {
    mergesort_remote( start, size, diverge_omp_num_threads(), 0 );
}

void sort_trafo_index_rank( trafo_index* start, index_t size ) {
    mergesort_rank( start, size, diverge_omp_num_threads(), 0 );
}

}
