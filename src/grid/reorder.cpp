/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "reorder.hpp"
#include <vector>
#include <cassert>
#include <cstring>

#include <algorithm>
#ifdef USE_BOOST_SORT

// ignore boost warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <boost/sort/sort.hpp>
#pragma GCC diagnostic pop
#define _CUSTOM_SORT_(begin, end, comp) boost::sort::block_indirect_sort(begin, end, comp, diverge_omp_num_threads())
#define _CUSTOM_STABLE_SORT_(begin, end, comp) boost::sort::parallel_stable_sort(begin, end, comp, diverge_omp_num_threads())

#else // USE_BOOST_SORT

#define _CUSTOM_SORT_(begin, end, comp) std::sort(begin, end, comp)
#define _CUSTOM_STABLE_SORT_(begin, end, comp) std::stable_sort(begin, end, comp)

#endif // USE_BOOST_SORT

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

// no boost warning about negative size_t stuff on intel compilers
#ifdef __INTEL_COMPILER
#pragma warning disable 68
#endif
void sort_trafo_index_remote( trafo_index* start, trafo_index* stop ) {
    _CUSTOM_SORT_( start, stop, []( const trafo_index& a, const trafo_index& b ) {
                return a.remote_index<b.remote_index;
            } );
}

void sort_trafo_index_rank( trafo_index* start, trafo_index* stop ) {
    _CUSTOM_STABLE_SORT_( start, stop, []( const trafo_index& a, const trafo_index& b ) {
                return a.remote_rank<b.remote_rank;
            } );
}

}
