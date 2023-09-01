#pragma once


#include "typedefs.hpp"

namespace grid {

struct dist_info {
    index_t* starts;
    index_t* stops;
    index_t* sizes;
    int nranks;
    int myrank;
    MPI_Comm comm;
};
typedef struct dist_info dist_info;

template<class IdxFunTypeNew, class IdxFunTypeOld>
void gen_maps( index_t* send_map, int* send_displs, int* send_counts,
        index_t* recv_map, int* recv_displs, int* recv_counts,
        const dist_info* dist, index_t nk, index_t nb, IdxFunTypeNew newindex_tFun,
        IdxFunTypeOld oldindex_tFun );

void gen_dist_info( dist_info* dist, index_t* vertex_starts, index_t* vertex_stops,
        MPI_Comm comm );
void free_dist_info( const dist_info* dist );

void full_pack_exchange( const complex128_t* vertex, complex128_t* vertex_T,
        const index_t* send_map, complex128_t* send_pack, const int* send_displs, const int* send_counts,
        const index_t* recv_map, complex128_t* recv_pack, const int* recv_displs, const int* recv_counts,
        const dist_info* dist, bool add, double scale, MPI_Comm comm );

int rank_of_index( index_t index, const dist_info* dist );

typedef struct {
    index_t local_index;
    index_t remote_index;
    int remote_rank;
} trafo_index;

void sort_trafo_index_remote( trafo_index* start, trafo_index* stop );
void sort_trafo_index_rank( trafo_index* start, trafo_index* stop );

template<typename IdxFunTypeNew, typename IdxFunTypeOld>
index_t debug_map( index_t nk, index_t nb, IdxFunTypeNew newindex_tFun, IdxFunTypeOld oldindex_tFun ) {
    index_t size_full = nk*nk*nk*nb*nb*nb*nb;
    for (index_t i=0; i<size_full; ++i) {
        index_t ii = newindex_tFun( oldindex_tFun( i, nk, nb ), nk, nb );
        if (ii != i) return i+1;
    }
    return 0;
}

template<class IdxFunTypeNew, class IdxFunTypeOld>
void gen_maps( index_t* send_map, int* send_displs, int* send_counts,
               index_t* recv_map, int* recv_displs, int* recv_counts,
               const dist_info* dist, index_t nk, index_t nb,
               IdxFunTypeNew newindex_tFun, IdxFunTypeOld oldindex_tFun ) {
    index_t start = dist->starts[dist->myrank];
    index_t stop = dist->stops[dist->myrank];

    trafo_index* send_map_ = (trafo_index*)calloc((stop-start),sizeof(trafo_index));
    trafo_index* recv_map_ = (trafo_index*)calloc((stop-start),sizeof(trafo_index));

    #pragma omp parallel for
    for (index_t i=start; i<stop; ++i) {
        index_t loc = i - start;
        send_map_[loc].local_index = i;
        send_map_[loc].remote_index = newindex_tFun( i, nk, nb );
        send_map_[loc].remote_rank = rank_of_index( send_map_[loc].remote_index, dist );

        recv_map_[loc].local_index = i;
        recv_map_[loc].remote_index = oldindex_tFun( i, nk, nb );
        recv_map_[loc].remote_rank = rank_of_index( recv_map_[loc].remote_index, dist );
    }

    sort_trafo_index_remote( send_map_, send_map_+stop-start );
    sort_trafo_index_rank( recv_map_, recv_map_+stop-start );

    for (int r=0; r<dist->nranks; ++r) {
        send_counts[r] = 0;
        recv_counts[r] = 0;
    }
    #pragma omp parallel for
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

    #pragma omp parallel for
    for (index_t j=0; j<dist->sizes[dist->myrank]; ++j) {
        send_map[j] = send_map_[j].local_index;
        recv_map[j] = recv_map_[j].local_index;
    }
    free(send_map_);
    free(recv_map_);

    #ifndef NDEBUG
    index_t dbg = debug_map<IdxFunTypeNew, IdxFunTypeOld>( nk, nb, newindex_tFun, oldindex_tFun );
    if (dbg != 0) mpi_err_printf( "not a unique map @ idx %li\n", dbg-1 );
    #endif
}

}
