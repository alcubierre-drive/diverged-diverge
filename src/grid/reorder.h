/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../misc/mpi_functions.h"
#include "../diverge_common.h"
#include "index_gymnastics.h"
#include "../misc/mergesort.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dist_info_t {
    index_t *qcounts, *qdispls,
            *counts, *displs;
    int myrank, nranks;
} dist_info_t;

dist_info_t* dist_info_init( index_t nk, index_t nb );
void dist_info_free( dist_info_t* dist );

void full_pack_exchange( const complex128_t* vertex, complex128_t* vertex_T,
        const index_t* send_map, complex128_t* send_pack, const int* send_displs, const int* send_counts,
        const index_t* recv_map, complex128_t* recv_pack, const int* recv_displs, const int* recv_counts,
        const dist_info_t* dist, bool add, double scale );

// implementation
typedef struct {
    index_t local_index;
    index_t remote_index;
    int remote_rank;
} trafo_index;

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( \
    ((x).remote_index - (y).remote_index) < 0 ? -1 : \
    ((x).remote_index - (y).remote_index) > 0 ? 1 : 0 \
)
MERGESORT_IMPL( mergesort_remote, trafo_index )
#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( (x).remote_rank - (y).remote_rank )
MERGESORT_IMPL( mergesort_rank, trafo_index )

static inline int rank_of_index( index_t index, const dist_info_t* dist ) {
    for (int i=0; i<dist->nranks; ++i)
        if ((dist->displs[i] <= index) && ((dist->counts[i] + dist->displs[i]) > index))
            return i;
    return -1;
}

#define GEN_MAPS_IMPL( name, newindex_tFun, oldindex_tFun ) \
static inline void gen_maps_##name( index_t* send_map, int* send_displs, int* send_counts, \
        index_t* recv_map, int* recv_displs, int* recv_counts, const dist_info_t* dist, \
        index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 ) { \
    const index_t displ = dist->displs[dist->myrank], \
                  count = dist->counts[dist->myrank]; \
 \
    trafo_index* send_map_ = calloc(count,sizeof(trafo_index)); \
    trafo_index* recv_map_ = calloc(count,sizeof(trafo_index)); \
 \
    _Pragma("omp parallel for num_threads(diverge_omp_num_threads())") \
    for (index_t loc=0; loc<count; ++loc) { \
        send_map_[loc].local_index = loc + displ; \
        send_map_[loc].remote_index = (newindex_tFun)( loc + displ, nk, nb, k1m2, k1p2 ); \
        send_map_[loc].remote_rank = rank_of_index( send_map_[loc].remote_index, dist ); \
 \
        recv_map_[loc].local_index = loc + displ; \
        recv_map_[loc].remote_index = (oldindex_tFun)( loc + displ, nk, nb, k1m2, k1p2 ); \
        recv_map_[loc].remote_rank = rank_of_index( recv_map_[loc].remote_index, dist ); \
    } \
 \
    mergesort_remote( send_map_, count, diverge_omp_num_threads(), 0 ); \
    mergesort_rank( recv_map_, count, diverge_omp_num_threads(), 0 ); \
 \
    for (int r=0; r<dist->nranks; ++r) { \
        send_counts[r] = 0; \
        recv_counts[r] = 0; \
    } \
    _Pragma("omp parallel num_threads(diverge_omp_num_threads())") \
    { \
        int* send_counts_local = calloc(dist->nranks, sizeof*send_counts_local); \
        int* recv_counts_local = calloc(dist->nranks, sizeof*recv_counts_local); \
        _Pragma("omp for") \
        for (index_t j=0; j<dist->counts[dist->myrank]; ++j) { \
            send_counts_local[send_map_[j].remote_rank]++; \
            recv_counts_local[recv_map_[j].remote_rank]++; \
        } \
        for (index_t r=0; r<dist->nranks; ++r) { \
            _Pragma("omp atomic") \
            send_counts[r] += send_counts_local[r]; \
            _Pragma("omp atomic") \
            recv_counts[r] += recv_counts_local[r]; \
        } \
        free( send_counts_local ); \
        free( recv_counts_local ); \
    } \
    send_displs[0] = 0; \
    recv_displs[0] = 0; \
    for (int r=1; r<dist->nranks; ++r) { \
        send_displs[r] = send_displs[r-1] + send_counts[r-1]; \
        recv_displs[r] = recv_displs[r-1] + recv_counts[r-1]; \
    } \
 \
    _Pragma("omp parallel for num_threads(diverge_omp_num_threads())") \
    for (index_t j=0; j<dist->counts[dist->myrank]; ++j) { \
        send_map[j] = send_map_[j].local_index; \
        recv_map[j] = recv_map_[j].local_index; \
    } \
    free(send_map_); \
    free(recv_map_); \
    /* \
    #ifndef NDEBUG \
    index_t dbg = 0; \
    index_t size_full = nk*nk*nk*nb*nb*nb*nb; \
    for (index_t i=0; i<size_full; ++i) { \
        index_t ii = (newindex_tFun)( (oldindex_tFun)( i, nk, nb, k1m2, k1p2 ), \
                nk, nb, k1m2, k1p2  ); \
        if (ii != i) { dbg = i+1; break; } \
    } \
    if (dbg != 0) mpi_err_printf( "not a unique map @ idx %li\n", dbg-1 ); \
    #endif \
    */ \
}

#ifdef __cplusplus
}
#endif
