/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../misc/mpi_functions.h"
#include "../diverge_common.h"
#include "index_gymnastics.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dist_info {
    index_t* starts;
    index_t* stops;
    index_t* sizes;
    int nranks;
    int myrank;
    MPI_Comm comm;
};
typedef struct dist_info dist_info;

void gen_dist_info( dist_info* dist, index_t* vertex_displs, index_t* vertex_counts, MPI_Comm comm );
void free_dist_info( const dist_info* dist );

void full_pack_exchange( const complex128_t* vertex, complex128_t* vertex_T,
        const index_t* send_map, complex128_t* send_pack, const int* send_displs, const int* send_counts,
        const index_t* recv_map, complex128_t* recv_pack, const int* recv_displs, const int* recv_counts,
        const dist_info* dist, bool add, double scale );

void gen_maps( index_t* send_map, int* send_displs, int* send_counts,
        index_t* recv_map, int* recv_displs, int* recv_counts, const dist_info* dist,
        index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2, const index_t* k3map,
        index_gymnastics_t newindex_tFun, index_gymnastics_t oldindex_tFun );

#ifdef __cplusplus
}
#endif
