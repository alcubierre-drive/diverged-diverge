/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../misc/mpi_functions.h"
#include "../diverge_common.h"
#include "index_gymnastics.h"

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

void gen_maps( index_t* send_map, int* send_displs, int* send_counts,
        index_t* recv_map, int* recv_displs, int* recv_counts, const dist_info_t* dist,
        index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2,
        index_gymnastics_t newindex_tFun, index_gymnastics_t oldindex_tFun );

#ifdef __cplusplus
}
#endif
