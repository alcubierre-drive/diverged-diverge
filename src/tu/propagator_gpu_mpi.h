/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../misc/bitary.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    index_t dest;
    int dest_rank, orig_rank;
    complex128_t val;
} pack_t;

typedef struct tu_prop_gpu_mpi_dist_t tu_prop_gpu_mpi_dist_t;

// must take non-null pointers p_recv_pack and p_n_recv_pack. The *values* may
// be NULL/0, respectively. Depending on whether this is the case, the function
// *allocates* or *reallocates* the recv_pack array in order to keep the memory
// usage minimal.
//
// TODO(LK) we should implement specific communication datastructures, s.t. only
// as much data as needed is communicated (reduction to 3/4 in size of send_pack
// and recv_pack).
tu_prop_gpu_mpi_dist_t* tu_prop_gpu_mpi_dist_init( index_t my_nk,
        index_t my_nk_off, index_t nk, index_t L_orbff_spin,
        pack_t** p_recv_pack, index_t* p_n_recv_pack );
void tu_prop_gpu_mpi_dist_free( tu_prop_gpu_mpi_dist_t* t );
void tu_prop_gpu_mpi_dist_comm( tu_prop_gpu_mpi_dist_t* t, complex128_t* L );

cbyte_t* tu_prop_gpu_mpi_dist_cuda_kernel_bitary( tu_prop_gpu_mpi_dist_t* t );

index_t tu_prop_gpu_mpi_dist_timing_num( const tu_prop_gpu_mpi_dist_t* t );
double tu_prop_gpu_mpi_dist_timing( const tu_prop_gpu_mpi_dist_t* t, index_t i );
const char* tu_prop_gpu_mpi_dist_timing_name( const tu_prop_gpu_mpi_dist_t* t, index_t i );

void sort_pack( pack_t* start, pack_t* end );

#ifdef __cplusplus
}
#endif

