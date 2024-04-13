#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_prop_gpu_mpi_dist_t tu_prop_gpu_mpi_dist_t;

tu_prop_gpu_mpi_dist_t* tu_prop_gpu_mpi_dist_init( index_t my_nk,
        index_t my_nk_off, index_t nk, index_t L_orbff_spin );
void tu_prop_gpu_mpi_dist_free( tu_prop_gpu_mpi_dist_t* t );
void tu_prop_gpu_mpi_dist_comm( tu_prop_gpu_mpi_dist_t* t, complex128_t* L_full_q, complex128_t* L );

typedef struct {
    index_t dest;
    int dest_rank, orig_rank;
    complex128_t val;
} pack_t;

index_t tu_prop_gpu_mpi_dist_timing_num( const tu_prop_gpu_mpi_dist_t* t );
double tu_prop_gpu_mpi_dist_timing( const tu_prop_gpu_mpi_dist_t* t, index_t i );
const char* tu_prop_gpu_mpi_dist_timing_name( const tu_prop_gpu_mpi_dist_t* t, index_t i );

void sort_pack( pack_t* start, pack_t* end );

#ifdef __cplusplus
}
#endif

