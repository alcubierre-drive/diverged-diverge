/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../misc/mpi_functions.h"
#include "diverge_interface_symmstruct.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __LINALG_H__
typedef struct claV3i_t { int v[3]; } claV3i_t;
typedef struct claV2ll_t { index_t v[2]; } claV2ll_t;
#endif//__LINALG_H__

typedef enum {
    tu_fft_algorithm_greedy,
    tu_fft_algorithm_split,
    tu_fft_algorithm_mpi,
    tu_fft_algorithm_mpi_new,
} tu_fft_algorithm_t;

typedef struct tu_data_t tu_data_t;

struct tu_data_t {
    symmstruct_t* symm;
    bool use_reduced_loop;
    bool use_cuda_mpi_loop;
    index_t nk;
    index_t nkf;
    index_t nktot;
    index_t nkibz;

    index_t my_nk;
    index_t my_nk_off;

    index_t n_orbff;
    index_t n_bonds;
    index_t* bond_sizes;
    index_t* bond_offsets;
    index_t* ob_to_orbff;

    index_t* mi_to_ofrom;
    index_t* mi_to_oto;
    index_t* mi_to_ffidx;
    index_t* mi_to_R;
    index_t* mi_to_tuffidx;

    double offload_cpu;
    double percent_mem_GPU_blocked;
    bool tu_selfenergy_flow;

    index_t mpi_fft_chunksize;
    tu_fft_algorithm_t which_fft_simple,
                       which_fft_greens,
                       which_fft_red;
    bool propagator_extra_cpu_timings;

    // reduction of computational cost of the loop
    VECTOR_DECL( claV3i_t, existing_ff )
    VECTOR_DECL( claV3i_t, existing_ff_fine )

    VECTOR_DECL( claV3i_t, unique_ff_combi )
    index_t* ff_idx_to_unique_ff_elem;

    VECTOR_DECL( index_t, unique_mi_l )
    VECTOR_DECL( index_t, unique_mi_r )
    VECTOR_DECL( index_t, lmap_l )
    VECTOR_DECL( index_t, lmap_r )
    VECTOR_DECL( index_t, lmap_off )
    VECTOR_DECL( index_t, lmap_len )
};

// tu info struct contains all relevant informations for the TU code
// which do not strictly belong to the model
void init_MPI_dist( diverge_model_t* model );
void init_ff_struct( const double max_dist, diverge_model_t* model );
void generate_reduced_loop_maps( diverge_model_t* model );

// below functions act as constructor/destructor for tu_data_t
// void diverge_model_internals_tu( diverge_model_t* model, double max_dist );
void tu_data_destructor( void* model );

#ifdef __cplusplus
}
#endif
