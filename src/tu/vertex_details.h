/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "vertex.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../misc/mpi_functions.h"
#include "../misc/gemm.h"
#include "diverge_interface.h"
#include "projection.h"
#include "projection_handler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct transpose_plan_t transpose_plan_t;

struct tu_vertex_t {
    complex128_t *Pch, *Cch, *Dch;
    complex128_t *pp_bubble_int, *ph_bubble_int;
    complex128_t *self_en;

    diverge_model_t* model;
    const tu_data_t* tu_data;
    index_t n_orbff;
    index_t n_orb;
    index_t n_bonds;
    index_t n_spin;
    index_t nk;
    index_t nkf;
    index_t my_nk;
    index_t my_nk_off;
    bool SU2;
    index_t full_vert_size;
    index_t full_bubble_size;
    index_t selfenergy_size;
    const tu_symmstruct_t* symm;

    char* GPU_memory_pool[32];
    char* GPU_memory_pool_managed[32];
    char* GPU_gemm_buffer[32][3];
    bool has_gpu_memory;
    single_gemm_Xt_t* GPU_reco_gemm;
    transpose_plan_t* transpose_plan;

    bool P_flow, C_flow, D_flow;
    bool C_to_P, D_to_P, P_to_C, D_to_C, P_to_D, C_to_D;

    bool use_self;

    // helper arrays
    complex128_t* bubble_helper;
    complex128_t* projection_helper;
    complex128_t* product_helper;

    // private
    bool owns;
};

// private functions
void tu_vertex_init_U_from_fullV( tu_vertex_t* V, diverge_model_t* model, tu_projection_t* proj );
void tu_vertex_initialize_GPU_memory( tu_vertex_t* V, tu_projection_t* proj, tu_prop_t* loop );
void tu_vertex_init_U( tu_vertex_t* V, bool enforce_exchange );
void tu_vertex_init_U_ff( tu_vertex_t* V, bool enforce_exchange );
void tu_vertex_project_fullV( const tu_vertex_t* V, const diverge_model_t* mod_ptr, complex128_t* fullV );

#ifdef __cplusplus
}
#endif
