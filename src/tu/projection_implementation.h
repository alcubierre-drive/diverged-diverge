/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "diverge_interface.h"
#include "cuda_init.h"

#ifdef __cplusplus
extern "C" {
#endif

void _to_C( complex128_t* proj_vertex, const complex128_t* I_channel,
        const claV2ll_t* rsmap_to, index_t rsmap_to_sz, const index_t* rsmap_from,
        const index_t* mom, const index_t* rsmap_off, const index_t* rsmap_len,
        const complex128_t* proj_prefac, const index_t realsp_size,
        const index_t num_spin, const index_t nkpts_vertex,
        const index_t my_nkpts_vertex, const index_t nr );
void _to_P( complex128_t* proj_vertex, const complex128_t* I_channel,
        const claV2ll_t* rsmap_to, index_t rsmap_to_sz, const index_t* rsmap_from,
        const index_t* mom, const index_t* rsmap_off, const index_t* rsmap_len,
        const complex128_t* proj_prefac, const index_t realsp_size,
        const index_t num_spin, const index_t nkpts_vertex,
        const index_t my_nkpts_vertex, const index_t nr );
void _to_D( complex128_t* proj_vertex, const complex128_t* I_channel,
        const claV2ll_t* rsmap_to, index_t rsmap_to_sz, const index_t* rsmap_from,
        const index_t* mom, const index_t* rsmap_off, const index_t* rsmap_len,
        const complex128_t* proj_prefac, const index_t realsp_size,
        const index_t num_spin, const index_t nkpts_vertex,
        const index_t my_nkpts_vertex, const index_t nr );

#ifdef USE_CUDA
void GPUproj_handler_ft( const cuda_proj_descr_t* cp, const char channel,
        complex128_t* proj_vertex, const complex128_t* I_channel,
        const index_t* rsmap_to, const index_t* rsmap_from, const index_t* mom_in,
        index_t mapsize, const complex128_t* proj_prefac, const index_t realsp_size,
        const index_t num_spin, const index_t my_nkpts_vertex, const size_t n_projpref );
#endif

#ifdef __cplusplus
}
#endif
