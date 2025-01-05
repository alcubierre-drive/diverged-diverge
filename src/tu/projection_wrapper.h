/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "diverge_interface.h"
#include "cuda_init.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Projection {
    cuda_proj_descr_t proj_descr;
    complex128_t* my_prefac_reco;
    complex128_t* my_prefac_proj;
    complex128_t* my_prefac_proj_rk;

    VECTOR_DECL( claV2ll_t, PC_CP_t )
    index_t* PC_CP_f;
    index_t* PC_CP_t_padded;
    VECTOR_DECL( index_t, PC_CP_mom )
    index_t* PC_CP_d;
    index_t* PC_CP_l;

    VECTOR_DECL( claV2ll_t, DC_CD_t )
    index_t* DC_CD_f;
    index_t* DC_CD_t_padded;
    VECTOR_DECL( index_t, DC_CD_mom )
    index_t* DC_CD_d;
    index_t* DC_CD_l;

    VECTOR_DECL( claV2ll_t, D_to_P_t )
    index_t* D_to_P_f;
    index_t* D_to_P_t_padded;
    VECTOR_DECL( index_t, D_to_P_mom )
    index_t* D_to_P_d;
    index_t* D_to_P_l;

    VECTOR_DECL( claV2ll_t, P_to_D_t )
    index_t* P_to_D_f;
    index_t* P_to_D_t_padded;
    VECTOR_DECL( index_t, P_to_D_mom )
    index_t* P_to_D_d;
    index_t* P_to_D_l;

    index_t PC_CP_f_sz, DC_CD_f_sz, D_to_P_f_sz, P_to_D_f_sz;

    index_t nk[3];
    index_t nr;
    index_t n_orb;
    index_t n_bonds;
    index_t n_orbff;
    const index_t* bond_sizes;
    const index_t* bond_offsets;
    const index_t* ob_to_orbff;
    const tu_formfactor_t* tu_ff;

    VECTOR_DECL( claV3i_t, existing_r )
    VECTOR_DECL( index_t, PC_r )
    VECTOR_DECL( index_t, DC_r )
    VECTOR_DECL( index_t, DP_r )
    VECTOR_DECL( index_t, PD_r )
};
typedef struct Projection Projection;

Projection* projection_init( const diverge_model_t* model );
void projection_free( Projection* p );

#ifdef __cplusplus
}
#endif
