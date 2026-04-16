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

struct tu_projection_t {
    cuda_proj_descr_t proj_descr;
    complex128_t* my_prefac_reco;
    complex128_t* my_prefac_proj;

    VECTOR_DECL( claV2ll_t, PC_CP_to )
    index_t* PC_CP_displ; // (PC_CP_to_sz,) describes _padded, _from, _mom
    index_t* PC_CP_count; // (PC_CP_to_sz,)
    VECTOR_DECL( claV2ll_t, PC_CP_to_padded )
    index_t* PC_CP_from; // (PC_CP_to_padded_sz, 2)
    index_t* PC_CP_mom; // (PC_CP_to_padded_sz, 2)

    VECTOR_DECL( claV2ll_t, DC_CD_to )
    index_t* DC_CD_displ; // (DC_CD_to_sz,) describes _padded, _from, _mom
    index_t* DC_CD_count; // (DC_CD_to_sz,)
    VECTOR_DECL( claV2ll_t, DC_CD_to_padded )
    index_t* DC_CD_from; // (DC_CD_to_padded_sz, 2)
    index_t* DC_CD_mom; // (DC_CD_to_padded_sz, 2)

    VECTOR_DECL( claV2ll_t, D_to_P_to )
    index_t* D_to_P_displ; // (D_to_P_to_sz,) describes _padded, _from, _mom
    index_t* D_to_P_count; // (D_to_P_to_sz,)
    VECTOR_DECL( claV2ll_t, D_to_P_to_padded )
    index_t* D_to_P_from; // (D_to_P_to_padded_sz, 2)
    index_t* D_to_P_mom; // (D_to_P_to_padded_sz, 2)

    VECTOR_DECL( claV2ll_t, P_to_D_to )
    index_t* P_to_D_displ; // (P_to_D_to_sz,) describes _padded, _from, _mom
    index_t* P_to_D_count; // (P_to_D_to_sz,)
    VECTOR_DECL( claV2ll_t, P_to_D_to_padded )
    index_t* P_to_D_from; // (P_to_D_to_padded_sz, 2)
    index_t* P_to_D_mom; // (P_to_D_to_padded_sz, 2)

    index_t nk[3];
    index_t n_orb;
    index_t n_bonds;
    index_t n_orbff;
    const index_t* bond_sizes;
    const index_t* bond_offsets;
    const index_t* ob_to_orbff;
    const tu_formfactor_t* tu_ff;

    claV3i_t* existing_r;
    index_t nr;

    VECTOR_DECL( index_t, PC_r )
    VECTOR_DECL( index_t, DC_r )
    VECTOR_DECL( index_t, DP_r )
    VECTOR_DECL( index_t, PD_r )
};
typedef struct tu_projection_t tu_projection_t;

void tu_projection_configure_implementation( int hashtable );
void tu_projection_configure_permutation( int inplace );
tu_projection_t* tu_projection_init( const diverge_model_t* model );
void tu_projection_free( tu_projection_t* p );

#ifdef __cplusplus
}
#endif
