/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_symmstruct_t tu_symmstruct_t;
typedef struct diverge_model_t diverge_model_t;

typedef struct {
    index_t sym;
    index_t isym;
} tu_whichsym_t;

struct tu_symmstruct_t {
    // if true, the vectors are filled
    bool use_symmetries;
    // things needed for the generation of the maps
    index_t* inv_symmetry;
    index_t n_sym;

    VECTOR_DECL( complex128_t, mi_prefac_in_out )
    VECTOR_DECL( complex128_t, mi_prefac_in_in )

    VECTOR_DECL( index_t, mi_maps_to )
    index_t *mi_map_len, *mi_map_off;
    index_t mi_cnt;

    VECTOR_DECL( double, beyond_UC )

    // general properties of the IBZ
    index_t* kmaps_to; // nk
    index_t* idx_ibz_in_fullmesh; // nkibz
    tu_whichsym_t* which_symmetry_used; // nk
    index_t nk, nkibz;

    VECTOR_DECL( complex128_t, prefactor_symmP )
    VECTOR_DECL( complex128_t, prefactor_symmCD )
    VECTOR_DECL( index_t, idx_map_symm )

    index_t *o2m_map_len, *o2m_map_off;
    index_t o2m_map_sz;

    // only required for MPI projections using symmetries
    index_t *from_ibz_2_bz, *from_ibz_2_bz_off, *from_ibz_2_bz_len;
    index_t my_nk_in_bz;
    index_t my_nk_in_bz_off;
};

void tu_symmstruct_delete( tu_symmstruct_t* symm );
void tu_init_symmetries( diverge_model_t* model );

#ifdef __cplusplus
}
#endif
