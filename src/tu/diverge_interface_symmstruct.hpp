/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "global.hpp"

struct symmstruct_t{
    // if true, the vectors are filled
    bool use_symmetries;
    //things needed for the generation of the maps;
    vector<index_t> inv_symmetry;

    vector<complex128_t> mi_prefac_in_out;
    vector<complex128_t> mi_prefac_in_in;
    vector<index_t> mi_maps_to;
    vector<index_t> mi_map_len;
    vector<index_t> mi_map_off;
    vector<Vec3d> beyond_UC;

    // general properties of the IBZ
    vector<index_t> kmaps_to;
    vector<index_t> idx_ibz_in_fullmesh;
    vector<array<index_t,2>> which_symmetry_used;

    vector<complex128_t> prefactor_symmP;
    vector<complex128_t> prefactor_symmCD;
    vector<index_t> idx_map_symm;

    vector<index_t> o2m_map_len;
    vector<index_t> o2m_map_off;

    // only required for MPI projections using symmetries
    vector<index_t> from_ibz_2_bz;
    vector<index_t> from_ibz_2_bz_off;
    vector<index_t> from_ibz_2_bz_len;
    index_t my_nk_in_bz;
    index_t my_nk_in_bz_off;
};

// allocates *ptr using realloc, i.e., must be free'd by user with free.
void symmstruct_serialize( const symmstruct_t* symm, char** ptr, uindex_t* nbytes );
// allocates symmstruct_t* with new, i.e., must be free'd by user with delete.
symmstruct_t* symmstruct_deserialize( const char* ptr );

// checks whether two symmtructs are equal, ignores my_* members
int symmstruct_equal( const symmstruct_t* A, const symmstruct_t* B );
