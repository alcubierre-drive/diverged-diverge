/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Function: diverge_model_generate_meshes
// mesh generation helper function, useful for your own models. dimension of
// buffers:
//
// kmesh(nk[0]*nk[1]*nk[2],3), kfmesh(nk[0]*nk[1]*nk[2]*nkf[0]*nkf[1]*nkf[2],3)
void diverge_model_generate_meshes(double* kmesh, double* kfmesh,
        const index_t nk[3], const index_t nkf[3], double lattice[3][3]);
// Function: diverge_model_generate_mom_basis
// use lattice vectors to generate reciprocal lattice vectors. corrects for
// left-handed coordinate systems.
void diverge_model_generate_mom_basis(double lattice[3][3], double kbasis[3][3]);

#ifdef __cplusplus
}
#endif


