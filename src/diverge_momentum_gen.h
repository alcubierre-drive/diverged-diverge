/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * mesh generation helper function, useful for your own models
 *
 * :param double* kmesh: array to store the coarse momentum mesh in, dimension
 *                       (nk[0],nk[1],nk[2],3).
 * :param double* kfmesh: array to store the fine momentum mesh in, dimension
 *                        (nk[0]*nkf[0],nk[1]*nkf[1],nk[2]*nkf[2],3).
 */
void diverge_model_generate_meshes(double* kmesh, double* kfmesh,
        const index_t nk[3], const index_t nkf[3], double lattice[3][3]);

/**
 * use lattice vectors to generate reciprocal lattice vectors. corrects for
 * left-handed coordinate systems.
 *
 * :param double[3][3] lattice: lattice vectors in C ordering (see
 *                              :c:member:`diverge_model_t.lattice`), input & output.
 * :param double[3][3] kbasis: reciprocal lattice vectors in C ordering, output.
 */
void diverge_model_generate_mom_basis(double lattice[3][3], double kbasis[3][3]);

#ifdef __cplusplus
}
#endif


