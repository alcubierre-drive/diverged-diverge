/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_lattice_t diverge_lattice_t;
/** struct for lattice vectors, required only to be used as return value */
struct diverge_lattice_t {
    /** the data, as 3x3 double array, with the first lattice vector being ``v[0]`` and so on. */
    double v[3][3];
};

/**
 * mesh generation helper function, useful for your own models
 *
 * :param double* kmesh: array to store the coarse momentum mesh in, dimension
 *                       (nk[0],nk[1],nk[2],3).
 * :param double* kfmesh: array to store the fine momentum mesh in, dimension
 *                        (nk[0]*nkf[0],nk[1]*nkf[1],nk[2]*nkf[2],3).
 *
 * :reutrns diverge_lattice_t: the reciprocal lattice vectors
 */
diverge_lattice_t diverge_model_generate_meshes(double* kmesh, double* kfmesh,
        const index_t nk[3], const index_t nkf[3], double lattice[3][3]);

/**
 * use lattice vectors to generate reciprocal lattice vectors. corrects for
 * left-handed coordinate systems.
 *
 * :param double[3][3] lattice: lattice vectors in C ordering (see
 *                              :c:member:`diverge_model_t.lattice`), input & output.
 * :param double[3][3] kbasis: reciprocal lattice vectors in C ordering, output.
 */
void diverge_model_generate_mom_basis(double* lattice, double* kbasis);

#ifdef __cplusplus
}
#endif


