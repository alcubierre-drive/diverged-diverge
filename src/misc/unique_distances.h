/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "qnan.h"
#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration
typedef struct diverge_model_t diverge_model_t;

/**
 * given a :c:type:`diverge_model_t` with lattice vectors and positions, this
 * function calculates all unique distances. Useful if e.g. interactions of
 * first, second, etc. nearest neighbors should be initialized.
 *
 * :param diverge_model_t* m: model to operate on
 * :param int Rmax: maximum integer distance of lattice vectors. For each
 *                  dimension, the loop walks through
 *                  :math:`-R_\mathrm{max}..R_\mathrm{max}` for the calculation
 *                  of all possible distances
 * :param int dim: dimension of the model. The first ``dim`` lattice vectors are
 *                 considered in the iteration via ``Rmax``.
 *
 * :returns: allocated vector of unique distances terminated by a quiet NaN (see
 *           src/misc/qnan.h). Must be free'd manually using ``free``.
 */
double* diverge_model_unique_distances( diverge_model_t* m, int Rmax, int dim );

/**
 * return the length of the unique distance array by checking for quiet NaN (see
 * src/misc/qnan.h). In practice, unique distances can be obtained as follows:
 *
 * .. sourcecode:: C
 *
 *  diverge_model_t* m = ...;
 *
 *  double* dists = diverge_model_unique_distances( m, 3, 2 ); // for a 2D model
 *  index_t n_dists = diverge_model_unique_distances_length( dists );
 *
 *  // do sth with dists here
 *
 *  free( dists );
 */
static inline index_t diverge_model_unique_distances_length( double* dists ) {
    index_t n_dists = -1; while (!qnan_isnan(dists[++n_dists])) {}
    return n_dists;
}

#ifdef __cplusplus
}
#endif
