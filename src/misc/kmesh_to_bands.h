/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration
typedef struct diverge_model_t diverge_model_t;

/**
 * finds fine mesh points (as indices) that lie on the segments that define the
 * irreducible path. Useful for, e.g., generation of the Hamiltonian (or
 * self-energy) along the irreducible path, or outputting a susceptibility.
 * Written such that the runtime is O(nk * n_ibz_path). Requires a previous call
 * to :c:func:`diverge_model_internals_common` in order to prepare the momentum
 * meshes.
 *
 * :param diverge_model_t m: the model to use (with bandstructure path defined in
 *                           :c:member:`diverge_model_t.ibz_path` and
 *                           :c:member:`diverge_model_t.n_ibz_path`)
 * :param index_t* indices: pointer to result array that is allocated and filled
 *                          with the indices that lie on the ibz path. indices
 *                          refer to fine mesh.
 * :param index_t n_indices: pointer to length of result array
 *
 * :returns: The number of points per segment in an array. both the returned
 *           array as well as the ``*indices`` array must be free'd manually.
 *
 * **Example**: :c:func:`diverge_kmesh_to_bands` or
 * :c:func:`diverge_kmesh_to_bands_crs` can be used as follows:
 *
 * .. sourcecode:: C
 *
 *  diverge_model_t* m;
 *  // fill m with appropriate content
 *  diverge_model_internals_common( m );
 *  index_t* pts;
 *  index_t n_pts;
 *  index_t* n_per_segment = diverge_kmesh_to_bands( m, &pts, &n_pts );
 *  // do something with pts, n_pts, and n_per_segment.
 *  free( pts );
 *  free( n_per_segment );
 */
index_t* diverge_kmesh_to_bands( diverge_model_t* m, index_t** indices, index_t* n_indices );

/** same as :c:func:`diverge_kmesh_to_bands`, but for the coarse mesh. */
index_t* diverge_kmesh_to_bands_crs( diverge_model_t* m, index_t** indices, index_t* n_indices );

#ifdef __cplusplus
}
#endif
