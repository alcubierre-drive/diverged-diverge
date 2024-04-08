/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_model.h"

// the interface with energies and bands is chosen because we want to make it
// possible to simulate models with a correlated subspace, i.e. more bands in
// the dispersion than in the vertex. generating a patching for these models
// thus requires the energy array E as a separate piece of information, as well
// as the number of bands nb.

#ifdef __cplusplus
#include <vector>
using std::vector;

/** alias of std::vector<:c:type:`index_t`> */
typedef vector<index_t> vector_index_t;
/**
 * uses internal energy array if NULL is provided for E. Works like
 * :c:func:`diverge_patching_find_fs_pts_C` but from C++, i.e., returning an
 * std::vector<:c:type:`index_t`> instead of messing with pointers.
 */
vector_index_t diverge_patching_find_fs_pts( diverge_model_t* mod, const double* E, index_t nb, index_t np_ibz, index_t np_search );
extern "C" {
#endif // __cplusplus

/**
 * C interface to the function :cpp:func:`diverge_patching_find_fs_pts`. must
 * free the pts memory with ``free(*pts)``.
 *
 * :param diverge_model_t mod: model to operate on
 * :param double E: energy array. May be NULL for operation on model internals
 * :param index_t nb: number of bands. May be -1 for operation on model
 *                    internals
 * :param index_t np_ibz: number of patches to generate in the irreducible BZ
 * :param index_t np_search: number of patches to include in the search array
 *                           within the irreducible BZ
 * :param index_t** pts: output pointer for the fermi surface index array.
 *                       allocated with standard malloc/calloc. must be free'd
 *                       manually.
 * :param index_t* npts: output pointer for the length of the fermi surface
 *                       array.
 */
void diverge_patching_find_fs_pts_C( diverge_model_t* mod, double* E, index_t nb,
        index_t np_ibz, index_t np_search, index_t** pts, index_t* npts );

/**
 * allocates everything inside the ``mom_patching_t*`` struct given Fermi
 * surface indices in the array ``patches``. The ``mom_patching_t*`` internals
 * as well as the struct itself are freed by the code when they are attached to
 * a ``diverge_model_t*`` instance.
 */
mom_patching_t* diverge_patching_from_indices( diverge_model_t* mod, const index_t* patches, index_t npatches );

/**
 * generate refinement with a heuristic function
 *
 * :param diverge_model_t* mod: model to operate on
 * :param mom_patching_t* patch: patching to operate on
 * :param double* E: energy array to operate on. uses internal energy array if
 *                   NULL is provided for E
 * :param index_t nb: #bands spanned by E. uses internal nb if NULL is set for
 *                    E.
 * :param index_t ngroups: #groups to use for autofine function
 * :param double alpha: :math:`(|E-E_\mathrm{min}|/|E_\mathrm{max}-
 *                      E_\mathrm{min}|)^\alpha` is used to calculate group
 *                      index g
 * :param double beta: :math:`(1-g/n_\mathrm{groups})^\beta` is used to
 *                     calculate number of clusters in group g
 * :param double gamma: if :math:`\gamma < \beta * n_\mathrm{groups}` use all
 *                      points in cluster
 */
void diverge_patching_autofine( diverge_model_t* mod, mom_patching_t* patch,
        const double* E, index_t nb, index_t ngroups, double alpha, double beta,
        double gamma );

/**
 * according to the realspace symmetries defined in the model the refinement is
 * symmetrized. useful if the refinement has been generated with the autofine
 * method.
 */
void diverge_patching_symmetrize_refinement( diverge_model_t* mod, mom_patching_t* patch );

/**
 * frees all resources of a :c:struct:`mom_patching_t` structure. Usage is
 * typically not needed, but for advanced patching calculations it may be
 * advantageous.
 */
void diverge_patching_free( mom_patching_t* patching );


#ifdef __cplusplus
}
#endif
