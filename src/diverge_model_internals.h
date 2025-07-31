/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_model.h"
#include "diverge_momentum_gen.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Check for errors in the user defined model
 */
int diverge_model_validate( diverge_model_t* model );

/**
 * Prepare the common part of the :c:type:`internals_t` struct,
 * including Hamiltonian, Eigenvectors/Eigenvalues, momentum meshes, etc.
 */
void diverge_model_internals_common( diverge_model_t* model );

/** Prepare internal structure for the computation mode specified as string in
 * mode; can be ``"grid"``, ``"patch"``, or ``"tu"``. Dispatches to the
 * corrseponding function, i.e.,
 *
 *  * :c:func:`diverge_model_internals_grid`,
 *  * :c:func:`diverge_model_internals_patch`,
 *  * :c:func:`diverge_model_internals_tu`.
 *
 * For :c:func:`diverge_model_internals_grid`, no additional parameters are
 * needed.
 *
 * For :c:func:`diverge_model_internals_patch`, one additional parameter
 * (``index_t np_ibz``) must be passed. Documentation found in
 * :c:func:`diverge_model_internals_patch`.
 *
 * For :c:func:`diverge_model_internals_tu`, one additional parameter (``double
 * max_dist``) must be passed. Documentation found in
 * :c:func:`diverge_model_internals_tu`.
 */
void diverge_model_internals_any( diverge_model_t* model, const char* mode, ... );

/**
 * Prepare the grid FRG part of the :c:type:`internals_t` structure.
 */
void diverge_model_internals_grid( diverge_model_t* model );

/**
 * Prepare the patch part of the :c:type:`internals_t` struct as well as the
 * :c:struct:`mom_patching_t` struct (:c:member:`diverge_model_t.patching`) if it
 * is not allocated.
 *
 * .. note::
 *  divERGe can only generate and operate on the npatch backend if there
 *  is no refinement set through :c:member:`diverge_model_t.nkf`, i.e., ones in
 *  those spots where :c:member:`diverge_model_t.nk` is nonzero. Some of the
 *  patching routines further assume a two-dimensional model -- not the actual
 *  FRG kernels, merely some helper functions.
 *
 * :param diverge_model_t model: model instance
 * :param index_t np_ibz: number of patches to search for in the IBZ if
 *                        :c:member:`diverge_model_t.patching` is NULL
 */
void diverge_model_internals_patch( diverge_model_t* model, index_t np_ibz );

/**
 * Prepare the TU part of the :c:type:`internals_t` struct.
 *
 * :param diverge_model_t model: model instance
 * :param double max_dist: distance up to which formfactors should be included.
 *                         If the return value of
 *                         :c:func:`diverge_max_dist_iobi` is passed (a special
 *                         'nan'), formfactors that adhere to the IOBI
 *                         approximation are generated.
 */
void diverge_model_internals_tu( diverge_model_t* model, double max_dist );

#define IOBI_FORMFACTOR_MAGIC_NUMBER 0
/**
 * return magic value for formfactor setup (max_dist value in
 * :c:func:`diverge_model_internals_tu`) of TUFRG such that only on-site
 * intraorbital formfactors are generated (IOBI approximation). Uses quiet nan
 * payload, i.e., returns 'nan'. Might break if non-IEEE floating point numbers
 * are in use.
 */
double diverge_max_dist_iobi( void );

/**
 * Reset all internal structures of the model and free their resources. Does not
 * reset the patching. Usage is discouraged.
 */
void diverge_model_internals_reset( diverge_model_t* model );

// DEVELOPERS
//
// For a new backend, you must write your own diverge_model_internals_backend
// initialization routine. There, you must initialize the vertex using the
// following hierarchy:
//
// if (model->ffill != NULL)
//     use this function to initialize the full vertex
// else
//     use model->vfill to initialize the three channels
//
// For examples see src/diverge_model_internals_grid.cpp or
// src/diverge_model_internals_patch.c
//
// Moreover you must take care of enforcing exchange symmetry in the non-SU2
// case if the user whishes to do so (i.e. model->internals->enforce_exchange is
// true). Note that *all* crossing relations need to be taken into account:
//
// V(1234) = 0.5 * ( V(1234) - V(1243) + V(2143) - V(2134) )
//
// The factor 0.5 exists for the reason of initializing density-density
// interactions in their 'natural' (i.e. D) channel only, and not both C and D
// channel.

/**
 * returns the internally generated energies (nk*nkf, nb).
 *
 * :c:func:`diverge_model_internals_common` must have been called, otherwise
 * NULL is returned.
 */
double*       diverge_model_internals_get_E(      const diverge_model_t* model );
/**
 * returns the internally generated orbital to band transform (nk*nkf, nb, nb)
 *
 * :c:func:`diverge_model_internals_common` must have been called, otherwise
 * NULL is returned.
 */
complex128_t* diverge_model_internals_get_U(      const diverge_model_t* model );
/**
 * returns the internally generated hamiltonian (nk*nkf, nb, nb)
 *
 * :c:func:`diverge_model_internals_common` must have been called, otherwise
 * NULL is returned.
 */
complex128_t* diverge_model_internals_get_H(      const diverge_model_t* model );
/**
 * returns the internally generated coarse momentum mesh (nk, 3)
 *
 * :c:func:`diverge_model_internals_common` must have been called, otherwise
 * NULL is returned.
 */
double*       diverge_model_internals_get_kmesh(  const diverge_model_t* model );
/**
 * returns the internally generated coarse momentum mesh (nk*nkf, 3)
 *
 * :c:func:`diverge_model_internals_common` must have been called, otherwise
 * NULL is returned.
 */
double*       diverge_model_internals_get_kfmesh( const diverge_model_t* model );
/**
 * returns the internal Green's function buffer (2, nk*nkf, nb, nb)
 *
 * :c:func:`diverge_model_internals_common` must have been called, otherwise
 * NULL is returned.
 */
gf_complex_t* diverge_model_internals_get_greens( const diverge_model_t* model );
/** returns the dimension (:c:func:`diverge_model_internals_common` should be called prior) */
index_t       diverge_model_internals_get_dim(    const diverge_model_t* model );
/** return the reciprocal lattice (:c:func:`diverge_model_internals_common` should be called prior) */
diverge_lattice_t diverge_model_internals_get_rlattice( const diverge_model_t* model );

#ifdef __cplusplus
}
#endif
