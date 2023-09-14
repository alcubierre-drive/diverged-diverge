/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Function: diverge_model_validate
// Check for errors in the user defined model
int diverge_model_validate( diverge_model_t* model );

// Function: diverge_model_internals_common
// Prepare the common part of the <internals_t> struct
void diverge_model_internals_common( diverge_model_t* model );

// Function: diverge_model_internals_grid
// Prepare the grid part of the <internals_t> struct
void diverge_model_internals_grid( diverge_model_t* model );
// Function: diverge_model_internals_patch
// Prepare the patch part of the <internals_t> struct as well as the
// <mom_patching_t> struct if it is not allocated.
//
// Parameters:
//  model - <diverge_model_t> instance
//  np_ibz - number of patches to search for in the IBZ if <mom_patching_t> is
//           NULL
void diverge_model_internals_patch( diverge_model_t* model, index_t np_ibz );
// Function: diverge_model_internals_tu
// Prepare the tu part of the <internals_t> struct
//
// Parameters:
//  model - <diverge_model_t> instance
//  max_dist - distance up to which formfactors should be included.
void diverge_model_internals_tu( diverge_model_t* model, double max_dist );

// Function: diverge_model_internals_reset
// Reset all internal structures of the model and free their resources. Does not
// reset the patching. Usage is discouraged.
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

// Namespace: diverge_model_internals_get_
// some getter functions to access internals. may break for non-standard
// models.
//
// Function: diverge_model_internals_get_E
// returns the internally generated energies (nk*nkf, nb).
//
// <diverge_model_internals_common> must have been called, otherwise NULL is
// returned.
double*       diverge_model_internals_get_E(      const diverge_model_t* model );
// Function: diverge_model_internals_get_U
// returns the internally generated orbital to band transform (nk*nkf, nb, nb)
//
// <diverge_model_internals_common> must have been called, otherwise NULL is
// returned.
complex128_t* diverge_model_internals_get_U(      const diverge_model_t* model );
// Function: diverge_model_internals_get_H
// returns the internally generated hamiltonian (nk*nkf, nb, nb)
//
// <diverge_model_internals_common> must have been called, otherwise NULL is
// returned.
complex128_t* diverge_model_internals_get_H(      const diverge_model_t* model );
// Function: diverge_model_internals_get_kmesh
// returns the internally generated coarse momentum mesh (nk, 3)
//
// <diverge_model_internals_common> must have been called, otherwise NULL is
// returned.
double*       diverge_model_internals_get_kmesh(  const diverge_model_t* model );
// Function: diverge_model_internals_get_kfmesh
// returns the internally generated coarse momentum mesh (nk*nkf, 3)
//
// <diverge_model_internals_common> must have been called, otherwise NULL is
// returned.
double*       diverge_model_internals_get_kfmesh( const diverge_model_t* model );
#ifdef __cplusplus
}
#endif
