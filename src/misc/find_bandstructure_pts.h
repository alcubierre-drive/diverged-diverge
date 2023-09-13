/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration
typedef struct diverge_model_t diverge_model_t;

// Function: find_bandstructure_pts
// finds fine mesh points (as indices) that lie on the segments that define the
// irreducible path. Useful for, e.g., generation of the Hamiltonian (or
// self-energy) along the irreducible path, or outputting a susceptibility.
// Written such that the runtime is O(nk * n_ibz_path). Requires a previous call
// to <diverge_model_internals_common> in order to prepare the momentum meshes.
//
// Parameters:
// m - the model to use (with bandstructure path defined in
//     <m->ibz_path at diverge_model_t.ibz_path[MAX_N_ORBS][3]> and
//     <m->n_ibz_path at diverge_model_t.n_ibz_path>)
// indices - pointer to result array that is allocated and filled with the
//           indices that lie on the ibz path. indices refer to fine mesh.
// n_indices - pointer to length of result array
//
// Returns:
// The number of points per segment in an array. both the returned array as well
// as the *indices array must be free'd manually.
//
// Example:
// <find_bandstructure_pts> or <find_bandstructure_pts_crs> can be used as
// follows:
// ---- C/C++ ----
// diverge_model_t* m;
// // fill m with appropriate content
// diverge_model_internals_common( m );
// index_t* pts;
// index_t n_pts;
// index_t* n_per_segment = find_bandstructure_pts( m, &pts, &n_pts );
// // do something with pts, n_pts, and n_per_segment.
// free( pts );
// free( n_per_segment );
// ---------------
index_t* find_bandstructure_pts( diverge_model_t* m, index_t** indices, index_t* n_indices );

// Function: find_bandstructure_pts_crs
// same as <find_bandstructure_pts>, but for the coarse mesh.
index_t* find_bandstructure_pts_crs( diverge_model_t* m, index_t** indices, index_t* n_indices );

// TODO(LK) need to prefix these functions with diverge_ somehow. Don't want
// incredibly long names, though. ideas?

#ifdef __cplusplus
}
#endif
