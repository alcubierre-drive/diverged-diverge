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
// Written such that the runtime is O(nk * n_ibz_path).
//
// Parameters:
// m - the model to use (with bandstructure path defined in m->ibz_path)
// indices - pointer to result array that is allocated and filled with the
//           indices that lie on the ibz path. indices refer to fine mesh.
// n_indices - pointer to length of result array
//
// Returns:
// The number of points per segment in an array. both the returned array as well
// as the *indices array must be free'd manually.
index_t* find_bandstructure_pts( diverge_model_t* m, index_t** indices, index_t* n_indices );

// Function: find_bandstructure_pts_crs
// same as <find_bandstructure_pts>, but for the coarse mesh.
index_t* find_bandstructure_pts_crs( diverge_model_t* m, index_t** indices, index_t* n_indices );

#ifdef __cplusplus
}
#endif
