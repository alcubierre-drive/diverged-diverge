#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

// Function: diverge_fukui
// Calculate Berry's curvature using the Fukui-Hatsugai method, normed to the
// Chern number, i.e., the sum of all elements for a certain chern band yields
// the Chern number of that band, not 2Pi times the Chern number. This function
// returns an allocated (calloc) array of shape (nk, nb, 3) that contains the
// Berry curvature along the three crystal axes; not along cartesian axes. The
// return value can be NULL on errors, and must be freed by users with a call to
// free or <diverge_mem_free>.
//
// Parameters:
// m - model instance. obtain all parameters from there by default. internals
//     must be set (<diverge_model_internals_common>) in order to have U
//     initialized. if NULL is passed, the function can still operate in case
//     all the other parameters are set accordingly.
// U - orbital to band matrices as (k,b,o) array. Use if != NULL.
// nb - nuber of bands. Use this if > 0.
// nk - number of k points (3,) array. Use if != NULL.
double* diverge_fukui( diverge_model_t* m, complex128_t* U, index_t nb, index_t* nk );

#ifdef __cplusplus
}
#endif
