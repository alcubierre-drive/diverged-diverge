#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

// Function: diverge_fukui
// calculate Berry's curvature using the Fukui-Hatsugai method. Returns an
// allocated (calloc) array of shape (nk, nb, 3) with the Berry curvature that
// must be freed by users (using free or <diverge_mem_free>). Returns NULL on
// errors.
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
