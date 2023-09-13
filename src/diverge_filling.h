/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Function: diverge_model_get_filling
// returns the filling value between 0 and 1
//
// Parameters:
// model - the model *with initialized common internals*
//         (<diverge_model_internals_common>) but without specific ones (via
//         <diverge_model_internals_tu>, <diverge_model_internals_patch>,
//         <diverge_model_internals_grid>)
// E - energy array. if (E == NULL) the internal energy array is used
// nb - number of bands. if (E == NULL) the internal number of bands is used
double diverge_model_get_filling( diverge_model_t* model, const double* E, index_t nb );

// Function: diverge_model_set_filling
// set the filling value
//
// Parameters:
// model - the model *with initialized common internals*
//         (<diverge_model_internals_common>) but without specific ones (via
//         <diverge_model_internals_tu>, <diverge_model_internals_patch>,
//         <diverge_model_internals_grid>)
// E - energy array. if (E == NULL) the internal energy array is used
//     *and modified*. Otherwise, E is used and modified
// nb - number of bands. if (E == NULL) the internal number of bands is used
// nu - filling value between 0 (all bands empty) and 1 (all bands full)
//
// Returns:
// the chemical potential used to set the desired filling value
double diverge_model_set_filling( diverge_model_t* model, double* E, index_t nb, double nu );

// Function: diverge_model_set_chempot
// set the chemical potential
//
// Parameters:
// model - the model *with initialized common internals*
//         (<diverge_model_internals_common>) but without specific ones (via
//         <diverge_model_internals_tu>, <diverge_model_internals_patch>,
//         <diverge_model_internals_grid>)
// E - energy array. if (E == NULL) the internal energy array is used
//     *and modified*. Otherwise, E is used and modified
// nb - number of bands. if (E == NULL) the internal number of bands is used
// mu - chemical potential
void diverge_model_set_chempot( diverge_model_t* model, double* E, index_t nb, double mu );

#ifdef __cplusplus
}
#endif
