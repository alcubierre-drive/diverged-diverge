/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * returns the filling value between 0 and 1
 *
 * :param diverge_model_t* model: the model *with* initialized *common* internals
 *                                (:c:func:`diverge_model_internals_common`) but
 *                                *without specific* ones (via
 *                                :c:func:`diverge_model_internals_tu`,
 *                                :c:func:`diverge_model_internals_patch`,
 *                                :c:func:`diverge_model_internals_grid`)
 * :param double* E: energy array. if (E == NULL) the internal energy array is used
 * :param index_t nb: number of bands. if (E == NULL) the internal number of bands is used
 */
double diverge_model_get_filling( diverge_model_t* model, const double* E, index_t nb );

/**
 * set the filling value
 *
 * :param diverge_model_t* model: the model *with* initialized *common* internals
 *                                (:c:func:`diverge_model_internals_common`) but
 *                                *without specific* ones (via
 *                                :c:func:`diverge_model_internals_tu`,
 *                                :c:func:`diverge_model_internals_patch`,
 *                                :c:func:`diverge_model_internals_grid`)
 * :param double* E: energy array. if (E == NULL) the internal energy array is used *and modified*.
 *                   Otherwise E is *used and modified*.
 * :param index_t nb: number of bands. if (E == NULL) the internal number of bands is used
 * :param double nu: filling value between 0 (all bands empty) and 1 (all bands full)
 *
 * :returns: the chemical potential used to set the desired filling value
 */
double diverge_model_set_filling( diverge_model_t* model, double* E, index_t nb, double nu );

/**
 * set the chemical potential
 *
 * :param diverge_model_t* model: the model *with* initialized *common* internals
 *                                (:c:func:`diverge_model_internals_common`) but
 *                                *without specific* ones (via
 *                                :c:func:`diverge_model_internals_tu`,
 *                                :c:func:`diverge_model_internals_patch`,
 *                                :c:func:`diverge_model_internals_grid`)
 * :param double* E: energy array. if (E == NULL) the internal energy array is used *and modified*.
 *                   Otherwise E is *used and modified*.
 * :param index_t nb: number of bands. if (E == NULL) the internal number of bands is used
 * :param double mu: chemical potential
 */
void diverge_model_set_chempot( diverge_model_t* model, double* E, index_t nb, double mu );

#ifdef __cplusplus
}
#endif
