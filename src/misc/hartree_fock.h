/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/** opaque typedef for hanlde used in HF simulations */
typedef struct diverge_hartree_fock_t diverge_hartree_fock_t;

/**
 * initialize a :c:type:`diverge_hartree_fock_t` handle given a model, filling, and the
 * interaction channels that should be taken into account.
 *
 * :param diverge_model_t* m: diverge model
 * :param double nu: filling value (0-1)
 * :param const char* chans: channels to consider for the interaction
 *   (equivalent to the *vertex* specific channels in
 *   :c:func:`diverge_flow_step_init`)
 */
diverge_hartree_fock_t* diverge_hartree_fock_init( diverge_model_t* model, double nu, const char* chans );

/** perform a self-consistent HF step at temperature T */
void diverge_hartree_fock_step( diverge_hartree_fock_t* h, double T );

/**
 * return the current self-energy. Use this array to set a seed. combine with
 * :c:func:`diverge_hartree_fock_get_sigma_prev` to implement a mixing scheme.
 */
complex128_t* diverge_hartree_fock_get_sigma( diverge_hartree_fock_t* h );

/**
 * return the self-energy of the previous iteration. Do not put a seed in here
 * or modify
 */
complex128_t* diverge_hartree_fock_get_sigma_prev( diverge_hartree_fock_t* h );

/** free the :c:type:`diverge_hartree_fock_t` handle. */
void diverge_hartree_fock_free( diverge_hartree_fock_t* h );

#ifdef __cplusplus
}
#endif
