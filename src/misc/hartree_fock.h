/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/** opaque typedef for the handle */
typedef struct diverge_hartree_fock_t diverge_hartree_fock_t;

/**
 * initialize a :c:type:`diverge_hartree_fock_t` handle given a model that has
 * both the TU and common internals set. The user additionally has to provide a
 * filling value :math:`\nu` and the interaction channels that should be taken
 * into account (equivalent to the *vertex* specific channels in
 * :c:func:`diverge_flow_step_init`). Must be freed with a call to
 * :c:func:`diverge_hartree_fock_free`.
 *
 * :param diverge_model_t* m: diverge model
 * :param double nu: filling value (0...1)
 * :param const char* chans: interaction channels as string
 */
diverge_hartree_fock_t* diverge_hartree_fock_init( diverge_model_t* model, double nu, const char* chans );

/**
 * perform a step of the Hartre-Fock self-consistency iteration (at temperature
 * T). "Step" here refers to the fact that the current self-energy (see
 * :c:func:`diverge_hartree_fock_get_sigma`) is used for calculating the Green's
 * function via :math:`G(\boldsymbol{k})^{-1} = i\omega - H(\boldsymbol{k}) -
 * \Sigma(\boldsymbol{k})`, which is used to evaluate the diagrams. On exit, the
 * resulting "new" self-energy is stored in the buffer pointed to by
 * :c:func:`diverge_hartree_fock_get_sigma`, and the (previous) self-energy that
 * acted as input to calculating the diagrams gets copied to the buffer pointed
 * to by :c:func:`diverge_hartree_fock_get_sigma_prev`.
 */
void diverge_hartree_fock_step( diverge_hartree_fock_t* h, double T );

/**
 * return the current self-energy buffer. Can be used to e.g. set a seed or
 * output of the final self-energy.
 */
complex128_t* diverge_hartree_fock_get_sigma( diverge_hartree_fock_t* h );

/**
 * return the self-energy buffer of the *previous* iteration. Do not put a seed
 * in here. Useful to combine with :c:func:`diverge_hartree_fock_get_sigma` for
 * mixing schemes or assessing convergence.
 */
complex128_t* diverge_hartree_fock_get_sigma_prev( diverge_hartree_fock_t* h );

/** free the :c:type:`diverge_hartree_fock_t` handle. */
void diverge_hartree_fock_free( diverge_hartree_fock_t* h );

#ifdef __cplusplus
}
#endif
