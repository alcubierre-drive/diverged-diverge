/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;
typedef struct rs_hopping_t rs_hopping_t;

/**
 * transforms hamiltonian :math:`H(\boldsymbol{k})` of a
 * :c:struct:`diverge_model_t` into an array of realspace hopping parameters
 * :c:struct:`rs_hopping_t`, i.e., an inverse Fourier transform.
 *
 * :param const diverge_model_t* model: the divERGe model to operate on. needs
 *      to have common internals set (:c:func:`diverge_model_internals_common`),
 *      such that the hamiltonian buffer is allocated and filled.
 * :param double hoplim: minimum absolute value that a hopping element must have
 *      in order to be included.
 * :param index_t* n_hop: output for the resulting length of the
 *      :c:struct:`rs_hopping_t` array
 *
 * :returns: :c:type:`rs_hopping_t` array that contains the real-space version
 *      of the input Bloch Hamiltonian (obtained via lattice Fourier transform).
 */
rs_hopping_t* diverge_model_ham2hop( const diverge_model_t* m, double hoplim,
        index_t* n_hop );

/**
 * same as :c:func:`diverge_model_ham2hop`, but with explicit control over the
 * buffer which is used for the Hamiltonian. Very useful to, e.g., extract the
 * selfenergy in efficient representation at the end of the flow.
 */
rs_hopping_t* diverge_model_ham2hop_fg( const diverge_model_t* m, double hoplim,
        index_t* n_hop, const complex128_t* H0 );

#ifdef __cplusplus
}
#endif
