/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/**
 * generate the Hamiltonian at the momentum vector :math:`\boldsymbol{k}`
 * (described by the array ``k[3]``) and write it into ``buf`` (which needs to
 * be of the size ``POW2(m->n_orb * m->n_spin)``). The chemical potential
 * :math:`\mu` is *not* included; to do so you have to subtract it from the
 * diagonal. The matrix pointed to by ``buf`` is overwritten.
 */
void diverge_model_ham_at_kpt( const diverge_model_t* m, const double* k, complex128_t* buf );
/**
 * generate the Hamiltonian at the coarse index ``k`` and write it into ``buf``.
 * ``k`` refers to a coarse mesh index, and ``buf`` must be sized the same way
 * as in :c:func:`diverge_model_ham_at_kpt`.
 */
void diverge_model_ham_at_kcidx( const diverge_model_t* m, index_t k, complex128_t* buf );
/**
 * generate the Hamiltonian at the fine index ``k`` and write it into ``buf``.
 * ``k`` refers to a fine mesh index, and ``buf`` must be sized the same way as
 * in :c:func:`diverge_model_ham_at_kpt`.
 */
void diverge_model_ham_at_kfidx( const diverge_model_t* m, index_t k, complex128_t* buf );

#ifdef __cplusplus
}
#endif
