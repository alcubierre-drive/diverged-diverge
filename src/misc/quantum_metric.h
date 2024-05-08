/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/**
 * Calculate the Fubini-Study metric :math:`g(\boldsymbol k)`. It is defined as
 * :math:`g(\boldsymbol k) = \frac{1}{2} \mathrm{Tr} \partial_i P_{\mathcal S}
 * \partial_i P_\mathcal{S}`, where :math:`P^{o,o'}_{\mathcal S}(\boldsymbol k)
 * = \sum_{b\in\mathcal S} U_{o,b}(\boldsymbol k) U_{o',b}^*(\boldsymbol k)`,
 * where :math:`\mathcal S` denotes a set of bands to sum over. Our
 * implementation only takes finite differences along each of the lattice
 * directions, but does *not* divide by the corresponding lengths. This must be
 * taken into account when interpreting :math:`g(\boldsymbol k)`.
 *
 * :param diverge_model_t m: the model instance. May be NULL in case all other
 *      parameters are given (non-NULL / not -1).
 * :param complex128_t* U: orbital to band matrices as (k,b,o) array;
 *      :math:`U_{o,b}(\boldsymbol k)`. May be NULL in case ``m`` contains
 *      common internals.
 * :param index_t nb: total number of bands in the system. May be -1 in case
 *      ``m`` is given.
 * :param index_t* nk: number of k points as array of shape (3,). May be NULL in
 *      case ``m`` is given.
 * :param index_t* which_bands: the bands to sum over. Describes the set
 *      :math:`\mathcal S`. Must not be NULL. Array of shape (``n_which_bands``,).
 * :param index_t n_which_bands: number of bands to sum over. Describes the set
 *      :math:`\mathcal S`, i.e., the array ``which_bands``.
 * :param int non_ortho_grad: boolean flag that determines whether the lattice
 *      vectors are taken into account in the calculation of :math:`\nabla f
 *      \cdot \nabla g`.
 *
 * :returns: a ``double`` array of size (nk,) with the values of
 *      :math:`g(\boldsymbol k)`. Must be free'd by the user by a call to free
 *      or :c:func:`diverge_mem_free`.
 */
double* diverge_qgt( diverge_model_t* m, complex128_t* U, index_t nb, index_t* nk,
                     index_t* which_bands, index_t n_which_bands, int non_ortho_grad );

#ifdef __cplusplus
}
#endif


