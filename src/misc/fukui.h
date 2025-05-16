/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/**
 * Calculate Berry's curvature :math:`\Omega(\boldsymbol{k})` using the
 * Fukui-Hatsugai method, normed to the Chern number, i.e., the sum of all
 * elements for a certain chern band yields the Chern number of that band, not
 * :math:`2\pi` times the Chern number.
 *
 * :param diverge_model_t m: model instance. obtain all parameters from there by
 *                           default. internals must be set
 *                           (:c:func:`diverge_model_internals_common`) in order
 *                           to have U initialized. if NULL is passed, the
 *                           function can still operate in case all the other
 *                           parameters are set accordingly.
 * :param complex128_t* U: orbital to band matrices as (k,b,o) array. Use if != NULL.
 * :param index_t nb: nuber of bands. Use this if > 0.
 * :param index_t* nk: number of k points (3,) array. Use if != NULL. If ``nk[0]
 *                     < 0 || nk[1] < 0 || nk[2] < 0``, include transformation
 *                     to improper gauge in the calculation of
 *                     :math:`\Omega(\boldsymbol{k})` and use ``abs(nk[i])`` as
 *                     number of :math:`\boldsymbol{k}` points. In case the
 *                     improper gauge is used, one can swap the sign convention
 *                     by changing the number of negative dimensions.
 *
 * :returns: ``double`` array of shape ``(nk, nb, 3)`` that contains the Berry
 *           curvature along the three crystal axes. Must be freed manually by a
 *           call to ``free`` or :c:func:`diverge_mem_free`. May be ``NULL`` on
 *           errors.
 */
double* diverge_fukui( diverge_model_t* m, complex128_t* U, index_t nb, index_t* nk );

/**
 * Calculate the non-abelian Berry curvature over a set of bands using Fukui's
 * method. As for :c:func:`diverge_fukui`, we norm it to the Chern number. The
 * input parameters also follow the function :c:func:`diverge_fukui`:
 *
 * :param diverge_model_t m: model instance. obtain all parameters from there by
 *                           default. internals must be set
 *                           (:c:func:`diverge_model_internals_common`) in order
 *                           to have U initialized. if NULL is passed, the
 *                           function can still operate in case all the other
 *                           parameters are set accordingly.
 * :param complex128_t* U: orbital to band matrices as (k,b,o) array. Use if != NULL.
 * :param index_t nb: nuber of bands. Use this if > 0.
 * :param index_t* nk: number of k points (3,) array. Use if != NULL. If ``nk[0]
 *                     < 0 || nk[1] < 0 || nk[2] < 0``, include transformation
 *                     to improper gauge in the calculation of
 *                     :math:`\Omega(\boldsymbol{k})` and use ``abs(nk[i])`` as
 *                     number of :math:`\boldsymbol{k}` points. In case the
 *                     improper gauge is used, one can swap the sign convention
 *                     by changing the number of negative dimensions.
 * :param index_t* which_bands: selection of bands to take the non-abelian
 *                              determinant over
 * :param index_t n_which_bands: number of bands in the selection (same usage as
 *                               in :c:func:`diverge_qgt`)
 *
 * :returns: ``double`` array of shape ``(nk, 3)`` that contains the non-abelian
 *           Berry curvature along the three crystal axes. Must be freed
 *           manually by a call to ``free`` or :c:func:`diverge_mem_free`. May
 *           be ``NULL`` on errors.
 */
double* diverge_fukui_non_abelian( diverge_model_t* m, complex128_t* U,
        index_t nb, index_t* nk, index_t* which_bands, index_t n_which_bands );

#ifdef __cplusplus
}
#endif
