/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

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
 * and :math:`\mathcal S` denotes a set of bands to sum over. Our implementation
 * only takes finite differences along each of the lattice directions, which are
 * transformed to cartesian coordinates with ``non_ortho_grad``.
 *
 * :param diverge_model_t m: the model instance. May be NULL in case all other
 *      parameters are given (non-NULL / not -1).
 * :param complex128_t* U: orbital to band matrices as (k,b,o) array;
 *      :math:`U_{o,b}(\boldsymbol k)`. May be NULL in case ``m`` contains
 *      common internals.
 * :param index_t nb: total number of bands in the system. May be -1 in case
 *      ``m`` is given.
 * :param index_t* nk: number of k points as array of shape (3,). May be NULL in
 *      case ``m`` is given. As for :c:func:`diverge_fukui`, if any of the
 *      provided mesh lengths is negative, this function transforms to the
 *      improper gauge; the model ``m`` must not be NULL for this to work.
 *      In case the improper gauge is used, one can swap the sign convention by
 *      changing the number of negative dimensions.
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

/**
 * Calculate the full quantum geometric tensor :math:`\eta_{\mu\nu}(\boldsymbol
 * k)`, which split into real- and imaginary parts yields the Berry curvature
 * :math:`-2\mathrm{Im}\, \eta_{\mu\nu}(\boldsymbol k) =
 * \Omega_{\mu\nu}(\boldsymbol k)` (see :c:func:`diverge_fukui_non_abelian`) and
 * the quantum metric :math:`\mathrm{Re}\, \eta_{\mu\nu}(\boldsymbol k) =
 * g_{\mu\nu}(\boldsymbol k)` (see :c:func:`diverge_qgt`). Following `[Resta,
 * EPJB 79, 121-137 (2011)] <http://dx.doi.org/10.1140/epjb/e2010-10874-4>`_, we
 * can obtain the metric tensor in a fully gauge-invariant way by calculating
 * :math:`\eta_{\mu\nu}(\boldsymbol k) = \mathrm{Tr}\big[ \partial_\mu \hat
 * P(\boldsymbol k) \hat Q(\boldsymbol k) \partial_\nu \hat P(\boldsymbol k)
 * \big]`. The projectors are defined as :math:`P_{o,o'}(\boldsymbol k) =
 * \sum_{b\in\mathcal S} U_{o,b}(\boldsymbol k) U_{o',b}^*(\boldsymbol k)` and
 * :math:`\hat Q(\boldsymbol k) = 1 - \hat P(\boldsymbol k)`, where
 * :math:`\mathcal S` denotes the set of isolated bands to calculate the quantum
 * metric over.
 *
 * .. note::
 *  The Berry curvature obtained from this function is not normalized
 *  intuitively. To obtain the Chern number (in a certain direction), you have
 *  to calculate
 *
 *  .. math::
 *
 *      \mathcal C_{\mu\nu} = \frac{-2\pi}{A N_{\boldsymbol k}} \sum_{\boldsymbol
 *      k} \mathrm{Im}\big( \Omega_{\mu\nu}(\boldsymbol k) -
 *      \Omega_{\nu\mu}(\boldsymbol k)\big) \,,
 *
 *  with :math:`A = (\boldsymbol a_1 \times \boldsymbol a_2) \cdot \boldsymbol
 *  a_3` the unit cell area.
 *
 *  The quantum metric (:c:func:`diverge_qgt`) is given in the same units:
 *
 *  .. math::
 *
 *      g(\boldsymbol k) = \sum_i \mathrm{Re}\big( \eta_{ii}(\boldsymbol k)
 *      \big) \,.
 *
 * :param diverge_model_t m: the model instance. May be NULL in case all other
 *      parameters are given (non-NULL / not -1).
 * :param complex128_t* U: orbital to band matrices as (k,b,o) array;
 *      :math:`U_{o,b}(\boldsymbol k)`. May be NULL in case ``m`` contains
 *      common internals.
 * :param index_t nb: total number of bands in the system. May be -1 in case
 *      ``m`` is given.
 * :param index_t* nk: number of k points as array of shape (3,). May be NULL in
 *      case ``m`` is given. As for :c:func:`diverge_fukui`, if any of the
 *      provided mesh lengths is negative, this function transforms to the
 *      improper gauge; the model ``m`` must not be NULL for this to work.
 *      In case the improper gauge is used, one can swap the sign convention by
 *      changing the number of negative dimensions.
 * :param index_t* which_bands: the bands to sum over. Describes the set
 *      :math:`\mathcal S`. Must not be NULL. Array of shape (``n_which_bands``,).
 * :param index_t n_which_bands: number of bands to sum over. Describes the set
 *      :math:`\mathcal S`, i.e., the array ``which_bands``.
 * :param int non_ortho_grad: boolean flag that determines whether the lattice
 *      vectors are taken into account in the calculation of :math:`\nabla f
 *      \cdot \nabla g`.
 *
 * :returns: a :c:type:`complex128_t` array of size (nk,3,3) with the values of
 *      :math:`\eta_{\mu\nu}(\boldsymbol k)`. Must be free'd by the user by a
 *      call to free or :c:func:`diverge_mem_free`.
 */
complex128_t* diverge_qgt_tensor( diverge_model_t* m, complex128_t* U, index_t nb,
        index_t* nk, index_t* which_bands, index_t n_which_bands, int non_ortho_grad );

#ifdef __cplusplus
}
#endif


