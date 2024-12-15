/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/**
 * create a linearly spaced array from ``min`` to ``max`` with ``num`` entries.
 * returned array must be free'd by user with ``free`` or
 * :c:func:`diverge_mem_free`. When ``num`` is negative, the endpoint is
 * excluded. Otherwise it's included.
 */
double* diverge_linspace( double min, double max, index_t num );

/**
 * calculate the DOS given an array of energies or a model that has its
 * internals set: :math:`\rho(\omega) = \sum_{k,b} \Im[ 1/(\omega - i\eta -
 * E_{kb}) ]`.
 *
 * :param const diverge_model_t* m: the model to operate on with common
 *      internals set via :c:func:`diverge_model_internals_common`.
 * :param const double* E: an optional energy array, may be ``NULL`` in which
 *      case the internal energy array is used.
 * :param index_t nb: an optional number of bands. is ignored if ``E == NULL``
 *      or if ``nb <= 0``
 * :param const double* omega: frequency array
 * :param index_t n_omega: number of frequencies
 * :param double eta: smearing. If ``eta < 0``, use an optimized algorithm that
 *      only considers energies within a small window for DOS calculation at
 *      each frequency, assumes linearly spaced array for ``omega``.
 *
 * :returns: ``double`` array with the density of states (shape n_omega). Result
 *      must be free'd with call to ``free`` or :c:func:`diverge_mem_free`.
 */
double* diverge_model_dos( const diverge_model_t* m, const double* E, index_t nb,
        const double* omega, index_t n_omega, double eta );

/**
 * calculate the LDOS given an array of energies and orbital to band matrices or
 * a model that has its internals set: :math:`\rho(\omega,o) = \sum_{k,b} \Im[
 * |U_{kb}^o|^2/(\omega - i\eta - E_{kb}) ]`.
 *
 * :param const diverge_model_t* m: the model to operate on with common
 *      internals set via :c:func:`diverge_model_internals_common`.
 * :param const double* E: an optional energy array, may be ``NULL`` in which
 *      case the internal energy array is used.
 * :param const complex128_t* U: an optional orbital to band array, may be
 *      ``NULL`` in which case the internal o2b array is used.
 * :param index_t nb: an optional number of bands. is ignored if ``E == NULL``
 *      or if ``nb <= 0``
 * :param const double* omega: frequency array
 * :param index_t n_omega: number of frequencies
 * :param double eta: smearing. If ``eta < 0``, use an optimized algorithm that
 *      only considers energies within a small window for LDOS calculation at
 *      each frequency, assumes linearly spaced array for ``omega``.
 *
 * :returns: ``double`` array with the local density of states (shape
 *      (n_omega,nb)). Result must be free'd with call to ``free`` or
 *      :c:func:`diverge_mem_free`.
 */
double* diverge_model_ldos( const diverge_model_t* m, const double* E,
        const complex128_t* U, index_t nb, const double* omega, index_t n_omega,
        double eta );

/**
 * set the factor :math:`F` that determines the number of bins :math:`B` from
 * smearing :math:`\eta` and omega spacing :math:`\mathrm{d}\omega` in the
 * optimized algorithms. Defaults to 10.0.
 *
 * :math:`B = \eta / \mathrm{d}\omega \cdot F`
 */
void diverge_optimized_dos_set_binfact( double binfact );

#ifdef __cplusplus
}
#endif
