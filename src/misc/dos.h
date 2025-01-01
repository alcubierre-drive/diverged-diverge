/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

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
 * Calculate an energy array given a filling array, i.e., do the transformation
 * :math:`\nu \rightarrow E(\nu)` for all ``n_nu`` values in the ``nu`` array.
 * Allocates and frees sorted copy of energy array.
 *
 * :param const diverge_model_t* m: the model to operate on with common
 *      internals set via :c:func:`diverge_model_internals_common`.
 * :param const double* E: an optional energy array, may be ``NULL`` in which
 *      case the internal energy array is used.
 * :param index_t nb: an optional number of bands. is ignored if ``E == NULL``
 *      or if ``nb <= 0``
 * :param const double* nu: filling array of size ``n_nu`` with values between
 *      zero and one (everything else is truncated to zero-one)
 * :param index_t n_nu: length of the filling array.
 *
 * :returns: ``double`` array with the energies corresponding to the fillings
 *     (size ``n_nu``) that must be free'd by the user via ``free`` or
 *     :c:func:`diverge_mem_free`.
 */
double* diverge_filling_to_energy( const diverge_model_t* m, const double* E,
        index_t nb, const double* nu, index_t n_nu );

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
 * :param const double* omega: frequency array in ascending order (else it is
 *      modified and sorted).
 * :param index_t n_omega: number of frequencies. If negative, don't allocate
 *      the memory for the result but instead *reallocate* ``omega`` to a region
 *      large enough to hold ``omega`` and the result (subsequently in memory).
 *      The new pointer is returned. Works only for the case where ``omega`` is
 *      allocated with ``malloc`` and friends.
 * :param double eta: smearing. If ``eta < 0``, use an optimized algorithm that
 *      only considers energies within a small window for DOS calculation at
 *      each frequency, assumes linearly spaced array for ``omega``.
 *
 * :returns: ``double`` array with the density of states (shape n_omega). Result
 *      must be free'd with call to ``free`` or :c:func:`diverge_mem_free`.
 */
double* diverge_model_dos( const diverge_model_t* m, const double* E, index_t nb,
        double* omega, index_t n_omega, double eta );

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
 * :param const double* omega: frequency array in ascending order (else it is
 *      modified and sorted).
 * :param index_t n_omega: number of frequencies. If negative, don't allocate
 *      the memory for the result but instead *reallocate* ``omega`` to a region
 *      large enough to hold ``omega`` and the result (subsequently in memory).
 *      The new pointer is returned. Works only for the case where ``omega`` is
 *      allocated with ``malloc`` and friends.
 * :param double eta: smearing. If ``eta < 0``, use an optimized algorithm that
 *      only considers energies within a small window for LDOS calculation at
 *      each frequency, assumes linearly spaced array for ``omega``.
 *
 * :returns: ``double`` array with the local density of states (shape
 *      (n_omega,nb)). Result must be free'd with call to ``free`` or
 *      :c:func:`diverge_mem_free`.
 */
double* diverge_model_ldos( const diverge_model_t* m, const double* E,
        const complex128_t* U, index_t nb, double* omega, index_t n_omega,
        double eta );

/**
 * set the factor :math:`e` that determines the width of energies :math:`W` to
 * be considered for summation at a given frequency :math:`\omega`, i.e.,
 *
 * :math:`W = [\omega - e\eta, \omega + e\eta]`
 *
 * Defaults to :math:`e = 15`, which is good enough for most features in a
 * regular DOS, but not good enough for excellent convergence of the Lorentzian
 * Kernel (which only falls off quadratically).
 */
void diverge_dos_set_eta_factor( double eta_factor );

/** set whether GPU is used for DOS calc */
void diverge_dos_use_gpu( int use_gpu );

void _dos_sort_doubles_( double* start, index_t size );
void _dos_sort_doubles_indirect_( index_t* start, index_t size, const double* vals );

#ifdef __cplusplus
}
#endif
