/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/** solve a single eigen problem for densely packed matrices in C order */
void single_eigen( const complex128_t* H, complex128_t* U, double* E, index_t N );
/**
 * solve a single SVD for densely packed matrices in C order. if N<0, use
 * non-hermitian solver (dim=|N|) and copy the *real part* of the eigenvalues
 */
void single_svd( const complex128_t* H, complex128_t* U, complex128_t* V, double* S, index_t N );

// ALL THE SORT FUNCTIONS ASSUME FORTRAN ORDER FOR U, IN CONTRAST TO THE SINGLE_EIGEN FUNCTIONS!!!

/**
 * sort eigenvectors and eigenvalues of on eigenproblem according to a given
 * algorithm. Notably, the sort function expects FORTRAN ordered eigenvectors,
 * whereas the :c:func:`single_eigen` and :c:func:`single_svd` functions return
 * C ordered eigenvectors.
 *
 * Sorting options (``which``):
 *
 *      :``0``: default (``'M'``)
 *      :``'A'``: alternating (max negative, max positive, ...)
 *      :``'P'``: max positive
 *      :``'N'``: max negative
 *      :``'M'``: magnitude
 *      :``'m'``: inverse magnitude
 */
void single_eigen_sort( complex128_t* U, double* E, index_t N, char which );

/**
 * same as :c:func:`single_eigen_sort`, same as above, but only copy over the
 * first ``num`` values. return the number of values written.
 */
index_t single_eigen_sort_part( complex128_t* U, double* E, index_t N, char which, index_t num );

/**
 * same as :c:func:`single_eigen_sort` in terms of sorting order given by
 * ``which``. In addition to the sorting order, we allow for a relative or
 * absolute cutoff in ``abs_limit``. if ``abs_limit<0``, use relative cutoff. if
 * ``abs_limit>0``, use absolute cutoff. returns the number of elements the
 * satify the cutoff condition. if ``abs_limit`` is all zero bytes, do not
 * cutoff the eigensolution.
 */
index_t single_eigen_sort_N( complex128_t* U, double* E, index_t N, char which, double abs_limit );

#ifdef __cplusplus
}
#endif
