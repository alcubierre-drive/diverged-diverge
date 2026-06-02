/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

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

/**
 * do power iteration for the maximum eigenvector of matrix ``A`` (NxN, Row
 * major). save eigenvector in ``v``. auxiliary buffer ``aux`` can be NULL or of
 * size ``2*N``. If ``niter != NULL``, use the value pointed to as maximum
 * number of iterations. If ``tol != NULL``, use the value pointed to as minimum
 * tolerance. If ``seed != NULL``, use these *four* integers as seed to
 * construct the random vector. The last integer *must be odd*.
 */
complex128_t power_iteration( const complex128_t* A, complex128_t* v, index_t N,
        complex128_t* aux, index_t* niter, double* tol, const int* iseed );

/**
 * power iteration for a batch of matrices, parallelized over the batch index,
 * hence useful for many rather smallish matrices (where BLAS calls are not
 * pereferable).
 *
 * :param const complex128_t* A_batched:
 *      the matrices that should be analyzed with power iteration, C-shape
 *      ``(num,N,N)``
 * :param complex128_t* U_batched:
 *      eigenvectors as output, C-shape ``(num,N)``. May be NULL, which results
 *      in the function allocating/freeing some memory.
 * :param complex128_t* E_batched:
 *      eigenvalues as output, C-shape ``(num,)``. Must not be NULL.
 * :param const index_t N:
 *      linear size of a single matrix
 * :param const index_t num:
 *      batch size, i.e., number of matrices in a row.
 * :param index_t* maxiter:
 *      number of iterations to perform for each batch index, C-shape
 *      ``(num,)``. May be NULL, in which case memory is allocated/freed and the
 *      default number of iterations is used.
 * :param double* tol:
 *      relative tolerance to achieve for each batch index, C-shape ``(num,)``.
 *      May be NULL, in which case memory is allocated/freed and the default
 *      tolerance is used.
 * :param complex128_t* aux:
 *      auxiliary memory which can be passed to have the function only perform
 *      the absolutely necessary allocations. Minimum size ``(n_threads,2,N)``,
 *      while ``n_threads <= num``. May be NULL, which results in the function
 *      taking care of the auxiliary buffer itself.
 */
void power_iteration_batched( const complex128_t* A_batched, complex128_t* U_batched,
        complex128_t* E_batched, const index_t N, const index_t num, index_t* maxiter,
        double* tol, complex128_t* aux );

/** configure random seed for batched power iteration */
void power_iteration_batched_cfg_seed( uint64_t seed );

#ifdef __cplusplus
}
#endif
