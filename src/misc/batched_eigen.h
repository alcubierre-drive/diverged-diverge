/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * batched eigensolver for Hermitian matrices, good for many small matrices.
 *
 * :param int* devices: list of CUDA devices, may be NULL, in which case control
 *      can be achieved through ``ndevices``: if ``ndevices == 0`` do CPU
 *      eigensolution. If ``ndevices == -1`` do GPU eigensolution with automatic
 *      device find. If ``ndevices == -2`` do GPU eigensolution with automatic
 *      device find and automatic chunking. If CUDA is not compiled in,
 *      ``devices`` is ignored.
 * :param int ndevices: number of CUDA devices. If ``devices != NULL`` and
 *      ``ndevices < 1`` do automatic setup as if ``deivecs == NULL``.
 *      Otherwise, use the device list provided in ``devices``.
 * :param complex128_t* H: input and output for the matrices to diagonalize.
 *      Matrices must be contiguous in memory and ordered as (i,o,p), where the
 *      latter two indices correspond to the matrix rows p and columns o (i.e.
 *      FORTRAN order). On output, the eigenvectors are stored as ``(i,b,o)``
 *      with ``i`` the index of the eigenproblem, ``b`` the index of the
 *      eigenvalue, and ``o`` the vector component.
 * :param double* E: output array for the eigenvalues. Given as (i,b) in memory,
 *      with each individual eigenproblem sorted in standard fashion.
 * :param dim: linear size of each individual problem (``b``, ``o``, ``p``)
 * :param Num: number of eigenproblems (``i``)
 */
void batched_eigen_r( int* devices, int ndevices, complex128_t* H, double* E, index_t dim, index_t Num );

/** force the CPU eigensolver built with Eigen3 for :c:func:`batched_eigen_r` */
void batched_eigen_r_set_eigen3_mode( int use_eigen3 );

/**
 * set the number of chunks to use in the batched eigen CUDA backend (which
 * operates with 32bit indices, thus enforcing an additional chunking). not
 * thread safe.
 */
void batched_eigen_set_nchunks( int nchunks );

/** get the number of chunks for CUDA. not thread safe. */
int batched_eigen_get_nchunks( void );

/**
 * we need this (annoying) function because some LAPACKE implementations won't
 * accept running with the implicitly set number of OMP threads
 */
void batched_eigen_set_nthreads( int nthr );

/**
 * disable or enable all ``mpi_..._printf()`` messaged in
 * :c:func:`batched_eigen` and :c:func:`batched_eigen_r`. not thread safe.
 */
void batched_eigen_shut_up( int silence );

#ifdef __cplusplus
}
#endif
