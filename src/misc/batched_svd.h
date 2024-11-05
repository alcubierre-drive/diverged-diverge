/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * batched SVD on square matrices. if ``n_dev < 0``, use GPU with automatic
 * devicing. if ``n_dev == 0``, use CPU. if CUDA is not compiled in, always use
 * CPU. Notably, there is no batched SVD algorithm in CUSOLVER. Therefore, we
 * manually up/download the matrices to the GPUs, making this solver efficient
 * as an eigensolver for larger matrices (see :c:func:`batched_eigen` for small
 * matrices).
 *
 * special features:
 *
 *   - if ``n_dev == INT_MIN``, do eigensolution instead of SVD. The result is
 *     written to U, and V is not touched.
 *   - if ``n_dev == INT_MIN+1`` same as above, but force CPU algorithm even if
 *     CUDA is compiled in
 *
 * ``U`` and ``V`` may be ``NULL``, in which case nothing is copied into the
 * output arrays. one of ``U`` or ``V`` may be the same as ``A``, in which case
 * the input is overwritten.
 */
void batched_svd( int* dev, int n_dev, complex128_t* A, complex128_t* U,
                  complex128_t* V, double* S, index_t dim, index_t cnt );

#ifdef __cplusplus
}
#endif
