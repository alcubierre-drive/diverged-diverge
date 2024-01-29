/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// batched SVD on square matrices. if n_dev < 0, use GPU with automatic
// devicing. if n_dev == 0, use CPU. if CUDA is not compiled in, always use CPU.
//
// U and V may be NULL, in which case nothing is copied into the output arrays.
// one of U or V may be the same as A, in which case the input is overwritten.
void batched_svd( int* dev, int n_dev, complex128_t* A, complex128_t* U,
                  complex128_t* V, double* S, index_t dim, index_t cnt );

#ifdef __cplusplus
}
#endif
