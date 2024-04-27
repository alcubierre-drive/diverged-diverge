/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * :math:`C_{n,m} = \beta C_{n,m] + \alpha A_{n,k} B_{k,m}`, with A, B, C
 * matrices in C order.
 */
void single_gemm( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t M, index_t N, index_t K, complex128_t alpha, complex128_t beta);

#ifdef __cplusplus
}
#endif
