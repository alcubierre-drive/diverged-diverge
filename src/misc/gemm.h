/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * calculates the GEMM :math:`C_{nm} = \alpha A_{nk} B_{km} + \beta C_{nm]` with
 * A, B, C matrices in C order.
 */
void single_gemm( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t M, index_t N, index_t K, complex128_t alpha, complex128_t beta);

/**
 * calculates the GEMM :math:`C_{12} = A_{13} B_{32}` for square matrices stored
 * in C order, i.e., Column-Major. Allows to do Hermitian transposition on
 * either A or B (if the integer argument is nozero).
 */
void single_square_gemm( const complex128_t* A, int AconjTrans,
        const complex128_t* B, int BconjTrans, complex128_t* C, index_t n );

#ifdef __cplusplus
}
#endif
