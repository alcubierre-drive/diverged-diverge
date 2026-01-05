/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * calculates the matrix product :math:`C = \beta C + \alpha A \cdot B`. A(N,K),
 * B(K,M) and C(N,M) are matrices in C order (Row Major).
 */
void single_gemm( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t M, index_t N, index_t K, complex128_t alpha, complex128_t beta );

/**
 * calculate the matrix product :math:`C(N,K) = alpha * A(N,M) * B(M,K) + beta *
 * C(N,K)` with all matrices in C order (Row Major), but with an Eigen3 backend
 * (so no parallelization issues but less ideal runtime).
 */
void single_gemm_serial( const complex128_t* A, const complex128_t* B,
        complex128_t* C, index_t N, index_t M, index_t K,
        complex128_t alpha, complex128_t beta );

/**
 * calculates the matrix product :math:`C = \beta C + \alpha A \cdot B^\dagger`.
 * A(N,K), B(M,K) and C(N,M) are matrices in C order (Row Major).
 */
void single_gemm_adjoint( const complex128_t* A, const complex128_t* B,
        complex128_t* C, index_t M, index_t N, index_t K, complex128_t alpha,
        complex128_t beta);

/**
 * calculates the GEMM :math:`C_{12} = A_{13} B_{32}` for square matrices stored
 * in C order, i.e., Column-Major. Allows to do Hermitian transposition on
 * either A or B (if the integer argument is nozero).
 */
void single_square_gemm( const complex128_t* A, int AconjTrans,
        const complex128_t* B, int BconjTrans, complex128_t* C, index_t n );

/** in-place transposition of square matrix ``A`` (dimension NxN) */
void transpose_inplace( complex128_t* A, index_t N );

/** out-of-place transposition of square matrix ``A`` into ``B`` (dimension NxN) */
void transpose_outofplace( complex128_t* B, const complex128_t* A, index_t N );

/**
 * quasi in-place transposition of rectangular matrix of any datatype (size
 * ``sizeof_A``), C-ordered ``A(N,M)``
 */
void transpose_inplace_rect_any( void* A, index_t sizeof_A, index_t N, index_t M );

/** out-of-place adjoint of square matrix ``A`` into ``B`` (dimension NxN) */
void transpose_conj_outofplace( complex128_t* B, const complex128_t* A, index_t N );

typedef struct single_gemm_Xt_t single_gemm_Xt_t;
single_gemm_Xt_t* single_gemm_Xt_init( void );
void single_gemm_Xt_free( single_gemm_Xt_t* x );
void single_gemm_Xt( single_gemm_Xt_t* x, const complex128_t* A,
        const complex128_t* B, complex128_t* C, index_t M, index_t N, index_t K,
        complex128_t alpha, complex128_t beta );

#ifdef __cplusplus
}
#endif
