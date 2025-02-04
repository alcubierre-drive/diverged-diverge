/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "gemm.h"

#ifdef USE_MKL
#include <mkl.h>
#else // USE_MKL
#include <cblas.h>
#endif // USE_MKL

void single_gemm( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t M, index_t N, index_t K, complex128_t alpha, complex128_t beta) {
    cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, &alpha, A, K,
            B, M, &beta, C, M );
}

void single_gemm_adjoint( const complex128_t* A, const complex128_t* B,
        complex128_t* C, index_t M, index_t N, index_t K, complex128_t alpha,
        complex128_t beta) {
    cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasConjTrans, N, M, K, &alpha, A, K,
            B, K, &beta, C, M );
}

void single_square_gemm( const complex128_t* A, int AconjTrans,
        const complex128_t* B, int BconjTrans, complex128_t* C, index_t n ) {
    const complex128_t one = 1.0, zero = 0.0;
    cblas_zgemm( CblasRowMajor, AconjTrans?CblasConjTrans:CblasNoTrans,
            BconjTrans?CblasConjTrans:CblasNoTrans, n, n, n, &one, A, n, B, n, &zero, C, n );
}

void transpose_inplace( complex128_t* A, index_t N ) {
    for (index_t i=0; i<N; ++i)
    for (index_t j=i+1; j<N; ++j) {
        const complex128_t tmp = A[IDX2(i,j,N)];
        A[IDX2(i,j,N)] = A[IDX2(j,i,N)];
        A[IDX2(j,i,N)] = tmp;
    }
}

void transpose_outofplace( complex128_t* B, const complex128_t* A, index_t N ) {
    for (index_t i=0; i<N; ++i) for (index_t j=0; j<N; ++j) B[IDX2(i,j,N)] = A[IDX2(j,i,N)];
}

void transpose_conj_outofplace( complex128_t* B, const complex128_t* A, index_t N ) {
    for (index_t i=0; i<N; ++i) for (index_t j=0; j<N; ++j) B[IDX2(i,j,N)] = conj(A[IDX2(j,i,N)]);
}
