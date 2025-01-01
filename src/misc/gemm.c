/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "gemm.h"

#ifdef USE_MKL
#include <mkl.h>
#else // USE_MKL
#include <cblas.h>
#endif // USE_MKL

void single_gemm( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t M, index_t N, index_t K, complex128_t alpha, complex128_t beta) {
    index_t LDA = M;
    index_t LDB = K;
    index_t LDC = M;
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha,
                B, LDA, A, LDB, &beta, C, LDC );
}

void single_square_gemm( const complex128_t* A, int AconjTrans,
        const complex128_t* B, int BconjTrans, complex128_t* C, index_t n ) {
    const complex128_t one = 1.0, zero = 0.0;
    cblas_zgemm( CblasRowMajor, AconjTrans?CblasConjTrans:CblasNoTrans,
            BconjTrans?CblasConjTrans:CblasNoTrans, n, n, n, &one, A, n, B, n, &zero, C, n );
}
