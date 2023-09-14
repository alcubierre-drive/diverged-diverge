/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "gemm.h"

#ifdef USE_MKL
#include <mkl.h>
#else // USE_MKL
#include <cblas.h>
#endif // USE_MKL

#if defined(USE_LAPACKE) && !defined(USE_MKL)
#include <lapacke.h>
#endif // LAPACKE

#include <omp.h>
#include <stdio.h>

void gemm( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t M, index_t N, index_t K, complex128_t alpha,
        complex128_t beta) {
    index_t LDA = M;
    index_t LDB = K;
    index_t LDC = M;
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha,
                Bstart, LDA, Astart, LDB, &beta, Cstart, LDC );
}
