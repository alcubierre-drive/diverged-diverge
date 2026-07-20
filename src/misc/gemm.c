/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "gemm.h"
#include "bitary.h"

#ifdef USE_MKL
#include <mkl.h>
#else // USE_MKL
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <cblas.h>
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif
#endif // USE_MKL

void single_gemm( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t M, index_t N, index_t K, complex128_t alpha, complex128_t beta) {
    const index_t LDA = M, LDB = K, LDC = M;
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha,
                B, LDA, A, LDB, &beta, C, LDC );
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

void transpose_inplace_rect_any( void* vdata, index_t data_sz, index_t rows, index_t cols ) {
    unsigned char* data = vdata;
    index_t size = rows * cols;
    index_t visited_count = 0;
    cbytes_t* visited = bitary_alloc(size);

    for (index_t start=0; visited_count<size; ++start) {
        if (bitary_get(visited,start)) continue;

        index_t current = start;
        unsigned char tmp[data_sz];
        memcpy( tmp, data + data_sz*current, data_sz );

        do {
            index_t next = (current % cols) * rows + (current / cols);
            unsigned char swap[data_sz];
            memcpy( swap, data + data_sz*next, data_sz );
            memcpy( data + data_sz*next, tmp, data_sz );

            bitary_set(visited, current);
            visited_count++;

            current = next;
            memcpy( tmp, swap, data_sz );
        } while (current != start);
    }

    free(visited);
}

void transpose_outofplace( complex128_t* B, const complex128_t* A, index_t N ) {
    for (index_t i=0; i<N; ++i) for (index_t j=0; j<N; ++j) B[IDX2(i,j,N)] = A[IDX2(j,i,N)];
}

void transpose_conj_outofplace( complex128_t* B, const complex128_t* A, index_t N ) {
    for (index_t i=0; i<N; ++i) for (index_t j=0; j<N; ++j) B[IDX2(i,j,N)] = conj(A[IDX2(j,i,N)]);
}

#ifdef USE_CUDA
#include <cublasXt.h>
#include "cuda_error.h"

struct single_gemm_Xt_t {
    cublasXtHandle_t t;
};
single_gemm_Xt_t* single_gemm_Xt_init( void ) {
    single_gemm_Xt_t* x = calloc(1, sizeof*x);
    CUBLAS_CHECK( cublasXtCreate( &x->t ) );
    int ndev = -1;
    CUDA_CHECK( cudaGetDeviceCount( &ndev ) );
    int* dev = calloc(ndev, sizeof*dev);
    for (int i=0; i<ndev; ++i) dev[i] = i;
    CUBLAS_CHECK( cublasXtDeviceSelect( x->t, ndev, dev ) );
    CUBLAS_CHECK( cublasXtSetBlockDim( x->t, 8192 ) );
    CUBLAS_CHECK( cublasXtSetCpuRatio( x->t, CUBLASXT_GEMM, CUBLASXT_DOUBLECOMPLEX, 0.0 ) );
    free( dev );
    return x;
}
void single_gemm_Xt_free( single_gemm_Xt_t* x ) {
    CUBLAS_CHECK( cublasXtDestroy( x->t ) );
    free( x );
}
void single_gemm_Xt( single_gemm_Xt_t* x, const complex128_t* A,
        const complex128_t* B, complex128_t* C, index_t M, index_t N, index_t K,
        complex128_t alpha, complex128_t beta ) {
    if (x) {
        const index_t LDA = M, LDB = K, LDC = M;
        CUBLAS_CHECK( cublasXtZgemm( x->t, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuDoubleComplex*)&alpha,
                    (const cuDoubleComplex*)B, LDA, (const cuDoubleComplex*)A, LDB, (cuDoubleComplex*)&beta,
                    (cuDoubleComplex*)C, LDC ) );
    } else {
        mpi_wrn_printf( "Xt GEMM not initialized, fallback to standard gemm\n" );
        single_gemm( A, B, C, M, N, K, alpha, beta );
    }
}

#else // USE_CUDA

single_gemm_Xt_t* single_gemm_Xt_init( void ) { return NULL; }
void single_gemm_Xt_free( single_gemm_Xt_t* x ) { (void)x; }
void single_gemm_Xt( single_gemm_Xt_t* x, const complex128_t* A,
        const complex128_t* B, complex128_t* C, index_t M, index_t N, index_t K,
        complex128_t alpha, complex128_t beta ) {
    (void)x;
    single_gemm( A, B, C, M, N, K, alpha, beta );
}

#endif // USE_CUDA
