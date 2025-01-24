/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_zgemd.h"
#include "mpi_functions.h"
#include "../diverge_Eigen3.hpp"

#ifdef FORCE_HANDWRITTEN_ZGEMD

#ifndef USE_MKL
#include <cblas.h>
#else
#include <mkl.h>
#endif

static void batched_zgemd_r_smart( complex128_t* S, complex128_t* U, double* E,
        complex128_t alpha, complex128_t beta, index_t N, index_t num,
        complex128_t* buf = NULL ) {

    bool free_aux = false;
    if (!buf) {
        free_aux = true;
        buf = (complex128_t*)malloc(sizeof(complex128_t)*N*N*num);
    }

    const double E_add = *std::min_element( E, E+N*num );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<num; ++i)
    for (index_t b=0; b<N; ++b)
    for (index_t o=0; o<N; ++o)
        buf[IDX3(i,b,o,N,N)] = U[IDX3(i,b,o,N,N)] * sqrt(E[IDX2(i,b,N)] - E_add + DIVERGE_EPS_MESH);

    batched_gemm_N_ll_num( buf, buf, S, N, alpha, beta, num, false, true );

    if (free_aux) free( buf );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<num; ++i)
    for (index_t b=0; b<N; ++b)
        S[IDX3(i,b,b,N,N)] -= alpha * ( DIVERGE_EPS_MESH-E_add );

}

void batched_zgemd_r( complex128_t* S, complex128_t* U, double* E,
        complex128_t alpha, complex128_t beta, index_t N, index_t num ) {
    batched_zgemd_r_smart( S, U, E, alpha, beta, N, num );
}

#else // FORCE_HANDWRITTEN_ZGEMD

void batched_zgemd_r( complex128_t* S, complex128_t* U, double* E,
        complex128_t alpha, complex128_t beta, index_t N, index_t num ) {
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<num; ++i) {
        Map<MatXcd> UU(U + i*N*N, N, N);
        Map<VecXd> EE(E + i*N, N);
        Map<MatXcd> SS(S + i*N*N, N, N);

        SS = UU * EE.asDiagonal() * UU.adjoint() * alpha + SS * beta;
    }
}

#endif // FORCE_HANDWRITTEN_ZGEMD
