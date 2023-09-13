/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

// TODO(LK) the batched zgemd in its 'smart' version is not working the way we'd
// want it to. In case anyone spots the bug: It should be as fast/faster than
// what's written below...
#if 0
#include "batched_zgemd.h"
#include "batched_gemms.h"
#include "../diverge_Eigen3.hpp"

void batched_zgemd_r( complex128_t* S, complex128_t* U, double* E,
        complex128_t alpha, complex128_t beta, index_t N, index_t num ) {

    complex128_t* buf = (complex128_t*)malloc(sizeof(complex128_t)*N*N*num);

    const double E_add = *std::min_element( E, E+N*num ) + DIVERGE_EPS_MESH;

    #pragma omp parallel for collapse(3)
    for (index_t i=0; i<num; ++i)
    for (index_t b=0; b<N; ++b)
    for (index_t o=0; o<N; ++o)
        buf[IDX3(i,b,o,N,N)] = U[IDX3(i,b,o,N,N)] * sqrt(E[IDX2(i,b,N)]+E_add);

    batched_gemm_N_ll_num( buf, buf, S, N, alpha, beta, num, false, true );

    free( buf );

    #pragma omp parallel for collapse(2)
    for (index_t i=0; i<num; ++i)
    for (index_t b=0; b<N; ++b)
        S[IDX3(i,b,b,N,N)] -= alpha * E_add;

}
#endif

#include "batched_zgemd.h"
#include "../diverge_Eigen3.hpp"

void batched_zgemd_r( complex128_t* S, complex128_t* U, double* E,
        complex128_t alpha, complex128_t beta, index_t N, index_t num ) {
    #pragma omp parallel for
    for (index_t i=0; i<num; ++i) {
        Map<MatXcd> UU(U + i*N*N, N, N);
        Map<VecXd> EE(E + i*N, N);
        Map<MatXcd> SS(S + i*N*N, N, N);

        SS = UU * EE.asDiagonal() * UU.adjoint() * alpha + SS * beta;
    }
}
