/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_zgemd.h"
#include "mpi_functions.h"
#include "../diverge_Eigen3.hpp"

void batched_zgemd_r( complex128_t* S, complex128_t* U, double* E,
        complex128_t alpha, complex128_t beta, index_t N, index_t num ) {
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<num; ++i) {
        Map<MatXcd> UU(U + i*N*N, N, N);
        Map<VecXd> EE(E + i*N, N);
        Map<cMatXcd> SS(S + i*N*N, N, N);

        SS = UU * EE.asDiagonal() * UU.adjoint() * alpha + SS * beta;
    }
}
