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
