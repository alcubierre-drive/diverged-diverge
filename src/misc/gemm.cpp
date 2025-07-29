#include "gemm.h"
#include "../diverge_Eigen3.hpp"

void single_gemm_serial( const complex128_t* A, const complex128_t* B,
        complex128_t* C, index_t N, index_t M, index_t K,
        complex128_t alpha, complex128_t beta ) {
    Map<const cMatXcd> _A(A, N,M);
    Map<const cMatXcd> _B(B, M,K);
    Map<cMatXcd> _C(C, N,K);
    _C = beta * _C + alpha * _A * _B;
}
