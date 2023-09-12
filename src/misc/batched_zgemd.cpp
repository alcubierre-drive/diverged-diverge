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
