#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// calculate batched 'zgemd' product
// out = U * diag(E) * adjoint(U) * alpha + out * beta
// for the arrays out( num, N, N ), U( num, N, N ), E( num, N )

// TODO implement GPU version
void batched_zgemd_r( complex128_t* out, complex128_t* U, double* E,
        complex128_t alpha, complex128_t beta, index_t N, index_t num );

#ifdef __cplusplus
}
#endif
