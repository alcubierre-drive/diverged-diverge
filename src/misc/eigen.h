#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void eigen_solve( const complex128_t* H, complex128_t* U, double* E, index_t N );
void singular_value_decomp( const complex128_t* H, complex128_t* U, complex128_t* V, double* S, index_t N );

// sorting options:
// 0: default ('M')
// 'A': alternating (max negative, max positive, ...)
// 'P': max positive
// 'N': max negative
// 'M': magnitude
// 'm': inverse magnitude
void eigensolution_sort( complex128_t* U, double* E, index_t N, char which );

#ifdef __cplusplus
}
#endif
