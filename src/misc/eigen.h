/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void single_eigen( const complex128_t* H, complex128_t* U, double* E, index_t N );
void single_svd( const complex128_t* H, complex128_t* U, complex128_t* V, double* S, index_t N );

// sorting options:
// 0: default ('M')
// 'A': alternating (max negative, max positive, ...)
// 'P': max positive
// 'N': max negative
// 'M': magnitude
// 'm': inverse magnitude
void single_eigen_sort( complex128_t* U, double* E, index_t N, char which );

// sorting options are the same as above (which). in addition, we allow for
// relative or absolute cutoff in abs_limit. if abs_limit<0, use relative
// cutoff. if abs_limit>0, use absolute cutoff. returns the number of elements
// the satify the cutoff condition. if abs_limit is all zero bytes, do not
// cutoff the eigensolution.
index_t single_eigen_sort_N( complex128_t* U, double* E, index_t N, char which, double abs_limit );

#ifdef __cplusplus
}
#endif
