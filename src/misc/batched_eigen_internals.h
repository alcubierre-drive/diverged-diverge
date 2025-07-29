/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int device;
    complex128_t* H;
    double* E;
    index_t n;
    index_t N;
    index_t nbatches;
} batched_eigen_t;

void* batched_eigen_gpu( void* );
void* batched_eigen_cpu( void* );

void batched_zheevd( const complex128_t* H, complex128_t* U, double* E, index_t num, index_t dim );

#ifdef __cplusplus
}
#endif
