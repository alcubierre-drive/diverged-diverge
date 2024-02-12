/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int device;
    complex128_t* H;
    complex128_t* E;
    double* Er;
    index_t n;
    index_t N;
    index_t nbatches;
} batched_eigen_t;

void* batched_eigen_gpu( void* );
void* batched_eigen_cpu( void* );

#ifdef __cplusplus
}
#endif
