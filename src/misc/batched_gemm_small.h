/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct batched_gemm_small_handle_t batched_gemm_small_handle_t;

batched_gemm_small_handle_t* batched_gemm_small_init( index_t n, index_t num );
void batched_gemm_small_destroy( batched_gemm_small_handle_t* h );

// to change the automatically generated mode (either CPU or GPU depending on
// how the compilation went) to something else, we can use this function on an
// existing handle
typedef enum {
    batched_gemm_small_mode_cpu,
    batched_gemm_small_mode_gpu,
} batched_gemm_small_mode_t;
void batched_gemm_small_mode( batched_gemm_small_handle_t* h, batched_gemm_small_mode_t mode );

// assumes C ordering for all matrices, with one matrix being in memory straight
// after the other and no additional strides within the matrices.
void batched_gemm_small( batched_gemm_small_handle_t* h, const complex128_t* A,
        const complex128_t* B, complex128_t* C );

#ifdef __cplusplus
}
#endif
