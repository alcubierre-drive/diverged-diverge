/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/** forward declaration of opaque struct */
typedef struct batched_gemm_small_handle_t batched_gemm_small_handle_t;

/** initialization */
batched_gemm_small_handle_t* batched_gemm_small_init( index_t n, index_t num );
/** free resources */
void batched_gemm_small_destroy( batched_gemm_small_handle_t* h );

/** collection of operation modes for :c:func:`batched_gemm_small` */
typedef enum {
    /** CPU mode */
    batched_gemm_small_mode_cpu,
    /** GPU mode */
    batched_gemm_small_mode_gpu,
} batched_gemm_small_mode_t;

/**
 * to change the automatically generated mode (either CPU or GPU depending on
 * how the compilation went) to something else, we can use this function on an
 * existing handle
 */
void batched_gemm_small_mode( batched_gemm_small_handle_t* h, batched_gemm_small_mode_t mode );

/**
 * assumes C ordering for all matrices, with one matrix being in memory straight
 * after the other and no additional strides within the matrices.
 */
void batched_gemm_small( batched_gemm_small_handle_t* h, const complex128_t* A,
        const complex128_t* B, complex128_t* C );

#ifdef __cplusplus
}
#endif
