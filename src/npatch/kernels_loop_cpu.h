/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "loop.h"

#ifdef __cplusplus
extern "C" {
#endif

// stupid march=native stuff happening when mixing CUDA/C/C++
typedef int16_t npatch_fill16_t;
typedef int32_t npatch_fill32_t;
typedef index_t npatch_fill64_t;

struct npatch_loop_kernel_info_t {
    npatch_loop_t* L;
    npatch_fill64_t _pad0;

    index_t start, stop;

    int device;
    npatch_fill32_t _pad1;
    npatch_fill64_t _pad2;

    bool needs_gpu_copy;
    bool calc_gf_on_gpu;
    npatch_fill16_t _pad3;
    npatch_fill32_t _pad4;
    npatch_fill64_t _pad5;

    complex128_t Lambda;

    double wtime;
    npatch_fill64_t _pad6;
    double wtime_gpus[128];
    double wtime_gfs[128];
};
typedef struct npatch_loop_kernel_info_t npatch_loop_kernel_info_t;

void* npatch_loop_gen_cpu( void* data );

#ifdef __cplusplus
}
#endif
