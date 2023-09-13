/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "kernels_loop_cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void* npatch_loop_gen_gpu( void* data );

void npatch_loop_gpu_teardown( void );

#ifdef __cplusplus
}
#endif
