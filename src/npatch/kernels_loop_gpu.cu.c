/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "kernels_loop_gpu.h"

#ifndef USE_CUDA

void* npatch_loop_gen_gpu( void* data ) {
    (void)(data);
    mpi_err_printf("CUDA not compiled\n");
    return NULL;
}

void npatch_loop_gpu_set_weights( int* weights ) {
    (void)weights;
}

void npatch_loop_gpu_teardown( void ) {
}

#endif
