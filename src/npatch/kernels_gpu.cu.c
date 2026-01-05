/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "kernels_gpu.h"

#ifndef USE_CUDA

void* npatch_gpu_flow_step_SU2_P( void* params ) {
    (void)params;
    mpi_err_printf("CUDA not compiled in\n");
    return NULL;
}
void* npatch_gpu_flow_step_SU2_C( void* params ) {
    (void)params;
    mpi_err_printf("CUDA not compiled in\n");
    return NULL;
}
void* npatch_gpu_flow_step_SU2_D1( void* params ) {
    (void)params;
    mpi_err_printf("CUDA not compiled in\n");
    return NULL;
}
void* npatch_gpu_flow_step_SU2_D2( void* params ) {
    (void)params;
    mpi_err_printf("CUDA not compiled in\n");
    return NULL;
}
void* npatch_gpu_flow_step_SU2_D3( void* params ) {
    (void)params;
    mpi_err_printf("CUDA not compiled in\n");
    return NULL;
}

void* npatch_gpu_flow_step_nonSU2_P( void* params ) {
    (void)params;
    mpi_err_printf("CUDA not compiled in\n");
    return NULL;
}
void* npatch_gpu_flow_step_nonSU2_C( void* params ) {
    (void)params;
    mpi_err_printf("CUDA not compiled in\n");
    return NULL;
}
void* npatch_gpu_flow_step_nonSU2_D1( void* params ) {
    (void)params;
    mpi_err_printf("CUDA not compiled in\n");
    return NULL;
}

void npatch_flow_gpu_setup( const int* gpus, int ngpus, const int nblocks[3] ) {
    (void)gpus;
    (void)ngpus;
    (void)nblocks;
}

int npatch_flow_gpu_device_count( void ) {
    return 0;
}

void npatch_flow_gpu_set_weights( const int* gpu_weights ) {
    (void)gpu_weights;
}

void npatch_setup_query_gpu( const char* property, void** data, index_t* size ) {
    (void)property;
    (void)data;
    (void)size;
}

void npatch_flow_gpu_reset(void) {}
void npatch_flow_gpu_teardown(void) {}

#endif
