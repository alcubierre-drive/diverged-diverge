/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "vertex.h"
#include "flow_step.h"
#include "loop.h"

#ifdef __cplusplus
extern "C" {
#endif

struct npatch_gpu_kernel_start_t {
    index_t start_at;
    index_t stop_at;
    index_t size;
    npatch_vertex_t* V;
    npatch_loop_t* L;
    complex128_t* dPhi;

    int device;

    char chan[16];
    bool SU2;

    double wtime;
    double wtime_gpus[128];
};

void npatch_flow_gpu_setup( const int* gpus, int ngpus, const int nblocks[3] );
int npatch_flow_gpu_device_count( void );

void npatch_flow_gpu_set_weights( const int* gpu_weights );
void npatch_flow_gpu_reset( void );
void npatch_flow_gpu_teardown( void );

void* npatch_gpu_flow_step_SU2_P( void* );
void* npatch_gpu_flow_step_SU2_C( void* );
void* npatch_gpu_flow_step_SU2_D1( void* );
void* npatch_gpu_flow_step_SU2_D2( void* );
void* npatch_gpu_flow_step_SU2_D3( void* );

void* npatch_gpu_flow_step_nonSU2_P( void* );
void* npatch_gpu_flow_step_nonSU2_C( void* );
void* npatch_gpu_flow_step_nonSU2_D1( void* );

void npatch_setup_query_gpu( const char* property, void** data, index_t* size );

#ifdef __cplusplus
}
#endif
