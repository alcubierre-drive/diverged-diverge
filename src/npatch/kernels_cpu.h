/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "vertex.h"
#include "loop.h"
#include "flow_step.h"

#ifdef __cplusplus
extern "C" {
#endif

struct npatch_cpu_kernel_start_t {
    index_t start_at;
    index_t stop_at;
    index_t size;
    npatch_vertex_t* V;
    npatch_loop_t* L;
    complex128_t* dPhi;

    double wtime;
};

void* npatch_cpu_flow_step_SU2_P( void* );
void* npatch_cpu_flow_step_SU2_C( void* );
void* npatch_cpu_flow_step_SU2_D1( void* );
void* npatch_cpu_flow_step_SU2_D2( void* );
void* npatch_cpu_flow_step_SU2_D3( void* );

void* npatch_cpu_flow_step_nonSU2_P( void* );
void* npatch_cpu_flow_step_nonSU2_C( void* );
void* npatch_cpu_flow_step_nonSU2_D1( void* );

#ifdef __cplusplus
}
#endif
