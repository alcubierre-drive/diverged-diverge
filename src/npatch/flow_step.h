/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct npatch_vertex_t npatch_vertex_t;
typedef struct npatch_loop_t npatch_loop_t;

struct npatch_flow_step_conf_t {
     bool SU2, P, C, D;
     double wtime_gpu, wtime_cpu, wtime_comm, wtime_max,
            wtime_loop_gpu, wtime_loop_cpu, wtime_loop_comm;
     double wtime_gpu_list[128];
     double wtime_loop_gpu_list[128];
     double wtime_loop_gpu_gf_list[128];

     double wtime_full;

     index_t niter;
};
typedef struct npatch_flow_step_conf_t npatch_flow_step_conf_t;

npatch_flow_step_conf_t* npatch_flow_step_conf_init( bool SU2 );
void npatch_flow_step_set_sub_D( bool D1, bool D2, bool D3 );
void npatch_flow_step_set_ignore_exchange_nonSU2( bool ignore );
void npatch_flow_step_set_vertex_resym( bool resym );

void npatch_setup_arch( const int* gpus, int ngpus, const int nblocks[3],
        float gpu_workshare, float loop_gpu_workshare );
void npatch_setup_query( const char* property, void** data, index_t* size );
void npatch_setup_reset( void );
void npatch_setup_teardown( void );

void npatch_flow_step( npatch_flow_step_conf_t* config, npatch_vertex_t* vertex, npatch_loop_t* loop,
                double* Lambda, double dLambda, double* chanmax );

// factor == 0.5 means we set up, factor == 0.25 means we average
void npatch_flow_step_enforce_exchange( complex128_t* V, complex128_t* aux,
        const index_t* k3map, index_t np, index_t nb, double factor );

#ifdef __cplusplus
}
#endif
