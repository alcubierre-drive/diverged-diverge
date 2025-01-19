/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../diverge_flow_step.h" // timing

typedef struct grid_loop_t grid_loop_t;
typedef struct grid_vertex_t grid_vertex_t;
typedef struct grid_flow_step_t grid_flow_step_t;

#ifdef __cplusplus
extern "C" {
#endif

grid_flow_step_t* grid_flow_step_init( grid_vertex_t* V, grid_loop_t* L,
        bool P/*= true*/, bool C/*= true*/, bool D/*= true*/ );
void grid_flow_step_free( grid_flow_step_t* s );
// returns an array of size 6 that must *not* be freed! Pmax, Cmax, Dmax, Vmax, LPmax, LDmax
const double* grid_flow_step_euler( grid_flow_step_t* s, double Lambda, double dLambda );

// returns an array of size 6 that must *not* be freed!
diverge_timing_t* grid_flow_step_timings( grid_flow_step_t* s );
index_t grid_flow_step_num_calls( const grid_flow_step_t* s );

#ifdef __cplusplus
}
#endif
