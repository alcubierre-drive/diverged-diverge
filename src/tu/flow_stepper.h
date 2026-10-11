/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../diverge_flow_step.h" // timing
#include "propagator.h"

typedef struct tu_projection_t tu_projection_t;
typedef struct tu_vertex_t tu_vertex_t;
typedef struct tu_selfenergy_t tu_selfenergy_t;

typedef struct tu_stepper_t tu_stepper_t;

#ifdef __cplusplus
extern "C" {
#endif

tu_stepper_t* tu_stepper_init( tu_projection_t* proj, tu_prop_t* loop, tu_vertex_t* vert );
void tu_stepper_free( tu_stepper_t* s );
// if eig != NULL, the projected vertices are passed to
// diverge_flow_step_eigchan_in_step (may abort the step). returns nonzero if
// the step was aborted (vertex unchanged)
int tu_stepper_euler( tu_stepper_t* s, tu_vertex_t* inout, double lambda, double dlambda, double maxs[6],
        diverge_flow_step_t* eig );

// useful if you want to hack around, chans: bitwise P,C,D,CD; returns prev.
uint64_t tu_stepper_prodignore( tu_stepper_t* s, uint64_t chans );

// returns static memory, don't free!
diverge_timing_t* tu_stepper_timings( tu_stepper_t* s, int* num );
index_t tu_stepper_num_calls( const tu_stepper_t* s );

// only needed for postproc
tu_vertex_t* tu_stepper_internal_vertex( tu_stepper_t* s );

#ifdef __cplusplus
}
#endif
