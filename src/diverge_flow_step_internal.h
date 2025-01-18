/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_flow_step.h"
#include "diverge_flow_type.h"
#include "diverge_internals_struct.h"

#include "grid/loops.h"
#include "grid/flow_step.h"

#include "tu/vertex.h"
#include "tu/projection.h"
#include "tu/propagator.h"
#include "tu/flow_stepper.h"

#include "npatch/loop.h"
#include "npatch/flow_step.h"

// DEVELOPERS
//
// For a new backend, you can add arbitrary structures to the
// diverge_flow_step_t structure defined below.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double t;
    char nam[56];
} diverge_flow_step_timing_t;

struct diverge_flow_step_t {
    diverge_model_t* mod;
    internals_t* intls;
    diverge_flow_type type;

    // P, C, D, E
    bool channel_selection[4];

    // grid classes
    grid_loop_t* grid_loop;
    grid_flow_step_t* grid_step;

    // patch structs
    npatch_loop_t* patch_loop;
    npatch_flow_step_conf_t* patch_step_conf;

    // tu structs
    tu_projection_t* tu_proj;
    tu_prop_t* tu_loop;
    tu_vertex_t* tu_vertex;
    tu_stepper_t* tu_step;

    double vertmax;
    double loopmax[2];
    double chanmax[3];

    double Lambda;
    double dLambda;

    index_t niter;
    VECTOR_DECL( diverge_flow_step_timing_t, timings )
};

#ifdef __cplusplus
}
#endif
