/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_flow_step.h"
#include "diverge_flow_type.h"
#include "diverge_internals_struct.h"

#include "grid/loops.hpp"
#include "grid/flow_step.hpp"

#include "tu/vertex.hpp"
#include "tu/projection.h"
#include "tu/propagator.h"
#include "tu/flow_stepper.hpp"

#include "npatch/loop.h"
#include "npatch/flow_step.h"

// DEVELOPERS
//
// For a new backend, you can add arbitrary C/C++ structs to the
// diverge_flow_step_t structure defined below.

struct diverge_flow_step_t {
    diverge_model_t* mod;
    internals_t* intls;
    diverge_flow_type type;

    // P, C, D, E
    bool channel_selection[4];

    // grid classes
    grid::Loops* grid_loop;
    grid::FlowStep* grid_step;

    // patch structs
    npatch_loop_t* patch_loop;
    npatch_flow_step_conf_t* patch_step_conf;

    // tu structs
    projection_t* tu_proj;
    tu_prop_t* tu_loop;
    Vertex* tu_vertex;
    tu_stepper* tu_step;

    double vertmax;
    double loopmax[2];
    double chanmax[3];

    vector<double>* timings;
    vector<string>* timings_descr;
    index_t niter;

    double Lambda;
    double dLambda;
};

