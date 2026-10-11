/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

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
#include "tu/rpa.h"

#include "npatch/loop.h"
#include "npatch/flow_step.h"

// DEVELOPERS
//
// For a new backend, you can add arbitrary structures to the
// diverge_flow_step_t structure defined below.

#ifdef __cplusplus
extern "C" {
#endif

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
    tu_rpa_t* tu_rpa;

    double vertmax;
    double loopmax[2];
    double chanmax[3];

    double Lambda;
    double dLambda;

    index_t niter;
    VECTOR_DECL( diverge_timing_t, timings )

    double t_eigchan_diag;

    // eigenvalues from within the step (TU only), see
    // DIVERGE_FLOW_STEP_EIGCHAN_PITER_FROM_STEP. order: p,m,h,c,d
    int step_eig_on;
    int step_eig_piter;
    double step_eig_abort;
    double step_eig[5];
};

// called by the TU stepper with each projected vertex (scale*buf is channel
// chan, q-local and symmetry reduced) if step_eig_on. stores the eigenvalue,
// returns nonzero if the step must be aborted
int diverge_flow_step_eigchan_in_step( diverge_flow_step_t* st, char chan, const complex128_t* buf, double scale );

extern int diverge_flow_step_eigchan_shut_up;
extern index_t diverge_flow_step_eigchan_piter_maxiter;
extern double diverge_flow_step_eigchan_piter_tol;

#ifdef __cplusplus
}
#endif
