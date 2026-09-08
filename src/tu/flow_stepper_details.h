/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "flow_stepper.h"

#ifdef __cplusplus
extern "C" {
#endif

// char is expected to be 'H' and 'P' for particle-hole and particle-particle,
// respectively
typedef void (*tu_loop_func_t)(char ph, complex128_t* buf, void* ctx);

struct tu_stepper_t {
    double w_time_GF, w_time_full, w_time_loop, w_time_product,
           w_time_projection, w_time_add_find_max, w_time_self,
           w_time_katanin;

    tu_selfenergy_t* self;
    tu_vertex_t* internal_vertex;
    tu_prop_t* loop;
    tu_projection_t* proj;
    bool channel_selection[4];
    double norm[3];
    index_t prodsize;
    index_t nk;
    index_t my_nk;
    index_t num_calls;
    complex128_t* coarseGF;
    bool tu_selfenergy_flow;

    // approx. Katanin
    bool katanin;
    complex128_t *kat_pp, *kat_ph;
    complex128_t* self_shift;
    index_t full_vert_size;
    index_t selfenergy_size;

    tu_loop_func_t loop_func;
    void* loop_func_ctx;

    diverge_timing_t timings[64];

    uint64_t channel_product_ignore_map;
};

#ifdef __cplusplus
}
#endif
