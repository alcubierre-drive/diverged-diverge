/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifndef DIVERGE_POST_MAGIC_NUMBER_TU
#define DIVERGE_POST_MAGIC_NUMBER_TU 'T'
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_vertex_t tu_vertex_t;
typedef struct tu_projection_t tu_projection_t;
typedef struct tu_rpa_t tu_rpa_t;
typedef struct diverge_model_t diverge_model_t;

typedef struct {
    bool skip_channel_calc;
    index_t tu_n_singular_values;
    char algo;
    double algo_zero;
    double cutoff;
    bool cutoff_rel;
    index_t tu_n_decomp_values;
    int bare_susceptibilities;
    int channel_calc_project;
    char tu_lingap_solver_mode[4];

    bool lingaps;
    bool susc_ff;
    bool susc_full;
    bool lingap_atscale;
    int save_self;
    int save_full;
    bool save_symmetries;

    char rpa_chans[6];
    char rpa_include_chi_pp;
    char rpa_include_V_pp;
    int rpa_nev;
    int rpa_ncv;
    int rpa_maxiter;
    double rpa_tol;

    float self_rs_dist;
    float self_rs_mval;
} tu_postproduction_conf_t;
#ifndef __cplusplus
static const tu_postproduction_conf_t tu_postproduction_conf_defaults = {
    .skip_channel_calc = false,
    .tu_n_singular_values = 1,
    .algo = 'a',
    .algo_zero = 1.e-11,
    .cutoff = 25.,
    .cutoff_rel = false,
    .tu_n_decomp_values = 0,
    .bare_susceptibilities = 0,
    .channel_calc_project = false,
    .tu_lingap_solver_mode = "sss",
    .lingaps = true,
    .susc_ff = false,
    .susc_full = false,
    .lingap_atscale = false,
    .save_self = 0,
    .save_full = 0,
    .save_symmetries = false,
    .rpa_chans = "CD",
    .rpa_include_chi_pp = 0,
    .rpa_include_V_pp = 0,
    .rpa_nev = 10,
    .rpa_ncv = 100,
    .rpa_maxiter = 100,
    .rpa_tol = 1.e-6,
    .self_rs_dist = 4.01,
    .self_rs_mval = 1.e-4,
};
#endif // __cplusplus

void tu_postproduction( tu_vertex_t* vertex, tu_vertex_t* internal_vertex,
        tu_projection_t* proj, tu_rpa_t* rpa, const diverge_model_t* model,
        const tu_postproduction_conf_t* c, const char* filename );

#ifdef __cplusplus
}
#endif
