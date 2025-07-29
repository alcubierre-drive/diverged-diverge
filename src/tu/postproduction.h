/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifndef TU_MAGIC_NUMBER_POST_PROCESSING
#define TU_MAGIC_NUMBER_POST_PROCESSING 'T'
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_vertex_t tu_vertex_t;
typedef struct tu_projection_t tu_projection_t;
typedef struct diverge_model_t diverge_model_t;

typedef struct {
    bool skip_channel_calc;
    index_t tu_n_singular_values;
    char algo;
    double cutoff;
    bool cutoff_rel;
    index_t tu_n_decomp_values;
    bool bare_susceptibilities;
    int channel_calc_project;
    char tu_lingap_solver_mode[4];

    bool lingaps;
    bool susc_ff;
    bool susc_full;
    bool lingap_atscale;
    bool save_self;
    bool save_full;
    bool save_symmetries;
} tu_postproduction_conf_t;
#ifndef __cplusplus
static const tu_postproduction_conf_t tu_postproduction_conf_defaults = {
    .skip_channel_calc = false,
    .tu_n_singular_values = 1,
    .algo = 'a',
    .cutoff = 25.,
    .cutoff_rel = false,
    .tu_n_decomp_values = 0,
    .bare_susceptibilities = false,
    .channel_calc_project = false,
    .tu_lingap_solver_mode = "sss",
    .lingaps = true,
    .susc_ff = false,
    .susc_full = false,
    .lingap_atscale = false,
    .save_self = false,
    .save_full = false,
    .save_symmetries = false,
};
#endif // __cplusplus
tu_postproduction_conf_t tu_postproduction_conf_defaults_CPP( void );

void tu_postproduction( tu_vertex_t* vertex, tu_vertex_t* internal_vertex,
        tu_projection_t* proj, const diverge_model_t* model,
        const tu_postproduction_conf_t* c, const char* filename );

#ifdef __cplusplus
}
#endif
