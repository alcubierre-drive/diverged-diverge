/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;
typedef struct grid_symm_transform_t grid_symm_transform_t;

typedef struct grid_loop_t grid_loop_t;
struct grid_loop_t {
    complex128_t *X_P, *X_C;
    double max_P, max_C;
    index_t qstart, qsize, nk, nb, Xsize;

    index_t nkf, *kfrev, *k1m2;
    double* kfweights;

    complex128_t* formfac;
    index_t nff;

    // timing
    double t_sym;

    // private
    grid_symm_transform_t* SYM;
    bool SYM_enabled;
    double greens_cache_lambda;
    diverge_model_t* dmod;
    complex128_t *greens_cache_p, *greens_cache_m,
                 *greens_proj_cache_p, *greens_proj_cache_m;
};

grid_loop_t* grid_loop_init( diverge_model_t* model );
void grid_loop_free( grid_loop_t* l );
complex128_t* grid_loop_calc( grid_loop_t* l, char chan, double Lambda );
void grid_loop_toggle_symmetrizer( grid_loop_t* l );

#ifdef __cplusplus
}
#endif
