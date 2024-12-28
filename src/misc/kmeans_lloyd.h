/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int dkm_use_random_seed( void );
void dkm_set_use_random_seed( int use );

#ifndef SKIP_CLAV2D_T
typedef struct { double v[2]; } claV2d_t;
#endif

typedef struct {
    claV2d_t* points;
    index_t* clusters;
    index_t k;
} dkm_result_t;

dkm_result_t kmeans_lloyd( const claV2d_t* pts, index_t n_pts, index_t k,
        double limit, index_t maxiter );
void dkm_result_free( dkm_result_t x );

#ifdef __cplusplus
}
#endif
