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

typedef struct { VECTOR_DECL( claV2d_t, v ) } vecVec2d_t;
typedef struct { VECTOR_DECL( uint32_t, v ) } vecU32_t;

typedef struct {
    vecVec2d_t points;
    vecU32_t clusters;
} dkm_result_t;

dkm_result_t kmeans_lloyd( const vecVec2d_t data, uint32_t k, double limit, index_t maxiter );
void dkm_result_free( dkm_result_t r );

#ifdef __cplusplus
}
#endif
