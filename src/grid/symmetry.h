/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

typedef struct grid_symm_transform_t grid_symm_transform_t;
grid_symm_transform_t* grid_symm_transform_init( diverge_model_t* m, index_t qdispl, index_t qcount );
void grid_symm_transform_free( grid_symm_transform_t* s );

complex128_t* grid_symm_transform_loop( grid_symm_transform_t* s, complex128_t* loop, char chan );
bool grid_symm_transform_loop_checksyms( grid_symm_transform_t* s, complex128_t* loop, char chan );

#ifdef __cplusplus
}
#endif
