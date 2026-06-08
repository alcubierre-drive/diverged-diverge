/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/** for python wrapper */
rs_hopping_t* diverge_mem_alloc_rs_hopping_t( index_t num );

/** for python wrapper */
rs_vertex_t* diverge_mem_alloc_rs_vertex_t( index_t num );

/** for python wrapper */
tu_formfactor_t* diverge_mem_alloc_tu_formfactor_t( index_t num );

/** for python wrapper */
double* diverge_mem_alloc_complex128_t( index_t num );

/** for python wrapper */
void* diverge_mem_alloc( index_t num );

/** for python wrapper, should not be required */
void diverge_mem_free( void* addr );

#ifdef __cplusplus
}
#endif
