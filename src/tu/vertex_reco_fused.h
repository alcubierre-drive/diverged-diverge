/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_vertex_t tu_vertex_t;
typedef struct tu_projection_t tu_projection_t;
typedef struct diverge_model_t diverge_model_t;
typedef struct tu_reco_fused_t tu_reco_fused_t;

/**
 * Tables that fold the symmetry reconstruction into the Fourier transform of
 * :c:func:`diverge_tu_vertex_reco_ft`, see vertex_reco_fused.c. Returns NULL if
 * the model does not admit the construction (caller falls back to the two-step
 * implementation).
 */
tu_reco_fused_t* tu_reco_fused_init( const diverge_model_t* model, const tu_projection_t* proj );
void tu_reco_fused_free( tu_reco_fused_t* f );

/** number of real space vectors the fused GEMM runs over (diagnostics) */
index_t tu_reco_fused_nu( const tu_reco_fused_t* f );
/** number of real space vectors of the projection the tables were built for */
index_t tu_reco_fused_nr( const tu_reco_fused_t* f );
/** BZ points left to the two-step path (high symmetry momenta) */
index_t tu_reco_fused_nslow( const tu_reco_fused_t* f );

/** fused replacement of diverge_tu_vertex_reco_ft(), CPU only */
void tu_reco_fused_ft( const tu_reco_fused_t* f, const tu_vertex_t* V, char chan,
        complex128_t* buf );

#ifdef __cplusplus
}
#endif
