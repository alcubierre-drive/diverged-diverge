/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_vertex_t tu_vertex_t;
typedef struct tu_projection_t tu_projection_t;
typedef struct diverge_model_t diverge_model_t;

void diverge_tu_sc_vertex( tu_vertex_t* V, tu_projection_t* P, char chan,
        const index_t* kidx, index_t n_kidx, const index_t* nk,
        complex128_t* out/*(n_kidx,n_spin,n_orb,n_spin,n_orb, n_kidx,n_spin,n_orb,n_spin,n_orb)*/ );
void diverge_tu_sc_greens_step( const diverge_model_t* m, const index_t* kidx,
        index_t n_kidx, double Lambda, double dLambda,
        complex128_t* out/*(n_kidx, n_spin,n_orb,n_spin,n_orb, n_spin,n_orb,n_spin,n_orb)*/ );
