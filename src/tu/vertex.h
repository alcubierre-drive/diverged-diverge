/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_vertex_t tu_vertex_t;
typedef struct tu_prop_t tu_prop_t;
typedef struct tu_projection_t tu_projection_t;
typedef struct diverge_model_t diverge_model_t;

tu_vertex_t* tu_vertex_init( diverge_model_t* model, tu_projection_t* proj, tu_prop_t* loop,
        bool P/*=true*/, bool C/*=true*/, bool D/*=true*/, bool self/*=false*/ );
void tu_vertex_free( tu_vertex_t* V );
tu_vertex_t* tu_vertex_copy( const tu_vertex_t* O );

// these functions are renamed such that we can export them
void diverge_tu_vertex_reco( const tu_vertex_t* V, char chan, complex128_t* buf, const complex128_t* inp/*=NULL*/ );
void diverge_tu_vertex_reco_ft( const tu_vertex_t* V, char chan, complex128_t* buf,
        const complex128_t* existing_r_exp, index_t nr, complex128_t* bufMPI/*=NULL*/ );
// two-step reference implementation of the above (reconstruct, then FT)
void diverge_tu_vertex_reco_ft_ref( const tu_vertex_t* V, char chan, complex128_t* buf,
        const complex128_t* existing_r_exp, index_t nr, complex128_t* bufMPI/*=NULL*/ );

void tu_vertex_add(tu_vertex_t* result, const tu_vertex_t* inp, const double factor);
void tu_vertex_set(tu_vertex_t* f, const tu_vertex_t* initial,const tu_vertex_t* inp, const double factor);
void tu_vertex_multiply(tu_vertex_t* f,const tu_vertex_t* inp, const double factor);
double tu_vertex_infinity_norm(const tu_vertex_t* V);
void tu_vertex_max(const tu_vertex_t* V, double out[6]);

// result = B × C × B * prefac
void tu_vertex_product( complex128_t* result, const complex128_t* B,
        const complex128_t* C, complex128_t* helper, index_t productsize,
        index_t nbatch, double prefac, bool buffered, char* GPUbuffer[32][3] );

typedef struct tu_data_phonons_t tu_data_phonons_t;
void tu_vertex_init_phonons( tu_vertex_t* V, tu_data_phonons_t* p );

#ifdef __cplusplus
}
#endif
