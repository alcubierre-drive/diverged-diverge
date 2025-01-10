/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

// this is a temporary interface for the C-style selfenergy
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Vertex Vertex;
typedef struct tu_prop_t tu_prop_t;
typedef struct projection_t projection_t;
typedef struct diverge_model_t diverge_model_t;

/*
diverge_model_t* Vertex_get_model( const Vertex* V );
const tu_data_t* Vertex_get_tu_data( const Vertex* V );
index_t Vertex_get_n_orb( const Vertex* V );
index_t Vertex_get_n_spin( const Vertex* V );
index_t Vertex_get_n_orbff( const Vertex* V );
index_t Vertex_get_nk( const Vertex* V );
index_t Vertex_get_nkf( const Vertex* V );
index_t Vertex_get_my_nk( const Vertex* V );
index_t Vertex_get_my_nk_off( const Vertex* V );
const symmstruct_t* Vertex_get_symm( const Vertex* V );
bool Vertex_get_P_flow( const Vertex* V );
bool Vertex_get_C_flow( const Vertex* V );
bool Vertex_get_D_flow( const Vertex* V );
complex128_t* Vertex_get_product_helper( const Vertex* V );
complex128_t* Vertex_get_bubble_helper( const Vertex* V );
index_t Vertex_get_full_vert_size( const Vertex* V );

int Vertex_get_X_flow( const Vertex* V, char X ) {
    if (X == 'P') return V->V->P_flow;
    if (X == 'C') return V->V->C_flow;
    if (X == 'D') return V->V->D_flow;
    return 0;
}

int Vertex_get_X_to_Y_proj( const Vertex* V, char X, char Y ) {
    char XY[3] = {0}; XY[0] = X; XY[1] = Y;
    if (!strcmp(XY,"PC")) return V->P_to_C;
    if (!strcmp(XY,"PD")) return V->P_to_D;
    if (!strcmp(XY,"CP")) return V->C_to_P;
    if (!strcmp(XY,"CD")) return V->C_to_D;
    if (!strcmp(XY,"DP")) return V->D_to_P;
    if (!strcmp(XY,"DC")) return V->D_to_C;
    return 0;
}

complex128_t* Vertex_get_chan( const Vertex* V, char X ) {
    if (X == 'P') return V->V->Pch;
    if (X == 'C') return V->V->Cch;
    if (X == 'D') return V->V->Dch;
    return NULL;
}
*/

Vertex* Vertex_init( diverge_model_t* model, projection_t* proj, tu_prop_t* loop,
        bool P/*=true*/, bool C/*=true*/, bool D/*=true*/, bool self/*=false*/ );
void Vertex_free( Vertex* V );
Vertex* Vertex_copy( const Vertex* O );

void Vertex_reco( const Vertex* V, char chan, complex128_t* buf, const complex128_t* inp/*=NULL*/ );
void Vertex_reco_ft( const Vertex* V, char chan, complex128_t* buf,
        const complex128_t* existing_r_exp, index_t nr, complex128_t* bufMPI/*=NULL*/ );

void add(Vertex* result, const Vertex* inp, const double factor);
void set(Vertex* f, const Vertex* initial,const Vertex* inp, const double factor);
void multiply(Vertex* f,const Vertex* inp, const double factor);
double infinity_norm(const Vertex* V);
void max(const Vertex* V, double out[6]);

#ifdef __cplusplus
}
#endif
