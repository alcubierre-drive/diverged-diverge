/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_flow_step_t diverge_flow_step_t;
typedef struct diverge_model_t diverge_model_t;
typedef struct tu_rpa_t tu_rpa_t;

struct tu_rpa_t {
    diverge_flow_step_t* s;
    diverge_model_t* m;
    index_t* kidx;
    index_t n_kidx;
    complex128_t* chi_pp;
    complex128_t* V_pp;

    double* vals;
    int nev;
    complex128_t* vecs;
    int legdim;
};

tu_rpa_t* tu_rpa_init( diverge_flow_step_t* s, const index_t* kidx, index_t n_kidx );
void tu_rpa_step( tu_rpa_t* t, double Lambda, double dLambda );
void tu_rpa_post( tu_rpa_t* t, const char* chans, int nev, int ncv, int maxiter, double tol );
void tu_rpa_free( tu_rpa_t* t );

#ifdef __cplusplus
}
#endif
