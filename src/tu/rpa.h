/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_flow_step_t diverge_flow_step_t;
typedef struct diverge_model_t diverge_model_t;

void diverge_flow_step_sc_vertex( diverge_flow_step_t* s, char chan, const index_t* kidx, index_t n_kidx,
        complex128_t* out/*(n_kidx,n_spin,n_orb,n_spin,n_orb, n_kidx,n_spin,n_orb,n_spin,n_orb)*/ );
void diverge_flow_step_sc_greens( diverge_flow_step_t* s, const index_t* kidx, index_t n_kidx, double Lambda,
        double dLambda, complex128_t* out/*(n_kidx, n_spin,n_orb,n_spin,n_orb, n_spin,n_orb,n_spin,n_orb)*/ );

// perform the vertex × greens·greens product in-place (V_pp)
void diverge_flow_step_sc_greens_apply( diverge_flow_step_t* s,
        const index_t n_kidx, const complex128_t* Chi_pp, complex128_t* V_pp );

/*

diverge_model_t* m = ...;
m->nkf[0] = m->nkf[1] = m->nkf[2] = 1; // fine mesh disabled!

diverge_model_internals_tu(m, 0.01); // we perform IOBI/RPA
diverge_flow_step_t* s = diverge_flow_step_init_any(m, "PCD");

// fill these with meaningful content (indices that refer to the coarse mesh)
index_t n_kidx = ...;
index_t* kidx = ...;

// and allocate buffer for the pp bubble
complex128_t* chi_pp = calloc( n_kidx*POW4(m->n_orb*m->n_spin), sizeof*chi_pp );

// integrate flow equations
diverge_euler_t eu = diverge_euler_defaults;
double vmax = 0.0;
do {
    diverge_flow_step_euler(s, eu.Lambda, eu.dLambda);
    diverge_flow_step_vertmax(s, &vmax);
    // we integrate the k-space chi_pp at all those scales the flow passes
    diverge_flow_step_sc_greens(s, kidx, n_kidx, eu.Lambda, eu.dLambda, chi_pp);
} while(diverge_euler_next(&eu, vmax));

// and then get the superconducting vertex
complex128_t* V_sc = calloc( POW2(n_kidx*POW2(m->n_orb*m->n_spin)), sizeof*V_sc );
diverge_flow_step_sc_vertex( s, 'c', kidx, n_kidx, V_sc );

// multiply V with chi
diverge_flow_step_sc_greens_apply( s, n_kidx, chi_pp, V_sc );

// and do something with it (e.g. iterative eigensolve)
int nev = 10, ncv = 100, maxit = 100;
double tol = 1.e-5;
transpose_inplace( V_sc, n_kidx*POW2(m->n_orb*m->n_spin) );
complex128_t* V = calloc( nev*n_kidx*POW2(m->n_orb*m->n_spin), sizeof *V );
complex128_t* E = calloc( nev, sizeof *E );
arnoldi_eigen_dense("SR", n_kidx*POW2(m->n_orb*m->n_spin), nev, ncv, V_sc, E,
                    V, tol, maxit);
free( chi_pp );
free( V_sc );

// now we can save V, E; and inspect!

*/

#ifdef __cplusplus
}
#endif
