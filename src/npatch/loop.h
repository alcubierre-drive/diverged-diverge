/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct npatch_loop_workshare_t npatch_loop_workshare_t;

// diverge typedefs
typedef struct diverge_model_t diverge_model_t;
typedef struct internals_t internals_t;
typedef struct mom_patching_t mom_patching_t;

typedef struct npatch_loop_t npatch_loop_t;
struct npatch_loop_t {
    gf_complex_t* Gm;
    gf_complex_t* Gp;
    index_t nk, nb, G_sz, L_sz;

    diverge_model_t* dmod;
    internals_t* dint;
    mom_patching_t* dpat;

    index_t np;
    index_t *p_idx, *p_cnt, *p_dis, *p_map;
    double *f_wgh, *wgh;
    index_t nk_flat[3];
    index_t *k3map;

    double* E_real;
    complex128_t* E;
    complex128_t* U;

    complex128_t* Lp;
    complex128_t* Lm;

    npatch_loop_workshare_t* workshare;
    double wtime_gpu, wtime_cpu, wtime_comm;
    double wtime_gpu_list[128];
    double wtime_gpu_gf_list[128];

    // recursion pattern when using a projected loop
    npatch_loop_t* projLoop;
    int is_proj_dispatcher;
    int is_proj;
};

npatch_loop_t* npatch_loop_init( diverge_model_t* dmodel );

void npatch_loop_gen( npatch_loop_t* loop, complex128_t Lambda );
void npatch_loop_free( npatch_loop_t* loop );

void npatch_loop_gpu_set_weights( int* weights );

#ifdef __cplusplus
}
#endif
