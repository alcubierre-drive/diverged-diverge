/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*internals_destructor_t)(void*);

struct internals_t {
    // common internals
    double* kmesh;
    double* kfmesh;
    complex128_t* ham;
    complex128_t* U;
    double* E;
    gf_complex_t* greens; // 2*nk*nkf*nb*nb

    int greens_shared;

    int periodic_direction[3];
    index_t dim;
    bool enforce_exchange;
    index_t has_common_internals;
    double mu;

    bool ignore_eigensolver_in_common;
    int use_batched_svd_algo_in_common;

    // for the flow step to automatically detect the method.
    // DEVELOPERS must set this in their diverge_model_internals_BACKEND to the
    // correct string
    char backend[1024];

    // DEVELOPERS
    //
    // this file is for interfacing with the source codes found in
    // subdirectories of src/. If you want to add a new backend, you can add
    // your own variables to the internals_t struct. You are yourself
    // responsible for initializing and deinitializing the data correctly.
    // Consider the structures without a subdirectory prefix (i.e. the pointers
    // above) to be filled at the time when your code runs. To see examples of
    // how the code specific internal structures are set, look at
    // src/diverge_model_internals_grid.cpp or
    // src/diverge_model_internals_patch.c

    // grid internals
    index_t grid_nk, grid_nkf, grid_nb, grid_kfzero;
    index_t *grid_kfmesh, // grid_nk * grid_nkf
            *grid_kfrev, // grid_nkf
            *grid_k1m2, *grid_k1p2; // grid_nk * grid_nk
    double* grid_weights; // grid_nkf
    complex128_t *grid_greens_p_buf, *grid_greens_m_buf; // grid_nk * grid_nkf * nb*nb

    void* grid_vertex;
    internals_destructor_t grid_vertex_destructor;

    // patch internals
    void* patch_vertex;
    internals_destructor_t patch_vertex_destructor;
    index_t* patch_k3map;
    complex128_t *patch_pmesh, *patch_kmesh;
    complex128_t patch_g1, patch_g2;
    index_t patch_nkx, patch_nky, patch_nk, patch_nb;

    double patch_gpu_workshare, patch_gpu_loop_workshare;
    int *patch_gpus, *patch_nblocks, patch_ngpus,
        *patch_gpu_weights, *patch_loop_gpu_weights;

    // tu internals
    void* tu_data;
    internals_destructor_t tu_data_destructor;
    int tu_symcheck_mpi;

    // symmetry internals for two point functions
    index_t* symm_map_mom_fine;
    index_t* symm_map_mom_crs;
    double* symm_beyond_UC;

    index_t* symm_map_orb;
    index_t* symm_orb_off;
    index_t* symm_orb_len;
    complex128_t* symm_pref;

    int symcheck_mpi;

};
typedef struct internals_t internals_t;

#ifndef GRID_FF_SHELL_DISTANCE
#define GRID_FF_SHELL_DISTANCE (3.01)
#endif

static inline index_t k1m2( index_t k1, index_t k2, const index_t nk[3] ) {
    index_t k1x = k1 / (nk[1]*nk[2]),
            k1y = (k1 % (nk[1]*nk[2])) / nk[2],
            k1z = k1 % nk[2];
    index_t k2x = k2 / (nk[1]*nk[2]),
            k2y = (k2 % (nk[1]*nk[2])) / nk[2],
            k2z = k2 % nk[2];
    index_t k1m2x = (k1x - k2x + nk[0]) % nk[0],
            k1m2y = (k1y - k2y + nk[1]) % nk[1],
            k1m2z = (k1z - k2z + nk[2]) % nk[2];
    return IDX3( k1m2x, k1m2y, k1m2z, nk[1], nk[2] );
}

static inline index_t k1p2( index_t k1, index_t k2, const index_t nk[3] ) {
    index_t k1x = k1 / (nk[1]*nk[2]),
            k1y = (k1 % (nk[1]*nk[2])) / nk[2],
            k1z = k1 % nk[2];
    index_t k2x = k2 / (nk[1]*nk[2]),
            k2y = (k2 % (nk[1]*nk[2])) / nk[2],
            k2z = k2 % nk[2];
    index_t k1p2x = (k1x + k2x) % nk[0],
            k1p2y = (k1y + k2y) % nk[1],
            k1p2z = (k1z + k2z) % nk[2];
    return IDX3( k1p2x, k1p2y, k1p2z, nk[1], nk[2] );
}

static inline index_t kidxc2f( index_t k, const index_t nk[3], const index_t nkf[3] ) {
    index_t kx = k / (nk[1]*nk[2]),
            ky = (k % (nk[1]*nk[2])) / nk[2],
            kz = k % nk[2];
    return IDX3( kx*nkf[0], ky*nkf[1], kz*nkf[2], nk[1]*nkf[1], nk[2]*nkf[2] );
}

static inline index_t kdim( const index_t nk[3] ) {
    return nk[0] * nk[1] * nk[2];
}

static inline index_t kdimtot( const index_t nk[3], const index_t nkf[3] ) {
    return kdim(nk) * kdim(nkf);
}

static inline void unravel( index_t idx, const index_t* dim, index_t ndim, index_t* out ) {
    for (index_t i=ndim-1; i>=0; i--) {
        out[i] = idx % dim[i];
        idx /= dim[i];
    }
}

static inline char* strdup_div( const char* str ) {
    char* r = (char*)malloc(strlen(str)+1);
    strcpy(r, str);
    return r;
}

static inline void strlwr_div( char* p ) {
    for ( ; *p; ++p) *p = tolower(*p);
}

static inline void strupr_div( char* p ) {
    for ( ; *p; ++p) *p = toupper(*p);
}

#ifdef __cplusplus
}
#endif
