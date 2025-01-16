/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef index_t (*index_gymnastics_t)(index_t, index_t, index_t, const index_t*, const index_t*);

static inline index_t index_gymnastics_V_to_P( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_V_to_C( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_V_to_D( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_P_to_V( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_C_to_V( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_D_to_V( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_D_to_P( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_P_to_D( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_D_to_C( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_C_to_D( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_D_swap_3_and_4( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_D_swap_1_and_2( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_V_swap_3_and_4( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );
static inline index_t index_gymnastics_V_swap_1_and_2( index_t idx, index_t nk, index_t nb, const index_t* k1m2, const index_t* k1p2 );

// implementation
static inline void unravel_I( index_t idx, index_t nk, index_t nb, index_t* out );
static inline void unravel( index_t idx, const index_t* dim, index_t ndim, index_t* out );
static inline void unravel_V( index_t idx, index_t nk, index_t nb, index_t* out );

static inline index_t index_gymnastics_V_to_P( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    (void)k1m2; (void)k1p2;
    index_t IDX[7]; unravel_V( idx, nk, nb, IDX );
    const index_t k1 = IDX[0], k2 = IDX[1], k3 = IDX[2],
                  b1 = IDX[3], b2 = IDX[4], b3 = IDX[5], b4 = IDX[6];
    return IDX7( k1p2[nk*k2+k1], k1, b1, b2, k3, b3, b4, nk,nb,nb, nk,nb,nb );
}

static inline index_t index_gymnastics_V_to_C( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    (void)k1m2; (void)k1p2;
    index_t IDX[7]; unravel_V( idx, nk, nb, IDX );
    const index_t k1 = IDX[0], k2 = IDX[1], k3 = IDX[2],
                  b1 = IDX[3], b2 = IDX[4], b3 = IDX[5], b4 = IDX[6];
    return IDX7( k1m2[nk*k3+k2], k1, b4, b1, k3, b2, b3, nk,nb,nb, nk,nb,nb );
}

static inline index_t index_gymnastics_V_to_D( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    (void)k1m2; (void)k1p2;
    index_t IDX[7]; unravel_V( idx, nk, nb, IDX );
    const index_t k1 = IDX[0], k2 = IDX[1], k3 = IDX[2],
                  b1 = IDX[3], b2 = IDX[4], b3 = IDX[5], b4 = IDX[6];
    return IDX7( k1m2[nk*k1+k3], k1, b3, b1, k1p2[nk*k1m2[nk*k2+k3]+k1], b2, b4, nk,nb,nb, nk,nb,nb );
}

static inline index_t index_gymnastics_P_to_V( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    (void)k1m2; (void)k1p2;
    index_t IDX[7]; unravel_I( idx, nk, nb, IDX );
    const index_t q = IDX[0], k  = IDX[1], b1 = IDX[2], b2 = IDX[3],
                              kp = IDX[4], b3 = IDX[5], b4 = IDX[6];
    return IDX7( k, k1m2[nk*q+k], kp, b1, b2, b3, b4, nk,nk, nb,nb,nb,nb );
}

static inline index_t index_gymnastics_C_to_V( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    (void)k1m2; (void)k1p2;
    index_t IDX[7]; unravel_I( idx, nk, nb, IDX );
    const index_t q = IDX[0], k  = IDX[1], b1 = IDX[2], b2 = IDX[3],
                              kp = IDX[4], b3 = IDX[5], b4 = IDX[6];
    return IDX7( k, k1m2[nk*kp+q], kp, b2, b3, b4, b1, nk,nk, nb,nb,nb,nb );
}

static inline index_t index_gymnastics_D_to_V( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    (void)k1m2; (void)k1p2;
    index_t IDX[7]; unravel_I( idx, nk, nb, IDX );
    const index_t q = IDX[0], k  = IDX[1], b1 = IDX[2], b2 = IDX[3],
                              kp = IDX[4], b3 = IDX[5], b4 = IDX[6];
    return IDX7( k, k1m2[nk*kp+q], k1m2[nk*k+q], b2, b3, b1, b4, nk,nk, nb,nb,nb,nb );
}

static inline index_t index_gymnastics_D_to_P( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    return index_gymnastics_V_to_P( index_gymnastics_D_to_V( idx, nk, nb, k1m2, k1p2 ),
            nk, nb, k1m2, k1p2 );
}

static inline index_t index_gymnastics_P_to_D( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    return index_gymnastics_V_to_D( index_gymnastics_P_to_V( idx, nk, nb, k1m2, k1p2 ),
            nk, nb, k1m2, k1p2 );
}

static inline index_t index_gymnastics_D_to_C( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    return index_gymnastics_V_to_C( index_gymnastics_D_to_V( idx, nk, nb, k1m2, k1p2 ),
            nk, nb, k1m2, k1p2 );
}

static inline index_t index_gymnastics_C_to_D( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    return index_gymnastics_V_to_D( index_gymnastics_C_to_V( idx, nk, nb, k1m2, k1p2 ),
            nk, nb, k1m2, k1p2 );
}

static inline index_t index_gymnastics_D_swap_3_and_4( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    return index_gymnastics_V_to_D( index_gymnastics_V_swap_3_and_4(
                index_gymnastics_D_to_V( idx, nk, nb, k1m2, k1p2 ),
                nk, nb, k1m2, k1p2 ), nk, nb, k1m2, k1p2 );
}

static inline index_t index_gymnastics_D_swap_1_and_2( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    return index_gymnastics_V_to_D( index_gymnastics_V_swap_1_and_2(
                index_gymnastics_D_to_V( idx, nk, nb, k1m2, k1p2 ),
                nk, nb, k1m2, k1p2 ), nk, nb, k1m2, k1p2 );
}

static inline index_t index_gymnastics_V_swap_3_and_4( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    index_t IDX[7]; unravel_V( idx, nk, nb, IDX );
    index_t k1 = IDX[0]; index_t k2 = IDX[1]; index_t k3 = IDX[2];
    index_t b1 = IDX[3]; index_t b2 = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    return IDX7( k1, k2, k1m2[nk*k1p2[nk*k1+k2]+k3], b1, b2, b4, b3, nk,nk, nb,nb,nb,nb );
}

static inline index_t index_gymnastics_V_swap_1_and_2( index_t idx, index_t nk, index_t nb,
        const index_t* k1m2, const index_t* k1p2 ) {
    (void)k1m2; (void)k1p2;
    index_t IDX[7]; unravel_V( idx, nk, nb, IDX );
    index_t k1 = IDX[0]; index_t k2 = IDX[1]; index_t k3 = IDX[2];
    index_t b1 = IDX[3]; index_t b2 = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    return IDX7( k2, k1, k3, b2, b1, b3, b4, nk,nk, nb,nb,nb,nb );
}

static inline void unravel( index_t idx, const index_t* dim, index_t ndim, index_t* out ) {
    for (index_t i=ndim-1; i>=0; i--) {
        out[i] = idx % dim[i];
        idx /= dim[i];
    }
}

static inline void unravel_V( index_t idx, index_t nk, index_t nb, index_t* out ) {
    const index_t dims[7] = {nk,nk,nk, nb,nb,nb,nb};
    unravel( idx, dims, 7, out );
}

static inline void unravel_I( index_t idx, index_t nk, index_t nb, index_t* out ) {
    const index_t dims[7] = {nk, nk,nb,nb, nk,nb,nb};
    unravel( idx, dims, 7, out );
}

#ifdef __cplusplus
}
#endif
