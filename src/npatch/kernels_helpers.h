#pragma once

#include "../diverge_common.h"

#define UNRAVEL_V( idx, x, y, z, a, b, c, d ) \
    index_t x, y, z, a, b, c, d; \
    c = idx; \
    d = np * np * nb * nb * nb * nb; \
    x = c / d; \
    c %= d; d /= np; \
    y = c / d; \
    c %= d; d /= np; \
    z = c / d; \
    c %= d; d /= nb; \
    a = c / d; \
    c %= d; d /= nb; \
    b = c / d; \
    c %= d; d /= nb; \
    c = c / d; \
    d = idx % d;

#define UNRAVEL_K3( idx, k1, k2, k3 ) \
    index_t k1 = idx / (np*np), \
            k2 = (idx % (np*np)) / np, \
            k3 = idx % np;

// k1*nk*nk*nb*nb*nb*nb+ k2*nk*nb*nb*nb*nb+ k3*nb*nb*nb*nb+ o1*nb*nb*nb+ o2*nb*nb+ o3*nb+ o4;
#define RAVEL_V( k1, k2, k3, o1, o2, o3, o4 ) ((((((k1*np+k2)*np+k3)*nb+o1)*nb+o2)*nb+o3)*nb+o4)

// k1*nk*nb*nb*nb*nb+ k2*nb*nb*nb*nb+ o1*nb*nb*nb+ o2*nb*nb+ o3+nb+ o4;
#define RAVEL_L( k1, k2, o1, o2, o3, o4 ) (((((k1*np+k2)*nb+o1)*nb+o2)*nb+o3)*nb+o4)

#define RAVEL_L3( Pi, P1, P2, o1, o2, o3, o4 ) ((((((Pi*np+P1)*np+P2)*nb+o1)*nb+o2)*nb+o3)*nb+o4)
#define RAVEL_O4( o1, o2, o3, o4 ) ((o1*nb+o2)*nb+o3)*nb+o4

static inline void complex128_t_atomic_add( complex128_t* addr, const complex128_t* val ) {
    double* addr_double = (double*)addr;
    const double* val_double = (const double*)val;
    #pragma omp atomic
    addr_double[0] += val_double[0];
    #pragma omp atomic
    addr_double[1] += val_double[1];
}

static inline void complex128_t_atomic_add_byval( complex128_t* addr, complex128_t val ) {
    double* addr_double = (double*)addr;
    const double* val_double = (const double*)&val;
    #pragma omp atomic
    addr_double[0] += val_double[0];
    #pragma omp atomic
    addr_double[1] += val_double[1];
}
