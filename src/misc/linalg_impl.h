/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include <math.h>

#ifndef LINALG_NO_LAPACKE
#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#define LINALG_LAPACKE_EIGH_d LAPACKE_dsyevd
#define LINALG_LAPACKE_EIGH_cd LAPACKE_zheevd
#define LINALG_LAPACKE_SVD_d LAPACKE_dgesvd
#define LINALG_LAPACKE_SVD_cd LAPACKE_zgesvd
#else // LINALG_NO_LAPACKE
#define LINALG_LAPACKE_EIGH_d( ... ) { (void)H; (void)pU; }
#define LINALG_LAPACKE_EIGH_cd( ... ) { (void)H; (void)pU; }
#define LINALG_LAPACKE_SVD_d( ... ) { (void)superb; (void)H; (void)pU; (void)pVT; }
#define LINALG_LAPACKE_SVD_cd( ... ) { (void)superb; (void)H; (void)pU; (void)pVT;  }
#endif // LINALG_NO_LAPACKE

#define LINALG_LAPACKE_EIGH_i( ... ) { (void)H; (void)pU; }
#define LINALG_LAPACKE_EIGH_ll( ... ) { (void)H; (void)pU; }
#define LINALG_LAPACKE_SVD_i( ... ) { (void)superb; (void)H; (void)pU; (void)pVT; }
#define LINALG_LAPACKE_SVD_ll( ... ) { (void)superb; (void)H; (void)pU; (void)pVT; }

#define LINALG_VEC_IMPL( cla_prefix, dim, type ) \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_zero( void ) { \
    cla_prefix##V##dim##type##_t result = {0}; \
    return result; \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_const( LINALG_DATA_TYPE_##type val ) { \
    cla_prefix##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = val; \
    return result; \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_map( LINALG_DATA_TYPE_##type* p ) { \
    cla_prefix##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = p[i]; \
    return result; \
} \
LINALG_EXPORT void cla_prefix##V##dim##type##_rmap( cla_prefix##V##dim##type##_t v, LINALG_DATA_TYPE_##type * p ) { \
    for (int i=0; i<dim; ++i) p[i] = v.v[i]; \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_scale( cla_prefix##V##dim##type##_t v, LINALG_DATA_TYPE_##type s ) { \
    cla_prefix##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = v.v[i] * s; \
    return result; \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_add( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ) { \
    cla_prefix##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = a.v[i] + b.v[i]; \
    return result; \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_sub( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ) { \
    return cla_prefix##V##dim##type##_add( a, cla_prefix##V##dim##type##_scale(b, -1.0) ); \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_conj( cla_prefix##V##dim##type##_t a ) { \
    cla_prefix##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = conj(a.v[i]); \
    return result; \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_odot( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ) { \
    cla_prefix##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = a.v[i] * b.v[i]; \
    return result; \
} \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##V##dim##type##_dot( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ) { \
    cla_prefix##V##dim##type##_t t = cla_prefix##V##dim##type##_odot( cla_prefix##V##dim##type##_conj(a), b ); \
    LINALG_DATA_TYPE_##type r = 0.0; \
    for (int i=0; i<dim; ++i) \
        r += t.v[i]; \
    return r; \
} \
LINALG_EXPORT double cla_prefix##V##dim##type##_norm2( cla_prefix##V##dim##type##_t a ) { \
    return cla_prefix##V##dim##type##_dot( a, a ); \
} \
LINALG_EXPORT double cla_prefix##V##dim##type##_norm( cla_prefix##V##dim##type##_t a ) { \
    return sqrt( cla_prefix##V##dim##type##_norm2( a ) ); \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_cross( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ) { \
    cla_prefix##V##dim##type##_t c; \
    for (int i=0; i<dim; ++i) { \
        c.v[i] = a.v[(i+1)%dim]*b.v[(i+2)%dim] - a.v[(i+2)%dim]*b.v[(i+1)%dim]; \
    } \
    return c; \
} \
 \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_map( LINALG_DATA_TYPE_##type* p ) { \
    cla_prefix##M##dim##type##_t result; \
    LINALG_DATA_TYPE_##type* ptr = result.m[0]; \
    for (int i=0; i<dim*dim; ++i) \
        ptr[i] = p[i]; \
    return result; \
} \
LINALG_EXPORT void cla_prefix##M##dim##type##_rmap( cla_prefix##M##dim##type##_t m, LINALG_DATA_TYPE_##type * p ) { \
    LINALG_DATA_TYPE_##type* ptr = m.m[0]; \
    for (int i=0; i<dim*dim; ++i) p[i] = ptr[i]; \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_zero( void ) { \
    cla_prefix##M##dim##type##_t result = {0}; \
    return result; \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_id( void ) { \
    cla_prefix##M##dim##type##_t result = {0}; \
    for (int i=0; i<dim; ++i) \
        result.m[i][i] = 1.0; \
    return result; \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_scale( cla_prefix##M##dim##type##_t v, LINALG_DATA_TYPE_##type s ) { \
    cla_prefix##M##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = v.m[i][j] * s; \
    return result; \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_add( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ) { \
    cla_prefix##M##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = a.m[i][j] + b.m[i][j]; \
    return result; \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_sub( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ) { \
    return cla_prefix##M##dim##type##_add( a, cla_prefix##M##dim##type##_scale(b, -1.0) ); \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_conj( cla_prefix##M##dim##type##_t a ) { \
    cla_prefix##M##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = conj(a.m[i][j]); \
    return result; \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_odot( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ) { \
    cla_prefix##M##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = a.m[i][j] * b.m[i][j]; \
    return result; \
} \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##M##dim##type##_dot( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ) { \
    cla_prefix##M##dim##type##_t t = cla_prefix##M##dim##type##_odot( cla_prefix##M##dim##type##_conj(a), b ); \
    LINALG_DATA_TYPE_##type r = 0.0; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        r += t.m[i][j]; \
    return r; \
} \
LINALG_EXPORT double cla_prefix##M##dim##type##_norm2( cla_prefix##M##dim##type##_t a ) { \
    return cla_prefix##M##dim##type##_dot( a, a ); \
} \
LINALG_EXPORT double cla_prefix##M##dim##type##_norm( cla_prefix##M##dim##type##_t a ) { \
    return sqrt( cla_prefix##M##dim##type##_norm2( a ) ); \
} \
 \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_matmul( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ) { \
    cla_prefix##M##dim##type##_t result = {0}; \
    for (int i=0; i<dim; ++i) \
    for (int k=0; k<dim; ++k) \
    for (int j=0; j<dim; ++j) \
        result.m[i][k] += a.m[i][j] * b.m[j][k]; \
    return result; \
} \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##M##dim##type##_trace( cla_prefix##M##dim##type##_t a ) { \
    LINALG_DATA_TYPE_##type result = 0.0; \
    for (int i=0; i<dim; ++i) \
        result += a.m[i][i]; \
    return result; \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##M##dim##type##_matvec( cla_prefix##M##dim##type##_t a, cla_prefix##V##dim##type##_t v ) { \
    cla_prefix##V##dim##type##_t result = {0}; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.v[i] += a.m[i][j] * v.v[j]; \
    return result; \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_transpose( cla_prefix##M##dim##type##_t a ) { \
    cla_prefix##M##dim##type##_t result = {0}; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = a.m[j][i]; \
    return result; \
} \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_adjoint( cla_prefix##M##dim##type##_t a ) { \
    return cla_prefix##M##dim##type##_transpose( cla_prefix##M##dim##type##_conj( a ) ); \
} \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##M##dim##type##_vecmat( cla_prefix##V##dim##type##_t v, cla_prefix##M##dim##type##_t a ) { \
    return cla_prefix##M##dim##type##_matvec( cla_prefix##M##dim##type##_transpose(a), v ); \
} \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_eigvalsh( cla_prefix##M##dim##type##_t H ) { \
    cla_prefix##V##dim##d_t val = {0}; \
    cla_prefix##M##dim##type##_t* pU = &H; \
    LINALG_LAPACKE_EIGH_##type ( LAPACK_ROW_MAJOR, 'N', 'U', dim, pU->m[0], dim, val.v ); \
    return val; \
} \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_eigh( cla_prefix##M##dim##type##_t H, cla_prefix##M##dim##type##_t* pU ) { \
    cla_prefix##V##dim##d_t val = {0}; \
    *pU = H; \
    LINALG_LAPACKE_EIGH_##type ( LAPACK_ROW_MAJOR, 'V', 'U', dim, pU->m[0], dim, val.v ); \
    return val; \
} \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_svd( cla_prefix##M##dim##type##_t H, cla_prefix##M##dim##type##_t* pU, cla_prefix##M##dim##type##_t* pVT ) { \
    cla_prefix##V##dim##d_t val = {0}; \
    double superb[dim] = {0}; \
    LINALG_LAPACKE_SVD_##type ( LAPACK_ROW_MAJOR, 'A', 'A', dim, dim, H.m[0], dim, val.v, pU->m[0], dim, pVT->m[0], dim, superb ); \
    return val; \
}
