#pragma once

#include <math.h>

#ifndef LINALG_NO_LAPACKE
#include <lapacke.h>
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
#define LINALG_LAPACKE_SVD_i( ... ) { (void)superb; (void)H; (void)pU; (void)pVT; }

#define LINALG_VEC_IMPL( dim, type ) \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_map( LINALG_DATA_TYPE_##type* p ) { \
    cla##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = p[i]; \
    return result; \
} \
LINALG_EXPORT void cla##V##dim##type##_rmap( cla##V##dim##type##_t v, LINALG_DATA_TYPE_##type * p ) { \
    for (int i=0; i<dim; ++i) p[i] = v.v[i]; \
} \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_scale( cla##V##dim##type##_t v, LINALG_DATA_TYPE_##type s ) { \
    cla##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = v.v[i] * s; \
    return result; \
} \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_add( cla##V##dim##type##_t a, cla##V##dim##type##_t b ) { \
    cla##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = a.v[i] + b.v[i]; \
    return result; \
} \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_sub( cla##V##dim##type##_t a, cla##V##dim##type##_t b ) { \
    return cla##V##dim##type##_add( a, cla##V##dim##type##_scale(b, -1.0) ); \
} \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_conj( cla##V##dim##type##_t a ) { \
    cla##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = conj(a.v[i]); \
    return result; \
} \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_odot( cla##V##dim##type##_t a, cla##V##dim##type##_t b ) { \
    cla##V##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
        result.v[i] = a.v[i] * b.v[i]; \
    return result; \
} \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla##V##dim##type##_dot( cla##V##dim##type##_t a, cla##V##dim##type##_t b ) { \
    cla##V##dim##type##_t t = cla##V##dim##type##_odot( cla##V##dim##type##_conj(a), b ); \
    LINALG_DATA_TYPE_##type r = 0.0; \
    for (int i=0; i<dim; ++i) \
        r += t.v[i]; \
    return r; \
} \
LINALG_EXPORT double cla##V##dim##type##_norm2( cla##V##dim##type##_t a ) { \
    return cla##V##dim##type##_dot( a, a ); \
} \
LINALG_EXPORT double cla##V##dim##type##_norm( cla##V##dim##type##_t a ) { \
    return sqrt( cla##V##dim##type##_norm2( a ) ); \
} \
 \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_map( LINALG_DATA_TYPE_##type* p ) { \
    cla##M##dim##type##_t result; \
    LINALG_DATA_TYPE_##type* ptr = result.m[0]; \
    for (int i=0; i<dim*dim; ++i) \
        ptr[i] = p[i]; \
    return result; \
} \
LINALG_EXPORT void cla##M##dim##type##_rmap( cla##M##dim##type##_t m, LINALG_DATA_TYPE_##type * p ) { \
    LINALG_DATA_TYPE_##type* ptr = m.m[0]; \
    for (int i=0; i<dim*dim; ++i) p[i] = ptr[i]; \
} \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_id( void ) { \
    cla##M##dim##type##_t result = {0}; \
    for (int i=0; i<dim; ++i) \
        result.m[i][i] = 1.0; \
    return result; \
} \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_scale( cla##M##dim##type##_t v, LINALG_DATA_TYPE_##type s ) { \
    cla##M##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = v.m[i][j] * s; \
    return result; \
} \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_add( cla##M##dim##type##_t a, cla##M##dim##type##_t b ) { \
    cla##M##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = a.m[i][j] + b.m[i][j]; \
    return result; \
} \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_sub( cla##M##dim##type##_t a, cla##M##dim##type##_t b ) { \
    return cla##M##dim##type##_add( a, cla##M##dim##type##_scale(b, -1.0) ); \
} \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_conj( cla##M##dim##type##_t a ) { \
    cla##M##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = conj(a.m[i][j]); \
    return result; \
} \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_odot( cla##M##dim##type##_t a, cla##M##dim##type##_t b ) { \
    cla##M##dim##type##_t result; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = a.m[i][j] * b.m[i][j]; \
    return result; \
} \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla##M##dim##type##_dot( cla##M##dim##type##_t a, cla##M##dim##type##_t b ) { \
    cla##M##dim##type##_t t = cla##M##dim##type##_odot( cla##M##dim##type##_conj(a), b ); \
    LINALG_DATA_TYPE_##type r = 0.0; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        r += t.m[i][j]; \
    return r; \
} \
LINALG_EXPORT double cla##M##dim##type##_norm2( cla##M##dim##type##_t a ) { \
    return cla##M##dim##type##_dot( a, a ); \
} \
LINALG_EXPORT double cla##M##dim##type##_norm( cla##M##dim##type##_t a ) { \
    return sqrt( cla##M##dim##type##_norm2( a ) ); \
} \
 \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_matmul( cla##M##dim##type##_t a, cla##M##dim##type##_t b ) { \
    cla##M##dim##type##_t result = {0}; \
    for (int i=0; i<dim; ++i) \
    for (int k=0; k<dim; ++k) \
    for (int j=0; j<dim; ++j) \
        result.m[i][k] += a.m[i][j] * b.m[j][k]; \
    return result; \
} \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla##M##dim##type##_trace( cla##M##dim##type##_t a ) { \
    LINALG_DATA_TYPE_##type result = 0.0; \
    for (int i=0; i<dim; ++i) \
        result += a.m[i][i]; \
    return result; \
} \
LINALG_EXPORT cla##V##dim##type##_t cla##M##dim##type##_matvec( cla##M##dim##type##_t a, cla##V##dim##type##_t v ) { \
    cla##V##dim##type##_t result = {0}; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.v[i] += a.m[i][j] * v.v[j]; \
    return result; \
} \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_transpose( cla##M##dim##type##_t a ) { \
    cla##M##dim##type##_t result = {0}; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        result.m[i][j] = a.m[j][i]; \
    return result; \
} \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_adjoint( cla##M##dim##type##_t a ) { \
    return cla##M##dim##type##_transpose( cla##M##dim##type##_conj( a ) ); \
} \
LINALG_EXPORT cla##V##dim##type##_t cla##M##dim##type##_vecmat( cla##V##dim##type##_t v, cla##M##dim##type##_t a ) { \
    return cla##M##dim##type##_matvec( cla##M##dim##type##_transpose(a), v ); \
} \
LINALG_EXPORT cla##V##dim##d_t cla##M##dim##type##_eigvalsh( cla##M##dim##type##_t H ) { \
    cla##V##dim##d_t val = {0}; \
    cla##M##dim##type##_t* pU = &H; \
    LINALG_LAPACKE_EIGH_##type ( LAPACK_ROW_MAJOR, 'N', 'U', dim, pU->m[0], dim, val.v ); \
    return val; \
} \
LINALG_EXPORT cla##V##dim##d_t cla##M##dim##type##_eigh( cla##M##dim##type##_t H, cla##M##dim##type##_t* pU ) { \
    cla##V##dim##d_t val = {0}; \
    *pU = H; \
    LINALG_LAPACKE_EIGH_##type ( LAPACK_ROW_MAJOR, 'V', 'U', dim, pU->m[0], dim, val.v ); \
    return val; \
} \
LINALG_EXPORT cla##V##dim##d_t cla##M##dim##type##_svd( cla##M##dim##type##_t H, cla##M##dim##type##_t* pU, cla##M##dim##type##_t* pVT ) { \
    cla##V##dim##d_t val = {0}; \
    double superb[dim] = {0}; \
    LINALG_LAPACKE_SVD_##type ( LAPACK_ROW_MAJOR, 'A', 'A', dim, dim, H.m[0], dim, val.v, pU->m[0], dim, pVT->m[0], dim, superb ); \
    return val; \
}
