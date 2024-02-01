#pragma once

#include <complex.h>
typedef _Complex double complex128_t;

#define LINALG_DATA_TYPE_d double
#define LINALG_DATA_TYPE_cd complex128_t
#define LINALG_DATA_TYPE_i int

#ifndef LINALG_EXPORT
#define LINALG_EXPORT 
#endif

#ifndef LINALG_IMPLEMENT
#define LINALG_IMPLEMENT \
    LINALG_IMPL( 2 ) \
    LINALG_IMPL( 3 ) \
    LINALG_IMPL( 4 )
#endif

#define LINALG_TYPES( dim ) \
    LINALG_VEC_TYPES( dim, d ) \
    LINALG_VEC_TYPES( dim, cd ) \
    LINALG_VEC_TYPES( dim, i )

#define LINALG_IMPL( dim ) \
    LINALG_TYPES( dim ) \
    LINALG_VEC_IMPL( dim, d ) \
    LINALG_VEC_IMPL( dim, cd ) \
    LINALG_VEC_IMPL( dim, i )

#define LINALG_VEC_TYPES( dim, type ) \
typedef struct cla##V##dim##type##_t { \
    LINALG_DATA_TYPE_##type v[dim]; \
} cla##V##dim##type##_t; \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_map( LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT void cla##V##dim##type##_rmap( cla##V##dim##type##_t v, LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_scale( cla##V##dim##type##_t v, LINALG_DATA_TYPE_##type s ); \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_add( cla##V##dim##type##_t a, cla##V##dim##type##_t b ); \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_sub( cla##V##dim##type##_t a, cla##V##dim##type##_t b ); \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_conj( cla##V##dim##type##_t a ); \
LINALG_EXPORT cla##V##dim##type##_t cla##V##dim##type##_odot( cla##V##dim##type##_t a, cla##V##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla##V##dim##type##_dot( cla##V##dim##type##_t a, cla##V##dim##type##_t b ); \
LINALG_EXPORT double cla##V##dim##type##_norm2( cla##V##dim##type##_t a ); \
LINALG_EXPORT double cla##V##dim##type##_norm( cla##V##dim##type##_t a ); \
\
typedef struct cla##M##dim##type##_t { \
    LINALG_DATA_TYPE_##type m [dim][dim]; \
} cla##M##dim##type##_t; \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_id( void ); \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_map( LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT void cla##M##dim##type##_rmap( cla##M##dim##type##_t m, LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_scale( cla##M##dim##type##_t v, LINALG_DATA_TYPE_##type s ); \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_add( cla##M##dim##type##_t a, cla##M##dim##type##_t b ); \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_sub( cla##M##dim##type##_t a, cla##M##dim##type##_t b ); \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_conj( cla##M##dim##type##_t a ); \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_odot( cla##M##dim##type##_t a, cla##M##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla##M##dim##type##_dot( cla##M##dim##type##_t a, cla##M##dim##type##_t b ); \
LINALG_EXPORT double cla##M##dim##type##_norm2( cla##M##dim##type##_t a ); \
LINALG_EXPORT double cla##M##dim##type##_norm( cla##M##dim##type##_t a ); \
\
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_matmul( cla##M##dim##type##_t a, cla##M##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla##M##dim##type##_trace( cla##M##dim##type##_t a ); \
LINALG_EXPORT cla##V##dim##type##_t cla##M##dim##type##_matvec( cla##M##dim##type##_t a, cla##V##dim##type##_t v ); \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_transpose( cla##M##dim##type##_t a ); \
LINALG_EXPORT cla##M##dim##type##_t cla##M##dim##type##_adjoint( cla##M##dim##type##_t a ); \
LINALG_EXPORT cla##V##dim##type##_t cla##M##dim##type##_vecmat( cla##V##dim##type##_t v, cla##M##dim##type##_t a ); \
LINALG_EXPORT cla##V##dim##d_t cla##M##dim##type##_eigvalsh( cla##M##dim##type##_t H ); \
LINALG_EXPORT cla##V##dim##d_t cla##M##dim##type##_eigh( cla##M##dim##type##_t H, cla##M##dim##type##_t* pU ); \
LINALG_EXPORT cla##V##dim##d_t cla##M##dim##type##_svd( cla##M##dim##type##_t H, cla##M##dim##type##_t* pU, cla##M##dim##type##_t* pVT );

#include "linalg_impl.h"
