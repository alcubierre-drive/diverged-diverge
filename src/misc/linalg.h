#pragma once

#ifdef __cplusplus
#include <complex>
#include <cstdint>
typedef std::complex<double> complex128_t;
#else
#include <complex.h>
#include <stdint.h>
typedef _Complex double complex128_t;
#endif

#define LINALG_DATA_TYPE_d double
#define LINALG_DATA_TYPE_cd complex128_t
#define LINALG_DATA_TYPE_i int
#define LINALG_DATA_TYPE_ll int64_t

#ifndef LINALG_EXPORT
#define LINALG_EXPORT static inline
#endif

#ifndef LINALG_PREFIX
#define LINALG_PREFIX cla
#endif

#ifndef LINALG_IMPLEMENT
#define LINALG_IMPLEMENT \
    LINALG_IMPL( LINALG_PREFIX, 2 ) \
    LINALG_IMPL( LINALG_PREFIX, 3 ) \
    LINALG_IMPL( LINALG_PREFIX, 4 )
#endif

#define LINALG_TYPES( cla_prefix, dim ) \
    LINALG_VEC_TYPES( cla_prefix, dim, d ) \
    LINALG_VEC_TYPES( cla_prefix, dim, cd ) \
    LINALG_VEC_TYPES( cla_prefix, dim, i ) \
    LINALG_VEC_TYPES( cla_prefix, dim, ll )

#define LINALG_IMPL( cla_prefix, dim ) \
    LINALG_TYPES( cla_prefix, dim ) \
    LINALG_VEC_IMPL( cla_prefix, dim, d ) \
    LINALG_VEC_IMPL( cla_prefix, dim, cd ) \
    LINALG_VEC_IMPL( cla_prefix, dim, i ) \
    LINALG_VEC_IMPL( cla_prefix, dim, ll )

#define LINALG_VEC_TYPES( cla_prefix, dim, type ) \
typedef struct cla_prefix##V##dim##type##_t { \
    LINALG_DATA_TYPE_##type v[dim]; \
} cla_prefix##V##dim##type##_t; \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_zero( void ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_map( LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT void cla_prefix##V##dim##type##_rmap( cla_prefix##V##dim##type##_t v, LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_scale( cla_prefix##V##dim##type##_t v, LINALG_DATA_TYPE_##type s ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_add( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_sub( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_conj( cla_prefix##V##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_odot( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##V##dim##type##_dot( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT double cla_prefix##V##dim##type##_norm2( cla_prefix##V##dim##type##_t a ); \
LINALG_EXPORT double cla_prefix##V##dim##type##_norm( cla_prefix##V##dim##type##_t a ); \
\
typedef struct cla_prefix##M##dim##type##_t { \
    LINALG_DATA_TYPE_##type m [dim][dim]; \
} cla_prefix##M##dim##type##_t; \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_zero( void ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_id( void ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_map( LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT void cla_prefix##M##dim##type##_rmap( cla_prefix##M##dim##type##_t m, LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_scale( cla_prefix##M##dim##type##_t v, LINALG_DATA_TYPE_##type s ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_add( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_sub( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_conj( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_odot( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##M##dim##type##_dot( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT double cla_prefix##M##dim##type##_norm2( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT double cla_prefix##M##dim##type##_norm( cla_prefix##M##dim##type##_t a ); \
\
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_matmul( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##M##dim##type##_trace( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##M##dim##type##_matvec( cla_prefix##M##dim##type##_t a, cla_prefix##V##dim##type##_t v ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_transpose( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_adjoint( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##M##dim##type##_vecmat( cla_prefix##V##dim##type##_t v, cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_eigvalsh( cla_prefix##M##dim##type##_t H ); \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_eigh( cla_prefix##M##dim##type##_t H, cla_prefix##M##dim##type##_t* pU ); \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_svd( cla_prefix##M##dim##type##_t H, cla_prefix##M##dim##type##_t* pU, cla_prefix##M##dim##type##_t* pVT );

#ifndef __cplusplus
#include "linalg_impl.h"
#endif
