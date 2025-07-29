/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include <omp.h>

#ifdef __cplusplus
#include <cstddef>
#include <cstring>
#include <cstdlib>
#ifndef MERGESORT_RESTRICT
#define MERGESORT_RESTRICT 
#endif
extern "C" {
#else
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#define MERGESORT_RESTRICT restrict
#endif

#ifndef MERGESORT_DEFAULT_TASK_SIZE
#define MERGESORT_DEFAULT_TASK_SIZE 8192
#endif
#ifndef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ((x)-(y))
#endif
#ifndef MERGESORT_COMP_R
#define MERGESORT_COMP_R( x, y, ctx ) ((x)-(y))
#endif


#define MERGESORT_BUILD_NAME( X, Y ) X##Y

#define MERGESORT_IMPL(MERGESORT_NAME, MERGESORT_TYPE) \
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse ) ( \
        MERGESORT_TYPE* MERGESORT_RESTRICT X, size_t n, MERGESORT_TYPE* MERGESORT_RESTRICT tmp, size_t ts ); \
\
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, ) ( MERGESORT_TYPE* base, \
        size_t nel, int nthr, size_t tasksize ) { \
    if (tasksize <= 0) tasksize = MERGESORT_DEFAULT_TASK_SIZE; \
    if (nthr <= 0) nthr = 1; \
    MERGESORT_TYPE* tmp = (MERGESORT_TYPE*)calloc(nel, sizeof(MERGESORT_TYPE)); \
    _Pragma("omp parallel num_threads(nthr)") \
    { \
        _Pragma("omp single") \
        MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse ) (base, nel, tmp, tasksize); \
    } \
    free( tmp ); \
} \
 \
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_aux ) ( \
        MERGESORT_TYPE* MERGESORT_RESTRICT X, size_t n, MERGESORT_TYPE* MERGESORT_RESTRICT tmp ) { \
    size_t i = 0; \
    size_t j = n/2; \
    size_t ti = 0; \
 \
    while (i<n/2 && j<n) { \
        if (MERGESORT_COMP( X[i], X[j] ) <= 0) \
            tmp[ti++] = X[i++]; \
        else \
            tmp[ti++] = X[j++]; \
    } \
    while (i<n/2) tmp[ti++] = X[i++]; \
    while (j<n) tmp[ti++] = X[j++]; \
 \
    memcpy( X, tmp, n*sizeof(MERGESORT_TYPE) ); \
} \
 \
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse ) ( \
        MERGESORT_TYPE* MERGESORT_RESTRICT X, size_t n, MERGESORT_TYPE* MERGESORT_RESTRICT tmp, size_t tasksize ) { \
    if (n < 2) return; \
 \
   _Pragma("omp task shared(X) if (n > tasksize)") \
   MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse ) (X, n/2, tmp, tasksize); \
 \
   _Pragma("omp task shared(X) if (n > tasksize)") \
   MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse ) (X+(n/2), n-(n/2), tmp+(n/2), tasksize); \
 \
   _Pragma("omp taskwait") \
   MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_aux ) (X, n, tmp); \
} \


#define MERGESORT_IMPL_R(MERGESORT_NAME, MERGESORT_TYPE) \
 \
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_r ) ( \
        MERGESORT_TYPE* MERGESORT_RESTRICT X, size_t n, void* ctx, MERGESORT_TYPE* MERGESORT_RESTRICT tmp, size_t ts ); \
 \
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, _r ) ( MERGESORT_TYPE* MERGESORT_RESTRICT base, \
        size_t nel, void* MERGESORT_RESTRICT ctx, int nthr, size_t tasksize ) { \
    if (tasksize <= 0) tasksize = MERGESORT_DEFAULT_TASK_SIZE; \
    if (nthr <= 0) nthr = 1; \
    MERGESORT_TYPE* tmp = (MERGESORT_TYPE*)calloc(nel, sizeof(MERGESORT_TYPE)); \
    _Pragma("omp parallel num_threads(nthr)") \
    { \
        _Pragma("omp single") \
        MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_r ) (base, nel, ctx, tmp, tasksize); \
    } \
    free( tmp ); \
} \
 \
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_aux_r ) ( \
        MERGESORT_TYPE* MERGESORT_RESTRICT X, size_t n, void* MERGESORT_RESTRICT ctx, MERGESORT_TYPE* MERGESORT_RESTRICT tmp ) { \
    (void)ctx; \
    size_t i = 0; \
    size_t j = n/2; \
    size_t ti = 0; \
 \
    while (i<n/2 && j<n) { \
        if (MERGESORT_COMP_R( X[i], X[j], ctx ) <= 0) \
            tmp[ti++] = X[i++]; \
        else \
            tmp[ti++] = X[j++]; \
    } \
    while (i<n/2) tmp[ti++] = X[i++]; \
    while (j<n) tmp[ti++] = X[j++]; \
 \
    memcpy( X, tmp, n*sizeof(MERGESORT_TYPE) ); \
} \
 \
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_r ) ( \
        MERGESORT_TYPE* MERGESORT_RESTRICT X, size_t n, void* MERGESORT_RESTRICT ctx, MERGESORT_TYPE* MERGESORT_RESTRICT tmp, size_t tasksize ) { \
    if (n < 2) return; \
 \
   _Pragma("omp task shared(X) if (n > tasksize)") \
   MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_r ) (X, n/2, ctx, tmp, tasksize); \
 \
   _Pragma("omp task shared(X) if (n > tasksize)") \
   MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_r ) (X+(n/2), n-(n/2), ctx, tmp+(n/2), tasksize); \
 \
   _Pragma("omp taskwait") \
   MERGESORT_BUILD_NAME( MERGESORT_NAME, _recurse_aux_r ) (X, n, ctx, tmp); \
}

#ifndef SERISORT_COMP
#define SERISORT_COMP( x, y ) ((x)-(y))
#endif
#ifndef SERISORT_COMP_R
#define SERISORT_COMP_R( x, y, ctx ) ((x)-(y))
#endif

#define SERISORT_BUILD_NAME( X, Y ) X##Y

#define SERISORT_IMPL(SERISORT_NAME, SERISORT_TYPE) \
static inline void SERISORT_BUILD_NAME( SERISORT_NAME, _s ) ( \
        SERISORT_TYPE* MERGESORT_RESTRICT X, size_t n, SERISORT_TYPE* MERGESORT_RESTRICT tmp ); \
\
static inline void SERISORT_BUILD_NAME( SERISORT_NAME, ) ( SERISORT_TYPE* base, \
        size_t nel ) { \
    SERISORT_TYPE* tmp = (SERISORT_TYPE*)calloc(nel, sizeof(SERISORT_TYPE)); \
    SERISORT_BUILD_NAME( SERISORT_NAME, _s ) (base, nel, tmp); \
    free( tmp ); \
} \
 \
static inline void SERISORT_BUILD_NAME( SERISORT_NAME, _s_aux ) ( \
        SERISORT_TYPE* MERGESORT_RESTRICT X, size_t n, SERISORT_TYPE* MERGESORT_RESTRICT tmp ) { \
    size_t i = 0; \
    size_t j = n/2; \
    size_t ti = 0; \
 \
    while (i<n/2 && j<n) { \
        if (SERISORT_COMP( X[i], X[j] ) <= 0) \
            tmp[ti++] = X[i++]; \
        else \
            tmp[ti++] = X[j++]; \
    } \
    while (i<n/2) tmp[ti++] = X[i++]; \
    while (j<n) tmp[ti++] = X[j++]; \
 \
    memcpy( X, tmp, n*sizeof(SERISORT_TYPE) ); \
} \
 \
static inline void SERISORT_BUILD_NAME( SERISORT_NAME, _s ) ( \
        SERISORT_TYPE* MERGESORT_RESTRICT X, size_t n, SERISORT_TYPE* MERGESORT_RESTRICT tmp ) { \
    if (n < 2) return; \
    SERISORT_BUILD_NAME( SERISORT_NAME, _s ) (X, n/2, tmp); \
    SERISORT_BUILD_NAME( SERISORT_NAME, _s ) (X+(n/2), n-(n/2), tmp+(n/2)); \
    SERISORT_BUILD_NAME( SERISORT_NAME, _s_aux ) (X, n, tmp); \
} \


#define SERISORT_IMPL_R(SERISORT_NAME, SERISORT_TYPE) \
 \
static inline void SERISORT_BUILD_NAME( SERISORT_NAME, _sr ) ( \
        SERISORT_TYPE* MERGESORT_RESTRICT X, size_t n, void* MERGESORT_RESTRICT ctx, SERISORT_TYPE* MERGESORT_RESTRICT tmp ); \
 \
static inline void SERISORT_BUILD_NAME( SERISORT_NAME, _r ) ( SERISORT_TYPE* MERGESORT_RESTRICT base, \
        size_t nel, void* MERGESORT_RESTRICT ctx ) { \
    SERISORT_TYPE* tmp = (SERISORT_TYPE*)calloc(nel, sizeof(SERISORT_TYPE)); \
    SERISORT_BUILD_NAME( SERISORT_NAME, _sr ) (base, nel, ctx, tmp); \
    free( tmp ); \
} \
 \
static inline void SERISORT_BUILD_NAME( SERISORT_NAME, _sr_aux ) ( \
        SERISORT_TYPE* MERGESORT_RESTRICT X, size_t n, void* MERGESORT_RESTRICT ctx, SERISORT_TYPE* MERGESORT_RESTRICT tmp ) { \
    (void)ctx; \
    size_t i = 0; \
    size_t j = n/2; \
    size_t ti = 0; \
 \
    while (i<n/2 && j<n) { \
        if (SERISORT_COMP_R( X[i], X[j], ctx ) <= 0) \
            tmp[ti++] = X[i++]; \
        else \
            tmp[ti++] = X[j++]; \
    } \
    while (i<n/2) tmp[ti++] = X[i++]; \
    while (j<n) tmp[ti++] = X[j++]; \
 \
    memcpy( X, tmp, n*sizeof(SERISORT_TYPE) ); \
} \
 \
static inline void SERISORT_BUILD_NAME( SERISORT_NAME, _sr ) ( \
        SERISORT_TYPE* MERGESORT_RESTRICT X, size_t n, void* MERGESORT_RESTRICT ctx, SERISORT_TYPE* MERGESORT_RESTRICT tmp ) { \
    if (n < 2) return; \
    SERISORT_BUILD_NAME( SERISORT_NAME, _sr ) (X, n/2, ctx, tmp); \
    SERISORT_BUILD_NAME( SERISORT_NAME, _sr ) (X+(n/2), n-(n/2), ctx, tmp+(n/2)); \
    SERISORT_BUILD_NAME( SERISORT_NAME, _sr_aux ) (X, n, ctx, tmp); \
}

#ifdef __cplusplus
}
#endif
