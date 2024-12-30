/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include <omp.h>

#ifdef __cplusplus
#include <cstddef>
#include <cstring>
#include <cstdlib>
extern "C" {
#else
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#endif

#ifndef MERGESORT_DEFAULT_TASK_SIZE
#define MERGESORT_DEFAULT_TASK_SIZE 16384
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
        MERGESORT_TYPE* X, size_t n, MERGESORT_TYPE* tmp, size_t ts ); \
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
        MERGESORT_TYPE* X, size_t n, MERGESORT_TYPE* tmp ) { \
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
        MERGESORT_TYPE* X, size_t n, MERGESORT_TYPE* tmp, size_t tasksize ) { \
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
        MERGESORT_TYPE* X, size_t n, void* ctx, MERGESORT_TYPE* tmp, size_t ts ); \
 \
static inline void MERGESORT_BUILD_NAME( MERGESORT_NAME, _r ) ( MERGESORT_TYPE* base, \
        size_t nel, void* ctx, int nthr, size_t tasksize ) { \
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
        MERGESORT_TYPE* X, size_t n, void* ctx, MERGESORT_TYPE* tmp ) { \
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
        MERGESORT_TYPE* X, size_t n, void* ctx, MERGESORT_TYPE* tmp, size_t tasksize ) { \
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

#ifdef __cplusplus
}
#endif
