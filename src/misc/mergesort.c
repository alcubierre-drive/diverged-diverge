#include "mergesort.h"
#include <omp.h>
#include <string.h>
#include <stdlib.h>

static inline void mergesort_recurse( char* X, size_t n, size_t width,
        int (*cmp)(const void*, const void*), char *tmp, size_t tasksize );
static inline void mergesort_recurse_aux( char* X, size_t n, size_t width,
        int (*cmp)(const void*, const void*), char* tmp );
static inline void mergesort_recurse_r( char* X, size_t n, size_t width,
        int (*cmp)(const void*, const void*, void*), void* ctx, char *tmp, size_t tasksize );
static inline void mergesort_recurse_r_aux( char* X, size_t n, size_t width,
        int (*cmp)(const void*, const void*, void*), void* ctx, char* tmp );

#ifndef MERGESORT_DEFAULT_TASK_SIZE
#define MERGESORT_DEFAULT_TASK_SIZE 1024
#endif

void mergesort( void* base, size_t nel, size_t width,
        int (*cmp)(const void*, const void*), int nthr, size_t tasksize ) {
    if (tasksize <= 0) tasksize = MERGESORT_DEFAULT_TASK_SIZE;
    if (nthr <= 0) nthr = 1;
    void* tmp = calloc(nel, width);
    #pragma omp parallel num_threads(nthr)
    {
        #pragma omp single
        mergesort_recurse(base, nel, width, cmp, tmp, tasksize);
    }
    free( tmp );
}

void mergesort_r( void* base, size_t nel, size_t width,
        int (*cmp)(const void*, const void*, void* ctx), void* ctx, int nthr, size_t tasksize ) {
    if (tasksize <= 0) tasksize = MERGESORT_DEFAULT_TASK_SIZE;
    if (nthr <= 0) nthr = 1;
    void* tmp = calloc(nel, width);
    #pragma omp parallel num_threads(nthr)
    {
        #pragma omp single
        mergesort_recurse_r(base, nel, width, cmp, ctx, tmp, tasksize);
    }
    free( tmp );
}


static inline void mergesort_recurse_aux( char* X, size_t n, size_t width,
        int (*cmp)(const void*, const void*), char* tmp ) {
    size_t i = 0;
    size_t j = n/2;
    size_t ti = 0;

    while (i<n/2 && j<n) {
        if ((*cmp)( X+i*width, X+j*width ) <= 0) {
            memcpy( tmp+ti*width, X+i*width, width );
            ti++; i++;
        } else {
            memcpy( tmp+ti*width, X+j*width, width );
            ti++; j++;
        }
    }
    while (i<n/2) { /* finish up lower half */
        memcpy( tmp+ti*width, X+i*width, width );
        ti++; i++;
    }
    while (j<n) { /* finish up upper half */
        memcpy( tmp+ti*width, X+j*width, width );
        ti++; j++;
    }
    memcpy( X, tmp, n*width );
}

static inline void mergesort_recurse( char* X, size_t n, size_t width,
        int (*cmp)(const void*, const void*), char *tmp, size_t tasksize ) {
    if (n < 2) return;

   #pragma omp task shared(X) if (n > tasksize)
   mergesort_recurse(X, n/2, width, cmp, tmp, tasksize);

   #pragma omp task shared(X) if (n > tasksize)
   mergesort_recurse(X+(n/2)*width, n-(n/2), width, cmp, tmp + (n/2)*width, tasksize);

   #pragma omp taskwait
   mergesort_recurse_aux(X, n, width, cmp, tmp);
}

static inline void mergesort_recurse_r_aux( char* X, size_t n, size_t width,
        int (*cmp)(const void*, const void*, void*), void* ctx, char* tmp ) {
    size_t i = 0;
    size_t j = n/2;
    size_t ti = 0;

    while (i<n/2 && j<n) {
        if ((*cmp)( X+i*width, X+j*width, ctx ) <= 0) {
            memcpy( tmp+ti*width, X+i*width, width );
            ti++; i++;
        } else {
            memcpy( tmp+ti*width, X+j*width, width );
            ti++; j++;
        }
    }
    while (i<n/2) { /* finish up lower half */
        memcpy( tmp+ti*width, X+i*width, width );
        ti++; i++;
    }
    while (j<n) { /* finish up upper half */
        memcpy( tmp+ti*width, X+j*width, width );
        ti++; j++;
    }
    memcpy( X, tmp, n*width );
}

static inline void mergesort_recurse_r( char* X, size_t n, size_t width,
        int (*cmp)(const void*, const void*, void*), void* ctx, char *tmp, size_t tasksize ) {
    if (n < 2) return;

   #pragma omp task shared(X) if (n > tasksize)
   mergesort_recurse_r(X, n/2, width, cmp, ctx, tmp, tasksize);

   #pragma omp task shared(X) if (n > tasksize)
   mergesort_recurse_r(X+(n/2)*width, n-(n/2), width, cmp, ctx, tmp + (n/2)*width, tasksize);

   #pragma omp taskwait
   mergesort_recurse_r_aux(X, n, width, cmp, ctx, tmp);
}

