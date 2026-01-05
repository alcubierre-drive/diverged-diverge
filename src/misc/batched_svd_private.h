/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BATCHED_SVD_THREAD_CNT_MAX
#define BATCHED_SVD_THREAD_CNT_MAX 32
#endif // BATCHED_SVD_THREAD_CNT_MAX

typedef struct batched_svd_cfg_t batched_svd_cfg_t;

struct batched_svd_cfg_t {
    batched_svd_cfg_t* parent;
    int idx;
    pthread_t threads[BATCHED_SVD_THREAD_CNT_MAX];

    int dev[BATCHED_SVD_THREAD_CNT_MAX];
    int n_dev;

    complex128_t* A;
    complex128_t* U;
    complex128_t* V;
    double* S;

    index_t dim;
    index_t cnt;

    int do_eigen;
};

void* batched_svd_gpu( void* data );
void* batched_svd_cpu( void* data );

#ifdef __cplusplus
}
#endif
