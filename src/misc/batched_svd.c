/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_svd.h"
#include "batched_svd_private.h"
#include "mpi_functions.h"
#include <limits.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "../misc/cuda_error.h"
#include "eigen.h"
#endif

void batched_svd( int* dev, int n_dev, complex128_t* A, complex128_t* U,
        complex128_t* V, double* S, index_t dim, index_t cnt ) {

    if (!S || !A) {
        mpi_err_printf( "batched SVD A or Sigma cannot be NULL\n" );
        return;
    }

    batched_svd_cfg_t cfg_parent = { .idx=-1, .A=A, .U=U, .V=V, .S=S, .dim=dim, .cnt=cnt };
    int use_gpu_fun = 0;

    if (n_dev == INT_MIN || n_dev == INT_MIN+1) {
        cfg_parent.do_eigen = 1;
        n_dev = n_dev == INT_MIN ? -1 : 0;
    }

    // default (CPU) setup
    cfg_parent.n_dev = MIN(diverge_omp_num_threads(), BATCHED_SVD_THREAD_CNT_MAX);
    for (int d=0; d<cfg_parent.n_dev; ++d)
        cfg_parent.dev[d] = d;

#ifdef USE_CUDA
    if (n_dev != 0) {
        use_gpu_fun = 1;
        if (dev == NULL || n_dev < 0) {
            CUDA_CHECK( cudaGetDeviceCount( &cfg_parent.n_dev ) );
            for (int d=0; d<cfg_parent.n_dev; ++d)
                cfg_parent.dev[d] = d;
        } else {
            cfg_parent.n_dev = n_dev;
            for (int d=0; d<n_dev; ++d)
                cfg_parent.dev[d] = dev[d];
        }
    }
#else // USE_CUDA
    (void)dev;
    (void)n_dev;
#endif // USE_CUDA

    batched_svd_cfg_t* configs = malloc( cfg_parent.n_dev*sizeof(batched_svd_cfg_t) );

    index_t counts[BATCHED_SVD_THREAD_CNT_MAX] = {0};
    index_t displs[BATCHED_SVD_THREAD_CNT_MAX] = {0};

    for (index_t c=0; c<cfg_parent.cnt; ++c)
        counts[c%cfg_parent.n_dev]++;

    for (int d=1; d<cfg_parent.n_dev; ++d)
        displs[d] = counts[d-1] + displs[d-1];

    for (int d=0; d<cfg_parent.n_dev; ++d) {
        memcpy( configs+d, &cfg_parent, sizeof(batched_svd_cfg_t) );
        configs[d].idx = d;
        configs[d].parent = &cfg_parent;

        configs[d].cnt = counts[d];
        configs[d].A = cfg_parent.A + POW2(cfg_parent.dim) * displs[d];

        if (cfg_parent.U) configs[d].U = cfg_parent.U + POW2(cfg_parent.dim) * displs[d];
        else configs[d].U = NULL;

        if (cfg_parent.V) configs[d].V = cfg_parent.V + POW2(cfg_parent.dim) * displs[d];
        else configs[d].V = NULL;

        configs[d].S = cfg_parent.S + cfg_parent.dim * displs[d];
    }

    for (int d=0; d<cfg_parent.n_dev; ++d)
        pthread_create( &cfg_parent.threads[d], NULL, use_gpu_fun ? &batched_svd_gpu : &batched_svd_cpu, configs+d );

    for (int d=0; d<cfg_parent.n_dev; ++d)
        pthread_join( cfg_parent.threads[d], NULL );

    free(configs);

}
