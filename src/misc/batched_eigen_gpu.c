/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_eigen.h"
#include "batched_eigen_internals.h"

#ifdef USE_CUDA

#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "../misc/cuda_error.h"

#include "mpi_log.h"

static inline void transpose_memcpy_batched( cuDoubleComplex* out, const complex128_t* in, index_t num, index_t N ) {
    cuDoubleComplex* ptr = (cuDoubleComplex*)in;
    for (index_t n=0; n<num; ++n)
    for (index_t i=0; i<N; ++i)
    for (index_t j=0; j<N; ++j)
        out[IDX3(n,i,j,N,N)] = ptr[IDX3(n,j,i,N,N)];
}

void* batched_eigen_gpu( void* data ) {
    batched_eigen_t* cfg = (batched_eigen_t*)data;
    if (cfg->N == 0 || cfg->n == 0 || cfg->H == NULL) return NULL;
    CUDA_CHECK( cudaSetDevice( cfg->device ) );
    cuDoubleComplex *H;
    double *E;

    int N_per_run = cfg->N / cfg->nbatches + (cfg->N % cfg->nbatches != 0);
    int N_runs = cfg->nbatches;

    CUDA_CHECK( cudaMallocManaged( (void**)&H, sizeof(cuDoubleComplex)*cfg->n*cfg->n*N_per_run, cudaMemAttachGlobal ) );
    CUDA_CHECK( cudaMallocManaged( (void**)&E, sizeof(double)*cfg->n*N_per_run, cudaMemAttachGlobal ) );
    memset( E, 0, sizeof(double)*cfg->n*N_per_run );

    cusolverDnHandle_t handle;
    cudaStream_t stream;
    syevjInfo_t info;
    int lwork;

    CUSOLVER_CHECK( cusolverDnCreate(&handle) );
    CUSOLVER_CHECK( cusolverDnCreateSyevjInfo(&info) );
    CUSOLVER_CHECK( cusolverDnGetStream( handle, &stream ) );

    CUSOLVER_CHECK( cusolverDnZheevjBatched_bufferSize( handle,
            CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, cfg->n,
            H, cfg->n, E, &lwork, info, N_per_run ) );
    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    cuDoubleComplex *work;
    CUDA_CHECK( cudaMallocManaged( (void**)&work, sizeof(cuDoubleComplex)*lwork, cudaMemAttachGlobal ) );
    int *pinfo;
    CUDA_CHECK( cudaMallocManaged( (void**)&pinfo, sizeof(int)*N_per_run, cudaMemAttachGlobal ) );
    CUDA_CHECK( cudaMemPrefetchAsync( E, sizeof(double)*cfg->n*N_per_run, cfg->device, stream ) );
    CUDA_CHECK( cudaMemPrefetchAsync( pinfo, sizeof(int)*N_per_run, cfg->device, stream ) );
    CUDA_CHECK( cudaMemPrefetchAsync( work, sizeof(cuDoubleComplex)*lwork, cfg->device, stream ) );
    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    for (int run=0; run<N_runs; ++run) {
        int N_this_run = (run+1)*N_per_run < cfg->N ? N_per_run : (cfg->N - run*N_per_run);
        if(N_this_run > 0) {
            transpose_memcpy_batched( H, cfg->H+N_per_run*run*cfg->n*cfg->n, N_this_run, cfg->n );
            CUDA_CHECK( cudaMemPrefetchAsync( H, sizeof(cuDoubleComplex)*cfg->n*cfg->n*N_this_run, cfg->device, stream ) );
            CUSOLVER_CHECK( cusolverDnZheevjBatched( handle,
                    CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, cfg->n,
                    H, cfg->n, E, work, lwork, pinfo, info, N_this_run ) );
            CUDA_CHECK( cudaStreamSynchronize( stream ) );
            memcpy( cfg->H + N_per_run*run*cfg->n*cfg->n, H, sizeof(cuDoubleComplex)*cfg->n*cfg->n*N_this_run );

            if (cfg->Er == NULL) {
                for (index_t i=0; i<N_this_run*cfg->n; ++i)
                    cfg->E[N_per_run*run*cfg->n+i] = E[i];
            } else {
                for (index_t i=0; i<N_this_run*cfg->n; ++i)
                    cfg->Er[N_per_run*run*cfg->n+i] = E[i];
            }
            CUDA_CHECK( cudaStreamSynchronize( stream ) );
        }
    }

    CUDA_CHECK( cudaFree( work ) );
    CUDA_CHECK( cudaFree( pinfo ) );
    CUDA_CHECK( cudaFree( H ) );
    CUDA_CHECK( cudaFree( E ) );

    CUSOLVER_CHECK( cusolverDnDestroy(handle) );
    CUSOLVER_CHECK( cusolverDnDestroySyevjInfo(info) );

    return NULL;
}

#else // USE_CUDA

void* batched_eigen_gpu( void* data ) {
    (void)data;
    mpi_err_printf("CUDA not compiled in.\n");
    return NULL;
}

#endif // USE_CUDA
