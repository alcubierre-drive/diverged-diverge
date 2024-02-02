/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_svd.h"
#include "batched_svd_private.h"

#ifdef USE_CUDA

#include <semaphore.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "../misc/cuda_error.h"

#include "mpi_log.h"

static inline void transpose_memcpy_128( void* out_, const void* in_, index_t N ) {
    complex128_t* out = (complex128_t*)out_;
    const complex128_t* in = (const complex128_t*)in_;
    for (index_t i=0; i<N; ++i)
    for (index_t j=0; j<N; ++j)
        out[IDX2(i,j,N)] = in[IDX2(j,i,N)];
}

typedef struct single_cusolver_solution_t single_cusolver_solution_t;

struct single_cusolver_solution_t {
    cusolverDnHandle_t handle;
    cudaStream_t stream;
    int dev;
    int do_eigen;

    cuDoubleComplex *dA, *dU, *dV;
    double *dS;
    complex128_t *A, *U, *V;
    double *S;

    void *dwork, *work;
    size_t nbytes_dwork, nbytes_work;
    int* dinfo;

    index_t dim;

    sem_t* sem;
    pthread_t thr;

    index_t* indices;
    index_t n_indices;
};

void* single_cusolver_thread( void* data ) {
    single_cusolver_solution_t* cfg = (single_cusolver_solution_t*)data;
    if (cfg->n_indices <= 0) return NULL;

    CUDA_CHECK( cudaSetDevice( cfg->dev ) );
    for (index_t ii=0; ii<cfg->n_indices; ++ii) {
        index_t i = cfg->indices[ii];
        transpose_memcpy_128( cfg->dA, cfg->A + i*POW2(cfg->dim), cfg->dim );
        CUDA_CHECK( cudaMemPrefetchAsync( cfg->dA, sizeof(complex128_t)*POW2(cfg->dim), cfg->dev, cfg->stream ) );
        CUDA_CHECK( cudaStreamSynchronize( cfg->stream ) );

        // those are exclusive
        sem_wait( cfg->sem );
        if (!cfg->do_eigen) {
            CUSOLVER_CHECK( cusolverDnXgesvd( cfg->handle, NULL, 'A', 'A',
                    cfg->dim, cfg->dim, CUDA_C_64F, cfg->dA, cfg->dim, CUDA_R_64F,
                    cfg->dS, CUDA_C_64F, cfg->dU, cfg->dim, CUDA_C_64F,
                    cfg->dV, cfg->dim, CUDA_C_64F, cfg->dwork, cfg->nbytes_dwork,
                    cfg->work, cfg->nbytes_work, cfg->dinfo) );
        } else {
            CUSOLVER_CHECK( cusolverDnXsyevd( cfg->handle, NULL,
                        CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                        cfg->dim, CUDA_C_64F, cfg->dA, cfg->dim, CUDA_R_64F,
                        cfg->dS, CUDA_C_64F, cfg->dwork, cfg->nbytes_dwork,
                        cfg->work, cfg->nbytes_work, cfg->dinfo ) );
        }
        CUDA_CHECK( cudaStreamSynchronize( cfg->stream ) );
        sem_post( cfg->sem );

        if (cfg->U && !cfg->do_eigen) transpose_memcpy_128( cfg->U + i*POW2(cfg->dim), cfg->dU, cfg->dim );
        else                          memcpy( cfg->U + i*POW2(cfg->dim), cfg->dA, sizeof(complex128_t)*POW2(cfg->dim) );

        if (cfg->V)
            memcpy( cfg->V + i*POW2(cfg->dim), cfg->dV, sizeof(complex128_t)*POW2(cfg->dim) );
        memcpy( cfg->S + i*cfg->dim, cfg->dS, sizeof(double)*cfg->dim );
    }

    return NULL;
}

void* batched_svd_gpu( void* data ) {

    batched_svd_cfg_t* cfg = (batched_svd_cfg_t*)data;

    if (cfg->cnt == 0) return NULL;

    int dev = cfg->dev[cfg->idx];
    CUDA_CHECK( cudaSetDevice( dev ) );

    index_t sz_mat = sizeof(cuDoubleComplex) * POW2(cfg->dim),
            sz_vec = sizeof(double) * cfg->dim;

    sem_t sol_sem;
    sem_init( &sol_sem, 0, 1 );
    single_cusolver_solution_t sol[2] = {0};
    for (int i=0; i<2; ++i) {
        CUDA_CHECK( cudaMallocManaged( (void**)&sol[i].dA, sz_mat, cudaMemAttachGlobal ) );
        CUDA_CHECK( cudaMallocManaged( (void**)&sol[i].dU, sz_mat, cudaMemAttachGlobal ) );
        CUDA_CHECK( cudaMallocManaged( (void**)&sol[i].dV, sz_mat, cudaMemAttachGlobal ) );
        CUDA_CHECK( cudaMallocManaged( (void**)&sol[i].dS, sz_vec, cudaMemAttachGlobal ) );
        sol[i].A = cfg->A;
        sol[i].U = cfg->U;
        sol[i].V = cfg->do_eigen ? NULL : cfg->V;
        sol[i].S = cfg->S;
        sol[i].dim = cfg->dim;
        sol[i].dev = dev;
        sol[i].sem = &sol_sem;
        sol[i].indices = malloc(sizeof(index_t) * cfg->cnt);
        sol[i].n_indices = 0;
        sol[i].do_eigen = cfg->do_eigen;
        CUSOLVER_CHECK( cusolverDnCreate( &sol[i].handle ) );
        CUSOLVER_CHECK( cusolverDnGetStream( sol[i].handle, &sol[i].stream ) );

        if (!sol[i].do_eigen) {
            CUSOLVER_CHECK( cusolverDnXgesvd_bufferSize( sol[i].handle, NULL, 'A', 'A',
                    cfg->dim, cfg->dim, CUDA_C_64F, sol[i].dA, cfg->dim, CUDA_R_64F,
                    sol[i].dS, CUDA_C_64F, sol[i].dU, cfg->dim, CUDA_C_64F,
                    sol[i].dV, cfg->dim, CUDA_C_64F, &sol[i].nbytes_dwork, &sol[i].nbytes_work ) );
        } else {
            CUSOLVER_CHECK( cusolverDnXsyevd_bufferSize( sol[i].handle, NULL,
                    CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, cfg->dim,
                    CUDA_C_64F, sol[i].dA, cfg->dim, CUDA_R_64F, sol[i].dS,
                    CUDA_C_64F, &sol[i].nbytes_dwork, &sol[i].nbytes_work ) );
        }

        CUDA_CHECK( cudaMalloc( (void**)&sol[i].dinfo, sizeof(int) ) );
        CUDA_CHECK( cudaMalloc( &sol[i].dwork, sol[i].nbytes_dwork ) );
        if (sol[i].nbytes_work > 0)
            sol[i].work = malloc(sol[i].nbytes_work);
        else
            sol[i].work = NULL;
    }

    // distribute indices
    for (index_t i=0; i<cfg->cnt; ++i) {
        int which = i >= cfg->cnt/2 ? 1 : 0;
        sol[which].indices[sol[which].n_indices++] = i;
    }

    pthread_t threads[2];
    // actual computation
    for (int i=0; i<2; ++i)
        pthread_create( &threads[i], NULL, &single_cusolver_thread, &sol[i] );
    for (int i=0; i<2; ++i)
        pthread_join( threads[i], NULL );


    for (int i=0; i<2; ++i) {
        CUSOLVER_CHECK( cusolverDnDestroy( sol[i].handle ) );
        CUDA_CHECK( cudaFree( sol[i].dA ) );
        CUDA_CHECK( cudaFree( sol[i].dU ) );
        CUDA_CHECK( cudaFree( sol[i].dV ) );
        CUDA_CHECK( cudaFree( sol[i].dS ) );
        CUDA_CHECK( cudaFree( sol[i].dinfo ) );
        CUDA_CHECK( cudaFree( sol[i].dwork ) );
        free( sol[i].work );
        free( sol[i].indices );
    }

    return NULL;
}

#else // USE_CUDA

void* batched_svd_gpu( void* data ) {
    (void)data;
    mpi_err_printf("CUDA not compiled in.\n");
    return NULL;
}

#endif // USE_CUDA
