#include "batched_gemm_small.h"
#include "init_internal_libs.h"
#include "gemm.h"

#ifndef USE_CUDA
typedef void* cublasHandle_t;
typedef void* cudaStream_t;
#else
#include <cublas_v2.h>
#endif

void batched_gemm_small_gpu( batched_gemm_small_handle_t* h, const complex128_t* A,
        const complex128_t* B, complex128_t* C );
void batched_gemm_small_cpu( batched_gemm_small_handle_t* h, const complex128_t* A,
        const complex128_t* B, complex128_t* C );

struct batched_gemm_small_handle_t {
    cublasHandle_t h[32];
    cudaStream_t s[32];
    index_t n;
    index_t num;

    int ndev;
    complex128_t* dA[32];
    complex128_t* dB[32];
    complex128_t* dC[32];

    index_t private_num[32];
    index_t private_mem_displ[32];

    batched_gemm_small_mode_t mode;
};

void batched_gemm_small( batched_gemm_small_handle_t* h, const complex128_t* A,
        const complex128_t* B, complex128_t* C ) {
    if (!h) {
        mpi_err_printf( "batched_gemm_small needs init via batched_gemm_small_init()\n" );
        return;
    }
    switch (h->mode) {
        case batched_gemm_small_mode_gpu:
            batched_gemm_small_gpu( h, A, B, C );
            break;
        case batched_gemm_small_mode_cpu:
        default:
            batched_gemm_small_cpu( h, A, B, C );
            break;
    }
}

void batched_gemm_small_mode( batched_gemm_small_handle_t* h, batched_gemm_small_mode_t mode ) {
    h->mode = mode;
}

#ifdef USE_CUDA
#include "cuda_error.h"
#include <pthread.h>

#include <cuda_runtime_api.h>

batched_gemm_small_handle_t* batched_gemm_small_init( index_t n, index_t num ) {
    batched_gemm_small_handle_t* h = calloc(1, sizeof(batched_gemm_small_handle_t));
    h->mode = batched_gemm_small_mode_gpu;

    CUDA_CHECK( cudaGetDeviceCount( &h->ndev ) );

    h->n = n;
    h->num = num;
    for (index_t i=0; i<num; ++i)
        h->private_num[i%h->ndev]++;
    for (int d=1; d<h->ndev; ++d)
        h->private_mem_displ[d] = h->private_mem_displ[d-1] + h->private_num[d-1]*POW2(n);

    for (int d=0; d<h->ndev; ++d) {
        CUDA_CHECK( cudaSetDevice(d) );
        CUDA_CHECK( cudaStreamCreate(&h->s[d]) );
        CUBLAS_CHECK( cublasCreate(&h->h[d]) );
        CUBLAS_CHECK( cublasSetStream(h->h[d], h->s[d]) );
        CUDA_CHECK( cudaMallocManaged((void**)&h->dA[d], sizeof(complex128_t)*h->private_num[d]*POW2(n),
                    cudaMemAttachGlobal) );
        CUDA_CHECK( cudaMallocManaged((void**)&h->dB[d], sizeof(complex128_t)*h->private_num[d]*POW2(n),
                    cudaMemAttachGlobal) );
        CUDA_CHECK( cudaMallocManaged((void**)&h->dC[d], sizeof(complex128_t)*h->private_num[d]*POW2(n),
                    cudaMemAttachGlobal) );
    }

    return h;
}

typedef struct {
    batched_gemm_small_handle_t* h;
    const complex128_t* A;
    const complex128_t* B;
    complex128_t* C;
    int d;
} batched_gemm_small_handle_thread_t;

#ifndef CUBLAS_BATCHED_GEMM_CALL
#define CUBLAS_BATCHED_GEMM_CALL cublasZgemmStridedBatched_64
#endif

static void* batched_gemm_small_dev( void* arg ) {
    batched_gemm_small_handle_thread_t* t = (batched_gemm_small_handle_thread_t*)arg;

    batched_gemm_small_handle_t* h = t->h;
    const complex128_t* A = t->A;
    const complex128_t* B = t->B;
    complex128_t* C = t->C;
    int d = t->d;
    cuDoubleComplex alpha = {0};
    alpha.x = 1.0;
    cuDoubleComplex beta = {0};
    CUDA_CHECK( cudaSetDevice(d) );
    memcpy( h->dA[d], A + h->private_mem_displ[d], sizeof(complex128_t)*h->private_num[d]*POW2(h->n) );
    memcpy( h->dB[d], B + h->private_mem_displ[d], sizeof(complex128_t)*h->private_num[d]*POW2(h->n) );
    CUBLAS_CHECK( CUBLAS_BATCHED_GEMM_CALL( h->h[d], CUBLAS_OP_N, CUBLAS_OP_N,
                h->n, h->n, h->n, &alpha,
                (cuDoubleComplex*)h->dB[d], h->n, POW2(h->n),
                (cuDoubleComplex*)h->dA[d], h->n, POW2(h->n), &beta,
                (cuDoubleComplex*)h->dC[d], h->n, POW2(h->n), h->private_num[d] ) );
    CUDA_CHECK( cudaStreamSynchronize(h->s[d]) );
    memcpy( C + h->private_mem_displ[d], h->dC[d], sizeof(complex128_t)*h->private_num[d]*POW2(h->n) );
    return NULL;
}

void batched_gemm_small_gpu( batched_gemm_small_handle_t* h, const complex128_t* A,
        const complex128_t* B, complex128_t* C ) {
    pthread_t threads[32];
    batched_gemm_small_handle_thread_t handles[32];
    for (int d=0; d<h->ndev; ++d) {
        handles[d] = (batched_gemm_small_handle_thread_t){.h = h, .A = A, .B = B, .C = C, .d = d};
        pthread_create( &threads[d], NULL, &batched_gemm_small_dev, &handles[d] );
    }
    for (int d=0; d<h->ndev; ++d)
        pthread_join( threads[d], NULL );
}

void batched_gemm_small_destroy( batched_gemm_small_handle_t* h ) {
    for (int d=0; d<h->ndev; ++d) {
        CUDA_CHECK( cudaFree(h->dA[d]) );
        CUDA_CHECK( cudaFree(h->dB[d]) );
        CUDA_CHECK( cudaFree(h->dC[d]) );
        CUBLAS_CHECK( cublasDestroy(h->h[d]) );
        CUDA_CHECK( cudaStreamDestroy(h->s[d]) );
    }
    free(h);
}

#else // USE_CUDA

batched_gemm_small_handle_t* batched_gemm_small_init( index_t n, index_t num ) {
    batched_gemm_small_handle_t* h = calloc(1, sizeof(batched_gemm_small_handle_t));
    h->mode = batched_gemm_small_mode_cpu;
    h->n = n;
    h->num = num;
    return h;
}

void batched_gemm_small_destroy( batched_gemm_small_handle_t* h ) {
    free(h);
}

void batched_gemm_small_gpu( batched_gemm_small_handle_t* h, const complex128_t* A,
        const complex128_t* B, complex128_t* C ) {
    mpi_err_printf( "batched_gemm_small_mode_gpu invalid, no CUDA, fallback to CPU\n" );
    batched_gemm_small_cpu( h, A, B, C );
}

#endif // USE_CUDA

void batched_gemm_small_cpu( batched_gemm_small_handle_t* h, const complex128_t* A,
        const complex128_t* B, complex128_t* C ) {
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<h->num; ++i)
        gemm( A+i*POW2(h->n), B+i*POW2(h->n), C+i*POW2(h->n), h->n, h->n, h->n, 1.0, 0.0 );
}
