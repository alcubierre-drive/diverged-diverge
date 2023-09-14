/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "kernels_helpers.h"
#include "kernels_gpu_helpers.hpp"
#include "../diverge_model.h"

typedef unsigned short ssindex_t;
typedef unsigned int sindex_t;

static void helper_kernel_alloc( index_t V_sz, index_t, index_t k3map_sz, index_t W_sz );
static void helper_kernel_memory_setup( index_t V_sz, index_t, index_t k3map_sz, index_t W_sz );
static void helper_kernel_free();

static index_t* distribute_gpu_workshare( index_t start, index_t stop, index_t factor );

static int* helper_kernel_gpus;
static int helper_kernel_ngpus;
static int* helper_kernel_gpu_weights = NULL;
static int helper_kernel_nblocks[3];
static cudaStream_t* helper_kernel_streams = NULL;

static void*** helper_kernel_memory = NULL;
static size_t* helper_kernel_memory_size = NULL;
static int helper_kernel_memory_count = 5;

static void call_gpu_kernel( cu_cdoub_t* dphi, const cu_cdoub_t* V, const cu_cdoub_t* L, const double* W,
        const index_t* k3map, index_t start, index_t stop, index_t np, index_t nb, bool SU2,
        const char* chan, cudaStream_t& stream, index_t n_spin );

void* npatch_gpu_flow_step_any ( void* params );
static void* exec_call_gpu_kernel( void* _params );

#define CONST_SU2 true
#define CONST_nonSU2 false
#define IMPL_ANY_GPU_FLOW_STEP( SU, CHAN ) \
void* npatch_gpu_flow_step_##SU##_##CHAN ( void* params ) { \
    double tick = diverge_mpi_wtime(); \
    struct npatch_gpu_kernel_start_t* args = (struct npatch_gpu_kernel_start_t*) params; \
    args->SU2 = CONST_##SU ; \
    strcpy( args->chan, #CHAN ); \
    void* result = npatch_gpu_flow_step_any( args ); \
    double tock = diverge_mpi_wtime(); \
    args->wtime = tock - tick; \
    return result; \
}

IMPL_ANY_GPU_FLOW_STEP( SU2, P )
IMPL_ANY_GPU_FLOW_STEP( SU2, C )
IMPL_ANY_GPU_FLOW_STEP( SU2, D1 )
IMPL_ANY_GPU_FLOW_STEP( SU2, D2 )
IMPL_ANY_GPU_FLOW_STEP( SU2, D3 )
IMPL_ANY_GPU_FLOW_STEP( nonSU2, P )
IMPL_ANY_GPU_FLOW_STEP( nonSU2, C )
IMPL_ANY_GPU_FLOW_STEP( nonSU2, D1 )

void* npatch_gpu_flow_step_SU2_P( void* );

void* npatch_gpu_flow_step_any ( void* params ) {
    struct npatch_gpu_kernel_start_t* shared_args = (struct npatch_gpu_kernel_start_t*) params;

    index_t np = shared_args->L->np;
    index_t nb = shared_args->L->nb;
    index_t nb4 = nb*nb*nb*nb;

    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * helper_kernel_ngpus);
    struct npatch_gpu_kernel_start_t* args = (struct npatch_gpu_kernel_start_t*)malloc(sizeof(struct npatch_gpu_kernel_start_t) * helper_kernel_ngpus);
    index_t* workshare = distribute_gpu_workshare( shared_args->start_at/nb4, shared_args->stop_at/nb4, nb4 );
    index_t* workshare_start = (index_t*)malloc(sizeof(index_t)*helper_kernel_ngpus);
    index_t* workshare_stop = (index_t*)malloc(sizeof(index_t)*helper_kernel_ngpus);
    workshare_start[0] = shared_args->start_at;
    for (int g=1; g<helper_kernel_ngpus; ++g) {
        workshare_start[g] = workshare_start[g-1] + workshare[g-1];
        workshare_stop[g-1] = workshare_start[g];
    }
    workshare_stop[helper_kernel_ngpus-1] = workshare_start[helper_kernel_ngpus-1] + workshare[helper_kernel_ngpus-1];

    index_t Phi_sz = 0;
    /* use maximum Phi_sz here */
    for (int g=0; g<helper_kernel_ngpus; ++g)
        Phi_sz = Phi_sz > (workshare_stop[g] - workshare_start[g]) ? Phi_sz : (workshare_stop[g] - workshare_start[g]);
    helper_kernel_alloc( sizeof(complex128_t)*shared_args->V->size, sizeof(complex128_t)*Phi_sz, sizeof(index_t)*np*np*np, sizeof(double)*np );

    for (int g=0; g<helper_kernel_ngpus; ++g) {
        args[g] = *shared_args;
        args[g].start_at = workshare_start[g];
        args[g].stop_at = workshare_stop[g];
        args[g].device = g;
        //printf("gpu %i: %lli -> %lli\n", g, workshare_start[g],
        //        workshare_stop[g]);
        pthread_create( threads+g, NULL, exec_call_gpu_kernel, args+g );
    }
    for (int g=0; g<helper_kernel_ngpus; ++g) {
        pthread_join( threads[g], NULL );
        shared_args->wtime_gpus[g%128] = args[g].wtime;
    }

    free(threads);
    free(args);
    free(workshare);
    free(workshare_start);
    free(workshare_stop);
    return NULL;
}

static void* exec_call_gpu_kernel( void* _params ) {
    double tick = diverge_mpi_wtime();
    struct npatch_gpu_kernel_start_t* params = (struct npatch_gpu_kernel_start_t*) _params;

    CUDA_CHECK( cudaSetDevice( helper_kernel_gpus[params->device] ) );

    index_t start = params->start_at,
            stop = params->stop_at;

    index_t sz = params->V->size;
    index_t np = params->L->np;
    index_t nb = params->L->nb;

    cu_cdoub_t *V_d, *dPhi_d, *L_d;
    index_t *k3map_d;

    V_d = (cu_cdoub_t*)(helper_kernel_memory[params->device][0]);
    L_d = (cu_cdoub_t*)(helper_kernel_memory[params->device][1]);
    k3map_d = (index_t*)(helper_kernel_memory[params->device][2]);
    dPhi_d = (cu_cdoub_t*)(helper_kernel_memory[params->device][3]);
    double* W_d = (double*)(helper_kernel_memory[params->device][4]);

    cu_cdoub_t* L = (cu_cdoub_t*)(!strcmp(params->chan, "P") ? params->L->Lm : params->L->Lp);
    cudaStream_t& stream = helper_kernel_streams[params->device];

    memcpy( V_d, params->V->V, sz*sizeof(cu_cdoub_t) );
    CUDA_CHECK( cudaMemPrefetchAsync( V_d, sz*sizeof(cu_cdoub_t), helper_kernel_gpus[params->device], stream ) );
    memcpy( L_d, L, sz*sizeof(cu_cdoub_t) );
    CUDA_CHECK( cudaMemPrefetchAsync( L_d, sz*sizeof(cu_cdoub_t), helper_kernel_gpus[params->device], stream  ) );
    memcpy( k3map_d, params->L->k3map, np*np*np*sizeof(index_t) );
    CUDA_CHECK( cudaMemPrefetchAsync( k3map_d, np*np*np*sizeof(index_t), helper_kernel_gpus[params->device], stream  ) );
    CUDA_CHECK( cudaMemset( dPhi_d, 0, sz*sizeof(cu_cdoub_t) ) );
    CUDA_CHECK( cudaMemcpy( W_d, params->L->wgh, np*sizeof(double), cudaMemcpyHostToDevice ) );

    CUDA_CHECK( cudaStreamSynchronize( stream ) );
    call_gpu_kernel( dPhi_d, V_d, L_d, W_d, k3map_d, start, stop, np, nb, params->SU2, params->chan, stream, abs(params->L->dmod->n_spin) );

    complex128_t* dPhi_copy = (complex128_t*)malloc(sizeof(complex128_t)*(stop-start));
    CUDA_CHECK( cudaMemcpy( dPhi_copy, dPhi_d+start, sizeof(complex128_t)*(stop-start), cudaMemcpyDeviceToHost ) );
    CUDA_CHECK( cudaDeviceSynchronize() );
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t i=start; i<stop; ++i)
        params->dPhi[i] += dPhi_copy[i-start];
    free(dPhi_copy);
    double tock = diverge_mpi_wtime();
    params->wtime = tock - tick;
    return NULL;
}

void npatch_flow_gpu_setup( const int* gpus, int ngpus, const int nblocks[3] ) {
    helper_kernel_ngpus = ngpus;
    helper_kernel_gpus = (int*)malloc(sizeof(int)*ngpus);
    helper_kernel_streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*ngpus);
    for (int i=0; i<ngpus; ++i) {
        if (gpus)
            helper_kernel_gpus[i] = gpus[i];
        else
            helper_kernel_gpus[i] = i;
        CUDA_CHECK( cudaSetDevice( helper_kernel_gpus[i] ) );
        CUDA_CHECK( cudaStreamCreate( helper_kernel_streams+i ) );
    }
    for (int i=0; i<2; ++i) {
        helper_kernel_nblocks[i] = nblocks[i];
    }
}

int npatch_flow_gpu_device_count( void ) {
    int ndev = 0;
    CUDA_CHECK( cudaGetDeviceCount(&ndev) );
    return ndev;
}

void npatch_flow_gpu_set_weights( const int* gpu_weights ) {
    if (gpu_weights) {
        helper_kernel_gpu_weights = (int*)malloc(sizeof(int)*helper_kernel_ngpus);
        for (int g=0; g<helper_kernel_ngpus; ++g)
            helper_kernel_gpu_weights[g] = gpu_weights[g];
    } else {
        helper_kernel_gpu_weights = NULL;
    }
}

void npatch_setup_query_gpu( const char* property, void** data, index_t* psize ) {
    index_t size = 0;
    if        (!strcmp(property,"ngpus")) {
        *data = malloc( (size = sizeof(int)) );
        *(int*)(*data) = helper_kernel_ngpus;
    } else if (!strcmp(property,"gpus")) {
        *data = malloc( (size = sizeof(int) * helper_kernel_ngpus) );
        for (int i=0; i<helper_kernel_ngpus; ++i) ((int*)(*data))[i] = helper_kernel_gpus[i];
    } else if (!strcmp(property,"nblocks")) {
        *data = malloc( (size = sizeof(int) * 3) );
        for (int i=0; i<3; ++i) ((int*)(*data))[i] = helper_kernel_nblocks[i];
    } else {
        mpi_wrn_printf( "property '%s' not found.", property );
        *data = malloc( (size = sizeof(char)) );
        *(char*)(*data) = '\0';
    }
    if (psize) *psize = size;
}

void npatch_flow_gpu_reset() {
    helper_kernel_free();
}

void npatch_flow_gpu_teardown() {
    helper_kernel_free();
    for (int i=0; i<helper_kernel_ngpus; ++i) {
        CUDA_CHECK( cudaStreamSynchronize( helper_kernel_streams[i] ) );
        CUDA_CHECK( cudaStreamDestroy( helper_kernel_streams[i] ) );
    }
    free(helper_kernel_gpus);
    free(helper_kernel_streams);
    if (helper_kernel_gpu_weights) free(helper_kernel_gpu_weights);
}

__global__ static void gpu_kernel_SU2_P    ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_SU2_C    ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_SU2_D1   ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_SU2_D2   ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_SU2_D3   ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_nonSU2_P ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_nonSU2_C ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_nonSU2_D1( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_oversize_SU2_P    ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_oversize_SU2_C    ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_oversize_SU2_D1   ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_oversize_SU2_D2   ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_oversize_SU2_D3   ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_oversize_nonSU2_P ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_oversize_nonSU2_C ( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );
__global__ static void gpu_kernel_oversize_nonSU2_D1( cu_cdoub_t* __restrict__ dphi, const
        cu_cdoub_t* __restrict__ V, const cu_cdoub_t* __restrict__ L, const double*
        __restrict__ W, const index_t* __restrict__ k3map, index_t phi_k_idx_start,
        index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv );

static void helper_kernel_memory_setup( index_t V_sz, index_t, index_t k3map_sz, index_t W_sz ) {
    helper_kernel_memory = (void***)malloc(sizeof(void**)*helper_kernel_ngpus);
    helper_kernel_memory_size = (size_t*)malloc(sizeof(size_t*)*helper_kernel_memory_count);
    helper_kernel_memory_size[0] = V_sz;
    helper_kernel_memory_size[1] = V_sz;
    helper_kernel_memory_size[2] = k3map_sz;
    helper_kernel_memory_size[3] = V_sz;
    helper_kernel_memory_size[4] = W_sz;

    double sizes_sum = 0.0;
    for (int i=0; i<helper_kernel_memory_count; ++i) sizes_sum += helper_kernel_memory_size[i];
    mpi_vrb_printf("initializing contractions on GPU, using %.1fGiB\n", sizes_sum/(1024.0*1024.0*1024.0));

    // this is important, as atomicAdd only works for device memory
    for (int dev=0; dev<helper_kernel_ngpus; ++dev) {
        CUDA_CHECK( cudaSetDevice( helper_kernel_gpus[dev] ) );
        helper_kernel_memory[dev] = (void**)malloc(sizeof(void*)*helper_kernel_memory_count);
        for (int cnt=0; cnt<helper_kernel_memory_count; ++cnt) {
            if (cnt == helper_kernel_memory_count - 1) {
                // alloc dPhi on the GPUs
                CUDA_CHECK( cudaMalloc( helper_kernel_memory[dev] + cnt,
                            helper_kernel_memory_size[cnt] ) );
            } else {
                // and everything else in unified memory
                CUDA_CHECK( cudaMallocManaged( helper_kernel_memory[dev] + cnt, helper_kernel_memory_size[cnt] ) );
            }
        }
    }
    size_t shared_size = sizeof(complex128_t) * (V_sz/sizeof(complex128_t)) / (k3map_sz/sizeof(index_t)); // nb*nb*nb*nb * sizeof(complex128_t)
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_SU2_P,     cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_SU2_C,     cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_SU2_D1,    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_SU2_D2,    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_SU2_D3,    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_nonSU2_P,  cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_nonSU2_C,  cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_nonSU2_D1, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_oversize_SU2_P,     cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_oversize_SU2_C,     cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_oversize_SU2_D1,    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_oversize_SU2_D2,    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_oversize_SU2_D3,    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_oversize_nonSU2_P,  cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_oversize_nonSU2_C,  cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
    CUDA_CHECK( cudaFuncSetAttribute(gpu_kernel_oversize_nonSU2_D1, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size) );
}

static void helper_kernel_alloc( index_t V_sz, index_t, index_t k3map_sz, index_t W_sz ) {
    if (helper_kernel_memory == NULL) {
        helper_kernel_memory_setup( V_sz, 0, k3map_sz, W_sz );
    }
}

static void helper_kernel_free() {
    if (helper_kernel_memory != NULL) {
        for (int dev=0; dev<helper_kernel_ngpus; ++dev) {
            CUDA_CHECK( cudaSetDevice( helper_kernel_gpus[dev] ) );
            for (int cnt=0; cnt<helper_kernel_memory_count; ++cnt) {
                CUDA_CHECK( cudaFree( helper_kernel_memory[dev][cnt] ) );
            }
            free( helper_kernel_memory[dev] );
        }
        free( helper_kernel_memory );
        helper_kernel_memory = NULL;
    }
    if (helper_kernel_memory_size != NULL) {
        free( helper_kernel_memory_size );
        helper_kernel_memory_size = NULL;
    }
}

static index_t* distribute_gpu_workshare( index_t start, index_t stop, index_t factor ) {
    index_t* workshare = (index_t*)calloc(sizeof(index_t), helper_kernel_ngpus);

    if (helper_kernel_gpu_weights == NULL) {
        for (index_t s=start; s<stop; ++s)
            workshare[s%helper_kernel_ngpus]++;
    } else {
        index_t total_weight = 0;
        for (int g=0; g<helper_kernel_ngpus; ++g) total_weight += helper_kernel_gpu_weights[g];

        int* gpu_per_tot_idx = (int*)calloc(sizeof(index_t), total_weight);
        for (int i=0; i<total_weight; ++i) {
            int t=0;
            for (int g=0; g<helper_kernel_ngpus; ++g)
                for (int w=0; w<helper_kernel_gpu_weights[g]; ++w)
                    gpu_per_tot_idx[t++] = g;
        }

        for (index_t s=start; s<stop; ++s)
            workshare[gpu_per_tot_idx[s%total_weight]]++;

        free( gpu_per_tot_idx );
    }

    for (index_t g=0; g<helper_kernel_ngpus; ++g) {
        //printf("%lli\n", workshare[g]);
        workshare[g] *= factor;
    }
    return workshare;
}

static void call_gpu_kernel( cu_cdoub_t* dphi, const cu_cdoub_t* V, const cu_cdoub_t* L, const double* W,
        const index_t* k3map, index_t start, index_t stop, index_t np, index_t nb, bool SU2,
        const char* chan, cudaStream_t& stream, index_t n_spin ) {
    int facInv = 0; // default;
    if (!strcmp(chan, "P") && !SU2) facInv = n_spin; // only change when in the one relevant case

    dim3 nblocks(helper_kernel_nblocks[0],helper_kernel_nblocks[1],1);

    index_t kstart = np * start / (nb*nb*nb*nb);
    index_t kstop = np * stop / (nb*nb*nb*nb);
    index_t nk_per_iter = nblocks.x*nblocks.y;
    index_t niter = (kstop-kstart) / nk_per_iter + ((kstop-kstart)%nk_per_iter > 0);
    index_t shared_size = sizeof(complex128_t)*nb*nb*nb*nb;
    // printf("%lli %lli %lli %lli %lli [(%i,%i),(%i,%i)]\n", kstart, kstop, nk_per_iter, niter,
    //         shared_size, nblocks.x, nblocks.y, nthreads.x, nthreads.y);
    for (index_t iter=0; iter<niter; ++iter) {
        if (nb*nb*nb*nb > MAX_THREADS_PER_BLOCK) {
            if (nb*nb*nb > MAX_THREADS_PER_BLOCK) {
                mpi_err_printf("cannot launch kernel with %lli=%lli**3 threads per block (max %i)\n",
                        nb*nb*nb, nb, MAX_THREADS_PER_BLOCK);
                return;
            }
            dim3 nthreads(nb,nb,nb);
            if (SU2) {
                if (!strcmp(chan,"P"))
                    gpu_kernel_oversize_SU2_P<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"C"))
                    gpu_kernel_oversize_SU2_C<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"D1"))
                    gpu_kernel_oversize_SU2_D1<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"D2"))
                    gpu_kernel_oversize_SU2_D2<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"D3"))
                    gpu_kernel_oversize_SU2_D3<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
            } else {
                if (!strcmp(chan,"P"))
                    gpu_kernel_oversize_nonSU2_P<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"C"))
                    gpu_kernel_oversize_nonSU2_C<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"D1"))
                    gpu_kernel_oversize_nonSU2_D1<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
            }
        } else {
            dim3 nthreads(nb*nb,nb*nb,1);
            if (SU2) {
                if (!strcmp(chan,"P"))
                    gpu_kernel_SU2_P<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"C"))
                    gpu_kernel_SU2_C<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"D1"))
                    gpu_kernel_SU2_D1<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"D2"))
                    gpu_kernel_SU2_D2<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"D3"))
                    gpu_kernel_SU2_D3<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
            } else {
                if (!strcmp(chan,"P"))
                    gpu_kernel_nonSU2_P<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"C"))
                    gpu_kernel_nonSU2_C<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
                if (!strcmp(chan,"D1"))
                    gpu_kernel_nonSU2_D1<<<nblocks,nthreads,shared_size,stream>>>( dphi, V, L, W, k3map, kstart + iter*nk_per_iter, kstop, np, nb, facInv );
            }
        }
    }
}

#define DO_PRAGMA(x) _Pragma (#x)
#define ESC_PARENS(...) __VA_ARGS__

#define INVOKE_UNRAVEL_V(...) UNRAVEL_V(__VA_ARGS__)
#define INVOKE_RAVEL_V(...) RAVEL_V(__VA_ARGS__)
#define INVOKE_RAVEL_L3(...) RAVEL_L3(__VA_ARGS__)

#define K3ADD( k1, k2, k3 ) k3map[(k1*np+k2)*np+k3]

#define UNRAVEL_K4( idx, k1, k2, k3, k4 ) \
    sindex_t k1, k2, k3, k4; \
    k1 = idx / (np*np*np); \
    idx %= (np*np*np); \
    k2 = idx / (np*np); \
    idx %= (np*np); \
    k3 = idx / np; \
    k4 = idx % np;

#define CONTRACT_P_Vl ESC_PARENS(k1, k2, k, o1, o2, p1, p2)
#define CONTRACT_P_Vr ESC_PARENS(k, K3ADD( k1, k2, k ), k3, p3, p4, o3, o4)
#define CONTRACT_P_L3 ESC_PARENS(k, k1, k2, p1, p2, p3, p4)
#define CONTRACT_P_nm const ssindex_t o1 = n, o2 = m, p3 = o, p4 = p;
#define CONTRACT_P_op const ssindex_t o1 = n, o2 = m, o3 = o, o4 = p;

#define CONTRACT_C_Vl ESC_PARENS(k1, K3ADD( k2, k, k3 ), k, o1, p2, p1, o4)
#define CONTRACT_C_Vr ESC_PARENS(k, k2, k3,                 p3, o2, o3, p4)
#define CONTRACT_C_L3 ESC_PARENS(k, k2, k3, p1, p4, p3, p2)
#define CONTRACT_C_nm const ssindex_t o1 = n, o4 = m, p3 = o, p4 = p;
#define CONTRACT_C_op const ssindex_t o1 = n, o4 = m, o2 = o, o3 = p;

#define CONTRACT_D1_Vl ESC_PARENS(k1, K3ADD( k, k3, k1 ), k3, o1, p1, o3, p2)
#define CONTRACT_D1_Vr ESC_PARENS(k, k2, K3ADD( k, k3, k1 ), p4, o2, p3, o4)
#define CONTRACT_D1_L3 ESC_PARENS(k, k3, k1, p2, p3, p4, p1)
#define CONTRACT_D1_nm const ssindex_t o1 = n, o3 = m, p3 = o, p4 = p;
#define CONTRACT_D1_op const ssindex_t o1 = n, o3 = m, o2 = o, o4 = p;

#define CONTRACT_D2_Vl ESC_PARENS(k1, K3ADD( k, k3, k1 ), k3, o1, p1, o3, p2)
#define CONTRACT_D2_Vr ESC_PARENS(k, k2, K3ADD( k1, k2, k3 ), p4, o2, o4, p3)
#define CONTRACT_D2_L3 ESC_PARENS(k, k3, k1, p2, p3, p4, p1)
#define CONTRACT_D2_nm const ssindex_t o1 = n, o3 = m, p3 = o, p4 = p;
#define CONTRACT_D2_op const ssindex_t o1 = n, o3 = m, o2 = o, o4 = p;

#define CONTRACT_D3_Vl ESC_PARENS(k1, K3ADD( k, k3, k1 ), k, o1, p2, p1, o3)
#define CONTRACT_D3_Vr ESC_PARENS(k, k2, K3ADD( k, k3, k1 ), p3, o2, p4, o4)
#define CONTRACT_D3_L3 ESC_PARENS(k, k3, k1, p1, p4, p3, p2)
#define CONTRACT_D3_nm const ssindex_t o1 = n, o3 = m, p3 = o, p4 = p;
#define CONTRACT_D3_op const ssindex_t o1 = n, o3 = m, o2 = o, o4 = p;

#define GPU_KERNEL_IMPL( su, chan, factor ) \
__global__ static void gpu_kernel_##su##_##chan ( cu_cdoub_t* dphi, const cu_cdoub_t* V, const cu_cdoub_t* L, const double* W, \
        const index_t* k3map, index_t phi_k_idx_start, index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv ) { \
    extern __shared__ cu_cdoub_t _kernel_cache[]; \
    const ssindex_t n = threadIdx.x / nb, \
                    m = threadIdx.x % nb, \
                    o = threadIdx.y / nb, \
                    p = threadIdx.y % nb; \
    const sindex_t oidx = threadIdx.x*blockDim.y + threadIdx.y; \
    \
    index_t phi_k_idx = blockIdx.x * gridDim.y + blockIdx.y + phi_k_idx_start; \
    if (phi_k_idx >= phi_k_idx_stop) return; \
    UNRAVEL_K4( phi_k_idx, k1, k2, k3, k ) \
    const double nrm = W[k] * ((facInv == 0) ? factor : 1./(double)facInv); \
    \
    _kernel_cache[oidx].x = 0.0; \
    _kernel_cache[oidx].y = 0.0; \
    { CONTRACT_##chan##_nm \
    for (ssindex_t p1=0; p1<nb; ++p1) \
    for (ssindex_t p2=0; p2<nb; ++p2) { \
        const index_t lidx = INVOKE_RAVEL_L3( CONTRACT_##chan##_L3 ); \
        const index_t vidx = INVOKE_RAVEL_V( CONTRACT_##chan##_Vl ); \
        cu_cdoub_t val = cuCmul( L[lidx], V[vidx] ); \
        cu_complex_atomic_add( _kernel_cache + oidx, &val ); \
    } cu_complex_scale( _kernel_cache + oidx, nrm ); } \
    \
    __syncthreads(); \
    \
    { CONTRACT_##chan##_op \
    const index_t pidx = RAVEL_V( k1, k2, k3, o1, o2, o3, o4 ); \
    for (ssindex_t p3=0; p3<nb; ++p3) \
    for (ssindex_t p4=0; p4<nb; ++p4) { \
        const sindex_t oidx = RAVEL_O4( n, m, p3, p4 ); \
        const index_t vidx = INVOKE_RAVEL_V( CONTRACT_##chan##_Vr ); \
        cu_cdoub_t val = cuCmul( _kernel_cache[oidx], V[vidx] ); \
        cu_complex_atomic_add( dphi + pidx, &val ); \
    }} \
}

#define GPU_KERNEL_IMPL_OVERSIZE( su, chan, factor ) \
__global__ static void gpu_kernel_oversize_##su##_##chan ( cu_cdoub_t* dphi, const cu_cdoub_t* V, const cu_cdoub_t* L, const double* W, \
        const index_t* k3map, index_t phi_k_idx_start, index_t phi_k_idx_stop, sindex_t np, ssindex_t nb, ssindex_t facInv ) { \
    extern __shared__ cu_cdoub_t _kernel_cache[]; \
    index_t phi_k_idx = blockIdx.x * gridDim.y + blockIdx.y + phi_k_idx_start; \
    if (phi_k_idx >= phi_k_idx_stop) return; \
    \
    const ssindex_t n = threadIdx.x, \
                    m = threadIdx.y, \
                    o = threadIdx.z; \
    \
    UNRAVEL_K4( phi_k_idx, k1, k2, k3, k ) \
    const double nrm = W[k] * ((facInv == 0) ? factor : 1./(double)facInv); \
    \
    for (ssindex_t p=0; p<nb; ++p) { \
        const sindex_t oidx = (((n*nb + m)*nb + o)*nb + p); \
        CONTRACT_##chan##_nm \
        _kernel_cache[oidx].x = 0.0; \
        _kernel_cache[oidx].y = 0.0; \
        for (ssindex_t p1=0; p1<nb; ++p1) \
        for (ssindex_t p2=0; p2<nb; ++p2) { \
            const index_t lidx = INVOKE_RAVEL_L3( CONTRACT_##chan##_L3 ); \
            const index_t vidx = INVOKE_RAVEL_V( CONTRACT_##chan##_Vl ); \
            cu_cdoub_t val = cuCmul( L[lidx], V[vidx] ); \
            cu_complex_atomic_add( _kernel_cache + oidx, &val ); \
        } \
        cu_complex_scale( _kernel_cache + oidx, nrm ); \
    } \
    \
    __syncthreads(); \
    \
    for (ssindex_t p=0; p<nb; ++p) { \
        CONTRACT_##chan##_op \
        const index_t pidx = RAVEL_V( k1, k2, k3, o1, o2, o3, o4 ); \
        for (ssindex_t p3=0; p3<nb; ++p3) \
        for (ssindex_t p4=0; p4<nb; ++p4) { \
            const sindex_t oidx = RAVEL_O4( n, m, p3, p4 ); \
            const index_t vidx = INVOKE_RAVEL_V( CONTRACT_##chan##_Vr ); \
            cu_cdoub_t val = cuCmul( _kernel_cache[oidx], V[vidx] ); \
            cu_complex_atomic_add( dphi + pidx, &val ); \
        } \
    } \
}

GPU_KERNEL_IMPL( SU2, P, 1.0 )
GPU_KERNEL_IMPL( SU2, C, 1.0 )
GPU_KERNEL_IMPL( SU2, D1, -2.0 )
GPU_KERNEL_IMPL( SU2, D2, 1.0 )
GPU_KERNEL_IMPL( SU2, D3, 1.0 )
GPU_KERNEL_IMPL( nonSU2, P, 0.5 )
GPU_KERNEL_IMPL( nonSU2, C, 1.0 )
GPU_KERNEL_IMPL( nonSU2, D1, -1.0 )

GPU_KERNEL_IMPL_OVERSIZE( SU2, P, 1.0 )
GPU_KERNEL_IMPL_OVERSIZE( SU2, C, 1.0 )
GPU_KERNEL_IMPL_OVERSIZE( SU2, D1, -2.0 )
GPU_KERNEL_IMPL_OVERSIZE( SU2, D2, 1.0 )
GPU_KERNEL_IMPL_OVERSIZE( SU2, D3, 1.0 )
GPU_KERNEL_IMPL_OVERSIZE( nonSU2, P, 0.5 )
GPU_KERNEL_IMPL_OVERSIZE( nonSU2, C, 1.0 )
GPU_KERNEL_IMPL_OVERSIZE( nonSU2, D1, -1.0 )
