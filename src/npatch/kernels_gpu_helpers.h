/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "kernels_gpu.h"
#include "../misc/mpi_functions.h"
#include "../misc/cuda_error.h"

#include <pthread.h>
#include <cuComplex.h>

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 1024
#endif

typedef cuDoubleComplex cu_cdoub_t;
typedef cuFloatComplex cu_cfloat_t;

#ifdef USE_GF_FLOATS
typedef cu_cfloat_t cu_gf_complex_t;
#define cu_gf_complex_mul cuCmulf
#define cu_gf_complex_div cuCdivf
#define cu_gf_complex_add cuCaddf
#define cu_gf_complex_sub cuCsubf
#define cu_gf_complex_conj cuConjf
#else
typedef cu_cdoub_t cu_gf_complex_t;
#define cu_gf_complex_mul cuCmul
#define cu_gf_complex_div cuCdiv
#define cu_gf_complex_add cuCadd
#define cu_gf_complex_sub cuCsub
#define cu_gf_complex_conj cuConj
#endif

#ifdef CUDA_CIRCUMVENT_CONSTEXPR
#define CUDA_CONSTEXPR(x) (x)
#else
#define CUDA_CONSTEXPR(x) constexpr(x)
#endif

#ifdef CUDA_DEFINE_ATOMIC_ADD
__device__ inline double atomicAdd_(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else // CUDA_DEFINE_ATOMIC_ADD
#define atomicAdd_ atomicAdd
#endif // CUDA_DEFINE_ATOMIC_ADD

static inline __device__ index_t mem_offset() {
    index_t blockId = blockIdx.x + blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;
    index_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
        threadIdx.x + threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;
    return threadId;
}

__device__ inline void cu_complex_atomic_add( cu_cdoub_t* address, const cu_cdoub_t* val ) {
    atomicAdd_( (double*)(address)+0, *((const double*)((val))+0) );
    atomicAdd_( (double*)(address)+1, *((const double*)((val))+1) );
}

__device__ inline void cu_complex_scale( cu_cdoub_t* x, double scale ) {
    *((double*)(x)+0) *= scale;
    *((double*)(x)+1) *= scale;
}

__device__ inline void cu_complex_atomic_add( cu_cdoub_t* address, const cu_cfloat_t* val ) {
    atomicAdd_( (double*)(address)+0, *((const float*)((val))+0) );
    atomicAdd_( (double*)(address)+1, *((const float*)((val))+1) );
}

__device__ inline void cu_complex_scale( cu_cfloat_t* x, float scale ) {
    *((float*)(x)+0) *= scale;
    *((float*)(x)+1) *= scale;
}

