/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../misc/cuda_error.h"
#include "../misc/mpi_log.h"
#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#else
typedef struct cufftHandle cufftHandle;
typedef struct cudaStream_t cudaStream_t;
typedef struct cuDoubleComplex cuDoubleComplex;
#endif

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

typedef struct cuda_prop_descr_t cuda_prop_descr_t;
typedef struct cuda_proj_descr_t cuda_proj_descr_t;

struct cuda_prop_descr_t {
    bool prop_has_gpu_setup;
    bool prop_memory_set_local;
    bool prop_memory_set_vertex;
    bool prop_preload_full_gf;

    int32_t _tasks_prop[32];
    int32_t _off_prop[32];
    int32_t _tasks_cpu;
    int32_t _off_cpu;

    int nthread[3];
    int nblock[3];
    size_t nk_per_thread;
    size_t n_kblocks;
    cufftHandle* gpu_fft;
    size_t worksize_fft[32];
    long long int dims[3];
    bool dims_set;

    index_t* mi_to_R[32];
    index_t* c_to_f_q[32];
    index_t* nkdev[32];
    cuDoubleComplex* GF_fft[32];
    cuDoubleComplex* GF_fft_conj[32];
    cuDoubleComplex* buf_GG[32];
    cuDoubleComplex* buf_res[32];
    cuDoubleComplex* fft_workspace[32];
};

#ifndef __cplusplus
static const cuda_prop_descr_t cuda_prop_defaults = {
    .prop_has_gpu_setup = false,
    .prop_memory_set_local = false,
    .prop_memory_set_vertex = false,
    .prop_preload_full_gf = true,

    ._tasks_prop = {0},
    ._off_prop = {0},
    ._tasks_cpu = 0,
    ._off_cpu = 0,

    .nthread = {512,1,1},
    .nblock = {18,1,1},
    .nk_per_thread = 1,
    .n_kblocks = 1,
    .gpu_fft = NULL,
    .worksize_fft = {0},
    .dims = {1,1,0},
    .dims_set = false,

    .mi_to_R = {0},
    .c_to_f_q = {0},
    .nkdev = {0},
    .GF_fft = {0},
    .GF_fft_conj = {0},
    .buf_GG = {0},
    .buf_res = {0},
    .fft_workspace = {0},
};
#endif // __cplusplus
cuda_prop_descr_t cuda_prop_defaults_CPP( void );

struct cuda_proj_descr_t {
    bool proj_has_gpu_setup;
    int _threadkpts_proj[32];
    int _qstart_proj[32];
    int nthread[3];
    int nblock[3];

    cuDoubleComplex* prefacs[32];
    index_t* rs_to[32];
    index_t* rs_from[32];
    index_t* mom[32];
    cuDoubleComplex* buf_ondev[32];
    cuDoubleComplex* buf_res0[32];
    cuDoubleComplex* buf_res1[32];
    cuDoubleComplex* buf_resh[32];
};

#ifndef __cplusplus
static const cuda_proj_descr_t cuda_proj_defaults = {
    .proj_has_gpu_setup = false,
    ._threadkpts_proj = {0},
    ._qstart_proj = {0},
    .nthread = {18,18,1},
    .nblock = {18,18,1},

    .prefacs = {0},
    .rs_to = {0},
    .rs_from = {0},
    .mom = {0},
    .buf_ondev = {0},
    .buf_res0 = {0},
    .buf_res1 = {0},
    .buf_resh = {0},
};
#endif // __cplusplus
cuda_proj_descr_t cuda_proj_defaults_CPP( void );


void cuda_reset_prop_descr( cuda_prop_descr_t* cp, index_t tasks_spin,
        index_t ntasks_dist, double offload/*=0.0*/ );
void cuda_setup_prop_descr( cuda_prop_descr_t* cp, index_t task_spin,
        index_t tasks_dist, const index_t* nkpts, double offload_cpu/*=0.1*/ );
void cuda_delete_prop_descr( cuda_prop_descr_t* cp );

// globals
typedef struct {
    int tu_deviceid[32];
    int tu_num_of_devices;
    bool tu_has_gpu_setup;
    cudaStream_t* tu_gpu_stream;
    cudaStream_t* tu_gpu_downstream;
} tu_gpu_info_t;
tu_gpu_info_t tu_gpu_info( void );

void cuda_setup( void );
void cuda_setup_projections( cuda_proj_descr_t* cp, index_t ntasks, index_t my_nk );
void cuda_deconstruct( void );

#ifdef __cplusplus
}
#endif
