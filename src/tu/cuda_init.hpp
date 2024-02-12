/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once
#include "../diverge_common.h"
#include "../misc/cuda_error.h"
#include "../misc/mpi_log.h"

#include <cassert>
#include <array>

using std::array;

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#else
typedef void cufftHandle;
typedef void cudaStream_t;
typedef void cuDoubleComplex;
#endif

struct cuda_prop_descr_t {
    bool prop_has_gpu_setup = false;
    bool prop_memory_set_local = false;
    bool prop_memory_set_vertex = false;
    bool prop_preload_full_gf = true;
    int _tasks_prop[32] = {0};
    int _off_prop[32] = {0};
    int _tasks_cpu = 0;
    int _off_cpu = 0;
    array<int,3> nthread {512,1,1};
    array<int,3> nblock {18,1,1};
    size_t nk_per_thread = 1;
    size_t n_kblocks = 1;
    cufftHandle* gpu_fft;
    size_t worksize_fft[32] = {0};
    long long int dims[3] = {1,1,0};
    bool dims_set = false;

    index_t* mi_to_R[32];
    index_t* c_to_f_q[32];
    index_t* nkdev[32];
    cuDoubleComplex* GF_fft[32];
    cuDoubleComplex* GF_fft_conj[32];
    cuDoubleComplex* buf_GG[32];
    cuDoubleComplex* buf_res[32];
    cuDoubleComplex* fft_workspace[32];
};

struct cuda_proj_descr_t {
    bool proj_has_gpu_setup = false;
    int _threadkpts_proj[32] = {0};
    int _qstart_proj[32] = {0};
    array<int,3> nthread {18,18,1};
    array<int,3> nblock {18,18,1};

    cuDoubleComplex* prefacs[32];
    index_t* rs_to[32];
    index_t* rs_from[32];
    index_t* mom[32];
    cuDoubleComplex* buf_ondev[32];
    cuDoubleComplex* buf_res0[32];
    cuDoubleComplex* buf_res1[32];
    cuDoubleComplex* buf_resh[32];
};


void cuda_reset_prop_descr(cuda_prop_descr_t* cp,index_t tasks_spin, 
                     index_t ntasks_dist, double offload = 0.0);

void cuda_setup_prop_descr(cuda_prop_descr_t* cp,  index_t task_spin,
                index_t tasks_dist, const index_t* nkpts,double offload_cpu = 0.1);
void cuda_delete_prop_descr(cuda_prop_descr_t* cp);

// globals
extern bool tu_has_gpu_setup;
extern int tu_num_of_devices;
extern int tu_deviceid[32];
extern cudaStream_t* tu_gpu_stream;
extern cudaStream_t* tu_gpu_downstream;

void cuda_setup();
void cuda_setup_projections(cuda_proj_descr_t* cp,index_t ntasks, index_t my_nk);
void cuda_deconstruct();
