/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t bool64_t;
typedef struct { char str[128]; } str1024_t;

typedef struct {
    // ABI relevant
    int64_t   _MAX_N_ORBS;
    int64_t   _MAX_NAME_LENGTH;
    int64_t   _MAX_N_SYM;
    int64_t   _MAX_ORBS_PER_SITE;
    double    _DIVERGE_EPS_MESH;
    bool64_t  _USE_GF_FLOATS;
    int64_t   _BATCHED_EIGEN_NCHUNKS_AUTONUM;
    int64_t   _BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB;
    int64_t   _DIVERGE_FLOW_STEP_EIGCHAN_PITER_ARPACK;
    int64_t   _DIVERGE_MODEL_MAGIC_NUMBER;
    int64_t   _DIVERGE_MODEL_NBANDSTRUCTURE;
    bool64_t  _DIVERGE_QNAN_DISABLE_CLANG_OPTS;
    double    _GRID_FF_SHELL_DISTANCE;
    bool64_t  _USE_MKL;
    bool64_t  _USE_MPI;
    bool64_t  _USE_NO_LAPACKE;
    bool64_t  _SHARED_MEM_MPI_IMPL;
    bool64_t  _SHARED_MEM_POSIX_IMPL;

    // internal to the library, not ABI relevant, informational only
    bool64_t  _USE_CUDA;
    int64_t   _BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS;
    bool64_t  _CUDA_CIRCUMVENT_CONSTEXPR;
    bool64_t  _CUDA_DEFINE_ATOMIC_ADD;
    bool64_t  _DIVERGE_GRID_PARALLEL_BATCHED_GEMM;
    int64_t   _DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER;
    double    _DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL;
    bool64_t  _DIVERGE_LOG_COLORLESS;
    bool64_t  _DIVERGE_LOG_GRAYSCALE;
    bool64_t  _DIVERGE_LOG_NAMELESS;
    bool64_t  _DIVERGE_NO_ABORT_ON_GPU_ERROR;
    bool64_t  _DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT;
    int64_t   _DIVERGE_OPENBLAS_LIMIT_THREADS;
    bool64_t  _DIVERGE_USE_ZHEEVD;
    int64_t   _BATCHED_SVD_THREAD_CNT_MAX;
    int64_t   _DIVERGE_GETLINE_MAX;
    bool64_t  _DIVERGE_SKIP_TESTS;
    bool64_t  _DIVERGE_SKIP_TESTS_LINALG;
    bool64_t  _CT_NO_FORK;
    bool64_t  _DIVERGE_TU_INPLACE_PERMUTE;
    bool64_t  _DIVERGE_TU_GF_COARSIFY_FFT;
    bool64_t  _DIVERGE_UNIQUE_ORBFF_FIND_HASHTABLE;
    bool64_t  _TU_POST_USE_BATCHED_SVD_U;
    bool64_t  _TU_PROJ_TIMING;
    bool64_t  _TU_VERTEX_RECO_GPU_ZGEMM;
    int64_t   _LDOS_KERNEL_GPU_DEV_MAXNOS;
    int64_t   _LDOS_KERNEL_GPU_DEV_NTHREADS;
    bool64_t  _CLOG_MISSING;
    bool64_t  _LEAVE_EIGEN_BROKEN;
    int64_t   _DIVERGE_POST_MAGIC_NUMBER_GRID;
    int64_t   _DIVERGE_POST_MAGIC_NUMBER_NPATCH;
    int64_t   _MAX_THREADS_PER_BLOCK;
    bool64_t  _MKL_FORCE_AVX_KERNEL_ON_EPYC;
    bool64_t  _NDEBUG;
    bool64_t  _NDEBUG_SKIP_ERROR_LINES;
    double    _NPATCH_ALMOST_ZERO;
    int64_t   _NPATCH_PERIODIC_NORM_NG;
    bool64_t  _SKIP_EIGEN3_INCLUDE_PREFIX;
    bool64_t  _SKIP_FLOW_TIMING;
    int64_t   _DIVERGE_POST_MAGIC_NUMBER_TU;
    bool64_t  _USE_ARPACK_EIGENSOLVER;
    bool64_t  _USE_COMPRESSED_OUTPUT;
    bool64_t  _USE_DYNAMIC_RECONSTRUCT_SCHEDULE;
    int64_t   _MERGESORT_DEFAULT_TASK_SIZE;
    bool64_t  _USE_ALTERNATIVE_TIME;
    bool64_t  _USE_EIGEN_BATCHED_GEMM;
    bool64_t  _USE_EIGEN3_CPU_EIGENSOLVER;
    int64_t   _USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE;
    bool64_t  _USE_NO_BLAS_VERTEX_LOOP;
    bool64_t  _USE_SERIAL_FFTW;
    bool64_t  _USE_SHARED_MEM;
    bool64_t  _MEMFD_NOT_SUPPORTED;
    bool64_t  _USE_MPI_FFTW;
    int64_t   _PACK_MPI_MULTIPLIER;
    bool64_t  _TU_GPU_MPI_ALLOC_PRINT;
    int64_t   _CUDA_MEM_ALIGN;
    bool64_t  _MPI_VERBOSE_ALLOCATION_PRINT;
    bool64_t  _TRANSPOSE_MPI_FFT_BUFFERS;
    bool64_t  _DIVERGE_OLD_W90_READER;
    bool64_t  _DIVERGE_TU_PROJ_TO_R_VTREE;
    bool64_t  _XVTREE_ALIGNMENT_PRINT;

    // pads to a multiple of 512 bits. adjust when adding a field
    char      _PADDING[8];

    // char arrays last, so that they stay aligned.
    str1024_t _CUBLAS_BATCHED_GEMM_CALL;
    str1024_t _CUDA_VERSION_STR;
    str1024_t _DIVERGE_LOG_PREFIX;
} diverge_constants_t;

#ifndef CTYPESGEN
#ifdef __cplusplus
static_assert( sizeof(diverge_constants_t) % 64 == 0,
        "diverge_constants_t must be padded to a multiple of 512 bits" );
#else
_Static_assert( sizeof(diverge_constants_t) % 64 == 0,
        "diverge_constants_t must be padded to a multiple of 512 bits" );
#endif
#endif

diverge_constants_t diverge_constants( void );
void diverge_constants_print( int header );

#ifdef __cplusplus
}
#endif
