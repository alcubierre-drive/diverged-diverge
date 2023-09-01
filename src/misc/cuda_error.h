#pragma once

#include "mpi_log.h"
#include "mpi_functions.h"

#ifndef DIVERGE_NO_ABORT_ON_GPU_ERROR
#define _GPU_ABORT_ON_ERR 1
#else
#define _GPU_ABORT_ON_ERR 0
#endif

#ifdef USE_CUDA

#define ERROR_RAISE( status, info ) { \
    mpi_err_printf( "CUDA: %s\n", info ); \
    if (_GPU_ABORT_ON_ERR) \
        diverge_mpi_exit( status ); \
}

#define CUSOLVER_CASE_DO( ENUM ) \
    case ENUM: \
        ERROR_RAISE( cusolver_status, #ENUM ); \
        break;
#define CUSOLVER_CHECK( call ) { \
    cusolverStatus_t cusolver_status = call; \
    switch(cusolver_status) { \
        CUSOLVER_CASE_DO( CUSOLVER_STATUS_NOT_INITIALIZED ) \
        CUSOLVER_CASE_DO( CUSOLVER_STATUS_ALLOC_FAILED ) \
        CUSOLVER_CASE_DO( CUSOLVER_STATUS_INVALID_VALUE ) \
        CUSOLVER_CASE_DO( CUSOLVER_STATUS_ARCH_MISMATCH ) \
        CUSOLVER_CASE_DO( CUSOLVER_STATUS_EXECUTION_FAILED ) \
        CUSOLVER_CASE_DO( CUSOLVER_STATUS_INTERNAL_ERROR ) \
        CUSOLVER_CASE_DO( CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED ) \
        default: break; \
    } \
}

#define CUBLAS_CASE_DO( ENUM ) \
    case ENUM: \
        ERROR_RAISE( cublas_status, #ENUM ); \
        break;
#define CUBLAS_CHECK( call ) { \
    cublasStatus_t cublas_status = call; \
    switch(cublas_status) { \
        CUBLAS_CASE_DO( CUBLAS_STATUS_ALLOC_FAILED ) \
        CUBLAS_CASE_DO( CUBLAS_STATUS_ARCH_MISMATCH ) \
        CUBLAS_CASE_DO( CUBLAS_STATUS_EXECUTION_FAILED ) \
        CUBLAS_CASE_DO( CUBLAS_STATUS_INTERNAL_ERROR ) \
        CUBLAS_CASE_DO( CUBLAS_STATUS_INVALID_VALUE ) \
        CUBLAS_CASE_DO( CUBLAS_STATUS_LICENSE_ERROR ) \
        CUBLAS_CASE_DO( CUBLAS_STATUS_MAPPING_ERROR ) \
        CUBLAS_CASE_DO( CUBLAS_STATUS_NOT_INITIALIZED ) \
        CUBLAS_CASE_DO( CUBLAS_STATUS_NOT_SUPPORTED ) \
        default: break; \
    } \
}

#define CUFFT_CASE_DO( ENUM ) \
    case ENUM: \
        ERROR_RAISE( cufft_status, #ENUM ); \
        break;
#define CUFFT_CHECK( call ) { \
    cufftResult cufft_status = call; \
    switch(cufft_status) { \
        CUFFT_CASE_DO( CUFFT_INVALID_PLAN ) \
        CUFFT_CASE_DO( CUFFT_ALLOC_FAILED ) \
        CUFFT_CASE_DO( CUFFT_INVALID_TYPE ) \
        CUFFT_CASE_DO( CUFFT_INVALID_VALUE ) \
        CUFFT_CASE_DO( CUFFT_INTERNAL_ERROR ) \
        CUFFT_CASE_DO( CUFFT_EXEC_FAILED ) \
        CUFFT_CASE_DO( CUFFT_SETUP_FAILED ) \
        CUFFT_CASE_DO( CUFFT_INVALID_SIZE ) \
        CUFFT_CASE_DO( CUFFT_UNALIGNED_DATA ) \
        default: break; \
    } \
}

#define CUDA_CHECK( call ) { \
    cudaError_t cuda_status = call; \
    if (cuda_status) ERROR_RAISE( cuda_status, cudaGetErrorName( cuda_status ) ); \
}

#else // USE_CUDA

#define CUDA_CHECK( call ) { call; }
#define CUBLAS_CHECK( call ) { call; }
#define CUSOLVER_CHECK( call ) { call; }
#define CUFFT_CHECK( call ) { call; }

#endif // USE_CUDA
