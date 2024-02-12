/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// C may overlap with A/B
void batched_gemm_overlapping( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num );
void batched_gemm( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num );
void batched_gemm_with_buffers( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, char* GPUbuffer[32][3], index_t N, complex128_t alpha, complex128_t beta, index_t num );

// specialized functions for vertex·loop products in grid code
void batched_gemm_vertex_loop( complex128_t* vertex, const complex128_t* loop, index_t qsize,
        index_t nk, index_t nb );
void batched_gemm_loop_vertex( const complex128_t* loop, complex128_t* vertex, index_t qsize,
        index_t nk, index_t nb );

void batched_gemm_cublas_init( index_t N );
void batched_gemm_cublas_clear( void );
void batched_gemm_cublas_w_buffer_clear( void );

// assumes col-major order of each individual matrix, A and B may overlap
void batched_gemm_N_ll_num( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t N, complex128_t alpha, complex128_t beta, index_t num, bool Aconj, bool Bconj );

#ifdef __cplusplus
}
#endif

