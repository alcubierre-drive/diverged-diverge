/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_gemms.h"
#include "mpi_functions.h"
#include <vector>

using std::vector;

#ifdef USE_MKL
#include <mkl.h>
#ifdef MKL_FORCE_AVX_KERNEL_ON_EPYC
extern "C" { int mkl_serv_intel_cpu_true() { return 1; } }
#endif
#else // USE_MKL
#include <cblas.h>
#endif // USE_MKL

#if defined(USE_LAPACKE) && !defined(USE_MKL)
#include <lapacke.h>
#endif // LAPACKE

#ifndef USE_NO_BLAS_VERTEX_LOOP

#ifdef USE_LAPACKE
#define BATCHED_GEMMS_ZLACPY_LAPACKE
#else // USE_LAPACKE
#undef BATCHED_GEMMS_ZLACPY_LAPACKE
#endif // USE_LAPACKE
#define BATCHED_GEMMS_ZLACPY
static void batched_gemms_zlacpy( index_t N, const complex128_t* A, complex128_t* B, index_t strideB );

#else // USE_NO_BLAS_VERTEX_LOOP

#undef BATCHED_GEMMS_ZLACPY

#endif // USE_NO_BLAS_VERTEX_LOOP

#include <omp.h>

#ifndef USE_CUDA

void batched_gemm_cublas_w_buffer_clear( void ) {}
void batched_gemm_cublas_clear( void ) {}
void batched_gemm_cublas_init( index_t N ) { (void)N; }

#ifdef USE_EIGEN_BATCHED_GEMM
#include "../diverge_Eigen3.hpp"
template<int dim>
void batched_gemm_overlapping_helper( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    typedef Matrix<complex128_t,dim,dim> Mat_cd;
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        Mat_cd mat_buf = Mat_cd::Zero(N,N);
        #pragma omp for
        for (index_t i=0; i<num; ++i) {
            const complex128_t *A = Astart + N*N*i,
                               *B = Bstart + N*N*i;
            const Map<Mat_cd> Amap(const_cast<complex128_t*>(A), N, N);
            const Map<Mat_cd> Bmap(const_cast<complex128_t*>(B), N, N);
            complex128_t *C = Cstart + N*N*i;
            Map<Mat_cd> Cmap(C, N, N);
            mat_buf = Cmap;
            mat_buf = Bmap * Amap * alpha + mat_buf * beta;
            Cmap = mat_buf;
        }
    }
}

void batched_gemm_overlapping( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    switch (N) {
        #define make_case( dim ) \
        case dim: batched_gemm_overlapping_helper<dim>( Astart, Bstart, Cstart, N, alpha, beta, num ); break;
        make_case( 1 )
        make_case( 2 )
        make_case( 3 )
        make_case( 4 )
        make_case( 5 )
        make_case( 6 )
        make_case( 7 )
        make_case( 8 )
        make_case( 9 )
        make_case( 10 )
        make_case( 11 )
        make_case( 12 )
        make_case( 13 )
        make_case( 14 )
        make_case( 15 )
        make_case( 16 )
        make_case( 17 )
        make_case( 18 )
        make_case( 19 )
        make_case( 20 )
        make_case( 21 )
        make_case( 22 )
        make_case( 23 )
        make_case( 24 )
        make_case( 25 )
        make_case( 26 )
        make_case( 27 )
        make_case( 28 )
        make_case( 29 )
        make_case( 30 )
        make_case( 31 )
        make_case( 32 )
        case -1:
        default: batched_gemm_overlapping_helper<-1>( Astart, Bstart, Cstart, N, alpha, beta, num ); break;
    }
}
void batched_gemm( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    batched_gemm_overlapping( Astart, Bstart, Cstart, N, alpha, beta, num );
}

#else // USE_EIGEN_BATCHED_GEMM

#ifdef DIVERGE_GRID_PARALLEL_BATCHED_GEMM

void batched_gemm_overlapping( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    int nthreads = diverge_omp_num_threads();
    void* cmemory = calloc(nthreads*N*N,sizeof(complex128_t));
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<num; ++i) {
        complex128_t* buf = (complex128_t*)cmemory + N*N*omp_get_thread_num();
        memcpy( buf, Cstart + N*N*i, sizeof(complex128_t)*N*N );
        cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha,
                Bstart + N*N*i, N, Astart + N*N*i, N, &beta, buf, N );
        memcpy( Cstart + N*N*i, buf, sizeof(complex128_t)*N*N );
    }
    free(cmemory);
}
void batched_gemm( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<num; ++i) {
        cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha,
                Bstart + N*N*i, N, Astart + N*N*i, N, &beta, Cstart + N*N*i, N );
    }
}

#else // DIVERGE_GRID_PARALLEL_BATCHED_GEMM

void batched_gemm_overlapping( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    void* cmemory = calloc(N*N,sizeof(complex128_t));
    for (index_t i=0; i<num; ++i) {
        complex128_t* buf = (complex128_t*)cmemory;
        memcpy( buf, Cstart + N*N*i, sizeof(complex128_t)*N*N );
        cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha,
                Bstart + N*N*i, N, Astart + N*N*i, N, &beta, buf, N );
        memcpy( Cstart + N*N*i, buf, sizeof(complex128_t)*N*N );
    }
    free(cmemory);
}
void batched_gemm( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    for (index_t i=0; i<num; ++i) {
        cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha,
                Bstart + N*N*i, N, Astart + N*N*i, N, &beta, Cstart + N*N*i, N );
    }
}

#endif // DIVERGE_GRID_PARALLEL_BATCHED_GEMM

#endif // USE_EIGEN_BATCHED_GEMM


void batched_gemm_with_buffers( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, char* GPUbuffer[32][3], index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    (void)GPUbuffer;
    batched_gemm(Astart, Bstart, Cstart, N, alpha, beta, num );
}

#endif // USE_CUDA




#ifndef USE_NO_BLAS_VERTEX_LOOP

void batched_gemm_vertex_loop( complex128_t* vertex, const complex128_t* loop, index_t qsize,
        index_t nk, index_t nb ) {
    index_t N = nb*nb;
    complex128_t alpha = 1.0;
    complex128_t beta = 0.0;
    complex128_t* buffers = (complex128_t*)calloc(N*N*diverge_omp_num_threads(), sizeof(complex128_t));
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<qsize; ++q)
    for (index_t k=0; k<nk; ++k)
    for (index_t kp=0; kp<nk; ++kp) {

        complex128_t* vmatrix = vertex + q*nk*N*nk*N + k*N*nk*N + kp*N;
        const complex128_t* lmatrix = loop + q*nk*N*N + kp*N*N;
        complex128_t* buffer = buffers + N*N*omp_get_thread_num();

        cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha,
            vmatrix, N*nk, lmatrix, N, &beta, buffer, N );
        batched_gemms_zlacpy( N, buffer, vmatrix, N*nk );
    }
    free(buffers);
}

void batched_gemm_loop_vertex( const complex128_t* loop, complex128_t* vertex, index_t qsize,
        index_t nk, index_t nb ) {
    index_t N = nb*nb;
    complex128_t alpha = 1.0;
    complex128_t beta = 0.0;
    complex128_t* buffers = (complex128_t*)calloc(N*N*diverge_omp_num_threads(), sizeof(complex128_t));
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<qsize; ++q)
    for (index_t k=0; k<nk; ++k)
    for (index_t kp=0; kp<nk; ++kp) {

        complex128_t* vmatrix = vertex + q*nk*N*nk*N + k*N*nk*N + kp*N;
        const complex128_t* lmatrix = loop + q*nk*N*N + k*N*N;
        complex128_t* buffer = buffers + N*N*omp_get_thread_num();

        cblas_zgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha,
                lmatrix, N, vmatrix, N*nk, &beta, buffer, N );
        batched_gemms_zlacpy( N, buffer, vmatrix, N*nk );
    }
    free(buffers);
}

#else // USE_NO_BLAS_VERTEX_LOOP

template<index_t N_>
void _batched_gemm_vertex_loop( complex128_t* vertex, const complex128_t* loop, index_t qsize,
        index_t nk, index_t nb ) {
    index_t N = nb*nb;
    complex128_t* buffers = (complex128_t*)calloc(N*N*diverge_omp_num_threads(),sizeof(complex128_t));
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<qsize; ++q)
    for (index_t k=0; k<nk; ++k)
    for (index_t kp=0; kp<nk; ++kp) {
        if constexpr(N_==-1) {
            complex128_t* buf = buffers + omp_get_thread_num()*N*N;
            for (index_t OL=0; OL<N; ++OL) for (index_t OR=0; OR<N; ++OR) {
                buf[OL*N+OR] = 0.0;
                for (index_t OO=0; OO<N; ++OO) {
                    buf[OL*N+OR] += vertex[q*nk*N*nk*N + k*N*nk*N + OL*nk*N + kp*N + OO] *
                        loop[q*nk*N*N + kp*N*N + OO*N + OR];
                }
            }
            for (index_t OL=0; OL<N; ++OL) for (index_t OR=0; OR<N; ++OR) {
                vertex[q*nk*N*nk*N + k*N*nk*N + OL*nk*N + kp*N + OR] = buf[OL*N+OR];
            }
        } else {
            complex128_t* buf = buffers + omp_get_thread_num()*N_*N_;
            for (index_t OL=0; OL<N_; ++OL) for (index_t OR=0; OR<N_; ++OR) {
                buf[OL*N_+OR] = 0.0;
                for (index_t OO=0; OO<N_; ++OO) {
                    buf[OL*N_+OR] += vertex[q*nk*N_*nk*N_ + k*N_*nk*N_ + OL*nk*N_ + kp*N_ + OO] *
                        loop[q*nk*N_*N_ + kp*N_*N_ + OO*N_ + OR];
                }
            }
            for (index_t OL=0; OL<N_; ++OL) for (index_t OR=0; OR<N_; ++OR) {
                vertex[q*nk*N_*nk*N_ + k*N_*nk*N_ + OL*nk*N_ + kp*N_ + OR] = buf[OL*N_+OR];
            }
        }
    }
    free(buffers);
}

template<index_t N_>
void _batched_gemm_loop_vertex( const complex128_t* loop, complex128_t* vertex, index_t qsize,
        index_t nk, index_t nb ) {
    index_t N = nb*nb;
    complex128_t* buffers = (complex128_t*)calloc(N*N*diverge_omp_num_threads(),sizeof(complex128_t));
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<qsize; ++q)
    for (index_t k=0; k<nk; ++k)
    for (index_t kp=0; kp<nk; ++kp) {
        if constexpr(N_==-1) {
            complex128_t* buf = buffers + omp_get_thread_num()*N*N;
            for (index_t OL=0; OL<N; ++OL) for (index_t OR=0; OR<N; ++OR) {
                buf[OL*N+OR] = 0.0;
                for (index_t OO=0; OO<N; ++OO) {
                    buf[OL*N+OR] +=  loop[q*nk*N*N + k*N*N + OL*N + OO] *
                        vertex[q*nk*N*nk*N + k*N*nk*N + OO*nk*N + kp*N + OR];
                }
            }
            for (index_t OL=0; OL<N; ++OL) for (index_t OR=0; OR<N; ++OR) {
                vertex[q*nk*N*nk*N + k*N*nk*N + OL*nk*N + kp*N + OR] = buf[OL*N+OR];
            }
        } else {
            complex128_t* buf = buffers + omp_get_thread_num()*N_*N_;
            for (index_t OL=0; OL<N_; ++OL) for (index_t OR=0; OR<N_; ++OR) {
                buf[OL*N_+OR] = 0.0;
                for (index_t OO=0; OO<N_; ++OO) {
                    buf[OL*N_+OR] +=  loop[q*nk*N_*N_ + k*N_*N_ + OL*N_ + OO] *
                        vertex[q*nk*N_*nk*N_ + k*N_*nk*N_ + OO*nk*N_ + kp*N_ + OR];
                }
            }
            for (index_t OL=0; OL<N_; ++OL) for (index_t OR=0; OR<N_; ++OR) {
                vertex[q*nk*N_*nk*N_ + k*N_*nk*N_ + OL*nk*N_ + kp*N_ + OR] = buf[OL*N_+OR];
            }
        }
    }
    free(buffers);
}

void batched_gemm_vertex_loop( complex128_t* vertex, const complex128_t* loop, index_t qsize,
        index_t nk, index_t nb ) {
    if (nb == 1)
        _batched_gemm_vertex_loop<1>( vertex, loop, qsize, nk, nb );
    else if (nb == 2)
        _batched_gemm_vertex_loop<4>( vertex, loop, qsize, nk, nb );
    else if (nb == 3)
        _batched_gemm_vertex_loop<9>( vertex, loop, qsize, nk, nb );
    else if (nb == 4)
        _batched_gemm_vertex_loop<16>( vertex, loop, qsize, nk, nb );
    else
        _batched_gemm_vertex_loop<-1>( vertex, loop, qsize, nk, nb );
}
void batched_gemm_loop_vertex( const complex128_t* loop, complex128_t* vertex, index_t qsize,
        index_t nk, index_t nb ) {
    if (nb == 1)
        _batched_gemm_loop_vertex<1>( loop, vertex, qsize, nk, nb );
    else if (nb == 2)
        _batched_gemm_loop_vertex<4>( loop, vertex, qsize, nk, nb );
    else if (nb == 3)
        _batched_gemm_loop_vertex<9>( loop, vertex, qsize, nk, nb );
    else if (nb == 4)
        _batched_gemm_loop_vertex<16>( loop, vertex, qsize, nk, nb );
    else
        _batched_gemm_loop_vertex<-1>( loop, vertex, qsize, nk, nb );
}

#endif // USE_NO_BLAS_VERTEX_LOOP

#ifdef BATCHED_GEMMS_ZLACPY
#ifdef BATCHED_GEMMS_ZLACPY_LAPACKE
static void batched_gemms_zlacpy( index_t N, const complex128_t* A, complex128_t* B, index_t strideB ) {
    LAPACKE_zlacpy( LAPACK_ROW_MAJOR, 'F', N, N, A, N, B, strideB );
}
#else // BATCHED_GEMMS_ZLACPY_LAPACKE
static void batched_gemms_zlacpy( index_t N, const complex128_t* A, complex128_t* B, index_t strideB ) {
    for (index_t i=0; i<N; ++i)
    for (index_t j=0; j<N; ++j)
        B[i*strideB+j] = A[i*N+j];
}
#endif // BATCHED_GEMMS_ZLACPY_LAPACKE
#endif

#include "../diverge_Eigen3.hpp"

template<int dim, bool Aconj, bool Bconj>
void batched_gemm_N_ll_num_helper( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t N, complex128_t alpha, complex128_t beta, index_t num ) {

    typedef Matrix<complex128_t,dim,dim> Mat_cd;

    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        Mat_cd mat_buf = Mat_cd::Zero(N,N);
        #pragma omp for
        for (index_t i=0; i<num; ++i) {
            Map<const Mat_cd> Amap( A + N*N*i, N, N);
            Map<const Mat_cd> Bmap( B + N*N*i, N, N);
            Map<Mat_cd> Cmap( C + N*N*i, N, N);

            mat_buf = Cmap;
            if constexpr(Aconj && Bconj)
                mat_buf = Amap.adjoint() * Bmap.adjoint() * alpha + mat_buf * beta;
            else if constexpr(Aconj && !Bconj)
                mat_buf = Amap.adjoint() * Bmap * alpha + mat_buf * beta;
            else if constexpr(!Aconj && Bconj)
                mat_buf = Amap * Bmap.adjoint() * alpha + mat_buf * beta;
            else
                mat_buf = Amap * Bmap * alpha + mat_buf * beta;
            Cmap = mat_buf;
        }
    }
}

void batched_gemm_N_ll_num( const complex128_t* A, const complex128_t* B, complex128_t* C,
        index_t N, complex128_t alpha, complex128_t beta, index_t num, bool Aconj, bool Bconj ) {
    switch (N) {
        #define make_case( dim ) \
        case dim: \
            if (!Aconj && !Bconj) \
                batched_gemm_N_ll_num_helper<dim,false,false>( A, B, C, N, alpha, beta, num ); \
            if (!Aconj && Bconj) \
                batched_gemm_N_ll_num_helper<dim,false,true>( A, B, C, N, alpha, beta, num ); \
            if (Aconj && !Bconj) \
                batched_gemm_N_ll_num_helper<dim,true,false>( A, B, C, N, alpha, beta, num ); \
            if (Aconj && Bconj) \
                batched_gemm_N_ll_num_helper<dim,true,true>( A, B, C, N, alpha, beta, num ); \
            break;
#ifndef BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS
#define BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS 4
#endif // BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS
        make_case( 1 )
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 2
        make_case( 2 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 3
        make_case( 3 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 4
        make_case( 4 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 5
        make_case( 5 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 6
        make_case( 6 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 7
        make_case( 7 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 8
        make_case( 8 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 9
        make_case( 9 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 10
        make_case( 10 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 11
        make_case( 11 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 12
        make_case( 12 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 13
        make_case( 13 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 14
        make_case( 14 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 15
        make_case( 15 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 16
        make_case( 16 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 17
        make_case( 17 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 18
        make_case( 18 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 19
        make_case( 19 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 20
        make_case( 20 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 21
        make_case( 21 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 22
        make_case( 22 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 23
        make_case( 23 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 24
        make_case( 24 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 25
        make_case( 25 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 26
        make_case( 26 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 27
        make_case( 27 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 28
        make_case( 28 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 29
        make_case( 29 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 30
        make_case( 30 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 31
        make_case( 31 )
        #endif
        #if BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS >= 32
        make_case( 32 )
        #endif
        default:
        make_case( -1 )
    }
}
