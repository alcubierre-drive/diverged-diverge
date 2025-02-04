/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_eigen.h"
#include "batched_eigen_internals.h"
#include "gemm.h"

// compile the batched zheevd for EIGEN usage, selectable through USE_EIGEN3_CPU_EIGENSOLVER
#include "../diverge_Eigen3.hpp"
#include "mpi_functions.h"

template<index_t N> using MatNcd = Eigen::Matrix<complex128_t,N,N>;
template<index_t N> using cMatNcd = Eigen::Matrix<complex128_t,N,N,Eigen::RowMajor>;
template<index_t N> using VecNd = Eigen::Matrix<double,N,1>;

template<index_t dim>
void batched_zheevd_templ( const complex128_t* H, complex128_t* U, double* E, index_t num ) {
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        Eigen::SelfAdjointEigenSolver<MatNcd<dim>> sol;
        #pragma omp for
        for (index_t i=0; i<num; ++i) {
            sol.compute( Eigen::Map<const cMatNcd<dim>>( H + i*dim*dim ) );
            Eigen::Map<MatNcd<dim>>( U + i*dim*dim ) = sol.eigenvectors();
            Eigen::Map<VecNd<dim>>( E + i*dim ) = sol.eigenvalues();
        }
    }
}

void batched_zheevd_any( const complex128_t* H, complex128_t* U, double* E, index_t num, index_t dim ) {
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        Eigen::SelfAdjointEigenSolver<MatXcd> sol;
        #pragma omp for
        for (index_t i=0; i<num; ++i) {
            sol.compute( Eigen::Map<const cMatXcd>( H + i*dim*dim, dim, dim ) );
            Eigen::Map<MatXcd>( U + i*dim*dim, dim, dim ) = sol.eigenvectors();
            Eigen::Map<VecXd>( E + i*dim, dim ) = sol.eigenvalues();
        }
    }
}

void batched_zheevd( const complex128_t* H, complex128_t* U, double* E, index_t num, index_t dim ) {
    switch (dim) {
        case 1:  batched_zheevd_templ<1> ( H, U, E, num ); break;
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 1
        case 2:  batched_zheevd_templ<2> ( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 2
        case 3:  batched_zheevd_templ<3> ( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 3
        case 4:  batched_zheevd_templ<4> ( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 4
        case 5:  batched_zheevd_templ<5> ( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 5
        case 6:  batched_zheevd_templ<6> ( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 6
        case 7:  batched_zheevd_templ<7> ( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 7
        case 8:  batched_zheevd_templ<8> ( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 8
        case 9:  batched_zheevd_templ<9> ( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 9
        case 10: batched_zheevd_templ<10>( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 10
        case 11: batched_zheevd_templ<11>( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 11
        case 12: batched_zheevd_templ<12>( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 12
        case 13: batched_zheevd_templ<13>( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 13
        case 14: batched_zheevd_templ<14>( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 14
        case 15: batched_zheevd_templ<15>( H, U, E, num ); break;
#endif
#if USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE > 15
        case 16: batched_zheevd_templ<16>( H, U, E, num ); break;
#endif
        default: batched_zheevd_any( H, U, E, num, dim ); break;
    }
}

#ifdef USE_LAPACKE

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

void* batched_eigen_cpu( void* data ) {
    batched_eigen_t* cfg = (batched_eigen_t*)data;

    index_t n = cfg->n;
    complex128_t lwork_query;
    double lrwork_query;
    int liwork_query;

    complex128_t* H_test = (complex128_t*)malloc(sizeof(complex128_t)*n*n);
    double* E_test = (double*)malloc(sizeof(double)*n);
    LAPACKE_zheevd_work( LAPACK_COL_MAJOR, 'V', 'U', n, H_test, n, E_test,
            &lwork_query, -1, &lrwork_query, -1, &liwork_query, -1 );
    free(E_test);

    index_t lwork_query_i = lroundl(lwork_query.real());
    complex128_t* work = (complex128_t*)malloc(sizeof(complex128_t)*lwork_query_i);
    double* rwork = (double*)malloc(sizeof(double)*lrwork_query);
    int* iwork = (int*)malloc(sizeof(int)*liwork_query);

    double* E = (double*)malloc(sizeof(double)*n);

    for (index_t i=0; i<cfg->N; ++i) {
        transpose_outofplace( H_test, cfg->H + n*n*i, n );
        LAPACKE_zheevd_work( LAPACK_COL_MAJOR, 'V', 'U', n, H_test, n, E, work,
                lwork_query_i, rwork, lrwork_query, iwork, liwork_query );
        memcpy( cfg->H + n*n*i, H_test, sizeof(complex128_t)*n*n );
        memcpy( cfg->E + n*i, E, sizeof(double)*n );
    }
    free(H_test);

    free(work);
    free(rwork);
    free(iwork);
    free(E);

    return NULL;
}

#else // USE_LAPACKE

#include "../diverge_Eigen3.hpp"

static void batched_eigen_cpu_helper( complex128_t* pH, double* pE, index_t n ) {
    Map<cMatXcd> cH( pH, n, n );
    Eigen::SelfAdjointEigenSolver<MatXcd> sol( cH );

    Map<MatXcd> H( pH, n, n );
    H = sol.eigenvectors();
    Map<VecXd> E( pE, n );
    E = sol.eigenvalues();
}

void* batched_eigen_cpu( void* data ) {
    batched_eigen_t* cfg = (batched_eigen_t*)data;

    index_t n = cfg->n;
    double* E = (double*)malloc(sizeof(double)*n);

    for (index_t i=0; i<cfg->N; ++i) {
        complex128_t* H = cfg->H + n*n*i;
        batched_eigen_cpu_helper( H, E, n );
        memcpy( cfg->E + n*i, E, sizeof(double)*n );
    }

    free(E);

    return NULL;
}

#endif // USE_LAPACKE
