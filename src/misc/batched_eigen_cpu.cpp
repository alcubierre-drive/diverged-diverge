/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_eigen.h"
#include "batched_eigen_internals.h"

#ifdef USE_LAPACKE

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

static inline void transpose_memcpy( complex128_t* out, const complex128_t* in, index_t N ) {
    for (index_t i=0; i<N; ++i)
    for (index_t j=0; j<N; ++j)
        out[IDX2(i,j,N)] = in[IDX2(j,i,N)];
}

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
        transpose_memcpy( H_test, cfg->H + n*n*i, n );
        LAPACKE_zheevd_work( LAPACK_COL_MAJOR, 'V', 'U', n, H_test, n, E, work,
                lwork_query_i, rwork, lrwork_query, iwork, liwork_query );
        memcpy( cfg->H + n*n*i, H_test, sizeof(complex128_t)*n*n );
        if (cfg->Er == NULL) {
            for (index_t j=0; j<n; ++j)
                cfg->E[i*n+j] = E[j];
        } else {
            for (index_t j=0; j<n; ++j)
                cfg->Er[i*n+j] = E[j];
        }
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
    typedef Matrix<complex128_t,-1,-1,Eigen::RowMajor> cMatXcd;
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
        if (cfg->Er == NULL) {
            for (index_t j=0; j<n; ++j)
                cfg->E[i*n+j] = E[j];
        } else {
            for (index_t j=0; j<n; ++j)
                cfg->Er[i*n+j] = E[j];
        }
    }

    free(E);

    return NULL;
}

#endif // USE_LAPACKE

