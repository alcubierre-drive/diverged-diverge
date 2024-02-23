/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "../diverge_Eigen3.hpp"

#include "eigen.h"

#include <algorithm>
#include <vector>
#include <cassert>

using std::vector;

#if defined(USE_LAPACKE) || defined(USE_MKL)
#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

void eigen_solve( const complex128_t* H, complex128_t* U, double* E, index_t N ) {
    assert(U != NULL);
    assert(E != NULL);
    assert(N > 0);
    if (H != NULL)
        if (U != H)
            memcpy( U, H, sizeof(complex128_t)*N*N );
    LAPACKE_zheev( LAPACK_ROW_MAJOR, 'V', 'U', N, (lapack_complex_double*)U, N, E );
}

void singular_value_decomp( const complex128_t* H, complex128_t* U, complex128_t* V, double* E, index_t N ) {
    assert(U != NULL);
    assert(V != NULL);
    assert(E != NULL);
    assert(N > 0);
    if (H != NULL)
        if (U != H)
            memcpy( U, H, sizeof(complex128_t)*N*N );
    double* scratch = (double*)calloc(N, sizeof(double));
    LAPACKE_zgesvd( LAPACK_ROW_MAJOR, 'O', 'A', N, N, U, N, E, NULL, N, V, N, scratch );
    free(scratch);
}

#else // LAPACKE/MKL

void eigen_solve( const complex128_t* H, complex128_t* U, double* E, index_t N ) {
    assert(U != NULL);
    assert(E != NULL);
    assert(N > 0);

    Map<MatXcd> mU(U, N, N);
    Map<VecXd> mE(E, N);

    if (!(H == NULL || H == U))
        mU = Map<MatXcd>(const_cast<complex128_t*>(H), N, N);

    Eigen::SelfAdjointEigenSolver<MatXcd> sol( mU.transpose() );
    mU = sol.eigenvectors().transpose();
    mE = sol.eigenvalues();
}

void singular_value_decomp( const complex128_t* H, complex128_t* U, complex128_t* V, double* E, index_t N ) {
    assert(U != NULL);
    assert(V != NULL);
    assert(E != NULL);
    assert(N > 0);

    Map<MatXcd> mU(U, N, N);
    Map<MatXcd> mV(V, N, N);
    Map<VecXd> mE(E, N);

    if (!(H == NULL || H == U))
        mU = Map<MatXcd>(const_cast<complex128_t*>(H), N, N);

    Eigen::JacobiSVD<MatXcd> SVD( mU.transpose(),
            Eigen::DecompositionOptions::ComputeFullU |
            Eigen::DecompositionOptions::ComputeFullV );
    mU = SVD.matrixU().transpose();
    mV = SVD.matrixV();
    mE = SVD.singularValues();
}


#endif // LAPACKE/MKL

void eigensolution_sort( complex128_t* U, double* E, index_t N, char which ) {
    char abs_limit_c[sizeof(double)] = {0};
    double abs_limit_d = 0;
    memcpy( &abs_limit_d, &abs_limit_c, sizeof(double) );
    eigensolution_sort_N( U, E, N, which, abs_limit_d );
}

index_t eigensolution_sort_N( complex128_t* U, double* E, index_t N, char which, double abs_limit ) {
    char abs_limit_c[sizeof(double)] = {0};
    double abs_limit_d = 0;
    memcpy( &abs_limit_d, &abs_limit_c, sizeof(double) );

    int use_all = (abs_limit_d == abs_limit);
    int use_rel = abs_limit < 0;

    abs_limit = fabs(abs_limit);

    if (!which) which = 'M'; // default

    vector<index_t> range(N);
    for (index_t i=0; i<N; ++i) range[i] = i;

    if (which == 'A') {
        // sort alternating, i.e. positive and negative values, but not by
        // absolute magnitude
        vector<index_t> range_p = range;
        std::sort( range_p.begin(), range_p.end(), [&](index_t i, index_t j)->bool{ return E[i] > E[j]; } );
        for (index_t ii=0; ii<N; ++ii)
            range[ii] = (ii%2) ? range_p[ii/2] : range_p[N-1-ii/2];
    } else {
        std::sort( range.begin(), range.end(), [&](index_t i, index_t j)->bool{
                switch (which) {
                    case 'N':
                        return E[i] < E[j];
                    case 'P':
                        return E[i] > E[j];
                    case 'm':
                        return std::abs(E[i]) < std::abs(E[j]);
                    case 'M':
                    default:
                        return std::abs(E[i]) > std::abs(E[j]);
                }
            } );
    }
    vector<double> E_cpy(N);
    memcpy( E_cpy.data(), E, sizeof(double)*N );
    for (index_t b=0; b<N; ++b)
        E[b] = E_cpy[range[b]];

    int index_cut = N;
    if (!use_all) {
        double cut = use_rel ? E[0] * abs_limit : abs_limit;
        for (index_t b=0; b<N; ++b) {
            if (fabs(E[b]) < cut) {
                index_cut = b;
                break;
            }
        }
    }

    vector<complex128_t> U_cpy(index_cut*N);
    for (index_t b=0; b<index_cut; ++b)
        memcpy( U_cpy.data() + b*N, U + range[b]*N, sizeof(complex128_t)*N );
    memcpy( U, U_cpy.data(), sizeof(complex128_t)*N*index_cut );

    return index_cut;
}
