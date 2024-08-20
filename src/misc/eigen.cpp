/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "../diverge_Eigen3.hpp"

#include "eigen.h"
#include "qnan.h"

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

void single_eigen( const complex128_t* H, complex128_t* U, double* E, index_t N ) {
    assert(U != NULL);
    assert(E != NULL);
    assert(N > 0);
    if (H != NULL)
        if (U != H)
            memcpy( U, H, sizeof(complex128_t)*N*N );
    LAPACKE_zheev( LAPACK_ROW_MAJOR, 'V', 'U', N, (lapack_complex_double*)U, N, E );
}

void single_svd( const complex128_t* H, complex128_t* U, complex128_t* V, double* E, index_t N ) {
    assert(U != NULL);
    assert(V != NULL);
    assert(E != NULL);
    if (N > 0) {
        if (H != NULL)
            if (U != H)
                memcpy( U, H, sizeof(complex128_t)*N*N );
        double* scratch = (double*)calloc(N, sizeof(double));
        LAPACKE_zgesvd( LAPACK_ROW_MAJOR, 'O', 'A', N, N, U, N, E, NULL, N, V, N, scratch );
        free(scratch);
    } else {
        // eigen (non-hermitian)
        N = std::abs(N);
        lapack_complex_double* A = (lapack_complex_double*)calloc(N*N, sizeof(complex128_t));
        memcpy( A, (H == NULL || H == U) ? U : H, sizeof(complex128_t)*N*N );
        VecXcd Ec = VecXcd::Zero(N);
        LAPACKE_zgeev( LAPACK_ROW_MAJOR, 'V', 'V', N, A, N, (lapack_complex_double*)Ec.data(), U, N, V, N );
        Map<VecXd> mE(E, N); mE = Ec.array().real().matrix();
        free( A );
    }
}

#else // LAPACKE/MKL

void single_eigen( const complex128_t* H, complex128_t* U, double* E, index_t N ) {
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

void single_svd( const complex128_t* H, complex128_t* U, complex128_t* V, double* E, index_t N ) {
    assert(U != NULL);
    assert(V != NULL);
    assert(E != NULL);

    if (N > 0) {
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
    } else {
        N = std::abs(N);
        MatXcd A = (H == NULL || H == U) ? Map<const MatXcd>(U, N, N) : Map<const MatXcd>(H, N, N);
        Map<MatXcd> mU(U, N, N);
        Map<MatXcd> mV(V, N, N);
        Map<VecXd> mE(E, N);

        Eigen::ComplexEigenSolver<MatXcd> sol( A.transpose() );
        mV = sol.eigenvectors().transpose();
        mU = sol.eigenvectors().inverse();
        mE = sol.eigenvalues().array().real().matrix();
    }
}


#endif // LAPACKE/MKL

void single_eigen_sort( complex128_t* U, double* E, index_t N, char which ) {
    single_eigen_sort_N( U, E, N, which, qnan_gen(0) );
}

index_t single_eigen_sort_part( complex128_t* U, double* E, index_t N, char which, index_t num ) {
    return single_eigen_sort_N( U, E, N, which, qnan_gen(num) );
}

index_t single_eigen_sort_N( complex128_t* U, double* E, index_t N, char which, double abs_limit ) {
    int use_all_or_num = 0;
    uint32_t num_limit = 0;
    if (qnan_isnan(abs_limit)) {
        num_limit = qnan_get(abs_limit);
        use_all_or_num = 1;
    }
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

    int index_cut = (num_limit != 0) ? num_limit : N;
    if (!use_all_or_num) { // overwrite index_cut
        double cut = use_rel ? fabs(E[0]) * abs_limit : abs_limit;
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
