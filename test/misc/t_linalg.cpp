#ifndef DIVERGE_SKIP_TESTS
#ifndef DIVERGE_SKIP_TESTS_LINALG

#define CT_SUITE_NAME linalg
#include "../ctest.h"
// #include "../catch.hpp"
#include "../../src/diverge_common.h"
#include "../../src/diverge_Eigen3.hpp"

#include "../../src/misc/batched_gemm_small.h"
#include "../../src/misc/batched_eigen.h"
#include "../../src/misc/batched_svd.h"

typedef Eigen::SelfAdjointEigenSolver<MatXcd> SolXcd;
typedef Eigen::JacobiSVD<MatXcd> SvdXcd;

static complex128_t* gen_random_matrix_batch( index_t dim, index_t num ) {
    complex128_t* b = (complex128_t*)calloc(num*POW2(dim), sizeof(complex128_t));
    for (index_t i=0; i<num; ++i)
        Map<cMatXcd>(b+i*POW2(dim), dim, dim) = cMatXcd::Random(dim, dim);
    return b;
}

static const index_t N = 50, num = 20;

CT_TEST_NAME( batched_gemm_small, "batched,gemm_small" ) {
    complex128_t *A = gen_random_matrix_batch( N, num ),
                 *B = gen_random_matrix_batch( N, num ),
                 *C = gen_random_matrix_batch( N, num );

    // C is only there to hold the result
    batched_gemm_small_handle_t* h = batched_gemm_small_init( N, num );
    batched_gemm_small( h, A, B, C );

    // we compute the 'acutal' result on the fly and save it into B
    for (index_t i=0; i<num; ++i) {
        cMatXcd _C = Map<cMatXcd>(A+i*N*N, N, N) * Map<cMatXcd>(B+i*N*N, N, N);
        Map<cMatXcd>(B+i*N*N, N, N) = _C;
    }

    // and then compare C with B
    for (index_t i=0; i<num; ++i) {
        double err = (Map<cMatXcd>(B+i*N*N, N, N) - Map<cMatXcd>(C+i*N*N, N, N)).norm();
        CT_CHECK(err < 1.e-7);
    }

    free(A);
    free(B);
    free(C);
    batched_gemm_small_destroy( h );
}

CT_TEST_NAME( batched_eigen, "batched,eigen" ) {
    complex128_t *A = gen_random_matrix_batch(N, num),
                 *H = gen_random_matrix_batch(N, num),
                 *U = gen_random_matrix_batch(N, num);

    for (index_t i=0; i<num; ++i)
        Map<cMatXcd>(H+i*N*N, N, N) = Map<cMatXcd>(A+i*N*N, N, N) + Map<cMatXcd>(A+i*N*N, N, N).adjoint();

    int mode = batched_eigen_r_get_eigen3_mode();

    double* E1 = (double*)calloc(N*num, sizeof(double));
    complex128_t* U1 = U;
    memcpy( U1, H, sizeof(complex128_t)*N*N*num );
    batched_eigen_r_set_eigen3_mode(0);
    batched_eigen_r( NULL, -1, U1, E1, N, num );

    double* E2 = (double*)calloc(N*num, sizeof(double));
    complex128_t* U2 = A;
    memcpy( U2, H, sizeof(complex128_t)*N*N*num );
    batched_eigen_r_set_eigen3_mode(1);
    batched_eigen_r( NULL, -1, U2, E2, N, num );

    batched_eigen_r_set_eigen3_mode(mode);

    for (index_t i=0; i<num; ++i) {
        Map<cMatXcd> H3(H+i*N*N, N, N);
        SolXcd Solv(H3);
        MatXcd U3 = Solv.eigenvectors();
        MatXd E3 = Solv.eigenvalues().asDiagonal();
        double N3 = (U3.adjoint() * H3 * U3 - E3).norm();

        MatXd xE2 = Map<VecXd>(E2+i*N, N).asDiagonal();
        MatXcd xU2 = Map<MatXcd>(U2+i*N*N, N,N);
        double N2 = (xU2.adjoint() * H3 * xU2 - xE2).norm();

        MatXd xE1 = Map<VecXd>(E1+i*N, N).asDiagonal();
        MatXcd xU1 = Map<MatXcd>(U1+i*N*N, N,N);
        double N1 = (xU1.adjoint() * H3 * xU1 - xE1).norm();

        CT_CHECK( N3 < 1.e-7 );
        CT_CHECK( N2 < 1.e-7 );
        CT_CHECK( N1 < 1.e-7 );

        CT_CHECK( (E3 - xE2).norm() < 1.e-7 );
        CT_CHECK( (E3 - xE1).norm() < 1.e-7 );
        CT_CHECK( (xE2 - xE1).norm() < 1.e-7 );
    }

    free(E1);
    free(E2);
    free(A);
    free(H);
    free(U);
}

CT_TEST_NAME( batched_eigen_svd, "batched,eigen_svd" ) {
    complex128_t *A = gen_random_matrix_batch(N, num),
                 *H = gen_random_matrix_batch(N, num),
                 *U = gen_random_matrix_batch(N, num);

    for (index_t i=0; i<num; ++i)
        Map<cMatXcd>(H+i*N*N, N, N) = Map<cMatXcd>(A+i*N*N, N, N) + Map<cMatXcd>(A+i*N*N, N, N).adjoint();

    double* E1 = (double*)calloc(N*num, sizeof(double));
    complex128_t* U1 = U;
    memcpy( U1, H, sizeof(complex128_t)*N*N*num );
    batched_eigen_r( NULL, -1, U1, E1, N, num );

    double* E2 = (double*)calloc(N*num, sizeof(double));
    complex128_t* U2 = A;
    memcpy( U2, H, sizeof(complex128_t)*N*N*num );
    batched_svd( NULL, INT_MIN, H, U2, NULL, E2, N, num );

    for (index_t i=0; i<num; ++i) {
        Map<cMatXcd> H3(H+i*N*N, N, N);
        MatXd xE2 = Map<VecXd>(E2+i*N, N).asDiagonal();
        MatXcd xU2 = Map<MatXcd>(U2+i*N*N, N,N);
        double N2 = (xU2.adjoint() * H3 * xU2 - xE2).norm();

        MatXd xE1 = Map<VecXd>(E1+i*N, N).asDiagonal();
        MatXcd xU1 = Map<MatXcd>(U1+i*N*N, N,N);
        double N1 = (xU1.adjoint() * H3 * xU1 - xE1).norm();

        CT_CHECK( N2 < 1.e-7 );
        CT_CHECK( N1 < 1.e-7 );

        CT_CHECK( (xE2 - xE1).norm() < 1.e-7 );
    }

    free(E1);
    free(E2);
    free(A);
    free(H);
    free(U);
}

CT_TEST_NAME( batched_svd, "batched,svd" ) {
    complex128_t *A = gen_random_matrix_batch(N, num),
                 *V = gen_random_matrix_batch(N, num),
                 *U = gen_random_matrix_batch(N, num);
    double* S = (double*)calloc(N*num, sizeof(double));

    batched_svd( NULL, -1, A, U, V, S, N, num );

    for (index_t i=0; i<num; ++i) {
        Map<cMatXcd> AA(A+i*N*N, N, N);
        SvdXcd sol(AA, Eigen::DecompositionOptions::ComputeFullU | Eigen::DecompositionOptions::ComputeFullV );
        MatXcd UU = sol.matrixU(),
               VV = sol.matrixV().transpose();
        VecXd SS = sol.singularValues();
        double X1 = (UU * SS.asDiagonal() * VV.conjugate() - AA).norm();
        CT_CHECK( X1 < 1.e-7 );

        Map<cMatXcd> U2( U+i*N*N, N, N ),
                     V2( V+i*N*N, N, N );
        Map<VecXd> S2( S+i*N, N );
        double X2 = (U2 * S2.asDiagonal() * V2.conjugate() - AA).norm();
        CT_CHECK( X2 < 1.e-7 );

        CT_CHECK( (SS - S2).norm() < 1.e-7 );
    }

    free(A);
    free(V);
    free(U);
    free(S);
}

#endif // DIVERGE_SKIP_TESTS_LINALG
#endif // DIVERGE_SKIP_TESTS
