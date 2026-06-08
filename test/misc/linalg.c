#define CT_SUITE_NAME linalg
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_common.h"
#include "../../src/misc/gemm.h"
#include "../../src/misc/eigen.h"
#include "../helper_functions.h"

#include "../../src/misc/batched_gemm_small.h"
#include "../../src/misc/batched_eigen.h"
#include "../../src/misc/batched_svd.h"
#include <limits.h>

static complex128_t* gen_random_matrix_batch( index_t dim, index_t num ) {
    complex128_t* b = calloc(num*POW2(dim), sizeof*b);
    for (index_t ijk=0; ijk<num*POW2(dim); ++ijk) b[ijk] = random_cmplx();
    return b;
}
static inline double vector_distance( const complex128_t* A, const complex128_t* B, index_t N ) {
    double res = 0.0;
    for (index_t i=0; i<N; ++i) {
        complex128_t sub = A[i] - B[i];
        res += creal(conj(sub) * sub);
    }
    return sqrt(res);
}
static inline double vector_distance_d( const double* A, const double* B, index_t N ) {
    double res = 0.0;
    for (index_t i=0; i<N; ++i) {
        double sub = A[i] - B[i]; res += POW2(sub);
    }
    return sqrt(res);
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
        complex128_t* _C = calloc(N*N, sizeof*_C);
        single_square_gemm(A+i*N*N, 0, B+i*N*N, 0, _C, N);
        memcpy(B+i*N*N, _C, sizeof*_C*N*N);
        free(_C);
    }

    // and then compare C with B
    for (index_t i=0; i<num; ++i) {
        double err = vector_distance(B, C, N*N);
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

    // set H = A + A^\dagger
    memcpy( H, A, sizeof*H * num * N*N );
    for (index_t i=0; i<num; ++i) {
        transpose_inplace(A+i*N*N, N);
        for (index_t ij=0; ij<N*N; ++ij) H[IDX2(i,ij,N*N)] += conj(A[IDX2(i,ij,N*N)]);
    }

    int mode = batched_eigen_r_get_eigen3_mode();

    double* E1 = calloc(N*num, sizeof(double));
    complex128_t* U1 = U;
    memcpy( U1, H, sizeof(complex128_t)*N*N*num );
    batched_eigen_r_set_eigen3_mode(0);
    batched_eigen_r( NULL, -1, U1, E1, N, num );

    double* E2 = calloc(N*num, sizeof(double));
    complex128_t* U2 = A;
    memcpy( U2, H, sizeof(complex128_t)*N*N*num );
    batched_eigen_r_set_eigen3_mode(1);
    batched_eigen_r( NULL, -1, U2, E2, N, num );

    batched_eigen_r_set_eigen3_mode(mode);

    complex128_t* tmp_U = calloc(N*N, sizeof*tmp_U);
    complex128_t* tmp_E = calloc(N*N, sizeof*tmp_E);
    complex128_t* tmp_1 = calloc(N*N, sizeof*tmp_1);
    complex128_t* tmp_2 = calloc(N*N, sizeof*tmp_2);
    double* tmp_Ed = calloc(N, sizeof*tmp_Ed);
    for (index_t i=0; i<num; ++i) {
        ZERO_OUT( tmp_U, N*N ); ZERO_OUT( tmp_E, N*N );
        ZERO_OUT( tmp_1, N*N ); ZERO_OUT( tmp_2, N*N );
        ZERO_OUT( tmp_Ed, N );
        const complex128_t* ham = H+i*N*N;

        single_eigen(ham, tmp_U, tmp_Ed, N);
        // make tmp_E the diagonal matrix of eigenvalues
        for (index_t j=0; j<N; ++j) tmp_E[IDX2(j,j,N)] = tmp_Ed[j];
        single_square_gemm(tmp_U, 1, ham, 0, tmp_1, N);
        single_square_gemm(tmp_1, 0, tmp_U, 0, tmp_2, N);
        double N3 = vector_distance(tmp_2, tmp_E, N*N);

        ZERO_OUT( tmp_U, N*N ); ZERO_OUT( tmp_E, N*N );
        ZERO_OUT( tmp_1, N*N ); ZERO_OUT( tmp_2, N*N );
        for (index_t j=0; j<N; ++j) tmp_E[IDX2(j,j,N)] = E2[IDX2(i,j,N)];
        transpose_inplace(U2+i*N*N, N);
        single_square_gemm(U2+i*N*N, 1, ham, 0, tmp_1, N);
        single_square_gemm(tmp_1, 0, U2+i*N*N, 0, tmp_2, N);
        double N2 = vector_distance(tmp_2, tmp_E, N*N);

        ZERO_OUT( tmp_U, N*N ); ZERO_OUT( tmp_E, N*N );
        ZERO_OUT( tmp_1, N*N ); ZERO_OUT( tmp_2, N*N );
        for (index_t j=0; j<N; ++j) tmp_E[IDX2(j,j,N)] = E1[IDX2(i,j,N)];
        transpose_inplace(U1+i*N*N, N);
        single_square_gemm(U1+i*N*N, 1, ham, 0, tmp_1, N);
        single_square_gemm(tmp_1, 0, U1+i*N*N, 0, tmp_2, N);
        double N1 = vector_distance(tmp_2, tmp_E, N*N);

        CT_CHECK( N3 < 1.e-7 );
        CT_CHECK( N2 < 1.e-7 );
        CT_CHECK( N1 < 1.e-7 );

        CT_CHECK( vector_distance_d(tmp_Ed, E2+i*N, N) < 1.e-7 );
        CT_CHECK( vector_distance_d(tmp_Ed, E1+i*N, N) < 1.e-7 );
        CT_CHECK( vector_distance_d(E2+i*N, E1+i*N, N) < 1.e-7 );
    }
    free( tmp_U );
    free( tmp_E );
    free( tmp_1 );
    free( tmp_2 );
    free( tmp_Ed );

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

    // set H = A + A^\dagger
    memcpy( H, A, sizeof*H * num * N*N );
    for (index_t i=0; i<num; ++i) {
        transpose_inplace(A+i*N*N, N);
        for (index_t ij=0; ij<N*N; ++ij) H[IDX2(i,ij,N*N)] += conj(A[IDX2(i,ij,N*N)]);
    }


    double* E1 = calloc(N*num, sizeof(double));
    complex128_t* U1 = U;
    memcpy( U1, H, sizeof(complex128_t)*N*N*num );
    batched_eigen_r( NULL, -1, U1, E1, N, num );

    double* E2 = calloc(N*num, sizeof(double));
    complex128_t* U2 = A;
    memcpy( U2, H, sizeof(complex128_t)*N*N*num );
    batched_svd( NULL, INT_MIN, H, U2, NULL, E2, N, num );

    complex128_t* tmp_U = calloc(N*N, sizeof*tmp_U);
    complex128_t* tmp_E = calloc(N*N, sizeof*tmp_E);
    complex128_t* tmp_1 = calloc(N*N, sizeof*tmp_1);
    complex128_t* tmp_2 = calloc(N*N, sizeof*tmp_2);
    for (index_t i=0; i<num; ++i) {
        ZERO_OUT( tmp_U, N*N ); ZERO_OUT( tmp_E, N*N );
        ZERO_OUT( tmp_1, N*N ); ZERO_OUT( tmp_2, N*N );
        const complex128_t* ham = H+i*N*N;

        ZERO_OUT( tmp_U, N*N ); ZERO_OUT( tmp_E, N*N );
        ZERO_OUT( tmp_1, N*N ); ZERO_OUT( tmp_2, N*N );
        for (index_t j=0; j<N; ++j) tmp_E[IDX2(j,j,N)] = E2[IDX2(i,j,N)];
        transpose_inplace(U2+i*N*N, N);
        single_square_gemm(U2+i*N*N, 1, ham, 0, tmp_1, N);
        single_square_gemm(tmp_1, 0, U2+i*N*N, 0, tmp_2, N);
        double N2 = vector_distance(tmp_2, tmp_E, N*N);

        ZERO_OUT( tmp_U, N*N ); ZERO_OUT( tmp_E, N*N );
        ZERO_OUT( tmp_1, N*N ); ZERO_OUT( tmp_2, N*N );
        for (index_t j=0; j<N; ++j) tmp_E[IDX2(j,j,N)] = E1[IDX2(i,j,N)];
        transpose_inplace(U1+i*N*N, N);
        single_square_gemm(U1+i*N*N, 1, ham, 0, tmp_1, N);
        single_square_gemm(tmp_1, 0, U1+i*N*N, 0, tmp_2, N);
        double N1 = vector_distance(tmp_2, tmp_E, N*N);

        CT_CHECK( N2 < 1.e-7 );
        CT_CHECK( N1 < 1.e-7 );

        CT_CHECK( vector_distance_d(E2+i*N, E1+i*N, N) < 1.e-7 );
    }
    free(tmp_U);
    free(tmp_E);
    free(tmp_1);
    free(tmp_2);

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
    double* S = calloc(N*num, sizeof(double));

    batched_svd( NULL, -1, A, U, V, S, N, num );

    complex128_t* sU = calloc(N*N, sizeof*sU);
    complex128_t* sV = calloc(N*N, sizeof*sV);
    double* sS = calloc(N, sizeof*sS);
    complex128_t* tmp_1 = calloc(N*N, sizeof*tmp_1);
    complex128_t* tmp_2 = calloc(N*N, sizeof*tmp_2);
    for (index_t i=0; i<num; ++i) {
        single_svd( A+i*N*N, sU, sV, sS, N );

        ZERO_OUT( tmp_2, N*N ); for (index_t j=0; j<N; ++j) tmp_2[IDX2(j,j,N)] = sS[j];
        single_square_gemm( sU, 0, tmp_2, 0, tmp_1, N );
        single_square_gemm( tmp_1, 0, sV, 0, tmp_2, N );
        CT_CHECK( vector_distance( tmp_2, A+i*N*N, N*N ) < 1.e-7 );

        ZERO_OUT( tmp_2, N*N ); for (index_t j=0; j<N; ++j) tmp_2[IDX2(j,j,N)] = S[IDX2(i,j,N)];
        transpose_outofplace( sV, V+i*N*N, N );
        single_square_gemm( U+i*N*N, 0, tmp_2, 0, tmp_1, N );
        single_square_gemm( tmp_1, 0, sV, 1, tmp_2, N );
        CT_CHECK( vector_distance( tmp_2, A+i*N*N, N*N ) < 1.e-7 );

        CT_CHECK( vector_distance_d( sS, S+i*N, N ) < 1.e-7 );
    }
    free( tmp_1 );
    free( tmp_2 );
    free( sU );
    free( sV );
    free( sS );

    free(A);
    free(V);
    free(U);
    free(S);
}

CT_TEST_NAME( batched_piter, "batched,piter" ) {
    complex128_t *A = gen_random_matrix_batch(N, num),
                 *H = gen_random_matrix_batch(N, num);

    // set H = A + A^\dagger; both A and H contain the matrix after this block
    memcpy( H, A, sizeof*H * num * N*N );
    for (index_t i=0; i<num; ++i) {
        transpose_inplace(A+i*N*N, N);
        for (index_t ij=0; ij<N*N; ++ij) H[IDX2(i,ij,N*N)] += conj(A[IDX2(i,ij,N*N)]);
    }
    memcpy( A, H, sizeof*A * num * N*N );

    double* E1 = calloc(N*num, sizeof(double));
    batched_eigen_r( NULL, -1, A, E1, N, num );

    complex128_t* E2 = calloc(N*num, sizeof(complex128_t));
    double TOL = 1.e-14;
    index_t MAXITER = 40000;
    double* tol = calloc(num, sizeof*tol);
    index_t* maxiter = calloc(num, sizeof*maxiter);
    for (index_t i=0; i<num; ++i) {
        tol[i] = TOL;
        maxiter[i] = MAXITER;
    }
    power_iteration_batched( H, NULL, E2, N, num, maxiter, tol, NULL );

    free(H);
    free(A);

    for (index_t i=0; i<num; ++i) {
        double maxval_eigen = 0.0;
        for (index_t a=0; a<N; ++a)
            maxval_eigen = MAX(maxval_eigen, fabs(E1[IDX2(i,a,N)]));
        double maxval_piter = cabs(E2[i]);

        CT_CHECK( fabs(maxval_piter - maxval_eigen) <= 1.e-7 * maxval_piter );
        CT_CHECK( maxiter[i] < MAXITER );
        CT_CHECK( tol[i] < TOL );
    }

    free(tol);
    free(maxiter);
    free(E1);
    free(E2);
}

#endif // DIVERGE_SKIP_TESTS
