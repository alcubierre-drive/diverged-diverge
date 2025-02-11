#include "eigen.h"

#ifdef USE_MKL
#include <mkl.h>
#ifndef DIVERGE_SKIP_ZDOTC
#define DIVERGE_DEFINE_ZDOTC
#endif
#else // USE_MKL
#include <cblas.h>
#include <lapacke.h>
#endif // USE_MKL

#ifdef DIVERGE_DEFINE_ZDOTC
static inline complex128_t cblas_zdotc(index_t dim, const complex128_t* w,
        index_t incw, const complex128_t* v, index_t incv) {
    complex128_t res = 0.0;
    cblas_zdotc_sub(dim, w, incw, v, incv, &res);
    return res;
}
#endif

static index_t power_iteration_maxiter = 40000;
static double power_iteration_tol = 1.e-6;
static const lapack_int power_iteration_seed[4] = {4,3,2,1};

static inline void normalize( complex128_t* v, index_t N ) {
    cblas_zdscal( N, 1./cblas_dznrm2(N,v,1), v, 1 );
}

complex128_t power_iteration( const complex128_t* A, complex128_t* v, index_t N,
        complex128_t* aux, index_t* niter, double* tol, const int* iseed ) {

    lapack_int seed[4]; memcpy( seed, iseed?iseed:power_iteration_seed, sizeof(seed) );
    const index_t maxiter = niter ? *niter : power_iteration_maxiter;
    const double mintol = tol ? *tol : power_iteration_tol;

    complex128_t* w = aux; if (!w) w = (complex128_t*)calloc(2*N, sizeof*w);
    complex128_t* u = w + N;

    const complex128_t alpha = 1.0, beta = 0.0;
    // gershgorin circle estimate
    for (index_t i=0; i<N; ++i) w[i] = 1.0;
    cblas_zgemv( CblasRowMajor, CblasNoTrans, N, N, &alpha, A, N, w, 1, &beta, u, 1 );
    for (index_t i=0; i<N; ++i) u[i] = cabs(u[i]);
    double max_eigval = cabs(cblas_zdotc( N, w, 1, u, 1 ));

    // power iteration
    LAPACKE_zlarnv( 4, seed, N, v );
    normalize( v, N );
    double ctol = -1.0;
    complex128_t eigenvalue = 0.0;
    complex128_t last_eigenvalue = 0.0;
    int i;
    for (i=0; i<maxiter; ++i) {
        last_eigenvalue = eigenvalue;
        XORSWAP_PTR( w, v );
        cblas_zgemv( CblasRowMajor, CblasNoTrans, N, N, &alpha, A, N, w, 1, &beta, v, 1 );
        normalize( v, N );
        cblas_zgemv( CblasRowMajor, CblasNoTrans, N, N, &alpha, A, N, v, 1, &beta, u, 1 );
        eigenvalue = cblas_zdotc( N, v, 1, u, 1 );
        ctol = cabs(last_eigenvalue - eigenvalue)/max_eigval;
        if (ctol <= mintol) break;
    }

    if (tol) *tol = ctol;
    if (niter) *niter = i;
    if (!aux) free(w);
    return eigenvalue;
}
