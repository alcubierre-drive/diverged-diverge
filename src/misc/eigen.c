#include "eigen.h"
#include "mt19937.h"
#include "mpi_functions.h"
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#ifndef DIVERGE_SKIP_ZDOTC
#define DIVERGE_DEFINE_ZDOTC
#endif
#else // USE_MKL
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <cblas.h>
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif
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

complex128_t power_iteration( const complex128_t* restrict A, complex128_t* restrict v,
        index_t N, complex128_t* restrict aux, index_t* restrict niter, double* restrict tol,
        const int* restrict iseed ) {

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

static const index_t  bpit_default_maxiter = 40000;
static const double   bpit_default_tol = 1.e-6;
static uint64_t       bpit_default_seed = 1234;

void power_iteration_batched_cfg_seed( uint64_t seed ) { bpit_default_seed = seed; }

static inline void fill_random(complex128_t* v, index_t N, mt19937_64_t* rng) {
    for (index_t i=0; i<N; ++i) {
        // Box-Miller
        double U1 = 0.0; while (U1 <= 0.0) U1 = mt64_next_dbl(rng);
        double U2 = 0.0; while (U2 <= 0.0) U2 = mt64_next_dbl(rng);
        double Zx = sqrt(-2.0*log(U1)) * cos(2.*M_PI*U2),
               Zy = sqrt(-2.0*log(U1)) * sin(2.*M_PI*U2);
        v[i] = Zx + I*Zy;
    }
}
static inline void normalize_noblas(complex128_t* v, index_t N) {
    double nrm2 = 0.0;
    for (index_t i=0; i<N; ++i) nrm2 += v[i]*conj(v[i]);
    double fac = 1./sqrt(nrm2);
    for (index_t i=0; i<N; ++i) v[i] *= fac;
}
static inline uint64_t lcg64(uint64_t seed) { return 6364136223846793005ul*seed + 1442695040888963407ul; }
static inline void matvec_noblas(const complex128_t* A, const complex128_t* v, const index_t N, complex128_t* w) {
    for (index_t i=0; i<N; ++i) {
        complex128_t wi = 0.0;
        for (index_t j=0; j<N; ++j) wi += A[IDX2(i,j,N)] * v[j];
        w[i] = wi;
    }
}
static inline complex128_t dot_noblas(const complex128_t* v1, const complex128_t* v2, const index_t N) {
    complex128_t res = 0.0;
    for (index_t i=0; i<N; ++i) res += conj(v1[i]) * v2[i];
    return res;
}

void power_iteration_batched( const complex128_t* A_batched, complex128_t* _U_batched,
        complex128_t* E_batched, const index_t N, const index_t num, index_t* _maxiter,
        double* _tol, complex128_t* _aux ) {
    int nthr = MIN(diverge_omp_num_threads(), num);

    mt19937_64_t* rng = calloc(nthr, sizeof*rng);
    for (int t=0; t<nthr; ++t) rng[t] = mt64_init((bpit_default_seed=lcg64(bpit_default_seed)));

    #define ALLOC_OPTIONAL(type, name, cnt) \
        type* name = _##name ? _##name : calloc((cnt), sizeof(type));
    #define FREE_OPTIONAL(name) \
        if (!_##name) free(name);
    ALLOC_OPTIONAL(complex128_t, U_batched, N*num);
    ALLOC_OPTIONAL(index_t,      maxiter,   num);
    ALLOC_OPTIONAL(double,       tol,       num);
    ALLOC_OPTIONAL(complex128_t, aux,       2*N*nthr);

    if (!_maxiter) for (index_t q=0; q<num; ++q) maxiter[q] = bpit_default_maxiter;
    if (!_tol)     for (index_t q=0; q<num; ++q) tol[q]     = bpit_default_tol;

    #pragma omp parallel for num_threads(nthr)
    for (index_t q=0; q<num; ++q) {
        int t = omp_get_thread_num();
        complex128_t *v = aux + 2*t*N,
                     *w = aux + 2*t*N + N;
       const complex128_t* A = A_batched + q*POW2(N);
       complex128_t* u = U_batched + q*N;

       double max_eigval_est = 0.0;
       for (index_t i=0; i<N; ++i) {
           double max = 0.0;
           for (index_t j=0; j<N; ++j) max += cabs(A[IDX2(i,j,N)]);
           max_eigval_est = MAX(max, max_eigval_est);
       }

       fill_random(v, N, rng+t);
       normalize_noblas(v, N);
       double ctol = -1.0;
       complex128_t eig = 0.0;
       complex128_t eig_prev = 0.0;
       int i = 0;
       for (; i<maxiter[q]; ++i) {
           eig_prev = eig;
           XORSWAP_PTR(w, v);
           matvec_noblas(A, w, N, v);
           normalize_noblas(v, N);
           matvec_noblas(A, v, N, u);
           eig = dot_noblas(v, u, N);
           ctol = cabs(eig - eig_prev)/max_eigval_est;
           if (ctol <= tol[q]) break;
       }
       maxiter[q] = i;
       tol[q] = ctol;
       E_batched[q] = eig;
       memcpy(u, v, sizeof*u * N);
    }

    FREE_OPTIONAL(U_batched);
    FREE_OPTIONAL(maxiter);
    FREE_OPTIONAL(tol);
    FREE_OPTIONAL(aux);
    free(rng);
}
