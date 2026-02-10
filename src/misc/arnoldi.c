#include "arnoldi.h"
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "gemm.h"

#ifdef USE_ARPACK_EIGENSOLVER
extern void znaupd_(
    int *ido, char *bmat, int *n, char *which,
    int *nev, double *tol,
    complex128_t *resid,
    int *ncv, complex128_t *v, int *ldv,
    int *iparam, int *ipntr,
    complex128_t *workd,
    complex128_t *workl,
    int *lworkl, double *rwork,
    int *info
);
extern void zneupd_(
    int *rvec, char *howmny, int *select,
    complex128_t *d,
    complex128_t *z, int *ldz,
    complex128_t *sigma,
    complex128_t *workev,
    char *bmat, int *n, char *which,
    int *nev, double *tol,
    complex128_t *resid,
    int *ncv, complex128_t *v, int *ldv,
    int *iparam, int *ipntr,
    complex128_t *workd,
    complex128_t *workl,
    int *lworkl, double *rwork,
    int *info
);
static inline void arpack_message( void ) {}
#else // USE_ARPACK_EIGENSOLVER
void znaupd_( int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
    complex128_t *resid, int *ncv, complex128_t *v, int *ldv, int *iparam,
    int *ipntr, complex128_t *workd, complex128_t *workl, int *lworkl,
    double *rwork, int *info  ) {
    *ido = 99;
    *info = 1;
    (void)bmat;
    (void)n;
    (void)which;
    (void)nev;
    (void)tol;
    (void)resid;
    (void)ncv;
    (void)v;
    (void)ldv;
    (void)iparam;
    (void)ipntr;
    (void)workd;
    (void)workl;
    (void)lworkl;
    (void)rwork;
}
void zneupd_( int *rvec, char *howmny, int *select, complex128_t *d,
        complex128_t *z, int *ldz, complex128_t *sigma,
        complex128_t *workev, char *bmat, int *n, char *which, int *nev,
        double *tol, complex128_t *resid, int *ncv, complex128_t *v,
        int *ldv, int *iparam, int *ipntr, complex128_t *workd,
        complex128_t *workl, int *lworkl, double *rwork, int *info ) {
    *info = 1;
    (void)rvec;
    (void)howmny;
    (void)select;
    (void)d;
    (void)z;
    (void)ldz;
    (void)sigma;
    (void)workev;
    (void)bmat;
    (void)n;
    (void)which;
    (void)nev;
    (void)tol;
    (void)resid;
    (void)ncv;
    (void)v;
    (void)ldv;
    (void)iparam;
    (void)ipntr;
    (void)workd;
    (void)workl;
    (void)lworkl;
    (void)rwork;
}
static inline void arpack_message( void ) {
    mpi_err_printf( "arpack not compiled in (-DUSE_ARPACK_EIGENSOLVER)\n" );
}
#endif // USE_ARPACK_EIGENSOLVER

int arnoldi_eigen( char* which, int n, int nev, int ncv,
        arnoldi_matvec_t matvec, void* userdata, complex128_t* eigvals,
        complex128_t* eigvecs, double tol, int maxit ) {
    arpack_message();
    int ido = 0;
    char bmat = 'I';
    int info = 0;

    int iparam[11] = {0};
    int ipntr[14] = {0};

    iparam[0] = 1;        // exact shifts
    iparam[2] = maxit;    // max iterations
    iparam[6] = 1;        // standard eigenproblem

    complex128_t* resid  = calloc(n, sizeof *resid);
    complex128_t* v      = calloc((size_t)n * ncv, sizeof *v);
    complex128_t* workd  = calloc(3 * n, sizeof *workd);

    int lworkl = 3 * ncv * ncv + 5 * ncv;
    complex128_t* workl  = calloc(lworkl, sizeof *workl);
    double* rwork        = calloc(ncv, sizeof *rwork);

    if (!resid || !v || !workd || !workl || !rwork)
        return -1;

    do {
        znaupd_( &ido, &bmat, &n, which, &nev, &tol, resid, &ncv, v, &n,
            iparam, ipntr, workd, workl, &lworkl, rwork, &info );

        if (ido == -1 || ido == 1) {
            complex128_t* x = &workd[ipntr[0] - 1];
            complex128_t* y = &workd[ipntr[1] - 1];
            matvec(n, x, y, userdata);
        }
    } while (ido != 99);

    if (info != 0) goto cleanup;

    int rvec = 1;             // compute eigenvectors
    char howmny = 'A';        // all converged
    complex128_t sigma = 0.0; // no shift
    int* select          = calloc(ncv, sizeof *select);
    complex128_t* workev = calloc(2*ncv, sizeof *workev);

    zneupd_( &rvec, &howmny, select, eigvals, eigvecs, &n, &sigma, workev,
        &bmat, &n, which, &nev, &tol, resid, &ncv, v, &n, iparam, ipntr, workd,
        workl, &lworkl, rwork, &info );

    free(select);
    free(workev);
cleanup:
    free(resid);
    free(v);
    free(workd);
    free(workl);
    free(rwork);
    return info;
}

void arnoldi_dense_matvec( int n, const complex128_t* x, complex128_t* y, void* userdata ) {
    const complex128_t* A = userdata;
    single_gemm_serial( A, x, y, n, n, 1, 1.0, 0.0 );
}

void arnoldi_dense_matvec_par( int n, const complex128_t* x, complex128_t* y, void* userdata ) {
    const complex128_t* A = userdata;
    single_gemm( A, x, y, 1, n, n, 1.0, 0.0 );
}

int arnoldi_eigen_dense( char* which, int n, int nev, int ncv,
        const complex128_t* Aij, complex128_t* eigvals, complex128_t* eigvecs,
        double tol, int maxit ) {
    return arnoldi_eigen( which, n, nev, ncv, &arnoldi_dense_matvec_par, (void*)Aij, eigvals, eigvecs, tol, maxit );
}
