#pragma once

#ifdef __cplusplus
#include <complex>
typedef std::complex<double> complex128_t;
extern "C" {
#else
#include <complex.h>
typedef _Complex double complex128_t;
#endif

// which -> "SR", "LR", "SI", "LI", "SM", "LM" -> Smallest/Largest Real/Imaginary
typedef void (*arpack_matvec)( int n, const complex128_t* x, complex128_t* y, void* userdata );
int arnoldi_eigen( char* which, int n, int nev, int ncv, arpack_matvec matvec,
        void* userdata, complex128_t* eigvals, complex128_t* eigvecs,
        double tol, int maxit );

// which -> "SR", "LR", "SI", "LI", "SM", "LM" -> Smallest/Largest Real/Imaginary
int arnoldi_eigen_dense( char* which, int n, int nev, int ncv,
        const complex128_t* Aij, complex128_t* eigvals, complex128_t* eigvecs,
        double tol, int maxit );

#ifdef __cplusplus
}
#endif
