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
typedef void (*arnoldi_matvec_t)( int n, const complex128_t* x, complex128_t* y, void* userdata );
int arnoldi_eigen( char* which, int n, int nev, int ncv, arnoldi_matvec_t matvec,
        void* userdata, complex128_t* eigvals, complex128_t* eigvecs,
        double tol, int maxit );

// specializations for dense matrix/vector setups

// which -> "SR", "LR", "SI", "LI", "SM", "LM" -> Smallest/Largest Real/Imaginary
void arnoldi_dense_matvec_par( int n, const complex128_t* x, complex128_t* y, void* userdata );
int arnoldi_eigen_dense( char* which, int n, int nev, int ncv,
        const complex128_t* Aij, complex128_t* eigvals, complex128_t* eigvecs,
        double tol, int maxit );

// which -> "SR", "LR", "SI", "LI", "SM", "LM" -> Smallest/Largest Real/Imaginary
void arnoldi_dense_vecmat_par( int n, const complex128_t* x, complex128_t* y, void* userdata );
int arnoldi_eigen_dense_T( char* which, int n, int nev, int ncv,
        const complex128_t* Aij, complex128_t* eigvals, complex128_t* eigvecs,
        double tol, int maxit );

// serial version, to use together with arnoldi_eigen
void arnoldi_dense_matvec( int n, const complex128_t* x, complex128_t* y, void* userdata );

int arnoldi_package_compiled_in( void );

#ifdef __cplusplus
}
#endif
