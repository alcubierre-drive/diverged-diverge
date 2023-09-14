/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifndef CTYPESGEN
#include "misc/mpi_log.h"
#endif

#ifdef __cplusplus

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdbool>
#include <cstdint>

#ifndef CTYPESGEN
// c++ is nasty.
#include <complex>
typedef std::complex<double> complex128_t;
typedef std::complex<float> complex64_t;
const complex128_t I128 (0.0,1.0) ;
#endif

extern "C" {

#else // __cplusplus

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef CTYPESGEN
#include <complex.h>
// Typedef: complex128_t
// 128bit complex number (two 64bit floating point numbers)
typedef _Complex double complex128_t;
// Typedef: complex64_t
// 64bit complex number (two 32bit floating point numbers)
typedef _Complex float complex64_t;
#endif

#endif // __cplusplus

#ifdef CTYPESGEN
struct complex128_t { double x; double y; };
typedef struct complex128_t complex128_t;
struct complex64_t { float x; float y; };
typedef struct complex64_t complex64_t;
#endif

#ifdef USE_GF_FLOATS
typedef complex64_t gf_complex_t;
#else // USE_GF_FLOATS
// Typedef: gf_complex_t
// Green's function complex type. uses <complex128_t> if USE_GF_FLOATS is not
// defined. Otherwise, <complex64_t> is used
typedef complex128_t gf_complex_t;
#endif // USE_GF_FLOATS
#define lapack_complex_double complex128_t
#define lapack_complex_float complex64_t

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#ifndef MAX_N_ORBS
#define MAX_N_ORBS 32768
#endif // MAX_N_ORBS

#ifndef MAX_NAME_LENGTH
#define MAX_NAME_LENGTH 1024
#endif // MAX_NAME_LENGTH

#ifndef MAX_N_SYM
#define MAX_N_SYM 256
#endif // MAX_N_SYM

#ifndef DIVERGE_EPS_MESH
#define DIVERGE_EPS_MESH 1.e-6
#endif // DIVERGE_EPS_MESH

// Typedef: index_t
// 64bit signed integer
typedef int64_t index_t;
// Typedef: uindex_t
// 64bit unsigned integer
typedef uint64_t uindex_t;

#define POW2(x) ((x)*(x))
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) ((x)*(x)*(x)*(x))

#define IDX2( i, j, nj ) \
    ( (i)*(nj) + (j) )
#define IDX3( i, j, k, nj, nk ) \
    ( (i)*(nj)*(nk) + IDX2(j, k, nk) )
#define IDX4( i, j, k, l, nj, nk, nl ) \
    ( (i)*(nj)*(nk)*(nl) + IDX3(j, k, l, nk, nl) )
#define IDX5( i, j, k, l, m, nj, nk, nl, nm ) \
    ( (i)*(nj)*(nk)*(nl)*(nm) + IDX4(j, k, l, m, nk, nl, nm) )
#define IDX6( i, j, k, l, m, n, nj, nk, nl, nm, nn ) \
    ( (i)*(nj)*(nk)*(nl)*(nm)*(nn) + IDX5(j, k, l, m, n, nk, nl, nm, nn) )
#define IDX7( i, j, k, l, m, n, o, nj, nk, nl, nm, nn, no ) \
    ( (i)*(nj)*(nk)*(nl)*(nm)*(nn)*(no) + IDX6(j, k, l, m, n, o, nk, nl, nm, nn, no) )
#define IDX8( i, j, k, l, m, n, o, p, nj, nk, nl, nm, nn, no, np ) \
    ( (i)*(nj)*(nk)*(nl)*(nm)*(nn)*(no)*(np) + IDX7(j, k, l, m, n, o, p, nk, nl, nm, nn, no, np) )

#define MAX(x,y) (x) > (y) ? (x) : (y)
#define MIN(x,y) (x) < (y) ? (x) : (y)

#ifndef BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB
#define BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB 4ul
#endif // BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB

#ifndef BATCHED_EIGEN_NCHUNKS_AUTONUM
#define BATCHED_EIGEN_NCHUNKS_AUTONUM 1000000
#endif // BATCHED_EIGEN_NCHUNKS_AUTONUM

#ifndef USE_NO_LAPACKE
#define USE_LAPACKE
#else
#undef USE_LAPACKE
#endif

#ifdef __cplusplus
}
#endif

