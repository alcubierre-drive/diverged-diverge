/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
inline double cabs( complex128_t x ) { return std::abs(x); }
extern "C" {
#endif

#ifndef NPATCH_PERIODIC_NORM_NG
#define NPATCH_PERIODIC_NORM_NG 2
#endif

#ifndef NPATCH_ALMOST_ZERO
#define NPATCH_ALMOST_ZERO 1.e-7
#endif

static inline double periodic_norm( const complex128_t x, const complex128_t g1, const complex128_t g2 ) {
    double nrm = 2.*M_PI;
    for (int i=-NPATCH_PERIODIC_NORM_NG; i<=NPATCH_PERIODIC_NORM_NG; ++i)
    for (int j=-NPATCH_PERIODIC_NORM_NG; j<=NPATCH_PERIODIC_NORM_NG; ++j) {
        double _nrm = cabs( x + (double)i*(g1) + (double)j*(g2) );
        if (_nrm < nrm) nrm = _nrm;
    }
    return nrm;
}

static inline index_t find_closest_patch( complex128_t ppt, const complex128_t* pmesh,
        index_t np, const complex128_t g1, const complex128_t g2 ) {
    index_t result = -1;
    double nrm = 2.*M_PI;
    for (index_t p=0; p<np; ++p) {
        double _nrm = periodic_norm( ppt - pmesh[p], g1, g2 );
        if (_nrm < nrm) {
            nrm = _nrm;
            result = p;
        }
    }
    return result;
}

#ifndef __cplusplus
#define RESTRICT restrict
#else
#define RESTRICT
#endif
static inline void atomic_add( complex128_t* RESTRICT addr, const complex128_t* RESTRICT val ) {
    double* paddr = (double*)addr;
    const double* pval = (const double*)val;
    #pragma omp atomic
    paddr[0] += pval[0];
    #pragma omp atomic
    paddr[1] += pval[1];
}

#ifdef __cplusplus
}
#endif
