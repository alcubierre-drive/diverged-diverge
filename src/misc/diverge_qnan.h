/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

/** compiled version of :c:func:`qnan_gen` */
double      diverge_qnan_gen( uint64_t payload );
/** compiled version of :c:func:`qnan_get` */
uint64_t    diverge_qnan_get( double x );
/** compiled version of :c:func:`qnan_isnan` */
int         diverge_qnan_isnan( double x );

/** compiled version of :c:func:`fnan_gen` */
double      diverge_fnan_gen( double payload );
/** compiled version of :c:func:`fnan_get` */
double      diverge_fnan_get( double x );
/** compiled version of :c:func:`fnan_isnan` */
int         diverge_fnan_isnan( double x );

/** compiled version of :c:func:`qnanf_gen` */
float       diverge_qnanf_gen( uint32_t payload );
/** compiled version of :c:func:`qnanf_get` */
uint32_t    diverge_qnanf_get( float x );
/** compiled version of :c:func:`qnanf_isnan` */
int         diverge_qnanf_isnan( float x );

/** compiled version of :c:func:`fnanf_gen` */
float       diverge_fnanf_gen( float payload );
/** compiled version of :c:func:`fnanf_get` */
float       diverge_fnanf_get( float x );
/** compiled version of :c:func:`fnanf_isnan` */
int         diverge_fnanf_isnan( float x );

#ifdef __cplusplus
}
#endif
