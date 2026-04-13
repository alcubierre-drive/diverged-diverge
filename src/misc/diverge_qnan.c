/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_qnan.h"
// just a wrapper around qnan.h (inline, static)
#include "qnan.h"

double diverge_qnan_gen( uint64_t payload ) { return qnan_gen( payload ); }
uint64_t diverge_qnan_get( double x ) { return qnan_get( x ); }
int diverge_qnan_isnan( double x ) { return qnan_isnan( x ); }
double diverge_fnan_gen( double payload ) { return fnan_gen( payload ); }
double diverge_fnan_get( double x ) { return fnan_get( x ); }
int diverge_fnan_isnan( double x ) { return fnan_isnan( x ); }
float diverge_qnanf_gen( uint32_t payload ) { return qnanf_gen( payload ); }
uint32_t diverge_qnanf_get( float x ) { return qnanf_get( x ); }
int diverge_qnanf_isnan( float x ) { return qnanf_isnan( x ); }
float diverge_fnanf_gen( float payload ) { return fnanf_gen( payload ); }
float diverge_fnanf_get( float x ) { return fnanf_get( x ); }
int diverge_fnanf_isnan( float x ) { return fnanf_isnan( x ); }
