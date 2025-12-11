/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

double      diverge_qnan_gen( uint64_t payload );
uint64_t    diverge_qnan_get( double x );
int         diverge_qnan_isnan( double x );

double      diverge_fnan_gen( double payload );
double      diverge_fnan_get( double x );
int         diverge_fnan_isnan( double x );

float       diverge_qnanf_gen( uint32_t payload );
uint32_t    diverge_qnanf_get( float x );
int         diverge_qnanf_isnan( float x );

float       diverge_fnanf_gen( float payload );
float       diverge_fnanf_get( float x );
int         diverge_fnanf_isnan( float x );

#ifdef __cplusplus
}
#endif
