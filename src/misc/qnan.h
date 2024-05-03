/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

double qnan_gen( uint32_t payload );
uint32_t qnan_get( double qnan );
int qnan_isnan( double qnan );

double fnan_gen( float payload );
float fnan_get( double fnan );
int fnan_isnan( double fnan );

#ifdef __cplusplus
}
#endif
