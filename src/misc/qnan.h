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

#ifdef __cplusplus
}
#endif
