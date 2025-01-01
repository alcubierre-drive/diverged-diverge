/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

bool check_iec_double( void );
bool check_iec_float( void );
bool check_little_endian( void );
bool check_8bit_char( void );

#ifdef __cplusplus
}
#endif
