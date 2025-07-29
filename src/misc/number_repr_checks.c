/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "number_repr_checks.h"
#include <stdint.h>
#include <limits.h>

bool check_64bit_double( void ) {
    return sizeof(double) == 8 && CHAR_BIT == 8;
}

bool check_32bit_float( void ) {
    return sizeof(float) == 4 && CHAR_BIT == 8;
}

bool check_little_endian( void ) {
    const int endian_check_int = 1;
    bool int_is_le = (*(char*)&endian_check_int) == 1;
    const int64_t endian_check_int64_t = 1;
    bool int64_t_is_le = (*(char*)&endian_check_int64_t) == 1;
    return int_is_le && int64_t_is_le;
}

bool check_8bit_char( void ) {
    return CHAR_BIT == 8;
}
