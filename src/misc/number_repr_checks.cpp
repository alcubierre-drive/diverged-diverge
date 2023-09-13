/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "number_repr_checks.h"
#include <cstdint>
#include <limits>
#include <climits>

bool check_iec_double( void ) {
    return std::numeric_limits<double>::is_iec559;
}

bool check_iec_float( void ) {
    return std::numeric_limits<float>::is_iec559;
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
