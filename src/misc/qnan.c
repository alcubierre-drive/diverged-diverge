/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "qnan.h"
#include <math.h>
#include <string.h>

#ifndef NAN

double qnan_gen( uint32_t payload_ ) {
    (void)payload_;
    double res;
    memset( &res, 0xFF, sizeof(double) );
    return res;
}
uint32_t qnan_get( double nan_ ) {
    (void)nan_;
    return 0;
}

#else // NAN

#define QNAN_BITS 0x7FFFFFFFFFFFFULL

double qnan_gen( uint32_t payload_ ) {
    uint64_t payload = payload_;
    union { uint64_t i; double d; } unan = {.d = NAN};
    unan.i |= QNAN_BITS & payload;
    return unan.d;
}
uint32_t qnan_get( double nan_ ) {
    union { uint64_t i; double d; } unan = {.d = nan_};
    uint64_t payload = (unan.i & QNAN_BITS);
    return payload & UINT32_MAX;
}

#endif // NAN
