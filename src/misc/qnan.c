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
int qnan_isnan( double qnan ) {
    (void)qnan;
    return 0;
}

#else // NAN

static const uint64_t NAN_MASK =     0xFFFF000000000000;
static const uint64_t PAYLOAD_MASK = 0x00000000FFFFFFFF;

typedef union {
    uint64_t i;
    double d;
} double_u;

double qnan_gen( uint32_t payload_ ) {
    uint64_t payload = payload_;
    double_u unan = {.i = NAN_MASK};
    unan.i |= PAYLOAD_MASK & payload;
    return unan.d;
}

uint32_t qnan_get( double nan_ ) {
    double_u unan = {.d = nan_};
    uint64_t payload = (unan.i & PAYLOAD_MASK);
    return payload;
}

int qnan_isnan( double nan_ ) {
    double_u qnan = {.d = nan_};
    return ((qnan.i & NAN_MASK) == NAN_MASK);
}

#endif // NAN
