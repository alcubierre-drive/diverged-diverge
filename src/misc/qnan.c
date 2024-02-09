/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "qnan.h"
#include <string.h>

static const uint64_t NAN_MASK = 0xFFFF000000000000;
static const uint32_t NOT_A_NAN = 0xFFFFFFFF;

int qnan_isnan( double qnan ) {
    uint64_t unan = 0;
    memcpy( &unan, &qnan, sizeof(unan) );
    return ((unan & NAN_MASK) == NAN_MASK);
}

double qnan_gen( uint32_t payload ) {
    double result = 0;
    memcpy( &result, &NAN_MASK, sizeof(NAN_MASK) );
    memcpy( &result, &payload, sizeof(payload) );
    return result;
}

uint32_t qnan_get( double qnan ) {
    if (!qnan_isnan( qnan ))
        return NOT_A_NAN;
    uint32_t result = 0;
    memcpy( &result, &qnan, sizeof(result) );
    return result;
}
