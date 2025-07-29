/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <cstring>
extern "C" {
#else
#include <stdint.h>
#include <string.h>
#endif

static const uint64_t NAN_MASK = 0xFFFF000000000000;
static const uint32_t NOT_A_NAN = 0xFFFFFFFF;

/** check whether a double holds a quiet NaN (same as :c:func:`fnan_isnan`) */
static inline int qnan_isnan( double qnan ) {
    uint64_t unan = 0;
    memcpy( &unan, &qnan, sizeof(unan) );
    return ((unan & NAN_MASK) == NAN_MASK);
}

/** generate a quiet NaN that has a 32bit unsigned integer payload (see :c:func:`fnan_gen`) */
static inline double qnan_gen( uint32_t payload ) {
    double result = 0;
    memcpy( &result, &NAN_MASK, sizeof(NAN_MASK) );
    memcpy( &result, &payload, sizeof(payload) );
    return result;
}

/** get the 32bit unsigned integer payload from a quiete NaN (see :c:func:`fnan_get`) */
static inline uint32_t qnan_get( double qnan ) {
    if (!qnan_isnan( qnan ))
        return NOT_A_NAN;
    uint32_t result = 0;
    memcpy( &result, &qnan, sizeof(result) );
    return result;
}

/** check whether a double holds a quiet NaN (same as :c:func:`qnan_isnan`) */
static inline int fnan_isnan( double fnan ) {
    uint64_t unan = 0;
    memcpy( &unan, &fnan, sizeof(unan) );
    return ((unan & NAN_MASK) == NAN_MASK);
}

/** generate a quiet NaN that has a 32bit floating point payload (see :c:func:`qnan_gen`) */
static inline double fnan_gen( float payload ) {
    double result = 0;
    memcpy( &result, &NAN_MASK, sizeof(NAN_MASK) );
    memcpy( &result, &payload, sizeof(payload) );
    return result;
}

/** get the 32bit floating point payload from a quiete NaN (see :c:func:`qnan_get`) */
static inline float fnan_get( double fnan ) {
    float result = 0;
    if (!fnan_isnan( fnan ))
        memcpy( &result, &NOT_A_NAN, sizeof(result) );
    else
        memcpy( &result, &fnan, sizeof(result) );
    return result;
}

#ifdef __cplusplus
}
#endif
