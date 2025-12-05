/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

typedef union { uint64_t bits; double f; } dbl_nanbox_u;
static const uint64_t DBL_NAN_MASK = 0x7ff8000000000000ULL,
                      DBL_NAN_LOAD = 0x0007FFFFFFFFFFFFULL;
static inline uint64_t _qnan_make_nan_payload( uint64_t payload ) {
    return DBL_NAN_MASK | (payload & DBL_NAN_LOAD);
}

/** generate a quiet NaN that has a 51bit unsigned integer payload */
static inline double qnan_gen( uint64_t payload ) {
    dbl_nanbox_u u;
    u.bits = _qnan_make_nan_payload(payload);
    return u.f;
}

/** check whether a double holds a quiet NaN (same as :c:func:`fnan_isnan`) */
static inline int qnan_isnan( double x ) {
    dbl_nanbox_u u;
    u.f = x;
    return (u.bits & DBL_NAN_MASK) == DBL_NAN_MASK;
}

/** get the 51bit unsigned integer payload from a quiet NaN */
static inline uint64_t qnan_get( double x ) {
    dbl_nanbox_u u;
    u.f = x;
    return u.bits & DBL_NAN_LOAD;
}

/** generate a quiet NaN that has a 51bit floating point payload (a double truncated in precision) */
static inline double fnan_gen( double payload ) {
    dbl_nanbox_u u;
    u.f = payload;
    u.bits = DBL_NAN_MASK | ((u.bits >> 13) & DBL_NAN_LOAD);
    return u.f;
}

/** get the precision truncated double (51bit floating point) payload from a quiet NaN */
static inline double fnan_get( double x ) {
    dbl_nanbox_u u;
    u.f = x;
    u.bits = (u.bits & DBL_NAN_LOAD) << 13;
    return u.f;
}

/** check whether a double holds a quiet NaN (same as :c:func:`qnan_isnan`) */
static inline int fnan_isnan( double x ) { return qnan_isnan(x); }

typedef union { uint32_t bits; float f; } nanbox_u;
static const uint32_t NAN_MASK = 0x7FC00000u,
                      NAN_LOAD = 0x003FFFFFu;
static inline uint32_t _qnanf_make_nan_payload( uint32_t payload ) {
    return NAN_MASK | (payload & NAN_LOAD);
}

/** generate a quiet NaNf that has a 22bit unsigned integer payload */
static inline float qnanf_gen( uint32_t payload ) {
    nanbox_u u;
    u.bits = _qnanf_make_nan_payload(payload);
    return u.f;
}

/** check whether a float holds a quiet NaNf (same as :c:func:`fnanf_isnsn`) */
static inline int qnanf_isnan( float x ) {
    nanbox_u u;
    u.f = x;
    return (u.bits & NAN_MASK) == NAN_MASK;
}

/** get the 22bit unsigned integer payload from a quiet NaNf */
static inline uint32_t qnanf_get( float x ) {
    nanbox_u u;
    u.f = x;
    return u.bits & NAN_LOAD;
}

/** generate a quiet NaNf that has a 22bit floating point payload (a float truncated in precision) */
static inline float fnanf_gen( float payload ) {
    nanbox_u u;
    u.f = payload;
    u.bits = NAN_MASK | ((u.bits >> 10) & NAN_LOAD);
    return u.f;
}

/** get the precision truncated float (22bit floating point) payload from a quiet NaN */
static inline float fnanf_get( float x ) {
    nanbox_u u;
    u.f = x;
    u.bits = (u.bits & NAN_LOAD) << 10;
    return u.f;
}

/** check whether a float holds a quiet NaNf (same as :c:func:`qnanf_isnsn`) */
static inline int fnanf_isnan( float x ) { return qnanf_isnan(x); }

#ifdef __cplusplus
}
#endif
