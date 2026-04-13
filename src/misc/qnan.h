/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <cassert>
extern "C" {
#else
#include <stdint.h>
#include <assert.h>
#endif

#define NAN_BITSHIFT 13
#define NANF_BITSHIFT 10

#ifdef DIVERGE_QNAN_DISABLE_CLANG_OPTS
_Pragma("clang optimize off")
#endif

typedef union { uint64_t bits; double f; } qnan_u;
static const uint64_t NAN_MASK = ((~(0ULL)) >> (64-NAN_BITSHIFT)) << (64-NAN_BITSHIFT);
static const uint64_t NAN_LOAD = ~NAN_MASK;

/** generate a quiet NaN that has a 51bit unsigned integer payload */
static inline double qnan_gen( uint64_t payload ) {
    assert( (payload & NAN_MASK) == 0 );
    qnan_u u;
    u.bits = NAN_MASK | (payload & NAN_LOAD);
    return u.f;
}

/** check whether a double holds a quiet NaN (same as :c:func:`fnan_isnan`) */
static inline int qnan_isnan( double x ) {
    qnan_u u;
    u.f = x;
    return (u.bits & NAN_MASK) == NAN_MASK;
}

/** get the 51bit unsigned integer payload from a quiet NaN */
static inline uint64_t qnan_get( double x ) {
    qnan_u u;
    u.f = x;
    return u.bits & NAN_LOAD;
}

/** generate a quiet NaN that has a 51bit floating point payload (a double truncated in precision) */
static inline double fnan_gen( double payload ) {
    qnan_u u;
    u.f = payload;
    u.bits = NAN_MASK | ((u.bits >> NAN_BITSHIFT) & NAN_LOAD);
    return u.f;
}

/** get the precision truncated double (51bit floating point) payload from a quiet NaN */
static inline double fnan_get( double x ) {
    qnan_u u;
    u.f = x;
    u.bits = (u.bits & NAN_LOAD) << NAN_BITSHIFT;
    return u.f;
}

/** check whether a double holds a quiet NaN (same as :c:func:`qnan_isnan`) */
static inline int fnan_isnan( double x ) { return qnan_isnan(x); }

typedef union { uint32_t bits; float f; } qnanf_u;
static const uint32_t NANF_MASK = ((~(0u)) >> (32-NANF_BITSHIFT)) << (32-NANF_BITSHIFT);
static const uint32_t NANF_LOAD = ~NANF_MASK;

/** generate a quiet NaNf that has a 22bit unsigned integer payload */
static inline float qnanf_gen( uint32_t payload ) {
    assert( (payload & NANF_MASK) == 0 );
    qnanf_u u;
    u.bits = NANF_MASK | (payload & NANF_LOAD);
    return u.f;
}

/** check whether a float holds a quiet NaNf (same as :c:func:`fnanf_isnsn`) */
static inline int qnanf_isnan( float x ) {
    qnanf_u u;
    u.f = x;
    return (u.bits & NANF_MASK) == NANF_MASK;
}

/** get the 22bit unsigned integer payload from a quiet NaNf */
static inline uint32_t qnanf_get( float x ) {
    qnanf_u u;
    u.f = x;
    return u.bits & NANF_LOAD;
}

/** generate a quiet NaNf that has a 22bit floating point payload (a float truncated in precision) */
static inline float fnanf_gen( float payload ) {
    qnanf_u u;
    u.f = payload;
    u.bits = NANF_MASK | ((u.bits >> NANF_BITSHIFT) & NANF_LOAD);
    return u.f;
}

/** get the precision truncated float (22bit floating point) payload from a quiet NaN */
static inline float fnanf_get( float x ) {
    qnanf_u u;
    u.f = x;
    u.bits = (u.bits & NANF_LOAD) << NANF_BITSHIFT;
    return u.f;
}

/** check whether a float holds a quiet NaNf (same as :c:func:`qnanf_isnsn`) */
static inline int fnanf_isnan( float x ) { return qnanf_isnan(x); }

#ifdef DIVERGE_QNAN_DISABLE_CLANG_OPTS
_Pragma("clang optimize on")
#endif

#ifdef __cplusplus
}
#endif
