/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdlib>
#include <climits>
#else // __cplusplus
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#endif // __cplusplus

#if (CHAR_BIT == 8)
#define CHAR_MOD(x) ((x)&7)
#else
#define CHAR_MOD(x) ((x)%CHAR_BIT)
#endif

#if (CHAR_BIT == 8)
#define CHAR_DIV(x) ((x)>>3)
#else
#define CHAR_DIV(x) ((x)/CHAR_BIT)
#endif

typedef unsigned char cbit;
typedef unsigned char cbyte_t;

#ifdef __CUDACC__
#define BITARY_HOST_DEV __host__ __device__
#else
#define BITARY_HOST_DEV
#endif

BITARY_HOST_DEV static inline void bit_set( cbyte_t* b, cbit bit ) {
    *b |= 1 << bit;
}

BITARY_HOST_DEV static inline void bit_unset( cbyte_t* b, cbit bit ) {
    *b &= ~(1 << bit);
}

BITARY_HOST_DEV static inline void bit_toggle( cbyte_t* b, cbit bit ) {
    *b ^= 1 << bit;
}

BITARY_HOST_DEV static inline cbit bit_get( const cbyte_t* b, cbit bit ) {
    return (*b >> bit) & 1;
}

BITARY_HOST_DEV static inline size_t bitary_nbytes( size_t nbit ) {
    return CHAR_DIV(nbit)+(CHAR_MOD(nbit) != 0);
}

BITARY_HOST_DEV static inline cbyte_t* bitary_alloc( size_t nbit ) {
    cbyte_t* result = (cbyte_t*)calloc(bitary_nbytes(nbit), sizeof(cbyte_t));
    return result;
}

BITARY_HOST_DEV static inline void bitary_set( cbyte_t* ary, size_t bitidx ) {
    bit_set( ary + CHAR_DIV(bitidx), CHAR_MOD(bitidx));
}

BITARY_HOST_DEV static inline void bitary_unset( cbyte_t* ary, size_t bitidx ) {
    bit_unset( ary + CHAR_DIV(bitidx), CHAR_MOD(bitidx));
}

BITARY_HOST_DEV static inline void bitary_toggle( cbyte_t* ary, size_t bitidx ) {
    bit_toggle( ary + CHAR_DIV(bitidx), CHAR_MOD(bitidx));
}

BITARY_HOST_DEV static inline cbit bitary_get( const cbyte_t* ary, size_t bitidx ) {
    return bit_get( ary + CHAR_DIV(bitidx), CHAR_MOD(bitidx));
}

BITARY_HOST_DEV static inline void bitary_set_all( cbyte_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_set( ary, i );
}

BITARY_HOST_DEV static inline void bitary_unset_all( cbyte_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_unset( ary, i );
}

BITARY_HOST_DEV static inline void bitary_toggle_all( cbyte_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_toggle( ary, i );
}
