/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

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

/** set bit number ``bit`` of byte ``b`` to one */
BITARY_HOST_DEV static inline void bit_set( cbyte_t* b, cbit bit ) {
    *b |= 1 << bit;
}

/** set bit number ``bit`` of byte ``b`` to zero */
BITARY_HOST_DEV static inline void bit_unset( cbyte_t* b, cbit bit ) {
    *b &= ~(1 << bit);
}

/** toggle bit number ``bit`` of byte ``b`` */
BITARY_HOST_DEV static inline void bit_toggle( cbyte_t* b, cbit bit ) {
    *b ^= 1 << bit;
}

/** get bit number ``bit`` of byte ``b`` */
BITARY_HOST_DEV static inline cbit bit_get( const cbyte_t* b, cbit bit ) {
    return (*b >> bit) & 1;
}

/** get number of bytes required to represent ``nbit`` contiguous bits in memory */
BITARY_HOST_DEV static inline size_t bitary_nbytes( size_t nbit ) {
    return CHAR_DIV(nbit)+(CHAR_MOD(nbit) != 0);
}

/**
 * allocate a bit array with at least ``nbit`` bits, all set to zero. The
 * returnet memory must be free'd with ``free`` or :c:func:`diverge_mem_free`.
 */
BITARY_HOST_DEV static inline cbyte_t* bitary_alloc( size_t nbit ) {
    cbyte_t* result = (cbyte_t*)calloc(bitary_nbytes(nbit), sizeof(cbyte_t));
    return result;
}

/** set the :math:`n`-th bit (``bitidx``) of the bit array ``ary`` to one */
BITARY_HOST_DEV static inline void bitary_set( cbyte_t* ary, size_t bitidx ) {
    bit_set( ary + CHAR_DIV(bitidx), CHAR_MOD(bitidx));
}

/** set the :math:`n`-th bit (``bitidx``) of the bit array ``ary`` to zero */
BITARY_HOST_DEV static inline void bitary_unset( cbyte_t* ary, size_t bitidx ) {
    bit_unset( ary + CHAR_DIV(bitidx), CHAR_MOD(bitidx));
}

/** toggle the :math:`n`-th bit (``bitidx``) of the bit array ``ary`` */
BITARY_HOST_DEV static inline void bitary_toggle( cbyte_t* ary, size_t bitidx ) {
    bit_toggle( ary + CHAR_DIV(bitidx), CHAR_MOD(bitidx));
}

/** get the :math:`n`-th bit (``bitidx``) of the bit array ``ary`` */
BITARY_HOST_DEV static inline cbit bitary_get( const cbyte_t* ary, size_t bitidx ) {
    return bit_get( ary + CHAR_DIV(bitidx), CHAR_MOD(bitidx));
}

/** set the first ``nbit`` bits of the bit array ``ary`` to one */
BITARY_HOST_DEV static inline void bitary_set_all( cbyte_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_set( ary, i );
}

/** set the first ``nbit`` bits of the bit array ``ary`` to zero */
BITARY_HOST_DEV static inline void bitary_unset_all( cbyte_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_unset( ary, i );
}

/** toggle the first ``nbit`` bits of the bit array ``ary`` */
BITARY_HOST_DEV static inline void bitary_toggle_all( cbyte_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_toggle( ary, i );
}
