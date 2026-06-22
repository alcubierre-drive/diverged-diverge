/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdlib>
#include <climits>
#include <cstdint>
#else // __cplusplus
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#endif // __cplusplus

#define BITS_MOD(x) ((x)&63)
#define BITS_DIV(x) ((x)>>6)

typedef uint64_t cbit;
typedef uint64_t cbytes_t;

#if defined(__has_builtin)
    #if __has_builtin(__builtin_popcountll)
        #define HAVE_BUILTIN_POPCOUNTLL 1
    #endif
#elif defined(__GNUC__) && ((__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
    #define HAVE_BUILTIN_POPCOUNTLL 1
#endif

#ifdef __CUDACC__
#define BITARY_HOST_DEV __host__ __device__
#define HAVE_BUILTIN_POPCOUNTLL 0
#else
#define BITARY_HOST_DEV
#endif

#if defined(HAVE_BUILTIN_POPCOUNTLL)
    #define DVG_POPCOUNT64(x) __builtin_popcountll(x)
#else
    static inline int generic_popcount64(uint64_t x) {
        x -= (x >> 1) & 0x5555555555555555ULL;
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
        return (int)((x * 0x0101010101010101ULL) >> 56);
    }
    #define DVG_POPCOUNT64(x) generic_popcount64(x)
#endif

/** set bit number ``bit`` of byte ``b`` to one */
BITARY_HOST_DEV static inline void bit_set( cbytes_t* b, cbit bit ) {
    *b |= 1ULL << bit;
}

/** set bit number ``bit`` of byte ``b`` to zero */
BITARY_HOST_DEV static inline void bit_unset( cbytes_t* b, cbit bit ) {
    *b &= ~(1ULL << bit);
}

/** toggle bit number ``bit`` of byte ``b`` */
BITARY_HOST_DEV static inline void bit_toggle( cbytes_t* b, cbit bit ) {
    *b ^= 1ULL << bit;
}

/** get bit number ``bit`` of byte ``b`` */
BITARY_HOST_DEV static inline cbit bit_get( const cbytes_t* b, cbit bit ) {
    return (*b >> bit) & 1ULL;
}

/** get number of bytes required to represent ``nbit`` contiguous bits in memory */
BITARY_HOST_DEV static inline size_t bitary_nwords( size_t nbit ) {
    return BITS_DIV(nbit)+(BITS_MOD(nbit) != 0);
}

/**
 * allocate a bit array with at least ``nbit`` bits, all set to zero. The
 * returnet memory must be free'd with ``free`` or :c:func:`diverge_mem_free`.
 */
BITARY_HOST_DEV static inline cbytes_t* bitary_alloc( size_t nbit ) {
    cbytes_t* result = (cbytes_t*)calloc(bitary_nwords(nbit), sizeof(cbytes_t));
    return result;
}

/** set the :math:`n`-th bit (``bitidx``) of the bit array ``ary`` to one */
BITARY_HOST_DEV static inline void bitary_set( cbytes_t* ary, size_t bitidx ) {
    bit_set( ary + BITS_DIV(bitidx), BITS_MOD(bitidx));
}
// required for omp atomic operation
#define BITARY_SET( ary, bitidx ) \
    (ary)[BITS_DIV(bitidx)] |= (1ULL << BITS_MOD(bitidx))

/** set the :math:`n`-th bit (``bitidx``) of the bit array ``ary`` to zero */
BITARY_HOST_DEV static inline void bitary_unset( cbytes_t* ary, size_t bitidx ) {
    bit_unset( ary + BITS_DIV(bitidx), BITS_MOD(bitidx));
}

/** toggle the :math:`n`-th bit (``bitidx``) of the bit array ``ary`` */
BITARY_HOST_DEV static inline void bitary_toggle( cbytes_t* ary, size_t bitidx ) {
    bit_toggle( ary + BITS_DIV(bitidx), BITS_MOD(bitidx));
}

/** get the :math:`n`-th bit (``bitidx``) of the bit array ``ary`` */
BITARY_HOST_DEV static inline cbit bitary_get( const cbytes_t* ary, size_t bitidx ) {
    return bit_get( ary + BITS_DIV(bitidx), BITS_MOD(bitidx));
}

/** set the first ``nbit`` bits of the bit array ``ary`` to one */
BITARY_HOST_DEV static inline void bitary_set_all( cbytes_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_set( ary, i );
}

/** set the first ``nbit`` bits of the bit array ``ary`` to zero */
BITARY_HOST_DEV static inline void bitary_unset_all( cbytes_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_unset( ary, i );
}

/** toggle the first ``nbit`` bits of the bit array ``ary`` */
BITARY_HOST_DEV static inline void bitary_toggle_all( cbytes_t* ary, size_t nbit ) {
    for (size_t i=0; i<nbit; ++i) bitary_toggle( ary, i );
}

BITARY_HOST_DEV static inline size_t bitary_count( cbytes_t* ary, size_t nbit ) {
    const size_t nw = bitary_nwords(nbit);
    size_t cnt = 0;
    for (size_t w=0; w<nw; ++w) cnt += DVG_POPCOUNT64(ary[w]);
    return cnt;
}

BITARY_HOST_DEV static inline size_t bitary_count_parallel( cbytes_t* ary, size_t nbit, int nthr ) {
    const size_t nw = bitary_nwords(nbit);
    size_t cnt = 0;
    #pragma omp parallel for num_threads(nthr) reduction(+:cnt)
    for (size_t w=0; w<nw; ++w) cnt += DVG_POPCOUNT64(ary[w]);
    return cnt;
}
