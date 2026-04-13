/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include <stdint.h>

#define MT64_NN 312
#define MT64_MM 156

#define MT64_MATRIX_A 0xB5026F5AA96619E9ULL
#define MT64_UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define MT64_LM 0x7FFFFFFFULL         /* Least significant 31 bits */

typedef struct {
    uint64_t mt[MT64_NN];
    int index;
} mt19937_64_t;

static inline mt19937_64_t mt64_init(uint64_t seed) {
    mt19937_64_t rng = { .mt={seed,}, .index=MT64_NN };
    for (int i=1; i<MT64_NN; ++i)
        rng.mt[i] = 6364136223846793005ULL * (rng.mt[i-1] ^ (rng.mt[i-1] >> 62)) + i;
    return rng;
}

static inline uint64_t mt64_next(mt19937_64_t *rng) {
    uint64_t x;
    static const uint64_t mag01[2] = {0ULL, MT64_MATRIX_A};

    if (rng->index >= MT64_NN) {
        int i;

        for (i=0; i<MT64_NN-MT64_MM; ++i) {
            x = (rng->mt[i] & MT64_UM) | (rng->mt[i+1] & MT64_LM);
            rng->mt[i] = rng->mt[i+MT64_MM] ^ (x >> 1) ^ mag01[x & 1ULL];
        }
        for (; i<MT64_NN-1; ++i) {
            x = (rng->mt[i] & MT64_UM) | (rng->mt[i+1] & MT64_LM);
            rng->mt[i] = rng->mt[i+(MT64_MM-MT64_NN)] ^ (x >> 1) ^ mag01[x & 1ULL];
        }
        x = (rng->mt[MT64_NN-1] & MT64_UM) | (rng->mt[0] & MT64_LM);
        rng->mt[MT64_NN-1] = rng->mt[MT64_MM-1] ^ (x >> 1) ^ mag01[x & 1ULL];

        rng->index = 0;
    }

    x = rng->mt[rng->index++];

    /* Tempering */
    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
}

static inline double mt64_next_dbl(mt19937_64_t *rng) {
    return (double)mt64_next(rng)/(double)UINT64_MAX;
}
