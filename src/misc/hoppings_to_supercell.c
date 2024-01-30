/* Copyright (C) 2024 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "hoppings_to_supercell.h"

static inline index_t pmod( index_t i, index_t n ) {
    return (n + (i % n)) % n;
}

static inline index_t pdiv( index_t i, index_t n ) {
    return (i - pmod(i,n))/n;
}

rs_hopping_t* rs_hopping_to_supercell( const rs_hopping_t* h, index_t* n_hop,
        index_t no, const index_t nr[3] ) {
    index_t Nr = nr[0]*nr[1]*nr[2];

    index_t result_cap = *n_hop*Nr;
    rs_hopping_t* result = calloc(result_cap, sizeof(rs_hopping_t));
    index_t result_sz = 0;
    #define result_push_back( ... ) { \
        if (result_sz >= result_cap-1) \
            result = realloc(result, sizeof(rs_hopping_t)*(result_cap*=2)); \
        result[result_sz++] = __VA_ARGS__; \
    }

    for (index_t i=0; i<*n_hop; ++i) {
        const rs_hopping_t* hp = h + i;

        index_t o1 = hp->o1,
                o2 = hp->o2,
                s1 = hp->s1,
                s2 = hp->s2;
        complex128_t t = hp->t;
        index_t Rx = hp->R[0], Ry = hp->R[1], Rz = hp->R[2];

        for (index_t r1=0; r1<nr[0]; ++r1)
        for (index_t r2=0; r2<nr[1]; ++r2)
        for (index_t r3=0; r3<nr[2]; ++r3) {

            index_t r1r = pmod(Rx + r1, nr[0]),
                    r1R = pdiv(Rx + r1, nr[0]);
            index_t r2r = pmod(Ry + r2, nr[1]),
                    r2R = pdiv(Ry + r2, nr[1]);
            index_t r3r = pmod(Rz + r3, nr[2]),
                    r3R = pdiv(Rz + r3, nr[2]);

            index_t rr1 = r1 * nr[1]*nr[2] + r2 * nr[2] + r3;
            index_t rr2 = r1r * nr[1]*nr[2] + r2r * nr[2] + r3r;

            index_t po1 = rr1 * no + o1,
                    po2 = rr2 * no + o2;

            result_push_back( (rs_hopping_t){ {r1R, r2R, r3R}, po1, po2, s1, s2, t } );
        }
    }

    *n_hop = result_sz;
    return result;
}

