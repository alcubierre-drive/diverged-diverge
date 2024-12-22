/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "read_W90.h"
#include "../diverge_model.h"
#include <assert.h>

#ifndef W90_GETLINE_MAX
#define W90_GETLINE_MAX 8192
#endif

static inline index_t W90_getline( char* line, FILE* stream );
static inline void unravel( index_t idx, index_t no, index_t ns, bool outer, index_t* o, index_t* s );

rs_hopping_t* diverge_read_W90( const char* fname, index_t n_spin, index_t* n_hop, index_t* n_orb_out ) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        mpi_err_printf("failed to open W90 file at %s \n", fname);
        return NULL;
    }

    int swap_orbitals = 0;
    if (n_orb_out) if (~(*n_orb_out) == 0) {
        mpi_wrn_printf( "read_W90: not swapping. deprecation. see docs if you _must_ swap orbitals.\n" );
    }
    if (n_orb_out) if (*n_orb_out == DIVERGE_W90_READER_SWAP_ORBITALS) {
        mpi_log_printf( "read_W90: swapping orbital order\n" );
        swap_orbitals = 1;
    }

    int i = 0;
    int weight_elem = 0;
    char line[W90_GETLINE_MAX] = {0};
    int outer = (n_spin>=0);
    n_spin = n_spin ? labs(n_spin) : 1;
    index_t n_orb = 0;

    VECTOR_INIT( double, weights );
    VECTOR_INIT( rs_hopping_t, hop_elem );

    index_t len = 0;
    int current_norm = 0;
    index_t R_cmp[3] = {0};
    rs_hopping_t helper = {0};
    int set_r = 0;
    while (W90_getline(line, f) != -1) {
        char* s = strtok(line, " ");
        if (i == 0) { // skip first line
        } else if (i == 1) { // number of orbitals + spins
            while (s != NULL) {
                if (strlen(s) > 0) n_orb = atol(s)/n_spin;
                s = strtok(NULL, " ");
            }
        } else if (i == 2) { // len
            while (s != NULL) {
                if (strlen(s) > 0) len = atol(s);
                s = strtok(NULL, " ");
            }
            mpi_vrb_printf( "W90 #Rvecs=%d, n_orb=%d, n_spin=%d\n", len, n_orb, n_spin );
            VECTOR_RESERVE( hop_elem, len * POW2( n_orb*n_spin ) );
            VECTOR_RESIZE( weights, len );
            memset( weights, 0, sizeof(*weights) * weights_sz );
        } else if ( i > 2 && i < (3+ceil(1.*(len)/15.0)) ) {
            // for each real space vector the corresponding weight,
            // maximally 15 in each line
            while (s != NULL) {
                if (strlen(s) > 0) weights[weight_elem++] = atol(s);
                s = strtok(NULL, " ");
            }
        } else {
            // hoppings
            // r_1, r_2, r_3, o_1, o_2, real(t), imag(t)
            unsigned current_elem = 0;
            assert(n_orb != 0);
            while (s != NULL) {
                if (strlen(s) > 0) {
                    // W90 is fortran ordered, diverge is C ordered!
                    switch (current_elem) {
                        case 0: case 1: case 2: helper.R[current_elem] = atol(s); break;
                        case 3: unravel(atol(s)-1, n_orb, n_spin, outer, &helper.o1, &helper.s1); break;
                        case 4: unravel(atol(s)-1, n_orb, n_spin, outer, &helper.o2, &helper.s2); break;
                        case 5: helper.t = atof(s); break;
                        case 6: helper.t += I128*atof(s); break;
                        default: break;
                    }
                    current_elem++;
                }
                s = strtok(NULL, " ");
            }
            if (!set_r) {
                R_cmp[0] = helper.R[0],
                R_cmp[1] = helper.R[1],
                R_cmp[2] = helper.R[2];
                helper.t /= (double)weights[current_norm];
                set_r = 1;
            } else {
                if ((helper.R[0]-R_cmp[0]) == 0 && (helper.R[1]-R_cmp[1]) == 0 &&
                    (helper.R[2]-R_cmp[2]) == 0 ) {
                     helper.t /= (double)weights[current_norm];
                } else {
                    helper.t /= (double)weights[++current_norm];
                    R_cmp[0] = helper.R[0],
                    R_cmp[1] = helper.R[1],
                    R_cmp[2] = helper.R[2];
                }
            }
            VECTOR_PUSH_BACK( hop_elem, helper );
        }
        i++;
    }
    free( weights );
    fclose( f );
    VECTOR_SHRINK_TO_FIT( hop_elem );

    if (n_orb_out != NULL)
        *n_orb_out = n_orb;

    if (swap_orbitals) for (rs_hopping_t* h=hop_elem; h!=hop_elem+hop_elem_sz; ++h) {
        XORSWAP( h->o1, h->o2 );
        XORSWAP( h->s1, h->s2 );
    }

    if (n_hop) *n_hop = hop_elem_sz;
    return hop_elem;
}

rs_hopping_t* diverge_read_W90_C( const char* fname, index_t n_spin, index_t* n_hop, index_t* n_orb_out ) {
    return diverge_read_W90( fname, n_spin, n_hop, n_orb_out );
}

static inline index_t W90_getline( char* line, FILE* stream ) {
    if (!line) return -1;
    fgets(line, W90_GETLINE_MAX-1, stream);
    return feof(stream) ? -1 : (index_t)strlen(line);
}

static inline void unravel( index_t idx, index_t no, index_t ns, bool outer, index_t* o, index_t* s ) {
    *s = outer ? idx / no : idx % ns;
    *o = outer ? idx % no : idx / ns;
}

