/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "read_W90.h"
#include "../diverge_model.h"
#include <fstream>
#include <string>
#include <cassert>

using std::string;
using std::getline;

static inline void unravel( index_t idx, index_t no, index_t ns, bool outer, index_t* o, index_t* s ) {
    *s = outer ? idx / no : idx % ns;
    *o = outer ? idx % no : idx / ns;
}

vector<rs_hopping_t> diverge_read_W90( const char* fname, index_t n_spin, index_t* n_orb_out ) {
    std::ifstream input(fname, std::ios::in);

    if (!(input.is_open())) {
        mpi_err_printf("failed to open W90 file at %s \n", fname);
        return {};
    }

    bool swap_orbitals = false;
    if (n_orb_out) if (~(*n_orb_out) == 0) {
        mpi_wrn_printf( "read_W90: not swapping. deprecation. see docs if you _must_ swap orbitals.\n" );
    }
    if (n_orb_out) if (*n_orb_out == DIVERGE_W90_READER_SWAP_ORBITALS) {
        mpi_log_printf( "read_W90: swapping orbital order\n" );
        swap_orbitals = true;
    }

    int i = 0;
    int weight_elem = 0;
    string line;
    bool outer = n_spin>=0 ? true:false;
    n_spin = n_spin ? std::abs(n_spin) : 1;
    index_t n_orb = 0;

    vector<double> weights;
    vector<rs_hopping_t> hop_elem;
    index_t len = 0;
    int current_norm = 0;
    index_t R_cmp[3] = {0,0,0};
    rs_hopping_t helper = {{0,0,0}, 0,0, 0,0, 0.};
    bool set_r = false;
    while (getline(input,line)) {
        std::istringstream iss(line);
        string s;
        if (i == 0) { // skip first line
        } else if (i == 1) { // number of orbitals + spins
            while (getline( iss, s, ' ' ))
                if (s.length() > 0) n_orb = stol(s)/n_spin;
        } else if (i==2) { // len
            while (getline( iss, s, ' ' ))
                if (s.length() > 0) len = stol(s);
            mpi_vrb_printf( "W90 #Rvecs=%d, n_orb=%d, n_spin=%d\n", len,n_orb,n_spin );
            hop_elem.reserve( len * POW2( n_orb * n_spin ) );
            weights.resize( len );
            memset( weights.data(), 0, sizeof(double)*weights.size() );
        } else if ( i > 2 && i < (3+ceil(1.*(len)/15.0)) ) {
            // for each real space vector the corresponding weight,
            // maximally 15 in each line
            while (getline( iss, s, ' ' ))
                if (s.length() > 0) weights[weight_elem++] = stol(s);
        } else {
            // hoppings
            // r_1, r_2, r_3, o_1, o_2, real(t), imag(t)
            unsigned current_elem = 0;
            assert(n_orb != 0);
            while (getline( iss, s, ' ' )) {
                if (s.length() > 0) {
                    // W90 is fortran ordered, diverge is C ordered!
                    switch (current_elem) {
                        case 0: case 1: case 2: helper.R[current_elem] = stol(s); break;
                        case 3: unravel(stol(s)-1, n_orb, n_spin, outer, &helper.o1, &helper.s1); break;
                        case 4: unravel(stol(s)-1, n_orb, n_spin, outer, &helper.o2, &helper.s2); break;
                        case 5: helper.t = stod(s); break;
                        case 6: helper.t += I128*stod(s); break;
                        default: break;
                    }
                    current_elem++;
                }
            }
            if (!set_r) {
                R_cmp[0] = helper.R[0],
                R_cmp[1] = helper.R[1],
                R_cmp[2] = helper.R[2];
                helper.t /= (double)weights[current_norm];
                set_r = true;
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
            hop_elem.push_back(helper);
        }
        i++;
    }
    input.close();

    if (n_orb_out != NULL)
        *n_orb_out = n_orb;

    if (swap_orbitals) for (rs_hopping_t& h: hop_elem) {
        XORSWAP( h.o1, h.o2 );
        XORSWAP( h.s1, h.s2 );
    }

    return hop_elem;
}

rs_hopping_t* diverge_read_W90_C( const char* fname, index_t n_spin, index_t* len, index_t* n_orb ) {
    vector<rs_hopping_t> res = diverge_read_W90( fname, n_spin, n_orb );
    if (res.size() == 0)
        return NULL;
    *len = res.size();
    rs_hopping_t* result = (rs_hopping_t*)malloc(sizeof(rs_hopping_t)*res.size());
    memcpy( result, res.data(), sizeof(rs_hopping_t)*res.size() );
    return result;
}
