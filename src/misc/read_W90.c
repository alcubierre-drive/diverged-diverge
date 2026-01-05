/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "read_W90.h"
#include "../diverge_model.h"
#include "text_io.h"

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

static inline void unravel( index_t idx, index_t no, index_t ns, bool outer, index_t* o, index_t* s );

#ifndef DIVERGE_OLD_W90_READER

#define VRB_ASSERT( cond ) { \
    int err = !(cond); \
    assert( err == 0 ); \
    if (err) { mpi_err_printf( "%s\n", #cond ); } \
}
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

    int outer = (n_spin>=0);
    n_spin = n_spin ? labs(n_spin) : 1;

    char line[DIVERGE_GETLINE_MAX] = {0};
    int nread = 0;
    // comment line
    if (diverge_getline(line, f) == -1) {
        fclose( f );
        return NULL;
    }
    // n_orb*n_spin
    index_t n_orb_n_spin = 0;
    nread = fscanf( f, "%li\n", &n_orb_n_spin );
    VRB_ASSERT( nread == 1 );
    index_t n_orb = n_orb_n_spin/n_spin;
    if (n_orb_out) *n_orb_out = n_orb;

    index_t n_wght = 0;
    nread = fscanf( f, "%li\n", &n_wght );
    VRB_ASSERT( nread == 1 );

    double* weights = calloc( n_wght, sizeof*weights );
    for (index_t h=0; h<n_wght; ++h) {
        nread = fscanf( f, "%lf", weights+h );
        VRB_ASSERT( nread == 1 );
    }

    VECTOR_INIT( rs_hopping_t, hop )
    while (!feof(f)) {
        index_t O1 = 0, O2 = 0; double t[2] = {0}; index_t R[3] = {0};
        nread = fscanf( f, "%li %li %li %li %li %lf %lf\n", &R[0], &R[1], &R[2],
                &O1, &O2, &t[0], &t[1] );
        VRB_ASSERT( nread == 7 || nread == 0 );
        if (nread == 7) {
            rs_hopping_t tmp = { .R={R[0],R[1],R[2]}, .t=t[0]+I*t[1] };
            unravel( O1-1, n_orb, n_spin, outer, &tmp.o1, &tmp.s1 );
            unravel( O2-1, n_orb, n_spin, outer, &tmp.o2, &tmp.s2 );
            VECTOR_PUSH_BACK( hop, tmp );
        }
    }
    fclose( f );

    VECTOR_INIT( claV3ll_t, unique_R );
    for (index_t h=0; h<hop_sz; ++h) {
        int is_new_R = 0;
        if (unique_R_sz > 0) {
            claV3ll_t dR = claV3ll_sub( unique_R[unique_R_sz-1], claV3ll_map(hop[h].R) );
            is_new_R = (claV3ll_dot( dR, dR ) != 0);
        } else {
            is_new_R = 1;
        }
        if (is_new_R) VECTOR_PUSH_BACK( unique_R, claV3ll_map(hop[h].R) );
        hop[h].t /= (double)weights[unique_R_sz-1];
        if (swap_orbitals) {
            XORSWAP( hop[h].o1, hop[h].o2 );
            XORSWAP( hop[h].s1, hop[h].s2 );
        }
    }
    free( unique_R );
    free( weights );

    *n_hop = hop_sz;
    return hop;
}

#else // DIVERGE_OLD_W90_READER

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
    char line[DIVERGE_GETLINE_MAX] = {0};
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
    while (diverge_getline(line, f) != -1) {
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

#endif // DIVERGE_OLD_W90_READER

rs_hopping_t* diverge_read_W90_C( const char* fname, index_t n_spin, index_t* n_hop, index_t* n_orb_out ) {
    return diverge_read_W90( fname, n_spin, n_hop, n_orb_out );
}

static inline void unravel( index_t idx, index_t no, index_t ns, bool outer, index_t* o, index_t* s ) {
    *s = outer ? idx / no : idx % ns;
    *o = outer ? idx % no : idx / ns;
}

double* diverge_read_wout( const char* fname, double* lattice/*3,3*/, index_t* n_wan ) {
    FILE* f = fopen( fname, "r" );
    if (!f) {
        mpi_err_printf( "could not open file %s\n", fname );
        return NULL;
    }

    index_t nread = 0;
    char line[DIVERGE_GETLINE_MAX] = {0};
    int linenum = 0;

    int errors = 0;

    int system_lattice_vecs = 0;
    int final_state = 0;

    int system_lattice_vecs_ntimes = 0;
    int final_state_ntimes = 0;

    VECTOR_INIT( double, pos )

    while ((nread = diverge_getline(line, f)) != -1) {
        linenum++;

        char* l = diverge_chomp(line, nread);
        if (system_lattice_vecs == 0 && !strcmp(l,"------")) system_lattice_vecs++;
        else if (system_lattice_vecs == 1 && !strcmp(l,"SYSTEM")) system_lattice_vecs++;
        else if (system_lattice_vecs == 2 && !strcmp(l,"------")) system_lattice_vecs++;
        else if (system_lattice_vecs == 3 && !strcmp(l,"")) system_lattice_vecs++;
        else if (system_lattice_vecs == 4 && !strcmp(l,"Lattice Vectors (Ang)")) { system_lattice_vecs++;
            system_lattice_vecs_ntimes++;
        } else if (system_lattice_vecs >= 5 && system_lattice_vecs < 8) { system_lattice_vecs++;
            int aind = -1;
            double avec[3] = {0};
            int nscan = sscanf( l, "a_%i %lf %lf %lf", &aind, &avec[0], &avec[1], &avec[2] );
            int err = 0;
            if (nscan != 4) {
                mpi_err_printf( "scanned only %i/4 params in line %i\n", nscan, linenum );
                err++;
            }
            if (aind > 3 || aind < 1) {
                mpi_err_printf( "found wrong lattice vector index (%i)\n", aind );
                err++;
            }
            if (!err) memcpy( lattice + (aind-1)*3, avec, sizeof avec );
            else errors += err;
        } else system_lattice_vecs = 0;

        if (final_state == 0 && !strcmp(l,"Final State")) { final_state++;
            final_state_ntimes++;
            pos_sz = 0; // reset position vector length when we encounter a new final state
        }
        else if (final_state > 0 && !strncmp(l,"WF centre and spread",20)) { final_state++;
            int wind = -1;
            double wpos[3] = {0};
            double wspread = 0.0;
            int nscan = sscanf( l, "WF centre and spread %i ( %lf, %lf, %lf ) %lf",
                    &wind, &wpos[0], &wpos[1], &wpos[2], &wspread );
            int err = 0;
            if (nscan != 5) {
                mpi_err_printf( "scanned only %i/5 params in line %i\n", nscan, linenum );
                err++;
            }
            if (wind < 1) {
                mpi_err_printf( "found wrong WF index (%i)\n", wind );
                err++;
            }
            if (!err) { for (int d=0; d<3; ++d) VECTOR_PUSH_BACK( pos, wpos[d] ); }
            else errors += err;
        } else if (final_state > 0 && !strncmp(l,"Sum of centres and spreads",26)) final_state = 0;
        else final_state = 0;
    }

    if (errors)
        mpi_err_printf( "encountered %i errors\n", errors );

    mpi_vrb_printf( "encountered %i lattice vector and %i WF center sections\n",
            system_lattice_vecs_ntimes, final_state_ntimes );

    *n_wan = pos_sz/3;

    fclose( f );
    return pos;
}

typedef struct { claV3i_t R; int16_t O1; int16_t O2; } supvec_t;
#define vtree_key_t supvec_t

typedef struct { VECTOR_DECL(claV3i_t, v) } vec_claV3i_t;
#define vtree_val_t vec_claV3i_t

// we have the freedom to sort the btree any way we want, it's just a fancy dict
typedef struct { int64_t d[2]; } supvec_mask_t;
static inline int64_t compare_supvec_t( supvec_t a, supvec_t b ) {
    assert( sizeof(supvec_t) == sizeof(supvec_mask_t) );
    supvec_mask_t *ma = (supvec_mask_t*)&a, *mb = (supvec_mask_t*)&b;
    ma->d[0] -= mb->d[0]; ma->d[1] -= mb->d[1];
    return ma->d[1] ? ma->d[1] : ma->d[0];
}
#define vtree_key_cmp( a, b ) compare_supvec_t( a, b )

#include "vtree.h"

static xvtree_t* wsvec_read( const char* fname ) {
    FILE* f = fopen( fname, "r" );
    if (!f) {
        mpi_err_printf("failed to open wsvec file at %s\n", fname);
        return NULL;
    }

    // get the file size for a "reasonable" estimate
    fseek( f, 0, SEEK_END ); index_t file_size = ftell( f ); rewind( f );

    // ignore first line
    char line[DIVERGE_GETLINE_MAX] = {0};
    if (diverge_getline(line, f) == -1) {
        fclose( f );
        return NULL;
    }

    xvtree_t* x = xvtree_init( MAX(file_size/10, 1024) );
    while (!feof(f)) {
        supvec_t s = {0};
        int nread = fscanf( f, "%i %i %i %hi %hi\n", &s.R.v[0], &s.R.v[1], &s.R.v[2], &s.O1, &s.O2 );
        if (nread == 0) break;

        VRB_ASSERT( nread == 5 );
        index_t nv = -1;
        nread = fscanf( f, "%li\n", &nv );
        VRB_ASSERT( nread == 1 );
        VRB_ASSERT( nv > 0 );

        vec_claV3i_t supvecs = {0};
        VECTOR_RESERVE( supvecs.v, nv );
        for (index_t v=0; v<nv; ++v) {
            claV3i_t S = {0};
            nread = fscanf( f, "%i %i %i\n", &S.v[0], &S.v[1], &S.v[2] );
            VRB_ASSERT( nread == 3 );
            VECTOR_PUSH_BACK( supvecs.v, S );
        }

        s.O1--; s.O2--;
        vtree_node_t* v = xvtree_find_or_insert( x, s );
        if (v->val.v_sz == 0) v->val = supvecs;
    }
    fclose( f );
    return x;
}

void xvtree_free_iter( vtree_node_t* v, void* ctx ) { (void)ctx; free(v->val.v); }

rs_hopping_t* diverge_W90_apply_wsvec( rs_hopping_t* hop, index_t* p_n_hop,
        int spin_outer, const char* wsvec ) {
    index_t n_hop = *p_n_hop;
    index_t n_spin = 0;
    index_t n_orb = 0;
    for (index_t i=0; i<n_hop; ++i) {
        if (hop[i].s1 >= 0) n_spin = MAX(n_spin, hop[i].s1);
        if (hop[i].s2 >= 0) n_spin = MAX(n_spin, hop[i].s2);
        n_orb = MAX(n_orb, hop[i].o1);
        n_orb = MAX(n_orb, hop[i].o2);
    }
    n_orb++;
    n_spin++;
    mpi_log_printf( "found n_orb=%li, n_spin=%li\n", n_orb, n_spin );

    xvtree_t* sup = wsvec_read( wsvec );
    VECTOR_INIT_CAP( rs_hopping_t, new_hop, n_hop );
    for (index_t i=0; i<n_hop; ++i) {
        index_t s1 = MAX(hop[i].s1,0);
        index_t s2 = MAX(hop[i].s2,0);
        supvec_t s = { {{hop[i].R[0], hop[i].R[1], hop[i].R[2]}},
            spin_outer ? IDX2(s1,hop[i].o1, n_orb) : IDX2(hop[i].o1,s1, n_spin),
            spin_outer ? IDX2(s2,hop[i].o2, n_orb) : IDX2(hop[i].o2,s2, n_spin) };
        vtree_node_t* vs = xvtree_find( sup, s );
        if (!vs) {
            mpi_err_printf( "could not find hopping (%i,%i,%i @ %hi:%hi)\n",
                    s.R.v[0],s.R.v[1],s.R.v[2], s.O1,s.O2 );
        } else {
            index_t n_supvecs = vs->val.v_sz;
            for (index_t s=0; s<n_supvecs; ++s) {
                claV3i_t supvec = vs->val.v[s];
                rs_hopping_t h = hop[i];
                for (int d=0; d<3; ++d) h.R[d] += supvec.v[d];
                h.t /= (double)n_supvecs;
                VECTOR_PUSH_BACK( new_hop, h );
            }
        }
    }
    xvtree_iter( sup, &xvtree_free_iter, NULL );
    xvtree_free( sup );

    free( hop );
    *p_n_hop = new_hop_sz;
    return new_hop;
}

diverge_model_t* diverge_read_W90_model( const char* hr_dat, const char* wout,
        const char* wsvec, index_t n_spin ) {
    diverge_model_t* m = diverge_model_init();

    m->n_spin = n_spin != 0 ? labs(n_spin) : 1;
    m->SU2 = n_spin == 0;
    m->hop = diverge_read_W90( hr_dat, n_spin, &m->n_hop, &m->n_orb );

    index_t n_wan = 0;
    double* pos = diverge_read_wout( wout, m->lattice[0], &n_wan );
    if (n_wan != m->n_orb) mpi_err_printf( "found inconsistent n_orb(%li) and n_wan(%li)\n", m->n_orb, n_wan );
    if (pos) memcpy( m->positions[0], pos, sizeof(double)*n_wan*3 );
    free( pos );

    m->hop = diverge_W90_apply_wsvec( m->hop, &m->n_hop, n_spin>=0, wsvec );
    return m;
}
