/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "hoppings_to_supercell.h"
#include "../diverge_model.h"
#include "mpi_functions.h"
#include <assert.h>

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

typedef struct {
    index_t N;
    claV3i_t N1, N2, N3;
    claV3i_t B1, B2, B3;

    claV3i_t* cells;
    index_t n_cells;
} supercell_t;

#define SGN( i ) ((i)>0 ? 1 : -1)

static void supercell_del( supercell_t* s ) {
    free( s->cells );
    free( s );
}

static supercell_t* supercell_gen( int* N1, int* N2, int* N3 ) {
    supercell_t* s = calloc(1, sizeof(supercell_t));
    s->N1 = claV3i_map( N1 );
    s->N2 = claV3i_map( N2 );
    s->N3 = claV3i_map( N3 );

    s->N = claV3i_dot( s->N1, claV3i_cross( s->N2, s->N3 ) );

    s->B1 = claV3i_scale( claV3i_cross( s->N2, s->N3 ), SGN(s->N) );
    s->B2 = claV3i_scale( claV3i_cross( s->N3, s->N1 ), SGN(s->N) );
    s->B3 = claV3i_scale( claV3i_cross( s->N1, s->N2 ), SGN(s->N) );

    s->N = labs(s->N);

    index_t cells_cap = 1;
    s->cells = calloc(cells_cap, sizeof(claV3i_t));
    s->n_cells = 0;
    #define cells_push( ... ) { \
        if (s->n_cells >= cells_cap) { \
            s->cells = realloc(s->cells, (cells_cap*=2) * sizeof(claV3i_t)); \
        } \
        s->cells[s->n_cells++] = __VA_ARGS__; \
    }

    claV3i_t corners[2][2][2] = {0};
    for (int n1=0; n1<2; ++n1) for (int n2=0; n2<2; ++n2) for (int n3=0; n3<2; ++n3)
        corners[n1][n2][n3] = claV3i_add( claV3i_add( claV3i_scale(s->N1,n1), claV3i_scale(s->N2,n2) ), claV3i_scale(s->N3,n3) );

    int n_lower[3] = {0}, n_upper[3] = {0};
    for (int n1=0; n1<2; ++n1) for (int n2=0; n2<2; ++n2) for (int n3=0; n3<2; ++n3) {
        for (int d=0; d<3; ++d) {
            n_lower[d] = MIN( n_lower[d], corners[n1][n2][n3].v[d] );
            n_upper[d] = MAX( n_upper[d], corners[n1][n2][n3].v[d] );
        }
    }

    for (int n1=n_lower[0]-1; n1<=n_upper[0]+1; ++n1)
    for (int n2=n_lower[1]-1; n2<=n_upper[1]+1; ++n2)
    for (int n3=n_lower[2]-1; n3<=n_upper[2]+1; ++n3) {
        claV3i_t R = {{n1, n2, n3}};
        int RB1 = claV3i_dot( R, s->B1 ),
            RB2 = claV3i_dot( R, s->B2 ),
            RB3 = claV3i_dot( R, s->B3 );
        if (RB1 >= 0 && RB1 < s->N && RB2 >= 0 && RB2 < s->N && RB3 >= 0 && RB3 < s->N)
            cells_push( R );
    }

    assert( s->n_cells == s->N );
    return s;
}

static inline index_t py_mod( index_t i, index_t n ) { return (n + (i % n)) % n; }
static inline index_t py_div( index_t i, index_t n ) { return (i - py_mod(i,n))/n; }

// maps unit cell lattice vector R to supercell s.
// returning supercell lattice vector S and supercell index idx
static void supercell_map( supercell_t* s, claV3i_t R, claV3i_t* S, index_t* idx ) {
    int RB1 = claV3i_dot(R,s->B1), RB2 = claV3i_dot(R,s->B2), RB3 = claV3i_dot(R,s->B3);
    int R1 = py_div(RB1, s->N), r1 = py_mod(RB1, s->N);
    int R2 = py_div(RB2, s->N), r2 = py_mod(RB2, s->N);
    int R3 = py_div(RB3, s->N), r3 = py_mod(RB3, s->N);

    claV3i_t r = claV3i_add( claV3i_add( claV3i_scale(s->N1,r1), claV3i_scale(s->N2,r2) ), claV3i_scale(s->N3,r3) );
    *idx = -1;

    for (index_t n=0; n<s->n_cells; ++n) {
        int nmatches = 0;
        for (int d=0; d<3; ++d)
            if (s->cells[n].v[d] == (r.v[d] / s->N)) nmatches++;
        if (nmatches == 3)
            *idx = n;
    }

    assert( *idx >= 0 && *idx < s->n_cells );

    *S = (claV3i_t){{R1, R2, R3}};
}

void rs_hopping_to_fractcell( diverge_model_t* m, int* N1, int* N2, int* N3 ) {
    assert(m->hop != NULL || m->n_hop == 0);

    if (N2 == NULL && N3 == NULL) {
        rs_hopping_to_supercell( m, N1 );
        return;
    }

    if (m->n_sym != 0) {
        mpi_wrn_printf( "found n_sym=%li and orb_symmetries=%p in supercell; this might lead to SIGSEGV\n", m->n_sym, m->orb_symmetries );
    }

    supercell_t* s = supercell_gen( N1, N2, N3 );

    mpi_vrb_printf( "model to supercell (%i,%i,%i) (%i,%i,%i), (%i,%i,%i)\n",
            N1[0], N1[1], N1[2],
            N2[0], N2[1], N2[2],
            N3[0], N3[1], N3[2] );
    mpi_vrb_printf( "with N=%i unit cells per supercell\n", s->N );

    claV3d_t L1 = claV3d_map(m->lattice[0]),
             L2 = claV3d_map(m->lattice[1]),
             L3 = claV3d_map(m->lattice[2]);

    index_t n_orb_mini = m->n_orb;

    double pos_cpy[sizeof(m->positions)/24][3];
    // update positions
    for (index_t c=0; c<s->n_cells; ++c) {
        for (index_t i=0; i<n_orb_mini; ++i) {
            claV3d_t pos = claV3d_map(m->positions[i]);
            pos = claV3d_add( pos, claV3d_scale( L1, s->cells[c].v[0] ) );
            pos = claV3d_add( pos, claV3d_scale( L2, s->cells[c].v[1] ) );
            pos = claV3d_add( pos, claV3d_scale( L3, s->cells[c].v[2] ) );
            claV3d_rmap(pos, pos_cpy[IDX2(c, i, n_orb_mini)]);
        }
    }
    memcpy( m->positions[0], pos_cpy[0], sizeof(m->positions) );

    // update #orbs
    m->n_orb *= s->N;

    // update lattice
    claV3d_t S1 = claV3d_add( claV3d_add( claV3d_scale(L1,N1[0]), claV3d_scale(L2,N1[1]) ), claV3d_scale(L3,N1[2]) );
    claV3d_t S2 = claV3d_add( claV3d_add( claV3d_scale(L1,N2[0]), claV3d_scale(L2,N2[1]) ), claV3d_scale(L3,N2[2]) );
    claV3d_t S3 = claV3d_add( claV3d_add( claV3d_scale(L1,N3[0]), claV3d_scale(L2,N3[1]) ), claV3d_scale(L3,N3[2]) );
    claV3d_rmap( S1, m->lattice[0] );
    claV3d_rmap( S2, m->lattice[1] );
    claV3d_rmap( S3, m->lattice[2] );

    rs_hopping_t* new_hop = calloc(m->n_hop * s->n_cells, sizeof(rs_hopping_t));
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t c=0; c<s->n_cells; ++c) {
        claV3i_t cell = s->cells[c];
        for (index_t h=0; h<m->n_hop; ++h) {
            rs_hopping_t* H = m->hop + h;

            claV3i_t R = {0};
            index_t r = -1;
            claV3i_t R_old = {{H->R[0], H->R[1], H->R[2]}};
            supercell_map( s, claV3i_add(R_old, cell), &R, &r );

            index_t new_hop_idx = c*m->n_hop + h;
            rs_hopping_t* Hnew = new_hop + new_hop_idx;

            memcpy( Hnew, H, sizeof(rs_hopping_t) );
            for (int i=0; i<3; ++i) Hnew->R[i] = R.v[i];
            Hnew->o1 += n_orb_mini * r;
            Hnew->o2 += n_orb_mini * c;
        }
    }
    free(m->hop);
    m->hop = new_hop;
    m->n_hop *= s->n_cells;

    supercell_del( s );
}

void rs_hopping_to_supercell( diverge_model_t* m, int* nr ) {
    int N[3][3] = {0};
    for (int i=0; i<3; ++i) N[i][i] = nr[i];
    rs_hopping_to_fractcell( m, N[0], N[1], N[2] );
}

void rs_hopping_to_fractcell_mat( diverge_model_t* m, int* N123 ) {
    rs_hopping_to_fractcell( m, N123+0, N123+3, N123+6 );
}
