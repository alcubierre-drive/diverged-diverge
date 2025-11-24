/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_phonons.h"
#include "diverge_common.h"
#include "diverge_model.h"
#include "misc/mpi_functions.h"
#include "tu/phonons.h"

#include "misc/shared_mem.h"
#include "misc/bitary.h"
#include "misc/mergesort.h"

#include "diverge_model.h"
#include "diverge_model_internals.h"
#include "diverge_internals_struct.h"
#include <assert.h>

rs_hopping_t* diverge_phonons_asr( rs_hopping_t* ph_hop, index_t* n_ph_hop ) {
    mpi_log_printf( "ASR (hoppings) with phonon orb index -> (site, xyz)\n" );
    index_t n_hop = *n_ph_hop;

    index_t n_orb = 0;
    for (index_t i=0; i<n_hop; ++i) {
        index_t om = MAX(ph_hop[i].o1, ph_hop[i].o2);
        n_orb = MAX(n_orb, om);
    }
    n_orb = (n_orb+1)/3;
    mpi_vrb_printf( "found (%li,3) orbitals\n", n_orb );

    complex128_t* self_forces = calloc(n_orb*9, sizeof*self_forces);
    for (index_t i=0; i<n_hop; ++i) {
        int ho1 = ph_hop[i].o1/3, hr1 = ph_hop[i].o1%3, hr2 = ph_hop[i].o2%3;
        self_forces[IDX3(ho1,hr1,hr2, 3,3)] += ph_hop[i].t;
    }

    ph_hop = realloc(ph_hop, (*n_ph_hop = (n_hop + (n_orb*9))) * sizeof*ph_hop);
    double forcemax = 0.0;
    for (index_t o=0; o<n_orb; ++o)
    for (index_t r1=0; r1<3; ++r1)
    for (index_t r2=0; r2<3; ++r2) {
        ph_hop[n_hop + IDX3(o,r1,r2, 3,3)] = (rs_hopping_t){
            .o1=IDX2(o,r1,3), .o2=IDX2(o,r2,3),
            .t=-self_forces[IDX3(o,r1,r2, 3,3)],
        };
        forcemax = MAX(forcemax, cabs(self_forces[IDX3(o,r1,r2, 3,3)]));
    }
    mpi_log_printf( "ASR (hoppings) maximal restoring force value: %.5e\n", forcemax );

    free(self_forces);
    return ph_hop;
}

void diverge_phonons_apply_asr( diverge_model_t* m ) {
    m->hop = diverge_phonons_asr( m->hop, &m->n_hop );
}

void* diverge_phonons_apply_sgnsqrt( diverge_model_t* m ) {
    assert( m->n_spin == 1 );
    double* E = diverge_model_internals_get_E(m);
    if (!m->internals->model_phonons_imag_freq)
        m->internals->model_phonons_imag_freq = bitary_alloc(kdimtot(m->nk,m->nkf)*m->n_orb);
    int nimag = 0;
    if (shared_exclusive_enter(E)) for (index_t k=0; k<kdimtot(m->nk,m->nkf)*m->n_orb; ++k) {
        int isimag = (E[k] < -DIVERGE_EPS_MESH);
        if (isimag) {
            bitary_set(m->internals->model_phonons_imag_freq, k);
            nimag++;
        }
        E[k] = (isimag?-1.0:1.0) * sqrt(fabs(E[k]));
    }
    shared_exclusive_wait(E);
    diverge_mpi_allreduce_int_max_inplace( &nimag, 1 );
    if (nimag)
        mpi_wrn_printf( "found %li imaginary phonon frequencies\n", nimag );
    return m->internals->model_phonons_imag_freq;
}

void* diverge_phonons_get_ifrq( diverge_model_t* m ) {
    return m->internals->model_phonons_imag_freq;
}

complex128_t* diverge_phonons_get_g( diverge_model_t* m ) {
    tu_data_phonons_t* p = m->internals->tu_data_phonons;
    return p ? tu_data_phonons_get_g(p) : NULL;
}

double diverge_phonons_change_regulator( double eta ) {
    return tu_data_phonons_change_regulator( eta );
}


typedef struct { int32_t R12[3], o1, s1, o2, s2, dim; } ph_vertex_asr_t;

ph_vertex_asr_t ph_vertex_to_asr( ph_vertex_t p ) {
    return (ph_vertex_asr_t){ .R12={p.R12[0],p.R12[1],p.R12[2]},
        .o1=p.o1, .s1=p.s1, .o2=p.o2, .s2=p.s2, .dim=p.OA%3 };
}

#define vtree_key_t ph_vertex_asr_t
#define vtree_val_t complex128_t
#include "misc/vtree.h"

typedef struct {
    ph_vertex_t* vtx;
    index_t n_vtx;
    index_t n_orb;
    double maximum;
} vtx_push_t;

static inline void vtx_push( vtree_node_t* node, void* ctx ) {
    vtx_push_t* vp = ctx;

    ph_vertex_t node_vtx = {
        .R12={ node->key.R12[0], node->key.R12[1], node->key.R12[2] },
        .o1 = node->key.o1,
        .s1 = node->key.s1,
        .o2 = node->key.o2,
        .s2 = node->key.s2,
        .g = -node->val / (double)vp->n_orb, // this is important!!
    };

    vp->maximum = MAX( vp->maximum, cabs(node->val) );

    for (index_t o=0; o<vp->n_orb; ++o) {
        node_vtx.OA = IDX2(o,node->key.dim,3);
        vp->vtx[vp->n_vtx++] = node_vtx;
    }
}

static int DIVERGE_PHONONS_ASR_VERTEX_USE_COUNT_DISPL_SORTED = 0;
static ph_vertex_t* diverge_phonons_asr_vertex_sorted( ph_vertex_t* vtx, index_t* n_vtx );

ph_vertex_t* diverge_phonons_asr_vertex( ph_vertex_t* vtx, index_t* n_vtx ) {
    mpi_log_printf( "ASR (vertices) with phonon orb index -> (site, xyz)\n" );
    if (DIVERGE_PHONONS_ASR_VERTEX_USE_COUNT_DISPL_SORTED)
        return diverge_phonons_asr_vertex_sorted( vtx, n_vtx );
    xvtree_t* x = xvtree_init(MIN(1024, *n_vtx));

    index_t n_orb = 0;
    for (index_t v=0; v<*n_vtx; ++v) {
        vtree_node_t* nd = xvtree_find_or_insert(x, ph_vertex_to_asr(vtx[v]));
        nd->val += vtx[v].g;
        n_orb = MAX(n_orb, vtx[v].OA);
    }
    n_orb = (n_orb+1)/3;
    mpi_vrb_printf( "found (%li,3) orbitals\n", n_orb );

    index_t n_add = xvtree_size( x ); // # vertices to add (w/o orbitals!)

    // and realloc that size (with the orbitals)
    vtx = realloc( vtx, sizeof*vtx * (*n_vtx + n_orb * n_add) );

    vtx_push_t vp = { .vtx=vtx, .n_vtx=*n_vtx, .n_orb=n_orb, .maximum=0 };
    xvtree_iter(x, &vtx_push, &vp);
    xvtree_free(x);
    *n_vtx += n_orb * n_add;
    assert( *n_vtx == vp.n_vtx );
    mpi_log_printf( "ASR (vertices) maximal restoring element value: %.5e\n", vp.maximum/(double)n_orb );
    return vtx;
}

#undef MERGESORT_COMP
static inline int ph_vertex_cmp( const ph_vertex_t A, const ph_vertex_t B ) {
    int Ro_cmp = memcmp( &A, &B, offsetof(ph_vertex_t, OA) );
    if (!Ro_cmp) {
        int dimA = A.OA % 3, dimB = B.OA % 3;
        return dimA - dimB;
    } else {
        return Ro_cmp;
    }
}
#define MERGESORT_COMP ph_vertex_cmp
MERGESORT_IMPL( ph_vertex_sort, ph_vertex_t )

static ph_vertex_t* diverge_phonons_asr_vertex_sorted( ph_vertex_t* vtx, index_t* n_vtx ) {
    // sort vertices according to the comparison function ph_vertex_cmp
    ph_vertex_sort( vtx, *n_vtx, diverge_omp_num_threads(), 0 );

    // count unique "formfactors" and the number of orbitals
    VECTOR_INIT( ph_vertex_t, asr_ff );
    VECTOR_INIT( index_t, asr_ff_displ );
    VECTOR_INIT( index_t, asr_ff_count );
    index_t n_orb = 0;
    {
        ph_vertex_t* prev_ff = NULL;
        for (index_t v=0; v<*n_vtx; ++v) {
            n_orb = MAX(n_orb, vtx[v].OA);
            if (!prev_ff) {
                prev_ff = vtx + v;
                VECTOR_PUSH_BACK( asr_ff, *prev_ff );
                VECTOR_PUSH_BACK( asr_ff_displ, v );
            } else {
                if (ph_vertex_cmp( *prev_ff, vtx[v] )) {
                    prev_ff = vtx+v;
                    VECTOR_PUSH_BACK( asr_ff, *prev_ff );
                    VECTOR_PUSH_BACK( asr_ff_displ, v );
                }
            }
        }
    }
    n_orb = (n_orb + 1)/3;
    mpi_vrb_printf( "found (%li,3) orbitals\n", n_orb );

    VECTOR_SHRINK_TO_FIT( asr_ff )
    VECTOR_PUSH_BACK( asr_ff_displ, *n_vtx ); // in order to easily count :)
    VECTOR_SHRINK_TO_FIT( asr_ff_displ )
    for (index_t a=0; a<asr_ff_sz; ++a) VECTOR_PUSH_BACK( asr_ff_count, asr_ff_displ[a+1] - asr_ff_displ[a] );
    VECTOR_SHRINK_TO_FIT( asr_ff_count )

    double gmax = 0.0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(max:gmax)
    for (index_t a=0; a<asr_ff_sz; ++a) {
        asr_ff[a].OA = asr_ff[a].OA % 3;
        asr_ff[a].RA[0] = asr_ff[a].RA[1] = asr_ff[a].RA[2] = 0;
        asr_ff[a].g = 0.0;
        for (index_t c=0; c<asr_ff_count[a]; ++c)
            asr_ff[a].g -= vtx[asr_ff_displ[a] + c].g / (double)n_orb;
        gmax = MAX(gmax, cabs(asr_ff[a].g));
    }
    mpi_log_printf( "ASR (vertices) maximal restoring element value: %.5e\n", gmax );
    VECTOR_FREE( asr_ff_count );
    VECTOR_FREE( asr_ff_displ );

    vtx = realloc(vtx, sizeof*vtx * (*n_vtx + n_orb * asr_ff_sz));
    for (index_t a=0; a<asr_ff_sz; ++a)
    for (index_t o=0; o<n_orb; ++o) {
        ph_vertex_t asr_v = asr_ff[a];
        asr_v.OA = IDX2(o,asr_v.OA,3);
        vtx[(*n_vtx)++] = asr_v;
    }

    VECTOR_FREE( asr_ff );

    return vtx;
}

int diverge_phonons_asr_vertex_set_sorted_mode( int sorted ) {
    XORSWAP( DIVERGE_PHONONS_ASR_VERTEX_USE_COUNT_DISPL_SORTED, sorted );
    return sorted;
}
