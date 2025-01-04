/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "../misc/btree.h"
#include "../misc/mergesort.h"
#include "../misc/bitary.h"

#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

#define SKIP_CLA_TYPES
#include "projection_wrapper.h"

static void init_CP_PC_proj( Projection* p, const diverge_model_t* model );
static void init_CD_DC_proj( Projection* p, const diverge_model_t* model );
static void init_D_to_P_proj( Projection* p, const diverge_model_t* model );
static void init_P_to_D_proj( Projection* p, const diverge_model_t* model );
static void mpi_communicate( Projection* p );
static void init_to_mom_map_existing_r( Projection* p );
static void init_prefac_vector( Projection* p, const diverge_model_t* model );

static inline int mergesort_permutation_r_comp( size_t i, size_t j, const claV2ll_t* _t ) {
    if (_t[i].v[1] == _t[j].v[1])
        return (_t[i].v[0] < _t[j].v[0]) ? -1 : (_t[i].v[0] > _t[j].v[0]) ? 1 : 0;
    return (_t[i].v[1] < _t[j].v[1]) ? -1 : (_t[i].v[1] > _t[j].v[1]) ? 1 : 0;
}
#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R(i,j,_vt) mergesort_permutation_r_comp(i,j,(const claV2ll_t*)_vt)
MERGESORT_IMPL_R( mergesort_permutation, size_t )

static inline size_t* sort_permutation( const claV2ll_t* _t, size_t _t_sz ) {
    size_t* p = (size_t*)calloc(_t_sz, sizeof*p);
    for (size_t i=0; i<_t_sz; ++i) p[i] = i;
    mergesort_permutation_r(p, _t_sz, (void*)_t, diverge_omp_num_threads(), 0);
    return p;
}

static inline void apply_permutation_in_place( claV2ll_t* vec, const size_t* p, size_t p_sz ) {
    cbyte_t* done = bitary_alloc(p_sz);
    for (size_t i=0; i<p_sz; ++i) {
        if (bitary_get(done, i)) continue;
        bitary_set(done,i);
        size_t prev_j = i;
        size_t j = p[i];
        while (i != j) {
            claV2ll_t tmp; tmp = vec[prev_j]; vec[prev_j] = vec[j]; vec[j] = tmp;
            bitary_set(done,j);
            prev_j = j;
            j = p[j];
        }
    }
    free(done);
}

typedef struct { const claV2ll_t* _fhelp; const index_t* _off; uindex_t k; } mergesort_fhelp_t;
static inline int mergesort_fhelp_comp( uindex_t i, uindex_t j, const mergesort_fhelp_t* h ) {
    const claV2ll_t fi = h->_fhelp[i+h->_off[h->k]],
                    fj = h->_fhelp[j+h->_off[h->k]];
    if (fi.v[1] == fj.v[1])
        return fi.v[0] < fi.v[0] ? -1 : fi.v[0] > fi.v[0] ? 1 : 0;
    return fi.v[1] < fi.v[1] ? -1 : fi.v[1] > fi.v[1] ? 1 : 0;
}
#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R( x, y, ctx ) mergesort_fhelp_comp( (x), (y), (const mergesort_fhelp_t*)ctx )
MERGESORT_IMPL_R( mergesort_fhelp, uindex_t )

typedef struct { VECTOR_DECL( claV2ll_t, v ) } vec_claV2ll_t;
typedef struct { VECTOR_DECL( index_t, v ) } vec_index_t;

static inline void sort_and_filter( vec_claV2ll_t* _t, vec_index_t* _t_p,
        vec_claV2ll_t* _fhelp, vec_index_t* _f,
        vec_claV2ll_t* _rhelp, vec_index_t* _r,
        vec_index_t* _len, vec_index_t* _off ) {
    size_t* p = sort_permutation( _t->v, _t->v_sz );
    apply_permutation_in_place( _t->v, p, _t->v_sz );
    apply_permutation_in_place( _fhelp->v, p, _t->v_sz );
    apply_permutation_in_place( _rhelp->v, p, _t->v_sz );
    free( p );

    uindex_t msize = _t->v_sz;
    VECTOR_RESIZE( _t_p->v, 2*msize );
    memcpy( _t_p->v, _t->v, 2*msize*sizeof(index_t) );

    VECTOR_RESIZE( _t->v, 0 );
    index_t l = _t_p->v[0], r = _t_p->v[1];
    index_t len = 0;
    VECTOR_RESIZE( _f->v, _t_p->v_sz );
    VECTOR_RESIZE( _r->v, _t_p->v_sz );
    for (uindex_t i=0; i<msize; ++i) {
        if (_t_p->v[2*i] == l && _t_p->v[2*i+1] == r) {
            len++;
        } else {
            VECTOR_PUSH_BACK( _len->v, len ); len = 1;
            VECTOR_PUSH_BACK( _t->v, (claV2ll_t){{l,r}} );
            l = _t_p->v[2*i]; r = _t_p->v[2*i+1];
        }
    }
    VECTOR_PUSH_BACK( _len->v, len );
    VECTOR_PUSH_BACK( _t->v, (claV2ll_t){{l,r}} );

    VECTOR_RESIZE( _off->v, _len->v_sz );
    _off->v[0] = 0;
    for (index_t i=1; i<_len->v_sz; ++i)
        _off->v[i] = _off->v[i-1] + _len->v[i-1];
    for (index_t k=0; k<_len->v_sz; ++k) {
        VECTOR_INIT_CAP( uindex_t, p, _len->v[k] ); p_sz = _len->v[k];
        for (index_t i=0; i<p_sz; ++i) p[i] = i;
        mergesort_fhelp_t ctx = { _fhelp->v, _off->v, k };
        mergesort_fhelp_r( p, p_sz, (void*)&ctx, diverge_omp_num_threads(), 0 );
        for (index_t q=0; q<_len->v[k]; ++q) {
            _f->v[2*(q+_off->v[k])] = _fhelp->v[_off->v[k]+p[q]].v[0];
            _f->v[2*(q+_off->v[k])+1] = _fhelp->v[_off->v[k]+p[q]].v[1];
            _r->v[2*(q+_off->v[k])] = _rhelp->v[_off->v[k]+p[q]].v[0];
            _r->v[2*(q+_off->v[k])+1] = _rhelp->v[_off->v[k]+p[q]].v[1];
        }
        _len->v[k] *= 2; _off->v[k] *= 2;
        VECTOR_FREE( p );
    }
}


static inline claV3i_t subtract_2( const tu_formfactor_t* tu_ff, index_t b1, index_t b2 ) {
    return (claV3i_t){{ tu_ff[b1].R[0]-tu_ff[b2].R[0],
                        tu_ff[b1].R[1]-tu_ff[b2].R[1],
                        tu_ff[b1].R[2]-tu_ff[b2].R[2] }};
}

static inline claV3i_t subtract_3( const tu_formfactor_t* tu_ff, index_t b1, index_t b2, index_t b3 ) {
    return (claV3i_t){{ tu_ff[b1].R[0]-tu_ff[b2].R[0]-tu_ff[b3].R[0],
                        tu_ff[b1].R[1]-tu_ff[b2].R[1]-tu_ff[b3].R[1],
                        tu_ff[b1].R[2]-tu_ff[b2].R[2]-tu_ff[b3].R[2] }};
}

static inline claV3i_t set_to( const tu_formfactor_t* tu_ff, index_t b1, int pref ) {
    return (claV3i_t){{ tu_ff[b1].R[0]*pref,
                        tu_ff[b1].R[1]*pref,
                        tu_ff[b1].R[2]*pref }};
}

#define TO_IDX( a, nk ) IDX3( (a).v[0], (a).v[1], (a).v[2], (nk)[1], (nk)[2] )

static inline int claV3i_equal( claV3i_t a, claV3i_t b ) {
    return (a.v[0] == b.v[0]) && (a.v[1] == b.v[1]) && (a.v[2] == b.v[2]);
}

Projection* projection_init( const diverge_model_t* model ) {
    Projection* p = (Projection*)calloc(1, sizeof*p);
    p->proj_descr = cuda_proj_defaults_CPP();

    tu_data_t* tu_data = model->internals->tu_data;
    p->n_orb = model->n_orb;
    p->n_bonds = tu_data->n_bonds;
    p->n_orbff = tu_data->n_orbff;
    p->bond_sizes = tu_data->bond_sizes;
    p->bond_offsets = tu_data->bond_offsets;
    p->ob_to_orbff = tu_data->ob_to_orbff;
    p->tu_ff = model->tu_ff;
    memcpy( p->nk, model->nk, sizeof(p->nk) );

    double tick, tock;
    tick = diverge_mpi_wtime();
    init_CP_PC_proj( p, model );
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("init_CP_PC_proj took %7.2fs\n", tock-tick);

    tick = diverge_mpi_wtime();
    init_CD_DC_proj( p, model );
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("init_CD_DC_proj took %7.2fs\n", tock-tick);

    tick = diverge_mpi_wtime();
    init_P_to_D_proj( p, model );
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("init_P_to_D_proj took %7.2fs\n", tock-tick);

    tick = diverge_mpi_wtime();
    init_D_to_P_proj( p, model );
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("init_D_to_P_proj took %7.2fs\n", tock-tick);

    tick = diverge_mpi_wtime();
    init_to_mom_map_existing_r( p );
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("constructing existing vectors took %7.2fs\n", tock-tick);

    mpi_communicate( p );
    init_prefac_vector( p, model );

    const index_t ntasks =  p->DC_CD_t_sz*p->nr*POW4(model->n_spin);
    cuda_setup_projections( &p->proj_descr, ntasks, tu_data->my_nk );
    return p;
}

void projection_free( Projection* p ) {
    free(p->my_prefac_reco);
    free(p->my_prefac_proj);
    free(p->my_prefac_proj_rk);

    free(p->PC_CP_t);
    free(p->PC_CP_f);
    free(p->PC_CP_t_padded);
    free(p->PC_CP_mom);
    free(p->PC_CP_d);
    free(p->PC_CP_l);

    free(p->DC_CD_t);
    free(p->DC_CD_f);
    free(p->DC_CD_t_padded);
    free(p->DC_CD_mom);
    free(p->DC_CD_d);
    free(p->DC_CD_l);

    free(p->D_to_P_t);
    free(p->D_to_P_f);
    free(p->D_to_P_t_padded);
    free(p->D_to_P_mom);
    free(p->D_to_P_d);
    free(p->D_to_P_l);

    free(p->P_to_D_t);
    free(p->P_to_D_f);
    free(p->P_to_D_t_padded);
    free(p->P_to_D_mom);
    free(p->P_to_D_d);
    free(p->P_to_D_l);

    free(p->existing_r);
    free(p->PC_r);
    free(p->DC_r);
    free(p->DP_r);
    free(p->PD_r);
}

static inline void mesh_normalize_1( claV3i_t* v, const index_t* nk ) {
    for (short d=0; d<3; ++d) { v->v[d] += nk[d]; v->v[d] %= nk[d]; }
}

static inline void mesh_normalize_2( claV3i_t* v, const index_t* nk ) {
    for (short d=0; d<3; ++d) { v->v[d] += nk[d]*2; v->v[d] %= nk[d]; }
}

static inline void mesh_normalize_4( claV3i_t* v, const index_t* nk ) {
    for (short d=0; d<3; ++d) { v->v[d] += nk[d]*4; v->v[d] %= nk[d]; }
}

static void init_CP_PC_proj( Projection* p, const diverge_model_t* model ) {
    if (diverge_mpi_comm_rank() != 0) return;
    VECTOR_INIT( claV2ll_t, f_helper );
    VECTOR_INIT( claV2ll_t, r_helper );
    VECTOR_RESERVE( p->PC_CP_t, POW2(p->n_orbff) );
    // calculate site dependent index map
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        VECTOR_INIT_CAP( claV2ll_t, my_t, POW2(p->n_orbff)/diverge_omp_num_threads() );
        VECTOR_INIT_CAP( claV2ll_t, my_help_f, POW2(p->n_orbff)/diverge_omp_num_threads() );
        VECTOR_INIT_CAP( claV2ll_t, my_rv, POW2(p->n_orbff)/diverge_omp_num_threads() );
        claV3i_t s1, s2;
        #pragma omp for collapse(3) schedule(dynamic)
        for (index_t o1=0; o1 < p->n_orb; ++o1)
        for (index_t o3=0; o3 < p->n_orb; ++o3) {
            for (index_t b1=0; b1<p->n_bonds; ++b1)
            for (index_t b3p=0; b3p<p->bond_sizes[o3]; ++b3p)
            if (b1 < p->bond_sizes[o1] && p->tu_ff[p->bond_offsets[o1]+b1].oto
                    == p->tu_ff[p->bond_offsets[o3]+b3p].oto) {
                for (index_t b1p=0; b1p<p->bond_sizes[o1]; ++b1p)
                for (index_t b3=0; b3<p->bond_sizes[o3]; ++b3)
                if (p->tu_ff[p->bond_offsets[o1]+b1p].oto == p->tu_ff[p->bond_offsets[o3]+b3].oto) {
                    s1 = subtract_2(p->tu_ff, p->bond_offsets[o3]+b3, p->bond_offsets[o3]+b3p);
                    s2 = subtract_2(p->tu_ff, p->bond_offsets[o1]+b1p, p->bond_offsets[o1]+b1);
                    mesh_normalize_2( &s1, model->nk );
                    mesh_normalize_2( &s2, model->nk );

                    if (claV3i_equal(s1,s2)) {
                        claV2ll_t trial = {{ p->ob_to_orbff[b1+p->n_bonds*o1],
                                             p->ob_to_orbff[b3+p->n_bonds*o3] }};
                        claV2ll_t trial2 = {{ p->ob_to_orbff[b1p+p->n_bonds*o1],
                                              p->ob_to_orbff[b3p+p->n_bonds*o3] }};
                        claV2ll_t rv = {{ TO_IDX(s1,model->nk), TO_IDX(s1,model->nk) }};
                        VECTOR_PUSH_BACK( my_rv, rv );
                        VECTOR_PUSH_BACK( my_t, trial );
                        VECTOR_PUSH_BACK( my_help_f, trial2 );
                    }
                }
            }
        }
        VECTOR_SHRINK_TO_FIT( my_t );
        VECTOR_SHRINK_TO_FIT( my_rv );
        VECTOR_SHRINK_TO_FIT( my_help_f );
        #pragma omp critical
        {
            for (index_t i=0; i<my_t_sz; ++i) VECTOR_PUSH_BACK( p->PC_CP_t, my_t[i] );
            for (index_t i=0; i<my_help_f_sz; ++i) VECTOR_PUSH_BACK( f_helper, my_help_f[i] );
            for (index_t i=0; i<my_rv_sz; ++i) VECTOR_PUSH_BACK( r_helper, my_rv[i] );
        }
        VECTOR_FREE( my_t );
        VECTOR_FREE( my_rv );
        VECTOR_FREE( my_help_f );
    }

    vec_claV2ll_t v_PC_CP_t = { .v = p->PC_CP_t, .v_sz = p->PC_CP_t_sz, .v_cap = p->PC_CP_t_cap },
                  v_f_helper = { .v = f_helper, .v_sz = f_helper_sz, .v_cap = f_helper_cap },
                  v_r_helper = { .v = r_helper, .v_sz = r_helper_sz, .v_cap = r_helper_cap };

    vec_index_t v_PC_CP_t_padded = {0},
                v_PC_CP_f = {0},
                v_PC_r = {0},
                v_PC_CP_l = {0},
                v_PC_CP_d = {0};

    sort_and_filter( &v_PC_CP_t, &v_PC_CP_t_padded, &v_f_helper, &v_PC_CP_f,
            &v_r_helper, &v_PC_r, &v_PC_CP_l, &v_PC_CP_d );

    p->PC_CP_t = v_PC_CP_t.v;
    p->PC_CP_t_sz = v_PC_CP_t.v_sz;
    p->PC_CP_t_cap = v_PC_CP_t.v_cap;

    p->PC_CP_f = v_PC_CP_f.v;
    p->PC_CP_t_padded = v_PC_CP_t_padded.v;
    p->PC_CP_l = v_PC_CP_l.v;
    p->PC_CP_d = v_PC_CP_d.v;

    p->PC_r = v_PC_r.v;
    p->PC_r_sz = v_PC_r.v_sz;
    p->PC_r_cap = v_PC_r.v_cap;

    VECTOR_FREE( f_helper );
    VECTOR_FREE( r_helper );
}


static void init_CD_DC_proj( Projection* p, const diverge_model_t* model ) {
    if (diverge_mpi_comm_rank() != 0) return;
    VECTOR_INIT( claV2ll_t, f_helper );
    VECTOR_INIT( claV2ll_t, r_helper );
    VECTOR_RESERVE( p->DC_CD_t, POW2(p->n_orbff) );
    // calculate site dependent index map
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        VECTOR_INIT_CAP( claV2ll_t, my_t, POW2(p->n_orbff)/diverge_omp_num_threads() );
        VECTOR_INIT_CAP( claV2ll_t, my_help_f, POW2(p->n_orbff)/diverge_omp_num_threads() );
        VECTOR_INIT_CAP( claV2ll_t, my_rv, POW2(p->n_orbff)/diverge_omp_num_threads() );
        claV3i_t s1, s2;
        #pragma omp for collapse(3) schedule(dynamic)
        for (index_t o1=0; o1<p->n_orb; ++o1)
        for (index_t o4=0; o4<p->n_orb; ++o4) {
            for (index_t b1p=0; b1p<p->n_bonds; ++b1p)
            if (b1p < p->bond_sizes[o1] && p->tu_ff[p->bond_offsets[o1]+b1p].oto == o4) {
                for (index_t b1=0; b1<p->bond_sizes[o1]; ++b1)
                for (index_t b4=0; b4<p->bond_sizes[o4]; ++b4) {
                    index_t o3 = p->tu_ff[p->bond_offsets[o1]+b1].oto;
                    for (index_t b3p=0; b3p<p->bond_sizes[o3]; ++b3p)
                    if (p->tu_ff[p->bond_offsets[o4]+b4].oto == p->tu_ff[p->bond_offsets[o3]+b3p].oto) {
                        s1 = subtract_3(p->tu_ff, p->bond_offsets[o1]+b1p,
                                p->bond_offsets[o3]+b3p, p->bond_offsets[o1]+b1);
                        s2 = set_to(p->tu_ff,p->bond_offsets[o4]+b4,-1.0);

                        mesh_normalize_4( &s1, model->nk );
                        mesh_normalize_1( &s2, model->nk );

                        if(claV3i_equal(s1,s2)) {
                            claV2ll_t trial = {{ p->ob_to_orbff[b1+p->n_bonds*o1],
                                                 p->ob_to_orbff[b4+p->n_bonds*o4] }};
                            claV2ll_t trial2 = {{ p->ob_to_orbff[b1p+p->n_bonds*o1],
                                                  p->ob_to_orbff[b3p+p->n_bonds*o3] }};
                            claV3i_t helper = set_to(p->tu_ff,p->bond_offsets[o3]+b3p,1.0);
                            claV2ll_t rv = {{ TO_IDX(s2,model->nk), TO_IDX(helper,model->nk) }};
                            VECTOR_PUSH_BACK( my_rv, rv );
                            VECTOR_PUSH_BACK( my_t, trial );
                            VECTOR_PUSH_BACK( my_help_f, trial2 );
                        }
                    }
                }
            }
        }
        VECTOR_SHRINK_TO_FIT( my_t );
        VECTOR_SHRINK_TO_FIT( my_rv );
        VECTOR_SHRINK_TO_FIT( my_help_f );
        #pragma omp critical
        {
            for (index_t i=0; i<my_t_sz; ++i) VECTOR_PUSH_BACK( p->DC_CD_t, my_t[i] );
            for (index_t i=0; i<my_help_f_sz; ++i) VECTOR_PUSH_BACK( f_helper, my_help_f[i] );
            for (index_t i=0; i<my_rv_sz; ++i) VECTOR_PUSH_BACK( r_helper, my_rv[i] );
        }
        VECTOR_FREE( my_t );
        VECTOR_FREE( my_rv );
        VECTOR_FREE( my_help_f );
    }
    vec_claV2ll_t v_DC_CD_t = { .v = p->DC_CD_t, .v_sz = p->DC_CD_t_sz, .v_cap = p->DC_CD_t_cap },
                  v_f_helper = { .v = f_helper, .v_sz = f_helper_sz, .v_cap = f_helper_cap },
                  v_r_helper = { .v = r_helper, .v_sz = r_helper_sz, .v_cap = r_helper_cap };

    vec_index_t v_DC_CD_t_padded = {0},
                v_DC_CD_f = {0},
                v_DC_r = {0},
                v_DC_CD_l = {0},
                v_DC_CD_d = {0};

    sort_and_filter( &v_DC_CD_t, &v_DC_CD_t_padded, &v_f_helper, &v_DC_CD_f,
            &v_r_helper, &v_DC_r, &v_DC_CD_l, &v_DC_CD_d );

    p->DC_CD_t = v_DC_CD_t.v;
    p->DC_CD_t_sz = v_DC_CD_t.v_sz;
    p->DC_CD_t_cap = v_DC_CD_t.v_cap;

    p->DC_CD_f = v_DC_CD_f.v;
    p->DC_CD_t_padded = v_DC_CD_t_padded.v;
    p->DC_CD_l = v_DC_CD_l.v;
    p->DC_CD_d = v_DC_CD_d.v;

    p->DC_r = v_DC_r.v;
    p->DC_r_sz = v_DC_r.v_sz;
    p->DC_r_cap = v_DC_r.v_cap;
    VECTOR_FREE( f_helper );
    VECTOR_FREE( r_helper );
}

static void init_D_to_P_proj( Projection* p, const diverge_model_t* model ) {
    if (diverge_mpi_comm_rank() != 0) return;
    VECTOR_INIT( claV2ll_t, f_helper );
    VECTOR_INIT( claV2ll_t, r_helper );
    VECTOR_RESERVE( p->D_to_P_t, POW2(p->n_orbff) );
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        VECTOR_INIT_CAP( claV2ll_t, my_t, POW2(p->n_orbff)/diverge_omp_num_threads() );
        VECTOR_INIT_CAP( claV2ll_t, my_help_f, POW2(p->n_orbff)/diverge_omp_num_threads() );
        VECTOR_INIT_CAP( claV2ll_t, my_rv, POW2(p->n_orbff)/diverge_omp_num_threads() );
        claV3i_t s1, s2;
        #pragma omp for collapse(3) schedule(dynamic)
        for (index_t o1=0; o1<p->n_orb; ++o1)
        for (index_t o3=0; o3<p->n_orb; ++o3)
        for (index_t b1p=0; b1p<p->n_bonds; ++b1p)
        if (b1p < p->bond_sizes[o1] && p->tu_ff[p->bond_offsets[o1]+b1p].oto == o3) {
            for (index_t b1=0; b1<p->bond_sizes[o1]; ++b1)
            for (index_t b3=0; b3<p->bond_sizes[o3]; ++b3) {
                index_t o4 = p->tu_ff[p->bond_offsets[o3]+b3].oto;
                for (index_t b4p=0; b4p<p->bond_sizes[o4]; ++b4p)
                if (p->tu_ff[p->bond_offsets[o4]+b4p].oto == p->tu_ff[p->bond_offsets[o1]+b1].oto) {

                    s1 = subtract_2(p->tu_ff, p->bond_offsets[o1]+b1p, p->bond_offsets[o1]+b1);
                    s2 = claV3i_add( set_to(p->tu_ff, p->bond_offsets[o4]+b4p,-1.0),
                                     set_to(p->tu_ff, p->bond_offsets[o3]+b3,-1.0) );
                    mesh_normalize_4( &s1, model->nk );
                    mesh_normalize_4( &s2, model->nk );

                    if (claV3i_equal(s1,s2)) {
                        claV2ll_t trial = {{ p->ob_to_orbff[b1+p->n_bonds*o1],
                                             p->ob_to_orbff[b3+p->n_bonds*o3] }};
                        claV2ll_t trial2 = {{ p->ob_to_orbff[b1p+p->n_bonds*o1],
                                              p->ob_to_orbff[b4p+p->n_bonds*o4] }};
                        claV3i_t helper = subtract_2(p->tu_ff, p->bond_offsets[o1]+b1p, p->bond_offsets[o1]+b1);
                        mesh_normalize_4( &helper, model->nk );
                        claV3i_t helper2 = set_to(p->tu_ff, p->bond_offsets[o4]+b4p, -1.0);
                        mesh_normalize_4( &helper2, model->nk );

                        claV2ll_t rv = {{ TO_IDX(helper,model->nk), TO_IDX(helper2,model->nk) }};
                        VECTOR_PUSH_BACK( my_rv, rv );
                        VECTOR_PUSH_BACK( my_t, trial );
                        VECTOR_PUSH_BACK( my_help_f, trial2 );
                    }
                }
            }
        }
        VECTOR_SHRINK_TO_FIT( my_t );
        VECTOR_SHRINK_TO_FIT( my_rv );
        VECTOR_SHRINK_TO_FIT( my_help_f );
        #pragma omp critical
        {
            for (index_t i=0; i<my_t_sz; ++i) VECTOR_PUSH_BACK( p->D_to_P_t, my_t[i] );
            for (index_t i=0; i<my_help_f_sz; ++i) VECTOR_PUSH_BACK( f_helper, my_help_f[i] );
            for (index_t i=0; i<my_rv_sz; ++i) VECTOR_PUSH_BACK( r_helper, my_rv[i] );
        }
        VECTOR_FREE( my_t );
        VECTOR_FREE( my_rv );
        VECTOR_FREE( my_help_f );
    }
    vec_claV2ll_t v_D_to_P_t = { .v = p->D_to_P_t, .v_sz = p->D_to_P_t_sz, .v_cap = p->D_to_P_t_cap },
                  v_f_helper = { .v = f_helper, .v_sz = f_helper_sz, .v_cap = f_helper_cap },
                  v_r_helper = { .v = r_helper, .v_sz = r_helper_sz, .v_cap = r_helper_cap };

    vec_index_t v_D_to_P_t_padded = {0},
                v_D_to_P_f = {0},
                v_DP_r = {0},
                v_D_to_P_l = {0},
                v_D_to_P_d = {0};

    sort_and_filter( &v_D_to_P_t, &v_D_to_P_t_padded, &v_f_helper, &v_D_to_P_f,
            &v_r_helper, &v_DP_r, &v_D_to_P_l, &v_D_to_P_d );

    p->D_to_P_t = v_D_to_P_t.v;
    p->D_to_P_t_sz = v_D_to_P_t.v_sz;
    p->D_to_P_t_cap = v_D_to_P_t.v_cap;

    p->D_to_P_f = v_D_to_P_f.v;
    p->D_to_P_t_padded = v_D_to_P_t_padded.v;
    p->D_to_P_l = v_D_to_P_l.v;
    p->D_to_P_d = v_D_to_P_d.v;

    p->DP_r = v_DP_r.v;
    p->DP_r_sz = v_DP_r.v_sz;
    p->DP_r_cap = v_DP_r.v_cap;
    VECTOR_FREE( f_helper );
    VECTOR_FREE( r_helper );
}

static void init_P_to_D_proj( Projection* p, const diverge_model_t* model ) {
    if (diverge_mpi_comm_rank() != 0) return;
    VECTOR_INIT( claV2ll_t, f_helper );
    VECTOR_INIT( claV2ll_t, r_helper );
    VECTOR_RESERVE( p->P_to_D_t, POW2(p->n_orbff) );
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        VECTOR_INIT_CAP( claV2ll_t, my_t, POW2(p->n_orbff)/diverge_omp_num_threads() );
        VECTOR_INIT_CAP( claV2ll_t, my_help_f, POW2(p->n_orbff)/diverge_omp_num_threads() );
        VECTOR_INIT_CAP( claV2ll_t, my_rv, POW2(p->n_orbff)/diverge_omp_num_threads() );
        claV3i_t s1, s2;
        #pragma omp for collapse(3) schedule(dynamic)
        for (index_t o1=0; o1<p->n_orb; ++o1)
        for (index_t o4=0; o4<p->n_orb; ++o4) {
            for (index_t b4=0; b4<p->n_bonds; ++b4)
            for (index_t b1p=0; b1p<p->bond_sizes[o1]; ++b1p)
            if (b4 < p->bond_sizes[o4] && p->tu_ff[p->bond_offsets[o4]+b4].oto
                    == p->tu_ff[p->bond_offsets[o1]+b1p].oto) {
                for (index_t b1=0; b1<p->bond_sizes[o1]; ++b1) {
                    index_t o3 = p->tu_ff[p->bond_offsets[o1]+b1].oto;
                    for (index_t b3p=0; b3p<p->bond_sizes[o3]; ++b3p)
                    if (p->tu_ff[p->bond_offsets[o3]+b3p].oto == o4) {
                        s1 = set_to(p->tu_ff, p->bond_offsets[o4]+b4, 1.0);
                        s2 = subtract_3(p->tu_ff, p->bond_offsets[o1]+b1p,
                                p->bond_offsets[o1]+b1, p->bond_offsets[o3]+b3p);
                        mesh_normalize_4( &s1, model->nk );
                        mesh_normalize_4( &s2, model->nk );
                        if (claV3i_equal(s1, s2)) {
                            claV2ll_t trial = {{ p->ob_to_orbff[b1+p->n_bonds*o1],
                                                 p->ob_to_orbff[b4+p->n_bonds*o4] }};
                            claV2ll_t trial2 = {{ p->ob_to_orbff[b1p+p->n_bonds*o1],
                                                  p->ob_to_orbff[b3p+p->n_bonds*o3] }};

                            claV3i_t helper = set_to(p->tu_ff, p->bond_offsets[o4]+b4, 1.0);

                            claV3i_t helper2 = claV3i_add(set_to(p->tu_ff, p->bond_offsets[o4]+b4, 1.0),
                                                          set_to(p->tu_ff, p->bond_offsets[o3]+b3p, 1.0));
                            mesh_normalize_4( &helper2, model->nk );
                            claV2ll_t rv = {{ TO_IDX(helper,model->nk), TO_IDX(helper2,model->nk) }};
                            VECTOR_PUSH_BACK( my_rv, rv );
                            VECTOR_PUSH_BACK( my_t, trial );
                            VECTOR_PUSH_BACK( my_help_f, trial2 );
                        }
                    }
                }
            }
        }
        VECTOR_SHRINK_TO_FIT( my_t );
        VECTOR_SHRINK_TO_FIT( my_rv );
        VECTOR_SHRINK_TO_FIT( my_help_f );
        #pragma omp critical
        {
            for (index_t i=0; i<my_t_sz; ++i) VECTOR_PUSH_BACK( p->P_to_D_t, my_t[i] );
            for (index_t i=0; i<my_help_f_sz; ++i) VECTOR_PUSH_BACK( f_helper, my_help_f[i] );
            for (index_t i=0; i<my_rv_sz; ++i) VECTOR_PUSH_BACK( r_helper, my_rv[i] );
        }
        VECTOR_FREE( my_t );
        VECTOR_FREE( my_rv );
        VECTOR_FREE( my_help_f );
    }
    vec_claV2ll_t v_P_to_D_t = { .v = p->P_to_D_t, .v_sz = p->P_to_D_t_sz, .v_cap = p->P_to_D_t_cap },
                  v_f_helper = { .v = f_helper, .v_sz = f_helper_sz, .v_cap = f_helper_cap },
                  v_r_helper = { .v = r_helper, .v_sz = r_helper_sz, .v_cap = r_helper_cap };

    vec_index_t v_P_to_D_t_padded = {0},
                v_P_to_D_f = {0},
                v_PD_r = {0},
                v_P_to_D_l = {0},
                v_P_to_D_d = {0};

    sort_and_filter( &v_P_to_D_t, &v_P_to_D_t_padded, &v_f_helper, &v_P_to_D_f,
            &v_r_helper, &v_PD_r, &v_P_to_D_l, &v_P_to_D_d );

    p->P_to_D_t = v_P_to_D_t.v;
    p->P_to_D_t_sz = v_P_to_D_t.v_sz;
    p->P_to_D_t_cap = v_P_to_D_t.v_cap;

    p->P_to_D_f = v_P_to_D_f.v;
    p->P_to_D_t_padded = v_P_to_D_t_padded.v;
    p->P_to_D_l = v_P_to_D_l.v;
    p->P_to_D_d = v_P_to_D_d.v;

    p->PD_r = v_PD_r.v;
    p->PD_r_sz = v_PD_r.v_sz;
    p->PD_r_cap = v_PD_r.v_cap;
    VECTOR_FREE( f_helper );
    VECTOR_FREE( r_helper );
}


static inline claV3i_t unravel_idx( index_t a, const index_t* nk ) {
    index_t x = a / (nk[1]*nk[2]);
    a -= x * nk[1] * nk[2];
    index_t y = a / nk[2];
    index_t z = a - y * nk[2];
    return (claV3i_t){{x,y,z}};
}

static inline void index_to_reduced_iterate( btree_t* node, void* ctx ) {
    Projection* p = ctx;
    node->val = p->existing_r_sz;
    VECTOR_PUSH_BACK( p->existing_r, unravel_idx( node->key, p->nk ) );
}

static void init_to_mom_map_existing_r( Projection* p ) {
    index_t t_sz = (p->PC_r_sz + p->DC_r_sz + p->DP_r_sz + p->PD_r_sz)/4;
    xbtree_t* t = xbtree_init( MAX(t_sz,1024) );

    for (index_t i=0; i<p->PC_r_sz; ++i) xbtree_find_or_insert( t, p->PC_r[i] );
    for (index_t i=0; i<p->DC_r_sz; ++i) xbtree_find_or_insert( t, p->DC_r[i] );
    for (index_t i=0; i<p->DP_r_sz; ++i) xbtree_find_or_insert( t, p->DP_r[i] );
    for (index_t i=0; i<p->PD_r_sz; ++i) xbtree_find_or_insert( t, p->PD_r[i] );

    VECTOR_RESERVE( p->existing_r, MAX(t_sz,1024) );
    xbtree_iter( t, &index_to_reduced_iterate, p );

    p->nr = p->existing_r_sz;
    VECTOR_RESIZE( p->PC_CP_mom, p->PC_r_sz );
    for (index_t i=0; i<p->PC_r_sz; ++i) p->PC_CP_mom[i] = xbtree_find( t, p->PC_r[i] )->val;
    VECTOR_RESIZE( p->DC_CD_mom, p->DC_r_sz );
    for (index_t i=0; i<p->DC_r_sz; ++i) p->DC_CD_mom[i] = xbtree_find( t, p->DC_r[i] )->val;
    VECTOR_RESIZE( p->P_to_D_mom, p->PD_r_sz );
    for (index_t i=0; i<p->PD_r_sz; ++i) p->P_to_D_mom[i] = xbtree_find( t, p->PD_r[i] )->val;
    VECTOR_RESIZE( p->D_to_P_mom, p->DP_r_sz );
    for (index_t i=0; i<p->DP_r_sz; ++i) p->D_to_P_mom[i] = xbtree_find( t, p->DP_r[i] )->val;

    xbtree_free(t);
}

void mpi_communicate( Projection* p ) {
    index_t n_len, n_tot;

#define MPICOPY( XX ) \
    if (diverge_mpi_comm_rank() == 0) { \
        n_len = p-> XX##_t_sz; \
        n_tot = p-> XX##_l[n_len-1] + p-> XX##_d[n_len-1]; \
    } \
    diverge_mpi_bcast_bytes( &n_len, sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( &n_tot, sizeof(index_t), 0 ); \
    if (diverge_mpi_comm_rank() != 0) { \
        p-> XX##_d = calloc(n_len, sizeof*p-> XX##_d); \
        p-> XX##_l = calloc(n_len, sizeof*p-> XX##_l); \
        VECTOR_RESIZE( p-> XX##_t, n_len ); \
        p-> XX##_t_padded = calloc(n_len, sizeof*p-> XX##_t_padded * 2); \
        VECTOR_RESIZE( p-> XX##_mom, n_tot ); \
        p-> XX##_f = calloc(n_tot, sizeof*p-> XX##_f); \
    } \
    diverge_mpi_bcast_bytes( p-> XX##_d, n_len*sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_l, n_len*sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_t, n_len*sizeof(claV2ll_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_t_padded, n_len*2*sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_mom, n_tot*sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_f, n_tot*sizeof(index_t), 0 );

    MPICOPY( PC_CP )
    MPICOPY( DC_CD )
    MPICOPY( D_to_P )
    MPICOPY( P_to_D )

    diverge_mpi_bcast_bytes( &p->nr, sizeof(p->nr), 0 );
    if (diverge_mpi_comm_rank() != 0)
        VECTOR_RESIZE( p->existing_r, p->nr );
    diverge_mpi_bcast_bytes( p->existing_r, p->nr*sizeof(claV3i_t), 0 );
}

static void init_prefac_vector( Projection* p, const diverge_model_t* model ) {

    tu_data_t* tu_data = model->internals->tu_data;
    index_t nk = kdim(model->nk);
    complex128_t* prefac_proj = calloc(nk*p->nr, sizeof(complex128_t));

    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t rv=0; rv<p->nr; ++rv)
    for (index_t kx=0; kx<model->nk[0]; ++kx)
    for (index_t ky=0; ky<model->nk[1]; ++ky)
    for (index_t kz=0; kz<model->nk[2]; ++kz) {
        index_t k = IDX3(kx, ky, kz, model->nk[1], model->nk[2]);
        prefac_proj[k+nk*rv] = cexp(2.*M_PI*I*(
                    p->existing_r[rv].v[0]*kx/(double)model->nk[0] +
                    p->existing_r[rv].v[1]*ky/(double)model->nk[1] +
                    p->existing_r[rv].v[2]*kz/(double)model->nk[2] ));
    }
    p->my_prefac_proj = calloc(tu_data->my_nk*p->nr, sizeof(complex128_t));
    p->my_prefac_proj_rk = calloc(tu_data->my_nk*p->nr, sizeof(complex128_t));
    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t rv=0; rv<p->nr; ++rv)
    for (index_t k=0; k<tu_data->my_nk; ++k) {
        p->my_prefac_proj[k+tu_data->my_nk*rv] =
            prefac_proj[tu_data->symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off] + rv*nk];
        p->my_prefac_proj_rk[rv + p->nr*k] =
            prefac_proj[tu_data->symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off] + rv*nk];
    }

    index_t my_nk_in_bz = tu_data->symm->my_nk_in_bz;
    index_t my_nk_in_bz_off = tu_data->symm->my_nk_in_bz_off;
    p->my_prefac_reco = calloc(my_nk_in_bz*p->nr, sizeof(complex128_t));
    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for(index_t k=0; k<my_nk_in_bz; ++k)
    for(index_t rv=0; rv<p->nr; ++rv) {
        index_t q_in_full = tu_data->symm->from_ibz_2_bz[k + my_nk_in_bz_off];
        p->my_prefac_reco[rv + p->nr*k] = prefac_proj[q_in_full+nk*rv];
    }
    free(prefac_proj);
}

