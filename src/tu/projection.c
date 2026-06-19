/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

#include "../misc/hashtable.h"
#include "../misc/mergesort.h"
#include "../misc/bitary.h"
#include <assert.h>

// this must follow the LINALG include to pre-define the types
#include "projection.h"

#define vtree_key_t index_t
#define vtree_val_t index_t
#include "../misc/vtree.h"

static void init_CP_PC_proj( tu_projection_t* p, const diverge_model_t* model );
static void init_CD_DC_proj( tu_projection_t* p, const diverge_model_t* model );
static void init_D_to_P_proj( tu_projection_t* p, const diverge_model_t* model );
static void init_P_to_D_proj( tu_projection_t* p, const diverge_model_t* model );
static void mpi_communicate( tu_projection_t* p );
static void init_to_mom_map_existing_r( tu_projection_t* p );
static void init_prefac_vector( tu_projection_t* p, const diverge_model_t* model );

static index_t* sort_permutation( const claV2ll_t* to, index_t to_sz );
static inline void apply_permutation_in_place( claV2ll_t* vec, const index_t* p, index_t p_sz );

typedef struct { VECTOR_DECL( claV2ll_t, v ) } vec_claV2ll_t;
typedef struct { VECTOR_DECL( index_t, v ) } vec_index_t;
static vec_claV2ll_t sort_and_filter( vec_claV2ll_t* to, vec_claV2ll_t* from_help, vec_index_t* from,
        vec_claV2ll_t* rvec_help, vec_index_t* rvec, vec_index_t* count, vec_index_t* displ );

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

tu_projection_t* tu_projection_init( const diverge_model_t* model ) {
    tu_projection_t* p = calloc(1, sizeof*p);
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

    const index_t ntasks = (p->DC_CD_to_padded_sz/2)*p->nr*POW4(model->n_spin);
    cuda_setup_projections( &p->proj_descr, ntasks, tu_data->my_nk );

    return p;
}

void tu_projection_free( tu_projection_t* p ) {
    free(p->my_prefac_reco);
    free(p->my_prefac_proj);

    free(p->PC_CP_to);
    free(p->PC_CP_to_padded);
    free(p->PC_CP_from);
    free(p->PC_CP_mom);
    free(p->PC_CP_displ);
    free(p->PC_CP_count);

    free(p->DC_CD_to);
    free(p->DC_CD_to_padded);
    free(p->DC_CD_from);
    free(p->DC_CD_mom);
    free(p->DC_CD_displ);
    free(p->DC_CD_count);

    free(p->D_to_P_to);
    free(p->D_to_P_to_padded);
    free(p->D_to_P_from);
    free(p->D_to_P_mom);
    free(p->D_to_P_displ);
    free(p->D_to_P_count);

    free(p->P_to_D_to);
    free(p->P_to_D_to_padded);
    free(p->P_to_D_from);
    free(p->P_to_D_mom);
    free(p->P_to_D_displ);
    free(p->P_to_D_count);

    free(p->existing_r);
    free(p->PC_r);
    free(p->DC_r);
    free(p->DP_r);
    free(p->PD_r);
    free(p);
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

static void init_CP_PC_proj( tu_projection_t* p, const diverge_model_t* model ) {
    if (diverge_mpi_comm_rank() != 0) return;
    VECTOR_INIT( claV2ll_t, f_helper );
    VECTOR_INIT( claV2ll_t, r_helper );
    VECTOR_RESERVE( p->PC_CP_to, POW2(p->n_orbff) );
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
        for (index_t i=0; i<my_t_sz; ++i) VECTOR_PUSH_BACK( p->PC_CP_to, my_t[i] );
        for (index_t i=0; i<my_help_f_sz; ++i) VECTOR_PUSH_BACK( f_helper, my_help_f[i] );
        for (index_t i=0; i<my_rv_sz; ++i) VECTOR_PUSH_BACK( r_helper, my_rv[i] );
        }

        VECTOR_FREE( my_t );
        VECTOR_FREE( my_help_f );
        VECTOR_FREE( my_rv );
    }

    vec_claV2ll_t v_PC_CP_to = { .v = p->PC_CP_to, .v_sz = p->PC_CP_to_sz, .v_cap = p->PC_CP_to_cap },
                  v_f_helper = { .v = f_helper, .v_sz = f_helper_sz, .v_cap = f_helper_cap },
                  v_r_helper = { .v = r_helper, .v_sz = r_helper_sz, .v_cap = r_helper_cap };

    vec_index_t v_PC_CP_from = {0},
                v_PC_r = {0},
                v_PC_CP_count = {0},
                v_PC_CP_displ = {0};

    vec_claV2ll_t padded = sort_and_filter( &v_PC_CP_to, &v_f_helper, &v_PC_CP_from,
            &v_r_helper, &v_PC_r, &v_PC_CP_count, &v_PC_CP_displ );

    VECTOR_COPY( p->PC_CP_to, v_PC_CP_to.v );
    VECTOR_COPY( p->PC_CP_to_padded, padded.v );
    VECTOR_COPY( p->PC_r, v_PC_r.v );

    p->PC_CP_from = v_PC_CP_from.v;
    p->PC_CP_count = v_PC_CP_count.v;
    p->PC_CP_displ = v_PC_CP_displ.v;

    VECTOR_FREE( f_helper );
    VECTOR_FREE( r_helper );
}


static void init_CD_DC_proj( tu_projection_t* p, const diverge_model_t* model ) {
    if (diverge_mpi_comm_rank() != 0) return;
    VECTOR_INIT( claV2ll_t, f_helper );
    VECTOR_INIT( claV2ll_t, r_helper );
    VECTOR_RESERVE( p->DC_CD_to, POW2(p->n_orbff) );
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
        for (index_t i=0; i<my_t_sz; ++i) VECTOR_PUSH_BACK( p->DC_CD_to, my_t[i] );
        for (index_t i=0; i<my_help_f_sz; ++i) VECTOR_PUSH_BACK( f_helper, my_help_f[i] );
        for (index_t i=0; i<my_rv_sz; ++i) VECTOR_PUSH_BACK( r_helper, my_rv[i] );
        }

        VECTOR_FREE( my_t );
        VECTOR_FREE( my_help_f );
        VECTOR_FREE( my_rv );
    }
    vec_claV2ll_t v_DC_CD_to = { .v = p->DC_CD_to, .v_sz = p->DC_CD_to_sz, .v_cap = p->DC_CD_to_cap },
                  v_f_helper = { .v = f_helper, .v_sz = f_helper_sz, .v_cap = f_helper_cap },
                  v_r_helper = { .v = r_helper, .v_sz = r_helper_sz, .v_cap = r_helper_cap };

    vec_index_t v_DC_CD_from = {0},
                v_DC_r = {0},
                v_DC_CD_count = {0},
                v_DC_CD_displ = {0};

    vec_claV2ll_t padded = sort_and_filter( &v_DC_CD_to, &v_f_helper, &v_DC_CD_from,
            &v_r_helper, &v_DC_r, &v_DC_CD_count, &v_DC_CD_displ );

    VECTOR_COPY( p->DC_CD_to, v_DC_CD_to.v );
    VECTOR_COPY( p->DC_CD_to_padded, padded.v );
    VECTOR_COPY( p->DC_r, v_DC_r.v );

    p->DC_CD_from = v_DC_CD_from.v;
    p->DC_CD_count = v_DC_CD_count.v;
    p->DC_CD_displ = v_DC_CD_displ.v;

    VECTOR_FREE( f_helper );
    VECTOR_FREE( r_helper );
}

static void init_D_to_P_proj( tu_projection_t* p, const diverge_model_t* model ) {
    if (diverge_mpi_comm_rank() != 0) return;
    VECTOR_INIT( claV2ll_t, f_helper );
    VECTOR_INIT( claV2ll_t, r_helper );
    VECTOR_RESERVE( p->D_to_P_to, POW2(p->n_orbff) );
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
        for (index_t i=0; i<my_t_sz; ++i) VECTOR_PUSH_BACK( p->D_to_P_to, my_t[i] );
        for (index_t i=0; i<my_help_f_sz; ++i) VECTOR_PUSH_BACK( f_helper, my_help_f[i] );
        for (index_t i=0; i<my_rv_sz; ++i) VECTOR_PUSH_BACK( r_helper, my_rv[i] );
        }

        VECTOR_FREE( my_t );
        VECTOR_FREE( my_help_f );
        VECTOR_FREE( my_rv );
    }
    vec_claV2ll_t v_D_to_P_to = { .v = p->D_to_P_to, .v_sz = p->D_to_P_to_sz, .v_cap = p->D_to_P_to_cap },
                  v_f_helper = { .v = f_helper, .v_sz = f_helper_sz, .v_cap = f_helper_cap },
                  v_r_helper = { .v = r_helper, .v_sz = r_helper_sz, .v_cap = r_helper_cap };

    vec_index_t v_D_to_P_from = {0},
                v_DP_r = {0},
                v_D_to_P_count = {0},
                v_D_to_P_displ = {0};

    vec_claV2ll_t padded = sort_and_filter( &v_D_to_P_to, &v_f_helper, &v_D_to_P_from,
            &v_r_helper, &v_DP_r, &v_D_to_P_count, &v_D_to_P_displ );

    VECTOR_COPY( p->D_to_P_to, v_D_to_P_to.v );
    VECTOR_COPY( p->D_to_P_to_padded, padded.v );
    VECTOR_COPY( p->DP_r, v_DP_r.v );

    p->D_to_P_from = v_D_to_P_from.v;
    p->D_to_P_count = v_D_to_P_count.v;
    p->D_to_P_displ = v_D_to_P_displ.v;

    VECTOR_FREE( f_helper );
    VECTOR_FREE( r_helper );
}

static void init_P_to_D_proj( tu_projection_t* p, const diverge_model_t* model ) {
    if (diverge_mpi_comm_rank() != 0) return;
    VECTOR_INIT( claV2ll_t, f_helper );
    VECTOR_INIT( claV2ll_t, r_helper );
    VECTOR_RESERVE( p->P_to_D_to, POW2(p->n_orbff) );
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
        for (index_t i=0; i<my_t_sz; ++i) VECTOR_PUSH_BACK( p->P_to_D_to, my_t[i] );
        for (index_t i=0; i<my_help_f_sz; ++i) VECTOR_PUSH_BACK( f_helper, my_help_f[i] );
        for (index_t i=0; i<my_rv_sz; ++i) VECTOR_PUSH_BACK( r_helper, my_rv[i] );
        }

        VECTOR_FREE( my_t );
        VECTOR_FREE( my_help_f );
        VECTOR_FREE( my_rv );
    }
    vec_claV2ll_t v_P_to_D_to = { .v = p->P_to_D_to, .v_sz = p->P_to_D_to_sz, .v_cap = p->P_to_D_to_cap },
                  v_f_helper = { .v = f_helper, .v_sz = f_helper_sz, .v_cap = f_helper_cap },
                  v_r_helper = { .v = r_helper, .v_sz = r_helper_sz, .v_cap = r_helper_cap };

    vec_index_t v_P_to_D_from = {0},
                v_PD_r = {0},
                v_P_to_D_count = {0},
                v_P_to_D_displ = {0};

    vec_claV2ll_t padded = sort_and_filter( &v_P_to_D_to, &v_f_helper, &v_P_to_D_from,
            &v_r_helper, &v_PD_r, &v_P_to_D_count, &v_P_to_D_displ );

    VECTOR_COPY( p->P_to_D_to, v_P_to_D_to.v );
    VECTOR_COPY( p->P_to_D_to_padded, padded.v );
    VECTOR_COPY( p->PD_r, v_PD_r.v );

    p->P_to_D_from = v_P_to_D_from.v;
    p->P_to_D_count = v_P_to_D_count.v;
    p->P_to_D_displ = v_P_to_D_displ.v;

    VECTOR_FREE( f_helper );
    VECTOR_FREE( r_helper );
}

static inline void tree_count( vtree_node_t* node, void* ctx ) { (void)node; index_t* pcount = ctx; (*pcount)++; }

static inline void table_writeout( index_t key, void* val, void* ctx ) {
    (void)val;
    vec_index_t* indices = ctx;
    VECTOR_PUSH_BACK( indices->v, key );
}

static inline index_t binsearch( const index_t* vals, index_t sz, index_t val ) {
    index_t low = 0, high = sz - 1;
    while (low <= high) {
        index_t mid = low + (high - low) / 2;
        if (vals[mid] == val)       return mid;
        else if (vals[mid] < val)   low = mid + 1;
        else                        high = mid - 1;
    }
    return -1;
}

#define UNRAVEL_IDX3( a, nk ) (claV3i_t){{a/(nk[1]*nk[2]), (a%(nk[1]*nk[2]))/nk[2], a%nk[2]}}
static inline void index_to_reduced_iterate( vtree_node_t* node, void* ctx ) {
    tu_projection_t* p = ctx;
    node->val = p->nr;
    p->existing_r[p->nr++] = UNRAVEL_IDX3( node->key, p->nk );
}

#undef MERGESORT_COMP
#define MERGESORT_COMP(x,y) ((x)-(y))
MERGESORT_IMPL( mergesort_index_t, index_t )

#ifdef DIVERGE_TU_PROJ_TO_R_VTREE
static int EXISTING_R_HASHTABLE_IMPL = 0;
#else // DIVERGE_TU_PROJ_TO_R_VTREE
static int EXISTING_R_HASHTABLE_IMPL = 1;
#endif // DIVERGE_TU_PROJ_TO_R_VTREE
void tu_projection_configure_implementation( int hashtable ) { EXISTING_R_HASHTABLE_IMPL = hashtable; }

static void init_to_mom_map_existing_r( tu_projection_t* p ) {
    index_t t_sz = (p->PC_r_sz + p->DC_r_sz + p->DP_r_sz + p->PD_r_sz)/4;
    if (EXISTING_R_HASHTABLE_IMPL) {
        hashtable_t* t = hashtable_init_fg( MAX(t_sz,1024), 16, INT64_MAX );

        for (index_t i=0; i<p->PC_r_sz; ++i) hashtable_insert_safe( t, p->PC_r[i], NULL );
        for (index_t i=0; i<p->DC_r_sz; ++i) hashtable_insert_safe( t, p->DC_r[i], NULL );
        for (index_t i=0; i<p->DP_r_sz; ++i) hashtable_insert_safe( t, p->DP_r[i], NULL );
        for (index_t i=0; i<p->PD_r_sz; ++i) hashtable_insert_safe( t, p->PD_r[i], NULL );

        vec_index_t keys = {0};
        hashtable_iterate( t, &table_writeout, &keys );
        VECTOR_SHRINK_TO_FIT( keys.v );
        hashtable_free( t );

        mergesort_index_t( keys.v, keys.v_sz, diverge_omp_num_threads(), 0 );
        p->nr = keys.v_sz;
        p->existing_r = calloc( keys.v_sz, sizeof*p->existing_r );
        for (index_t nr=0; nr<keys.v_sz; ++nr)
            p->existing_r[nr] = UNRAVEL_IDX3( keys.v[nr], p->nk );

        p->PC_CP_mom = calloc(p->PC_r_sz, sizeof*p->PC_CP_mom);
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<p->PC_r_sz; ++i) p->PC_CP_mom[i] = binsearch( keys.v, keys.v_sz, p->PC_r[i] );
        p->DC_CD_mom = calloc(p->DC_r_sz, sizeof*p->DC_CD_mom);
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<p->DC_r_sz; ++i) p->DC_CD_mom[i] = binsearch( keys.v, keys.v_sz, p->DC_r[i] );
        p->P_to_D_mom = calloc(p->PD_r_sz, sizeof*p->P_to_D_mom);
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<p->PD_r_sz; ++i) p->P_to_D_mom[i] = binsearch( keys.v, keys.v_sz, p->PD_r[i] );
        p->D_to_P_mom = calloc(p->DP_r_sz, sizeof*p->D_to_P_mom);
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<p->DP_r_sz; ++i) p->D_to_P_mom[i] = binsearch( keys.v, keys.v_sz, p->DP_r[i] );

        VECTOR_FREE( keys.v );
    } else {
        xvtree_t* t = xvtree_init( MAX(t_sz,1024) );

        for (index_t i=0; i<p->PC_r_sz; ++i) xvtree_find_or_insert( t, p->PC_r[i] );
        for (index_t i=0; i<p->DC_r_sz; ++i) xvtree_find_or_insert( t, p->DC_r[i] );
        for (index_t i=0; i<p->DP_r_sz; ++i) xvtree_find_or_insert( t, p->DP_r[i] );
        for (index_t i=0; i<p->PD_r_sz; ++i) xvtree_find_or_insert( t, p->PD_r[i] );

        index_t tree_size = 0; xvtree_iter( t, &tree_count, &tree_size );
        p->existing_r = calloc( tree_size, sizeof*p->existing_r );
        xvtree_iter_ascending( t, &index_to_reduced_iterate, p );

        p->PC_CP_mom = calloc(p->PC_r_sz, sizeof*p->PC_CP_mom);
        for (index_t i=0; i<p->PC_r_sz; ++i) p->PC_CP_mom[i] = xvtree_find( t, p->PC_r[i] )->val;
        p->DC_CD_mom = calloc(p->DC_r_sz, sizeof*p->DC_CD_mom);
        for (index_t i=0; i<p->DC_r_sz; ++i) p->DC_CD_mom[i] = xvtree_find( t, p->DC_r[i] )->val;
        p->P_to_D_mom = calloc(p->PD_r_sz, sizeof*p->P_to_D_mom);
        for (index_t i=0; i<p->PD_r_sz; ++i) p->P_to_D_mom[i] = xvtree_find( t, p->PD_r[i] )->val;
        p->D_to_P_mom = calloc(p->DP_r_sz, sizeof*p->D_to_P_mom);
        for (index_t i=0; i<p->DP_r_sz; ++i) p->D_to_P_mom[i] = xvtree_find( t, p->DP_r[i] )->val;

        xvtree_free(t);
    }
}

void mpi_communicate( tu_projection_t* p ) {
    index_t to_sz, to_padded_sz;

#define MPICOPY( XX ) \
    if (diverge_mpi_comm_rank() == 0) { \
        to_sz = p-> XX##_to_sz; \
        to_padded_sz = p-> XX##_to_padded_sz; \
    } \
    diverge_mpi_bcast_bytes( &to_sz, sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( &to_padded_sz, sizeof(index_t), 0 ); \
    if (diverge_mpi_comm_rank() != 0) { \
        p-> XX##_displ = calloc(to_sz, sizeof*p-> XX##_displ); \
        p-> XX##_count = calloc(to_sz, sizeof*p-> XX##_count); \
        VECTOR_RESIZE( p-> XX##_to, to_sz ); \
        VECTOR_RESIZE( p-> XX##_to_padded, to_padded_sz ); \
        p-> XX##_mom = calloc(to_padded_sz*2, sizeof*p-> XX##_mom); \
        p-> XX##_from = calloc(to_padded_sz*2, sizeof*p-> XX##_from); \
    } \
    diverge_mpi_bcast_bytes( p-> XX##_displ, to_sz*sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_count, to_sz*sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_to, to_sz*sizeof(claV2ll_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_to_padded, to_padded_sz*sizeof(claV2ll_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_mom, to_padded_sz*2*sizeof(index_t), 0 ); \
    diverge_mpi_bcast_bytes( p-> XX##_from, to_padded_sz*2*sizeof(index_t), 0 );

    MPICOPY( PC_CP )
    MPICOPY( DC_CD )
    MPICOPY( D_to_P )
    MPICOPY( P_to_D )

    diverge_mpi_bcast_bytes( &p->nr, sizeof(p->nr), 0 );
    if (diverge_mpi_comm_rank() != 0) p->existing_r = calloc(p->nr, sizeof*p->existing_r);
    diverge_mpi_bcast_bytes( p->existing_r, p->nr*sizeof(claV3i_t), 0 );
}

static void init_prefac_vector( tu_projection_t* p, const diverge_model_t* model ) {

    tu_data_t* tu_data = model->internals->tu_data;
    index_t nk = kdim(model->nk);
    complex128_t* prefac_proj = calloc(nk*p->nr, sizeof(complex128_t));

    #pragma omp parallel for collapse(4) num_threads(diverge_omp_num_threads())
    for (index_t rv=0; rv<p->nr; ++rv)
    for (index_t kx=0; kx<model->nk[0]; ++kx)
    for (index_t ky=0; ky<model->nk[1]; ++ky)
    for (index_t kz=0; kz<model->nk[2]; ++kz) {
        index_t k = IDX3(kx, ky, kz, model->nk[1], model->nk[2]);
        prefac_proj[k+nk*rv] = cexp(2.*M_PI*I*(
                    (double)p->existing_r[rv].v[0]*kx/(double)model->nk[0] +
                    (double)p->existing_r[rv].v[1]*ky/(double)model->nk[1] +
                    (double)p->existing_r[rv].v[2]*kz/(double)model->nk[2] ));
    }
    p->my_prefac_proj = calloc(tu_data->my_nk*p->nr, sizeof(complex128_t));
    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t rv=0; rv<p->nr; ++rv)
    for (index_t k=0; k<tu_data->my_nk; ++k) {
        p->my_prefac_proj[k+tu_data->my_nk*rv] =
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

static inline int mergesort_permutation_r_comp( index_t i, index_t j, const claV2ll_t* _t ) {
    if (_t[i].v[1] == _t[j].v[1])
        return (_t[i].v[0] < _t[j].v[0]) ? -1 : (_t[i].v[0] > _t[j].v[0]) ? 1 : 0;
    return (_t[i].v[1] < _t[j].v[1]) ? -1 : (_t[i].v[1] > _t[j].v[1]) ? 1 : 0;
}
#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R(i,j,_vt) mergesort_permutation_r_comp(i,j,(const claV2ll_t*)_vt)
MERGESORT_IMPL_R( mergesort_permutation, index_t )
static index_t* sort_permutation( const claV2ll_t* to, index_t to_sz ) {
    index_t* p = calloc(to_sz, sizeof*p);
    for (index_t i=0; i<to_sz; ++i) p[i] = i;
    mergesort_permutation_r(p, to_sz, (void*)to, diverge_omp_num_threads(), 0);
    return p;
}

static inline void apply_permutation_in_place( claV2ll_t* vec, const index_t* p, index_t p_sz ) {
    cbytes_t* done = bitary_alloc(p_sz);
    for (index_t i=0; i<p_sz; ++i) {
        if (bitary_get(done, i)) continue;
        bitary_set(done,i);
        index_t prev_j = i;
        index_t j = p[i];
        while (i != j) {
            claV2ll_t tmp; tmp = vec[prev_j]; vec[prev_j] = vec[j]; vec[j] = tmp;
            bitary_set(done,j);
            prev_j = j;
            j = p[j];
        }
    }
    free(done);
}

static inline void apply_permutation( claV2ll_t* vec, const index_t* p, index_t p_sz ) {
    claV2ll_t* new_vec = malloc(p_sz * sizeof*new_vec);
    for (index_t i=0; i<p_sz; ++i) new_vec[i] = vec[p[i]];
    memcpy( vec, new_vec, sizeof*vec * p_sz );
    free( new_vec );
}

#ifdef DIVERGE_TU_FAST_PERMUTE
static void (*permutation_func)( claV2ll_t*, const index_t*, index_t ) = &apply_permutation;
#else // DIVERGE_TU_FAST_PERMUTE
static void (*permutation_func)( claV2ll_t*, const index_t*, index_t ) = &apply_permutation_in_place;
#endif // DIVERGE_TU_FAST_PERMUTE

void tu_projection_set_permutation_algorithm( int fast ) {
    permutation_func = fast ? &apply_permutation : &apply_permutation_in_place;
}

typedef struct { const claV2ll_t* _fhelp; const index_t* _off; index_t k; } serisort_fhelp_t;
static inline int serisort_fhelp_comp( index_t i, index_t j, const serisort_fhelp_t* h ) {
    const claV2ll_t fi = h->_fhelp[i+h->_off[h->k]],
                    fj = h->_fhelp[j+h->_off[h->k]];
    return (fi.v[1] == fj.v[1]) ? (fi.v[0] - fj.v[0]) : (fi.v[1] - fj.v[1]);
}
#undef SERISORT_COMP_R
#define SERISORT_COMP_R( x, y, ctx ) serisort_fhelp_comp( (x), (y), (const serisort_fhelp_t*)ctx )
SERISORT_IMPL_R( serisort_fhelp, index_t )

static vec_claV2ll_t sort_and_filter( vec_claV2ll_t* to, vec_claV2ll_t* from_help, vec_index_t* from,
        vec_claV2ll_t* rvec_help, vec_index_t* rvec, vec_index_t* count, vec_index_t* displ ) {
    index_t* p = sort_permutation( to->v, to->v_sz );
    #pragma omp parallel num_threads(MIN(diverge_omp_num_threads(),3))
    {
        #pragma omp single nowait
        (*permutation_func)( to->v, p, to->v_sz );
        #pragma omp single nowait
        (*permutation_func)( from_help->v, p, to->v_sz );
        #pragma omp single nowait
        (*permutation_func)( rvec_help->v, p, to->v_sz );
    }
    free( p );

    index_t msize = to->v_sz;
    vec_claV2ll_t to_padded = {0};
    VECTOR_RESIZE( to_padded.v, msize );
    memcpy( to_padded.v, to->v, msize*sizeof(claV2ll_t) );
    claV2ll_t lr = to->v[0];

    to->v_sz = 0;
    index_t len = 0;
    for (index_t i=0; i<msize; ++i) {
        if (claV2ll_equal(to_padded.v[i], lr)) {
            len++;
        } else {
            VECTOR_PUSH_BACK( count->v, len ); len = 1;
            VECTOR_PUSH_BACK( to->v, lr );
            lr = to_padded.v[i];
        }
    }
    VECTOR_PUSH_BACK( count->v, len );
    VECTOR_PUSH_BACK( to->v, lr );
    VECTOR_SHRINK_TO_FIT( to->v );

    VECTOR_RESIZE( from->v, msize*2 );
    VECTOR_RESIZE( rvec->v, msize*2 );
    VECTOR_RESIZE( displ->v, count->v_sz );
    displ->v[0] = 0;
    index_t maxlen = count->v[0];
    for (index_t i=1; i<count->v_sz; ++i) {
        displ->v[i] = displ->v[i-1] + count->v[i-1];
        maxlen = MAX(maxlen, count->v[i]);
    }
    index_t* pp = calloc(maxlen, sizeof*pp);
    index_t* pt = calloc(maxlen, sizeof*pt);
    for (index_t k=0; k<count->v_sz; ++k) {
        // set the index array
        for (index_t i=0; i<count->v[k]; ++i) pp[i] = i;
        // sort it
        serisort_fhelp_t ctx = { from_help->v, displ->v, k };
        serisort_fhelp_sr( pp, count->v[k], (void*)&ctx, pt );
        // and push back the sorted stuff
        for (index_t q=0; q<count->v[k]; ++q)
        for (int d=0; d<2; ++d) {
            from->v[2*(q+displ->v[k])+d] = from_help->v[displ->v[k]+pp[q]].v[d];
            rvec->v[2*(q+displ->v[k])+d] = rvec_help->v[displ->v[k]+pp[q]].v[d];
        }
        count->v[k] *= 2; displ->v[k] *= 2;
    }
    free( pp );
    free( pt );

    return to_padded;
}
