/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "helpers.h"
#include "loops.h"
#include "symmetry.h"
#include "vertex_memory.h"

#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../misc/mergesort.h"
#include "../misc/mpi_functions.h"
#include "../misc/btree.h"
#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

static void frg_loop_calc_P_greens( const bool isProjectedLoop, complex128_t* mem,
        const complex128_t* greens_p, const complex128_t* greens_m, index_t qstart,
        index_t qstop, index_t nk, index_t nkf, index_t nb, const index_t* kfrev,
        const double* kfweights, const index_t* k1m2 );
static void frg_loop_calc_C_greens( const bool isProjectedLoop, complex128_t* mem,
        const complex128_t* greens_p, const complex128_t* greens_m, index_t qstart,
        index_t qstop, index_t nk, index_t nkf, index_t nb, const index_t* kfrev,
        const double* kfweights, const index_t* k1m2 );
static void frg_loop_calc_greens( const char chan, const bool isProjectedLoop,
        complex128_t* mem, const complex128_t* greens_p, const complex128_t* greens_m,
        index_t qstart, index_t qstop, index_t nk, index_t nkf, index_t nb,
        const index_t* kfrev, const double* kfweights, const index_t* k1m2 );
static void transform_to_gbuf( const gf_complex_t* greens, complex128_t* greens_p,
        complex128_t* greens_m, const index_t* grid_kfmesh, index_t grid_nk_nkf,
        index_t div_nk_nkf, index_t nb );
static complex128_t* generate_trivial_formfactors( diverge_model_t* dmod, index_t* nff );

grid_loop_t* grid_loop_init( diverge_model_t* dmod ) {
    grid_loop_t* l = calloc(1, sizeof*l);

    l->dmod = dmod;
    l->nk = l->dmod->internals->grid_nk;
    l->nb = l->dmod->internals->grid_nb;
    l->greens_cache_p = (complex128_t*)l->dmod->internals->grid_greens_p_buf;
    l->greens_cache_m = (complex128_t*)l->dmod->internals->grid_greens_m_buf;

    l->nkf = l->dmod->internals->grid_nkf;
    l->k1m2 = l->dmod->internals->grid_k1m2;
    l->kfrev = l->dmod->internals->grid_kfrev;
    l->kfweights = l->dmod->internals->grid_weights;

    if (l->dmod->gproj != NULL) {
        l->greens_proj_cache_p = calloc(l->nk*l->nkf*l->nb*l->nb, sizeof(complex128_t));
        l->greens_proj_cache_m = calloc(l->nk*l->nkf*l->nb*l->nb, sizeof(complex128_t));
    }

    l->qstart = ((grid_vertex_t*)dmod->internals->grid_vertex)->qdispl;
    l->qsize = ((grid_vertex_t*)dmod->internals->grid_vertex)->qcount;

    l->Xsize = l->qsize*l->nk*POW4(l->nb);
    l->X_P = malloc( l->Xsize*sizeof(complex128_t) );
    l->X_C = malloc( l->Xsize*sizeof(complex128_t) );

    if (dmod->n_sym > 0) {
        l->SYM_enabled = true;
        l->SYM = grid_symm_transform_init( dmod, l->qstart, l->qsize );
    }

    l->formfac = generate_trivial_formfactors( dmod, &l->nff );
    return l;
}

void grid_loop_free( grid_loop_t* l ) {
    free(l->X_P);
    free(l->X_C);

    free(l->greens_proj_cache_p);
    free(l->greens_proj_cache_m);

    if (l->SYM) grid_symm_transform_free(l->SYM);

    free(l->formfac);
    free(l);
}

static void greens_cache_fill( grid_loop_t* l, double Lambda ) {
    if (l->greens_cache_lambda == Lambda) return;
    l->greens_cache_lambda = Lambda;

    if ((*l->dmod->gfill)(l->dmod, I*Lambda, l->dmod->internals->greens) != greensfunc_op_cpu)
        mpi_err_printf("grid FRG does not support GPU Green functions\n");
    transform_to_gbuf( l->dmod->internals->greens, l->greens_cache_p, l->greens_cache_m,
            l->dmod->internals->grid_kfmesh, l->dmod->internals->grid_nk * l->dmod->internals->grid_nkf,
            kdimtot(l->dmod->nk,l->dmod->nkf), l->nb );

    if (l->dmod->gproj != NULL) {
        if ((*l->dmod->gproj)(l->dmod, I*Lambda, l->dmod->internals->greens) != greensfunc_op_cpu)
            mpi_err_printf("grid FRG does not support GPU Green functions\n");
        transform_to_gbuf( l->dmod->internals->greens, l->greens_proj_cache_p, l->greens_proj_cache_m,
            l->dmod->internals->grid_kfmesh, l->dmod->internals->grid_nk * l->dmod->internals->grid_nkf,
            kdimtot(l->dmod->nk,l->dmod->nkf), l->nb );
    }
}

static complex128_t* symmetrize( grid_loop_t* l, complex128_t* loop, char chan ) {
    if (l->SYM_enabled && l->SYM != NULL) {
        double tick = diverge_mpi_wtime();
        loop = grid_symm_transform_loop( l->SYM, loop, chan );
        l->t_sym += diverge_mpi_wtime() - tick;
    }
    return loop;
}

complex128_t* grid_loop_calc( grid_loop_t* l, char chan, double Lambda ) {
    greens_cache_fill( l, Lambda );
    complex128_t* loopbuf = chan == 'P' ? l->X_P : l->X_C;
    frg_loop_calc_greens( chan, false, loopbuf, l->greens_cache_p, l->greens_cache_m,
            l->qstart, l->qstart+l->qsize, l->nk, l->nkf, l->nb, l->kfrev, l->kfweights, l->k1m2 );

    if (l->dmod->gproj != NULL)
        frg_loop_calc_greens( chan, true, loopbuf, l->greens_proj_cache_p, l->greens_proj_cache_m,
            l->qstart, l->qstart+l->qsize, l->nk, l->nkf, l->nb, l->kfrev, l->kfweights, l->k1m2 );

    loopbuf = symmetrize( l, loopbuf, chan );

    if (chan == 'P') l->max_P = absmax( loopbuf, l->Xsize );
    else             l->max_C = absmax( loopbuf, l->Xsize );

    return loopbuf;
}

void grid_loop_toggle_symmetrizer( grid_loop_t* l ) {
    l->SYM_enabled = !l->SYM_enabled;
}

static void frg_loop_calc_P_greens( const bool isProjectedLoop, complex128_t* mem,
        const complex128_t* greens_p, const complex128_t* greens_m, index_t qstart,
        index_t qstop, index_t nk, index_t nkf, index_t nb, const index_t* kfrev,
        const double* kfweights, const index_t* k1m2 ) {

    index_t LoopSize = (qstop-qstart)*nk*nb*nb*nb*nb;

    const double norm = 1./2./M_PI * (isProjectedLoop?-1.0:1.0);

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t Loopindex=0; Loopindex<LoopSize; ++Loopindex) {
        index_t q = Loopindex/(nk*nb*nb*nb*nb) + qstart,
              k = (Loopindex%(nk*nb*nb*nb*nb))/(nb*nb*nb*nb),
              o1 = (Loopindex%(nb*nb*nb*nb))/(nb*nb*nb),
              o2 = (Loopindex%(nb*nb*nb))/(nb*nb),
              o3 = (Loopindex%(nb*nb))/nb,
              o4 = Loopindex%nb;

        if (!isProjectedLoop) mem[Loopindex] = 0;

        index_t k1 = k;
        index_t k2 = k1m2[q*nk+k];

        for (index_t kf1=0; kf1<nkf; ++kf1) {
            index_t kf2 = kfrev[kf1];
            mem[Loopindex] += norm * kfweights[kf1] * ( greens_m[k1*nkf*nb*nb+kf1*nb*nb+o1*nb+o3] *
                                                        greens_p[k2*nkf*nb*nb+kf2*nb*nb+o2*nb+o4] +
                                                        greens_p[k1*nkf*nb*nb+kf1*nb*nb+o1*nb+o3] *
                                                        greens_m[k2*nkf*nb*nb+kf2*nb*nb+o2*nb+o4] );
        }
    }
}

static void frg_loop_calc_C_greens( const bool isProjectedLoop, complex128_t* mem,
        const complex128_t* greens_p, const complex128_t* greens_m, index_t qstart,
        index_t qstop, index_t nk, index_t nkf, index_t nb, const index_t* kfrev,
        const double* kfweights, const index_t* k1m2 ) {
    (void)kfrev;

    index_t LoopSize = (qstop-qstart)*nk*nb*nb*nb*nb;

    const double norm = 1./2./M_PI * (isProjectedLoop?-1.0:1.0);

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t Loopindex=0; Loopindex<LoopSize; ++Loopindex) {
        index_t q = Loopindex/(nk*nb*nb*nb*nb) + qstart,
              k = (Loopindex%(nk*nb*nb*nb*nb))/(nb*nb*nb*nb),
              O1 = (Loopindex%(nb*nb*nb*nb))/(nb*nb*nb),
              O2 = (Loopindex%(nb*nb*nb))/(nb*nb),
              O3 = (Loopindex%(nb*nb))/nb,
              O4 = Loopindex%nb;
        index_t o1 = O2, o2 = O3, o3 = O4, o4 = O1;

        if (!isProjectedLoop) mem[Loopindex] = 0;

        index_t k1 = k;
        index_t k2 = k1m2[k*nk+q];

        for (index_t kf1=0; kf1<nkf; ++kf1) {
            index_t kf2 = kf1;
            mem[Loopindex] += norm * kfweights[kf1] * ( greens_m[k1*nkf*nb*nb+kf1*nb*nb+o1*nb+o3] *
                                                        greens_m[k2*nkf*nb*nb+kf2*nb*nb+o2*nb+o4] +
                                                        greens_p[k1*nkf*nb*nb+kf1*nb*nb+o1*nb+o3] *
                                                        greens_p[k2*nkf*nb*nb+kf2*nb*nb+o2*nb+o4] );
        }
    }
}

static void frg_loop_calc_greens( const char chan, const bool isProjectedLoop,
        complex128_t* mem, const complex128_t* greens_p, const complex128_t* greens_m,
        index_t qstart, index_t qstop, index_t nk, index_t nkf, index_t nb,
        const index_t* kfrev, const double* kfweights, const index_t* k1m2 ) {
    if (chan == 'P') frg_loop_calc_P_greens( isProjectedLoop, mem, greens_p, greens_m,
                        qstart, qstop, nk, nkf, nb, kfrev, kfweights, k1m2 );
    else             frg_loop_calc_C_greens( isProjectedLoop, mem, greens_p, greens_m,
                        qstart, qstop, nk, nkf, nb, kfrev, kfweights, k1m2 );
}

static void transform_to_gbuf( const gf_complex_t* greens, complex128_t* greens_p,
        complex128_t* greens_m, const index_t* grid_kfmesh, index_t grid_nk_nkf,
        index_t div_nk_nkf, index_t nb ) {

    const gf_complex_t* div_gp = greens;
    const gf_complex_t* div_gm = greens + div_nk_nkf*nb*nb;
    const index_t nb2 = nb*nb;

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kg=0; kg<grid_nk_nkf; ++kg) {
        index_t kd = grid_kfmesh[kg];
        for (index_t bb=0; bb<nb2; ++bb)
            greens_p[kg*nb2+bb] = -div_gp[kd*nb2+bb];
    }

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kg=0; kg<grid_nk_nkf; ++kg) {
        index_t kd = grid_kfmesh[kg];
        for (index_t bb=0; bb<nb2; ++bb)
            greens_m[kg*nb2+bb] = -div_gm[kd*nb2+bb];
    }

}

static inline index_t claV3ll_to_key( claV3ll_t vec, index_t mabs ) {
    index_t  idx = IDX3( mabs+vec.v[0], mabs+vec.v[1], mabs+vec.v[2], 2*mabs+1, 2*mabs+1 );
    index_t nidx = IDX3( mabs-vec.v[0], mabs-vec.v[1], mabs-vec.v[2], 2*mabs+1, 2*mabs+1 );
    return MIN(idx, nidx);
}

static inline claV3ll_t key_to_claV3ll_t( index_t key, index_t mabs ) {
    claV3ll_t result = {0};
    index_t dim[3] = {2*mabs+1, 2*mabs+1, 2*mabs+1};
    unravel( key, dim, 3, result.v );
    return claV3ll_sub( result, (claV3ll_t){{mabs, mabs, mabs}} );
}

typedef struct { VECTOR_DECL( claV3ll_t, v ) index_t mabs; } vec_claV3ll_t;
static inline void candidates_write_out( btree_t* node, void* ctx ) {
    vec_claV3ll_t* ff = ctx;
    VECTOR_PUSH_BACK( ff->v, key_to_claV3ll_t(node->key, ff->mabs) );
}

static inline int mergesort_formfactors_comp( claV3ll_t i, claV3ll_t j, const claM3d_t* lat ) {
    double ni = claV3d_norm(claM3d_vecmat( (claV3d_t){{i.v[0],i.v[1],i.v[2]}}, *lat )),
           nj = claV3d_norm(claM3d_vecmat( (claV3d_t){{j.v[0],j.v[1],j.v[2]}}, *lat ));
    return ni > nj ? 1 : ni < nj ? -1 : 0;
}
#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R( x, y, ctx ) mergesort_formfactors_comp( x, y, ctx )
MERGESORT_IMPL_R( mergesort_formfactors, claV3ll_t )

static inline void normalize( complex128_t* v, index_t sz ) {
    double norm = 0.0;
    for (index_t i=0; i<sz; ++i) norm += POW2(cabs(v[i]));
    norm = 1./sqrt(norm);
    for (index_t i=0; i<sz; ++i) v[i] *= norm;
}

static complex128_t* generate_trivial_formfactors( diverge_model_t* dmod, index_t* nff ) {
    internals_t* intls = dmod->internals;
    claV3d_t lattice[3];
    for (int d=0; d<3; ++d) lattice[d] = claV3d_map(dmod->lattice[d]);

    VECTOR_INIT_CAP( claV3ll_t, rs_ff_candidates, POW2(GRID_FF_SHELL_DISTANCE*2+1) );
    double norm_fac = claV3d_norm(lattice[0]);
    for (int d=1; d<intls->dim; ++d) norm_fac += claV3d_norm(lattice[d]);
    norm_fac /= (double)intls->dim;
    if (norm_fac < DIVERGE_EPS_MESH) {
        mpi_wrn_printf( "tiny values for lattice vector norms (%.5e)\n", norm_fac );
        norm_fac = 1.0;
    }

    int run_dim[3] = {3*GRID_FF_SHELL_DISTANCE, 3*GRID_FF_SHELL_DISTANCE, 3*GRID_FF_SHELL_DISTANCE};
    for (int d=intls->dim; d<3; ++d) run_dim[d] = 0;
    index_t mabs = 0;
    for (int r1=-run_dim[0]; r1<=run_dim[0]; ++r1)
    for (int r2=-run_dim[1]; r2<=run_dim[1]; ++r2)
    for (int r3=-run_dim[2]; r3<=run_dim[2]; ++r3) {
        claV3d_t R = claV3d_add(claV3d_add(claV3d_scale(lattice[0], r1),
                                           claV3d_scale(lattice[1], r2)),
                                           claV3d_scale(lattice[2], r3));
        if ((claV3d_norm(R)/norm_fac) < GRID_FF_SHELL_DISTANCE && (r1 != 0 || r2 != 0 || r3 != 0)) {
            VECTOR_PUSH_BACK( rs_ff_candidates, (claV3ll_t){{r1,r2,r3}} );
            mabs = MAX(abs(r1), mabs); mabs = MAX(abs(r2), mabs); mabs = MAX(abs(r3), mabs);
        }
    }
    mabs++;

    // only the 'positive' ones, claV3ll_to_key maps positive/negative vecs to the same integer
    xbtree_t* rs_ff = xbtree_init( MAX(rs_ff_candidates_sz/2+1, 512) );
    for (index_t x=0; x<rs_ff_candidates_sz; ++x)
        xbtree_find_or_insert( rs_ff, claV3ll_to_key(rs_ff_candidates[x], mabs) );

    vec_claV3ll_t rs_ff_vec = { .v=rs_ff_candidates, .v_sz=0,
        .v_cap=rs_ff_candidates_cap, .mabs=mabs };

    xbtree_iter( rs_ff, &candidates_write_out, &rs_ff_vec );
    xbtree_free( rs_ff );

    mergesort_formfactors_r( rs_ff_vec.v, rs_ff_vec.v_sz, &lattice, diverge_omp_num_threads(), 0 );
    *nff = rs_ff_vec.v_sz*2+1;

    if (*nff > intls->grid_nk) {
        mpi_wrn_printf( "small grid (nk=%li), truncating #formfactors (nff=%li)\n",
                intls->grid_nk, *nff );
        rs_ff_vec.v_sz = MAX(intls->grid_nk/2-1,0);
        *nff = intls->grid_nk;
    }
    complex128_t* formfac = malloc(sizeof(complex128_t)*(*nff)*intls->grid_nk);
    complex128_t* buf = formfac;
    for (index_t k=0; k<intls->grid_nk; ++k) buf[k] = 1.0;
    buf += intls->grid_nk;
    for (index_t ff=0; ff<rs_ff_vec.v_sz; ++ff) {
        claV3d_t Rvec = claM3d_vecmat( (claV3d_t){{rs_ff_vec.v[ff].v[0],
                rs_ff_vec.v[ff].v[1],rs_ff_vec.v[ff].v[2]}}, claM3d_map(dmod->lattice[0]) );
        for (index_t sign=0; sign<2; ++sign) {
            for (index_t k=0; k<intls->grid_nk; ++k) {
                claV3d_t Kvec = claV3d_map(intls->kmesh+3*k);
                buf[k] = sign ? cos(claV3d_dot(Rvec,Kvec)) : sin(claV3d_dot(Rvec,Kvec));
            }
            buf += intls->grid_nk;
        }
    }

    VECTOR_FREE( rs_ff_candidates );

    for (index_t ff=0; ff<*nff; ++ff) normalize(formfac+ff*intls->grid_nk, intls->grid_nk);
    return formfac;
}

