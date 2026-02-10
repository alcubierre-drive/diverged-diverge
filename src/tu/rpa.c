/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "rpa.h"
#include "vertex.h"
#include "vertex_details.h"
#include "../diverge_flow_step.h"
#include "../diverge_flow_step_internal.h"
#include "../misc/arnoldi.h"
#include "../misc/eigen.h"
#include "../misc/gemm.h"
#include "../diverge_model.h"

#define vhash_key_t index_t
#define vhash_val_t index_t
#define vhash_key_empty -1
#define vhash_val_empty -1
#include "../misc/vhashtable.h"
VHASH_IMPLEMENT(rpahash)

static char tu_rpa_post_which_ev[16] = "SR";
static char tu_rpa_post_sorting = 'N';
void tu_rpa_post_set_which_ev( char sort ) {
    switch (sort) {
        case 'P':
            tu_rpa_post_sorting = 'P';
            strcpy( tu_rpa_post_which_ev, "LR" );
            break;
        case 'N':
            tu_rpa_post_sorting = 'N';
            strcpy( tu_rpa_post_which_ev, "SR" );
            break;
        case 'M':
            tu_rpa_post_sorting = 'M';
            strcpy( tu_rpa_post_which_ev, "LM" );
            break;
        default: break;
    }
}

tu_rpa_t* tu_rpa_init( diverge_flow_step_t* s, const index_t* kidx, index_t n_kidx ) {
    diverge_flow_type tp = diverge_flow_step_type(s);
    if (tp != diverge_tu && tp != diverge_tu_phonons) {
        mpi_err_printf( "tu-rpa only possible with TUFRG\n" );
        return NULL;
    }
    diverge_model_t* m = s->mod;
    if (kdim(m->nkf) != 1) {
        mpi_err_printf( "tu-rpa only possible with nkf=1\n" );
        return NULL;
    }
    tu_rpa_t* t = calloc(1, sizeof*t);
    t->s = s;
    t->m = m;
    t->n_kidx = n_kidx;
    t->kidx = calloc(n_kidx, sizeof*t->kidx);
    memcpy( t->kidx, kidx, sizeof*t->kidx * n_kidx );

    t->chi_pp = calloc( t->n_kidx * POW4(m->n_orb*m->n_spin), sizeof*t->chi_pp );
    return t;
}

void tu_rpa_step( tu_rpa_t* t, double Lambda, double dLambda ) {
    diverge_model_t* m = t->m;
    complex128_t cLambda = I*Lambda;
    gf_complex_t* gbuf = m->internals->greens;
    (*m->gfill)(m, cLambda, gbuf);
    const index_t NK = kdim(m->nk);
    const double prefac = dLambda/(2.*M_PI);

    const index_t nso = m->n_orb*m->n_spin;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t ik=0; ik<t->n_kidx; ++ik) {
        index_t kk = t->kidx[ik];
        index_t mk = k1m2(0,kk,m->nk);
        for (index_t so1=0; so1<nso; ++so1)
        for (index_t so2=0; so2<nso; ++so2)
        for (index_t so3=0; so3<nso; ++so3)
        for (index_t so4=0; so4<nso; ++so4)
            t->chi_pp[IDX5(ik,so1,so2,so3,so4, nso,nso,nso,nso)] += prefac * (
                gbuf[IDX4(0,kk,so1,so3, NK,nso,nso)] * gbuf[IDX4(1,mk,so2,so4, NK,nso,nso)] +
                gbuf[IDX4(1,mk,so3,so1, NK,nso,nso)] * gbuf[IDX4(0,kk,so4,so2, NK,nso,nso)] );
    }
}

// allocates and fills t->V_pp given chan following flow_step naming
static void tu_rpa_post_vert( tu_rpa_t* t, char chan );
// applies t->chi_pp to t->V_pp (rhs)
static void tu_rpa_post_apply_greens( tu_rpa_t* t );

void tu_rpa_post( tu_rpa_t* t, const char* chans, int nev, int ncv, int maxiter, double tol ) {
    mpi_log_printf( "RPA postprocessing (TU) with channels %s\n", chans );
    for (const char* c=chans; *c; c++)
        tu_rpa_post_vert( t, *c );
    tu_rpa_post_apply_greens( t );
    t->legdim = POW2(t->m->n_orb*t->m->n_spin)*t->n_kidx;
    transpose_inplace( t->V_pp, t->legdim );

    t->nev = nev;
    t->vecs = calloc(nev*t->legdim, sizeof*t->vecs);
    t->vals = calloc(MAX(t->legdim,2*nev), sizeof*t->vals);
    if (!arnoldi_eigen_dense( tu_rpa_post_which_ev, t->legdim, nev, ncv, t->V_pp,
                (complex128_t*)t->vals, t->vecs, tol, maxiter )) {
        for (int i=0; i<nev; ++i) t->vals[i] = t->vals[2*i]; // copy real part
        for (int i=nev; i<t->legdim; ++i) t->vals[i] = 0.0; // set everything else to zero
    } else { // could not do arnoldi, fall back to dense eigen
        mpi_wrn_printf( "arpack not compiled in, fall back to lapack\n" );
        t->vecs = realloc(t->vecs, sizeof*t->vecs * POW2(t->legdim));
        complex128_t* buf = calloc(POW2(t->legdim), sizeof*buf);
        single_svd(t->V_pp, t->vecs, buf, t->vals, -t->legdim );
        transpose_inplace( t->vecs, t->legdim );
        free( buf );
    }
    single_eigen_sort_part( t->vecs, t->vals, t->legdim, tu_rpa_post_sorting, t->nev );

    free( t->V_pp );   t->V_pp   = NULL;
    free( t->chi_pp ); t->chi_pp = NULL;
}

void tu_rpa_free( tu_rpa_t* t ) {
    free( t->kidx );
    if (t->chi_pp) free( t->chi_pp );
    if (t->V_pp) free( t->V_pp );
    if (t->vals) free( t->vals );
    if (t->vecs) free( t->vecs );
    free( t );
}

static void tu_rpa_post_vert( tu_rpa_t* t, char chan ) {
    const index_t* nk = t->m->nk;
    tu_vertex_t* V = t->s->tu_vertex;
    tu_projection_t* P = t->s->tu_proj;

    int full_vtx = islower(chan);
    chan = toupper(chan);
    if (chan != 'C' && chan != 'D' && chan != 'M' && chan != 'H') {
        mpi_err_printf( "invalid channel '%c', use cdCDmh, reset to 'C'\n", chan );
        chan = 'C';
    }
    if (chan == 'M' || chan == 'H') full_vtx = 1;

    complex128_t* Vbuf = NULL;
    if (full_vtx) {
        if (chan == 'C') C_projection(V->projection_helper, V, P);
        if (chan == 'D') D_projection(V->projection_helper, V, P);
        if (chan == 'M' || chan == 'H') {
            C_projection(V->projection_helper, V, P);
            for (index_t i=0; i<V->full_vert_size; ++i)
                V->projection_helper[i] *= -1.0;
            if (chan == 'H') {
                complex128_t* helper = calloc(V->full_vert_size, sizeof*helper);
                D_projection(helper, V, P);
                for (index_t i=0; i<V->full_vert_size; ++i)
                    V->projection_helper[i] += 2.0*helper[i];
                free(helper);
            }
            if (chan == 'M') chan = 'C';
            if (chan == 'H') chan = 'D';
        }
        Vbuf = V->projection_helper;
    } else {
        Vbuf = (chan == 'C') ? V->Cch : V->Dch;
    }

    index_t q_off = V->my_nk_off,
            q_len = V->my_nk;
    if (V->symm->use_symmetries) {
        q_off = V->symm->my_nk_in_bz_off;
        q_len = V->symm->my_nk_in_bz;
    }

    rpahashtable_t* ktab = rpahashtable_init( 64*t->n_kidx, 8 );
    for (index_t ik=0; ik<t->n_kidx; ++ik)
        rpahashtable_insert_safe(ktab, t->kidx[ik], ik);

    complex128_t* Vbuf_reco = calloc(POW2(V->n_orbff*POW2(V->n_spin)) * q_len, sizeof*Vbuf_reco);
    diverge_tu_vertex_reco( V, chan, Vbuf_reco, Vbuf );

    const index_t legdim = POW2(V->n_orb*V->n_spin) * t->n_kidx;
    const index_t *mi_to_ofrom = V->tu_data->mi_to_ofrom,
                  *mi_to_oto = V->tu_data->mi_to_oto;
    if (!t->V_pp) t->V_pp = calloc( POW2(legdim), sizeof*t->V_pp );

    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t ik=0; ik<t->n_kidx; ++ik)
    for (index_t iq=0; iq<q_len; ++iq) {
        index_t kk = t->kidx[ik],
                qq = q_off+iq;
        index_t kp = (chan == 'C') ? k1m2(qq,kk, nk) : k1p2(qq,kk, nk);
        index_t ikp = rpahashtable_find(ktab, kp);
        if (ikp != -1) {
            for (index_t m1=0; m1<V->n_orbff; ++m1)
            for (index_t s1=0; s1<V->n_spin; ++s1)
            for (index_t s2=0; s2<V->n_spin; ++s2)
            for (index_t m2=0; m2<V->n_orbff; ++m2)
            for (index_t s3=0; s3<V->n_spin; ++s3)
            for (index_t s4=0; s4<V->n_spin; ++s4) {
                if (chan == 'C') { // (1,4,3,2)
                    index_t o1 = mi_to_ofrom[m1], o4 = mi_to_oto[m1],
                            o3 = mi_to_ofrom[m2], o2 = mi_to_oto[m2];
                    index_t left  = IDX5(ik, s1,o1,s2,o2, V->n_spin,V->n_orb,V->n_spin,V->n_orb),
                            right = IDX5(ikp,s3,o3,s4,o4, V->n_spin,V->n_orb,V->n_spin,V->n_orb);
                    t->V_pp[IDX2(left, right, legdim)] += Vbuf_reco[IDX7(s4,s3,m2,s2,s1,m1,iq,
                            V->n_spin,V->n_orbff, V->n_spin,V->n_spin,V->n_orbff, q_len)];
                }
                if (chan == 'D') { // (1,3,4,2)
                    index_t o1 = mi_to_ofrom[m1], o3 = mi_to_oto[m1],
                            o4 = mi_to_ofrom[m2], o2 = mi_to_oto[m2];
                    index_t left  = IDX5(ik, s1,o1,s2,o2, V->n_spin,V->n_orb,V->n_spin,V->n_orb),
                            right = IDX5(ikp,s3,o3,s4,o4, V->n_spin,V->n_orb,V->n_spin,V->n_orb);
                    t->V_pp[IDX2(left, right, legdim)] += Vbuf_reco[IDX7(s4,s3,m2,s2,s1,m1,iq,
                            V->n_spin,V->n_orbff, V->n_spin,V->n_spin,V->n_orbff, q_len)];
                }
            }
        }
    }

    diverge_mpi_allreduce_complex_sum_inplace(t->V_pp, POW2(legdim));
    free( Vbuf_reco );
    rpahashtable_free( ktab );
}

static void tu_rpa_post_apply_greens( tu_rpa_t* t ) {
    diverge_model_t* m = t->m;
    const index_t nso2 = POW2(m->n_orb*m->n_spin);
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        complex128_t* buf1 = calloc(POW2(nso2), sizeof*buf1);
        complex128_t* buf2 = calloc(POW2(nso2), sizeof*buf2);
        #pragma omp for collapse(2)
        for (index_t k1=0; k1<t->n_kidx; ++k1)
        for (index_t k2=0; k2<t->n_kidx; ++k2) {
            ZERO_OUT( buf1, POW2(nso2) );
            ZERO_OUT( buf2, POW2(nso2) );
            // copy V_pp[k1,k2] for all orbs to buf1
            for (index_t so12=0; so12<nso2; ++so12)
            for (index_t so34=0; so34<nso2; ++so34)
                buf1[IDX2(so12,so34, nso2)] = t->V_pp[IDX4(k1,so12,k2,so34, nso2,t->n_kidx,nso2)];
            // perform the GEMM
            single_gemm_serial( buf1, t->chi_pp + k2*POW2(nso2), buf2, nso2, nso2, nso2, -0.5, 0.0 );
            // copy buf2 for all orbs to V_pp[k1,k2]
            for (index_t so12=0; so12<nso2; ++so12)
            for (index_t so34=0; so34<nso2; ++so34)
                t->V_pp[IDX4(k1,so12,k2,so34, nso2,t->n_kidx,nso2)] = buf2[IDX2(so12,so34, nso2)];
        }
        free(buf1);
        free(buf2);
    }
}

