#include "rpa.h"
#include "vertex.h"
#include "vertex_details.h"
#include "../diverge_flow_step.h"
#include "../diverge_flow_step_internal.h"
#include "../misc/hashtable.h"
#include "../diverge_model.h"

void diverge_flow_step_sc_vertex( diverge_flow_step_t* s, char chan, const index_t* kidx,
        index_t n_kidx, complex128_t* out) {

    diverge_flow_type tp = diverge_flow_step_type(s);
    if (tp != diverge_tu) {
        mpi_err_printf( "sc vertex only possible with TUFRG\n" );
        return;
    }
    if (kdim(s->mod->nkf) != 1) {
        mpi_err_printf( "sc vertex only possible with nkf=1\n" );
        return;
    }

    const index_t* nk = s->mod->nk;
    tu_vertex_t* V = s->tu_vertex;
    tu_projection_t* P = s->tu_proj;

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

    hashtable_t* ktab = hashtable_init( 8*n_kidx, 8 );
    for (index_t ik=0; ik<n_kidx; ++ik)
        hashtable_insert_safe(ktab, kidx[ik], (void*)(ik+1));

    complex128_t* Vbuf_reco = calloc(POW2(V->n_orbff*POW2(V->n_spin)) * q_len, sizeof*Vbuf_reco);
    diverge_tu_vertex_reco( V, chan, Vbuf_reco, Vbuf );

    const index_t legdim = POW2(V->n_orb*V->n_spin) * n_kidx;
    const index_t *mi_to_ofrom = V->tu_data->mi_to_ofrom,
                  *mi_to_oto = V->tu_data->mi_to_oto;
    ZERO_OUT( out, POW2(legdim) );

    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t ik=0; ik<n_kidx; ++ik)
    for (index_t iq=0; iq<q_len; ++iq) {
        index_t kk = kidx[ik],
                qq = q_off+iq;
        index_t kp = (chan == 'C') ? k1m2(qq,kk, nk) : k1p2(qq,kk, nk);
        index_t ikp = (index_t)(hashtable_find(ktab, kp))-1;
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
                    out[IDX2(left, right, legdim)] += Vbuf_reco[IDX7(s4,s3,m2,s2,s1,m1,iq,
                            V->n_spin,V->n_orbff, V->n_spin,V->n_spin,V->n_orbff, q_len)];
                }
                if (chan == 'D') { // (1,3,4,2)
                    index_t o1 = mi_to_ofrom[m1], o3 = mi_to_oto[m1],
                            o4 = mi_to_ofrom[m2], o2 = mi_to_oto[m2];
                    index_t left  = IDX5(ik, s1,o1,s2,o2, V->n_spin,V->n_orb,V->n_spin,V->n_orb),
                            right = IDX5(ikp,s3,o3,s4,o4, V->n_spin,V->n_orb,V->n_spin,V->n_orb);
                    out[IDX2(left, right, legdim)] += Vbuf_reco[IDX7(s4,s3,m2,s2,s1,m1,iq,
                            V->n_spin,V->n_orbff, V->n_spin,V->n_spin,V->n_orbff, q_len)];
                }
            }
        }
    }

    diverge_mpi_allreduce_complex_sum_inplace(out, POW2(legdim));
    free( Vbuf_reco );
    hashtable_free( ktab );
}

void diverge_flow_step_sc_greens( diverge_flow_step_t* s, const index_t* kidx,
        index_t n_kidx, double Lambda, double dLambda, complex128_t* out ) {
    if (kdim(s->mod->nkf) != 1) {
        mpi_err_printf( "sc greens only possible with nkf=1\n" );
        return;
    }

    diverge_model_t* m = s->mod;
    complex128_t cLambda = I*Lambda;
    gf_complex_t* gbuf = m->internals->greens;
    (*m->gfill)(m, cLambda, gbuf);
    const index_t NK = kdim(m->nk);
    const double prefac = dLambda/(2.*M_PI);

    index_t legdim = POW2(m->n_orb*m->n_spin);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t ik=0; ik<n_kidx; ++ik) {
        index_t kk = kidx[ik];
        index_t mk = k1m2(0,kk,m->nk);
        for (index_t s1=0; s1<m->n_spin; ++s1)
        for (index_t o1=0; o1<m->n_orb;  ++o1)
        for (index_t s2=0; s2<m->n_spin; ++s2)
        for (index_t o2=0; o2<m->n_orb;  ++o2)
        for (index_t s3=0; s3<m->n_spin; ++s3)
        for (index_t o3=0; o3<m->n_orb;  ++o3)
        for (index_t s4=0; s4<m->n_spin; ++s4)
        for (index_t o4=0; o4<m->n_orb;  ++o4) {
            index_t left  = IDX4(s1,o1,s2,o2, m->n_orb,m->n_spin,m->n_orb),
                    right = IDX4(s3,o3,s4,o4, m->n_orb,m->n_spin,m->n_orb);
            out[IDX3(ik,left,right, legdim,legdim)] += prefac * (
                gbuf[IDX3(0,kk,left,NK,legdim)] * gbuf[IDX3(1,mk,right,NK,legdim)] +
                gbuf[IDX3(1,mk,left,NK,legdim)] * gbuf[IDX3(0,kk,right,NK,legdim)] );
        }
    }
}
