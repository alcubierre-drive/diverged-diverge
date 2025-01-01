/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "symmetrize_vertex.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../diverge_symmetrize.h"
#include "../misc/mpi_functions.h"

static double dot(double* A, double* B) {
    return A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
}

void symmetrize_npatch_vertex( diverge_model_t* model, complex128_t* buf, const index_t* map, index_t np, complex128_t* aux ) {
    if (model->n_sym == 0 || model->orb_symmetries == NULL) return;

    diverge_generate_symm_maps(model);
    index_t nk = kdim(model->nk);
    index_t nb = model->n_orb * model->n_spin;
    index_t n_spin = model->n_spin;
    index_t n_orb = model->n_orb;
    index_t size = POW3(np) * POW4(nb);

    bool free_aux = false;
    if (!aux) {
        aux = malloc(sizeof(complex128_t)*size + sizeof(index_t)*(1+nk));
        memset( aux, 0, sizeof(complex128_t)*size + sizeof(index_t)*(1+nk) );
        free_aux = true;
    }

    index_t* imap = (index_t*)(aux + size) + 1;
    index_t* symm_map_mom = model->internals->symm_map_mom_crs;
    index_t n_sym = model->n_sym;
    if (!imap[-1]) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for(index_t k = 0; k < nk; ++k) imap[k] = -1;
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for(index_t k = 0; k < np; ++k) imap[map[k]] = k;

        index_t missing = 0;
        #pragma omp parallel for collapse(2) reduction(+:missing) num_threads(diverge_omp_num_threads())
        for ( index_t m1 = 0; m1 < np; ++m1 )
        for ( index_t s = 0; s < n_sym; ++s ) {
            index_t mk1 = imap[symm_map_mom[s + n_sym * map[m1]]];
            missing += (mk1 == -1);
        }

        if (missing > 0) {
            mpi_err_printf( "map is not symmetric, cannot symmetrize\n" );
            if (free_aux) free(aux);
            return;
        }
        imap[-1] = 1;
    }

    // TODO the routine is not thoroughly tested for true patching systems...
    int niter=1;
    while (niter--) {

    memcpy( aux, buf, sizeof(complex128_t)*size );
    memset( buf, 0, sizeof(complex128_t)*size );

    double* symm_beyond_UC = model->internals->symm_beyond_UC;
    double* kmesh = model->internals->kmesh;
    index_t* symm_map_orb = model->internals->symm_map_orb;
    index_t* symm_orb_off = model->internals->symm_orb_off;
    index_t* symm_orb_len = model->internals->symm_orb_len;
    complex128_t* symm_pref = model->internals->symm_pref;

    double norm = 1./(double) n_sym;

    #pragma omp parallel for collapse(11) num_threads(diverge_omp_num_threads())
    for (index_t p1=0; p1<np; ++p1)
    for (index_t p2=0; p2<np; ++p2)
    for (index_t p3=0; p3<np; ++p3)
    for (index_t oo1=0; oo1<n_orb; ++oo1)
    for (index_t ss1=0; ss1<n_spin; ++ss1)
    for (index_t oo2=0; oo2<n_orb; ++oo2)
    for (index_t ss2=0; ss2<n_spin; ++ss2)
    for (index_t oo3=0; oo3<n_orb; ++oo3)
    for (index_t ss3=0; ss3<n_spin; ++ss3)
    for (index_t oo4=0; oo4<n_orb; ++oo4)
    for (index_t ss4=0; ss4<n_spin; ++ss4) {
        index_t o1 = IDX2( ss1, oo1, n_orb ),
                o2 = IDX2( ss2, oo2, n_orb ),
                o3 = IDX2( ss3, oo3, n_orb ),
                o4 = IDX2( ss4, oo4, n_orb );
        index_t k1 = map[p1],
                k2 = map[p2],
                k3 = map[p3];
        index_t k4 = k1m2( k1p2( k1, k2, model->nk ), k3, model->nk );
        for (index_t s=0; s<n_sym; ++s) {
            index_t os1 = IDX2( o1, s, n_sym ),
                    os2 = IDX2( o2, s, n_sym ),
                    os3 = IDX2( o3, s, n_sym ),
                    os4 = IDX2( o4, s, n_sym );
            index_t ks1 = symm_map_mom[IDX2(k1,s,n_sym)],
                    ks2 = symm_map_mom[IDX2(k2,s,n_sym)],
                    ks3 = symm_map_mom[IDX2(k3,s,n_sym)],
                    ks4 = symm_map_mom[IDX2(k4,s,n_sym)];
            index_t mp1 = imap[ks1],
                    mp2 = imap[ks2],
                    mp3 = imap[ks3];
            for (index_t i1=0; i1<symm_orb_len[os1]; ++i1)
            for (index_t i2=0; i2<symm_orb_len[os2]; ++i2)
            for (index_t i3=0; i3<symm_orb_len[os3]; ++i3)
            for (index_t i4=0; i4<symm_orb_len[os4]; ++i4) {
                index_t xo1 = symm_orb_off[os1] + i1,
                        xo2 = symm_orb_off[os2] + i2,
                        xo3 = symm_orb_off[os3] + i3,
                        xo4 = symm_orb_off[os4] + i4;
                index_t mo1 = symm_map_orb[xo1],
                        mo2 = symm_map_orb[xo2],
                        mo3 = symm_map_orb[xo3],
                        mo4 = symm_map_orb[xo4];
                double kdotR11 = dot(symm_beyond_UC+3*IDX2(oo1,s,n_sym), kmesh+3*ks1);
                double kdotR22 = dot(symm_beyond_UC+3*IDX2(oo2,s,n_sym), kmesh+3*ks2);
                double kdotR33 = dot(symm_beyond_UC+3*IDX2(oo3,s,n_sym), kmesh+3*ks3);
                double kdotR44 = dot(symm_beyond_UC+3*IDX2(oo4,s,n_sym), kmesh+3*ks4);
                buf[IDX7(p1,p2,p3, o1,o2,o3,o4, np,np, nb,nb,nb,nb)] +=
                    aux[IDX7(mp1,mp2,mp3, mo1,mo2,mo3,mo4, np,np, nb,nb,nb,nb)] *
                        (symm_pref[xo1] * (cos(kdotR11) - I * sin(kdotR11))) *
                        (symm_pref[xo2] * (cos(kdotR22) - I * sin(kdotR22))) *
                    conj(symm_pref[xo3] * (cos(kdotR33) - I * sin(kdotR33))) *
                    conj(symm_pref[xo4] * (cos(kdotR44) - I * sin(kdotR44))) * norm;
            }
        }
    }

    } // niter
    if (free_aux) free(aux);
}

