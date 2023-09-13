/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model.h"
#include "diverge_symmetrize.h"
#include "diverge_internals_struct.h"

#define alloc_aux( size ) \
    bool free_aux = false; \
    if (!aux) { \
        aux = malloc(size); \
        free_aux = true; \
    } \
    { \
    memcpy( aux, buf, size ); \
    memset( buf, 0, size ); \
    }

#define free_aux() { \
    if (free_aux) free(aux); \
}
static double dot(double* A, double* B) {
    return A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
}

double diverge_symmetrize_2pt_coarse( diverge_model_t* model, complex128_t* buf, complex128_t* aux ) {
    if (model->n_sym == 0 || model->orb_symmetries == NULL) return -1.0;

    diverge_generate_symm_maps(model);
    index_t nk = kdim( model->nk );
    index_t n_os = model->n_orb * model->n_spin;
    index_t n_spin = model->n_spin;
    index_t n_orb = model->n_orb;
    index_t size = POW2(n_os) * nk;

    alloc_aux( sizeof(complex128_t)*size );

    index_t n_sym = model->n_sym;
    double norm = 1./(double) n_sym;

    index_t* symm_map_mom = model->internals->symm_map_mom_crs;
    double* symm_beyond_UC = model->internals->symm_beyond_UC;
    double* kmesh = model->internals->kmesh;
    index_t* symm_map_orb = model->internals->symm_map_orb;
    index_t* symm_orb_off = model->internals->symm_orb_off;
    index_t* symm_orb_len = model->internals->symm_orb_len;
    complex128_t* symm_pref = model->internals->symm_pref;

    #pragma omp parallel for collapse(5)
    for (index_t k=0; k<nk; ++k)
    for (index_t oo1=0; oo1<n_orb; ++oo1)
    for (index_t ss1=0; ss1<n_spin; ++ss1)
    for (index_t oo2=0; oo2<n_orb; ++oo2)
    for (index_t ss2=0; ss2<n_spin; ++ss2) {
        index_t o1 = oo1 + n_orb * ss1;
        index_t o2 = oo2 + n_orb * ss2;
        for (index_t s=0; s<n_sym; ++s) {
            index_t osym1 = IDX2( o1, s, n_sym ),
                    osym2 = IDX2( o2, s, n_sym ),
                    ksym = symm_map_mom[IDX2(k,s,n_sym)];
            for (index_t i=0; i<symm_orb_len[osym1]; ++i)
            for (index_t j=0; j<symm_orb_len[osym2]; ++j) {
                index_t xo1 = symm_orb_off[osym1] + i,
                        xo2 = symm_orb_off[osym2] + j;
                double kdotR11 = dot(symm_beyond_UC+3*IDX2(oo1,s,n_sym), kmesh+3*ksym);
                double kdotR22 = dot(symm_beyond_UC+3*IDX2(oo2,s,n_sym), kmesh+3*ksym);

                buf[IDX3(k,o1,o2,n_os,n_os)] += aux[IDX3(ksym,symm_map_orb[xo1],symm_map_orb[xo2], n_os,n_os)] *
                                                (symm_pref[xo1] * (cos(kdotR11) - I * sin(kdotR11))) *
                                            conj(symm_pref[xo2] * (cos(kdotR22) - I * sin(kdotR22))) * norm;
            }
        }
    }
    double max_diff = 0.;
    #pragma omp parallel for reduction(max:max_diff)
    for (index_t idx=0; idx<nk*n_os*n_os; ++idx)
        max_diff = MAX(max_diff, cabs(buf[idx]-aux[idx]));

    free_aux();
    return max_diff;
}

double diverge_symmetrize_2pt_fine( diverge_model_t* model, complex128_t* buf, complex128_t* aux ) {
    if (model->n_sym == 0 || model->orb_symmetries == NULL) return -1.0;

    diverge_generate_symm_maps(model);
    index_t nk = kdimtot( model->nk, model->nkf );
    index_t n_os = model->n_orb * model->n_spin;
    index_t n_spin = model->n_spin;
    index_t n_orb = model->n_orb;
    index_t size = POW2(n_os) * nk;

    alloc_aux( sizeof(complex128_t)*size );

    index_t n_sym = model->n_sym;
    double norm = 1./(double) n_sym;

    index_t* symm_map_mom = model->internals->symm_map_mom_fine;
    double* symm_beyond_UC = model->internals->symm_beyond_UC;
    double* kmesh = model->internals->kfmesh;
    index_t* symm_map_orb = model->internals->symm_map_orb;
    index_t* symm_orb_off = model->internals->symm_orb_off;
    index_t* symm_orb_len = model->internals->symm_orb_len;
    complex128_t* symm_pref = model->internals->symm_pref;

    #pragma omp parallel for collapse(5)
    for (index_t k=0; k<nk; ++k)
    for (index_t oo1=0; oo1<n_orb; ++oo1)
    for (index_t ss1=0; ss1<n_spin; ++ss1)
    for (index_t oo2=0; oo2<n_orb; ++oo2)
    for (index_t ss2=0; ss2<n_spin; ++ss2) {
        index_t o1 = oo1 + n_orb * ss1;
        index_t o2 = oo2 + n_orb * ss2;
        for (index_t s=0; s<n_sym; ++s) {
            index_t osym1 = IDX2( o1, s, n_sym ),
                    osym2 = IDX2( o2, s, n_sym ),
                    ksym = symm_map_mom[IDX2(k,s,n_sym)];
            for (index_t i=0; i<symm_orb_len[osym1]; ++i)
            for (index_t j=0; j<symm_orb_len[osym2]; ++j) {
                index_t xo1 = symm_orb_off[osym1] + i,
                        xo2 = symm_orb_off[osym2] + j;
                double kdotR11 = dot(symm_beyond_UC+3*IDX2(oo1,s,n_sym), kmesh+3*ksym);
                double kdotR22 = dot(symm_beyond_UC+3*IDX2(oo2,s,n_sym), kmesh+3*ksym);

                buf[IDX3(k,o1,o2,n_os,n_os)] += aux[IDX3(ksym,symm_map_orb[xo1],symm_map_orb[xo2], n_os,n_os)] *
                                                (symm_pref[xo1] * (cos(kdotR11) - I * sin(kdotR11))) *
                                            conj(symm_pref[xo2] * (cos(kdotR22) - I * sin(kdotR22))) * norm;
            }
        }
    }
    double max_diff = 0.;
    #pragma omp parallel for reduction(max:max_diff)
    for (index_t idx=0; idx<nk*n_os*n_os; ++idx) {
        max_diff = MAX(max_diff, cabs(buf[idx]-aux[idx]));
    }

    free_aux();
    return max_diff;
}

double diverge_symmetrize_greens( diverge_model_t* model, gf_complex_t* buf, gf_complex_t* aux ) {
    if (model->n_sym == 0 || model->orb_symmetries == NULL) return -1.0;

    diverge_generate_symm_maps(model);
    index_t nk = kdimtot( model->nk, model->nkf );
    index_t n_os = model->n_orb * model->n_spin;
    index_t n_spin = model->n_spin;
    index_t n_orb = model->n_orb;
    index_t size = POW2(n_os) * nk;

    alloc_aux( sizeof(gf_complex_t)*size );

    index_t n_sym = model->n_sym;
    double norm = 1./(double) n_sym;

    index_t* symm_map_mom = model->internals->symm_map_mom_fine;
    double* symm_beyond_UC = model->internals->symm_beyond_UC;
    double* kmesh = model->internals->kfmesh;
    index_t* symm_map_orb = model->internals->symm_map_orb;
    index_t* symm_orb_off = model->internals->symm_orb_off;
    index_t* symm_orb_len = model->internals->symm_orb_len;
    complex128_t* symm_pref = model->internals->symm_pref;

    #pragma omp parallel for collapse(5)
    for (index_t k=0; k<nk; ++k)
    for (index_t oo1=0; oo1<n_orb; ++oo1)
    for (index_t ss1=0; ss1<n_spin; ++ss1)
    for (index_t oo2=0; oo2<n_orb; ++oo2)
    for (index_t ss2=0; ss2<n_spin; ++ss2) {
        index_t o1 = oo1 + n_orb * ss1;
        index_t o2 = oo2 + n_orb * ss2;
        for (index_t s=0; s<n_sym; ++s) {
            index_t osym1 = IDX2( o1, s, n_sym ),
                    osym2 = IDX2( o2, s, n_sym ),
                    ksym = symm_map_mom[IDX2(k,s,n_sym)];
            for (index_t i=0; i<symm_orb_len[osym1]; ++i)
            for (index_t j=0; j<symm_orb_len[osym2]; ++j) {
                index_t xo1 = symm_orb_off[osym1] + i,
                        xo2 = symm_orb_off[osym2] + j;
                double kdotR11 = dot(symm_beyond_UC+3*IDX2(oo1,s,n_sym), kmesh+3*ksym);
                double kdotR22 = dot(symm_beyond_UC+3*IDX2(oo2,s,n_sym), kmesh+3*ksym);

                buf[IDX3(k,o1,o2,n_os,n_os)] += aux[IDX3(ksym,symm_map_orb[xo1],symm_map_orb[xo2], n_os,n_os)] *
                                                (symm_pref[xo1] * (cos(kdotR11) - I * sin(kdotR11))) *
                                            conj(symm_pref[xo2] * (cos(kdotR22) - I * sin(kdotR22))) * norm;
            }
        }
    }
    double max_diff = 0.;
    #pragma omp parallel for reduction(max:max_diff)
    for (index_t idx=0; idx<nk*n_os*n_os; ++idx) {
        max_diff = MAX(max_diff, cabs(buf[idx]-aux[idx]));
    }

    free_aux();
    return max_diff;
}

double diverge_symmetrize_mom_coarse( diverge_model_t* model, double* buf, index_t sub, double* aux ) {
    if (model->n_sym == 0 || model->orb_symmetries == NULL) return -1.0;

    diverge_generate_symm_maps(model);
    index_t nk = kdim( model->nk );

    alloc_aux( sizeof(double)*nk*sub );

    index_t n_sym = model->n_sym;
    double norm = 1./(double) n_sym;

    index_t* symm_map_mom = model->internals->symm_map_mom_crs;
    double max_diff = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:max_diff)
    for( index_t k = 0; k < nk; ++k )
    for( index_t o = 0; o < sub; ++o ) {
        for( index_t s = 0; s < n_sym; ++s )
            buf[k*sub+o] += aux[symm_map_mom[s + n_sym * k]*sub + o];
        buf[k*sub+o] *= norm;
        max_diff = MAX(max_diff, fabs(buf[k*sub+o] - aux[k*sub+o]));
    }
    free_aux();
    return max_diff;
}

double diverge_symmetrize_mom_fine( diverge_model_t* model, double* buf, index_t sub, double* aux ) {
    if (model->n_sym == 0 || model->orb_symmetries == NULL) return -1.0;

    diverge_generate_symm_maps(model);
    index_t nk = kdimtot( model->nk, model->nkf );

    alloc_aux( sizeof(double)*nk*sub );

    index_t n_sym = model->n_sym;
    double norm = 1./(double) n_sym;

    index_t* symm_map_mom = model->internals->symm_map_mom_fine;
    double max_diff = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:max_diff)
    for( index_t k = 0; k < nk; ++k )
    for( index_t o = 0; o < sub; ++o ) {
        for( index_t s = 0; s < n_sym; ++s )
            buf[k*sub+o] += aux[symm_map_mom[s + n_sym * k]*sub + o];
        buf[k*sub+o] *= norm;
        max_diff = MAX(max_diff, fabs(buf[k*sub+o] - aux[k*sub+o]));
    }
    free_aux();
    return max_diff;
}

