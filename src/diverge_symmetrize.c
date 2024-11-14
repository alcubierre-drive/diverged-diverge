/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model.h"
#include "diverge_symmetrize.h"
#include "diverge_internals_struct.h"
#include "misc/mpi_functions.h"
#include "diverge_model_internals.h"
#include <fftw3.h>

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

    #pragma omp parallel for collapse(5) num_threads(diverge_omp_num_threads())
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
    #pragma omp parallel for reduction(max:max_diff) num_threads(diverge_omp_num_threads())
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

    #pragma omp parallel for collapse(5) num_threads(diverge_omp_num_threads())
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
    #pragma omp parallel for reduction(max:max_diff) num_threads(diverge_omp_num_threads())
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

    #pragma omp parallel for collapse(5) num_threads(diverge_omp_num_threads())
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
    #pragma omp parallel for reduction(max:max_diff) num_threads(diverge_omp_num_threads())
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
    #pragma omp parallel for collapse(2) reduction(max:max_diff) num_threads(diverge_omp_num_threads())
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
    #pragma omp parallel for collapse(2) reduction(max:max_diff) num_threads(diverge_omp_num_threads())
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

rs_hopping_t* diverge_symmetrize_hoppings( diverge_model_t* model,
        rs_hopping_t* hop_, index_t* n_hop_, double hoplim ) {
    // some preparations
    if (model->n_sym == 0 || model->orb_symmetries == NULL) {
        mpi_err_printf( "need symmetries in order to symmetrize hoppings\n" );
        return hop_;
    }
    int delete_common_internals = 0;
    index_t nk_old[3] = {0};
    index_t nkf_old[3] = {0};
    if (!model->internals->has_common_internals) {
        mpi_log_printf( "common internals not set, so we generate them…\n" );
        if (!hop_) {
            index_t Rmax[3] = {0,0,0};
            for (rs_hopping_t* h = model->hop; h != model->hop + model->n_hop; ++h)
                for (int d=0; d<3; ++d) Rmax[d] = MAX(labs(h->R[d]), Rmax[d]);
            memcpy( nk_old, model->nk, sizeof(nk_old) );
            memcpy( nkf_old, model->nkf, sizeof(nkf_old) );
            // this one is quite unstable; we shouldn't rely on it.
            for (int d=0; d<3; ++d) {
                model->nk[d] = 2*Rmax[d]+1;
                model->nkf[d] = 1;
            }
            diverge_model_internals_common(model);
            delete_common_internals = 1;
        } else {
            mpi_err_printf( "must operate on internal hoppings (i.e. hop == NULL)\n" );
            return hop_;
        }
    }
    diverge_generate_symm_maps(model);

    rs_hopping_t* hop = model->hop;
    index_t* n_hop = &model->n_hop;
    int swap_model_hoppings = 1;
    if (hop_ && n_hop_) {
        hop = hop_;
        n_hop = n_hop_;
        swap_model_hoppings = 0;
    }
    mpi_vrb_printf( "initial n_hop = %li\n", *n_hop );
    index_t nr[3] = {0};
    index_t nk_tot = 1;
    for (int d=0; d<3; ++d) {
        nr[d] = model->nk[d] * model->nkf[d];
        nk_tot *= nr[d];
    }
    const index_t n_orb = model->n_orb;
    const index_t n_spin = model->n_spin;
    const index_t n_kpts = kdimtot(model->nk, model->nkf);

    index_t buf_sz = sizeof(complex128_t)*n_kpts*POW2(n_orb*n_spin);
    fftw_complex* fftw_buf = fftw_malloc(buf_sz);
    complex128_t* fftw_cbuf = (complex128_t*)fftw_buf;
    index_t stride_fft = POW2(n_orb*n_spin);
    fftw_iodim64 dims[3];
    dims[0].n = nr[0], dims[1].n = nr[1], dims[2].n = nr[2];
    dims[0].is = nr[2]*nr[1]*stride_fft, dims[1].is = nr[2]*stride_fft, dims[2].is = stride_fft;
    dims[0].os = nr[2]*nr[1]*stride_fft, dims[1].os = nr[2]*stride_fft, dims[2].os = stride_fft;
    fftw_iodim64 howmany;
    howmany.n = stride_fft, howmany.is = 1, howmany.os = 1;
    fftw_plan p_forward = fftw_plan_guru64_dft(3, dims, 1, &howmany, fftw_buf, fftw_buf, -1, FFTW_ESTIMATE);
    fftw_plan p_backward = fftw_plan_guru64_dft(3, dims, 1, &howmany, fftw_buf, fftw_buf, 1, FFTW_ESTIMATE);

    // fill hoppings
    memset((void*)fftw_cbuf, 0, buf_sz);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t idx = 0; idx<*n_hop; ++idx) {
        index_t o1 = hop[idx].o1,
                s1 = hop[idx].s1,
                o2 = hop[idx].o2,
                s2 = hop[idx].s2;
        index_t R[3]; for (int i=0; i<3; ++i) R[i] = hop[idx].R[i];

        index_t x_id = (nr[0] + R[0]) % nr[0],
                y_id = (nr[1] + R[1]) % nr[1],
                z_id = (nr[2] + R[2]) % nr[2];
        index_t Ridx = IDX3(x_id, y_id, z_id, nr[1], nr[2]);

        if (cabs(hop[idx].t) > hoplim) {
            double* out = (double*)(fftw_cbuf + IDX5(Ridx, s1, o1, s2, o2, n_spin, n_orb, n_spin, n_orb));
            double* in = (double*)(&hop[idx].t);
            #pragma omp atomic
            out[0] += in[0];
            #pragma omp atomic
            out[1] += in[1];
        }
    }

    // FFT, symmetrize, IFFT
    fftw_execute( p_forward );
    diverge_symmetrize_2pt_fine( model, fftw_cbuf, NULL );
    fftw_execute( p_backward );
    for (index_t i=0; i<nk_tot*POW2(n_orb*n_spin); ++i) fftw_cbuf[i] *= 1./(double)nk_tot;
    fftw_destroy_plan( p_forward );
    fftw_destroy_plan( p_backward );

    // find the new number of hoppings
    *n_hop = 0;
    for (index_t Rx=-nr[0]/2; Rx<=nr[0]/2; Rx++)
    for (index_t Ry=-nr[1]/2; Ry<=nr[1]/2; Ry++)
    for (index_t Rz=-nr[2]/2; Rz<=nr[2]/2; Rz++)
    for (index_t o1=0; o1<n_orb; ++o1)
    for (index_t o2=0; o2<n_orb; ++o2)
    for (index_t s1=0; s1<n_spin; ++s1)
    for (index_t s2=0; s2<n_spin; ++s2) {
        index_t Rx_ = (Rx + nr[0]) % nr[0],
                Ry_ = (Ry + nr[1]) % nr[1],
                Rz_ = (Rz + nr[2]) % nr[2];
        complex128_t elem = fftw_cbuf[IDX7(Rx_,Ry_,Rz_,s1,o1,s2,o2, nr[1],nr[2],n_spin,n_orb,n_spin,n_orb)];
        if (cabs(elem) > hoplim) (*n_hop)++;
    }

    hop = realloc(hop, sizeof(rs_hopping_t) * (*n_hop));
    if (swap_model_hoppings) model->hop = hop;
    *n_hop = 0;
    for (index_t Rx=-nr[0]/2; Rx<=nr[0]/2; Rx++)
    for (index_t Ry=-nr[1]/2; Ry<=nr[1]/2; Ry++)
    for (index_t Rz=-nr[2]/2; Rz<=nr[2]/2; Rz++)
    for (index_t o1=0; o1<n_orb; ++o1)
    for (index_t o2=0; o2<n_orb; ++o2)
    for (index_t s1=0; s1<n_spin; ++s1)
    for (index_t s2=0; s2<n_spin; ++s2) {
        index_t Rx_ = (Rx + nr[0]) % nr[0],
                Ry_ = (Ry + nr[1]) % nr[1],
                Rz_ = (Rz + nr[2]) % nr[2];
        complex128_t elem = fftw_cbuf[IDX7(Rx_,Ry_,Rz_,s1,o1,s2,o2, nr[1],nr[2],n_spin,n_orb,n_spin,n_orb)];
        if (cabs(elem) > hoplim) hop[(*n_hop)++] = (rs_hopping_t){ {Rx, Ry, Rz}, o1,o2, s1,s2, elem };
    }
    mpi_vrb_printf( "final n_hop = %li\n", *n_hop );

    fftw_free( fftw_buf );

    if (delete_common_internals) {
        diverge_model_internals_reset(model);
        memcpy( model->nk, nk_old, sizeof(nk_old) );
        memcpy( model->nkf, nkf_old, sizeof(nkf_old) );
    }
    return hop;
}
