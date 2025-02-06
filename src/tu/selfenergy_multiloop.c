#include "selfenergy_multiloop.h"
#include <fftw3.h>
#include "../misc/batched_gemms.h"
#include "../misc/batched_zgemd.h"
#include "../misc/batched_eigen.h"
#include "../misc/gemm.h"
#include "../diverge_patching_helpers.h" // atomic_add
#include "../diverge_model_internals.h"
#include "selfenergy.h"

#ifndef DIVERGE_MULTILOOP_ZERO_ENERGY
#define DIVERGE_MULTILOOP_ZERO_ENERGY 1.e-3
#endif

static inline double G_Lamdba_integrate_kernel( const double Lambda, const double eps ) {
    double result = 0.5;
    if (fabs(eps) > DIVERGE_MULTILOOP_ZERO_ENERGY) {
        result = atan(Lambda/eps)/M_PI;
        result += (eps < 0.0) ? 1.0 : 0.0;
    }
    return result;
}

void tu_multiloop_t_bar_GF( complex128_t* G_int, const double Lambda,
        void* workspace/*sizeof(double)*nk*nb*/, const double* E_eff,
        const complex128_t* U_eff, const index_t nk, const index_t nb ) {
    double* aux = workspace;
    // prepare the \int \dd k0 G_b(k0) buffer
    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
    for (index_t b=0; b<nb; ++b)
        aux[IDX2(k,b,nb)] = G_Lamdba_integrate_kernel( Lambda, E_eff[IDX2(k,b,nb)] );
    batched_zgemd_r( G_int, U_eff, aux, 1.0, 0.0, nb, nk );
}

void tu_multiloop_t_GF( complex128_t* G_int, const double Lambda,
        void* workspace/*sizeof(complex128_t)*nk*POW2(nb)*/, const double* E_eff,
        const complex128_t* U_eff, const complex128_t* d_Sigma_t_bar,
        const index_t nk, const index_t nb ) {
    complex128_t* aux = workspace;
    const index_t nb2 = POW2(nb);

    // aux should contain the band space d_Sigma_t_bar
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k) transpose_outofplace( aux+k*nb2, d_Sigma_t_bar+k*nb2, nb );
    batched_gemm_N_ll_num( U_eff, aux, aux, nb, 1.0, 0.0, nk, 1, 0 );
    batched_gemm_N_ll_num( aux, U_eff, aux, nb, 1.0, 0.0, nk, 0, 0 );

    // we then multiply it by \int \dd k0 G_b(k0)G_b'(k0)
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
    for (index_t b=0; b<nb; ++b)
    for (index_t c=0; c<nb; ++c) {
        const double eps_b = E_eff[IDX2(k,b,nb)],
                     eps_c = E_eff[IDX2(k,c,nb)];
        if (fabs(eps_b-eps_c) < DIVERGE_MULTILOOP_ZERO_ENERGY) {
            const double eps = 0.5 * (eps_b + eps_c);
            aux[IDX3(k,b,c,nb,nb)] *= -Lambda/M_PI/(POW2(Lambda)+POW2(eps));
        } else {
            aux[IDX3(k,b,c,nb,nb)] *= (G_Lamdba_integrate_kernel(Lambda,eps_b)-G_Lamdba_integrate_kernel(Lambda,eps_c)) /
                                      (eps_b - eps_c);
        }
    }

    // and transform this (b,b') matrix back to (o,p) space
    batched_gemm_N_ll_num( aux, U_eff, aux, nb, 1.0, 0.0, nk, 0, 1 );
    batched_gemm_N_ll_num( U_eff, aux, aux, nb, 1.0, 0.0, nk, 0, 0 );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k) transpose_outofplace( G_int+k*nb2, aux+k*nb2, nb );
}

struct tu_multiloop_correction_t {
    index_t nk, nb;

    const complex128_t* S;
    const complex128_t* mU;
    const double* mE;
    complex128_t* U;
    double* E;
    void* workspace;

    complex128_t* G_int;
    complex128_t* mFRG_Sigma_t;
    complex128_t* mFRG_Sigma_t_bar;

    greensfunc_coarse_t* coarsify;
};

tu_multiloop_correction_t* tu_multiloop_correction_init( diverge_model_t* model, const complex128_t* Sigma ) {
    tu_multiloop_correction_t* m = calloc(1, sizeof*m);
    m->nk = kdimtot(model->nk, model->nkf);
    m->nb = model->n_orb * model->n_spin;
    m->workspace = calloc(m->nk * POW2(m->nb), sizeof(complex128_t));
    m->S = Sigma;
    m->mU = diverge_model_internals_get_U(model);
    m->mE = diverge_model_internals_get_E(model);
    m->U = calloc(m->nk * POW2(m->nb), sizeof*m->U);
    m->E = calloc(m->nk * m->nb, sizeof*m->E);

    m->G_int = calloc(m->nk * POW2(m->nb), sizeof*m->G_int);
    m->mFRG_Sigma_t = calloc(m->nk * POW2(m->nb), sizeof*m->mFRG_Sigma_t);
    m->mFRG_Sigma_t_bar = calloc(m->nk * POW2(m->nb), sizeof*m->mFRG_Sigma_t_bar);

    m->coarsify = greensfunc_coarse_init( model->nk, model->nkf, m->nb, model->internals->dim );
    return m;
}

void tu_multiloop_correction_free( tu_multiloop_correction_t* m ) {
    free( m->workspace );
    free( m->U );
    free( m->E );
    free( m->G_int );
    free( m->mFRG_Sigma_t );
    free( m->mFRG_Sigma_t_bar );
    greensfunc_coarse_free( m->coarsify );
    free( m );
}

void tu_multiloop_correction_calc( tu_multiloop_correction_t* m, double Lambda,
        const complex128_t* Pch, const complex128_t* Cch, const complex128_t* Dch,
        const complex128_t* dPch, const complex128_t* dDch, tu_selfenergy_t* self,
        complex128_t* out ) {
    // copy self-energy to buffer, add Hamiltonian, and diagonalize
    memcpy( m->U, m->S, sizeof*m->U * m->nk * POW2(m->nb) );
    batched_zgemd_r( m->U, m->mU, m->mE, 1.0, 1.0, m->nb, m->nk );
    batched_eigen_shut_up(1);
    batched_eigen_r( NULL, -2, m->U, m->E, m->nb, m->nk );
    batched_eigen_shut_up(0);

    // now to the first correction (Sigma_t_bar)
    tu_multiloop_t_bar_GF( m->G_int, Lambda, m->workspace, m->E, m->U, m->nk, m->nb );
    tu_selfenergy_flow( self, m->mFRG_Sigma_t_bar, dPch, NULL, dDch,
            greensfunc_coarse_calc( m->coarsify, m->G_int ), 1.0 );

    // and here the second one (Sigma_t)
    tu_multiloop_t_GF( m->G_int, Lambda, m->workspace, m->E, m->U,
            m->mFRG_Sigma_t_bar, m->nk, m->nb );
    tu_selfenergy_flow( self, m->mFRG_Sigma_t, Pch, Cch, Dch,
            greensfunc_coarse_calc( m->coarsify, m->G_int ), 1.0 );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<m->nk*POW2(m->nb); ++i)
        out[i] += m->mFRG_Sigma_t[i] + m->mFRG_Sigma_t_bar[i];
}

struct greensfunc_coarse_t {
    fftw_plan fine_in, coarse_out;
    fftw_complex *buf, *cbuf;
    index_t nk[3], nkf[3], nkkf[3], nb, nb2, Nkkf, Nk;
    int dim, fftw_plans_set;
};

greensfunc_coarse_t* greensfunc_coarse_init( const index_t* nk, const index_t* nkf, index_t nb, int dim ) {
    greensfunc_coarse_t* g = calloc(1, sizeof*g);
    memcpy( g->nk, nk, sizeof(g->nk) );
    memcpy( g->nkf, nkf, sizeof(g->nkf) );
    for (int d=0; d<3; ++d) g->nkkf[d] = g->nk[d] * g->nkf[d];
    g->nb = nb;
    g->nb2 = POW2(g->nb);
    g->dim = dim;
    g->Nkkf = kdim(g->nkkf);
    g->Nk = kdim(g->nk);
    g->cbuf = fftw_alloc_complex(g->Nk * g->nb2);
    if (g->Nkkf != g->Nk) {
        g->buf = fftw_alloc_complex(g->Nkkf * g->nb2);
        const fftw_iodim64 dims_fin[3] = {
            { .n = g->nkkf[0], .is = g->nkkf[1]*g->nkkf[2]*g->nb2, .os = g->nkkf[1]*g->nkkf[2]*g->nb2, },
            { .n = g->nkkf[1], .is =            g->nkkf[2]*g->nb2, .os =            g->nkkf[2]*g->nb2, },
            { .n = g->nkkf[2], .is =                       g->nb2, .os =                       g->nb2, },
        };
        const fftw_iodim64 howmany = { .n = g->nb2, .is = 1, .os = 1 };
        g->fine_in = fftw_plan_guru64_dft(3, dims_fin, 1, &howmany, g->buf, g->buf, -1, FFTW_MEASURE);
        const fftw_iodim64 dims_crs[3] = {
            { .n = g->nk[0], .is = g->nk[1]*g->nk[2]*g->nb2, .os = g->nk[1]*g->nk[2]*g->nb2, },
            { .n = g->nk[1], .is =          g->nk[2]*g->nb2, .os =          g->nk[2]*g->nb2, },
            { .n = g->nk[2], .is =                   g->nb2, .os =                   g->nb2, },
        };
        g->coarse_out = fftw_plan_guru64_dft(3, dims_crs, 1, &howmany, g->cbuf, g->cbuf, 1, FFTW_MEASURE);
        g->fftw_plans_set = 1;
    }
    return g;
}

void greensfunc_coarse_free( greensfunc_coarse_t* g ) {
    fftw_free( g->cbuf );
    if (g->Nkkf != g->Nk) {
        fftw_destroy_plan( g->fine_in );
        fftw_destroy_plan( g->coarse_out );
        fftw_free( g->buf );
    }
    free( g );
}

complex128_t* greensfunc_coarse_calc( greensfunc_coarse_t* g, const gf_complex_t* G_fine ) {
    complex128_t* cbuf = (complex128_t*)g->cbuf;
    complex128_t* fbuf = (complex128_t*)g->buf;
    if (g->Nkkf == g->Nk) {
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t k=0; k<g->Nk; ++k)
        for (index_t o=0; o<g->nb; ++o)
        for (index_t b=0; b<g->nb; ++b)
            cbuf[IDX3(o,b,k,g->nb,g->Nk)] = G_fine[IDX3(k,b,o,g->nb,g->nb)];
        return cbuf;
    }

    // copy fine GF to fine buffer and normalize
    for (index_t i=0; i<g->Nkkf*g->nb2; ++i) fbuf[i] = G_fine[i] / (double)g->Nkkf;

    // real-space GF on the fine mesh
    fftw_execute_dft( g->fine_in, g->buf, g->buf );

    // zero-un-padding, save the result in cbuf
    ZERO_OUT( cbuf, g->Nk*g->nb2 );
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t kc1=-g->nk[0]/2; kc1<=g->nk[0]/2; ++kc1)
    for (index_t kc2=-g->nk[1]/2; kc2<=g->nk[1]/2; ++kc2)
    for (index_t kc3=-g->nk[2]/2; kc3<=g->nk[2]/2; ++kc3) {
        index_t kc[] = {kc1, kc2, kc3};
        int boundary[3] = {0};
        for (int d=0; d<g->dim; ++d) boundary[d] = (labs(kc[d])==g->nk[d]/2) && (g->nk[d]%2);
        double factor = 1.;
        for (int d=0; d<g->dim; ++d) factor *= boundary[d] ? 0.5 : 1.0;
        const index_t kc1i = (kc1 + g->nk[0]) % g->nk[0],
                      kc2i = (kc2 + g->nk[1]) % g->nk[1],
                      kc3i = (kc3 + g->nk[2]) % g->nk[2];
        const index_t kcidx = IDX3(kc1i,kc2i,kc3i,g->nk[1],g->nk[2]);

        const index_t kf1i = (kc1 + g->nkkf[0]) % g->nkkf[0],
                      kf2i = (kc2 + g->nkkf[1]) % g->nkkf[1],
                      kf3i = (kc3 + g->nkkf[2]) % g->nkkf[2];
        const index_t kfidx = IDX3(kf1i,kf2i,kf3i,g->nkkf[1],g->nkkf[2]);
        for (index_t bb=0; bb<g->nb2; ++bb) {
            complex128_t cbuf_add = factor * fbuf[IDX2(kfidx,bb,g->nb2)];
            atomic_add( cbuf + IDX2(kcidx,bb,g->nb2), &cbuf_add );
        }
    }

    // FFT back to momentum space on the coarse mesh
    fftw_execute_dft( g->coarse_out, cbuf, cbuf );

    // and transpose the result into fbuf (which always has enough space!)
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<g->Nk; ++k)
    for (index_t o=0; o<g->nb; ++o)
    for (index_t b=0; b<g->nb; ++b)
        fbuf[IDX3(o,b,k,g->nb,g->Nk)] = cbuf[IDX3(k,b,o,g->nb,g->nb)];
    return fbuf;
}

