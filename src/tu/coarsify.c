#include "coarsify.h"
#include "../diverge_patching_helpers.h" // atomic_add
#include <fftw3.h>

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

