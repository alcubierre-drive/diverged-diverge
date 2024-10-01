/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "ham2hop.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "mpi_functions.h"
#include <fftw3.h>

rs_hopping_t* diverge_model_ham2hop( const diverge_model_t* model, double hoplim,
        index_t* n_hop_ ) {

    index_t nr[3] = {0};
    index_t nk_tot = 1;
    for (int d=0; d<3; ++d) {
        nr[d] = model->nk[d] * model->nkf[d];
        nk_tot *= nr[d];
    }
    const index_t n_orb = model->n_orb;
    const index_t n_spin = model->n_spin;
    const index_t n_kpts = kdimtot(model->nk, model->nkf);

    const complex128_t* H0 = model->internals->ham;

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
    fftw_plan p_backward = fftw_plan_guru64_dft(3, dims, 1, &howmany, fftw_buf, fftw_buf, 1, FFTW_ESTIMATE);

    memcpy( fftw_buf, H0, buf_sz );
    fftw_execute( p_backward );
    for (index_t i=0; i<nk_tot*POW2(n_orb*n_spin); ++i) fftw_cbuf[i] *= 1./(double)nk_tot;

    fftw_destroy_plan( p_backward );

    // find the new number of hoppings
    index_t n_hop = 0;
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
        if (cabs(elem) > hoplim) n_hop++;
    }

    rs_hopping_t* res = calloc(n_hop, sizeof(rs_hopping_t));
    n_hop = 0;
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
        if (cabs(elem) > hoplim) res[n_hop++] = (rs_hopping_t){ {Rx, Ry, Rz}, o1,o2, s1,s2, elem };
    }

    fftw_free( fftw_buf );

    *n_hop_ = n_hop;
    return res;
}
