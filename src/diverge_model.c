/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model.h"
#include "diverge_internals_struct.h"
#include "misc/init_interaction_channels.h"
#include "misc/mpi_functions.h"
#include "diverge_patching.h"
#include "diverge_patching_helpers.h" // atomic_add
#include "diverge_model_internals.h"
#include "misc/dependency_tree.h"
#include "misc/qnan.h"
#include "misc/init_internal_libs.h"
#include "misc/mergesort.h"
#include <fftw3.h>
#include <assert.h>

diverge_model_t* diverge_model_init(void) {
    diverge_model_t* result = calloc(1,sizeof*result);
    result->hfill = &diverge_hamilton_generator_default;
    result->gfill = &diverge_greensfunc_generator_default;
    result->gproj = NULL;
    result->vfill = &diverge_channel_vertex_generator_default;
    result->ffill = NULL;
    result->n_vert_chan[0] = -1;
    result->n_spin = 1;
    result->SU2 = 1;
    result->internals = calloc(1,sizeof*result->internals);
    result->internals->has_common_internals = 0;
    result->internals->tu_symcheck_mpi = environ_tu_symcheck_mpi();
    result->internals->symcheck_mpi = environ_symcheck_mpi();
    dependency_tree_insert_model(result);
    return result;
}

void diverge_model_free_set_options(diverge_model_t* model, int ignore_hop,
        int ignore_vert, int ignore_orb_symmetries) {
    model->internals->ignore_hop_free = ignore_hop;
    model->internals->ignore_vert_free = ignore_vert;
    model->internals->ignore_orb_symmetries_free = ignore_orb_symmetries;
}

void diverge_model_free(diverge_model_t* model) {
    if (model->internals->tu_phonon_model_dont_free) {
        mpi_err_printf( "not allowed to free model %p: it's a phonon model.\n", model );
        return;
    }
    dependency_tree_remove_model(model);

    if (model->patching) {
        diverge_patching_free(model->patching);
        model->patching = NULL;
    }
    free(model->tu_ff);

    if (model->data_destructor) (*model->data_destructor)(model->data);
    else if (model->nbytes_data > 0) free(model->data);
    else if (model->data) mpi_log_printf( "data ptr set but not free'd (nbytes_data == 0)\n" );

    if (!model->internals->ignore_orb_symmetries_free) free(model->orb_symmetries);
    if (!model->internals->ignore_hop_free) free(model->hop);
    if (!model->internals->ignore_vert_free) free(model->vert);

    if (model->internals) diverge_model_internals_reset(model);
    free(model->internals);
    free(model);
}

static int diverge_hamilton_generator_add_planning_flag = FFTW_ESTIMATE;
void diverge_hamilton_generator_set_fftw_options( int flags ) {
    diverge_hamilton_generator_add_planning_flag = flags;
}

void diverge_hamilton_generator_add(const diverge_model_t* model, complex128_t* buf) {
    const index_t n_hop = model->n_hop;
    const rs_hopping_t* hop = model->hop;
    const index_t n_kx = model->nk[0]*model->nkf[0];
    const index_t n_ky = model->nk[1]*model->nkf[1];
    const index_t n_kz = model->nk[2]*model->nkf[2];
    const index_t n_kpts = n_kx*n_ky*n_kz;
    const index_t n_spin = model->n_spin;
    const index_t n_orb = model->n_orb;
    const bool SU2 = model->SU2;

    index_t buf_sz = sizeof(complex128_t)*n_kpts*POW2(n_orb*n_spin);
    fftw_complex* fftw_buf = (fftw_complex*)fftw_malloc(buf_sz);
    complex128_t* fftw_cbuf = (complex128_t*)fftw_buf;

    index_t stride_fft = POW2(n_orb*n_spin);
    fftw_iodim64 dims[3] = {{.n = n_kx, .is = n_kz*n_ky*stride_fft, .os = n_kz*n_ky*stride_fft,},
                            {.n = n_ky, .is = n_kz*stride_fft,      .os = n_kz*stride_fft,},
                            {.n = n_kz, .is = stride_fft,           .os = stride_fft},};
    fftw_iodim64 howmany = { .n = stride_fft, .is = 1, .os = 1 };
    fftw_plan p = fftw_plan_guru64_dft(3, dims, 1, &howmany, fftw_buf, fftw_buf, -1,
            diverge_hamilton_generator_add_planning_flag);

    memset((void*)fftw_cbuf, 0, buf_sz);

    double invalid_hoppings_norm = 0;
    // fill fftw_cbuf as H[R]
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(+:invalid_hoppings_norm)
    for (index_t idx = 0; idx < n_hop; ++idx) {
        index_t o1 = hop[idx].o1,
                s1 = hop[idx].s1,
                o2 = hop[idx].o2,
                s2 = hop[idx].s2;
        index_t R[3]; for (int i=0; i<3; ++i) R[i] = hop[idx].R[i];

        const index_t x_id = (n_kx + R[0]) % n_kx,
                      y_id = (n_ky + R[1]) % n_ky,
                      z_id = (n_kz + R[2]) % n_kz;
        const index_t Ridx = IDX3(x_id, y_id, z_id, n_ky, n_kz);
        if (Ridx < 0) {
            invalid_hoppings_norm += cabs(hop[idx].t);
            continue;
        }

        const index_t sconfig[] = {s1,s2};
        const index_t sconfig_m1[] = {0,0, 1,1};
        const index_t* sc = (s1 == -1) ? sconfig_m1 : sconfig;
        const index_t nsc = (s1 == -1 && !SU2) ?  2 :       1;
        for (index_t isc=0; isc<nsc; ++isc) {
            s1 = sc[IDX2(isc,0,2)];
            s2 = sc[IDX2(isc,1,2)];
            atomic_add( fftw_cbuf + IDX5(Ridx, s1, o1, s2, o2, n_spin, n_orb, n_spin, n_orb), &hop[idx].t );
        }
    }

    if ((n_kpts*POW2(n_spin*n_orb) < n_hop && n_kpts > 1) || (invalid_hoppings_norm > 1e-9))
        mpi_wrn_printf("hamilton_generator: #hoppings > #kpts -> potential error %.5e\n", invalid_hoppings_norm);

    mpi_vrb_printf( "hamilton_generator fft…\n" );
    if(n_kpts > 1)
        fftw_execute(p);
    fftw_destroy_plan(p);

    mpi_vrb_printf( "hamilton_generator outputting elements…\n" );
    for (index_t i=0; i<n_kpts*POW2(n_orb*n_spin); ++i)
        buf[i] += fftw_cbuf[i];
    fftw_free(fftw_buf);
}

void diverge_hamilton_generator_default(const diverge_model_t* model, complex128_t* buf) {
    ZERO_OUT( buf, POW2(model->n_orb*model->n_spin)*kdimtot(model->nk, model->nkf) );
    diverge_hamilton_generator_add( model, buf );
}

greensfunc_op_t diverge_greensfunc_generator_default(const diverge_model_t* model, complex128_t Lambda, gf_complex_t* buf) {

    const index_t nkf = kdimtot(model->nk, model->nkf);
    const index_t nb = model->n_spin * model->n_orb;
    const complex128_t* U = (const complex128_t*) model->internals->U;
    const double* E = model->internals->E;

    for (int sign=0; sign<2; ++sign) {
        complex128_t L = sign ? Lambda : conj(Lambda);
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t k=0; k<nkf; ++k)
        for (index_t o=0; o<nb; ++o)
        for (index_t p=0; p<nb; ++p) {
            complex128_t tmp = 0.0;
            for (index_t b=0; b<nb; ++b)
                tmp += U[IDX3(k,b,o,nb,nb)] * conj(U[IDX3(k,b,p,nb,nb)]) / (L - E[IDX2(k,b,nb)]);
            buf[IDX4(sign, k, o, p, nkf, nb, nb)] = tmp;
        }
    }
    return greensfunc_op_cpu;
}

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( (x).chan - (y).chan )
MERGESORT_IMPL( vertsort, rs_vertex_t )

#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R( x, y, ctx ) ( \
    IDX3((x).R[0],(x).R[1],(x).R[2],((index_t*)(ctx))[1], ((index_t*)(ctx))[2]) - \
    IDX3((y).R[0],(y).R[1],(y).R[2],((index_t*)(ctx))[1], ((index_t*)(ctx))[2]) \
)
MERGESORT_IMPL_R( vertsort, rs_vertex_t )


static inline void sort_rs_vertex_elements(diverge_model_t* model) {
    if (model->n_vert_chan[0] != -1) return;
    vertsort_r( model->vert, model->n_vert, model->nk, diverge_omp_num_threads(), 0 );
    vertsort( model->vert, model->n_vert, diverge_omp_num_threads(), 0 );
    char chans[] = "CDP";
    rs_vertex_t* vptr = model->vert;
    const rs_vertex_t* vptr_end = model->vert + model->n_vert;
    for (int i=0; i<3; ++i) {
        model->n_vert_chan[i] = 0;
        while (vptr != vptr_end && vptr->chan == chans[i]) {
            vptr++;
            model->n_vert_chan[i]++;
        }
    }
}

int diverge_channel_vertex_generator_default(diverge_model_t* model, char channel, complex128_t* buf) {
    sort_rs_vertex_elements(model);

    index_t n_kx = model->nk[0],
            n_ky = model->nk[1],
            n_kz = model->nk[2],
            n_spin = model->n_spin,
            n_orb = model->n_orb;

    index_t stride_fft = POW2(n_orb*n_spin*n_spin);

    fftw_iodim64 dims[3];
    dims[0].n = n_kx, dims[1].n = n_ky, dims[2].n = n_kz;
    dims[0].is = n_kz*n_ky*stride_fft, dims[1].is = n_kz*stride_fft, dims[2].is = stride_fft;
    dims[0].os = n_kz*n_ky*stride_fft, dims[1].os = n_kz*stride_fft, dims[2].os = stride_fft;

    fftw_iodim64 howmany;
    howmany.n = stride_fft, howmany.is = 1, howmany.os = 1;

    index_t buf_sz = sizeof(complex128_t)*kdim(model->nk)*POW2(n_orb*n_spin*n_spin);
    fftw_complex* fftw_buf = (fftw_complex*)fftw_malloc(buf_sz);
    fftw_plan p = fftw_plan_guru64_dft(3, dims, 1, &howmany, fftw_buf, fftw_buf, -1, FFTW_ESTIMATE);

    memset(fftw_buf, 0, buf_sz);

    int filled = 0;
    if (channel == 'P') {
        filled = init_P_channel(model, (complex128_t*)fftw_buf);
    } else if (channel == 'C') {
        filled = init_C_channel(model, (complex128_t*)fftw_buf);
    } else if (channel == 'D') {
        filled = init_D_channel(model, (complex128_t*)fftw_buf);
    } else {
        mpi_wrn_printf("unknown channel supplied, using 'D'.\n");
        filled = init_D_channel(model, (complex128_t*)fftw_buf);
    }

    if (filled)
        fftw_execute(p);
    else
        memset((void*)buf, 0, buf_sz);
    fftw_destroy_plan(p);
    memcpy((void*)buf, fftw_buf, buf_sz);
    fftw_free(fftw_buf);

    return filled;
}

double diverge_model_ibz_path_skip_segment( void ) {
    uint64_t number = 0;
    memcpy( &number, "skip", MIN(sizeof(uint64_t),sizeof("skip")) );
    return qnan_gen( number );
}

