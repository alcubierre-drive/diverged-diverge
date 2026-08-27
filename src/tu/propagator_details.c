#include "propagator_details.h"
#include "../misc/shared_mem.h"
#include "../misc/invert.h"

void reconstruct_from_reduced( const tu_prop_t* t, complex128_t* loop ) {
    index_t n_unique = t->tu_data->unique_mi_l_sz;

#ifndef USE_DYNAMIC_RECONSTRUCT_SCHEDULE
    #pragma omp parallel for collapse(4) num_threads(diverge_omp_num_threads())
#else // USE_DYNAMIC_RECONSTRUCT_SCHEDULE
    #pragma omp parallel for collapse(4) schedule(dynamic,1) num_threads(diverge_omp_num_threads())
#endif // USE_DYNAMIC_RECONSTRUCT_SCHEDULE
    for (index_t q=0; q<t->my_nk; ++q)
    for (index_t sr=0; sr<POW2(t->n_spin); ++sr)
    for (index_t sl=0; sl<POW2(t->n_spin); ++sl)
    for (index_t m=0; m<n_unique; ++m) {
        index_t gmo1 = t->tu_data->unique_mi_l[m];
        index_t gmo3 = t->tu_data->unique_mi_r[m];
        complex128_t from = loop[gmo1+t->n_orbff*(sl +POW2(t->n_spin)*(gmo3+t->n_orbff*
                                (sr + POW2(t->n_spin)*q)))];
        for (index_t l=1; l<t->tu_data->lmap_len[m]; ++l) {
            index_t mo1 = t->tu_data->lmap_l[l+t->tu_data->lmap_off[m]];
            index_t mo3 = t->tu_data->lmap_r[l+t->tu_data->lmap_off[m]];
            loop[mo1+t->n_orbff*(sl +POW2(t->n_spin)*(mo3+t->n_orbff*
                                (sr + POW2(t->n_spin)*q)))] = from;
        }
    }
}

static const char* tu_fft_algorithm_name( tu_fft_algorithm_t algo ) {
    switch (algo) {
        case tu_fft_algorithm_greedy: return "greedy";
        case tu_fft_algorithm_split: return "split";
        case tu_fft_algorithm_mpi: return "mpi";
        case tu_fft_algorithm_mpi_new: return "mpi_new";
    }
    return "invalid";
}

static fftw_plan plan_transpose( int rows, int cols, complex128_t* in, complex128_t* out ) {
    fftw_iodim howmany_dims[2];
    howmany_dims[0].n  = rows;
    howmany_dims[0].is = cols;
    howmany_dims[0].os = 1;
    howmany_dims[1].n  = cols;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = rows;
    return fftw_plan_guru_dft( 0, NULL, 2, howmany_dims, (fftw_complex*)in, (fftw_complex*)out, 0, FFTW_MEASURE);
}

void cpu_fft_algorithm_init( tu_prop_t* t, tu_fft_algorithm_t greens,
        tu_fft_algorithm_t simple, tu_fft_algorithm_t red ) {

    if (t->nktot <= 1) return;

    mpi_vrb_printf( "TU loop algorithm choice: GF: %s, simple: %s, reduced: %s\n",
            tu_fft_algorithm_name( greens ), tu_fft_algorithm_name( simple ), tu_fft_algorithm_name( red ) );

    int fftwnthr = diverge_fftw_num_threads();
    const index_t* vnk = t->model->nk;
    const index_t* vnkf = t->model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]), (int)(vnk[2]*vnkf[2])};
    int dims_coarse[] = {(int)vnk[0], (int)vnk[1], (int)vnk[2]};

    mpi_vrb_printf( "TU_prop CPU init greens…\n" );
    switch (greens) {
        case tu_fft_algorithm_mpi_new:
            if (!t->mpi_gf)
                t->mpi_gf = tu_mpi_gf_init( vnk, vnkf, t->model->n_orb*t->model->n_spin, t->gf_buf, t->model );
            break;
        case tu_fft_algorithm_mpi:
            if (!t->fft_nons2_backward_mpi)
                t->fft_nons2_backward_mpi = fftw_mpi_plan_init( dims, POW2(t->n_spin*t->n_orb),
                    t->GF_fft_plus, t->GF_fft_plus, FFTW_BACKWARD, (complex128_t*)t->gf_buf );
            break;
        case tu_fft_algorithm_split:
        case tu_fft_algorithm_greedy:
        default:
            if (!t->fft_nons2_backward)
                t->fft_nons2_backward = fftw_plan_many_dft( 3, dims, POW2(t->n_spin*t->n_orb),
                    (fftw_complex*)t->GF_fft_plus, dims, 1, t->nktot,
                    (fftw_complex*)t->GF_fft_plus, dims, 1, t->nktot,
                    FFTW_BACKWARD, FFTW_MEASURE );
            break;
    }

    mpi_vrb_printf( "TU_prop CPU init loop…\n" );
    tu_fft_algorithm_t simple_red_algos[] = {simple, red};
    for (int i=0; i<2; ++i) switch (simple_red_algos[i]) {
        case tu_fft_algorithm_mpi_new:
            if (!t->mpi_loop) t->mpi_loop = tu_mpi_loop_init( t->model->nk, t->model->nkf,
                    t->chunksize_max, t->c2cmap, t->my_nk, t->n_spin, t->n_orb, t->n_orbff,
                    t->tu_data->unique_mi_l, t->tu_data->unique_mi_l_sz,
                    t->tu_data->unique_mi_r, t->mi_to_ofrom, t->mi_to_oto, t->mi_to_R,
                    &t->GF_fft_plus, &t->GF_fft_conj_plus, t, t->model );
            break;
        case tu_fft_algorithm_mpi:
            if (!t->buffer_ggm_many)
                t->buffer_ggm_many = (complex128_t*)fftw_alloc_complex(t->nktot * t->chunksize_max);
            if (!t->buffer_ggp_many)
                t->buffer_ggp_many = (complex128_t*)fftw_alloc_complex(t->nktot * t->chunksize_max);
            if (!t->fft_many_backward_mpi)
                t->fft_many_backward_mpi = fftw_mpi_plan_init( dims, t->chunksize_max,
                    t->buffer_ggm_many, t->buffer_ggp_many, FFTW_BACKWARD, NULL );
            if (!t->fft_many_forward_mpi)
                t->fft_many_forward_mpi = fftw_mpi_plan_init( dims, t->chunksize_max,
                    t->buffer_ggm_many, t->buffer_ggp_many, FFTW_FORWARD, NULL );
            if (!t->fft_T_to)
                t->fft_T_to = plan_transpose( t->chunksize_max, t->nktot, t->buffer_ggm_many, t->buffer_ggp_many );
            if (!t->fft_T_from)
                t->fft_T_from = plan_transpose( t->nktot, t->chunksize_max,t->buffer_ggm_many,t->buffer_ggp_many );
            break;
        case tu_fft_algorithm_split:
            if (!t->buffer_ggm_many)
                t->buffer_ggm_many = (complex128_t*)fftw_alloc_complex(t->nktot * t->chunksize_max);
            if (!t->buffer_ggp_many)
                t->buffer_ggp_many = (complex128_t*)fftw_alloc_complex(t->nktot * t->chunksize_max);
            // coarse-mesh-only batched transforms
            if (!t->fft_many_backward_coarse)
                t->fft_many_backward_coarse = fftw_plan_many_dft( 3, dims_coarse, t->chunksize_max,
                    (fftw_complex*)t->buffer_ggp_many, dims_coarse, 1, t->nktot,
                    (fftw_complex*)t->buffer_ggp_many, dims_coarse, 1, t->nktot,
                    FFTW_BACKWARD, FFTW_MEASURE );
            if (!t->fft_many_forward_coarse)
                t->fft_many_forward_coarse = fftw_plan_many_dft( 3, dims_coarse, t->chunksize_max,
                    (fftw_complex*)t->buffer_ggp_many, dims_coarse, 1, t->nktot,
                    (fftw_complex*)t->buffer_ggp_many, dims_coarse, 1, t->nktot,
                    FFTW_FORWARD, FFTW_MEASURE );
            break;
        case tu_fft_algorithm_greedy:
        default:
            #ifndef USE_SERIAL_FFTW
            fftw_plan_with_nthreads(1);
            #endif
            if (!t->buffer_ggm) {
                t->buffer_ggm = calloc( diverge_omp_num_threads(), sizeof*t->buffer_ggm );
                for (index_t i=0; i<diverge_omp_num_threads(); ++i)
                    t->buffer_ggm[i] = (complex128_t*)fftw_alloc_complex(t->nktot);
            }
            if (!t->buffer_ggp) {
                t->buffer_ggp = calloc( diverge_omp_num_threads(), sizeof*t->buffer_ggp );
                for (index_t i=0; i<diverge_omp_num_threads(); ++i)
                    t->buffer_ggp[i] = (complex128_t*)fftw_alloc_complex(t->nktot);
            }
            // coarse-mesh-only transforms
            if (!t->fft_single_backward_coarse)
                t->fft_single_backward_coarse = fftw_plan_many_dft( 3, dims_coarse, 1,
                    (fftw_complex*)t->buffer_ggp[0], dims_coarse, 1, t->nk,
                    (fftw_complex*)t->buffer_ggp[0], dims_coarse, 1, t->nk,
                    FFTW_BACKWARD, FFTW_MEASURE );
            if (!t->fft_single_forward_coarse)
                t->fft_single_forward_coarse = fftw_plan_many_dft( 3, dims_coarse, 1,
                    (fftw_complex*)t->buffer_ggp[0], dims_coarse, 1, t->nk,
                    (fftw_complex*)t->buffer_ggp[0], dims_coarse, 1, t->nk,
                    FFTW_FORWARD, FFTW_MEASURE );
            #ifndef USE_SERIAL_FFTW
            fftw_plan_with_nthreads(fftwnthr);
            #endif
            break;
    }
    mpi_vrb_printf( "TU_prop CPU init done\n" );
}

void cpu_fft_algorithm_cleanup( tu_prop_t* t ) {
    if (t->nktot <= 1) return;

    if (t->fft_single_forward_coarse) fftw_destroy_plan(t->fft_single_forward_coarse);
    if (t->fft_single_backward_coarse) fftw_destroy_plan(t->fft_single_backward_coarse);
    if (t->fft_nons2_backward) fftw_destroy_plan(t->fft_nons2_backward);
    if (t->fft_many_forward_coarse) fftw_destroy_plan(t->fft_many_forward_coarse);
    if (t->fft_many_backward_coarse) fftw_destroy_plan(t->fft_many_backward_coarse);
    if (t->buffer_ggm_many) fftw_free(t->buffer_ggm_many);
    if (t->buffer_ggp_many) fftw_free(t->buffer_ggp_many);

    for (index_t i=0; i<diverge_omp_num_threads(); ++i) {
        if (t->buffer_ggm) fftw_free( t->buffer_ggm[i] );
        if (t->buffer_ggp) fftw_free( t->buffer_ggp[i] );
    }
    free( t->buffer_ggm );
    free( t->buffer_ggp );

    if (t->fft_many_backward_mpi) fftw_mpi_plan_free(t->fft_many_backward_mpi);
    if (t->fft_many_forward_mpi) fftw_mpi_plan_free(t->fft_many_forward_mpi);
    if (t->fft_T_to) fftw_destroy_plan(t->fft_T_to);
    if (t->fft_T_from) fftw_destroy_plan(t->fft_T_from);
    if (t->fft_nons2_backward_mpi) fftw_mpi_plan_free(t->fft_nons2_backward_mpi);

    if (t->mpi_loop) tu_mpi_loop_destroy( t->mpi_loop );
    if (t->mpi_gf) tu_mpi_gf_destroy( t->mpi_gf );
}

void swap_GF_cfrg( tu_prop_t* t ) {
    XORSWAP_PTR( t->GF_fft_plus, t->cFRG_GF_fft_plus );
    XORSWAP_PTR( t->GF_fft_conj_plus, t->cFRG_GF_fft_conj_plus );
}

void GF_zero_T_any( tu_prop_t* t, greensfunc_generator_t gf, complex128_t lambda,
        complex128_t* self_energy, gf_complex_t* gbuf ) {
    greensfunc_op_t res = gf( t->model, lambda, gbuf );
    if (res != greensfunc_op_cpu) mpi_err_printf( "GPU GFs not supported\n" );

    if (self_energy && !t->skip_selfenergy) {
        int enter = 1;
        if (t->model->internals->greens_shared) enter = shared_exclusive_enter(gbuf);
        if (enter) {
            const index_t nb = t->n_spin*t->n_orb;
            const index_t nb2 = POW2(nb), nktot = t->nktot;
            #pragma omp parallel num_threads(diverge_omp_num_threads())
            {
                gf_complex_t* Sbuf = calloc(nb2, sizeof*Sbuf);
                gf_complex_t* Gbuf = calloc(nb2, sizeof*Gbuf);
                #pragma omp for collapse(2)
                for (index_t l=0; l<2; ++l)
                for (index_t k=0; k<nktot; ++k) {
                    // fill self-energy
                    for (index_t ij=0; ij<nb2; ++ij) Sbuf[ij] = self_energy[IDX2(k,ij,nb2)];
                    // invert the free GF
                    cfmatrix_invert_F( gbuf+(l*nktot+k)*nb2, nb, Gbuf );
                    // subtract the selfenergy from the inverted GF
                    for (index_t ij=0; ij<nb2; ++ij) Gbuf[ij] -= Sbuf[ij];
                    // and invert it back to gbuf
                    cfmatrix_invert_F( Gbuf, nb, gbuf+(l*nktot+k)*nb2 );
                }
                free( Sbuf );
                free( Gbuf );
            }
        }
        if (t->model->internals->greens_shared) shared_exclusive_wait(gbuf);
    }
}

void GF_zero_T_FFT_any( tu_prop_t* t, complex128_t* gfft_plus,
        complex128_t* gfft_conj_plus, gf_complex_t* gbuf ) {

    if (t->which_fft_greens == tu_fft_algorithm_mpi_new) {
        tu_mpi_gf_fft_and_transpose( t->mpi_gf, gbuf );
        return;
    }

    const gf_complex_t weight = 1./(double)t->nktot;
    const index_t nb = t->n_spin*t->n_orb;
    const index_t nb2 = POW2(nb), nktot = t->nktot;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t o=0; o<nb; ++o)
    for (index_t b=0; b<nb; ++b)
    for (index_t k=0; k<nktot; ++k)
        gfft_plus[IDX3(o,b,k,nb,nktot)] = gbuf[IDX3(k,b,o,nb,nb)] * weight;

    #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t b=0; b<nb; ++b)
    for (index_t o=0; o<nb; ++o)
    for (index_t k=0; k<nktot; ++k)
        gfft_conj_plus[IDX3(o,b,k,nb,nktot)] = gbuf[IDX3(k,b,o,nb,nb) + nb2*nktot] * weight;
    if (nktot > 1) {
        if (t->which_fft_greens == tu_fft_algorithm_mpi) {
            fftw_mpi_plan_execute( t->fft_nons2_backward_mpi, gfft_plus, gfft_plus, false );
            fftw_mpi_plan_execute( t->fft_nons2_backward_mpi, gfft_conj_plus, gfft_conj_plus, false );
        } else {
            double tick = 0, tock = 0;
            if (t->extra_timings_enabled) tick = diverge_mpi_wtime();
            fftw_execute_dft( t->fft_nons2_backward,(fftw_complex*)gfft_plus,(fftw_complex*)gfft_plus );
            fftw_execute_dft( t->fft_nons2_backward,(fftw_complex*)gfft_conj_plus,(fftw_complex*)gfft_conj_plus );
            if (t->extra_timings_enabled) tock = diverge_mpi_wtime(),
                                          t->extra_timings[0].t += tock - tick;
        }
    }
}
