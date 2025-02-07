#include "propagator.h"
#include "propagator_details.h"
#include "propagator_cpu_mpi.h"
#include "propagator_cpu_mpi_fft.h"
#include "propagator_gpu_mpi.h"
#include "cuda_init.h"
#include "../misc/shared_mem.h"
#include "coarsify.h"

#include <assert.h>
#include <pthread.h>

tu_prop_t* tu_prop_init( diverge_model_t* model ) {
    tu_prop_t* t = calloc(1, sizeof*t);

    // set default nonzero members
    t->do_GF_fft_allocs = true;
    t->cuda_descr = cuda_prop_defaults;

    t->gf_buf = model->internals->greens;
    t->model = model;
    t->tu_data = model->internals->tu_data;
    t->mi_to_ofrom = t->tu_data->mi_to_ofrom;
    t->mi_to_oto = t->tu_data->mi_to_oto;
    t->mi_to_R = t->tu_data->mi_to_R;
    t->n_orbff = t->tu_data->n_orbff;
    t->n_orb = model->n_orb;
    t->n_bonds = t->tu_data->n_bonds;
    t->n_spin = model->n_spin;
    t->nk = t->tu_data->nk;
    t->nkf = t->tu_data->nkf;
    t->nktot = t->nk*t->nkf;
    t->nkibz = t->tu_data->nkibz;
    t->my_nk = t->tu_data->my_nk;
    t->my_nk_off = t->tu_data->my_nk_off;
    t->SU2 = (model->SU2 > 0);
    t->cFRG_loop = (model->gproj != NULL);
    t->which_fft_simple = t->tu_data->which_fft_simple;
    t->which_fft_red = t->tu_data->which_fft_red;
    t->which_fft_greens = t->tu_data->which_fft_greens;

    tu_data_t* tu_data_write = model->internals->tu_data;
    if (tu_data_write->use_cuda_mpi_loop) {
        if (diverge_mpi_comm_size() <= 1) {
            mpi_wrn_printf( "reset use_cuda_mpi_loop to 0 since we only have one MPI rank\n" );
            tu_data_write->use_cuda_mpi_loop = 0;
        }
    }

    // algorithm choice
    if (t->which_fft_greens == tu_fft_algorithm_mpi_new && t->nktot == 1)
        t->which_fft_greens = tu_fft_algorithm_mpi;
    #ifdef USE_CUDA
    if (t->which_fft_greens == tu_fft_algorithm_mpi_new) {
        t->which_fft_greens = tu_fft_algorithm_mpi;
        mpi_wrn_printf( "TU-loop: cannot use new MPI GF algo with CUDA\n" );
    }
    #endif // USE_CUDA
    if (sizeof(complex128_t) != sizeof(gf_complex_t) &&
        (t->which_fft_simple != tu_fft_algorithm_mpi_new ||
         t->which_fft_red != tu_fft_algorithm_mpi_new)) {
        t->which_fft_greens = tu_fft_algorithm_mpi;
        mpi_wrn_printf( "TU-loop: can only use entirely new MPI algorithms with USE_GF_FLOATS\n" );
    }

    // The 1D problem is described here:
    // https://fftw.org/doc/Basic-and-advanced-distribution-interfaces.html
    if (model->internals->dim <= 1) {
        if (t->which_fft_red == tu_fft_algorithm_mpi_new ||
            t->which_fft_red == tu_fft_algorithm_mpi) {
            t->which_fft_red = tu_fft_algorithm_greedy;
            mpi_wrn_printf( "which_fft_red <- tu_fft_algorithm_greedy (fftw3_mpi cannot do 1D/many)\n" );
        }
        if (t->which_fft_simple == tu_fft_algorithm_mpi_new ||
            t->which_fft_simple == tu_fft_algorithm_mpi) {
            t->which_fft_simple = tu_fft_algorithm_greedy;
            mpi_wrn_printf( "which_fft_simple <- tu_fft_algorithm_greedy (fftw3_mpi cannot do 1D/many)\n" );
        }
        if (t->which_fft_greens == tu_fft_algorithm_mpi_new ||
            t->which_fft_greens == tu_fft_algorithm_mpi) {
            t->which_fft_greens = tu_fft_algorithm_greedy;
            mpi_wrn_printf( "which_fft_greens <- tu_fft_algorithm_greedy (fftw3_mpi cannot do 1D/many)\n" );
        }
    }

    // algorithm preparations
    t->do_GF_fft_allocs = !(t->which_fft_greens == tu_fft_algorithm_mpi_new);

    t->chunksize_max = (t->tu_data->mpi_fft_chunksize < 0) ?
        diverge_omp_num_threads() : t->tu_data->mpi_fft_chunksize;
    t->extra_timings_enabled = t->tu_data->propagator_extra_cpu_timings;
    t->n_extra_timings = 11;
    index_t task_spin = POW4(t->n_spin);
    index_t tasks_dist = t->tu_data->use_reduced_loop ? t->tu_data->unique_mi_l_sz : POW2(t->n_orbff);
    index_t nkff[3]; for (int i=0; i<3; ++i) nkff[i] = model->nk[i] * model->nkf[i];

    cuda_setup_prop_descr(&t->cuda_descr, task_spin, tasks_dist, nkff, t->tu_data->offload_cpu);

    mpi_vrb_printf( "TU-loop: using GF of %.2fGiB\n",
            (double)(sizeof(complex128_t)*t->nktot*POW2(t->n_spin*t->n_orb)) / (1024.*1024.*1024.) );
    if (t->do_GF_fft_allocs) {
        t->GF_fft_plus = (complex128_t*)fftw_alloc_complex(t->nktot* POW2(t->n_spin*t->n_orb));
        t->GF_fft_conj_plus = (complex128_t*)fftw_alloc_complex(t->nktot* POW2(t->n_spin*t->n_orb));
    } else {
        t->GF_fft_plus = (complex128_t*)t->gf_buf;
        t->GF_fft_conj_plus = (complex128_t*)(t->gf_buf + t->nktot*POW2(t->n_spin*t->n_orb));
    }
    if (t->cFRG_loop) {
        t->gf_buf_cfrg = (model->internals->greens_shared ? shared_calloc : calloc)(
                2*t->nktot*POW2(t->n_spin*t->n_orb), sizeof*t->gf_buf_cfrg );
        if (t->do_GF_fft_allocs) {
            t->cFRG_GF_fft_plus = (complex128_t*)fftw_alloc_complex(t->nktot* POW2(t->n_spin*t->n_orb));
            t->cFRG_GF_fft_conj_plus = (complex128_t*)fftw_alloc_complex(t->nktot* POW2(t->n_spin*t->n_orb));
        } else {
            t->cFRG_GF_fft_plus = (complex128_t*)t->gf_buf_cfrg;
            t->cFRG_GF_fft_conj_plus = (complex128_t*)(t->gf_buf_cfrg + t->nktot*POW2(t->n_spin*t->n_orb));
        }
    }

    #ifdef USE_CUDA
    CUDA_CHECK(cudaHostRegister(t->GF_fft_plus, t->nktot*POW2(t->n_spin*t->n_orb)*sizeof(fftw_complex),
                cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(t->GF_fft_conj_plus, t->nktot*POW2(t->n_spin*t->n_orb)*sizeof(fftw_complex),
                cudaHostRegisterDefault));
    if (t->cFRG_loop) {
        CUDA_CHECK(cudaHostRegister( t->cFRG_GF_fft_plus, t->nktot*POW2(t->n_spin*t->n_orb)*sizeof(fftw_complex),
                    cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister( t->cFRG_GF_fft_conj_plus, t->nktot*POW2(t->n_spin*t->n_orb)*
                    sizeof(fftw_complex), cudaHostRegisterDefault));
    }
    #endif // USE_CUDA

    t->c2fmap = calloc(t->my_nk, sizeof*t->c2fmap);
    t->c2fmap_sz = t->my_nk;
    if (t->tu_data->symm->use_symmetries) {
        for (index_t i=0; i<t->my_nk; ++i)
            t->c2fmap[i] = kidxc2f(t->tu_data->symm->idx_ibz_in_fullmesh[i+t->my_nk_off], model->nk, model->nkf);
    } else {
        for (index_t i=0; i<t->my_nk; ++i)
            t->c2fmap[i] = kidxc2f(i+t->my_nk_off, model->nk, model->nkf);
    }
    cpu_fft_algorithm_init( t, t->which_fft_greens, t->which_fft_simple, t->which_fft_red );

    t->coarsify = t->tu_data->GF_fft_coarsify ? greensfunc_coarse_init( model->nk,
            model->nkf, model->n_orb*model->n_spin, model->internals->dim ) : NULL;

    return t;
}

void tu_prop_free( tu_prop_t* t ) {
    #ifdef USE_CUDA
    if (t->cuda_descr.prop_memory_set_local) {
        for (int i=0; i<tu_gpu_info().tu_num_of_devices; ++i) {
            CUDA_CHECK( cudaFree(t->cuda_descr.mi_to_R[i]) );
            CUDA_CHECK( cudaFree(t->cuda_descr.c_to_f_q[i]) );
            CUDA_CHECK( cudaFree(t->cuda_descr.nkdev[i]) );
            CUDA_CHECK( cudaFree(t->cuda_descr.GF_fft[i]) );
            CUDA_CHECK( cudaFree(t->cuda_descr.GF_fft_conj[i]) );
            CUDA_CHECK( cudaFree(t->cuda_descr.buf_GG[i]) );
            CUDA_CHECK( cudaFree(t->cuda_descr.buf_res[i]) );
            CUDA_CHECK( cudaFree(t->cuda_descr.fft_workspace[i]) );
        }
    }
    CUDA_CHECK(cudaHostUnregister(t->GF_fft_plus));
    CUDA_CHECK(cudaHostUnregister(t->GF_fft_conj_plus));

    if (t->cFRG_loop) {
        CUDA_CHECK(cudaHostUnregister(t->cFRG_GF_fft_plus));
        CUDA_CHECK(cudaHostUnregister(t->cFRG_GF_fft_conj_plus));
    }
    #endif // USE_CUDA

    if (t->do_GF_fft_allocs) {
        fftw_free(t->GF_fft_plus);
        fftw_free(t->GF_fft_conj_plus);
    }
    if (t->cFRG_loop) {
        (t->model->internals->greens_shared ? shared_free : free)(t->gf_buf_cfrg);

        if (t->do_GF_fft_allocs) {
            fftw_free(t->cFRG_GF_fft_plus);
            fftw_free(t->cFRG_GF_fft_conj_plus);
        }
    }
    cuda_delete_prop_descr(&t->cuda_descr);

    cpu_fft_algorithm_cleanup( t );

    if (t->gpu_mpi_dist_pp) tu_prop_gpu_mpi_dist_free(t->gpu_mpi_dist_pp);
    if (t->gpu_mpi_dist_ph) tu_prop_gpu_mpi_dist_free(t->gpu_mpi_dist_ph);

    free( t->c2fmap );
    free( t->c2fmap_full );

    if (t->coarsify) greensfunc_coarse_free(t->coarsify);

    free( t );
}

void tu_prop_GF_zero_T( tu_prop_t* t, complex128_t lambda, complex128_t* self_energy ) {
    GF_zero_T_any( t, t->model->gfill, lambda, self_energy, t->gf_buf );
    GF_zero_T_FFT_any( t, t->GF_fft_plus, t->GF_fft_conj_plus, t->gf_buf );
    if (t->cFRG_loop) {
        GF_zero_T_any( t, t->model->gproj, lambda, self_energy, t->gf_buf_cfrg );
        GF_zero_T_FFT_any( t, t->cFRG_GF_fft_plus, t->cFRG_GF_fft_conj_plus, t->gf_buf_cfrg );
    }
}

void tu_prop_GF_for_self( tu_prop_t* t, complex128_t* coarseGF ) {
    // we here write the buffer into a *transposed* form, because the TUFRG
    // backend for self-energies expects the o1/o2 index order the other way
    // round.
    const index_t nb = (t->n_spin*t->n_orb), nk = t->nk;
    const index_t nb2 = POW2(nb);

    if (t->coarsify) {
        memcpy( coarseGF, greensfunc_coarse_calc( t->coarsify, t->gf_buf ), sizeof*coarseGF * nk*POW2(nb) );
        if (t->cFRG_loop) {
            complex128_t* coarseGF_cFRG = greensfunc_coarse_calc( t->coarsify, t->gf_buf_cfrg );
            #pragma omp parallel for num_threads(diverge_omp_num_threads())
            for (index_t i=0; i<nk*POW2(nb); ++i) coarseGF[i] -= coarseGF_cFRG[i];
        }
        return;
    }

    #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t b=0; b<nb; ++b)
    for (index_t o=0; o<nb; ++o)
    for (index_t k=0; k<nk; ++k)
        coarseGF[k+nk*(b+nb*o)] = t->gf_buf[o+nb*b+nb2*kidxc2f(k,t->model->nk,t->model->nkf)];
    if (t->cFRG_loop) {
        #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t b=0; b<nb; ++b)
        for (index_t o=0; o<nb; ++o)
        for (index_t k=0; k<nk; ++k)
            coarseGF[k+nk*(b+nb*o)] -= t->gf_buf_cfrg[o+nb*b+nb2*kidxc2f(k,t->model->nk,t->model->nkf)];
    }
}

void tu_prop_set_extra_timings( tu_prop_t* t, bool enabled ) { t->extra_timings_enabled = enabled; }

diverge_timing_t* tu_prop_get_extra_timings( tu_prop_t* t, int* ntimings ) {
    if (!t->extra_timings_enabled) { *ntimings = 0; return NULL; }
    t->n_extra_timings = 11; // reset
    double tvec[3] = {0};
    if (t->fft_nons2_backward_mpi) {
        fftw_mpi_plan_timings( t->fft_nons2_backward_mpi, tvec );
        for (int i=1; i<4; ++i) t->extra_timings[i].t = tvec[i-1];
    }
    if (!t->mpi_loop) {
        if (t->fft_many_backward_mpi) {
            fftw_mpi_plan_timings( t->fft_many_backward_mpi, tvec );
            for (int i=5; i<8; ++i) t->extra_timings[i].t = tvec[i-5];
        }
        if (t->fft_many_forward_mpi) {
            fftw_mpi_plan_timings( t->fft_many_forward_mpi, tvec );
            for (int i=8; i<11; ++i) t->extra_timings[i].t = tvec[i-8];
        }
    } else {
        t->n_extra_timings = 4;
        for (int i=0; i<TU_MPI_LOOP_TIMING_NUM; ++i)
            t->extra_timings[t->n_extra_timings++].t = t->mpi_loop->timings[i];
    }
    // overwrite first couple of timings of GF is the MPI one
    if (t->mpi_gf)
        for (int i=0; i<TU_MPI_GF_TIMING_NUM; ++i)
            t->extra_timings[i].t = t->mpi_gf->timings[i];

    if (t->gpu_mpi_dist_pp)
        for (index_t i=0; i<tu_prop_gpu_mpi_dist_timing_num( t->gpu_mpi_dist_pp ); ++i)
            t->extra_timings[t->n_extra_timings++].t = tu_prop_gpu_mpi_dist_timing( t->gpu_mpi_dist_pp, i );
    if (t->gpu_mpi_dist_ph)
        for (index_t i=0; i<tu_prop_gpu_mpi_dist_timing_num( t->gpu_mpi_dist_ph ); ++i)
            t->extra_timings[t->n_extra_timings++].t = tu_prop_gpu_mpi_dist_timing( t->gpu_mpi_dist_ph, i );

    *ntimings = t->n_extra_timings;

    // and the names
    int name_idx = 0;
    if (!t->mpi_gf) {
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft GF serial", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft GF mpi reorder", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft GF mpi fft", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft GF mpi comm", 55 );
    } else {
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_gf_timing_trans", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_gf_timing_ffttrans", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_gf_timing_fft", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_gf_timing_comm", 55 );
    }
    if (!t->mpi_loop) {
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft many serial", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft many mpi+ reorder", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft many mpi+ fft", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft many mpi+ comm", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft many mpi- reorder", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft many mpi- fft", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "loop-fft many mpi- comm", 55 );
    } else {
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_prepare", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_reorder", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_transpose", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_fft", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_insert", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_mpibufprep", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_comm", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_insert_missing", 55 );
        strncpy( t->extra_timings[name_idx++].nam, "tu_mpi_loop_timing_postproc", 55 );
    }
    if (t->gpu_mpi_dist_pp)
        for (index_t i=0; i<tu_prop_gpu_mpi_dist_timing_num( t->gpu_mpi_dist_pp ); ++i) {
            strcpy( t->extra_timings[name_idx].nam, "pp:" );
            strncat( t->extra_timings[name_idx].nam, tu_prop_gpu_mpi_dist_timing_name(t->gpu_mpi_dist_pp,i), 55 );
            name_idx++;
        }
    if (t->gpu_mpi_dist_ph)
        for (index_t i=0; i<tu_prop_gpu_mpi_dist_timing_num( t->gpu_mpi_dist_ph ); ++i) {
            strcpy( t->extra_timings[name_idx].nam, "ph:" );
            strncat( t->extra_timings[name_idx].nam, tu_prop_gpu_mpi_dist_timing_name(t->gpu_mpi_dist_ph,i), 55 );
            name_idx++;
        }

    assert( name_idx == t->n_extra_timings );
    return t->extra_timings;
}

#ifdef USE_CUDA

typedef struct {
    tu_prop_t* t;
    complex128_t* tmpgpu;
    void (*gpu_loop)(tu_prop_t*, complex128_t*);
} gpu_thread_t;

void* gpu_thread_fun( void* data ) {
    gpu_thread_t* g = data;
    g->gpu_loop( g->t, g->tmpgpu );
    return NULL;
}

void tu_prop_ph_loop( tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu ) {
    if (t->nktot==1) {
        S_loop_static_rs( t, false, false, L );
        if (t->cFRG_loop) {
            swap_GF_cfrg( t );
            S_loop_static_rs( t, false, true, L );
            swap_GF_cfrg( t );
        }
    } else {
        bool do_free = false;
        if (t->cuda_descr.prop_has_gpu_setup && tmpgpu == NULL) {
            do_free = true;
            index_t my_nk_tot = (t->tu_data->use_cuda_mpi_loop && t->tu_data->use_reduced_loop) ?
                t->nkibz : t->my_nk;
            CUDA_CHECK( cudaMallocHost((void**)&tmpgpu,
                        my_nk_tot*POW2(t->n_orbff*POW2(t->n_spin))*sizeof(*tmpgpu)) );
        }
        ZERO_OUT( L, t->my_nk*POW2(t->n_orbff*POW2(t->n_spin)) );

        // set the right function pointer, either _red or _simple
        void (*cpu_loop)(tu_prop_t*, bool, bool, complex128_t*, index_t, index_t, bool) =
            t->tu_data->use_reduced_loop ? &S_loop_static_red : &S_loop_static_simple;
        gpu_thread_t gt = {.t = t, .tmpgpu = tmpgpu, .gpu_loop = t->tu_data->use_reduced_loop ?
            &S_ph_loop_static_red_gpu : &S_ph_loop_static_simple_gpu };

        pthread_t gpu_thread = {0};
        pthread_create( &gpu_thread, NULL, &gpu_thread_fun, &gt );
        (*cpu_loop)( t, false, false, L, t->cuda_descr._tasks_cpu, t->cuda_descr._off_cpu,
                !t->cuda_descr.prop_has_gpu_setup );
        pthread_join( gpu_thread, NULL );

        if (t->cFRG_loop) {
            swap_GF_cfrg( t );
            if (t->cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop_cfrg_step1( t, L, tmpgpu );

            gpu_thread = (pthread_t){0};
            pthread_create( &gpu_thread, NULL, &gpu_thread_fun, &gt );
            (*cpu_loop)( t, false, true, L, t->cuda_descr._tasks_cpu, t->cuda_descr._off_cpu,
                    !t->cuda_descr.prop_has_gpu_setup );
            pthread_join( gpu_thread, NULL );

            swap_GF_cfrg( t );
            if (t->cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop_cfrg_step2( t, L, tmpgpu );
        } else {
            if (t->cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop( t, L, tmpgpu );
        }
        if (do_free) CUDA_CHECK( cudaFreeHost(tmpgpu) );
    }
}

void tu_prop_pp_loop( tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu ) {
    if (t->nktot==1) {
        S_loop_static_rs( t, true, false, L );
        if (t->cFRG_loop) {
            swap_GF_cfrg( t );
            S_loop_static_rs( t, true, true, L );
            swap_GF_cfrg( t );
        }
    } else {
        bool do_free = false;
        if (t->cuda_descr.prop_has_gpu_setup && tmpgpu == NULL) {
            do_free = true;
            index_t my_nk_tot = (t->tu_data->use_cuda_mpi_loop && t->tu_data->use_reduced_loop) ?
                t->nkibz : t->my_nk;
            CUDA_CHECK( cudaMallocHost((void**)&tmpgpu,
                        my_nk_tot*POW2(t->n_orbff*POW2(t->n_spin))*sizeof(*tmpgpu)) );
        }
        ZERO_OUT( L, t->my_nk*POW2(t->n_orbff*POW2(t->n_spin)) );

        // set the right function pointer, either _red or _simple
        void (*cpu_loop)(tu_prop_t*, bool, bool, complex128_t*, index_t, index_t, bool) =
            t->tu_data->use_reduced_loop ? &S_loop_static_red : &S_loop_static_simple;
        gpu_thread_t gt = {.t = t, .tmpgpu = tmpgpu, .gpu_loop = t->tu_data->use_reduced_loop ?
            &S_pp_loop_static_red_gpu : &S_pp_loop_static_simple_gpu };

        pthread_t gpu_thread = {0};
        pthread_create( &gpu_thread, NULL, &gpu_thread_fun, &gt );
        (*cpu_loop)( t, true, false, L, t->cuda_descr._tasks_cpu, t->cuda_descr._off_cpu,
                !t->cuda_descr.prop_has_gpu_setup );
        pthread_join( gpu_thread, NULL );

        if (t->cFRG_loop) {
            swap_GF_cfrg( t );
            if (t->cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop_cfrg_step1( t, L, tmpgpu );

            gpu_thread = (pthread_t){0};
            pthread_create( &gpu_thread, NULL, &gpu_thread_fun, &gt );
            (*cpu_loop)( t, true, true, L, t->cuda_descr._tasks_cpu, t->cuda_descr._off_cpu,
                    !t->cuda_descr.prop_has_gpu_setup );
            pthread_join( gpu_thread, NULL );

            swap_GF_cfrg( t );
            if (t->cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop_cfrg_step2( t, L, tmpgpu );
        } else {
            if (t->cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop( t, L, tmpgpu );
        }
        if (do_free) CUDA_CHECK( cudaFreeHost(tmpgpu) );
    }
}

#else // USE_CUDA

void tu_prop_ph_loop( tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu ) {
    (void)tmpgpu;

    // set the right function pointer, either _red or _simple
    void (*cpu_loop)(tu_prop_t*, bool, bool, complex128_t*, index_t, index_t, bool) =
            t->tu_data->use_reduced_loop ? &S_loop_static_red : &S_loop_static_simple;

    if (t->nktot==1) {
        S_loop_static_rs( t, false, false, L );
    } else {
        bool rescale = !t->cuda_descr.prop_has_gpu_setup;
        if (t->cFRG_loop) rescale = false;
        ZERO_OUT( L, t->my_nk*POW2(t->n_orbff*POW2(t->n_spin)) );
        (*cpu_loop)( t, false, false, L, t->cuda_descr._tasks_cpu, t->cuda_descr._off_cpu, rescale );
    }
    if (t->cFRG_loop) {
        swap_GF_cfrg( t );
        if (t->nktot==1) {
            S_loop_static_rs( t, false, true, L );
        } else {
            (*cpu_loop)( t, false, true, L, t->cuda_descr._tasks_cpu, t->cuda_descr._off_cpu,
                                    !t->cuda_descr.prop_has_gpu_setup );
        }
        swap_GF_cfrg( t );
    }
}

void tu_prop_pp_loop( tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu ) {
    (void)tmpgpu;

    // set the right function pointer, either _red or _simple
    void (*cpu_loop)(tu_prop_t*, bool, bool, complex128_t*, index_t, index_t, bool) =
            t->tu_data->use_reduced_loop ? &S_loop_static_red : &S_loop_static_simple;

    if (t->nktot==1) {
        S_loop_static_rs( t, true, false, L );
    } else {
        bool rescale = !t->cuda_descr.prop_has_gpu_setup;
        if (t->cFRG_loop) rescale = false;
        ZERO_OUT( L, t->my_nk*POW2(t->n_orbff*POW2(t->n_spin)) );
        (*cpu_loop)( t, true, false, L, t->cuda_descr._tasks_cpu, t->cuda_descr._off_cpu, rescale );
    }
    if (t->cFRG_loop) {
        swap_GF_cfrg( t );
        if (t->nktot==1) {
            S_loop_static_rs( t, true, true, L );
        } else {
            (*cpu_loop)( t, true, true, L, t->cuda_descr._tasks_cpu, t->cuda_descr._off_cpu,
                                    !t->cuda_descr.prop_has_gpu_setup );
        }
        swap_GF_cfrg( t );
    }
}

#endif // USE_CUDA
