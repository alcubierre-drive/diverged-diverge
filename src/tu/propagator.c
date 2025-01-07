#include "propagator.h"
#include "propagator_details.h"
#include "propagator_cpu_mpi.h"
#include "propagator_cpu_mpi_fft.h"
#include "propagator_gpu_mpi.h"
#include "cuda_init.h"
#include "../misc/shared_mem.h"

static tu_prop_timing_t extra_timings[32];

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
    #endif
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
    t->extra_timings = extra_timings;
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
        if (model->internals->greens_shared) {
            t->gf_buf_cfrg = shared_calloc(2*t->nktot*POW2(t->n_spin*t->n_orb),sizeof(gf_complex_t));
            mpi_wrn_printf("enabled shared GF for cFRG TU loop\n");
        } else {
            t->gf_buf_cfrg = calloc(2*t->nktot*POW2(t->n_spin*t->n_orb),sizeof(gf_complex_t));
        }
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
    #endif

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

    return t;
}

void tu_prop_free( tu_prop_t* t ) {
    #ifdef USE_CUDA
    if (t->cuda_descr.prop_memory_set_local) {
        for (int i=0; i<tu_gpu_info().tu_num_of_devices; ++i) {
            cudaFree(t->cuda_descr.mi_to_R[i]);
            cudaFree(t->cuda_descr.c_to_f_q[i]);
            cudaFree(t->cuda_descr.nkdev[i]);
            cudaFree(t->cuda_descr.GF_fft[i]);
            cudaFree(t->cuda_descr.GF_fft_conj[i]);
            cudaFree(t->cuda_descr.buf_GG[i]);
            cudaFree(t->cuda_descr.buf_res[i]);
            cudaFree(t->cuda_descr.fft_workspace[i]);
        }
    }
    CUDA_CHECK(cudaHostUnregister(t->GF_fft_plus));
    CUDA_CHECK(cudaHostUnregister(t->GF_fft_conj_plus));

    if (t->cFRG_loop) {
        CUDA_CHECK(cudaHostUnregister(t->cFRG_GF_fft_plus));
        CUDA_CHECK(cudaHostUnregister(t->cFRG_GF_fft_conj_plus));
    }
    #endif

    if (t->do_GF_fft_allocs) {
        fftw_free(t->GF_fft_plus);
        fftw_free(t->GF_fft_conj_plus);
    }
    if (t->cFRG_loop) {
        if (t->model->internals->greens_shared)
            shared_free(t->gf_buf_cfrg);
        else
            free(t->gf_buf_cfrg);

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

    free( t );
}

