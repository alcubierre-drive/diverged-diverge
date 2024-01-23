/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "propagator.hpp"
#include "cuda_init.hpp"
#include "../misc/batched_eigen.h"
#include "../misc/init_internal_libs.h"
#include <iostream>

// this function is needed for the C MPI kernel
extern "C" void tu_loop_reconstruct_from_reduced( void* handle, complex128_t* loop ) {
    tu_loop_t* t = (tu_loop_t*)handle;
    t->reconstruct_from_reduced( loop );
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

tu_loop_t::tu_loop_t(diverge_model_t* modelin)
:
    gf_buf{modelin->internals->greens},
    model{modelin},
    tu_data{(tu_data_t*)(model -> internals-> tu_data)},
    mi_to_ofrom{tu_data->mi_to_ofrom},
    mi_to_oto{tu_data->mi_to_oto},
    mi_to_R{tu_data->mi_to_R},
    n_orbff{tu_data->n_orbff},
    n_orb{model->n_orb},
    n_bonds{tu_data-> n_bonds},
    n_spin{model->n_spin},
    nk{tu_data->nk},
    nkf{tu_data->nkf},
    nktot{nk*nkf},
    my_nk{tu_data->my_nk},
    my_nk_off{tu_data->my_nk_off},
    SU2{model->SU2 > 0},
    cFRG_loop{model->gproj!=NULL},
    which_fft_simple{tu_data->which_fft_simple},
    which_fft_red{tu_data->which_fft_red},
    which_fft_greens{tu_data->which_fft_greens}
{

    if (which_fft_greens == tu_fft_algorithm_mpi_new && nktot == 1)
        which_fft_greens = tu_fft_algorithm_mpi;
    #ifdef USE_CUDA
    if (which_fft_greens == tu_fft_algorithm_mpi_new) {
        which_fft_greens = tu_fft_algorithm_mpi;
        mpi_wrn_printf( "TU-loop: cannot use new MPI GF algo with CUDA\n" );
    }
    #endif
    if (sizeof(complex128_t) != sizeof(gf_complex_t) &&
        (which_fft_simple != tu_fft_algorithm_mpi_new ||
         which_fft_red != tu_fft_algorithm_mpi_new)) {
        which_fft_greens = tu_fft_algorithm_mpi;
        mpi_wrn_printf( "TU-loop: can only use entirely new MPI algorithms with USE_GF_FLOATS\n" );
    }
    do_GF_fft_allocs = !(which_fft_greens == tu_fft_algorithm_mpi_new);

    chunksize_max = tu_data->mpi_fft_chunksize < 0 ? diverge_omp_num_threads() : tu_data->mpi_fft_chunksize;
    extra_timings_enabled = tu_data->propagator_extra_cpu_timings;
    extra_tvec = new vector<double>( 11, 0.0 );
    index_t task_spin = POW4(n_spin);
    index_t tasks_dist = tu_data->use_reduced_loop ? (index_t)tu_data->unique_mi_l.size() : POW2(n_orbff);
    index_t nkff[3];
    for (int i=0; i<3; ++i) nkff[i] = model->nk[i] * model->nkf[i];
    cuda_setup_prop_descr(&cuda_descr,task_spin,tasks_dist, nkff,tu_data->offload_cpu);

    mpi_vrb_printf( "TU-loop: using GF of %.2fGiB\n",
            (double)(sizeof(complex128_t)*nktot*POW2(n_spin*n_orb)) / (1024.*1024.*1024.) );
    if (do_GF_fft_allocs) {
        GF_fft_plus = (complex128_t*)fftw_alloc_complex(nktot* POW2(n_spin*n_orb));
        GF_fft_conj_plus = (complex128_t*)fftw_alloc_complex(nktot* POW2(n_spin*n_orb));
    } else {
        GF_fft_plus = (complex128_t*)gf_buf;
        GF_fft_conj_plus = (complex128_t*)(gf_buf + nktot*POW2(n_spin*n_orb));
    }
    if(cFRG_loop) {
        gf_buf_cfrg = (gf_complex_t*)calloc(2*nktot*POW2(n_spin*n_orb),sizeof(gf_complex_t));
        if (do_GF_fft_allocs) {
            cFRG_GF_fft_plus = (complex128_t*)fftw_alloc_complex(nktot* POW2(n_spin*n_orb));
            cFRG_GF_fft_conj_plus = (complex128_t*)fftw_alloc_complex(nktot* POW2(n_spin*n_orb));
        } else {
            cFRG_GF_fft_plus = (complex128_t*)gf_buf_cfrg;
            cFRG_GF_fft_conj_plus = (complex128_t*)(gf_buf_cfrg + nktot*POW2(n_spin*n_orb));
        }
    }
    #ifdef USE_CUDA
    CUDA_CHECK(cudaHostRegister(GF_fft_plus,nktot* POW2(n_spin*n_orb)*sizeof(fftw_complex), cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(GF_fft_conj_plus,nktot* POW2(n_spin*n_orb)*sizeof(fftw_complex), cudaHostRegisterDefault));
    if(cFRG_loop) {
        CUDA_CHECK(cudaHostRegister(cFRG_GF_fft_plus,nktot* POW2(n_spin*n_orb)*sizeof(fftw_complex), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(cFRG_GF_fft_conj_plus,nktot* POW2(n_spin*n_orb)*sizeof(fftw_complex), cudaHostRegisterDefault));
    }
    #endif

    c2fmap.resize(my_nk);
    if(tu_data->symm->use_symmetries) {
        for(index_t i = 0; i < my_nk;++i)
            c2fmap[i] = kidxc2f(tu_data->symm->idx_ibz_in_fullmesh[i+my_nk_off],
                                model->nk,model->nkf);
    }else{
        for(index_t i = 0; i < my_nk;++i)
            c2fmap[i] = kidxc2f(i+my_nk_off,model->nk,model->nkf);
    }
    cpu_fft_algorithm_init( which_fft_greens, which_fft_simple, which_fft_red );
}


vector<double> tu_loop_t::extra_timings( void ) {
    if (!extra_timings_enabled) return vector<double>();
    if (fft_nons2_backward_mpi) {
        array<double,3> t = fft_nons2_backward_mpi->timings();
        extra_tvec->at(1) = t[0];
        extra_tvec->at(2) = t[1];
        extra_tvec->at(3) = t[2];
    }
    if (!mpi_loop) {
        if (fft_many_backward_mpi) {
            array<double,3> t = fft_many_backward_mpi->timings();
            extra_tvec->at(5) = t[0];
            extra_tvec->at(6) = t[1];
            extra_tvec->at(7) = t[2];
        }
        if (fft_many_forward_mpi) {
            array<double,3> t = fft_many_forward_mpi->timings();
            extra_tvec->at(8) = t[0];
            extra_tvec->at(9) = t[1];
            extra_tvec->at(10) = t[2];
        }
    } else {
        extra_tvec->resize(4);
        for (int i=0; i<TU_MPI_LOOP_TIMING_NUM; ++i)
            extra_tvec->push_back( mpi_loop->timings[i] );
    }
    // overwrite first couple of timings of GF is the MPI one
    if (mpi_gf)
        for (int i=0; i<TU_MPI_GF_TIMING_NUM; ++i)
            extra_tvec->at(i) = mpi_gf->timings[i];
    return *extra_tvec;
}

vector<string> tu_loop_t::extra_timings_descr( void ) {
    if (!extra_timings_enabled) return vector<string>();
    vector<string> descr;
    if (!mpi_gf)
        descr = { "loop-fft GF serial",
                  "loop-fft GF mpi reorder",
                  "loop-fft GF mpi fft",
                  "loop-fft GF mpi comm", };
    else
        descr = { "tu_mpi_gf_timing_trans",
                  "tu_mpi_gf_timing_ffttrans",
                  "tu_mpi_gf_timing_fft",
                  "tu_mpi_gf_timing_comm" };
    if (!mpi_loop)
        for (const auto& s: { "loop-fft many serial",
                              "loop-fft many mpi+ reorder",
                              "loop-fft many mpi+ fft",
                              "loop-fft many mpi+ comm",
                              "loop-fft many mpi- reorder",
                              "loop-fft many mpi- fft",
                              "loop-fft many mpi- comm" }) descr.push_back(s);
    else
        for (const auto& s: { "tu_mpi_loop_timing_prepare",
                              "tu_mpi_loop_timing_reorder",
                              "tu_mpi_loop_timing_transpose",
                              "tu_mpi_loop_timing_fft",
                              "tu_mpi_loop_timing_insert",
                              "tu_mpi_loop_timing_mpibufprep",
                              "tu_mpi_loop_timing_comm",
                              "tu_mpi_loop_timing_insert_missing",
                              "tu_mpi_loop_timing_postproc" }) descr.push_back(s);
    return descr;
}

void tu_loop_t::extra_timings_set( bool enabled ) {
    if (fft_many_backward_mpi) fft_many_backward_mpi->timings_enabled = enabled;
    if (fft_many_forward_mpi) fft_many_forward_mpi->timings_enabled = enabled;
    if (fft_nons2_backward_mpi) fft_nons2_backward_mpi->timings_enabled = enabled;
    this->extra_timings_enabled = enabled;
}

void tu_loop_t::cpu_fft_algorithm_cleanup( void ) {
    if (nktot <= 1) return;

    if (fft_single_forward) fftw_destroy_plan(fft_single_forward);
    if (fft_single_backward) fftw_destroy_plan(fft_single_backward);
    if (fft_nons2_backward) fftw_destroy_plan(fft_nons2_backward);
    if (fft_many_forward) fftw_destroy_plan(fft_many_forward);
    if (fft_many_backward) fftw_destroy_plan(fft_many_backward);
    if (buffer_ggm_many) fftw_free(buffer_ggm_many);
    if (buffer_ggp_many) fftw_free(buffer_ggp_many);

    for (complex128_t* buf : buffer_ggm) fftw_free(buf);
    for (complex128_t* buf : buffer_ggp) fftw_free(buf);

    if (fft_many_backward_mpi) delete fft_many_backward_mpi;
    if (fft_many_forward_mpi) delete fft_many_forward_mpi;
    if (fft_T_to) fftw_destroy_plan(fft_T_to);
    if (fft_T_from) fftw_destroy_plan(fft_T_from);
    if (fft_nons2_backward_mpi) delete fft_nons2_backward_mpi;

    if (mpi_loop) tu_mpi_loop_destroy( mpi_loop );
}

static std::string tu_fft_algorithm_name( tu_fft_algorithm_t algo ) {
    switch (algo) {
        case tu_fft_algorithm_greedy: return "greedy";
        case tu_fft_algorithm_split: return "split";
        case tu_fft_algorithm_mpi: return "mpi";
        case tu_fft_algorithm_mpi_new: return "mpi_new";
    }
    return "invalid";
}

void tu_loop_t::cpu_fft_algorithm_init( tu_fft_algorithm_t greens,
        tu_fft_algorithm_t simple, tu_fft_algorithm_t red ) {

    if (nktot <= 1) return;

    mpi_vrb_printf( "TU loop algorithm choice: GF: %s, simple: %s, reduced: %s\n",
            tu_fft_algorithm_name( greens ).c_str(),
            tu_fft_algorithm_name( simple ).c_str(),
            tu_fft_algorithm_name( red ).c_str() );

    int fftwnthr = diverge_fftw_num_threads();
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]), (int)(vnk[2]*vnkf[2])};

    switch (greens) {
        case tu_fft_algorithm_mpi_new:
            if (!mpi_gf)
                mpi_gf = tu_mpi_gf_init( vnk, vnkf, model->n_orb*model->n_spin, gf_buf, model );
            break;
        case tu_fft_algorithm_mpi:
            if (!fft_nons2_backward_mpi)
                fft_nons2_backward_mpi = new fftw_mpi_plan( dims, POW2(n_spin*n_orb),
                    GF_fft_plus, GF_fft_plus, FFTW_BACKWARD, (complex128_t*)gf_buf );
            break;
        case tu_fft_algorithm_split:
        case tu_fft_algorithm_greedy:
        default:
            if (!fft_nons2_backward)
                fft_nons2_backward = fftw_plan_many_dft( 3, dims, POW2(n_spin*n_orb),
                    (fftw_complex*)GF_fft_plus, dims, 1, nktot,
                    (fftw_complex*)GF_fft_plus, dims, 1, nktot,
                    FFTW_BACKWARD, FFTW_MEASURE );
            break;
    }

    tu_fft_algorithm_t simple_red_algos[] = {simple, red};
    for (int i=0; i<2; ++i) switch (simple_red_algos[i]) {
        case tu_fft_algorithm_mpi_new:
            if (!mpi_loop) mpi_loop = tu_mpi_loop_init( model->nk, model->nkf,
                    chunksize_max, c2fmap.data(), my_nk, n_spin, n_orb, n_orbff,
                    tu_data->unique_mi_l.data(), tu_data->unique_mi_l.size(),
                    tu_data->unique_mi_r.data(), mi_to_ofrom, mi_to_oto, mi_to_R,
                    &GF_fft_plus, &GF_fft_conj_plus, this, model );
            break;
        case tu_fft_algorithm_mpi:
            if (!buffer_ggm_many)
                buffer_ggm_many = (complex128_t*)fftw_alloc_complex(nktot * chunksize_max);
            if (!buffer_ggp_many)
                buffer_ggp_many = (complex128_t*)fftw_alloc_complex(nktot * chunksize_max);
            if (!fft_many_backward_mpi)
                fft_many_backward_mpi = new fftw_mpi_plan( dims, chunksize_max,
                    buffer_ggm_many, buffer_ggp_many, FFTW_BACKWARD );
            if (!fft_many_forward_mpi)
                fft_many_forward_mpi = new fftw_mpi_plan( dims, chunksize_max,
                    buffer_ggm_many, buffer_ggp_many, FFTW_FORWARD );
            if (!fft_T_to)
                fft_T_to = plan_transpose( chunksize_max, nktot, buffer_ggm_many, buffer_ggp_many );
            if (!fft_T_from)
                fft_T_from = plan_transpose( nktot, chunksize_max, buffer_ggm_many, buffer_ggp_many );
            break;
        case tu_fft_algorithm_split:
            if (!buffer_ggm_many)
                buffer_ggm_many = (complex128_t*)fftw_alloc_complex(nktot * chunksize_max);
            if (!buffer_ggp_many)
                buffer_ggp_many = (complex128_t*)fftw_alloc_complex(nktot * chunksize_max);
            if (!fft_many_backward)
                fft_many_backward = fftw_plan_many_dft( 3, dims, chunksize_max,
                    (fftw_complex*)buffer_ggm_many, dims, 1, nktot,
                    (fftw_complex*)buffer_ggp_many, dims, 1, nktot,
                    FFTW_BACKWARD, FFTW_MEASURE );
            if (!fft_many_forward)
                fft_many_forward = fftw_plan_many_dft( 3, dims, chunksize_max,
                    (fftw_complex*)buffer_ggm_many, dims, 1, nktot,
                    (fftw_complex*)buffer_ggp_many, dims, 1, nktot,
                    FFTW_FORWARD, FFTW_MEASURE );
            break;
        case tu_fft_algorithm_greedy:
        default:
            fftw_plan_with_nthreads(1);
            if (buffer_ggm.size() == 0u) {
                buffer_ggm.resize(diverge_omp_num_threads());
                for (auto& buf: buffer_ggm) buf = (complex128_t*)fftw_alloc_complex(nktot);
            }
            if (buffer_ggp.size() == 0u) {
                buffer_ggp.resize(diverge_omp_num_threads());
                for (auto& buf: buffer_ggp) buf = (complex128_t*)fftw_alloc_complex(nktot);
            }
            if (!fft_single_backward)
                fft_single_backward = fftw_plan_many_dft( 3, dims, 1,
                    (fftw_complex*)buffer_ggm[0], dims, 1, nktot,
                    (fftw_complex*)buffer_ggp[0], dims, 1, nktot,
                    FFTW_BACKWARD, FFTW_MEASURE );
            if (!fft_single_forward)
                fft_single_forward = fftw_plan_many_dft( 3, dims, 1,
                    (fftw_complex*)buffer_ggm[0], dims, 1, nktot,
                    (fftw_complex*)buffer_ggp[0], dims, 1, nktot,
                    FFTW_FORWARD, FFTW_MEASURE );
            fftw_plan_with_nthreads(fftwnthr);
            break;
    }
}



tu_loop_t::~tu_loop_t() {
    #ifdef USE_CUDA
    if(cuda_descr.prop_memory_set_local) {
        for(int i = 0; i<tu_num_of_devices;++i) {
            cudaFree(cuda_descr.mi_to_R[i]);
            cudaFree(cuda_descr.c_to_f_q[i]);
            cudaFree(cuda_descr.nkdev[i]);
            cudaFree(cuda_descr.GF_fft[i]);
            cudaFree(cuda_descr.GF_fft_conj[i]);
            cudaFree(cuda_descr.buf_GG[i]);
            cudaFree(cuda_descr.buf_res[i]);
            cudaFree(cuda_descr.fft_workspace[i]);
        }
    }
    CUDA_CHECK(cudaHostUnregister(GF_fft_plus));
    CUDA_CHECK(cudaHostUnregister(GF_fft_conj_plus));

    if(cFRG_loop) {
        CUDA_CHECK(cudaHostUnregister(cFRG_GF_fft_plus));
        CUDA_CHECK(cudaHostUnregister(cFRG_GF_fft_conj_plus));
    }
    #endif

    if (do_GF_fft_allocs) {
        fftw_free(GF_fft_plus);
        fftw_free(GF_fft_conj_plus);
    }
    if(cFRG_loop) {
        free(gf_buf_cfrg);
        if (do_GF_fft_allocs) {
            fftw_free(cFRG_GF_fft_plus);
            fftw_free(cFRG_GF_fft_conj_plus);
        }
    }
    cuda_delete_prop_descr(&cuda_descr);

    cpu_fft_algorithm_cleanup();

    delete extra_tvec;
}

void tu_loop_t::GF_zero_T( complex128_t lambda, complex128_t* self_energy ) {
    GF_zero_T_any( model->gfill, lambda, self_energy, gf_buf );
    GF_zero_T_FFT_any( GF_fft_plus, GF_fft_conj_plus, gf_buf );
    if (cFRG_loop) {
        GF_zero_T_any( model->gproj, lambda, self_energy, gf_buf_cfrg );
        GF_zero_T_FFT_any( cFRG_GF_fft_plus, cFRG_GF_fft_conj_plus, gf_buf_cfrg );
    }
}

void tu_loop_t::GF_zero_T_any( greensfunc_generator_t gf, complex128_t lambda, complex128_t* self_energy, gf_complex_t* gbuf ) {
    greensfunc_op_t res = gf( model, lambda, gbuf );
    if (res != greensfunc_op_cpu) mpi_err_printf( "GPU GFs not supported\n" );
    index_t nb = n_spin*n_orb;
    index_t nb2 = POW2(n_spin*n_orb);

    typedef Matrix<gf_complex_t,-1,-1> MatXgfct;
    if (self_energy != nullptr) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for(index_t k=0; k<nktot*2; ++k) {
            index_t kk = k >= nktot ? k - nktot : k;
            Map<MatXcd> S(self_energy+nb2*kk,nb,nb);
            Map<MatXgfct> GF0(gbuf+nb2*k,nb,nb);
            GF0 = GF0.inverse() - S.cast<gf_complex_t>();
            GF0 = GF0.inverse();
        }
    }
}

void tu_loop_t::GF_zero_T_FFT_any( complex128_t* gfft_plus, complex128_t* gfft_conj_plus, gf_complex_t* gbuf ) {

    if (which_fft_greens == tu_fft_algorithm_mpi_new) {
        tu_mpi_gf_fft_and_transpose( mpi_gf, gbuf );
        return;
    }

    const gf_complex_t weight = 1./(double)nktot;
    index_t nb = n_spin*n_orb;
    index_t nb2 = POW2(n_spin*n_orb);

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
        if (which_fft_greens == tu_fft_algorithm_mpi) {
            fft_nons2_backward_mpi->execute( gfft_plus, gfft_plus );
            fft_nons2_backward_mpi->execute( gfft_conj_plus, gfft_conj_plus );
        } else {
            double tick = 0, tock = 0;
            if (extra_timings_enabled) tick = diverge_mpi_wtime();
            fftw_execute_dft( fft_nons2_backward,(fftw_complex*)gfft_plus,(fftw_complex*)gfft_plus );
            fftw_execute_dft( fft_nons2_backward,(fftw_complex*)gfft_conj_plus,(fftw_complex*)gfft_conj_plus );
            if (extra_timings_enabled) tock = diverge_mpi_wtime();
            extra_tvec->at(0) += tock - tick;
        }
    }
}

void tu_loop_t::get_GF_for_self(complex128_t* coarseGF) {
    index_t nb = (n_spin*n_orb);
    index_t nb2 = POW2(n_spin*n_orb);
    #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t b = 0; b < nb; ++b)
    for(index_t o = 0; o < nb; ++o)
    for(index_t k = 0; k < nk; ++k) {
        coarseGF[k+nk*(o+nb*b)] = gf_buf[o+nb*b+nb2*kidxc2f(k,model->nk,model->nkf)];
    }
    if(cFRG_loop) {
        #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
        for(index_t b = 0; b < nb; ++b)
        for(index_t o = 0; o < nb; ++o)
        for(index_t k = 0; k < nk; ++k) {
            coarseGF[k+nk*(o+nb*b)] -= gf_buf_cfrg[o+nb*b+nb2*kidxc2f(k,model->nk,model->nkf)];
        }
    }
}


void tu_loop_t::swap_GF_cfrg() {
    complex128_t* buffer;
    buffer = GF_fft_plus;
    GF_fft_plus = cFRG_GF_fft_plus;
    cFRG_GF_fft_plus = buffer;

    buffer = GF_fft_conj_plus;
    GF_fft_conj_plus = cFRG_GF_fft_conj_plus;
    cFRG_GF_fft_conj_plus = buffer;
}

void tu_loop_t::reconstruct_from_reduced(complex128_t* loop) const {
    index_t n_unique = (index_t)tu_data->unique_mi_l.size();
    #pragma omp for collapse(4) schedule(dynamic,1)
    for(index_t q = 0; q<my_nk;++q)
    for(index_t sr = 0; sr<POW2(n_spin);++sr)
    for(index_t sl = 0; sl<POW2(n_spin);++sl)
    for(index_t m = 0; m<n_unique;++m){
        index_t gmo1 = tu_data->unique_mi_l[m];
        index_t gmo3 = tu_data->unique_mi_r[m];
        complex128_t from = loop[gmo1+n_orbff*(sl +POW2(n_spin)*(gmo3+n_orbff*
                                (sr + POW2(n_spin)*q)))];
        for(index_t l = 1; l < tu_data->lmap_len[m];++l){
            index_t mo1 = tu_data->lmap_l[l+tu_data->lmap_off[m]];
            index_t mo3 = tu_data->lmap_r[l+tu_data->lmap_off[m]];
            loop[mo1+n_orbff*(sl +POW2(n_spin)*(mo3+n_orbff*
                                (sr + POW2(n_spin)*q)))] = from;
        }
    }
}


#ifdef USE_CUDA
void tu_loop_t::ph_loop(complex128_t* L, complex128_t* tmpgpu) {
    if(nktot==1){
        S_loop_static_rs<false>(L);
        if(cFRG_loop) {
            swap_GF_cfrg();
            S_loop_static_rs<false>(L);
            swap_GF_cfrg();
        }
    }else{
        bool do_free = false;
        if(cuda_descr.prop_has_gpu_setup && tmpgpu == nullptr) {
            do_free = true;
            cudaMallocHost(&tmpgpu, my_nk*POW2(n_orbff*POW2(n_spin))*sizeof(complex128_t));
        }
        memset((void*)L,0,sizeof(complex128_t)*my_nk*POW2(n_orbff*POW2(n_spin)));
        if(tu_data->use_reduced_loop) {
            std::thread cpu = std::thread([&](){
                S_loop_static_red<false>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, !cuda_descr.prop_has_gpu_setup);});
            std::thread gpu = std::thread([&](){
                S_ph_loop_static_red_gpu(tmpgpu);});
            cpu.join(); gpu.join();
        } else {
            std::thread cpu = std::thread([&](){
                S_loop_static_simple<false>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, !cuda_descr.prop_has_gpu_setup);});
            std::thread gpu = std::thread([&](){
                S_ph_loop_static_simple_gpu(tmpgpu);});
            cpu.join(); gpu.join();
        }

        if(cFRG_loop) {
            swap_GF_cfrg();
            if(cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop_cfrg_step1(L, tmpgpu);
            if(tu_data->use_reduced_loop) {
                std::thread cpu = std::thread([&](){
                    S_loop_static_red<false,true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, !cuda_descr.prop_has_gpu_setup);});
                std::thread gpu = std::thread([&](){
                    S_ph_loop_static_red_gpu(tmpgpu);});
                cpu.join(); gpu.join();
            } else {
                std::thread cpu = std::thread([&](){
                    S_loop_static_simple<false,true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, !cuda_descr.prop_has_gpu_setup);});
                std::thread gpu = std::thread([&](){
                    S_ph_loop_static_simple_gpu(tmpgpu);});
                cpu.join(); gpu.join();
            }
            swap_GF_cfrg();
            if(cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop_cfrg_step2(L, tmpgpu);
        }else{
            if(cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop(L, tmpgpu);
        }
        if(do_free) cudaFreeHost(tmpgpu);
    }
}

void tu_loop_t::pp_loop(complex128_t* L, complex128_t* tmpgpu) {
    if(nktot==1){
        S_loop_static_rs<true>(L);
        if(cFRG_loop) {
            swap_GF_cfrg();
            S_loop_static_rs<true>(L);
            swap_GF_cfrg();
        }
    }else{
        bool do_free = false;
        if(cuda_descr.prop_has_gpu_setup && tmpgpu == nullptr) {
            do_free = true;
            cudaMallocHost(&tmpgpu, my_nk*POW2(n_orbff*POW2(n_spin))*sizeof(complex128_t));
        }
        memset((void*)L,0,sizeof(complex128_t)*my_nk*POW2(n_orbff*POW2(n_spin)));
        if(tu_data->use_reduced_loop) {
            std::thread cpu = std::thread([&](){
                S_loop_static_red<true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, !cuda_descr.prop_has_gpu_setup);});
            std::thread gpu = std::thread([&](){
                S_pp_loop_static_red_gpu(tmpgpu);});
            cpu.join(); gpu.join();
        }else{
            std::thread cpu = std::thread([&](){
                S_loop_static_simple<true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, !cuda_descr.prop_has_gpu_setup);});
            std::thread gpu = std::thread([&](){
                S_pp_loop_static_simple_gpu(tmpgpu);});
            cpu.join(); gpu.join();
        }
        if(cFRG_loop) {
            swap_GF_cfrg();
            if(cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop_cfrg_step1(L, tmpgpu);
            if(tu_data->use_reduced_loop) {
                std::thread cpu = std::thread([&](){
                    S_loop_static_red<true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, !cuda_descr.prop_has_gpu_setup);});
                std::thread gpu = std::thread([&](){
                    S_pp_loop_static_red_gpu(tmpgpu);});
                cpu.join(); gpu.join();
            }else{
                std::thread cpu = std::thread([&](){
                    S_loop_static_simple<true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, !cuda_descr.prop_has_gpu_setup);});
                std::thread gpu = std::thread([&](){
                    S_pp_loop_static_simple_gpu(tmpgpu);});
                cpu.join(); gpu.join();
            }
            swap_GF_cfrg();
            if(cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop_cfrg_step2(L, tmpgpu);
        }else{
            if(cuda_descr.prop_has_gpu_setup) merge_gpu_cpu_loop(L, tmpgpu);
        }
        if(do_free) cudaFreeHost(tmpgpu);
    }
}

void tu_loop_t::S_ph_loop_static_simple_gpu(complex128_t* L) {
    if(!cuda_descr.prop_memory_set_local && !cuda_descr.prop_memory_set_vertex) {
        mpi_wrn_printf("Memory for GPU calculation was not allocated by vertex \n");
        allocate_GPU_memory();
    }
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]),
                                         (vnk[2]*vnkf[2])};
    ph_bubble_momenta_handler_fft(&cuda_descr, L, GF_fft_plus,GF_fft_conj_plus,
                            n_orb, n_bonds, tu_data->bond_sizes,
                            n_orbff, n_spin, nktot, my_nk, c2fmap.data(),
                            dims,  mi_to_ofrom, 
                            mi_to_oto, mi_to_R);
}

void tu_loop_t::S_pp_loop_static_simple_gpu(complex128_t* L) {
    if(!cuda_descr.prop_memory_set_local && !cuda_descr.prop_memory_set_vertex) {
        mpi_wrn_printf("Memory for GPU calculation was not allocated by vertex \n");
        allocate_GPU_memory();
    }
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]),
                                         (vnk[2]*vnkf[2])};
    pp_bubble_momenta_handler_fft(&cuda_descr, L, GF_fft_plus,GF_fft_conj_plus,
                            n_orb, n_bonds, tu_data->bond_sizes,
                            n_orbff, n_spin, nktot, my_nk, c2fmap.data(),
                            dims, mi_to_ofrom, mi_to_oto, mi_to_R);
}

void tu_loop_t::S_ph_loop_static_red_gpu(complex128_t* L) {
    if(!cuda_descr.prop_memory_set_local && !cuda_descr.prop_memory_set_vertex) {
        mpi_wrn_printf("Memory for GPU calculation was not allocated by vertex \n");
        allocate_GPU_memory();
    }
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]),
                                         (vnk[2]*vnkf[2])};
    ph_bubble_momenta_handler_fft_red(&cuda_descr, L, GF_fft_plus,GF_fft_conj_plus,
                            n_orb, n_orbff, n_spin, nktot, my_nk, c2fmap.data(),
                            dims, mi_to_ofrom, mi_to_oto, mi_to_R, tu_data->unique_mi_l.data(),
                                      tu_data->unique_mi_r.data(),tu_data->unique_mi_l.size());
}

void tu_loop_t::S_pp_loop_static_red_gpu(complex128_t* L) {
    if(!cuda_descr.prop_memory_set_local && !cuda_descr.prop_memory_set_vertex) {
        mpi_wrn_printf("Memory for GPU calculation was not allocated by vertex \n");
        allocate_GPU_memory();
    }
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]),
                                         (vnk[2]*vnkf[2])};
    pp_bubble_momenta_handler_fft_red(&cuda_descr, L, GF_fft_plus,GF_fft_conj_plus,
                            n_orb, n_orbff, n_spin, nktot, my_nk, c2fmap.data(),
                            dims, mi_to_ofrom, mi_to_oto, mi_to_R, tu_data->unique_mi_l.data(),
                                      tu_data->unique_mi_r.data(),tu_data->unique_mi_l.size());
}



void tu_loop_t::merge_gpu_cpu_loop_cfrg_step1(complex128_t* L, complex128_t* tmpgpu) const {
    index_t blocksize = POW2(n_orbff*POW2(n_spin));
    Map<MatXcd> Lmap(L,blocksize,my_nk);
    Map<MatXcd> tmpmap(tmpgpu,my_nk,blocksize);
    Lmap += tmpmap.transpose();
}

void tu_loop_t::merge_gpu_cpu_loop_cfrg_step2(complex128_t* L, complex128_t* tmpgpu) const {
    index_t blocksize = POW2(n_orbff*POW2(n_spin));
    Map<MatXcd> Lmap(L,blocksize,my_nk);
    Map<MatXcd> tmpmap(tmpgpu,my_nk,blocksize);
    Lmap -= tmpmap.transpose();
    if(tu_data->use_reduced_loop) reconstruct_from_reduced(L);
    scale_loop(L, -1./(2.*M_PI), my_nk*POW4(n_spin)*POW2(n_orbff));
}

void tu_loop_t::merge_gpu_cpu_loop(complex128_t* L, complex128_t* tmpgpu) const {
    index_t blocksize = POW2(n_orbff*POW2(n_spin));
    Map<MatXcd> Lmap(L,blocksize,my_nk);
    Map<MatXcd> tmpmap(tmpgpu,my_nk,blocksize);
    Lmap += tmpmap.transpose();
    if(tu_data->use_reduced_loop) reconstruct_from_reduced(L);
    scale_loop(L, -1./(2.*M_PI), my_nk*POW4(n_spin)*POW2(n_orbff));
}

#ifndef CUDA_MEM_ALIGN
#define CUDA_MEM_ALIGN 256
#endif

static index_t ensure_align(index_t numToRound) {

    index_t remainder = numToRound % 256;
    if (remainder == 0)
        return numToRound;

    return numToRound + 256 - remainder;
}

void tu_loop_t::allocate_GPU_memory() {
    if(cuda_descr.prop_memory_set_vertex)
        return;
    cuda_descr.prop_memory_set_local = true;

    index_t prop_memsize[11] = {0};
    const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;

    for(int i = 0; i<tu_num_of_devices;++i) {
        cudaSetDevice(i);
        prop_memsize[0] = ensure_align(sizeof(index_t)*n_orbff*3); // mi_to_R
        prop_memsize[1] = ensure_align(sizeof(index_t)*my_nk); // c_to_f_q
        prop_memsize[2] = ensure_align(sizeof(index_t)*3); // nkdev
        prop_memsize[3] = ensure_align(_GF_size); // GF_fft
        prop_memsize[4] = ensure_align(_GF_size); // GF_fft_conj
        prop_memsize[5] = ensure_align(sizeof(cuDoubleComplex)*nktot); // buf_GG
        prop_memsize[6] = ensure_align(sizeof(cuDoubleComplex)*my_nk); // buf_res 0
        prop_memsize[7] = ensure_align(sizeof(cuDoubleComplex)*my_nk); // buf_res 1
        prop_memsize[8] = ensure_align(sizeof(cuDoubleComplex)*cuda_descr.worksize_fft[i]); // fft_workspace 0

        cudaMalloc(&(cuda_descr.mi_to_R[i]), prop_memsize[0]);
        cudaMalloc(&(cuda_descr.c_to_f_q[i]), prop_memsize[1]);
        cudaMalloc(&(cuda_descr.nkdev[i]), prop_memsize[2]);
        cudaMalloc(&(cuda_descr.GF_fft[i]), prop_memsize[3]);
        cudaMalloc(&(cuda_descr.GF_fft_conj[i]), prop_memsize[4]);
        cudaMalloc(&(cuda_descr.buf_GG[i]), prop_memsize[5]);
        cudaMalloc(&(cuda_descr.buf_res[i]), prop_memsize[6]);
        cudaMalloc(&(cuda_descr.fft_workspace[i]), prop_memsize[8]);
        cudaSetDevice(0);
    }
}

#else


void tu_loop_t::ph_loop(complex128_t* L, complex128_t* tmpgpu) {
    (void)tmpgpu;
    if(nktot==1){
        S_loop_static_rs<false>(L);
    }else{
        bool rescale = !cuda_descr.prop_has_gpu_setup;
        if(cFRG_loop) rescale = false;
        memset((void*)L,0,sizeof(complex128_t)*my_nk*POW2(n_orbff*POW2(n_spin)));
        if(tu_data->use_reduced_loop) {
            S_loop_static_red<false>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, rescale);
        }else{
            S_loop_static_simple<false>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, rescale);
        }
    }
    if(cFRG_loop) {
        swap_GF_cfrg();
        if(nktot==1){
            S_loop_static_rs<false,true>(L);
        }else{
            if(tu_data->use_reduced_loop) {
                S_loop_static_red<false,true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu,
                                    !cuda_descr.prop_has_gpu_setup);
            }else{
                S_loop_static_simple<false,true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu,
                                        !cuda_descr.prop_has_gpu_setup);
            }
        }
        swap_GF_cfrg();
    }
}


void tu_loop_t::pp_loop(complex128_t* L, complex128_t* tmpgpu) {
    (void)tmpgpu;
    if(nktot==1){
        S_loop_static_rs<true>(L);
    }else{
        bool rescale = !cuda_descr.prop_has_gpu_setup;
        if(cFRG_loop) rescale = false;
        memset((void*)L,0,sizeof(complex128_t)*my_nk*POW2(n_orbff*POW2(n_spin)));
        if(tu_data->use_reduced_loop) {
            S_loop_static_red<true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, rescale);
        }else{
            S_loop_static_simple<true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu, rescale);
        }
    }
    if(cFRG_loop) {
        swap_GF_cfrg();
        if(nktot==1){
            S_loop_static_rs<true,true>(L);
        }else{
            if(tu_data->use_reduced_loop) {
                S_loop_static_red<true,true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu,
                                    !cuda_descr.prop_has_gpu_setup);
            }else{
                S_loop_static_simple<true,true>(L, cuda_descr._tasks_cpu, cuda_descr._off_cpu,
                                        !cuda_descr.prop_has_gpu_setup);
            }
        }
        swap_GF_cfrg();
    }
}

void tu_loop_t::allocate_GPU_memory(){;}

void tu_loop_t::S_ph_loop_static_simple_gpu(complex128_t* L){(void)L;}

void tu_loop_t::S_pp_loop_static_simple_gpu(complex128_t* L){(void)L;}

void tu_loop_t::S_ph_loop_static_red_gpu(complex128_t* L){(void)L;}

void tu_loop_t::S_pp_loop_static_red_gpu(complex128_t* L){(void)L;}

void tu_loop_t::merge_gpu_cpu_loop(complex128_t* L, complex128_t* tmpgpu) const {(void)L; (void)tmpgpu;}
void tu_loop_t::merge_gpu_cpu_loop_cfrg_step1(complex128_t* L, complex128_t* tmpgpu) const {(void)L; (void)tmpgpu;}
void tu_loop_t::merge_gpu_cpu_loop_cfrg_step2(complex128_t* L, complex128_t* tmpgpu) const {(void)L; (void)tmpgpu;}

#endif


