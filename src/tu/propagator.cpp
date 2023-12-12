/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "propagator.hpp"
#include "cuda_init.hpp"
#include "../misc/batched_eigen.h"
#include "../misc/init_internal_libs.h"
#include <iostream>

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
    n_frq{tu_data->n_frq},
    SU2{model->SU2 > 0},
    cFRG_loop{model->gproj!=NULL}
{
    index_t task_spin = POW4(n_spin);
    index_t tasks_dist = tu_data->use_reduced_loop ? (index_t)tu_data->unique_mi_l.size() : POW2(n_orbff);
    index_t nkff[3];
    for (int i=0; i<3; ++i) nkff[i] = model->nk[i] * model->nkf[i];
    cuda_setup_prop_descr(&cuda_descr,task_spin,tasks_dist, nkff,tu_data->offload_cpu);

    GF_fft_plus = (complex128_t*)fftw_alloc_complex(nktot* POW2(n_spin*n_orb));
    GF_fft_conj_plus = (complex128_t*)fftw_alloc_complex(nktot* POW2(n_spin*n_orb));
    if(cFRG_loop) {
        gf_buf_cfrg = (gf_complex_t*)malloc(2 * nktot* POW2(n_spin*n_orb)*sizeof(gf_complex_t));
        cFRG_GF_fft_plus = (complex128_t*)fftw_alloc_complex(nktot* POW2(n_spin*n_orb));
        cFRG_GF_fft_conj_plus = (complex128_t*)fftw_alloc_complex(nktot* POW2(n_spin*n_orb));
    }
    #ifdef USE_CUDA
    CUDA_CHECK(cudaHostRegister(GF_fft_plus,nktot* POW2(n_spin*n_orb)*sizeof(fftw_complex), cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(GF_fft_conj_plus,nktot* POW2(n_spin*n_orb)*sizeof(fftw_complex), cudaHostRegisterDefault));
    if(cFRG_loop) {
        CUDA_CHECK(cudaHostRegister(cFRG_GF_fft_plus,nktot* POW2(n_spin*n_orb)*sizeof(fftw_complex), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(cFRG_GF_fft_conj_plus,nktot* POW2(n_spin*n_orb)*sizeof(fftw_complex), cudaHostRegisterDefault));
    }
    #endif

    int fftwnthr = diverge_fftw_num_threads();
    fftw_plan_with_nthreads(1);
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]),
                                         (int)(vnk[2]*vnkf[2])};
    if(nktot>1) {
        buffer_ggm.resize(diverge_omp_num_threads());
        buffer_ggp.resize(diverge_omp_num_threads());
        for (int i=0; i<diverge_omp_num_threads(); ++i) {
            buffer_ggm[i] = (complex128_t*)fftw_alloc_complex(nktot);
            buffer_ggp[i] = (complex128_t*)fftw_alloc_complex(nktot);
        }
        fft_single_backward = fftw_plan_many_dft( 3, dims, 1,
                (fftw_complex*)buffer_ggm[0], dims, 1, nktot,
                (fftw_complex*)buffer_ggp[0], dims, 1, nktot,
                FFTW_BACKWARD, FFTW_MEASURE );
        fft_single_forward = fftw_plan_many_dft( 3, dims, 1,
                (fftw_complex*)buffer_ggm[0], dims, 1, nktot,
                (fftw_complex*)buffer_ggp[0], dims, 1, nktot,
                FFTW_FORWARD, FFTW_MEASURE );
        fft_nons2_backward = fftw_plan_many_dft( 3, dims, POW2(n_spin*n_orb),
                (fftw_complex*)GF_fft_plus, dims, 1, nktot,
                (fftw_complex*)GF_fft_plus, dims, 1, nktot,
                FFTW_BACKWARD, FFTW_MEASURE );
    }
    fftw_plan_with_nthreads(fftwnthr);
    c2fmap.resize(my_nk);
    if(tu_data->symm->use_symmetries) {
        for(index_t i = 0; i < my_nk;++i)
            c2fmap[i] = kidxc2f(tu_data->symm->idx_ibz_in_fullmesh[i+my_nk_off],
                                model->nk,model->nkf);
    }else{
        for(index_t i = 0; i < my_nk;++i)
            c2fmap[i] = kidxc2f(i+my_nk_off,model->nk,model->nkf);
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
            cudaFree(cuda_descr.buf_GG[i][0]);
            cudaFree(cuda_descr.buf_GG[i][1]);
            cudaFree(cuda_descr.buf_res[i][0]);
            cudaFree(cuda_descr.buf_res[i][1]);
            cudaFree(cuda_descr.fft_workspace[i][0]);
            cudaFree(cuda_descr.fft_workspace[i][1]);
        }
    }
    CUDA_CHECK(cudaHostUnregister(GF_fft_plus));
    CUDA_CHECK(cudaHostUnregister(GF_fft_conj_plus));

    if(cFRG_loop) {
        CUDA_CHECK(cudaHostUnregister(cFRG_GF_fft_plus));
        CUDA_CHECK(cudaHostUnregister(cFRG_GF_fft_conj_plus));
    }
    #endif

    fftw_free(GF_fft_plus);
    fftw_free(GF_fft_conj_plus);
    if(cFRG_loop) {
        free(gf_buf_cfrg);
        fftw_free(cFRG_GF_fft_plus);
        fftw_free(cFRG_GF_fft_conj_plus);
    }
    cuda_delete_prop_descr(&cuda_descr);
    if(nktot>1) {
        fftw_destroy_plan(fft_single_forward);
        fftw_destroy_plan(fft_single_backward);
        fftw_destroy_plan(fft_nons2_backward);
        for(int i = 0; i < diverge_omp_num_threads();++i) {
            fftw_free(buffer_ggm[i]);
            fftw_free(buffer_ggp[i]);
        }
    }
}


void tu_loop_t::GF_zero_T(const complex128_t lambda, complex128_t* self_energy) {
    greensfunc_op_t res = model->gfill(model, lambda, gf_buf);
    if(res != greensfunc_op_cpu)
        mpi_err_printf("Not supported");
    index_t nb = n_spin*n_orb;
    index_t nb2 = POW2(n_spin*n_orb);

    typedef Matrix<gf_complex_t,-1,-1> MatXgfct;
    if(self_energy != nullptr) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for(index_t k=0; k<nktot*2; ++k) {
            index_t kk = k >= nktot ? k - nktot : k;
            Map<MatXcd> S(self_energy+nb2*kk,nb,nb);
            Map<MatXgfct> GF0(gf_buf+nb2*k,nb,nb);
            GF0 = GF0.inverse() - S.cast<gf_complex_t>();
            GF0 = GF0.inverse();
        }
    }

    const gf_complex_t weight = 1./(double)nktot;
    #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t b = 0; b < nb; ++b)
    for(index_t o = 0; o < nb; ++o)
    for(index_t k = 0; k < nktot; ++k) {
        GF_fft_plus[k+nktot*(b+nb*o)] = gf_buf[o+nb*b+nb2*k]*weight;
    }

    #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t b = 0; b < nb; ++b)
    for(index_t o = 0; o < nb; ++o)
    for(index_t k = 0; k < nktot; ++k) {
        GF_fft_conj_plus[k+nktot*(b+nb*o)] = gf_buf[o+nb*b+nb2*k + nb2*nktot]*weight;
    }
    if(nktot > 1) {
        fftw_execute_dft( fft_nons2_backward, (fftw_complex*)GF_fft_plus, (fftw_complex*)GF_fft_plus );
        fftw_execute_dft( fft_nons2_backward, (fftw_complex*)GF_fft_conj_plus, (fftw_complex*)GF_fft_conj_plus );
    }
    if(cFRG_loop) {
        greensfunc_op_t res = model->gproj(model, lambda, gf_buf_cfrg);
        if(res != greensfunc_op_cpu)
            mpi_err_printf("Not supported");

        if(self_energy != nullptr) {
            #pragma omp parallel for num_threads(diverge_omp_num_threads())
            for(index_t k=0; k<nktot*2; ++k) {
                index_t kk = k >= nktot ? k - nktot : k;
                Map<MatXcd> S(self_energy+nb2*kk,nb,nb);
                Map<MatXgfct> GF0(gf_buf_cfrg+nb2*k,nb,nb);
                GF0 = GF0.inverse() - S.cast<gf_complex_t>();
                GF0 = GF0.inverse();
            }
        }

        #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
        for(index_t b = 0; b < nb; ++b)
        for(index_t o = 0; o < nb; ++o)
        for(index_t k = 0; k < nktot; ++k) {
            cFRG_GF_fft_plus[k+nktot*(b+nb*o)] = gf_buf_cfrg[o+nb*b+nb2*k]*weight;
        }

        #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
        for(index_t b = 0; b < nb; ++b)
        for(index_t o = 0; o < nb; ++o)
        for(index_t k = 0; k < nktot; ++k) {
            cFRG_GF_fft_conj_plus[k+nktot*(b+nb*o)] = gf_buf_cfrg[o+nb*b+nb2*k + nb2*nktot]*weight;
        }

        if(nktot > 1) {
            fftw_execute_dft( fft_nons2_backward, (fftw_complex*)cFRG_GF_fft_plus, (fftw_complex*)cFRG_GF_fft_plus );
            fftw_execute_dft( fft_nons2_backward, (fftw_complex*)cFRG_GF_fft_conj_plus, (fftw_complex*)cFRG_GF_fft_conj_plus );
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
                            n_orb, n_orbff, n_spin, nktot, my_nk, c2fmap.data(),
                            dims, mi_to_ofrom, mi_to_oto, mi_to_R);
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
                            n_orb, n_orbff, n_spin, nktot, my_nk, c2fmap.data(),
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
    scale_loop(L, -1./(2.*M_PI), my_nk*POW4(n_spin)*POW2(n_orbff)*n_frq);
}

void tu_loop_t::merge_gpu_cpu_loop(complex128_t* L, complex128_t* tmpgpu) const {
    index_t blocksize = POW2(n_orbff*POW2(n_spin));
    Map<MatXcd> Lmap(L,blocksize,my_nk);
    Map<MatXcd> tmpmap(tmpgpu,my_nk,blocksize);
    Lmap += tmpmap.transpose();
    if(tu_data->use_reduced_loop) reconstruct_from_reduced(L);
    scale_loop(L, -1./(2.*M_PI), my_nk*POW4(n_spin)*POW2(n_orbff)*n_frq);
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
        prop_memsize[0] = ensure_align(sizeof(index_t)*n_orbff*3); // mi_to_R
        prop_memsize[1] = ensure_align(sizeof(index_t)*my_nk); // c_to_f_q
        prop_memsize[2] = ensure_align(sizeof(index_t)*3); // nkdev
        prop_memsize[3] = ensure_align(_GF_size); // GF_fft
        prop_memsize[4] = ensure_align(_GF_size); // GF_fft_conj
        prop_memsize[5] = ensure_align(sizeof(cuDoubleComplex)*nktot); // buf_GG 0
        prop_memsize[6] = ensure_align(sizeof(cuDoubleComplex)*nktot); // buf_GG 1
        prop_memsize[7] = ensure_align(sizeof(cuDoubleComplex)*my_nk); // buf_res 0
        prop_memsize[8] = ensure_align(sizeof(cuDoubleComplex)*my_nk); // buf_res 1
        prop_memsize[9] = ensure_align(sizeof(cuDoubleComplex)*cuda_descr.worksize_fft[i]); // fft_workspace 0
        prop_memsize[10] = ensure_align(sizeof(cuDoubleComplex)*cuda_descr.worksize_fft[i]); // fft_workspace 1

        cudaMalloc(&(cuda_descr.mi_to_R[i]), prop_memsize[0]);
        cudaMalloc(&(cuda_descr.c_to_f_q[i]), prop_memsize[1]);
        cudaMalloc(&(cuda_descr.nkdev[i]), prop_memsize[2]);
        cudaMalloc(&(cuda_descr.GF_fft[i]), prop_memsize[3]);
        cudaMalloc(&(cuda_descr.GF_fft_conj[i]), prop_memsize[4]);
        cudaMalloc(&(cuda_descr.buf_GG[i][0]), prop_memsize[5]);
        cudaMalloc(&(cuda_descr.buf_GG[i][1]), prop_memsize[6]);
        cudaMalloc(&(cuda_descr.buf_res[i][0]), prop_memsize[7]);
        cudaMalloc(&(cuda_descr.buf_res[i][1]), prop_memsize[8]);
        cudaMalloc(&(cuda_descr.fft_workspace[i][0]), prop_memsize[9]);
        cudaMalloc(&(cuda_descr.fft_workspace[i][1]), prop_memsize[10]);
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


