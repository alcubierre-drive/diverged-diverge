/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once
#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "diverge_interface.hpp"
#include "global.hpp"
#include "cuda_init.hpp"
#include <fftw3.h>
#include <thread>

inline void scale_loop(complex128_t* const loop, const double pref, const index_t len) {
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t idx = 0; idx < len; ++idx) {
        ((double*)(loop+idx))[0] *= pref;
        ((double*)(loop+idx))[1] *= pref;
    }
}

void ph_bubble_momenta_handler_fft(cuda_prop_descr_t* cp, complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin);
void pp_bubble_momenta_handler_fft(cuda_prop_descr_t* cp, complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin);

void pp_bubble_momenta_handler_fft_red(cuda_prop_descr_t* cp, complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin,const index_t* const umi_l,const index_t* const umi_r,
    const index_t n_unique_pairs);
void ph_bubble_momenta_handler_fft_red(cuda_prop_descr_t* cp, complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin,const index_t* const umi_l,const index_t* const umi_r,
    const index_t n_unique_pairs);

class tu_loop_t {
public:
    tu_loop_t(diverge_model_t*);
    ~tu_loop_t();

    void GF_zero_T(const complex128_t lambda, complex128_t* self_energy = nullptr);
    void ph_loop(complex128_t* L, complex128_t* tmpgpu = nullptr);
    void pp_loop(complex128_t* L, complex128_t* tmpgpu = nullptr);

    DELETE_CONSTRUCTOR_RULE_OF_FIVE(tu_loop_t);

    gf_complex_t* gf_buf;
    template<bool subract = false>
    void S_ph_loop_static_rs(complex128_t* L) const;
    template<bool subract = false>
    void S_pp_loop_static_rs(complex128_t* L) const;

    template<bool pp, bool subract = false>
    void S_loop_static_simple(complex128_t* L, index_t ntasks = -1,
                              index_t taskoff = 0, bool rescale = true) const;
    template<bool pp, bool subract = false>
    void S_loop_static_red(complex128_t* L, index_t ntasks = -1,
                              index_t taskoff = 0, bool rescale = true) const;

    void S_ph_loop_static_simple_gpu(complex128_t* L);
    void S_pp_loop_static_simple_gpu(complex128_t* L);
    void S_ph_loop_static_red_gpu(complex128_t* L);
    void S_pp_loop_static_red_gpu(complex128_t* L);

    void merge_gpu_cpu_loop(complex128_t* L, complex128_t* tmpgpu) const;

    void merge_gpu_cpu_loop_cfrg_step1(complex128_t* L, complex128_t* tmpgpu) const;
    void merge_gpu_cpu_loop_cfrg_step2(complex128_t* L, complex128_t* tmpgpu) const;

    diverge_model_t* model;
    const tu_data_t* const tu_data;

    index_t* mi_to_ofrom;
    index_t* mi_to_oto;
    index_t* mi_to_R;

    const index_t n_orbff;
    const index_t n_orb;
    const index_t n_bonds;
    const index_t n_spin;
    const index_t nk;
    const index_t nkf;
    const index_t nktot;
    const index_t my_nk;
    const index_t my_nk_off;
    const index_t n_frq;
    const bool SU2;
    const double* bos_frq;

    void get_GF_for_self(complex128_t* coarseGF);

    void reconstruct_from_reduced(complex128_t* L) const;

    vector<index_t> c2fmap;
    cuda_prop_descr_t cuda_descr;


private:
    bool cFRG_loop;

    fftw_plan fft_nons2_backward,
              fft_single_backward,
              fft_single_forward;

    vector<complex128_t*> buffer_ggm;
    vector<complex128_t*> buffer_ggp;
    complex128_t* GF_fft_plus;
    complex128_t* GF_fft_conj_plus;

    gf_complex_t* gf_buf_cfrg = nullptr;
    complex128_t* cFRG_GF_fft_plus = nullptr;
    complex128_t* cFRG_GF_fft_conj_plus = nullptr;
    void allocate_GPU_memory();
    void swap_GF_cfrg();
};


#include "propagator_cpu.hpp"
