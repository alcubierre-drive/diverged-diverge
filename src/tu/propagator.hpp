/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "../misc/bitary.h"

#include "diverge_interface.hpp"
#include "global.hpp"
#include "cuda_init.hpp"

#include "propagator_cpu_mpi.h"
#include "propagator_cpu_mpi_fft.h"
#include "propagator_gpu_mpi.h"

#include <fftw3.h>
#include <thread>
#include <string>
using std::string;

inline void scale_loop(complex128_t* const loop, const double pref, const index_t len) {
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t idx = 0; idx < len; ++idx) {
        ((double*)(loop+idx))[0] *= pref;
        ((double*)(loop+idx))[1] *= pref;
    }
}

void ph_bubble_momenta_handler_fft(cuda_prop_descr_t* cp,  complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb, const index_t n_bonds, const index_t* const bond_sizes,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* ofrom, const index_t* oto,
    const index_t* const mi_to_Rin);
void pp_bubble_momenta_handler_fft(cuda_prop_descr_t* cp, complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,const index_t n_bonds, const index_t* const bond_sizes,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* ofrom, const index_t* oto,
    const index_t* const mi_to_Rin);

void pp_bubble_momenta_handler_fft_red(cuda_prop_descr_t* cp, complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin,const index_t* const umi_l,const index_t* const umi_r,
    const index_t n_unique_pairs, cbyte_t* L_touched_map );
void ph_bubble_momenta_handler_fft_red(cuda_prop_descr_t* cp, complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin,const index_t* const umi_l,const index_t* const umi_r,
    const index_t n_unique_pairs, cbyte_t* L_touched_map );

class tu_loop_t {
public:
    tu_loop_t(diverge_model_t*);
    ~tu_loop_t();

    void GF_zero_T( complex128_t lambda, complex128_t* self_energy = nullptr );
    void ph_loop( complex128_t* L, complex128_t* tmpgpu = nullptr );
    void pp_loop( complex128_t* L, complex128_t* tmpgpu = nullptr );

    tu_loop_t() = delete;
    tu_loop_t(const tu_loop_t&) = delete;
    tu_loop_t& operator=(const tu_loop_t&) = delete;
    tu_loop_t(tu_loop_t&&) = delete;
    tu_loop_t& operator=(tu_loop_t&&) = delete;


    gf_complex_t* gf_buf;
    template<bool pp, bool subract = false>
    void S_loop_static_rs(complex128_t* L) const;

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

    // not quite tested...
    void S_loop_static_red_gpu_mpi( complex128_t* L, bool pp );

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
    const index_t nkibz;
    const index_t my_nk;
    const index_t my_nk_off;
    const bool SU2;
    const double* bos_frq;

    void get_GF_for_self(complex128_t* coarseGF);

    void reconstruct_from_reduced(complex128_t* L) const;

    vector<index_t> c2fmap;
    cuda_prop_descr_t cuda_descr;

    // only CPU timings
    vector<double> extra_timings( void );
    vector<string> extra_timings_descr( void );
    void extra_timings_set( bool enabled );

private:
    bool extra_timings_enabled;
    vector<double>* extra_tvec;
    bool cFRG_loop;

    tu_fft_algorithm_t which_fft_simple, which_fft_red, which_fft_greens;
    index_t chunksize_max;

    // tu_fft_algorithm_greedy
    // greens
    fftw_plan fft_nons2_backward = NULL;
    // simple & red
    fftw_plan fft_single_backward = NULL,
              fft_single_forward = NULL;
    vector<complex128_t*> buffer_ggm,
                          buffer_ggp;

    // tu_fft_algorithm_split
    // simple & red
    fftw_plan fft_many_backward = NULL,
              fft_many_forward = NULL;
    complex128_t *buffer_ggm_many = NULL,
                 *buffer_ggp_many = NULL;

    // tu_fft_algorithm_mpi
    // simple & red
    fftw_mpi_plan_t *fft_many_backward_mpi = NULL,
                    *fft_many_forward_mpi = NULL;
    // transpose a matrix (size: many)
    fftw_plan fft_T_to = NULL,
              fft_T_from = NULL;
    // greens
    fftw_mpi_plan_t *fft_nons2_backward_mpi = NULL;

    void cpu_fft_algorithm_init( tu_fft_algorithm_t greens, tu_fft_algorithm_t simple, tu_fft_algorithm_t red );
    void cpu_fft_algorithm_cleanup( void );

    // new MPI algorithms
    tu_mpi_loop_t* mpi_loop = NULL;
    tu_mpi_gf_t* mpi_gf = NULL;

    bool do_GF_fft_allocs = true;
    complex128_t* GF_fft_plus;
    complex128_t* GF_fft_conj_plus;

    gf_complex_t* gf_buf_cfrg = nullptr;
    complex128_t* cFRG_GF_fft_plus = nullptr;
    complex128_t* cFRG_GF_fft_conj_plus = nullptr;

    // GPU MPI algorithm (very memory intensive)
    bool has_memory_resized = false;
    int32_t tasks_prop[32];
    int32_t off_prop[32];
    int32_t tasks_prop_mpi[32];
    int32_t off_prop_mpi[32];
    bool has_tasks_off_saved = false;
    vector<index_t> c2fmap_full;
    tu_prop_gpu_mpi_dist_t* gpu_mpi_dist_pp = NULL;
    tu_prop_gpu_mpi_dist_t* gpu_mpi_dist_ph = NULL;
    // those are shared between the two pp/ph
    pack_t* gpu_mpi_dist_recv_pack = NULL;
    index_t gpu_mpi_dist_recv_pack_sz = 0;

    void allocate_GPU_memory();
    void swap_GF_cfrg();

    void GF_zero_T_any( greensfunc_generator_t gf, complex128_t lambda, complex128_t* self_energy, gf_complex_t* buf );
    void GF_zero_T_FFT_any( complex128_t* gfft_plus, complex128_t* gfft_conj_plus, gf_complex_t* gbuf );
};


#include "propagator_cpu.hpp"
