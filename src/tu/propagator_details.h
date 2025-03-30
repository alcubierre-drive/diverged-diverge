#pragma once

#include "propagator.h"
#include "propagator_cpu_mpi.h"
#include "propagator_cpu_mpi_fft.h"
#include "propagator_gpu_mpi.h"
#include "cuda_init.h"
#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct greensfunc_coarse_t greensfunc_coarse_t;

struct tu_prop_t {
    gf_complex_t* gf_buf;
    diverge_model_t* model;
    const tu_data_t* tu_data;

    index_t* mi_to_ofrom;
    index_t* mi_to_oto;
    index_t* mi_to_R;

    index_t n_orbff;
    index_t n_orb;
    index_t n_bonds;
    index_t n_spin;
    index_t nk;
    index_t nkf;
    index_t nktot;
    index_t nkibz;
    index_t my_nk;
    index_t my_nk_off;
    bool SU2;
    double* bos_frq;

    index_t* c2fmap;
    index_t c2fmap_sz;
    cuda_prop_descr_t cuda_descr;

    bool cFRG_loop;

    // only there to allow HF-like simulations, i.e., patching the GF generator
    // and *not* adding the self-energy to the GF
    bool skip_selfenergy;

    tu_fft_algorithm_t which_fft_simple, which_fft_red, which_fft_greens;
    index_t chunksize_max;

    // tu_fft_algorithm_greedy
    // greens
    fftw_plan fft_nons2_backward;
    // simple & red
    fftw_plan fft_single_backward,
              fft_single_forward;

    // to put the GF on a coarse mesh using FFTs
    greensfunc_coarse_t* coarsify;

    complex128_t **buffer_ggm, **buffer_ggp;
    index_t buffer_ggmp_sz;

    // tu_fft_algorithm_split
    // simple & red
    fftw_plan fft_many_backward,
              fft_many_forward;
    complex128_t *buffer_ggm_many,
                 *buffer_ggp_many;

    // tu_fft_algorithm_mpi
    // simple & red
    fftw_mpi_plan_t *fft_many_backward_mpi,
                    *fft_many_forward_mpi;
    // transpose a matrix (size: many)
    fftw_plan fft_T_to,
              fft_T_from;
    // greens
    fftw_mpi_plan_t *fft_nons2_backward_mpi;

    // new MPI algorithms
    tu_mpi_loop_t* mpi_loop;
    tu_mpi_gf_t* mpi_gf;

    bool do_GF_fft_allocs;
    complex128_t* GF_fft_plus;
    complex128_t* GF_fft_conj_plus;

    gf_complex_t* gf_buf_cfrg;
    complex128_t* cFRG_GF_fft_plus;
    complex128_t* cFRG_GF_fft_conj_plus;

    // GPU MPI algorithm (very memory intensive)
    bool has_memory_resized;
    int32_t tasks_prop[32];
    int32_t off_prop[32];
    int32_t tasks_prop_mpi[32];
    int32_t off_prop_mpi[32];
    bool has_tasks_off_saved;
    index_t* c2fmap_full;
    index_t c2fmap_full_sz;
    tu_prop_gpu_mpi_dist_t* gpu_mpi_dist_pp;
    tu_prop_gpu_mpi_dist_t* gpu_mpi_dist_ph;
    // those are shared between the two pp/ph
    pack_t* gpu_mpi_dist_recv_pack;
    index_t gpu_mpi_dist_recv_pack_sz;

    // only CPU timings
    diverge_timing_t extra_timings[64];
    int n_extra_timings;
    bool extra_timings_enabled;
};

/* ================== *
 * extended interface *
 * ================== */

// these are in propagator_cpu.c
void S_loop_static_rs( const tu_prop_t* t, bool pp, bool subract/*=false*/,
        complex128_t* L );
void S_loop_static_simple( tu_prop_t* t, bool pp, bool subtract/*=false*/,
        complex128_t* L, index_t ntasks/*=-1*/, index_t taskoff/*=0*/,
        bool rescale/*=true*/ );
void S_loop_static_red( tu_prop_t* t, bool pp, bool subtract/*=false*/,
        complex128_t* L, index_t ntasks/*=-1*/, index_t taskoff/*=0*/,
        bool rescale/*=true*/ );

// this is in propagator_gpu_mpi_wrap.h
void S_loop_static_red_gpu_mpi( tu_prop_t* t, complex128_t* L, bool pp );

// these are in propagator_details.c
void cpu_fft_algorithm_init( tu_prop_t* t, tu_fft_algorithm_t greens,
        tu_fft_algorithm_t simple, tu_fft_algorithm_t red );
void cpu_fft_algorithm_cleanup( tu_prop_t* t );
void reconstruct_from_reduced( const tu_prop_t* t, complex128_t* L );
void swap_GF_cfrg( tu_prop_t* t );
void GF_zero_T_any( tu_prop_t* t, greensfunc_generator_t gf, complex128_t lambda,
        complex128_t* self_energy, gf_complex_t* buf );
void GF_zero_T_FFT_any( tu_prop_t* t, complex128_t* gfft_plus,
        complex128_t* gfft_conj_plus, gf_complex_t* gbuf );

// these are in propagator_gpu.c
void merge_gpu_cpu_loop( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu );
void merge_gpu_cpu_loop_cfrg_step1( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu );
void merge_gpu_cpu_loop_cfrg_step2( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu );
void allocate_GPU_memory( tu_prop_t* t );
void S_ph_loop_static_simple_gpu( tu_prop_t* t, complex128_t* L );
void S_pp_loop_static_simple_gpu( tu_prop_t* t, complex128_t* L );
void S_ph_loop_static_red_gpu( tu_prop_t* t, complex128_t* L );
void S_pp_loop_static_red_gpu( tu_prop_t* t, complex128_t* L );

static inline void scale_loop(complex128_t* const loop, const double pref, const index_t len) {
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t idx=0; idx<len; ++idx) loop[idx] *= pref;
}

// these are in propagator_gpu.cu
#ifdef USE_CUDA
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
#endif // USE_CUDA

#ifdef __cplusplus
}
#endif
