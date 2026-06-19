/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../misc/mpi_functions.h"
#include "../misc/bitary.h"
#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned propagator_cpu_mpi_planning_patience;

typedef struct diverge_model_t diverge_model_t;

typedef struct {
    index_t s1, s2, s3, s4, m;
} ptask_t;

typedef struct {
    index_t k;
    index_t q;
    int source_rank;
    int target_rank;
} kmap_t;

enum {
    tu_mpi_loop_timing_prepare = 0,
    tu_mpi_loop_timing_reorder,
    tu_mpi_loop_timing_transpose,
    tu_mpi_loop_timing_fft,
    tu_mpi_loop_timing_insert,
    tu_mpi_loop_timing_mpibufprep,
    tu_mpi_loop_timing_comm,
    tu_mpi_loop_timing_insert_missing,
    tu_mpi_loop_timing_postproc,
    TU_MPI_LOOP_TIMING_NUM
};

typedef struct {
    int rank;
    int size;
    MPI_Comm comm;

    index_t vnk[3];
    index_t vnkf[3];
    index_t chunksize_max;

    index_t dims[3];

    index_t kx_count, kx_displ;
    complex128_t* ggm_buf;
    complex128_t* ggp_buf;

    index_t alloc;

    fftw_plan fft_forward,
              fft_backward,
              fft_transpose;

    int* kx_counts;
    int* kx_displs;

    // internal info
    int* sendcounts;
    int* senddispls;
    int* recvcounts;
    int* recvdispls;
    kmap_t* kcomms;
    index_t kcomms_size;
    kmap_t* kc_send;
    index_t kc_send_count;
    kmap_t* kc_recv;
    index_t kc_recv_count;

    int *xsendc, *xsendd,
        *xrecvc, *xrecvd;

    double timings[TU_MPI_LOOP_TIMING_NUM];
    int enable_timings;

    // boolean array that tracks whether c2fmap content is local
    cbytes_t* c2fmap_local;

    // loop info
    index_t my_nk;
    const index_t* c2fmap;

    index_t n_spin;
    index_t n_orb;
    index_t n_orbff;
    const index_t* unique_mi_l;
    index_t unique_mi_l_size;
    const index_t* unique_mi_r;
    const index_t* mi_to_ofrom;
    const index_t* mi_to_oto;
    const index_t* mi_to_R;

    complex128_t** p_GF_fft_plus;
    complex128_t** p_GF_fft_conj_plus;
    void* tu_loop_handle;

    diverge_model_t* model;
} tu_mpi_loop_t;

tu_mpi_loop_t* tu_mpi_loop_init( const index_t nk[3], const index_t nkf[3],
        index_t chunksize_max, const index_t* c2fmap, index_t my_nk,
        index_t n_spin, index_t n_orb, index_t n_orbff,
        const index_t* unique_mi_l, index_t unique_mi_l_size,
        const index_t* unique_mi_r, const index_t* mi_to_ofrom,
        const index_t* mi_to_oto, const index_t* mi_to_R,
        complex128_t** GF_fft_plus, complex128_t** GF_fft_conj_plus,
        void* handle, diverge_model_t* model );
void tu_mpi_loop_calc( tu_mpi_loop_t* t, complex128_t* loop,
        index_t ntasks, index_t taskoff, bool rescale, bool pp, bool simple,
        bool subtract );
void tu_mpi_loop_destroy( tu_mpi_loop_t* t );

enum {
    tu_mpi_gf_timing_trans = 0,
    tu_mpi_gf_timing_ffttrans,
    tu_mpi_gf_timing_fft,
    tu_mpi_gf_timing_comm,
    TU_MPI_GF_TIMING_NUM
};

#ifdef USE_GF_FLOATS
#define GF_FFTW( name ) fftwf_##name
#else
#define GF_FFTW( name ) fftw_##name
#endif

typedef struct {
    index_t nk[3];
    index_t nkf[3];
    index_t nb;

    index_t dims[3];
    index_t kx_count, kx_displ;
    GF_FFTW(plan) fft;
    GF_FFTW(plan) transpose[2]; // gf_buf[0] and gf_buf[nk*nb*nb]
    gf_complex_t* fft_buf;

    double timings[TU_MPI_GF_TIMING_NUM];
    int enable_timings;

    MPI_Comm comm;
    int size;
    int rank;
    int* counts;
    int* displs;

    diverge_model_t* model;
} tu_mpi_gf_t;

tu_mpi_gf_t* tu_mpi_gf_init( const index_t nk[3], const index_t nkf[3], index_t nb, gf_complex_t* gf_buf,
        diverge_model_t* model );
void tu_mpi_gf_fft_and_transpose( tu_mpi_gf_t* t, gf_complex_t* gf_buf );
void tu_mpi_gf_destroy( tu_mpi_gf_t* t );

#ifdef __cplusplus
}
#endif

