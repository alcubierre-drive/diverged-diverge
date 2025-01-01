/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_symmetrize.h"
#include "diverge_interface.h"

#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct selfenergy_t selfenergy_t;
struct selfenergy_t {
    fftw_plan ifft_self_oop;
    fftw_plan fft_self_oop;
    fftw_plan ifft_self_ip;
    fftw_plan fft_self_ip;

    fftw_plan fft_self_oop_gfmesh;

    fftw_plan ifft_fullself_ip;
    fftw_plan fft_fullself_ip;

    fftw_plan ifft_vert_oop;
    fftw_plan fft_vert_oop;
    fftw_plan ifft_vert_ip;
    fftw_plan fft_vert_ip;

    //inputs
    diverge_model_t* model;
    const tu_data_t* tu_data;
    index_t n_orb;
    index_t n_spin;
    index_t ns;
    index_t n_orbff;
    index_t n_bonds;
    index_t nk;
    index_t nkf;
    index_t nktot;
    index_t my_nk;
    index_t my_nk_off;
    index_t nff;
    bool SU2;

    complex128_t* ffkmesh;

    const index_t* ob_to_orbff;
    const claV3i_t* existing_ff;
    index_t existing_ff_sz;

    const tu_formfactor_t* tu_ff;
    const index_t* bond_sizes;
    const index_t* bond_offsets;
    const symmstruct_t* symm;
    bool P_flow;
    bool C_flow;
    bool D_flow;

    complex128_t* reco_helper;
    complex128_t* interpol_helper;

    complex128_t* ff_helper;
    complex128_t* greensfkt_helper;
};

selfenergy_t* selfenergy_init( void* vertex_ptr );
void selfenergy_free( selfenergy_t* s );

void selfenergy_flow( selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, const complex128_t* C_in,
        const complex128_t* D_in, const complex128_t* GF,
        const double prefac_sub );
void selfenergy_fft_interpolate( selfenergy_t* s, complex128_t* self_en );

#ifdef __cplusplus
}
#endif
