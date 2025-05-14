/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_phonons.h"
#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO the structure may be public on this level, but it would be better to
// just expose a tiny little bit of it via getters
typedef struct tu_data_phonons_t tu_data_phonons_t;
struct tu_data_phonons_t {
    diverge_model_t* pmod;

    index_t nk[3];
    index_t nkf[3];
    index_t nO;

    ph_vertex_t* vtx;
    index_t n_vtx;

    complex128_t* g_ob_qO; // (my_nk, n_spin,n_spin,n_orbff, nO)
    complex128_t* tmp_g; // (my_nk, n_spin,n_spin,n_orbff, nO)

    complex128_t *gbuf, *gbuf_zero; // (ll, my_nk, nO, nO), with ll=(0,Lambda)
    double gbuf_last_Lambda, gbuf_zero_last_Lambda;

    int buffered; // optional buffered operation for gen_vert()
    complex128_t *Dch, *Dch_zero; // (my_nk, POW2(POW2(n_spin)*n_orbff))
    double Dch_last_Lambda, Dch_zero_last_Lambda;

    index_t n_spin, n_orbff;
    index_t my_nk;
    index_t my_nk_off;
    const index_t* idx_ibz_in_fullmesh;
};

// TODO those are for testing
extern int tu_data_phonons_init___fftw_sign;

tu_data_phonons_t* tu_data_phonons_init( diverge_model_t* ph, diverge_model_t* el,
        bool buffered );
void tu_data_phonons_free( tu_data_phonons_t* t );

// generates D channel vertex scaled by fac, adds to existing content of buffer
void tu_data_phonons_gen_vert( tu_data_phonons_t* t, double Lambda,
        complex128_t* out, double fac );
diverge_model_t* tu_data_phonons_get_model( tu_data_phonons_t* t );


#ifdef __cplusplus
}
#endif
