/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ph_vertex_t {
    char _chan;
    index_t R12[3];
    int32_t o1,s1, o2,s2;
    // that's o1 and o2. we have s1..s4 left, which we map to
    index_t O1,R1[3];
    complex128_t g;
} ph_vertex_t;

struct tu_data_phonons_t {
    diverge_model_t* pmod;
    // we have to set a special GF generator in pmod that does E^2/(E^2+Lambda^2)
    // and has the correct limit

    index_t nk[3];
    complex128_t* gbuf; // (ll, my_nk, nO, nO), with ll=(0,Lambda)
    index_t nO;

    ph_vertex_t* vtx;
    index_t n_vtx;

    complex128_t* g_ob_qO; // (my_nk, n_spin,n_spin,n_orbff, nO)
    complex128_t* tmp_g; // (my_nk, n_spin,n_spin,n_orbff, nO)

    index_t n_spin, n_orbff;
    index_t my_nk;
    index_t my_nk_off;
    const index_t* idx_ibz_in_fullmesh;
};

tu_data_phonons_t* tu_data_phonons_init( diverge_model_t* pmod );
void tu_data_phonons_init_vertex( tu_data_phonons_t* t, tu_data_t* d,
        const tu_formfactor_t* tu_ff, index_t n_tu_ff, index_t n_spin );

void tu_data_phonons_free( tu_data_phonons_t* t );

void tu_data_phonons_gfill( tu_data_phonons_t* t, double Lambda );
// calculates V = g D g*
void phonon_gemm( const complex128_t* g, const complex128_t* D,
        complex128_t* V, complex128_t* tmp, index_t linsz, index_t nO );
void tu_data_phonons_vertices( tu_data_phonons_t* t, double Lambda,
        complex128_t* V_Lambda, complex128_t* V_zero );

#ifdef __cplusplus
}
#endif
