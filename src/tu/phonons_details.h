/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "phonons.h"

#ifdef __cplusplus
extern "C" {
#endif

struct tu_data_phonons_t {
    diverge_model_t* pmod;

    index_t nk[3];
    index_t nkf[3];
    index_t nO;

    ph_vertex_t* vtx;
    index_t n_vtx;

    complex128_t* g_ob_qO; // (my_nk, n_spin,n_spin,n_orbff, nO)
    complex128_t* g_ob_qO_fullmesh; // only present if requested (nk, n_spin,n_spin,n_orbff, nO)

    int ignore_imag_freq;
    int save_g_on_full_qmesh;
    complex128_t *gbuf, *gbuf_zero; // (my_nk, nO)
    double gbuf_last_Lambda, gbuf_zero_last_Lambda;

    complex128_t *Dch, *Dch_zero; // (my_nk, POW2(POW2(n_spin)*n_orbff))
    double Dch_last_Lambda, Dch_zero_last_Lambda;

    index_t n_spin, n_orbff;
    index_t my_nk;
    index_t my_nk_off;
    const index_t* idx_ibz_in_fullmesh;

    int enable_refinement;
    refinement_t* refine; // dv_vec_free(), not normal free!
    index_t n_refine;

    complex128_t* g_ob_qO_refined; // (n_refine, n_spin,n_spin,n_orbff, nO)
    complex128_t *gbuf_ref, *gbuf_ref_zero; // (n_refined, nO)
    double gbuf_ref_last_Lambda, gbuf_ref_zero_last_Lambda;

    diverge_phonon_vertex_generator_t V;
    diverge_flow_step_t* V_step;
    void* V_ctx;
};


#ifdef __cplusplus
}
#endif
