/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_phonons.h"
#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_data_phonons_t tu_data_phonons_t;

// reset the imaginary regulator for acoustic modes to some new value. return
// the old value. do not reset if qnan_isnan(reg), return the already set value
// in that case.
double tu_data_phonons_change_regulator( double reg );

// same behavior as for tu_data_phonons_change_regulator, but regarding the
// refinement epsilon for mesh comparison with zero
double tu_data_phonons_change_refinement_eps( double eps );

tu_data_phonons_t* tu_data_phonons_init( diverge_model_t* ph, diverge_model_t* el,
        bool ignore_imag_freq, bool enable_refinement, bool save_g_on_full_qmesh );
void tu_data_phonons_free( tu_data_phonons_t* t );

// generates D channel vertex scaled by fac, adds to existing content of buffer
void tu_data_phonons_gen_vert( tu_data_phonons_t* t, double Lambda,
        complex128_t* out, double fac );

// phonon GF on the (rank-local) IBZ (my_nk, my_nk_off)
complex128_t* tu_data_phonons_gfill( tu_data_phonons_t* t, double Lambda,
        bool ignore_imag_freq, bool refined );

void tu_data_phonons_get_g_dim( tu_data_phonons_t* t, int fullmesh, index_t g_dim[3] );

typedef struct diverge_flow_step_t diverge_flow_step_t;
// interface for custom ph vert generation (non-refined)
void tu_data_phonons_set_vertex_generator( tu_data_phonons_t* t,
        diverge_phonon_vertex_generator_t V, diverge_flow_step_t* step, void* ctx );


#ifdef __cplusplus
}
#endif
