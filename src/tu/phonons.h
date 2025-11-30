/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

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
        bool buffered, bool ignore_imag_freq, bool enable_refinement );
void tu_data_phonons_free( tu_data_phonons_t* t );

// generates D channel vertex scaled by fac, adds to existing content of buffer
void tu_data_phonons_gen_vert( tu_data_phonons_t* t, double Lambda,
        complex128_t* out, double fac );
diverge_model_t* tu_data_phonons_get_model( tu_data_phonons_t* t );
complex128_t* tu_data_phonons_get_g( tu_data_phonons_t* t );

// we expose an interface to the refined g
complex128_t* tu_data_phonons_get_g_ref( tu_data_phonons_t* t );

// and to the refinement itself
refinement_t* tu_data_phonons_get_refine( tu_data_phonons_t* t, index_t* n_refine );


#ifdef __cplusplus
}
#endif
