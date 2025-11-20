/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_phonons.h"
#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_data_phonons_t tu_data_phonons_t;

tu_data_phonons_t* tu_data_phonons_init( diverge_model_t* ph, diverge_model_t* el,
        bool buffered, bool ignore_imag_freq, const index_t* acoustic_modes,
        index_t n_acoustic_modes );
void tu_data_phonons_free( tu_data_phonons_t* t );

void tu_data_phonons_set_acoustic_modes( tu_data_phonons_t* t,
        const index_t* acoustic_modes, index_t n_acoustic_modes );

// generates D channel vertex scaled by fac, adds to existing content of buffer
void tu_data_phonons_gen_vert( tu_data_phonons_t* t, double Lambda,
        complex128_t* out, double fac );
diverge_model_t* tu_data_phonons_get_model( tu_data_phonons_t* t );
complex128_t* tu_data_phonons_get_g( tu_data_phonons_t* t );


#ifdef __cplusplus
}
#endif
