/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_phonons.h"
#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_data_phonons_t tu_data_phonons_t;

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
