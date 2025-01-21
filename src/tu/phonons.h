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

typedef struct tu_data_phonons_t tu_data_phonons_t;

tu_data_phonons_t* tu_data_phonons_init( diverge_model_t* pmod, diverge_model_t* elmod );
void tu_data_phonons_free( tu_data_phonons_t* t );

// generates D channel vertex scaled by fac, optionally adds to existing buffer
void tu_data_phonons_gen_vert( tu_data_phonons_t* t, double Lambda,
        complex128_t* out, bool add, double fac );


#ifdef __cplusplus
}
#endif
