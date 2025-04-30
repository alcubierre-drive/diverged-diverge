/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/** electron-phonon vertex type definition. */
typedef struct ph_vertex_t {
    /** padding to be equivalent to :c:type:`rs_vertex_t` */
    char _chan;
    /**
     * electronic bond (R vector, analogous to :c:type:`tu_formfactor_t` or
     * :c:type:`rs_hopping_t`)
     */
    index_t R12[3];
    /** electron 1, orbital (see :c:type:`rs_hopping_t`)*/
    int32_t o1;
    /** electron 1, spin (see :c:type:`rs_hopping_t`) */
    int32_t s1;
    /** electron 2, orbital (see :c:type:`rs_hopping_t`) */
    int32_t o2;
    /** electron 2, spin (see :c:type:`rs_hopping_t`) */
    int32_t s2;

    // o1,s1, o2,s2 take the space of o1,o2 in the rs_vertex_t struct. We have
    // s1,...,s4 left which is occupied by O1, R1[3].

    /** phononic 'orbital' (site,xyz) multiindex */
    index_t O1;
    /** phononic 'transfer' vector (only an FFT of :math:`\boldsymbol{q}`) */
    index_t R1[3];

    /** coupling strength */
    complex128_t g;
} ph_vertex_t;

// all the other functions are invisible to the user; this header should only be
// consulted for the definition of ph_vertex_t.

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
