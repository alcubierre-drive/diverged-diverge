/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Formfactor Mapping for Advanced Initialization
 * ==============================================
 *
 * These functions provide a way to explicitly loop over the lattice/orbitals
 * and ask for the corresponding formfactor indices in O(1) time (implemented as
 * :release_tree:`splitmix64 hashtable <src/misc/vhashtable.h>`). Very useful
 * for the advanced vertex Initialization feature in TUFRG
 * (:c:member:`rs_vertex_t.chan` ``=='Z'``).
 */

/** forward declaration of main map */
typedef struct tu_formfactor_map_t tu_formfactor_map_t;

/** compound datatype for the result of a mapping operation */
typedef struct {
    /** formfactor index (following :c:member:`diverge_model_t.tu_ff`) */
    index_t t;
    /** formfactor distance */
    double d;
} tu_formfactor_idx_t;

/** the result in case the requested formfactor could not be found (-1,NAN) */
static const tu_formfactor_idx_t tu_formfactor_idx_invalid = { -1, NAN };

/** initialize the formfactor map from a model *with TU internals* */
tu_formfactor_map_t* tu_formfactor_map_init( diverge_model_t* m );

/** get a formfactor index from Rvectors and oribtals in O(1) time */
tu_formfactor_idx_t tu_formfactor_map_get( tu_formfactor_map_t* tmap,
        index_t Rx, index_t Ry, index_t Rz, index_t ofrom, index_t oto );

/** get a formfactor index from a TU formfactor in O(1) time */
tu_formfactor_idx_t tu_formfactor_map_get_tu( tu_formfactor_map_t* tmap,
        tu_formfactor_t t );

/**
 * return the inverted formfactor (which is nontrivial for TU formfactors, but
 * trivial for Rxyz op ones due to the -nk padding in TU formfactors)
 */
tu_formfactor_idx_t tu_formfactor_map_get_tu_inv( tu_formfactor_map_t* tmap,
        tu_formfactor_t t );

/** clean up resources */
void tu_formfactor_map_free( tu_formfactor_map_t* tmap );

/**
 * generate *real* formfactors (with minus signs) from *TU* formfactors. Must be
 * freed by the user.
 */
tu_formfactor_t* tu_formfactor_periodicity( diverge_model_t* m );

#ifdef __cplusplus
}
#endif
