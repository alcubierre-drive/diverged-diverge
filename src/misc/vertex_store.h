/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../diverge_flow_step.h"

typedef struct tu_vertex_storage_t tu_vertex_storage_t;
typedef struct tu_vertex_storage_qpt_t tu_vertex_storage_qpt_t;

/** utility struct to save qpoint and ints name */
struct tu_vertex_storage_qpt_t {
    /** qpt index (wrt ibz, cf. :c:func:`diverge_flow_step_ibz`) */
    index_t qpt;
    /** and a name that you can set if you'd like to */
    char name[24];
};

/** utility struct to allow saving of TUFRG vertices at each flow step. */
struct tu_vertex_storage_t {
    /**
     * channel info, follows :c:func:`diverge_flow_step_vertex`. Channel may be
     * inaccurate in case :c:func:`tu_vertex_storage_snap_chan` has been used.
     */
    char chan;

    /** number of times :c:func:`tu_vertex_storage_snap` has been called on this object for storage */
    index_t n_steps;

    /** number of momentum points */
    index_t n_qpts;
    /** and actual momentum points */
    tu_vertex_storage_qpt_t* qpts;

    /** linear dimension of vertex (n_orbff,n_spin,n_spin) */
    index_t linsz;

    // private
    VECTOR_DECL( complex128_t, memory )
    VECTOR_DECL( double, Lambda )
    VECTOR_DECL( char, chans )
    int* qpts_ranks;
    diverge_flow_step_t* s;
};

/**
 * initialize the vertex storage for a specific channel (following nomenclature
 * in :c:func:`diverge_flow_step_vertex`
 */
tu_vertex_storage_t* tu_vertex_storage_init( diverge_flow_step_t* s, char chan,
        const index_t* qpts, index_t n_qpts );
/**
 * initialize the vertex storage for a specific channel, but with names for
 * qpts. nomenclature follows :c:func:`diverge_flow_step_vertex`.
 */
tu_vertex_storage_t* tu_vertex_storage_init_fg( diverge_flow_step_t* s, char chan,
        const tu_vertex_storage_qpt_t* qpts, index_t n_qpts );

/** get the stored array at qpt index ``iq`` and snap index ``istep`` on rank 0 */
complex128_t* tu_vertex_storage_get( tu_vertex_storage_t* t, index_t istep, index_t iq );

/**
 * add one more step to the storage object (which keeps a reference of
 * :c:struct:`diverge_flow_step_t` around)
 */
void tu_vertex_storage_snap( tu_vertex_storage_t* t );

/** same as :c:func:`tu_vertex_storage_snap` but for arbitrary channel */
void tu_vertex_storage_snap_chan( tu_vertex_storage_t* t, char chan );

/** same as :c:func:`tu_vertex_storage_snap` but with scale information */
void tu_vertex_storage_lsnap( tu_vertex_storage_t* t, double Lambda );

/** same as :c:func:`tu_vertex_storage_snap_chan` but with scale information */
void tu_vertex_storage_lsnap_chan( tu_vertex_storage_t* t, double Lambda, char chan );

/** pop last step from the storage object */
void tu_vertex_storage_pop( tu_vertex_storage_t* t );

/** output a :c:struct:`tu_vertex_storage_t` object to disk */
void tu_vertex_storage_to_file( tu_vertex_storage_t* t, const char* fname );

/** free a :c:struct:`tu_vertex_storage_t` object */
void tu_vertex_storage_free( tu_vertex_storage_t* t );

