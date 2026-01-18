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
 * initialize the vertex storage for a specific channel (following channel
 * nomenclature of :c:func:`diverge_flow_step_vertex`). For control over the
 * qpts' names use either the alternative constructor
 * :c:func:`tu_vertex_storage_init_fg` or access (and change) the names via
 * :c:member:`tu_vertex_storage_t.qpts`.
 */
tu_vertex_storage_t* tu_vertex_storage_init( diverge_flow_step_t* s, char chan,
        const index_t* qpts, index_t n_qpts );

/**
 * initialize the vertex storage for a specific channel, but with named qpts
 * (following channel nomenclature of :c:func:`diverge_flow_step_vertex`).
 * otherwise equivalent to :c:func:`tu_vertex_storage_init`.
 */
tu_vertex_storage_t* tu_vertex_storage_init_fg( diverge_flow_step_t* s, char chan,
        const tu_vertex_storage_qpt_t* qpts, index_t n_qpts );

/**
 * get the stored array at qpt index ``iq`` and snap index ``istep`` on rank 0,
 * all other ranks return NULL.
 */
complex128_t* tu_vertex_storage_get( tu_vertex_storage_t* t, index_t istep, index_t iq );

/**
 * add one more vertex to the storage object (which keeps a reference of
 * :c:struct:`diverge_flow_step_t` around that was given in the constructor
 * :c:func:`tu_vertex_storage_init` in order to obtain the vertices).
 */
void tu_vertex_storage_snap( tu_vertex_storage_t* t );
/** same as :c:func:`tu_vertex_storage_snap` but for arbitrary channel */
void tu_vertex_storage_snap_chan( tu_vertex_storage_t* t, char chan );
/** same as :c:func:`tu_vertex_storage_snap` but with scale information */
void tu_vertex_storage_lsnap( tu_vertex_storage_t* t, double Lambda );
/** same as :c:func:`tu_vertex_storage_snap_chan` but with scale information */
void tu_vertex_storage_lsnap_chan( tu_vertex_storage_t* t, double Lambda, char chan );
/**
 * convenience wrapper around :c:func:`tu_vertex_storage_snap_chan` that
 * iterates over *all* channels that are included in the
 * :c:struct:`diverge_flow_step_t`
 */
void tu_vertex_storage_snap_all( tu_vertex_storage_t* t );
/** same as :c:func:`tu_vertex_storage_snap_all` but with scale information */
void tu_vertex_storage_lsnap_all( tu_vertex_storage_t* t, double Lambda );

/** pop last step from the storage object. does not release memory; operates lazily. */
void tu_vertex_storage_pop( tu_vertex_storage_t* t );

/**
 * output a :c:struct:`tu_vertex_storage_t` object to a file ``fname``. This
 * file can be read using the ``diverge.output`` python library
 * (:py:meth:`diverge.output.read` returns
 * :py:class:`diverge.output.tu_vertex_store`). The output includes all channel
 * and scale information.
 *
 * file format specification info: ``header`` is 128 elements of type
 * :c:type:`index_t` (64 bit signed integers)
 *
 * all displacement/length information is in bytes.
 *
 *  :``header[0-2]``:     "tu_vertex_storage_t" as byte array, i.e., the three
 *                        numbers (7310593875232388468, 7449361081668165496,
 *                        7626597) when interpreted as int64_t
 *  :``header[124]``:     file size (in bytes)
 *  :``header[125-126]``: file format version string [displ, size].
 *  :``header[127]``:     health check bit field (should be all zero!)
 *  :``header[3]``:       ``chan``
 *  :``header[4]``:       ``linsz``
 *  :``header[5]``:       ``n_steps``
 *  :``header[6]``:       ``n_qpts``
 *  :``header[7-8]``:     [displ, size] for ``qpts`` shape: (n_qpts,)
 *  :``header[9-10]``:    [displ, size] for ``verts`` shape:
 *                        (n_steps,n_qpts,linsz,linsz)
 *  :``header[11-12]``:   [displ, size] for ``Lambda`` shape: (n_steps,)
 *  :``header[13-14]``:   [displ, size] for ``chans`` shape: (n_steps,)
 */
void tu_vertex_storage_to_file( tu_vertex_storage_t* t, const char* fname );

/** free a :c:struct:`tu_vertex_storage_t` object */
void tu_vertex_storage_free( tu_vertex_storage_t* t );

