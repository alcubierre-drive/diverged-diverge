/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../diverge_flow_step.h"

typedef struct tu_highsym_vertex_t tu_highsym_vertex_t;

/**
 * utility struct to allow saving of TUFRG vertices at each flow step.
 *
 * .. note::
 *      this functionality is restricted to TUFRG and in its current
 *      implementation only works when all channels are included in the
 *      simulation.
 */
struct tu_highsym_vertex_t {
    /** number of steps that have a saved vertex */
    index_t nsteps;
    /** number of :c:type:`complex128_t` elements in a single step */
    index_t stepsz;

    /** number of momentum points to save the vertices on */
    index_t nqpts;
    /**
     * number of orbitals/formfactors *and spins* (i.e. the linear dimension,
     * this is strictly speaking *not* ``n_orbff``) in each channel
     */
    index_t n_orbff_P, n_orbff_C, n_orbff_D;

    /**
     * array of which momentum transfer indices are actually saved. only used
     * for output via :c:func:`tu_highsym_vertex_to_file`, must be set manually
     */
    index_t* qpts;
    /**
     * description of the qpts, names of the high-symmetry points. only used for
     * output via :c:func:`tu_highsym_vertex_to_file`
     */
    char* qpts_descr_nams;
    /**
     * description of the qpts, number of symmetry-equivalent points per name;
     * also only used for output via :c:func:`tu_highsym_vertex_to_file`
     */
    index_t* qpts_descr_lens;
    /**
     * description of the qpts, number of high-symmetry names (only used for
     * output via :c:func:`tu_highsym_vertex_to_file`) 
     */
    index_t n_qpts_descr;

    // those are private
    complex128_t* memory;
    index_t memory_idx;
    index_t memory_cap;
};

/**
 * store the channels of the current flow step into ``vert``. if ``vert ==
 * NULL``, initialize a new :c:struct:`tu_highsym_vertex_t` object.
 *
 * :param diverge_flow_step_t* step: flow step object (must be TUFRG!)
 * :param index_t* qpts: indices of the coarse k-mesh to save the vertex on. In
 *                       case MPI and/or symmetries are used, these indices
 *                       refer to the reduced mesh
 * :param index_t nqpts: length of the ``qpts`` array.
 * :param tu_highsym_vertex_t* vert: object to work on. If NULL, a new object is
 *                                   initialized and returned (similar behavior
 *                                   to the realloc library function).
 */
tu_highsym_vertex_t* tu_highsym_vertex_store( diverge_flow_step_t* step,
        index_t* qpts, index_t nqpts, tu_highsym_vertex_t* vert );

/**
 * save the contents of a :c:struct:`tu_highsym_vertex_t` to disk. As of now,
 * the file format is not fixed. It works along the lines of the other file
 * formats encountered in the divERGe world:
 *
 * A header of 128 64-bit (signed) integers describes the file type, and the
 * contents, which are appended to this header. For a description, see the
 * source file (src/misc/vertex_store.c)
 */
void tu_highsym_vertex_to_file( tu_highsym_vertex_t* vert, const char* fname );

/** free a :c:struct:`tu_highsym_vertex_t` object */
void tu_highsym_vertex_free( tu_highsym_vertex_t* vert );

