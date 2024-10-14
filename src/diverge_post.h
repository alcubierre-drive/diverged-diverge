/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_flow_step.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_postprocess_conf_t diverge_postprocess_conf_t;
/**
 * Default values are obtained using
 * :c:func:`diverge_postprocess_conf_defaults_CPP` from C/C++/Python and the
 * static constant variable :c:var:`diverge_postprocess_conf_defaults` from C.
 *
 * .. note::
 *  This documentation does *not* include the defaults directly, but
 *  they are given in the (linked) source file ``src/diverge_post.h``.
 *
 * .. note::
 *  Member variables with prefix are used as follows
 *
 *  ``patch_``
 *   used with the patch backend
 *  ``grid_``
 *   used with the grid backend
 *  ``tu_``
 *   used with the tu backend
 */
struct diverge_postprocess_conf_t {
    /** include the channelized matrices in the save */
    bool    patch_q_matrices;
    /** use dV instead of V for the channelized matrices */
    bool    patch_q_matrices_use_dV;
    /** how many eigenvectors. -1: full (no eigen), 0: all, >0: number */
    int     patch_q_matrices_nv;
    /** restrict analysis to q_norm <= <val>. if <0  or >1, no restriction. */
    double  patch_q_matrices_max_rel;
    /** sort q matrices eigenvalues by 'M'agnitude, 'P'ositive, 'N'egative, or 'A'lternating values */
    char    patch_q_matrices_eigen_which;
    /** include vertex in save */
    bool    patch_V;
    /** include dV in save */
    bool    patch_dV;
    /** include plus loop in save */
    bool    patch_Lp;
    /** include minus loop in save */
    bool    patch_Lm;

    /** output the q=0 linearized gap vertex of the P channel to the file (if != "") */
    char    grid_lingap_vertex_file_P[MAX_NAME_LENGTH];
    /** output the q=0 linearized gap vertex of the C channel to the file (if != "") */
    char    grid_lingap_vertex_file_C[MAX_NAME_LENGTH];
    /** output the q=0 linearized gap vertex of the D channel to the file (if != "") */
    char    grid_lingap_vertex_file_D[MAX_NAME_LENGTH];
    /** number of singular values to save for the lineraized gap solutions */
    int     grid_n_singular_values;
    /** include the loop in the linearized gap solution */
    bool    grid_use_loop;
    /** write the full vertex to this file (if != "") */
    char    grid_vertex_file[MAX_NAME_LENGTH];
    /** choose the channel to write the vertex in */
    char    grid_vertex_chan;

    /**
     * choose how to handle non hermitian entries of the channel
     * diagonalization. options: 'a'(utomatic), 's'(vd), 'e'(igen), 'b'(atched
     * eigen), 'B'(atched SVD)
     */
    char    tu_which_solver_mode;
    /** if true, the eigenvectors of each channel are not calculated. useful for RPA calculations */
    bool    tu_skip_channel_calc;
    /** sets a theshold above which all eigen/singular vectors will be stored (magnitude) */
    double  tu_storing_threshold;
    /** use a relative storing threshold instead of the aboslute one */
    bool    tu_storing_relative;
    /** number of singular/eigen values to save for the lineraized gap solutions */
    index_t tu_n_singular_values;
    /** calculate the linearized gap solution */
    bool    tu_lingap;
    /** calculate the susceptibilities from tufrg in o_1-o_4 notation */
    bool    tu_susceptibilities_full;
    /** calculate the susceptibilities from tufrg in formfactor space */
    bool    tu_susceptibilities_ff;
    /** store the selfenergy if it was included in the calculation */
    bool    tu_selfenergy;
    /** store the vertices in the formfactor basis (*requires a lot of memory*) */
    bool    tu_channels;
    /** store the symmetry transfromations for the left multiindex (works for eigenvectors) */
    bool    tu_symmetry_maps;
    /** store the N leading values of the channel eigendecomposition/SVD. ignored if zero */
    index_t tu_n_decomp_values;
    /**
     * use the at-scale propagator instead of the integrated one in the
     * linearized gap, useful for RPA
     */
    bool    tu_lingap_atscale;
    /**
     * choose how to handle diagonalizations in the linearized gap equations for
     * each of the channels (PCD, as index), useful for RPA. options: ``'s'``
     * (svd), ``'E'`` (eigen, sort by absolute value), ``'e'`` (eigen, sort
     * alternating), ``'t'`` (eigen, sort from largest), ``'b'`` (eigen, sort
     * from smallest)
     */
    char    tu_lingap_solver_mode[4];

    // DEVELOPERS
    //
    // For a new backend, you may choose to add postprocessing flags here and
    // their defaults below
};

#if !defined(__cplusplus) && !defined(CTYPESGEN)
/** defaults for C */
static const diverge_postprocess_conf_t diverge_postprocess_conf_defaults = {
    .patch_q_matrices = true,
    .patch_q_matrices_use_dV = false,
    .patch_q_matrices_nv = 0,
    .patch_q_matrices_max_rel = 0.9,
    .patch_q_matrices_eigen_which = 'M',
    .patch_V = false,
    .patch_dV = false,
    .patch_Lp = false,
    .patch_Lm = false,

    .grid_lingap_vertex_file_P = {'\0'},
    .grid_lingap_vertex_file_C = {'\0'},
    .grid_lingap_vertex_file_D = {'\0'},
    .grid_n_singular_values = 20,
    .grid_use_loop = true,
    .grid_vertex_file = {'\0'},
    .grid_vertex_chan = '0',

    .tu_which_solver_mode = 'a',
    .tu_skip_channel_calc = false,
    .tu_storing_threshold = 25.,
    .tu_storing_relative = 0,
    .tu_n_singular_values = 1,
    .tu_lingap = true,
    .tu_susceptibilities_full = false,
    .tu_susceptibilities_ff = false,
    .tu_selfenergy = false,
    .tu_channels = false,
    .tu_symmetry_maps = false,
    .tu_n_decomp_values = 0,
    .tu_lingap_atscale = false,
    .tu_lingap_solver_mode = "sss",
};
#endif

/** return default values in C, C++, and Python */
diverge_postprocess_conf_t diverge_postprocess_conf_defaults_CPP( void );

/**
 * Post-processing and output to disk as .dvg file. Those files can be read by
 * the ``diverge.output`` Python library (see :ref:`Simulation Output`).
 * File format specification depends on the chosen backend. In general, the
 * header (shown in the figures) is followed by data. For the TUFRG backend,
 * displacements are *in bytes* while sizes are in units of the respective
 * datatype. For the other backends, all units are :c:type:`index_t` (i.e.
 * 64bit). Default values from :c:struct:`diverge_postprocess_conf_t` (i.e.
 * :c:func:`diverge_postprocess_conf_defaults_CPP`) are taken. For more control,
 * use :c:func:`diverge_postprocess_and_write_finegrained` or its pyhton wrapper
 * :func:`diverge.postprocess_and_write_PY`.
 *
 * Grid-FRG:
 *
 * .. image:: img/diverge_grid_output.png
 *      :width: 800
 *      :alt: specification for grid frg output files
 *
 * Patch-FRG:
 *
 * .. image:: img/diverge_patch_output.png
 *      :width: 800
 *      :alt: specification for patch frg output files
 *
 * TUFRG:
 *
 * .. image:: img/diverge_tu_output.png
 *      :width: 800
 *      :alt: specification for TUFRG output files
 *
 */
void diverge_postprocess_and_write( diverge_flow_step_t* st, const char* file );

/**
 * same as :c:func:`diverge_postprocess_and_write` but with control over the
 * parameters in :c:struct:`diverge_postprocess_conf_t`.
 * :c:func:`diverge_postprocess_and_write` uses the defaults
 * (:c:var:`diverge_postprocess_conf_defaults` or
 * :c:func:`diverge_postprocess_conf_defaults_CPP`).
 */
void diverge_postprocess_and_write_finegrained( diverge_flow_step_t* st, const char* file,
        const diverge_postprocess_conf_t* cfg );

// TODO(LK) we might want to include the Metznerish Way of doing a channel
// specific inverse BSE equation such that post-processing output doesn't depend
// on the maximal stopping scale of the flow equations. Need to integrate up the
// bubble though...

#ifdef __cplusplus
}
#endif

