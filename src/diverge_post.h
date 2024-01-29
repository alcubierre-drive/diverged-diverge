/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_flow_step.h"

#ifdef __cplusplus
extern "C" {
#endif

// Struct: diverge_postprocess_conf_t
//
// Info:
// patch_ - used with the patch backend
// grid_ - used with the grid backend
// tu_ - used with the tu backend
typedef struct {
    // Variable: patch_q_matrices
    // include the channelized matrices in the save
    bool    patch_q_matrices;
    // Variable: patch_q_matrices_use_dV
    // use dV instead of V for the channelized matrices
    bool    patch_q_matrices_use_dV;
    // Variable: patch_q_matrices_nv
    // how many eigenvectors. -1: full (no eigen), 0: all, >0: number
    int     patch_q_matrices_nv;
    // Variable: patch_q_matrices_max_rel
    // restrict analysis to q_norm <= <val>. if <0  or >1, no restriction.
    double  patch_q_matrices_max_rel;
    // Variable: patch_q_matrices_eigen_which
    // sort q matrices eigenvalues by 'M'agnitude, 'P'ositive, 'N'egative, or 'A'lternating values
    char    patch_q_matrices_eigen_which;
    // Variable: patch_V
    // include vertex in save
    bool    patch_V;
    // Variable: patch_dV
    // include dV in save
    bool    patch_dV;
    // Variable: patch_Lp
    // include plus loop in save
    bool    patch_Lp;
    // Variable: patch_Lm
    // include minus loop in save
    bool    patch_Lm;

    // Variable: grid_lingap_vertex_file_P[MAX_NAME_LENGTH]
    // output the q=0 linearized gap vertex of each channel to the file (if != "")
    char    grid_lingap_vertex_file_P[MAX_NAME_LENGTH];
    // Variable: grid_lingap_vertex_file_C[MAX_NAME_LENGTH]
    char    grid_lingap_vertex_file_C[MAX_NAME_LENGTH];
    // Variable: grid_lingap_vertex_file_D[MAX_NAME_LENGTH]
    char    grid_lingap_vertex_file_D[MAX_NAME_LENGTH];
    // Variable: grid_n_singular_values
    // number of singular values to save for the lineraized gap solutions
    int     grid_n_singular_values;
    // Variable: grid_use_loop
    // include the loop in the linearized gap solution
    bool    grid_use_loop;
    // Variable: grid_vertex_file[MAX_NAME_LENGTH]
    // write the full vertex to this file (if != "")
    char    grid_vertex_file[MAX_NAME_LENGTH];
    // Variable: grid_vertex_chan
    // choose the channel to write the vertex in
    char    grid_vertex_chan;

    // Variable: tu_which_solver_mode
    // choose how to handle non hermitian entries of the channels options
    // 'a'(utomatic), 's'(vd), 'e'(igen), 'b'(atched eigen), 'B'(atched SVD)
    char    tu_which_solver_mode;
    // Variable: tu_skip_channel_calc
    // if true, the eigenvectors of each channel are not calculated - usefull for RPA
    bool    tu_skip_channel_calc;
    // Variable: tu_storing_threshold
    // sets a theshold above which all eigen/singular vectors will be stored (magnitude)
    double  tu_storing_threshold;
    // Variable: tu_storing_relative
    // use a relative storing threshold instead of the aboslute one
    bool    tu_storing_relative;
    // Variable: tu_n_singular_values
    // number of singular values to save for the lineraized gap solutions
    index_t tu_n_singular_values;
    // Variable: tu_lingap
    // calculate the linearized gap solution
    bool    tu_lingap;
    // Variable: tu_susceptibilities_full
    // calculate the susceptibilities from tufrg in o_1-o_4 notation
    bool    tu_susceptibilities_full;
    // Variable: tu_susceptibilities_ff
    // calculate the susceptibilities from tufrg in formfactor space
    bool    tu_susceptibilities_ff;
    // Variable: tu_selfenergy
    // store the selfenergy if it was included in the calculation
    bool    tu_selfenergy;
    // Variable: tu_channels
    // store the vertices in the formfactor basis -- Requires a lot of memory
    bool    tu_channels;
    // Variable: tu_symmetry_maps
    // store the symmetry transfromations for the left multiindex -- works for
    // eigenvectors
    bool    tu_symmetry_maps;

    // DEVELOPERS
    //
    // For a new backend, you may choose to add postprocessing flags here and
    // their defaults below
} diverge_postprocess_conf_t;

#if !defined(__cplusplus) && !defined(CTYPESGEN)
// defaults for C
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
    .tu_symmetry_maps = false
};
#endif

// Function: diverge_postprocess_conf_defaults_CPP
// return default values in C, C++, and Python
diverge_postprocess_conf_t diverge_postprocess_conf_defaults_CPP( void );

// Function: diverge_postprocess_and_write
// Post-processing and output to disk. File format specification depends on the
// chosen backend. In general, the header (shown in the figures) is followed by
// data. For the TUFRG backend, displacements are _in bytes_ while sizes are in
// units of the respective datatype. For the other backends, all units are
// <index_t> (i.e. 64bit)
//
// Grid-FRG:
//
// (see diverge_grid_output.png)
//
// Patch-FRG:
//
// (see diverge_patch_output.png)
//
// TUFRG:
//
// (see diverge_tu_output.png)
void diverge_postprocess_and_write( diverge_flow_step_t* st, const char* file );

// Function: diverge_postprocess_and_write_finegrained
// same as <diverge_postprocess_and_write> but with control over the parameters
// in <diverge_postprocess_conf_t>. <diverge_postprocess_and_write> uses the
// defaults.
void diverge_postprocess_and_write_finegrained( diverge_flow_step_t* st, const char* file,
        const diverge_postprocess_conf_t* cfg );

// TODO(LK) we might want to include the Metznerish Way of doing a channel
// specific inverse BSE equation such that post-processing output doesn't depend
// on the maximal stopping scale of the flow equations. Need to integrate up the
// bubble though...

#ifdef __cplusplus
}
#endif

