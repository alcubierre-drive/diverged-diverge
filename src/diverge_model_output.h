/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DIVERGE_MODEL_MAGIC_NUMBER
#define DIVERGE_MODEL_MAGIC_NUMBER ((index_t)'D'+(index_t)'M')
#endif

#ifndef DIVERGE_MODEL_NBANDSTRUCTURE
#define DIVERGE_MODEL_NBANDSTRUCTURE 300
#endif

typedef struct diverge_model_t diverge_model_t;

/* Function: diverge_model_to_file
 * returns a checksum of the model as a string (must not be free'd as it is
 * statically owned by libdivERGe)
 *
 * file format specification
 *
 * header is 128 elements of type index_t (64 bit signed integers)
 *
 * all displacement/length information is in bytes
 *
 * header[0] - DIVERGE_MODEL_MAGIC_NUMBER
 * header[1-2] - name [displ, size]
 * header[3] - dim
 * header[4-6] - nk
 * header[7-9] - nkf
 *
 * _header[10-22] is 0 if patching == NULL_
 *
 * header[10] - n_patches
 * header[11-12] - patches [displ, size]
 * header[13-14] - weights [displ, size]
 * header[15-16] - p_count [displ, size]
 * header[17-18] - p_displ [displ, size]
 * header[19-20] - p_map [displ, size]
 * header[21-22] - p_weights [displ, size]
 *
 * hedaer[23-24] - ibz_path [displ, size]
 * header[25] - n_orb
 * header[26-34] - lattice [interpret as double[3][3]]
 * header[35-36] - positions [displ, size]
 * header[37] - n_sym
 * header[38-39] - orb_symmetries [displ, size]
 * header[40-41] - rs_symmetries [displ, size]
 * header[42] - n_hop
 * header[43-44] - hop [displ, size]
 * header[45] - SU2
 * header[46] - n_spin
 * header[47] - n_vert
 * header[48-49] - vert [displ, size]
 * header[50] - n_tu_ff
 * header[51-52] - tu_ff [displ, size]
 * header[53-55] - n_vert_chan
 * header[56-57] - data [displ, size]
 * header[58-66] - reciprocal latttice [interpret as double[3][3]]
 *
 * _by default all the following are zero. They can be controlled via the
 * finegrained function <diverge_model_to_file_finegrained at
 * diverge_model_output_conf_t.diverge_model_to_file_finegrained>_
 *
 * header[80-81] - kmesh [displ, size]
 * header[82-83] - kfmesh [displ, size]
 * header[84-85] - ham [displ, size]
 * header[86-87] - U [displ, size]
 * header[88-89] - E [displ, size]
 * header[90-91] - kmesh ibz_path [displ, size]
 * header[92-93] - kfmesh ibz_path [displ, size]
 *
 * Both the kmesh and kfmesh ibz_path can only be calculated if ibz_path is set.
 * These two objects require a bit of internal structure to be read correctly:
 * The memory layout is:
 *
 * - N (number of segments, <index_t>)
 * - lengh_segment_0 (<index_t>)
 * - …
 * - length_segment_N
 * - M (number of points, <index_t>)
 * - index_point_0 (<index_t>)
 * - …
 * - index_point_M
 * - point_0_x (double)
 * - point_0_y
 * - point_0_z
 * - …
 * - point_M_x
 * - point_M_y
 * - point_M_z
 *
 * Note that [displ, size] accounts for the *full* displacement and size of the
 * memory layout described above
 *
 * _if ibz_path is set, else all the following are zero_
 *
 * header[100-101] - bandstructure [displ, size]
 *
 * _structure offest/size information, useful for python wrapping_
 *
 * header[102] - offsetof(rs_hopping_t, R)
 * header[103] - offsetof(rs_hopping_t, o1)
 * header[104] - offsetof(rs_hopping_t, o2)
 * header[105] - offsetof(rs_hopping_t, s1)
 * header[106] - offsetof(rs_hopping_t, s2)
 * header[107] - offsetof(rs_hopping_t, t)
 * header[108] - sizeof(rs_hopping_t)
 * header[109] - offsetof(rs_vertex_t, chan)
 * header[110] - offsetof(rs_vertex_t, R)
 * header[111] - offsetof(rs_vertex_t, o1)
 * header[112] - offsetof(rs_vertex_t, o2)
 * header[113] - offsetof(rs_vertex_t, s1)
 * header[114] - offsetof(rs_vertex_t, s2)
 * header[115] - offsetof(rs_vertex_t, s3)
 * header[116] - offsetof(rs_vertex_t, s4)
 * header[117] - offsetof(rs_vertex_t, V)
 * header[118] - sizeof(rs_vertex_t)
 * header[119] - offsetof(tu_formfactor_t, R[3])
 * header[120] - offsetof(tu_formfactor_t, ofrom)
 * header[121] - offsetof(tu_formfactor_t, oto)
 * header[122] - offsetof(tu_formfactor_t, d)
 * header[123] - offsetof(tu_formfactor_t, ffidx)
 * header[124] - sizeof(tu_formfactor_t)
 *
 * _header[127]_:
 * bit field (64bits) that is all zero when the numerical representations are
 * correct. if not, errors are indicated in the following manner:
 *
 * - the leftmost (first) bit is one when floats are not represented in
 *   iec559
 * - the second bit is one when doubles are not represented in iec559
 * - the third bit is one when integers (int and index_t) are not little
 *   endian
 * - the fourth bit is one when 1 byte (sizeof(char)) is not the same as 8
 *   bit.
 */
char* diverge_model_to_file( diverge_model_t* mod, const char* fname );

// Struct: diverge_model_output_conf_t
// to have fine-grained output control over diverge_model_to_file
//
// Members (all bool/int):
//
// kc - coarse kmesh
// kf - fine kmesh
// kc_ibz_path - include the indices on the coarse mesh that describe the ibz
//               path. must have model->n_ibz_path > 0.
// kf_ibz_path - include the indices on the fine mesh that describe the ibz
//               path. must have model->n_ibz_path > 0.
// H - Hamiltonian on fine mesh
// U - Orbital to band matrices on fine mesh
// E - energies on fine mesh
typedef struct diverge_model_output_conf_t {
    int kc;
    int kf;
    int kc_ibz_path;
    int kf_ibz_path;
    int H;
    int U;
    int E;
} diverge_model_output_conf_t;

#if !defined(__cplusplus) && !defined(CTYPESGEN)
// defaults for C
static const diverge_model_output_conf_t diverge_model_output_conf_defaults = {
    .kc = 0,
    .kf = 0,
    .kc_ibz_path = 0,
    .kf_ibz_path = 0,
    .H = 0,
    .U = 0,
    .E = 0,
};
#endif // __cplusplus

// Function: diverge_model_output_conf_defaults_CPP
// return the defaults of the <diverge_model_output_conf_t> structure from
// CPP/Python. For C you can use a plain, simple
// === C/C++ ===
// diverge_model_output_conf_t outconf = diverge_model_output_conf_defaults;
// =============
diverge_model_output_conf_t diverge_model_output_conf_defaults_CPP( void );

// Function: diverge_model_to_file_finegrained
// offers the same functionality as <diverge_model_to_file> including
// fine-grained controls over all internal structures. Returns the md5sum of the
// written file as *static* string.
//
// Parameters:
// mod - model
// fname - file name of the output
// cfg - pointer to <diverge_model_output_conf_t>, can be NULL for defaults.
char* diverge_model_to_file_finegrained( diverge_model_t* mod, const char* fname,
        const diverge_model_output_conf_t* cfg );

// TODO(LK) add this to the output config; as additional parameter that doesn't
// have precedence over sth that's set by the function?
// Function: diverge_model_output_set_npath
// set the number of points per IBZ segment for the bandstructure calculation.
// defaults to DIVERGE_MODEL_NBANDSTRUCTURE (300). must be called before
// <diverge_model_to_file>. If set to -1, use <diverge_kmesh_to_bands> on the
// internal energy array and do not generate a band structure. requires common
// internals to be set (due to the energy array).
void diverge_model_output_set_npath( int npath );

#ifdef __cplusplus
}
#endif
