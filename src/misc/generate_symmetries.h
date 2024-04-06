/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MAX_ORBS_PER_SITE
#define MAX_ORBS_PER_SITE 20
#endif // MAX_ORBS_PER_SITE

/**
 * enumerates all possible orbitals supported by divERGe's symmetry generator.
 * For orbitals other than :math:`s`, :math:`p_y`, :math:`p_z`, :math:`p_x`, see
 * the source code.
 */
typedef enum real_harmonics_t {
    /** s orbital */
    orb_s = 0,

    /** py orbital */
    orb_pm1 = 1,
    orb_py = 1,
    /** pz orbital */
    orb_p_0 = 2,
    orb_pz = 2,
    /** px orbital */
    orb_p_1 = 3,
    orb_px = 3,

    orb_dm2 = 4,
    orb_dm1 = 5,
    orb_d0 = 6,
    orb_d1 = 7,
    orb_d2 = 8,

    orb_dxy = 4,
    orb_dyz = 5,
    orb_dz2 = 6,
    orb_dxz = 7,
    orb_dx2y2 = 8,

    // Wikipedia
    orb_fm3 = 9,
    orb_fm2 = 10,
    orb_fm1 = 11,
    orb_f_0 = 12,
    orb_f_1 = 13,
    orb_f_2 = 14,
    orb_f_3 = 15,

    orb_gm4 = 16,
    orb_gm3 = 17,
    orb_gm2 = 18,
    orb_gm1 = 19,
    orb_g_0 = 20,
    orb_g_1 = 21,
    orb_g_2 = 22,
    orb_g_3 = 23,
    orb_g_4 = 24
} real_harmonics_t;

/**
 * defines one symmetry operation
 */
typedef struct sym_op_t {
    /**
     * single character: possible choices are 'R'(rotation), 'S'(spin rotation),
     * 'M'(mirror), 'I'(inversion), 'F'(spin flip), 'E'(identity)
     */
    char type;
    /**
     * the normal vector around which to generate the symmetry, in real space
     * (irrelevant for 'I', 'F', 'E')
     */
    double normal_vector[3];
    /**
     * angle of rotation (relevant for 'R', 'S')
     */
    double angle;
} sym_op_t;

/**
 * defines a single orbital (in the language of ``n_orb`` defined in
 * :c:struct:`diverge_model_t`) on a single site.
 */
typedef struct site_descr_t {
    /** array of amplitudes in the real harmonic basis */
    complex128_t amplitude[MAX_ORBS_PER_SITE];
    /** array of real harmonics defining the basis (see :c:struct:`real_harmonics_t`) */
    real_harmonics_t function[MAX_ORBS_PER_SITE];
// default initialization in c++ is weird...
#ifdef __cplusplus
    index_t n_functions = 0;
    double xaxis[MAX_ORBS_PER_SITE][3] = {{0}};
    double zaxis[MAX_ORBS_PER_SITE][3] = {{0}};
#else
    /**
     * length of the :c:member:`site_descr_t.function` and
     * :c:member:`site_descr_t.amplitude` arrays, i.e., number of real harmonic basis
     * functions of the site
     */
    index_t n_functions;
    /**
     * orientation of the x axis of each of the real harmonic basis functions
     * defined in the amplitude/function arrays. defaults to the cartesian x
     * axis, OPTIONAL.
     */
    double xaxis[MAX_ORBS_PER_SITE][3];
    /**
     * orientation of the z axis of each of the real harmonic basis functions
     * defined in the amplitude/function arrays. defaults to the cartesian z
     * axis, OPTIONAL.
     */
    double zaxis[MAX_ORBS_PER_SITE][3];
#endif
} site_descr_t;

/**
 * generates symmetry transform for the orbitals decribed in the :c:struct:`site_descr_t`
 * array orbs and a symmetry operation described by the :c:struct:`sym_op_t` array syms.
 * Outputs the orbital and real-space symmetry matrices
 *
 * :param index_t n_spin: number of spins
 * :param site_descr_t* orbs: site descriptor array of length ``n_orbs``
 * :param index_t n_orbs: number of 'orbitals' (cf. :c:struct:`diverge_model_t`)
 * :param sym_op_t* syms: symmetry operation array (of length ``n_syms``) of the
 *                        atomic symmetry operations, first element (syms[0])
 *                        acts first, then second (syms[1]), etc.
 * :param index_t n_syms: number of atomic symmetry operations to build symmetry
 *                        operation at hand
 * :param double* rs_trafo: output for the realspace matrix (size must be >= 9)
 * :param complex128_t* orb_trafo: output for the orbital space matrix (size
 *                                 must be >= n_orbs*n_orbs)
 */
void diverge_generate_symm_trafo( index_t n_spin, const site_descr_t* orbs, index_t n_orbs,
    const sym_op_t* syms, index_t n_syms, double* rs_trafo, complex128_t* orb_trafo );

#ifdef __cplusplus // extern "C"
}
#endif
