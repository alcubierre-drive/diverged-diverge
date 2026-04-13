/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

typedef struct diverge_model_t diverge_model_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * map model to a cubic supercell. requires :c:member:`diverge_model_t.hop`,
 * :c:member:`diverge_model_t.n_hop`, and :c:member:`diverge_model_t.lattice` to
 * be set and modifies those members, as well as
 * :c:member:`diverge_model_t.positions` and :c:member:`diverge_model_t.n_orb`.
 *
 * .. note:: The orbital sign convention can be changed by a call to
 *           :c:func:`diverge_supercell_set_sign`.
 *
 * Special case of :c:func:`rs_hopping_to_fractcell` with :math:`N^j_i =
 * \delta_{ij} n^r_i`.
 *
 * .. note :: This function does not touch, "see", or change the vertex elements.
 *            Users are hence advised to set interaction components *after*
 *            mapping the non-interacting parts of the model to a supercell.
 *
 *
 * :param diverge_model_t* m: input/output model
 * :param int[3] nr: description of the cubic supercell :math:`n^r`. the new
 *                   lattice vectors are given as :math:`L_i = n^r_i l_i`, with
 *                   :math:`l_i` the old lattice vectors.
 */
void rs_hopping_to_supercell( diverge_model_t* m, int* nr );

/**
 * map hopping parameters and model to supercell and allow matrix supercells
 * wrt. to the old lattice vectors. requires :c:member:`diverge_model_t.hop`,
 * :c:member:`diverge_model_t.n_hop`, and :c:member:`diverge_model_t.lattice` to
 * be set and modifies those members, as well as
 * :c:member:`diverge_model_t.positions` and :c:member:`diverge_model_t.n_orb`.
 *
 * .. note:: The orbital sign convention can be changed by a call to
 *           :c:func:`diverge_supercell_set_sign`.
 *
 * .. note :: This function does not touch, "see", or change the vertex elements.
 *            Users are hence advised to set interaction components *after*
 *            mapping the non-interacting parts of the model to a supercell.
 *
 * The supercell lattice vectors are given as :math:`L_j = \sum_i N^j_i l_i`. In
 * case N2 == N3 == NULL, make call to :c:func:`rs_hopping_to_supercell` with N1
 * as nr.
 *
 * :param diverge_model_t* m: input/output model
 * :param int[3] N1: integer coordinates :math:`N^1` for first supercell vector
 * :param int[3] N2: integer coordinates :math:`N^2` for first supercell vector
 * :param int[3] N3: integer coordinates :math:`N^3` for first supercell vector
 */
void rs_hopping_to_fractcell( diverge_model_t* m, int* N1, int* N2, int* N3 );

/**
 * same as :c:func:`rs_hopping_to_fractcell` but with matrix ``N123[3][3]`` for
 * supercell, where each row corresponds to one supercell vector.
 */
void rs_hopping_to_fractcell_mat( diverge_model_t* m, int* N123 );

/** same as :c:func:`rs_hopping_to_supercell`, but also maps (real space) vertex elements */
void diverge_model_to_supercell( diverge_model_t* m, int* nr );
/** same as :c:func:`rs_hopping_to_fractcell`, but also maps (real space) vertex elements */
void diverge_model_to_fractcell( diverge_model_t* m, int* N1, int* N2, int* N3 );
/** same as :c:func:`rs_hopping_to_fractcell_mat`, but also maps (real space) vertex elements */
void diverge_model_to_fractcell_mat( diverge_model_t* m, int* N123 );

/**
 * to swap the orbital convention in supercells, use this function to set -1
 * instead of the default, i.e., 1. Not thread-safe. Call before generation.
 */
void diverge_supercell_set_sign( double sign );

/**
 * to swap the orbital convention only in the hoppings, not in the positions.
 * returns whether original state is used. Also not thread-safe, call before
 * generation.
 */
int diverge_supercell_swap_hop( void );


/**
 * structure to set/get configuration of supercell, cf.
 * :c:func:`diverge_supercell_configure`.
 */
typedef struct {
    /**
     * sign (±1, cf. :c:func:`diverge_supercell_set_sign`). If
     * ``qnanf_isnan(sign)`` do not change this value and only return current
     * setting.
     */
    float sign;
    /**
     * swap orbitals (0|1, cf. :c:func:`diverge_supercell_swap_hop`). If
     * ``swap==-1`` do not change this value and only return current setting.
     */
    int swap;
} diverge_supercell_configure_t;

/**
 * advanced configuration interface that also returns previous state. See
 * :c:struct:`diverge_supercell_configure_t` for parameters. Comprehensive
 * wrapper around both :c:func:`diverge_supercell_set_sign` and
 * :c:func:`diverge_supercell_swap_hop`.
 */
diverge_supercell_configure_t diverge_supercell_configure( diverge_supercell_configure_t cfg );

#ifdef __cplusplus
}
#endif
