/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"
#include "misc/merge_orb_w_rs_symm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/**
 * generates symmetry mappings needed to perform the tasks below. For certain
 * models, the positions might not be given accurately enough. You can set the
 * precision of position and orbital mapping via
 * :c:func:`diverge_generate_symm_maps_set_precision`.
 */
void diverge_generate_symm_maps( diverge_model_t* model );

/**
 * set the precision for symmetry map generation.
 *
 * :param double ASSERT_POSITION_BEYOND: for assertion whether the beyond unit
 *      cell portion of the position mapping is violated, ~0
 * :param double ORBITAL_OVERLAP: minimal overlap of orbitals that is considered
 *      nonzero, ~0
 * :param double POS_EQUAL: finding whether obtials have same position, ~0
 * :param double MIN_ORBITAL_WEIGHT: minimal orbital weight that is considered
 *      'one', ~1
 */
void diverge_generate_symm_maps_set_precision( double ASSERT_POSITION_BEYOND,
        double ORBITAL_OVERLAP, double POS_EQUAL, double MIN_ORBITAL_WEIGHT );

/**
 * symmetrizes coarse mesh 2pt function stored in buf. returns maximum
 * difference between symmetrized and un-symmetrized buffer
 *
 * :param diverge_model_t model: the divERGe model
 * :param complex128_t* buf: buffer to symmetrize, length (nk, nb, nb) with nb = n_orb * n_spin
 * :param complex128_t* aux: auxiliary buffer that may be either NULL or of the same size as buf
 */
double diverge_symmetrize_2pt_coarse( diverge_model_t* model, complex128_t* buf, complex128_t* aux );

/**
 * symmetrizes fine mesh 2pt function stored in buf. returns maximum difference
 * between symmetrized and un-symmetrized buffer
 *
 * :param diverge_model_t model: the divERGe model
 * :param complex128_t* buf: buffer to symmetrize, length (nkf, nb, nb) with nb = n_orb * n_spin
 * :param complex128_t* aux: auxiliary buffer that may be either NULL or of the same size as buf
 */
double diverge_symmetrize_2pt_fine( diverge_model_t* model, complex128_t* buf, complex128_t* aux );

/**
 * symmetrizes 2pt function of type ``gf_complex_t`` on a fine mesh (as the
 * Green's function).  returns maximum difference between symmetrized and
 * un-symmetrized buffer.
 *
 * :param diverge_model_t model: the divERGe model
 * :param gf_complex_t* buf: buffer to symmetrize, length (nkf, nb, nb) with nb = n_orb * n_spin
 * :param gf_complex_t* aux: auxiliary buffer that may be either NULL or of the same size as buf
 */
double diverge_symmetrize_greens( diverge_model_t* model, gf_complex_t* buf, gf_complex_t* aux );

/**
 * symmetrizes coarse mesh momentum space function (no orbital/spin indices)
 * stored in buf. returns maximum difference between symmetrized and
 * un-symmetrized buffer
 *
 * :param diverge_model_t* model: the divERGe model
 * :param double* buf: buffer of size (nk, sub)
 * :param index_t sub: number of internal indices (e.g. bands). all indices belonging to the
 *                     sub dimension are symmetrized
 * :param double* aux: auxiliary buffer that may be either NULL or of the same size as buf
 */
double diverge_symmetrize_mom_coarse( diverge_model_t* model, double* buf, index_t sub, double* aux );

/**
 * symmetrizes fine mesh momentum space function (no orbital/spin indices)
 * stored in buf. returns maximum difference between symmetrized and
 * un-symmetrized buffer
 *
 * :param diverge_model_t* model: the divERGe model
 * :param double* buf: buffer of size (nkf, sub)
 * :param index_t sub: number of internal indices (e.g. bands). all indices belonging to the
 *                     sub dimension are symmetrized
 * :param double* aux: auxiliary buffer that may be either NULL or of the same size as buf
 */
double diverge_symmetrize_mom_fine( diverge_model_t* model, double* buf, index_t sub, double* aux );

/**
 * symmetrizes hopping parameters of a divERGe model. Operates on the momentum
 * mesh associated to the ``model``.
 *
 * .. note::
 *  In case ``hop == NULL``, this function operates on the internal hopping
 *  parameter array :c:member:`diverge_model_t.hop`. If the model's internals
 *  struct isn't initialized, we generate it automagically *and delete it again
 *  thereafter*. For safe operation, we advise to *not* rely on this feature.
 *
 * :param diverge_model_t* model: the divERGe model which needs to have common
 *                                internals set (using
 *                                :c:func:`diverge_model_internals_common`) and
 *                                nonzero number of symmetries/allocated orbital
 *                                symmetries (:c:member:`diverge_model_t.n_sym`,
 *                                :c:member:`diverge_model_t.orb_symmetries`).
 * :param rs_hopping_t* hop: hopping parameters to symmetrize. Reallocates (via
 *                           ``realloc``) and returns the new memory. If ``hop
 *                           == NULL``, the function operates on the internal
 *                           hopping parameters, ignores ``n_hop``, and swaps
 *                           out the old pointer to the realloced one.
 * :param index_t* n_hop: pointer to the length of the ``hop`` array. If
 *                       ``n_hop == NULL`` operate on the model's hopping
 *                       parameters and ignore ``n_hop`` and ``hop``.
 * :param double hoplim: absolute value up to which include hopping parameters
 *                       in the symmetrized hoppings
 *
 * :returns: :c:type:`rs_hopping_t` array that is the realloced version of the
 *                                  input, On exit and if ``!= NULL``,
 *                                  ``*n_hop`` is set to the new number of
 *                                  hopping parameters. Otherwise, all relevant
 *                                  information is updated in the ``model``.
 */
rs_hopping_t* diverge_symmetrize_hoppings( diverge_model_t* model,
        rs_hopping_t* hop, index_t* n_hop, double hoplim );

#ifdef __cplusplus
}
#endif
