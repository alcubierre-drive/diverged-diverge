/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration
typedef struct rs_hopping_t rs_hopping_t;
typedef struct diverge_model_t diverge_model_t;

#ifndef DIVERGE_W90_READER_SWAP_ORBITALS
#define DIVERGE_W90_READER_SWAP_ORBITALS -2
#endif

/**
 * reads Wannier90's \*_hr.dat file into divERGe's :c:struct:`rs_hopping_t` data
 * structure.
 *
 * :param const char* fname: /path/to/material_hr.dat
 * :param index_t n_spin: default value 0 amounts to :math:`SU(2)` symmetric model. if
 *                        :math:`n_\mathrm{spin} \neq 0`,
 *                        :math:`|n_\mathrm{spin}| = 2S+1` with :math:`S` the
 *                        physical spin (i.e., for :math:`S=1/2` we have
 *                        :math:`|n_\mathrm{spin}|=2`). the sign determines
 *                        whether the spin index is the one which increases
 *                        memory slowly (negative) or fast (positive) in the W90
 *                        file. Note that the divERGe convention is always
 *                        (s,o), i.e. spin indices increasing memory fast
 *                        ('outer indices').
 * :param index_t* len: output for the number of hoppings that have been read.
 * :param index_t* n_orb: output for the maximum number of orbitals that were
 *                        found in the hr.dat file. not considered if NULL. In
 *                        case ``n_orb != NULL`` and ``*n_orb ==
 *                        DIVERGE_W90_READER_SWAP_ORBITALS``, swap the orbital
 *                        indices.
 *
 * .. note:: In previous versions, setting ``~(*n_orb) == 0`` would swap the
 *           orbital ordering. Since we recently changed the *default* ordering
 *           of o1/o2 for the W90 reader, the magic number to get swapped
 *           ordering changed. In this way, previous codes that explicitly
 *           swapped the ordering are still correct. If you provide ``~(*n_orb)
 *           == 0``, a corresponding deprecation warning is issued.
 *
 * Example:
 *
 * .. sourcecode:: C
 *
 *  diverge_model_t* model = diverge_model_init();
 *  // set some model parameters...
 *  model->SU2 = 1;
 *  model->n_spin = 1;
 *  model->hop = diverge_read_W90_C( "/path/to/material_hr.dat",
 *                                  0, &model->n_hop, &model->n_orb );
 *  // do something with the hoppings here!
 */
rs_hopping_t* diverge_read_W90_C( const char* fname, index_t n_spin, index_t* len, index_t* n_orb );

/**
 * We do not support the C++ interface any longer. Instead, this function just
 * calls :c:func:`diverge_read_W90_C`.
 */
rs_hopping_t* diverge_read_W90( const char* fname, index_t n_spin, index_t* len, index_t* n_orb );

/**
 * read the lattice vectors and Wannier centers from a .wout file
 *
 * :param const char* fname: ``prefix.wout`` filename
 * :param double* lattice: array of shape (3,3) that the lattice vectors are
 *      stored into
 * :param index_t* n_wan: pointer to an :c:type:`index_t` where the number of
 *      WFs is stored on exit
 *
 * :returns: array of shape ``(*n_wan,3)`` with the Wannier centers. This array
 *      must be free'd by the user (call to ``free`` or
 *      :c:func:`diverge_mem_free`)
 */
double* diverge_read_wout( const char* fname, double* lattice, index_t* n_wan );

/**
 * apply WS symmetry as given in the wsvec W90 file to hopping parameter array,
 * act like realloc. The ``spin_outer`` parameter refers to the explanation
 * given in :c:func:`diverge_read_W90_C`, i.e., ``spin_outer = (n_spin > 0)``.
 */
rs_hopping_t* diverge_W90_apply_wsvec( rs_hopping_t* hop, index_t* p_n_hop,
        int spin_outer, const char* wsvec );

/**
 * convenience combination of :c:func:`diverge_read_W90`,
 * :c:func:`diverge_read_wout`, and :c:func:`diverge_W90_apply_wsvec` to
 * directly yield a :c:type:`diverge_model_t`.
 */
diverge_model_t* diverge_read_W90_model( const char* hr_dat, const char* wout,
        const char* wsvec, index_t n_spin );

#ifdef __cplusplus
}
#endif
