/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/**
 * read FPLO Hamiltonian files (``+hamdata``) into :c:struct:`diverge_model_t`
 * objects. Only sets/reads the members listed below and leaves everything else
 * untouched:
 *
 * * initializes the :c:struct:`diverge_model_t` via :c:func:`diverge_model_init`
 * * sets the hopping parameters (:c:member:`diverge_model_t.hop`)
 * * sets the number of orbitals (:c:member:`diverge_model_t.n_orb`)
 * * sets the number of spins (:c:member:`diverge_model_t.n_spin`)
 * * sets the real space lattice (:c:member:`diverge_model_t.lattice`)
 *   and positions (:c:member:`diverge_model_t.positions`)
 * * returns NULL on errors / if FPLO reader not compiled in
 *
 * .. note::
 *  This function is not compiled into the library by default. Enable
 *  compilation of the FPLO reader by passing the ``USE_FPLO_READER`` flag in
 *  the :ref:`Compilation` process (see :ref:`Compilation DEFINES`).
 */
diverge_model_t* diverge_read_fplo( const char* fname );

#ifdef __cplusplus
}
#endif
