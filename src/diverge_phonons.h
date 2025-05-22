/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"
#include "misc/wigner.h"

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration
typedef struct rs_hopping_t rs_hopping_t;
typedef struct diverge_model_t diverge_model_t;

/**
 * electron-phonon vertex structure in real space. Some member types are chosen
 * in a specific way to yield a type that's as large as :c:type:`rs_vertex_t`.
 *
 * The three-leg vertex is characterized by three quantum numbers. Two
 * electronic ones (1,2) and one bosonic one (A). Its real-space version
 * therefore reads
 *
 * :math:`g^{1,2}_A \equiv
 * g^{(\boldsymbol0,o_1,s_1),(\boldsymbol{R}_{12},o_2,s_2)}_{(O_A,\boldsymbol{R}_A)} \,.`
 *
 * In literature, :math:`g` is commonly encountered in momentum space. This is
 * equivalent to performing an FFT on both :math:`\boldsymbol{R}_{12}` and
 * :math:`\boldsymbol{R}_A`:
 *
 * :math:`g^{(o_1,s_1),(o_2,s_2)}_{O_A}(\boldsymbol q, \boldsymbol k) =
 * \sum_{\boldsymbol R_{12}, \boldsymbol R_A} e^{-i\boldsymbol q\cdot
 * \boldsymbol R_A} e^{-i \boldsymbol k\cdot\boldsymbol R_{12}}
 * g^{(\boldsymbol0,o_1,s_1),(\boldsymbol{R}_{12},o_2,s_2)}_{(O_A,\boldsymbol{R}_A)}
 * \,.`
 *
 * To get comfortable with real-space electron-phonon coupling, you can have a
 * look at the examples.
 */
typedef struct ph_vertex_t {
    /** padding to be equivalent to :c:type:`rs_vertex_t` */
    char _chan;
    /**
     * electronic bond (R vector, analogous to :c:type:`tu_formfactor_t` or
     * :c:type:`rs_hopping_t`)
     */
    index_t R12[3];
    /** electron 1, orbital (see :c:type:`rs_hopping_t`)*/
    int32_t o1;
    /** electron 1, spin (see :c:type:`rs_hopping_t`) */
    int32_t s1;
    /** electron 2, orbital (see :c:type:`rs_hopping_t`) */
    int32_t o2;
    /** electron 2, spin (see :c:type:`rs_hopping_t`) */
    int32_t s2;

    // o1,s1, o2,s2 take the space of o1,o2 in the rs_vertex_t struct. We have
    // s1,...,s4 left which is occupied by OA, RA[3].

    /** phononic 'orbital' (site,xyz) multiindex */
    index_t OA;
    /** phononic 'transfer' vector (only an FFT of :math:`\boldsymbol{q}`) */
    index_t RA[3];

    /** coupling strength */
    complex128_t g;
} ph_vertex_t;

/**
 * apply the acoustic sum rule (ASR) to a set of phonon hopping parameters. This
 * function *invalidates* the pointer ``ph_hop``, use as if ``realloc`` were
 * called (i.e. use the returned pointer!).
 *
 * :param rs_hopping_t* ph_hop: hopping parameter array (spring constants), must
 *      be allocated via calloc/malloc/realloc (or their divERGe wrappers)
 *      because we use realloc on this pointer internally!
 * :param index_t* n_ph_hop: number of hoppings, points to new number of
 *      hoppings on return.
 *
 * :returns rs_hopping_t*: hopping parameter array, reallocated.
 */
rs_hopping_t* diverge_phonons_asr( rs_hopping_t* ph_hop, index_t* n_ph_hop );

/**
 * same as :c:func:`diverge_phonons_asr` except that it reallocates m->hop and
 * sets m->n_hop, i.e.:
 *
 * .. sourcecode:: C
 *
 *  m->hop = diverge_phonons_asr( m->hop, &m->n_hop );
 */
void diverge_phonons_apply_asr( diverge_model_t* m );

/**
 * convert eigenvalues to frequencies (by taking :math:`\sqrt{D}`) and fill an
 * internal array that tracks whether at a certain point in momentum space for a
 * certain mode we have *imaginary* frequencies. This bit-array may be retrieved
 * with :c:func:`diverge_phonons_get_ifrq`
 */
void diverge_phonons_apply_sgnsqrt( diverge_model_t* m );

/**
 * return a bit-array (cf. :c:func:`bitary_get`) of shape (nk, nb) that tracks
 * whether at a certain k point and mode the phonon frequency is *imaginary*.
 */
void* diverge_phonons_get_ifrq( diverge_model_t* m );

/**
 * returns the electron-phonon coupling as an array (my_nk, n_spin, n_spin,
 * n_orbff, nO), with my_nk the symmetry-reduced BZ (see TU backend) and nO the
 * number of phonon modes. In case this array is written to the changes are
 * applied to the RG flow (*this is the actual buffer, not a copy*). May return
 * NULL in case phonon internals are not initialized.
 */
complex128_t* diverge_phonons_get_g( diverge_model_t* m );

#ifdef __cplusplus
}
#endif
