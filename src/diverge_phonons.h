/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"
#include "misc/wigner.h"

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration
typedef struct rs_hopping_t rs_hopping_t;
typedef struct diverge_model_t diverge_model_t;
typedef struct diverge_flow_step_t diverge_flow_step_t;

/**
 * electron-phonon vertex structure in real space. Some member types are chosen
 * in a specific way to yield a type that's as large as :c:type:`rs_vertex_t`.
 * **This equivalence is important and must not be violated!**
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
    /**
     * mode selection for the phononic 'transfer' vector. If ``mode == '0'`` use
     * the lattice vector FFT (:c:member:`RA`). Otherwise the phonon is situated
     * at the floating point position :c:member:`RA_pos` (in crystal
     * coordinates; the real position is :math:`R_A^\mathrm{pos} \cdot L` with
     * :math:`L` the lattice vector matrix) and an explicit floating point
     * fourier transform (i.e. not an FFT) is performed for the respective
     * component.
     */
    char mode;
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
    int32_t RA[3];
    /**
     * phononic 'transfer' vector (fourier transform of :math:`\boldsymbol{q}`),
     * but in crystal coordinates (i.e. with the possibility to use fractional
     * coordinates). Use this when :c:member:`mode` is set to something nonzero.
     * Otherwise use the lattice vector position :c:member:`RA` (and therefore
     * also an FFT!)
     */
    float RA_pos[3];

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
 * certain mode we have *imaginary* frequencies. If used, the returned pointer
 * must be interpreted as bit-array, which may also be retrieved with
 * :c:func:`diverge_phonons_get_ifrq`
 */
void* diverge_phonons_apply_sgnsqrt( diverge_model_t* m );

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
 *
 * ``if (refined)``, return the *refined* electron-phonon coupling buffer, size
 * being (n_refine, n_spin, n_spin, n_orbff, nO). The size of refinement can be
 * returned with :c:func:`diverge_phonons_get_refine`.
 */
complex128_t* diverge_phonons_get_g( diverge_model_t* m, int refined );

/** refinement structure, simple tuple of (kf_index, weight). */
typedef struct {
    /** index of the refined point in the fine mesh */
    index_t kf_idx;
    /**
     * weight of the refined point. All refined points are weighted such that
     * the sum of the weights is one.
     */
    double weight;
} refinement_t;

/**
 * returns the refinement (or NULL) of the phonon-Gamma point vertex.
 * ``n_refine`` is set to the length of the refinement.
 */
refinement_t* diverge_phonons_get_refine( diverge_model_t* m, index_t* n_refine );

/**
 * Changes the phonon propagator regulator :math:`\eta` for the
 * :math:`iq_0 \to 0` limit:
 *
 * .. math::
 *
 *  D_\nu(\boldsymbol{q}, iq_0) = \mathrm{Re}\bigg[ \frac{1}{(iq_0)^2 - i\eta - \omega_{\nu}(\boldsymbol{q})^2} \bigg]
 *
 * This is highly relevant for the application of the theory to acoustic modes.
 * The function returns the old value for :math:`\eta`. If ``qnan_isnan(eta)``
 * the regulator is left unchanged; this serves the purpose of allowing to look
 * up the current value.
 */
double diverge_phonons_change_regulator( double eta );

/**
 * apply an acoustic sum rule to the electron phonon coupling vertices. assumes
 * three dimensional phonon "orbitals", i.e., (o,d) ordering of the OA index
 * (with o=0...n_sites-1 and d=0..2). operates on *all* vertex elements
 * irrespective of their :c:member:`ph_vertex_t.mode` setting. Ensures that
 *
 * .. math::
 *
 *  \sum_{\boldsymbol{R}_A, o_A} g_{xy}(\boldsymbol{R}_A, (o_A, d_A)) = 0
 *
 * with :math:`x,y` electronic indices (orbital, spin, formfactor) and
 * :math:`o_A, d_A` the phononic sites/directions.
 *
 * :param ph_vertex_t* vtx: vertex array, must be allocated via
 *      calloc/malloc/realloc (or their divERGe wrappers) because we use realloc
 *      on this pointer internally!
 * :param index_t* n_vtx: number of hoppings, points to new number of
 *      vertices on return.
 *
 * :returns ph_vertex_t*: vertex array, reallocated.
 */
ph_vertex_t* diverge_phonons_asr_vertex( ph_vertex_t* vtx, index_t* n_vtx );

/**
 * change implementation behavior of :c:func:`diverge_phonons_asr_vertex` to an
 * implementation that sorts all vertex elements and then looks at
 * displacements/counts of unequal 'formfactor' components in order to sum them
 * up and thus executes in parallel. should be faster for very large models, but
 * slower for small models. returns previous mode (in order to save/restore
 * state).
 */
int diverge_phonons_asr_vertex_set_sorted_mode( int sorted );


/**
 * enable refinement of the Gamma point D channel vertex value, must be called
 * prior to :c:func:`diverge_model_internals_tu_phonons`.
 *
 * :param diverge_model_t* el: electronic model
 * :param int enable_refinement: whether to enable refinement or not
 * :param double eps_mesh: This value is considered zero for the construction of
 *      the refinement. the previous value is returned. setting a non-default
 *      value can be avioded by passing a quiet nan (:c:func:`qnan_gen()`).
 *
 * :returns double: previous value for ``eps_mesh``, in order to have some
 *      introspection.
 */
double diverge_phonons_enable_refinement( diverge_model_t* el, int enable_refinement, double eps_mesh );

/**
 * typedef for custom electron-phonon vertex generation. TUFRG details have to
 * be either put into ctx or obtained from step. should generate the
 * Lambda-retarded vertex on this rank's IBZ wedge into buf
 * (n_ibz,n_orbff,n_orbff).
 */
typedef void (*diverge_phonon_vertex_generator_t)( diverge_flow_step_t* step,
        double Lambda, complex128_t* buf, void* ctx );

/**
 * allow for custom electron-phonon vertex generation (only for non-refined
 * vertices!). reset to default by passing NULL for V (+the other variables if
 * you wish).
 */
void diverge_phonons_set_vertex_generator( diverge_model_t* el,
        diverge_phonon_vertex_generator_t V, diverge_flow_step_t* step, void* ctx );

#ifdef __cplusplus
}
#endif
