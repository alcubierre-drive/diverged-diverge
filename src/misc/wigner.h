#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;
typedef struct ph_vertex_t ph_vertex_t;

/**
 * bit-or of this constant and
 * :c:macro:`DIVERGE_WIGNER_CONFIG_APPLY_ASR_VERTICES` to set which ASR to apply
 * for model read by :c:func:`diverge_phonons_read_ifc` and
 * :c:func:`diverge_phonons_read_ifc_and_elph`
 */
#define DIVERGE_WIGNER_CONFIG_APPLY_ASR_HOPPINGS 0x1
/**
 * bit-or of this constant and
 * :c:macro:`DIVERGE_WIGNER_CONFIG_APPLY_ASR_HOPPINGS` to set which ASR to apply
 * for model read by :c:func:`diverge_phonons_read_ifc_and_elph`
 */
#define DIVERGE_WIGNER_CONFIG_APPLY_ASR_VERTICES 0x2
/**
 * bit-or this constant with the other ones to set how to apply ASR for
 * :c:func:`diverge_phonons_read_ifc_and_elph`. This constant only sets the
 * algorithm to the sorted one, but doesn't enable the ASR, i.e., it has to be
 * used in conjunction with :c:macro:`DIVERGE_WIGNER_CONFIG_APPLY_ASR_VERTICES`.
 */
#define DIVERGE_WIGNER_CONFIG_APPLY_ASR_VERTICES_SORTED 0x4
/**
 * bit-or this constant with the other ones to use the wigner data's #orbitals
 * instead of automatic detection
 */
#define DIVERGE_WIGNER_CONFIG_APPLY_ASR_NO_ORBDETECT 0x8

/**
 * configuration for reading wigner(.fmt) files. defaults in
 * :c:var:`diverge_wigner_config_default` (or
 * :c:func:`diverge_wigner_config_defaults_CPP`).
 */
typedef struct {
    /** number of supercells to look for WS data in, defaults to 2 */
    index_t nsc;
    /** sign structure of the lattice vectors, defaults to - */
    int sgn;
    /** minimum distance to consider 'the same', defaults to 1.e-7 */
    double eps;
    /** dimension of the model, defaults to -1, i.e., automatic */
    int dim;
    /**
     * apply the acoustic sum rule to hoppings/vertices (bit-or of
     * :c:macro:`DIVERGE_WIGNER_CONFIG_APPLY_ASR_HOPPINGS`,
     * :c:macro:`DIVERGE_WIGNER_CONFIG_APPLY_ASR_VERTICES`, and
     * :c:macro:`DIVERGE_WIGNER_CONFIG_APPLY_ASR_VERTICES_SORTED`)
     */
    int apply_asr;
    /** apply supercell degeneracy map */
    int apply_ndegen;
    /** apply mass division */
    int apply_massdiv;
    /** apply energy rescale from rydberg to eV */
    int apply_rydberg;
} diverge_wigner_config_t;

#ifndef __cplusplus
/** defaults for :c:type:`diverge_wigner_config_t` */
static const diverge_wigner_config_t diverge_wigner_config_defaults = {
    .nsc = 2, .sgn = -1, .eps = 1.e-7, .dim = -1, .apply_asr = 0,
    .apply_ndegen = 1, .apply_massdiv = 1, .apply_rydberg = 1,
};
#endif
/**
 * defaults for :c:type:`diverge_wigner_config_t` (just a wrapper around
 * :c:var:`diverge_wigner_config_defaults` which is not available in C++)
 */
diverge_wigner_config_t diverge_wigner_config_defaults_CPP( void );

/** read force constants from an .ifc file. optionally provide non-default config in ``wc`` */
diverge_model_t* diverge_phonons_read_ifc( const char* fname, const diverge_wigner_config_t* wc );

/**
 * read electron-phonon coupling from a wigner.fmt file ``wigner`` and an
 * .epmatwp file ``epmatwp``.
 *
 * :param const char* wigner: ``wigner.fmt`` file name
 * :param const char* epmatwp: ``epmatwp`` file name
 * :param const double* masses: The atom masses (as provided by
 *      :c:func:`diverge_phonons_model_get_masses` given a model built with
 *      :c:func:`diverge_phonons_read_ifc`, array of shape n_orb) that *must* be
 *      provided.
 * :param double maxrel: relative maximum value cutoff (in real space). Let
 *      :math:`M` be the maximum of the electron-phonon coupling and :math:`R`
 *      this cutoff. Then all elements :math:`g_{Rx}^{R'ab}` that fulfill
 *      :math:`|g_{Rx}^{R'ab}| < RM` are discarded.
 * :param index_t* p_vtx_sz: length of the returned array.
 * :param int apply_asr: apply the vertex acoustic sum rule (cf.
 *      :c:func:`diverge_phonons_asr_vertex`).
 *
 * :returns: array of type :c:type:`ph_vertex_t` with the real-space elph
 *      elements, shape (``*p_vtx_sz``,).
 */
ph_vertex_t* diverge_phonons_read_elph( const char* wigner, const char* epmatwp,
        const double* masses, double maxrel, index_t* p_vtx_sz, int apply_asr );

/**
 * provide the memory region where atom masses are stored in a phonon model
 * created with :c:func:`diverge_phonons_read_ifc`.
 */
double* diverge_phonons_model_get_masses( diverge_model_t* m );

/**
 * convenience wrapper around :c:func:`diverge_phonons_read_ifc` and
 * :c:func:`diverge_phonons_read_elph` that reads the non-interacting part of
 * the model as well as the electron-phonon coupling---which is stored in
 * :c:member:`diverge_model_t.vert` and :c:member:`diverge_model_t.n_vert`.
 */
diverge_model_t* diverge_phonons_read_ifc_and_elph( const char* ifc,
        const char* wigner, const char* epmatwp, double maxrel,
        const diverge_wigner_config_t* wc );

#ifdef __cplusplus
}
#endif
