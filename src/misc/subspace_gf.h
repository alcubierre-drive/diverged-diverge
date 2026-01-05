/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/** opaque typedef for hanlde used in subspace Green's functions */
typedef struct subspace_gfill_multi_t subspace_gfill_multi_t;

/**
 * initialize the :c:type:`subspace_gfill_multi_t` handle, which must be free'd
 * using :c:func:`subspace_gfill_multi_free`.
 *
 * :param double* E: dispersion array of the *full* model
 * :param complex128_t* U: orbital to band transformation of the *full* model
 * :param index_t nk: number of kpoints (``kdimtot(nk, nkf)``)
 * :param const index_t* orbs: selection of orbital/spin indices to construct
 *                             subspace for
 * :param index_t n_orbs: size of subspace
 * :param index_t nb: number of bands in the *full* model
 * :param index_t b_per_bb: number of bands to consider in a single CUDA thread
 *                          (for the hand-written GEMM). Note that the maximum
 *                          number of threads per block is not checked against,
 *                          so make this large enough to accomodate
 *                          ``dim3(nb/b_per_bb, n_orbs, n_orbs)`` threads per
 *                          block.
 */
subspace_gfill_multi_t* subspace_gfill_multi_init( double* E, complex128_t* U,
        index_t nk, const index_t* orbs, index_t n_orbs, index_t nb, index_t b_per_bb );

/**
 * free the :c:type:`subspace_gfill_multi_t` handle.
 */
void subspace_gfill_multi_free( subspace_gfill_multi_t* m );

/**
 * calculate subspace GFs using the handle obtained with
 * :c:func:`subspace_gfill_multi_init`. Safe to call multiple times, but not
 * safe to use in a scenario where the target buffer resides in shared memory.
 *
 * Example:
 *
 * .. sourcecode:: C
 *
 *  static greensfunc_op_t subspace_gf( const diverge_model_t* model, complex128_t Lambda, complex128_t* buf );
 *
 *  int main(int argc, char** argv) {
 *      diverge_init( &argc, &argv );
 *
 *      // some initialization, other code, ...
 *
 *      diverge_model_t* fullmodel = ...; // set up the full model for kinetics
 *      diverge_model_t* subspace = ...; // set up the subspace model for interactions
 *
 *      index_t orbital_selection[] = { 0, 1, 5, 6 }; // choice of orbitals
 *      index_t n_orbital_selection = sizeof(orbital_selection)/sizeof(orbital_selection[0]);
 *
 *      index_t b_per_bb = 3; // adjust accordingly
 *      // before calling the function below, diverge_model_internals_common()
 *      // must have been called on fullmodel
 *      subspace_gfill_multi_t* subspace_gf_data = subspace_gfill_multi_init(
 *          diverge_model_internals_get_E(fullmodel),
 *          diverge_model_internals_get_U(fullmodel),
 *          kdimtot(fullmodel->nk, fullmodel->nkf),
 *          orbital_selection,
 *          n_orbital_selection,
 *          fullmodel->n_orb * fullmodel->n_spin,
 *          b_per_bb
 *      );
 *
 *      subspace->data = subspace_gf_data;
 *      subspace->gfill = &subspace_gf;
 *
 *      // do something with the subspace model
 *
 *      subspace_gfill_multi_free( subspace_gf_data );
 *      diverge_finalize();
 *  }
 *
 *  static greensfunc_op_t subspace_gf( const diverge_model_t* model, complex128_t Lambda, complex128_t* buf ) {
 *      subspace_gfill_multi_exec( model->data, buf, Lambda );
 *      return greensfunc_op_cpu;
 *  }
 */
void subspace_gfill_multi_exec( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda );

/**
 * same as :c:func:`subspace_gfill_multi_exec` except for an additional chemical
 * potential that is subtracted from the eigenvalues. Useful for GPU
 * calculations, because there, the eigenvalue buffer is copied to the device
 * and *not* automatically updated.
 */
void subspace_gfill_multi_exec_mu( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda, double mu_add );

#ifdef __cplusplus
}
#endif
