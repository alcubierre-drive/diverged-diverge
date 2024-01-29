/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"
#include "misc/merge_orb_w_rs_symm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

// Function: diverge_generate_symm_maps
// generates symmetry mappings needed to perform the tasks below
void diverge_generate_symm_maps( diverge_model_t* model );

// Function: diverge_symmetrize_2pt_coarse
// symmetrizes coarse mesh 2pt function stored in buf. returns maximum
// difference between symmetrized and un-symmetrized buffer
//
// Parameters:
// model - the divERGe model
// buf - buffer to symmetrize, length (nk, nb, nb) with nb = n_orb * n_spin
// aux - auxiliary buffer that may be either NULL or of the same size as buf
double diverge_symmetrize_2pt_coarse( diverge_model_t* model, complex128_t* buf, complex128_t* aux );

// Function: diverge_symmetrize_2pt_fine
// symmetrizes fine mesh 2pt function stored in buf. returns maximum difference
// between symmetrized and un-symmetrized buffer
//
// Parameters:
// model - the divERGe model
// buf - buffer to symmetrize, length (nkf, nb, nb) with nb = n_orb * n_spin
// aux - auxiliary buffer that may be either NULL or of the same size as buf
double diverge_symmetrize_2pt_fine( diverge_model_t* model, complex128_t* buf, complex128_t* aux );

// Function: diverge_symmetrize_greens
// symmetrizes 2pt function of type <gf_complex_t> on a fine mesh (as the
// Green's function).  returns maximum difference between symmetrized and
// un-symmetrized buffer.
//
// Parameters:
// model - the divERGe model
// buf - buffer to symmetrize, length (nkf, nb, nb) with nb = n_orb * n_spin
// aux - auxiliary buffer that may be either NULL or of the same size as buf
double diverge_symmetrize_greens( diverge_model_t* model, gf_complex_t* buf, gf_complex_t* aux );

// Function: diverge_symmetrize_mom_coarse
// symmetrizes coarse mesh momentum space function (no orbital/spin indices)
// stored in buf. returns maximum difference between symmetrized and
// un-symmetrized buffer
//
// Parameters:
// model - the divERGe model
// buf - buffer of size (nk, sub)
// sub - number of internal indices (e.g. bands). all indices belonging to the
//       sub dimension are symmetrized
// aux - auxiliary buffer that may be either NULL or of the same size as buf
double diverge_symmetrize_mom_coarse( diverge_model_t* model, double* buf, index_t sub, double* aux );

// Function: diverge_symmetrize_mom_fine
// symmetrizes fine mesh momentum space function (no orbital/spin indices)
// stored in buf.returns maximum difference between symmetrized and
// un-symmetrized buffer
//
// Parameters:
// model - the divERGe model
// buf - buffer of size (nkf, sub)
// sub - number of internal indices (e.g. bands). all indices belonging to the
//       sub dimension are symmetrized
// aux - auxiliary buffer that may be either NULL or of the same size as buf
double diverge_symmetrize_mom_fine( diverge_model_t* model, double* buf, index_t sub, double* aux );

#ifdef __cplusplus
}
#endif
