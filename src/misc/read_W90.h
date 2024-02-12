/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus

#include <vector>
using std::vector;

extern "C" {
#endif

// forward declaration
typedef struct rs_hopping_t rs_hopping_t;

// Function: diverge_read_W90_C
// reads Wannier90's *_hr.dat file into divERGe's <rs_hopping_t> data structure.
//
// Parameters:
// fname - /path/to/material_hr.dat
// n_spin - default value 0 amounts to SU(2) symmetric model. if nspin != 0,
//         abs(nspin) = (S+1/2)*2 with S the physical spin (i.e., for S=1/2 we
//         have abs(nspin)=2). the sign determines whether the spin index is the
//         one which increases memory slowly (negative) or fast (positive) in
//         the W90 file. Note that the divERGe convention is always (s,o), i.e.
//         spin indices increasing memory fast ('outer indices').
// n_orb - output for the maximum number of orbitals that were found in the
//         hr.dat file. not considered if NULL.
//
// Example:
// ===C/C++===
// diverge_model_t* model = diverge_model_init();
// // set some model parameters...
// model->SU2 = 1;
// model->n_spin = 1;
// model->hop = diverge_read_W90_C( "/path/to/material_hr.dat",
//                                  0, &model->n_hop, NULL );
// // do something with the hoppings here!
// ===========
rs_hopping_t* diverge_read_W90_C( const char* fname, index_t n_spin, index_t* len, index_t* n_orb );

#ifdef __cplusplus
} // extern "C"

// Function: diverge_read_W90
// C++ function corresponding to <diverge_read_W90_C> that facilitates the usage
// by returning an std::vector< <rs_hopping_t> > instead of messing with pointers.
vector<rs_hopping_t> diverge_read_W90( const char* fname, index_t n_spin, index_t* n_orb );
#endif
