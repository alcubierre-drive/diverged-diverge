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
rs_hopping_t* diverge_read_W90_C( const char* fname, index_t n_spin, index_t* len );

#ifdef __cplusplus
} // extern "C"

vector<rs_hopping_t> diverge_read_W90( const char* fname, index_t n_spin );
#endif
