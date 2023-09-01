#pragma once

#include "../diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Function: diverge_mem_alloc_rs_hopping_t
// for python wrapper
rs_hopping_t* diverge_mem_alloc_rs_hopping_t( index_t num );

// Function: diverge_mem_alloc_rs_vertex_t
// for python wrapper
rs_vertex_t* diverge_mem_alloc_rs_vertex_t( index_t num );

// Function: diverge_mem_alloc_tu_formfactor_t
// for python wrapper
tu_formfactor_t* diverge_mem_alloc_tu_formfactor_t( index_t num );

// Function: diverge_mem_alloc_complex128_t
// for python wrapper
double* diverge_mem_alloc_complex128_t( index_t num );

// Function: diverge_mem_free
// for python wrapper, should not be required
void diverge_mem_free( void* addr );

#ifdef __cplusplus
}
#endif
