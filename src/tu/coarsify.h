#pragma once

#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct greensfunc_coarse_t greensfunc_coarse_t;
greensfunc_coarse_t* greensfunc_coarse_init( const index_t* nk, const index_t* nkf, index_t nb, int dim );
void greensfunc_coarse_free( greensfunc_coarse_t* g );
complex128_t* greensfunc_coarse_calc( greensfunc_coarse_t* g, const gf_complex_t* G_fine );

#ifdef __cplusplus
}
#endif
