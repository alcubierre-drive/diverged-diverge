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

typedef struct tu_selfenergy_t tu_selfenergy_t;
typedef struct tu_multiloop_correction_t tu_multiloop_correction_t;
tu_multiloop_correction_t* tu_multiloop_correction_init( diverge_model_t* model, const complex128_t* self_en );
void tu_multiloop_correction_calc( tu_multiloop_correction_t* m, double Lambda,
        const complex128_t* Pch, const complex128_t* Cch, const complex128_t* Dch,
        const complex128_t* dPch, const complex128_t* dDch, tu_selfenergy_t* self,
        complex128_t* out ); // out += correction, not scaled by dLambda
void tu_multiloop_correction_free( tu_multiloop_correction_t* m );

#ifdef __cplusplus
}
#endif
