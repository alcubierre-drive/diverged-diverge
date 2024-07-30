#pragma once

#include "../diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

greensfunc_op_t diverge_largemat_gf( const diverge_model_t* model, complex128_t Lambda, complex128_t* buf );

#ifdef __cplusplus
}
#endif
