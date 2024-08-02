/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Green's function generator that is faster than the default
 * :c:func:`diverge_greensfunc_generator_default` for systems with a significant
 * number of orbitals. Performs actual BLAS calls to GEMM instead of manually
 * passing through the (o,b,p) loop. Shared memory GFs as well as 32bit floating
 * point precision *not supported*.
 */
greensfunc_op_t diverge_largemat_gf( const diverge_model_t* model, complex128_t Lambda, complex128_t* buf );

#ifdef __cplusplus
}
#endif
