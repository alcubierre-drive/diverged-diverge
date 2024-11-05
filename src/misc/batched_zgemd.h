/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * calculate batched 'zgemd' product
 *
 * :math:`O = U * \mathrm{diag}(E) * U^\dagger * \alpha + O * \beta`
 *
 * for the arrays out( num, N, N ) (:math:`O`), U( num, N, N ), E( num, N )
 *
 * .. note:: currently, no GPU version is implemented.
 */
void batched_zgemd_r( complex128_t* out, complex128_t* U, double* E,
        complex128_t alpha, complex128_t beta, index_t N, index_t num );

#ifdef __cplusplus
}
#endif
