/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void gemm( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t M, index_t N, index_t K, complex128_t alpha,
        complex128_t beta);

#ifdef __cplusplus
}
#endif
