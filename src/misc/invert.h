/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void matrix_invert( const double* A, index_t sz, double* Ainv );
void cmatrix_invert( const complex128_t* A, index_t sz, complex128_t* Ainv );

#ifdef __cplusplus
}
#endif
