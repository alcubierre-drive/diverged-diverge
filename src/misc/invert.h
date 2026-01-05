/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void matrix_invert( const double* A, index_t sz, double* Ainv );
void cmatrix_invert( const complex128_t* A, index_t sz, complex128_t* Ainv );
void cfmatrix_invert_F( const gf_complex_t* A, index_t sz, gf_complex_t* Ainv );
complex128_t cmatrix_determinant( const complex128_t* A, index_t sz );

#ifdef __cplusplus
}
#endif
