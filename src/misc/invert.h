/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/** invert the (sz×sz) matrix ``A`` and output result to ``Ainv`` */
void matrix_invert( const double* A, index_t sz, double* Ainv );
/** same as :c:func:`matrix_invert` but for complex numbers */
void cmatrix_invert( const complex128_t* A, index_t sz, complex128_t* Ainv );
/**
 * same as :c:func:`cmatrix_invert` but for FORTRAN ordering and---depending on
 * the compilation settings---different precision.
 */
void cfmatrix_invert_F( const gf_complex_t* A, index_t sz, gf_complex_t* Ainv );
/** calculate the determinant of a complex square matrix */
complex128_t cmatrix_determinant( const complex128_t* A, index_t sz );

#ifdef __cplusplus
}
#endif
