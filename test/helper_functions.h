#ifndef DIVERGE_SKIP_TESTS

#pragma once

#include "../src/diverge_common.h"
#include "../src/diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

bool check_hermicity(complex128_t* data, index_t linsize,index_t batches);
complex128_t trace_cd(complex128_t* data, index_t linsize, index_t batches);
double trace_d(double* data, index_t linsize, index_t batches);
complex128_t trace_batch_lin(complex128_t* data, index_t linsize, index_t batches);
bool check_equal_cd(complex128_t* data1, complex128_t* data2, index_t size, double errbnd/*=1e-12*/);
bool check_equal_d(double* data1, double* data2, index_t size, double errbnd/*=1e-12*/);
bool check_abs_sorted_equal_cd(complex128_t* data1, complex128_t* data2, index_t size);
bool check_abs_sorted_equal_d(double* data1, double* data2, index_t size);

complex128_t ff( const double* k, const double* r );
void fill_Rvec_from_vec(const diverge_model_t* model, index_t* R, double* Rvec);
void fill_Rvec(const diverge_model_t* model, double* Rvec);
void random_seed( uint64_t seed ); /* re-seed rng */
double random_dbl( void ); /*uniform -1...1*/
complex128_t random_cmplx( void ); /*uniform -1...1*/

#ifdef __cplusplus
}
#endif

#endif // DIVERGE_SKIP_TESTS
