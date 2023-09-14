/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void batched_eigen( int* devices, int ndevices, complex128_t* H, complex128_t* E, index_t dim, index_t Num );
void batched_eigen_r( int* devices, int ndevices, complex128_t* H, double* E, index_t dim, index_t Num );
// if devices == NULL, ndevices serves as control parameter:
//  · for ndevices ==  0: do CPU eigensolution
//  · for ndevices == -1: do GPU eigensolution with automatic device find
//  · for ndevices == -2: do GPU eigensolution with automatic device find &
//                        automatic chunking
// if devices != NULL and ndevices < 1: do autosetup as if devices were NULL

void batched_eigen_set_nchunks( int nchunks );
int batched_eigen_get_nchunks( void );

void batched_eigen_shut_up( int silence );

#ifdef __cplusplus
}
#endif
