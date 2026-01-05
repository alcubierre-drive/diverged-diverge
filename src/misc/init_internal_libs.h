/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void init_internal_libs( void );
void finalize_internal_libs( void );

int diverge_has_init_internal_libs( void );

void diverge_internal_force_thread_limit( int nthr );
int diverge_internal_omp_num_threads( void );
int diverge_internal_fftw_num_threads( void );

int environ_tu_symcheck_mpi( void );
int environ_symcheck_mpi( void );

#ifdef __cplusplus
}
#endif
