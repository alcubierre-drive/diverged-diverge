/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

void init_internal_libs( void );
void finalize_internal_libs( void );

// get the number of threads to use in diverge's OMP loops
int diverge_omp_num_threads( void );

// limit the number of threads for *every* OMP/FFTW parallel region; except for
// cases where the number is specifically set higher.
void diverge_force_thread_limit( int nthr );

