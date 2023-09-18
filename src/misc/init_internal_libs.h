/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

void init_internal_libs( void );
void finalize_internal_libs( void );

// TODO(LK) move declarations and definitions somewhere else
void diverge_force_thread_limit( int nthr );
int diverge_omp_num_threads();

