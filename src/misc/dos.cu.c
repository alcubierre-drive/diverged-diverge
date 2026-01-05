/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "dos.cu.h"

#ifndef USE_CUDA

void ldos_kernel_gpu( const double* E, index_t nE, const complex128_t* U, index_t nos,
        const double* omega, index_t n_omega, double eta, double window, double* ldos ) {
    (void)E; (void)nE; (void)U; (void)nos; (void)omega; (void)n_omega; (void)eta; (void)window; (void)ldos;
    mpi_err_printf( "CUDA NOT COMPILED IN\n" );
}

#endif // USE_CUDA
