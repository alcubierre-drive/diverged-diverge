#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ldos_kernel_gpu( const double* E, index_t nE, const complex128_t* U, index_t nos,
        const double* omega, index_t n_omega, double eta, double window, double* ldos );
void ldos_kernel_gpu_setsort( int gpusort );

#ifdef __cplusplus
}
#endif
