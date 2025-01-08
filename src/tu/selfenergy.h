/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_symmetrize.h"
#include "diverge_interface.h"

#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct selfenergy_t selfenergy_t;

selfenergy_t* selfenergy_init( void* vertex_ptr );
void selfenergy_free( selfenergy_t* s );

void selfenergy_flow( selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, const complex128_t* C_in,
        const complex128_t* D_in, const complex128_t* GF,
        const double prefac_sub );
void selfenergy_fft_interpolate( selfenergy_t* s, complex128_t* self_en );
// needed for tests
fftw_plan selfenergy_get_ifft_self_ip( selfenergy_t* s );

#ifdef __cplusplus
}
#endif
