/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_symmetrize.h"
#include "diverge_interface.h"
#include "vertex.h"

#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_selfenergy_t tu_selfenergy_t;

tu_selfenergy_t* tu_selfenergy_init( tu_vertex_t* vertex_ptr );
void tu_selfenergy_free( tu_selfenergy_t* s );

void tu_selfenergy_flow( tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, const double prefac_sub, const double lambda );
void tu_selfenergy_fft_interpolate( tu_selfenergy_t* s, complex128_t* self_en );
// needed for tests
fftw_plan tu_selfenergy_get_ifft_self_ip( tu_selfenergy_t* s );

#ifdef __cplusplus
}
#endif
