#pragma once

#include "../diverge_common.h"
#include "../diverge_model.h"
#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_prop_t tu_prop_t;

tu_prop_t* tu_prop_init( diverge_model_t* model );
void tu_prop_free( tu_prop_t* t );

// TODO these are missing
void tu_prop_ph_loop( tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu/*=NULL*/ );
void tu_prop_pp_loop( tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu/*=NULL*/ );
void tu_prop_GF_zero_T( tu_prop_t* t, complex128_t lambda, complex128_t* self_energy/*=NULL*/ );
void tu_prop_GF_for_self( tu_prop_t* t, complex128_t* coarseGF );

typedef struct {
    double t;
    char name[56];
} tu_prop_timing_t;
tu_prop_timing_t* tu_prop_get_extra_timings( const tu_prop_t* t, int* ntimings ); // dont free
void tu_prop_set_extra_timings( tu_prop_t* t, bool enabled );

#ifdef __cplusplus
}
#endif
