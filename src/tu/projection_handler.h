/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "diverge_interface.h"
#include "projection_implementation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct projection_t projection_t;
typedef struct tu_vertex_t tu_vertex_t;

void P_projection(complex128_t* proj_vertex, const tu_vertex_t* vertex, const projection_t* proj);
void D_projection(complex128_t* proj_vertex, const tu_vertex_t* vertex, const projection_t* proj);
void C_projection(complex128_t* proj_vertex, const tu_vertex_t* vertex, const projection_t* proj);

void P_projection_no_GPU(complex128_t* proj_vertex, const tu_vertex_t* vertex, const projection_t* proj);
void D_projection_no_GPU(complex128_t* proj_vertex, const tu_vertex_t* vertex, const projection_t* proj);
void C_projection_no_GPU(complex128_t* proj_vertex, const tu_vertex_t* vertex, const projection_t* proj);

#ifdef __cplusplus
}
#endif
