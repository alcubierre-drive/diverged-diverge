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

typedef struct Projection Projection;
typedef struct Vertex Vertex;

void P_projection(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj);
void D_projection(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj);
void C_projection(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj);

void P_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj);
void D_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj);
void C_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj);

#ifdef __cplusplus
}
#endif
