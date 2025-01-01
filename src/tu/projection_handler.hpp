/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "diverge_interface.h"
#include "vertex.hpp"
#include "projection_wrapper.hpp"
#include "projection_implementation.hpp"

void P_projection(complex128_t* proj_vertex, Vertex* const vertex,
                const Projection* const proj);

void D_projection(complex128_t* proj_vertex, Vertex* const vertex,
                const Projection* const proj);

void C_projection(complex128_t* proj_vertex, Vertex* const vertex,
                const Projection* const proj);

void P_projection_no_GPU(complex128_t* proj_vertex, Vertex* const vertex,
                const Projection* const proj);

void D_projection_no_GPU(complex128_t* proj_vertex, Vertex* const vertex,
                const Projection* const proj);

void C_projection_no_GPU(complex128_t* proj_vertex, Vertex* const vertex,
                const Projection* const proj);
