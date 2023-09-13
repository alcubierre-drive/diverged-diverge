/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "diverge_interface.hpp"
#include "global.hpp"
#include "vertex.hpp"
#include "projection_wrapper.hpp"
#include "projection_implementation.hpp"



inline void add_to_all_frq(complex128_t* const to, const complex128_t* const inc, uint nb,
                            uint rest);

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
