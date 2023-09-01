#pragma once
#include "../../src/tu/global.hpp"
#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection_wrapper.hpp"
#include "../../src/diverge_model.h"
#include "../helper_functions.hpp"

void t_P_projection(complex128_t* proj_vertex,
                    const Vertex* const vertex,
                    const diverge_model_t* const model);

void t_C_projection(complex128_t* proj_vertex,
                    const Vertex* const vertex,
                    const diverge_model_t* const model);

void t_D_projection(complex128_t* proj_vertex,
                    const Vertex* const vertex,
                    const diverge_model_t* const model);
