#ifndef DIVERGE_SKIP_TESTS

#pragma once
#include "../../src/tu/vertex.h"
#include "../../src/tu/projection.h"
#include "../../src/diverge_model.h"
#include "../helper_functions.hpp"

void t_P_projection(complex128_t* proj_vertex,
                    const tu_vertex_t* const vertex,
                    const diverge_model_t* const model);

void t_C_projection(complex128_t* proj_vertex,
                    const tu_vertex_t* const vertex,
                    const diverge_model_t* const model);

void t_D_projection(complex128_t* proj_vertex,
                    const tu_vertex_t* const vertex,
                    const diverge_model_t* const model);

#endif // DIVERGE_SKIP_TESTS
