/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "diverge_interface.h"
#include "vertex.h"
#include "propagator.h"
#include "selfenergy.h"
#include "projection.h"
#include "projection_handler.h"
#include "../misc/batched_gemms.h"

#include <string>
#include <vector>

using std::string;
using std::vector;

void vertex_product(complex128_t* result, const complex128_t* const  B,
                    const complex128_t* const C, complex128_t* helper,
                    char* GPUbuffer[32][3] ,
                    const index_t productsize, const index_t nbatch,
                    const double prefac = 1.0);

class tu_stepper{
public:
    tu_stepper(projection_t*, tu_prop_t* loop, Vertex* Vertex);
    vector<double> tu_step_euler(Vertex* inout, double lambda, double dlambda);

    ~tu_stepper(){
        Vertex_free(internal_vertex);
        if(channel_selection[3]) {
            selfenergy_free(self);
            free(coarseGF);
        }
    }
    selfenergy_t* self;
    Vertex* internal_vertex;

    vector<double> get_timings( void ) const {
        vector<double> result = { w_time_GF, w_time_loop, w_time_projection,
            w_time_product, w_time_add_find_max, w_time_self, w_time_full};
        int ntimings = 0;
        tu_prop_timing_t* timings = tu_prop_get_extra_timings( loop, &ntimings );
        for (int i=0; i<ntimings; ++i) result.push_back(timings[i].t);
        return result;
    }
    vector<string> get_timings_descr( void ) const {
        vector<string> result = { "Greens function", "loop", "projection",
            "contract", "max and add", "self", "full step" };
        int ntimings = 0;
        tu_prop_timing_t* timings = tu_prop_get_extra_timings( loop, &ntimings );
        for (int i=0; i<ntimings; ++i) result.push_back(timings[i].name);
        return result;
    }
    index_t get_num_calls( void ) const {
        return num_calls;
    }

    double w_time_GF = 0.;
    double w_time_full = 0.;
    double w_time_loop = 0.;
    double w_time_product = 0.;
    double w_time_projection = 0.;
    double w_time_add_find_max = 0.;
    double w_time_self = 0.;


private:
    tu_prop_t* loop;
    projection_t* proj;
    bool channel_selection[4];
    double norm[3];
    index_t prodsize;
    index_t nk;
    index_t my_nk;
    index_t num_calls = 0;
    complex128_t* coarseGF;
    bool tu_selfenergy_flow;
};
