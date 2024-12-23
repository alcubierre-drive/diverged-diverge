/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "diverge_interface.hpp"
#include "global.hpp"
#include "vertex.hpp"
#include "propagator.hpp"
#include "selfenergy.hpp"
#include "projection_wrapper.hpp"
#include "projection_handler.hpp"
#include "../misc/batched_gemms.h"

#include <string>
using std::string;

void vertex_product(complex128_t* result, const complex128_t* const  B,
                    const complex128_t* const C, complex128_t* helper,
                    char* GPUbuffer[32][3] ,
                    const index_t productsize, const index_t nbatch,
                    const double prefac = 1.0);

class tu_stepper{
public:
    tu_stepper(Projection*, tu_loop_t* loop, Vertex* Vertex);
    vector<double> tu_step_euler(Vertex* inout, double lambda, double dlambda);

    ~tu_stepper(){
        delete internal_vertex;
        if(channel_selection[3]) {
            delete self;
            free(coarseGF);
        }
    }
    selfEnergy* self;
    Vertex* internal_vertex;

    vector<double> get_timings( void ) const {
        vector<double> result = { w_time_GF, w_time_loop, w_time_projection,
            w_time_product, w_time_add_find_max, w_time_self, w_time_full};
        for (const double& t: loop->extra_timings()) result.push_back(t);
        return result;
    }
    vector<string> get_timings_descr( void ) const {
        vector<string> result = { "Greens function", "loop", "projection",
            "contract", "max and add", "self", "full step" };
        for (const string& s: loop->extra_timings_descr()) result.push_back(s);
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
    tu_loop_t* loop;
    Projection* proj;
    bool channel_selection[4];
    double norm[3];
    index_t prodsize;
    index_t nk;
    index_t my_nk;
    index_t num_calls = 0;
    complex128_t* coarseGF;
    bool tu_selfenergy_flow;
};
