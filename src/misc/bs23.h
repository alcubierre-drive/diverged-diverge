#pragma once

#include <stdint.h>
typedef int64_t index_t;

typedef struct bs23 {
    index_t ydim;
    double *yn, *k1, *k2, *k3; // we require 4 buffers for the BS integrator
    int has_k1;
} bs23_t;

// dy/dt = k = f(t,y)
typedef void (bs23_fun_t)( double t, double* y, double* k, index_t num, void* userdata );

double bs23_step( bs23_t* step, bs23_fun_t* fun, double t, double dt, void* userdata );
