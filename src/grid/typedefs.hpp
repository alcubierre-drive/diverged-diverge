/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include <vector>
#include <array>
#include <string>
#include <complex>
#include <iostream>
#include <sstream>

#include "../diverge_Eigen3.hpp"
#include "../misc/mpi_functions.h"

namespace grid {

using std::vector;
using std::array;
using std::string;
using std::stringstream;

template<typename T>
double take_time( T x ) {
    double tick = diverge_mpi_wtime();
    x();
    double tock = diverge_mpi_wtime();
    return tock - tick;
}

static inline double absmax( const complex128_t* ary, index_t cnt ) {
    double result = 0.0;
    #pragma omp parallel for reduction(max:result) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<cnt; ++i)
        result = std::max(result, std::abs(ary[i]));
    return result;
}

}
