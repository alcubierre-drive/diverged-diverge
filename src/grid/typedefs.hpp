/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../misc/mpi_functions.h"

#include <vector>
#include <array>
#include <string>

namespace grid {

using std::vector;
using std::array;
using std::string;

#define TICK_TOCK( output, ... ) { \
    double tick = diverge_mpi_wtime(); \
    __VA_ARGS__; \
    double tock = diverge_mpi_wtime(); \
    output += tock - tick; \
}

static inline double absmax( const complex128_t* ary, index_t cnt ) {
    double result = 0.0;
    #pragma omp parallel for reduction(max:result) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<cnt; ++i)
        result = std::max(result, std::abs(ary[i]));
    return result;
}

}
