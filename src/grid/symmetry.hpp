/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "reorder.h"
#include "../diverge_Eigen3.hpp"

#include <vector>
#include <array>

using std::vector;
using std::array;

#include "symmetry.h"

namespace grid {

class LoopSymmetrizer {
    public:
        LoopSymmetrizer( grid_symm_transform_t* s, index_t qdispl, index_t qcount );
        ~LoopSymmetrizer();
        complex128_t* operator()( complex128_t* loop_in, char chan );
        void setDebugMode() { _debug = true; }
        bool check( complex128_t* loop_in, char chan );
    private:
        int operator()( complex128_t* loop_in, complex128_t* loop_out, index_t sym, char chan );
        bool _can_transform;
        index_t _nsym;
        index_t _loop_size;
        bool _debug = false;
        grid_symm_comm_t *_comms_P, *_comms_C;
        vector<complex128_t> _send_buf;
        vector<complex128_t> _recv_buf;
        vector<complex128_t> _loop_buf;
};

}
