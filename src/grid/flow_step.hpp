/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "typedefs.hpp"

typedef struct grid_loop_t grid_loop_t;

namespace grid {

class VertexMemory;

class FlowStep {
    public:
        FlowStep( VertexMemory* V, grid_loop_t* L, bool P = true, bool C = true, bool D = true );
        // returns Pmax, Cmax, Dmax, Vmax, LPmax, LDmax
        vector<double> operator()( double Lambda, double dLambda );
        void print_timings( void );
        bool is_SU2( void ) { return _SU2; }
        void set_D_channel_degeneracy( double nrm );

        vector<double> get_timings( void ) const {
            return { _reorder_time, _loop_time, _loop_symm_time,
                     _loop_gemm_time, _contract_time, _max_time };
        }
        vector<string> get_timings_descr( void ) const {
            return { "reorder", "loop", "loop-symm", "loop-gemm", "contract", "max" };
        }
        index_t get_num_calls( void ) const {
            return _num_calls;
        }
    private:
        VertexMemory* _V;
        grid_loop_t* _L;
        bool _P, _C, _D;
        bool _SU2;
        double _kfnorm;

        complex128_t* _LC = nullptr;
        complex128_t* _LP = nullptr;

        void _set_dv_zero();
        double _get_dv( char channel, double Lambda, double dLambda );
        double _add_dv_to_v();

        array<double,5> _norms;
        double _reorder_time = 0;
        double _loop_time = 0;
        double _loop_symm_time = 0;
        double _loop_gemm_time = 0;
        double _contract_time = 0;
        double _max_time = 0;
        int _num_calls = 0;
};

}
