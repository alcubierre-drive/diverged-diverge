/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "flow_step.hpp"
#include "loops.hpp"
#include "vertex_memory.hpp"
#include "symmetry.hpp"

#include "../misc/batched_gemms.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>
#include <fstream>

namespace grid {

template<typename T>
double dont_take_time( T x ) {
    x();
    return 0;
}
#ifndef SKIP_FLOW_TIMING
#define elapsed_time( fun ) take_time( [&](){ fun; } )
#else // SKIP_FLOW_TIMING
#define elapsed_time( fun ) dont_take_time( [&](){ fun; } )
#endif // SKIP_FLOW_TIMING

static double kfnormgen( const double* kfweights, index_t nkf ) {
    double norm = 0.0;
    for (index_t kf=0; kf<nkf; ++kf) norm += kfweights[kf];
    return norm;
}

FlowStep::FlowStep( VertexMemory* V, Loops* L, LoopSymmetrizer* LS, bool P, bool C, bool D ) {
    _P = P;
    _C = C;
    _D = D;
    _SU2 = V->get_SU2();
    _V = V;
    _L = L;
    _LS = LS;
    _kfnorm = kfnormgen( L->get_kfweights(), L->get_nkf() );

    for (unsigned i=0; i<_norms.size(); ++i)
        _norms[i] = 1.0/(double)_L->get_nk()/_kfnorm;

    if (_SU2) {
        _norms[2] *= -2.0;
        if (V->was_sub_D_modified()) {
            array<bool,3> subD = V->get_sub_D();
            if (!subD[0]) _norms[2] = 0;
            if (!subD[1]) _norms[3] = 0;
            if (!subD[2]) _norms[4] = 0;
            mpi_log_printf( "enabled D channel diagrams: RPA(%i) VC1(%i) VC2(%i)\n", subD[0], subD[1], subD[2] );
        }
    } else {
        if (C != D)
            mpi_err_printf( "non-SU2 only uses D *and* C\n" );
        _norms[0] *=  0.5;
        _norms[2] *= -1.0;
    }
}

vector<double> FlowStep::operator()( double Lambda, double dLambda ) {
    _set_dv_zero();
    vector<double> result;
    result.reserve(6);
    if (_SU2) {
        double Pmax = 0;
        double Cmax = 0;
        double Dmax = 0;
        if (_P) Pmax = _get_dv( 'P', Lambda, dLambda );
        if (_C) Cmax = _get_dv( 'C', Lambda, dLambda );
        if (_D) Dmax = _get_dv( 'D', Lambda, dLambda );
        result.push_back(Pmax);
        result.push_back(Cmax);
        result.push_back(Dmax);
    } else {
        double Pmax = 0;
        double Dmax = 0;
        if (_P) Pmax = _get_dv( 'P', Lambda, dLambda );
        if (_D) Dmax = _get_dv( 'D', Lambda, dLambda );
        result.push_back(Pmax);
        result.push_back(Dmax);
        result.push_back(Dmax);
    }

    result.push_back( _add_dv_to_v() );
    _num_calls++;

    array<double,2> loopmax = _L->get_max();
    for (unsigned i=0; i<loopmax.size(); ++i) {
        result.push_back( loopmax[i] );
    }

    for (unsigned i=0; i<result.size(); ++i) {
        diverge_mpi_allreduce_double_max( MPI_IN_PLACE, result.data()+i, 1 );
    }

    return result;
}

void FlowStep::_set_dv_zero() {
    complex128_t* d_vertex = _V->vertex_buffers[1];
    index_t Vsize = _V->get_size();
    memset( (void*)d_vertex, 0, Vsize*sizeof(complex128_t) );
    _LP = nullptr;
    _LC = nullptr;
}

double FlowStep::_add_dv_to_v() {
    complex128_t* vertex = _V->vertex_buffers[0];
    complex128_t* d_vertex = _V->vertex_buffers[1];
    index_t Vsize = _V->get_size();
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<Vsize; ++i) vertex[i] += d_vertex[i];
    double Vmax = absmax( vertex, Vsize );
    return Vmax;
}

double FlowStep::_get_dv( char channel, double Lambda, double dLambda ) {
    index_t qsize = _L->get_qsize();
    index_t nk = _L->get_nk();
    index_t nb = _L->get_nb();
    complex128_t* v = _V->vertex_buffers[0];
    complex128_t* left = _V->vertex_buffers[2];
    complex128_t* right = _V->vertex_buffers[3];
    complex128_t* dv = _V->vertex_buffers[1];
    double vmax = 0.0;
    index_t Vsize = _V->get_size();
    VertexMemory& V = *_V;
    complex128_t* _LP_sym = nullptr;
    complex128_t* _LC_sym = nullptr;
    auto symmetrize_P = [&](){
        if (_LP_sym == nullptr) {
            if (_LS)
                _LP_sym = (*_LS)(_LP, 'P');
            else
                _LP_sym = _LP;
        }
    };
    auto symmetrize_C = [&](){
        if (_LC_sym == nullptr) {
            if (_LS)
                _LC_sym = (*_LS)(_LC, 'C');
            else
                _LC_sym = _LC;
        }
    };
    if (channel == 'P') {
        _loop_time +=      elapsed_time( if (!_LP) _LP = _L->X_P(Lambda); );
        _reorder_time +=   elapsed_time( V( v, left,  "D_to_P", false, 1.0 ) );
        _reorder_time +=   elapsed_time( V( v, right, "D_to_P", false, 1.0 ) );
        _loop_symm_time += elapsed_time( symmetrize_P() );
        _loop_gemm_time += elapsed_time( batched_gemm_vertex_loop( left, _LP_sym, qsize, nk, nb ) );
        _contract_time +=  elapsed_time( batched_gemm_overlapping( left, right, right, nk*nb*nb, 1.0, 0.0, qsize ) );
        _max_time +=       elapsed_time( vmax = absmax( right, Vsize ) * _norms[0] * dLambda );
        _reorder_time +=   elapsed_time( V( right, dv, "P_to_D", true, _norms[0]*dLambda ) );
    }
    if (channel == 'C') {
        if (!_SU2) return 0;
        _loop_time +=      elapsed_time( if(!_LC) _LC = _L->X_C(Lambda); );
        _reorder_time +=   elapsed_time( V( v, left,  "D_to_C", false, 1.0 ) );
        _reorder_time +=   elapsed_time( V( v, right, "D_to_C", false, 1.0 ) );
        _loop_symm_time += elapsed_time( symmetrize_C() );
        _loop_gemm_time += elapsed_time( batched_gemm_vertex_loop( left, _LC_sym, qsize, nk, nb ) );
        _contract_time +=  elapsed_time( batched_gemm_overlapping( left, right, right, nk*nb*nb, 1.0, 0.0, qsize ) );
        _max_time +=       elapsed_time( vmax = absmax( right, Vsize ) * _norms[1] * dLambda );
        _reorder_time +=   elapsed_time( V( right, dv, "C_to_D", true, _norms[1]*dLambda ) );
    }
    if (channel == 'D') {
        _loop_time +=      elapsed_time( if(!_LC) _LC = _L->X_C(Lambda); );
        if (_SU2) {
            // 'right' is the result

            // vertex corrections
            // left vertex C channel
            _reorder_time +=   elapsed_time( V( v, left, "D_to_C", false, 1.0 ) );
            _loop_symm_time += elapsed_time( symmetrize_C() );
            _loop_gemm_time += elapsed_time( batched_gemm_vertex_loop( left, _LC_sym, qsize, nk, nb ) );
            // make 'v' the right vertex and 'right' the result
            _contract_time +=  elapsed_time( batched_gemm( left, v, right, nk*nb*nb, _norms[3]*dLambda, 0.0, qsize ) );
            // right vertex C channel
            // make 'left' right
            _reorder_time +=   elapsed_time( V( v, left, "D_to_C", false, 1.0 ) );
            // make 'send_pack' left (*after* reordering call)
            _reorder_time +=   elapsed_time( V( v, V.get_send_pack(), false, 1.0 ) );
            _loop_gemm_time += elapsed_time( batched_gemm_vertex_loop( V.get_send_pack(), _LC_sym, qsize, nk, nb ) );
            // write result in 'right'
            _contract_time +=  elapsed_time( batched_gemm( V.get_send_pack(), left, right, nk*nb*nb, _norms[4]*dLambda, 1.0, qsize ) );

            // rpa
            // use 'send_pack' as left vertex *with* loop and 'v' as right
            // write result to 'right'
            _contract_time +=  elapsed_time( batched_gemm( V.get_send_pack(), v, right, nk*nb*nb, _norms[2]*dLambda, 1.0, qsize ) );
            // get max of result ('right')
            _max_time +=       elapsed_time( vmax = absmax( right, Vsize ) );
            // add result ('right') to dv
            _reorder_time +=   elapsed_time( V( right, dv, true, 1.0 ) );
        } else {
            _reorder_time +=   elapsed_time( V( v, left, false, 1.0 ) );
            _loop_symm_time += elapsed_time( symmetrize_C() );
            _loop_gemm_time += elapsed_time( batched_gemm_vertex_loop( left, _LC_sym, qsize, nk, nb ) );
            _contract_time +=  elapsed_time( batched_gemm( left, v, right, nk*nb*nb, 1.0, 0.0, qsize ) );
            _max_time +=       elapsed_time( vmax = absmax( right, Vsize ) * _norms[2] * dLambda );
            _reorder_time +=   elapsed_time( V( right, dv, true, _norms[2]*dLambda ) );
            _reorder_time +=   elapsed_time( V( right, dv, "D_swap_3_and_4", true, _norms[1]*dLambda ) );
        }
    }
    return std::abs(vmax);
}

void FlowStep::print_timings() {
    if (_num_calls < 1) return;

    diverge_mpi_allreduce_double_max( MPI_IN_PLACE, &_reorder_time,   1 );
    diverge_mpi_allreduce_double_max( MPI_IN_PLACE, &_loop_time,      1 );
    diverge_mpi_allreduce_double_max( MPI_IN_PLACE, &_loop_gemm_time, 1 );
    diverge_mpi_allreduce_double_max( MPI_IN_PLACE, &_contract_time,  1 );
    diverge_mpi_allreduce_double_max( MPI_IN_PLACE, &_max_time,       1 );
    diverge_mpi_allreduce_double_max( MPI_IN_PLACE, &_loop_symm_time, 1 );
    int nranks = diverge_mpi_comm_size();
    _num_calls *= (double)nranks;
    mpi_tim_printf( "time per step reorder:   %5.2fs\n", _reorder_time/_num_calls   );
    mpi_tim_printf( "time per step loop:      %5.2fs\n", _loop_time/_num_calls      );
    mpi_tim_printf( "time per step loop_symm: %5.2fs\n", _loop_symm_time/_num_calls );
    mpi_tim_printf( "time per step loop_gemm: %5.2fs\n", _loop_gemm_time/_num_calls );
    mpi_tim_printf( "time per step contract:  %5.2fs\n", _contract_time/_num_calls  );
    mpi_tim_printf( "time per step max:       %5.2fs\n", _max_time/_num_calls       );
    _num_calls /= (double)nranks;
}

void FlowStep::set_D_channel_degeneracy( double nrm ) {
    if (_SU2)
        _norms[2] = -1.0 * nrm;
    else
        mpi_wrn_printf( "trying to set D-channel degeneracy for non-SU2 system\n" );
}

} // grid
