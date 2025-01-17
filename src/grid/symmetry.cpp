/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "symmetry.hpp"
#include "../diverge_internals_struct.h"

namespace grid {

LoopSymmetrizer::LoopSymmetrizer( grid_symm_transform_t* s, index_t qdispl, index_t qcount ) {
    if (!s) {
        _can_transform = false;
        _nsym = 0;
    } else {
        _can_transform = true;
        grid_symm_transform_init_communicator( s, qdispl, qcount );
        _nsym = grid_symm_transform_nsym(s);
        _loop_size = grid_symm_transform_loopsize(s);
        _comms_P = (grid_symm_comm_t*)calloc(_nsym, sizeof*_comms_P);
        _comms_C = (grid_symm_comm_t*)calloc(_nsym, sizeof*_comms_P);
        for (int i=0; i<_nsym; ++i) {
            _comms_P[i] = grid_symm_transform_comm(s, i, 'P');
            _comms_C[i] = grid_symm_transform_comm(s, i, 'C');
        }
        _send_buf.resize( _loop_size );
        _recv_buf.resize( _loop_size );
        _loop_buf.resize( _loop_size );
    }
}

LoopSymmetrizer::~LoopSymmetrizer() {
    for (int i=0; i<_nsym; ++i)
        grid_symm_comm_free( _comms_P[i] ),
        grid_symm_comm_free( _comms_C[i] );
    free( _comms_P );
    free( _comms_C );
}

int LoopSymmetrizer::operator()( complex128_t* loop_in, complex128_t* loop_out, index_t sym, char chan ) {
    if (!_can_transform) return 1;

    const grid_symm_comm_t& comm = chan == 'P' ? _comms_P[sym] : _comms_C[sym];

    const index_t* send_map = comm.reorder_before_send;
    const complex128_t* send_phase = comm.phase_before_send;

    const index_t* send_counts = comm.send_counts;
    const index_t* send_displs = comm.send_displs;
    const index_t* recv_counts = comm.recv_counts;
    const index_t* recv_displs = comm.recv_displs;

    const index_t* recv_map = comm.reorder_after_recv;

    if (!comm.orbital_matrix) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t x=0; x<_loop_size; ++x)
            _send_buf[x] = loop_in[send_map[x]] * send_phase[x];
    } else {
        const index_t nb = comm.nb;
        const complex128_t* orbital_matrix = comm.orbital_matrix;
        ZERO_OUT( _send_buf.data(), _loop_size );
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t qk=0; qk<_loop_size/(nb*nb*nb*nb); ++qk) {
            index_t x = qk*nb*nb*nb*nb;
            for (index_t o=0; o<nb*nb*nb*nb; ++o)
            for (index_t op=0; op<nb*nb*nb*nb; ++op) {

                const index_t dim[4] = {nb,nb,nb,nb};
                index_t O[4]; unravel( o, dim, 4, O );
                index_t Op[4]; unravel( op, dim, 4, Op );
                index_t o1,o2,o3,o4,
                        o1p,o2p,o3p,o4p;
                if (chan == 'P') {
                    o1 = O[0], o2 = O[1], o3 = O[2], o4 = O[3],
                    o1p = Op[0], o2p = Op[1], o3p = Op[2], o4p = Op[3];
                } else {
                    o1 = O[1], o2 = O[2], o3 = O[3], o4 = O[0],
                    o1p = Op[1], o2p = Op[2], o3p = Op[3], o4p = Op[0];
                }


                _send_buf[x+op] += loop_in[send_map[x+o]] *
                    std::conj(orbital_matrix[IDX2(o1p,o1,nb)]) *
                    std::conj(orbital_matrix[IDX2(o2p,o2,nb)]) *
                             (orbital_matrix[IDX2(o3p,o3,nb)]) *
                             (orbital_matrix[IDX2(o4p,o4,nb)]) * send_phase[x+o];
            }
        }
    }

    diverge_mpi_alltoallv_bytes( _send_buf.data(), send_counts, send_displs,
            _recv_buf.data(), recv_counts, recv_displs, sizeof(complex128_t) );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t x=0; x<_loop_size; ++x) {
        loop_out[recv_map[x]] += _recv_buf[x];
    }

    return 0;
}

bool LoopSymmetrizer::check( complex128_t* loop, char chan ) {
    if (!_can_transform) return true;
    complex128_t* buf = _loop_buf.data();
    bool errors = false;
    mpi_err_printf( "checking '%c'\n", chan );
    for (int i=0; i<_nsym; ++i) {
        ZERO_OUT( buf, _loop_size );
        this->operator()( loop, buf, i, chan );
        for (index_t x=0; x<_loop_size; ++x) {
            double err = std::abs( buf[x] - loop[x] );
            if (err > 1.e-13) {
                mpi_wrn_printf( "chan '%c', sym %i loop not symmetric (%li; %.5e%+.5ej - %.5e%+.5ej)\n",
                        chan, i, x, buf[x].real(), buf[x].real(), loop[x].real(), loop[x].imag() );
                errors = true;
            }
        }
    }
    return errors;
}

complex128_t* LoopSymmetrizer::operator()( complex128_t* loop, char chan ) {
    if (!_can_transform) return loop;
    complex128_t* out = _loop_buf.data();
    ZERO_OUT( out, _loop_size );
    for (int i=0; i<_nsym; ++i)
        if (this->operator()( loop, out, i, chan )) {
            return loop;
        }
    double norm = 1./(double)_nsym;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (int i=0; i<_loop_size; ++i)
        out[i] *= norm;

    return out;
}

}
