/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "reorder.h"
#include "../diverge_Eigen3.hpp"

#include <vector>
#include <array>

using std::vector;
using std::array;

namespace grid {

typedef struct symm_props symm_props;
// TODO grid re-symmetrization not thoroughly tested for the case of
// multi-orbital multi-site. Luckily, only a maniac would do a grid FRG
// calculation for such systems.
class SymmetryTransform {
    public:
        SymmetryTransform(
                const vector<Mat3d>& symmetries,
                const vector<Vec3d>& basis,
                const array<Vec3d,3>& lattice,
                const array<Vec3d,3>& reciprocal,
                const vector<Vec3d>& kmesh );
        ~SymmetryTransform();

        void transform_P( const index_t* k1m2, const index_t* k1p2, index_t qstart,
                index_t qstop, index_t nk, index_t nb, index_t symm );
        void transform_C( const index_t* k1m2, const index_t* k1p2, index_t qstart,
                index_t qstop, index_t nk, index_t nb, index_t symm );

        index_t get_nsym( void ) { return nsym; }

        template<char chan>
        vector<complex128_t>& get_phases( index_t symm ) {
            if (chan == 'P')
                return _symmetry_phases_P[symm];
            else
                return _symmetry_phases_C[symm];
        }
        template<char chan>
        vector<index_t>& get_map( index_t symm ) {
            if (chan == 'P')
                return _symmetry_maps_P[symm];
            else
                return _symmetry_maps_C[symm];
        }

        void setOrbitalMatrix( index_t symm, const MatXcd& symmat ) {
            if (symm < nsym && symm >= 0) _symmetry_matrices[symm] = symmat;
        }
        vector<MatXcd> getOrbitalMatrices( void ) const { return _symmetry_matrices; }
        bool getMatrixUse( void ) const { return _use_matrices; }
        void setMatrixUse( bool diverge_style = false ) {
            _use_matrices = true;
            _reset_bond_symmprops(diverge_style);
        }

        void clean_mem( index_t symm );

    private:
        vector<Mat3d> _symmetries;
        vector<Vec3d> _basis;
        array<Vec3d,3> _lattice;
        array<Vec3d,3> _reciprocal;
        vector<Vec3d> _kmesh;
        vector<vector<symm_props>> _symm_props;
        vector<vector<index_t>> _k_symmap;

        index_t nsym;

        vector<vector<complex128_t>> _symmetry_phases_P;
        vector<vector<index_t>> _symmetry_maps_P;
        vector<vector<complex128_t>> _symmetry_phases_C;
        vector<vector<index_t>> _symmetry_maps_C;

        vector<MatXcd> _symmetry_matrices;
        bool _use_matrices = false;
        void _reset_bond_symmprops( bool diverge_style );
};

typedef struct {
    int current_rank;
    int target_rank;
    index_t current_index;
    index_t target_index;
    complex128_t phase;
} SymmetryOperation;

typedef struct {
    vector<index_t> reorder_before_send;
    vector<complex128_t> phase_before_send;
    vector<index_t> send_counts, send_displs,
                       recv_counts, recv_displs;
    vector<index_t> reorder_after_recv;
    MatXcd orbital_matrix;
    bool use_orbital_matrix;
    char chan;
} SymmetryCommunication;

class SymmetryCommunicatorFactory {
    public:
        SymmetryCommunicatorFactory( SymmetryTransform* st, dist_info* mpi_info,
                const index_t* k1m2, const index_t* k1p2, index_t qstart, index_t qstop,
                index_t nk, index_t nb );
        ~SymmetryCommunicatorFactory();

        index_t get_nsym() { return _st->get_nsym(); }
        index_t get_loop_size() { return _loop_size; }
        index_t get_qstart() { return _qstart; }

        SymmetryCommunication operator()( index_t symm, char chan ) {
            return gen_communication( symm, chan );
        }
    private:
        SymmetryTransform* _st;
        dist_info* _mpi;
        vector<index_t> _qstartstops();
        vector<index_t> _qstarts;
        vector<index_t> _qstops;

        index_t _qstart, _qstop, _nk, _nb;
        index_t _loop_size;

        template<char chan>
        SymmetryOperation* get_symop( index_t symm ) {
            if constexpr(chan == 'P')
                return _symops_P[symm];
            else
                return _symops_C[symm];
            return nullptr;
        }

        SymmetryCommunication gen_communication( index_t symm, char chan );

        vector<SymmetryOperation*> _symops_P;
        vector<SymmetryOperation*> _symops_C;
};

class LoopSymmetrizer {
    public:
        LoopSymmetrizer( SymmetryCommunicatorFactory* factory, dist_info* mpi_info );
        complex128_t* operator()( complex128_t* loop_in, char chan );
        void setDebugMode() { _debug = true; }
        bool check( complex128_t* loop_in, char chan );
    private:
        int operator()( complex128_t* loop_in, complex128_t* loop_out, index_t sym, char chan );
        bool _can_transform;
        index_t _nsym;
        index_t _loop_size;
        bool _debug = false;
        vector<SymmetryCommunication> _comms_P;
        vector<SymmetryCommunication> _comms_C;
        dist_info* _mpi;
        vector<complex128_t> _send_buf;
        vector<complex128_t> _recv_buf;
        vector<complex128_t> _loop_buf;
};

}
