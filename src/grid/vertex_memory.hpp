/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "typedefs.hpp"
typedef struct dist_info dist_info;

namespace grid {

class IndexGenerator;

class VertexMemory {
    public:
        VertexMemory( MPI_Comm comm, index_t nk, index_t nb, index_t* k1m2,
                index_t* k1p2, bool SU2, index_t* k3map = nullptr );

        ~VertexMemory();

        // elements: v, dv, left, right.
        // v and dv in D channel
        array<complex128_t*,4> vertex_buffers;

        void init_from_qbuf( const complex128_t* qbuf_in, char chan_in, index_t nspin );
        vector<index_t> qshare(void);

        // creates trafo if no transform is found
        void operator()( const complex128_t* buf_in, complex128_t* buf_out, char chan1, char chan2,
                bool add, double scale );

        void operator()( const complex128_t* buf_in, complex128_t* buf_out, string trafo,
                bool add, double scale );
        void operator()( const complex128_t* buf_in, complex128_t* buf_out,
                bool add, double scale );

        // these are needed for the SU(2) D channel contraction
        complex128_t* get_send_pack(void) const { return send_pack; }
        complex128_t* get_recv_pack(void) const { return recv_pack; }

        dist_info* get_mpi_info(void) const { return _mpi_info; }
        bool get_SU2(void) const { return _SU2; }
        index_t get_size(void) const { return _size; }

        // enable/disable the D channel diagrams one by one
        void set_sub_D( int which[3] ) { _was_sub_D_modified = true; for (int i=0; i<3; ++i) _sub_D[i] = which[i]; }
        bool was_sub_D_modified(void) const { return _was_sub_D_modified; }
        array<bool,3> get_sub_D(void) const { return _sub_D; }

        void overwrite_transform( int send, int recv, string new_name );

        // additional buffer may be NULL. In this case, buf_in is overwritten
        void change_basis( complex128_t* buf_in, complex128_t* buf_out, complex128_t* buf_additional, complex128_t* trafo );

        // anti-symmetrizes vertex. factor 0.25 means we average. 0.5 means we
        // 'extend' the C/D channel vertex to all channels
        void exchange_symmetry(double factor = 0.5);

        void write_to_file( string fname, char channel );

        void get_vertex( complex128_t* buf, char channel );

        void print_memory_requirement(void);

        void remove_projected_components( complex128_t* buf, complex128_t* aux_buf,
                char channel, const vector<index_t>& qvec, const vector<complex128_t*>& U_matrices );

        IndexGenerator* get_index_gen(void) { return _InGen; }

    private:
        dist_info* _mpi_info;
        void _alloc_map_buffers();
        void _gen_index_maps();
        void _alloc_vertex_buffers();
        void _alloc_first_vertex_and_send_buffers();
        void _alloc_other_vertex_buffers();
        void _dealloc_buffers();
        size_t _memory_requirement();
        int _find_send_transform(string name);
        int _find_recv_transform(string name);
        const index_t _nk;
        const index_t _nb;
        const bool _SU2;
        index_t _size;
        IndexGenerator* _InGen;

        void _change_basis_k123( complex128_t* vertex_inout, complex128_t* U );

        static constexpr int num_maps = 8;
        array<index_t*,num_maps> send_recv_maps;
        array<int*,num_maps> send_recv_displs, send_recv_counts;
        array<string,num_maps> send_recv_descr;
        complex128_t* send_pack;
        complex128_t* recv_pack;

        bool _was_sub_D_modified = false;
        array<bool,3> _sub_D;

        const index_t *k1m2, *k1p2, *k3map;
};

}

extern "C" void grid_VertexMemory_set_sub_D( void* vtx, int which[3] );
