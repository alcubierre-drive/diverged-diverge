/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "typedefs.hpp"
#include "reorder.h"

namespace grid {

class IndexGenerator;

class VertexMemory {
    public:
        VertexMemory( index_t nk, index_t nb, index_t* k1m2, index_t* k1p2, bool SU2 );

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

        bool get_SU2(void) const { return _SU2; }
        index_t get_size(void) const { return _size; }

        // enable/disable the D channel diagrams one by one
        void set_sub_D( int which[3] ) { _was_sub_D_modified = true; for (int i=0; i<3; ++i) _sub_D[i] = which[i]; }
        bool was_sub_D_modified(void) const { return _was_sub_D_modified; }
        array<bool,3> get_sub_D(void) const { return _sub_D; }

        void overwrite_transform( int send, int recv, string new_name );

        // anti-symmetrizes vertex. factor 0.25 means we average. 0.5 means we
        // 'extend' the C/D channel vertex to all channels
        void exchange_symmetry(double factor = 0.5);

        void write_to_file( string fname, char channel );

        void get_vertex( complex128_t* buf, char channel );

        void print_memory_requirement(void);

        index_t get_qdispl( void ) const { return _dist->qdispls[_dist->myrank]; }
        index_t get_qcount( void ) const { return _dist->qcounts[_dist->myrank]; }

    private:
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

        // index distribution info
        index_t _myrank, _nranks;
        index_t *_qcounts, *_qdispls, *_counts, *_displs;
        dist_info_t* _dist;

        static constexpr int num_maps = 8;
        array<index_t*,num_maps> send_recv_maps;
        array<int*,num_maps> send_recv_displs, send_recv_counts;
        array<string,num_maps> send_recv_descr;
        complex128_t* send_pack;
        complex128_t* recv_pack;

        bool _was_sub_D_modified = false;
        array<bool,3> _sub_D;

        const index_t *k1m2, *k1p2;
};

}

extern "C" void grid_VertexMemory_set_sub_D( void* vtx, int which[3] );
