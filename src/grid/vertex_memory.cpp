/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "vertex_memory.hpp"
#include "index_gymnastics.hpp"
#include "reorder.hpp"
#include "index_generator.hpp"
#include "loops.hpp"
#include "../diverge_Eigen3.hpp"
#include <algorithm>
#include <cstring>
#include <vector>
#include <string>
#include <omp.h>
// for atomic_add
#include "../diverge_patching_helpers.h"

namespace grid {

VertexMemory::VertexMemory( MPI_Comm comm, index_t nk, index_t nb,
        index_t* k1m2, index_t* k1p2, bool SU2, index_t* k3map ) :
    _nk(nk), _nb(nb), _SU2(SU2) {

    _InGen = new IndexGenerator( nk, nb, comm );
    _mpi_info = new dist_info;
    gen_dist_info( _mpi_info, _InGen->starts, _InGen->stops, comm );

    print_memory_requirement();

    _comm = comm;
    _IG = new IndexGymnastics( k1m2, k1p2, k3map );
    _alloc_map_buffers();
    _gen_index_maps();
    _alloc_vertex_buffers();
}

VertexMemory::~VertexMemory() {
    _dealloc_buffers();
    delete _IG;
    free_dist_info( _mpi_info );
    delete _mpi_info;
    delete _InGen;
}

vector<index_t> VertexMemory::qshare(void) {
    index_t qstart = _InGen->qstarts[_mpi_info->myrank],
            qstop = _InGen->qstops[_mpi_info->myrank];
    vector<index_t> result;
    result.reserve(qstop-qstart);
    for (index_t q=qstart; q<qstop; ++q)
        result.push_back(q);
    return result;
}

// if (!SU2)
//  qbuf_in: (nq, ns,ns,no, ns,ns,no)
// else
//  qbuf_in: (nq, no, no)
void VertexMemory::init_from_qbuf( const complex128_t* qbuf_in, char chan_in, index_t ns ) {
    if (chan_in != 'P' && chan_in != 'C' && chan_in != 'D') {
        mpi_wrn_printf("invalid channel '%c', fallback to 'D'.\n", chan_in);
        chan_in = 'D';
    }
    const vector<index_t> k1list = qshare();
    const index_t nk1 = k1list.size();

    complex128_t* vbuf_in = vertex_buffers[2];
    complex128_t* vbuf = vertex_buffers[0];

    if (_SU2) ns = 1;
    index_t no = _nb / ns;
    memset((void*)vbuf_in, 0, sizeof(complex128_t)*nk1*POW2(_nk)*POW4(_nb));

    const index_t *k1p2 = _IG->get_k1p2(),
                  *k1m2 = _IG->get_k1m2();

    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t k1i=0; k1i<nk1; ++k1i)
    for (index_t k2=0; k2<_nk; ++k2)
    for (index_t k3=0; k3<_nk; ++k3) {
        index_t k1 = k1list[k1i];
        index_t qq = 0;
        switch (chan_in) {
            case 'P': qq = k1p2[k1*_nk+k2]; break;
            case 'C': qq = k1m2[k3*_nk+k2]; break;
            case 'D': qq = k1m2[k1*_nk+k3]; break;
            default: break;
        }
        for (index_t OA=0; OA<no; ++OA)
        for (index_t OB=0; OB<no; ++OB) {
            index_t o1 = 0, o2 = 0, o3 = 0, o4 = 0;
            switch (chan_in) {
                case 'P': o1 = OA; o2 = OA; o3 = OB; o4 = OB; break;
                case 'C': o1 = OA; o2 = OB; o3 = OB; o4 = OA; break;
                case 'D': o1 = OA; o2 = OB; o3 = OA; o4 = OB; break;
                default: break;
            }
            for (index_t SA=0; SA<ns; ++SA)
            for (index_t SB=0; SB<ns; ++SB)
            for (index_t SC=0; SC<ns; ++SC)
            for (index_t SD=0; SD<ns; ++SD) {
                index_t s1 = 0, s2 = 0, s3 = 0, s4 = 0;
                switch (chan_in) {
                    case 'P': s1 = SA; s2 = SB; s3 = SC; s4 = SD; break;
                    case 'C': s1 = SA; s2 = SD; s3 = SC; s4 = SB; break;
                    case 'D': s1 = SA; s2 = SD; s3 = SB; s4 = SC; break;
                    default: break;
                }
                index_t qbuf_in_idx = IDX7( qq, SA, SB, OA, SC, SD, OB, ns, ns, no, ns, ns, no );
                index_t b1 = IDX2(s1, o1, no),
                        b2 = IDX2(s2, o2, no),
                        b3 = IDX2(s3, o3, no),
                        b4 = IDX2(s4, o4, no);
                index_t vbuf_in_idx = IDX7( k1i, k2, k3, b1, b2, b3, b4, _nk, _nk, _nb, _nb, _nb, _nb );
                atomic_add( vbuf_in + vbuf_in_idx, qbuf_in + qbuf_in_idx );
            }
        }
    }

    overwrite_transform( 0, 1, "V_to_D" );
    (*this)( vbuf_in, vbuf, "V_to_D", true, 1.0 );
    overwrite_transform( 0, 1, "D_to_P" );
}

void VertexMemory::print_memory_requirement() {
    index_t memory = _memory_requirement();
    vector<index_t> memory_per_rank(_mpi_info->nranks);
    diverge_mpi_allgather_index( &memory, memory_per_rank.data(), 1 );

    double mean_mem = 0.;
    for (unsigned i=0; i<memory_per_rank.size(); ++i)
        mean_mem += memory_per_rank[i];
    mean_mem /= (double) memory_per_rank.size();
    double min_mem = *std::min_element( memory_per_rank.begin(), memory_per_rank.end() );
    double max_mem = *std::max_element( memory_per_rank.begin(), memory_per_rank.end() );
    double GB = 1024.*1024.*1024.;

    mpi_vrb_printf( "grid vertex, mem per rank: %.1fGB-(%.1fGB)-%.1fGB\n", min_mem/GB, mean_mem/GB, max_mem/GB );
}

void VertexMemory::remove_projected_components( complex128_t* buf, complex128_t* aux_buf,
        char channel, const vector<index_t>& qvec,
        const vector<complex128_t*>& U_matrices ) {
    VertexMemory& V = *this;
    string back_trafo = "";
    if (channel == 'P') {
        V( buf, aux_buf, "D_to_P", false, 1.0 );
        back_trafo = "P_to_D";
    }
    if (channel == 'C') {
        if (_SU2) {
            V( buf, aux_buf, "D_to_C", false, 1.0 );
            back_trafo = "C_to_D";
        } else {
            V( buf, aux_buf, "D_swap_3_and_4", false, 1.0 );
            back_trafo = "D_swap_3_and_4";
        }
    }
    if (channel == 'D') {
        aux_buf = buf;
    }
    for (size_t i=0; i<qvec.size(); ++i) {
        index_t global_start_idx = qvec[i] * _nk * _nk * _nb * _nb * _nb * _nb;
        if (global_start_idx >= _mpi_info->starts[_mpi_info->myrank] &&
            global_start_idx < _mpi_info->stops[_mpi_info->myrank]) {
            index_t local_start_idx = global_start_idx - _mpi_info->starts[_mpi_info->myrank];
            Map<MatXcd> V_chan_mat(aux_buf + local_start_idx, _nk*_nb*_nb, _nk*_nb*_nb);
            Map<MatXcd> U_proj(U_matrices[i], _nk*_nb*_nb, 1);
            MatXcd V_projected = U_proj.adjoint() * V_chan_mat * U_proj;
            V_chan_mat -= U_proj * V_projected * U_proj.adjoint();
        }
    }
    if (back_trafo != "" && aux_buf != buf)
        V(aux_buf, buf, back_trafo, false, 1.0 );
}

void VertexMemory::operator()( const complex128_t* buf_in, complex128_t* buf_out, char chan1, char chan2, bool add, double scale ) {

    if (chan1 == chan2) {
        (*this)( buf_in, buf_out, add, scale );
        return;
    }

    string trafo_name = chan1 + (string)"_to_" + chan2;
    if ( ((chan1 == 'D' && chan2 == 'C') || (chan2 == 'D' && chan1 == 'C')) && !_SU2 )
        trafo_name = "D_swap_3_and_4";

    int send_id = _find_send_transform(trafo_name);
    int recv_id = _find_recv_transform(trafo_name);

    bool update_trafo = false;
    if (send_id == -1 || recv_id == -1) {
        overwrite_transform( 0, 1, trafo_name );
        update_trafo = true;
    }

    (*this)( buf_in, buf_out, trafo_name, add, scale );

    if (update_trafo)
        overwrite_transform( 0, 1, "D_to_P" );
}

void VertexMemory::operator()( const complex128_t* buf_in, complex128_t* buf_out,
        string trafo, bool add, double scale ) {
    int send_id = _find_send_transform(trafo);
    int recv_id = _find_recv_transform(trafo);
    if (send_id == -1 || recv_id == -1) {
        mpi_err_printf( "could not find '%s' vertex transform\n", trafo.c_str() );
        return;
    }
    full_pack_exchange( buf_in, buf_out,
            send_recv_maps[send_id], send_pack, send_recv_displs[send_id], send_recv_counts[send_id],
            send_recv_maps[recv_id], recv_pack, send_recv_displs[recv_id], send_recv_counts[recv_id],
            _mpi_info, add, scale, _comm );
}

void VertexMemory::operator()( const complex128_t* buf_in, complex128_t* buf_out, bool add,
        double scale ) {
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<_size; ++i) {
        if (add)
            buf_out[i] += buf_in[i] * scale;
        else
            buf_out[i] = buf_in[i] * scale;
    }
}

int VertexMemory::_find_send_transform(string name) {
    int result = -1;
    for (unsigned i=0; i<send_recv_descr.size(); ++i) {
        if (send_recv_descr[i] == (name + "_send")) result = i;
    }
    return result;
}

int VertexMemory::_find_recv_transform(string name) {
    int result = -1;
    for (unsigned i=0; i<send_recv_descr.size(); ++i) {
        if (send_recv_descr[i] == (name + "_recv")) result = i;
    }
    return result;
}

void VertexMemory::_alloc_map_buffers() {
    _size = _mpi_info->sizes[_mpi_info->myrank];
    int nranks = _mpi_info->nranks;
    for (size_t i=0; i<send_recv_maps.size()-(_SU2==0); ++i) {
        send_recv_maps[i] = (index_t*) calloc(_size, sizeof(index_t));
        send_recv_displs[i] = (int*) calloc(nranks, sizeof(int));
        send_recv_counts[i] = (int*) calloc(nranks, sizeof(int));
    }
}

void VertexMemory::_alloc_vertex_buffers() {
    _alloc_first_vertex_and_send_buffers();
    _alloc_other_vertex_buffers();
}

void VertexMemory::_alloc_first_vertex_and_send_buffers() {
    vertex_buffers[0] = (complex128_t*) calloc(_size, sizeof(complex128_t));
    send_pack = (complex128_t*) calloc(_size, sizeof(complex128_t));
    recv_pack = (complex128_t*) calloc(_size, sizeof(complex128_t));
}

void VertexMemory::_alloc_other_vertex_buffers() {
    for (size_t i=1; i<vertex_buffers.size(); ++i)
        vertex_buffers[i] = (complex128_t*) calloc(_size, sizeof(complex128_t));
}

void VertexMemory::_dealloc_buffers() {
    free(send_pack);
    free(recv_pack);
    for (size_t i=0; i<vertex_buffers.size(); ++i)
        free(vertex_buffers[i]);
    for (size_t i=0; i<send_recv_maps.size()-(_SU2==0); ++i) {
        free(send_recv_maps[i]);
        free(send_recv_displs[i]);
        free(send_recv_counts[i]);
    }
}

void VertexMemory::_gen_index_maps() {
    auto gen_D_to_P_index_map = [&](){
    gen_maps( send_recv_maps[0], send_recv_displs[0], send_recv_counts[0],
              send_recv_maps[1], send_recv_displs[1], send_recv_counts[1],
              _mpi_info, _nk, _nb, InGyFunPtr( _IG, D_to_P ), InGyFunPtr( _IG, P_to_D ) );
    send_recv_descr[0] = "D_to_P_send";
    send_recv_descr[1] = "D_to_P_recv";
    };
    mpi_tim_printf( "gen_D_to_P_index_map() took %5.2fs\n", take_time( gen_D_to_P_index_map ) );

    auto gen_P_to_D_index_map = [&](){
    gen_maps( send_recv_maps[2], send_recv_displs[2], send_recv_counts[2],
              send_recv_maps[3], send_recv_displs[3], send_recv_counts[3],
              _mpi_info, _nk, _nb, InGyFunPtr( _IG, P_to_D ), InGyFunPtr( _IG, D_to_P ) );
    send_recv_descr[2] = "P_to_D_send";
    send_recv_descr[3] = "P_to_D_recv";
    };
    mpi_tim_printf( "gen_P_to_D_index_map() took %5.2fs\n", take_time( gen_P_to_D_index_map ) );

    if (_SU2) {
        auto gen_D_to_C_index_map = [&](){
        gen_maps( send_recv_maps[4], send_recv_displs[4], send_recv_counts[4],
                  send_recv_maps[5], send_recv_displs[5], send_recv_counts[5],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, D_to_C ), InGyFunPtr( _IG, C_to_D ) );
        send_recv_descr[4] = "D_to_C_send";
        send_recv_descr[5] = "D_to_C_recv";
        };
        mpi_tim_printf( "gen_D_to_C_index_map() took %5.2fs\n", take_time( gen_D_to_C_index_map ) );

        auto gen_C_to_D_index_map = [&](){
        gen_maps( send_recv_maps[6], send_recv_displs[6], send_recv_counts[6],
                  send_recv_maps[7], send_recv_displs[7], send_recv_counts[7],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, C_to_D ), InGyFunPtr( _IG, D_to_C ) );
        send_recv_descr[6] = "C_to_D_send";
        send_recv_descr[7] = "C_to_D_recv";
        };
        mpi_tim_printf( "gen_C_to_D_index_map() took %5.2fs\n", take_time( gen_C_to_D_index_map ) );
    } else {
        auto gen_D_swap_3_and_4_index_map = [&](){
        gen_maps( send_recv_maps[4], send_recv_displs[4], send_recv_counts[4],
                  send_recv_maps[5], send_recv_displs[5], send_recv_counts[5],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, D_swap_3_and_4 ), InGyFunPtr( _IG, D_swap_3_and_4 ) );
        send_recv_descr[4] = "D_swap_3_and_4_send";
        send_recv_descr[5] = "D_swap_3_and_4_recv";
        send_recv_descr[6] = "None";
        send_recv_descr[7] = "None";
        };
        mpi_tim_printf( "gen_D_swap_3_and_4_index_map() took %5.2fs\n", take_time( gen_D_swap_3_and_4_index_map ) );
    }
}

size_t VertexMemory::_memory_requirement() {
    index_t size = _mpi_info->sizes[_mpi_info->myrank];
    size_t total_memsize = sizeof(complex128_t) * (vertex_buffers.size()+2) * size +
                           sizeof(index_t) * (send_recv_maps.size()-(_SU2==0)) * size;
    return total_memsize;
}

void VertexMemory::overwrite_transform( int send, int recv, string new_name ) {
    if (new_name == "D_to_P")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, D_to_P ), InGyFunPtr( _IG, P_to_D ) );
    else if (new_name == "P_to_D")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, P_to_D ), InGyFunPtr( _IG, D_to_P ) );
    else if (new_name == "D_to_C")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, D_to_C ), InGyFunPtr( _IG, C_to_D ) );
    else if (new_name == "C_to_D")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, C_to_D ), InGyFunPtr( _IG, D_to_C ) );
    else if (new_name == "D_to_V")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, D_to_V ), InGyFunPtr( _IG, V_to_D ) );
    else if (new_name == "V_to_D")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, V_to_D ), InGyFunPtr( _IG, D_to_V ) );
    else if (new_name == "C_to_V")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, C_to_V ), InGyFunPtr( _IG, V_to_C ) );
    else if (new_name == "V_to_C")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, V_to_C ), InGyFunPtr( _IG, C_to_V ) );
    else if (new_name == "P_to_V")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, P_to_V ), InGyFunPtr( _IG, V_to_P ) );
    else if (new_name == "V_to_P")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, V_to_P ), InGyFunPtr( _IG, P_to_V ) );
    else if (new_name == "D_swap_3_and_4")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, D_swap_3_and_4 ), InGyFunPtr( _IG, D_swap_3_and_4 ) );
    else if (new_name == "D_swap_1_and_2")
        gen_maps( send_recv_maps[send], send_recv_displs[send], send_recv_counts[send],
                  send_recv_maps[recv], send_recv_displs[recv], send_recv_counts[recv],
                  _mpi_info, _nk, _nb, InGyFunPtr( _IG, D_swap_1_and_2 ), InGyFunPtr( _IG, D_swap_1_and_2 ) );
    else {
        mpi_wrn_printf( "cannot find index gymnastics '%s'. skipping.\n", new_name.c_str() );
        return;
    }

    send_recv_descr[send] = new_name + "_send";
    send_recv_descr[recv] = new_name + "_recv";
}

void VertexMemory::change_basis( complex128_t* buf_in, complex128_t* buf_out, complex128_t* buf_additional, complex128_t* trafo ) {
    complex128_t* buf = buf_additional == nullptr ? buf_in : buf_additional;
    (*this)( buf_in, buf_out, "D_to_V", false, 1.0 );
    _change_basis_k123( buf_out, trafo );
    memcpy( buf, buf_out, _size*sizeof(complex128_t) );
    (*this)( buf, buf_out, "V_to_D", false, 1.0 );
}

void VertexMemory::exchange_symmetry(double factor) {
    complex128_t* vertex = vertex_buffers[0];
    complex128_t* buf1 = vertex_buffers[1];
    complex128_t* buf2 = vertex_buffers[2];
    overwrite_transform( 0, 1, "D_swap_1_and_2" );

    (*this)( vertex, buf1, false, 1.0 );
    (*this)( vertex, buf1, "D_swap_1_and_2", true, -1.0 );

    // move to buf2
    (*this)( buf1, buf2, false, 1.0 );

    (*this)( buf2, buf1, false, 1.0 );
    (*this)( buf2, buf1, "D_swap_3_and_4", true, -1.0 );

    (*this)( buf1, vertex, false, factor );
    overwrite_transform( 0, 1, "D_to_P" );
}

void VertexMemory::write_to_file( string fname, char channel ) {
    bool create_transform = false;
    bool do_transform = false;
    string trafo_name = "D_to_";
    if (channel == 'P' || channel == 'V' || channel == 'C') {
        trafo_name.push_back(channel);
        int send_trafo = _find_send_transform( trafo_name );
        int recv_trafo = _find_recv_transform( trafo_name );
        if (send_trafo == -1 || recv_trafo == -1) {
            create_transform = true;
            do_transform = true;
        } else {
            do_transform = true;
        }
    } else if (channel == 'D') {
    } else {
        mpi_wrn_printf( "invalid transformation channel '%c' using D.\n", channel );
    }

    if (do_transform) {
        if (create_transform) overwrite_transform( 0, 1, trafo_name );
        (*this)( vertex_buffers[0], vertex_buffers[1], trafo_name, false, 1.0 );
        if (create_transform) overwrite_transform( 0, 1, "D_to_P" );
    } else {
        (*this)( vertex_buffers[0], vertex_buffers[1], false, 1.0 );
    }

    diverge_mpi_write_cdoub_to_file( fname.c_str(), vertex_buffers[1],
            _mpi_info->starts[_mpi_info->myrank], get_size() );
}


void VertexMemory::get_vertex( complex128_t* buf, char channel ) {
    bool create_transform = false;
    bool do_transform = false;
    string trafo_name = "D_to_";
    if (channel == 'P' || channel == 'V' || channel == 'C') {
        trafo_name.push_back(channel);
        int send_trafo = _find_send_transform( trafo_name );
        int recv_trafo = _find_recv_transform( trafo_name );
        if (send_trafo == -1 || recv_trafo == -1) {
            create_transform = true;
            do_transform = true;
        } else {
            do_transform = true;
        }
    } else if (channel == 'D') {
    } else {
        mpi_wrn_printf( "invalid transformation channel '%c' using D.\n", channel );
    }

    if (do_transform) {
        if (create_transform) overwrite_transform( 0, 1, trafo_name );
        (*this)( vertex_buffers[0], vertex_buffers[1], trafo_name, false, 1.0 );
        if (create_transform) overwrite_transform( 0, 1, "D_to_P" );
    } else {
        (*this)( vertex_buffers[0], vertex_buffers[1], false, 1.0 );
    }
    vector<index_t> qlist = qshare();
    index_t rest = POW4(_nb)*POW2(_nk);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (uindex_t i=0; i<qlist.size(); ++i) {
        for (index_t r=0; r<rest; ++r) {
            buf[r+rest * qlist[i]] = vertex_buffers[1][r+rest*i];
        }
    }
    diverge_mpi_allreduce_complex_sum_inplace(buf, rest*_nk);
}

static inline index_t o2b_idx_order( const index_t& x, const index_t& y, const index_t& nb ) {
    return x*nb+y;
}

void VertexMemory::_change_basis_k123( complex128_t* V, complex128_t* U ) {
    index_t nb = _nb;
    index_t nk = _nk;
    index_t bandstuff = nb*nb*nb*nb;
    index_t start = _mpi_info->starts[_mpi_info->myrank];
    index_t stop = _mpi_info->stops[_mpi_info->myrank];
    index_t k1start = start / (nk*nk*bandstuff);
    index_t k1stop = stop / (nk*nk*bandstuff);

    const index_t* k1m2 = _IG->get_k1m2();
    const index_t* k1p2 = _IG->get_k1p2();
    const index_t* k3map = _IG->get_k3map();

    const bool use_k3map = k3map != nullptr;

    complex128_t* buffers = (complex128_t*) malloc(sizeof(complex128_t)*bandstuff*diverge_omp_num_threads());
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (int k1=k1start; k1<k1stop; ++k1)
    for (int k2=0; k2<nk; ++k2)
    for (int k3=0; k3<nk; ++k3) {
        complex128_t* vertex_inout = V+((k1-k1start)*nk*nk + k2*nk + k3)*bandstuff;
        int k4 = use_k3map ? k3map[nk*nk*k1 + nk*k2 + k3] :
                             k1m2[k3+nk*k1p2[k1*nk+k2]];

        complex128_t* vertex_buf = buffers + bandstuff * omp_get_thread_num();
        memset( (void*)vertex_buf, 0, sizeof(complex128_t)*bandstuff );
        for (int b1=0; b1<nb; ++b1)
        for (int b2=0; b2<nb; ++b2)
        for (int b3=0; b3<nb; ++b3)
        for (int b4=0; b4<nb; ++b4) {
            for (int c1=0; c1<nb; ++c1)
            for (int c2=0; c2<nb; ++c2)
            for (int c3=0; c3<nb; ++c3)
            for (int c4=0; c4<nb; ++c4) {
                int index_out = b1*nb*nb*nb+b2*nb*nb+b3*nb+b4;
                int index_in  = c1*nb*nb*nb+c2*nb*nb+c3*nb+c4;
                complex128_t Umatrix_index_out_index_in =
                    std::conj(U[k1*nb*nb+ o2b_idx_order(c1,b1,nb) ]) *
                    std::conj(U[k2*nb*nb+ o2b_idx_order(c2,b2,nb) ]) *
                    U[k3*nb*nb+ o2b_idx_order(c3,b3,nb) ] *
                    U[k4*nb*nb+ o2b_idx_order(c4,b4,nb) ];
                vertex_buf[index_out] += Umatrix_index_out_index_in * vertex_inout[index_in];
            }
        }
        memcpy( vertex_inout, vertex_buf, sizeof(complex128_t)*bandstuff );
    }
    free( buffers );
}

}
