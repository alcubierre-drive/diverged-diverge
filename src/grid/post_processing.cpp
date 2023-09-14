/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "post_processing.hpp"

#include "../misc/batched_gemms.h"
#include "vertex_memory.hpp"
#include "loops.hpp"
#include "reorder.hpp"

#include <thread>
#include <numeric>
#include <algorithm>
#include <cstring>

#include "../misc/eigen.h"

namespace grid {

class LinearizedGapSolution {
    public:
        LinearizedGapSolution( VertexMemory& V, Loops& L, double T, char chan,
                unsigned n, bool use_loop, string vertex_file );
        ~LinearizedGapSolution();

        complex128_t* get_U() const { return _singular_matrix_U; }
        complex128_t* get_V() const { return _singular_matrix_V; }
        double* get_Sigma() const { return _singular_values; }

        // pack U, V, Sigma and write how many doubles this array has in total
        // return nullptr if not _is_q_zero
        double* pack_data( index_t* count_doubles );
        void unpack_data( const double* data );
    private:
        VertexMemory* _V;
        Loops* _L;

        void _calc( double T, char chan, bool use_loop, string vertex_file );

        unsigned _num_singular_values;
        complex128_t* _singular_matrix_U = nullptr;
        complex128_t* _singular_matrix_V = nullptr;
        double* _singular_values = nullptr;

        complex128_t* _eigen_vectors = nullptr;
        double* _eigen_values = nullptr;

        index_t _nk, _nb, _qsize;

        index_t _q_zero;
        bool _is_q_zero;
};

LinearizedGapSolution::LinearizedGapSolution( VertexMemory& V, Loops& L,
        double T, char chan, unsigned n, bool use_loop, string vertex_file ) {
    _V = &V;
    _L = &L;
    _num_singular_values = n;

    _nk = _L->get_nk();
    _nb = _L->get_nb();
    _qsize = _L->get_qsize();

    _q_zero = _L->get_kzero();
    _is_q_zero = (_q_zero >= _L->get_qstart()) && (_q_zero < _L->get_qstop());

    if (_is_q_zero) {
        _singular_matrix_U = (complex128_t*)calloc(_nk*_nb*_nb*_num_singular_values,sizeof(complex128_t));
        _singular_matrix_V = (complex128_t*)calloc(_nk*_nb*_nb*_num_singular_values,sizeof(complex128_t));
        _singular_values = (double*)calloc(_nk*_nb*_nb,sizeof(double));

        _eigen_vectors = (complex128_t*)calloc(_nk*_nb*_nb*_num_singular_values,sizeof(complex128_t));
        _eigen_values = (double*)calloc(_nk*_nb*_nb,sizeof(double));
    }

    _calc( T, chan, use_loop, vertex_file );
}

LinearizedGapSolution::~LinearizedGapSolution() {
    if (_singular_matrix_U) free(_singular_matrix_U);
    if (_singular_matrix_V) free(_singular_matrix_V);
    if (_singular_values) free(_singular_values);

    if (_eigen_vectors) free(_eigen_vectors);
    if (_eigen_values) free(_eigen_values);
}

void LinearizedGapSolution::_calc( double T, char chan, bool use_loop, string vertex_file ) {

    VertexMemory& V = *_V;
    complex128_t* loop = nullptr;
    if (chan == 'P') {
        std::thread lc( [&](){ if (use_loop) loop = _L->X_P( T ); } );
        V( V.vertex_buffers[0], V.vertex_buffers[1], "D_to_P", false, 1.0 );
        lc.join();
        if (use_loop) loop = _L->symmetrize( loop, 'P' );
    } else if (chan == 'C') {
        std::thread lc( [&](){ if (use_loop) loop = _L->X_C( T ); } );
        V( V.vertex_buffers[0], V.vertex_buffers[1], "D_to_C", false, 1.0 );
        lc.join();
        if (use_loop) loop = _L->symmetrize( loop, 'C' );
    } else /*chan == 'D'*/ {
        std::thread lc( [&](){ if (use_loop) loop = _L->X_C( T ); } );
        V( V.vertex_buffers[0], V.vertex_buffers[1], false, 1.0 );
        lc.join();
        if (use_loop) loop = _L->symmetrize( loop, 'C' );
    }

    if (!_is_q_zero) return;

    if (vertex_file != "") {
        FILE* v = fopen( vertex_file.c_str(), "w" );
        mpi_log_printf( "writing MF q=0 '%c' vertex to file\n", chan )
        if (!v) {
            mpi_err_printf( "could not open file\n" );
        } else {
            fwrite( V.vertex_buffers[1] + _q_zero * _nk*_nk*_nb*_nb*_nb*_nb,
                sizeof(complex128_t), _nk*_nk*_nb*_nb*_nb*_nb, v );
            fclose( v );
        }
    }

    index_t N = _nk*_nb*_nb;
    complex128_t* v = V.vertex_buffers[1] + (_q_zero - _L->get_qstart())*N*N;
    complex128_t* U_ = V.vertex_buffers[2];
    complex128_t* V_ = V.vertex_buffers[3];

    // eigensolution of vertex (hermitian!)
    eigen_solve( v, U_, _eigen_values, N );
    eigensolution_sort( U_, _eigen_values, N, 0 );
    for (index_t i=0; i<N; ++i) {
        for (index_t v=0; v<_num_singular_values; ++v) {
            _eigen_vectors[v*N+i] = *(V_ + i*N + v);
        }
    }

    if (loop)
        batched_gemm_vertex_loop( V.vertex_buffers[1], loop, _qsize, _nk, _nb );
    singular_value_decomp( v, U_, V_, _singular_values, N );
    for (index_t i=0; i<N; ++i) {
        for (index_t v=0; v<_num_singular_values; ++v) {
            _singular_matrix_U[v*N+i] = *(U_ + i*N + v );
            _singular_matrix_V[v*N+i] = std::conj( *(V_ + v*N + i) );
        }
    }
}

double* LinearizedGapSolution::pack_data( index_t* count_doubles ) {
    index_t N = _nk*_nb*_nb;
    index_t Nmat = N*_num_singular_values*2;
    *count_doubles = Nmat*3 + _num_singular_values*2;

    if (!_is_q_zero) {
        return nullptr;
    } else {
        double* result = (double*)malloc(sizeof(double)* *count_doubles);
        for (index_t i=0; i<Nmat; ++i) {
            result[i] = ((double*)(_singular_matrix_U))[i];
        }
        for (index_t i=0; i<Nmat; ++i) {
            result[Nmat+i] = ((double*)(_singular_matrix_V))[i];
        }
        for (index_t i=0; i<_num_singular_values; ++i) {
            result[Nmat*2+i] = _singular_values[i];
        }
        for (index_t i=0; i<Nmat; ++i) {
            result[Nmat*2+_num_singular_values+i] = ((double*)_eigen_vectors)[i];
        }
        for (index_t i=0; i<_num_singular_values; ++i) {
            result[Nmat*3+_num_singular_values+i] = _eigen_values[i];
        }
        return result;
    }
}

void LinearizedGapSolution::unpack_data( const double* data ) {
    index_t N = _nk*_nb*_nb;
    index_t Nmat = N * _num_singular_values * 2;
    if (_singular_matrix_U != nullptr ||
        _singular_matrix_V != nullptr ||
        _singular_values != nullptr ||
        _eigen_vectors != nullptr ||
        _eigen_values != nullptr)
            mpi_wrn_printf( "non-null arrays overwritten\n" );
    _singular_matrix_U = (complex128_t*)malloc(sizeof(double)*Nmat);
    _singular_matrix_V = (complex128_t*)malloc(sizeof(double)*Nmat);
    _singular_values = (double*)malloc(sizeof(double)*_num_singular_values);
    _eigen_vectors = (complex128_t*)malloc(sizeof(double)*Nmat);
    _eigen_values = (double*)malloc(sizeof(double)*_num_singular_values);
    for (index_t i=0; i<Nmat; ++i) {
        ((double*)(_singular_matrix_U))[i] = data[i];
    }
    for (index_t i=0; i<Nmat; ++i) {
        ((double*)(_singular_matrix_V))[i] = data[Nmat+i];
    }
    for (index_t i=0; i<_num_singular_values; ++i) {
        _singular_values[i] = data[Nmat*2+i];
    }
    for (index_t i=0; i<Nmat; ++i) {
        ((double*)_eigen_vectors)[i] = data[Nmat*2+_num_singular_values+i];
    }
    for (index_t i=0; i<_num_singular_values; ++i) {
        _eigen_values[i] = data[Nmat*3+_num_singular_values+i];
    }
}

PostProcessing::PostProcessing( VertexMemory& V, Loops &L ) {
    _V = &V;
    _L = &L;
    _SU2 = _V->get_SU2();
    _qsize = _L->get_qsize();
    _nk = _L->get_nk();
    _nb = _L->get_nb();
    _nff = _L->get_nff();
    _Xsize = _qsize * _nb * _nb * _nb * _nb * _nff;
    _comm = _V->get_mpi_info()->comm;
}

PostProcessing::~PostProcessing() {
    if (_susc_P_dist) free(_susc_P_dist);
    if (_susc_C_dist) free(_susc_C_dist);
    if (_susc_D_dist) free(_susc_D_dist);
    if (_lgap_P) delete _lgap_P;
    if (_lgap_C) delete _lgap_C;
    if (_lgap_D) delete _lgap_D;
}

void PostProcessing::set_num_singular_values( unsigned n ) {
    _num_singular_values = n;
}

void PostProcessing::set_loop_usage( bool loop ) {
    _use_loop = loop;
}

void PostProcessing::linearized_gap( double T, char chan, string vertex_file ) {
    if (chan == 'P')
        _lgap_P = new LinearizedGapSolution( *_V, *_L, T, chan, _num_singular_values, _use_loop, vertex_file );
    else if (chan == 'D')
        _lgap_D = new LinearizedGapSolution( *_V, *_L, T, chan, _num_singular_values, _use_loop, vertex_file );
    else if (chan == 'C' && _SU2)
        _lgap_C = new LinearizedGapSolution( *_V, *_L, T, chan, _num_singular_values, _use_loop, vertex_file );
    else
        mpi_wrn_printf( "invalid channel '%c'\n", chan );
}

void PostProcessing::susceptibility( double T, char chan ) {
    complex128_t* sus_buf = nullptr;
    if (chan == 'P') {
        _susc_P_dist = (complex128_t*)calloc(_Xsize,sizeof(complex128_t));
        sus_buf = _susc_P_dist;
    } else if (chan == 'D') {
        _susc_D_dist = (complex128_t*)calloc(_Xsize,sizeof(complex128_t));
        sus_buf = _susc_D_dist;
    } else if (chan == 'C' && _SU2) {
        _susc_C_dist = (complex128_t*)calloc(_Xsize,sizeof(complex128_t));
        sus_buf = _susc_C_dist;
    } else {
        mpi_wrn_printf( "invalid channel '%c'\n", chan );
    }
    if (sus_buf) _calc_sus( sus_buf, T, chan );
}

void PostProcessing::_calc_sus( complex128_t* buf, double T, char chan ) {
    VertexMemory& V = *_V;
    complex128_t* loop = nullptr;
    complex128_t* ff = _L->get_ff();
    if (chan == 'P') {
        std::thread lc( [&](){ loop = _L->X_P( T ); } );
        V( V.vertex_buffers[0], V.vertex_buffers[1], "D_to_P", false, 1.0 );
        lc.join();
        loop = _L->symmetrize( loop, 'P' );
    } else if (chan == 'C') {
        std::thread lc( [&](){ loop = _L->X_C( T ); } );
        V( V.vertex_buffers[0], V.vertex_buffers[1], "D_to_C", false, 1.0 );
        lc.join();
        loop = _L->symmetrize( loop, 'C' );
    } else /*chan == 'D'*/ {
        std::thread lc( [&](){ loop = _L->X_C( T ); } );
        V( V.vertex_buffers[0], V.vertex_buffers[1], false, 1.0 );
        lc.join();
        loop = _L->symmetrize( loop, 'C' );
    }
    batched_gemm_vertex_loop( V.vertex_buffers[1], loop, _qsize, _nk, _nb );
    batched_gemm_loop_vertex( loop, V.vertex_buffers[1], _qsize, _nk, _nb );
    #pragma omp parallel for collapse(6)
    for (index_t q=0; q<_qsize; ++q)
    for (index_t f=0; f<_nff; ++f)
    for (index_t b1=0; b1<_nb; ++b1)
    for (index_t b2=0; b2<_nb; ++b2)
    for (index_t b3=0; b3<_nb; ++b3)
    for (index_t b4=0; b4<_nb; ++b4) {
        for (index_t k=0; k<_nk; ++k)
        for (index_t kp=0; kp<_nk; ++kp) {
            index_t Vertexindex_t = q*_nk*_nb*_nb*_nk*_nb*_nb + k*_nb*_nb*_nk*_nb*_nb +
                                b1*_nb*_nk*_nb*_nb + b2*_nk*_nb*_nb + kp*_nb*_nb +
                                b3*_nb + b4;
            index_t Suscindex_t = q*_nff*_nb*_nb*_nb*_nb + f*_nb*_nb*_nb*_nb +
                              b1*_nb*_nb*_nb + b2*_nb*_nb + b3*_nb + b4;
            buf[Suscindex_t] += V.vertex_buffers[1][Vertexindex_t] * ff[f*_nk+k] * ff[f*_nk+kp];
        }
    }
    #pragma omp parallel for
    for (index_t i=0; i<_Xsize; ++i) {
        buf[i] /= (double)(_nk*_nk);
    }
}

vector<index_t> PostProcessing::_qstartstops() {
    int nranks = diverge_mpi_comm_size();
    array<index_t,2> q = { _L->get_qstart(), _L->get_qstop() };
    vector<index_t> qstartstops( nranks*2 );
    diverge_mpi_allgather_index( q.data(), qstartstops.data(), 2 );
    return qstartstops;
}

int PostProcessing::_find_q_zero_rank() {
    vector<index_t> qstartstops = _qstartstops();
    index_t q_zero = _L->get_kzero();
    for (unsigned i=0; i<qstartstops.size()/2; ++i) {
        index_t q0 = qstartstops[2*i];
        index_t q1 = qstartstops[2*i+1];
        if ((q_zero >= q0) && (q_zero < q1))
            return i;
    }
    return -1;
}

void PostProcessing::_communicate_linearized_gaps_to_master() {
    double* pack_P = nullptr;
    double* pack_C = nullptr;
    double* pack_D = nullptr;
    index_t pack_P_size = -1;
    index_t pack_C_size = -1;
    index_t pack_D_size = -1;
    if (_lgap_P) pack_P = _lgap_P->pack_data( &pack_P_size );
    if (_lgap_C) pack_C = _lgap_C->pack_data( &pack_C_size );
    if (_lgap_D) pack_D = _lgap_D->pack_data( &pack_D_size );

    bool has_P = pack_P_size >= 0;
    bool has_C = pack_C_size >= 0;
    bool has_D = pack_D_size >= 0;
    if (! (has_P || has_C || has_D) ) return;

    bool is_sender = (pack_P != nullptr) || (pack_C != nullptr) || (pack_D != nullptr);
    bool is_master = _V->get_mpi_info()->myrank == 0;
    int q_zero_rank = _find_q_zero_rank();

    if (is_sender && is_master) {
        if (pack_P) free(pack_P);
        if (pack_C) free(pack_C);
        if (pack_D) free(pack_D);
        return;
    }

    if (is_sender) {
        int to_rank = 0;
        if (has_P) {
            diverge_mpi_send_double( pack_P, pack_P_size, to_rank, 'P' );
            free(pack_P);
        }
        if (has_C) {
            diverge_mpi_send_double( pack_C, pack_C_size, to_rank, 'C' );
            free(pack_C);
        }
        if (has_D) {
            diverge_mpi_send_double( pack_D, pack_D_size, to_rank, 'D' );
            free(pack_D);
        }
    }

    if (is_master) {
        int from_rank = q_zero_rank;
        if (from_rank < 0) mpi_err_printf( "invalid q=0 rank\n" );
        if (has_P) {
            pack_P = (double*)calloc(pack_P_size,sizeof(double));
            diverge_mpi_recv_double( pack_P, pack_P_size, from_rank, 'P' );
            _lgap_P->unpack_data( pack_P );
            free(pack_P);
        }
        if (has_C) {
            pack_C = (double*)calloc(pack_C_size,sizeof(double));
            diverge_mpi_recv_double( pack_C, pack_C_size, from_rank, 'C' );
            _lgap_C->unpack_data( pack_C );
            free(pack_C);
        }
        if (has_D) {
            pack_D = (double*)calloc(pack_D_size,sizeof(double));
            diverge_mpi_recv_double( pack_D, pack_D_size, from_rank, 'D' );
            _lgap_D->unpack_data( pack_D );
            free(pack_D);
        }
    }
}

void PostProcessing::_communicate_susceptibility_to_master( char chan ) {

    complex128_t* send_buf = (chan == 'P') ? _susc_P_dist :
                      (chan == 'C') ? _susc_C_dist :
                      /*chan == 'D'*/ _susc_D_dist;
    if (send_buf == nullptr) return;

    diverge_mpi_barrier();
    vector<index_t> qstartstops = _qstartstops();
    vector<int> susc_counts( qstartstops.size()/2 );
    for (unsigned i=0; i<susc_counts.size(); ++i) {
        susc_counts[i] = (qstartstops[2*i+1] - qstartstops[2*i]) * _nff*_nb*_nb*_nb*_nb;
    }
    vector<int> susc_displs( susc_counts.size() );
    susc_displs[0] = 0;
    for (unsigned i=1; i<susc_displs.size(); ++i) {
        susc_displs[i] = susc_displs[i-1] + susc_counts[i-1];
    }
    int myrank = _V->get_mpi_info()->myrank;

    complex128_t* recv_buf = (chan == 'P') ? _V->vertex_buffers[1] :
                      (chan == 'C') ? _V->vertex_buffers[2] :
                      /*chan == 'D'*/ _V->vertex_buffers[3];
    diverge_mpi_gatherv_cdoub( send_buf, susc_counts[myrank], recv_buf, susc_counts.data(), susc_displs.data(), 0 );
    diverge_mpi_barrier();
}

void PostProcessing::communicate() {
    _communicate_linearized_gaps_to_master();
    _communicate_susceptibility_to_master('P');
    _communicate_susceptibility_to_master('C');
    _communicate_susceptibility_to_master('D');
    _has_communicated = true;
}


// header consists of 64 index_t specifying the counts and
// displacements (in units of doubles) of the channels, i.e.:
// nk, nb, nff, SU2, formfactor_displ, formfactor_count,
// P_susc_displ, P_susc_count, P_lin_displ, P_lin_count,
// C_susc_displ, C_susc_count, C_lin_displ, C_lin_count,
// D_susc_displ, D_susc_count, D_lin_displ, D_lin_count,
// lingap_num_ev, lingap_matrix_size (linear),
// ...
// file_size
//
// header is followed by data (size*sizeof(double))
//
// returns total file size.
index_t* PostProcessing::to_file( const string& fname ) {
    if (!_has_communicated) {
        mpi_err_printf( "must have communicated before to_file can be called\n" );
        return nullptr;
    }
    index_t size_lP = 0;
    index_t size_lC = 0;
    index_t size_lD = 0;
    double* lP = nullptr;
    double* lC = nullptr;
    double* lD = nullptr;
    if (_lgap_P) lP = _lgap_P->pack_data( &size_lP );
    if (_lgap_C) lC = _lgap_C->pack_data( &size_lC );
    if (_lgap_D) lD = _lgap_D->pack_data( &size_lD );

    index_t size_susc_P = _susc_P_dist == nullptr ? 0 : _nk*_nb*_nb*_nb*_nb*_nff*2;
    index_t size_susc_C = _susc_C_dist == nullptr ? 0 : _nk*_nb*_nb*_nb*_nb*_nff*2;
    index_t size_susc_D = _susc_D_dist == nullptr ? 0 : _nk*_nb*_nb*_nb*_nb*_nff*2;

    index_t* header = (index_t*) calloc(64,sizeof(index_t));
    int i=0;

    header[i] = MAGIC_NUMBER_POST_PROCESSING; i++;
    header[i] = _nk; i++;
    header[i] = _nb; i++;
    header[i] = _nff; i++;
    header[i] = _SU2; i++;

    // formfactors
    header[i] = 64; i++;
    header[i] = _nff * _nk * 2; i++;

    // susc P
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_susc_P; i++;
    // lin gap P
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_lP; i++;

    // susc C
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_susc_C; i++;
    // lin gap C
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_lC; i++;

    // susc D
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_susc_D; i++;
    // lin gap D
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_lD; i++;

    index_t total_size = header[i-2] + header[i-1];

    // lingap info
    header[i] = _num_singular_values; i++;
    header[i] = _nk*_nb*_nb; i++;

    header[62] = MAGIC_NUMBER_POST_PROCESSING;
    header[63] = total_size;

    if (_V->get_mpi_info()->myrank != 0) return header;

    FILE* outfile = fopen( fname.c_str(), "w" );
    if (!outfile) {
        mpi_err_printf( "could not open file '%s'\n", fname.c_str() );
        return header;
    }
    fwrite( header, sizeof(index_t), 64, outfile );
    fwrite( _L->get_ff(), sizeof(double), header[6], outfile );

    fwrite( _V->vertex_buffers[1], sizeof(double), header[8], outfile );
    if (lP) {
        fwrite( lP, sizeof(double), header[10], outfile );
        free(lP);
    }

    fwrite( _V->vertex_buffers[2], sizeof(double), header[12], outfile );
    if (lC) {
        fwrite( lC, sizeof(double), header[14], outfile );
        free(lC);
    }

    fwrite( _V->vertex_buffers[3], sizeof(double), header[16], outfile );
    if (lD) {
        fwrite( lD, sizeof(double), header[18], outfile );
        free(lD);
    }

    fclose( outfile );
    return header;
}

void print_post_processing_file_header( index_t* header ) {
    if (header == nullptr) {
        mpi_err_printf( "header is empty\n" );
        return;
    }
    index_t nk = header[1];
    index_t nb = header[2];
    index_t nff = header[3];
    index_t SU2 = header[4];

    index_t ff[2] = { header[5], header[6] };
    index_t P_susc[2] = { header[7], header[8] };
    index_t P_lin_gap[2] = { header[9], header[10] };
    index_t C_susc[2] = { header[11], header[12] };
    index_t C_lin_gap[2] = { header[13], header[14] };
    index_t D_susc[2] = { header[15], header[16] };
    index_t D_lin_gap[2] = { header[17], header[18] };

    index_t lingap_num_ev = header[19];
    index_t lingap_matrix_sz = header[20];

    index_t magic_number = header[62];
    index_t file_size = header[63];

    mpi_fil_printf( "file size %li (*64bit), magic number %li\n", file_size, magic_number );
    mpi_fil_printf( "nk=%li nb=%li nff=%li SU2=%i nev=%li mat_sz=%li\n", nk, nb, nff,
            SU2, lingap_num_ev, lingap_matrix_sz );
    mpi_fil_printf( "ff      %10li (+%10li; *64bit)\n", ff[0]       , ff[1] );
    mpi_fil_printf( "P susc  %10li (+%10li; *64bit)\n", P_susc[0]   , P_susc[1] );
    mpi_fil_printf( "P lingap%10li (+%10li; *64bit)\n", P_lin_gap[0], P_lin_gap[1] );
    mpi_fil_printf( "C susc  %10li (+%10li; *64bit)\n", C_susc[0]   , C_susc[1] );
    mpi_fil_printf( "C lingap%10li (+%10li; *64bit)\n", C_lin_gap[0], C_lin_gap[1] );
    mpi_fil_printf( "D susc  %10li (+%10li; *64bit)\n", D_susc[0]   , D_susc[1] );
    mpi_fil_printf( "D lingap%10li (+%10li; *64bit)\n", D_lin_gap[0], D_lin_gap[1] );
}

}
