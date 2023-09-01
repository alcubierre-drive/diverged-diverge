#pragma once

#include <vector>
#include <string>
#include "typedefs.hpp"

#ifndef MAGIC_NUMBER_POST_PROCESSING
#define MAGIC_NUMBER_POST_PROCESSING 'P'
#endif

namespace grid {

class VertexMemory;
class Loops;
class LinearizedGapSolution;

class PostProcessing {
    public:
        PostProcessing( VertexMemory& V, Loops& L );
        ~PostProcessing();

        void susceptibility( double T, char chan );
        void linearized_gap( double T, char chan, string vertex_file );

        void set_num_singular_values( unsigned n );
        void set_loop_usage( bool loop );
        void set_vertex_write( string fname, char chan );

        // only call communicate after all calculations!
        void communicate();

        // header consists of 64 index_t specifying the counts and
        // displacements (in units of doubles) of the channels, i.e.:
        // nk, nb, nff, SU2, formfactor_displ, formfactor_count,
        // P_susc_displ, P_susc_count, P_lin_displ, P_lin_count,
        // C_susc_displ, C_susc_count, C_lin_displ, C_lin_count,
        // D_susc_displ, D_susc_count, D_lin_displ, D_lin_count
        // lingap_num_ev, lingap_matrix_size (linear),
        // ...
        // file_size
        // header is followed by data (size*sizeof(double))
        //
        // returns total file size.
        index_t* to_file( const string& fname );
    private:
        VertexMemory* _V;
        Loops* _L;

        index_t _qsize, _nb, _nk, _nff, _Xsize;

        bool _use_loop = true;
        unsigned _num_singular_values = 10;
        complex128_t* _susc_P_dist = nullptr;
        complex128_t* _susc_C_dist = nullptr;
        complex128_t* _susc_D_dist = nullptr;

        void _calc_sus( complex128_t* buf, double T, char chan );

        LinearizedGapSolution* _lgap_P = nullptr;
        LinearizedGapSolution* _lgap_C = nullptr;
        LinearizedGapSolution* _lgap_D = nullptr;

        bool _SU2;

        vector<index_t> _qstartstops();
        int _find_q_zero_rank();
        MPI_Comm _comm;

        void _communicate_linearized_gaps_to_master();
        void _communicate_susceptibility_to_master( char chan );

        bool _has_communicated = false;
};

void print_post_processing_file_header( index_t* header );

}
