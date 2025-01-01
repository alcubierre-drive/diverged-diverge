/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

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

        /* header is index_t[64], and all displacement/size information is in
         * units of 64bit (i.e. double).
         *
         * header[0] - MAGIC_NUMBER_POST_PROCESSING
         * header[60-61] - file format version string [displ, size]. if both are
         *                 zero, default to v0.4 for diverge.output.read().
         *
         * header[1] - nk
         * header[2] - nb
         * header[3] - nff
         * header[4] - SU2
         * header[5-6] - formfactors [displ, size]
         * header[7-8] - susc P [displ, size]
         * header[9-10] - lin gap P [displ, size]
         * header[11-12] - susc C [displ, size]
         * header[13-14] - lin gap C [displ, size]
         * header[15-16] - susc D [displ, size]
         * header[17-18] - lin gap D [displ, size]
         * header[19] - lin gap num values/vectors
         * header[20] - lin gap matrix size (one leg, i.e., nk*nb*nb)
         *
         * header[62] - MAGIC_NUMBER_POST_PROCESSING
         * header[63] - total file size (in 64bit units)
         */
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
