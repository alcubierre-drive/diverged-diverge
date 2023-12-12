#pragma once

#include "mpi_functions.h"
#include "../diverge_common.h"
#include <fftw3.h>

class fftw_mpi_plan {
    public:
        fftw_mpi_plan( index_t dims[3], index_t howmany, complex128_t* in, complex128_t* out, int dir );
        fftw_mpi_plan( int dims[3], index_t howmany, complex128_t* in, complex128_t* out, int dir );
        ~fftw_mpi_plan();
        void execute( complex128_t* in, complex128_t* out );
    private:
        ptrdiff_t local_n0;
        ptrdiff_t local_0_start;
        fftw_plan p;
        complex128_t* in;
        complex128_t* out;
        complex128_t* in_mpi = NULL;
        complex128_t* out_mpi = NULL;
        complex128_t* buf = NULL;
        index_t dims[3];
        index_t howmany;

        int nranks, rank;
        int* counts;
        int* displs;

        void constructor( index_t dims[3], index_t howmany, complex128_t* in, complex128_t* out, int dir );
};
