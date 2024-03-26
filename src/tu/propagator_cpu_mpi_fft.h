#pragma once

#include "../misc/mpi_functions.h"
#include "../diverge_common.h"
#include <fftw3.h>

typedef struct fftw_mpi_plan_t fftw_mpi_plan_t;

fftw_mpi_plan_t* fftw_mpi_plan_init( int dims[3], index_t howmany,
        complex128_t* in, complex128_t* out, int dir, complex128_t* buf );
fftw_mpi_plan_t* fftw_mpi_plan_init_l( index_t dims[3], index_t howmany,
        complex128_t* in, complex128_t* out, int dir, complex128_t* buf );
void fftw_mpi_plan_free( fftw_mpi_plan_t* p );

void fftw_mpi_plan_execute( fftw_mpi_plan_t* p, complex128_t* in,
        complex128_t* out, bool input_transposed );
void fftw_mpi_plan_enable_timings( fftw_mpi_plan_t* p, bool enabled );
void fftw_mpi_plan_timings( fftw_mpi_plan_t* p, double tvec[3] ); // reorder, fft, comm
index_t fftw_mpi_plan_get_local_n0( fftw_mpi_plan_t* p );
index_t fftw_mpi_plan_get_local_0_start( fftw_mpi_plan_t* p );
complex128_t* fftw_mpi_plan_get_in( fftw_mpi_plan_t* p );
complex128_t* fftw_mpi_plan_get_out( fftw_mpi_plan_t* p );

