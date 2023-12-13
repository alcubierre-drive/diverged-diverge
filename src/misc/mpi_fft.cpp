#include "mpi_fft.hpp"
#include "mpi_functions.h"

fftw_mpi_plan::fftw_mpi_plan( int dims_[3], index_t howmany, complex128_t* in, complex128_t* out, int dir ) {
    index_t dims[3] = {dims_[0], dims_[1], dims_[2]};
    constructor( dims, howmany, in, out, dir );
}

fftw_mpi_plan::fftw_mpi_plan( index_t dims[3], index_t howmany, complex128_t* in, complex128_t* out, int dir ) {
    constructor( dims, howmany, in, out, dir );
}

#if defined(USE_MPI_FFTW) && defined(USE_MPI)
#include <fftw3-mpi.h>

void fftw_mpi_plan::constructor( index_t dims_[3], index_t howmany_, complex128_t* in_,
        complex128_t* out_, int dir ) {
    in = in_;
    out = out_;
    howmany = howmany_;
    nranks = diverge_mpi_comm_size();
    rank = diverge_mpi_comm_rank();
    dims[0] = dims_[0];
    dims[1] = dims_[1];
    dims[2] = dims_[2];
    tvec[0] = tvec[1] = tvec[2] = 0.0;
    timings_enabled = true;
    ptrdiff_t alloc = fftw_mpi_local_size_many( 3, dims, howmany, FFTW_MPI_DEFAULT_BLOCK,
            diverge_mpi_get_comm(), &local_n0, &local_0_start );
    in_mpi = (complex128_t*)fftw_alloc_complex(alloc);
    if (in != out)
        out_mpi = (complex128_t*)fftw_alloc_complex(alloc);
    else
        out_mpi = in_mpi;
    buf = (complex128_t*)fftw_alloc_complex(howmany*dims[0]*dims[1]*dims[2]);
    p = fftw_mpi_plan_many_dft( 3, dims, howmany, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
            (fftw_complex*)in_mpi, (fftw_complex*)out_mpi, diverge_mpi_get_comm(), dir, FFTW_MEASURE );

    counts = (int*)calloc(nranks, sizeof(int));
    displs = (int*)calloc(nranks, sizeof(int));
    int count = local_n0*dims[1]*dims[2]*howmany;
    int displ = local_0_start*dims[1]*dims[2]*howmany;
    diverge_mpi_allgather_int( &count, counts, 1 );
    diverge_mpi_allgather_int( &displ, displs, 1 );
}

void fftw_mpi_plan::execute( complex128_t* in, complex128_t* out ) {
    double tick = 0, tock = 0;

    if (timings_enabled) tick = diverge_mpi_wtime();
    #pragma omp parallel for collapse(3)
    for (index_t h=0; h<howmany; ++h)
    for (index_t i=0; i<local_n0; ++i)
    for (index_t j=0; j<dims[1]*dims[2]; ++j)
        in_mpi[IDX3( i, j, h, dims[1]*dims[2], howmany )] =
            in[IDX3( h, i+local_0_start, j, dims[0], dims[1]*dims[2] )];
    if (timings_enabled) tock = diverge_mpi_wtime();
    tvec[0] += tock-tick;

    if (timings_enabled) tick = diverge_mpi_wtime();
    fftw_mpi_execute_dft( p, (fftw_complex*)in_mpi, (fftw_complex*)out_mpi );
    if (timings_enabled) tock = diverge_mpi_wtime();
    tvec[1] += tock-tick;

    if (timings_enabled) tick = diverge_mpi_wtime();
    memcpy( buf+displs[rank], out_mpi, counts[rank]*sizeof(complex128_t) );
    if (timings_enabled) tock = diverge_mpi_wtime();
    tvec[0] += tock-tick;

    if (timings_enabled) tick = diverge_mpi_wtime();
    diverge_mpi_allgatherv( buf, counts, displs );
    if (timings_enabled) tock = diverge_mpi_wtime();
    tvec[2] += tock-tick;

    if (timings_enabled) tick = diverge_mpi_wtime();
    #pragma omp parallel for collapse(2)
    for (index_t h=0; h<howmany; ++h)
    for (index_t r=0; r<nranks; ++r) {
        complex128_t* start = buf+displs[r];
        index_t _local_0_start = displs[r]/howmany/dims[1]/dims[2],
                _local_n0 = counts[r]/howmany/dims[1]/dims[2];

        for (index_t i=0; i<_local_n0; ++i)
        for (index_t j=0; j<dims[1]*dims[2]; ++j) {
            out[IDX3( h, _local_0_start+i, j, dims[0], dims[1]*dims[2] )] =
                start[IDX3( i, j, h, dims[1]*dims[2], howmany )];
        }
    }
    if (timings_enabled) tock = diverge_mpi_wtime();
    tvec[0] += tock-tick;
}

fftw_mpi_plan::~fftw_mpi_plan() {
    if (in_mpi != out_mpi)
        fftw_free(out_mpi);
    fftw_free(in_mpi);
    fftw_free(buf);
    fftw_destroy_plan(p);
    free(counts);
    free(displs);
}
#else // USE_MPI_FFTW && USE_MPI
void fftw_mpi_plan::constructor( index_t dims_[3], index_t howmany, complex128_t* in,
        complex128_t* out, int dir ) {
    tvec[0] = tvec[1] = tvec[2] = 0.0;
    timings_enabled = true;
    int dims[3] = { static_cast<int>(dims_[0]), static_cast<int>(dims_[1]), static_cast<int>(dims_[2]) };
    p = fftw_plan_many_dft( 3, dims, howmany, (fftw_complex*)in, dims, 1, dims[0]*dims[1]*dims[2],
            (fftw_complex*)out, dims, 1, dims[0]*dims[1]*dims[2], dir, FFTW_MEASURE );
}

void fftw_mpi_plan::execute( complex128_t* in, complex128_t* out ) {
    double tick = 0, tock = 0;
    if (timings_enabled) tick = diverge_mpi_wtime();
    fftw_execute_dft( p, (fftw_complex*)in, (fftw_complex*)out );
    if (timings_enabled) tock = diverge_mpi_wtime();
    tvec[1] += tock-tick;
}

fftw_mpi_plan::~fftw_mpi_plan() {
    fftw_destroy_plan(p);
}
#endif // USE_MPI_FFTW && USE_MPI
