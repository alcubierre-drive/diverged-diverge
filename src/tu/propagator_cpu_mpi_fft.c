#include "propagator_cpu_mpi_fft.h"
#include "../misc/mpi_functions.h"
#include "propagator_cpu_mpi.h"

unsigned fftw_mpi_plan_patience = FFTW_MEASURE;

struct fftw_mpi_plan_t {
    bool timings_enabled:1;
    bool free_buf:1;
    ptrdiff_t local_n0;
    ptrdiff_t local_0_start;
    fftw_plan p;
    complex128_t* in;
    complex128_t* out;
    complex128_t* in_mpi;
    complex128_t* out_mpi;
    complex128_t* buf;
    index_t dims[3];
    index_t howmany;

    int nranks, rank;
    int* counts;
    int* displs;

    double tvec[3];
};

fftw_mpi_plan_t* fftw_mpi_plan_init( int dims[3], index_t howmany,
        complex128_t* in, complex128_t* out, int dir, complex128_t* buf ) {
    index_t dims_[3] = {dims[0], dims[1], dims[2]};
    return fftw_mpi_plan_init_l( dims_, howmany, in, out, dir, buf );
}

void fftw_mpi_plan_enable_timings( fftw_mpi_plan_t* p, bool enabled ) {
    p->timings_enabled = enabled;
}
void fftw_mpi_plan_timings( fftw_mpi_plan_t* p, double tvec[3] ) {
    memcpy( tvec, p->tvec, sizeof(double)*3 );
}
index_t fftw_mpi_plan_get_local_n0( fftw_mpi_plan_t* p ) {
    return p->local_n0;
}
index_t fftw_mpi_plan_get_local_0_start( fftw_mpi_plan_t* p ) {
    return p->local_0_start;
}
complex128_t* fftw_mpi_plan_get_in( fftw_mpi_plan_t* p ) {
    return p->in;
}
complex128_t* fftw_mpi_plan_get_out( fftw_mpi_plan_t* p ) {
    return p->out;
}

#if defined(USE_MPI_FFTW) && defined(USE_MPI)
#include <fftw3-mpi.h>

static inline complex128_t* fftw_alloc_complex_wrap_( size_t sz, const char* f, int l ) {
    complex128_t* result = (complex128_t*)fftw_alloc_complex( sz );
    mpi_vrb_printf( "allocating %.2fGiB (%s:%i)\n",
            (double)(sz*sizeof(complex128_t)) / (double)(1024.*1024.*1024.), f, l );
    return result;
}
#define fftw_alloc_complex_wrap( sz ) fftw_alloc_complex_wrap_( sz, __FILE__, __LINE__ )


fftw_mpi_plan_t* fftw_mpi_plan_init_l( index_t dims[3], index_t howmany,
        complex128_t* in, complex128_t* out, int dir, complex128_t* buf ) {

    mpi_wrn_printf( "Using the old MPI-FFT algorithm for TU propagators is deprecated\n" );

    fftw_mpi_plan_t* p = calloc(1, sizeof(fftw_mpi_plan_t));
    p->free_buf = true;
    p->timings_enabled = true;
    p->in = in;
    p->out = out;
    p->howmany = howmany;
    p->nranks = diverge_mpi_comm_size();
    p->rank = diverge_mpi_comm_rank();
    p->dims[0] = dims[0];
    p->dims[1] = dims[1];
    p->dims[2] = dims[2];
    p->tvec[0] = p->tvec[1] = p->tvec[2] = 0.0;
    ptrdiff_t alloc = fftw_mpi_local_size_many( 3, dims, howmany, FFTW_MPI_DEFAULT_BLOCK,
            diverge_mpi_get_comm(), &p->local_n0, &p->local_0_start );
    p->in_mpi = fftw_alloc_complex_wrap(alloc);
    if (p->in != p->out)
        p->out_mpi = fftw_alloc_complex_wrap(alloc);
    else
        p->out_mpi = p->in_mpi;

    if (buf) {
        p->buf = buf;
        p->free_buf = false;
    } else {
        p->buf = fftw_alloc_complex_wrap(p->howmany*p->dims[0]*p->dims[1]*p->dims[2]);
    }

    p->p = fftw_mpi_plan_many_dft( 3, p->dims, p->howmany, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
            (fftw_complex*)p->in_mpi, (fftw_complex*)p->out_mpi, diverge_mpi_get_comm(), dir,
            fftw_mpi_plan_patience );

    p->counts = calloc(p->nranks, sizeof(int));
    p->displs = calloc(p->nranks, sizeof(int));
    int count = p->local_n0*p->dims[1]*p->dims[2]*p->howmany;
    int displ = p->local_0_start*p->dims[1]*p->dims[2]*p->howmany;
    diverge_mpi_allgather_int( &count, p->counts, 1 );
    diverge_mpi_allgather_int( &displ, p->displs, 1 );

    return p;
}

void fftw_mpi_plan_execute( fftw_mpi_plan_t* p, complex128_t* in,
        complex128_t* out, bool input_transposed ) {

    double tick = 0, tock = 0;

    if (!input_transposed) {

        if (p->timings_enabled) tick = diverge_mpi_wtime();
        // reorder from (h,i+l0,j) to (i,j,h)
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t h=0; h<p->howmany; ++h)
        for (index_t i=0; i<p->local_n0; ++i)
        for (index_t j=0; j<p->dims[1]*p->dims[2]; ++j)
            p->in_mpi[IDX3( i, j, h, p->dims[1]*p->dims[2], p->howmany )] =
                   in[IDX3( h, i+p->local_0_start, j, p->dims[0], p->dims[1]*p->dims[2] )];
        if (p->timings_enabled) tock = diverge_mpi_wtime();
        p->tvec[0] += tock-tick;

    } else {

        if (p->timings_enabled) tick = diverge_mpi_wtime();
        memcpy( p->in_mpi, in + p->local_0_start * p->dims[1] * p->dims[2] * p->howmany,
                sizeof(complex128_t) * p->local_n0 * p->dims[1] * p->dims[2] * p->howmany );
        if (p->timings_enabled) tock = diverge_mpi_wtime();
        p->tvec[0] += tock-tick;

    }

    if (p->timings_enabled) tick = diverge_mpi_wtime();
    fftw_mpi_execute_dft( p->p, (fftw_complex*)p->in_mpi, (fftw_complex*)p->out_mpi );
    if (p->timings_enabled) tock = diverge_mpi_wtime();
    p->tvec[1] += tock-tick;

    if (!input_transposed) {

        if (p->timings_enabled) tick = diverge_mpi_wtime();
        memcpy( p->buf+p->displs[p->rank], p->out_mpi, p->counts[p->rank]*sizeof(complex128_t) );
        if (p->timings_enabled) tock = diverge_mpi_wtime();
        p->tvec[0] += tock-tick;

        if (p->timings_enabled) tick = diverge_mpi_wtime();
        diverge_mpi_allgatherv( p->buf, p->counts, p->displs );
        if (p->timings_enabled) tock = diverge_mpi_wtime();
        p->tvec[2] += tock-tick;

        if (p->timings_enabled) tick = diverge_mpi_wtime();
        // reorder from (i,j,h) to (h,i+l0,j)
        #pragma omp parallel for collapse(2)
        for (index_t h=0; h<p->howmany; ++h)
        for (index_t r=0; r<p->nranks; ++r) {
            complex128_t* start = p->buf+p->displs[r];
            index_t _local_0_start = p->displs[r]/p->howmany/p->dims[1]/p->dims[2],
                    _local_n0 = p->counts[r]/p->howmany/p->dims[1]/p->dims[2];

            for (index_t i=0; i<_local_n0; ++i)
            for (index_t j=0; j<p->dims[1]*p->dims[2]; ++j) {
                out[IDX3( h, _local_0_start+i, j, p->dims[0], p->dims[1]*p->dims[2] )] =
                    start[IDX3( i, j, h, p->dims[1]*p->dims[2], p->howmany )];
            }
        }
        if (p->timings_enabled) tock = diverge_mpi_wtime();
        p->tvec[0] += tock-tick;

    } else {

        if (p->timings_enabled) tick = diverge_mpi_wtime();
        memcpy( out+p->displs[p->rank], p->out_mpi, p->counts[p->rank]*sizeof(complex128_t) );
        if (p->timings_enabled) tock = diverge_mpi_wtime();
        p->tvec[0] += tock-tick;

        if (p->timings_enabled) tick = diverge_mpi_wtime();
        diverge_mpi_allgatherv( out, p->counts, p->displs );
        if (p->timings_enabled) tock = diverge_mpi_wtime();
        p->tvec[2] += tock-tick;

    }
}

void fftw_mpi_plan_free( fftw_mpi_plan_t* p ) {
    if (p->in_mpi != p->out_mpi)
        fftw_free(p->out_mpi);
    fftw_free(p->in_mpi);
    if (p->free_buf) fftw_free(p->buf);
    fftw_destroy_plan(p->p);
    free(p->counts);
    free(p->displs);
    free(p);
}
#else // USE_MPI_FFTW && USE_MPI
fftw_mpi_plan_t* fftw_mpi_plan_init_l( index_t dims_[3], index_t howmany_,
        complex128_t* in_, complex128_t* out_, int dir, complex128_t* buf_ ) {

    mpi_wrn_printf( "Using the old MPI-FFT algorithm for TU propagators is deprecated\n" );

    fftw_mpi_plan_t* p = calloc(1, sizeof(fftw_mpi_plan_t));
    (void)buf_;
    p->tvec[0] = p->tvec[1] = p->tvec[2] = 0.0;
    p->timings_enabled = true;
    p->dims[0] = dims_[0];
    p->dims[1] = dims_[1];
    p->dims[2] = dims_[2];
    p->howmany = howmany_;
    p->local_0_start = 0;
    p->local_n0 = p->dims[0];
    int mydims[3] = { dims_[0], dims_[1], dims_[2] };
    p->in = in_;
    p->out = out_;
    p->p = fftw_plan_many_dft( 3, mydims, p->howmany, (fftw_complex*)p->in, mydims, 1, mydims[0]*mydims[1]*mydims[2],
            (fftw_complex*)p->out, mydims, 1, mydims[0]*mydims[1]*mydims[2], dir,
            fftw_mpi_plan_patience );
    return p;
}

void fftw_mpi_plan_execute( fftw_mpi_plan_t* p, complex128_t* in, complex128_t* out, bool input_transposed ) {
    double tick = 0, tock = 0;
    if (p->timings_enabled) tick = diverge_mpi_wtime();
    if (!input_transposed) {
        fftw_execute_dft( p->p, (fftw_complex*)in, (fftw_complex*)out );
    } else {
        p->buf = (complex128_t*)fftw_alloc_complex( p->dims[0]*p->dims[1]*p->dims[2]*p->howmany );

        // transpose
        for (index_t i=0; i<p->dims[0]*p->dims[1]*p->dims[2]; ++i)
        for (index_t h=0; h<p->howmany; ++h)
            p->buf[IDX2( h, i, p->dims[0]*p->dims[1]*p->dims[2] )] = in[IDX2( i, h, p->howmany )];

        // fft
        if (in == out) {
            fftw_execute_dft( p->p, (fftw_complex*)p->buf, (fftw_complex*)p->buf );
        } else {
            fftw_execute_dft( p->p, (fftw_complex*)p->buf, (fftw_complex*)out );
            memcpy( p->buf, out, sizeof(complex128_t)*p->dims[0]*p->dims[1]*p->dims[2]*p->howmany );
        }

        // transpose
        for (index_t i=0; i<p->dims[0]*p->dims[1]*p->dims[2]; ++i)
        for (index_t h=0; h<p->howmany; ++h)
            out[IDX2( i, h, p->howmany )] = p->buf[IDX2( h, i, p->dims[0]*p->dims[1]*p->dims[2] )];

        fftw_free( (fftw_complex*)p->buf );
    }
    if (p->timings_enabled) tock = diverge_mpi_wtime();
    p->tvec[1] += tock-tick;
}

void fftw_mpi_plan_free( fftw_mpi_plan_t* p ) {
    fftw_destroy_plan(p->p);
    free(p);
}
#endif // USE_MPI_FFTW && USE_MPI
