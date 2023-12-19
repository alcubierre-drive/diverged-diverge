/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "init_internal_libs.h"
#include "dependency_tree.h"
#include "shared_mem.h"

#include "../diverge_common.h"
#include <fftw3.h>
#include <omp.h>
#include <unistd.h>

#if defined(USE_MPI) && defined(USE_MPI_FFTW)
#include <fftw3-mpi.h>
#endif

static int dvg_has_internal_libs_init = 0;

static int dvg_has_fftw_init = 0;
static int dvg_has_mpi_fftw_init = 0;
static int dvg_previous_log_color = -1;

static int dvg_omp_num_threads = -1;
static int dvg_fftw_num_threads = -1;
static int dvg_omp_implicit_num_threads = -1;

static void init_internal_libs_omp_implicit( int nthr );
static void init_internal_libs_omp_diverge( int nthr );
static void init_internal_libs_fftw( int nthr );
static void read_environment( void );

void init_internal_libs( void ) {

    read_environment();

    init_internal_libs_omp_implicit( dvg_omp_implicit_num_threads );
    init_internal_libs_omp_diverge( dvg_omp_num_threads );
    init_internal_libs_fftw( dvg_fftw_num_threads );

    // logging
    dvg_previous_log_color = mpi_log_get_colors();
    #ifndef DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT
    if (dvg_previous_log_color)
        mpi_log_set_colors(isatty(STDERR_FILENO));
    #endif

    dependency_tree_init();
    shared_malloc_init();

    dvg_has_internal_libs_init = 1;

}

void finalize_internal_libs( void ) {

    dependency_tree_free();
    shared_malloc_finalize();

    if (dvg_has_mpi_fftw_init) {
        #if defined(USE_MPI_FFTW) && defined(USE_MPI)
        fftw_mpi_cleanup();
        #endif
        dvg_has_mpi_fftw_init = 0;
    }

    if (dvg_has_fftw_init) {
        #ifndef USE_SERIAL_FFTW
        fftw_cleanup_threads();
        #endif
        fftw_cleanup();
        dvg_has_fftw_init = 0;
    }
    mpi_log_set_colors(dvg_previous_log_color);
}

int diverge_omp_num_threads( void ) {
    if (!dvg_has_internal_libs_init)
        return 1;
    else
        return dvg_omp_num_threads;
}


int diverge_fftw_num_threads( void ) {
    if (!dvg_has_internal_libs_init)
        return 1;
    else
        return dvg_fftw_num_threads;
}

void diverge_force_thread_limit( int nthr ) {
    init_internal_libs_omp_diverge( nthr );
    init_internal_libs_omp_implicit( nthr );
    init_internal_libs_fftw( nthr );
}

static void read_environment( void ) {
    // read strings
    char* env_dvg_omp_num_threads = getenv( "DIVERGE_OMP_NUM_THREADS" );
    char* env_dvg_fftw_num_threads = getenv( "DIVERGE_FFTW_NUM_THREADS" );
    char* env_dvg_omp_implicit_num_threads = getenv( "DIVERGE_OMP_IMPLICIT_NUM_THREADS" );

    // set values
    if (env_dvg_omp_num_threads) {
        dvg_omp_num_threads = atoi(env_dvg_omp_num_threads);
        mpi_log_printf( "using %d threads for divERGe OpenMP\n", dvg_omp_num_threads );
    }
    if (env_dvg_fftw_num_threads) {
        dvg_fftw_num_threads = atoi(env_dvg_fftw_num_threads);
        mpi_log_printf( "using %d threads for FFTW\n", dvg_fftw_num_threads );
    }
    if (env_dvg_omp_implicit_num_threads) {
        dvg_omp_implicit_num_threads = atoi(env_dvg_omp_implicit_num_threads);
        mpi_log_printf( "using %d threads for implicit OpenMP\n", dvg_omp_implicit_num_threads );
    }

    // and check
    if (dvg_omp_num_threads <= 0) dvg_omp_num_threads = omp_get_max_threads();
    if (dvg_fftw_num_threads <= 0) dvg_fftw_num_threads = omp_get_max_threads();
    if (dvg_omp_implicit_num_threads <= 0) dvg_omp_implicit_num_threads = omp_get_max_threads();
}

static void init_internal_libs_fftw( int nthr ) {
    dvg_fftw_num_threads = nthr;
    #ifndef USE_SERIAL_FFTW
    if (!dvg_has_fftw_init)
        fftw_init_threads();
    fftw_plan_with_nthreads( dvg_fftw_num_threads );
    #else
    dvg_fftw_num_threads = 1;
    #endif
    dvg_has_fftw_init = 1;

    #if defined(USE_MPI_FFTW) && defined(USE_MPI)
    if (!dvg_has_mpi_fftw_init)
        fftw_mpi_init();
    #endif
    dvg_has_mpi_fftw_init = 1;
}

static void init_internal_libs_omp_implicit( int nthr ) {
    dvg_omp_implicit_num_threads = nthr;

    #ifdef DIVERGE_OPENBLAS_LIMIT_THREADS
    if (dvg_omp_implicit_num_threads > DIVERGE_OPENBLAS_LIMIT_THREADS) {
        mpi_wrn_printf("export OMP_NUM_THREADS=%i (due to OpenBLAS config)\n",
                DIVERGE_OPENBLAS_LIMIT_THREADS);
        dvg_omp_implicit_num_threads = DIVERGE_OPENBLAS_LIMIT_THREADS;
    }
    #endif

    omp_set_num_threads( dvg_omp_implicit_num_threads );
}

static void init_internal_libs_omp_diverge( int nthr ) {
    dvg_omp_num_threads = nthr;
}

