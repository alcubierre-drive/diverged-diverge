/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "init_internal_libs.h"
#include "dependency_tree.h"
#include "shared_mem.h"
#include "mpi_functions.h"

#include "../diverge_common.h"
#include <fftw3.h>
#include <omp.h>
#include <unistd.h>
#include <limits.h>

#if defined(USE_MPI) && defined(USE_MPI_FFTW)
#include <fftw3-mpi.h>
#endif

static int dvg_has_internal_libs_init = 0;

#define INT_BIT (CHAR_BIT * sizeof(int))
typedef struct {
    int val:(INT_BIT-2);
    int set:1;
    int warn:1;
} tag_t;

static struct {
    tag_t fftw_num_threads;
    tag_t omp_num_threads;
    tag_t omp_implicit_num_threads;
    tag_t symcheck_mpi;
    tag_t tu_symcheck_mpi;
    tag_t shared_malloc_extra_colors;
} dvg_env = {0};

static int dvg_has_fftw_init = 0;
static int dvg_has_mpi_fftw_init = 0;
static int dvg_previous_log_color = -1;

static void init_internal_libs_omp_implicit( int nthr );
static void init_internal_libs_omp_diverge( int nthr );
static void init_internal_libs_fftw( int nthr );
static void read_environment( void );

int environ_tu_symcheck_mpi( void ) {
    return dvg_env.tu_symcheck_mpi.set;
}

int environ_symcheck_mpi( void ) {
    return dvg_env.symcheck_mpi.set;
}

void init_internal_libs( void ) {
    read_environment();

    // we need to equalize the number of threads on each rank.
#define EQUALIZE_X( name, x ) { \
    int equalize_buf = dvg_env.name.x; \
    diverge_mpi_allreduce_int_max_inplace( &equalize_buf, 1 ); \
    dvg_env.name.x = equalize_buf; \
}
#define EQUALIZE( name ) { \
    EQUALIZE_X( name, val ); \
    EQUALIZE_X( name, set ); \
    EQUALIZE_X( name, warn ); \
}
    EQUALIZE( fftw_num_threads );
    EQUALIZE( omp_implicit_num_threads );
    EQUALIZE( omp_num_threads );
    EQUALIZE( symcheck_mpi );
    EQUALIZE( tu_symcheck_mpi );
    EQUALIZE( shared_malloc_extra_colors );

    init_internal_libs_omp_implicit( dvg_env.omp_implicit_num_threads.val );
    init_internal_libs_omp_diverge( dvg_env.omp_num_threads.val );
    init_internal_libs_fftw( dvg_env.fftw_num_threads.val );

    // logging
    dvg_previous_log_color = mpi_log_get_colors();
    #ifndef DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT
    if (dvg_previous_log_color)
        mpi_log_set_colors(isatty(STDERR_FILENO));
    #endif

#define NOTIFY_OR_WARN( nam, ... ) { \
    if (dvg_env.nam.warn) { \
        mpi_wrn_printf( __VA_ARGS__ ); \
    } else if (dvg_env.nam.set) { \
        mpi_log_printf( __VA_ARGS__ ); \
    } \
}
    NOTIFY_OR_WARN( omp_num_threads, "using %d threads for divERGe OpenMP\n", dvg_env.omp_num_threads.val );
    NOTIFY_OR_WARN( fftw_num_threads, "using %d threads for FFTW\n", dvg_env.fftw_num_threads.val );
    NOTIFY_OR_WARN( omp_implicit_num_threads, "using %d threads for implicit OpenMP\n", dvg_env.omp_implicit_num_threads.val );
    NOTIFY_OR_WARN( shared_malloc_extra_colors, "using extra colors for MPI shared memory distribution\n" );
    NOTIFY_OR_WARN( symcheck_mpi, "enable symmetry check for MPI\n" );
    NOTIFY_OR_WARN( tu_symcheck_mpi, "enable TU symmetry check for MPI\n" );

    dependency_tree_init();
    shared_malloc_init( dvg_env.shared_malloc_extra_colors.set );

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

int diverge_internal_omp_num_threads( void ) {
    if (!dvg_has_internal_libs_init)
        return 1;
    else
        return dvg_env.omp_num_threads.val;
}


int diverge_internal_fftw_num_threads( void ) {
    if (!dvg_has_internal_libs_init)
        return 1;
    else
        return dvg_env.fftw_num_threads.val;
}

void diverge_internal_force_thread_limit( int nthr ) {
    init_internal_libs_omp_diverge( nthr );
    init_internal_libs_omp_implicit( nthr );
    init_internal_libs_fftw( nthr );
}

static void read_environment( void ) {
    // read strings
    char* env_dvg_omp_num_threads = getenv( "DIVERGE_OMP_NUM_THREADS" );
    char* env_dvg_fftw_num_threads = getenv( "DIVERGE_FFTW_NUM_THREADS" );
    char* env_dvg_omp_implicit_num_threads = getenv( "DIVERGE_OMP_IMPLICIT_NUM_THREADS" );
    char* env_dvg_shared_malloc_extra_colors = getenv( "DIVERGE_SHARED_MALLOC_EXTRA_COLORS" );
    char* env_dvg_symcheck_mpi = getenv( "DIVERGE_SYMCHECK_MPI" );
    char* env_dvg_tu_symcheck_mpi = getenv( "DIVERGE_TU_SYMCHECK_MPI" );

#define SET_ENV_VAR( nam ) \
    if ((dvg_env.nam.set = (env_dvg_##nam != NULL))) { \
        dvg_env.nam.val = atoi( env_dvg_##nam ); \
    }
    SET_ENV_VAR( omp_num_threads );
    SET_ENV_VAR( fftw_num_threads );
    SET_ENV_VAR( omp_implicit_num_threads );
    SET_ENV_VAR( shared_malloc_extra_colors );
    SET_ENV_VAR( symcheck_mpi );
    SET_ENV_VAR( tu_symcheck_mpi );

    // and check
#define CHECK_ENV_VAR( nam, default ) \
    if (dvg_env.nam.val <= 0) { \
        dvg_env.nam.warn = (dvg_env.nam.val < 0); \
        dvg_env.nam.val = default; \
    }
    CHECK_ENV_VAR( omp_num_threads, omp_get_max_threads() );
    CHECK_ENV_VAR( fftw_num_threads, omp_get_max_threads() );
    CHECK_ENV_VAR( omp_implicit_num_threads, omp_get_max_threads() );
}

static void init_internal_libs_fftw( int nthr ) {
    dvg_env.fftw_num_threads.val = nthr;
    #ifndef USE_SERIAL_FFTW
    if (!dvg_has_fftw_init)
        fftw_init_threads();
    fftw_plan_with_nthreads( dvg_env.fftw_num_threads.val );
    #else
    dvg_env.fftw_num_threads.val = 1;
    #endif
    dvg_has_fftw_init = 1;

    #if defined(USE_MPI_FFTW) && defined(USE_MPI)
    if (!dvg_has_mpi_fftw_init)
        fftw_mpi_init();
    #endif
    dvg_has_mpi_fftw_init = 1;
}

static void init_internal_libs_omp_implicit( int nthr ) {
    dvg_env.omp_implicit_num_threads.val = nthr;

    #ifdef DIVERGE_OPENBLAS_LIMIT_THREADS
    if (dvg_env.omp_implicit_num_threads.val > DIVERGE_OPENBLAS_LIMIT_THREADS) {
        mpi_wrn_printf("export OMP_NUM_THREADS=%i (due to OpenBLAS config)\n",
                DIVERGE_OPENBLAS_LIMIT_THREADS);
        dvg_env.omp_implicit_num_threads.val = DIVERGE_OPENBLAS_LIMIT_THREADS;
    }
    #endif

    omp_set_num_threads( dvg_env.omp_implicit_num_threads.val );
}

static void init_internal_libs_omp_diverge( int nthr ) {
    dvg_env.omp_num_threads.val = nthr;
}

