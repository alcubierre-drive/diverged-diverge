/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_constants.h"
#include "misc/generate_symmetries.h"
#include <limits.h>
#include "diverge_flow_step.h"
#include "diverge_model_output.h"
#include "diverge_internals_struct.h"
#include "diverge_patching_helpers.h"
#include "grid/post_processing.h"
#include "tu/postproduction.h"
#include "misc/batched_svd_private.h"
#include "misc/mergesort.h"
#include "misc/text_io.h"
#include "misc/batched_eigen.h"

/* defaults that live in a translation unit (or a CUDA-only header) that cannot
 * be included from here. keep in sync with the file given in the comment. */
#ifndef DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER // diverge_flow_step.c
#define DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER 1000
#endif
#ifndef DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL // diverge_flow_step.c
#define DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL 1.e-6
#endif
#ifndef CUBLAS_BATCHED_GEMM_CALL // misc/batched_gemm_small.c
#define CUBLAS_BATCHED_GEMM_CALL cublasZgemmStridedBatched_64
#endif
#ifndef BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS // misc/batched_gemms_cpu.cpp
#define BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS 4
#endif
#ifndef CUDA_VERSION_STR // misc/compilation_status.c
#define CUDA_VERSION_STR "CUDA"
#endif
#ifndef DIVERGE_LOG_PREFIX // misc/mpi_log.c
#define DIVERGE_LOG_PREFIX "divERGe"
#endif
#ifndef LDOS_KERNEL_GPU_DEV_MAXNOS // misc/dos.cu
#define LDOS_KERNEL_GPU_DEV_MAXNOS 512
#endif
#ifndef LDOS_KERNEL_GPU_DEV_NTHREADS // misc/dos.cu
#define LDOS_KERNEL_GPU_DEV_NTHREADS 512
#endif
#ifndef DIVERGE_POST_MAGIC_NUMBER_NPATCH // npatch/output.c
#define DIVERGE_POST_MAGIC_NUMBER_NPATCH 'V'
#endif
#ifndef MAX_THREADS_PER_BLOCK // npatch/kernels_gpu_helpers.h
#define MAX_THREADS_PER_BLOCK 1024
#endif
#ifndef CUDA_MEM_ALIGN // tu/propagator_gpu.c
#define CUDA_MEM_ALIGN 256
#endif
#ifndef PACK_MPI_MULTIPLIER // tu/propagator_gpu_mpi.c
#define PACK_MPI_MULTIPLIER 128
#endif

/* DEFINEs that carry a value but have no default at all. the 0 below marks
 * them as unset, PRINT_LI_OPT emits #undef/-U for those. */
#ifndef DIVERGE_OPENBLAS_LIMIT_THREADS
#define DIVERGE_OPENBLAS_LIMIT_THREADS 0
#endif
#ifndef USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE
#define USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE 0
#endif

/* test/ctest.h turns CT_NO_FORK on for MPI builds, mirror that rule here */
#if !defined(CT_NO_FORK) && defined(USE_MPI)
#define CT_NO_FORK
#endif

/* the remaining DEFINEs are pure on/off switches, i.e. they are queried with
 * #ifdef only and may be defined to nothing. map them to 0/1 constants. */
#ifdef USE_CUDA
#define HAS_USE_CUDA 1
#else
#define HAS_USE_CUDA 0
#endif
#ifdef CUDA_CIRCUMVENT_CONSTEXPR
#define HAS_CUDA_CIRCUMVENT_CONSTEXPR 1
#else
#define HAS_CUDA_CIRCUMVENT_CONSTEXPR 0
#endif
#ifdef CUDA_DEFINE_ATOMIC_ADD
#define HAS_CUDA_DEFINE_ATOMIC_ADD 1
#else
#define HAS_CUDA_DEFINE_ATOMIC_ADD 0
#endif
#ifdef DIVERGE_GRID_PARALLEL_BATCHED_GEMM
#define HAS_DIVERGE_GRID_PARALLEL_BATCHED_GEMM 1
#else
#define HAS_DIVERGE_GRID_PARALLEL_BATCHED_GEMM 0
#endif
#ifdef DIVERGE_LOG_COLORLESS
#define HAS_DIVERGE_LOG_COLORLESS 1
#else
#define HAS_DIVERGE_LOG_COLORLESS 0
#endif
#ifdef DIVERGE_LOG_GRAYSCALE
#define HAS_DIVERGE_LOG_GRAYSCALE 1
#else
#define HAS_DIVERGE_LOG_GRAYSCALE 0
#endif
#ifdef DIVERGE_LOG_NAMELESS
#define HAS_DIVERGE_LOG_NAMELESS 1
#else
#define HAS_DIVERGE_LOG_NAMELESS 0
#endif
#ifdef DIVERGE_NO_ABORT_ON_GPU_ERROR
#define HAS_DIVERGE_NO_ABORT_ON_GPU_ERROR 1
#else
#define HAS_DIVERGE_NO_ABORT_ON_GPU_ERROR 0
#endif
#ifdef DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT
#define HAS_DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT 1
#else
#define HAS_DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT 0
#endif
#ifdef DIVERGE_USE_ZHEEVD
#define HAS_DIVERGE_USE_ZHEEVD 1
#else
#define HAS_DIVERGE_USE_ZHEEVD 0
#endif
#ifdef DIVERGE_SKIP_TESTS
#define HAS_DIVERGE_SKIP_TESTS 1
#else
#define HAS_DIVERGE_SKIP_TESTS 0
#endif
#ifdef DIVERGE_SKIP_TESTS_LINALG
#define HAS_DIVERGE_SKIP_TESTS_LINALG 1
#else
#define HAS_DIVERGE_SKIP_TESTS_LINALG 0
#endif
#ifdef CT_NO_FORK
#define HAS_CT_NO_FORK 1
#else
#define HAS_CT_NO_FORK 0
#endif
#ifdef DIVERGE_QNAN_DISABLE_CLANG_OPTS
#define HAS_DIVERGE_QNAN_DISABLE_CLANG_OPTS 1
#else
#define HAS_DIVERGE_QNAN_DISABLE_CLANG_OPTS 0
#endif
#ifdef DIVERGE_TU_INPLACE_PERMUTE
#define HAS_DIVERGE_TU_INPLACE_PERMUTE 1
#else
#define HAS_DIVERGE_TU_INPLACE_PERMUTE 0
#endif
#ifdef DIVERGE_TU_GF_COARSIFY_FFT
#define HAS_DIVERGE_TU_GF_COARSIFY_FFT 1
#else
#define HAS_DIVERGE_TU_GF_COARSIFY_FFT 0
#endif
#ifdef DIVERGE_UNIQUE_ORBFF_FIND_HASHTABLE
#define HAS_DIVERGE_UNIQUE_ORBFF_FIND_HASHTABLE 1
#else
#define HAS_DIVERGE_UNIQUE_ORBFF_FIND_HASHTABLE 0
#endif
#ifdef TU_POST_USE_BATCHED_SVD_U
#define HAS_TU_POST_USE_BATCHED_SVD_U 1
#else
#define HAS_TU_POST_USE_BATCHED_SVD_U 0
#endif
#ifdef TU_PROJ_TIMING
#define HAS_TU_PROJ_TIMING 1
#else
#define HAS_TU_PROJ_TIMING 0
#endif
#ifdef TU_VERTEX_RECO_GPU_ZGEMM
#define HAS_TU_VERTEX_RECO_GPU_ZGEMM 1
#else
#define HAS_TU_VERTEX_RECO_GPU_ZGEMM 0
#endif
#ifdef CLOG_MISSING
#define HAS_CLOG_MISSING 1
#else
#define HAS_CLOG_MISSING 0
#endif
#ifdef LEAVE_EIGEN_BROKEN
#define HAS_LEAVE_EIGEN_BROKEN 1
#else
#define HAS_LEAVE_EIGEN_BROKEN 0
#endif
#ifdef MKL_FORCE_AVX_KERNEL_ON_EPYC
#define HAS_MKL_FORCE_AVX_KERNEL_ON_EPYC 1
#else
#define HAS_MKL_FORCE_AVX_KERNEL_ON_EPYC 0
#endif
#ifdef NDEBUG
#define HAS_NDEBUG 1
#else
#define HAS_NDEBUG 0
#endif
#ifdef NDEBUG_SKIP_ERROR_LINES
#define HAS_NDEBUG_SKIP_ERROR_LINES 1
#else
#define HAS_NDEBUG_SKIP_ERROR_LINES 0
#endif
#ifdef SKIP_EIGEN3_INCLUDE_PREFIX
#define HAS_SKIP_EIGEN3_INCLUDE_PREFIX 1
#else
#define HAS_SKIP_EIGEN3_INCLUDE_PREFIX 0
#endif
#ifdef SKIP_FLOW_TIMING
#define HAS_SKIP_FLOW_TIMING 1
#else
#define HAS_SKIP_FLOW_TIMING 0
#endif
#ifdef USE_ARPACK_EIGENSOLVER
#define HAS_USE_ARPACK_EIGENSOLVER 1
#else
#define HAS_USE_ARPACK_EIGENSOLVER 0
#endif
#ifdef USE_COMPRESSED_OUTPUT
#define HAS_USE_COMPRESSED_OUTPUT 1
#else
#define HAS_USE_COMPRESSED_OUTPUT 0
#endif
#ifdef USE_DYNAMIC_RECONSTRUCT_SCHEDULE
#define HAS_USE_DYNAMIC_RECONSTRUCT_SCHEDULE 1
#else
#define HAS_USE_DYNAMIC_RECONSTRUCT_SCHEDULE 0
#endif
#ifdef USE_ALTERNATIVE_TIME
#define HAS_USE_ALTERNATIVE_TIME 1
#else
#define HAS_USE_ALTERNATIVE_TIME 0
#endif
#ifdef USE_EIGEN_BATCHED_GEMM
#define HAS_USE_EIGEN_BATCHED_GEMM 1
#else
#define HAS_USE_EIGEN_BATCHED_GEMM 0
#endif
#ifdef USE_EIGEN3_CPU_EIGENSOLVER
#define HAS_USE_EIGEN3_CPU_EIGENSOLVER 1
#else
#define HAS_USE_EIGEN3_CPU_EIGENSOLVER 0
#endif
#ifdef USE_MKL
#define HAS_USE_MKL 1
#else
#define HAS_USE_MKL 0
#endif
#ifdef USE_MPI
#define HAS_USE_MPI 1
#else
#define HAS_USE_MPI 0
#endif
#ifdef USE_NO_BLAS_VERTEX_LOOP
#define HAS_USE_NO_BLAS_VERTEX_LOOP 1
#else
#define HAS_USE_NO_BLAS_VERTEX_LOOP 0
#endif
#ifdef USE_NO_LAPACKE
#define HAS_USE_NO_LAPACKE 1
#else
#define HAS_USE_NO_LAPACKE 0
#endif
#ifdef USE_SERIAL_FFTW
#define HAS_USE_SERIAL_FFTW 1
#else
#define HAS_USE_SERIAL_FFTW 0
#endif
#ifdef USE_SHARED_MEM
#define HAS_USE_SHARED_MEM 1
#else
#define HAS_USE_SHARED_MEM 0
#endif
#ifdef SHARED_MEM_MPI_IMPL
#define HAS_SHARED_MEM_MPI_IMPL 1
#else
#define HAS_SHARED_MEM_MPI_IMPL 0
#endif
#ifdef SHARED_MEM_POSIX_IMPL
#define HAS_SHARED_MEM_POSIX_IMPL 1
#else
#define HAS_SHARED_MEM_POSIX_IMPL 0
#endif
#ifdef MEMFD_NOT_SUPPORTED
#define HAS_MEMFD_NOT_SUPPORTED 1
#else
#define HAS_MEMFD_NOT_SUPPORTED 0
#endif
#ifdef USE_MPI_FFTW
#define HAS_USE_MPI_FFTW 1
#else
#define HAS_USE_MPI_FFTW 0
#endif
#ifdef TU_GPU_MPI_ALLOC_PRINT
#define HAS_TU_GPU_MPI_ALLOC_PRINT 1
#else
#define HAS_TU_GPU_MPI_ALLOC_PRINT 0
#endif
#ifdef MPI_VERBOSE_ALLOCATION_PRINT
#define HAS_MPI_VERBOSE_ALLOCATION_PRINT 1
#else
#define HAS_MPI_VERBOSE_ALLOCATION_PRINT 0
#endif
#ifdef TRANSPOSE_MPI_FFT_BUFFERS
#define HAS_TRANSPOSE_MPI_FFT_BUFFERS 1
#else
#define HAS_TRANSPOSE_MPI_FFT_BUFFERS 0
#endif
#ifdef DIVERGE_OLD_W90_READER
#define HAS_DIVERGE_OLD_W90_READER 1
#else
#define HAS_DIVERGE_OLD_W90_READER 0
#endif
#ifdef DIVERGE_TU_PROJ_TO_R_VTREE
#define HAS_DIVERGE_TU_PROJ_TO_R_VTREE 1
#else
#define HAS_DIVERGE_TU_PROJ_TO_R_VTREE 0
#endif
#ifdef XVTREE_ALIGNMENT_PRINT
#define HAS_XVTREE_ALIGNMENT_PRINT 1
#else
#define HAS_XVTREE_ALIGNMENT_PRINT 0
#endif

#define DIVERGE_STRINGIFY_(X) #X
/** stringify a DEFINE that expands to a token instead of a string literal */
#define DIVERGE_STRINGIFY(X) DIVERGE_STRINGIFY_(X)

static str1024_t str1024( const char* s ) {
    str1024_t r = {{0}};
    snprintf( r.str, sizeof(r.str), "%s", s );
    return r;
}

diverge_constants_t diverge_constants( void ) {
    return (diverge_constants_t){
        ._MAX_N_ORBS = MAX_N_ORBS,
        ._MAX_NAME_LENGTH = MAX_NAME_LENGTH,
        ._MAX_N_SYM = MAX_N_SYM,
        ._MAX_ORBS_PER_SITE = MAX_ORBS_PER_SITE,
        ._DIVERGE_EPS_MESH = DIVERGE_EPS_MESH,
        ._USE_GF_FLOATS = sizeof(gf_complex_t) != sizeof(complex128_t),
        ._BATCHED_EIGEN_NCHUNKS_AUTONUM = BATCHED_EIGEN_NCHUNKS_AUTONUM,
        ._BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB = BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB,
        ._USE_CUDA = HAS_USE_CUDA,
        ._BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS = BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS,
        ._CUBLAS_BATCHED_GEMM_CALL = str1024( DIVERGE_STRINGIFY(CUBLAS_BATCHED_GEMM_CALL) ),
        ._CUDA_CIRCUMVENT_CONSTEXPR = HAS_CUDA_CIRCUMVENT_CONSTEXPR,
        ._CUDA_DEFINE_ATOMIC_ADD = HAS_CUDA_DEFINE_ATOMIC_ADD,
        ._CUDA_VERSION_STR = str1024( CUDA_VERSION_STR ),
        ._DIVERGE_GRID_PARALLEL_BATCHED_GEMM = HAS_DIVERGE_GRID_PARALLEL_BATCHED_GEMM,
        ._DIVERGE_FLOW_STEP_EIGCHAN_PITER_ARPACK = DIVERGE_FLOW_STEP_EIGCHAN_PITER_ARPACK,
        ._DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER = DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER,
        ._DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL = DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL,
        ._DIVERGE_LOG_COLORLESS = HAS_DIVERGE_LOG_COLORLESS,
        ._DIVERGE_LOG_GRAYSCALE = HAS_DIVERGE_LOG_GRAYSCALE,
        ._DIVERGE_LOG_NAMELESS = HAS_DIVERGE_LOG_NAMELESS,
        ._DIVERGE_LOG_PREFIX = str1024( DIVERGE_LOG_PREFIX ),
        ._DIVERGE_MODEL_MAGIC_NUMBER = DIVERGE_MODEL_MAGIC_NUMBER,
        ._DIVERGE_MODEL_NBANDSTRUCTURE = DIVERGE_MODEL_NBANDSTRUCTURE,
        ._DIVERGE_NO_ABORT_ON_GPU_ERROR = HAS_DIVERGE_NO_ABORT_ON_GPU_ERROR,
        ._DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT = HAS_DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT,
        ._DIVERGE_OPENBLAS_LIMIT_THREADS = DIVERGE_OPENBLAS_LIMIT_THREADS,
        ._DIVERGE_USE_ZHEEVD = HAS_DIVERGE_USE_ZHEEVD,
        ._BATCHED_SVD_THREAD_CNT_MAX = BATCHED_SVD_THREAD_CNT_MAX,
        ._DIVERGE_GETLINE_MAX = DIVERGE_GETLINE_MAX,
        ._DIVERGE_SKIP_TESTS = HAS_DIVERGE_SKIP_TESTS,
        ._DIVERGE_SKIP_TESTS_LINALG = HAS_DIVERGE_SKIP_TESTS_LINALG,
        ._CT_NO_FORK = HAS_CT_NO_FORK,
        ._DIVERGE_QNAN_DISABLE_CLANG_OPTS = HAS_DIVERGE_QNAN_DISABLE_CLANG_OPTS,
        ._DIVERGE_TU_INPLACE_PERMUTE = HAS_DIVERGE_TU_INPLACE_PERMUTE,
        ._DIVERGE_TU_GF_COARSIFY_FFT = HAS_DIVERGE_TU_GF_COARSIFY_FFT,
        ._DIVERGE_UNIQUE_ORBFF_FIND_HASHTABLE = HAS_DIVERGE_UNIQUE_ORBFF_FIND_HASHTABLE,
        ._TU_POST_USE_BATCHED_SVD_U = HAS_TU_POST_USE_BATCHED_SVD_U,
        ._TU_PROJ_TIMING = HAS_TU_PROJ_TIMING,
        ._TU_VERTEX_RECO_GPU_ZGEMM = HAS_TU_VERTEX_RECO_GPU_ZGEMM,
        ._LDOS_KERNEL_GPU_DEV_MAXNOS = LDOS_KERNEL_GPU_DEV_MAXNOS,
        ._LDOS_KERNEL_GPU_DEV_NTHREADS = LDOS_KERNEL_GPU_DEV_NTHREADS,
        ._CLOG_MISSING = HAS_CLOG_MISSING,
        ._GRID_FF_SHELL_DISTANCE = GRID_FF_SHELL_DISTANCE,
        ._LEAVE_EIGEN_BROKEN = HAS_LEAVE_EIGEN_BROKEN,
        ._DIVERGE_POST_MAGIC_NUMBER_GRID = DIVERGE_POST_MAGIC_NUMBER_GRID,
        ._DIVERGE_POST_MAGIC_NUMBER_NPATCH = DIVERGE_POST_MAGIC_NUMBER_NPATCH,
        ._MAX_THREADS_PER_BLOCK = MAX_THREADS_PER_BLOCK,
        ._MKL_FORCE_AVX_KERNEL_ON_EPYC = HAS_MKL_FORCE_AVX_KERNEL_ON_EPYC,
        ._NDEBUG = HAS_NDEBUG,
        ._NDEBUG_SKIP_ERROR_LINES = HAS_NDEBUG_SKIP_ERROR_LINES,
        ._NPATCH_ALMOST_ZERO = NPATCH_ALMOST_ZERO,
        ._NPATCH_PERIODIC_NORM_NG = NPATCH_PERIODIC_NORM_NG,
        ._SKIP_EIGEN3_INCLUDE_PREFIX = HAS_SKIP_EIGEN3_INCLUDE_PREFIX,
        ._SKIP_FLOW_TIMING = HAS_SKIP_FLOW_TIMING,
        ._DIVERGE_POST_MAGIC_NUMBER_TU = DIVERGE_POST_MAGIC_NUMBER_TU,
        ._USE_ARPACK_EIGENSOLVER = HAS_USE_ARPACK_EIGENSOLVER,
        ._USE_COMPRESSED_OUTPUT = HAS_USE_COMPRESSED_OUTPUT,
        ._USE_DYNAMIC_RECONSTRUCT_SCHEDULE = HAS_USE_DYNAMIC_RECONSTRUCT_SCHEDULE,
        ._MERGESORT_DEFAULT_TASK_SIZE = MERGESORT_DEFAULT_TASK_SIZE,
        ._USE_ALTERNATIVE_TIME = HAS_USE_ALTERNATIVE_TIME,
        ._USE_EIGEN_BATCHED_GEMM = HAS_USE_EIGEN_BATCHED_GEMM,
        ._USE_EIGEN3_CPU_EIGENSOLVER = HAS_USE_EIGEN3_CPU_EIGENSOLVER,
        ._USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE = USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE,
        ._BATCHED_EIGEN_FORCE_CPU = BATCHED_EIGEN_FORCE_CPU,
        ._USE_MKL = HAS_USE_MKL,
        ._USE_MPI = HAS_USE_MPI,
        ._USE_NO_BLAS_VERTEX_LOOP = HAS_USE_NO_BLAS_VERTEX_LOOP,
        ._USE_NO_LAPACKE = HAS_USE_NO_LAPACKE,
        ._USE_SERIAL_FFTW = HAS_USE_SERIAL_FFTW,
        ._USE_SHARED_MEM = HAS_USE_SHARED_MEM,
        ._SHARED_MEM_MPI_IMPL = HAS_SHARED_MEM_MPI_IMPL,
        ._SHARED_MEM_POSIX_IMPL = HAS_SHARED_MEM_POSIX_IMPL,
        ._MEMFD_NOT_SUPPORTED = HAS_MEMFD_NOT_SUPPORTED,
        ._USE_MPI_FFTW = HAS_USE_MPI_FFTW,
        ._PACK_MPI_MULTIPLIER = PACK_MPI_MULTIPLIER,
        ._TU_GPU_MPI_ALLOC_PRINT = HAS_TU_GPU_MPI_ALLOC_PRINT,
        ._CUDA_MEM_ALIGN = CUDA_MEM_ALIGN,
        ._MPI_VERBOSE_ALLOCATION_PRINT = HAS_MPI_VERBOSE_ALLOCATION_PRINT,
        ._TRANSPOSE_MPI_FFT_BUFFERS = HAS_TRANSPOSE_MPI_FFT_BUFFERS,
        ._DIVERGE_OLD_W90_READER = HAS_DIVERGE_OLD_W90_READER,
        ._DIVERGE_TU_PROJ_TO_R_VTREE = HAS_DIVERGE_TU_PROJ_TO_R_VTREE,
        ._XVTREE_ALIGNMENT_PRINT = HAS_XVTREE_ALIGNMENT_PRINT,

        ._PADDING = {0},
    };
}

void diverge_constants_print( int header ) {
    diverge_constants_t c = diverge_constants();
    const char* def_str   = header ? "#define "         : "-D";
    const char* undef_str = header ? "#undef "          : "-U";
    const char* fmt_str   = header ? "#define %%s %s\n" : "-D%%s=%s ";
    char fmt[1024] = {0};
    #define PRINT_LI( C ) { \
        sprintf( fmt, fmt_str, "%li" ); \
        printf( fmt, #C, c._##C ); \
    }
    #define PRINT_DBL( C ) { \
        sprintf( fmt, fmt_str, "%.12e" ); \
        printf( fmt, #C, c._##C ); \
    }
    /* a switch given as -DFOO on the command line expands to 1, so emit
     * '#define FOO 1' to not trigger a redefinition warning downstream */
    #define PRINT_DEF( D ) { \
        if (c._##D) sprintf( fmt, header ? "%s%%s 1\n" : "%s%%s ", def_str ); \
        else        sprintf( fmt, header ? "%s%%s\n"   : "%s%%s ", undef_str ); \
        printf( fmt, #D ); \
    }
    /* NDEBUG must not leak into the generated header: diverge_config.h is
     * included from diverge.h, so a '#undef NDEBUG' would silently re-enable
     * asserts in a program that compiles against us with -DNDEBUG itself.
     * still reported by -printconstants, which nothing includes. */
    #define PRINT_DEF_NOCONFIG( D ) { \
        if (!header) { \
            sprintf( fmt, "%s%%s ", c._##D ? def_str : undef_str ); \
            printf( fmt, #D ); \
        } \
    }
    /* DEFINEs without default: value 0 means 'was never defined'. note that
     * the bodies below cannot delegate to the macros above, as that would
     * macro expand C (which is a DEFINE itself) on rescan. */
    #define PRINT_LI_OPT( C ) { \
        if (c._##C == 0) { \
            sprintf( fmt, header ? "%s%%s\n" : "%s%%s ", undef_str ); \
            printf( fmt, #C ); \
        } else { \
            sprintf( fmt, fmt_str, "%li" ); \
            printf( fmt, #C, c._##C ); \
        } \
    }
    /* string valued DEFINEs are printed including their quotes. an empty
     * string means 'was never defined' */
    #define PRINT_STR( C ) { \
        if (c._##C.str[0] == '\0') { \
            sprintf( fmt, header ? "%s%%s\n" : "%s%%s ", undef_str ); \
            printf( fmt, #C ); \
        } else { \
            sprintf( fmt, fmt_str, header ? "\"%s\"" : "\\\"%s\\\"" ); \
            printf( fmt, #C, c._##C.str ); \
        } \
    }
    /* DEFINEs that expand to a token (a function name, say) instead of to a
     * string literal. transported as a string, printed without quotes */
    #define PRINT_TOK( C ) { \
        sprintf( fmt, fmt_str, "%s" ); \
        printf( fmt, #C, c._##C.str ); \
    }

    /* the DEFINEs below are referenced from a header that diverge.h or
     * diverge_misc.h pulls in, so a consumer of the library must agree with
     * us on them. everything after them is internal to the library. */
    if (header) printf( "// ABI relevant\n" );

    PRINT_LI( MAX_N_ORBS );
    PRINT_LI( MAX_NAME_LENGTH );
    PRINT_LI( MAX_N_SYM );
    PRINT_LI( MAX_ORBS_PER_SITE );
    PRINT_DBL( DIVERGE_EPS_MESH );
    PRINT_DEF( USE_GF_FLOATS );
    PRINT_LI( BATCHED_EIGEN_NCHUNKS_AUTONUM );
    PRINT_LI( BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB );
    PRINT_LI( DIVERGE_FLOW_STEP_EIGCHAN_PITER_ARPACK );
    PRINT_LI( DIVERGE_MODEL_MAGIC_NUMBER );
    PRINT_LI( DIVERGE_MODEL_NBANDSTRUCTURE );
    PRINT_DEF( DIVERGE_QNAN_DISABLE_CLANG_OPTS );
    PRINT_DBL( GRID_FF_SHELL_DISTANCE );
    PRINT_DEF( USE_MKL );
    PRINT_DEF( USE_MPI );
    PRINT_DEF( USE_NO_LAPACKE );
    PRINT_DEF( SHARED_MEM_MPI_IMPL );
    PRINT_DEF( SHARED_MEM_POSIX_IMPL );

    if (header) printf( "\n// not ABI relevant\n"
                        "#ifdef DIVERGE_CONFIG_ALL\n\n" );

    PRINT_DEF( USE_CUDA );
    PRINT_LI( BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS );
    PRINT_DEF( CUDA_CIRCUMVENT_CONSTEXPR );
    PRINT_DEF( CUDA_DEFINE_ATOMIC_ADD );
    PRINT_DEF( DIVERGE_GRID_PARALLEL_BATCHED_GEMM );
    PRINT_LI( DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER );
    PRINT_DBL( DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL );
    PRINT_DEF( DIVERGE_LOG_COLORLESS );
    PRINT_DEF( DIVERGE_LOG_GRAYSCALE );
    PRINT_DEF( DIVERGE_LOG_NAMELESS );
    PRINT_DEF( DIVERGE_NO_ABORT_ON_GPU_ERROR );
    PRINT_DEF( DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT );
    PRINT_LI_OPT( DIVERGE_OPENBLAS_LIMIT_THREADS );
    PRINT_DEF( DIVERGE_USE_ZHEEVD );
    PRINT_LI( BATCHED_SVD_THREAD_CNT_MAX );
    PRINT_LI( DIVERGE_GETLINE_MAX );
    PRINT_DEF( DIVERGE_SKIP_TESTS );
    PRINT_DEF( DIVERGE_SKIP_TESTS_LINALG );
    PRINT_DEF( CT_NO_FORK );
    PRINT_DEF( DIVERGE_TU_INPLACE_PERMUTE );
    PRINT_DEF( DIVERGE_TU_GF_COARSIFY_FFT );
    PRINT_DEF( DIVERGE_UNIQUE_ORBFF_FIND_HASHTABLE );
    PRINT_DEF( TU_POST_USE_BATCHED_SVD_U );
    PRINT_DEF( TU_PROJ_TIMING );
    PRINT_DEF( TU_VERTEX_RECO_GPU_ZGEMM );
    PRINT_LI( LDOS_KERNEL_GPU_DEV_MAXNOS );
    PRINT_LI( LDOS_KERNEL_GPU_DEV_NTHREADS );
    PRINT_DEF( CLOG_MISSING );
    PRINT_DEF( LEAVE_EIGEN_BROKEN );
    PRINT_LI( DIVERGE_POST_MAGIC_NUMBER_GRID );
    PRINT_LI( DIVERGE_POST_MAGIC_NUMBER_NPATCH );
    PRINT_LI( MAX_THREADS_PER_BLOCK );
    PRINT_DEF( MKL_FORCE_AVX_KERNEL_ON_EPYC );
    PRINT_DEF_NOCONFIG( NDEBUG );
    PRINT_DEF( NDEBUG_SKIP_ERROR_LINES );
    PRINT_DBL( NPATCH_ALMOST_ZERO );
    PRINT_LI( NPATCH_PERIODIC_NORM_NG );
    PRINT_DEF( SKIP_EIGEN3_INCLUDE_PREFIX );
    PRINT_DEF( SKIP_FLOW_TIMING );
    PRINT_LI( DIVERGE_POST_MAGIC_NUMBER_TU );
    PRINT_DEF( USE_ARPACK_EIGENSOLVER );
    PRINT_DEF( USE_COMPRESSED_OUTPUT );
    PRINT_DEF( USE_DYNAMIC_RECONSTRUCT_SCHEDULE );
    PRINT_LI( MERGESORT_DEFAULT_TASK_SIZE );
    PRINT_DEF( USE_ALTERNATIVE_TIME );
    PRINT_DEF( USE_EIGEN_BATCHED_GEMM );
    PRINT_DEF( USE_EIGEN3_CPU_EIGENSOLVER );
    PRINT_LI_OPT( USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE );
    PRINT_LI( BATCHED_EIGEN_FORCE_CPU )
    PRINT_DEF( USE_NO_BLAS_VERTEX_LOOP );
    PRINT_DEF( USE_SERIAL_FFTW );
    PRINT_DEF( USE_SHARED_MEM );
    PRINT_DEF( MEMFD_NOT_SUPPORTED );
    PRINT_DEF( USE_MPI_FFTW );
    PRINT_LI( PACK_MPI_MULTIPLIER );
    PRINT_DEF( TU_GPU_MPI_ALLOC_PRINT );
    PRINT_LI( CUDA_MEM_ALIGN );
    PRINT_DEF( MPI_VERBOSE_ALLOCATION_PRINT );
    PRINT_DEF( TRANSPOSE_MPI_FFT_BUFFERS );
    PRINT_DEF( DIVERGE_OLD_W90_READER );
    PRINT_DEF( DIVERGE_TU_PROJ_TO_R_VTREE );
    PRINT_DEF( XVTREE_ALIGNMENT_PRINT );

    PRINT_TOK( CUBLAS_BATCHED_GEMM_CALL );
    PRINT_STR( CUDA_VERSION_STR );
    PRINT_STR( DIVERGE_LOG_PREFIX );

    if (header) printf( "\n#endif // DIVERGE_CONFIG_ALL\n" );

    else printf( "\n" );

    #undef PRINT_LI
    #undef PRINT_DBL
    #undef PRINT_DEF
    #undef PRINT_DEF_NOCONFIG
    #undef PRINT_LI_OPT
    #undef PRINT_STR
    #undef PRINT_TOK
}
