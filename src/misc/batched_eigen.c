/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_eigen.h"
#include "batched_eigen_internals.h"
#include "mpi_functions.h"

#include <pthread.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "../misc/cuda_error.h"
#include "eigen.h"
#endif

static int helper_batched_eigen_nchunks = -1;
static int helper_batched_eigen_shut_up = 0;
static int helper_batched_eigen_nthreads = -1;

#ifdef USE_EIGEN3_CPU_EIGENSOLVER
static int helper_batched_eigen_eigen3cpu = 1;
#else
static int helper_batched_eigen_eigen3cpu = 0;
#endif

void batched_eigen_r_set_eigen3_mode( int use_eigen3 ) {
    helper_batched_eigen_eigen3cpu = use_eigen3;
}

static void batched_eigen_wrap_r( int* devices, int ndevices, complex128_t* H,
        void* E_, bool real, index_t n, index_t N );

void batched_eigen_shut_up( int silence ) {
    helper_batched_eigen_shut_up = silence;
}

void batched_eigen_set_nthreads( int nthr ) {
    helper_batched_eigen_nthreads = nthr;
}

void batched_eigen( int* devices, int ndevices, complex128_t* H, complex128_t* E, index_t n, index_t N ) {
    batched_eigen_wrap_r( devices, ndevices, H, E, false, n, N );
}

void batched_eigen_r( int* devices, int ndevices, complex128_t* H, double* E, index_t n, index_t N ) {
#ifdef USE_CUDA
    if (ndevices == 0 && helper_batched_eigen_eigen3cpu) {
#else
    if (helper_batched_eigen_eigen3cpu) {
#endif
        mpi_vrb_printf( "dispatching to batched_zheevd eigensolver (Eigen3@CPU)\n" );
        batched_zheevd( H, H, E, N, n );
    } else {
        batched_eigen_wrap_r( devices, ndevices, H, E, true, n, N );
    }
}

void batched_eigen_set_nchunks( int nchunks ) {
    helper_batched_eigen_nchunks = nchunks;
}

int batched_eigen_get_nchunks( void ) {
    return helper_batched_eigen_nchunks;
}

static void batched_eigen_wrap_r( int* devices, int ndevices, complex128_t* H,
        void* E_, bool real, index_t n, index_t N ) {

    bool should_free_device_array = false;
    bool should_reset_nchunks = false;
    bool work_on_cpu = false;

    int nchunks_reset = -1;

    // reset to CPU operation
    #ifndef USE_CUDA
    devices = NULL;
    ndevices = 0;
    #endif

    if (devices == NULL) {

        autosetup:

        if (ndevices == 0) {
            work_on_cpu = true;
        } else if (ndevices == -2) {
            should_reset_nchunks = true;
            nchunks_reset = batched_eigen_get_nchunks();
            index_t nbytes_total = n * n * N * sizeof(complex128_t);
            index_t nbytes_per_chunk = BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB * 1000ul * 1000ul * 1000ul;
            int nchunks = MAX(nbytes_total / nbytes_per_chunk, 10);
            if (N/nchunks > BATCHED_EIGEN_NCHUNKS_AUTONUM)
                nchunks = N/BATCHED_EIGEN_NCHUNKS_AUTONUM;
            if(nchunks >= N) nchunks = N;
            batched_eigen_set_nchunks( nchunks );
        }

        #ifdef USE_CUDA
        CUDA_CHECK( cudaGetDeviceCount(&ndevices) );
        #endif

        if (!work_on_cpu) {
            should_free_device_array = true;
            devices = (int*)malloc(sizeof(int)*ndevices);
            for (int d=0; d<ndevices; ++d)
                devices[d] = d;

            if (!helper_batched_eigen_shut_up) {
                if (should_reset_nchunks) {
                    mpi_vrb_printf("batched eigen: %i@%i devices@chunks\n", ndevices, batched_eigen_get_nchunks());
                } else {
                    mpi_vrb_printf("batched eigen: %i devices\n", ndevices);
                }
            }
        }
    } else {
        if (ndevices < 1) {
            mpi_wrn_printf("ndevices set to '%i', doing automatic setup\n", ndevices);
            goto autosetup;
        }
    }

    if (work_on_cpu) {
        ndevices = helper_batched_eigen_nthreads > 0 ? helper_batched_eigen_nthreads : diverge_omp_num_threads();
        devices = (int*)calloc(ndevices, sizeof(int));
        should_free_device_array = true;
    }

    index_t* counts = (index_t*)calloc(ndevices, sizeof(index_t));
    for (index_t i=0; i<N; ++i)
        counts[i%ndevices]++;
    index_t* displs = (index_t*)malloc(ndevices*sizeof(index_t));
    displs[0] = 0;
    for (int i=1; i<ndevices; ++i)
        displs[i] = displs[i-1] + counts[i-1];

    batched_eigen_t* configs = (batched_eigen_t*)calloc(ndevices,sizeof(batched_eigen_t));
    for (int i=0; i<ndevices; ++i) {
        configs[i].device = devices[i];
        configs[i].nbatches = helper_batched_eigen_nchunks == -1 ? 1 : helper_batched_eigen_nchunks;
        configs[i].n = n;
        configs[i].N = counts[i];
        configs[i].H = H + n*n * displs[i];
        if (real)
            configs[i].Er = (double*)E_ + n * displs[i];
        else
            configs[i].E = (complex128_t*)E_ + n * displs[i];
    }

    pthread_t* threads = (pthread_t*)calloc(ndevices,sizeof(pthread_t));
    for (int d=0; d<ndevices; ++d) {
        if (work_on_cpu)
            pthread_create( threads+d, NULL, batched_eigen_cpu, configs+d );
        else
            pthread_create( threads+d, NULL, batched_eigen_gpu, configs+d );
    }
    for (int d=0; d<ndevices; ++d)
        pthread_join( threads[d], NULL );
    free(threads);

    free(configs);
    free(counts);
    free(displs);

    if (should_free_device_array) free(devices);
    if (should_reset_nchunks) batched_eigen_set_nchunks(nchunks_reset);

}
