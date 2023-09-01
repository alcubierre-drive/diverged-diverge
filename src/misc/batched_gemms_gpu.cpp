#include "batched_gemms.h"

#ifdef USE_CUDA

#include "../misc/cuda_error.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

static int cublas_ndevices = 1;
static int cublas_deviceid[256] = {0};
static bool cublas_has_setup = false;
static index_t cublas_setup_num = -1;

static bool cublas_w_buffer_setup = false;
static cublasHandle_t* cublas_handles_w_buffer;
static cublasHandle_t* cublas_handles;
static void** helper_cublas_host_memory[3];
static void** helper_cublas_dev_memory[3];

void batched_gemm_cublas_w_buffer_init() {
    if (cublas_w_buffer_setup)
        return;
    cublas_w_buffer_setup = true;
    int ndevices = 0;
    CUDA_CHECK( cudaGetDeviceCount( &ndevices ) );
    cublas_handles_w_buffer = (cublasHandle_t*)malloc(sizeof(cublasHandle_t)*ndevices);
    for (int d=0; d<ndevices; ++d) {
        CUBLAS_CHECK( cublasCreate( &(cublas_handles_w_buffer[d]) ) );
    }
}

void batched_gemm_cublas_w_buffer_clear() {
     if (!cublas_w_buffer_setup)
        return;
    cublas_w_buffer_setup = false;

    for (int d=0; d<cublas_ndevices; ++d) {
        CUDA_CHECK( cudaSetDevice( d ) );
        CUBLAS_CHECK( cublasDestroy( cublas_handles_w_buffer[d] ) );
    }
    free( cublas_handles_w_buffer );
}

void batched_gemm_cublas_init( index_t N ) {
    if (cublas_has_setup && cublas_setup_num == N)
        return;
    // free resources before re-alloc
    if (cublas_has_setup && cublas_setup_num != N)
        batched_gemm_cublas_clear();

    int ndevices = 0;
    CUDA_CHECK( cudaGetDeviceCount( &ndevices ) );

    if (ndevices < 1) {
        mpi_err_printf( "exiting GPU setup due to previous errors.\n" );
        cublas_has_setup = false;
        cublas_setup_num = -1;
        return;
    } else {
        mpi_log_printf("initializing %i GPUs for batched GEMMs with N=%li\n", ndevices, N );
        cublas_setup_num = N;
        cublas_has_setup = true;
    }

    cublas_ndevices = ndevices;
    for (int i=0; i<ndevices; ++i)
        cublas_deviceid[i] = i;

    cublas_handles = (cublasHandle_t*)malloc(sizeof(cublasHandle_t)*ndevices);
    for (int i=0; i<3; ++i) {
        helper_cublas_host_memory[i] = (void**)malloc(sizeof(void*)*ndevices);
        helper_cublas_dev_memory[i] = (void**)malloc(sizeof(void*)*ndevices);
    }
    for (int d=0; d<ndevices; ++d) {
        CUDA_CHECK( cudaSetDevice(d) );
        for (int i=0; i<3; ++i) {
            CUDA_CHECK( cudaHostAlloc( &(helper_cublas_host_memory[i][d]), N*N*sizeof(complex128_t), cudaHostAllocDefault ) );
            memset( helper_cublas_host_memory[i][d], 0, N*N*sizeof(complex128_t) );
            CUDA_CHECK( cudaMalloc( &(helper_cublas_dev_memory[i][d]), N*N*sizeof(complex128_t) ) );
        }
        CUBLAS_CHECK( cublasCreate( &(cublas_handles[d]) ) );
    }
}

void batched_gemm_cublas_clear( void ) {
    if (!cublas_has_setup)
        return;
    cublas_has_setup = false;
    cublas_setup_num = -1;

    for (int d=0; d<cublas_ndevices; ++d) {
        CUDA_CHECK( cudaSetDevice( d ) );
        for (int i=0; i<3; ++i) {
            CUDA_CHECK( cudaFreeHost( helper_cublas_host_memory[i][d] ) );
            CUDA_CHECK( cudaFree( helper_cublas_dev_memory[i][d] ) );
        }
        CUBLAS_CHECK( cublasDestroy( cublas_handles[d] ) );
    }

    free( cublas_handles );
    for (int i=0; i<3; ++i) {
        free( helper_cublas_host_memory[i] );
        free( helper_cublas_dev_memory[i] );
    }
}

void batched_gemm_overlapping( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {

    batched_gemm_cublas_init(N);

    index_t numdev = num/cublas_ndevices > 0 ? cublas_ndevices : 1;
    index_t* dev_starts = (index_t*)malloc(sizeof(index_t)*numdev);
    index_t* dev_stops = (index_t*)malloc(sizeof(index_t)*numdev);

    for (index_t d=0; d<numdev; ++d) {
        dev_starts[d] = num/numdev*d;
        dev_stops[d] = num/numdev*(d+1);
    }
    dev_stops[numdev-1] = num;

    auto batch_part_on_dev = [&]( index_t d ){
        CUDA_CHECK( cudaSetDevice( d ) );
        for (index_t i=dev_starts[d]; i<dev_stops[d]; ++i) {
            memcpy( helper_cublas_host_memory[0][d], Astart+N*N*i, N*N*sizeof(complex128_t) );
            memcpy( helper_cublas_host_memory[1][d], Bstart+N*N*i, N*N*sizeof(complex128_t) );
            memcpy( helper_cublas_host_memory[2][d], Cstart+N*N*i, N*N*sizeof(complex128_t) );

            for (int i=0; i<3; ++i) {
                CUDA_CHECK( cudaMemcpy( helper_cublas_dev_memory[i][d], helper_cublas_host_memory[i][d], N*N*sizeof(complex128_t), cudaMemcpyHostToDevice ) );
            }

            // swap A and B to trick cublas into col-major operation
            CUBLAS_CHECK( cublasZgemm( cublas_handles[d], CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                (cuDoubleComplex*)&alpha,
                (cuDoubleComplex*)helper_cublas_dev_memory[1][d], N,
                (cuDoubleComplex*)helper_cublas_dev_memory[0][d], N,
                (cuDoubleComplex*)&beta,
                (cuDoubleComplex*)helper_cublas_dev_memory[2][d], N ) );

            CUDA_CHECK( cudaMemcpy( helper_cublas_host_memory[2][d], helper_cublas_dev_memory[2][d], N*N*sizeof(complex128_t), cudaMemcpyDeviceToHost ) );
            memcpy( Cstart+N*N*i, helper_cublas_host_memory[2][d], N*N*sizeof(complex128_t) );
        }
    };

    #pragma omp parallel for num_threads(numdev)
    for (index_t d=0; d<numdev; ++d) {
        batch_part_on_dev(d);
    }
    free(dev_stops);
    free(dev_starts);
}

// pinnen memory implementation is by definition 'overlapping'
void batched_gemm( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    batched_gemm_overlapping( Astart, Bstart, Cstart, N, alpha, beta, num );
}


void batched_gemm_with_buffers( const complex128_t* Astart, const complex128_t* Bstart,
        complex128_t* Cstart, char* AGPU[32][3], index_t N, complex128_t alpha, complex128_t beta, index_t num ) {
    
    batched_gemm_cublas_w_buffer_init();

    index_t numdev = num/cublas_ndevices > 0 ? cublas_ndevices : 1;
    index_t* dev_starts = (index_t*)malloc(sizeof(index_t)*numdev);
    index_t* dev_stops = (index_t*)malloc(sizeof(index_t)*numdev);

    for (index_t d=0; d<numdev; ++d) {
        dev_starts[d] = num/numdev*d;
        dev_stops[d] = num/numdev*(d+1);
    }
    dev_stops[numdev-1] = num;

    auto batch_part_on_dev = [&]( index_t d ){
        CUDA_CHECK( cudaSetDevice( d ) );
        for (index_t i=dev_starts[d]; i<dev_stops[d]; ++i) {
            CUDA_CHECK( cudaMemcpy( AGPU[d][0], Astart+N*N*i, N*N*sizeof(cuDoubleComplex) , cudaMemcpyHostToDevice ) );
            CUDA_CHECK( cudaMemcpy( AGPU[d][1], Bstart+N*N*i, N*N*sizeof(cuDoubleComplex) , cudaMemcpyHostToDevice ) );

            // swap A and B to trick cublas into col-major operation
            CUBLAS_CHECK( cublasZgemm( cublas_handles_w_buffer[d], CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                (cuDoubleComplex*)&alpha,
                (cuDoubleComplex*)AGPU[d][1], N,
                (cuDoubleComplex*)AGPU[d][0], N,
                (cuDoubleComplex*)&beta,
                (cuDoubleComplex*)AGPU[d][2], N ) );
            CUDA_CHECK( cudaMemcpy( Cstart+N*N*i, AGPU[d][2], N*N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost ) );
        }
    };

    #pragma omp parallel for num_threads(numdev)
    for (index_t d=0; d<numdev; ++d) {
        batch_part_on_dev(d);
    }
    free(dev_stops);
    free(dev_starts);
}

#endif // USE_CUDA
