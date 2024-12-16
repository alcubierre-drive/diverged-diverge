#include "dos.cu.h"
#include "dos_sort.h"
#include "cuda_error.h"
#include <cuComplex.h>
#include <omp.h>

__host__ __device__
static inline index_t find_bin_irreg( const double* vals, index_t n_vals, double test ) {
    if (test >= vals[n_vals-1]) return n_vals-1;
    if (test <= vals[0]) return -1;
    index_t low = 0,
            high = n_vals-1;
    while (low <= high) {
        index_t mid = low + (high - low)/2;
        if (vals[mid] <= test)
            low = mid + 1;
        else
            high = mid -1;
    }
    return low - 1;
}

__device__
static inline double cabs2( const cuDoubleComplex U ) {
    const double* pU = (const double*)&U;
    return POW2(pU[0]) + POW2(pU[1]);
}

__host__ __device__
static inline index_t regularize( index_t idx, index_t sz ) {
    return idx >= sz ? (sz - 1) : (idx <= 0) ? 0 : idx;
}

#ifndef LDOS_KERNEL_GPU_DEV_NTHREADS
#define LDOS_KERNEL_GPU_DEV_NTHREADS 512
#endif

#ifndef LDOS_KERNEL_GPU_DEV_MAXNOS
#define LDOS_KERNEL_GPU_DEV_MAXNOS 512
#endif

__global__
static void ldos_kernel_gpu_dev( const double* E, index_t nE, const cuDoubleComplex* U,
        index_t nos, const double* omega, double eta, double window, double* ldos ) {
    __shared__ index_t binlimits[2];
    __shared__ double ldos_local[LDOS_KERNEL_GPU_DEV_MAXNOS];

    const index_t omega_idx = blockIdx.x;
    if (threadIdx.x == 0) {
        binlimits[0] = regularize( find_bin_irreg( E, nE, omega[omega_idx]-window ), nE );
    }
    if (threadIdx.x == 1) {
        binlimits[1] = regularize( find_bin_irreg( E, nE, omega[omega_idx]+window ), nE );
        if (binlimits[1] == nE-1) binlimits[1] = nE;
    }
    if (threadIdx.x == 2) {
        for (index_t o=0; o<nos; ++o) ldos_local[o] = 0;
    }
    __syncthreads();

    index_t bincount_total = binlimits[1] - binlimits[0];
    index_t bincount = bincount_total / LDOS_KERNEL_GPU_DEV_NTHREADS +
        (bincount_total % LDOS_KERNEL_GPU_DEV_NTHREADS != 0);
    index_t bindispl = bincount * threadIdx.x;
    for (index_t b=0; b<bincount; ++b) {
        index_t bin = bindispl + b;
        if (bin >= bincount_total) break;
        bin += binlimits[0];

        cuDoubleComplex denom {0};
        denom.x = omega[omega_idx] - E[bin];
        denom.y = -eta;
        cuDoubleComplex nomin {0};
        nomin.x = 1;
        nomin = cuCdiv(nomin, denom);

        for (index_t o=0; o<nos; ++o)
            atomicAdd( &ldos_local[o], cabs2(U[IDX2(bin,o,nos)]) * nomin.y );
    }
    __syncthreads();

    index_t ocount = nos / LDOS_KERNEL_GPU_DEV_NTHREADS + (nos % LDOS_KERNEL_GPU_DEV_NTHREADS != 0);
    index_t odispl = threadIdx.x * ocount;
    for (index_t o=0; o<ocount; ++o)
        if (odispl+o < nos)
            ldos[IDX2(omega_idx,odispl+o,nos)] = ldos_local[odispl+o];
}

__global__ static void setComplexOne( cuDoubleComplex* ary ) {
    cuDoubleComplex val; val.x = 1; val.y = 0;
    ary[IDX2(blockIdx.x,threadIdx.x,blockDim.x)] = val;
}

void ldos_kernel_gpu( const double* E_, index_t nE, const complex128_t* U_, index_t nos,
        const double* omega, index_t n_omega, double eta, double window, double* ldos ) {
    int ndev = -1;
    CUDA_CHECK( cudaGetDeviceCount( &ndev ) );
    index_t* omega_counts = diverge_distribute( n_omega, ndev );
    index_t* omega_displs = omega_counts + ndev;

    if (LDOS_KERNEL_GPU_DEV_MAXNOS < nos)
        mpi_err_printf( "number of orbitals (%li) not supported by GPU LDOS (maxnos=%li)\n",
                nos, LDOS_KERNEL_GPU_DEV_MAXNOS );

    // GPU sorting doesn't really work...
    index_t* E_sort = (index_t*)calloc( nE, sizeof*E_sort );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<nE; ++i) E_sort[i] = i;
    dos_sort_doubles_indirect( E_sort, nE, E_ );

    complex128_t* U = U_ ? (complex128_t*)calloc( nE*nos, sizeof*U ) : NULL;
    double* E = (double*)calloc( nE, sizeof*E );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<nE; ++i) {
        E[i] = E_[E_sort[i]];
        if (U_) for (index_t o=0; o<nos; ++o)
            U[IDX2(i,o,nos)] = U_[IDX2(E_sort[i],o,nos)];
    }
    free( E_sort );

    #pragma omp parallel for num_threads(ndev)
    for (int dev=0; dev<ndev; ++dev) {

        CUDA_CHECK( cudaSetDevice( dev ) );

        // find block of energies on this device
        index_t E_displ = regularize( find_bin_irreg( E, nE, omega[omega_displs[dev]]-window ), nE );
        index_t E_displ_count = regularize(
                        find_bin_irreg(E,nE,omega[omega_displs[dev]+omega_counts[dev]-1]+window), nE );
        if (E_displ_count == nE-1) E_displ_count = nE;
        index_t E_count = E_displ_count - E_displ;

        // allocate+copy energies on device
        double* dE;
        CUDA_CHECK( cudaMalloc( (void**)&dE, E_count * sizeof(*dE) ) );
        CUDA_CHECK( cudaMemcpy( dE, E + E_displ, E_count*sizeof(*dE), cudaMemcpyHostToDevice ) );

        // allocate+copy/set o2b on device
        cuDoubleComplex* dU;
        CUDA_CHECK( cudaMalloc( (void**)&dU, E_count * sizeof(*dU) * nos ) );
        if (U) {
            CUDA_CHECK( cudaMemcpy( dU, U + E_displ*nos, E_count*nos*sizeof(*dU), cudaMemcpyHostToDevice ) )
        } else {
            setComplexOne<<<dim3(E_count,1,1),dim3(nos,1,1)>>>( dU );
            CUDA_CHECK( cudaDeviceSynchronize() );
        }

        // allocate+copy omega on device
        double* domega;
        CUDA_CHECK( cudaMalloc( (void**)&domega, omega_counts[dev]*sizeof(*domega) ) );
        CUDA_CHECK( cudaMemcpy( domega, omega+omega_displs[dev], omega_counts[dev]*sizeof(*domega),
                    cudaMemcpyHostToDevice ) );

        // allocate ldos on device
        double* dldos;
        CUDA_CHECK( cudaMalloc( (void**)&dldos, omega_counts[dev]*nos*sizeof(*dldos) ) );

        // call the kernel
        ldos_kernel_gpu_dev<<<dim3(omega_counts[dev],1,1),dim3(LDOS_KERNEL_GPU_DEV_NTHREADS,1,1)>>>( dE,
                E_count, dU, nos, domega, eta, window, dldos );
        CUDA_CHECK( cudaDeviceSynchronize() );

        // copy back stuff
        CUDA_CHECK( cudaMemcpy( ldos + omega_displs[dev]*nos, dldos, omega_counts[dev]*nos*sizeof(*ldos),
                    cudaMemcpyDeviceToHost ) );

        CUDA_CHECK( cudaFree( dE ) );
        CUDA_CHECK( cudaFree( dU ) );
        CUDA_CHECK( cudaFree( domega ) );
        CUDA_CHECK( cudaFree( dldos ) );
    }

    free( omega_counts );
    free( U );
    free( E );
}
