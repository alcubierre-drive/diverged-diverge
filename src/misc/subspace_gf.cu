#include "subspace_gf.h"
#include "subspace_gf.cu.h"
#include <cuComplex.h>
#include <vector>
#include <thread>

using std::vector;
using std::thread;

typedef cuDoubleComplex ccomplex128_t;

// assumes all arrays to be GPU
__global__ void subspace_gfill_gpu( const double* E, const ccomplex128_t* U,
        ccomplex128_t Lambda, ccomplex128_t* out, index_t nb,
        const index_t* orbs, index_t n_orbs, index_t b_per_bb ) {

    extern __shared__ ccomplex128_t bufs[];

    const index_t k = blockIdx.x;

    const index_t bb = threadIdx.x;
    const index_t o1 = threadIdx.y;
    const index_t o2 = threadIdx.z;

    index_t bstart = bb * b_per_bb,
            bstop = (bb + 1) * b_per_bb;

    ccomplex128_t* buf = bufs + IDX3(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.y, blockDim.z);
    buf->x = buf->y = 0;

    for (index_t b=bstart; b<bstop; ++b) {
        if (b >= nb) continue;
        ccomplex128_t cLambda_X = Lambda;
        cLambda_X.x -= E[IDX2(k,b,nb)];
        *buf = cuCadd( *buf, cuCdiv( cuCmul( U[IDX3(k,b,orbs[o1],nb,nb)], cuConj(U[IDX3(k,b,orbs[o2],nb,nb)]) ), cLambda_X ) );
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i=1; i<blockDim.x; ++i) {
            bufs[IDX3(0,o1,o2,n_orbs,n_orbs)] = cuCadd( bufs[IDX3(0,o1,o2,n_orbs,n_orbs)],
                                                        bufs[IDX3(i,o1,o2,n_orbs,n_orbs)] );
        }
        out[IDX3(k,o1,o2,n_orbs,n_orbs)] = bufs[IDX3(0,o1,o2,n_orbs,n_orbs)];
    }
}

typedef struct {
    double* E;
    ccomplex128_t* U;
    ccomplex128_t* out;

    index_t kdispl;
    index_t kcount;

    index_t nk;
    index_t nb;

    index_t* orbs;
    index_t n_orbs;

    index_t b_per_bb;

    int dev;
} subspace_gfill_gpu_handle_t;

subspace_gfill_gpu_handle_t* subspace_gfill_gpu_init_dev( double* E, ccomplex128_t* U,
        index_t kdispl, index_t kcount, index_t nk, const index_t* orbs, index_t
        n_orbs, index_t nb, index_t b_per_bb, int dev ) {

    subspace_gfill_gpu_handle_t* h = (subspace_gfill_gpu_handle_t*) calloc(1, sizeof(subspace_gfill_gpu_handle_t));

    h->dev = dev;
    cudaSetDevice(h->dev);

    h->kdispl = kdispl;
    h->kcount = kcount;

    h->nk = nk;
    h->nb = nb;
    h->n_orbs = n_orbs;
    cudaMalloc( (void**)(&h->orbs), sizeof(index_t)*h->n_orbs );
    cudaMemcpy( h->orbs, orbs, sizeof(index_t)*h->n_orbs, cudaMemcpyHostToDevice );
    h->b_per_bb = b_per_bb;

    cudaMallocManaged( (void**)(&h->E), sizeof(double)*h->kcount*h->nb );
    cudaMallocManaged( (void**)(&h->U), sizeof(ccomplex128_t)*h->kcount*h->nb*h->nb );
    cudaMallocManaged( (void**)(&h->out), sizeof(ccomplex128_t)*h->kcount*h->nb*h->nb );

    memcpy( h->E, E+h->kdispl*nb, sizeof(double)*h->kcount*nb );
    memcpy( h->U, U+h->kdispl*nb*nb, sizeof(ccomplex128_t)*h->kcount*nb*nb );
    return h;
}

void subspace_gfill_gpu_kwrap( subspace_gfill_gpu_handle_t* h, ccomplex128_t Lambda ) {
    index_t biter = h->nb/h->b_per_bb;
    subspace_gfill_gpu<<<dim3(h->kcount,1,1), dim3(biter, h->n_orbs, h->n_orbs), sizeof(ccomplex128_t)*biter*h->n_orbs*h->n_orbs>>>(
            h->E, h->U, Lambda, h->out, h->nb, h->orbs, h->n_orbs, h->b_per_bb );
}

void subspace_gfill_gpu_free_dev( subspace_gfill_gpu_handle_t* h ) {
    cudaFree( h->E );
    cudaFree( h->U );
    cudaFree( h->out );
    cudaFree( h->orbs );
    free( h );
}

struct subspace_gfill_multi_t {
    subspace_gfill_gpu_handle_t* handles[32];
    int ndev;
};

subspace_gfill_multi_t* subspace_gfill_multi_init_cu( double* E, complex128_t* U, index_t nk, const index_t* orbs,
        index_t n_orbs, index_t nb, index_t b_per_bb ) {

    subspace_gfill_multi_t* m = (subspace_gfill_multi_t*)calloc(1, sizeof(subspace_gfill_multi_t));
    cudaGetDeviceCount( &m->ndev );

    vector<index_t> kcounts(m->ndev, 0);
    vector<index_t> kdispls(m->ndev, 0);
    for (index_t k=0; k<nk; ++k) kcounts[k%m->ndev]++;
    for (int d=1; d<m->ndev; ++d)
        kdispls[d] = kdispls[d-1] + kcounts[d-1];
    for (int d=0; d<m->ndev; ++d) {
        m->handles[d] = subspace_gfill_gpu_init_dev( E, (ccomplex128_t*)U, kdispls[d], kcounts[d], nk, orbs, n_orbs, nb, b_per_bb, d );
    }

    return m;
}

void subspace_gfill_multi_free_cu( subspace_gfill_multi_t* m ) {
    for (int d=0; d<m->ndev; ++d)
        subspace_gfill_gpu_free_dev( m->handles[d] );
    free( m );
}

void subspace_gfill_multi_exec_cu( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda_ ) {

    ccomplex128_t Lambda_p, Lambda_m;
    *(complex128_t*)&Lambda_p = Lambda_;
    *(complex128_t*)&Lambda_m = -Lambda_;

    vector<thread> threads;
    for (int d=0; d<m->ndev; ++d)
        threads.push_back( thread( [&]( int device ){
            subspace_gfill_gpu_handle_t* h = m->handles[device];
            cudaSetDevice( h->dev );
            subspace_gfill_gpu_kwrap( h, Lambda_p );
            cudaDeviceSynchronize();
            memcpy( dest + h->kdispl*h->n_orbs*h->n_orbs, h->out,
                    sizeof(complex128_t)*h->kcount*h->n_orbs*h->n_orbs );
            subspace_gfill_gpu_kwrap( h, Lambda_m );
            cudaDeviceSynchronize();
            memcpy( dest + h->nk*h->n_orbs*h->n_orbs + h->kdispl*h->n_orbs*h->n_orbs, h->out,
                    sizeof(complex128_t)*h->kcount*h->n_orbs*h->n_orbs );
        }, d ) );

    for (auto& t: threads)
        t.join();
}
