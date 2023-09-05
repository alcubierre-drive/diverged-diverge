#include "kernels_helpers.h"
#include "kernels_gpu_helpers.hpp"

#include "kernels_loop_gpu.h"
#include "flow_step.h"

#include <pthread.h>

__device__ static inline index_t _loop_add_k_gpu( index_t k1, index_t k2, index_t nk1d ) {
    return ((k1/nk1d + k2/nk1d)%nk1d)*nk1d + (k1%nk1d + k2%nk1d)%nk1d;
}
__device__ static inline index_t _loop_sub_k_gpu( index_t k1, index_t k2, index_t nk1d ) {
    return ((k1/nk1d - k2/nk1d + nk1d)%nk1d)*nk1d + (k1%nk1d - k2%nk1d + nk1d)%nk1d;
}

static int* helper_klp_gpus = NULL;
static int* helper_klp_weights = NULL;

static int helper_klp_ngpus = 0;
static int helper_klp_nblocks[3] = {1,1,1};
//static int helper_klp_nthreads[3] = {16,16,1};

static void** helper_klp_dispptr = NULL;

static int helper_klp_nstreams = 0;
static cudaStream_t* helper_klp_streams = NULL;

static void*** helper_klp_mem = NULL;

#define helper_klp_cnt 9
static size_t helper_klp_sz[helper_klp_cnt] = {0,0,0,0, 0,0,0,0, 0};

static bool helper_klp_setup( const int* gpus, int ngpus, const int nblocks[3], size_t nk, size_t np, size_t nb, size_t map_sz );
// returns whether GPU data needs copy

static void start_and_stop( index_t start, index_t stop, index_t scale, int ngpus,
        index_t** starts, index_t** stops );

// only valid up to 1024 threads per block
template<char Channel>
__global__ void npatch_loop_gen_gpu_krnl( index_t np, index_t nk_flat, index_t nb,
        index_t p, index_t p1p2k_offset, double global_norm,
        cu_cdoub_t * __restrict__ Lp,
        cu_cdoub_t * __restrict__ Lm,
        const cu_gf_complex_t * __restrict__ Gp,
        const cu_gf_complex_t * __restrict__ Gm,
        const index_t * __restrict__ p_idx,
        const index_t * __restrict__ kf_ary,
        index_t nkf,
        const double * __restrict__ f_wgh ) {
    index_t p1p2k = p1p2k_offset + blockIdx.x * gridDim.y + blockIdx.y;
    if (p1p2k >= nkf*np*np) return;
    unsigned short o1 = threadIdx.x / nb,
                   o3 = threadIdx.x % nb,
                   o2 = threadIdx.y / nb,
                   o4 = threadIdx.y % nb;
    index_t p1 = p1p2k / (np*nkf),
            p2 = (p1p2k % (np*nkf)) / nkf,
            kfi = p1p2k % nkf;
    index_t k = p_idx[p],
            k1 = p_idx[p1],
            k2 = p_idx[p2],
            kf = kf_ary[kfi];
    index_t L_idx = RAVEL_V( p, p1, p2, o1, o2, o3, o4 );
    index_t _kl = _loop_add_k_gpu(k, kf, nk_flat);

    double kf_norm = global_norm * f_wgh[kfi];

    if CUDA_CONSTEXPR(Channel == 'P') {

        index_t _krc = _loop_add_k_gpu(_loop_sub_k_gpu(k1, k2, nk_flat), _kl, nk_flat);
        cu_gf_complex_t Lp_ = cu_gf_complex_add( cu_gf_complex_mul( Gm[_kl*nb*nb+o1*nb+o3], Gm[_krc*nb*nb+o2*nb+o4] ),
                                                 cu_gf_complex_mul( Gp[_kl*nb*nb+o1*nb+o3], Gp[_krc*nb*nb+o2*nb+o4] ) );
        cu_complex_scale( &Lp_, kf_norm );
        cu_complex_atomic_add( Lp + L_idx, &Lp_ );

    } else {

        index_t _krp = _loop_sub_k_gpu(_loop_add_k_gpu(k1, k2, nk_flat), _kl, nk_flat);
        cu_gf_complex_t Lm_ = cu_gf_complex_add( cu_gf_complex_mul( Gm[_kl*nb*nb+o1*nb+o3], Gp[_krp*nb*nb+o2*nb+o4] ),
                                                 cu_gf_complex_mul( Gp[_kl*nb*nb+o1*nb+o3], Gm[_krp*nb*nb+o2*nb+o4] ) );
        cu_complex_scale( &Lm_, kf_norm );
        cu_complex_atomic_add( Lm + L_idx, &Lm_ );

    }
}

// alternative implementation when nb gets too large
template<char Channel>
__global__ void npatch_loop_gen_gpu_krnl3( index_t np, index_t nk_flat, index_t nb,
        index_t p, index_t p1p2k_offset, double global_norm,
        cu_cdoub_t * __restrict__ Lp,
        cu_cdoub_t * __restrict__ Lm,
        const cu_gf_complex_t * __restrict__ Gp,
        const cu_gf_complex_t * __restrict__ Gm,
        const index_t * __restrict__ p_idx,
        const index_t * __restrict__ kf_ary,
        index_t nkf,
        const double * __restrict__ f_wgh ) {
    index_t p1p2k = p1p2k_offset + blockIdx.x * gridDim.y + blockIdx.y;
    if (p1p2k >= nkf*np*np) return;
    unsigned short o1 = blockIdx.z,
                   o3 = threadIdx.x,
                   o2 = threadIdx.y,
                   o4 = threadIdx.z;
    index_t p1 = p1p2k / (np*nkf),
            p2 = (p1p2k % (np*nkf)) / nkf,
            kfi = p1p2k % nkf;
    index_t k = p_idx[p],
            k1 = p_idx[p1],
            k2 = p_idx[p2],
            kf = kf_ary[kfi];
    index_t L_idx = RAVEL_V( p, p1, p2, o1, o2, o3, o4 );
    index_t _kl = _loop_add_k_gpu(k, kf, nk_flat);

    double kf_norm = global_norm * f_wgh[kfi];

    if CUDA_CONSTEXPR(Channel == 'P') {

        index_t _krc = _loop_add_k_gpu(_loop_sub_k_gpu(k1, k2, nk_flat), _kl, nk_flat);
        cu_gf_complex_t Lp_ = cu_gf_complex_add( cu_gf_complex_mul( Gm[_kl*nb*nb+o1*nb+o3], Gm[_krc*nb*nb+o2*nb+o4] ),
                                                 cu_gf_complex_mul( Gp[_kl*nb*nb+o1*nb+o3], Gp[_krc*nb*nb+o2*nb+o4] ) );
        cu_complex_scale( &Lp_, kf_norm );
        cu_complex_atomic_add( Lp + L_idx, &Lp_ );

    } else {

        index_t _krp = _loop_sub_k_gpu(_loop_add_k_gpu(k1, k2, nk_flat), _kl, nk_flat);
        cu_gf_complex_t Lm_ = cu_gf_complex_add( cu_gf_complex_mul( Gm[_kl*nb*nb+o1*nb+o3], Gp[_krp*nb*nb+o2*nb+o4] ),
                                                 cu_gf_complex_mul( Gp[_kl*nb*nb+o1*nb+o3], Gm[_krp*nb*nb+o2*nb+o4] ) );
        cu_complex_scale( &Lm_, kf_norm );
        cu_complex_atomic_add( Lm + L_idx, &Lm_ );

    }
}

__global__ void loop_gen_gf_krnl( const __restrict__ cu_gf_complex_t* d_E_fine,
        const __restrict__ cu_gf_complex_t* d_U_fine, index_t nk, index_t nb,
        index_t offset, cu_gf_complex_t Lambda, __restrict__ cu_gf_complex_t* G ) {

    extern __shared__ cu_cdoub_t cache[];

    index_t kidx = offset + blockIdx.x * gridDim.y + blockIdx.y;

    if (kidx >= nk) return;

    index_t o1 = threadIdx.x,
            o2 = threadIdx.y,
            b = threadIdx.z;

    cu_cdoub_t E, U1, U2;
    E.x = Lambda.x - d_E_fine[kidx*nb+b].x;
    E.y = Lambda.y - d_E_fine[kidx*nb+b].y;

    #ifdef DIVERGE_MODEL_USE_O2B_C_ORDER
    U1.x =  d_U_fine[kidx*nb*nb+o1*nb+b].x;
    U1.y = -d_U_fine[kidx*nb*nb+o1*nb+b].y;
    U2.x =  d_U_fine[kidx*nb*nb+o2*nb+b].x;
    U2.y =  d_U_fine[kidx*nb*nb+o2*nb+b].y;
    #else
    U1.x =  d_U_fine[kidx*nb*nb+o1+nb*b].x;
    U1.y = -d_U_fine[kidx*nb*nb+o1+nb*b].y;
    U2.x =  d_U_fine[kidx*nb*nb+o2+nb*b].x;
    U2.y =  d_U_fine[kidx*nb*nb+o2+nb*b].y;
    #endif

    U1 = cuCmul( U1, U2 );
    cache[o1*nb*nb+o2*nb+b] = cuCdiv( U1, E );

    __syncthreads();

    if (b != 0) return;

    index_t c_Gidx = o1*nb*nb + o2*nb + 0;
    for (int bb=1; bb<nb; ++bb)
        cache[c_Gidx] = cuCadd( cache[c_Gidx], cache[o1*nb*nb+o2*nb+bb] );

    index_t Gidx = kidx * nb*nb + o1*nb + o2;
    G[Gidx] = { (float)cache[c_Gidx].x, (float)cache[c_Gidx].y };
}

void* npatch_loop_gen_gpu_impl( void* data ) {

    double tick = diverge_mpi_wtime();

    npatch_loop_kernel_info_t* loop_info = (npatch_loop_kernel_info_t*)data;
    npatch_loop_t* L = loop_info->L;
    complex128_t * Lp = L->Lp,
                 * Lm = L->Lm;
    const gf_complex_t * Gp = L->Gp,
                       * Gm = L->Gm;
    const index_t * p_idx = L->p_idx,
                  * p_cnt = L->p_cnt,
                  * p_dis = L->p_dis,
                  * p_map = L->p_map;
    const double * f_wgh = L->f_wgh;
    const void* host_mem[helper_klp_cnt] = { Lp, Lm, Gp, Gm, p_idx, p_cnt, p_dis, p_map, f_wgh };

    const index_t nb = L->nb,
                  np = L->np;
    const index_t *nk_flat = L->nk_flat;
    const double global_norm = 0.5/M_PI;
    const index_t L0 = loop_info->start,
                  L1 = loop_info->stop;

    cu_cdoub_t* ptr_Lambda = (cu_cdoub_t*)&(loop_info->Lambda);
    cu_cdoub_t dbl_Lambda = *ptr_Lambda;

    cu_gf_complex_t pLambda, mLambda;
    pLambda.x = dbl_Lambda.x;
    pLambda.y = dbl_Lambda.y;
    mLambda.x = -dbl_Lambda.x;
    mLambda.y = -dbl_Lambda.y;

    int g = loop_info->device;

    cu_cdoub_t *d_Lp = (cu_cdoub_t*) helper_klp_mem[g][0],
               *d_Lm = (cu_cdoub_t*) helper_klp_mem[g][1];
    cu_gf_complex_t *d_Gp = (cu_gf_complex_t*) helper_klp_mem[g][2],
                    *d_Gm = (cu_gf_complex_t*) helper_klp_mem[g][3];
    index_t *d_p_idx = (index_t*) helper_klp_mem[g][4],
            /* *d_p_cnt = (index_t*) helper_klp_mem[g][5], */
            /* *d_p_dis = (index_t*) helper_klp_mem[g][6], */
            *d_p_map = (index_t*) helper_klp_mem[g][7];
    double *d_f_wgh = (double*) helper_klp_mem[g][8];

    CUDA_CHECK( cudaSetDevice( helper_klp_gpus[g] ) );

    cudaStream_t krnl_stream = helper_klp_streams[g];

    double tick_gf = diverge_mpi_wtime();

    if (!loop_info->calc_gf_on_gpu) {
        for (int x=2; x<4; ++x) { // copy GF
            CUDA_CHECK( cudaMemcpy( helper_klp_mem[g][x], host_mem[x], helper_klp_sz[x], cudaMemcpyHostToHost ) );
            CUDA_CHECK( cudaMemPrefetchAsync( helper_klp_mem[g][x], helper_klp_sz[x], helper_klp_gpus[g], krnl_stream ) );
        }
    } else {
        index_t per_iter = helper_klp_nblocks[0] * helper_klp_nblocks[1];
        index_t nk = nk_flat[0]*nk_flat[1];
        index_t niter = nk / per_iter + ((nk % per_iter) > 0);
        cu_gf_complex_t* d_E_fine = (cu_gf_complex_t*)helper_klp_dispptr[g];
        cu_gf_complex_t* d_U_fine = ((cu_gf_complex_t*)helper_klp_dispptr[g]) + nk*nb;
        for (index_t iter=0; iter<niter; ++iter) {
            dim3 blk(helper_klp_nblocks[0], helper_klp_nblocks[1], 1);
            dim3 thr(nb,nb,nb);
            loop_gen_gf_krnl<<<blk,thr,nb*nb*nb*sizeof(cu_cdoub_t),krnl_stream>>>(
                    d_E_fine, d_U_fine, nk, nb, iter*per_iter, pLambda, d_Gp );
            loop_gen_gf_krnl<<<blk,thr,nb*nb*nb*sizeof(cu_cdoub_t),krnl_stream>>>(
                    d_E_fine, d_U_fine, nk, nb, iter*per_iter, mLambda, d_Gm );
        }
    }

    if (loop_info->needs_gpu_copy) {
        for (int x=4; x<helper_klp_cnt; ++x) { // do not copy loops, only copy stuff if needed
            CUDA_CHECK( cudaMemcpy( helper_klp_mem[g][x], host_mem[x], helper_klp_sz[x], cudaMemcpyHostToHost ) );
            CUDA_CHECK( cudaMemPrefetchAsync( helper_klp_mem[g][x], helper_klp_sz[x], helper_klp_gpus[g], krnl_stream ) );
        }
    }

    if (loop_info->needs_gpu_copy || !loop_info->calc_gf_on_gpu)
        CUDA_CHECK( cudaStreamSynchronize(krnl_stream) );

    // but instead set them zero
    for (int x=0; x<2; ++x)
        CUDA_CHECK( cudaMemset( helper_klp_mem[g][x], 0, helper_klp_sz[x] ) );

    CUDA_CHECK( cudaStreamSynchronize(krnl_stream) );
    double tock_gf = diverge_mpi_wtime();
    loop_info->wtime_gfs[g] = tock_gf - tick_gf;

    index_t pstart = L0 / (np*np*nb*nb*nb*nb),
            pstop = L1 / (np*np*nb*nb*nb*nb);
    for (index_t p=pstart; p<pstop; ++p) {
        index_t* d_kf_ary = d_p_map + p_dis[p];
        double* d_wf_ary = d_f_wgh + p_dis[p];
        index_t n_kf = p_cnt[p];
        index_t total_count = np*np*n_kf;
        index_t per_iter = helper_klp_nblocks[0] * helper_klp_nblocks[1];
        index_t n_iter = total_count / per_iter + (total_count % per_iter > 0);
        for (index_t iter=0; iter<n_iter; ++iter) {
            index_t p1p2k_offset = per_iter * iter;
            if (nb*nb*nb*nb > MAX_THREADS_PER_BLOCK) {
                dim3 blk(helper_klp_nblocks[0],helper_klp_nblocks[1], nb);
                dim3 thr(nb, nb, nb);
                if (nb*nb*nb > MAX_THREADS_PER_BLOCK) {
                    mpi_err_printf("cannot launch kernel with %lli=%lli**3 threads per block (max %i)\n",
                            nb*nb*nb, nb, MAX_THREADS_PER_BLOCK);
                    return NULL;
                }
                npatch_loop_gen_gpu_krnl3<'P'><<<blk,thr,0,krnl_stream>>>( np, nk_flat[1], nb, p, p1p2k_offset, global_norm,
                        d_Lp, d_Lm, d_Gp, d_Gm, d_p_idx, d_kf_ary, n_kf, d_wf_ary );
                npatch_loop_gen_gpu_krnl3<'M'><<<blk,thr,0,krnl_stream>>>( np, nk_flat[1], nb, p, p1p2k_offset, global_norm,
                        d_Lp, d_Lm, d_Gp, d_Gm, d_p_idx, d_kf_ary, n_kf, d_wf_ary );
            } else {
                dim3 blk(helper_klp_nblocks[0],helper_klp_nblocks[1], 1);
                dim3 thr(nb*nb, nb*nb, 1);
                npatch_loop_gen_gpu_krnl<'P'><<<blk,thr,0,krnl_stream>>>( np, nk_flat[1], nb, p, p1p2k_offset, global_norm,
                        d_Lp, d_Lm, d_Gp, d_Gm, d_p_idx, d_kf_ary, n_kf, d_wf_ary );
                npatch_loop_gen_gpu_krnl<'M'><<<blk,thr,0,krnl_stream>>>( np, nk_flat[1], nb, p, p1p2k_offset, global_norm,
                        d_Lp, d_Lm, d_Gp, d_Gm, d_p_idx, d_kf_ary, n_kf, d_wf_ary );
            }
        }
    }

    CUDA_CHECK( cudaStreamSynchronize( krnl_stream ) );

    CUDA_CHECK( cudaMemcpy( Lp + L0, d_Lp + L0, sizeof(complex128_t)*(L1-L0), cudaMemcpyHostToHost ) );
    CUDA_CHECK( cudaMemcpy( Lm + L0, d_Lm + L0, sizeof(complex128_t)*(L1-L0), cudaMemcpyHostToHost ) );

    double tock = diverge_mpi_wtime();
    loop_info->wtime = tock - tick;

    return NULL;
}

void* npatch_loop_gen_gpu( void* data ) {
    npatch_loop_kernel_info_t* loop_info = (npatch_loop_kernel_info_t*)data;
    const index_t nb = loop_info->L->nb,
                  np = loop_info->L->np;
    const index_t *nk_flat = loop_info->L->nk_flat;
    const index_t L0 = loop_info->start,
                  L1 = loop_info->stop;
    const index_t * p_cnt = loop_info->L->p_cnt,
                  * p_dis = loop_info->L->p_dis;

    double tick = diverge_mpi_wtime();

    int* p_ngpus; index_t s_ngpus;
    int* p_gpus; index_t s_gpus;
    int* p_nblocks; index_t s_nblocks;
    npatch_setup_query( "ngpus", (void**)&p_ngpus, &s_ngpus );
    npatch_setup_query( "gpus", (void**)&p_gpus, &s_gpus );
    npatch_setup_query( "nblocks", (void**)&p_nblocks, &s_nblocks );

    index_t map_sz = p_cnt[np-1] + p_dis[np-1];
    bool needs_gpu_copy = helper_klp_setup( p_gpus, *p_ngpus, p_nblocks, nk_flat[0]*nk_flat[1], np, nb, map_sz );

    // setup stuff on GPUs
    if (loop_info->calc_gf_on_gpu && (helper_klp_dispptr == NULL)) {
        index_t nknb = nk_flat[0]*nk_flat[1]*nb;

        // copy to reduced dtype on host
        gf_complex_t* h_E_fine = (gf_complex_t*)malloc(sizeof(cu_gf_complex_t)*nknb);
        gf_complex_t* h_U_fine = (gf_complex_t*)malloc(sizeof(cu_gf_complex_t)*nknb*nb);

        bool complex_E = loop_info->L->E_real == NULL;
        if (complex_E) {
            const complex128_t* _E_fine = loop_info->L->E;
            const complex128_t* _U_fine = loop_info->L->U;
            #pragma omp parallel for schedule(static)
            for (index_t kb=0; kb<nknb; ++kb) {
                h_E_fine[kb] = _E_fine[kb];
                for (index_t b=0; b<nb; ++b)
                    h_U_fine[kb*nb+b] = _U_fine[kb*nb+b];
            }
        } else {
            const double* _E_fine = loop_info->L->E_real;
            const complex128_t* _U_fine = loop_info->L->U;
            #pragma omp parallel for schedule(static)
            for (index_t kb=0; kb<nknb; ++kb) {
                h_E_fine[kb] = _E_fine[kb];
                for (index_t b=0; b<nb; ++b)
                    h_U_fine[kb*nb+b] = _U_fine[kb*nb+b];
            }
        }

        helper_klp_dispptr = (void**)malloc(sizeof(void*)*helper_klp_ngpus);
        for (int g=0; g<helper_klp_ngpus; ++g) {
            CUDA_CHECK( cudaSetDevice(helper_klp_gpus[g]) );
            CUDA_CHECK( cudaMallocManaged( helper_klp_dispptr+g, sizeof(cu_gf_complex_t)*nknb*(1+nb) ) );
            memcpy( helper_klp_dispptr[g], h_E_fine, sizeof(cu_gf_complex_t)*nknb );
            memcpy( ((cu_gf_complex_t*)helper_klp_dispptr[g]) + nknb, h_U_fine, sizeof(cu_gf_complex_t)*nknb*nb );
            CUDA_CHECK( cudaMemPrefetchAsync( helper_klp_dispptr[g], sizeof(cu_gf_complex_t)*nknb*(nb+1), helper_klp_gpus[g], helper_klp_streams[g] ) );
        }
        double additional_mem = sizeof(cu_gf_complex_t)*nknb*(1+nb);
        mpi_vrb_printf( "initializing GF on GPUs, using additional %.1fGiB\n", additional_mem/(1024.*1024.*1024.) );

        // and sync
        for (int g=0; g<helper_klp_ngpus; ++g)
            CUDA_CHECK( cudaStreamSynchronize( helper_klp_streams[g] ) );


        free(h_U_fine);
        free(h_E_fine);
    }

    free( p_ngpus );
    free( p_gpus );
    free( p_nblocks );

    index_t *starts, *stops;
    pthread_t* gpu_threads = (pthread_t*)calloc(helper_klp_ngpus, sizeof(pthread_t));
    npatch_loop_kernel_info_t* gpu_args = (npatch_loop_kernel_info_t*)calloc(helper_klp_ngpus, sizeof(npatch_loop_kernel_info_t));
    start_and_stop( L0, L1, np*np*nb*nb*nb*nb, helper_klp_ngpus, &starts, &stops );

    for (int g=0; g<helper_klp_ngpus; ++g) {
        memcpy( gpu_args+g, data, sizeof(npatch_loop_kernel_info_t) );
        gpu_args[g].device = g;
        gpu_args[g].start = starts[g];
        gpu_args[g].stop = stops[g];
        gpu_args[g].needs_gpu_copy = needs_gpu_copy;
        pthread_create( gpu_threads+g, NULL, npatch_loop_gen_gpu_impl, gpu_args+g );
    }

    for (int g=0; g<helper_klp_ngpus; ++g) {
        pthread_join( gpu_threads[g], NULL );
        loop_info->wtime_gpus[g%128] = gpu_args[g].wtime;
        loop_info->wtime_gfs[g%128] = gpu_args[g].wtime_gfs[g%128];
    }

    free( gpu_threads );
    free( gpu_args );
    free( starts );
    free( stops );

    double tock = diverge_mpi_wtime();
    loop_info->wtime += (tock - tick);

    return NULL;
}

void npatch_loop_gpu_set_weights( int* weights ) {
    if (!weights) {
        free(helper_klp_weights);
        helper_klp_weights = NULL;
        return;
    }
    helper_klp_weights = (int*)malloc(sizeof(int)*helper_klp_ngpus);
    for (int g=0; g<helper_klp_ngpus; ++g)
        helper_klp_weights[g] = weights[g];
}

static bool helper_klp_setup( const int* gpus, int ngpus, const int nblocks[3], size_t nk, size_t np, size_t nb, size_t map_sz ) {

    bool reset_gpus = false;
    bool reset_sizes = false;

    size_t L_sz = np*np*np * nb*nb*nb*nb;
    size_t G_sz = nk * nb*nb;

    if (helper_klp_gpus == NULL ||
        helper_klp_ngpus != ngpus ||
        helper_klp_nblocks[0] != nblocks[0] ||
        helper_klp_nblocks[1] != nblocks[1] ||
        helper_klp_nblocks[2] != nblocks[2] ||
        helper_klp_mem == NULL || helper_klp_streams == NULL)
            reset_gpus = true;

    size_t sizes_want[helper_klp_cnt] = {
        L_sz * sizeof(complex128_t),
        L_sz * sizeof(complex128_t),
        G_sz * sizeof(gf_complex_t),
        G_sz * sizeof(gf_complex_t),
        np * sizeof(index_t),
        np * sizeof(index_t),
        np * sizeof(index_t),
        map_sz * sizeof(index_t),
        map_sz * sizeof(double)
    };

    for (int x=0; x<helper_klp_cnt; ++x)
        reset_sizes += helper_klp_sz[x] < sizes_want[x];

    if (reset_gpus || reset_sizes) {
        if (helper_klp_mem) {
            for (int i=0; i<helper_klp_ngpus; ++i) {
                for (int x=0; x<helper_klp_cnt; ++x) {
                    void* mem = helper_klp_mem[i][x];
                    CUDA_CHECK( cudaFree( mem ) );
                }
                free( helper_klp_mem[i] );
            }
            free( helper_klp_mem );
        }
        free(helper_klp_gpus);
        if (helper_klp_weights) {
            free(helper_klp_weights);
            helper_klp_weights = NULL;
        }
    }

    if (reset_gpus) {
        helper_klp_ngpus = ngpus;
        helper_klp_gpus = (int*)malloc(sizeof(int)*helper_klp_ngpus);
        memcpy( helper_klp_gpus, gpus, sizeof(int)*helper_klp_ngpus );
        for (int i=0; i<3; ++i)
            helper_klp_nblocks[i] = nblocks[i];
    }

    if (reset_gpus || reset_sizes) {

        double sizes_sum = 0.0;
        for (int i=0; i<sizeof(sizes_want)/sizeof(size_t); ++i) sizes_sum += sizes_want[i];
        mpi_vrb_printf("initializing loops on GPU, using %.1fGiB\n", sizes_sum/(1024.0*1024.0*1024.0));

        for (int x=0; x<helper_klp_cnt; ++x)
            helper_klp_sz[x] = sizes_want[x];
        helper_klp_mem = (void***)malloc(sizeof(void**)*helper_klp_ngpus);
        for (int i=0; i<helper_klp_ngpus; ++i) {
            helper_klp_mem[i] = (void**)malloc(sizeof(void*)*helper_klp_cnt);
            for (int x=0; x<helper_klp_cnt; ++x) {
                CUDA_CHECK( cudaMallocManaged( helper_klp_mem[i] + x, helper_klp_sz[x] ) );
            }
        }
    }

    if (reset_gpus) {
        for (int s=0; s<helper_klp_nstreams; ++s)
            CUDA_CHECK( cudaStreamDestroy( helper_klp_streams[s] ) );
        if (helper_klp_streams) free(helper_klp_streams);
        helper_klp_nstreams = helper_klp_ngpus;
        helper_klp_streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*helper_klp_nstreams);
        for (int g=0; g<helper_klp_ngpus; ++g) {
            CUDA_CHECK( cudaSetDevice( helper_klp_gpus[g] ) );
            CUDA_CHECK( cudaStreamCreate( helper_klp_streams+g ) );
        }
    }

    return reset_gpus || reset_sizes;
}

void npatch_loop_gpu_teardown( void ) {
    if (helper_klp_mem) {
        for (int i=0; i<helper_klp_ngpus; ++i) {
            for (int x=0; x<helper_klp_cnt; ++x) {
                void* mem = helper_klp_mem[i][x];
                CUDA_CHECK( cudaFree( mem ) );
            }
            free( helper_klp_mem[i] );
        }
        free( helper_klp_mem );
        helper_klp_mem = NULL;
    }
    if (helper_klp_gpus) {
        free(helper_klp_gpus);
        helper_klp_gpus = NULL;
        helper_klp_ngpus = 0;
    }
    if (helper_klp_weights) {
        free(helper_klp_weights);
        helper_klp_weights = NULL;
    }
    if (helper_klp_streams) {
        for (int s=0; s<helper_klp_nstreams; ++s)
            CUDA_CHECK( cudaStreamDestroy( helper_klp_streams[s] ) );
        free(helper_klp_streams);
        helper_klp_streams = NULL;
        helper_klp_nstreams = 0;
    }
}

static void start_and_stop( index_t start, index_t stop, index_t scale, int ngpus,
        index_t** p_starts, index_t** p_stops ) {

    int* pp_gpu_counts = (int*)calloc(ngpus, sizeof(int));
    int* pp_gpu_displs = (int*)calloc(ngpus, sizeof(int));
    if (helper_klp_weights) {
        pp_gpu_counts[0] = helper_klp_weights[0];
        pp_gpu_displs[0] = 0;
        for (int i=1; i<ngpus; ++i) {
            pp_gpu_displs[i] = pp_gpu_counts[i-1] + pp_gpu_displs[i-1];
            pp_gpu_counts[i] = helper_klp_weights[i];
        }
    } else {
        for (int i=0; i<ngpus; ++i) {
            pp_gpu_counts[i] = 1;
            pp_gpu_displs[i] = i;
        }
    }
    index_t ngpus_virt = pp_gpu_counts[ngpus-1] + pp_gpu_displs[ngpus-1];

    index_t* counts_virt = (index_t*)calloc(ngpus_virt, sizeof(index_t));
    index_t* counts = (index_t*)calloc(ngpus, sizeof(index_t));

    start /= scale;
    stop /= scale;
    for (int s=start; s<stop; ++s)
        counts_virt[s%ngpus_virt]++;

    for (int g=0; g<ngpus; ++g)
        for (int gg=pp_gpu_displs[g]; gg<pp_gpu_displs[g]+pp_gpu_counts[g]; ++gg)
            counts[g] += counts_virt[gg];
    free(counts_virt);

    *p_starts = (index_t*)calloc(ngpus,sizeof(index_t));
    *p_stops = (index_t*)calloc(ngpus,sizeof(index_t));
    index_t* starts = *p_starts;
    index_t* stops = *p_stops;
    starts[0] = start;
    stops[0] = start + counts[0];
    for (int g=1; g<ngpus; ++g) {
        starts[g] = stops[g-1];
        stops[g] = stops[g-1] + counts[g];
    }
    for (int g=0; g<ngpus; ++g) {
        starts[g] *= scale;
        stops[g] *= scale;
        counts[g] *= scale;
        //printf("gpu%i: %lli-%lli (%lli)\n", g, starts[g], stops[g], counts[g]);
    }
    free(counts);
}
