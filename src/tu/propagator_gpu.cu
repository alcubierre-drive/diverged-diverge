/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#define EIGEN_NO_CUDA

#include "propagator_details.h"
#include "cuda_init.h"
#include "../npatch/kernels_gpu_helpers.h"

static inline __device__ index_t mem_offset_tu() {
    return blockIdx.x*blockDim.x + threadIdx.x;
}

__global__ void GF_prod_ph(cuDoubleComplex* __restrict__ const buf,
                    const cuDoubleComplex* __restrict__ const GF1,
                    const cuDoubleComplex* __restrict__ const GFconj1,
                    const cuDoubleComplex* __restrict__ const GF2,
                    const cuDoubleComplex* __restrict__ const GFconj2,
                    const index_t* __restrict__ R1,
                    const index_t* __restrict__ R2,
                    const index_t* __restrict__ nk,
                    const index_t n_blocks,
                    const index_t nk_per_thread);

__global__ void GF_prod_pp(cuDoubleComplex* __restrict__ const buf,
                    const cuDoubleComplex* __restrict__ const GF1,
                    const cuDoubleComplex* __restrict__ const GFconj1,
                    const cuDoubleComplex* __restrict__ const GF2,
                    const cuDoubleComplex* __restrict__ const GFconj2,
                    const index_t* __restrict__ R1,
                    const index_t* __restrict__ R2,
                    const index_t* __restrict__ nk,
                    const index_t n_blocks,
                    const index_t nk_per_thread);

__global__ void to_loop_simple(cuDoubleComplex* __restrict__ const loop,
                    const cuDoubleComplex* __restrict__ const res,const index_t my_nk,
                    const index_t* const idx_map);

// see fold_to_coarse() in propagator_cpu.c for the "FFT trick" reduction
static void fold_to_coarse_gemv(cuda_prop_descr_t* __restrict__ cp, int dev, cudaStream_t stream,
                    const index_t* __restrict__ nkref, const index_t* __restrict__ nkref_coarse);


void ph_bubble_momenta_handler_fft(cuda_prop_descr_t* __restrict__ cp,
        complex128_t* const __restrict__ L, const complex128_t* const __restrict__ GF,
        const complex128_t* const __restrict__ GFconj, const index_t n_orb, const index_t n_bonds,
        const index_t* const __restrict__ bond_sizes, const index_t n_orbff, const index_t n_spin,
        const index_t nktot, const index_t my_nk,
        const index_t* __restrict__ nkref,
        const index_t* const __restrict__ c2c_map, const index_t* nkref_coarse,
        const index_t* __restrict__ mi_to_from,
        const index_t* __restrict__ mi_to_oto, const index_t* const __restrict__ mi_to_Rin) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    tu_gpu_info_t ii = tu_gpu_info();
    #pragma omp parallel for num_threads(ii.tu_num_of_devices)
    for (int dev=0; dev<ii.tu_num_of_devices; ++dev) {
        cudaSetDevice(dev);
        index_t* mi_to_R = cp->mi_to_R[dev];
        index_t* c_to_c_q = cp->c_to_c_q[dev];
        index_t* nkdev = cp->nkdev[dev];
        cuDoubleComplex* GF_fft = cp->GF_fft[dev];
        cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

        cudaStream_t up = ii.tu_gpu_stream[dev];
        cudaStream_t down = ii.tu_gpu_downstream[dev];

        const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;
        index_t off_GF1 = 0;
        index_t off_GF2 = (n_spin*n_orb > 1) ? nktot : 0;
        cufftDoubleComplex* loopfft_coarse = (cufftDoubleComplex*) cp->buf_GG_coarse[dev];
        CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft_coarse[dev], cp->fft_workspace_coarse[dev]));
        CUFFT_CHECK(cufftSetStream(cp->gpu_fft_coarse[dev], down));

        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        CUDA_CHECK(cudaMemcpyAsync(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault,down));
        CUDA_CHECK(cudaMemcpyAsync(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault,down));
        CUDA_CHECK(cudaMemcpyAsync(c_to_c_q,c2c_map,sizeof(index_t)*my_nk,cudaMemcpyDefault,down));
        if(cp->prop_preload_full_gf){
            CUDA_CHECK(cudaMemcpy(GF_fft,(cuDoubleComplex*)GF, _GF_size,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft_conj,(cuDoubleComplex*)GFconj, _GF_size,cudaMemcpyDefault));
        }
        cudaDeviceSynchronize();

        index_t off_old1 = -1;
        index_t off_old2 = -1;
        for(index_t block = 0; block < cp->_tasks_prop[dev]; ++block) {
            index_t offmq = block +cp->_off_prop[dev];
            const index_t s4 = offmq/(n_orbff*n_orbff*n_spin*n_spin*n_spin);
            offmq-=s4*(n_orbff*n_orbff*n_spin*n_spin*n_spin);

            const index_t s2 = offmq/(n_orbff*n_orbff*n_spin*n_spin);
            offmq-=s2*(n_orbff*n_orbff*n_spin*n_spin);

            const index_t mo3 = (offmq/(n_orbff*n_spin*n_spin));
            offmq-=mo3*(n_orbff*n_spin*n_spin);

            const index_t s3 = offmq/(n_orbff*n_spin);
            offmq-=s3*(n_orbff*n_spin);

            const index_t s1 = offmq/(n_orbff);
            const index_t mo1 = offmq - s1*(n_orbff);

            const index_t off_orb1 = nktot*(mi_to_from[mo1]+n_orb*(s1+n_spin
                                                *(mi_to_from[mo3]+n_orb*(s3))));
            const index_t off_orb2 = nktot*(mi_to_oto[mo3]+n_orb*(s2+n_spin
                                                *(mi_to_oto[mo1]+n_orb*(s4))));

            const index_t* R1 = mi_to_R+3*mo1;
            const index_t* R2 = mi_to_R+3*mo3;

            if(cp->prop_preload_full_gf) {
                off_GF1 = off_orb1;
                off_GF2 = off_orb2;
            }else{
                if(off_old1 != off_orb1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft,(cuDoubleComplex*)(GF+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj,(cuDoubleComplex*)(GFconj+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                }
                if(off_old2 != off_orb2 && n_spin*n_orb > 1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft+off_GF2,(cuDoubleComplex*)(GF+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj+off_GF2,(cuDoubleComplex*)(GFconj+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                }
                off_old1 = off_orb1;
                off_old2 = off_orb2;
            }
            CUDA_CHECK( cudaStreamSynchronize( down ) );
            GF_prod_ph<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                        dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,up>>>(
                                cp->buf_GG[dev],GF_fft+off_GF1,GF_fft+off_GF2,
                                GF_fft_conj+off_GF1,GF_fft_conj+off_GF2, R1, R2,
                                nkdev,cp->n_kblocks, cp->nk_per_thread);
            CUDA_CHECK( cudaStreamSynchronize( up ) );

            fold_to_coarse_gemv(cp, dev, down, nkref, nkref_coarse);
            CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft_coarse[dev], loopfft_coarse, loopfft_coarse, CUFFT_INVERSE));
            to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                            dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                        cp->buf_res[dev], cp->buf_GG_coarse[dev],my_nk,c_to_c_q);

            const index_t loop_idx = mo1+n_orbff*(s1+s4*n_spin
                                +POW2(n_spin)*(mo3+n_orbff*
                                (s3 + s2*n_spin)));
            cudaMemcpyAsync((cuDoubleComplex*)(L+(loop_idx)*my_nk),cp->buf_res[dev], my_nk*sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToHost,down);
        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}

void pp_bubble_momenta_handler_fft(cuda_prop_descr_t* __restrict__ cp,
        complex128_t* const __restrict__ L, const complex128_t* const __restrict__ GF,
        const complex128_t* const __restrict__ GFconj, const index_t n_orb, const index_t n_bonds,
        const index_t* const __restrict__ bond_sizes, const index_t n_orbff, const index_t n_spin,
        const index_t nktot, const index_t my_nk,
        const index_t* __restrict__ nkref,
        const index_t* const __restrict__ c2c_map, const index_t* nkref_coarse,
        const index_t* __restrict__ mi_to_from,
        const index_t* __restrict__ mi_to_oto, const index_t* const __restrict__ mi_to_Rin) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    tu_gpu_info_t ii = tu_gpu_info();
    #pragma omp parallel for num_threads(ii.tu_num_of_devices)
    for (int dev=0; dev<ii.tu_num_of_devices; ++dev) {
        cudaSetDevice(dev);
        index_t* mi_to_R = cp->mi_to_R[dev];
        index_t* c_to_c_q = cp->c_to_c_q[dev];
        index_t* nkdev = cp->nkdev[dev];
        cuDoubleComplex* GF_fft = cp->GF_fft[dev];
        cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

        cudaStream_t up = ii.tu_gpu_stream[dev];
        cudaStream_t down = ii.tu_gpu_downstream[dev];

        const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;
        index_t off_GF1 = 0;
        index_t off_GF2 = (n_spin*n_orb > 1) ? nktot : 0;
        cufftDoubleComplex* loopfft_coarse = (cufftDoubleComplex*) cp->buf_GG_coarse[dev];
        CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft_coarse[dev], cp->fft_workspace_coarse[dev]));
        CUFFT_CHECK(cufftSetStream(cp->gpu_fft_coarse[dev], down));

        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        CUDA_CHECK(cudaMemcpyAsync(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault,down));
        CUDA_CHECK(cudaMemcpyAsync(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault,down));
        CUDA_CHECK(cudaMemcpyAsync(c_to_c_q,c2c_map,sizeof(index_t)*my_nk,cudaMemcpyDefault,down));
        if(cp->prop_preload_full_gf){
            CUDA_CHECK(cudaMemcpy(GF_fft,(cuDoubleComplex*)GF, _GF_size,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft_conj,(cuDoubleComplex*)GFconj, _GF_size,cudaMemcpyDefault));
        }
        cudaDeviceSynchronize();

        index_t off_old1 = -1;
        index_t off_old2 = -1;
        for(index_t block = 0; block < cp->_tasks_prop[dev]; ++block) {
            index_t offmq = block +cp->_off_prop[dev];
            const index_t s4 = offmq/(n_orbff*n_orbff*n_spin*n_spin*n_spin);
            offmq-=s4*(n_orbff*n_orbff*n_spin*n_spin*n_spin);

            const index_t s3 = offmq/(n_orbff*n_orbff*n_spin*n_spin);
            offmq-=s3*(n_orbff*n_orbff*n_spin*n_spin);

            const index_t mo3 = (offmq/(n_orbff*n_spin*n_spin));
            offmq-=mo3*(n_orbff*n_spin*n_spin);

            const index_t s2 = offmq/(n_orbff*n_spin);
            offmq-=s2*(n_orbff*n_spin);

            const index_t s1 = offmq/(n_orbff);
            const index_t mo1 = offmq - s1*(n_orbff);

            const index_t off_orb1 = nktot*(mi_to_from[mo1]+n_orb*(s1+n_spin
                                                *(mi_to_from[mo3]+n_orb*(s3))));
            const index_t off_orb2 = nktot*(mi_to_oto[mo1]+n_orb*(s2+n_spin
                                                *(mi_to_oto[mo3]+n_orb*(s4))));
            const index_t* R1 = mi_to_R+3*mo1;
            const index_t* R2 = mi_to_R+3*mo3;

            if(cp->prop_preload_full_gf) {
                off_GF1 = off_orb1;
                off_GF2 = off_orb2;
            }else{
                if(off_old1 != off_orb1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft,(cuDoubleComplex*)(GF+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj,(cuDoubleComplex*)(GFconj+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                }
                if(off_old2 != off_orb2 && n_spin*n_orb > 1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft+off_GF2,(cuDoubleComplex*)(GF+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj+off_GF2,(cuDoubleComplex*)(GFconj+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                }
                off_old1 = off_orb1;
                off_old2 = off_orb2;
            }

            CUDA_CHECK( cudaStreamSynchronize( down ) );
            GF_prod_pp<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                        dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,up>>>(
                                cp->buf_GG[dev],GF_fft+off_GF1,GF_fft+off_GF2,
                                GF_fft_conj+off_GF1,GF_fft_conj+off_GF2, R1, R2,
                                nkdev,cp->n_kblocks, cp->nk_per_thread);
            CUDA_CHECK( cudaStreamSynchronize( up ) );

            fold_to_coarse_gemv(cp, dev, down, nkref, nkref_coarse);
            CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft_coarse[dev], loopfft_coarse, loopfft_coarse, CUFFT_FORWARD));
            to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                        dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                        cp->buf_res[dev], cp->buf_GG_coarse[dev],my_nk,c_to_c_q);
            const index_t loop_idx = mo1+n_orbff*(s1+s2*n_spin
                                +POW2(n_spin)*(mo3+n_orbff*
                                (s3 + s4*n_spin)));
            cudaMemcpyAsync((cuDoubleComplex*)(L+(loop_idx)*my_nk),cp->buf_res[dev], my_nk*sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToHost,down);
        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}

void ph_bubble_momenta_handler_fft_red( cuda_prop_descr_t* __restrict__ cp,
        complex128_t* const __restrict__ L, const complex128_t* const __restrict__ GF,
        const complex128_t* const __restrict__ GFconj, const index_t n_orb, const index_t n_orbff,
        const index_t n_spin, const index_t nktot, const index_t my_nk,
        const index_t* __restrict__ nkref,
        const index_t* const __restrict__ c2c_map, const index_t* __restrict__ nkref_coarse,
        const index_t* const __restrict__ mi_to_ofrom, const index_t* const __restrict__ mi_to_oto,
        const index_t* const __restrict__ mi_to_Rin, const index_t* const __restrict__ umi_l,
        const index_t* const __restrict__ umi_r, const index_t n_unique_pairs,
        cbytes_t* __restrict__ L_touched_map ) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    tu_gpu_info_t ii = tu_gpu_info();
    #pragma omp parallel for num_threads(ii.tu_num_of_devices)
    for (int dev=0; dev<ii.tu_num_of_devices; ++dev) {
        cudaSetDevice(dev);
        index_t* mi_to_R = cp->mi_to_R[dev];
        index_t* c_to_c_q = cp->c_to_c_q[dev];
        index_t* nkdev = cp->nkdev[dev];
        cuDoubleComplex* GF_fft = cp->GF_fft[dev];
        cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

        cudaStream_t up = ii.tu_gpu_stream[dev];
        cudaStream_t down = ii.tu_gpu_downstream[dev];
        const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;

        index_t off_GF1 = 0;
        index_t off_GF2 = (n_spin*n_orb > 1) ? nktot : 0;
        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft_coarse[dev], (void*)cp->fft_workspace_coarse[dev]));
        CUFFT_CHECK(cufftSetStream(cp->gpu_fft_coarse[dev], down));
        CUDA_CHECK(cudaMemcpy(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(c_to_c_q,c2c_map,sizeof(index_t)*my_nk,cudaMemcpyDefault));
        if(cp->prop_preload_full_gf) {
            CUDA_CHECK(cudaMemcpy(GF_fft,(cuDoubleComplex*)GF, _GF_size,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft_conj,(cuDoubleComplex*)GFconj, _GF_size,cudaMemcpyDefault));
        }
        cudaDeviceSynchronize();

        index_t off_old1 = -1;
        index_t off_old2 = -1;
        cufftDoubleComplex* loopfft_coarse = (cufftDoubleComplex*) cp->buf_GG_coarse[dev];
        for(index_t block = 0; block < cp->_tasks_prop[dev]; ++block) {
            index_t offmq = block +cp->_off_prop[dev];
            const index_t s4 = offmq/(n_unique_pairs*n_spin*n_spin*n_spin);
            offmq-=s4*(n_unique_pairs*n_spin*n_spin*n_spin);

            const index_t s2 = offmq/(n_unique_pairs*n_spin*n_spin);
            offmq-=s2*(n_unique_pairs*n_spin*n_spin);

            const index_t s3 = offmq/(n_unique_pairs*n_spin);
            offmq-=s3*(n_unique_pairs*n_spin);

            const index_t s1 = offmq/(n_unique_pairs);
            const index_t m = offmq - s1*(n_unique_pairs);
            const index_t mo1 = umi_l[m];
            const index_t mo3 = umi_r[m];

            const index_t off_orb1 = nktot*(mi_to_ofrom[mo1]+n_orb*(s1+n_spin
                                                *(mi_to_ofrom[mo3]+n_orb*(s3))));
            const index_t off_orb2 = nktot*(mi_to_oto[mo3]+n_orb*(s2+n_spin
                                                *(mi_to_oto[mo1]+n_orb*(s4))));

            const index_t* R1 = mi_to_R+3*mo1;
            const index_t* R2 = mi_to_R+3*mo3;
            if(cp->prop_preload_full_gf) {
                off_GF1 = off_orb1;
                off_GF2 = off_orb2;
            }else{
                if(off_old1 != off_orb1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft,(cuDoubleComplex*)(GF+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj,(cuDoubleComplex*)(GFconj+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                }
                if(off_old2 != off_orb2 && n_spin*n_orb > 1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft+off_GF2,(cuDoubleComplex*)(GF+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj+off_GF2,(cuDoubleComplex*)(GFconj+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                }
                off_old1 = off_orb1;
                off_old2 = off_orb2;
            }
            CUDA_CHECK( cudaStreamSynchronize( down ) );
            GF_prod_ph<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                         dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,up>>>(
                                cp->buf_GG[dev],GF_fft+off_GF1,GF_fft+off_GF2,
                                GF_fft_conj+off_GF1,GF_fft_conj+off_GF2, R1, R2,
                                nkdev,cp->n_kblocks, cp->nk_per_thread);
            CUDA_CHECK( cudaStreamSynchronize( up ) );
            fold_to_coarse_gemv(cp, dev, down, nkref, nkref_coarse);
            CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft_coarse[dev], loopfft_coarse, loopfft_coarse, CUFFT_INVERSE));

            to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                         dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                        cp->buf_res[dev], cp->buf_GG_coarse[dev],my_nk,c_to_c_q);
            const index_t loop_idx = mo1+n_orbff*(s1+s4*n_spin
                                    +POW2(n_spin)*(mo3+n_orbff*(s3 + s2*n_spin)));
            cudaMemcpyAsync((cuDoubleComplex*)(L+loop_idx*my_nk),cp->buf_res[dev], my_nk*sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToHost,down);
            if (L_touched_map) {
                for (index_t k=0; k<my_nk; ++k) {
                    #pragma omp atomic
                    BITARY_SET( L_touched_map, loop_idx*my_nk + k );
                }
            }

        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}

void pp_bubble_momenta_handler_fft_red( cuda_prop_descr_t* __restrict__ cp,
        complex128_t* const __restrict__ L, const complex128_t* const __restrict__ GF,
        const complex128_t* const __restrict__ GFconj, const index_t n_orb, const index_t n_orbff,
        const index_t n_spin, const index_t nktot, const index_t my_nk,
        const index_t* __restrict__ nkref,
        const index_t* const __restrict__ c2c_map, const index_t* __restrict__ nkref_coarse,
        const index_t* const __restrict__ mi_to_ofrom, const index_t* const __restrict__ mi_to_oto,
        const index_t* const __restrict__ mi_to_Rin, const index_t* const __restrict__ umi_l,
        const index_t* const __restrict__ umi_r, const index_t n_unique_pairs,
        cbytes_t* __restrict__ L_touched_map ) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    tu_gpu_info_t ii = tu_gpu_info();
    #pragma omp parallel for num_threads(ii.tu_num_of_devices)
    for (int dev=0; dev<ii.tu_num_of_devices; ++dev) {
        cudaSetDevice(dev);
        index_t* mi_to_R = cp->mi_to_R[dev];
        index_t* c_to_c_q = cp->c_to_c_q[dev];
        index_t* nkdev = cp->nkdev[dev];
        cuDoubleComplex* GF_fft = cp->GF_fft[dev];
        cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

        cudaStream_t up = ii.tu_gpu_stream[dev];
        cudaStream_t down = ii.tu_gpu_downstream[dev];
        const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;

        index_t off_GF1 = 0;
        index_t off_GF2 = (n_spin*n_orb > 1) ? nktot : 0;
        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft_coarse[dev], (void*)cp->fft_workspace_coarse[dev]));
        CUFFT_CHECK(cufftSetStream(cp->gpu_fft_coarse[dev], down));
        CUDA_CHECK(cudaMemcpy(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(c_to_c_q,c2c_map,sizeof(index_t)*my_nk,cudaMemcpyDefault));
        if(cp->prop_preload_full_gf) {
            CUDA_CHECK(cudaMemcpy(GF_fft,(cuDoubleComplex*)GF, _GF_size,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft_conj,(cuDoubleComplex*)GFconj, _GF_size,cudaMemcpyDefault));
        }
        cudaDeviceSynchronize();

        index_t off_old1 = -1;
        index_t off_old2 = -1;
        cufftDoubleComplex* loopfft_coarse = (cufftDoubleComplex*) cp->buf_GG_coarse[dev];
        for(index_t block = 0; block < cp->_tasks_prop[dev]; ++block) {
            index_t offmq = block+cp->_off_prop[dev];
            const index_t s4 = offmq/(n_unique_pairs*n_spin*n_spin*n_spin);
            offmq-=s4*(n_unique_pairs*n_spin*n_spin*n_spin);

            const index_t s3 = offmq/(n_unique_pairs*n_spin*n_spin);
            offmq-=s3*(n_unique_pairs*n_spin*n_spin);

            const index_t s2 = offmq/(n_unique_pairs*n_spin);
            offmq-=s2*(n_unique_pairs*n_spin);

            const index_t s1 = offmq/(n_unique_pairs);
            const index_t m = offmq - s1*(n_unique_pairs);
            const index_t mo1 = umi_l[m];
            const index_t mo3 = umi_r[m];

            const index_t off_orb1 = nktot*(mi_to_ofrom[mo1]+n_orb*(s1+n_spin
                                                *(mi_to_ofrom[mo3]+n_orb*(s3))));
            const index_t off_orb2 = nktot*(mi_to_oto[mo1]+n_orb*(s2+n_spin
                                                *(mi_to_oto[mo3]+n_orb*(s4))));

            const index_t* R1 = mi_to_R+3*mo1;
            const index_t* R2 = mi_to_R+3*mo3;
            if(cp->prop_preload_full_gf) {
                off_GF1 = off_orb1;
                off_GF2 = off_orb2;
            }else{
                if(off_old1 != off_orb1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft,(cuDoubleComplex*)(GF+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj,(cuDoubleComplex*)(GFconj+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                }
                if(off_old2 != off_orb2 && n_spin*n_orb > 1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft+off_GF2,(cuDoubleComplex*)(GF+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj+off_GF2,(cuDoubleComplex*)(GFconj+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                }
                off_old1 = off_orb1;
                off_old2 = off_orb2;
            }

            CUDA_CHECK( cudaStreamSynchronize( down ) );
            GF_prod_pp<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                         dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,up>>>(
                                cp->buf_GG[dev],GF_fft+off_GF1,GF_fft+off_GF2,
                                GF_fft_conj+off_GF1,GF_fft_conj+off_GF2, R1, R2,
                                nkdev,cp->n_kblocks, cp->nk_per_thread);
            CUDA_CHECK( cudaStreamSynchronize( up ) );
            fold_to_coarse_gemv(cp, dev, down, nkref, nkref_coarse);
            CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft_coarse[dev], loopfft_coarse, loopfft_coarse, CUFFT_FORWARD));

            to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                         dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                        cp->buf_res[dev], cp->buf_GG_coarse[dev],my_nk,c_to_c_q);
            const index_t loop_idx = mo1+n_orbff*(s1+s2*n_spin
                                    +POW2(n_spin)*(mo3+n_orbff*(s3 + s4*n_spin)));
            cudaMemcpyAsync((cuDoubleComplex*)(L+loop_idx*my_nk),cp->buf_res[dev], my_nk*sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToHost,down);
            if (L_touched_map) {
                for (index_t k=0; k<my_nk; ++k) {
                    #pragma omp atomic
                    BITARY_SET( L_touched_map, loop_idx*my_nk + k );
                }
            }
        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}


__global__ void GF_prod_pp(cuDoubleComplex* __restrict__ const buf,
                    const cuDoubleComplex* __restrict__ const GF_off_1,
                    const cuDoubleComplex* __restrict__ const GF_off_2,
                    const cuDoubleComplex* __restrict__ const GFconj_off_1,
                    const cuDoubleComplex* __restrict__ const GFconj_off_2,
                    const index_t* __restrict__ const R1,
                    const index_t* __restrict__ const R2,
                    const index_t* __restrict__ const nk,
                    const index_t n_blocks,
                    const index_t nk_per_thread)
{
    index_t offmq = mem_offset_tu();
    if(offmq>=n_blocks) return;

    const index_t kbegin = offmq*nk_per_thread;
    index_t kx = kbegin /(nk[1]*nk[2]);
    index_t ky = (kbegin-kx*nk[1]*nk[2])/nk[2];
    index_t kz = (kbegin-kx*nk[1]*nk[2]-ky*nk[2]);
    kx = (kx-R1[0]+R2[0]+nk[0]);
    ky = (ky-R1[1]+R2[1]+nk[1]);
    kz = (kz-R1[2]+R2[2]+nk[2]);
    while(kx >= nk[0]) kx -= nk[0];
    while(ky >= nk[1]) ky -= nk[1];
    while(kz >= nk[2]) kz -= nk[2];
    for(index_t k = kbegin; k < kbegin + nk_per_thread;++k) {
        buf[k] = cuCmul(GFconj_off_1[kz + nk[2]*(ky+nk[1]*kx)],GF_off_2[k]);
        buf[k] = cuCadd(buf[k],cuCmul(GF_off_1[kz + nk[2]*(ky+nk[1]*kx)],GFconj_off_2[k]));
        ++kz;
        if (kz >= nk[2]) { kz -= nk[2],++ky;
            if(ky >= nk[1]) { ky-=nk[1], ++kx;
                if(kx >= nk[0]) {kx-=nk[0];}}}
    }
}


__global__ void GF_prod_ph(cuDoubleComplex* __restrict__ const buf,
                    const cuDoubleComplex* __restrict__ const GF_off_1,
                    const cuDoubleComplex* __restrict__ const GF_off_2,
                    const cuDoubleComplex* __restrict__ const GFconj_off_1,
                    const cuDoubleComplex* __restrict__ const GFconj_off_2,
                    const index_t* __restrict__ const R1,
                    const index_t* __restrict__ const R2,
                    const index_t* __restrict__ const nk,
                    const index_t n_blocks,
                    const index_t nk_per_thread)
{
    index_t offmq = mem_offset_tu();
    if(offmq>=n_blocks) return;
    const index_t kbegin = offmq*nk_per_thread;

    index_t kx = kbegin /(nk[1]*nk[2]);
    index_t ky = (kbegin-kx*nk[1]*nk[2])/nk[2];
    index_t kz = (kbegin-(kx*nk[1]+ky)*nk[2]);
    kx = (-kx-R1[0]+R2[0]+3*nk[0]);
    ky = (-ky-R1[1]+R2[1]+3*nk[1]);
    kz = (-kz-R1[2]+R2[2]+3*nk[2]);
    while(kx >= nk[0]) kx -= nk[0];
    while(ky >= nk[1]) ky -= nk[1];
    while(kz >= nk[2]) kz -= nk[2];
    for(index_t k = kbegin; k < kbegin+nk_per_thread; ++k) {
        buf[k] = cuCmul(GF_off_1[kz + nk[2]*(ky+nk[1]*kx)],GF_off_2[k]);
        buf[k] = cuCadd(buf[k], cuCmul(GFconj_off_1[kz + nk[2]*(ky+nk[1]*kx)],GFconj_off_2[k]));
        ++kz;
        if (kz >= nk[2]) { kz -= nk[2],++ky;
            if(ky >= nk[1]) { ky-=nk[1], ++kx;
                if(kx >= nk[0]) {kx-=nk[0]; }}}
    }
}

__global__ void to_loop_simple(cuDoubleComplex* __restrict__ const loop,
                        const cuDoubleComplex* __restrict__ const res, const index_t my_nk,
                        const index_t* __restrict__ const idx_map)
{
    index_t q = mem_offset_tu();
    if(q>=my_nk) return;
    loop[q] = res[idx_map[q]];
}

// FFT trick: put \sum_{a=0}^{M} G[i+N*a] into G_c[i] and do FFT of G_c (see
// fold_to_coarse() in propagator_cpu.c). We can implement the sum over all
// k dimensions as consecutive strided batched zgemv operations with
// ones-vectors.
static void fold_to_coarse_gemv(cuda_prop_descr_t* __restrict__ cp, int dev, cudaStream_t stream,
                    const index_t* __restrict__ nkref, const index_t* __restrict__ nkref_coarse)
{
    const index_t Nx = nkref[0], Ny = nkref[1], Nz = nkref[2];
    const index_t Ncx = nkref_coarse[0], Ncy = nkref_coarse[1], Ncz = nkref_coarse[2];
    const index_t M0 = Nx/Ncx, M1 = Ny/Ncy, M2 = Nz/Ncz;

    cublasHandle_t handle = cp->cublas_handle[dev];
    CUBLAS_CHECK( cublasSetStream(handle, stream) );
    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    const cuDoubleComplex* ones = cp->fold_ones[dev];

    cuDoubleComplex* buf    = cp->buf_GG[dev];
    cuDoubleComplex* coarse = cp->buf_GG_coarse[dev];

    // x: buf (Nx,Ny,Nz) -> coarse (Ncx,Ny,Nz)
    const index_t K0 = Ncx*Ny*Nz;
    CUBLAS_CHECK( cublasZgemvStridedBatched(handle, CUBLAS_OP_N, (int)K0, (int)M0, &one,
                buf, (int)K0, (long long)K0, ones, 1, 0, &zero, coarse, 1, (long long)K0, 1) );

    // y: coarse (Ncx,Ny,Nz) -> buf (Ncx,Ncy,Nz), batched over Ncx
    const index_t K1 = Ncy*Nz;
    CUBLAS_CHECK( cublasZgemvStridedBatched(handle, CUBLAS_OP_N, (int)K1, (int)M1, &one,
                coarse, (int)K1, (long long)(Ny*Nz), ones, 1, 0, &zero, buf, 1, (long long)K1, (int)Ncx) );

    // z: buf (Ncx,Ncy,Nz) -> coarse (Ncx,Ncy,Ncz), batched over Ncx*Ncy;
    // this is the final result cufftExecZ2Z() reads
    const index_t K2 = Ncz;
    const index_t batch = Ncx*Ncy;
    CUBLAS_CHECK( cublasZgemvStridedBatched(handle, CUBLAS_OP_N, (int)K2, (int)M2, &one,
                buf, (int)K2, (long long)Nz, ones, 1, 0, &zero, coarse, 1, (long long)K2, (int)batch) );
}
