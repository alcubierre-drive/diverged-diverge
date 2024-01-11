/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "propagator.hpp"
#include "../misc/cuda_error.h"

#include <cuComplex.h>

#include <algorithm>
#include <numeric>
#include <thread>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include "cuda_init.hpp"

static inline __device__ index_t mem_offset() {
    return blockIdx.x*blockDim.x + threadIdx.x;
}

__global__ void GF_prod_ph(__restrict__ cuDoubleComplex* const buf,
                    __restrict__ const cuDoubleComplex* const GF1,
                    __restrict__ const cuDoubleComplex* const GFconj1,
                    __restrict__ const cuDoubleComplex* const GF2,
                    __restrict__ const cuDoubleComplex* const GFconj2,
                    __restrict__ const index_t* R1,
                    __restrict__ const index_t* R2,
                    __restrict__ const index_t* nk,
                    const index_t n_blocks,
                    const index_t nk_per_thread);

__global__ void GF_prod_pp(__restrict__ cuDoubleComplex* const buf,
                    __restrict__ const cuDoubleComplex* const GF1,
                    __restrict__ const cuDoubleComplex* const GFconj1,
                    __restrict__ const cuDoubleComplex* const GF2,
                    __restrict__ const cuDoubleComplex* const GFconj2,
                    __restrict__ const index_t* R1,
                    __restrict__ const index_t* R2,
                    __restrict__ const index_t* nk,
                    const index_t n_blocks,
                    const index_t nk_per_thread);

__global__ void to_loop_simple(__restrict__ cuDoubleComplex* const loop,
                    __restrict__ const cuDoubleComplex* const res,const index_t my_nk,
                    const index_t* const coarse_in_fine_map);


void ph_bubble_momenta_handler_fft(cuda_prop_descr_t* cp,  complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin) {
    cuda_setup();

    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    std::vector<std::thread> allthreads(tu_num_of_devices);
    for (index_t dev=0; dev<(index_t)tu_num_of_devices; ++dev) {
        allthreads[dev] = std::thread( [&](int dev){
            cudaSetDevice(dev);
            index_t* mi_to_R = cp->mi_to_R[dev];
            index_t* c_to_f_q = cp->c_to_f_q[dev];
            index_t* nkdev = cp->nkdev[dev];
            cuDoubleComplex* GF_fft = cp->GF_fft[dev];
            cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

            cudaStream_t up = tu_gpu_stream[dev];
            cudaStream_t down = tu_gpu_downstream[dev];

            const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;
            const index_t off_GF = (n_spin*n_orb > 1) ? nktot : 0;
            
            CUDA_CHECK(cudaMemcpyAsync(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault,down));
            CUDA_CHECK(cudaMemcpyAsync(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault,down));
            CUDA_CHECK(cudaMemcpyAsync(c_to_f_q,c2f_map,sizeof(index_t)*my_nk,cudaMemcpyDefault,down));
            cufftDoubleComplex* loopfft = (cufftDoubleComplex*) cp->buf_GG[dev];
            
            CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft[dev], cp->fft_workspace[dev]));
            CUFFT_CHECK(cufftSetStream(cp->gpu_fft[dev], down));
            index_t off_old1 = -1;
            index_t off_old2 = -1;
            for(index_t block = 0; block < cp->_tasks_prop[dev]; ++block) {
                index_t offmq = block + cp->_off_prop[dev];
                const index_t s2 = offmq/(n_orbff*n_orbff*n_spin*n_spin*n_spin);
                offmq-=s2*(n_orbff*n_orbff*n_spin*n_spin*n_spin);

                const index_t s3 = offmq/(n_orbff*n_orbff*n_spin*n_spin);
                offmq-=s3*(n_orbff*n_orbff*n_spin*n_spin);

                const index_t mo3 = (offmq/(n_orbff*n_spin*n_spin));
                offmq-=mo3*(n_orbff*n_spin*n_spin);

                const index_t s4 = offmq/(n_orbff*n_spin);
                offmq-=s4*(n_orbff*n_spin);

                const index_t s1 = offmq/(n_orbff);
                const index_t mo1 = offmq - s1*(n_orbff);

                const index_t off_orb1 = nktot*(mi_to_ofrom[mo1]+n_orb*(s1+n_spin
                                                    *(mi_to_ofrom[mo3]+n_orb*(s3))));
                const index_t off_orb2 = nktot*(mi_to_oto[mo3]+n_orb*(s2+n_spin
                                                    *(mi_to_oto[mo1]+n_orb*(s4))));

                const index_t* R1 = mi_to_R+3*mo1;
                const index_t* R2 = mi_to_R+3*mo3;

                if(off_old1 != off_orb1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft,(cuDoubleComplex*)(GF+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj,(cuDoubleComplex*)(GFconj+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                }
                if(off_old2 != off_orb2 && n_spin*n_orb > 1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft+off_GF,(cuDoubleComplex*)(GF+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj+off_GF,(cuDoubleComplex*)(GFconj+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                }

                off_old1 = off_orb1;
                off_old2 = off_orb2;
                
                GF_prod_ph<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                             dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,up>>>(
                                    cp->buf_GG[dev],GF_fft,GF_fft+off_GF,
                                    GF_fft_conj,GF_fft_conj+off_GF, R1, R2,
                                    nkdev,cp->n_kblocks, cp->nk_per_thread);
                CUDA_CHECK( cudaStreamSynchronize( up ) );
                                
                CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft[dev], loopfft, loopfft, CUFFT_INVERSE));

                CUDA_CHECK( cudaStreamSynchronize( down ) );
                to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                                 dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                            cp->buf_res[dev][0], cp->buf_GG[dev],my_nk,c_to_f_q);

                const index_t loop_idx = mo1+n_orbff*(s1+s4*n_spin
                                    +POW2(n_spin)*(mo3+n_orbff*
                                    (s3 + s2*n_spin)));
                cudaMemcpyAsync((cuDoubleComplex*)(L+(loop_idx)*my_nk),cp->buf_res[dev][0], my_nk*sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToHost,down);
            }
            cudaDeviceSynchronize();
        },dev);
    }
    for (std::thread& t: allthreads) t.join();
    cudaDeviceSynchronize();
}


void pp_bubble_momenta_handler_fft(cuda_prop_descr_t* cp,  complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin) {
    cuda_setup();
    
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    std::vector<std::thread> allthreads(tu_num_of_devices);
    for (index_t dev=0; dev<(index_t)tu_num_of_devices; ++dev) {
        allthreads[dev] = std::thread( [&](int dev){
            cudaSetDevice(dev);
            index_t* mi_to_R = cp->mi_to_R[dev];
            index_t* c_to_f_q = cp->c_to_f_q[dev];
            index_t* nkdev = cp->nkdev[dev];
            cuDoubleComplex* GF_fft = cp->GF_fft[dev];
            cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

            cudaStream_t up = tu_gpu_stream[dev];
            cudaStream_t down = tu_gpu_downstream[dev];

            const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
            CUFFT_CHECK(cufftSetStream(cp->gpu_fft[dev], down));
            CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft[dev], (void*)cp->fft_workspace[dev]));

            cudaDeviceSynchronize();
            CUDA_CHECK(cudaMemcpy(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(c_to_f_q,c2f_map,sizeof(index_t)*my_nk,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft,(cuDoubleComplex*)GF, _GF_size,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft_conj,(cuDoubleComplex*)GFconj, _GF_size,cudaMemcpyDefault));
            const index_t off_GF = (n_spin*n_orb > 1) ? nktot : 0;

            for(index_t block = 0; block < cp->_tasks_prop[dev]; ++block) {
                
                index_t offmq = block+cp->_off_prop[dev];

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

                const index_t off_orb1 = nktot*(mi_to_ofrom[mo1]+n_orb*(s1+n_spin
                                                    *(mi_to_ofrom[mo3]+n_orb*(s3))));
                const index_t off_orb2 = nktot*(mi_to_oto[mo1]+n_orb*(s2+n_spin
                                                    *(mi_to_oto[mo3]+n_orb*(s4))));
                const index_t* R1 = mi_to_R+3*mo1;
                const index_t* R2 = mi_to_R+3*mo3;


                GF_prod_pp<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                             dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                cp->buf_GG[dev],GF_fft+off_orb1,GF_fft+off_orb2,
                                GF_fft_conj+off_orb1,GF_fft_conj+off_orb2,R1,R2,
                                nkdev,cp->n_kblocks, cp->nk_per_thread);

                cufftDoubleComplex* loopfft = (cufftDoubleComplex*) cp->buf_GG[dev];
                CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft[dev], loopfft, loopfft, CUFFT_FORWARD));
                to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                             dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                            cp->buf_res[dev][0], cp->buf_GG[dev],my_nk,c_to_f_q);
                const index_t loop_idx = mo1+n_orbff*(s1+s2*n_spin
                                    +POW2(n_spin)*(mo3+n_orbff*
                                    (s3 + s4*n_spin)));
                cudaMemcpyAsync((cuDoubleComplex*)(L+(loop_idx)*my_nk),cp->buf_res[dev][0], my_nk*sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToHost,down);
            }
            cudaDeviceSynchronize();
        },dev);
    }
    for (std::thread& t: allthreads) t.join();
    cudaDeviceSynchronize();
}

void ph_bubble_momenta_handler_fft_red(cuda_prop_descr_t* cp,  complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin,const index_t* const umi_l,const index_t* const umi_r,
    const index_t n_unique_pairs) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));

    std::vector<std::thread> allthreads(tu_num_of_devices);
    for (index_t dev=0; dev<(index_t)tu_num_of_devices; ++dev) {
        allthreads[dev] = std::thread( [&](int dev){
            cudaSetDevice(dev);
            index_t* mi_to_R = cp->mi_to_R[dev];
            index_t* c_to_f_q = cp->c_to_f_q[dev];
            index_t* nkdev = cp->nkdev[dev];
            cuDoubleComplex* GF_fft = cp->GF_fft[dev];
            cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

            cudaStream_t up = tu_gpu_stream[dev];
            cudaStream_t down = tu_gpu_downstream[dev];
            cudaStream_t swap;
            const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;

            const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
            CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft[dev], (void*)cp->fft_workspace[dev]));
            CUFFT_CHECK(cufftSetStream(cp->gpu_fft[dev], down));
            CUDA_CHECK(cudaMemcpy(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(c_to_f_q,c2f_map,sizeof(index_t)*my_nk,cudaMemcpyDefault));
            index_t off_old1 = -1;
            index_t off_old2 = -1;

            cufftDoubleComplex* loopfft = (cufftDoubleComplex*) cp->buf_GG[dev];
            for(index_t block = 0; block < cp->_tasks_prop[dev]; ++block) {
                index_t offmq = block +cp->_off_prop[dev];
                const index_t s2 = offmq/(n_unique_pairs*n_spin*n_spin*n_spin);
                offmq-=s2*(n_unique_pairs*n_spin*n_spin*n_spin);

                const index_t s3 = offmq/(n_unique_pairs*n_spin*n_spin);
                offmq-=s3*(n_unique_pairs*n_spin*n_spin);

                const index_t s4 = offmq/(n_unique_pairs*n_spin);
                offmq-=s4*(n_unique_pairs*n_spin);

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
                

                if(off_old1 != off_orb1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft,(cuDoubleComplex*)(GF+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj,(cuDoubleComplex*)(GFconj+off_orb1), nkbyte,cudaMemcpyHostToDevice,up));
                }
                if(off_old2 != off_orb2 && n_spin*n_orb > 1) {
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft+off_GF,(cuDoubleComplex*)(GF+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                    CUDA_CHECK(cudaMemcpyAsync(GF_fft_conj+off_GF,(cuDoubleComplex*)(GFconj+off_orb2), nkbyte,cudaMemcpyHostToDevice,up));
                }

                off_old1 = off_orb1;
                off_old2 = off_orb2;

                GF_prod_ph<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                             dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,up>>>(
                                    cp->buf_GG[dev],GF_fft,GF_fft+off_GF,
                                    GF_fft_conj,GF_fft_conj+off_GF, R1, R2,
                                    nkdev,cp->n_kblocks, cp->nk_per_thread);
                CUDA_CHECK( cudaStreamSynchronize( up ) );

                CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft[dev], loopfft, loopfft, CUFFT_INVERSE));

                CUDA_CHECK( cudaStreamSynchronize( down ) );
                to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                             dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                            cp->buf_res[dev][0], cp->buf_GG[dev],my_nk,c_to_f_q);
                const index_t loop_idx = mo1+n_orbff*(s1+s4*n_spin
                                        +POW2(n_spin)*(mo3+n_orbff*(s3 + s2*n_spin)));
                cudaMemcpyAsync((cuDoubleComplex*)(L+loop_idx*my_nk),cp->buf_res[dev][0], my_nk*sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToHost,down);

            }
            cudaDeviceSynchronize();
        },dev);
    }
    for (std::thread& t: allthreads) t.join();
    cudaDeviceSynchronize();
}


void pp_bubble_momenta_handler_fft_red(cuda_prop_descr_t* cp,  complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin,const index_t* const umi_l,const index_t* const umi_r,
    const index_t n_unique_pairs) {
    cuda_setup();
    
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));

    std::vector<std::thread> allthreads(tu_num_of_devices);
    for (index_t dev=0; dev<(index_t)tu_num_of_devices; ++dev) {
        allthreads[dev] = std::thread( [&](int dev){
            cudaSetDevice(dev);
            index_t* mi_to_R = cp->mi_to_R[dev];
            index_t* c_to_f_q = cp->c_to_f_q[dev];
            index_t* nkdev = cp->nkdev[dev];
            cuDoubleComplex* GF_fft = cp->GF_fft[dev];
            cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

            cudaStream_t up = tu_gpu_stream[dev];
            cudaStream_t down = tu_gpu_downstream[dev];
            cudaStream_t swap;
            const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
            CUFFT_CHECK(cufftSetStream(cp->gpu_fft[dev], down));
            CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft[dev], (void*)cp->fft_workspace[dev]));

            cudaDeviceSynchronize();
            CUDA_CHECK(cudaMemcpy(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(c_to_f_q,c2f_map,sizeof(index_t)*my_nk,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft,(cuDoubleComplex*)GF, _GF_size,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft_conj,(cuDoubleComplex*)GFconj, _GF_size,cudaMemcpyDefault));

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

                GF_prod_pp<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                             dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                    cp->buf_GG[dev],GF_fft+off_orb1,GF_fft+off_orb2,
                                    GF_fft_conj+off_orb1,GF_fft_conj+off_orb2,R1,R2,
                                    nkdev,cp->n_kblocks, cp->nk_per_thread);
                cufftDoubleComplex* loopfft = (cufftDoubleComplex*) cp->buf_GG[dev];
                CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft[dev], loopfft, loopfft, CUFFT_FORWARD));
                to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                             dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                            cp->buf_res[dev][0], cp->buf_GG[dev],my_nk,c_to_f_q);
                const index_t loop_idx = mo1+n_orbff*(s1+s2*n_spin
                                    +POW2(n_spin)*(mo3+n_orbff*
                                    (s3 + s4*n_spin)));
                cudaMemcpyAsync((cuDoubleComplex*)(L+loop_idx*my_nk),cp->buf_res[dev][0], my_nk*sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToHost,down);
            }
            cudaDeviceSynchronize();
        },dev);
    }
    for (std::thread& t: allthreads) t.join();
    cudaDeviceSynchronize();
}


__global__ void GF_prod_pp(__restrict__ cuDoubleComplex* const buf,
                    __restrict__ const cuDoubleComplex* const GF_off_1,
                    __restrict__ const cuDoubleComplex* const GF_off_2,
                    __restrict__ const cuDoubleComplex* const GFconj_off_1,
                    __restrict__ const cuDoubleComplex* const GFconj_off_2,
                    __restrict__ const index_t* const R1,
                    __restrict__ const index_t* const R2,
                    __restrict__ const index_t* const nk,
                    const index_t n_blocks,
                    const index_t nk_per_thread)
{
    index_t offmq = mem_offset();
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


__global__ void GF_prod_ph(__restrict__ cuDoubleComplex* const buf,
                    __restrict__ const cuDoubleComplex* const GF_off_1,
                    __restrict__ const cuDoubleComplex* const GF_off_2,
                    __restrict__ const cuDoubleComplex* const GFconj_off_1,
                    __restrict__ const cuDoubleComplex* const GFconj_off_2,
                    __restrict__ const index_t* const R1,
                    __restrict__ const index_t* const R2,
                    __restrict__ const index_t* const nk,
                    const index_t n_blocks,
                    const index_t nk_per_thread)
{
    index_t offmq = mem_offset();
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

__global__ void to_loop_simple(__restrict__ cuDoubleComplex* const loop,
                        __restrict__ const cuDoubleComplex* const res, const index_t my_nk,
                        __restrict__ const index_t* const c2fmap)
{
    index_t q = mem_offset();
    if(q>=my_nk) return;
    loop[q] = res[c2fmap[q]];
}
