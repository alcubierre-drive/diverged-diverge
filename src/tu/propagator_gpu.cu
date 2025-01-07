/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#define EIGEN_NO_CUDA

#include "propagator_details.h"
#include "cuda_init.h"

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
    const index_t n_orb, const index_t n_bonds, const index_t* const bond_sizes,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* mi_to_from,
    const index_t* mi_to_oto,
    const index_t* const mi_to_Rin) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    tu_gpu_info_t ii = tu_gpu_info();
    #pragma omp parallel for num_threads(ii.tu_num_of_devices)
    for (int dev=0; dev<ii.tu_num_of_devices; ++dev) {
        cudaSetDevice(dev);
        index_t* mi_to_R = cp->mi_to_R[dev];
        index_t* c_to_f_q = cp->c_to_f_q[dev];
        index_t* nkdev = cp->nkdev[dev];
        cuDoubleComplex* GF_fft = cp->GF_fft[dev];
        cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

        cudaStream_t up = ii.tu_gpu_stream[dev];
        cudaStream_t down = ii.tu_gpu_downstream[dev];

        const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;
        index_t off_GF1 = 0;
        index_t off_GF2 = (n_spin*n_orb > 1) ? nktot : 0;
        cufftDoubleComplex* loopfft = (cufftDoubleComplex*) cp->buf_GG[dev];
        CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft[dev], cp->fft_workspace[dev]));
        CUFFT_CHECK(cufftSetStream(cp->gpu_fft[dev], down));

        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        CUDA_CHECK(cudaMemcpyAsync(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault,down));
        CUDA_CHECK(cudaMemcpyAsync(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault,down));
        CUDA_CHECK(cudaMemcpyAsync(c_to_f_q,c2f_map,sizeof(index_t)*my_nk,cudaMemcpyDefault,down));
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

            CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft[dev], loopfft, loopfft, CUFFT_INVERSE));
            to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                            dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                        cp->buf_res[dev], cp->buf_GG[dev],my_nk,c_to_f_q);

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


void pp_bubble_momenta_handler_fft(cuda_prop_descr_t* cp,  complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb, const index_t n_bonds, const index_t* const bond_sizes,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* mi_to_from,
    const index_t* mi_to_oto,
    const index_t* const mi_to_Rin) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    tu_gpu_info_t ii = tu_gpu_info();
    #pragma omp parallel for num_threads(ii.tu_num_of_devices)
    for (int dev=0; dev<ii.tu_num_of_devices; ++dev) {
        cudaSetDevice(dev);
        index_t* mi_to_R = cp->mi_to_R[dev];
        index_t* c_to_f_q = cp->c_to_f_q[dev];
        index_t* nkdev = cp->nkdev[dev];
        cuDoubleComplex* GF_fft = cp->GF_fft[dev];
        cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

        cudaStream_t up = ii.tu_gpu_stream[dev];
        cudaStream_t down = ii.tu_gpu_downstream[dev];

        const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;
        index_t off_GF1 = 0;
        index_t off_GF2 = (n_spin*n_orb > 1) ? nktot : 0;
        cufftDoubleComplex* loopfft = (cufftDoubleComplex*) cp->buf_GG[dev];
        CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft[dev], cp->fft_workspace[dev]));
        CUFFT_CHECK(cufftSetStream(cp->gpu_fft[dev], down));

        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        CUDA_CHECK(cudaMemcpyAsync(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault,down));
        CUDA_CHECK(cudaMemcpyAsync(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault,down));
        CUDA_CHECK(cudaMemcpyAsync(c_to_f_q,c2f_map,sizeof(index_t)*my_nk,cudaMemcpyDefault,down));
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

            CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft[dev], loopfft, loopfft, CUFFT_FORWARD));
            to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                        dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                        cp->buf_res[dev], cp->buf_GG[dev],my_nk,c_to_f_q);
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

void ph_bubble_momenta_handler_fft_red(cuda_prop_descr_t* cp,  complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin,const index_t* const umi_l,const index_t* const umi_r,
    const index_t n_unique_pairs, cbyte_t* L_touched_map ) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    tu_gpu_info_t ii = tu_gpu_info();
    #pragma omp parallel for num_threads(ii.tu_num_of_devices)
    for (int dev=0; dev<ii.tu_num_of_devices; ++dev) {
        cudaSetDevice(dev);
        index_t* mi_to_R = cp->mi_to_R[dev];
        index_t* c_to_f_q = cp->c_to_f_q[dev];
        index_t* nkdev = cp->nkdev[dev];
        cuDoubleComplex* GF_fft = cp->GF_fft[dev];
        cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

        cudaStream_t up = ii.tu_gpu_stream[dev];
        cudaStream_t down = ii.tu_gpu_downstream[dev];
        const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;

        index_t off_GF1 = 0;
        index_t off_GF2 = (n_spin*n_orb > 1) ? nktot : 0;
        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft[dev], (void*)cp->fft_workspace[dev]));
        CUFFT_CHECK(cufftSetStream(cp->gpu_fft[dev], down));
        CUDA_CHECK(cudaMemcpy(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(c_to_f_q,c2f_map,sizeof(index_t)*my_nk,cudaMemcpyDefault));
        if(cp->prop_preload_full_gf) {
            CUDA_CHECK(cudaMemcpy(GF_fft,(cuDoubleComplex*)GF, _GF_size,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft_conj,(cuDoubleComplex*)GFconj, _GF_size,cudaMemcpyDefault));
        }
        cudaDeviceSynchronize();

        index_t off_old1 = -1;
        index_t off_old2 = -1;
        cufftDoubleComplex* loopfft = (cufftDoubleComplex*) cp->buf_GG[dev];
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
            CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft[dev], loopfft, loopfft, CUFFT_INVERSE));

            to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                         dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                        cp->buf_res[dev], cp->buf_GG[dev],my_nk,c_to_f_q);
            const index_t loop_idx = mo1+n_orbff*(s1+s4*n_spin
                                    +POW2(n_spin)*(mo3+n_orbff*(s3 + s2*n_spin)));
            cudaMemcpyAsync((cuDoubleComplex*)(L+loop_idx*my_nk),cp->buf_res[dev], my_nk*sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToHost,down);
            if (L_touched_map) {
                for (index_t k=0; k<my_nk; ++k) {
                    bitary_set( L_touched_map, loop_idx*my_nk + k );
                }
            }

        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}


void pp_bubble_momenta_handler_fft_red(cuda_prop_descr_t* cp,  complex128_t* const L,
    const complex128_t* const GF, const complex128_t* const GFconj,
    const index_t n_orb,
    const index_t n_orbff, const index_t n_spin, const index_t nktot,
    const index_t my_nk, const index_t* const c2f_map, const index_t* nkref,
    const index_t* const mi_to_ofrom, const index_t* const mi_to_oto,
    const index_t* const mi_to_Rin,const index_t* const umi_l,const index_t* const umi_r,
    const index_t n_unique_pairs, cbyte_t* L_touched_map ) {
    cuda_setup();
    memset(L,0,POW4(n_spin)*POW2(n_orbff)*my_nk*sizeof(complex128_t));
    tu_gpu_info_t ii = tu_gpu_info();
    #pragma omp parallel for num_threads(ii.tu_num_of_devices)
    for (int dev=0; dev<ii.tu_num_of_devices; ++dev) {
        cudaSetDevice(dev);
        index_t* mi_to_R = cp->mi_to_R[dev];
        index_t* c_to_f_q = cp->c_to_f_q[dev];
        index_t* nkdev = cp->nkdev[dev];
        cuDoubleComplex* GF_fft = cp->GF_fft[dev];
        cuDoubleComplex* GF_fft_conj = cp->GF_fft_conj[dev];

        cudaStream_t up = ii.tu_gpu_stream[dev];
        cudaStream_t down = ii.tu_gpu_downstream[dev];
        const index_t nkbyte = sizeof(cuDoubleComplex)*nktot;

        index_t off_GF1 = 0;
        index_t off_GF2 = (n_spin*n_orb > 1) ? nktot : 0;
        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        CUFFT_CHECK(cufftSetWorkArea(cp->gpu_fft[dev], (void*)cp->fft_workspace[dev]));
        CUFFT_CHECK(cufftSetStream(cp->gpu_fft[dev], down));
        CUDA_CHECK(cudaMemcpy(nkdev,nkref, sizeof(index_t)*3,cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(mi_to_R,mi_to_Rin, sizeof(index_t)*n_orbff*3,cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(c_to_f_q,c2f_map,sizeof(index_t)*my_nk,cudaMemcpyDefault));
        if(cp->prop_preload_full_gf) {
            CUDA_CHECK(cudaMemcpy(GF_fft,(cuDoubleComplex*)GF, _GF_size,cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(GF_fft_conj,(cuDoubleComplex*)GFconj, _GF_size,cudaMemcpyDefault));
        }
        cudaDeviceSynchronize();

        index_t off_old1 = -1;
        index_t off_old2 = -1;
        cufftDoubleComplex* loopfft = (cufftDoubleComplex*) cp->buf_GG[dev];
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
            CUFFT_CHECK(cufftExecZ2Z(cp->gpu_fft[dev], loopfft, loopfft, CUFFT_FORWARD));

            to_loop_simple<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                         dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>(
                                        cp->buf_res[dev], cp->buf_GG[dev],my_nk,c_to_f_q);
            const index_t loop_idx = mo1+n_orbff*(s1+s2*n_spin
                                    +POW2(n_spin)*(mo3+n_orbff*(s3 + s4*n_spin)));
            cudaMemcpyAsync((cuDoubleComplex*)(L+loop_idx*my_nk),cp->buf_res[dev], my_nk*sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToHost,down);
            if (L_touched_map) {
                for (index_t k=0; k<my_nk; ++k) {
                    bitary_set( L_touched_map, loop_idx*my_nk + k );
                }
            }
        }
        cudaDeviceSynchronize();
    }
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
