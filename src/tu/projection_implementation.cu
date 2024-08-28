/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#define EIGEN_NO_CUDA

#include "projection_implementation.hpp"

#include "../misc/cuda_error.h"

#include <cuComplex.h>

#include <algorithm>
#include <numeric>
#include <thread>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include "cuda_init.hpp"
#include "../misc/mpi_log.h"

static inline __device__ size_t mem_offset() {
    size_t blockId = blockIdx.x + blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;
    size_t threadId = blockId * (blockDim.x * blockDim.y) +
        threadIdx.x + threadIdx.y * blockDim.x;
    return threadId;
}

__global__ void Proj_to_P_ft(__restrict__ cuDoubleComplex* const TO,
                    __restrict__ const cuDoubleComplex* const FROM,
                    const size_t n_orbff,const size_t my_nk,
                    const size_t n_spin, const size_t current_qTO,
                    const index_t len_maps, const size_t n_projpref,
                    const index_t* const rsmap_to,
                    const index_t* const rsmap_from,
                    const index_t* const mom,
                    const cuDoubleComplex* const proj_prefac);

__global__ void Proj_to_C_ft(__restrict__ cuDoubleComplex* const TO,
                    __restrict__ const cuDoubleComplex* const FROM,
                    const size_t n_orbff,const size_t my_nk,
                    const size_t n_spin, const size_t current_qTO,
                    const index_t len_maps, const size_t n_projpref,
                    const index_t* const rsmap_to,
                    const index_t* const rsmap_from,
                    const index_t* const mom,
                    const cuDoubleComplex* const proj_prefac);

__global__ void Proj_to_D_ft(__restrict__ cuDoubleComplex* const TO,
                    __restrict__ const cuDoubleComplex* const FROM,
                    const size_t n_orbff,const size_t my_nk,
                    const size_t n_spin, const size_t current_qTO,
                    const index_t len_maps, const size_t n_projpref,
                    const index_t* const rsmap_to,
                    const index_t* const rsmap_from,
                    const index_t* const mom,
                    const cuDoubleComplex* const proj_prefac);

void GPUproj_handler_ft(const cuda_proj_descr_t* cp,
                    const char channel, complex128_t* proj_vertex,const complex128_t* const I_channel,
                    const vector<index_t>& rsmap_to,
                    const vector<index_t>& rsmap_from,
                    const vector<index_t>& mom_in,
                    const complex128_t* const proj_prefac,
                    const index_t n_orbff,
                    const index_t n_spin,const index_t my_nk,
                    const size_t n_projpref) {
    if(channel != 'P' && channel != 'D' && channel != 'C')
        return;

    cuda_setup();
    const index_t blocksize = POW2(n_orbff*POW2(n_spin));
    cudaDeviceSynchronize();

    vector<std::thread> allthreads(tu_num_of_devices);
    for (size_t dev=0; dev<(size_t)tu_num_of_devices; ++dev) {
        allthreads[dev] = std::thread( [&](int dev){
            cudaSetDevice(dev);

            cuDoubleComplex* prefacs = cp->prefacs[dev];
            index_t* rs_to = cp->rs_to[dev];
            index_t* rs_from = cp->rs_from[dev];
            index_t* mom = cp->mom[dev];
            cuDoubleComplex* buf_ondev = cp->buf_ondev[dev];
            cuDoubleComplex* buf_res0 = cp->buf_res0[dev];
            cuDoubleComplex* buf_res1 = cp->buf_res1[dev];
            cuDoubleComplex* buf_resh;
            cudaStream_t up = tu_gpu_stream[dev];
            cudaStream_t down = tu_gpu_downstream[dev];
            cudaStream_t swap;

            CUDA_CHECK(cudaMemcpyAsync(prefacs, proj_prefac,sizeof(cuDoubleComplex)*n_projpref*my_nk,cudaMemcpyDefault,down));
            CUDA_CHECK(cudaMemcpyAsync(buf_ondev, I_channel, sizeof(cuDoubleComplex)*POW2(n_orbff*POW2(n_spin))*n_projpref,cudaMemcpyDefault,down));
            CUDA_CHECK(cudaMemcpyAsync(rs_to, rsmap_to.data(), sizeof(index_t)*rsmap_to.size(),cudaMemcpyDefault,down));
            CUDA_CHECK(cudaMemcpyAsync(rs_from, rsmap_from.data(), sizeof(index_t)*rsmap_from.size(),cudaMemcpyDefault,down));
            CUDA_CHECK(cudaMemcpyAsync(mom, mom_in.data(), sizeof(index_t)*mom_in.size(),cudaMemcpyDefault,down));
            for(size_t cq = 0; cq < cp->_threadkpts_proj[dev];++cq) {

                CUDA_CHECK(cudaMemsetAsync(buf_res0,0,sizeof(cuDoubleComplex)*POW2(n_orbff*POW2(n_spin)),down));
                size_t qTO = cp->_qstart_proj[dev]+cq;
                if (channel == 'P') {
                    Proj_to_P_ft<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                                    dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>
                            (buf_res0, buf_ondev, n_orbff,my_nk, n_spin,
                                     qTO,rsmap_to.size()/2,n_projpref,rs_to, rs_from, mom, prefacs);
                }else if(channel == 'C') {
                    Proj_to_C_ft<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                                 dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>
                            (buf_res0, buf_ondev, n_orbff,my_nk, n_spin,
                                     qTO,rsmap_to.size()/2,n_projpref,rs_to, rs_from, mom, prefacs);
                }else if(channel == 'D') {
                    Proj_to_D_ft<<<dim3(cp->nblock[0],cp->nblock[1],cp->nblock[2]),
                                 dim3(cp->nthread[0],cp->nthread[1],cp->nthread[2]),0,down>>>
                            (buf_res0, buf_ondev, n_orbff,my_nk, n_spin,
                                     qTO,rsmap_to.size()/2,n_projpref,rs_to, rs_from, mom, prefacs);
                }
                CUDA_CHECK(cudaMemcpyAsync(proj_vertex+qTO*blocksize,
                        buf_res0, sizeof(cuDoubleComplex)*blocksize,
                        cudaMemcpyDeviceToHost,down));
                swap = up; up = down; down = swap;
                buf_resh = buf_res0;buf_res0 = buf_res1; buf_res1 = buf_resh;
            }
            cudaDeviceSynchronize();

            cudaSetDevice(tu_deviceid[0]);
        },dev);
    }
    for (std::thread& t: allthreads) t.join();
    cudaDeviceSynchronize();
}

__global__ void Proj_to_P_ft(__restrict__ cuDoubleComplex* const TO, __restrict__ const cuDoubleComplex* const FROM,
                       const size_t n_orbff,const size_t my_nk,
                    const size_t n_spin, const size_t current_qTO,
                    const index_t len_maps, const size_t n_projpref,
                    const index_t* const rsmap_to,
                    const index_t* const rsmap_from,
                    const index_t* const mom,
                    const cuDoubleComplex* const proj_prefac)
{
    size_t offmq = mem_offset();

    if(offmq>=len_maps*n_spin*n_spin*n_spin*n_spin) return;
    const size_t s4 = offmq/(len_maps*n_spin*n_spin
                            *n_spin);
    offmq-=s4*(len_maps*n_spin*n_spin
                            *n_spin);
    const size_t s3 = offmq/(len_maps*n_spin*n_spin);
    offmq-=s3*(len_maps*n_spin*n_spin);

    const size_t s2 = offmq/(len_maps*n_spin);
    offmq-=s2*(len_maps*n_spin);

    const size_t s1 = offmq/(len_maps);
    const size_t orb_map = offmq - s1*(len_maps);


    const cuDoubleComplex buf_shift = FROM[mom[2*orb_map]+n_projpref*
                (rsmap_from[2*orb_map]+n_orbff*(
                        (s1+s2*n_spin)+POW2(n_spin)*(
                            rsmap_from[2*orb_map+1]+n_orbff*(
                                (s3+s4*n_spin)))))];
    cuDoubleComplex tmp = cuCmul(buf_shift,proj_prefac[current_qTO+my_nk
                                    *mom[2*orb_map+1]]);

    const index_t idx_TO = rsmap_to[2*orb_map]+n_orbff*(
            (s1+s2*n_spin)+POW2(n_spin)*(
                rsmap_to[2*orb_map+1]+n_orbff*(
                    (s3+s4*n_spin))));
    atomicAdd(((double*)(TO+idx_TO)), cuCreal(tmp));
    atomicAdd(((double*)(TO+idx_TO))+1, cuCimag(tmp));
}


__global__ void Proj_to_C_ft(__restrict__ cuDoubleComplex* const TO, __restrict__ const cuDoubleComplex* const FROM,
                       const size_t n_orbff,const size_t my_nk,
                    const size_t n_spin, const size_t current_qTO,
                    const index_t len_maps, const size_t n_projpref,
                    const index_t* const rsmap_to,
                    const index_t* const rsmap_from,
                    const index_t* const mom,
                    const cuDoubleComplex* const proj_prefac)
{
    size_t offmq = mem_offset();

    if(offmq>=len_maps*n_spin*n_spin
                            *n_spin*n_spin) return;
    const size_t s4 = offmq/(len_maps*n_spin*n_spin
                            *n_spin);
    offmq-=s4*(len_maps*n_spin*n_spin
                            *n_spin);
    const size_t s3 = offmq/(len_maps*n_spin*n_spin);
    offmq-=s3*(len_maps*n_spin*n_spin);

    const size_t s2 = offmq/(len_maps*n_spin);
    offmq-=s2*(len_maps*n_spin);

    const size_t s1 = offmq/(len_maps);
    const size_t orb_map = offmq - s1*(len_maps);


    const cuDoubleComplex buf_shift = FROM[mom[2*orb_map]+n_projpref*
                (rsmap_from[2*orb_map]+n_orbff*(
                        (s1+s2*n_spin)+POW2(n_spin)*(
                            rsmap_from[2*orb_map+1]+n_orbff*(
                                (s3+s4*n_spin)))))];
    cuDoubleComplex tmp = cuCmul(buf_shift,proj_prefac[current_qTO+my_nk
                                                        *mom[2*orb_map+1]]);

    const index_t idx_TO = rsmap_to[2*orb_map]+n_orbff*(
            (s1+s4*n_spin)+POW2(n_spin)*(
                rsmap_to[2*orb_map+1]+n_orbff*(
                    (s3+s2*n_spin))));
    atomicAdd(((double*)(TO+idx_TO)), cuCreal(tmp));
    atomicAdd(((double*)(TO+idx_TO))+1, cuCimag(tmp));

}


__global__ void Proj_to_D_ft(__restrict__ cuDoubleComplex* const TO, __restrict__ const cuDoubleComplex* const FROM,
                    const size_t n_orbff,const size_t my_nk,
                    const size_t n_spin, const size_t current_qTO,
                    const index_t len_maps, const size_t n_projpref,
                    const index_t* const rsmap_to,
                    const index_t* const rsmap_from,
                    const index_t* const mom,
                    const cuDoubleComplex* const proj_prefac)
{
    size_t offmq = mem_offset();

    if(offmq>=len_maps*n_spin*n_spin
                            *n_spin*n_spin) return;
    const size_t s4 = offmq/(len_maps*n_spin*n_spin
                            *n_spin);
    offmq-=s4*(len_maps*n_spin*n_spin
                            *n_spin);
    const size_t s3 = offmq/(len_maps*n_spin*n_spin);
    offmq-=s3*(len_maps*n_spin*n_spin);

    const size_t s2 = offmq/(len_maps*n_spin);
    offmq-=s2*(len_maps*n_spin);

    const size_t s1 = offmq/(len_maps);
    const size_t orb_map = offmq - s1*(len_maps);

    const cuDoubleComplex buf_shift = FROM[mom[2*orb_map]+n_projpref*
                (rsmap_from[2*orb_map]+n_orbff*(
                        (s1+s2*n_spin)+POW2(n_spin)*(
                            rsmap_from[2*orb_map+1]+n_orbff*(
                                (s3+s4*n_spin)))))];
    cuDoubleComplex tmp = cuCmul(buf_shift,proj_prefac[current_qTO+my_nk
                                                    *mom[2*orb_map+1]]);

    const index_t idx_TO = rsmap_to[2*orb_map]+n_orbff*(
            (s1+s3*n_spin)+POW2(n_spin)*(
                rsmap_to[2*orb_map+1]+n_orbff*(
                    (s4+s2*n_spin))));
    atomicAdd(((double*)(TO+idx_TO)), cuCreal(tmp));
    atomicAdd(((double*)(TO+idx_TO))+1, cuCimag(tmp));
}
