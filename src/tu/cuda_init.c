/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "cuda_init.h"
#include <assert.h>

cuda_prop_descr_t cuda_prop_defaults_CPP( void ) { return cuda_prop_defaults; }
cuda_proj_descr_t cuda_proj_defaults_CPP( void ) { return cuda_proj_defaults; }

// globals
bool tu_has_gpu_setup = false;
int tu_num_of_devices = 0;
int tu_deviceid[32] = {0};
cudaStream_t* tu_gpu_stream = NULL;
cudaStream_t* tu_gpu_downstream = NULL;

#ifdef USE_CUDA
static int enablePeerAccess(const int nbGpus) {
    int currentDevice = 0;
    CUDA_CHECK( cudaGetDevice( &currentDevice ) );

    // Remark: access granted by this cudaDeviceEnablePeerAccess is unidirectional
    // Rows and columns represents a connectivity matrix between GPUs in the system
    for(int row=0; row < nbGpus; row++) {
        CUDA_CHECK( cudaSetDevice(row) );
        for(int col=0; col < nbGpus; col++) {
            if( row != col ){
                int canAccessPeer = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer( &canAccessPeer, row, col ));
                if ( canAccessPeer ){
                    CUDA_CHECK(cudaDeviceEnablePeerAccess( col, 0 ));
                }else{
                    mpi_err_printf("peer access %d->%d impossible\n", row, col );
                }
            }
        }
    }
    CUDA_CHECK( cudaSetDevice( currentDevice ) );
    return 0;
}


void cuda_setup( void ) {
    if(tu_has_gpu_setup)
        return;
    tu_has_gpu_setup = true;
    CUDA_CHECK(cudaGetDeviceCount(&tu_num_of_devices));
    mpi_vrb_printf("TU kernels using %d GPUs\n", tu_num_of_devices);
    tu_gpu_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t) *tu_num_of_devices);
    tu_gpu_downstream = (cudaStream_t*)malloc(sizeof(cudaStream_t) *tu_num_of_devices);

    for(int i = 0; i<tu_num_of_devices;++i) {
        tu_deviceid[i] = i;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreateWithFlags(&tu_gpu_stream[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&tu_gpu_downstream[i], cudaStreamNonBlocking));
    }

    CUDA_CHECK( cudaSetDevice(tu_deviceid[0]) );
    // enable NVlink
    if(tu_num_of_devices>1) {
        int err = enablePeerAccess( tu_num_of_devices );
        (void)err;
        assert( err == 0 );
    }
    CUDA_CHECK( cudaSetDevice(tu_deviceid[0]) );
    mpi_vrb_printf("successful TU GPU initialization.\n");
}

void cuda_deconstruct( void ) {
    if(!tu_has_gpu_setup)
        return;
    tu_has_gpu_setup = false;

    for(int i = 0; i<tu_num_of_devices;++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamDestroy(tu_gpu_stream[i]));
        CUDA_CHECK(cudaStreamDestroy(tu_gpu_downstream[i]));
    }
    CUDA_CHECK(cudaSetDevice(tu_deviceid[0]));
    free(tu_gpu_stream);
    free(tu_gpu_downstream);
    CUDA_CHECK(cudaDeviceReset());
}


void cuda_delete_prop_descr(cuda_prop_descr_t* cp) {
    if(!cp->prop_has_gpu_setup)
        return;
    cp->prop_has_gpu_setup = false;
    for(int i = 0; i<tu_num_of_devices;++i) {CUFFT_CHECK(cufftDestroy(cp->gpu_fft[i]));}
    free(cp->gpu_fft);
}

#else

void cuda_delete_prop_descr(cuda_prop_descr_t* cp) {(void)cp;}
void cuda_setup( void ) {}
void cuda_deconstruct( void ) {}
#endif

void cuda_reset_prop_descr(cuda_prop_descr_t* cp,index_t tasks_spin, index_t tasks_dist, double offload_cpu) {
    mpi_wrn_printf("cuda_reset_prop_descr(...) used for testing. use with caution!\n");
    if(!cp->prop_has_gpu_setup) {
        mpi_err_printf("wrong call order. call cuda_setup_prop_descr(...) first!\n");
        return ;
    }
    cp->prop_has_gpu_setup = false;
    cuda_setup_prop_descr(cp,tasks_spin,tasks_dist,NULL,offload_cpu);
}


void cuda_setup_projections(cuda_proj_descr_t* cp,index_t ntasks, index_t my_nk) {
    #ifndef USE_CUDA
    (void)cp,(void)ntasks,(void)my_nk;
    #else

    if(cp->proj_has_gpu_setup) return;
    if(!tu_has_gpu_setup) cuda_setup();
    cp->proj_has_gpu_setup = true;
    double numt = cp->nthread[0]*cp->nthread[1]*cp->nthread[2];
    double min_blocks = ((ntasks+1)+numt-1)/numt;
    double min_xy = sqrt(min_blocks);
    cp->nblock[0] = (int)ceil(min_xy); cp->nblock[1] = (int)ceil(min_xy); cp->nblock[2] = 1;

    int rest = (my_nk)%tu_num_of_devices;
    int dev_i = 0;
    for (int i=0; i<tu_num_of_devices; ++i)
        cp->_threadkpts_proj[i] = (my_nk-rest)/tu_num_of_devices;

    while (rest>0) {
        cp->_threadkpts_proj[dev_i]+=1; rest-=1; dev_i+=1;
    }

    for (int i=0; i<tu_num_of_devices; ++i) cp->_qstart_proj[i] = 0;
    for (int ii=1; ii<tu_num_of_devices;++ii)
        cp->_qstart_proj[ii] = cp->_threadkpts_proj[ii-1]+cp->_qstart_proj[ii-1];
    for (int ii=0; ii<tu_num_of_devices;++ii)
        mpi_vrb_printf("TU:proj GPU %d: nq=%4d (+%4d)\n", ii, cp->_threadkpts_proj[ii] , cp->_qstart_proj[ii]);

    #endif
}

void cuda_setup_prop_descr(cuda_prop_descr_t* cp,  index_t tasks_spin,
                index_t tasks_dist, const index_t* nkpts, double offload_cpu) {
#ifndef USE_CUDA
    (void)tasks_spin,(void)tasks_dist,(void)nkpts,(void)offload_cpu;
    cp->_tasks_cpu = tasks_dist;
    cp->_off_cpu = 0;
#else
    if(cp->prop_has_gpu_setup) return;
    if(!tu_has_gpu_setup) cuda_setup();
    cp->prop_has_gpu_setup = true;
    cp->_tasks_cpu = (index_t)(offload_cpu*tasks_dist);
    cp->_off_cpu = tasks_dist - cp->_tasks_cpu;
    index_t _tasks_gpu = cp->_off_cpu*tasks_spin;

    int rest = _tasks_gpu%tu_num_of_devices;
    int dev_i = 0;
    for (int i=0; i<tu_num_of_devices; ++i)
        cp->_tasks_prop[i] = (_tasks_gpu-rest)/tu_num_of_devices;
    while (rest>0) {
        cp->_tasks_prop[dev_i]+=1; rest-=1; dev_i+=1;
    }
    for (int i=0; i<tu_num_of_devices; ++i) cp->_off_prop[i] = 0;
    for (int ii=1; ii<tu_num_of_devices; ++ii)
        cp->_off_prop[ii] = cp->_tasks_prop[ii-1]+cp->_off_prop[ii-1];
    for (int ii=0; ii<tu_num_of_devices; ++ii)
        mpi_vrb_printf("TU:prop GPU %d: ntasks=%9d (+%9d)\n", ii, cp->_tasks_prop[ii], cp->_off_prop[ii]);
    mpi_vrb_printf("TU:prop CPU  : ntasks=%9d (+%9d)\n", cp->_tasks_cpu, cp->_off_cpu);
    if(!cp->dims_set) {
        cp->dims[0] = nkpts[0], cp->dims[1] = nkpts[1],
        cp->dims[2] = nkpts[2];
        cp->dims_set = true;
    }
    double numt = cp->nthread[0]*cp->nthread[1]*cp->nthread[2];
    double min_blocks = (cp->dims[0]*cp->dims[1]*cp->dims[2]+numt-1)/numt;
    cp->nblock[0] =(int)min_blocks; cp->nblock[1] = cp->nblock[2] = 1;

    long long int nk = cp->dims[0]*cp->dims[1]*cp->dims[2];
    cp->gpu_fft = (cufftHandle*)malloc(sizeof(cufftHandle)*tu_num_of_devices);
    for(int i = 0; i<tu_num_of_devices;++i) {
        CUDA_CHECK( cudaSetDevice(tu_deviceid[i]) );
        CUFFT_CHECK(cufftCreate(&(cp->gpu_fft[i])));
        CUFFT_CHECK(cufftSetAutoAllocation(cp->gpu_fft[i], false));
        CUFFT_CHECK(cufftMakePlanMany64(cp->gpu_fft[i], 3, cp->dims, NULL,
                    1, nk, NULL, 1, nk, CUFFT_Z2Z, 1,
                    &(cp->worksize_fft[i])));
        CUFFT_CHECK(cufftGetSize(cp->gpu_fft[i], &(cp->worksize_fft[i])));
    }
    CUDA_CHECK( cudaSetDevice(0) );

    while(nk%cp->nk_per_thread!=0 && cp->nk_per_thread>1) {
        cp->nk_per_thread -= 1;
    }
    cp->n_kblocks = nk/cp->nk_per_thread;
    mpi_vrb_printf("TU prop GPU setup: nblock=%d, nk_per_thread=%d, nk=%d\n", cp->n_kblocks, cp->nk_per_thread, nk);
#endif
}
