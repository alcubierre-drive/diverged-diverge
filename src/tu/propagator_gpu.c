#include "propagator_details.h"

#ifdef USE_CUDA

void merge_gpu_cpu_loop( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu) {
    index_t blocksize = POW2(t->n_orbff*POW2(t->n_spin));
    for (index_t b=0; b<blocksize; ++b)
    for (index_t k=0; k<t->my_nk; ++k)
        L[IDX2(k,b,blocksize)] += tmpgpu[IDX2(b,k,t->my_nk)];
    if (t->tu_data->use_reduced_loop) reconstruct_from_reduced( t, L );
    scale_loop( L, -1./(2.*M_PI), t->my_nk*POW4(t->n_spin)*POW2(t->n_orbff) );
}

void merge_gpu_cpu_loop_cfrg_step1( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu ) {
    index_t blocksize = POW2(t->n_orbff*POW2(t->n_spin));
    for (index_t b=0; b<blocksize; ++b)
    for (index_t k=0; k<t->my_nk; ++k)
        L[IDX2(k,b,blocksize)] += tmpgpu[IDX2(b,k,t->my_nk)];
}

void merge_gpu_cpu_loop_cfrg_step2( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu ) {
    index_t blocksize = POW2(t->n_orbff*POW2(t->n_spin));
    for (index_t b=0; b<blocksize; ++b)
    for (index_t k=0; k<t->my_nk; ++k)
        L[IDX2(k,b,blocksize)] -= tmpgpu[IDX2(b,k,t->my_nk)];
    if (t->tu_data->use_reduced_loop) reconstruct_from_reduced( t, L );
    scale_loop( L, -1./(2.*M_PI), t->my_nk*POW4(t->n_spin)*POW2(t->n_orbff) );
}

#ifndef CUDA_MEM_ALIGN
#define CUDA_MEM_ALIGN 256
#endif

static index_t ensure_align(index_t numToRound) {
    index_t remainder = numToRound % CUDA_MEM_ALIGN;
    if (remainder == 0) return numToRound;
    return numToRound + CUDA_MEM_ALIGN - remainder;
}

void allocate_GPU_memory( tu_prop_t* t ) {
    if (t->cuda_descr.prop_memory_set_vertex)
        return;
    t->cuda_descr.prop_memory_set_local = true;

    index_t prop_memsize[11] = {0};
    const index_t _GF_size = sizeof(cuDoubleComplex)*t->n_orb*t->n_spin*t->nktot*t->n_orb*t->n_spin;

    for (int i=0; i<tu_gpu_info().tu_num_of_devices; ++i) {
        CUDA_CHECK( cudaSetDevice(i) );
        index_t my_nk_tot = (t->tu_data->use_reduced_loop && t->tu_data->use_cuda_mpi_loop) ? t->nkibz : t->my_nk;
        prop_memsize[0] = ensure_align(sizeof(index_t)*t->n_orbff*3); // mi_to_R
        prop_memsize[1] = ensure_align(sizeof(index_t)*my_nk_tot); // c_to_f_q
        prop_memsize[2] = ensure_align(sizeof(index_t)*3); // nkdev
        prop_memsize[3] = ensure_align(_GF_size); // GF_fft
        prop_memsize[4] = ensure_align(_GF_size); // GF_fft_conj
        prop_memsize[5] = ensure_align(sizeof(cuDoubleComplex)*t->nktot); // buf_GG
        prop_memsize[6] = ensure_align(sizeof(cuDoubleComplex)*my_nk_tot); // buf_res 0
        prop_memsize[7] = ensure_align(sizeof(cuDoubleComplex)*my_nk_tot); // buf_res 1
        prop_memsize[8] = ensure_align(sizeof(cuDoubleComplex)*t->cuda_descr.worksize_fft[i]); // fft_workspace 0

        CUDA_CHECK( cudaMalloc((void**)&(t->cuda_descr.mi_to_R[i]), prop_memsize[0]) );
        CUDA_CHECK( cudaMalloc((void**)&(t->cuda_descr.c_to_f_q[i]), prop_memsize[1]) );
        CUDA_CHECK( cudaMalloc((void**)&(t->cuda_descr.nkdev[i]), prop_memsize[2]) );
        CUDA_CHECK( cudaMalloc((void**)&(t->cuda_descr.GF_fft[i]), prop_memsize[3]) );
        CUDA_CHECK( cudaMalloc((void**)&(t->cuda_descr.GF_fft_conj[i]), prop_memsize[4]) );
        CUDA_CHECK( cudaMalloc((void**)&(t->cuda_descr.buf_GG[i]), prop_memsize[5]) );
        CUDA_CHECK( cudaMalloc((void**)&(t->cuda_descr.buf_res[i]), prop_memsize[6]) );
        CUDA_CHECK( cudaMalloc((void**)&(t->cuda_descr.fft_workspace[i]), prop_memsize[8]) );
        CUDA_CHECK( cudaSetDevice(0) );
    }
}

void S_ph_loop_static_simple_gpu( tu_prop_t* t, complex128_t* L ) {
    if (!t->cuda_descr.prop_memory_set_local && !t->cuda_descr.prop_memory_set_vertex) {
        mpi_wrn_printf("Memory for GPU calculation was not allocated by vertex\n");
        allocate_GPU_memory( t );
    }
    const index_t* vnk = t->model->nk;
    const index_t* vnkf = t->model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]), (vnk[2]*vnkf[2])};
    ph_bubble_momenta_handler_fft(&t->cuda_descr, L, t->GF_fft_plus, t->GF_fft_conj_plus,
                            t->n_orb, t->n_bonds, t->tu_data->bond_sizes, t->n_orbff,
                            t->n_spin, t->nktot, t->my_nk, t->c2fmap, dims, t->mi_to_ofrom,
                            t->mi_to_oto, t->mi_to_R);
}

void S_pp_loop_static_simple_gpu( tu_prop_t* t, complex128_t* L ) {
    if (!t->cuda_descr.prop_memory_set_local && !t->cuda_descr.prop_memory_set_vertex) {
        mpi_wrn_printf("Memory for GPU calculation was not allocated by vertex\n");
        allocate_GPU_memory( t );
    }
    const index_t* vnk = t->model->nk;
    const index_t* vnkf = t->model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]), (vnk[2]*vnkf[2])};
    pp_bubble_momenta_handler_fft(&t->cuda_descr, L, t->GF_fft_plus,t->GF_fft_conj_plus,
                            t->n_orb, t->n_bonds, t->tu_data->bond_sizes, t->n_orbff,
                            t->n_spin, t->nktot, t->my_nk, t->c2fmap, dims, t->mi_to_ofrom,
                            t->mi_to_oto, t->mi_to_R);
}

void S_ph_loop_static_red_gpu( tu_prop_t* t, complex128_t* L ) {
    if(!t->cuda_descr.prop_memory_set_local && !t->cuda_descr.prop_memory_set_vertex) {
        mpi_wrn_printf("Memory for GPU calculation was not allocated by vertex\n");
        allocate_GPU_memory( t );
    }

    if (t->tu_data->use_cuda_mpi_loop) {
        S_loop_static_red_gpu_mpi( t, L, false );
        return;
    }

    const index_t* vnk = t->model->nk;
    const index_t* vnkf = t->model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]), (vnk[2]*vnkf[2])};
    ph_bubble_momenta_handler_fft_red(&t->cuda_descr, L, t->GF_fft_plus, t->GF_fft_conj_plus,
                            t->n_orb, t->n_orbff, t->n_spin, t->nktot, t->my_nk, t->c2fmap,
                            dims, t->mi_to_ofrom, t->mi_to_oto, t->mi_to_R, t->tu_data->unique_mi_l,
                            t->tu_data->unique_mi_r, t->tu_data->unique_mi_l_sz, NULL);
}

void S_pp_loop_static_red_gpu( tu_prop_t* t, complex128_t* L ) {
    if (!t->cuda_descr.prop_memory_set_local && !t->cuda_descr.prop_memory_set_vertex) {
        mpi_wrn_printf("Memory for GPU calculation was not allocated by vertex\n");
        allocate_GPU_memory( t );
    }

    if (t->tu_data->use_cuda_mpi_loop) {
        S_loop_static_red_gpu_mpi( t, L, true );
        return;
    }

    const index_t* vnk = t->model->nk;
    const index_t* vnkf = t->model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]),
                                         (vnk[2]*vnkf[2])};
    pp_bubble_momenta_handler_fft_red(&t->cuda_descr, L, t->GF_fft_plus, t->GF_fft_conj_plus,
                            t->n_orb, t->n_orbff, t->n_spin, t->nktot, t->my_nk, t->c2fmap,
                            dims, t->mi_to_ofrom, t->mi_to_oto, t->mi_to_R, t->tu_data->unique_mi_l,
                            t->tu_data->unique_mi_r, t->tu_data->unique_mi_l_sz, NULL);
}

#else // USE_CUDA

void merge_gpu_cpu_loop( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu) {
    (void)t; (void)L; (void)tmpgpu; }
void merge_gpu_cpu_loop_cfrg_step1( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu ) {
    (void)t; (void)L; (void)tmpgpu; }
void merge_gpu_cpu_loop_cfrg_step2( const tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu ) {
    (void)t; (void)L; (void)tmpgpu; }
void S_ph_loop_static_simple_gpu( tu_prop_t* t, complex128_t* L ) { (void)t; (void)L; }
void S_pp_loop_static_simple_gpu( tu_prop_t* t, complex128_t* L ) { (void)t; (void)L; }
void S_ph_loop_static_red_gpu( tu_prop_t* t, complex128_t* L ) { (void)t; (void)L; }
void S_pp_loop_static_red_gpu( tu_prop_t* t, complex128_t* L ) { (void)t; (void)L; }
void allocate_GPU_memory( tu_prop_t* t ) { (void)t; }

#endif // USE_CUDA
