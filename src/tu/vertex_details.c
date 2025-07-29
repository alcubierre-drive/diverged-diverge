#include "vertex_details.h"
#include "propagator_details.h"

static inline complex128_t ff( const double* k, const double* r ) {
    return cexp( -I*(k[0]*r[0]+k[1]*r[1]+k[2]*r[2]) );
}

static inline void fill_Rvec( const diverge_model_t* model, double* Rvec ) {
    const tu_formfactor_t* tu_ff = model->tu_ff;
    for (index_t i=0; i<model->n_tu_ff; ++i)
    for (index_t d=0; d<3; ++d)
        Rvec[IDX2(i,d,3)] = tu_ff[i].R[0] * model->lattice[0][d]
                          + tu_ff[i].R[1] * model->lattice[1][d]
                          + tu_ff[i].R[2] * model->lattice[2][d];
}

#ifdef USE_CUDA
#ifndef CUDA_MEM_ALIGN
#define CUDA_MEM_ALIGN 256
#endif
static inline index_t ensure_align(index_t numToRound) {
    index_t remainder = numToRound % CUDA_MEM_ALIGN;
    if (remainder == 0) return numToRound;
    return numToRound + CUDA_MEM_ALIGN - remainder;
}
#endif

#define MAX3( A, B, C ) MAX(MAX(A,B),C)

void tu_vertex_initialize_GPU_memory( tu_vertex_t* V, tu_projection_t* proj, tu_prop_t* loop ) {
    (void)V; (void)proj; (void)loop;
#ifdef USE_CUDA
    if (!tu_gpu_info().tu_has_gpu_setup || !loop->cuda_descr.prop_has_gpu_setup ||
        !proj->proj_descr.proj_has_gpu_setup) {
        mpi_err_printf("CUDA initailized wrongly. TUFRG will fail!\n");
        return;
    }
    if (loop->cuda_descr.prop_memory_set_local) {
        mpi_err_printf("Memory for loop already allocated, free it first!\n");
        loop->cuda_descr.prop_memory_set_local = false;
        for (int i=0; i<tu_gpu_info().tu_num_of_devices; ++i) {
            CUDA_CHECK( cudaFree(loop->cuda_descr.mi_to_R[i]) );
            CUDA_CHECK( cudaFree(loop->cuda_descr.c_to_f_q[i]) );
            CUDA_CHECK( cudaFree(loop->cuda_descr.nkdev[i]) );
            CUDA_CHECK( cudaFree(loop->cuda_descr.GF_fft[i]) );
            CUDA_CHECK( cudaFree(loop->cuda_descr.GF_fft_conj[i]) );
            CUDA_CHECK( cudaFree(loop->cuda_descr.buf_GG[i]) );
            CUDA_CHECK( cudaFree(loop->cuda_descr.buf_res[i]) );
            CUDA_CHECK( cudaFree(loop->cuda_descr.fft_workspace[i]) );
        }
    }
    V->has_gpu_memory = true;
    index_t nktot = V->nk * V->nkf;
    size_t available_memory[256] = {0};
    for(int i=0; i<tu_gpu_info().tu_num_of_devices; ++i) {
        size_t total;
        CUDA_CHECK( cudaSetDevice( i ) );
        CUDA_CHECK( cudaMemGetInfo( &available_memory[i], &total ) );
        mpi_vrb_printf("TU GPU %d has %.1f GB available\n", i, available_memory[i] / POW3(1024.));
        available_memory[i] *= V->tu_data->percent_mem_GPU_blocked;

        index_t prop_mem = 0;
        index_t prop_mem_managed = 0;
        index_t proj_mem = 0;
        index_t proj_mem_managed = 0;
        index_t gemm_mem = 0;
        index_t gemm_mem_managed = 0;

        index_t proj_memsize[7] = {0};
        index_t proj_memoffset[7] = {0};
        index_t proj_memoffset_managed[7] = {0};
        index_t proj_memtype[7] = {0};

        proj_memsize[0] = ensure_align(sizeof(cuDoubleComplex)*proj->nr*V->my_nk); // prefacs
        proj_memsize[1] = ensure_align(sizeof(index_t)*proj->PC_CP_to_padded_sz*2); // rs_to
        proj_memsize[2] = ensure_align(sizeof(index_t)*proj->PC_CP_to_padded_sz*2); // rs_from
        proj_memsize[3] = ensure_align(sizeof(index_t)*proj->PC_CP_to_padded_sz*2); // mom
        proj_memsize[4] = ensure_align(sizeof(cuDoubleComplex)*POW2(V->n_orbff*POW2(V->n_spin))*proj->nr); // buf_ondev
        proj_memsize[5] = ensure_align(sizeof(cuDoubleComplex)*POW2(V->n_orbff*POW2(V->n_spin))); // buf_res0
        proj_memsize[6] = ensure_align(sizeof(cuDoubleComplex)*POW2(V->n_orbff*POW2(V->n_spin))); // buf_res1

        for (index_t mem=0; mem<7; ++mem) {
            proj_memtype[mem] = (proj_memsize[mem] < (index_t)available_memory[i] - proj_mem) ? 0:1;
            proj_mem += (proj_memtype[mem] == 0) ? proj_memsize[mem] : 0;
            proj_mem_managed += (proj_memtype[mem] == 1) ? proj_memsize[mem] : 0;
            if (mem>0) {
                proj_memoffset[mem] = (proj_memtype[mem-1] == 0) 
                                ? proj_memsize[mem-1] + proj_memoffset[mem-1]
                                : proj_memoffset[mem-1];
                proj_memoffset_managed[mem] = (proj_memtype[mem-1] == 1) 
                                ? proj_memsize[mem-1] + proj_memoffset_managed[mem-1]
                                : proj_memoffset_managed[mem-1];
            }
        }

        index_t prop_memsize[8] = {0};
        index_t prop_memoffset[8] = {0};
        index_t prop_memoffset_managed[8] = {0};
        index_t prop_memtype[8] = {0};
        const index_t _GF_size = sizeof(cuDoubleComplex)*V->n_orb*V->n_spin*nktot*V->n_orb*V->n_spin;
        cuda_prop_descr_t* cloop = &(loop->cuda_descr);

        index_t my_nk_tot = (V->tu_data->use_reduced_loop && V->tu_data->use_cuda_mpi_loop) ? V->nk : V->my_nk;
        prop_memsize[0] = ensure_align(sizeof(index_t)*V->n_orbff*3); // mi_to_R
        prop_memsize[1] = ensure_align(sizeof(index_t)*my_nk_tot); // c_to_f_q
        prop_memsize[2] = ensure_align(sizeof(index_t)*3); // nkdev
        prop_memsize[3] = ensure_align(_GF_size); // GF_fft
        prop_memsize[4] = ensure_align(_GF_size); // GF_fft_conj
        prop_memsize[5] = ensure_align(sizeof(cuDoubleComplex)*nktot); // buf_GG
        prop_memsize[6] = ensure_align(sizeof(cuDoubleComplex)*my_nk_tot); // buf_res 0
        prop_memsize[7] = ensure_align(sizeof(cuDoubleComplex)*loop->cuda_descr.worksize_fft[i]); // fft_workspace
        index_t all_propmem = 0;
        for (index_t mem = 0; mem < 8; ++mem)
            all_propmem += prop_memsize[mem];

        if (all_propmem > (index_t)(available_memory[i])) {
            mpi_vrb_printf("Fallback to shuffeling algorithm\n");
            prop_memsize[3] = ensure_align(sizeof(cuDoubleComplex)*nktot*2); // GF_fft
            prop_memsize[4] = ensure_align(sizeof(cuDoubleComplex)*nktot*2); // GF_fft_conj
            cloop->prop_preload_full_gf = false;
            all_propmem = 0;
            for (index_t mem=0; mem<8; ++mem)
                all_propmem += prop_memsize[mem];
            if (all_propmem > (index_t)(available_memory[i]))
                mpi_wrn_printf("Unified Memory usage required => massive slowdown\n" );
        }

        for (index_t mem=0; mem<8; ++mem) {
            prop_memtype[mem] = (prop_memsize[mem] < (index_t)available_memory[i] - prop_mem) ? 0:1;
            prop_mem += (prop_memtype[mem] == 0) ? prop_memsize[mem] : 0;
            prop_mem_managed += (prop_memtype[mem] == 1) ? prop_memsize[mem] : 0;
            if (mem > 0) {
                prop_memoffset[mem] = (prop_memtype[mem-1] == 0)
                    ? prop_memsize[mem-1] + prop_memoffset[mem-1]
                    : prop_memoffset[mem-1];
                prop_memoffset_managed[mem] = (prop_memtype[mem-1] == 1)
                    ? prop_memsize[mem-1] + prop_memoffset_managed[mem-1]
                    : prop_memoffset_managed[mem-1];
            }
        }

        index_t gemm_buffer_size = sizeof(cuDoubleComplex)*POW2(V->n_orbff*POW2(V->n_spin));
        index_t gemm_memsize[3] = {0};
        index_t gemm_memoffset[3] = {0};
        index_t gemm_memoffset_managed[3] = {0};
        index_t gemm_memtype[3] = {0};

        gemm_memsize[0] = ensure_align(gemm_buffer_size);
        gemm_memsize[1] = ensure_align(gemm_buffer_size);
        gemm_memsize[2] = ensure_align(gemm_buffer_size);

        for (index_t mem=0; mem<3; ++mem) {
            gemm_memtype[mem] = (gemm_memsize[mem] < (index_t)available_memory[i] - gemm_mem) ? 0:1;
            gemm_mem += (gemm_memtype[mem] == 0) ? gemm_memsize[mem] : 0;
            gemm_mem_managed += (gemm_memtype[mem] == 1) ? gemm_memsize[mem] : 0;
            if (mem > 0) {
                gemm_memoffset[mem] = (gemm_memtype[mem-1] == 0) ?
                    gemm_memsize[mem-1] + gemm_memoffset[mem-1] : gemm_memoffset[mem-1];
                gemm_memoffset_managed[mem] = (gemm_memtype[mem-1] == 1) ?
                    gemm_memsize[mem-1] + gemm_memoffset_managed[mem-1] : gemm_memoffset_managed[mem-1];
            }
        }
        index_t memory_GPU = MAX3( gemm_mem, prop_mem, proj_mem );
        index_t memory_managed = MAX3( gemm_mem_managed, prop_mem_managed, proj_mem_managed );
        // XXX needed to set the values to sth always >= 0
        memory_GPU = MAX( memory_GPU, 128*16 );
        memory_managed = MAX( memory_managed, 128*16 );
        mpi_vrb_printf("TU GPU pool with %.1f GB on-device and %.1f GB managed\n",
                (double)memory_GPU/POW3(1024.), (double)memory_managed/POW3(1024.));
        CUDA_CHECK( cudaMalloc((void**)&V->GPU_memory_pool[i], memory_GPU) );
        CUDA_CHECK( cudaMallocManaged((void**)&V->GPU_memory_pool_managed[i], memory_managed, cudaMemAttachGlobal) );

        V->GPU_gemm_buffer[i][0] = (gemm_memtype[0] == 0) ? V->GPU_memory_pool[i] + gemm_memoffset[0]
                                                          : V->GPU_memory_pool_managed[i] + gemm_memoffset_managed[0];
        V->GPU_gemm_buffer[i][1] = (gemm_memtype[1] == 0) ? V->GPU_memory_pool[i] + gemm_memoffset[1]
                                                          : V->GPU_memory_pool_managed[i] + gemm_memoffset_managed[1];
        V->GPU_gemm_buffer[i][2] = (gemm_memtype[2] == 0) ? V->GPU_memory_pool[i] + gemm_memoffset[2]
                                                          : V->GPU_memory_pool_managed[i] + gemm_memoffset_managed[2];

        cloop->mi_to_R[i] = (index_t*)((prop_memtype[0] == 0) ? V->GPU_memory_pool[i] + prop_memoffset[0]
                                                              : V->GPU_memory_pool_managed[i] + prop_memoffset_managed[0]);
        cloop->c_to_f_q[i] = (index_t*)((prop_memtype[1] == 0) ? V->GPU_memory_pool[i] + prop_memoffset[1]
                                                               : V->GPU_memory_pool_managed[i] + prop_memoffset_managed[1]);
        cloop->nkdev[i] = (index_t*)((prop_memtype[2] == 0) ? V->GPU_memory_pool[i] + prop_memoffset[2]
                                                            : V->GPU_memory_pool_managed[i] + prop_memoffset_managed[2]);
        cloop->GF_fft[i] = (cuDoubleComplex*)((prop_memtype[3] == 0) ? V->GPU_memory_pool[i] + prop_memoffset[3]
                                                                     : V->GPU_memory_pool_managed[i] + prop_memoffset_managed[3]);
        cloop->GF_fft_conj[i] = (cuDoubleComplex*)((prop_memtype[4] == 0) ? V->GPU_memory_pool[i] + prop_memoffset[4]
                                                                          : V->GPU_memory_pool_managed[i] + prop_memoffset_managed[4]);
        cloop->buf_GG[i] = (cuDoubleComplex*)((prop_memtype[5] == 0) ? V->GPU_memory_pool[i] + prop_memoffset[5]
                                                                        : V->GPU_memory_pool_managed[i] + prop_memoffset_managed[5]);
        cloop->buf_res[i] = (cuDoubleComplex*)((prop_memtype[6] == 0) ? V->GPU_memory_pool[i] + prop_memoffset[6]
                                                                         : V->GPU_memory_pool_managed[i] + prop_memoffset_managed[6]);
        cloop->fft_workspace[i] = (cuDoubleComplex*)((prop_memtype[7] == 0) ? V->GPU_memory_pool[i] + prop_memoffset[7]
                                                                               : V->GPU_memory_pool_managed[i] + prop_memoffset_managed[7]);

        cuda_proj_descr_t* cproj = &(proj->proj_descr);
        cproj->prefacs[i] = (cuDoubleComplex*)((proj_memtype[0] == 0) ? V->GPU_memory_pool[i] + proj_memoffset[0]
                                                                      : V->GPU_memory_pool_managed[i] + proj_memoffset_managed[0]);
        cproj->rs_to[i] = (index_t*)((proj_memtype[1] == 0) ? V->GPU_memory_pool[i] + proj_memoffset[1]
                                                            : V->GPU_memory_pool_managed[i] + proj_memoffset_managed[1]);
        cproj->rs_from[i] = (index_t*)((proj_memtype[2] == 0) ? V->GPU_memory_pool[i] + proj_memoffset[2]
                                                              : V->GPU_memory_pool_managed[i] + proj_memoffset_managed[2]);
        cproj->mom[i] = (index_t*)((proj_memtype[3] == 0) ? V->GPU_memory_pool[i] + proj_memoffset[3]
                                                          : V->GPU_memory_pool_managed[i] + proj_memoffset_managed[3]);
        cproj->buf_ondev[i] = (cuDoubleComplex*)((proj_memtype[4] == 0) ? V->GPU_memory_pool[i] + proj_memoffset[4]
                                                                        : V->GPU_memory_pool_managed[i] + proj_memoffset_managed[4]);
        cproj->buf_res0[i] = (cuDoubleComplex*)((proj_memtype[5] == 0) ? V->GPU_memory_pool[i] + proj_memoffset[5]
                                                                       : V->GPU_memory_pool_managed[i] + proj_memoffset_managed[5]);
        cproj->buf_res1[i] = (cuDoubleComplex*)((proj_memtype[6] == 0) ? V->GPU_memory_pool[i] + proj_memoffset[6]
                                                                       : V->GPU_memory_pool_managed[i] + proj_memoffset_managed[6]);
    }
    loop->cuda_descr.prop_memory_set_vertex = true;
    CUDA_CHECK( cudaSetDevice( 0 ) );
#endif
}


void tu_vertex_init_U_from_fullV( tu_vertex_t* V, diverge_model_t* model, tu_projection_t* proj ) {
    mpi_wrn_printf("initialization from full\n")
    double* Rvec = calloc(model->n_tu_ff, sizeof(double)*3);
    double* kmesh = model->internals->kmesh;

    const tu_formfactor_t* tu_ff = model->tu_ff;
    const index_t* bond_sizes = V->tu_data->bond_sizes;
    const index_t* bond_offsets = V->tu_data->bond_offsets;
    const index_t* ob_to_orbff = V->tu_data->ob_to_orbff;
    fill_Rvec(model,Rvec);

    complex128_t* buf = calloc(POW4(V->n_orb*V->n_spin), sizeof(complex128_t));
    const index_t* nkd = model->nk;
    if (V->D_flow) {
        ZERO_OUT(V->Dch, V->full_vert_size);
        // D-channel q = k_1 - k_3 = k_4 - k_2
        for (index_t q=0; q<V->my_nk; ++q)
        for (index_t k1=0; k1<V->nk; ++k1)
        for (index_t k4=0; k4<V->nk; ++k4) {
            index_t k3 = k1m2(k1,V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off],nkd);
            index_t k2 = k1m2(k4,V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off],nkd);
            (*(model->ffill))( model, k1, k2, k3, buf );
            #pragma omp parallel for collapse(6) schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t s4=0; s4<V->n_spin; ++s4)
            for (index_t s3=0; s3<V->n_spin; ++s3)
            for (index_t s2=0; s2<V->n_spin; ++s2)
            for (index_t s1=0; s1<V->n_spin; ++s1)
            for (index_t o1=0; o1<V->n_orb; ++o1)
            for (index_t o4=0; o4<V->n_orb; ++o4)
            for (index_t b1=0; b1<bond_sizes[o1]; ++b1)
            for (index_t b4=0; b4<bond_sizes[o4]; ++b4) {
                index_t o2 = tu_ff[bond_offsets[o4]+b4].oto,
                        o3 = tu_ff[bond_offsets[o1]+b1].oto;
                index_t ob1 = ob_to_orbff[b1+V->n_bonds*o1],
                        ob4 = ob_to_orbff[b4+V->n_bonds*o4];
                index_t id_buf = IDX8(s1,o1,s2,o2,s3,o3,s4,o4,V->n_orb,V->n_spin,V->n_orb,V->n_spin,V->n_orb,V->n_spin,V->n_orb);
                index_t iddch = IDX7(q,s2,s4,ob4,s3,s1,ob1,V->n_spin,V->n_spin,V->n_orbff,V->n_spin,V->n_spin,V->n_orbff);
                V->Dch[iddch] += buf[id_buf] * ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1))
                                        * conj(ff(kmesh+3*k4,Rvec+3*(bond_offsets[o4]+b4)))/(double)POW2(V->nk);

            }
        }
    }

    if (V->C_flow) {
        ZERO_OUT(V->Cch, V->full_vert_size);
        C_projection(V->Cch, V, proj);
        for (index_t i=0; i<V->full_vert_size; ++i) V->Cch[i] *= -1.0;
        // C-channel q = k_1 - k_4 = k_3 - k_2
        for (index_t q=0; q<V->my_nk; ++q)
        for (index_t k1=0; k1<V->nk; ++k1)
        for (index_t k3=0; k3<V->nk; ++k3) {
            index_t k2 = k1m2(k3,V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off],nkd);
            (*(model->ffill))( model, k1, k2, k3, buf );
            #pragma omp parallel for collapse(6) schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t s4=0; s4<V->n_spin; ++s4)
            for (index_t s3=0; s3<V->n_spin; ++s3)
            for (index_t s2=0; s2<V->n_spin; ++s2)
            for (index_t s1=0; s1<V->n_spin; ++s1)
            for (index_t o1=0; o1<V->n_orb; ++o1)
            for (index_t o3=0; o3<V->n_orb; ++o3)
            for (index_t b1=0; b1<bond_sizes[o1]; ++b1)
            for (index_t b3=0; b3<bond_sizes[o3]; ++b3) {
                index_t o2 = tu_ff[bond_offsets[o3]+b3].oto,
                        o4 = tu_ff[bond_offsets[o1]+b1].oto;
                index_t ob1 = ob_to_orbff[b1+V->n_bonds*o1],
                        ob3 = ob_to_orbff[b3+V->n_bonds*o3];
                index_t id_buf = IDX8(s1,o1,s2,o2,s3,o3,s4,o4,V->n_orb,V->n_spin,V->n_orb,V->n_spin,V->n_orb,V->n_spin,V->n_orb);
                index_t iddch = IDX7(q,s2,s3,ob3,s4,s1,ob1,V->n_spin,V->n_spin,V->n_orbff,V->n_spin,V->n_spin,V->n_orbff);
                V->Cch[iddch] += buf[id_buf] * ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1))
                                        * conj(ff(kmesh+3*k3,Rvec+3*(bond_offsets[o3]+b3)))/(double)POW2(V->nk);

            }
        }
    }

    if (V->P_flow) {
        ZERO_OUT(V->Pch, V->full_vert_size);
        P_projection(V->Pch,V,proj);
        for (index_t i=0; i<V->full_vert_size; ++i) V->Pch[i] *= -1.0;
        // P-channel q = k_1 + k_2 = k_3 + k_4
        for (index_t q=0; q<V->my_nk; ++q)
        for (index_t k1=0; k1<V->nk; ++k1)
        for (index_t k3=0; k3<V->nk; ++k3) {
            index_t k2 = k1m2(V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off], k1, nkd);
            (*(model->ffill))( model, k1, k2, k3, buf );
            #pragma omp parallel for collapse(6) schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t s4=0; s4<V->n_spin; ++s4)
            for (index_t s3=0; s3<V->n_spin; ++s3)
            for (index_t s2=0; s2<V->n_spin; ++s2)
            for (index_t s1=0; s1<V->n_spin; ++s1)
            for (index_t o1=0; o1<V->n_orb; ++o1)
            for (index_t o3=0; o3<V->n_orb; ++o3)
            for (index_t b1=0; b1<bond_sizes[o1]; ++b1)
            for (index_t b3=0; b3<bond_sizes[o3]; ++b3) {
                index_t o2 = tu_ff[bond_offsets[o1]+b1].oto,
                        o4 = tu_ff[bond_offsets[o3]+b3].oto;
                index_t ob1 = ob_to_orbff[b1+V->n_bonds*o1],
                        ob3 = ob_to_orbff[b3+V->n_bonds*o3];
                index_t id_buf = IDX8(s1,o1,s2,o2,s3,o3,s4,o4,V->n_orb,V->n_spin,V->n_orb,V->n_spin,V->n_orb,V->n_spin,V->n_orb);
                index_t iddch = IDX7(q,s4,s3,ob3,s2,s1,ob1,V->n_spin,V->n_spin,V->n_orbff,V->n_spin,V->n_spin,V->n_orbff);
                V->Pch[iddch] += buf[id_buf] * ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1))
                                        * conj(ff(kmesh+3*k3,Rvec+3*(bond_offsets[o3]+b3)))/(double)POW2(V->nk);

            }
        }
    }
    free(Rvec);
    free(buf);
}

void tu_vertex_init_U( tu_vertex_t* V, bool enforce_exchange ) {
    complex128_t* helper_crossing = NULL;
    complex128_t* helper = calloc(POW2(V->n_orb)*V->nk*POW4(V->n_spin), sizeof(complex128_t));
    if (enforce_exchange) {
        mpi_vrb_printf("enforcing exchange\n");
        helper_crossing = calloc(POW2(V->n_orb)*V->nk*POW4(V->n_spin), sizeof(complex128_t));
    }
    if (V->model->vfill(V->model,'P',helper)) {
        if (!enforce_exchange) {
            #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<V->my_nk; ++q)
            for (index_t s4=0; s4<V->n_spin; ++s4)
            for (index_t s3=0; s3<V->n_spin; ++s3)
            for (index_t o3=0; o3<V->n_orb; ++o3)
            for (index_t s2=0; s2<V->n_spin; ++s2)
            for (index_t s1=0; s1<V->n_spin; ++s1)
            for (index_t o1=0; o1<V->n_orb; ++o1) {
                index_t s21 = IDX2(s2, s1, V->n_spin),
                        s43 = IDX2(s4, s3, V->n_spin);
                index_t s12 = IDX2(s1, s2, V->n_spin),
                        s34 = IDX2(s3, s4, V->n_spin);
                complex128_t val = helper[IDX5(V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off],
                                s12, o1, s34, o3, V->n_spin*V->n_spin, V->n_orb, V->n_spin*V->n_spin, V->n_orb)];
                V->Pch[IDX5(q, s43, o3, s21, o1,
                        V->n_spin*V->n_spin, V->n_orbff, V->n_spin*V->n_spin, V->n_orbff)] = val;
            }
        } else {
            #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<V->my_nk; ++q)
            for (index_t s4=0; s4<V->n_spin; ++s4)
            for (index_t s3=0; s3<V->n_spin; ++s3)
            for (index_t o3=0; o3<V->n_orb; ++o3)
            for (index_t s2=0; s2<V->n_spin; ++s2)
            for (index_t s1=0; s1<V->n_spin; ++s1)
            for (index_t o1=0; o1<V->n_orb; ++o1) {
                index_t full_q = V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off];
                index_t r1234 = IDX7(q, s4, s3, o3, s2, s1, o1,
                                V->n_spin,V->n_spin, V->n_orbff, V->n_spin,V->n_spin, V->n_orbff),
                        i1234 = IDX7(full_q, s1, s2, o1, s3, s4, o3,
                                V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                        i2134 = IDX7(full_q, s2, s1, o1, s3, s4, o3,
                                V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                        i1243 = IDX7(full_q, s1, s2, o1, s4, s3, o3,
                                V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                        i2143 = IDX7(full_q, s2, s1, o1, s4, s3, o3,
                                V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb);
                V->Pch[r1234] = 0.5 * (helper[i1234] - helper[i1243] + helper[i2143] - helper[i2134]);
            }
        }
    }
    if (!enforce_exchange) {
        if (V->model->vfill(V->model,'C',helper)) {
            #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<V->my_nk; ++q)
            for (index_t s4=0; s4<V->n_spin; ++s4)
            for (index_t s3=0; s3<V->n_spin; ++s3)
            for (index_t o3=0; o3<V->n_orb; ++o3)
            for (index_t s2=0; s2<V->n_spin; ++s2)
            for (index_t s1=0; s1<V->n_spin; ++s1)
            for (index_t o1=0; o1<V->n_orb; ++o1) {
                index_t s21 = IDX2(s2, s1, V->n_spin),
                        s43 = IDX2(s4, s3, V->n_spin);
                index_t s12 = IDX2(s1, s2, V->n_spin),
                        s34 = IDX2(s3, s4, V->n_spin);
                complex128_t val = helper[IDX5(V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off],
                                s12, o1, s34, o3, V->n_spin*V->n_spin, V->n_orb, V->n_spin*V->n_spin, V->n_orb)];
                V->Cch[IDX5(q, s43, o3, s21, o1,
                        V->n_spin*V->n_spin, V->n_orbff, V->n_spin*V->n_spin, V->n_orbff)] = val;
            }
        }
        if (V->model->vfill(V->model,'D',helper)) {
            #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<V->my_nk; ++q)
            for (index_t s4=0; s4<V->n_spin; ++s4)
            for (index_t s3=0; s3<V->n_spin; ++s3)
            for (index_t o3=0; o3<V->n_orb; ++o3)
            for (index_t s2=0; s2<V->n_spin; ++s2)
            for (index_t s1=0; s1<V->n_spin; ++s1)
            for (index_t o1=0; o1<V->n_orb; ++o1) {
                index_t s21 = IDX2(s2, s1, V->n_spin),
                        s43 = IDX2(s4, s3, V->n_spin);
                index_t s12 = IDX2(s1, s2, V->n_spin),
                        s34 = IDX2(s3, s4, V->n_spin);
                complex128_t val = helper[IDX5(V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off],
                                s12, o1, s34, o3, V->n_spin*V->n_spin, V->n_orb, V->n_spin*V->n_spin, V->n_orb)];
                V->Dch[IDX5(q, s43, o3, s21, o1,
                        V->n_spin*V->n_spin, V->n_orbff, V->n_spin*V->n_spin, V->n_orbff)] = val;
            }
        }
    } else {
        if (!V->model->vfill(V->model,'D',helper_crossing)) {
            ZERO_OUT(helper_crossing, POW4(V->n_spin)*POW2(V->n_orb)*V->nk);
        }
        if (!V->model->vfill(V->model,'C',helper)) {
            ZERO_OUT(helper, POW4(V->n_spin)*POW2(V->n_orb)*V->nk);
        }
        #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
        for (index_t q=0; q<V->my_nk; ++q)
        for (index_t s4=0; s4<V->n_spin; ++s4)
        for (index_t s3=0; s3<V->n_spin; ++s3)
        for (index_t o3=0; o3<V->n_orb; ++o3)
        for (index_t s2=0; s2<V->n_spin; ++s2)
        for (index_t s1=0; s1<V->n_spin; ++s1)
        for (index_t o1=0; o1<V->n_orb; ++o1) {
            index_t full_q = V->symm->idx_ibz_in_fullmesh[q+V->my_nk_off];
            index_t mfull_q = k1m2(0, full_q, V->model->nk);
            index_t rc1234 = IDX7(q, s2, s3, o3, s4, s1, o1,
                                    V->n_spin,V->n_spin, V->n_orbff, V->n_spin,V->n_spin, V->n_orbff),
                    rd1234 = IDX7(q, s2, s4, o3, s3, s1, o1,
                                    V->n_spin,V->n_spin, V->n_orbff, V->n_spin,V->n_spin, V->n_orbff),
                    id1234 = IDX7(full_q, s1, s3, o1, s4, s2, o3,
                                    V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                    id1243 = IDX7(full_q, s1, s4, o1, s3, s2, o3,
                                    V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                    id2134 = IDX7(mfull_q, s2, s3, o3, s4, s1, o1,
                                    V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                    id2143 = IDX7(mfull_q, s2, s4, o3, s3, s1, o1,
                                    V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                    ic1234 = IDX7(full_q, s1, s4, o1, s3, s2, o3,
                                    V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                    ic1243 = IDX7(full_q, s1, s3, o1, s4, s2, o3,
                                    V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                    ic2134 = IDX7(mfull_q, s2, s4, o3, s3, s1, o1,
                                    V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb),
                    ic2143 = IDX7(mfull_q, s2, s3, o3, s4, s1, o1,
                                    V->n_spin,V->n_spin, V->n_orb, V->n_spin,V->n_spin, V->n_orb);
            complex128_t val_D = 0.5 * (helper_crossing[id1234] + helper_crossing[id2143] -
                                        helper[ic2134] - helper[ic1243]);
            complex128_t val_C = 0.5 * (helper[ic1234] + helper[ic2143] -
                                        helper_crossing[id2134] - helper_crossing[id1243]);
            V->Dch[rd1234] = val_D;
            V->Cch[rc1234] = val_C;
        }
    }
    free(helper);
    free(helper_crossing);
}

void tu_vertex_project_fullV( const tu_vertex_t* V, const diverge_model_t* model, complex128_t* full_V ) {
    double* Rvec = calloc(model->n_tu_ff,sizeof(double)*3);
    double* kmesh = model->internals->kmesh;
    fill_Rvec(model,Rvec);
    const index_t full_size = POW4(V->n_spin*V->n_orb)*POW3(V->nk);
    const tu_formfactor_t* tu_ff = model->tu_ff;
    const index_t* bond_sizes = V->tu_data->bond_sizes;
    const index_t* bond_offsets = V->tu_data->bond_offsets;
    const index_t* ob_to_orbff = V->tu_data->ob_to_orbff;
    const index_t n_bonds = V->tu_data->n_bonds;

    const index_t* nkd = model->nk;
    ZERO_OUT(full_V, full_size);
    complex128_t* buffer = calloc(V->nk*POW2(V->n_orbff*POW2(V->n_spin)),sizeof(complex128_t));

    index_t no = V->n_orb*V->n_spin;
    // unproject P
    if (V->P_flow) {
        tu_vertex_reco( V, 'P', buffer, NULL );
        #pragma omp parallel for collapse(8) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for (index_t s1=0; s1<V->n_spin; ++s1)
        for (index_t s2=0; s2<V->n_spin; ++s2)
        for (index_t s3=0; s3<V->n_spin; ++s3)
        for (index_t s4=0; s4<V->n_spin; ++s4)
        for (index_t o1=0; o1<V->n_orb; ++o1)
        for (index_t o3=0; o3<V->n_orb; ++o3)
        for (index_t k2=0; k2<V->nk; ++k2)
        for (index_t k1=0; k1<V->nk; ++k1) {
            index_t q_P = k1p2(k1,k2,nkd);

            for (index_t b1=0; b1<bond_sizes[o1]; ++b1)
            for (index_t b3=0; b3<bond_sizes[o3]; ++b3) {
                const index_t so1 = tu_ff[bond_offsets[o1]+b1].ofrom+V->n_orb*s1;
                const index_t so2 = tu_ff[bond_offsets[o1]+b1].oto+V->n_orb*s2;
                const index_t so3 = tu_ff[bond_offsets[o3]+b3].ofrom+V->n_orb*s3;
                const index_t so4 = tu_ff[bond_offsets[o3]+b3].oto+V->n_orb*s4;
                for (index_t k3=0; k3<V->nk; ++k3) {
                    index_t idxV = IDX7(k1,k2,k3,so1,so2,so3,so4, V->nk,V->nk,no,no,no,no);
                    index_t idxP =  q_P+V->nk*(ob_to_orbff[b1+n_bonds*o1]
                                +V->n_orbff*(s1+V->n_spin*(s2+V->n_spin*(
                            ob_to_orbff[b3+n_bonds*o3]+V->n_orbff*(s3+V->n_spin*
                                s4)))));

                    full_V[idxV] += conj(ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1)))*
                                         ff(kmesh+3*k3,Rvec+3*(bond_offsets[o3]+b3))* buffer[idxP];
                }
            }
        }
    }
    // unproject C
    if (V->C_flow) {
        tu_vertex_reco( V, 'C', buffer, NULL );
        #pragma omp parallel for collapse(8) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for (index_t s1=0; s1<V->n_spin; ++s1)
        for (index_t s2=0; s2<V->n_spin; ++s2)
        for (index_t s3=0; s3<V->n_spin; ++s3)
        for (index_t s4=0; s4<V->n_spin; ++s4)
        for (index_t o1=0; o1<V->n_orb; ++o1)
        for (index_t o3=0; o3<V->n_orb; ++o3)
        for (index_t k2=0; k2<V->nk; ++k2)
        for (index_t k1=0; k1<V->nk; ++k1) {
            for (index_t b1=0; b1<bond_sizes[o1]; ++b1)
            for (index_t b3=0; b3<bond_sizes[o3]; ++b3) {
                const index_t so1 = o1+V->n_orb*s1;
                const index_t so2 = tu_ff[bond_offsets[o3]+b3].oto+V->n_orb*s2;
                const index_t so3 = o3+V->n_orb*s3;
                const index_t so4 = tu_ff[bond_offsets[o1]+b1].oto+V->n_orb*s4;

                for (index_t k3=0; k3<V->nk;++k3) {
                    index_t idxV = IDX7(k1,k2,k3,so1,so2,so3,so4, V->nk,V->nk,no,no,no,no);
                    index_t idxC = k1m2(k3,k2,nkd)
                        +V->nk*(ob_to_orbff[b1+n_bonds*o1]
                                +V->n_orbff*(s1+V->n_spin*(s2+V->n_spin*(
                            ob_to_orbff[b3+n_bonds*o3]+V->n_orbff*(s3+V->n_spin*
                                s4)))));

                    full_V[idxV] += conj(ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1)))*
                                         ff(kmesh+3*k3,Rvec+3*(bond_offsets[o3]+b3))* buffer[idxC];
                }
            }
        }
    }
    // unproject D
    if (V->D_flow) {
        tu_vertex_reco( V, 'D', buffer, NULL );
        #pragma omp parallel for collapse(8) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for (index_t s1=0; s1<V->n_spin; ++s1)
        for (index_t s2=0; s2<V->n_spin; ++s2)
        for (index_t s3=0; s3<V->n_spin; ++s3)
        for (index_t s4=0; s4<V->n_spin; ++s4)
        for (index_t o1=0; o1<V->n_orb; ++o1)
        for (index_t o4=0; o4<V->n_orb; ++o4)
        for (index_t k2=0; k2<V->nk; ++k2)
        for (index_t k1=0; k1<V->nk; ++k1) {
            for (index_t b1=0; b1<bond_sizes[o1]; ++b1)
            for (index_t b4=0; b4<bond_sizes[o4]; ++b4) {
                const index_t so1 = o1+V->n_orb*s1;
                const index_t so2 = tu_ff[bond_offsets[o4]+b4].oto+V->n_orb*s2;
                const index_t so3 = tu_ff[bond_offsets[o1]+b1].oto+V->n_orb*s3;
                const index_t so4 = o4+V->n_orb*s4;

                for (index_t k3=0; k3<V->nk; ++k3) {
                    index_t idxV = IDX7(k1,k2,k3,so1,so2,so3,so4, V->nk,V->nk,no,no,no,no);
                    index_t idxD = k1m2(k1,k3,nkd)
                        +V->nk*(ob_to_orbff[b1+n_bonds*o1]
                            +V->n_orbff*(s1+V->n_spin*(s2+V->n_spin*(
                            ob_to_orbff[b4+n_bonds*o4]+V->n_orbff*(s3+V->n_spin*
                                (s4))))));

                    double uidx[3] = {kmesh[3*k1]+kmesh[3*k2]-kmesh[3*k3],
                                      kmesh[3*k1+1]+kmesh[3*k2+1]-kmesh[3*k3+1],
                                      kmesh[3*k1+2]+kmesh[3*k2+2]-kmesh[3*k3+2]};

                    full_V[idxV] += conj(ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1)))*
                                         ff(uidx,Rvec+3*(bond_offsets[o4]+b4))* buffer[idxD];
                }
            }
        }
    }
    free(Rvec);
    free(buffer);
}
