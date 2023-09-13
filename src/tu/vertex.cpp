/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "vertex.hpp"

Vertex::Vertex(diverge_model_t* modelin, Projection* proj, tu_loop_t* loop, 
                bool P, bool C, bool D, bool self)
:
    model{modelin},
    tu_data{(tu_data_t*)(model -> internals-> tu_data)},
    n_orbff{tu_data->n_orbff},
    n_orb{model->n_orb},
    n_bonds{tu_data-> n_bonds},
    n_spin{model->n_spin},
    nk{tu_data->nk},
    nkf{tu_data->nkf},
    my_nk{tu_data->my_nk},
    my_nk_off{tu_data->my_nk_off},
    n_frq{tu_data->n_frq},
    SU2{model->SU2 > 0},
    full_vert_size{POW2(n_orbff)*my_nk*POW4(n_spin)*n_frq},
    full_bubble_size{full_vert_size},
    selfenergy_size{self ? POW2(n_orb*n_spin)*n_frq*nkf*nk : 0},
    symm{tu_data->symm},
    P_flow{P},
    C_flow{C || D},
    D_flow{D || (!SU2 && C)},
    use_self{self}
{
    owns = true;
    index_t memsize = 0;
    if(P_flow){
        Pch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));
        pp_bubble_int = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));
        memsize += 2*2*full_vert_size;
    }
    if(C_flow){Cch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t)); memsize += 2*full_vert_size;}
    if(D_flow){Dch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t)); memsize += 2*full_vert_size;}
    if(use_self) {
        self_en = (complex128_t*)calloc(selfenergy_size, sizeof(complex128_t));
        memsize += 2*selfenergy_size;
        init_selfenergy();
        index_t n = nk;
        bubble_helper = (complex128_t*)calloc(POW2(n_orbff)*n*POW4(n_spin), sizeof(complex128_t));
        product_helper = (complex128_t*)calloc(POW2(n_orbff)*n*POW4(n_spin), sizeof(complex128_t));
        projection_helper = (complex128_t*)calloc(POW2(n_orbff)*my_nk*POW4(n_spin), sizeof(complex128_t));
        memsize += 2*POW2(n_orbff)*n*POW4(n_spin);
        memsize += POW2(n_orbff)*my_nk*POW4(n_spin);
    }else{
        index_t n = my_nk > symm->my_nk_in_bz ? my_nk : symm->my_nk_in_bz;
        bubble_helper = (complex128_t*)calloc(POW2(n_orbff)*n*POW4(n_spin), sizeof(complex128_t));
        memsize += POW2(n_orbff)*n*POW4(n_spin);
        n = my_nk > proj->get_nr() ? my_nk : proj->get_nr();
        product_helper = (complex128_t*)calloc(POW2(n_orbff)*n*POW4(n_spin), sizeof(complex128_t));
        memsize += POW2(n_orbff)*n*POW4(n_spin);
        projection_helper = (complex128_t*)calloc(POW2(n_orbff)*my_nk*POW4(n_spin), sizeof(complex128_t));
        memsize += POW2(n_orbff)*my_nk*POW4(n_spin);
    }
    if(C_flow || D_flow) {
        ph_bubble_int = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));
        memsize += 2*full_vert_size;
    }
    mpi_vrb_printf("TU euler requires %.1f GB\n", memsize * sizeof(complex128_t) / POW3(1024.))
    initialize_GPU_memory(proj, loop);
    if (model->ffill != NULL) {
        init_U_from_fullV(model,proj);
    }else{
        init_U(model -> internals -> enforce_exchange);
    }
    mpi_vrb_printf("TU vertex size: %.1f GB\n", (double)memsize/ POW3(1024.));
}

void Vertex::init_U(bool enforce_exchange) {
    complex128_t* helper_crossing = NULL;
    complex128_t* helper = (complex128_t*)calloc(POW2(n_orb)*nk*POW4(n_spin), sizeof(complex128_t));
    if(enforce_exchange){
        mpi_vrb_printf("enforcing exchange \n");
        helper_crossing = (complex128_t*)calloc(POW2(n_orb)*nk*POW4(n_spin), sizeof(complex128_t));
    }
    if(P_flow) {
        if(!enforce_exchange) {
            model->vfill(model,'P',helper);
            #pragma omp parallel for schedule(static) collapse(4)
            for (index_t q=0; q< my_nk; ++q)
            for (index_t s4=0; s4<n_spin; ++s4)
            for (index_t s3=0; s3<n_spin; ++s3)
            for (index_t o3=0; o3<n_orb; ++o3)
            for (index_t s2=0; s2<n_spin; ++s2)
            for (index_t s1=0; s1<n_spin; ++s1)
            for (index_t o1=0; o1<n_orb; ++o1) {
                index_t s21 = IDX2(s2, s1, n_spin),
                        s43 = IDX2(s4, s3, n_spin);
                index_t s12 = IDX2(s1, s2, n_spin),
                        s34 = IDX2(s3, s4, n_spin);
                complex128_t val = helper[IDX5(symm->idx_ibz_in_fullmesh[q+my_nk_off],
                                s12, o1, s34, o3, n_spin*n_spin, n_orb, n_spin*n_spin, n_orb)];
                Pch[IDX5(q, s43, o3, s21, o1,n_spin*n_spin, n_orbff, n_spin*n_spin, n_orbff)] = val;
            }
        } else {
            model->vfill(model,'P',helper);
            #pragma omp parallel for schedule(static) collapse(4)
            for (index_t q=0; q < my_nk; ++q)
            for (index_t s4=0; s4<n_spin; ++s4)
            for (index_t s3=0; s3<n_spin; ++s3)
            for (index_t o3=0; o3<n_orb; ++o3)
            for (index_t s2=0; s2<n_spin; ++s2)
            for (index_t s1=0; s1<n_spin; ++s1)
            for (index_t o1=0; o1<n_orb; ++o1) {
                index_t full_q = symm->idx_ibz_in_fullmesh[q+my_nk_off];
                index_t r1234 = IDX7(q, s4, s3, o3, s2, s1, o1,n_spin,n_spin, n_orbff, n_spin,n_spin, n_orbff),
                        i1234 = IDX7(full_q, s1, s2, o1, s3, s4, o3,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                        i2134 = IDX7(full_q, s2, s1, o1, s3, s4, o3,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                        i1243 = IDX7(full_q, s1, s2, o1, s4, s3, o3,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                        i2143 = IDX7(full_q, s2, s1, o1, s4, s3, o3,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb);
                Pch[r1234] = 0.5 * (helper[i1234] - helper[i1243] +
                                    helper[i2143] - helper[i2134]);
            }
        }
    }
    if(!enforce_exchange || !(D_flow && C_flow))
    {
        if(C_flow) {
            model->vfill(model,'C',helper);
            #pragma omp parallel for schedule(static) collapse(4)
            for (index_t q=0; q< my_nk; ++q)
            for (index_t s4=0; s4<n_spin; ++s4)
            for (index_t s3=0; s3<n_spin; ++s3)
            for (index_t o3=0; o3<n_orb; ++o3)
            for (index_t s2=0; s2<n_spin; ++s2)
            for (index_t s1=0; s1<n_spin; ++s1)
            for (index_t o1=0; o1<n_orb; ++o1) {
                index_t s21 = IDX2(s2, s1, n_spin),
                        s43 = IDX2(s2, s4, n_spin);
                index_t s12 = IDX2(s1, s2, n_spin),
                        s34 = IDX2(s3, s4, n_spin);
                complex128_t val = helper[IDX5(symm->idx_ibz_in_fullmesh[q+my_nk_off],
                                s12, o1, s34, o3, n_spin*n_spin, n_orb, n_spin*n_spin, n_orb)];
                Cch[IDX5(q, s43, o3, s21, o1,n_spin*n_spin, n_orbff, n_spin*n_spin, n_orbff)] = val;
            }
        }
        if(D_flow) {
            model->vfill(model,'D',helper);
            #pragma omp parallel for schedule(static) collapse(4)
            for (index_t q=0; q< my_nk; ++q)
            for (index_t s4=0; s4<n_spin; ++s4)
            for (index_t s3=0; s3<n_spin; ++s3)
            for (index_t o3=0; o3<n_orb; ++o3)
            for (index_t s2=0; s2<n_spin; ++s2)
            for (index_t s1=0; s1<n_spin; ++s1)
            for (index_t o1=0; o1<n_orb; ++o1) {
                index_t s21 = IDX2(s2, s1, n_spin),
                        s43 = IDX2(s4, s3, n_spin);
                index_t s12 = IDX2(s1, s2, n_spin),
                        s34 = IDX2(s3, s4, n_spin);
                complex128_t val = helper[IDX5(symm->idx_ibz_in_fullmesh[q+my_nk_off], 
                                s12, o1, s34, o3, n_spin*n_spin, n_orb, n_spin*n_spin, n_orb)];
                Dch[IDX5(q, s43, o3, s21, o1,n_spin*n_spin, n_orbff, n_spin*n_spin, n_orbff)] = val;
            }
        }
    }else{
        auto to_minus_q = [](index_t q, index_t* nk) {
            index_t q_x = q/(nk[2]*nk[1]);
            q -= q_x*nk[2]*nk[1];
            index_t q_y = q / nk[2];
            index_t q_z = q - q_y * nk[2];
            q_x = (-q_x+nk[0])%nk[0];
            q_y = (-q_y+nk[1])%nk[1];
            q_z = (-q_z+nk[2])%nk[2];
            return q_z + nk[2] * (q_y + nk[1] * q_x);
        };

        model->vfill(model,'D',helper_crossing);
        model->vfill(model,'C',helper);
        #pragma omp parallel for schedule(static) collapse(4)
        for (index_t q=0; q< my_nk; ++q)
        for (index_t s4=0; s4<n_spin; ++s4)
        for (index_t s3=0; s3<n_spin; ++s3)
        for (index_t o3=0; o3<n_orb; ++o3)
        for (index_t s2=0; s2<n_spin; ++s2)
        for (index_t s1=0; s1<n_spin; ++s1)
        for (index_t o1=0; o1<n_orb; ++o1) {
            index_t full_q = symm->idx_ibz_in_fullmesh[q+my_nk_off];
            index_t mfull_q = to_minus_q(full_q, model->nk);
            index_t rc1234 = IDX7(q, s2, s3, o3, s4, s1, o1,n_spin,n_spin, n_orbff, n_spin,n_spin, n_orbff),
                    rd1234 = IDX7(q, s2, s4, o3, s3, s1, o1,n_spin,n_spin, n_orbff, n_spin,n_spin, n_orbff),
                    id1234 = IDX7(full_q, s1, s3, o1, s4, s2, o3,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                    id1243 = IDX7(full_q, s1, s4, o1, s3, s2, o3,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                    id2134 = IDX7(mfull_q, s2, s3, o3, s4, s1, o1,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                    id2143 = IDX7(mfull_q, s2, s4, o3, s3, s1, o1,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                    ic1234 = IDX7(full_q, s1, s4, o1, s3, s2, o3,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                    ic1243 = IDX7(full_q, s1, s3, o1, s4, s2, o3,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                    ic2134 = IDX7(mfull_q, s2, s4, o3, s3, s1, o1,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb),
                    ic2143 = IDX7(mfull_q, s2, s3, o3, s4, s1, o1,n_spin,n_spin, n_orb, n_spin,n_spin, n_orb);
            complex128_t val_D = 0.5 * (helper_crossing[id1234] + helper_crossing[id2143] -
                                        helper[ic2134] - helper[ic1243]);
            complex128_t val_C = 0.5 * (helper[ic1234] + helper[ic2143] -
                                        helper_crossing[id2134] - helper_crossing[id1243]);
            Dch[rd1234] = val_D;
            Cch[rc1234] = val_C;
        }
    }
    free(helper);
    free(helper_crossing);
}


void Vertex::init_selfenergy() {
    const index_t hamsize = POW2(n_orb*n_spin)*nkf*nk;
    mfill(self_en,hamsize*n_frq,0.0);
}

void Vertex::unset_selfenergy() {
    const complex128_t* ham = (complex128_t*)model->internals->ham;
    const index_t hamsize = POW2(n_orb*n_spin)*nkf*nk;
    #pragma omp parallel for collapse(2)
    for(index_t b_idx = 0; b_idx < n_frq;++b_idx)
    for(index_t ii = 0; ii<hamsize;++ii) {
            self_en[ii+b_idx*hamsize] -= ham[ii];
    }
}

Vertex::Vertex(const Vertex& other)
:
    model{other.model},tu_data{other.tu_data},n_orbff{other.n_orbff},
    n_orb{other.n_orb}, n_bonds{other.n_bonds},n_spin{other.n_spin},
    nk{other.nk}, nkf{other.nkf}, my_nk{other.my_nk}, my_nk_off{other.my_nk_off},
    n_frq{other.n_frq}, SU2{other.SU2}, full_vert_size{other.full_vert_size},
    full_bubble_size{other.full_bubble_size},
    selfenergy_size{other.selfenergy_size}, symm{other.symm},P_flow{other.P_flow},
    C_flow{other.C_flow},D_flow{other.D_flow},use_self{other.use_self}
{
    Pch = nullptr;
    Cch = nullptr;
    Dch = nullptr;
    bubble_helper = other.bubble_helper;
    product_helper = other.product_helper;
    projection_helper = other.projection_helper;
    if(P_flow)Pch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));
    if(C_flow)Cch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));
    if(D_flow)Dch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));

    if(use_self)self_en = (complex128_t*)calloc(selfenergy_size, sizeof(complex128_t));
    if(P_flow)pp_bubble_int = (complex128_t*)calloc(full_bubble_size, sizeof(complex128_t));
    if(C_flow || D_flow)ph_bubble_int = (complex128_t*)calloc(full_bubble_size, sizeof(complex128_t));

    if(P_flow)mcopy(Pch,other.get_P_ptr_cst(),this->full_vert_size);
    if(C_flow)mcopy(Cch,other.get_C_ptr_cst(),this->full_vert_size);
    if(D_flow)mcopy(Dch,other.get_D_ptr_cst(),this->full_vert_size);
    if(P_flow)mcopy(pp_bubble_int,other.get_PP_bubble_cst(),this->full_bubble_size);
    if(C_flow || D_flow)mcopy(ph_bubble_int,other.get_PH_bubble_cst(),this->full_bubble_size);
    if(use_self)mcopy(self_en,other.get_E_ptr_cst(),this->selfenergy_size);
    owns = false;
    for(int i = 0; i < 32; ++i)
    for(int j = 0; j < 3; ++j) {
        GPU_gemm_buffer[i][j] = other.GPU_gemm_buffer[i][j];
    }
}


Vertex& Vertex::operator=(const Vertex& other) {
    if(P_flow)mcopy(Pch,other.get_P_ptr_cst(),this->full_vert_size);
    if(C_flow)mcopy(Cch,other.get_C_ptr_cst(),this->full_vert_size);
    if(D_flow)mcopy(Dch,other.get_D_ptr_cst(),this->full_vert_size);

    if(P_flow)mcopy(pp_bubble_int,other.get_PP_bubble_cst(),this->full_bubble_size);
    if(C_flow || D_flow)mcopy(ph_bubble_int,other.get_PH_bubble_cst(),this->full_bubble_size);
    if(use_self)mcopy(self_en,other.get_E_ptr_cst(),this->selfenergy_size);

    return *this;
}
#ifndef CUDA_MEM_ALIGN
#define CUDA_MEM_ALIGN 256
#endif
#ifdef USE_CUDA
static index_t ensure_align(index_t numToRound) {

    index_t remainder = numToRound % 256;
    if (remainder == 0)
        return numToRound;

    return numToRound + 256 - remainder;
}
#endif

void Vertex::initialize_GPU_memory(Projection* proj, tu_loop_t* loop) {
    (void)proj;(void)loop;
#ifdef USE_CUDA
    if(!tu_has_gpu_setup || !loop->cuda_descr.prop_has_gpu_setup || !proj->proj_descr.proj_has_gpu_setup) {
        mpi_err_printf("CUDA initailized wrongly. TUFRG will fail!\n");
        return ;
    }
    if(loop->cuda_descr.prop_memory_set_local) {
        mpi_err_printf("Memory for loop already allocated, free it first!\n");
        loop->cuda_descr.prop_memory_set_local = false;
        for(int i = 0; i<tu_num_of_devices;++i) {
            cudaFree(loop->cuda_descr.mi_to_R[i]);
            cudaFree(loop->cuda_descr.c_to_f_q[i]);
            cudaFree(loop->cuda_descr.nkdev[i]);
            cudaFree(loop->cuda_descr.GF_fft[i]);
            cudaFree(loop->cuda_descr.GF_fft_conj[i]);
            cudaFree(loop->cuda_descr.buf_GG[i][0]);
            cudaFree(loop->cuda_descr.buf_GG[i][1]);
            cudaFree(loop->cuda_descr.buf_res[i][0]);
            cudaFree(loop->cuda_descr.buf_res[i][1]);
            cudaFree(loop->cuda_descr.fft_workspace[i][0]);
            cudaFree(loop->cuda_descr.fft_workspace[i][1]);
        }
    }
    has_gpu_memory = true;
    index_t nktot = nk * nkf;
    size_t available_memory[256] = {0};
    for(int i = 0; i<tu_num_of_devices;++i) {
        size_t total;
        CUDA_CHECK( cudaSetDevice( i ) );
        CUDA_CHECK( cudaMemGetInfo( &available_memory[i], &total ) );
        mpi_vrb_printf("TU GPU %d has %.1f GB available\n", i, available_memory[i] / POW3(1024.));
        available_memory[i] *= 0.95;

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

        proj_memsize[0] = ensure_align(sizeof(cuDoubleComplex)*proj->get_nr()*my_nk); // prefacs
        proj_memsize[1] = ensure_align(sizeof(index_t)*proj->get_CP_PC_map_TO_padded().size()); // rs_to
        proj_memsize[2] = ensure_align(sizeof(index_t)*proj->get_CP_PC_map_FROM().size()); // rs_from
        proj_memsize[3] = ensure_align(sizeof(index_t)*proj->get_CP_PC_prefac_map().size()); // mom
        proj_memsize[4] = ensure_align(sizeof(cuDoubleComplex)*POW2(n_orbff*POW2(n_spin))*proj->get_nr()); // buf_ondev
        proj_memsize[5] = ensure_align(sizeof(cuDoubleComplex)*POW2(n_orbff*POW2(n_spin))); // buf_res0
        proj_memsize[6] = ensure_align(sizeof(cuDoubleComplex)*POW2(n_orbff*POW2(n_spin))); // buf_res1

        for (index_t mem = 0; mem < 7; ++mem) {
            proj_memtype[mem] = (proj_memsize[mem] < (index_t)available_memory[i] - proj_mem) ? 0:1;
            proj_mem += (proj_memtype[mem] == 0) ? proj_memsize[mem] : 0;
            proj_mem_managed += (proj_memtype[mem] == 1) ? proj_memsize[mem] : 0;
            if(mem > 0) {
                proj_memoffset[mem] = (proj_memtype[mem-1] == 0) 
                                ? proj_memsize[mem-1] + proj_memoffset[mem-1] : proj_memoffset[mem-1];
                proj_memoffset_managed[mem] = (proj_memtype[mem-1] == 1) 
                                ? proj_memsize[mem-1] + proj_memoffset_managed[mem-1] : proj_memoffset_managed[mem-1];
            }
        }

        index_t prop_memsize[11] = {0};
        index_t prop_memoffset[11] = {0};
        index_t prop_memoffset_managed[11] = {0};
        index_t prop_memtype[11] = {0};
        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;

        prop_memsize[0] = ensure_align(sizeof(index_t)*n_orbff*3); // mi_to_R
        prop_memsize[1] = ensure_align(sizeof(index_t)*my_nk); // c_to_f_q
        prop_memsize[2] = ensure_align(sizeof(index_t)*3); // nkdev
        prop_memsize[3] = ensure_align(_GF_size); // GF_fft
        prop_memsize[4] = ensure_align(_GF_size); // GF_fft_conj
        prop_memsize[5] = ensure_align(sizeof(cuDoubleComplex)*nktot); // buf_GG 0
        prop_memsize[6] = ensure_align(sizeof(cuDoubleComplex)*nktot); // buf_GG 1
        prop_memsize[7] = ensure_align(sizeof(cuDoubleComplex)*my_nk); // buf_res 0
        prop_memsize[8] = ensure_align(sizeof(cuDoubleComplex)*my_nk); // buf_res 1
        prop_memsize[9] = ensure_align(sizeof(cuDoubleComplex)*loop->cuda_descr.worksize_fft[i]); // fft_workspace 0
        prop_memsize[10] = ensure_align(sizeof(cuDoubleComplex)*loop->cuda_descr.worksize_fft[i]); // fft_workspace 1

        for (index_t mem = 0; mem < 11; ++mem) {
            prop_memtype[mem] = (prop_memsize[mem] < (index_t)available_memory[i] - prop_mem) ? 0:1;
            prop_mem += (prop_memtype[mem] == 0) ? prop_memsize[mem] : 0;
            prop_mem_managed += (prop_memtype[mem] == 1) ? prop_memsize[mem] : 0;
            if(mem > 0) {
                prop_memoffset[mem] = (prop_memtype[mem-1] == 0) ? prop_memsize[mem-1] + prop_memoffset[mem-1] : prop_memoffset[mem-1];
                prop_memoffset_managed[mem] = (prop_memtype[mem-1] == 1) ?
                    prop_memsize[mem-1] + prop_memoffset_managed[mem-1] : prop_memoffset_managed[mem-1];
            }
        }

        index_t gemm_buffer_size = sizeof(cuDoubleComplex)*POW2(n_orbff*POW2(n_spin));
        index_t gemm_memsize[3] = {0};
        index_t gemm_memoffset[3] = {0};
        index_t gemm_memoffset_managed[3] = {0};
        index_t gemm_memtype[3] = {0};

        gemm_memsize[0] = ensure_align(gemm_buffer_size);
        gemm_memsize[1] = ensure_align(gemm_buffer_size);
        gemm_memsize[2] = ensure_align(gemm_buffer_size);

        for (index_t mem = 0; mem < 3; ++mem) {
            gemm_memtype[mem] = (gemm_memsize[mem] < (index_t)available_memory[i] - gemm_mem) ? 0:1;
            gemm_mem += (gemm_memtype[mem] == 0) ? gemm_memsize[mem] : 0;
            gemm_mem_managed += (gemm_memtype[mem] == 1) ? gemm_memsize[mem] : 0;
            if(mem > 0) {
                gemm_memoffset[mem] = (gemm_memtype[mem-1] == 0) ?
                    gemm_memsize[mem-1] + gemm_memoffset[mem-1] : gemm_memoffset[mem-1];
                gemm_memoffset_managed[mem] = (gemm_memtype[mem-1] == 1) ?
                    gemm_memsize[mem-1] + gemm_memoffset_managed[mem-1] : gemm_memoffset_managed[mem-1];
            }
        }
        index_t memory_GPU = std::max( std::max( gemm_mem, prop_mem ) , proj_mem  );
        index_t memory_managed = std::max( std::max( gemm_mem_managed, prop_mem_managed ) , proj_mem_managed  );
        // XXX needed to set the values to sth always >= 0
        memory_GPU = MAX( memory_GPU, 128*16 );
        memory_managed = MAX( memory_managed, 128*16 );
        mpi_vrb_printf("TU GPU pool with %.1f GB on-device and %.1f GB managed\n", memory_GPU / POW3(1024.), memory_managed / POW3(1024.));
        CUDA_CHECK( cudaMalloc(&GPU_memory_pool[i], memory_GPU) );
        CUDA_CHECK( cudaMallocManaged(&GPU_memory_pool_managed[i], memory_managed) );

        GPU_gemm_buffer[i][0] = (gemm_memtype[0] == 0) ? GPU_memory_pool[i] + gemm_memoffset[0]
                                                       : GPU_memory_pool_managed[i] + gemm_memoffset_managed[0];
        GPU_gemm_buffer[i][1] = (gemm_memtype[1] == 0) ? GPU_memory_pool[i] + gemm_memoffset[1]
                                                       : GPU_memory_pool_managed[i] + gemm_memoffset_managed[1];
        GPU_gemm_buffer[i][2] = (gemm_memtype[2] == 0) ? GPU_memory_pool[i] + gemm_memoffset[2]
                                                       : GPU_memory_pool_managed[i] + gemm_memoffset_managed[2];

        cuda_prop_descr_t* cloop = &(loop->cuda_descr);
        cloop->mi_to_R[i] = (index_t*)((prop_memtype[0] == 0) ? GPU_memory_pool[i] + prop_memoffset[0]
                                                              : GPU_memory_pool_managed[i] + prop_memoffset_managed[0]);
        cloop->c_to_f_q[i] = (index_t*)((prop_memtype[1] == 0) ? GPU_memory_pool[i] + prop_memoffset[1]
                                                               : GPU_memory_pool_managed[i] + prop_memoffset_managed[1]);
        cloop->nkdev[i] = (index_t*)((prop_memtype[2] == 0) ? GPU_memory_pool[i] + prop_memoffset[2]
                                                            : GPU_memory_pool_managed[i] + prop_memoffset_managed[2]);
        cloop->GF_fft[i] = (cuDoubleComplex*)((prop_memtype[3] == 0) ? GPU_memory_pool[i] + prop_memoffset[3]
                                                                     : GPU_memory_pool_managed[i] + prop_memoffset_managed[3]);
        cloop->GF_fft_conj[i] = (cuDoubleComplex*)((prop_memtype[4] == 0) ? GPU_memory_pool[i] + prop_memoffset[4]
                                                                          : GPU_memory_pool_managed[i] + prop_memoffset_managed[4]);
        cloop->buf_GG[i][0] = (cuDoubleComplex*)((prop_memtype[5] == 0) ? GPU_memory_pool[i] + prop_memoffset[5]
                                                                        : GPU_memory_pool_managed[i] + prop_memoffset_managed[5]);
        cloop->buf_GG[i][1] = (cuDoubleComplex*)((prop_memtype[6] == 0) ? GPU_memory_pool[i] + prop_memoffset[6]
                                                                        : GPU_memory_pool_managed[i] + prop_memoffset_managed[6]);
        cloop->buf_res[i][0] = (cuDoubleComplex*)((prop_memtype[7] == 0) ? GPU_memory_pool[i] + prop_memoffset[7]
                                                                         : GPU_memory_pool_managed[i] + prop_memoffset_managed[7]);
        cloop->buf_res[i][1] = (cuDoubleComplex*)((prop_memtype[8] == 0) ? GPU_memory_pool[i] + prop_memoffset[8]
                                                                         : GPU_memory_pool_managed[i] + prop_memoffset_managed[8]);
        cloop->fft_workspace[i][0] = (cuDoubleComplex*)((prop_memtype[9] == 0) ? GPU_memory_pool[i] + prop_memoffset[9]
                                                                               : GPU_memory_pool_managed[i] + prop_memoffset_managed[9]);
        cloop->fft_workspace[i][1] = (cuDoubleComplex*)((prop_memtype[10] == 0) ? GPU_memory_pool[i] + prop_memoffset[10]
                                                                                : GPU_memory_pool_managed[i] + prop_memoffset_managed[10]);

        cuda_proj_descr_t* cproj = &(proj->proj_descr);
        cproj->prefacs[i] = (cuDoubleComplex*)((proj_memtype[0] == 0) ? GPU_memory_pool[i] + proj_memoffset[0]
                                                                      : GPU_memory_pool_managed[i] + proj_memoffset_managed[0]);
        cproj->rs_to[i] = (index_t*)((proj_memtype[1] == 0) ? GPU_memory_pool[i] + proj_memoffset[1]
                                                            : GPU_memory_pool_managed[i] + proj_memoffset_managed[1]);
        cproj->rs_from[i] = (index_t*)((proj_memtype[2] == 0) ? GPU_memory_pool[i] + proj_memoffset[2]
                                                              : GPU_memory_pool_managed[i] + proj_memoffset_managed[2]);
        cproj->mom[i] = (index_t*)((proj_memtype[3] == 0) ? GPU_memory_pool[i] + proj_memoffset[3]
                                                          : GPU_memory_pool_managed[i] + proj_memoffset_managed[3]);
        cproj->buf_ondev[i] = (cuDoubleComplex*)((proj_memtype[4] == 0) ? GPU_memory_pool[i] + proj_memoffset[4]
                                                                        : GPU_memory_pool_managed[i] + proj_memoffset_managed[4]);
        cproj->buf_res0[i] = (cuDoubleComplex*)((proj_memtype[5] == 0) ? GPU_memory_pool[i] + proj_memoffset[5]
                                                                       : GPU_memory_pool_managed[i] + proj_memoffset_managed[5]);
        cproj->buf_res1[i] = (cuDoubleComplex*)((proj_memtype[6] == 0) ? GPU_memory_pool[i] + proj_memoffset[6]
                                                                       : GPU_memory_pool_managed[i] + proj_memoffset_managed[6]);
    }
    loop->cuda_descr.prop_memory_set_vertex = true;
    CUDA_CHECK( cudaSetDevice( 0 ) );
#endif
}


Vertex::~Vertex()
{
    free(Cch);
    free(Pch);
    free(Dch);
    free(pp_bubble_int);
    free(ph_bubble_int);
    free(self_en);
    if(owns) {
        free(bubble_helper);
        free(product_helper);
        free(projection_helper);
        projection_helper = nullptr;
        product_helper = nullptr;
        bubble_helper = nullptr;
    }
#ifdef USE_CUDA
    if(has_gpu_memory) {
        for(int i = 0; i<tu_num_of_devices;++i) {
            CUDA_CHECK( cudaFree(GPU_memory_pool[i]) );
            CUDA_CHECK( cudaFree(GPU_memory_pool_managed[i]) );
        }
    }
#endif
}

void add(Vertex &result,
            const Vertex &inp, const double factor) {
    #pragma omp parallel
    {
        if(result.P_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                result.Pch[i] += factor*(inp.Pch[i]);
            }
        }
        if(result.C_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                result.Cch[i] += factor*(inp.Cch[i]);
            }
        }
        if(result.D_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                result.Dch[i] += factor*(inp.Dch[i]);
            }
        }
        if(result.P_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i<inp.full_bubble_size;++i)
            {
                result.pp_bubble_int[i] += factor*(inp.pp_bubble_int[i]);
            }
        }
        if(result.C_flow || result.D_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i< inp.full_bubble_size;++i) {
                result.ph_bubble_int[i] += factor*(inp.ph_bubble_int[i]);
            }
        }
        #pragma omp for schedule(static)
        for(index_t i = 0; i< inp.selfenergy_size;++i) {
            result.self_en[i] += factor*(inp.self_en[i]);
        }
    }
}


void set(Vertex &f,const Vertex &initial,const Vertex &inp, const double factor) {
    #pragma omp parallel
    {
        if(f.P_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                f.Pch[i]
                    = initial.Pch[i] + factor*(inp.Pch[i]);
            }
        }
        if(f.C_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                f.Cch[i]
                    = initial.Cch[i] + factor*(inp.Cch[i]);
            }
        }
        if(f.D_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                f.Dch[i]
                    = initial.Dch[i] + factor*(inp.Dch[i]);
            }
        }
        if(f.P_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i<inp.full_bubble_size;++i) {
                f.pp_bubble_int[i]
                    = initial.pp_bubble_int[i] +  factor*(inp.pp_bubble_int[i]);
            }
        }
        if(f.D_flow || f.C_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i< inp.full_bubble_size;++i) {
                f.ph_bubble_int[i]
                    = initial.ph_bubble_int[i] + factor*(inp.ph_bubble_int[i]);
            }
        }
        #pragma omp for schedule(static)
        for(index_t i = 0; i< inp.selfenergy_size;++i) {
            f.self_en[i]
                = initial.self_en[i] + factor*(inp.self_en[i]);
        }
    }
}


void multiply(Vertex &f,const Vertex &inp, const double factor) {
    #pragma omp parallel
    {
        if(f.P_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                f.Pch[i] = factor*(inp.Pch[i]);
            }
        }
        if(f.C_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                f.Cch[i] =  factor*(inp.Cch[i]);
            }
        }
        if(f.D_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i < inp.full_vert_size; ++i) {
                f.Dch[i] = factor*(inp.Dch[i]);
            }
        }

        if(f.P_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i<inp.full_bubble_size;++i) {
                f.pp_bubble_int[i] = factor*(inp.pp_bubble_int[i]);
            }
        }
        if(f.C_flow || f.D_flow) {
            #pragma omp for schedule(static)
            for(index_t i = 0; i< inp.full_bubble_size;++i)
            {
                f.ph_bubble_int[i] = factor*(inp.ph_bubble_int[i]);
            }
        }
        #pragma omp for schedule(static)
        for(index_t i = 0; i< inp.selfenergy_size;++i)
        {
            f.self_en[i] = factor*(inp.self_en[i]);
        }
    }
}


double infinity_norm(const Vertex &inp)
{
    double max_coeff = -0.1;
    double max_trial;
    if(inp.P_flow) {
        max_trial = max_abs(inp.Pch,inp.full_vert_size);
        max_trial = diverge_mpi_max(&max_trial);
        if(max_trial>max_coeff){max_coeff = max_trial;}
    }
    if(inp.C_flow) {
        max_trial = max_abs(inp.Cch,inp.full_vert_size);
        max_trial = diverge_mpi_max(&max_trial);
        if(max_trial>max_coeff){max_coeff = max_trial;}
    }
    if(inp.D_flow) {
        max_trial = max_abs(inp.Dch,inp.full_vert_size);
        max_trial = diverge_mpi_max(&max_trial);
        if(max_trial>max_coeff){max_coeff = max_trial;}
    }

    if(inp.P_flow) {
        max_trial =  max_abs(inp.pp_bubble_int,inp.full_bubble_size);
        max_trial = diverge_mpi_max(&max_trial);
        if(max_trial>max_coeff){max_coeff = max_trial;}
    }
    if(inp.C_flow || inp.D_flow) {
        max_trial =  max_abs(inp.ph_bubble_int,inp.full_bubble_size);
        max_trial = diverge_mpi_max(&max_trial);
        if(max_trial>max_coeff){max_coeff = max_trial;}
    }
    if(inp.selfenergy_size>0) {
        max_trial =  max_abs(inp.self_en,inp.selfenergy_size);
        max_trial = diverge_mpi_max(&max_trial);
        if(max_trial>max_coeff){max_coeff = max_trial;}
    }
    return max_coeff;
}

vector<double> max(const Vertex &inp)
{
    double max_trial;
    vector<double> maximal_vals = {0.,0.,0.,0.,0.};
    if(inp.P_flow) {
        max_trial = max_abs(inp.Pch,inp.full_vert_size);
        maximal_vals[0] = diverge_mpi_max(&max_trial);
    }
    if(inp.C_flow) {
        max_trial = max_abs(inp.Cch,inp.full_vert_size);
        maximal_vals[1] = diverge_mpi_max(&max_trial);
    }
    if(inp.D_flow) {
        max_trial = max_abs(inp.Dch,inp.full_vert_size);
        maximal_vals[2] = diverge_mpi_max(&max_trial);
    }
    if(inp.P_flow) {
        max_trial =  max_abs(inp.pp_bubble_int,inp.full_bubble_size);
        maximal_vals[3] = diverge_mpi_max(&max_trial);
    }
    if(inp.C_flow || inp.D_flow) {
        max_trial =  max_abs(inp.ph_bubble_int,inp.full_bubble_size);
        maximal_vals[4] = diverge_mpi_max(&max_trial);
    }
    return maximal_vals;
}
