/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "vertex.hpp"
#include "projection_handler.h"

diverge_model_t* Vertex_get_model( void* V_ ) { return static_cast<Vertex*>(V_)->model; }
const tu_data_t* Vertex_get_tu_data( void* V_ ) { return static_cast<Vertex*>(V_)->tu_data; }
index_t Vertex_get_n_orb( void* V_ ) { return static_cast<Vertex*>(V_)->n_orb; }
index_t Vertex_get_n_spin( void* V_ ) { return static_cast<Vertex*>(V_)->n_spin; }
index_t Vertex_get_n_orbff( void* V_ ) { return static_cast<Vertex*>(V_)->n_orbff; }
index_t Vertex_get_nk( void* V_ ) { return static_cast<Vertex*>(V_)->nk; }
index_t Vertex_get_nkf( void* V_ ) { return static_cast<Vertex*>(V_)->nkf; }
index_t Vertex_get_my_nk( void* V_ ) { return static_cast<Vertex*>(V_)->my_nk; }
index_t Vertex_get_my_nk_off( void* V_ ) { return static_cast<Vertex*>(V_)->my_nk_off; }
const symmstruct_t* Vertex_get_symm( void* V_ ) { return static_cast<Vertex*>(V_)->symm; }
bool Vertex_get_P_flow( void* V_ ) { return static_cast<Vertex*>(V_)->P_flow; }
bool Vertex_get_C_flow( void* V_ ) { return static_cast<Vertex*>(V_)->C_flow; }
bool Vertex_get_D_flow( void* V_ ) { return static_cast<Vertex*>(V_)->D_flow; }
complex128_t* Vertex_get_product_helper( void* V_ ) { return static_cast<Vertex*>(V_)->product_helper; }
complex128_t* Vertex_get_bubble_helper( void* V_ ) { return static_cast<Vertex*>(V_)->bubble_helper; }

static inline double max_abs(const complex128_t* start, index_t length) {
    return Map<const VecXcd>(start, length).lpNorm<Eigen::Infinity>();
}

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
    SU2{model->SU2 > 0},
    full_vert_size{POW2(n_orbff)*my_nk*POW4(n_spin)},
    full_bubble_size{full_vert_size},
    selfenergy_size{self ? POW2(n_orb*n_spin)*nkf*nk : 0},
    symm{tu_data->symm},
    P_flow{P},
    C_flow{C || (!SU2 && D)},
    D_flow{D || (!SU2 && C)},
    use_self{self}
{
    owns = true;
    index_t memsize = 0;
    Pch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));
    pp_bubble_int = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));
    memsize += 2*2*full_vert_size;
    Cch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t)); 
    memsize += 2*full_vert_size;
    Dch = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t)); 
    memsize += 2*full_vert_size;
    if(use_self) {
        self_en = (complex128_t*)calloc(selfenergy_size, sizeof(complex128_t));
        memsize += 2*selfenergy_size;
        index_t nk_local = nk;
        index_t bubble_nk_local = (tu_data->use_cuda_mpi_loop && tu_data->use_reduced_loop) ? nk : nk_local;
        bubble_helper = (complex128_t*)calloc(POW2(n_orbff)*bubble_nk_local*POW4(n_spin), sizeof(complex128_t));
        product_helper = (complex128_t*)calloc(POW2(n_orbff)*nk_local*POW4(n_spin), sizeof(complex128_t));
        projection_helper = (complex128_t*)calloc(POW2(n_orbff)*my_nk*POW4(n_spin), sizeof(complex128_t));
        memsize += 2*POW2(n_orbff)*nk_local*POW4(n_spin);
        memsize += POW2(n_orbff)*my_nk*POW4(n_spin);
    }else{
        index_t nk_local = my_nk > symm->my_nk_in_bz ? my_nk : symm->my_nk_in_bz;
        index_t bubble_nk_local = (tu_data->use_cuda_mpi_loop && tu_data->use_reduced_loop) ? nk : nk_local;
        bubble_helper = (complex128_t*)calloc(POW2(n_orbff)*bubble_nk_local*POW4(n_spin), sizeof(complex128_t));
        memsize += POW2(n_orbff)*nk_local*POW4(n_spin);
        nk_local = my_nk > proj->nr ? my_nk : proj->nr;
        product_helper = (complex128_t*)calloc(POW2(n_orbff)*nk_local*POW4(n_spin), sizeof(complex128_t));
        memsize += POW2(n_orbff)*nk_local*POW4(n_spin);
        projection_helper = (complex128_t*)calloc(POW2(n_orbff)*my_nk*POW4(n_spin), sizeof(complex128_t));
        memsize += POW2(n_orbff)*my_nk*POW4(n_spin);
    }
    if(C_flow || D_flow) {
        ph_bubble_int = (complex128_t*)calloc(full_vert_size, sizeof(complex128_t));
        memsize += 2*full_vert_size;
    }
    initialize_GPU_memory(proj, loop);
    if (model->ffill != NULL) {
        init_U_from_fullV(model,proj);
    }else{
        init_U(model -> internals -> enforce_exchange);
        if(!(P_flow && C_flow && D_flow)) {
            mpi_wrn_printf("Not all channels included - please ensure a suitable choice of formfactors\n")
            bool dummy1, dummy2;
            if(D_flow) {
                dummy1 = C_flow; dummy2 = P_flow;
                C_flow = P_flow = true;
                mpi_vrb_printf("Interactions only in D \n");
                D_projection(Dch, this,proj);
                if (C_flow) ZERO_OUT(Cch,full_vert_size);
                if (P_flow) ZERO_OUT(Pch,full_vert_size);
                C_flow = dummy1; P_flow = dummy2;
            }else if(C_flow) {
                dummy1 = D_flow; dummy2 = P_flow;
                P_flow = D_flow = true;
                mpi_vrb_printf("Interactions only in C \n");
                C_projection(Cch, this,proj);
                if (P_flow) ZERO_OUT(Pch,full_vert_size);
                D_flow = dummy1; P_flow = dummy2;
            }else{
                dummy1 = D_flow; dummy2 = C_flow;
                C_flow = D_flow = true;
                mpi_vrb_printf("Interactions only in P \n");
                P_projection(Pch, this,proj);
                D_flow = dummy1; C_flow = dummy2;
            }
        }
    }
    if(!P_flow){
        free(Pch); Pch = nullptr;
        free(pp_bubble_int); pp_bubble_int = nullptr;
        memsize -= 2*2*full_vert_size;
    }
    if(!C_flow){free(Cch); Cch = nullptr; memsize -= 2*full_vert_size;}
    if(!D_flow){free(Dch); Dch = nullptr; memsize -= 2*full_vert_size;}
    
    mpi_vrb_printf("TU euler requires %.1f GB\n", memsize * sizeof(complex128_t) / POW3(1024.))
    mpi_vrb_printf("TU vertex size: %.1f GB\n",  (double)full_vert_size *sizeof(complex128_t) / POW3(1024.));
}

void Vertex::init_U(bool enforce_exchange) {
    complex128_t* helper_crossing = NULL;
    complex128_t* helper = (complex128_t*)calloc(POW2(n_orb)*nk*POW4(n_spin), sizeof(complex128_t));
    if(enforce_exchange) {
        mpi_vrb_printf("enforcing exchange \n");
        helper_crossing = (complex128_t*)calloc(POW2(n_orb)*nk*POW4(n_spin), sizeof(complex128_t));
    }
    if(model->vfill(model,'P',helper)) {
        if(!enforce_exchange) {
            #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
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
            #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
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
    if(!enforce_exchange) {
        if(model->vfill(model,'C',helper)) {
            #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
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
                Cch[IDX5(q, s43, o3, s21, o1,n_spin*n_spin, n_orbff, n_spin*n_spin, n_orbff)] = val;
            }
        }
        if(model->vfill(model,'D',helper)) {
            #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
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

        if(!model->vfill(model,'D',helper_crossing))
            {memset((void*)helper_crossing,0,sizeof(complex128_t)*POW4(n_spin)*POW2(n_orb)*nk);}
        if(!model->vfill(model,'C',helper))
            {memset((void*)helper,0,sizeof(complex128_t)*POW4(n_spin)*POW2(n_orb)*nk);}
        #pragma omp parallel for schedule(static) collapse(4) num_threads(diverge_omp_num_threads())
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

Vertex::Vertex(const Vertex& other)
:
    model{other.model},tu_data{other.tu_data},n_orbff{other.n_orbff},
    n_orb{other.n_orb}, n_bonds{other.n_bonds},n_spin{other.n_spin},
    nk{other.nk}, nkf{other.nkf}, my_nk{other.my_nk}, my_nk_off{other.my_nk_off},
    SU2{other.SU2}, full_vert_size{other.full_vert_size},
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

    if (P_flow) memcpy(Pch,other.get_P_ptr_cst(),sizeof*Pch * this->full_vert_size);
    if (C_flow) memcpy(Cch,other.get_C_ptr_cst(),sizeof*Cch * this->full_vert_size);
    if (D_flow) memcpy(Dch,other.get_D_ptr_cst(),sizeof*Dch * this->full_vert_size);
    if (P_flow) memcpy(pp_bubble_int,other.get_PP_bubble_cst(),sizeof*pp_bubble_int * this->full_bubble_size);
    if (C_flow||D_flow) memcpy(ph_bubble_int,other.get_PH_bubble_cst(),
            sizeof*ph_bubble_int * this->full_bubble_size);
    if (use_self) memcpy(self_en,other.get_E_ptr_cst(),sizeof*self_en * this->selfenergy_size);
    owns = false;
    for(int i = 0; i < 32; ++i)
    for(int j = 0; j < 3; ++j) {
        GPU_gemm_buffer[i][j] = other.GPU_gemm_buffer[i][j];
    }
}


Vertex& Vertex::operator=(const Vertex& other) {
    if (P_flow) memcpy(Pch,other.get_P_ptr_cst(),sizeof*Pch * this->full_vert_size);
    if (C_flow) memcpy(Cch,other.get_C_ptr_cst(),sizeof*Cch * this->full_vert_size);
    if (D_flow) memcpy(Dch,other.get_D_ptr_cst(),sizeof*Dch * this->full_vert_size);

    if (P_flow) memcpy(pp_bubble_int,other.get_PP_bubble_cst(),sizeof*pp_bubble_int * this->full_bubble_size);
    if (C_flow||D_flow) memcpy(ph_bubble_int,other.get_PH_bubble_cst(),
            sizeof*ph_bubble_int * this->full_bubble_size);
    if (use_self) memcpy(self_en,other.get_E_ptr_cst(),sizeof*self_en * this->selfenergy_size);

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
            cudaFree(loop->cuda_descr.buf_GG[i]);
            cudaFree(loop->cuda_descr.buf_res[i]);
            cudaFree(loop->cuda_descr.fft_workspace[i]);
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
        available_memory[i] *= tu_data->percent_mem_GPU_blocked;

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

        proj_memsize[0] = ensure_align(sizeof(cuDoubleComplex)*proj->nr*my_nk); // prefacs
        proj_memsize[1] = ensure_align(sizeof(index_t)*proj->PC_CP_t_padded.size()); // rs_to
        proj_memsize[2] = ensure_align(sizeof(index_t)*proj->PC_CP_f.size()); // rs_from
        proj_memsize[3] = ensure_align(sizeof(index_t)*proj->PC_CP_mom.size()); // mom
        proj_memsize[4] = ensure_align(sizeof(cuDoubleComplex)*POW2(n_orbff*POW2(n_spin))*proj->nr); // buf_ondev
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

        index_t prop_memsize[8] = {0};
        index_t prop_memoffset[8] = {0};
        index_t prop_memoffset_managed[8] = {0};
        index_t prop_memtype[8] = {0};
        const index_t _GF_size = sizeof(cuDoubleComplex)*n_orb*n_spin*nktot*n_orb*n_spin;
        cuda_prop_descr_t* cloop = &(loop->cuda_descr);

        index_t my_nk_tot = (tu_data->use_reduced_loop && tu_data->use_cuda_mpi_loop) ? nk : my_nk;
        prop_memsize[0] = ensure_align(sizeof(index_t)*n_orbff*3); // mi_to_R
        prop_memsize[1] = ensure_align(sizeof(index_t)*my_nk_tot); // c_to_f_q
        prop_memsize[2] = ensure_align(sizeof(index_t)*3); // nkdev
        prop_memsize[3] = ensure_align(_GF_size); // GF_fft
        prop_memsize[4] = ensure_align(_GF_size); // GF_fft_conj
        prop_memsize[5] = ensure_align(sizeof(cuDoubleComplex)*nktot); // buf_GG
        prop_memsize[6] = ensure_align(sizeof(cuDoubleComplex)*my_nk_tot); // buf_res 0
        prop_memsize[7] = ensure_align(sizeof(cuDoubleComplex)*loop->cuda_descr.worksize_fft[i]); // fft_workspace
        index_t all_propmem = 0;
        for (index_t mem = 0; mem < 8; ++mem) {
            all_propmem += prop_memsize[mem];
        }

        if(all_propmem > (index_t)(available_memory[i])) {
            mpi_vrb_printf("Fallback to shuffeling algorithm \n");
            prop_memsize[3] = ensure_align(sizeof(cuDoubleComplex)*nktot*2); // GF_fft
            prop_memsize[4] = ensure_align(sizeof(cuDoubleComplex)*nktot*2); // GF_fft_conj
            cloop->prop_preload_full_gf = false;
            all_propmem = 0;
            for (index_t mem = 0; mem < 8; ++mem) {
                all_propmem += prop_memsize[mem];
            }
            if(all_propmem > (index_t)(available_memory[i])) {
                mpi_wrn_printf("Unified Memory usage required - massively slows down code!\n");
            }
        }

        for (index_t mem = 0; mem < 8; ++mem) {
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
        cloop->buf_GG[i] = (cuDoubleComplex*)((prop_memtype[5] == 0) ? GPU_memory_pool[i] + prop_memoffset[5]
                                                                        : GPU_memory_pool_managed[i] + prop_memoffset_managed[5]);
        cloop->buf_res[i] = (cuDoubleComplex*)((prop_memtype[6] == 0) ? GPU_memory_pool[i] + prop_memoffset[6]
                                                                         : GPU_memory_pool_managed[i] + prop_memoffset_managed[6]);
        cloop->fft_workspace[i] = (cuDoubleComplex*)((prop_memtype[7] == 0) ? GPU_memory_pool[i] + prop_memoffset[7]
                                                                               : GPU_memory_pool_managed[i] + prop_memoffset_managed[7]);

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
    #pragma omp parallel num_threads(diverge_omp_num_threads())
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
    #pragma omp parallel num_threads(diverge_omp_num_threads())
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
    #pragma omp parallel num_threads(diverge_omp_num_threads())
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
