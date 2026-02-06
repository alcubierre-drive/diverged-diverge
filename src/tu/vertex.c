/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "vertex.h"
#include "vertex_details.h"
#include "propagator_details.h"
#include "../misc/batched_gemms.h"
#include "../misc/gemm.h"

tu_vertex_t* tu_vertex_init( diverge_model_t* modelin, tu_projection_t* proj, tu_prop_t* loop,
        bool P/*=true*/, bool C/*=true*/, bool D/*=true*/, bool self/*=false*/ ) {
    tu_vertex_t* V = calloc(1, sizeof*V);
    V->model = modelin;
    V->tu_data = V->model->internals->tu_data;
    V->n_orbff = V->tu_data->n_orbff;
    V->n_orb = V->model->n_orb;
    V->n_bonds = V->tu_data->n_bonds;
    V->n_spin = V->model->n_spin;
    V->nk = V->tu_data->nk;
    V->nkf = V->tu_data->nkf;
    V->my_nk = V->tu_data->my_nk;
    V->my_nk_off = V->tu_data->my_nk_off;
    V->SU2 = V->model->SU2 > 0;
    V->full_vert_size = POW2(V->n_orbff)*V->my_nk*POW4(V->n_spin);
    V->full_bubble_size = V->full_vert_size;
    V->selfenergy_size = self ? POW2(V->n_orb*V->n_spin)*V->nkf*V->nk : 0;
    V->symm = V->tu_data->symm;
    V->P_flow = P;
    V->C_flow = C || (!V->SU2 && D);
    V->D_flow = D || (!V->SU2 && C);
    V->use_self = self;

    // set all crossprojections to true
    V->C_to_P = V->D_to_P = V->P_to_C = V->D_to_C = V->P_to_D = V->C_to_D = true;

    // we own the memory
    V->owns = true;

    double memsize = 0;
    V->Pch = calloc(V->full_vert_size, sizeof(complex128_t));
    V->pp_bubble_int = calloc(V->full_vert_size, sizeof(complex128_t));
    memsize += 2*2*V->full_vert_size;
    V->Cch = calloc(V->full_vert_size, sizeof(complex128_t)); 
    memsize += 2*V->full_vert_size;
    V->Dch = calloc(V->full_vert_size, sizeof(complex128_t)); 
    memsize += 2*V->full_vert_size;
    if (V->use_self) {
        V->self_en = calloc(V->selfenergy_size, sizeof(complex128_t));
        memsize += 2*V->selfenergy_size;
        index_t nk_local = V->nk;
        index_t bubble_nk_local = (V->tu_data->use_cuda_mpi_loop && V->tu_data->use_reduced_loop) ? V->nk :
            nk_local;
        V->bubble_helper = calloc(POW2(V->n_orbff)*bubble_nk_local*POW4(V->n_spin), sizeof(complex128_t));
        V->product_helper = calloc(POW2(V->n_orbff)*nk_local*POW4(V->n_spin), sizeof(complex128_t));
        V->projection_helper = calloc(POW2(V->n_orbff)*V->my_nk*POW4(V->n_spin), sizeof(complex128_t));
        memsize += 2*POW2(V->n_orbff)*nk_local*POW4(V->n_spin);
        memsize += POW2(V->n_orbff)*V->my_nk*POW4(V->n_spin);
    } else {
        index_t nk_local = V->my_nk > V->symm->my_nk_in_bz ? V->my_nk : V->symm->my_nk_in_bz;
        index_t bubble_nk_local = (V->tu_data->use_cuda_mpi_loop && V->tu_data->use_reduced_loop) ? V->nk :
            nk_local;
        V->bubble_helper = calloc(POW2(V->n_orbff)*bubble_nk_local*POW4(V->n_spin), sizeof(complex128_t));
        memsize += POW2(V->n_orbff)*nk_local*POW4(V->n_spin);
        nk_local = V->my_nk > proj->nr ? V->my_nk : proj->nr;
        V->product_helper = calloc(POW2(V->n_orbff)*nk_local*POW4(V->n_spin), sizeof(complex128_t));
        memsize += POW2(V->n_orbff)*nk_local*POW4(V->n_spin);
        V->projection_helper = calloc(POW2(V->n_orbff)*V->my_nk*POW4(V->n_spin), sizeof(complex128_t));
        memsize += POW2(V->n_orbff)*V->my_nk*POW4(V->n_spin);
    }
    if (V->C_flow || V->D_flow) {
        V->ph_bubble_int = calloc(V->full_vert_size, sizeof(complex128_t));
        memsize += 2*V->full_vert_size;
    }
    tu_vertex_initialize_GPU_memory( V, proj, loop );
    if (V->model->ffill != NULL) {
        tu_vertex_init_U_from_fullV( V, V->model, proj );
    } else {
        tu_vertex_init_U( V, V->model->internals->enforce_exchange );
        if (!(V->P_flow && V->C_flow && V->D_flow)) {
            mpi_wrn_printf("not all channels (P:%i,C:%i,D:%i): ensure correct formfactors\n",
                    V->P_flow, V->C_flow, V->D_flow);
            bool dummy1, dummy2;
            if (V->D_flow) {
                dummy1 = V->C_flow; dummy2 = V->P_flow;
                V->C_flow = V->P_flow = true;
                mpi_vrb_printf("Interactions only in D\n");
                D_projection(V->Dch, V, proj);
                if (V->C_flow) ZERO_OUT(V->Cch, V->full_vert_size);
                if (V->P_flow) ZERO_OUT(V->Pch, V->full_vert_size);
                V->C_flow = dummy1; V->P_flow = dummy2;
            } else if (V->C_flow) {
                dummy1 = V->D_flow; dummy2 = V->P_flow;
                V->P_flow = V->D_flow = true;
                mpi_vrb_printf("Interactions only in C\n");
                C_projection(V->Cch, V, proj);
                if (V->P_flow) ZERO_OUT(V->Pch, V->full_vert_size);
                V->D_flow = dummy1; V->P_flow = dummy2;
            } else {
                dummy1 = V->D_flow; dummy2 = V->C_flow;
                V->C_flow = V->D_flow = true;
                mpi_vrb_printf("Interactions only in P\n");
                P_projection(V->Pch, V, proj);
                V->D_flow = dummy1; V->C_flow = dummy2;
            }
        }
    }
    if (!V->P_flow) {
        free(V->Pch); V->Pch = NULL;
        free(V->pp_bubble_int); V->pp_bubble_int = NULL;
        memsize -= 2*2*V->full_vert_size;
    }
    if (!V->C_flow) { free(V->Cch); V->Cch = NULL; memsize -= 2*V->full_vert_size; }
    if (!V->D_flow) {free(V->Dch); V->Dch = NULL; memsize -= 2*V->full_vert_size; }

    mpi_vrb_printf("TU euler requires %.1fGB\n", memsize/POW3(1024.) * sizeof(complex128_t))
    mpi_vrb_printf("TU vertex size: %.1fGB\n",  (double)V->full_vert_size/POW3(1024.) * sizeof(complex128_t));

#ifdef TU_VERTEX_RECO_GPU_ZGEMM
    V->GPU_reco_gemm = single_gemm_Xt_init();
#endif

    return V;
}

void tu_vertex_free( tu_vertex_t* V ) {
    free(V->Cch);
    free(V->Pch);
    free(V->Dch);
    free(V->pp_bubble_int);
    free(V->ph_bubble_int);
    free(V->self_en);
    if (V->owns) {
        free(V->bubble_helper);
        free(V->product_helper);
        free(V->projection_helper);
    }
#ifdef USE_CUDA
    if (V->has_gpu_memory) {
        for (int i = 0; i<tu_gpu_info().tu_num_of_devices;++i) {
            CUDA_CHECK( cudaFree(V->GPU_memory_pool[i]) );
            CUDA_CHECK( cudaFree(V->GPU_memory_pool_managed[i]) );
        }
    }
#endif
#ifdef TU_VERTEX_RECO_GPU_ZGEMM
    if (V->owns) single_gemm_Xt_free( V->GPU_reco_gemm );
#endif
    free( V );
}

tu_vertex_t* tu_vertex_copy( const tu_vertex_t* O ) {
    tu_vertex_t* V = calloc(1, sizeof*V);

    V->model = O->model;
    V->tu_data = O->tu_data;
    V->n_orbff = O->n_orbff;
    V->n_orb = O->n_orb;
    V->n_bonds = O->n_bonds;
    V->n_spin = O->n_spin;
    V->nk = O->nk;
    V->nkf = O->nkf;
    V->my_nk = O->my_nk;
    V->my_nk_off = O->my_nk_off;
    V->SU2 = O->SU2;
    V->full_vert_size = O->full_vert_size;
    V->full_bubble_size = O->full_bubble_size;
    V->selfenergy_size = O->selfenergy_size;
    V->symm = O->symm;
    V->P_flow = O->P_flow;
    V->C_flow = O->C_flow;
    V->D_flow = O->D_flow;
    V->use_self = O->use_self;
    V->C_to_P = O->C_to_P;
    V->D_to_P = O->D_to_P;
    V->P_to_C = O->P_to_C;
    V->D_to_C = O->D_to_C;
    V->P_to_D = O->P_to_D;
    V->C_to_D = O->C_to_D;

    V->bubble_helper = O->bubble_helper;
    V->product_helper = O->product_helper;
    V->projection_helper = O->projection_helper;
    V->owns = false;

    if (V->P_flow) V->Pch = calloc(V->full_vert_size, sizeof(complex128_t));
    if (V->C_flow) V->Cch = calloc(V->full_vert_size, sizeof(complex128_t));
    if (V->D_flow) V->Dch = calloc(V->full_vert_size, sizeof(complex128_t));

    if (V->use_self) V->self_en = calloc(V->selfenergy_size, sizeof(complex128_t));
    if (V->P_flow) V->pp_bubble_int = calloc(V->full_bubble_size, sizeof(complex128_t));
    if (V->C_flow || V->D_flow) V->ph_bubble_int = calloc(V->full_bubble_size, sizeof(complex128_t));

    if (V->P_flow) memcpy(V->Pch,O->Pch,sizeof*V->Pch * V->full_vert_size);
    if (V->C_flow) memcpy(V->Cch,O->Cch,sizeof*V->Cch * V->full_vert_size);
    if (V->D_flow) memcpy(V->Dch,O->Dch,sizeof*V->Dch * V->full_vert_size);
    if (V->P_flow) memcpy(V->pp_bubble_int,O->pp_bubble_int,sizeof*V->pp_bubble_int * V->full_bubble_size);
    if (V->C_flow||V->D_flow) memcpy(V->ph_bubble_int,O->ph_bubble_int,
            sizeof*V->ph_bubble_int * V->full_bubble_size);
    if (V->use_self) memcpy(V->self_en,O->self_en,sizeof*V->self_en * V->selfenergy_size);
    for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 3; ++j) {
        V->GPU_gemm_buffer[i][j] = O->GPU_gemm_buffer[i][j];
    }
    V->GPU_reco_gemm = O->GPU_reco_gemm;

    return V;
}

void tu_vertex_add(tu_vertex_t* result, const tu_vertex_t* inp, const double factor) {
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        if (result->P_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) result->Pch[i] += factor*(inp->Pch[i]);
        }
        if (result->C_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) result->Cch[i] += factor*(inp->Cch[i]);
        }
        if (result->D_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) result->Dch[i] += factor*(inp->Dch[i]);
        }
        if (result->P_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_bubble_size; ++i)
                result->pp_bubble_int[i] += factor*(inp->pp_bubble_int[i]);
        }
        if (result->C_flow || result->D_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_bubble_size; ++i)
                result->ph_bubble_int[i] += factor*(inp->ph_bubble_int[i]);
        }
        #pragma omp for
        for (index_t i=0; i<inp->selfenergy_size; ++i)
            result->self_en[i] += factor*(inp->self_en[i]);
    }
}

void tu_vertex_set(tu_vertex_t* f, const tu_vertex_t* initial, const tu_vertex_t* inp, const double factor) {
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        if (f->P_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) f->Pch[i] = initial->Pch[i] + factor*(inp->Pch[i]);
        }
        if (f->C_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) f->Cch[i] = initial->Cch[i] + factor*(inp->Cch[i]);
        }
        if (f->D_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) f->Dch[i] = initial->Dch[i] + factor*(inp->Dch[i]);
        }
        if (f->P_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_bubble_size; ++i)
                f->pp_bubble_int[i] = initial->pp_bubble_int[i] +  factor*(inp->pp_bubble_int[i]);
        }
        if (f->D_flow || f->C_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_bubble_size; ++i)
                f->ph_bubble_int[i] = initial->ph_bubble_int[i] + factor*(inp->ph_bubble_int[i]);
        }
        #pragma omp for
        for (index_t i=0; i<inp->selfenergy_size; ++i)
            f->self_en[i] = initial->self_en[i] + factor*(inp->self_en[i]);
    }
}

void tu_vertex_multiply(tu_vertex_t* f, const tu_vertex_t* inp, const double factor) {
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        if (f->P_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) f->Pch[i] = factor*(inp->Pch[i]);
        }
        if (f->C_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) f->Cch[i] =  factor*(inp->Cch[i]);
        }
        if (f->D_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_vert_size; ++i) f->Dch[i] = factor*(inp->Dch[i]);
        }

        if (f->P_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_bubble_size; ++i) f->pp_bubble_int[i] = factor*(inp->pp_bubble_int[i]);
        }
        if (f->C_flow || f->D_flow) {
            #pragma omp for
            for (index_t i=0; i<inp->full_bubble_size; ++i) f->ph_bubble_int[i] = factor*(inp->ph_bubble_int[i]);
        }
        #pragma omp for
        for (index_t i=0; i<inp->selfenergy_size; ++i) f->self_en[i] = factor*(inp->self_en[i]);
    }
}

static inline double absmax( const complex128_t* start, index_t length ) {
    double result = 0.0;
    #pragma omp parallel for reduction(max:result) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<length; ++i) {
        double mabs = cabs(start[i]); result = MAX(result, mabs);
    }
    return result;
}

void tu_vertex_max(const tu_vertex_t* V, double out[6]) {
    ZERO_OUT( out, 6 );
    if (V->P_flow) {
        double max_trial = absmax(V->Pch,V->full_vert_size);
        out[0] = diverge_mpi_max(&max_trial);
    }
    if (V->C_flow) {
        double max_trial = absmax(V->Cch,V->full_vert_size);
        out[1] = diverge_mpi_max(&max_trial);
    }
    if (V->D_flow) {
        double max_trial = absmax(V->Dch,V->full_vert_size);
        out[2] = diverge_mpi_max(&max_trial);
    }

    if (V->P_flow) {
        double max_trial = absmax(V->pp_bubble_int,V->full_bubble_size);
        out[3] = diverge_mpi_max(&max_trial);
    }
    if (V->C_flow || V->D_flow) {
        double max_trial = absmax(V->ph_bubble_int,V->full_bubble_size);
        out[4] = diverge_mpi_max(&max_trial);
    }
    if (V->selfenergy_size>0) {
        double max_trial = absmax(V->self_en,V->selfenergy_size);
        out[5] = diverge_mpi_max(&max_trial);
    }
}

double tu_vertex_infinity_norm( const tu_vertex_t* V ) {
    double maxs[6] = {0};
    tu_vertex_max( V, maxs );

    double max = maxs[0];
    for (int i=1; i<6; ++i) max = MAX(max, maxs[i]);
    return max;
}

void diverge_tu_vertex_reco_ft( const tu_vertex_t* restrict V, char chan, complex128_t* restrict buf,
        const complex128_t* restrict existing_r_exp, index_t nr, complex128_t* restrict buf_q/*=NULL*/ ) {
    const char channel = chan;
    bool was_null = false;
    if (!buf_q) {
        was_null = true;
        index_t mvsq = V->symm->use_symmetries ? V->symm->my_nk_in_bz: V->my_nk;
        index_t my_vertsize = mvsq*POW2(V->n_orbff*POW2(V->n_spin));
        buf_q = calloc(my_vertsize,sizeof(complex128_t));
    }

    const index_t n_spin = V->n_spin,
                  n_orbff = V->n_orbff;

    const index_t* restrict symm_from_ibz_2_bz = V->symm->from_ibz_2_bz;
    const index_t* restrict symm_kmaps_to = V->symm->kmaps_to;
    const index_t my_nk_off = V->tu_data->my_nk_off;
    const index_t* restrict symm_o2m_map_len = V->symm->o2m_map_len;
    const index_t* restrict symm_o2m_map_off = V->symm->o2m_map_off;
    const index_t* restrict symm_idx_map_symm = V->symm->idx_map_symm;
    const complex128_t* restrict symm_prefactor_symmP = V->symm->prefactor_symmP;
    const complex128_t* restrict symm_prefactor_symmCD = V->symm->prefactor_symmCD;
    const complex128_t* restrict Pch = V->Pch;
    const complex128_t* restrict Cch = V->Cch;
    const complex128_t* restrict Dch = V->Dch;

    const index_t rest = POW2(n_orbff*POW2(n_spin));
    if (V->symm->use_symmetries) {
        const index_t my_nk_in_bz = V->symm->my_nk_in_bz;
        const index_t my_nk_in_bz_off = V->symm->my_nk_in_bz_off;

        ZERO_OUT(buf_q,rest*my_nk_in_bz);
        const index_t nsq = POW2(n_spin);
        const index_t legdim = nsq*n_orbff;
        #pragma omp parallel for collapse(6) num_threads(diverge_omp_num_threads())
        for (index_t s4=0; s4<n_spin; ++s4)
        for (index_t s3=0; s3<n_spin; ++s3)
        for (index_t mo3=0; mo3<n_orbff; ++mo3)
        for (index_t s2=0; s2<n_spin; ++s2)
        for (index_t s1=0; s1<n_spin; ++s1)
        for (index_t mo1=0; mo1<n_orbff; ++mo1)
        for (index_t q=0; q<my_nk_in_bz; ++q) {
            const index_t q_in_full = symm_from_ibz_2_bz[q + my_nk_in_bz_off];
            const index_t qFROM_IBZ = symm_kmaps_to[q_in_full] - my_nk_off;
            const index_t idx_TO = q + my_nk_in_bz*(mo1 + n_orbff*((s1+n_spin*s2)+nsq*
                    (mo3 + n_orbff*(s3+n_spin*s4))));

            if (channel == 'P') {
                index_t lidx = IDX4(s2,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s4,s3,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l=0; l < symm_o2m_map_len[lidx]; ++l)
                for(index_t r=0; r < symm_o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm_idx_map_symm[l+symm_o2m_map_off[lidx]] + legdim * (
                            symm_idx_map_symm[r+symm_o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += symm_prefactor_symmP[l+symm_o2m_map_off[lidx]]*
                                conj(symm_prefactor_symmP[r+symm_o2m_map_off[ridx]])* Pch[idx_FROM];
                }
            }
            if (channel == 'C') {
                index_t lidx = IDX4(s4,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s3,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l=0; l < symm_o2m_map_len[lidx]; ++l)
                for(index_t r=0; r < symm_o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm_idx_map_symm[l+symm_o2m_map_off[lidx]] + legdim * (
                            symm_idx_map_symm[r+symm_o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += symm_prefactor_symmCD[l+symm_o2m_map_off[lidx]]*
                                conj(symm_prefactor_symmCD[r+symm_o2m_map_off[ridx]])* Cch[idx_FROM];
                }
            }
            if (channel == 'D') {
                index_t lidx = IDX4(s3,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s4,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l=0; l < symm_o2m_map_len[lidx]; ++l)
                for(index_t r=0; r < symm_o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm_idx_map_symm[l+symm_o2m_map_off[lidx]] + legdim * (
                            symm_idx_map_symm[r+symm_o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += symm_prefactor_symmCD[l+symm_o2m_map_off[lidx]]*
                                conj(symm_prefactor_symmCD[r+symm_o2m_map_off[ridx]])* Dch[idx_FROM];
                }
            }
        }
#ifdef TU_VERTEX_RECO_GPU_ZGEMM
        single_gemm_Xt(V->GPU_reco_gemm,buf_q,existing_r_exp,buf,nr,rest,my_nk_in_bz,1./(double)V->nk,0.0);
#else
        single_gemm(buf_q,existing_r_exp,buf,nr,rest,my_nk_in_bz,1./(double)V->nk,0.0);
#endif
    } else {
        #pragma omp parallel for collapse(6) num_threads(diverge_omp_num_threads())
        for (index_t s4=0; s4<n_spin; ++s4)
        for (index_t s3=0; s3<n_spin; ++s3)
        for (index_t mo3=0; mo3<n_orbff; ++mo3)
        for (index_t s2=0; s2<n_spin; ++s2)
        for (index_t s1=0; s1<n_spin; ++s1)
        for (index_t mo1=0; mo1<n_orbff; ++mo1)
        for (index_t q=0; q<V->my_nk; ++q) {
            const index_t idx_TO = q + V->my_nk*(mo1+n_orbff*(s1+n_spin*
                                (s2+n_spin*(mo3+n_orbff*(s3+n_spin*s4)))));
            if (channel == 'P') {
                buf_q[idx_TO] = Pch[mo1+n_orbff*(s1+n_spin*
                                    (s2+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                        (s4+n_spin*q)))))];
            }
            if (channel == 'C') {
                buf_q[idx_TO] = Cch[mo1+n_orbff*(s1+n_spin*
                                    (s4+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                        (s2+n_spin*q)))))];
            }
            if (channel == 'D') {
                buf_q[idx_TO] = Dch[mo1+n_orbff*(s1+n_spin*
                                    (s3+n_spin*(mo3+n_orbff*(s4+n_spin*
                                                        (s2+n_spin*q)))))];
            }
        }
#ifdef TU_VERTEX_RECO_GPU_ZGEMM
        single_gemm_Xt(V->GPU_reco_gemm,buf_q,existing_r_exp,buf,nr,rest,V->my_nk,1./(double)V->nk,0.0);
#else
        single_gemm(buf_q,existing_r_exp,buf,nr,rest,V->my_nk,1./(double)V->nk,0.0);
#endif
    }
    diverge_mpi_allreduce_complex_sum_inplace(buf, nr*rest);

    if (was_null) free(buf_q);
}

void diverge_tu_vertex_reco( const tu_vertex_t* V, char chan, complex128_t* buf_q,
        const complex128_t* inp/*=NULL*/ ) {
    const char channel = chan;
    const index_t rest = POW2(V->n_orbff*POW2(V->n_spin));
    ZERO_OUT( buf_q, rest*V->nk );
    if (!inp) {
        inp = (channel == 'P') ? V->Pch :
              (channel == 'C') ? V->Cch :
              (channel == 'D') ? V->Dch : NULL;
    }
    const complex128_t* const helper = inp;

    if (V->symm->use_symmetries) {
        const index_t my_nk_in_bz = V->symm->my_nk_in_bz;
        const index_t my_nk_in_bz_off = V->symm->my_nk_in_bz_off;

        const index_t nsq = POW2(V->n_spin);
        const index_t legdim = nsq*V->n_orbff;
        #pragma omp parallel for schedule(static) collapse(7) num_threads(diverge_omp_num_threads())
        for (index_t s4=0; s4<V->n_spin; ++s4)
        for (index_t s3=0; s3<V->n_spin; ++s3)
        for (index_t mo3=0; mo3<V->n_orbff; ++mo3)
        for (index_t s2=0; s2<V->n_spin; ++s2)
        for (index_t s1=0; s1<V->n_spin; ++s1)
        for (index_t mo1=0; mo1<V->n_orbff; ++mo1)
        for (index_t q=0; q<my_nk_in_bz; ++q) {
            const index_t q_in_full = V->symm->from_ibz_2_bz[q + my_nk_in_bz_off];
            const index_t qFROM_IBZ = V->symm->kmaps_to[q_in_full] - V->tu_data->my_nk_off;
            const index_t idx_TO = q_in_full + V->nk*(mo1 + V->n_orbff*((s1+V->n_spin*s2)+nsq*
                    (mo3 + V->n_orbff*(s3+V->n_spin*s4))));

            if (channel == 'P') {
                index_t lidx = IDX4(s2,s1,mo1,q,V->n_spin,V->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s4,s3,mo3,q,V->n_spin,V->n_orbff,my_nk_in_bz);

                for (index_t l=0; l < V->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r < V->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        V->symm->idx_map_symm[l+V->symm->o2m_map_off[lidx]] + legdim * (
                            V->symm->idx_map_symm[r+V->symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += V->symm->prefactor_symmP[l+V->symm->o2m_map_off[lidx]]*
                                conj(V->symm->prefactor_symmP[r+V->symm->o2m_map_off[ridx]])* helper[idx_FROM];
                }
            }
            if (channel == 'C') {
                index_t lidx = IDX4(s4,s1,mo1,q,V->n_spin,V->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s3,mo3,q,V->n_spin,V->n_orbff,my_nk_in_bz);

                for (index_t l=0; l < V->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r < V->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        V->symm->idx_map_symm[l+V->symm->o2m_map_off[lidx]] + legdim * (
                            V->symm->idx_map_symm[r+V->symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += V->symm->prefactor_symmCD[l+V->symm->o2m_map_off[lidx]]*
                                conj(V->symm->prefactor_symmCD[r+V->symm->o2m_map_off[ridx]])* helper[idx_FROM];
                }
            }
            if (channel == 'D') {
                index_t lidx = IDX4(s3,s1,mo1,q,V->n_spin,V->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s4,mo3,q,V->n_spin,V->n_orbff,my_nk_in_bz);

                for (index_t l=0; l < V->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r < V->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        V->symm->idx_map_symm[l+V->symm->o2m_map_off[lidx]] + legdim * (
                            V->symm->idx_map_symm[r+V->symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += V->symm->prefactor_symmCD[l+V->symm->o2m_map_off[lidx]]*
                                conj(V->symm->prefactor_symmCD[r+V->symm->o2m_map_off[ridx]])* helper[idx_FROM];
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static) collapse(7) num_threads(diverge_omp_num_threads())
        for (index_t s4=0; s4<V->n_spin; ++s4)
        for (index_t s3=0; s3<V->n_spin; ++s3)
        for (index_t mo3=0; mo3<V->n_orbff; ++mo3)
        for (index_t s2=0; s2<V->n_spin; ++s2)
        for (index_t s1=0; s1<V->n_spin; ++s1)
        for (index_t mo1=0; mo1<V->n_orbff; ++mo1)
        for (index_t q=0; q<V->my_nk; ++q) {
            const index_t idx_TO = q + V->my_nk_off+ V->nk*(mo1+V->n_orbff*(s1+V->n_spin*
                                (s2+V->n_spin*(mo3+V->n_orbff*(s3+V->n_spin*
                                                    s4)))));
            if (channel == 'P') {
                buf_q[idx_TO] = helper[mo1+V->n_orbff*(s1+V->n_spin*
                                    (s2+V->n_spin*(mo3+V->n_orbff*(s3+V->n_spin*
                                                        (s4+V->n_spin*q)))))];
            }
            if (channel == 'C') {
                buf_q[idx_TO] = helper[mo1+V->n_orbff*(s1+V->n_spin*
                                    (s4+V->n_spin*(mo3+V->n_orbff*(s3+V->n_spin*
                                                        (s2+V->n_spin*q)))))];
            }
            if (channel == 'D') {
                buf_q[idx_TO] = helper[mo1+V->n_orbff*(s1+V->n_spin*
                                    (s3+V->n_spin*(mo3+V->n_orbff*(s4+V->n_spin*
                                                        (s2+V->n_spin*q)))))];
            }
        }
    }
    diverge_mpi_allreduce_complex_sum_inplace(buf_q, V->nk*rest);
}

void diverge_tu_vertex_reco_at_q( const tu_vertex_t* V, char chan, complex128_t* buf_q,
        index_t q_search, const complex128_t* inp/*=NULL*/ ) {
    const char channel = chan;
    const index_t rest = POW2(V->n_orbff*POW2(V->n_spin));
    ZERO_OUT( buf_q, rest );
    if (!inp) {
        inp = (channel == 'P') ? V->Pch :
              (channel == 'C') ? V->Cch :
              (channel == 'D') ? V->Dch : NULL;
    }
    const complex128_t* const helper = inp;

    if (V->symm->use_symmetries) {
        const index_t my_nk_in_bz = V->symm->my_nk_in_bz;
        const index_t my_nk_in_bz_off = V->symm->my_nk_in_bz_off;

        const index_t nsq = POW2(V->n_spin);
        const index_t legdim = nsq*V->n_orbff;
        #pragma omp parallel for schedule(static) collapse(7) num_threads(diverge_omp_num_threads())
        for (index_t s4=0; s4<V->n_spin; ++s4)
        for (index_t s3=0; s3<V->n_spin; ++s3)
        for (index_t mo3=0; mo3<V->n_orbff; ++mo3)
        for (index_t s2=0; s2<V->n_spin; ++s2)
        for (index_t s1=0; s1<V->n_spin; ++s1)
        for (index_t mo1=0; mo1<V->n_orbff; ++mo1)
        for (index_t q=0; q<my_nk_in_bz; ++q) {
            const index_t q_in_full = V->symm->from_ibz_2_bz[q + my_nk_in_bz_off];
            if (q_in_full != q_search) continue;
            const index_t qFROM_IBZ = V->symm->kmaps_to[q_in_full] - V->tu_data->my_nk_off;
            const index_t idx_TO = mo1 + V->n_orbff*((s1+V->n_spin*s2)+nsq*
                    (mo3 + V->n_orbff*(s3+V->n_spin*s4)));

            if (channel == 'P') {
                index_t lidx = IDX4(s2,s1,mo1,q,V->n_spin,V->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s4,s3,mo3,q,V->n_spin,V->n_orbff,my_nk_in_bz);

                for (index_t l=0; l < V->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r < V->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        V->symm->idx_map_symm[l+V->symm->o2m_map_off[lidx]] + legdim * (
                            V->symm->idx_map_symm[r+V->symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += V->symm->prefactor_symmP[l+V->symm->o2m_map_off[lidx]]*
                                conj(V->symm->prefactor_symmP[r+V->symm->o2m_map_off[ridx]])* helper[idx_FROM];
                }
            }
            if (channel == 'C') {
                index_t lidx = IDX4(s4,s1,mo1,q,V->n_spin,V->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s3,mo3,q,V->n_spin,V->n_orbff,my_nk_in_bz);

                for (index_t l=0; l < V->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r < V->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        V->symm->idx_map_symm[l+V->symm->o2m_map_off[lidx]] + legdim * (
                            V->symm->idx_map_symm[r+V->symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += V->symm->prefactor_symmCD[l+V->symm->o2m_map_off[lidx]]*
                                conj(V->symm->prefactor_symmCD[r+V->symm->o2m_map_off[ridx]])* helper[idx_FROM];
                }
            }
            if (channel == 'D') {
                index_t lidx = IDX4(s3,s1,mo1,q,V->n_spin,V->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s4,mo3,q,V->n_spin,V->n_orbff,my_nk_in_bz);

                for (index_t l=0; l < V->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r < V->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        V->symm->idx_map_symm[l+V->symm->o2m_map_off[lidx]] + legdim * (
                            V->symm->idx_map_symm[r+V->symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] += V->symm->prefactor_symmCD[l+V->symm->o2m_map_off[lidx]]*
                                conj(V->symm->prefactor_symmCD[r+V->symm->o2m_map_off[ridx]])* helper[idx_FROM];
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static) collapse(7) num_threads(diverge_omp_num_threads())
        for (index_t s4=0; s4<V->n_spin; ++s4)
        for (index_t s3=0; s3<V->n_spin; ++s3)
        for (index_t mo3=0; mo3<V->n_orbff; ++mo3)
        for (index_t s2=0; s2<V->n_spin; ++s2)
        for (index_t s1=0; s1<V->n_spin; ++s1)
        for (index_t mo1=0; mo1<V->n_orbff; ++mo1)
        for (index_t q=0; q<V->my_nk; ++q) {
            if ((q + V->my_nk_off) != q_search) continue;
            const index_t idx_TO = mo1+V->n_orbff*(s1+V->n_spin*
                                (s2+V->n_spin*(mo3+V->n_orbff*(s3+V->n_spin*
                                                    s4))));
            if (channel == 'P') {
                buf_q[idx_TO] = helper[mo1+V->n_orbff*(s1+V->n_spin*
                                    (s2+V->n_spin*(mo3+V->n_orbff*(s3+V->n_spin*
                                                        (s4+V->n_spin*q)))))];
            }
            if (channel == 'C') {
                buf_q[idx_TO] = helper[mo1+V->n_orbff*(s1+V->n_spin*
                                    (s4+V->n_spin*(mo3+V->n_orbff*(s3+V->n_spin*
                                                        (s2+V->n_spin*q)))))];
            }
            if (channel == 'D') {
                buf_q[idx_TO] = helper[mo1+V->n_orbff*(s1+V->n_spin*
                                    (s3+V->n_spin*(mo3+V->n_orbff*(s4+V->n_spin*
                                                        (s2+V->n_spin*q)))))];
            }
        }
    }
    diverge_mpi_allreduce_complex_sum_inplace(buf_q, rest);
}

void tu_vertex_product( complex128_t* result, const complex128_t* B,
        const complex128_t* C, complex128_t* helper, index_t productsize,
        index_t nbatch, double prefac, bool buffered, char* GPUbuffer[32][3] ) {
    if (buffered) {
        batched_gemm_with_buffers(B, C, helper, GPUbuffer, productsize, 1., 0., nbatch);
        batched_gemm_with_buffers(helper, B, result, GPUbuffer, productsize, prefac, 0., nbatch);
    } else {
        batched_gemm(B, C, helper, productsize, 1., 0., nbatch);
        batched_gemm(helper, B, result, productsize, prefac, 0., nbatch);
    }
}
