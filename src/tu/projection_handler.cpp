/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "cuda_init.h"
#include "projection_handler.h"
#include "vertex.hpp"

static inline void complex128_array_add( complex128_t* out, const complex128_t* in, index_t sz ) {
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<sz; ++i) out[i] += in[i];
}

static void timings( double time_cpyfill, double time_reco, double time_handler_ft, double time_add ) {
    if (time_cpyfill > 0.0 && time_reco > 0.0 && time_handler_ft > 0.0 && time_add > 0.0)
        mpi_tim_printf( "cpy+fill:%4.1fs, reco:%4.1fs, proj:%4.1fs, add:%4.1fs\n",
                time_cpyfill, time_reco, time_handler_ft, time_add );
}

#ifdef TU_PROJ_TIMING
#define TICK_TOCK( output, ... ) { \
    double tick = diverge_mpi_wtime(); \
    __VA_ARGS__; \
    double tock = diverge_mpi_wtime(); \
    output += tock - tick; \
}
#else
#define TICK_TOCK( output, ... ) { \
    __VA_ARGS__; \
    output -= 1.0; \
}
#endif

void P_projection(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj) {
    if(!vertex->P_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Pch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;

    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices) {
        if(vertex->C_flow && vertex->C_to_P) {
            TICK_TOCK(time_reco, vertex->template reco_ft<'C'>(
                            buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));

            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'P', bufMPI, buf,
                        (index_t*)proj->PC_CP_t, proj->PC_CP_t_sz,
                        proj->PC_CP_f, proj->PC_CP_mom,
                        proj->PC_CP_mom_sz, proj->my_prefac_proj, vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk, proj->nr));

            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->D_flow && vertex->D_to_P) {
            TICK_TOCK(time_reco, vertex->template reco_ft<'D'>(
                        buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));

            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'P', bufMPI, buf,
                        (index_t*)proj->D_to_P_t, proj->D_to_P_t_sz,
                        proj->D_to_P_f, proj->D_to_P_mom,
                        proj->D_to_P_mom_sz, proj->my_prefac_proj, vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk, proj->nr));

            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return ;
    }
    #endif // USE_CUDA
    if(vertex->C_flow && vertex->C_to_P) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'C'>(
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->PC_CP_t,
            proj->PC_CP_t_sz, proj->PC_CP_f,
            proj->PC_CP_mom, proj->PC_CP_d,
            proj->PC_CP_l, proj->my_prefac_proj, vertex->n_orbff,
            vertex->n_spin,vertex->nk,vertex->my_nk, proj->nr));

        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
               vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->D_flow && vertex->D_to_P) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'D'>(
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->D_to_P_t,
                    proj->D_to_P_t_sz, proj->D_to_P_f,
                    proj->D_to_P_mom, proj->D_to_P_d,
                    proj->D_to_P_l, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
            vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void D_projection(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj) {
    if(!vertex->D_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Dch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices)
    {
        if(vertex->C_flow && vertex->C_to_D) {
            TICK_TOCK(time_reco, vertex->template reco_ft<'C'>(buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'D', bufMPI, buf,
                        (index_t*)proj->DC_CD_t, proj->DC_CD_t_sz,
                        proj->DC_CD_f, proj->DC_CD_mom,
                        proj->DC_CD_mom_sz, proj->my_prefac_proj, vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk, proj->nr));
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->P_flow && vertex->P_to_D) {
            TICK_TOCK(time_reco, vertex->template reco_ft<'P'>(
                        buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'D', bufMPI, buf,
                        (index_t*)proj->P_to_D_t, proj->P_to_D_t_sz,
                        proj->P_to_D_f, proj->P_to_D_mom,
                        proj->P_to_D_mom_sz, proj->my_prefac_proj, vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk, proj->nr));
             TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return ;
    }
    #endif // USE_CUDA
    if(vertex->C_flow && vertex->C_to_D) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'C'>(buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI,buf,proj->DC_CD_t,
                    proj->DC_CD_t_sz, proj->DC_CD_f,
                    proj->DC_CD_mom, proj->DC_CD_d,
                    proj->DC_CD_l, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin,vertex->nk,vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_D) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'P'>(buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI,buf,proj->P_to_D_t,
                proj->P_to_D_t_sz, proj->P_to_D_f,
                proj->P_to_D_mom, proj->P_to_D_d,
                proj->P_to_D_l, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void C_projection(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj) {
    if(!vertex->C_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Cch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;

    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices)
    {
        if(vertex->P_flow && vertex->P_to_C) {
            TICK_TOCK(time_reco, vertex->template reco_ft<'P'>(buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'C', bufMPI, buf,
                    (index_t*)proj->PC_CP_t, proj->PC_CP_t_sz,
                    proj->PC_CP_f, proj->PC_CP_mom,
                    proj->PC_CP_mom_sz, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->my_nk, proj->nr));
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->D_flow && vertex->D_to_C) {
            TICK_TOCK(time_reco, vertex->template reco_ft<'D'>(buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'C', bufMPI, buf,
                    (index_t*)proj->DC_CD_t, proj->DC_CD_t_sz,
                    proj->DC_CD_f, proj->DC_CD_mom,
                    proj->DC_CD_mom_sz, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->my_nk, proj->nr));
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return ;
    }
    #endif // USE_CUDA

    if(vertex->D_flow && vertex->D_to_C) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'D'>(
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI,buf, proj->DC_CD_t,
                proj->DC_CD_t_sz, proj->DC_CD_f,
                proj->DC_CD_mom, proj->DC_CD_d,
                proj->DC_CD_l, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_C) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'P'>(
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI,buf, proj->PC_CP_t,
                proj->PC_CP_t_sz, proj->PC_CP_f,
                proj->PC_CP_mom, proj->PC_CP_d,
                proj->PC_CP_l, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void P_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj) {
    if(!vertex->P_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Pch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->C_flow && vertex->C_to_P) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'C'>(buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->PC_CP_t,
            proj->PC_CP_t_sz, proj->PC_CP_f,
            proj->PC_CP_mom, proj->PC_CP_d,
            proj->PC_CP_l, proj->my_prefac_proj, vertex->n_orbff,
            vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));

        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
               vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->D_flow && vertex->D_to_P) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'D'>(buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->D_to_P_t,
            proj->D_to_P_t_sz, proj->D_to_P_f,
            proj->D_to_P_mom, proj->D_to_P_d,
            proj->D_to_P_l, proj->my_prefac_proj, vertex->n_orbff,
            vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
            vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void D_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj) {
    if(!vertex->D_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Dch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->C_flow && vertex->C_to_D) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'C'>(
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI, buf, proj->DC_CD_t,
                    proj->DC_CD_t_sz, proj->DC_CD_f,
                    proj->DC_CD_mom, proj->DC_CD_d,
                    proj->DC_CD_l, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_D) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'P'>(
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI, buf, proj->P_to_D_t,
                proj->P_to_D_t_sz, proj->P_to_D_f,
                proj->P_to_D_mom, proj->P_to_D_d,
                proj->P_to_D_l, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void C_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const Projection* proj) {
    if(!vertex->C_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Cch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->D_flow && vertex->D_to_C) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'D'>(
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI, buf, proj->DC_CD_t,
                proj->DC_CD_t_sz, proj->DC_CD_f,
                proj->DC_CD_mom, proj->DC_CD_d,
                proj->DC_CD_l, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_C) {
        TICK_TOCK(time_reco, vertex->template reco_ft<'P'>(
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI, buf, proj->PC_CP_t,
                    proj->PC_CP_t_sz, proj->PC_CP_f,
                    proj->PC_CP_mom, proj->PC_CP_d,
                    proj->PC_CP_l, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}
