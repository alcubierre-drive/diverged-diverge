/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "projection_handler.hpp"
#include "cuda_init.h"

inline void add_to_all_frq(complex128_t* const to, const complex128_t* const inc,
                           index_t rest) {
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t r = 0; r < rest; ++r) {
        to[r] += inc[r];
    }
}

template<typename T>
double __elapsed_time__( T x ) {
    double tick = diverge_mpi_wtime();
    x();
    double tock = diverge_mpi_wtime();
    return tock-tick;
}
template<typename T>
double __no_elapsed_time__( T x ) {
    x();
    return -1.0;
}

void timings(double time_cpyfill, double time_reco, double time_handler_ft, double time_add) {
    if(time_cpyfill > 0.0) {
        mpi_log_printf("cpy+fill: %3.3fs    reco: %3.3fs \n to_channel: %3.3fs    add: %3.3fs \n",
                       time_cpyfill,time_reco,time_handler_ft,time_add);
    }
}
#ifdef TU_PROJ_TIMING
#define elapsed_time( fun ) __elapsed_time__( [&](){ fun; } )
#else // TU_PROJ_TIMING
#define elapsed_time( fun ) __no_elapsed_time__( [&](){ fun; } )
#endif // TU_PROJ_TIMING

void P_projection(complex128_t* proj_vertex, Vertex* const vertex,
                const Projection* const proj) {
    if(!vertex->P_flow) return;
    double time_cpyfill = 0.0;double time_reco = 0.0;
    double time_handler_ft = 0.0;double time_add = 0.0;
    time_cpyfill+=elapsed_time(memcpy(proj_vertex, vertex->Pch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;

    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices) {
        if(vertex->C_flow && vertex->C_to_P) {
            time_reco+=elapsed_time(vertex->template reco_ft<'C'>(
                            buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
            time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));

            time_handler_ft+=elapsed_time(GPUproj_handler_ft(&(proj->proj_descr),
                            'P',bufMPI, buf, proj->get_CP_PC_map_TO_padded(),
                proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->my_nk,
                proj->get_nr()));

            time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->D_flow && vertex->D_to_P) {
            time_reco+=elapsed_time(vertex->template reco_ft<'D'>(
                        buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
            time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));

            time_handler_ft+=elapsed_time(GPUproj_handler_ft(&(proj->proj_descr),'P',bufMPI, buf,
                proj->get_D_to_P_map_TO_padded(),
                proj->get_D_to_P_map_FROM(), proj->get_D_to_P_prefac_map(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->my_nk,
                proj->get_nr()));

            time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return ;
    }
    #endif // USE_CUDA
    if(vertex->C_flow && vertex->C_to_P) {
        time_reco+=elapsed_time(vertex->template reco_ft<'C'>(
                    buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_P(bufMPI, buf, proj->get_CP_PC_map_TO(),
            proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
            proj->get_CP_PC_offsets(), proj->get_CP_PC_lengths(),
            proj->get_my_prefac_proj(), vertex->n_orbff,
            vertex->n_spin,vertex->nk,vertex->my_nk,
            proj->get_nr()));

        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
               vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->D_flow && vertex->D_to_P) {
        time_reco+=elapsed_time(vertex->template reco_ft<'D'>(
                    buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_P(bufMPI, buf, proj->get_D_to_P_map_TO(),
            proj->get_D_to_P_map_FROM(), proj->get_D_to_P_prefac_map(),
            proj->get_D_to_P_offsets(), proj->get_D_to_P_lengths(),
            proj->get_my_prefac_proj(), vertex->n_orbff,
            vertex->n_spin,vertex->nk, vertex->my_nk,
            proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
            vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}


void D_projection(complex128_t* proj_vertex,
                    Vertex* const vertex,
                    const Projection* const proj) {
    if(!vertex->D_flow) return;
    double time_cpyfill = 0.0;double time_reco = 0.0;
    double time_handler_ft = 0.0;double time_add = 0.0;
    time_cpyfill+=elapsed_time(memcpy(proj_vertex, vertex->Dch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices)
    {
        if(vertex->C_flow && vertex->C_to_D) {
            time_reco+=elapsed_time(vertex->template reco_ft<'C'>(buf,
                        proj->my_prefac_reco,proj->get_nr(),bufMPI));
            time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            time_handler_ft+=elapsed_time(GPUproj_handler_ft(&(proj->proj_descr),'D',bufMPI, buf,
                    proj->get_CD_DC_map_TO_padded(),
                    proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                    proj->get_my_prefac_proj(), vertex->n_orbff,
                    vertex->n_spin,vertex->my_nk,
                    proj->get_nr()));
            time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->P_flow && vertex->P_to_D) {
            time_reco+=elapsed_time(vertex->template reco_ft<'P'>(
                        buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
            time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            time_handler_ft+=elapsed_time(GPUproj_handler_ft(&(proj->proj_descr),'D',bufMPI, buf,
                    proj->get_P_to_D_map_TO_padded(),
                    proj->get_P_to_D_map_FROM(), proj->get_P_to_D_prefac_map(),
                    proj->get_my_prefac_proj(), vertex->n_orbff,
                    vertex->n_spin,vertex->my_nk,
                    proj->get_nr()));
             time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return ;
    }
    #endif // USE_CUDA
    if(vertex->C_flow && vertex->C_to_D) {
        time_reco+=elapsed_time(vertex->template reco_ft<'C'>(buf,
                    proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_D(bufMPI,buf,proj->get_CD_DC_map_TO(),
                proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                proj->get_CD_DC_offsets(), proj->get_CD_DC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_D) {
        time_reco+=elapsed_time(vertex->template reco_ft<'P'>(buf,
                    proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_D(bufMPI,buf,proj->get_P_to_D_map_TO(),
                proj->get_P_to_D_map_FROM(), proj->get_P_to_D_prefac_map(),
                proj->get_P_to_D_offsets(), proj->get_P_to_D_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}


void C_projection(complex128_t* proj_vertex,
                    Vertex* const vertex,
                    const Projection* const proj) {
    if(!vertex->C_flow) return;
    double time_cpyfill = 0.0;double time_reco = 0.0;
    double time_handler_ft = 0.0;double time_add = 0.0;
    time_cpyfill+=elapsed_time(memcpy(proj_vertex, vertex->Cch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;

    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices)
    {
        if(vertex->P_flow && vertex->P_to_C) {
            time_reco+=elapsed_time(vertex->template reco_ft<'P'>(buf,
                        proj->my_prefac_reco,proj->get_nr(),bufMPI));
            time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            time_handler_ft+=elapsed_time(GPUproj_handler_ft(&(proj->proj_descr),'C',bufMPI, buf,
                    proj->get_CP_PC_map_TO_padded(),
                    proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
                    proj->get_my_prefac_proj(), vertex->n_orbff,
                    vertex->n_spin,vertex->my_nk,
                    proj->get_nr()));
            time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->D_flow && vertex->D_to_C) {
            time_reco+=elapsed_time(vertex->template reco_ft<'D'>(buf,
                        proj->my_prefac_reco,proj->get_nr(),bufMPI));
            time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            
            time_handler_ft+=elapsed_time(GPUproj_handler_ft(
                        &(proj->proj_descr),'C',bufMPI, buf, proj->get_CD_DC_map_TO_padded(),
                        proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                        proj->get_my_prefac_proj(), vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk,
                        proj->get_nr()));
            time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return ;
    }
    #endif // USE_CUDA

    if(vertex->D_flow && vertex->D_to_C) {
        time_reco+=elapsed_time(vertex->template reco_ft<'D'>(
                    buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_C(bufMPI,buf, proj->get_CD_DC_map_TO(),
                proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                proj->get_CD_DC_offsets(), proj->get_CD_DC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_C) {
        time_reco+=elapsed_time(vertex->template reco_ft<'P'>(
                    buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_C(bufMPI,buf, proj->get_CP_PC_map_TO(),
                proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
                proj->get_CP_PC_offsets(), proj->get_CP_PC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk, vertex->my_nk,
                proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}


void P_projection_no_GPU(complex128_t* proj_vertex,
                    Vertex* const vertex,const Projection* const proj) {
    if(!vertex->P_flow) return;
    double time_cpyfill = 0.0;double time_reco = 0.0;
    double time_handler_ft = 0.0;double time_add = 0.0;
    time_cpyfill+=elapsed_time(memcpy(proj_vertex, vertex->Pch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->C_flow && vertex->C_to_P) {
        time_reco+=elapsed_time(vertex->template reco_ft<'C'>(buf,
                    proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_P(bufMPI, buf, proj->get_CP_PC_map_TO(),
            proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
            proj->get_CP_PC_offsets(), proj->get_CP_PC_lengths(),
            proj->get_my_prefac_proj(), vertex->n_orbff,
            vertex->n_spin,vertex->nk,vertex->my_nk,
            proj->get_nr()));

        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
               vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->D_flow && vertex->D_to_P) {
        time_reco+=elapsed_time(vertex->template reco_ft<'D'>(buf,
                    proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_P(bufMPI, buf, proj->get_D_to_P_map_TO(),
            proj->get_D_to_P_map_FROM(), proj->get_D_to_P_prefac_map(),
            proj->get_D_to_P_offsets(), proj->get_D_to_P_lengths(),
            proj->get_my_prefac_proj(), vertex->n_orbff,
            vertex->n_spin,vertex->nk, vertex->my_nk,
            proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
            vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}


void D_projection_no_GPU(complex128_t* proj_vertex,
                    Vertex* const vertex,
                    const Projection* const proj) {
    if(!vertex->D_flow) return;
    double time_cpyfill = 0.0;double time_reco = 0.0;
    double time_handler_ft = 0.0;double time_add = 0.0;
    time_cpyfill+=elapsed_time(memcpy(proj_vertex, vertex->Dch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->C_flow && vertex->C_to_D) {
        time_reco+=elapsed_time(vertex->template reco_ft<'C'>(
                    buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_D(bufMPI,buf,proj->get_CD_DC_map_TO(),
                proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                proj->get_CD_DC_offsets(), proj->get_CD_DC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_D) {
        time_reco+=elapsed_time(vertex->template reco_ft<'P'>(
                    buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_D(bufMPI,buf,proj->get_P_to_D_map_TO(),
                proj->get_P_to_D_map_FROM(), proj->get_P_to_D_prefac_map(),
                proj->get_P_to_D_offsets(), proj->get_P_to_D_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}


void C_projection_no_GPU(complex128_t* proj_vertex,
                    Vertex* const vertex,
                    const Projection* const proj) {
    if(!vertex->C_flow) return;
    double time_cpyfill = 0.0;double time_reco = 0.0;
    double time_handler_ft = 0.0;double time_add = 0.0;
    time_cpyfill+=elapsed_time(memcpy(proj_vertex, vertex->Cch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->D_flow && vertex->D_to_C) {
        time_reco+=elapsed_time(vertex->template reco_ft<'D'>(
                    buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_C(bufMPI,buf, proj->get_CD_DC_map_TO(),
                proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                proj->get_CD_DC_offsets(), proj->get_CD_DC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_C) {
        time_reco+=elapsed_time(vertex->template reco_ft<'P'>(
                    buf,proj->my_prefac_reco,proj->get_nr(),bufMPI));
        time_cpyfill+=elapsed_time(ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        time_handler_ft+=elapsed_time(_to_C(bufMPI,buf, proj->get_CP_PC_map_TO(),
                proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
                proj->get_CP_PC_offsets(), proj->get_CP_PC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk, vertex->my_nk,
                proj->get_nr()));
        time_add+=elapsed_time(add_to_all_frq(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}
