/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "cuda_init.h"
#include "projection_handler.h"
#include "projection.h"
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

void P_projection(complex128_t* proj_vertex, const Vertex* vertex, const projection_t* proj) {
    if(!Vertex_get_X_flow( (void*)vertex, 'P' )) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, Vertex_get_chan((void*)vertex, 'P'), sizeof*proj_vertex * Vertex_get_full_vert_size((void*)vertex)));
    complex128_t* buf = Vertex_get_product_helper((void*)vertex);
    complex128_t* bufMPI = Vertex_get_bubble_helper((void*)vertex);

    #ifdef USE_CUDA
    if (Vertex_get_my_nk((void*)vertex) > (index_t)tu_gpu_info().tu_num_of_devices) {
        if(Vertex_get_X_flow( (void*)vertex, 'C' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'C', 'P' )) {
            TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'C',
                            buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));

            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'P', bufMPI, buf,
                        (index_t*)proj->PC_CP_to_padded, proj->PC_CP_from, proj->PC_CP_mom,
                        proj->PC_CP_to_padded_sz, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                        Vertex_get_n_spin((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));

            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        }
        if(Vertex_get_X_flow( (void*)vertex, 'D' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'D', 'P' )) {
            TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'D',
                        buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));

            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'P', bufMPI, buf,
                        (index_t*)proj->D_to_P_to_padded, proj->D_to_P_from, proj->D_to_P_mom,
                        proj->D_to_P_to_padded_sz, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                        Vertex_get_n_spin((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));

            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return;
    }
    #endif // USE_CUDA
    if(Vertex_get_X_flow( (void*)vertex, 'C' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'C', 'P' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'C',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->PC_CP_to,
            proj->PC_CP_to_sz, proj->PC_CP_from,
            proj->PC_CP_mom, proj->PC_CP_displ,
            proj->PC_CP_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
            Vertex_get_n_spin((void*)vertex),Vertex_get_nk((void*)vertex),Vertex_get_my_nk((void*)vertex), proj->nr));

        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
               Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    if(Vertex_get_X_flow( (void*)vertex, 'D' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'D', 'P' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'D',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->D_to_P_to,
                    proj->D_to_P_to_sz, proj->D_to_P_from,
                    proj->D_to_P_mom, proj->D_to_P_displ,
                    proj->D_to_P_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                    Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
            Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void D_projection(complex128_t* proj_vertex, const Vertex* vertex, const projection_t* proj) {
    if(!Vertex_get_X_flow( (void*)vertex, 'D' )) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, Vertex_get_chan((void*)vertex, 'D'), sizeof*proj_vertex * Vertex_get_full_vert_size((void*)vertex)));
    complex128_t* buf = Vertex_get_product_helper((void*)vertex);
    complex128_t* bufMPI = Vertex_get_bubble_helper((void*)vertex);
    #ifdef USE_CUDA
    if (Vertex_get_my_nk((void*)vertex) > (index_t)tu_gpu_info().tu_num_of_devices) {
        if(Vertex_get_X_flow( (void*)vertex, 'C' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'C', 'D' )) {
            TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'C',buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'D', bufMPI, buf,
                        (index_t*)proj->DC_CD_to_padded, proj->DC_CD_from, proj->DC_CD_mom,
                        proj->DC_CD_to_padded_sz, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                        Vertex_get_n_spin((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        }
        if(Vertex_get_X_flow( (void*)vertex, 'P' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'P', 'D' )) {
            TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'P',
                        buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'D', bufMPI, buf,
                        (index_t*)proj->P_to_D_to_padded, proj->P_to_D_from, proj->P_to_D_mom,
                        proj->P_to_D_to_padded_sz, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                        Vertex_get_n_spin((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
             TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                        Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return;
    }
    #endif // USE_CUDA
    if(Vertex_get_X_flow( (void*)vertex, 'C' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'C', 'D' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'C',buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI,buf,proj->DC_CD_to,
                    proj->DC_CD_to_sz, proj->DC_CD_from,
                    proj->DC_CD_mom, proj->DC_CD_displ,
                    proj->DC_CD_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                    Vertex_get_n_spin((void*)vertex),Vertex_get_nk((void*)vertex),Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    if(Vertex_get_X_flow( (void*)vertex, 'P' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'P', 'D' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'P',buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI,buf,proj->P_to_D_to,
                proj->P_to_D_to_sz, proj->P_to_D_from,
                proj->P_to_D_mom, proj->P_to_D_displ,
                proj->P_to_D_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void C_projection(complex128_t* proj_vertex, const Vertex* vertex, const projection_t* proj) {
    if(!Vertex_get_X_flow( (void*)vertex, 'C' )) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, Vertex_get_chan((void*)vertex, 'C'), sizeof*proj_vertex * Vertex_get_full_vert_size((void*)vertex)));
    complex128_t* buf = Vertex_get_product_helper((void*)vertex);
    complex128_t* bufMPI = Vertex_get_bubble_helper((void*)vertex);

    #ifdef USE_CUDA
    if (Vertex_get_my_nk((void*)vertex) > (index_t)tu_gpu_info().tu_num_of_devices) {
        if(Vertex_get_X_flow( (void*)vertex, 'P' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'P', 'C' )) {
            TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'P',buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'C', bufMPI, buf,
                    (index_t*)proj->PC_CP_to_padded, proj->PC_CP_from, proj->PC_CP_mom,
                    proj->PC_CP_to_padded_sz, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                    Vertex_get_n_spin((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        }
        if(Vertex_get_X_flow( (void*)vertex, 'D' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'D', 'C' )) {
            TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'D',buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'C', bufMPI, buf,
                    (index_t*)proj->DC_CD_to_padded, proj->DC_CD_from, proj->DC_CD_mom,
                    proj->DC_CD_to_padded_sz, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                    Vertex_get_n_spin((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return;
    }
    #endif // USE_CUDA

    if(Vertex_get_X_flow( (void*)vertex, 'D' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'D', 'C' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'D',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI,buf, proj->DC_CD_to,
                proj->DC_CD_to_sz, proj->DC_CD_from,
                proj->DC_CD_mom, proj->DC_CD_displ,
                proj->DC_CD_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    if(Vertex_get_X_flow( (void*)vertex, 'P' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'P', 'C' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'P',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI,buf, proj->PC_CP_to,
                proj->PC_CP_to_sz, proj->PC_CP_from,
                proj->PC_CP_mom, proj->PC_CP_displ,
                proj->PC_CP_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void P_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const projection_t* proj) {
    if(!Vertex_get_X_flow( (void*)vertex, 'P' )) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, Vertex_get_chan((void*)vertex, 'P'), sizeof*proj_vertex * Vertex_get_full_vert_size((void*)vertex)));
    complex128_t* buf = Vertex_get_product_helper((void*)vertex);
    complex128_t* bufMPI = Vertex_get_bubble_helper((void*)vertex);
    if(Vertex_get_X_flow( (void*)vertex, 'C' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'C', 'P' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'C',buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->PC_CP_to,
            proj->PC_CP_to_sz, proj->PC_CP_from,
            proj->PC_CP_mom, proj->PC_CP_displ,
            proj->PC_CP_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
            Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));

        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
               Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    if(Vertex_get_X_flow( (void*)vertex, 'D' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'D', 'P' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'D',buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->D_to_P_to,
            proj->D_to_P_to_sz, proj->D_to_P_from,
            proj->D_to_P_mom, proj->D_to_P_displ,
            proj->D_to_P_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
            Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
            Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void D_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const projection_t* proj) {
    if(!Vertex_get_X_flow( (void*)vertex, 'D' )) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, Vertex_get_chan((void*)vertex, 'D'), sizeof*proj_vertex * Vertex_get_full_vert_size((void*)vertex)));
    complex128_t* buf = Vertex_get_product_helper((void*)vertex);
    complex128_t* bufMPI = Vertex_get_bubble_helper((void*)vertex);
    if(Vertex_get_X_flow( (void*)vertex, 'C' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'C', 'D' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'C',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI, buf, proj->DC_CD_to,
                    proj->DC_CD_to_sz, proj->DC_CD_from,
                    proj->DC_CD_mom, proj->DC_CD_displ,
                    proj->DC_CD_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                    Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    if(Vertex_get_X_flow( (void*)vertex, 'P' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'P', 'D' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'P',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI, buf, proj->P_to_D_to,
                proj->P_to_D_to_sz, proj->P_to_D_from,
                proj->P_to_D_mom, proj->P_to_D_displ,
                proj->P_to_D_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void C_projection_no_GPU(complex128_t* proj_vertex, const Vertex* vertex, const projection_t* proj) {
    if(!Vertex_get_X_flow( (void*)vertex, 'C' )) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, Vertex_get_chan((void*)vertex, 'C'), sizeof*proj_vertex * Vertex_get_full_vert_size((void*)vertex)));
    complex128_t* buf = Vertex_get_product_helper((void*)vertex);
    complex128_t* bufMPI = Vertex_get_bubble_helper((void*)vertex);
    if(Vertex_get_X_flow( (void*)vertex, 'D' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'D', 'C' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'D',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI, buf, proj->DC_CD_to,
                proj->DC_CD_to_sz, proj->DC_CD_from,
                proj->DC_CD_mom, proj->DC_CD_displ,
                proj->DC_CD_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    if(Vertex_get_X_flow( (void*)vertex, 'P' ) && Vertex_get_X_to_Y_proj( (void*)vertex, 'P', 'C' )) {
        TICK_TOCK(time_reco, Vertex_reco_ft( (void*)vertex, 'P',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI, buf, proj->PC_CP_to,
                    proj->PC_CP_to_sz, proj->PC_CP_from,
                    proj->PC_CP_mom, proj->PC_CP_displ,
                    proj->PC_CP_count, proj->my_prefac_proj, Vertex_get_n_orbff((void*)vertex),
                    Vertex_get_n_spin((void*)vertex), Vertex_get_nk((void*)vertex), Vertex_get_my_nk((void*)vertex), proj->nr));
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 Vertex_get_my_nk((void*)vertex)*POW2(Vertex_get_n_orbff((void*)vertex)*POW2(Vertex_get_n_spin((void*)vertex)))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}
