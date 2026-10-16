/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "cuda_init.h"
#include "projection_handler.h"
#include "projection.h"
#include "vertex.h"
#include "vertex_details.h"
#include "../grid/helpers.h" // TICK_TOCK macro

static inline void complex128_array_add( complex128_t* restrict out, const complex128_t* restrict in, index_t sz ) {
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<sz; ++i) out[i] += in[i];
}

#define PROJ_HOOK(FROM, TO) { if (proj->hook != NULL) { \
    (*proj->hook)((diverge_flow_step_vertex_t){ .ary=bufMPI, .q_0=vertex->my_nk_off, \
        .q_1=vertex->my_nk_off+vertex->my_nk, .nk=vertex->nk, .n_orbff=vertex->n_orbff, \
        .n_spin=vertex->n_spin, .backend=FROM, .channel=TO }, proj->hook_ctx); \
} }

static void timings( double time_cpyfill, double time_reco, double time_handler_ft, double time_add ) {
    if (time_cpyfill > 0.0 && time_reco > 0.0 && time_handler_ft > 0.0 && time_add > 0.0)
        mpi_tim_printf( "cpy+fill:%4.1fs, reco:%4.1fs, proj:%4.1fs, add:%4.1fs\n",
                time_cpyfill, time_reco, time_handler_ft, time_add );
}

#ifndef TU_PROJ_TIMING
#undef TICK_TOCK
#define TICK_TOCK( output, ... ) { \
    __VA_ARGS__; \
    output -= 1.0; \
}
#endif

void P_projection(complex128_t* proj_vertex, const tu_vertex_t* vertex, const tu_projection_t* proj) {
    if(!vertex->P_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Pch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;

    #ifdef USE_CUDA
    if (vertex->my_nk > (index_t)tu_gpu_info().tu_num_of_devices) {
        if(vertex->C_flow && vertex->C_to_P) {
            TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'C',
                            buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));

            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'P', bufMPI, buf,
                        (index_t*)proj->PC_CP_to_padded, proj->PC_CP_from, proj->PC_CP_mom,
                        proj->PC_CP_to_padded_sz, proj->my_prefac_proj, vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk, proj->nr));

            PROJ_HOOK('C', 'P');
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->D_flow && vertex->D_to_P) {
            TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'D',
                        buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));

            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'P', bufMPI, buf,
                        (index_t*)proj->D_to_P_to_padded, proj->D_to_P_from, proj->D_to_P_mom,
                        proj->D_to_P_to_padded_sz, proj->my_prefac_proj, vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk, proj->nr));

            PROJ_HOOK('D', 'P');
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return;
    }
    #endif // USE_CUDA
    if(vertex->C_flow && vertex->C_to_P) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'C',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->PC_CP_to,
            proj->PC_CP_to_sz, proj->PC_CP_from,
            proj->PC_CP_mom, proj->PC_CP_displ,
            proj->PC_CP_count, proj->my_prefac_proj, vertex->n_orbff,
            vertex->n_spin,vertex->nk,vertex->my_nk, proj->nr));

        PROJ_HOOK('C', 'P');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
               vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->D_flow && vertex->D_to_P) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'D',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->D_to_P_to,
                    proj->D_to_P_to_sz, proj->D_to_P_from,
                    proj->D_to_P_mom, proj->D_to_P_displ,
                    proj->D_to_P_count, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('D', 'P');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
            vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void D_projection(complex128_t* proj_vertex, const tu_vertex_t* vertex, const tu_projection_t* proj) {
    if(!vertex->D_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Dch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    #ifdef USE_CUDA
    if (vertex->my_nk > (index_t)tu_gpu_info().tu_num_of_devices) {
        if(vertex->C_flow && vertex->C_to_D) {
            TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'C',buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'D', bufMPI, buf,
                        (index_t*)proj->DC_CD_to_padded, proj->DC_CD_from, proj->DC_CD_mom,
                        proj->DC_CD_to_padded_sz, proj->my_prefac_proj, vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk, proj->nr));
            PROJ_HOOK('C', 'D');
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->P_flow && vertex->P_to_D) {
            TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'P',
                        buf,proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'D', bufMPI, buf,
                        (index_t*)proj->P_to_D_to_padded, proj->P_to_D_from, proj->P_to_D_mom,
                        proj->P_to_D_to_padded_sz, proj->my_prefac_proj, vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk, proj->nr));
             PROJ_HOOK('P', 'D');
             TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return;
    }
    #endif // USE_CUDA
    if(vertex->C_flow && vertex->C_to_D) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'C',buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI,buf,proj->DC_CD_to,
                    proj->DC_CD_to_sz, proj->DC_CD_from,
                    proj->DC_CD_mom, proj->DC_CD_displ,
                    proj->DC_CD_count, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin,vertex->nk,vertex->my_nk, proj->nr));
        PROJ_HOOK('C', 'D');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_D) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'P',buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI,buf,proj->P_to_D_to,
                proj->P_to_D_to_sz, proj->P_to_D_from,
                proj->P_to_D_mom, proj->P_to_D_displ,
                proj->P_to_D_count, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('P', 'D');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void C_projection(complex128_t* proj_vertex, const tu_vertex_t* vertex, const tu_projection_t* proj) {
    if(!vertex->C_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Cch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;

    #ifdef USE_CUDA
    if (vertex->my_nk > (index_t)tu_gpu_info().tu_num_of_devices) {
        if(vertex->P_flow && vertex->P_to_C) {
            TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'P',buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'C', bufMPI, buf,
                    (index_t*)proj->PC_CP_to_padded, proj->PC_CP_from, proj->PC_CP_mom,
                    proj->PC_CP_to_padded_sz, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->my_nk, proj->nr));
            PROJ_HOOK('P', 'C');
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        if(vertex->D_flow && vertex->D_to_C) {
            TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'D',buf,
                        proj->my_prefac_reco,proj->nr,bufMPI));
            TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
            TICK_TOCK(time_handler_ft, GPUproj_handler_ft(&(proj->proj_descr), 'C', bufMPI, buf,
                    (index_t*)proj->DC_CD_to_padded, proj->DC_CD_from, proj->DC_CD_mom,
                    proj->DC_CD_to_padded_sz, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->my_nk, proj->nr));
            PROJ_HOOK('D', 'C');
            TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        }
        timings(time_cpyfill,time_reco,time_handler_ft,time_add);
        return;
    }
    #endif // USE_CUDA

    if(vertex->D_flow && vertex->D_to_C) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'D',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI,buf, proj->DC_CD_to,
                proj->DC_CD_to_sz, proj->DC_CD_from,
                proj->DC_CD_mom, proj->DC_CD_displ,
                proj->DC_CD_count, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('D', 'C');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_C) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'P',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI,buf, proj->PC_CP_to,
                proj->PC_CP_to_sz, proj->PC_CP_from,
                proj->PC_CP_mom, proj->PC_CP_displ,
                proj->PC_CP_count, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('P', 'C');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void P_projection_no_GPU(complex128_t* proj_vertex, const tu_vertex_t* vertex, const tu_projection_t* proj) {
    if(!vertex->P_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Pch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->C_flow && vertex->C_to_P) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'C',buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->PC_CP_to,
            proj->PC_CP_to_sz, proj->PC_CP_from,
            proj->PC_CP_mom, proj->PC_CP_displ,
            proj->PC_CP_count, proj->my_prefac_proj, vertex->n_orbff,
            vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));

        PROJ_HOOK('C', 'P');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
               vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->D_flow && vertex->D_to_P) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'D',buf,
                    proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_P(bufMPI, buf, proj->D_to_P_to,
            proj->D_to_P_to_sz, proj->D_to_P_from,
            proj->D_to_P_mom, proj->D_to_P_displ,
            proj->D_to_P_count, proj->my_prefac_proj, vertex->n_orbff,
            vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('D', 'P');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
            vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void D_projection_no_GPU(complex128_t* proj_vertex, const tu_vertex_t* vertex, const tu_projection_t* proj) {
    if(!vertex->D_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Dch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->C_flow && vertex->C_to_D) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'C',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI, buf, proj->DC_CD_to,
                    proj->DC_CD_to_sz, proj->DC_CD_from,
                    proj->DC_CD_mom, proj->DC_CD_displ,
                    proj->DC_CD_count, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('C', 'D');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_D) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'P',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_D(bufMPI, buf, proj->P_to_D_to,
                proj->P_to_D_to_sz, proj->P_to_D_from,
                proj->P_to_D_mom, proj->P_to_D_displ,
                proj->P_to_D_count, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('P', 'D');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}

void C_projection_no_GPU(complex128_t* proj_vertex, const tu_vertex_t* vertex, const tu_projection_t* proj) {
    if(!vertex->C_flow) return;
    double time_cpyfill = 0.0, time_reco = 0.0, time_handler_ft = 0.0, time_add = 0.0;
    TICK_TOCK(time_cpyfill, memcpy(proj_vertex, vertex->Cch, sizeof*proj_vertex * vertex->full_vert_size));
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    if(vertex->D_flow && vertex->D_to_C) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'D',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI, buf, proj->DC_CD_to,
                proj->DC_CD_to_sz, proj->DC_CD_from,
                proj->DC_CD_mom, proj->DC_CD_displ,
                proj->DC_CD_count, proj->my_prefac_proj, vertex->n_orbff,
                vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('D', 'C');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    if(vertex->P_flow && vertex->P_to_C) {
        TICK_TOCK(time_reco, diverge_tu_vertex_reco_ft( vertex, 'P',
                    buf,proj->my_prefac_reco,proj->nr,bufMPI));
        TICK_TOCK(time_cpyfill, ZERO_OUT(bufMPI,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
        TICK_TOCK(time_handler_ft, _to_C(bufMPI, buf, proj->PC_CP_to,
                    proj->PC_CP_to_sz, proj->PC_CP_from,
                    proj->PC_CP_mom, proj->PC_CP_displ,
                    proj->PC_CP_count, proj->my_prefac_proj, vertex->n_orbff,
                    vertex->n_spin, vertex->nk, vertex->my_nk, proj->nr));
        PROJ_HOOK('P', 'C');
        TICK_TOCK(time_add, complex128_array_add(proj_vertex, bufMPI,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin))));
    }
    timings(time_cpyfill,time_reco,time_handler_ft,time_add);
}
