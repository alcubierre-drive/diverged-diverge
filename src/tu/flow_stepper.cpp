/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "flow_stepper.hpp"
#include "vertex_details.h"
#include "../misc/mpi_functions.h"

tu_stepper::tu_stepper(projection_t* projin, tu_prop_t* loopin, tu_vertex_t* vertexin)
:
    loop{loopin},
    proj{projin},
    prodsize{vertexin->n_orbff*POW2(vertexin->n_spin)},
    nk{vertexin->nk},
    my_nk{vertexin->my_nk},
    tu_selfenergy_flow{vertexin->tu_data->tu_selfenergy_flow}
{
    channel_selection[0] = vertexin->P_flow;
    channel_selection[1] = vertexin->C_flow;
    channel_selection[2] = vertexin->D_flow;
    channel_selection[3] = vertexin->use_self;
    if(vertexin->SU2) {
        norm[0] = -1.0;
        norm[1] = -1.0;
        norm[2] =  2.0;
    }else{
        norm[0] = -0.5;
        norm[1] = -1.0;
        norm[2] =  1.0;
    }
    internal_vertex=tu_vertex_copy(vertexin);

    if(channel_selection[3]){
        coarseGF = (complex128_t*)calloc(nk*POW2(vertexin->n_orb*vertexin->n_spin),sizeof(complex128_t));
        self = selfenergy_init(vertexin);
    }
}

vector<double> tu_stepper::tu_step_euler(tu_vertex_t* inout, double lambda, double dlambda) {
    double flow_step_tick = diverge_mpi_wtime();
    double tick, tock;
    num_calls += 1;
    tick = diverge_mpi_wtime();
    // create GF buffers here s.t. the buffers can be used in
    // loop->p?_loop *and* be overwritten
    if (channel_selection[3]) {
        tu_prop_GF_zero_T(loop, lambda*I128, inout->self_en);
        tu_prop_GF_for_self(loop, coarseGF);
    } else {
        tu_prop_GF_zero_T(loop, lambda*I128, NULL);
    }
    tock = diverge_mpi_wtime();
    w_time_GF += tock-tick;
    if(!tu_selfenergy_flow) {
        if(inout->SU2){
            if(channel_selection[1] || channel_selection[2]){
                tick = diverge_mpi_wtime();
                tu_prop_ph_loop( loop, internal_vertex->ph_bubble_int, internal_vertex->bubble_helper );
                tock = diverge_mpi_wtime();
                w_time_loop += tock-tick;
                if(channel_selection[2] && channel_selection[1]){
                    tick = diverge_mpi_wtime();
                    C_projection(internal_vertex->Dch, inout, proj);
                    tock = diverge_mpi_wtime();
                    w_time_projection += tock-tick;

                    tick = diverge_mpi_wtime();
                    tu_vertex_product(internal_vertex->Cch, internal_vertex->Dch,
                                internal_vertex->ph_bubble_int, inout->product_helper,
                                prodsize,my_nk,norm[1], true, internal_vertex->GPU_gemm_buffer);
                    tock = diverge_mpi_wtime();
                    w_time_product += tock-tick;

                    tick = diverge_mpi_wtime();
                    D_projection(inout->projection_helper, inout, proj);
                    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
                    for(index_t ii = 0; ii <  inout->full_vert_size;++ii)
                        internal_vertex->projection_helper[ii] -= 0.5*internal_vertex->Dch[ii];
                    tock = diverge_mpi_wtime();
                    w_time_projection += tock-tick;

                    tick = diverge_mpi_wtime();
                    tu_vertex_product(internal_vertex->Dch, internal_vertex->projection_helper,
                                internal_vertex->ph_bubble_int,inout->product_helper,
                                prodsize,my_nk,norm[2], true, internal_vertex->GPU_gemm_buffer);
                    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
                    for(index_t ii = 0; ii <  inout->full_vert_size;++ii)
                        internal_vertex->Dch[ii] += 0.5*internal_vertex->Cch[ii];
                    tock = diverge_mpi_wtime();
                    w_time_product += tock-tick;
                }else if(channel_selection[1]){
                    tick = diverge_mpi_wtime();
                    C_projection(internal_vertex->projection_helper, inout, proj);
                    tock = diverge_mpi_wtime();
                    w_time_projection += tock-tick;

                    tick = diverge_mpi_wtime();
                    tu_vertex_product(internal_vertex->Cch, internal_vertex->projection_helper,
                                internal_vertex->ph_bubble_int,inout->product_helper,
                                prodsize,my_nk,norm[1], true, internal_vertex->GPU_gemm_buffer);
                    tock = diverge_mpi_wtime();
                    w_time_product += tock-tick;
                }else{
                    tick = diverge_mpi_wtime();
                    D_projection(internal_vertex->projection_helper, inout, proj);
                    tock = diverge_mpi_wtime();
                    w_time_projection += tock-tick;

                    tick = diverge_mpi_wtime();
                    tu_vertex_product(internal_vertex->Dch, internal_vertex->projection_helper,
                                internal_vertex->ph_bubble_int,inout->product_helper,
                                prodsize,my_nk,norm[2], true, internal_vertex->GPU_gemm_buffer);
                    tock = diverge_mpi_wtime();
                    w_time_product += tock-tick;
                }
            }
        }else{
            tick = diverge_mpi_wtime();
            if(channel_selection[1])
                C_projection(internal_vertex->projection_helper, inout, proj);
            tock = diverge_mpi_wtime();
            w_time_projection += tock-tick;


            tick = diverge_mpi_wtime();
            if(channel_selection[1] || channel_selection[2]) {
                tu_prop_ph_loop( loop, internal_vertex->ph_bubble_int, internal_vertex->bubble_helper );
            }
            tock = diverge_mpi_wtime();
            w_time_loop += tock-tick;

            tick = diverge_mpi_wtime();
            if(channel_selection[1])
                tu_vertex_product(internal_vertex->Cch, internal_vertex->projection_helper,
                        internal_vertex->ph_bubble_int,inout->product_helper,
                        prodsize,my_nk,norm[1], true, internal_vertex->GPU_gemm_buffer);
            tock = diverge_mpi_wtime();
            w_time_product += tock-tick;

            tick = diverge_mpi_wtime();
            if(channel_selection[2])
                D_projection(internal_vertex->projection_helper, inout, proj);
            tock = diverge_mpi_wtime();
            w_time_projection += tock-tick;

            tick = diverge_mpi_wtime();
            if(channel_selection[2])
                tu_vertex_product(internal_vertex->Dch,internal_vertex->projection_helper,
                            internal_vertex->ph_bubble_int, inout->product_helper,
                            prodsize,my_nk,norm[2], true, internal_vertex->GPU_gemm_buffer);
            tock = diverge_mpi_wtime();
            w_time_product += tock-tick;
        }
        if(channel_selection[0]) {
            tick = diverge_mpi_wtime();
            P_projection(inout->projection_helper, inout, proj);
            tock = diverge_mpi_wtime();
            w_time_projection += tock-tick;

            tick = diverge_mpi_wtime();
            tu_prop_pp_loop( loop, internal_vertex->pp_bubble_int, internal_vertex->bubble_helper );
            tock = diverge_mpi_wtime();
            w_time_loop += tock-tick;

            tick = diverge_mpi_wtime();
            tu_vertex_product(internal_vertex->Pch, inout->projection_helper,
                        internal_vertex->pp_bubble_int,inout->product_helper,
                        prodsize,my_nk,norm[0], true, internal_vertex->GPU_gemm_buffer);
            tock = diverge_mpi_wtime();
            w_time_product += tock-tick;
        }
    }else{
        if(channel_selection[0])
            memset((void*)(internal_vertex->Pch),0,sizeof(complex128_t)*internal_vertex->full_vert_size);
        if(channel_selection[1])
            memset((void*)(internal_vertex->Dch),0,sizeof(complex128_t)*internal_vertex->full_vert_size);
        if(channel_selection[2])
            memset((void*)(internal_vertex->Cch),0,sizeof(complex128_t)*internal_vertex->full_vert_size);
    }
    if (channel_selection[3]) {
        tick = diverge_mpi_wtime();
        selfenergy_flow( self, internal_vertex->self_en, inout->Pch,
                        inout->Cch, inout->Dch, coarseGF, 1. );
        tock = diverge_mpi_wtime();
        w_time_self += tock-tick;
    }
    tick = diverge_mpi_wtime();
    tu_vertex_add(inout, internal_vertex, dlambda);
    vector<double> ret = {0,0,0,0,0,0};
    tu_vertex_max(inout, ret.data());

    tock = diverge_mpi_wtime();
    w_time_add_find_max += tock-tick;
    w_time_full += diverge_mpi_wtime() - flow_step_tick;
    return ret;
}
