/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "flow_stepper.h"
#include "flow_stepper_details.h"

#include "vertex.h"
#include "vertex_details.h"
#include "propagator.h"
#include "propagator_details.h"
#include "projection.h"
#include "projection_handler.h"

#include "diverge_interface.h"
#include "selfenergy.h"
#include "../grid/helpers.h" // TICK_TOCK macro

tu_stepper_t* tu_stepper_init( tu_projection_t* proj, tu_prop_t* loop, tu_vertex_t* vert ) {
    tu_stepper_t* s = calloc(1, sizeof*s);
    s->loop = loop;
    s->proj = proj;
    s->prodsize = vert->n_orbff*POW2(vert->n_spin);
    s->nk = vert->nk;
    s->my_nk = vert->my_nk;
    s->tu_selfenergy_flow = vert->tu_data->tu_selfenergy_flow;
    s->channel_selection[0] = vert->P_flow;
    s->channel_selection[1] = vert->C_flow;
    s->channel_selection[2] = vert->D_flow;
    s->channel_selection[3] = vert->use_self;
    if (vert->SU2) {
        s->norm[0] = s->norm[1] = -1.0;
        s->norm[2] =  2.0;
    } else {
        s->norm[0] = -0.5;
        s->norm[1] = -1.0;
        s->norm[2] =  1.0;
    }
    s->internal_vertex = tu_vertex_copy(vert);
    s->loop_func = NULL;

    if (s->channel_selection[3]) {
        s->coarseGF = calloc(s->nk*POW2(vert->n_orb*vert->n_spin),sizeof(complex128_t));
        s->self = tu_selfenergy_init(vert);
    }
    return s;
}

void tu_stepper_free( tu_stepper_t* s ) {
    tu_vertex_free( s->internal_vertex );
    if (s->channel_selection[3]) {
        tu_selfenergy_free(s->self);
        free(s->coarseGF);
    }
    free( s );
}

void tu_stepper_euler( tu_stepper_t* s, tu_vertex_t* inout, double lambda, double dlambda, double maxs[6] ) {
    double flow_step_tick = diverge_mpi_wtime();
    s->num_calls += 1;
    // create GF buffers here s.t. the buffers can be used in
    // loop->p?_loop *and* be overwritten
    if (s->channel_selection[3]) {
        tu_prop_GF_zero_T(s->loop, lambda*I, inout->self_en);
        tu_prop_GF_for_self(s->loop, s->coarseGF);
    } else if (s->loop_func == NULL) {
        tu_prop_GF_zero_T(s->loop, lambda*I, NULL);
    }
    s->w_time_GF += diverge_mpi_wtime()-flow_step_tick;
    if (!s->tu_selfenergy_flow) {
        if (inout->SU2) {
            if (s->channel_selection[1] || s->channel_selection[2]) {
                if (s->loop_func == NULL) {
                    TICK_TOCK( s->w_time_loop, tu_prop_ph_loop( s->loop, s->internal_vertex->ph_bubble_int,
                               s->internal_vertex->bubble_helper ) );
                } else {
                    TICK_TOCK( s->w_time_loop, (*s->loop_func)('H', s->internal_vertex->ph_bubble_int,
                               s->loop_func_ctx) );
                }
                if (s->channel_selection[2] && s->channel_selection[1]) {
                    TICK_TOCK( s->w_time_projection, C_projection(s->internal_vertex->Dch, inout, s->proj) );

                    TICK_TOCK( s->w_time_product, tu_vertex_product(s->internal_vertex->Cch,
                                s->internal_vertex->Dch, s->internal_vertex->ph_bubble_int,
                                inout->product_helper, s->prodsize,s->my_nk,s->norm[1],
                                true, s->internal_vertex->GPU_gemm_buffer) );

                    TICK_TOCK( s->w_time_projection, D_projection(inout->projection_helper, inout, s->proj) );

                    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
                    for (index_t ii=0; ii<inout->full_vert_size; ++ii)
                        s->internal_vertex->projection_helper[ii] -= 0.5*s->internal_vertex->Dch[ii];

                    TICK_TOCK( s->w_time_product, tu_vertex_product(s->internal_vertex->Dch,
                                s->internal_vertex->projection_helper,
                                s->internal_vertex->ph_bubble_int,inout->product_helper,
                                s->prodsize,s->my_nk,s->norm[2], true, s->internal_vertex->GPU_gemm_buffer) );

                    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
                    for (index_t ii=0; ii<inout->full_vert_size; ++ii)
                        s->internal_vertex->Dch[ii] += 0.5*s->internal_vertex->Cch[ii];

                } else if (s->channel_selection[1]) {
                    TICK_TOCK( s->w_time_projection, C_projection(s->internal_vertex->projection_helper,
                                inout, s->proj) );

                    TICK_TOCK( s->w_time_product, tu_vertex_product(s->internal_vertex->Cch,
                                s->internal_vertex->projection_helper,
                                s->internal_vertex->ph_bubble_int,inout->product_helper,
                                s->prodsize,s->my_nk,s->norm[1], true, s->internal_vertex->GPU_gemm_buffer) );
                } else {
                    TICK_TOCK( s->w_time_projection, D_projection(s->internal_vertex->projection_helper,
                                inout, s->proj) );

                    TICK_TOCK( s->w_time_product, tu_vertex_product(s->internal_vertex->Dch,
                                s->internal_vertex->projection_helper,
                                s->internal_vertex->ph_bubble_int,inout->product_helper,
                                s->prodsize,s->my_nk,s->norm[2], true, s->internal_vertex->GPU_gemm_buffer) );
                }
            }
        } else {
            if (s->channel_selection[1])
                TICK_TOCK( s->w_time_projection, C_projection(s->internal_vertex->projection_helper,
                            inout, s->proj) );

            if (s->channel_selection[1] || s->channel_selection[2]) {
                if (s->loop_func == NULL) {
                    TICK_TOCK( s->w_time_loop, tu_prop_ph_loop( s->loop, s->internal_vertex->ph_bubble_int,
                               s->internal_vertex->bubble_helper ) );
                } else {
                    TICK_TOCK( s->w_time_loop, (*s->loop_func)('H', s->internal_vertex->ph_bubble_int,
                               s->loop_func_ctx) );
                }
            }

            if (s->channel_selection[1])
                TICK_TOCK( s->w_time_product, tu_vertex_product(s->internal_vertex->Cch,
                            s->internal_vertex->projection_helper, s->internal_vertex->ph_bubble_int,
                            inout->product_helper, s->prodsize,s->my_nk,s->norm[1], true,
                            s->internal_vertex->GPU_gemm_buffer) );

            if (s->channel_selection[2])
                TICK_TOCK( s->w_time_projection, D_projection(s->internal_vertex->projection_helper,
                            inout, s->proj) );

            if (s->channel_selection[2])
                TICK_TOCK( s->w_time_product, tu_vertex_product(s->internal_vertex->Dch,
                            s->internal_vertex->projection_helper, s->internal_vertex->ph_bubble_int,
                            inout->product_helper, s->prodsize,s->my_nk,s->norm[2], true,
                            s->internal_vertex->GPU_gemm_buffer) );
        }
        if (s->channel_selection[0]) {
            TICK_TOCK( s->w_time_projection, P_projection(inout->projection_helper, inout, s->proj) );

            if (s->loop_func == NULL) {
                TICK_TOCK( s->w_time_loop, tu_prop_pp_loop( s->loop, s->internal_vertex->pp_bubble_int,
                           s->internal_vertex->bubble_helper ) );
            } else {
                TICK_TOCK( s->w_time_loop, (*s->loop_func)('P', s->internal_vertex->pp_bubble_int,
                           s->loop_func_ctx) );
            }

            TICK_TOCK( s->w_time_product, tu_vertex_product(s->internal_vertex->Pch,
                        inout->projection_helper, s->internal_vertex->pp_bubble_int,inout->product_helper,
                        s->prodsize,s->my_nk,s->norm[0], true, s->internal_vertex->GPU_gemm_buffer) );
        }
    } else {
        if (s->channel_selection[0]) ZERO_OUT( s->internal_vertex->Pch, s->internal_vertex->full_vert_size );
        if (s->channel_selection[1]) ZERO_OUT( s->internal_vertex->Dch, s->internal_vertex->full_vert_size );
        if (s->channel_selection[2]) ZERO_OUT( s->internal_vertex->Cch, s->internal_vertex->full_vert_size );
    }
    if (s->channel_selection[3]) {
        TICK_TOCK( s->w_time_self, tu_selfenergy_flow( s->self, s->internal_vertex->self_en,
                    inout->Pch, inout->Cch, inout->Dch, s->coarseGF, 1. ) );
    }
    TICK_TOCK( s->w_time_add_find_max, tu_vertex_add(inout, s->internal_vertex, dlambda),
                                       tu_vertex_max(inout, maxs) );

    s->w_time_full += diverge_mpi_wtime()-flow_step_tick;
}

diverge_timing_t* tu_stepper_timings( tu_stepper_t* s, int* num ) {
    int cnt; if (!num) num = &cnt; *num = 0;
    #define push_timing( name ) { \
        s->timings[(*num)++] = (diverge_timing_t){ .t=s->w_time_##name, .nam=#name }; \
    }
    push_timing( GF );
    push_timing( loop );
    push_timing( projection );
    push_timing( product );
    push_timing( add_find_max );
    push_timing( self );
    push_timing( full );
    int nprop = 0;
    diverge_timing_t* tprop = tu_prop_get_extra_timings( s->loop, &nprop );
    for (int i=0; i<nprop; ++i)
        s->timings[(*num)++] = tprop[i];
    return s->timings;
}

index_t tu_stepper_num_calls( const tu_stepper_t* s ) {
    return s->num_calls;
}

tu_vertex_t* tu_stepper_internal_vertex( tu_stepper_t* s ) {
    return s->internal_vertex;
}
