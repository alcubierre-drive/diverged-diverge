/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_flow_step_internal.h"
#include "diverge_flow_type.h"
#include "misc/batched_gemms.h"
#include "misc/batched_eigen.h"
#include "misc/batched_svd.h"
#include "misc/batched_zgemd.h"
#include "misc/eigen.h"
#include "misc/mergesort.h"
#include "diverge_filling.h"
#include "diverge_model_internals.h"
#include "misc/dependency_tree.h"
#include "misc/shared_mem.h"
#include "misc/arnoldi.h"

#include "tu/vertex_details.h"
#include "tu/phonons.h"

#include "grid/vertex_memory.h"

#include "npatch/vertex.h"
#include "npatch/kernels_gpu.h"
#include "npatch/loop.h"

#include <float.h>
#include <limits.h>
#include <assert.h>
#include <omp.h>

static void diverge_flow_step_select_channels( diverge_flow_step_t* st, const char* chans_ );

diverge_flow_step_t* diverge_flow_step_init_any( diverge_model_t* mod, const char* channels ) {
    return diverge_flow_step_init( mod, mod->internals->backend, channels );
}

diverge_flow_step_t* diverge_flow_step_init( diverge_model_t* mod, const char* method, const char* channels ) {
    diverge_flow_step_t* st = calloc(1, sizeof(diverge_flow_step_t));
    st->mod = mod;
    st->intls = mod->internals;
    st->step_eig_abort = INFINITY;
    for (int i=0; i<5; ++i) st->step_eig[i] = -1.0;

    dependency_tree_insert_flow(st);

    st->type = diverge_flow_type_fromstr( method );
    diverge_flow_step_select_channels( st, channels );

    if (st->type == diverge_patch) {
        npatch_setup_arch( mod->internals->patch_gpus, mod->internals->patch_ngpus,
                mod->internals->patch_nblocks, mod->internals->patch_gpu_workshare,
                mod->internals->patch_gpu_loop_workshare );

        if (mod->internals->patch_loop_gpu_weights)
            npatch_loop_gpu_set_weights( mod->internals->patch_loop_gpu_weights );
        if (mod->internals->patch_gpu_weights)
            npatch_flow_gpu_set_weights( mod->internals->patch_gpu_weights );
    }
    // DEVELOPERS
    //
    // If you have global (de)initialization routines that don't allocate
    // massive amounts of memory or take significant time they can be called
    // analogous to the above routine

    st->niter = 0;

    // DEVELOPERS
    //
    // For more costly initialization routines, add a new case to the below
    // switch() {} statement with all your initialization code. You may operate
    // on all structures given in diverge_model_t as well as its internals and
    // your code specific internals.
    switch (st->type) {
        case diverge_grid:
            if (st->intls->grid_vertex == NULL) {
                mpi_err_printf("internals for grid FRG not set\n");
                goto end_error;
            }
            st->grid_loop = grid_loop_init(st->mod);
            st->grid_step = grid_flow_step_init( (grid_vertex_t*)st->intls->grid_vertex,
                    st->grid_loop, st->channel_selection[0], st->channel_selection[1],
                    st->channel_selection[2] );
            diverge_mpi_barrier();
            break;
        case diverge_patch:
            if (st->intls->patch_vertex == NULL || st->mod->patching == NULL) {
                mpi_err_printf("internals and patching for patch FRG not set\n");
                goto end_error;
            }
            st->patch_loop = npatch_loop_init( st->mod );
            st->patch_step_conf = npatch_flow_step_conf_init( st->mod->SU2 > 0 );
            st->patch_step_conf->P = st->channel_selection[0];
            st->patch_step_conf->C = st->channel_selection[1];
            st->patch_step_conf->D = st->channel_selection[2];
            break;
        case diverge_tu:
            if (st->intls->tu_data == NULL) {
                mpi_err_printf("internals for TU FRG not set\n");
                goto end_error;
            }
            st->tu_proj = tu_projection_init(mod);
            st->tu_loop = tu_prop_init(mod);
            st->tu_vertex = tu_vertex_init(mod, st->tu_proj, st->tu_loop, st->channel_selection[0],
                    st->channel_selection[1], st->channel_selection[2], st->channel_selection[3]);
            st->tu_step = tu_stepper_init(st->tu_proj, st->tu_loop, st->tu_vertex);
            break;
        case diverge_tu_phonons:
            if (!st->intls->tu_data || !st->intls->tu_data_phonons) {
                mpi_err_printf("internals for TU FRG × Phonons not set\n");
                goto end_error;
            }
            st->tu_proj = tu_projection_init(mod);
            st->tu_loop = tu_prop_init(mod);
            st->tu_vertex = tu_vertex_init(mod, st->tu_proj, st->tu_loop, st->channel_selection[0],
                    st->channel_selection[1], st->channel_selection[2], st->channel_selection[3]);
            tu_vertex_init_phonons(st->tu_vertex, st->intls->tu_data_phonons);
            st->tu_step = tu_stepper_init(st->tu_proj, st->tu_loop, st->tu_vertex);
            break;
        default:
        case diverge_flow_type_invalid:
            mpi_err_printf("invalid flow type '%s', use 'grid', 'patch', 'tu', 'tu_phonons'.\n", method);
            goto end_error;
            break;
    }
    return st;

// we want to clean up when we encounter an error
end_error:
    diverge_flow_step_free(st);
    return NULL;
}

static void diverge_flow_step_select_channels( diverge_flow_step_t* st, const char* chans_ ) {
    st->channel_selection[0] = st->channel_selection[1] = st->channel_selection[2] =
                               st->channel_selection[3] = false;
    for (const char* c = chans_; *c; ++c) {
        switch (*c) {
            case 'P':
                st->channel_selection[0] = true;
                break;
            case 'C':
                st->channel_selection[1] = true;
                break;
            case 'D':
                st->channel_selection[2] = true;
                break;
            case 'S':
                st->channel_selection[3] = true;
                break;
            default:
                break;
        }
    }
}

void diverge_flow_step_free( diverge_flow_step_t* st ) {
    // DEVELOPERS
    //
    // For a new backend, you can add deinitailization code in the below
    // switch() {} and global deinitailization code anywhere in this function
    switch (st->type) {
        case diverge_patch:
            free( st->patch_step_conf );
            npatch_loop_free( st->patch_loop );
            npatch_setup_teardown();
            break;
        case diverge_grid:
            grid_flow_step_free( st->grid_step );
            grid_loop_free( st->grid_loop );
            break;
        case diverge_tu_phonons:
        case diverge_tu:
            tu_vertex_free( st->tu_vertex );
            tu_projection_free( st->tu_proj );
            tu_stepper_free( st->tu_step );
            tu_prop_free( st->tu_loop );
            if (st->tu_rpa) tu_rpa_free( st->tu_rpa );
            break;
        default:
        case diverge_flow_type_invalid:
            break;
    }
    VECTOR_FREE( st->timings );

    dependency_tree_remove_flow(st);

    free(st);
    batched_gemm_cublas_clear();
    batched_gemm_cublas_w_buffer_clear();
}

void diverge_flow_step_euler( diverge_flow_step_t* st, double Lambda, double dLambda ) {
    double chanmax[6];
    int idx = 0;
    diverge_timing_t* ttt = NULL;
    st->Lambda = Lambda;
    st->dLambda = dLambda;
    for (int i=0; i<5; ++i) st->step_eig[i] = -1.0;
    // DEVELOPERS
    //
    // This is where the heavy lifiting happens. Depending on the backend, we
    // branch to one of the case statements. Add your backend as another case
    // statement and perform an Euler flow step there. Use all the structures
    // defined in diverge_flow_step_t, diverge_model_t and its internals struct.
    //
    // Please save
    // (i)   timing information in st->timings (times as doubles) and
    //       st->timings_descr (descriptions as strings)
    // (ii)  the channel maxima (P,C,D) in st->chanmax
    // (iii) the vertex maximum in st->vertmax
    // (iv)  the loop maxima in st->loopmax
    // (v)   the number of total iterations in st->niter
    //
    // Note that dLambda is negative. Moreover, you are responsible to call the
    // Green's function generator; the Green's function at a given frequency is
    // the only object your code may interface to. It is calculated by calling
    //
    // complex128_t cmplx_Lambda(0, st->Lambda);
    // greensfunc_op_t where = (*st->mod->gfill)( st->mod, cmplx_Lambda, st->intls->greens );
    //
    // The above lines fill the Green's functions at cmplx_Lambda and
    // std::conj(cmplx_Lambda) into the buffer st->intls->greens. You are free
    // to fill the Greens functions into your own buffers. Moreover, you must
    // check for the return value of the Green's function filler (i.e. where)
    // and act accordingly. See src/npatch/loop.c or src/grid/loops.c for
    // examples.
    switch (st->type) {
        case diverge_patch:
            npatch_flow_step( st->patch_step_conf, (npatch_vertex_t*)st->intls->patch_vertex,
                    st->patch_loop, &Lambda, dLambda, chanmax );
            st->loopmax[0] = chanmax[0];
            st->loopmax[1] = chanmax[1];
            st->chanmax[0] = chanmax[2];
            st->chanmax[1] = chanmax[3];
            st->chanmax[2] = chanmax[4];
            st->vertmax = chanmax[5];

            // save timings
            st->timings_sz = 0;
            VECTOR_RESERVE( st->timings, 8 + 3*128 );
            #define PATCH_TIMINGS_PUSH_BACK( name ) \
                VECTOR_PUSH_BACK(st->timings, (diverge_timing_t){ \
                        .t = st->patch_step_conf->name, \
                        .nam = #name, \
                });
            PATCH_TIMINGS_PUSH_BACK( wtime_gpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_cpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_comm )
            PATCH_TIMINGS_PUSH_BACK( wtime_max )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_gpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_cpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_comm )
            PATCH_TIMINGS_PUSH_BACK( wtime_full )
            for (int i=0; i<128; ++i) {
                VECTOR_PUSH_BACK(st->timings, (diverge_timing_t){.t=st->patch_step_conf->wtime_gpu_list[i]});
                snprintf(st->timings[st->timings_sz-1].nam, 55, "wtime_gpu_%03i", i);
                VECTOR_PUSH_BACK(st->timings, (diverge_timing_t){.t=st->patch_step_conf->wtime_loop_gpu_list[i]});
                snprintf(st->timings[st->timings_sz-1].nam, 55, "wtime_loop_gpu_%03i", i);
                VECTOR_PUSH_BACK(st->timings, (diverge_timing_t){.t=st->patch_step_conf->wtime_loop_gpu_gf_list[i]});
                snprintf(st->timings[st->timings_sz-1].nam, 55, "wtime_gf_gpu_%03i", i);
            }
            st->niter = st->patch_step_conf->niter;
            break;
        case diverge_grid:
            memcpy( chanmax, grid_flow_step_euler( st->grid_step, Lambda, dLambda ), sizeof(double)*6 );
            st->chanmax[0] = chanmax[0];
            st->chanmax[1] = chanmax[1];
            st->chanmax[2] = chanmax[2];
            st->vertmax = chanmax[3];
            st->loopmax[0] = chanmax[4];
            st->loopmax[1] = chanmax[5];

            VECTOR_RESIZE(st->timings, 6);
            memcpy(st->timings, grid_flow_step_timings(st->grid_step), sizeof(diverge_timing_t)*6);
            st->niter = grid_flow_step_num_calls( st->grid_step );
            break;
        case diverge_tu_phonons:
        case diverge_tu:
            if (!tu_stepper_euler( st->tu_step, st->tu_vertex, Lambda, dLambda, chanmax,
                        st->step_eig_on ? st : NULL ) && st->tu_rpa)
                tu_rpa_step( st->tu_rpa, Lambda, dLambda );

            st->chanmax[0] = chanmax[0];
            st->chanmax[1] = chanmax[1];
            st->chanmax[2] = chanmax[2];
            st->vertmax = MAX(MAX(chanmax[0], chanmax[1]), chanmax[2]);
            st->loopmax[0] = chanmax[3];
            st->loopmax[1] = chanmax[4];

            ttt = tu_stepper_timings(st->tu_step, &idx);
            VECTOR_RESERVE( st->timings, idx+1 );
            memcpy( st->timings, ttt, sizeof*st->timings * (st->timings_sz=idx) );
            if (st->tu_rpa) { VECTOR_PUSH_BACK( st->timings, (diverge_timing_t){
                    .t=st->tu_rpa->time_step,
                    .nam="tu_rpa_step",
            }); }

            st->niter = tu_stepper_num_calls( st->tu_step );
            break;
        default:
        case diverge_flow_type_invalid:
            break;
    }
    VECTOR_PUSH_BACK(st->timings, (diverge_timing_t){
            .t=st->t_eigchan_diag,
            .nam="eigchan_diag"
    });
}

index_t diverge_flow_step_niter( diverge_flow_step_t* st ) {
    return st->niter;
}
index_t diverge_flow_step_ntimings( diverge_flow_step_t* st ) {
    return st->timings_sz;
}
diverge_timing_t* diverge_flow_step_timings( diverge_flow_step_t* st ) {
    return st->timings;
}
double diverge_flow_step_timing( diverge_flow_step_t* st, index_t idx ) {
    return st->timings[idx].t;
}
const char* diverge_flow_step_timing_descr( diverge_flow_step_t* st, index_t idx ) {
    return st->timings[idx].nam;
}

static double dvg_timings[512] = {0};
double* diverge_flow_step_timing_vec( diverge_flow_step_t* st ) {
    for (index_t i=0; i<st->timings_sz; ++i)
        if (i < 511) dvg_timings[i] = st->timings[i].t;
    return dvg_timings;
}

static char dvg_timings_descr[512][64] = {0};
char** diverge_flow_step_timing_descr_vec( diverge_flow_step_t* st ) {
    for (index_t i=0; i<st->timings_sz; ++i)
        if (i < 511) strncpy( dvg_timings_descr[i], st->timings[i].nam, 63 );
    return (char**)dvg_timings_descr;
}

double diverge_flow_step_vertmax_r( diverge_flow_step_t* st, double* vertmax ) {
    double vmax = 0; diverge_flow_step_vertmax( st, &vmax );
    if (vertmax) *vertmax = vmax;
    return vmax;
}

void diverge_flow_step_vertmax( diverge_flow_step_t* st, double vertmax[1] ) {
    memcpy( vertmax, &st->vertmax, sizeof(double) );
}
void diverge_flow_step_loopmax( diverge_flow_step_t* st, double loopmax[2] ) {
    memcpy( loopmax, st->loopmax, sizeof(double)*2 );
}
void diverge_flow_step_chanmax( diverge_flow_step_t* st, double chanmax[3] ) {
    memcpy( chanmax, st->chanmax, sizeof(double)*3 );
}

double diverge_flow_step_lambda( diverge_flow_step_t* st ) {
    return st->Lambda;
}
double diverge_flow_step_dlambda( diverge_flow_step_t* st ) {
    return st->dLambda;
}

diverge_flow_step_vertex_t diverge_flow_step_vertex( diverge_flow_step_t* st, char chan ) {
    diverge_flow_step_vertex_t result = { .backend = 'I', .channel = chan };

    diverge_model_t* mod = st->mod;
    internals_t* intls = st->intls;

    grid_vertex_t* grid_V = (grid_vertex_t*)intls->grid_vertex;
    npatch_vertex_t* patch_V = (npatch_vertex_t*)intls->patch_vertex;
    tu_vertex_t* tu_V = st->tu_vertex;
    tu_projection_t* tu_Proj = st->tu_proj;

    // DEVELOPERS
    //
    // You may add a routine to access the vertex objects (as pointers) during
    // the flow here. See the three case: statements for the existing backends.
    // If you wish not to do so, you can put your backend identifier below the
    // default: case
    switch (st->type) {
        case diverge_grid:
            if (toupper(chan) != 'P' &&
                toupper(chan) != 'C' &&
                toupper(chan) != 'D' &&
                toupper(chan) != 'V') {
                mpi_wrn_printf("for grid vertex chan must be one of 'PCDVpcdv'. default: 'D'.\n");
                result.channel = 'D';
            }
            result.ary = grid_V->vertex_buffers[2];
            grid_vertex_project_chans( grid_V, islower(chan) ? grid_V->vertex_buffers[1] :
                    grid_V->vertex_buffers[0], result.ary, 'D', toupper(result.channel), false, 1.0 );
            result.q_0 = grid_V->qdispl;
            result.q_1 = result.q_0 + grid_V->qcount;
            result.nk = intls->grid_nk;
            result.n_orbff = mod->n_orb;
            result.n_spin = mod->n_spin;
            result.backend = 'G';
            break;
        case diverge_patch:
            if (chan != 'V' && chan != 'v') {
                mpi_wrn_printf("for patch vertex chan must be 'V' or 'v'. default: 'V.'\n");
                result.channel = 'V';
            }
            result.ary = (result.channel == 'v') ? patch_V->dV : patch_V->V;
            result.q_0 = 0;
            result.q_1 = mod->patching->n_patches;
            result.nk = mod->patching->n_patches;
            result.n_orbff = mod->n_orb;
            result.n_spin = mod->n_spin;
            result.backend = 'P';
            break;
        case diverge_tu_phonons:
        case diverge_tu:
            if (toupper(chan) != 'P' &&
                toupper(chan) != 'C' &&
                toupper(chan) != 'D' &&
                toupper(chan) != 'S' &&
                chan != 'm' && chan != 'h' &&
                toupper(chan) != 'X' &&
                toupper(chan) != 'L' &&
                toupper(chan) != 'K' &&
                chan != '0' && chan != '1' && chan != '2' && chan != '3' && chan != '4' && chan != '5' && chan != '6') {
                mpi_wrn_printf("for TUFRG vertex chan must be one of 'PCDSpcdsmh'. default: 'D'.\n");
                result.channel = chan = 'D';
            }

            if (toupper(chan) == 'X' || toupper(chan) == 'L' || toupper(chan) == 'K') {
                if (!isupper(chan)) {
                    ((toupper(chan-8) == 'P') ? P_projection :
                     (toupper(chan-8) == 'C') ? C_projection :
                    /*toupper(chan-8) == 'D'*/  D_projection )( tu_V->projection_helper, tu_V, tu_Proj );
                }
                mpi_vrb_printf( "recombining TUFRG vertex to channel '%c'\n", chan-8 );
                result.ary = calloc( tu_V->nk*POW2(tu_V->n_orbff * POW2(tu_V->n_spin)),
                        sizeof*result.ary );
                diverge_tu_vertex_reco( tu_V, toupper(chan-8), result.ary,
                        (!isupper(chan))?tu_V->projection_helper:NULL );
                result.nk = tu_V->nk;
                result.n_orbff = tu_V->n_orbff;
                result.n_spin = tu_V->n_spin;
                result.backend = 't';
                break;
            }

            // '3' == lp: Lambda-integrated pp bubble (bare pp susceptibility)
            // '4' == lm: Lambda-integrated ph bubble (bare ph susceptibility)
            if (chan == '3' || chan == '4') {
                result.ary = (chan == '3') ? tu_V->pp_bubble_int : tu_V->ph_bubble_int;
                if (!result.ary) {
                    mpi_wrn_printf("loop '%c' not allocated (channel not flowing)\n", chan);
                    break;                      // leaves backend == 'I'
                }
                result.q_0 = tu_V->my_nk_off;
                result.q_1 = tu_V->my_nk_off + tu_V->my_nk;
                result.nk  = 0;
                result.n_orbff = tu_V->n_orbff;
                result.n_spin  = mod->n_spin;
                result.backend = 'T';
                break;
            }

            // '5'/'6': as '3'/'4', but reconstructed to the full BZ on every rank.
            // Newly allocated, user frees. Canonical leg order, (f,g,q).
            if (chan == '5' || chan == '6') {
                const complex128_t* src = (chan == '5') ? tu_V->pp_bubble_int
                                                        : tu_V->ph_bubble_int;
                if (!src) {
                    mpi_wrn_printf("loop '%c' not allocated\n", chan);
                    break;
                }
                result.ary = calloc( tu_V->nk*POW2(tu_V->n_orbff*POW2(tu_V->n_spin)),
                        sizeof*result.ary );
                diverge_tu_vertex_reco( tu_V, (chan == '5') ? 'P' : 'C', result.ary, src );
                result.nk = tu_V->nk;
                result.n_orbff = tu_V->n_orbff;
                result.n_spin = tu_V->n_spin;
                result.backend = 't';
                break;
            }

            if (toupper(chan) != 'S') {
                if (isupper(chan) || chan == '0' || chan == '1' || chan == '2') {
                    result.ary = (chan == 'P') ? tu_V->Pch :
                                 (chan == 'C') ? tu_V->Cch :
                                 (chan == 'D') ? tu_V->Dch :
                                 (chan == '0') ? tu_stepper_internal_vertex(st->tu_step)->Pch :
                                 (chan == '1') ? tu_stepper_internal_vertex(st->tu_step)->Cch :
                                 (chan == '2') ? tu_stepper_internal_vertex(st->tu_step)->Dch :
                                                 NULL;
                } else if (chan != 'm' && chan != 'h') {
                    result.ary = tu_V->projection_helper;
                    if (chan == 'p') P_projection(tu_V->projection_helper, tu_V, tu_Proj);
                    if (chan == 'c') C_projection(tu_V->projection_helper, tu_V, tu_Proj);
                    if (chan == 'd') D_projection(tu_V->projection_helper, tu_V, tu_Proj);
                } else {
                    result.ary = tu_V->projection_helper;
                    C_projection(tu_V->projection_helper, tu_V, tu_Proj);
                    for (index_t i=0; i<tu_V->full_vert_size; ++i) tu_V->projection_helper[i] *= -1.0;
                    if (chan == 'h') {
                        if (!tu_V->C_flow) ZERO_OUT(tu_V->projection_helper, tu_V->full_vert_size);
                        complex128_t* helper = calloc(tu_V->full_vert_size, sizeof*helper);
                        D_projection(helper, tu_V, tu_Proj);
                        const double factor = tu_V->C_flow ? 2.0 : 1.0;
                        for (index_t i=0; i<tu_V->full_vert_size; ++i)
                            tu_V->projection_helper[i] += factor*helper[i];
                        free( helper );
                    }
                }
                result.q_0 = tu_V->my_nk_off;
                result.q_1 = tu_V->my_nk_off + tu_V->my_nk;
                result.nk = 0;
                result.n_orbff = tu_V->n_orbff;
            } else {
                result.ary = tu_V->self_en;
                result.q_0 = 0;
                result.q_1 = 0;
                result.nk = tu_V->nkf * tu_V->nk;
                result.n_orbff = mod->n_orb;
            }
            result.n_spin = mod->n_spin;
            result.backend = 'T';
            break;
        default:
        case diverge_flow_type_invalid:
            break;
    }

    return result;
}

void diverge_flow_step_refill( diverge_flow_step_t* s, double nu, void* workspace ) {
    if (s->type != diverge_tu && s->type != diverge_tu_phonons) {
        mpi_err_printf("refill only possible with TUFRG\n");
        return;
    }
    if (!s->channel_selection[3]) {
        mpi_err_printf("refill only possible with channel 'S'\n");
        return;
    }

    // sizes and ptrs
    index_t nk = kdimtot(s->mod->nk, s->mod->nkf),
            nb = s->mod->n_orb * s->mod->n_spin;
    double* E = diverge_model_internals_get_E( s->mod );


    // workspace
    bool free_workspace = false;
    if (!workspace) {
        free_workspace = true;
        workspace = calloc(nk*nb*nb*2 + nk*nb, sizeof(double));
    }

    // find mu
    double mu = diverge_flow_step_refill_Hself( s, nu, workspace );

    // only continue if no error
    if (mu != DBL_MAX) {
        // and readjust the energies accordingly
        if (shared_exclusive_enter(E) == 1) {
            #pragma omp parallel for num_threads(diverge_omp_num_threads())
            for (index_t kb=0; kb<nk*nb; ++kb)
                E[kb] -= mu;
        }
        shared_exclusive_wait(E);
    }

    // and free it
    if (free_workspace) free( workspace );
}

double diverge_flow_step_refill_Hself( diverge_flow_step_t* s, double nu, void* workspace ) {
    if (s->type != diverge_tu && s->type != diverge_tu_phonons) {
        mpi_err_printf("refill_Hself only possible with TUFRG\n");
        return 0;
    }
    if (!s->channel_selection[3]) {
        mpi_err_printf("refill_Hself only possible with channel 'S'\n");
        return 0;
    }

    diverge_model_t* m = s->mod;
    // get the data ptrs
    complex128_t *S = diverge_flow_step_vertex( s, 'S' ).ary,
                 *U = diverge_model_internals_get_U( m );
    double* E = diverge_model_internals_get_E( m );

    // and some sizes
    index_t nb = m->n_orb * m->n_spin;
    index_t nk = kdimtot(m->nk, m->nkf);

    // workspace to buffers
    complex128_t* Utmp = (complex128_t*)workspace;
    double* Etmp = (double*)( Utmp + nk*nb*nb );

    // add S to UEU* (in Utmp), i.e., H_eff = UEU* + S
    memcpy( Utmp, S, sizeof(complex128_t)*nk*nb*nb );
    batched_zgemd_r( Utmp, U, E, 1.0, 1.0, nb, nk );
    // do the eigensolution
    batched_eigen_shut_up(1);
    batched_eigen_r( NULL, -2, Utmp, Etmp, nb, nk );
    batched_eigen_shut_up(0);

    if (nu < -1.0 || nu > 2.0) {
        mpi_vrb_printf( "refilling at nu < -1.0 || nu > 2.0 doesn't work, returning %.2e\n", DBL_MAX );
        return DBL_MAX;
    } else {
        // get the chemical potential corresponding to the filling
        return diverge_model_set_filling( m, Etmp, nb, nu );
    }
}


double diverge_flow_step_get_filling_Hself( diverge_flow_step_t* s, void* workspace ) {
    if (s->type != diverge_tu && s->type != diverge_tu_phonons) {
        mpi_err_printf("refill_Hself only possible with TUFRG\n");
        return 0;
    }
    if (!s->channel_selection[3]) {
        mpi_err_printf("refill_Hself only possible with channel 'S'\n");
        return 0;
    }

    diverge_model_t* m = s->mod;
    // get the data ptrs
    complex128_t *S = diverge_flow_step_vertex( s, 'S' ).ary,
                 *U = diverge_model_internals_get_U( m );
    double* E = diverge_model_internals_get_E( m );

    // and some sizes
    index_t nb = m->n_orb * m->n_spin;
    index_t nk = kdimtot(m->nk, m->nkf);

    // workspace to buffers
    complex128_t* Utmp = (complex128_t*)workspace;
    double* Etmp = (double*)( Utmp + nk*nb*nb );

    // add S to UEU* (in Utmp), i.e., H_eff = UEU* + S
    memcpy( Utmp, S, sizeof(complex128_t)*nk*nb*nb );
    batched_zgemd_r( Utmp, U, E, 1.0, 1.0, nb, nk );
    // do the eigensolution
    batched_eigen_shut_up(1);
    batched_eigen_r( NULL, -2, Utmp, Etmp, nb, nk );
    batched_eigen_shut_up(0);

    return diverge_model_get_filling( m, Etmp, nb );
}

diverge_flow_type diverge_flow_step_type( const diverge_flow_step_t* st ) {
    return st->type;
}

void diverge_flow_step_get_channels( const diverge_flow_step_t* st, int chans[4] ) {
    for (int i=0; i<4; ++i) chans[i] = st->channel_selection[i];
}

void diverge_flow_step_set_interchannel( diverge_flow_step_t* st, const char* feedback ) {
    if (st->type != diverge_tu && st->type != diverge_tu_phonons) {
        mpi_err_printf( "interchannel feedback control only implemented for TUFRG\n" );
        return;
    }
    if (!st->intls->tu_data) {
        mpi_err_printf("internals for TU FRG not set\n");
        return;
    }
    if (!feedback) {
        mpi_vrb_printf( "X->Y feedback enabled (X,Y ∈ {P,C,D})\n" );
        st->tu_vertex->C_to_P = st->tu_vertex->D_to_P = true;
        st->tu_vertex->P_to_C = st->tu_vertex->D_to_C = true;
        st->tu_vertex->C_to_D = st->tu_vertex->P_to_D = true;
        mpi_vrb_printf( "ph->X feedback enabled (X ∈ {P,C,D})\n" );
        st->tu_vertex->ph_to_P = st->tu_vertex->ph_to_C = st->tu_vertex->ph_to_D = true;
        return;
    }
    if (!strncmp(feedback, "ph:", 3)) {
        st->tu_vertex->ph_to_P = st->tu_vertex->ph_to_C = st->tu_vertex->ph_to_D = false;
        feedback += 3;
        int target_chan = 0;
        for (const char* c = feedback; *c; ++c) {
            bool yes = (*c == 'y' || *c == 'Y' || *c == '1' || *c == '+');
            bool no  = (*c == 'n' || *c == 'N' || *c == '0' || *c == '-');
            if (!yes && !no) {
                mpi_wrn_printf( "cannot parse phonon channel #%i to y|n (%c)\n", target_chan, *c );
                continue;
            }
            assert( no == !yes );
            switch (target_chan) {
                case 0:
                    st->tu_vertex->ph_to_P = yes;
                    break;
                case 1:
                    st->tu_vertex->ph_to_C = yes;
                    break;
                case 2:
                    st->tu_vertex->ph_to_D = yes;
                    break;
                default:
                    mpi_wrn_printf( "too many channels (012->%i)\n", target_chan );
                    break;
            }
            target_chan++;
        }
        if (st->tu_vertex->ph_to_P) mpi_vrb_printf("ph->P feedback enabled\n");
        if (st->tu_vertex->ph_to_C) mpi_vrb_printf("ph->C feedback enabled\n");
        if (st->tu_vertex->ph_to_D) mpi_vrb_printf("ph->D feedback enabled\n");
    } else {
        int target_chan = 0;
        st->tu_vertex->C_to_P = st->tu_vertex->D_to_P = false;
        st->tu_vertex->P_to_C = st->tu_vertex->D_to_C = false;
        st->tu_vertex->C_to_D = st->tu_vertex->P_to_D = false;
        for (const char* c = feedback; *c; ++c) {
            if (*c == '|') {
                target_chan += 1;
            } else {
                switch (*c) {
                    case 'P':
                        if (target_chan == 1) st->tu_vertex->P_to_C = true;
                        if (target_chan == 2) st->tu_vertex->P_to_D = true;
                        break;
                    case 'C':
                        if (target_chan == 0) st->tu_vertex->C_to_P = true;
                        if (target_chan == 2) st->tu_vertex->C_to_D = true;
                        break;
                    case 'D':
                        if (target_chan == 0) st->tu_vertex->D_to_P = true;
                        if (target_chan == 1) st->tu_vertex->D_to_C = true;
                        break;
                    default:
                        mpi_wrn_printf( "unknown channel '%c' in feedback to channel #%i\n", *c, target_chan );
                        break;
                }
            }
        }
        if (st->tu_vertex->C_to_P) mpi_vrb_printf("C->P feedback enabled\n");
        if (st->tu_vertex->D_to_P) mpi_vrb_printf("D->P feedback enabled\n");
        if (st->tu_vertex->P_to_C) mpi_vrb_printf("P->C feedback enabled\n");
        if (st->tu_vertex->D_to_C) mpi_vrb_printf("D->C feedback enabled\n");
        if (st->tu_vertex->P_to_D) mpi_vrb_printf("P->D feedback enabled\n");
        if (st->tu_vertex->C_to_D) mpi_vrb_printf("C->D feedback enabled\n");
    }
}

const index_t* diverge_flow_step_mi_to_tuffidx( const diverge_flow_step_t* st ) {
    if (st->type != diverge_tu && st->type != diverge_tu_phonons)
        return NULL;
    tu_data_t* td = st->intls->tu_data;
    if (!td)
        return NULL;
    return td->mi_to_tuffidx;
}

index_t* diverge_flow_step_tuffidx_to_mi( const diverge_flow_step_t* st ) {
    const index_t* to = diverge_flow_step_mi_to_tuffidx( st );
    if (!to) return NULL;
    tu_data_t* td = st->intls->tu_data;
    index_t* from = calloc(td->n_orbff + 1, sizeof*from);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<td->n_orbff; ++i) {
        for (index_t j=0; j<td->n_orbff; ++j) if (to[j] == i) from[i] = j;
    }
    from[td->n_orbff] = -1;
    return from;
}

const index_t* diverge_flow_step_ibz( const diverge_flow_step_t* st ) {
    if (st->type != diverge_tu && st->type != diverge_tu_phonons) return NULL;
    tu_data_t* td = st->intls->tu_data;
    if (!td) return NULL;
    if (!td->symm) return NULL;
    return td->symm->idx_ibz_in_fullmesh;
}

index_t diverge_flow_step_n_ibz( const diverge_flow_step_t* st ) {
    if (diverge_flow_step_ibz( st )) {
        tu_data_t* td = st->intls->tu_data;
        return td->nkibz;
    }
    return 0;
}

#ifndef DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER
#define DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER 1000
#endif
#ifndef DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL
#define DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL 1.e-6
#endif
int diverge_flow_step_eigchan_shut_up = 0;
index_t diverge_flow_step_eigchan_piter_maxiter = DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER;
double diverge_flow_step_eigchan_piter_tol = DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL;

static double eigchan_of_vertex( diverge_flow_step_t* st, diverge_flow_step_vertex_t v, int piter );
static int step_eig_idx( char chan );

double diverge_flow_step_eigchan( diverge_flow_step_t* st, char chan, int piter ) {
    if (piter == DIVERGE_FLOW_STEP_EIGCHAN_PITER_FROM_STEP) {
        int i = step_eig_idx( chan );
        if (i < 0 && !diverge_flow_step_eigchan_shut_up)
            mpi_wrn_printf( "channel '%c' not computed within the step\n", chan );
        return i < 0 ? -1.0 : st->step_eig[i];
    }
    return eigchan_of_vertex( st, diverge_flow_step_vertex( st, chan ), piter );
}

static int step_eig_idx( char chan ) {
    const char* chans = "pmhcd";
    const char* c = strchr( chans, chan );
    return (chan && c) ? (int)(c - chans) : -1;
}

int diverge_flow_step_eigchan_in_step( diverge_flow_step_t* st, char chan, const complex128_t* buf, double scale ) {
    tu_vertex_t* tu_V = st->tu_vertex;
    diverge_flow_step_vertex_t v = {
        .ary = (complex128_t*)buf, // not written to
        .q_0 = tu_V->my_nk_off,
        .q_1 = tu_V->my_nk_off + tu_V->my_nk,
        .nk = 0,
        .n_orbff = tu_V->n_orbff,
        .n_spin = st->mod->n_spin,
        .backend = 'T',
        .channel = chan,
    };
    double val = fabs(scale) * eigchan_of_vertex( st, v, st->step_eig_piter );
    st->step_eig[step_eig_idx(chan)] = val;
    return val > st->step_eig_abort;
}

typedef enum {
    eigchan_piter,
    eigchan_arpack,
    eigchan_gersch,
    eigchan_svd,
} eigchan_method_t;

static inline double cabs_fast( const complex128_t z ) {
    const double* zd = (const double*)&z;
    return sqrt(POW2(zd[0]) + POW2(zd[1]));
}

static double eigchan_batched( eigchan_method_t m, const complex128_t* A,
        index_t linsz, index_t cnt, index_t* nnc ) {
    double result = 0.0;
    if (cnt <= 0) return result;
    switch (m) {
        case eigchan_piter: {
            double* tol_ary = calloc(cnt, sizeof*tol_ary);
            index_t* maxiter_ary = calloc(cnt, sizeof*maxiter_ary);
            complex128_t* eigen = calloc(cnt, sizeof*eigen);

            #pragma omp parallel for num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<cnt; ++q) {
                maxiter_ary[q] = diverge_flow_step_eigchan_piter_maxiter;
                tol_ary[q] = diverge_flow_step_eigchan_piter_tol;
            }

            power_iteration_batched( A, NULL, eigen, linsz, cnt, maxiter_ary, tol_ary, NULL );

            index_t nq_not_converged = 0;
            double maxval = 0.0;
            #pragma omp parallel for num_threads(diverge_omp_num_threads()) \
                                     reduction(+:nq_not_converged) \
                                     reduction(max:maxval)
            for (index_t q=0; q<cnt; ++q) {
                if (maxiter_ary[q] == diverge_flow_step_eigchan_piter_maxiter ||
                    tol_ary[q] > diverge_flow_step_eigchan_piter_tol)
                        nq_not_converged++;
                maxval = MAX(maxval, creal(eigen[q] * conj(eigen[q])));
            }
            *nnc += nq_not_converged;

            free( tol_ary );
            free( maxiter_ary );
            free( eigen );

            result = sqrt(maxval);
        } break;
        case eigchan_arpack: {
            for (index_t q=0; q<cnt; ++q) {
                int ncv = 10;
                complex128_t eigvals[ncv];
                complex128_t eigvecs[ncv*linsz];
                memset( eigvals, 0, sizeof(eigvals) );
                memset( eigvecs, 0, sizeof(eigvecs) );
                char* which = "LM";
                arnoldi_eigen_dense( which, linsz, 1, ncv, A+q*POW2(linsz), eigvals, eigvecs, 1.e-6, 100 );
                for (int i=0; i<ncv; ++i)
                    result = MAX(result, cabs_fast(eigvals[i]));
            }
        } break;
        case eigchan_gersch: {
            double Vmax = 0.0;
            #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(max:Vmax) collapse(2)
            for (index_t q=0; q<cnt; ++q)
            for (index_t l=0; l<linsz; ++l) {
                // Gerschgorin approximant: diagonal element is the disc's center
                // and the sum of all other elements' absolute value the radius
                complex128_t C_l = A[IDX3(q,l,l, linsz,linsz)];
                double R_l = 0.0;
                for (index_t m=0; m<linsz; ++m)
                    if (m != l) R_l += cabs_fast(A[IDX3(q,l,m, linsz,linsz)]);
                // worst case: extend into the direction of the center (which should
                // be real anyways)
                double Vmax_l = cabs_fast(cexp(I*carg(C_l))*R_l + C_l);
                Vmax = MAX(Vmax, Vmax_l);
            }
            result = Vmax;
        } break;
        case eigchan_svd: {
            double* singular_values = calloc( cnt * linsz, sizeof* singular_values );
            // A not written to
            batched_svd( NULL, -1, (complex128_t*)A, NULL, NULL, singular_values, linsz, cnt );
            double smax = 0.0;
            #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(max:smax)
            for (index_t i=0; i<linsz*cnt; ++i) smax = MAX(smax, singular_values[i]);
            free( singular_values );
            result = smax;
        } break;
    }
    return result;
}

static void absmatvec( const complex128_t* A, index_t N, const double* w, double* y, int trans ) {
    if (trans) {
        ZERO_OUT(y, N);
        for (index_t i=0; i<N; ++i)
        for (index_t j=0; j<N; ++j)
            y[j] += cabs_fast(A[IDX2(i,j,N)]) * w[i];
    } else {
        for (index_t i=0; i<N; ++i) {
            double s = 0.0;
            for (index_t j=0; j<N; ++j)
                s += cabs_fast(A[IDX2(i,j,N)]) * w[j];
            y[i] = s;
        }
    }
}

// aux: (4*N,), return the minumum of (i) Gershgorin rows and columns (or for SVD
// the Schur bound), (ii) the Frobenius norm, (iii) the Collatz-Wielandt bound,
// which is essentially power iteration of abs(A) with an all-ones vector w.
static double eigchan_bound( const complex128_t* A, index_t N, int svd, index_t iter, double* aux ) {
    double *c = aux, *w = aux+N, *y = aux+2*N, *z = aux+3*N;
    ZERO_OUT(c, N);
    double rmax = 0.0, cmax = 0.0, fro = 0.0;
    for (index_t i=0; i<N; ++i) {
        double r = 0.0;
        for (index_t j=0; j<N; ++j) {
            double a = cabs_fast(A[IDX2(i,j,N)]);
            r += a;
            c[j] += a;
            fro += a*a;
        }
        y[i] = r;
        rmax = MAX(rmax, r);
    }
    for (index_t j=0; j<N; ++j) cmax = MAX(cmax, c[j]);
    double bound = MIN( svd ? sqrt(rmax*cmax) : MIN(rmax, cmax), sqrt(fro) );

    // y = |A| w, fill w with 1s
    for (index_t i=0; i<N; ++i) w[i] = 1.0;
    for (index_t k=0; k<iter && bound > 0.0; ++k) {
        // z = B w
        if (svd) absmatvec( A, N, y, z, 1 );
        else memcpy( z, y, sizeof(double)*N );
        double ratio = 0.0, zmax = 0.0;
        for (index_t i=0; i<N; ++i) {
            ratio = MAX(ratio, z[i]/w[i]);
            zmax = MAX(zmax, z[i]);
        }
        bound = MIN( bound, svd ? sqrt(ratio) : ratio );
        if (k+1 == iter || zmax <= 0.0) break;
        // power step, w kept strictly positive (zero rows give z_i = 0)
        for (index_t i=0; i<N; ++i) w[i] = MAX(z[i]/zmax, 1.e-12);
        absmatvec( A, N, w, y, 0 );
    }
    return bound;
}

typedef struct {
    double b;
    index_t q;
} eigchan_bound_t;

#undef SERISORT_COMP
#define SERISORT_COMP(x, y) (((x).b < (y).b) - ((y).b < (x).b))
SERISORT_IMPL(eigchan_bound_sort, eigchan_bound_t)

static double eigchan_iterative( eigchan_method_t m, const complex128_t* A,
        index_t linsz, index_t nq, index_t* nnc ) {
    index_t L2 = POW2(linsz);
    int nthr = diverge_omp_num_threads();

    eigchan_bound_t* bnd = calloc(MAX(nq,1), sizeof*bnd);
    // aux: (nthr,4*linsz) for eigchan_bound, then (nq,) eigchan_bound_t for sorting
    double* aux = calloc(MAX(4*linsz*nthr, nq*(index_t)(sizeof(eigchan_bound_t)/sizeof(double))), sizeof*aux);
    #pragma omp parallel for num_threads(nthr)
    for (index_t q=0; q<nq; ++q) {
        bnd[q].q = q;
        bnd[q].b = eigchan_bound(A+q*L2, linsz, m==eigchan_svd, 3, aux+4*linsz*omp_get_thread_num());
    }
    eigchan_bound_sort_s(bnd, nq, (eigchan_bound_t*)aux);
    free(aux);

    // one per rank
    double lam = nq > 0 ? eigchan_batched(m, A+bnd[0].q*L2, linsz, 1, nnc) : 0.0;
    diverge_mpi_allreduce_double_max_inplace(&lam, 1);

    // then the remaining qpts on this rank
    index_t pos = MIN(nq, 1);
    for (; pos < nq && bnd[pos].b > lam; ++pos)
        lam = MAX(lam, eigchan_batched(m, A+bnd[pos].q*L2, linsz, 1, nnc));
    free(bnd);
    // and finally communicate each local max
    diverge_mpi_allreduce_double_max_inplace(&lam, 1);

    return lam;
}

static double eigchan_of_vertex( diverge_flow_step_t* st, diverge_flow_step_vertex_t v, int piter ) {
    index_t nq = (v.q_1 - v.q_0);
    index_t linsz = v.backend == 'T' ? v.n_orbff * POW2(v.n_spin) : POW2(v.n_orbff*v.n_spin) * v.nk;
    double tick = diverge_mpi_wtime();

    int iterative = 1;
    switch (piter) {
        case DIVERGE_FLOW_STEP_EIGCHAN_PITER_ITERATIVE_SVD:
            piter = 0;
            break;
        case DIVERGE_FLOW_STEP_EIGCHAN_PITER_ITERATIVE_PITER:
            piter = 1;
            break;
        case DIVERGE_FLOW_STEP_EIGCHAN_PITER_ITERATIVE_ARPACK:
            piter = DIVERGE_FLOW_STEP_EIGCHAN_PITER_ARPACK;
            break;
        default:
            iterative = 0;
            break;
    }

    eigchan_method_t m = eigchan_svd;
    if (piter > 0) {
        m = eigchan_piter;
    } else if (piter == DIVERGE_FLOW_STEP_EIGCHAN_PITER_ARPACK) {
        m = eigchan_arpack;
        if (linsz < 10) {
            if (!diverge_flow_step_eigchan_shut_up) {
                mpi_wrn_printf( "linsz too small (%li), using power iteration\n", linsz );
            }
            m = eigchan_piter;
        } else if (!arnoldi_package_compiled_in()) {
            if (!diverge_flow_step_eigchan_shut_up) {
                mpi_wrn_printf( "ARPACK not compiled in, using power iteration\n" );
            }
            m = eigchan_piter;
        }
    } else if (piter < 0) {
        m = eigchan_gersch;
    }

    index_t nq_not_converged = 0;
    double result = 0.0;
    if (iterative) {
        result = eigchan_iterative( m, v.ary, linsz, nq, &nq_not_converged );
    } else {
        result = eigchan_batched( m, v.ary, linsz, nq, &nq_not_converged );
        diverge_mpi_allreduce_double_max_inplace(&result, 1);
    }

    if (m == eigchan_piter) {
        diverge_mpi_allreduce_index_sum_inplace(&nq_not_converged, 1);
        if (nq_not_converged != 0 && !diverge_flow_step_eigchan_shut_up)
            mpi_wrn_printf( "%li qpts not converged in power iteration\n", nq_not_converged );
    }

    double tock = diverge_mpi_wtime();
    st->t_eigchan_diag += tock - tick;
    return result;
}

void diverge_flow_step_attach_rpa( diverge_flow_step_t* s, index_t* kidx, index_t n_kidx ) {
    int free_kidx = 0;
    if (!kidx || n_kidx < 0) {
        free_kidx = 1;
        n_kidx = kdim(s->mod->nk);
        kidx = calloc(n_kidx, sizeof*kidx);
        for (index_t k=0; k<n_kidx; ++k) kidx[k] = k;
    }
    s->tu_rpa = tu_rpa_init( s, kidx, n_kidx );
    if (free_kidx) free( kidx );
}
