/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_flow_step_internal.h"
#include "diverge_flow_type.h"
#include "misc/batched_gemms.h"
#include "misc/batched_eigen.h"
#include "misc/batched_zgemd.h"
#include "diverge_filling.h"
#include "diverge_model_internals.h"
#include "misc/dependency_tree.h"

#include "tu/vertex_details.h"

#include "grid/vertex_memory.h"

#include "npatch/vertex.h"
#include "npatch/kernels_gpu.h"
#include "npatch/loop.h"

#include <float.h>

static void diverge_flow_step_select_channels( diverge_flow_step_t* st, const char* chans_ );

diverge_flow_step_t* diverge_flow_step_init_any( diverge_model_t* mod, const char* channels ) {
    return diverge_flow_step_init( mod, mod->internals->backend, channels );
}

diverge_flow_step_t* diverge_flow_step_init( diverge_model_t* mod, const char* method, const char* channels ) {
    diverge_flow_step_t* st = (diverge_flow_step_t*)calloc(1, sizeof(diverge_flow_step_t));
    st->mod = mod;
    st->intls = mod->internals;

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
            VECTOR_RESIZE( st->timings, 8 + 3*128 );
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
            mpi_err_printf("invalid flow type '%s'. valid choices are 'grid', 'patch', 'tu'.\n", method);
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
        case diverge_tu:
            tu_vertex_free( st->tu_vertex );
            tu_projection_free( st->tu_proj );
            tu_stepper_free( st->tu_step );
            tu_prop_free( st->tu_loop );
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
            idx = 0;
            #define PATCH_TIMINGS_PUSH_BACK( name ) \
                st->timings[idx].t = st->patch_step_conf->name; \
                strcpy( st->timings[idx++].nam, #name );
            PATCH_TIMINGS_PUSH_BACK( wtime_gpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_cpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_comm )
            PATCH_TIMINGS_PUSH_BACK( wtime_max )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_gpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_cpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_comm )
            PATCH_TIMINGS_PUSH_BACK( wtime_full )
            for (int i=0; i<128; ++i) {
                st->timings[8+128*0+i].t = st->patch_step_conf->wtime_gpu_list[i];
                snprintf(st->timings[8+128*0+i].nam, 55, "wtime_gpu_%03i", i);
                st->timings[8+128*1+i].t = st->patch_step_conf->wtime_loop_gpu_list[i];
                snprintf(st->timings[8+128*1+i].nam, 55, "wtime_loop_gpu_%03i", i);
                st->timings[8+128*2+i].t = st->patch_step_conf->wtime_loop_gpu_gf_list[i];
                snprintf(st->timings[8+128*2+i].nam, 55, "wtime_gf_gpu_%03i", i);
            }
            st->niter = st->patch_step_conf->niter;
            return;
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
            return;
        case diverge_tu:
            tu_stepper_euler( st->tu_step, st->tu_vertex, Lambda, dLambda, chanmax );

            st->chanmax[0] = chanmax[0];
            st->chanmax[1] = chanmax[1];
            st->chanmax[2] = chanmax[2];
            st->vertmax = MAX(MAX(chanmax[0], chanmax[1]), chanmax[2]);
            st->loopmax[0] = chanmax[3];
            st->loopmax[1] = chanmax[4];

            ttt = tu_stepper_timings(st->tu_step, &idx);
            VECTOR_RESIZE( st->timings, idx );
            memcpy( st->timings, ttt, sizeof(diverge_timing_t)*idx );
            st->niter = tu_stepper_num_calls( st->tu_step );
            return;
        default:
        case diverge_flow_type_invalid:
            return;
    }
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
    diverge_flow_step_vertex_t result;
    result.ary = NULL;
    result.q_0 = result.q_1 = result.nk = result.n_orbff = result.n_spin = 0;
    result.backend = 'I';
    result.channel = chan;

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
        case diverge_tu:
            if (toupper(chan) != 'P' &&
                toupper(chan) != 'C' &&
                toupper(chan) != 'D' &&
                toupper(chan) != 'S') {
                mpi_wrn_printf("for TUFRG vertex chan must be one of 'PCDSpcds'. default: 'D'.\n");
                result.channel = chan = 'D';
            }

            if (toupper(chan) != 'S') {
                if (isupper(chan)) {
                    result.ary = (chan == 'P') ? tu_V->Pch :
                                 (chan == 'C') ? tu_V->Cch :
                                /*chan == 'D'*/  tu_V->Dch;
                } else {
                    result.ary = tu_V->projection_helper;
                    if (chan == 'p') P_projection(tu_V->projection_helper, tu_V, tu_Proj);
                    if (chan == 'c') C_projection(tu_V->projection_helper, tu_V, tu_Proj);
                    if (chan == 'd') D_projection(tu_V->projection_helper, tu_V, tu_Proj);
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
    if (s->type != diverge_tu) {
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
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t kb=0; kb<nk*nb; ++kb)
            E[kb] -= mu;
    }

    // and free it
    if (free_workspace) free( workspace );
}

double diverge_flow_step_refill_Hself( diverge_flow_step_t* s, double nu, void* workspace ) {
    if (s->type != diverge_tu) {
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
    if (s->type != diverge_tu) {
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
    if (st->type != diverge_tu) {
        mpi_err_printf( "interchannel feedback control only implemented for TUFRG\n" );
        return;
    }
    if (st->intls->tu_data == NULL) {
        mpi_err_printf("internals for TU FRG not set\n");
        return;
    }
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
                    if (target_chan == 1) st->tu_vertex->P_to_C=true;
                    if (target_chan == 2) st->tu_vertex->P_to_D=true;
                    break;
                case 'C':
                    if (target_chan == 0) st->tu_vertex->C_to_P=true;
                    if (target_chan == 2) st->tu_vertex->C_to_D=true;
                    break;
                case 'D':
                    if (target_chan == 0) st->tu_vertex->D_to_P=true;
                    if (target_chan == 1) st->tu_vertex->D_to_C=true;
                    break;
                default:
                    mpi_wrn_printf( "unknown channel in feedback to chan no.%i\n", target_chan );
                    break;
            }
        }
    }
    if(st->tu_vertex->C_to_P) mpi_vrb_printf("C to P feedback active\n");
    if(st->tu_vertex->D_to_P) mpi_vrb_printf("D to P feedback active\n");
    if(st->tu_vertex->P_to_C) mpi_vrb_printf("P to C feedback active\n");
    if(st->tu_vertex->D_to_C) mpi_vrb_printf("D to C feedback active\n");
    if(st->tu_vertex->P_to_D) mpi_vrb_printf("P to D feedback active\n");
    if(st->tu_vertex->C_to_D) mpi_vrb_printf("C to D feedback active\n");
}
