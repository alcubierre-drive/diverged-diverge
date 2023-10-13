/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_flow_step_internal.hpp"
#include "misc/batched_gemms.h"
#include "misc/batched_eigen.h"
#include "misc/batched_zgemd.h"
#include "diverge_filling.h"
#include "diverge_model_internals.h"
#include "misc/dependency_tree.h"

#include "grid/vertex_memory.hpp"

#include "npatch/vertex.h"
#include "npatch/kernels_gpu.h"
#include "npatch/loop.h"

#include <cfloat>

static void diverge_flow_step_select_channels( diverge_flow_step_t* st, const char* chans_ );

diverge_flow_step_t* diverge_flow_step_init( diverge_model_t* mod, const char* method_, const char* channels ) {
    diverge_flow_step_t* st = (diverge_flow_step_t*)calloc(sizeof(diverge_flow_step_t),1);
    st->mod = mod;
    st->intls = mod->internals;

    dependency_tree_insert_flow(st);

    diverge_flow_step_select_channels( st, channels );

    // DEVELOPERS
    //
    // For a new backend, you must add a selection to your new flow_type in the
    // below if statements.
    char* method = strdup_div(method_);
    strlwr_div(method);
    if (!strcmp(method, "grid"))
        st->type = diverge_grid;
    else if (!strcmp(method, "patch"))
        st->type = diverge_patch;
    else if (!strcmp(method, "tu"))
        st->type = diverge_tu;
    else
        st->type = diverge_flow_type_invalid;
    free(method);

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

    st->timings = new vector<double>();
    st->timings_descr = new vector<string>();
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
            st->grid_loop = new grid::Loops(st->mod);
            st->grid_step = new grid::FlowStep( (grid::VertexMemory*)st->intls->grid_vertex,
                    st->grid_loop, st->grid_loop->get_symmetrizer(),
                    st->channel_selection[0], st->channel_selection[1],
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
            st->timings->resize( 8 + 3*128 );
            st->timings_descr->resize( 8 + 3*128 );
            break;
        case diverge_tu:
            if (st->intls->tu_data == NULL) {
                mpi_err_printf("internals for TU FRG not set\n");
                goto end_error;
            }
            st->tu_proj = new Projection(mod);
            st->tu_loop = new tu_loop_t(mod);
            st->tu_vertex = new Vertex(mod, st->tu_proj,st->tu_loop, st->channel_selection[0],
                    st->channel_selection[1], st->channel_selection[2], st->channel_selection[3]);
            st->tu_step = new tu_stepper(st->tu_proj, st->tu_loop, st->tu_vertex);
            break;
        default:
        case diverge_flow_type_invalid:
            mpi_err_printf("invalid flow type '%s'. valid choices are 'grid', 'patch', 'tu'.\n", method_);
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
            delete st->grid_step;
            delete st->grid_loop;
            break;
        case diverge_tu:
            delete st->tu_vertex;
            delete st->tu_proj;
            delete st->tu_step;
            delete st->tu_loop;
            break;
        default:
        case diverge_flow_type_invalid:
            break;
    }
    delete st->timings;
    delete st->timings_descr;

    dependency_tree_remove_flow(st);

    free(st);
    batched_gemm_cublas_clear();
    batched_gemm_cublas_w_buffer_clear();
}

void diverge_flow_step_euler( diverge_flow_step_t* st, double Lambda, double dLambda ) {
    vector<double> chanmax_vec;
    double patch_cm[6];
    int idx = 0;
    char tmpchar[64];
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
    // and act accordingly. See src/npatch/loop.c or src/grid/loops.cpp for
    // examples.
    switch (st->type) {
        case diverge_patch:
            npatch_flow_step( st->patch_step_conf, (npatch_vertex_t*)st->intls->patch_vertex,
                    st->patch_loop, &Lambda, dLambda, patch_cm );
            st->loopmax[0] = patch_cm[0];
            st->loopmax[1] = patch_cm[1];
            st->chanmax[0] = patch_cm[2];
            st->chanmax[1] = patch_cm[3];
            st->chanmax[2] = patch_cm[4];
            st->vertmax = patch_cm[5];

            // save timings
            idx = 0;
            #define PATCH_TIMINGS_PUSH_BACK( name ) \
            st->timings->at(idx) = st->patch_step_conf->name; \
            st->timings_descr->at(idx++) = #name;
            PATCH_TIMINGS_PUSH_BACK( wtime_gpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_cpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_comm )
            PATCH_TIMINGS_PUSH_BACK( wtime_max )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_gpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_cpu )
            PATCH_TIMINGS_PUSH_BACK( wtime_loop_comm )
            PATCH_TIMINGS_PUSH_BACK( wtime_full )
            memcpy( st->timings->data() + 8 + 128*0, st->patch_step_conf->wtime_gpu_list, sizeof(double)*128 );
            memcpy( st->timings->data() + 8 + 128*1, st->patch_step_conf->wtime_loop_gpu_list, sizeof(double)*128 );
            memcpy( st->timings->data() + 8 + 128*2, st->patch_step_conf->wtime_loop_gpu_gf_list, sizeof(double)*128 );
            for (int i=0; i<128; ++i) {
                sprintf(tmpchar, "wtime_gpu_%03i", i);
                st->timings_descr->at(8+128*0+i) = tmpchar;
                sprintf(tmpchar, "wtime_loop_gpu_%03i", i);
                st->timings_descr->at(8+128*1+i) = tmpchar;
                sprintf(tmpchar, "wtime_gf_gpu_%03i", i);
                st->timings_descr->at(8+128*2+i) = tmpchar;
            }
            st->niter = st->patch_step_conf->niter;
            return;
        case diverge_grid:
            chanmax_vec = (*st->grid_step)( Lambda, dLambda );
            st->chanmax[0] = chanmax_vec[0];
            st->chanmax[1] = chanmax_vec[1];
            st->chanmax[2] = chanmax_vec[2];
            st->vertmax = chanmax_vec[3];
            st->loopmax[0] = chanmax_vec[4];
            st->loopmax[1] = chanmax_vec[5];

            // save timings
            *st->timings = st->grid_step->get_timings();
            *st->timings_descr = st->grid_step->get_timings_descr();
            st->niter = st->grid_step->get_num_calls();
            return;
        case diverge_tu:
            chanmax_vec = st->tu_step->tu_step_euler(st->tu_vertex, Lambda, dLambda);

            st->chanmax[0] = chanmax_vec[0];
            st->chanmax[1] = chanmax_vec[1];
            st->chanmax[2] = chanmax_vec[2];
            st->vertmax = std::max(chanmax_vec[0], std::max(chanmax_vec[1],chanmax_vec[2]));
            st->loopmax[0] = chanmax_vec[3];
            st->loopmax[1] = chanmax_vec[4];

            *st->timings = st->tu_step->get_timings();
            *st->timings_descr = st->tu_step->get_timings_descr();
            st->niter = st->tu_step->get_num_calls();
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
    return st->timings->size();
}
double diverge_flow_step_timing( diverge_flow_step_t* st, index_t idx ) {
    return st->timings->at(idx);
}
const char* diverge_flow_step_timing_descr( diverge_flow_step_t* st, index_t idx ) {
    return st->timings_descr->at(idx).c_str();
}

static double dvg_timings[129];
double* diverge_flow_step_timing_vec( diverge_flow_step_t* st ) {
    for (uindex_t i=0; i<st->timings->size(); ++i)
        if (i < 128)
            dvg_timings[i] = st->timings->at(i);
    memset( dvg_timings+128, 0, sizeof(double) );
    return dvg_timings;
}

static char dvg_timings_descr[129][1024];
char** diverge_flow_step_timing_descr_vec( diverge_flow_step_t* st ) {
    for (uindex_t i=0; i<st->timings_descr->size(); ++i)
        if (i < 128) {
            strncpy( dvg_timings_descr[i], st->timings_descr->at(i).c_str(), 1023 );
            dvg_timings_descr[i][1023] = '\0';
        }
    memset( dvg_timings_descr[128], 0, 1024 );
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

    grid::VertexMemory* grid_V = (grid::VertexMemory*)intls->grid_vertex;
    npatch_vertex_t* patch_V = (npatch_vertex_t*)intls->patch_vertex;
    Vertex* tu_V = st->tu_vertex;

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
            (*grid_V)( islower(chan) ? grid_V->vertex_buffers[1] : grid_V->vertex_buffers[0],
                    result.ary, 'D', toupper(result.channel), false, 1.0 );
            result.q_0 = grid_V->qshare()[0];
            result.q_1 = result.q_0 + grid_V->qshare().size();
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
                result.channel = 'D';
            }
            result.ary = (toupper(chan) == 'P') ? tu_V->Pch :
                         (toupper(chan) == 'C') ? tu_V->Cch :
                         (toupper(chan) == 'D') ? tu_V->Dch :
                                                  tu_V->self_en;
            if (toupper(chan) == 'S') {
                result.q_0 = 0;
                result.q_1 = 0;
                result.nk = tu_V->nkf * tu_V->nk;
                result.n_orbff = mod->n_orb;
            } else {
                result.q_0 = tu_V->my_nk_off;
                result.q_1 = tu_V->my_nk_off + tu_V->my_nk;
                result.nk = 0;
                result.n_orbff = tu_V->n_orbff;
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

    // add S to UEU* (in Utmp)
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k) {
        Map<MatXcd> UU(U + k*nb*nb, nb, nb);
        Map<VecXd> EE(E + k*nb, nb);
        Map<MatXcd> SS(S + k*nb*nb, nb, nb);

        Map<MatXcd>(Utmp + k*nb*nb, nb, nb) = UU * EE.asDiagonal() * UU.adjoint() + SS;
    }
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
