/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_post.h"
#include "diverge_flow_step_internal.hpp"

#include "npatch/output.h"
#include "grid/vertex_memory.hpp"
#include "grid/post_processing.hpp"
#include "tu/vertex.hpp"
#include "tu/postproduction.hpp"

static void grid_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file );
static void patch_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file );
static void tu_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file );

void diverge_postprocess_and_write( diverge_flow_step_t* st, const char* file ) {
    diverge_postprocess_and_write_finegrained( st, file, NULL );
}

// DEVELOPERS
//
// For a new backend, you can add a new post processing routine similar to the
// above three and call it in the switch() {} statement below

void diverge_postprocess_and_write_finegrained( diverge_flow_step_t* st,
        const char* file, const diverge_postprocess_conf_t* pc ) {
    mpi_log_printf( "writing postprocessing file %s\n", file );
    diverge_postprocess_conf_t pc_def = diverge_postprocess_conf_defaults_CPP();
    if (pc == NULL) pc = &pc_def;

    switch (st->type) {
        case diverge_tu:
            tu_pp( st, pc, file );
            break;
        case diverge_grid:
            grid_pp( st, pc, file );
            break;
        case diverge_patch:
            patch_pp( st, pc, file );
            break;
        case diverge_flow_type_invalid: default:
            mpi_err_printf("invalid flow type. cannot do postprocessing.\n");
            break;
    }
}

static void grid_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file ) {

    grid::VertexMemory* V = (grid::VertexMemory*)st->intls->grid_vertex;

    if (strcmp(pc->grid_vertex_file, "")) {
        mpi_log_printf( "writing vertex to file '%s'.\n", pc->grid_vertex_file );
        V->write_to_file( pc->grid_vertex_file, pc->grid_vertex_chan );
    }

    index_t nsing = st->intls->grid_nk * st->intls->grid_nb * st->intls->grid_nb;
    nsing = MIN(nsing, pc->grid_n_singular_values);
    if (nsing != pc->grid_n_singular_values)
        mpi_wrn_printf("reset # singular values to %li\n", nsing);
    grid::PostProcessing post_process( *V, *st->grid_loop );
    post_process.set_num_singular_values( nsing );
    post_process.set_loop_usage( pc->grid_use_loop );

    mpi_vrb_printf( "susceptibilities\n" );
    post_process.susceptibility( st->Lambda, 'P' );
    if (st->mod->SU2 > 0)
        post_process.susceptibility( st->Lambda, 'C' );
    post_process.susceptibility( st->Lambda, 'D' );

    mpi_vrb_printf( "linearized gaps\n" );
    post_process.linearized_gap( st->Lambda, 'P', pc->grid_lingap_vertex_file_P );
    if (st->mod->SU2 > 0)
        post_process.linearized_gap( st->Lambda, 'C', pc->grid_lingap_vertex_file_C );
    post_process.linearized_gap( st->Lambda, 'D', pc->grid_lingap_vertex_file_D );

    mpi_vrb_printf( "communicating results\n" );
    post_process.communicate();

    index_t* file_header = post_process.to_file( file );
    grid::print_post_processing_file_header( file_header );
    free( file_header );
}

static void patch_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file ) {
    npatch_output_post_config_t patch_pc;
    patch_pc.q_matrices = pc->patch_q_matrices;
    patch_pc.q_matrices_use_dV = pc->patch_q_matrices_use_dV;
    patch_pc.q_matrices_nv = pc->patch_q_matrices_nv;
    patch_pc.q_matrices_max_rel = pc->patch_q_matrices_max_rel;
    patch_pc.q_matrices_eigen_which = pc->patch_q_matrices_eigen_which;
    patch_pc.V = pc->patch_V;
    patch_pc.dV = pc->patch_dV;
    patch_pc.Lp = pc->patch_Lp;
    patch_pc.Lm = pc->patch_Lm;
    npatch_output_post( (npatch_vertex_t*)st->intls->patch_vertex, st->patch_loop, &patch_pc, file );
}

static void tu_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file ) {
    tu_postproduction tu_pc(st->tu_vertex->model, st->tu_vertex, st->tu_proj, pc->tu_skip_channel_calc);
    tu_pc.tu_n_singular_values = pc->tu_n_singular_values;
    tu_pc.algo = pc->tu_which_solver_mode;
    tu_pc.cutoff = pc->tu_storing_threshold;
    tu_pc.cutoff_rel = pc->tu_storing_relative;
    tu_pc.run_and_save(file,pc->tu_lingap,pc->tu_susceptibilities_ff,pc->tu_susceptibilities_full,
                        pc->tu_selfenergy, pc->tu_channels, pc->tu_symmetry_maps);
}
