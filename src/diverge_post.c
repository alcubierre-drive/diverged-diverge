/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_post.h"
#include "diverge_flow_step_internal.h"

#include "npatch/output.h"
#include "grid/vertex_memory.h"
#include "grid/post_processing.h"
#include "tu/vertex.h"
#include "tu/postproduction.h"
#include "misc/qnan.h"

static void grid_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file );
static void patch_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file );
static void tu_pp( diverge_flow_step_t* st,
        const diverge_postprocess_conf_t* pc, const char* file );

void diverge_postprocess_and_write_fg( diverge_flow_step_t* st, const char* file,
        const diverge_postprocess_conf_t* cfg ) {
    diverge_postprocess_and_write_finegrained( st, file, cfg );
}

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
    diverge_postprocess_conf_t pc_def = diverge_postprocess_conf_defaults;
    if (pc == NULL) pc = &pc_def;

    switch (st->type) {
        case diverge_tu_phonons:
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
    grid_vertex_t* V = (grid_vertex_t*)st->intls->grid_vertex;

    if (strcmp(pc->grid_vertex_file, "")) {
        mpi_log_printf( "writing vertex to file '%s'.\n", pc->grid_vertex_file );
        grid_vertex_to_file( V, pc->grid_vertex_file, pc->grid_vertex_chan );
    }

    index_t nsing = st->intls->grid_nk * st->intls->grid_nb * st->intls->grid_nb;
    nsing = MIN(nsing, pc->grid_n_singular_values);
    if (nsing != pc->grid_n_singular_values)
        mpi_wrn_printf("reset # singular values to %li\n", nsing);
    grid_postproc_t* pp = grid_postproc_init( V, st->grid_loop );
    grid_postproc_set_num_singular_values( pp, nsing );
    grid_postproc_set_loop_usage( pp, pc->grid_use_loop );

    mpi_vrb_printf( "susceptibilities\n" );
    grid_postproc_susc( pp, st->Lambda, 'P' );
    if (st->mod->SU2 > 0) grid_postproc_susc( pp, st->Lambda, 'C' );
    grid_postproc_susc( pp, st->Lambda, 'D' );

    mpi_vrb_printf( "linearized gaps\n" );
    grid_postproc_lingap( pp, st->Lambda, 'P', pc->grid_lingap_vertex_file_P,
            pc->grid_lingap_vertex_loop_file_P );
    if (st->mod->SU2 > 0) grid_postproc_lingap( pp, st->Lambda, 'C', pc->grid_lingap_vertex_file_C,
            pc->grid_lingap_vertex_loop_file_C );
    grid_postproc_lingap( pp, st->Lambda, 'D', pc->grid_lingap_vertex_file_D,
            pc->grid_lingap_vertex_loop_file_D );

    mpi_vrb_printf( "communicating results\n" );
    grid_postproc_communicate( pp );

    grid_postproc_print_header( grid_postproc_to_file( pp, file ) );
    grid_postproc_free( pp );
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
    tu_postproduction_conf_t tc = tu_postproduction_conf_defaults;
    tc.skip_channel_calc = pc->tu_skip_channel_calc;
    tc.tu_n_singular_values = pc->tu_n_singular_values;
    tc.algo = pc->tu_which_solver_mode;
    tc.algo_zero = pc->tu_which_solver_mode_hermiticity_limit;
    tc.cutoff = pc->tu_storing_threshold;
    tc.cutoff_rel = pc->tu_storing_relative;
    tc.tu_n_decomp_values = pc->tu_n_decomp_values;
    tc.bare_susceptibilities = pc->tu_susceptibilities_bare;
    tc.channel_calc_project = pc->tu_channel_calc_project;
    memcpy( tc.tu_lingap_solver_mode, pc->tu_lingap_solver_mode, sizeof(tc.tu_lingap_solver_mode) );
    tc.lingaps = pc->tu_lingap;
    tc.susc_ff = pc->tu_susceptibilities_ff;
    tc.susc_full = pc->tu_susceptibilities_full;
    tc.lingap_atscale = pc->tu_lingap_atscale;
    tc.save_self = pc->tu_selfenergy;
    tc.save_full = pc->tu_channels;
    tc.save_symmetries = pc->tu_symmetry_maps;
    // RPA attachment
    memcpy( tc.rpa_chans, pc->tu_rpa_chans, sizeof(pc->tu_rpa_chans) );
    tc.rpa_include_chi_pp = pc->tu_rpa_include_chi_pp;
    tc.rpa_include_V_pp   = pc->tu_rpa_include_V_pp;
    tc.rpa_nev     = pc->tu_rpa_nev;
    tc.rpa_ncv     = pc->tu_rpa_ncv;
    tc.rpa_maxiter = pc->tu_rpa_maxiter;
    tc.rpa_tol     = pc->tu_rpa_tol;
    tu_rpa_post_set_which_ev( pc->tu_rpa_sorting );
    tc.self_rs_dist = pc->tu_selfenergy_rs_dist;
    tc.self_rs_mval = pc->tu_selfenergy_rs_mval;

    // DEVELOPERS
    //
    // For a new backend, you may choose to add postprocessing flags here and
    // their defaults below
    tu_postproduction( st->tu_vertex, tu_stepper_internal_vertex(st->tu_step),
            st->tu_proj, st->tu_rpa, st->mod, &tc, file );
}

diverge_postprocess_conf_t diverge_postprocess_conf_defaults_CPP( void ) {
    return diverge_postprocess_conf_defaults;
}

double diverge_postprocess_conf_tu_storing_threshold_qrelative( double threshold ) {
    return fnan_gen(threshold);
}
