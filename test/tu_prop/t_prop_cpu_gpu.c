#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME tu_prop_gpu
#include "../ctest.h"

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/propagator_details.h"
#include "../../src/tu/cuda_init.h"
#include "../models/all.h"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.h"
#include "../../src/tu/vertex.h"
#include "../../src/tu/projection.h"
#include "../../src/misc/compilation_status.h"

static inline int compilation_options( void ) {
    // no floats for greens functions, no MPI, but CUDA
    if ((sizeof(gf_complex_t) != sizeof(complex128_t)) ||
        (diverge_mpi_comm_size() != 1) ||
        (!diverge_compilation_status_cuda(NULL)))
        return 1;
    else
        return 0;
}

CT_TEST_NAME(tu_prop_gpu, "tu_prop,gpu") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=6, .nky=6);
    diverge_model_internals_tu(mom, 1.5);

    tu_prop_t* PropMom = tu_prop_init(mom);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk,sizeof(complex128_t));
    complex128_t* helper = (complex128_t*)calloc(POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmg = (complex128_t*)calloc(POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );
    S_loop_static_simple( PropMom, false, false, lm, -1, 0, true );
    S_loop_static_red( PropMom, false, false, lmr, -1, 0, true );
    S_ph_loop_static_red_gpu( PropMom, helper );
    merge_gpu_cpu_loop( PropMom, lmrg, helper );

    cuda_reset_prop_descr(&(PropMom->cuda_descr),POW4(PropMom->n_spin), POW2(PropMom->n_orbff),0.0);

    S_ph_loop_static_simple_gpu( PropMom, helper );
    merge_gpu_cpu_loop( PropMom, lmg, helper );

    CT_CHECK(check_equal_cd(lmr,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    CT_CHECK(check_equal_cd(lmrg,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    CT_CHECK(check_equal_cd(lmg,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    ZERO_OUT( lmg, POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk );
    ZERO_OUT( lmrg, POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk );

    S_loop_static_simple( PropMom, true, false, lm, -1, 0, true );
    S_loop_static_red( PropMom, true, false, lmr, -1, 0, true );
    S_pp_loop_static_simple_gpu( PropMom, helper );
    merge_gpu_cpu_loop( PropMom, lmg, helper );

    cuda_reset_prop_descr(&(PropMom->cuda_descr),POW4(PropMom->n_spin), PropMom->tu_data->unique_mi_l_sz,0.0);

    S_pp_loop_static_red_gpu( PropMom, helper );
    merge_gpu_cpu_loop( PropMom, lmrg, helper );

    CT_CHECK(check_equal_cd(lmr,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    CT_CHECK(check_equal_cd(lmrg,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    CT_CHECK(check_equal_cd(lmg,lm,POW2(n_orbffm)*PropMom->nk,1e-12));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lm);
    free(helper);
    free(lmrg);
    free(lmg);
}

CT_TEST_NAME(tu_prop_offload, "tu_prop,gpu,offload") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=6, .nky=6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    tu_prop_t* PropMom = tu_prop_init(mom);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );

    S_loop_static_simple( PropMom, false, false, lmr, -1, 0, true );
    tu_prop_ph_loop(PropMom, lmrg, NULL);
    CT_CHECK(check_equal_cd(lmrg,lmr,POW2(n_orbffm)*PropMom->nk,1e-12));

    S_loop_static_simple( PropMom, true, false, lmr, -1, 0, true );
    tu_prop_pp_loop(PropMom, lmrg, NULL);
    CT_CHECK(check_equal_cd(lmrg,lmr,POW2(n_orbffm)*PropMom->nk,1e-12));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}


CT_TEST_NAME(tu_prop_nofull, "tu_prop,gpu,nofull") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=6, .nky=6);
    diverge_model_internals_tu(mom, 1.5);

    tu_prop_t* PropMom = tu_prop_init(mom);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );
    PropMom->cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* helper = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );
    S_loop_static_simple( PropMom, false, false, lm, -1, 0, true );
    S_loop_static_red( PropMom, false, false, lmr, -1, 0, true );
    S_ph_loop_static_red_gpu(PropMom, helper);
    merge_gpu_cpu_loop(PropMom, lmrg, helper);

    cuda_reset_prop_descr(&(PropMom->cuda_descr),POW4(PropMom->n_spin), POW2(PropMom->n_orbff),0.0);
    S_ph_loop_static_simple_gpu(PropMom, helper);
    merge_gpu_cpu_loop(PropMom, lmg, helper);

    CT_CHECK(check_equal_cd(lmr,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    CT_CHECK(check_equal_cd(lmrg,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    CT_CHECK(check_equal_cd(lmg,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    ZERO_OUT(lmg, POW2(n_orbffm*mom->n_spin)*PropMom->nk);
    ZERO_OUT(lmrg, POW2(n_orbffm*mom->n_spin)*PropMom->nk);

    S_loop_static_simple( PropMom, true, false, lm, -1, 0, true );
    S_loop_static_red( PropMom, true, false, lmr, -1, 0, true );
    S_pp_loop_static_simple_gpu(PropMom, helper);
    merge_gpu_cpu_loop(PropMom, lmg, helper);

    cuda_reset_prop_descr(&(PropMom->cuda_descr),POW4(PropMom->n_spin), PropMom->tu_data->unique_mi_l_sz, 0.0);
    S_pp_loop_static_red_gpu(PropMom, helper);
    merge_gpu_cpu_loop(PropMom, lmrg, helper);

    CT_CHECK(check_equal_cd(lmr,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    CT_CHECK(check_equal_cd(lmrg,lm,POW2(n_orbffm)*PropMom->nk,1e-12));
    CT_CHECK(check_equal_cd(lmg,lm,POW2(n_orbffm)*PropMom->nk,1e-12));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lm);
    free(helper);
    free(lmrg);
    free(lmg);
}

CT_TEST_NAME(tu_prop_nofull_offload, "tu_prop,gpu,offload,nofull") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=6, .nky=6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    tu_prop_t* PropMom = tu_prop_init(mom);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );
    PropMom->cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );

    S_loop_static_simple( PropMom, false, false, lmr, -1, 0, true );
    tu_prop_ph_loop(PropMom, lmrg, NULL);
    CT_CHECK(check_equal_cd(lmrg,lmr,POW2(n_orbffm)*PropMom->nk,1e-12));

    S_loop_static_simple( PropMom, true, false, lmr, -1, 0, true );
    tu_prop_pp_loop(PropMom, lmrg, NULL);
    CT_CHECK(check_equal_cd(lmrg,lmr,POW2(n_orbffm)*PropMom->nk,1e-12));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}

CT_TEST_NAME(tu_prop_offload_simple, "tu_prop,gpu,offload,simple") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=6, .nky=6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    ((tu_data_t*)mom->internals->tu_data)->use_reduced_loop = false;
    tu_prop_t* PropMom = tu_prop_init(mom);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );

    S_loop_static_simple( PropMom, false, false, lmr, -1, 0, true );
    tu_prop_ph_loop(PropMom, lmrg, NULL);
    CT_CHECK(check_equal_cd(lmrg,lmr,POW2(n_orbffm)*PropMom->nk,1e-12));

    S_loop_static_simple( PropMom, true, false, lmr, -1, 0, true );
    tu_prop_pp_loop(PropMom, lmrg, NULL);
    CT_CHECK(check_equal_cd(lmrg,lmr,POW2(n_orbffm)*PropMom->nk,1e-12));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}

CT_TEST_NAME(tu_prop_offload_simple_nofull, "tu_prop,gpu,offload,simple,nofull") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=6, .nky=6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    ((tu_data_t*)mom->internals->tu_data)->use_reduced_loop = false;
    tu_prop_t* PropMom = tu_prop_init(mom);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );
    PropMom->cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );

    S_loop_static_simple( PropMom, false, false, lmr, -1, 0, true );
    tu_prop_ph_loop(PropMom, lmrg, NULL);
    CT_CHECK(check_equal_cd(lmrg,lmr,POW2(n_orbffm)*PropMom->nk,1e-12));

    S_loop_static_simple( PropMom, true, false, lmr, -1, 0, true );
    tu_prop_pp_loop(PropMom, lmrg, NULL);
    CT_CHECK(check_equal_cd(lmrg,lmr,POW2(n_orbffm)*PropMom->nk,1e-12));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}

#endif // DIVERGE_SKIP_TESTS
