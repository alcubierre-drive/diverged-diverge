#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../../src/tu/diverge_interface.h"
#include "../model_examples/old.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/propagator_details.h"
#include "../../src/tu/cuda_init.h"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/tu/vertex.h"
#include "../../src/tu/projection.h"

#ifndef USE_GF_FLOATS
#ifndef USE_MPI
#ifdef USE_CUDA
TEST_CASE("momentum loop cpu gpu variants", "[prop][prop_gpu]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);

    tu_prop_t* PropMom = tu_prop_init(mom);
    projection_t* ProjMom = projection_init(mom);
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

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom->nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom->nk));
    ZERO_OUT( lmg, POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk );
    ZERO_OUT( lmrg, POW2(n_orbffm*POW2(mom->n_spin))*PropMom->nk );

    S_loop_static_simple( PropMom, true, false, lm, -1, 0, true );
    S_loop_static_red( PropMom, true, false, lmr, -1, 0, true );
    S_pp_loop_static_simple_gpu( PropMom, helper );
    merge_gpu_cpu_loop( PropMom, lmg, helper );

    cuda_reset_prop_descr(&(PropMom->cuda_descr),POW4(PropMom->n_spin), PropMom->tu_data->unique_mi_l_sz,0.0);

    S_pp_loop_static_red_gpu( PropMom, helper );
    merge_gpu_cpu_loop( PropMom, lmrg, helper );

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom->nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom->nk));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lm);
    free(helper);
    free(lmrg);
    free(lmg);
}

TEST_CASE("momentum loop cpu gpu offloading", "[prop][prop_offload]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    tu_prop_t* PropMom = tu_prop_init(mom);
    projection_t* ProjMom = projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );

    S_loop_static_simple( PropMom, false, false, lmr, -1, 0, true );
    tu_prop_ph_loop(PropMom, lmrg, NULL);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom->nk));

    S_loop_static_simple( PropMom, true, false, lmr, -1, 0, true );
    tu_prop_pp_loop(PropMom, lmrg, NULL);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom->nk));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}


TEST_CASE("momentum loop cpu gpu variants  no full GF", "[prop][prop_gpu_nofull]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);

    tu_prop_t* PropMom = tu_prop_init(mom);
    projection_t* ProjMom = projection_init(mom);
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

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom->nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom->nk));
    ZERO_OUT(lmg, POW2(n_orbffm*mom->n_spin)*PropMom->nk);
    ZERO_OUT(lmrg, POW2(n_orbffm*mom->n_spin)*PropMom->nk);

    S_loop_static_simple( PropMom, true, false, lm, -1, 0, true );
    S_loop_static_red( PropMom, true, false, lmr, -1, 0, true );
    S_pp_loop_static_simple_gpu(PropMom, helper);
    merge_gpu_cpu_loop(PropMom, lmg, helper);

    cuda_reset_prop_descr(&(PropMom->cuda_descr),POW4(PropMom->n_spin), PropMom->tu_data->unique_mi_l_sz, 0.0);
    S_pp_loop_static_red_gpu(PropMom, helper);
    merge_gpu_cpu_loop(PropMom, lmrg, helper);

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom->nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom->nk));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lm);
    free(helper);
    free(lmrg);
    free(lmg);
}

TEST_CASE("momentum loop cpu gpu offloading no full GF", "[prop][prop_offload]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    tu_prop_t* PropMom = tu_prop_init(mom);
    projection_t* ProjMom = projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );
    PropMom->cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );

    S_loop_static_simple( PropMom, false, false, lmr, -1, 0, true );
    tu_prop_ph_loop(PropMom, lmrg, NULL);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom->nk));

    S_loop_static_simple( PropMom, true, false, lmr, -1, 0, true );
    tu_prop_pp_loop(PropMom, lmrg, NULL);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom->nk));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}

TEST_CASE("momentum loop cpu gpu offloading simple", "[prop][prop_offload]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    ((tu_data_t*)mom->internals->tu_data)->use_reduced_loop = false;
    tu_prop_t* PropMom = tu_prop_init(mom);
    projection_t* ProjMom = projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );

    S_loop_static_simple( PropMom, false, false, lmr, -1, 0, true );
    tu_prop_ph_loop(PropMom, lmrg, NULL);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom->nk));

    S_loop_static_simple( PropMom, true, false, lmr, -1, 0, true );
    tu_prop_pp_loop(PropMom, lmrg, NULL);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom->nk));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}
TEST_CASE("momentum loop cpu gpu offloading no full GF simple", "[prop][prop_offload]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    ((tu_data_t*)mom->internals->tu_data)->use_reduced_loop = false;
    tu_prop_t* PropMom = tu_prop_init(mom);
    projection_t* ProjMom = projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );
    PropMom->cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );

    S_loop_static_simple( PropMom, false, false, lmr, -1, 0, true );
    tu_prop_ph_loop(PropMom, lmrg, NULL);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom->nk));

    S_loop_static_simple( PropMom, true, false, lmr, -1, 0, true );
    tu_prop_pp_loop(PropMom, lmrg, NULL);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom->nk));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}

#endif
#endif
#endif

#endif // DIVERGE_SKIP_TESTS
