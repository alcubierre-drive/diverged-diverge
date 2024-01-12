#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../../src/tu/diverge_interface.hpp"
#include "../model_examples/old.h"
#include "../../src/tu/global.hpp"
#include "../../src/tu/propagator.hpp"
#include "../../src/tu/cuda_init.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection_wrapper.hpp"

#ifndef USE_GF_FLOATS
#ifndef USE_MPI
#ifdef USE_CUDA
TEST_CASE("momentum loop cpu gpu variants", "[prop][prop_gpu]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);

    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);


    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* helper = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);
    PropMom.S_loop_static_simple<false>(lm);
    PropMom.S_loop_static_red<false>(lmr);
    PropMom.S_ph_loop_static_red_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmrg, helper);

    cuda_reset_prop_descr(&(PropMom.cuda_descr),POW4(PropMom.n_spin), POW2(PropMom.n_orbff),0.0);

    PropMom.S_ph_loop_static_simple_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmg, helper);
    
    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom.nk));
    PropMom.S_loop_static_simple<true>(lm);
    PropMom.S_loop_static_red<true>(lmr);
    memset((void*)lmg,0,POW2(n_orbffm*mom->n_spin)*PropMom.nk*sizeof(complex128_t));
    memset((void*)lmrg,0,POW2(n_orbffm*mom->n_spin)*PropMom.nk*sizeof(complex128_t));
    PropMom.S_pp_loop_static_simple_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmg, helper);
    cuda_reset_prop_descr(&(PropMom.cuda_descr),POW4(PropMom.n_spin), PropMom.tu_data->unique_mi_l.size(),0.0);
    PropMom.S_pp_loop_static_red_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmrg, helper);
    
    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom.nk));

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
    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);

    PropMom.S_loop_static_simple<false>(lmr);
    PropMom.ph_loop(lmrg);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom.nk));

    PropMom.S_loop_static_simple<true>(lmr);
    PropMom.pp_loop(lmrg);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom.nk));

    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}


TEST_CASE("momentum loop cpu gpu variants  no full GF", "[prop][prop_gpu]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);

    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);
    PropMom.cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* helper = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);
    PropMom.S_loop_static_simple<false>(lm);
    PropMom.S_loop_static_red<false>(lmr);
    PropMom.S_ph_loop_static_red_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmrg, helper);

    cuda_reset_prop_descr(&(PropMom.cuda_descr),POW4(PropMom.n_spin), POW2(PropMom.n_orbff),0.0);

    PropMom.S_ph_loop_static_simple_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmg, helper);

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom.nk));
    PropMom.S_loop_static_simple<true>(lm);
    PropMom.S_loop_static_red<true>(lmr);
    memset((void*)lmg,0,POW2(n_orbffm*mom->n_spin)*PropMom.nk*sizeof(complex128_t));
    memset((void*)lmrg,0,POW2(n_orbffm*mom->n_spin)*PropMom.nk*sizeof(complex128_t));
    PropMom.S_pp_loop_static_simple_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmg, helper);
    cuda_reset_prop_descr(&(PropMom.cuda_descr),POW4(PropMom.n_spin), PropMom.tu_data->unique_mi_l.size(),0.0);
    PropMom.S_pp_loop_static_red_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmrg, helper);

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom.nk));

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
    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);
    PropMom.cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);

    PropMom.S_loop_static_simple<false>(lmr);
    PropMom.ph_loop(lmrg);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom.nk));

    PropMom.S_loop_static_simple<true>(lmr);
    PropMom.pp_loop(lmrg);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom.nk));

    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}

TEST_CASE("momentum loop cpu gpu variants simple", "[prop][prop_gpu]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->use_reduced_loop = false;
    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);


    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* helper = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);
    PropMom.S_loop_static_simple<false>(lm);
    PropMom.S_loop_static_red<false>(lmr);
    PropMom.S_ph_loop_static_red_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmrg, helper);

    cuda_reset_prop_descr(&(PropMom.cuda_descr),POW4(PropMom.n_spin), POW2(PropMom.n_orbff),0.0);

    PropMom.S_ph_loop_static_simple_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmg, helper);

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom.nk));
    PropMom.S_loop_static_simple<true>(lm);
    PropMom.S_loop_static_red<true>(lmr);
    memset((void*)lmg,0,POW2(n_orbffm*mom->n_spin)*PropMom.nk*sizeof(complex128_t));
    memset((void*)lmrg,0,POW2(n_orbffm*mom->n_spin)*PropMom.nk*sizeof(complex128_t));
    PropMom.S_pp_loop_static_simple_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmg, helper);
    cuda_reset_prop_descr(&(PropMom.cuda_descr),POW4(PropMom.n_spin), PropMom.tu_data->unique_mi_l.size(),0.0);
    PropMom.S_pp_loop_static_red_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmrg, helper);

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom.nk));

    diverge_model_free(mom);
    free(lmr);
    free(lm);
    free(helper);
    free(lmrg);
    free(lmg);
}

TEST_CASE("momentum loop cpu gpu offloading simple", "[prop][prop_offload]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    ((tu_data_t*)mom->internals->tu_data)->use_reduced_loop = false;
    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);

    PropMom.S_loop_static_simple<false>(lmr);
    PropMom.ph_loop(lmrg);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom.nk));

    PropMom.S_loop_static_simple<true>(lmr);
    PropMom.pp_loop(lmrg);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom.nk));

    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}


TEST_CASE("momentum loop cpu gpu variants  no full GF simple", "[prop][prop_gpu]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->use_reduced_loop = false;

    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);
    PropMom.cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* helper = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);
    PropMom.S_loop_static_simple<false>(lm);
    PropMom.S_loop_static_red<false>(lmr);
    PropMom.S_ph_loop_static_red_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmrg, helper);

    cuda_reset_prop_descr(&(PropMom.cuda_descr),POW4(PropMom.n_spin), POW2(PropMom.n_orbff),0.0);

    PropMom.S_ph_loop_static_simple_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmg, helper);

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom.nk));
    PropMom.S_loop_static_simple<true>(lm);
    PropMom.S_loop_static_red<true>(lmr);
    memset((void*)lmg,0,POW2(n_orbffm*mom->n_spin)*PropMom.nk*sizeof(complex128_t));
    memset((void*)lmrg,0,POW2(n_orbffm*mom->n_spin)*PropMom.nk*sizeof(complex128_t));
    PropMom.S_pp_loop_static_simple_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmg, helper);
    cuda_reset_prop_descr(&(PropMom.cuda_descr),POW4(PropMom.n_spin), PropMom.tu_data->unique_mi_l.size(),0.0);
    PropMom.S_pp_loop_static_red_gpu(helper);
    PropMom.merge_gpu_cpu_loop(lmrg, helper);

    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmrg,lm,POW2(n_orbffm)*PropMom.nk));
    CHECK(check_equal(lmg,lm,POW2(n_orbffm)*PropMom.nk));

    diverge_model_free(mom);
    free(lmr);
    free(lm);
    free(helper);
    free(lmrg);
    free(lmg);
}

TEST_CASE("momentum loop cpu gpu offloading no full GF simple", "[prop][prop_offload]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mom, 1.5);
    ((tu_data_t*)mom->internals->tu_data)->offload_cpu = 0.5;
    ((tu_data_t*)mom->internals->tu_data)->use_reduced_loop = false;
    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);
    PropMom.cuda_descr.prop_preload_full_gf = false;

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmrg = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);

    PropMom.S_loop_static_simple<false>(lmr);
    PropMom.ph_loop(lmrg);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom.nk));

    PropMom.S_loop_static_simple<true>(lmr);
    PropMom.pp_loop(lmrg);
    CHECK(check_equal(lmrg,lmr,POW2(n_orbffm)*PropMom.nk));

    diverge_model_free(mom);
    free(lmr);
    free(lmrg);
}

#endif
#endif
#endif

#endif // DIVERGE_SKIP_TESTS
