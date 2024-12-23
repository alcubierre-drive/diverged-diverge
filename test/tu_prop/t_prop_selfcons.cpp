#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../../src/tu/diverge_interface.hpp"
#include "../model_examples/old.h"
#include "../../src/tu/propagator.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection_wrapper.hpp"

#ifndef USE_GF_FLOATS
#ifndef USE_MPI
TEST_CASE("momentum vs realspace tu_loop_t", "[prop][prop_orbVSreal]")
{
    diverge_model_t* mom = gen_square_lat_hub(12, 12, 1, 1);
    diverge_model_t* orb = gen_square_lat_hub(1, 1, 12, 12);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mom, 1.1);

    tu_loop_t PropMom (mom);
    tu_loop_t PropOrb (orb);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);
    Projection ProjOrb (orb);
    Vertex Vertorb(mom, &ProjOrb, &PropOrb);


    index_t n_orbffo = PropOrb.n_orbff;
    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lo = (complex128_t*)calloc(POW2(n_orbffo*orb->n_spin)*PropOrb.nk,sizeof(complex128_t));
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropOrb.GF_zero_T(I128*0.1);
    PropMom.GF_zero_T(I128*0.1);

    CHECK(std::abs(trace(PropMom.gf_buf,PropMom.n_orb,PropMom.nk)
                  -trace(PropOrb.gf_buf,PropOrb.n_orb,PropOrb.nk)) < 1e-10);

    PropOrb.S_loop_static_rs<true>(lo);
    PropMom.S_loop_static_simple<true>(lm);
    PropMom.S_loop_static_red<true>(lmr);
    CHECK(std::abs(trace(lm,n_orbffm,PropMom.nk)
                  -trace(lo,n_orbffo,PropOrb.nk)) < 1e-10);

    CHECK(std::abs(trace(lmr,n_orbffm,PropMom.nk)
                  -trace(lo,n_orbffo,PropOrb.nk)) < 1e-10);

    CHECK(std::abs(trace(lmr,n_orbffm,PropMom.nk)
                  -trace(lm,n_orbffm,PropMom.nk)) < 1e-10);

    PropMom.S_loop_static_simple<false>(lm);
    PropMom.S_loop_static_red<false>(lmr);
    PropOrb.S_loop_static_rs<false>(lo);

    CHECK(std::abs(trace(lm,n_orbffm,PropMom.nk)
                  -trace(lo,n_orbffo,PropOrb.nk)) < 1e-10);

    CHECK(std::abs(trace(lmr,n_orbffm,PropMom.nk)
                  -trace(lo,n_orbffo,PropOrb.nk)) < 1e-10);

    CHECK(std::abs(trace(lmr,n_orbffm,PropMom.nk)
                  -trace(lm,n_orbffm,PropMom.nk)) < 1e-10);

    diverge_model_free(orb);
    diverge_model_free(mom);
    free(lo);
    free(lmr);
    free(lm);
}


TEST_CASE("momentum vs realspace basis tu_loop_t", "[prop][prop_basis]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(12, 12);
    diverge_model_t* orb = gen_honeycomb_lat_hub(1, 1, 12, 12);

    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mom, 1.1);

    tu_loop_t PropMom (mom);
    tu_loop_t PropOrb (orb);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);
    Projection ProjOrb (orb);
    Vertex Vertorb(mom, &ProjOrb, &PropOrb);


    index_t n_orbffo = PropOrb.n_orbff;
    index_t n_orbffm = PropMom.n_orbff;
    CHECK(check_abs_sorted_equal(mom->internals->E,orb->internals->E,PropOrb.n_orb*PropOrb.nk));
    CHECK(std::abs(trace(mom->internals->ham,PropMom.n_orb,PropMom.nk)
                  -trace(orb->internals->ham,PropOrb.n_orb,PropOrb.nk)) < 1e-10);
    CHECK(std::abs(trace(mom->internals->ham,PropMom.n_orb,PropMom.nk)
                  -trace(mom->internals->E,1,PropOrb.n_orb*PropOrb.nk)) < 1e-10);
    CHECK(std::abs(trace(orb->internals->ham,PropOrb.n_orb,PropOrb.nk)
                  -trace(orb->internals->E,1,PropOrb.n_orb*PropOrb.nk)) < 1e-10);


    complex128_t* lo = (complex128_t*)calloc(POW2(n_orbffo*orb->n_spin)*PropOrb.nk,sizeof(complex128_t));
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropOrb.GF_zero_T(I128*0.1);
    PropMom.GF_zero_T(I128*0.1);

    CHECK(std::abs(trace(mom->internals->greens,PropMom.n_orb,PropMom.nk)
                  -trace(orb->internals->greens,PropOrb.n_orb,PropOrb.nk)) < 1e-10);

    PropOrb.S_loop_static_rs<true>(lo);
    PropMom.S_loop_static_simple<true>(lm);
    PropMom.S_loop_static_red<true>(lmr);
    CHECK(std::abs(trace(lm,n_orbffm,PropMom.nk)
                  -trace(lo,n_orbffo,PropOrb.nk)) < 1e-9);

    CHECK(std::abs(trace(lmr,n_orbffm,PropMom.nk)
                  -trace(lo,n_orbffo,PropOrb.nk)) < 1e-9);

    CHECK(std::abs(trace(lmr,n_orbffm,PropMom.nk)
                  -trace(lm,n_orbffm,PropMom.nk)) < 1e-9);

    PropMom.S_loop_static_simple<false>(lm);
    PropMom.S_loop_static_red<false>(lmr);
    PropOrb.S_loop_static_rs<false>(lo);

    CHECK(std::abs(trace(lm,n_orbffm,PropMom.nk)
                  -trace(lo,n_orbffo,PropOrb.nk)) < 1e-9);

    CHECK(std::abs(trace(lmr,n_orbffm,PropMom.nk)
                  -trace(lo,n_orbffo,PropOrb.nk)) < 1e-9);

    CHECK(std::abs(trace(lmr,n_orbffm,PropMom.nk)
                  -trace(lm,n_orbffm,PropMom.nk)) < 1e-9);

    diverge_model_free(orb);
    diverge_model_free(mom);
    free(lo);
    free(lmr);
    free(lm);
}


TEST_CASE("momentum loop variants", "[prop][prop_var]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(12, 12);
    diverge_model_internals_tu(mom, 1.1);

    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);
    PropMom.S_loop_static_simple<false>(lm);
    PropMom.S_loop_static_red<false>(lmr);
    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));
    PropMom.S_loop_static_simple<true>(lm);
    PropMom.S_loop_static_red<true>(lmr);
    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk));

    diverge_model_free(mom);
    free(lmr);
    free(lm);
}
#endif


TEST_CASE("momentum loop variants refine", "[prop][prop_ref][cmp]")
{
    diverge_model_t* mom = gen_square_lat_hub(12, 12,1,1,1,-0.1, 3, -0.4,15, 15);
    diverge_model_internals_tu(mom, 1.1);

    tu_loop_t PropMom (mom);
    Projection ProjMom (mom);
    Vertex VertMom(mom, &ProjMom, &PropMom);

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);
    PropMom.S_loop_static_simple<false>(lm);
    PropMom.S_loop_static_red<false>(lmr);
    bool PH_equal = check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk);
    PropMom.S_loop_static_simple<true>(lm);
    PropMom.S_loop_static_red<true>(lmr);
    bool PP_equal = check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk);

    diverge_model_free(mom);
    free(lmr);
    free(lm);
    CHECK( PH_equal );
    CHECK( PP_equal );
}

TEST_CASE("momentum loop variants refine convergence", "[prop][prop_ref][conv]")
{
    diverge_model_t* mom = gen_square_lat_hub(90, 90,1,1,1,-0.1, 3, -0.4,3, 3);
    diverge_model_t* mom2 = gen_square_lat_hub(90, 90,1,1,1,-0.1, 3, -0.4,1, 1);
    diverge_model_internals_tu(mom, 4.1);
    diverge_model_internals_tu(mom2, 4.1);

    tu_loop_t PropMom (mom);
    tu_loop_t PropMom2 (mom2);

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);
    PropMom2.GF_zero_T(I128*0.1);
    PropMom.S_loop_static_simple<false>(lm);
    PropMom2.S_loop_static_simple<false>(lmr);
    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk, 0.1));
    PropMom.S_loop_static_simple<true>(lm);
    PropMom2.S_loop_static_simple<true>(lmr);
    CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom.nk, 0.1));

    diverge_model_free(mom);
    diverge_model_free(mom2);
    free(lmr);
    free(lm);
}


#endif

#endif // DIVERGE_SKIP_TESTS
