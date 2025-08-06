#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME tu_prop_spaces
#include "../ctest.h"

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../../src/tu/diverge_interface.h"
#include "../model_examples/old.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/propagator_details.h"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/tu/vertex.h"
#include "../../src/tu/projection.h"

#ifndef USE_GF_FLOATS
#ifndef USE_MPI
CT_TEST_NAME(tu_prop_orb_real, "tu_prop,orb:real") {
    diverge_model_t* mom = gen_square_lat_hub(12, 12, 1, 1);
    diverge_model_t* orb = gen_square_lat_hub(1, 1, 12, 12);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mom, 1.1);

    tu_prop_t* PropMom = tu_prop_init(mom);
    tu_prop_t* PropOrb = tu_prop_init(orb);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_projection_t* ProjOrb = tu_projection_init(orb);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );
    tu_vertex_t* Vertorb = tu_vertex_init( mom, ProjOrb, PropOrb, 1,1,1,0 );

    index_t n_orbffo = PropOrb->n_orbff;
    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lo = (complex128_t*)calloc(POW2(n_orbffo*orb->n_spin)*PropOrb->nk,sizeof(complex128_t));
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T(PropOrb, I128*0.1, NULL);
    tu_prop_GF_zero_T(PropMom, I128*0.1, NULL);

    CT_CHECK(std::abs(trace(PropMom->gf_buf,PropMom->n_orb,PropMom->nk)
                  -trace(PropOrb->gf_buf,PropOrb->n_orb,PropOrb->nk)) < 1e-10);

    S_loop_static_rs(PropOrb, true, false, lo);
    S_loop_static_simple(PropMom, true, false, lm, -1, 0, true);
    S_loop_static_red(PropMom, true, false, lmr, -1, 0, true);
    CT_CHECK(std::abs(trace(lm,n_orbffm,PropMom->nk)
                  -trace(lo,n_orbffo,PropOrb->nk)) < 1e-10);

    CT_CHECK(std::abs(trace(lmr,n_orbffm,PropMom->nk)
                  -trace(lo,n_orbffo,PropOrb->nk)) < 1e-10);

    CT_CHECK(std::abs(trace(lmr,n_orbffm,PropMom->nk)
                  -trace(lm,n_orbffm,PropMom->nk)) < 1e-10);

    S_loop_static_simple(PropMom, false, false, lm, -1, 0, true);
    S_loop_static_red(PropMom, false, false, lmr, -1, 0, true);
    S_loop_static_rs(PropOrb, false, false, lo);

    CT_CHECK(std::abs(trace(lm,n_orbffm,PropMom->nk)
                  -trace(lo,n_orbffo,PropOrb->nk)) < 1e-10);

    CT_CHECK(std::abs(trace(lmr,n_orbffm,PropMom->nk)
                  -trace(lo,n_orbffo,PropOrb->nk)) < 1e-10);

    CT_CHECK(std::abs(trace(lmr,n_orbffm,PropMom->nk)
                  -trace(lm,n_orbffm,PropMom->nk)) < 1e-10);

    tu_vertex_free( VertMom );
    tu_vertex_free( Vertorb );
    tu_prop_free( PropMom );
    tu_prop_free( PropOrb );
    tu_projection_free( ProjMom );
    tu_projection_free( ProjOrb );
    diverge_model_free(orb);
    diverge_model_free(mom);
    free(lo);
    free(lmr);
    free(lm);
}


CT_TEST_NAME(tu_prop_basis, "tu_prop,basis") {
    diverge_model_t* mom = gen_honeycomb_lat_hub(12, 12);
    diverge_model_t* orb = gen_honeycomb_lat_hub(1, 1, 12, 12);

    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mom, 1.1);

    tu_prop_t* PropMom = tu_prop_init( mom );
    tu_prop_t* PropOrb = tu_prop_init( orb );
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_projection_t* ProjOrb = tu_projection_init(orb);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );
    tu_vertex_t* Vertorb = tu_vertex_init( mom, ProjOrb, PropOrb, 1,1,1,0 );

    index_t n_orbffo = PropOrb->n_orbff;
    index_t n_orbffm = PropMom->n_orbff;
    CT_CHECK(check_abs_sorted_equal(mom->internals->E,orb->internals->E,PropOrb->n_orb*PropOrb->nk));
    CT_CHECK(std::abs(trace(mom->internals->ham,PropMom->n_orb,PropMom->nk)
                  -trace(orb->internals->ham,PropOrb->n_orb,PropOrb->nk)) < 1e-10);
    CT_CHECK(std::abs(trace(mom->internals->ham,PropMom->n_orb,PropMom->nk)
                  -trace(mom->internals->E,1,PropOrb->n_orb*PropOrb->nk)) < 1e-10);
    CT_CHECK(std::abs(trace(orb->internals->ham,PropOrb->n_orb,PropOrb->nk)
                  -trace(orb->internals->E,1,PropOrb->n_orb*PropOrb->nk)) < 1e-10);


    complex128_t* lo = (complex128_t*)calloc(POW2(n_orbffo*orb->n_spin)*PropOrb->nk,sizeof(complex128_t));
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T(PropOrb, I128*0.1, NULL);
    tu_prop_GF_zero_T(PropMom, I128*0.1, NULL);

    CT_CHECK(std::abs(trace(mom->internals->greens,PropMom->n_orb,PropMom->nk)
                  -trace(orb->internals->greens,PropOrb->n_orb,PropOrb->nk)) < 1e-10);

    S_loop_static_rs(PropOrb, true, false, lo);
    S_loop_static_simple(PropMom, true, false, lm, -1, 0, true);
    S_loop_static_red(PropMom, true, false, lmr, -1, 0, true);
    CT_CHECK(std::abs(trace(lm,n_orbffm,PropMom->nk)
                  -trace(lo,n_orbffo,PropOrb->nk)) < 1e-9);

    CT_CHECK(std::abs(trace(lmr,n_orbffm,PropMom->nk)
                  -trace(lo,n_orbffo,PropOrb->nk)) < 1e-9);

    CT_CHECK(std::abs(trace(lmr,n_orbffm,PropMom->nk)
                  -trace(lm,n_orbffm,PropMom->nk)) < 1e-9);

    S_loop_static_simple(PropMom, false, false, lm, -1, 0, true);
    S_loop_static_red(PropMom, false, false, lmr, -1, 0, true);
    S_loop_static_rs(PropOrb, false, false, lo);

    CT_CHECK(std::abs(trace(lm,n_orbffm,PropMom->nk)
                  -trace(lo,n_orbffo,PropOrb->nk)) < 1e-9);

    CT_CHECK(std::abs(trace(lmr,n_orbffm,PropMom->nk)
                  -trace(lo,n_orbffo,PropOrb->nk)) < 1e-9);

    CT_CHECK(std::abs(trace(lmr,n_orbffm,PropMom->nk)
                  -trace(lm,n_orbffm,PropMom->nk)) < 1e-9);

    tu_vertex_free( Vertorb );
    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_prop_free( PropOrb );
    tu_projection_free( ProjMom );
    tu_projection_free( ProjOrb );
    diverge_model_free(orb);
    diverge_model_free(mom);
    free(lo);
    free(lmr);
    free(lm);
}


CT_TEST_NAME(tu_prop_variants, "tu_prop,variants") {
    diverge_model_t* mom = gen_honeycomb_lat_hub(12, 12);
    diverge_model_internals_tu(mom, 1.1);

    tu_prop_t* PropMom = tu_prop_init( mom );
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T(PropMom, I128*0.1, NULL);
    S_loop_static_simple(PropMom, false, false, lm, -1, 0, true);
    S_loop_static_red(PropMom, false, false, lmr, -1, 0, true);
    CT_CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk));
    S_loop_static_simple(PropMom, true, false, lm, -1, 0, true);
    S_loop_static_red(PropMom, true, false, lmr, -1, 0, true);
    CT_CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk));

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lm);
}
#endif


CT_TEST_NAME(tu_prop_variants_refine, "tu_prop,variants,refined") {
    diverge_model_t* mom = gen_square_lat_hub(12, 12,1,1,1,-0.1, 3, -0.4,15, 15);
    diverge_model_internals_tu(mom, 1.1);

    tu_prop_t* PropMom = tu_prop_init( mom );
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, 1,1,1,0 );

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T(PropMom, I128*0.1, NULL);
    S_loop_static_simple(PropMom, false, false, lm, -1, 0, true);
    S_loop_static_red(PropMom, false, false, lmr, -1, 0, true);
    bool PH_equal = check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk);
    S_loop_static_simple(PropMom, true, false, lm, -1, 0, true);
    S_loop_static_red(PropMom, true, false, lmr, -1, 0, true);
    bool PP_equal = check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk);

    tu_vertex_free( VertMom );
    tu_prop_free( PropMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(lmr);
    free(lm);
    CT_CHECK( PH_equal );
    CT_CHECK( PP_equal );
}

CT_TEST_NAME(tu_prop_variants_refine_conv, "tu_prop,variants,refined,convergence") {
    diverge_model_t* mom = gen_square_lat_hub(90, 90,1,1,1,-0.1, 3, -0.4,3, 3);
    diverge_model_t* mom2 = gen_square_lat_hub(90, 90,1,1,1,-0.1, 3, -0.4,1, 1);
    diverge_model_internals_tu(mom, 4.1);
    diverge_model_internals_tu(mom2, 4.1);

    tu_prop_t* PropMom = tu_prop_init( mom );
    tu_prop_t* PropMom2 = tu_prop_init( mom2 );

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));
    complex128_t* lmr = (complex128_t*)calloc(POW2(n_orbffm*mom->n_spin)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T(PropMom, I128*0.1, NULL);
    tu_prop_GF_zero_T(PropMom2, I128*0.1, NULL);
    S_loop_static_simple(PropMom, false, false, lm, -1, 0, true);
    S_loop_static_simple(PropMom2, false, false, lmr, -1, 0, true);
    CT_CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk, 0.1));
    S_loop_static_simple(PropMom, true, false, lm, -1, 0, true);
    S_loop_static_simple(PropMom2, true, false, lmr, -1, 0, true);
    CT_CHECK(check_equal(lmr,lm,POW2(n_orbffm)*PropMom->nk, 0.1));

    tu_prop_free( PropMom );
    tu_prop_free( PropMom2 );
    diverge_model_free(mom);
    diverge_model_free(mom2);
    free(lmr);
    free(lm);
}


#endif

#endif // DIVERGE_SKIP_TESTS
