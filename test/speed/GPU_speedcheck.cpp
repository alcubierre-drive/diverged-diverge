#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../../src/tu/diverge_interface.h"
#include "../model_examples/old.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/propagator_details.h"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/tu/vertex_details.h"
#include "../../src/tu/projection.h"
#include "../../src/tu/projection_handler.h"

#ifdef USE_CUDA
TEST_CASE("loop speed", "[.][loopspeed]") {
    MPILoglevelWrapper LL(5);
    diverge_model_t* mom = gen_square_lat_hub(30, 30,2,2,1,-0.1, 3, -0.4,45,45);
    diverge_model_internals_tu(mom, 2.1);
    tu_prop_t* PropMom = tu_prop_init(mom);

    index_t n_orbffm = PropMom->n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm)*PropMom->nk,sizeof(complex128_t));

    tu_prop_GF_zero_T( PropMom, I128*0.1, NULL );
    double tick = diverge_mpi_wtime();
    for (index_t i=0; i<2; ++i)
        S_ph_loop_static_red_gpu( PropMom, lm );
    double tock = diverge_mpi_wtime();
    mpi_tim_printf("time 2 times ph loop reduced %4.4f \n", tock -tick);
    tick = diverge_mpi_wtime();
    for (index_t i=0; i<2; ++i)
        S_pp_loop_static_red_gpu( PropMom, lm );
    tock = diverge_mpi_wtime();
    mpi_tim_printf("time 2 times pp loop reduced %4.4f \n", tock -tick);

    tu_prop_free( PropMom );
    diverge_model_free(mom);
    free(lm);
}

TEST_CASE("loop proj", "[.][projspeed]") {
    MPILoglevelWrapper LL(5);
    diverge_model_t* mom = gen_honeycomb_lat_hub(30,30,1,1);
    diverge_model_internals_tu(mom, 2.5);
    projection_t* ProjMom = projection_init(mom);
    tu_prop_t* LoopMom = tu_prop_init(mom);
    Vertex* VertMom = Vertex_init(mom, ProjMom, LoopMom, 1,1,1,0);
    complex128_t* r2 = (complex128_t*)calloc(VertMom->full_vert_size,sizeof(complex128_t));

    double tick = diverge_mpi_wtime();
    P_projection(r2,VertMom,ProjMom);
    C_projection(r2,VertMom,ProjMom);
    D_projection(r2,VertMom,ProjMom);

    double tock = diverge_mpi_wtime();
    mpi_log_printf("time 3 projections took %4.4f \n", tock -tick);

    Vertex_free( VertMom );
    tu_prop_free( LoopMom );
    projection_free( ProjMom );
    diverge_model_free(mom);
    free(r2);
}

#endif

#endif // DIVERGE_SKIP_TESTS
