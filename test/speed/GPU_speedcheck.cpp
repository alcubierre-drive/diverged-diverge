#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../../src/tu/diverge_interface.h"
#include "../model_examples/old.h"
#include "../../src/tu/propagator.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection.h"
#include "../../src/tu/projection_handler.h"

#ifdef USE_CUDA
TEST_CASE("loop speed", "[.][loopspeed]") {
    MPILoglevelWrapper LL(5);
    diverge_model_t* mom = gen_square_lat_hub(30, 30,2,2,1,-0.1, 3, -0.4,45,45);
    diverge_model_internals_tu(mom, 2.1);
    tu_loop_t PropMom (mom);

    index_t n_orbffm = PropMom.n_orbff;
    complex128_t* lm = (complex128_t*)calloc(POW2(n_orbffm)*PropMom.nk,sizeof(complex128_t));

    PropMom.GF_zero_T(I128*0.1);
    double tick = diverge_mpi_wtime();
    for(index_t i = 0; i < 2; ++i){
        PropMom.S_ph_loop_static_red_gpu(lm);
    }
    double tock = diverge_mpi_wtime();
    mpi_log_printf("time 2 times ph loop reduced %4.4f \n", tock -tick);
    tick = diverge_mpi_wtime();
    for(index_t i = 0; i < 2; ++i){
        PropMom.S_pp_loop_static_red_gpu(lm);
    }
    tock = diverge_mpi_wtime();
    mpi_log_printf("time 2 times pp loop reduced %4.4f \n", tock -tick);

    diverge_model_free(mom);
    free(lm);
}

TEST_CASE("loop proj", "[.][projspeed]") {
    MPILoglevelWrapper LL(5);
    diverge_model_t* mom = gen_honeycomb_lat_hub(30,30,1,1);
    diverge_model_internals_tu(mom, 2.5);
    projection_t* ProjMom = projection_init(mom);
    tu_loop_t LoopMom (mom);
    Vertex VertMom(mom, ProjMom, &LoopMom);
    complex128_t* r2 = (complex128_t*)calloc(VertMom.full_vert_size,sizeof(complex128_t));

    double tick = diverge_mpi_wtime();
    P_projection(r2,&VertMom,ProjMom);
    C_projection(r2,&VertMom,ProjMom);
    D_projection(r2,&VertMom,ProjMom);

    double tock = diverge_mpi_wtime();
    mpi_log_printf("time 3 projections took %4.4f \n", tock -tick);

    projection_free( ProjMom );
    diverge_model_free(mom);
    free(r2);
}

#endif

#endif // DIVERGE_SKIP_TESTS
