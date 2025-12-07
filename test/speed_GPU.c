#define CT_SUITE_NAME speed
#include "ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../src/diverge_model.h"
#include "../src/diverge_momentum_gen.h"
#include "../src/tu/diverge_interface.h"
#include "../src/tu/propagator.h"
#include "../src/tu/propagator_details.h"
#include "../src/diverge_model_internals.h"
#include "../src/tu/vertex_details.h"
#include "../src/tu/projection.h"
#include "../src/tu/projection_handler.h"
#include "helper_functions.h"
#include "models/all.h"

CT_TEST_HIDDEN(loopspeed) {
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=30, .nky=30, .lx=2, .ly=2, .nkxf=45, .nkyf=45);
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

CT_TEST_HIDDEN(projspeed) {
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=30, .nky=30);
    diverge_model_internals_tu(mom, 2.5);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_prop_t* LoopMom = tu_prop_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init(mom, ProjMom, LoopMom, 1,1,1,0);
    complex128_t* r2 = (complex128_t*)calloc(VertMom->full_vert_size,sizeof(complex128_t));

    double tick = diverge_mpi_wtime();
    P_projection(r2,VertMom,ProjMom);
    C_projection(r2,VertMom,ProjMom);
    D_projection(r2,VertMom,ProjMom);

    double tock = diverge_mpi_wtime();
    mpi_tim_printf("time 3 projections took %4.4f \n", tock -tick);

    tu_vertex_free( VertMom );
    tu_prop_free( LoopMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(r2);
}

#endif // DIVERGE_SKIP_TESTS
