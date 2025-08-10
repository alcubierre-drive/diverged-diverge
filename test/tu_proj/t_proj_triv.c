#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME tu_proj_triv
#include "../ctest.h"

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/vertex_details.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/projection.h"
#include "../../src/tu/projection_handler.h"
#include "../models/all.h"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.h"
#include "trivial_proj.h"

CT_TEST_NAME( proj_triv, "proj,triv" ) {
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=6, .nky=6, .lx=1, .ly=1);
    diverge_model_internals_tu(mom, 1.1);

    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_prop_t* LoopMom = tu_prop_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, LoopMom, 1,1,1,0 );

    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t j = 0; j < VertMom->full_vert_size; ++j) {
        VertMom->Pch[j] = random_cmplx();
        VertMom->Cch[j] = random_cmplx();
        VertMom->Dch[j] = random_cmplx();
    }

    tu_vertex_t* VertOrb = tu_vertex_copy( VertMom );

    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->my_nk)) < 1e-10);
    CT_CHECK(cabs(trace_cd(VertOrb->Cch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMom->Cch,VertMom->n_orbff,VertMom->my_nk)) < 1e-10);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->my_nk)) < 1e-10);

    complex128_t* r2 = (complex128_t*)calloc(VertOrb->full_vert_size,sizeof(complex128_t));
    complex128_t* r1 = (complex128_t*)calloc(VertOrb->full_vert_size,sizeof(complex128_t));

    P_projection(r2,VertMom,ProjMom);
    t_P_projection(r1,VertOrb,mom);
    CT_CHECK(check_equal_cd(r1,r2,POW2(VertOrb->n_orbff)*VertOrb->my_nk,1e-12));

    C_projection(r2,VertMom,ProjMom);
    t_C_projection(r1,VertOrb,mom);
    CT_CHECK(check_equal_cd(r1,r2,POW2(VertOrb->n_orbff)*VertOrb->my_nk,1e-12));

    D_projection(r2,VertMom,ProjMom);
    t_D_projection(r1,VertOrb,mom);
    CT_CHECK(check_equal_cd(r1,r2,POW2(VertOrb->n_orbff)*VertOrb->my_nk,1e-12));

    tu_vertex_free( VertOrb );
    tu_vertex_free( VertMom );
    tu_prop_free( LoopMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(r1);
    free(r2);
}

CT_TEST_NAME( proj_triv_honey, "proj,triv,honeycomb" ) {
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=6, .nky=6, .lx=1, .ly=1);
    diverge_model_internals_tu(mom, 0.7);

    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_prop_t* LoopMom = tu_prop_init(mom);
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, LoopMom, 1,1,1,0 );

    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t j = 0; j < VertMom->full_vert_size; ++j) {
        VertMom->Pch[j] = random_cmplx();
        VertMom->Cch[j] = random_cmplx();
        VertMom->Dch[j] = random_cmplx();
    }

    tu_vertex_t* VertOrb = tu_vertex_copy( VertMom );

    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->my_nk)) < 1e-10);
    CT_CHECK(cabs(trace_cd(VertOrb->Cch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMom->Cch,VertMom->n_orbff,VertMom->my_nk)) < 1e-10);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->my_nk)) < 1e-10);

    complex128_t* r2 = (complex128_t*)calloc(VertOrb->full_vert_size,sizeof(complex128_t));
    complex128_t* r1 = (complex128_t*)calloc(VertOrb->full_vert_size,sizeof(complex128_t));

    P_projection(r2,VertMom,ProjMom);
    t_P_projection(r1,VertOrb,mom);
    CT_CHECK(check_equal_cd(r1,r2,POW2(VertOrb->n_orbff)*VertOrb->my_nk,1e-12));

    C_projection(r2,VertMom,ProjMom);
    t_C_projection(r1,VertOrb,mom);
    CT_CHECK(check_equal_cd(r1,r2,POW2(VertOrb->n_orbff)*VertOrb->my_nk,1e-12));

    D_projection(r2,VertMom,ProjMom);
    t_D_projection(r1,VertOrb,mom);
    CT_CHECK(check_equal_cd(r1,r2,POW2(VertOrb->n_orbff)*VertOrb->my_nk,1e-12));

    tu_vertex_free( VertOrb );
    tu_vertex_free( VertMom );
    tu_prop_free( LoopMom );
    tu_projection_free( ProjMom );
    diverge_model_free(mom);
    free(r1);
    free(r2);
}

#endif // DIVERGE_SKIP_TESTS
