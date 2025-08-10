#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME single_channel_flow
#include "ctest.h"

#include "../src/diverge_model.h"
#include "../src/diverge_momentum_gen.h"
#include "../src/tu/diverge_interface.h"
#include "../src/tu/propagator.h"
#include "../src/diverge_flow_step.h"
#include "../src/diverge_flow_step_internal.h"
#include "../src/tu/vertex_details.h"
#include "../src/tu/projection.h"
#include "../src/tu/projection_handler.h"
#include "../src/diverge_model_internals.h"
#include "helper_functions.h"

diverge_model_t* RPA_model(int n_spin, int kx, int kxf) {
    diverge_model_t* mod1 = diverge_model_init();
    mod1->n_orb = 2;
    mod1->nk[0]=kx;mod1->nkf[0]=kxf;
    mod1->n_spin = n_spin;
    mod1->SU2 = (n_spin == 1) ? true : false;
    mod1->lattice[0][0] = mod1->lattice[1][1] = mod1->lattice[2][2] = 1.0;
    mod1->n_hop = 6*n_spin;
    mod1->hop = (rs_hopping_t*)calloc(mod1->n_hop,sizeof(rs_hopping_t));
    for(int s = 0; s < n_spin; ++ s) {
        mod1->hop[0+6*s] = (rs_hopping_t){ {1,0,0}, 0,0,s,s, 1.0 };
        mod1->hop[1+6*s] = (rs_hopping_t){ {-1,0,0}, 0,0,s,s, 1.0 };
        mod1->hop[2+6*s] = (rs_hopping_t){ {1,0,0}, 1,1,s,s, 1.0 };
        mod1->hop[3+6*s] = (rs_hopping_t){ {-1,0,0}, 1,1,s,s, 1.0 };
        mod1->hop[4+6*s] = (rs_hopping_t){ {0,0,0}, 1,0,s,s, 0.5 };
        mod1->hop[5+6*s] = (rs_hopping_t){ {0,0,0}, 0,1,s,s, 0.5 };
    }
    index_t n_orb = mod1->n_orb;
    double U = 1.0;
    double J = 0.1;
    VECTOR_INIT_CAP(rs_vertex_t, Rvert, 12*10);
    rs_vertex_t helper;
    helper.s1 = -1, helper.R[0] = 0, helper.R[1] = 0, helper.R[2] = 0;

    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o2 = 0; o2 < n_orb; ++o2) {
        helper.o1 = o1;
        helper.o2 = o2;
        if( o1 == o2 ) {
            helper.V = U;
            helper.chan = 'D';
            VECTOR_PUSH_BACK(Rvert, helper);
        }else{
            // P channel
            helper.V = J;
            helper.chan = 'P';
            VECTOR_PUSH_BACK(Rvert, helper);
            helper.V = J;
            helper.chan = 'C';
            VECTOR_PUSH_BACK(Rvert, helper);
            helper.V = (U - 2.* J);
            helper.chan = 'D';
            VECTOR_PUSH_BACK(Rvert, helper);
        }
    }

    mod1->n_vert = Rvert_sz;
    mod1->vert = Rvert;
    diverge_model_internals_common( mod1 );
    return mod1;
}

CT_TEST_NAME(RPA_P, "RPA,P_channel") {
    int ll = mpi_loglevel_get(); mpi_loglevel_set(5);
    diverge_model_t* mod1 = RPA_model(1,10,5);
    diverge_model_t* mod2 = RPA_model(1,10,5);
    diverge_model_internals_tu( mod1, 1.01 );
    diverge_model_internals_tu( mod2, 1.01 );
    diverge_flow_step_t* s1 = diverge_flow_step_init( mod1, "tu", "PCD" );
    diverge_flow_step_t* s2 = diverge_flow_step_init( mod2, "tu", "P" );
    tu_vertex_t* v1 = s1->tu_vertex;
    tu_vertex_t* v2 = s2->tu_vertex;
    tu_vertex_t* v3 = tu_stepper_internal_vertex(s1->tu_step);
    P_projection(v1->projection_helper,v1,s1->tu_proj);
    CT_CHECK(check_equal_cd(v1->projection_helper,v2->Pch,v1->full_vert_size,1e-12));

    double Lambda = 1.0;
    double dLambda = -0.1;
    diverge_flow_step_euler( s2, Lambda, dLambda );

    tu_prop_GF_zero_T( s1->tu_loop, Lambda*I128, NULL );
    P_projection(v1->projection_helper, v1, s1->tu_proj);
    tu_prop_pp_loop( s1->tu_loop, v3->pp_bubble_int, v3->bubble_helper);
    tu_vertex_product(v3->Pch, v1->projection_helper, v3->pp_bubble_int,v1->product_helper,
                v1->n_orbff*POW2(v1->n_spin),v1->my_nk,-1.0,true,v3->GPU_gemm_buffer);
    ZERO_OUT(v3->Cch,v3->full_vert_size);
    ZERO_OUT(v3->Dch,v3->full_vert_size);
    tu_vertex_add(v1,v3, dLambda);
    P_projection(v1->projection_helper, v1, s1->tu_proj);

    CT_CHECK(check_equal_cd(v1->projection_helper,v2->Pch,v1->full_vert_size,1e-12));
    CT_CHECK(check_equal_cd(v1->pp_bubble_int,v2->pp_bubble_int,v1->full_vert_size,1e-12));
    diverge_flow_step_free( s1 );
    diverge_flow_step_free( s2 );
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
    mpi_loglevel_set(ll);
}

CT_TEST_NAME(RPA_C, "RPA,C_channel") {
    int ll = mpi_loglevel_get(); mpi_loglevel_set(5);
    diverge_model_t* mod1 = RPA_model(1,10,5);
    diverge_model_t* mod2 = RPA_model(1,10,5);
    diverge_model_internals_tu( mod1, 1.01 );
    diverge_model_internals_tu( mod2, 1.01 );
    diverge_flow_step_t* s1 = diverge_flow_step_init( mod1, "tu", "PCD" );
    diverge_flow_step_t* s2 = diverge_flow_step_init( mod2, "tu", "C" );
    tu_vertex_t* v1 = s1->tu_vertex;
    tu_vertex_t* v2 = s2->tu_vertex;
    tu_vertex_t* v3 = tu_stepper_internal_vertex(s1->tu_step);
    C_projection(v1->projection_helper,v1,s1->tu_proj);
    CT_CHECK(check_equal_cd(v1->projection_helper,v2->Cch,v1->full_vert_size,1e-12));

    double Lambda = 1.0;
    double dLambda = -0.1;
    diverge_flow_step_euler( s2, Lambda, dLambda );

    tu_prop_GF_zero_T(s1->tu_loop, Lambda*I128, NULL);
    C_projection(v1->projection_helper, v1, s1->tu_proj);
    tu_prop_ph_loop( s1->tu_loop, v3->ph_bubble_int, v3->bubble_helper);
    tu_vertex_product(v3->Cch, v1->projection_helper, v3->ph_bubble_int,v1->product_helper,
                v1->n_orbff*POW2(v1->n_spin),v1->my_nk,-1.0, true, v3->GPU_gemm_buffer);
    ZERO_OUT(v3->Pch,v3->full_vert_size);
    ZERO_OUT(v3->Dch,v3->full_vert_size);
    tu_vertex_add(v1,v3, dLambda);
    C_projection(v1->projection_helper,v1,s1->tu_proj);
    CT_CHECK(check_equal_cd(v1->projection_helper,v2->Cch,v1->full_vert_size,1e-12));
    CT_CHECK(check_equal_cd(v1->ph_bubble_int,v2->ph_bubble_int,v1->full_vert_size,1e-12));
    diverge_flow_step_free( s1 );
    diverge_flow_step_free( s2 );
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
    mpi_loglevel_set(ll);
}
#endif // DIVERGE_SKIP_TESTS
