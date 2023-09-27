#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../../src/tu/diverge_interface.hpp"
#include "../model_examples/old.h"
#include "../../src/tu/global.hpp"
#include "../../src/tu/propagator.hpp"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.hpp"
#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection_wrapper.hpp"
#include "../../src/tu/projection_handler.hpp"
#include "../../src/tu/selfenergy.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"

diverge_model_t* RPA_model(int n_spin, int kx = 1, int kxf = 1) {
    diverge_model_t* mod1 = diverge_model_init();
    mod1->n_orb = 2;
    mod1->nk[0]=kx;mod1->nkf[0]=kxf;
    mod1->n_spin = n_spin;
    mod1->SU2 = (n_spin == 1) ? true : false;
    mod1->lattice[0][0] = mod1->lattice[1][1] = mod1->lattice[2][2] = 1.0;
    mod1->n_hop = 6*n_spin;
    mod1->hop = (rs_hopping_t*)calloc(mod1->n_hop,sizeof(rs_hopping_t));
    for(int s = 0; s < n_spin; ++ s) {
        mod1->hop[0+6*s] = rs_hopping_t{ {1,0,0}, 0,0,s,s, 1.0 };
        mod1->hop[1+6*s] = rs_hopping_t{ {-1,0,0}, 0,0,s,s, 1.0 };
        mod1->hop[2+6*s] = rs_hopping_t{ {1,0,0}, 1,1,s,s, 1.0 };
        mod1->hop[3+6*s] = rs_hopping_t{ {-1,0,0}, 1,1,s,s, 1.0 };
        mod1->hop[4+6*s] = rs_hopping_t{ {0,0,0}, 1,0,s,s, 0.5 };
        mod1->hop[5+6*s] = rs_hopping_t{ {0,0,0}, 0,1,s,s, 0.5 };
    }
    index_t n_orb = mod1->n_orb;
    double U = 1.0;
    double J = 0.1;
    vector<rs_vertex_t> Rvert;
    Rvert.reserve(12*10);
    rs_vertex_t helper;
    helper.s1 = -1, helper.R[0] = 0, helper.R[1] = 0, helper.R[2] = 0;

    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o2 = 0; o2 < n_orb; ++o2) {
        helper.o1 = o1;
        helper.o2 = o2;
        if( o1 == o2 ) {
            helper.V = U;
            helper.chan = 'D';
            Rvert.push_back(helper);
        }else{
            // P channel
            helper.V = J;
            helper.chan = 'P';
            Rvert.push_back(helper);
            helper.V = J;
            helper.chan = 'C';
            Rvert.push_back(helper);
            helper.V = (U - 2.* J);
            helper.chan = 'D';
            Rvert.push_back(helper);
        }
    }

    mod1->n_vert = Rvert.size();
    mod1->vert = (rs_vertex_t*)calloc(mod1->n_vert,sizeof(rs_vertex_t));
    memcpy( mod1->vert, Rvert.data(), sizeof(rs_vertex_t)*mod1->n_vert );
    diverge_model_internals_common( mod1 );
    return mod1;
}

TEST_CASE("P flow","[RPA]") {
    mpi_loglevel_set( 5 );
    diverge_model_t* mod1 = RPA_model(1,10,5);
    diverge_model_t* mod2 = RPA_model(1,10,5);
    diverge_model_internals_tu( mod1, 1.01 );
    diverge_model_internals_tu( mod2, 1.01 );
    diverge_flow_step_t* s1 = diverge_flow_step_init( mod1, "tu", "PCD" );
    diverge_flow_step_t* s2 = diverge_flow_step_init( mod2, "tu", "P" );
    Vertex* v1 = s1->tu_vertex;
    Vertex* v2 = s2->tu_vertex;
    Vertex* v3 = s1->tu_step->internal_vertex;
    P_projection(v1->projection_helper,v1,s1->tu_proj);
    CHECK(check_equal(v1->projection_helper,v2->Pch,v1->full_vert_size));

    double Lambda = 1.0;
    double dLambda = -0.1;
    diverge_flow_step_euler( s2, Lambda, dLambda );
    
    s1->tu_loop->GF_zero_T(Lambda*I128);
    P_projection(v1->projection_helper, v1, s1->tu_proj);
    s1->tu_loop->pp_loop(v3->pp_bubble_int,v3->bubble_helper);
    vertex_product(v3->Pch, v1->projection_helper,
                v3->pp_bubble_int,v1->product_helper,
                v3->GPU_gemm_buffer,v1->n_orbff*POW2(v1->n_spin),v1->my_nk,-1.0);
    mfill(v3->Cch,v3->full_vert_size,0.0);
    mfill(v3->Dch,v3->full_vert_size,0.0);
    add(*v1,*v3, dLambda);
    P_projection(v1->projection_helper, v1, s1->tu_proj);
    
    CHECK(check_equal(v1->projection_helper,v2->Pch,v1->full_vert_size));
    CHECK(check_equal(v1->pp_bubble_int,v2->pp_bubble_int,v1->full_vert_size));
    diverge_flow_step_free( s1 );
    diverge_flow_step_free( s2 );
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}
TEST_CASE("C flow","[RPA]") {
    mpi_loglevel_set( 5 );
    diverge_model_t* mod1 = RPA_model(1,10,5);
    diverge_model_t* mod2 = RPA_model(1,10,5);
    diverge_model_internals_tu( mod1, 1.01 );
    diverge_model_internals_tu( mod2, 1.01 );
    diverge_flow_step_t* s1 = diverge_flow_step_init( mod1, "tu", "PCD" );
    diverge_flow_step_t* s2 = diverge_flow_step_init( mod2, "tu", "C" );
    Vertex* v1 = s1->tu_vertex;
    Vertex* v2 = s2->tu_vertex;
    Vertex* v3 = s1->tu_step->internal_vertex;
    C_projection(v1->projection_helper,v1,s1->tu_proj);
    CHECK(check_equal(v1->projection_helper,v2->Cch,v1->full_vert_size));

    double Lambda = 1.0;
    double dLambda = -0.1;
    diverge_flow_step_euler( s2, Lambda, dLambda );
    
    s1->tu_loop->GF_zero_T(Lambda*I128);
    C_projection(v1->projection_helper, v1, s1->tu_proj);
    s1->tu_loop->ph_loop(v3->ph_bubble_int,v3->bubble_helper);
    vertex_product(v3->Cch, v1->projection_helper,
                v3->ph_bubble_int,v1->product_helper,
                v3->GPU_gemm_buffer,v1->n_orbff*POW2(v1->n_spin),v1->my_nk,-1.0);
    mfill(v3->Pch,v3->full_vert_size,0.0);
    mfill(v3->Dch,v3->full_vert_size,0.0);
    add(*v1,*v3, dLambda);    
    C_projection(v1->projection_helper,v1,s1->tu_proj);
    CHECK(check_equal(v1->projection_helper,v2->Cch,v1->full_vert_size));
    CHECK(check_equal(v1->ph_bubble_int,v2->ph_bubble_int,v1->full_vert_size));
    diverge_flow_step_free( s1 );
    diverge_flow_step_free( s2 );
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}
#endif // DIVERGE_SKIP_TESTS
