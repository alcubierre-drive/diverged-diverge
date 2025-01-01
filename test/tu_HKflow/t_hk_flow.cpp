#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../../src/tu/diverge_interface.h"
#include "../model_examples/old.h"
#include "../../src/tu/propagator.hpp"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.hpp"
#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection_wrapper.hpp"
#include "../../src/tu/projection_handler.hpp"
#include "../../src/grid/vertex_memory.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include<iostream>

diverge_model_t* test_model(int n_spin, int kx = 1, int kxf = 1) {
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
            helper.V = U - 2.* J;
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

template<char chan>
double err_X(complex128_t* pSU, complex128_t* pnSU, int n_orb, int n_orbff) {
    double err = 0.0;
    for (index_t op3=0; op3<n_orb; ++op3){
        for (index_t op1=0; op1<n_orb; ++op1){
            index_t o1, o3;
            if constexpr(chan == 'D') {
                o1 = op1 + n_orbff * (3);
                o3 = op3 + n_orbff * (0);
            } else if constexpr(chan == 'C') {
                o1 = op1 + n_orbff * (1);
                o3 = op3 + n_orbff * (1);
            } else if constexpr(chan == 'P') {
                o1 = op1 + n_orbff * (1);
                o3 = op3 + n_orbff * (1);
            }
            double trial = std::abs(pSU[op1+n_orbff*op3] - pnSU[o1+n_orbff*4*o3]);
            if(trial > err) err = trial;
        }
    }
    double mpi_err = 0;
    diverge_mpi_allreduce_double_max( &err, &mpi_err, 1 );
    return mpi_err;
}

TEST_CASE("two orbital model in 1d","[hk]") {
    CHECK( diverge_mpi_comm_size() == 1 );
    if (diverge_mpi_comm_size() > 1) {
        mpi_err_printf( "cannot test two orbital model in 1d [hk] with more than one MPI rank\n" );
        return;
    }
    MPILoglevelWrapper LL(5);
    diverge_model_t* mod1 = test_model(1);
    diverge_model_t* mod2 = test_model(2);
    diverge_model_internals_tu( mod1, 4.01 );
    diverge_model_internals_tu( mod2, 4.01 );
    diverge_flow_step_t* s1 = diverge_flow_step_init( mod1, "tu", "PCD" );
    diverge_flow_step_t* s2 = diverge_flow_step_init( mod2, "tu", "PCD" );
    Vertex* v1 = s1->tu_vertex;
    Vertex* v2 = s2->tu_vertex;
    CHECK(err_X<'D'>(v1->Dch, v2->Dch, 2, v2->n_orbff) < 1e-9);
    CHECK(err_X<'C'>(v1->Cch, v2->Cch, 2, v2->n_orbff) < 1e-9);
    CHECK(err_X<'P'>(v1->Pch, v2->Pch, 2, v2->n_orbff) < 1e-9);


    index_t fsize = POW4(mod2->n_orb*mod2->n_spin);
    complex128_t* vertex = (complex128_t*)calloc(fsize,sizeof(complex128_t));
    s2->tu_vertex->project_fullV(mod2,vertex);
    index_t norb = mod2->n_orb*mod2->n_spin;
    for(index_t s4 = 0; s4<mod2->n_spin;++s4)
    for(index_t s3 = 0; s3<mod2->n_spin;++s3)
    for(index_t s2 = 0; s2<mod2->n_spin;++s2)
    for(index_t s1 = 0; s1<mod2->n_spin;++s1)
    for(index_t o4 = 0; o4<mod2->n_orb;++o4)
    for(index_t o3 = 0; o3<mod2->n_orb;++o3)
    for(index_t o2 = 0; o2<mod2->n_orb;++o2)
    for(index_t o1 = 0; o1<mod2->n_orb;++o1)
    {
        const index_t so1 = o1+mod2->n_orb*s1;
        const index_t so2 = o2+mod2->n_orb*s2;
        const index_t so3 = o3+mod2->n_orb*s3;
        const index_t so4 = o4+mod2->n_orb*s4;
        if(s1 == s4 && s2 == s3){
            if(s1!=s3){
                if(o1==o4 && o3==o2){
                    if(o1==o2) {
                        CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+1.0) < 1e-10);
                    }else
                        CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+1.0-2.*0.1) < 1e-10);
                }else if(o1==o3 && o4==o2 && o2!=o1)
                    CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+0.1) < 1e-10);
                else if(o1==o2 && o3==o4&& o1!=o3)
                    CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+0.1) < 1e-10);
            }else{
                if(o1==o4 && o2==o3 && o1!=o3)
                    CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+1.0*1.0-3.0*0.1) < 1e-10);
            }
        }else if(s1 == s3 && s4 == s2){
            if(s1!=s4){
                if(o1==o3 && o2==o4){
                    if(o1==o2) {
                        CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-1.0) < 1e-10);
                    }else
                         CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-1.0+2.*0.1) < 1e-10);
                }else if(o1==o4 && o3==o2 && o3!=o1)
                     CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-0.1) < 1e-10);
                else if(o1==o2 && o3==o4 && o1!=o3)
                     CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-0.1) < 1e-10);
            }else{
                if(o1==o3 && o2==o4 && o1!=o2)
                     CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-1.0*1.0+3.0*0.1) < 1e-10);
            }
        }
    }


    for(int s1 = 0; s1 < norb; ++ s1)
    for(int s2 = 0; s2 < norb; ++ s2)
    for(int s3 = 0; s3 < norb; ++ s3)
    for(int s4 = 0; s4 < norb; ++ s4){
        index_t i1234 = IDX4( s4, s3, s2, s1,norb,norb,norb),
                i2134 = IDX4( s4, s3, s1, s2,norb,norb,norb),
                i1243 = IDX4( s3, s4, s2, s1,norb,norb,norb),
                i2143 = IDX4( s3, s4, s1, s2,norb,norb,norb);
        if(std::abs(vertex[i1234]) > 1e-10 ){
            CHECK(std::abs(vertex[i1234]+vertex[i2134]) < 1e-10);
            CHECK(std::abs(vertex[i1234]+vertex[i1243]) < 1e-10);
            CHECK(std::abs(vertex[i1234]-vertex[i2143]) < 1e-10);
        }
    }

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<5; ++i) {
        diverge_flow_step_euler( s1, Lambda, dLambda );
        diverge_flow_step_euler( s2, Lambda, dLambda );

        Lambda += dLambda;
    }
    CHECK(err_X<'D'>(v1->Dch, v2->Dch, 2, v2->n_orbff) < 1e-9);
    CHECK(err_X<'C'>(v1->Cch, v2->Cch, 2, v2->n_orbff) < 1e-9);
    CHECK(err_X<'P'>(v1->Pch, v2->Pch, 2, v2->n_orbff) < 1e-9);
    diverge_flow_step_free( s1 );
    diverge_flow_step_free( s2 );
    free(vertex);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}


TEST_CASE("two orbital model in 1d grid","[hk-grid]") {
    MPILoglevelWrapper LL(5);
    diverge_model_t* mod2 = test_model(2);
    diverge_model_internals_grid( mod2 );
    diverge_flow_step_t* s2 = diverge_flow_step_init( mod2, "grid", "PCD" );

    index_t fsize = POW4(mod2->n_orb*mod2->n_spin);
    complex128_t* vertex = (complex128_t*)calloc(fsize,sizeof(complex128_t));
    ((grid::VertexMemory*)s2->intls->grid_vertex)->get_vertex( vertex, 'V' );
    index_t norb = mod2->n_orb*mod2->n_spin;
    for(index_t s4 = 0; s4<mod2->n_spin;++s4)
    for(index_t s3 = 0; s3<mod2->n_spin;++s3)
    for(index_t s2 = 0; s2<mod2->n_spin;++s2)
    for(index_t s1 = 0; s1<mod2->n_spin;++s1)
    for(index_t o4 = 0; o4<mod2->n_orb;++o4)
    for(index_t o3 = 0; o3<mod2->n_orb;++o3)
    for(index_t o2 = 0; o2<mod2->n_orb;++o2)
    for(index_t o1 = 0; o1<mod2->n_orb;++o1)
    {
        const index_t so1 = o1+mod2->n_orb*s1;
        const index_t so2 = o2+mod2->n_orb*s2;
        const index_t so3 = o3+mod2->n_orb*s3;
        const index_t so4 = o4+mod2->n_orb*s4;
        if(s1 == s4 && s2 == s3){
            if(s1!=s3){
                if(o1==o4 && o3==o2){
                    if(o1==o2) {
                        CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+1.0) < 1e-10);
                    }else
                        CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+1.0-2.*0.1) < 1e-10);
                }else if(o1==o3 && o4==o2 && o2!=o1)
                    CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+0.1) < 1e-10);
                else if(o1==o2 && o3==o4&& o1!=o3)
                    CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+0.1) < 1e-10);
            }else{
                if(o1==o4 && o2==o3 && o1!=o3)
                    CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]+1.0*1.0-3.0*0.1) < 1e-10);
            }
        }else if(s1 == s3 && s4 == s2){
            if(s1!=s4){
                if(o1==o3 && o2==o4){
                    if(o1==o2) {
                        CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-1.0) < 1e-10);
                    }else
                         CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-1.0+2.*0.1) < 1e-10);
                }else if(o1==o4 && o3==o2 && o3!=o1)
                     CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-0.1) < 1e-10);
                else if(o1==o2 && o3==o4 && o1!=o3)
                     CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-0.1) < 1e-10);
            }else{
                if(o1==o3 && o2==o4 && o1!=o2)
                     CHECK(std::abs(vertex[so4+norb*(so3+norb*(so2+norb*so1))]-1.0*1.0+3.0*0.1) < 1e-10);
            }
        }
    }

    diverge_flow_step_free( s2 );
    free(vertex);
    diverge_model_free( mod2 );
}

TEST_CASE("two orbital model in 1d mom","[mhk]") {
    MPILoglevelWrapper LL(5);
    diverge_model_t* mod1 = test_model(1,10,5);
    diverge_model_t* mod2 = test_model(2,10,5);
    diverge_model_internals_tu( mod1, 1.01 );
    diverge_model_internals_tu( mod2, 1.01 );
    diverge_flow_step_t* s1 = diverge_flow_step_init( mod1, "tu", "SPCD" );
    diverge_flow_step_t* s2 = diverge_flow_step_init( mod2, "tu", "SPCD" );
    Vertex* v1 = s1->tu_vertex;
    Vertex* v2 = s2->tu_vertex;
    CHECK(err_X<'D'>(v1->Dch, v2->Dch, 2, v2->n_orbff) < 1e-9);
    CHECK(err_X<'C'>(v1->Cch, v2->Cch, 2, v2->n_orbff) < 1e-9);
    CHECK(err_X<'P'>(v1->Pch, v2->Pch, 2, v2->n_orbff) < 1e-9);

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<5; ++i) {
        diverge_flow_step_euler( s1, Lambda, dLambda );
        diverge_flow_step_euler( s2, Lambda, dLambda );
        Lambda += dLambda;
    }
    CHECK(err_X<'D'>(v1->Dch, v2->Dch, 2, v2->n_orbff) < 1e-9);
    CHECK(err_X<'C'>(v1->Cch, v2->Cch, 2, v2->n_orbff) < 1e-9);
    CHECK(err_X<'P'>(v1->Pch, v2->Pch, 2, v2->n_orbff) < 1e-9);
    diverge_flow_step_free( s1 );
    diverge_flow_step_free( s2 );
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}

#endif // DIVERGE_SKIP_TESTS
