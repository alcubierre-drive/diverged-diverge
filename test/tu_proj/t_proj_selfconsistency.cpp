#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../../src/tu/diverge_interface.hpp"
#include "../model_examples/old.h"
#include "../../src/tu/global.hpp"
#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection_wrapper.hpp"
#include "../../src/tu/projection_handler.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include <stdio.h>

#ifndef USE_MPI
TEST_CASE("momentum vs realspace projections ", "[proj][proj_orbVSreal]")
{
    diverge_model_t* mom = gen_square_lat_hub(6, 6);
    diverge_model_t* orb = gen_square_lat_hub(1, 1, 6, 6);
    diverge_model_internals_tu(orb, 1.2);
    diverge_model_internals_tu(mom, 1.2);

    Projection ProjMom (mom);
    Projection ProjOrb (orb);
    
    tu_loop_t LoopMom (mom);
    tu_loop_t LoopOrb (orb);

    Vertex VertMom(mom, &ProjMom, &LoopMom);
    Vertex VertOrb(orb, &ProjOrb, &LoopOrb);

    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    tu_data_t* tu_o = (tu_data_t*)(orb -> internals-> tu_data);
    tu_data_t* tu_m = (tu_data_t*)(mom -> internals-> tu_data);
    double* Rveco = (double*)calloc(orb->n_tu_ff*3,sizeof(double));
    double* Rvecm = (double*)calloc(mom->n_tu_ff*3,sizeof(double));
    fill_Rvec(orb, Rveco);
    fill_Rvec(mom, Rvecm);

    index_t n_orbffo = VertOrb.n_orbff;
    index_t n_orbffm = VertMom.n_orbff;
    tu_formfactor_t* tu_ffo = orb->tu_ff;
    tu_formfactor_t* tu_ffm = mom->tu_ff;
    index_t* bso = tu_o->bond_sizes;
    index_t* boo = tu_o->bond_offsets;
    index_t* bsm = tu_m->bond_sizes;
    index_t* bom = tu_m->bond_offsets;
    for(index_t j = 0; j < orb->n_orb; ++j)
    for(index_t bj = 0; bj < bso[j]; ++bj)
    for(index_t k = 0; k < tu_o->nk; ++k) {
        VertOrb.Pch[j + n_orbffo * tu_ffo[boo[j]+bj].oto+POW2(n_orbffo)*k]
                    += ff(orb->internals->kmesh+3*k,Rveco + (boo[j]+bj)*3);
    }
    for(index_t j = 0; j < mom->n_orb; ++j)
    for(index_t bj = 0; bj < bsm[j]; ++bj)
    for(index_t k = 0; k < tu_m->nk; ++k) {
        VertMom.Pch[j + n_orbffm * tu_ffm[bom[j]+bj].oto+POW2(n_orbffm)*k]
                    += ff(mom->internals->kmesh+3*k,Rvecm + (boo[j]+bj)*3);
    }
    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    D_projection(VertMom.Dch,&VertMom,&ProjMom);
    D_projection(VertOrb.Dch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    C_projection(VertMom.Cch,&VertMom,&ProjMom);
    C_projection(VertOrb.Cch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Cch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Cch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    P_projection(VertMom.Pch,&VertMom,&ProjMom);
    P_projection(VertOrb.Pch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    D_projection(VertMom.Dch,&VertMom,&ProjMom);
    D_projection(VertOrb.Dch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);

    diverge_model_free(orb);
    diverge_model_free(mom);
    free(Rveco);
    free(Rvecm);
}

TEST_CASE("momentum vs mixed projections ", "[proj][proj_mixVSreal]")
{
    diverge_model_t* mom = gen_square_lat_hub(12, 12);
    diverge_model_t* orb = gen_square_lat_hub(4, 6, 3, 2);
    diverge_model_internals_tu(orb, 1.2);
    diverge_model_internals_tu(mom, 1.2);

    Projection ProjMom (mom);
    Projection ProjOrb (orb);

    tu_loop_t LoopMom (mom);
    tu_loop_t LoopOrb (orb);

    Vertex VertMom(mom, &ProjMom, &LoopMom);
    Vertex VertOrb(orb, &ProjOrb, &LoopOrb);
    double* Rveco = (double*)calloc(orb->n_tu_ff*3,sizeof(double));
    double* Rvecm = (double*)calloc(mom->n_tu_ff*3,sizeof(double));
    fill_Rvec(orb, Rveco);
    fill_Rvec(mom, Rvecm);

    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    tu_data_t* tu_o = (tu_data_t*)(orb -> internals-> tu_data);
    tu_data_t* tu_m = (tu_data_t*)(mom -> internals-> tu_data);

    index_t n_orbffo = VertOrb.n_orbff;
    index_t n_orbffm = VertMom.n_orbff;
    tu_formfactor_t* tu_ffo = orb->tu_ff;
    tu_formfactor_t* tu_ffm = mom->tu_ff;
    index_t* bso = tu_o->bond_sizes;
    index_t* boo = tu_o->bond_offsets;
    index_t* bsm = tu_m->bond_sizes;
    index_t* bom = tu_m->bond_offsets;

    for(index_t j = 0; j < orb->n_orb; ++j)
    for(index_t bj = 0; bj < bso[j]; ++bj)
    for(index_t k = 0; k < tu_o->nk; ++k) {
        VertOrb.Pch[j + n_orbffo * tu_ffo[boo[j]+bj].oto+POW2(n_orbffo)*k]
                    += ff(orb->internals->kmesh+3*k,Rveco + (boo[j]+bj)*3);
    }
    for(index_t j = 0; j < mom->n_orb; ++j)
    for(index_t bj = 0; bj < bsm[j]; ++bj)
    for(index_t k = 0; k < tu_m->nk; ++k) {
        VertMom.Pch[j + n_orbffm * tu_ffm[bom[j]+bj].oto+POW2(n_orbffm)*k]
                    += ff(mom->internals->kmesh+3*k,Rvecm + (boo[j]+bj)*3);
    }

    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    D_projection(VertMom.Dch,&VertMom,&ProjMom);
    D_projection(VertOrb.Dch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    C_projection(VertMom.Cch,&VertMom,&ProjMom);
    C_projection(VertOrb.Cch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Cch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Cch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    P_projection(VertMom.Pch,&VertMom,&ProjMom);
    P_projection(VertOrb.Pch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    D_projection(VertMom.Dch,&VertMom,&ProjMom);
    D_projection(VertOrb.Dch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);

    diverge_model_free(orb);
    diverge_model_free(mom);
    free(Rveco);
    free(Rvecm);
}

TEST_CASE("momentum vs mixed projections graphene", "[proj][proj_mixVSrealgr]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6, 6);
    diverge_model_t* orb = gen_honeycomb_lat_hub(1, 1, 6, 6);
    diverge_model_internals_tu(orb, 1.2);
    diverge_model_internals_tu(mom, 1.2);

    Projection ProjMom (mom);
    Projection ProjOrb (orb);

    tu_loop_t LoopMom (mom);
    tu_loop_t LoopOrb (orb);

    Vertex VertMom(mom, &ProjMom, &LoopMom);
    Vertex VertOrb(orb, &ProjOrb, &LoopOrb);
    double* Rveco = (double*)calloc(orb->n_tu_ff*3,sizeof(double));
    double* Rvecm = (double*)calloc(mom->n_tu_ff*3,sizeof(double));
    fill_Rvec(orb, Rveco);
    fill_Rvec(mom, Rvecm);

    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    tu_data_t* tu_o = (tu_data_t*)(orb -> internals-> tu_data);
    tu_data_t* tu_m = (tu_data_t*)(mom -> internals-> tu_data);

    index_t n_orbffo = VertOrb.n_orbff;
    index_t n_orbffm = VertMom.n_orbff;
    tu_formfactor_t* tu_ffo = orb->tu_ff;
    tu_formfactor_t* tu_ffm = mom->tu_ff;
    index_t* bso = tu_o->bond_sizes;
    index_t* boo = tu_o->bond_offsets;
    index_t* bsm = tu_m->bond_sizes;
    index_t* bom = tu_m->bond_offsets;

    for(index_t j = 0; j < orb->n_orb; ++j)
    for(index_t bj = 0; bj < bso[j]; ++bj)
    for(index_t k = 0; k < tu_o->nk; ++k) {
        VertOrb.Pch[j + n_orbffo * tu_ffo[boo[j]+bj].oto+POW2(n_orbffo)*k]
                    += ff(orb->internals->kmesh+3*k,Rveco + (boo[j]+bj)*3);
    }
    for(index_t j = 0; j < mom->n_orb; ++j)
    for(index_t bj = 0; bj < bsm[j]; ++bj)
    for(index_t k = 0; k < tu_m->nk; ++k) {
        VertMom.Pch[j + n_orbffm * tu_ffm[bom[j]+bj].oto+POW2(n_orbffm)*k]
                    += ff(mom->internals->kmesh+3*k,Rvecm + (boo[j]+bj)*3);
    }

    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    D_projection(VertMom.Dch,&VertMom,&ProjMom);
    D_projection(VertOrb.Dch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    C_projection(VertMom.Cch,&VertMom,&ProjMom);
    C_projection(VertOrb.Cch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Cch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Cch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    P_projection(VertMom.Pch,&VertMom,&ProjMom);
    P_projection(VertOrb.Pch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.nk)) < 1e-10);
    D_projection(VertMom.Dch,&VertMom,&ProjMom);
    D_projection(VertOrb.Dch,&VertOrb,&ProjOrb);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.nk)) < 1e-10);

    diverge_model_free(orb);
    diverge_model_free(mom);
    free(Rveco);
    free(Rvecm);
}
#endif

#endif // DIVERGE_SKIP_TESTS
