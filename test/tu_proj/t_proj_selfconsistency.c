#define CT_SUITE_NAME tu_proj_spaces
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

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

static inline int compilation_options( void ) {
    // MPI should be off
    return diverge_mpi_comm_size() != 1;
}

CT_TEST_NAME( proj_orb_real, "proj,orb:real" ) {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=6, .nky=6);
    diverge_model_t* orb = KWARGS(gen_square_lat_hub, .nkx=1, .nky=1, .lx=6, .ly=6);
    diverge_model_internals_tu(orb, 1.2);
    diverge_model_internals_tu(mom, 1.2);

    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_projection_t* ProjOrb = tu_projection_init(orb);

    tu_prop_t* LoopMom = tu_prop_init(mom);
    tu_prop_t* LoopOrb = tu_prop_init(orb);

    tu_vertex_t* VertMom = tu_vertex_init(mom, ProjMom, LoopMom, 1,1,1,0);
    tu_vertex_t* VertOrb = tu_vertex_init(orb, ProjOrb, LoopOrb, 1,1,1,0);

    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    tu_data_t* tu_o = (tu_data_t*)(orb -> internals-> tu_data);
    tu_data_t* tu_m = (tu_data_t*)(mom -> internals-> tu_data);
    double* Rveco = (double*)calloc(orb->n_tu_ff*3,sizeof(double));
    double* Rvecm = (double*)calloc(mom->n_tu_ff*3,sizeof(double));
    fill_Rvec(orb, Rveco);
    fill_Rvec(mom, Rvecm);

    index_t n_orbffo = VertOrb->n_orbff;
    index_t n_orbffm = VertMom->n_orbff;
    tu_formfactor_t* tu_ffo = orb->tu_ff;
    tu_formfactor_t* tu_ffm = mom->tu_ff;
    index_t* bso = tu_o->bond_sizes;
    index_t* boo = tu_o->bond_offsets;
    index_t* bsm = tu_m->bond_sizes;
    index_t* bom = tu_m->bond_offsets;
    for(index_t j = 0; j < orb->n_orb; ++j)
    for(index_t bj = 0; bj < bso[j]; ++bj)
    for(index_t k = 0; k < tu_o->nk; ++k) {
        VertOrb->Pch[j + n_orbffo * tu_ffo[boo[j]+bj].oto+POW2(n_orbffo)*k]
                    += ff(orb->internals->kmesh+3*k,Rveco + (boo[j]+bj)*3);
    }
    for(index_t j = 0; j < mom->n_orb; ++j)
    for(index_t bj = 0; bj < bsm[j]; ++bj)
    for(index_t k = 0; k < tu_m->nk; ++k) {
        VertMom->Pch[j + n_orbffm * tu_ffm[bom[j]+bj].oto+POW2(n_orbffm)*k]
                    += ff(mom->internals->kmesh+3*k,Rvecm + (boo[j]+bj)*3);
    }
    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    D_projection(VertMom->Dch,VertMom,ProjMom);
    D_projection(VertOrb->Dch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    C_projection(VertMom->Cch,VertMom,ProjMom);
    C_projection(VertOrb->Cch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Cch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Cch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    P_projection(VertMom->Pch,VertMom,ProjMom);
    P_projection(VertOrb->Pch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    D_projection(VertMom->Dch,VertMom,ProjMom);
    D_projection(VertOrb->Dch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);

    tu_vertex_free(VertMom);
    tu_vertex_free(VertOrb);
    tu_prop_free(LoopMom);
    tu_prop_free(LoopOrb);
    tu_projection_free( ProjMom );
    tu_projection_free( ProjOrb );
    diverge_model_free(orb);
    diverge_model_free(mom);
    free(Rveco);
    free(Rvecm);
}

CT_TEST_NAME( proj_mix_real, "proj,mix:real" ) {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=12, .nky=12);
    diverge_model_t* orb = KWARGS(gen_square_lat_hub, .nkx=4,  .nky=6, .lx=3, .ly=2);
    diverge_model_internals_tu(orb, 1.2);
    diverge_model_internals_tu(mom, 1.2);

    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_projection_t* ProjOrb = tu_projection_init(orb);

    tu_prop_t* LoopMom = tu_prop_init(mom);
    tu_prop_t* LoopOrb = tu_prop_init(orb);

    tu_vertex_t* VertMom = tu_vertex_init(mom, ProjMom, LoopMom, 1,1,1,0);
    tu_vertex_t* VertOrb = tu_vertex_init(orb, ProjOrb, LoopOrb, 1,1,1,0);
    double* Rveco = (double*)calloc(orb->n_tu_ff*3,sizeof(double));
    double* Rvecm = (double*)calloc(mom->n_tu_ff*3,sizeof(double));
    fill_Rvec(orb, Rveco);
    fill_Rvec(mom, Rvecm);

    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    tu_data_t* tu_o = (tu_data_t*)(orb -> internals-> tu_data);
    tu_data_t* tu_m = (tu_data_t*)(mom -> internals-> tu_data);

    index_t n_orbffo = VertOrb->n_orbff;
    index_t n_orbffm = VertMom->n_orbff;
    tu_formfactor_t* tu_ffo = orb->tu_ff;
    tu_formfactor_t* tu_ffm = mom->tu_ff;
    index_t* bso = tu_o->bond_sizes;
    index_t* boo = tu_o->bond_offsets;
    index_t* bsm = tu_m->bond_sizes;
    index_t* bom = tu_m->bond_offsets;

    for(index_t j = 0; j < orb->n_orb; ++j)
    for(index_t bj = 0; bj < bso[j]; ++bj)
    for(index_t k = 0; k < tu_o->nk; ++k) {
        VertOrb->Pch[j + n_orbffo * tu_ffo[boo[j]+bj].oto+POW2(n_orbffo)*k]
                    += ff(orb->internals->kmesh+3*k,Rveco + (boo[j]+bj)*3);
    }
    for(index_t j = 0; j < mom->n_orb; ++j)
    for(index_t bj = 0; bj < bsm[j]; ++bj)
    for(index_t k = 0; k < tu_m->nk; ++k) {
        VertMom->Pch[j + n_orbffm * tu_ffm[bom[j]+bj].oto+POW2(n_orbffm)*k]
                    += ff(mom->internals->kmesh+3*k,Rvecm + (boo[j]+bj)*3);
    }

    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    D_projection(VertMom->Dch,VertMom,ProjMom);
    D_projection(VertOrb->Dch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    C_projection(VertMom->Cch,VertMom,ProjMom);
    C_projection(VertOrb->Cch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Cch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Cch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    P_projection(VertMom->Pch,VertMom,ProjMom);
    P_projection(VertOrb->Pch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    D_projection(VertMom->Dch,VertMom,ProjMom);
    D_projection(VertOrb->Dch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);

    tu_vertex_free(VertMom);
    tu_vertex_free(VertOrb);
    tu_prop_free(LoopMom);
    tu_prop_free(LoopOrb);
    tu_projection_free(ProjMom);
    tu_projection_free(ProjOrb);
    diverge_model_free(orb);
    diverge_model_free(mom);
    free(Rveco);
    free(Rvecm);
}

CT_TEST_NAME( proj_mix_real_gr, "proj,mix:real,graphene" ) {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_honeycomb_lat_hub, .nkx=6, .nky=6);
    diverge_model_t* orb = KWARGS(gen_honeycomb_lat_hub, .nkx=1, .nky=1, .lx=6, .ly=6);
    diverge_model_internals_tu(orb, 1.2);
    diverge_model_internals_tu(mom, 1.2);

    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_projection_t* ProjOrb = tu_projection_init(orb);

    tu_prop_t* LoopMom = tu_prop_init(mom);
    tu_prop_t* LoopOrb = tu_prop_init(orb);

    tu_vertex_t* VertMom = tu_vertex_init(mom, ProjMom, LoopMom, 1,1,1,0);
    tu_vertex_t* VertOrb = tu_vertex_init(orb, ProjOrb, LoopOrb, 1,1,1,0);
    double* Rveco = (double*)calloc(orb->n_tu_ff*3,sizeof(double));
    double* Rvecm = (double*)calloc(mom->n_tu_ff*3,sizeof(double));
    fill_Rvec(orb, Rveco);
    fill_Rvec(mom, Rvecm);

    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    tu_data_t* tu_o = (tu_data_t*)(orb -> internals-> tu_data);
    tu_data_t* tu_m = (tu_data_t*)(mom -> internals-> tu_data);

    index_t n_orbffo = VertOrb->n_orbff;
    index_t n_orbffm = VertMom->n_orbff;
    tu_formfactor_t* tu_ffo = orb->tu_ff;
    tu_formfactor_t* tu_ffm = mom->tu_ff;
    index_t* bso = tu_o->bond_sizes;
    index_t* boo = tu_o->bond_offsets;
    index_t* bsm = tu_m->bond_sizes;
    index_t* bom = tu_m->bond_offsets;

    for(index_t j = 0; j < orb->n_orb; ++j)
    for(index_t bj = 0; bj < bso[j]; ++bj)
    for(index_t k = 0; k < tu_o->nk; ++k) {
        VertOrb->Pch[j + n_orbffo * tu_ffo[boo[j]+bj].oto+POW2(n_orbffo)*k]
                    += ff(orb->internals->kmesh+3*k,Rveco + (boo[j]+bj)*3);
    }
    for(index_t j = 0; j < mom->n_orb; ++j)
    for(index_t bj = 0; bj < bsm[j]; ++bj)
    for(index_t k = 0; k < tu_m->nk; ++k) {
        VertMom->Pch[j + n_orbffm * tu_ffm[bom[j]+bj].oto+POW2(n_orbffm)*k]
                    += ff(mom->internals->kmesh+3*k,Rvecm + (boo[j]+bj)*3);
    }

    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    D_projection(VertMom->Dch,VertMom,ProjMom);
    D_projection(VertOrb->Dch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    C_projection(VertMom->Cch,VertMom,ProjMom);
    C_projection(VertOrb->Cch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Cch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Cch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    P_projection(VertMom->Pch,VertMom,ProjMom);
    P_projection(VertOrb->Pch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->nk)) < 1e-10);
    D_projection(VertMom->Dch,VertMom,ProjMom);
    D_projection(VertOrb->Dch,VertOrb,ProjOrb);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->nk)
                  -trace_cd(VertMom->Dch,VertMom->n_orbff,VertMom->nk)) < 1e-10);

    tu_vertex_free(VertMom);
    tu_vertex_free(VertOrb);
    tu_prop_free(LoopMom);
    tu_prop_free(LoopOrb);
    tu_projection_free( ProjMom );
    tu_projection_free( ProjOrb );
    diverge_model_free(orb);
    diverge_model_free(mom);
    free(Rveco);
    free(Rvecm);
}

#endif // DIVERGE_SKIP_TESTS
