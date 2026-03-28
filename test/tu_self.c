#define CT_SUITE_NAME tu_self
#include "ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../src/diverge_model.h"
#include "../src/diverge_momentum_gen.h"
#include "../src/tu/diverge_interface.h"
#include "../src/tu/propagator.h"
#include "../src/tu/propagator_details.h"
#include "../src/diverge_flow_step.h"
#include "../src/diverge_flow_step_internal.h"
#include "../src/tu/vertex_details.h"
#include "../src/tu/projection.h"
#include "../src/tu/projection_handler.h"
#include "../src/tu/selfenergy.h"
#include "../src/diverge_model_internals.h"
#include "../src/misc/compilation_status.h"
#include "helper_functions.h"
#include "models/all.h"

static inline int compilation_options( void ) {
    // no gf_floats, no MPI
    if ((sizeof(complex128_t) != sizeof(gf_complex_t)) || diverge_compilation_status_mpi())
        return 1;
    else
        return 0;
}

void init_random_PCD(diverge_model_t* model, tu_vertex_t* vertex, tu_projection_t * proj, double* value) {
    tu_data_t* tu_data = (tu_data_t*)(model -> internals-> tu_data);
    tu_symmstruct_t* symm = tu_data->symm;
    double* Rveco = (double*)calloc(model->n_tu_ff*3,sizeof(double));
    fill_Rvec(model, Rveco);

    index_t n_orbffo = vertex->n_orbff;
    index_t n_spin2 = POW2(vertex->n_spin);
    tu_formfactor_t* tu_ffo = model->tu_ff;
    index_t* bso = tu_data->bond_sizes;
    index_t* boo = tu_data->bond_offsets;
    for(index_t j = 0; j < model->n_orb; ++j)
    for(index_t bj = 0; bj < bso[j]; ++bj)
    for(index_t sl = 0; sl < n_spin2; ++sl)
    for(index_t sr = 0; sr < n_spin2; ++sr)
    for(index_t k = 0; k < tu_data->my_nk; ++k) {
        vertex->Pch[j + n_orbffo * (sl + n_spin2*(tu_ffo[boo[j]+bj].oto+n_orbffo*(sr +n_spin2*k)))]
                    += value[0]*ff(model->internals->kmesh+3*(symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off]),
                                            Rveco + (boo[j]+bj)*3);
        vertex->Cch[j + n_orbffo * (sl + n_spin2*(tu_ffo[boo[j]+bj].oto+n_orbffo*(sr +n_spin2*k)))]
                    += value[1]*conj(ff(model->internals->kmesh+3*(symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off]),
                                            Rveco + (boo[j]+bj)*3));
        vertex->Dch[j + n_orbffo * (sl + n_spin2*(tu_ffo[boo[j]+bj].oto+n_orbffo*(sr +n_spin2*k)))]
                    += value[2]*conj(ff(model->internals->kmesh+3*(symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off]),
                                            Rveco + (boo[j]+bj)*3));
    }
    D_projection(vertex->Dch,vertex,proj);
    C_projection(vertex->Cch,vertex,proj);
    P_projection(vertex->Pch,vertex,proj);
    free(Rveco);
}

CT_TEST_NAME(real_vs_k_SU2,"self,real:mom,SU2") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=12, .nky=12);
    diverge_model_t* orb = KWARGS(gen_square_lat_hub, .nkx=1, .nky=1, .lx=12, .ly=12);
    diverge_model_t* mix = KWARGS(gen_square_lat_hub, .nkx=4, .nky=4, .lx=3,  .ly=3);
    diverge_model_internals_tu(mom, 1.1);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mix, 1.1);

    tu_prop_t* PropMom = tu_prop_init(mom);
    tu_prop_t* PropOrb = tu_prop_init(orb);
    tu_prop_t* PropMix = tu_prop_init(mix);

    tu_projection_t* ProjOrb = tu_projection_init(orb);
    tu_projection_t* ProjMom = tu_projection_init(mom);
    tu_projection_t* ProjMix = tu_projection_init(mix);

    tu_vertex_t* VertOrb = tu_vertex_init( orb, ProjOrb, PropOrb, true,true,true,true );
    tu_vertex_t* VertMom = tu_vertex_init( mom, ProjMom, PropMom, true,true,true,true );
    tu_vertex_t* VertMix = tu_vertex_init( mix, ProjMix, PropMix, true,true,true,true );
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,VertOrb,ProjOrb,value);
    init_random_PCD(mom,VertMom,ProjMom,value);
    init_random_PCD(mix,VertMix,ProjMix,value);

    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMom->Pch,VertMom->n_orbff,VertMom->my_nk)) < 1e-8);
    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMix->Pch,VertMix->n_orbff,VertMix->my_nk)) < 1e-8);

    tu_selfenergy_t* selfOrb = tu_selfenergy_init(VertOrb);
    tu_selfenergy_t* selfMom = tu_selfenergy_init(VertMom);
    tu_selfenergy_t* selfMix = tu_selfenergy_init(VertMix);

    const double Lambda = 1e-1;
    tu_prop_GF_zero_T(PropMom, Lambda, NULL);
    tu_prop_GF_zero_T(PropOrb, Lambda, NULL);
    tu_prop_GF_zero_T(PropMix, Lambda, NULL);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropMix, gfmix);

    tu_selfenergy_flow(selfOrb, VertOrb->self_en, VertOrb->Pch, VertOrb->Cch,
                    VertOrb->Dch, PropOrb->gf_buf, 1.0, Lambda);
    tu_selfenergy_flow(selfMom, VertMom->self_en, VertMom->Pch, VertMom->Cch,
                    VertMom->Dch, PropMom->gf_buf, 1.0, Lambda);
    tu_selfenergy_flow(selfMix, VertMix->self_en, VertMix->Pch, VertMix->Cch,
                    VertMix->Dch, gfmix, 1.0, Lambda);
    CT_CHECK(cabs(trace_cd(PropMom->gf_buf,PropMom->n_orb,PropMom->nk)
                  -trace_cd(PropMix->gf_buf,PropMix->n_orb,PropMix->nk)) < 1e-8);
    free(gfmix);
    CT_CHECK(cabs(trace_cd(VertMom->self_en,PropMom->n_orb,PropMom->nk)
                  -trace_cd(VertOrb->self_en,PropOrb->n_orb,PropOrb->nk)) < 1e-8);

    CT_CHECK(cabs(trace_cd(VertMix->self_en,PropMix->n_orb,PropMix->nk)
                  -trace_cd(VertOrb->self_en,PropOrb->n_orb,PropOrb->nk)) < 1e-8);

    tu_vertex_free( VertOrb );
    tu_vertex_free( VertMom );
    tu_vertex_free( VertMix );
    tu_selfenergy_free( selfOrb );
    tu_selfenergy_free( selfMom );
    tu_selfenergy_free( selfMix );
    tu_prop_free( PropMom );
    tu_prop_free( PropOrb );
    tu_prop_free( PropMix );
    tu_projection_free( ProjOrb );
    tu_projection_free( ProjMom );
    tu_projection_free( ProjMix );
    diverge_model_free(orb);
    diverge_model_free(mom);
    diverge_model_free(mix);
}

CT_TEST_NAME(honey,"self,honey") {
    if (compilation_options()) return;
    diverge_model_t* mix = KWARGS(gen_honeycomb_lat_hub, .nkx=12, .nky=12);
    diverge_model_t* orb = KWARGS(gen_honeycomb_lat_hub, .nkx=4, .nky=4, .lx=3, .ly=3);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mix, 1.1);

    tu_prop_t* PropOrb = tu_prop_init(orb);
    tu_prop_t* PropMix = tu_prop_init(mix);

    tu_projection_t* ProjOrb = tu_projection_init(orb);
    tu_projection_t* ProjMix = tu_projection_init(mix);

    tu_vertex_t* VertOrb = tu_vertex_init( orb, ProjOrb, PropOrb, true,true,true,true );
    tu_vertex_t* VertMix = tu_vertex_init( mix, ProjMix, PropMix,  true,true,true,true );
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,VertOrb,ProjOrb,value);
    init_random_PCD(mix,VertMix,ProjMix,value);

    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMix->Pch,VertMix->n_orbff,VertMix->my_nk)) < 1e-8);
    CT_CHECK(cabs(trace_cd(VertOrb->Cch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMix->Cch,VertMix->n_orbff,VertMix->my_nk)) < 1e-8);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff,VertOrb->my_nk)
                  -trace_cd(VertMix->Dch,VertMix->n_orbff,VertMix->my_nk)) < 1e-8);

    tu_selfenergy_t* selfOrb = tu_selfenergy_init( VertOrb );
    tu_selfenergy_t* selfMix = tu_selfenergy_init( VertMix );

    const double Lambda = 1e-1;
    tu_prop_GF_zero_T(PropOrb, Lambda, NULL);
    tu_prop_GF_zero_T(PropMix, Lambda, NULL);
    CT_CHECK(cabs(trace_cd(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix->nk)
                  -trace_cd(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb->nk)) < 1e-8);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropMix, gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertOrb->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropOrb, gfmix2);

    CT_CHECK(cabs(trace_batch_lin(gfmix,PropMix->n_orb*VertOrb->n_spin,PropMix->nk)
                  -trace_batch_lin(gfmix2,PropOrb->n_orb*VertOrb->n_spin,PropOrb->nk)) < 1e-8);

    tu_selfenergy_flow(selfOrb, VertOrb->self_en, VertOrb->Pch, VertOrb->Cch,
                    VertOrb->Dch, gfmix2, 1.0, Lambda);

    tu_selfenergy_flow(selfMix, VertMix->self_en, VertMix->Pch, VertMix->Cch,
                    VertMix->Dch, gfmix, 1.0, Lambda);
    free(gfmix);
    free(gfmix2);

    CT_CHECK(cabs(trace_cd(VertMix->self_en,PropMix->n_orb,PropMix->nk)
                  -trace_cd(VertOrb->self_en,PropOrb->n_orb,PropOrb->nk)) < 1e-8);

    tu_vertex_free( VertOrb );
    tu_vertex_free( VertMix );
    tu_prop_free( PropOrb );
    tu_prop_free( PropMix );
    tu_projection_free( ProjOrb );
    tu_projection_free( ProjMix );
    tu_selfenergy_free( selfOrb );
    tu_selfenergy_free( selfMix );
    diverge_model_free(orb);
    diverge_model_free(mix);
}

CT_TEST_NAME(honey_sym,"self,honeycomb,sym") {
    if (compilation_options()) return;
    diverge_model_t* mix = KWARGS(gen_honeycomb_lat_hub_sym, .nkx=12, .nky=12);
    diverge_model_t* orb = KWARGS(gen_honeycomb_lat_hub, .nkx=4, .nky=4, .lx=3, .ly=3);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mix, 1.1);

    tu_prop_t* PropOrb = tu_prop_init(orb);
    tu_prop_t* PropMix = tu_prop_init(mix);

    tu_projection_t* ProjOrb = tu_projection_init(orb);
    tu_projection_t* ProjMix = tu_projection_init(mix);

    tu_vertex_t* VertOrb = tu_vertex_init( orb, ProjOrb, PropOrb, true,true,true,true );
    tu_vertex_t* VertMix = tu_vertex_init( mix, ProjMix, PropMix,  true,true,true,true );
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,VertOrb,ProjOrb,value);
    init_random_PCD(mix,VertMix,ProjMix,value);

    tu_selfenergy_t* selfOrb = tu_selfenergy_init( VertOrb );
    tu_selfenergy_t* selfMix = tu_selfenergy_init( VertMix );

    const double Lambda = 1e-1;
    tu_prop_GF_zero_T(PropOrb, Lambda, NULL);
    tu_prop_GF_zero_T(PropMix, Lambda, NULL);
    CT_CHECK(cabs(trace_cd(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix->nk)
                  -trace_cd(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb->nk)) < 1e-8);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropMix, gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertOrb->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropOrb, gfmix2);

    CT_CHECK(cabs(trace_batch_lin(gfmix,PropMix->n_orb*VertOrb->n_spin,PropMix->nk)
                  -trace_batch_lin(gfmix2,PropOrb->n_orb*VertOrb->n_spin,PropOrb->nk)) < 1e-8);

    tu_selfenergy_flow(selfOrb, VertOrb->self_en, VertOrb->Pch, VertOrb->Cch,
                    VertOrb->Dch, gfmix2, 1.0, Lambda);

    tu_selfenergy_flow(selfMix, VertMix->self_en, VertMix->Pch, VertMix->Cch,
                    VertMix->Dch, gfmix, 1.0, Lambda);
    free(gfmix);
    free(gfmix2);

    CT_CHECK(cabs(trace_cd(VertMix->self_en,PropMix->n_orb,PropMix->nk)
                  -trace_cd(VertOrb->self_en,PropOrb->n_orb,PropOrb->nk)) < 1e-8);

    tu_vertex_free( VertMix );
    tu_vertex_free( VertOrb );
    tu_prop_free( PropOrb );
    tu_prop_free( PropMix );
    tu_projection_free( ProjOrb );
    tu_projection_free( ProjMix );
    tu_selfenergy_free( selfOrb );
    tu_selfenergy_free( selfMix );
    diverge_model_free(orb);
    diverge_model_free(mix);
}

CT_TEST_NAME(honey_rashba,"self,honeycomb,rashba") {
    if (compilation_options()) return;
    diverge_model_t* mix = KWARGS(gen_honeycomb_lat_rash, .nkx=12, .nky=12);
    diverge_model_t* orb = KWARGS(gen_honeycomb_lat_rash, .nkx=4, .nky=4, .lx=3, .ly=3);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mix, 1.1);

    tu_prop_t* PropOrb = tu_prop_init(orb);
    tu_prop_t* PropMix = tu_prop_init(mix);
    CT_CHECK(cabs(trace_cd(mix->internals->ham,mix->n_orb*mix->n_spin,PropMix->nk)
                  -trace_cd(orb->internals->ham,orb->n_orb*orb->n_spin,PropOrb->nk)) < 1e-8);

    tu_projection_t* ProjOrb = tu_projection_init(orb);
    tu_projection_t* ProjMix = tu_projection_init(mix);

    tu_vertex_t* VertOrb = tu_vertex_init( orb, ProjOrb, PropOrb, true,true,true,true );
    tu_vertex_t* VertMix = tu_vertex_init( mix, ProjMix, PropMix,  true,true,true,true );
    double value[3] = {0.1,-0.4,1.8};
    index_t n_sp2 = POW2(VertOrb->n_spin);
    init_random_PCD(orb,VertOrb,ProjOrb,value);
    init_random_PCD(mix,VertMix,ProjMix,value);

    CT_CHECK(cabs(trace_cd(VertOrb->Pch,VertOrb->n_orbff*n_sp2,VertOrb->my_nk)
                  -trace_cd(VertMix->Pch,VertMix->n_orbff*n_sp2,VertMix->my_nk)) < 1e-8);
    CT_CHECK(cabs(trace_cd(VertOrb->Cch,VertOrb->n_orbff*n_sp2,VertOrb->my_nk)
                  -trace_cd(VertMix->Cch,VertMix->n_orbff*n_sp2,VertMix->my_nk)) < 1e-8);
    CT_CHECK(cabs(trace_cd(VertOrb->Dch,VertOrb->n_orbff*n_sp2,VertOrb->my_nk)
                  -trace_cd(VertMix->Dch,VertMix->n_orbff*n_sp2,VertMix->my_nk)) < 1e-8);

    tu_selfenergy_t* selfOrb = tu_selfenergy_init( VertOrb );
    tu_selfenergy_t* selfMix = tu_selfenergy_init( VertMix );

    const double Lambda = 1e-1;
    tu_prop_GF_zero_T(PropOrb, Lambda, NULL);
    tu_prop_GF_zero_T(PropMix, Lambda, NULL);
    CT_CHECK(cabs(trace_cd(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix->nk)
                  -trace_cd(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb->nk)) < 1e-8);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropMix, gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertOrb->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropOrb, gfmix2);

    CT_CHECK(cabs(trace_batch_lin(gfmix,PropMix->n_orb*VertOrb->n_spin,PropMix->nk)
                  -trace_batch_lin(gfmix2,PropOrb->n_orb*VertOrb->n_spin,PropOrb->nk)) < 1e-8);

    tu_selfenergy_flow(selfOrb, VertOrb->self_en, VertOrb->Pch, VertOrb->Cch,
                    VertOrb->Dch, gfmix2, 1.0, Lambda);

    tu_selfenergy_flow(selfMix, VertMix->self_en, VertMix->Pch, VertMix->Cch,
                    VertMix->Dch, gfmix, 1.0, Lambda);
    free(gfmix);
    free(gfmix2);

    CT_CHECK(cabs(trace_cd(VertMix->self_en,PropMix->n_orb*VertOrb->n_spin,PropMix->nk)
                  -trace_cd(VertOrb->self_en,PropOrb->n_orb*VertOrb->n_spin,PropOrb->nk)) < 1e-8);

    tu_vertex_free( VertMix );
    tu_vertex_free( VertOrb );
    tu_prop_free( PropOrb );
    tu_prop_free( PropMix );
    tu_projection_free( ProjOrb );
    tu_projection_free( ProjMix );
    tu_selfenergy_free( selfOrb );
    tu_selfenergy_free( selfMix );
    diverge_model_free(orb);
    diverge_model_free(mix);
}

CT_TEST_NAME(honey_rash_ref,"self,rashba,refined") {
    if (compilation_options()) return;
    diverge_model_t* mix = KWARGS(gen_honeycomb_lat_rash, .nkx=44*3, .nky=44*3);
    diverge_model_t* orb = KWARGS(gen_honeycomb_lat_rash, .nkx=44, .nky=44, .nkxf=3, .nkyf=3);
    diverge_model_internals_tu(orb, 0.0);
    diverge_model_internals_tu(mix, 0.0);

    tu_prop_t* PropOrb = tu_prop_init(orb);
    tu_prop_t* PropMix = tu_prop_init(mix);
    CT_CHECK(cabs(trace_cd(mix->internals->ham,mix->n_orb*mix->n_spin,PropMix->nktot)
                  -trace_cd(orb->internals->ham,orb->n_orb*orb->n_spin,PropOrb->nktot)) < 1e-8);

    tu_projection_t* ProjOrb = tu_projection_init(orb);
    tu_projection_t* ProjMix = tu_projection_init(mix);

    tu_vertex_t* VertOrb = tu_vertex_init( orb, ProjOrb, PropOrb, true,true,true,true );
    tu_vertex_t* VertMix = tu_vertex_init( mix, ProjMix, PropMix,  true,true,true,true );
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,VertOrb,ProjOrb,value);
    init_random_PCD(mix,VertMix,ProjMix,value);

    tu_selfenergy_t* selfOrb = tu_selfenergy_init( VertOrb );
    tu_selfenergy_t* selfMix = tu_selfenergy_init( VertMix );

    const double Lambda = 4e-1;
    tu_prop_GF_zero_T(PropOrb, I128*Lambda, NULL);
    tu_prop_GF_zero_T(PropMix, I128*Lambda, NULL);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropMix, gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropOrb, gfmix2);
    CT_CHECK(cabs(trace_cd(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix->nktot)
                  -trace_cd(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb->nktot)) < 1e-8);

    tu_selfenergy_flow(selfOrb, VertOrb->self_en, VertOrb->Pch, VertOrb->Cch,
                    VertOrb->Dch, gfmix2, 1.0, Lambda);
    tu_selfenergy_flow(selfMix, VertMix->self_en, VertMix->Pch, VertMix->Cch,
                    VertMix->Dch, gfmix, 1.0, Lambda);

    free(gfmix);
    free(gfmix2);
    CT_CHECK(cabs(trace_cd(VertMix->self_en,PropMix->n_orb*VertOrb->n_spin,PropMix->nktot)
                  -trace_cd(VertOrb->self_en,PropOrb->n_orb*VertOrb->n_spin,PropOrb->nktot)) < 1e-3);

    tu_vertex_free( VertMix );
    tu_vertex_free( VertOrb );
    tu_prop_free( PropOrb );
    tu_prop_free( PropMix );
    tu_projection_free( ProjOrb );
    tu_projection_free( ProjMix );
    tu_selfenergy_free( selfOrb );
    tu_selfenergy_free( selfMix );
    diverge_model_free(orb);
    diverge_model_free(mix);
}

CT_TEST_NAME(hub_refined,"self,refined") {
    if (compilation_options()) return;
    diverge_model_t* mix = KWARGS(gen_square_lat_hub_sym, .nkx=100*3, .nky=100*3);
    diverge_model_t* orb = KWARGS(gen_square_lat_hub_sym, .nkx=100, .nky=100, .nkxf=3, .nkyf=3);
    diverge_model_internals_tu(orb, 0.0);
    diverge_model_internals_tu(mix, 0.0);

    tu_prop_t* PropOrb = tu_prop_init(orb);
    tu_prop_t* PropMix = tu_prop_init(mix);
    CT_CHECK(cabs(trace_cd(mix->internals->ham,mix->n_orb*mix->n_spin,PropMix->nktot)
                  -trace_cd(orb->internals->ham,orb->n_orb*orb->n_spin,PropOrb->nktot)) < 1e-8);

    tu_projection_t* ProjOrb = tu_projection_init(orb);
    tu_projection_t* ProjMix = tu_projection_init(mix);

    tu_vertex_t* VertOrb = tu_vertex_init( orb, ProjOrb, PropOrb, true,true,true,true );
    tu_vertex_t* VertMix = tu_vertex_init( mix, ProjMix, PropMix, true,true,true,true );
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,VertOrb,ProjOrb,value);
    init_random_PCD(mix,VertMix,ProjMix,value);

    tu_selfenergy_t* selfOrb = tu_selfenergy_init( VertOrb );
    tu_selfenergy_t* selfMix = tu_selfenergy_init( VertMix );

    const double Lambda = 0.4;
    tu_prop_GF_zero_T(PropOrb, I128*Lambda, NULL);
    tu_prop_GF_zero_T(PropMix, I128*Lambda, NULL);
    CT_CHECK(cabs(trace_cd(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix->nktot)
                  -trace_cd(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb->nktot)) < 1e-8);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropMix, gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    complex128_t* gf3 = (complex128_t*)calloc(VertMix->selfenergy_size,sizeof(complex128_t));
    tu_prop_GF_for_self(PropOrb, gfmix2);
    tu_prop_GF_for_self(PropOrb, gf3);

    tu_selfenergy_flow(selfOrb, VertOrb->self_en, VertOrb->Pch, VertOrb->Cch,
                    VertOrb->Dch, gfmix2, 1.0, Lambda);
    tu_selfenergy_flow(selfMix, VertMix->self_en, VertMix->Pch, VertMix->Cch,
                    VertMix->Dch, gfmix, 1.0, Lambda);

    fftw_execute_dft(tu_selfenergy_get_ifft_self_ip(selfOrb), (fftw_complex*)gfmix2, (fftw_complex*)gfmix2);
    tu_selfenergy_fft_interpolate(selfOrb, gfmix2);

    CT_CHECK(cabs(trace_cd(gfmix,PropMix->n_orb*VertOrb->n_spin,PropMix->nktot)
                  -trace_cd(gfmix2,PropOrb->n_orb*VertOrb->n_spin,PropOrb->nktot)) < 1e-3);
    free(gfmix);
    free(gfmix2);
    free(gf3);
    CT_CHECK(cabs(trace_cd(VertMix->self_en,PropMix->n_orb*VertOrb->n_spin,PropMix->nktot)
                  -trace_cd(VertOrb->self_en,PropOrb->n_orb*VertOrb->n_spin,PropOrb->nktot)) < 1e-3);
    tu_vertex_free( VertMix );
    tu_vertex_free( VertOrb );
    tu_prop_free( PropOrb );
    tu_prop_free( PropMix );
    tu_projection_free( ProjOrb );
    tu_projection_free( ProjMix );
    tu_selfenergy_free( selfOrb );
    tu_selfenergy_free( selfMix );
    diverge_model_free(orb);
    diverge_model_free(mix);
}


CT_TEST_NAME(real_vs_k_SU2_other,"self,real:mom,SU2,other") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=12, .nky=12);
    diverge_model_t* orb = KWARGS(gen_square_lat_hub, .nkx=1, .nky=1, .lx=12, .ly=12);
    diverge_model_internals_tu(mom, 1.1);
    diverge_model_internals_tu(orb, 1.1);

    diverge_flow_step_t* stepo = diverge_flow_step_init(orb, "TU", "PCDS");
    diverge_flow_step_t* stepm = diverge_flow_step_init(mom, "TU", "PCDS");

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<5; ++i) {
        if (Lambda < 0) break;
        diverge_flow_step_euler( stepo, Lambda, dLambda );
        diverge_flow_step_euler( stepm, Lambda, dLambda );
        Lambda += dLambda;
    }
    mpi_log_printf( "trace_cd_orb:%.5e, trace_cd_mom:%.5e\n",
            cabs(trace_cd(stepo->tu_vertex->self_en,stepo->tu_vertex->n_orb,stepo->tu_vertex->nk)),
            cabs(trace_cd(stepm->tu_vertex->self_en,stepm->tu_vertex->n_orb,stepm->tu_vertex->nk)) );
    CT_CHECK(cabs(trace_cd(stepo->tu_vertex->self_en,stepo->tu_vertex->n_orb,stepo->tu_vertex->nk)
                  -trace_cd(stepm->tu_vertex->self_en,stepm->tu_vertex->n_orb,stepm->tu_vertex->nk)) < 1e-8);
    diverge_flow_step_free(stepo);
    diverge_flow_step_free(stepm);
    diverge_model_free(orb);
    diverge_model_free(mom);
}

CT_TEST_NAME(pure,"self,pure") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=4, .nky=4);
    diverge_model_internals_tu(mom, 1.1);
    ((tu_data_t*)mom->internals->tu_data)->tu_selfenergy_flow = true;

    diverge_flow_step_t* step = diverge_flow_step_init(mom, "TU", "PCDS");

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<5; ++i) {
        if (Lambda < 0) break;
        diverge_flow_step_euler( step, Lambda, dLambda );
        Lambda += dLambda;
    }
    diverge_flow_step_free(step);
    diverge_model_free(mom);
}

CT_TEST_NAME(pchan,"self,P_channel") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=4, .nky=4);
    diverge_model_internals_tu(mom, 1.1);

    diverge_flow_step_t* step = diverge_flow_step_init(mom, "TU", "PS");

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<5; ++i) {
        if (Lambda < 0) break;
        diverge_flow_step_euler( step, Lambda, dLambda );
        Lambda += dLambda;
    }
    diverge_flow_step_free(step);
    diverge_model_free(mom);
}

CT_TEST_NAME(dchan,"self,D_channel") {
    if (compilation_options()) return;
    diverge_model_t* mom = KWARGS(gen_square_lat_hub, .nkx=4, .nky=4);
    diverge_model_internals_tu(mom, 1.1);

    diverge_flow_step_t* step = diverge_flow_step_init(mom, "TU", "DS");

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<5; ++i) {
        if (Lambda < 0) break;
        diverge_flow_step_euler( step, Lambda, dLambda );
        Lambda += dLambda;
    }
    diverge_flow_step_free(step);
    diverge_model_free(mom);
}

#endif // DIVERGE_SKIP_TESTS
