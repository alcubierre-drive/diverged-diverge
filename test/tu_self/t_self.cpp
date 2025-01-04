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
#include "../../src/tu/projection_wrapper.h"
#include "../../src/tu/projection_handler.h"
#include "../../src/tu/selfenergy.h"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"

#ifndef USE_GF_FLOATS
void init_random_PCD(diverge_model_t* model, Vertex* vertex, Projection * proj, double* value) {
    tu_data_t* tu_data = (tu_data_t*)(model -> internals-> tu_data);
    symmstruct_t* symm = tu_data->symm;
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
                    += value[1]*std::conj(ff(model->internals->kmesh+3*(symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off]),
                                            Rveco + (boo[j]+bj)*3));
        vertex->Dch[j + n_orbffo * (sl + n_spin2*(tu_ffo[boo[j]+bj].oto+n_orbffo*(sr +n_spin2*k)))]
                    += value[2]*std::conj(ff(model->internals->kmesh+3*(symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off]),
                                            Rveco + (boo[j]+bj)*3));
    }
    D_projection(vertex->Dch,vertex,proj);
    C_projection(vertex->Cch,vertex,proj);
    P_projection(vertex->Pch,vertex,proj);
    free(Rveco);
}


#ifndef USE_MPI
TEST_CASE("self energy Realspace vs k space SU2 static","[self][self_kVSr_SU2]"){
    diverge_model_t* mom = gen_square_lat_hub(12, 12);
    diverge_model_t* orb = gen_square_lat_hub(1, 1, 12, 12);
    diverge_model_t* mix = gen_square_lat_hub(4, 4, 3, 3);
    diverge_model_internals_tu(mom, 1.1);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mix, 1.1);

    tu_loop_t PropMom (mom);
    tu_loop_t PropOrb (orb);
    tu_loop_t PropMix (mix);

    Projection* ProjOrb = projection_init(orb);
    Projection* ProjMom = projection_init(mom);
    Projection* ProjMix = projection_init(mix);

    Vertex VertOrb(orb, ProjOrb,&PropOrb, true,true,true,true);
    Vertex VertMom(mom, ProjMom,&PropMom, true,true,true,true);
    Vertex VertMix(mix, ProjMix,&PropMix, true,true,true,true);
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,&VertOrb,ProjOrb,value);
    init_random_PCD(mom,&VertMom,ProjMom,value);
    init_random_PCD(mix,&VertMix,ProjMix,value);

    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.my_nk)) < 1e-8);
    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMix.Pch,VertMix.n_orbff,VertMix.my_nk)) < 1e-8);

    selfenergy_t* selfOrb = selfenergy_init(&VertOrb);
    selfenergy_t* selfMom = selfenergy_init(&VertMom);
    selfenergy_t* selfMix = selfenergy_init(&VertMix);

    const double Lambda = 1e-1;
    PropMom.GF_zero_T(Lambda);
    PropOrb.GF_zero_T(Lambda);
    PropMix.GF_zero_T(Lambda);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    PropMix.get_GF_for_self(gfmix);

    selfenergy_flow(selfOrb, VertOrb.self_en, VertOrb.Pch, VertOrb.Cch,
                    VertOrb.Dch, PropOrb.gf_buf, 1.0);
    selfenergy_flow(selfMom, VertMom.self_en, VertMom.Pch, VertMom.Cch,
                    VertMom.Dch, PropMom.gf_buf, 1.0);
    selfenergy_flow(selfMix, VertMix.self_en, VertMix.Pch, VertMix.Cch,
                    VertMix.Dch, gfmix, 1.0);
    CHECK(std::abs(trace(PropMom.gf_buf,PropMom.n_orb,PropMom.nk)
                  -trace(PropMix.gf_buf,PropMix.n_orb,PropMix.nk)) < 1e-8);
    free(gfmix);
    CHECK(std::abs(trace(VertMom.self_en,PropMom.n_orb,PropMom.nk)
                  -trace(VertOrb.self_en,PropOrb.n_orb,PropOrb.nk)) < 1e-8);

    CHECK(std::abs(trace(VertMix.self_en,PropMix.n_orb,PropMix.nk)
                  -trace(VertOrb.self_en,PropOrb.n_orb,PropOrb.nk)) < 1e-8);

    selfenergy_free( selfOrb );
    selfenergy_free( selfMom );
    selfenergy_free( selfMix );

    projection_free( ProjOrb );
    projection_free( ProjMom );
    projection_free( ProjMix );

    diverge_model_free(orb);
    diverge_model_free(mom);
    diverge_model_free(mix);
}

TEST_CASE("self energy honey","[self][self_honey]") {
    diverge_model_t* mix = gen_honeycomb_lat_hub(12, 12);
    diverge_model_t* orb = gen_honeycomb_lat_hub(4, 4, 3, 3);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mix, 1.1);

    tu_loop_t PropOrb (orb);
    tu_loop_t PropMix (mix);

    Projection* ProjOrb = projection_init(orb);
    Projection* ProjMix = projection_init(mix);

    Vertex VertOrb(orb, ProjOrb,&PropOrb, true,true,true,true);
    Vertex VertMix(mix, ProjMix,&PropMix,  true,true,true,true);
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,&VertOrb,ProjOrb,value);
    init_random_PCD(mix,&VertMix,ProjMix,value);

    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMix.Pch,VertMix.n_orbff,VertMix.my_nk)) < 1e-8);
    CHECK(std::abs(trace(VertOrb.Cch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMix.Cch,VertMix.n_orbff,VertMix.my_nk)) < 1e-8);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMix.Dch,VertMix.n_orbff,VertMix.my_nk)) < 1e-8);

    selfenergy_t* selfOrb = selfenergy_init( (void*)&VertOrb );
    selfenergy_t* selfMix = selfenergy_init( (void*)&VertMix );

    const double Lambda = 1e-1;
    PropOrb.GF_zero_T(Lambda);
    PropMix.GF_zero_T(Lambda);
    CHECK(std::abs(trace(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix.nk)
                  -trace(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb.nk)) < 1e-8);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    PropMix.get_GF_for_self(gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertOrb.selfenergy_size,sizeof(complex128_t));
    PropOrb.get_GF_for_self(gfmix2);

    CHECK(std::abs(trace_batch_lin(gfmix,PropMix.n_orb*VertOrb.n_spin,PropMix.nk)
                  -trace_batch_lin(gfmix2,PropOrb.n_orb*VertOrb.n_spin,PropOrb.nk)) < 1e-8);

    selfenergy_flow(selfOrb, VertOrb.self_en, VertOrb.Pch, VertOrb.Cch,
                    VertOrb.Dch, gfmix2, 1.0);

    selfenergy_flow(selfMix, VertMix.self_en, VertMix.Pch, VertMix.Cch,
                    VertMix.Dch, gfmix, 1.0);
    free(gfmix);
    free(gfmix2);

    CHECK(std::abs(trace(VertMix.self_en,PropMix.n_orb,PropMix.nk)
                  -trace(VertOrb.self_en,PropOrb.n_orb,PropOrb.nk)) < 1e-8);

    projection_free( ProjOrb );
    projection_free( ProjMix );

    selfenergy_free( selfOrb );
    selfenergy_free( selfMix );

    diverge_model_free(orb);
    diverge_model_free(mix);
}

TEST_CASE("self energy honey sym","[self][self_honey]") {
    diverge_model_t* mix = gen_honeycomb_lat_hub_sym(12, 12);
    diverge_model_t* orb = gen_honeycomb_lat_hub(4, 4, 3, 3);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mix, 1.1);

    tu_loop_t PropOrb (orb);
    tu_loop_t PropMix (mix);

    Projection* ProjOrb = projection_init(orb);
    Projection* ProjMix = projection_init(mix);

    Vertex VertOrb(orb, ProjOrb,&PropOrb, true,true,true,true);
    Vertex VertMix(mix, ProjMix,&PropMix,  true,true,true,true);
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,&VertOrb,ProjOrb,value);
    init_random_PCD(mix,&VertMix,ProjMix,value);

    selfenergy_t* selfOrb = selfenergy_init( (void*)&VertOrb );
    selfenergy_t* selfMix = selfenergy_init( (void*)&VertMix );

    const double Lambda = 1e-1;
    PropOrb.GF_zero_T(Lambda);
    PropMix.GF_zero_T(Lambda);
    CHECK(std::abs(trace(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix.nk)
                  -trace(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb.nk)) < 1e-8);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    PropMix.get_GF_for_self(gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertOrb.selfenergy_size,sizeof(complex128_t));
    PropOrb.get_GF_for_self(gfmix2);

    CHECK(std::abs(trace_batch_lin(gfmix,PropMix.n_orb*VertOrb.n_spin,PropMix.nk)
                  -trace_batch_lin(gfmix2,PropOrb.n_orb*VertOrb.n_spin,PropOrb.nk)) < 1e-8);

    selfenergy_flow(selfOrb, VertOrb.self_en, VertOrb.Pch, VertOrb.Cch,
                    VertOrb.Dch, gfmix2, 1.0);

    selfenergy_flow(selfMix, VertMix.self_en, VertMix.Pch, VertMix.Cch,
                    VertMix.Dch, gfmix, 1.0);
    free(gfmix);
    free(gfmix2);

    CHECK(std::abs(trace(VertMix.self_en,PropMix.n_orb,PropMix.nk)
                  -trace(VertOrb.self_en,PropOrb.n_orb,PropOrb.nk)) < 1e-8);

    projection_free( ProjOrb );
    projection_free( ProjMix );

    selfenergy_free( selfOrb );
    selfenergy_free( selfMix );

    diverge_model_free(orb);
    diverge_model_free(mix);
}

TEST_CASE("self energy honey rash","[self][self_rash]") {
    diverge_model_t* mix = gen_honeycomb_lat_rash(12, 12);
    diverge_model_t* orb = gen_honeycomb_lat_rash(4, 4, 3, 3);
    diverge_model_internals_tu(orb, 1.1);
    diverge_model_internals_tu(mix, 1.1);

    tu_loop_t PropOrb (orb);
    tu_loop_t PropMix (mix);
    CHECK(std::abs(trace(mix->internals->ham,mix->n_orb*mix->n_spin,PropMix.nk)
                  -trace(orb->internals->ham,orb->n_orb*orb->n_spin,PropOrb.nk)) < 1e-8);

    Projection* ProjOrb = projection_init(orb);
    Projection* ProjMix = projection_init(mix);

    Vertex VertOrb(orb, ProjOrb,&PropOrb, true,true,true,true);
    Vertex VertMix(mix, ProjMix,&PropMix,  true,true,true,true);
    double value[3] = {0.1,-0.4,1.8};
    index_t n_sp2 = POW2(VertOrb.n_spin);
    init_random_PCD(orb,&VertOrb,ProjOrb,value);
    init_random_PCD(mix,&VertMix,ProjMix,value);

    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff*n_sp2,VertOrb.my_nk)
                  -trace(VertMix.Pch,VertMix.n_orbff*n_sp2,VertMix.my_nk)) < 1e-8);
    CHECK(std::abs(trace(VertOrb.Cch,VertOrb.n_orbff*n_sp2,VertOrb.my_nk)
                  -trace(VertMix.Cch,VertMix.n_orbff*n_sp2,VertMix.my_nk)) < 1e-8);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff*n_sp2,VertOrb.my_nk)
                  -trace(VertMix.Dch,VertMix.n_orbff*n_sp2,VertMix.my_nk)) < 1e-8);

    selfenergy_t* selfOrb = selfenergy_init( (void*)&VertOrb );
    selfenergy_t* selfMix = selfenergy_init( (void*)&VertMix );

    const double Lambda = 1e-1;
    PropOrb.GF_zero_T(Lambda);
    PropMix.GF_zero_T(Lambda);
    CHECK(std::abs(trace(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix.nk)
                  -trace(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb.nk)) < 1e-8);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    PropMix.get_GF_for_self(gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertOrb.selfenergy_size,sizeof(complex128_t));
    PropOrb.get_GF_for_self(gfmix2);

    CHECK(std::abs(trace_batch_lin(gfmix,PropMix.n_orb*VertOrb.n_spin,PropMix.nk)
                  -trace_batch_lin(gfmix2,PropOrb.n_orb*VertOrb.n_spin,PropOrb.nk)) < 1e-8);

    selfenergy_flow(selfOrb, VertOrb.self_en, VertOrb.Pch, VertOrb.Cch,
                    VertOrb.Dch, gfmix2, 1.0);

    selfenergy_flow(selfMix, VertMix.self_en, VertMix.Pch, VertMix.Cch,
                    VertMix.Dch, gfmix, 1.0);
    free(gfmix);
    free(gfmix2);

    CHECK(std::abs(trace(VertMix.self_en,PropMix.n_orb*VertOrb.n_spin,PropMix.nk)
                  -trace(VertOrb.self_en,PropOrb.n_orb*VertOrb.n_spin,PropOrb.nk)) < 1e-8);

    projection_free( ProjOrb );
    projection_free( ProjMix );
    selfenergy_free( selfOrb );
    selfenergy_free( selfMix );
    diverge_model_free(orb);
    diverge_model_free(mix);
}


inline static void fftw_execute_dft_helper(const fftw_plan p, complex128_t* in,
                                                                complex128_t* out){
    fftw_execute_dft(p,reinterpret_cast<fftw_complex*>(in),
                        reinterpret_cast<fftw_complex*>(out));
}

TEST_CASE("self energy honey rash ref","[self][self_ref]") {
    diverge_model_t* mix = gen_honeycomb_lat_rash(44*3, 44*3, 1, 1, 1, 1);
    diverge_model_t* orb = gen_honeycomb_lat_rash(44, 44, 1, 1, 3, 3);
    diverge_model_internals_tu(orb, 0.0);
    diverge_model_internals_tu(mix, 0.0);

    tu_loop_t PropOrb (orb);
    tu_loop_t PropMix (mix);
    CHECK(std::abs(trace(mix->internals->ham,mix->n_orb*mix->n_spin,PropMix.nktot)
                  -trace(orb->internals->ham,orb->n_orb*orb->n_spin,PropOrb.nktot)) < 1e-8);

    Projection* ProjOrb = projection_init(orb);
    Projection* ProjMix = projection_init(mix);

    Vertex VertOrb(orb, ProjOrb,&PropOrb, true,true,true,true);
    Vertex VertMix(mix, ProjMix,&PropMix,  true,true,true,true);
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,&VertOrb,ProjOrb,value);
    init_random_PCD(mix,&VertMix,ProjMix,value);

    selfenergy_t* selfOrb = selfenergy_init( (void*)&VertOrb );
    selfenergy_t* selfMix = selfenergy_init( (void*)&VertMix );

    const double Lambda = 4e-1;
    PropOrb.GF_zero_T(I128*Lambda);
    PropMix.GF_zero_T(I128*Lambda);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    PropMix.get_GF_for_self(gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    PropOrb.get_GF_for_self(gfmix2);
    CHECK(std::abs(trace(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix.nktot)
                  -trace(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb.nktot)) < 1e-8);

    selfenergy_flow(selfOrb, VertOrb.self_en, VertOrb.Pch, VertOrb.Cch,
                    VertOrb.Dch, gfmix2, 1.0);
    selfenergy_flow(selfMix, VertMix.self_en, VertMix.Pch, VertMix.Cch,
                    VertMix.Dch, gfmix, 1.0);

    free(gfmix);
    free(gfmix2);
    CHECK(std::abs(trace(VertMix.self_en,PropMix.n_orb*VertOrb.n_spin,PropMix.nktot)
                  -trace(VertOrb.self_en,PropOrb.n_orb*VertOrb.n_spin,PropOrb.nktot)) < 1e-3);

    projection_free( ProjOrb );
    projection_free( ProjMix );
    selfenergy_free( selfOrb );
    selfenergy_free( selfMix );
    diverge_model_free(orb);
    diverge_model_free(mix);
}

TEST_CASE("self energy hub ref","[self][self_href]") {
    diverge_model_t* mix = gen_square_lat_hub_sym(100*3, 100*3, 1, 1, 1, 0.1, 3, 0.5);
    diverge_model_t* orb = gen_square_lat_hub_sym(100, 100, 3, 3, 1, 0.1, 3, 0.5);
    diverge_model_internals_tu(orb, 0.0);
    diverge_model_internals_tu(mix, 0.0);

    tu_loop_t PropOrb (orb);
    tu_loop_t PropMix (mix);
    CHECK(std::abs(trace(mix->internals->ham,mix->n_orb*mix->n_spin,PropMix.nktot)
                  -trace(orb->internals->ham,orb->n_orb*orb->n_spin,PropOrb.nktot)) < 1e-8);

    Projection* ProjOrb = projection_init(orb);
    Projection* ProjMix = projection_init(mix);

    Vertex VertOrb(orb, ProjOrb,&PropOrb, true,true,true,true);
    Vertex VertMix(mix, ProjMix,&PropMix,  true,true,true,true);
    double value[3] = {0.1,-0.4,1.8};
    init_random_PCD(orb,&VertOrb,ProjOrb,value);
    init_random_PCD(mix,&VertMix,ProjMix,value);

    selfenergy_t* selfOrb = selfenergy_init( (void*)&VertOrb );
    selfenergy_t* selfMix = selfenergy_init( (void*)&VertMix );

    const double Lambda = 0.4;
    PropOrb.GF_zero_T(I128*Lambda);
    PropMix.GF_zero_T(I128*Lambda);
    CHECK(std::abs(trace(mix->internals->greens,mix->n_orb*mix->n_spin,PropMix.nktot)
                  -trace(orb->internals->greens,orb->n_orb*orb->n_spin,PropOrb.nktot)) < 1e-8);
    complex128_t* gfmix = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    PropMix.get_GF_for_self(gfmix);
    complex128_t* gfmix2 = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    complex128_t* gf3 = (complex128_t*)calloc(VertMix.selfenergy_size,sizeof(complex128_t));
    PropOrb.get_GF_for_self(gfmix2);
    PropOrb.get_GF_for_self(gf3);

    selfenergy_flow(selfOrb, VertOrb.self_en, VertOrb.Pch, VertOrb.Cch,
                    VertOrb.Dch, gfmix2, 1.0);
    selfenergy_flow(selfMix, VertMix.self_en, VertMix.Pch, VertMix.Cch,
                    VertMix.Dch, gfmix, 1.0);

    fftw_execute_dft_helper(selfOrb->ifft_self_ip, gfmix2, gfmix2);
    selfenergy_fft_interpolate(selfOrb, gfmix2);

    CHECK(std::abs(trace(gfmix,PropMix.n_orb*VertOrb.n_spin,PropMix.nktot)
                  -trace(gfmix2,PropOrb.n_orb*VertOrb.n_spin,PropOrb.nktot)) < 1e-3);
    free(gfmix);
    free(gfmix2);
    free(gf3);
    CHECK(std::abs(trace(VertMix.self_en,PropMix.n_orb*VertOrb.n_spin,PropMix.nktot)
                  -trace(VertOrb.self_en,PropOrb.n_orb*VertOrb.n_spin,PropOrb.nktot)) < 1e-3);
    projection_free( ProjOrb );
    projection_free( ProjMix );
    selfenergy_free( selfOrb );
    selfenergy_free( selfMix );
    diverge_model_free(orb);
    diverge_model_free(mix);
}


TEST_CASE("self energy Realspace vs k space SU2 flow","[self][self_flow]"){
    diverge_model_t* mom = gen_square_lat_hub(12, 12);
    diverge_model_t* orb = gen_square_lat_hub(1, 1, 12, 12);
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
    CHECK(std::abs(trace(stepo->tu_vertex->self_en,stepo->tu_vertex->n_orb,stepo->tu_vertex->nk)
                  -trace(stepm->tu_vertex->self_en,stepm->tu_vertex->n_orb,stepm->tu_vertex->nk)) < 1e-8);
    diverge_flow_step_free(stepo);
    diverge_flow_step_free(stepm);
    diverge_model_free(orb);
    diverge_model_free(mom);
}

TEST_CASE("pure selfenergy flow","[self][self_flow]"){
    diverge_model_t* mom = gen_square_lat_hub(4, 4);
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

TEST_CASE("P+self","[self][self_flow]"){
    diverge_model_t* mom = gen_square_lat_hub(4, 4);
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

TEST_CASE("D+self","[self][self_flow]"){
    diverge_model_t* mom = gen_square_lat_hub(4, 4);
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

#endif // MPI
#endif

#endif // DIVERGE_SKIP_TESTS
