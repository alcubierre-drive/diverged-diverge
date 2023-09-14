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
#include "trivial_proj.hpp"
#include <random>


TEST_CASE("check vs trivial projections ", "[proj][proj_triv]")
{
    diverge_model_t* mom = gen_square_lat_hub(6,6,1,1);
    diverge_model_internals_tu(mom, 1.1);

    Projection ProjMom (mom);
    tu_loop_t LoopMom (mom);
    Vertex VertMom(mom, &ProjMom, &LoopMom);

    std::random_device rd{};
    std::mt19937 re{rd()};
    std::normal_distribution<double> unif{0, 5.0};

    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t j = 0; j < VertMom.full_vert_size; ++j) {
        VertMom.Pch[j] = unif(re)+I128*unif(re);
        VertMom.Cch[j] = unif(re)+I128*unif(re);
        VertMom.Dch[j] = unif(re)+I128*unif(re);
    }

    Vertex VertOrb(VertMom);

    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.my_nk)) < 1e-10);
    CHECK(std::abs(trace(VertOrb.Cch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMom.Cch,VertMom.n_orbff,VertMom.my_nk)) < 1e-10);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.my_nk)) < 1e-10);

    complex128_t* r2 = (complex128_t*)calloc(VertOrb.full_vert_size,sizeof(complex128_t));
    complex128_t* r1 = (complex128_t*)calloc(VertOrb.full_vert_size,sizeof(complex128_t));

    P_projection(r2,&VertMom,&ProjMom);
    t_P_projection(r1,&VertOrb,mom);
    CHECK(check_equal(r1,r2,POW2(VertOrb.n_orbff)*VertOrb.my_nk));

    C_projection(r2,&VertMom,&ProjMom);
    t_C_projection(r1,&VertOrb,mom);
    CHECK(check_equal(r1,r2,POW2(VertOrb.n_orbff)*VertOrb.my_nk));

    D_projection(r2,&VertMom,&ProjMom);
    t_D_projection(r1,&VertOrb,mom);
    CHECK(check_equal(r1,r2,POW2(VertOrb.n_orbff)*VertOrb.my_nk));

    diverge_model_free(mom);
    free(r1);
    free(r2);
}

TEST_CASE("check vs trivial projections honey ", "[proj][proj_triv_honey]")
{
    diverge_model_t* mom = gen_honeycomb_lat_hub(6,6,1,1);
    diverge_model_internals_tu(mom, 0.7);

    Projection ProjMom (mom);
    tu_loop_t LoopMom (mom);
    Vertex VertMom(mom, &ProjMom, &LoopMom);

        std::random_device rd{};
        std::mt19937 re{rd()};
        std::normal_distribution<double> unif{0, 5.0};

    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t j = 0; j < VertMom.full_vert_size; ++j) {
        VertMom.Pch[j] = unif(re)+I128*unif(re);
        VertMom.Cch[j] = unif(re)+I128*unif(re);
        VertMom.Dch[j] = unif(re)+I128*unif(re);
    }

    Vertex VertOrb(VertMom);

    CHECK(std::abs(trace(VertOrb.Pch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMom.Pch,VertMom.n_orbff,VertMom.my_nk)) < 1e-10);
    CHECK(std::abs(trace(VertOrb.Cch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMom.Cch,VertMom.n_orbff,VertMom.my_nk)) < 1e-10);
    CHECK(std::abs(trace(VertOrb.Dch,VertOrb.n_orbff,VertOrb.my_nk)
                  -trace(VertMom.Dch,VertMom.n_orbff,VertMom.my_nk)) < 1e-10);

    complex128_t* r2 = (complex128_t*)calloc(VertOrb.full_vert_size,sizeof(complex128_t));
    complex128_t* r1 = (complex128_t*)calloc(VertOrb.full_vert_size,sizeof(complex128_t));

    P_projection(r2,&VertMom,&ProjMom);
    t_P_projection(r1,&VertOrb,mom);
    CHECK(check_equal(r1,r2,POW2(VertOrb.n_orbff)*VertOrb.my_nk));

    C_projection(r2,&VertMom,&ProjMom);
    t_C_projection(r1,&VertOrb,mom);
    CHECK(check_equal(r1,r2,POW2(VertOrb.n_orbff)*VertOrb.my_nk));

    D_projection(r2,&VertMom,&ProjMom);
    t_D_projection(r1,&VertOrb,mom);
    CHECK(check_equal(r1,r2,POW2(VertOrb.n_orbff)*VertOrb.my_nk));

    diverge_model_free(mom);
    free(r1);
    free(r2);
}
