#ifndef DIVERGE_SKIP_TESTS

// #include "../catch.hpp"
#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../model_examples/old.h"
#include "../../src/diverge_flow_step_internal.h"
#include "../../src/diverge_model_internals.h"
#include "../model_examples/model_constructor.hpp"
#include "../helper_functions.hpp"

#define CT_SUITE_NAME tu_completeness
#include "../ctest.h"

CT_TEST(tu_honeycomb) {
    diverge_model_t* model = gen_honeycomb_lat_hub(6,6,1,1);
    diverge_model_internals_tu(model, 19.0);
    tu_data_t* tu_data = (tu_data_t*)(model -> internals-> tu_data);
    tu_formfactor_t* tu_ff = model->tu_ff;
    double* Rvec = (double*)calloc(model->n_tu_ff*3,sizeof(double));
    fill_Rvec(model, Rvec);
    index_t nk = tu_data->nk;
    index_t n_orbff = tu_data->n_orbff;
    index_t n_orb = model->n_orb;
    double* kmesh = model->internals->kmesh;

    for(index_t k = 0; k < nk; ++k)
    for(index_t kp = 0; kp < nk; ++kp)
    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o2 = 0; o2 < n_orb; ++o2) {
        complex128_t help = 0.0;
        for(index_t o = 0; o < n_orbff; ++o)
        if(tu_ff[o].oto == o1 && o2 == tu_ff[o].oto){
            help += ff(kmesh+3*k,Rvec+3*o)
                    *std::conj(ff(kmesh+3*kp,Rvec+3*o));
        }
        if(k==kp && o1==o2) {
            CT_CHECK(std::abs(help-(double)nk*(double)n_orb) < 1e-10);
        }else{
            CT_CHECK(std::abs(help) < 1e-10);
        }
    }

    for(index_t b1 = 0; b1 < n_orbff; ++b1)
    for(index_t b2 = 0; b2 < n_orbff; ++b2)
    if(tu_ff[b1].ofrom == tu_ff[b2].ofrom){
        complex128_t help = 0.0;
        for(index_t o = 0; o < n_orb; ++o)
        for(index_t k = 0; k < nk; ++k)
        if(tu_ff[b1].oto == o && o == tu_ff[b2].oto){
            help += ff(kmesh+3*k,Rvec+3*b1)
                    *std::conj(ff(kmesh+3*k,Rvec+3*b2));
        }
        if(b1==b2) {
            CT_CHECK(std::abs(help-(double)nk) < 1e-10);
        }else{
            CT_CHECK(std::abs(help) < 1e-10);
        }
    }

    free(Rvec);
    diverge_model_free(model);
}

#endif // DIVERGE_SKIP_TESTS
