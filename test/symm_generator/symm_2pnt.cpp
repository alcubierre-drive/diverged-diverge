#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME twopoint_sym
#include "../ctest.h"

#include "../../src/diverge_momentum_gen.h"
#include "../../src/diverge_symmetrize.h"
#include "../../src/diverge_common.h"
#include "../../src/diverge_model.h"
#include "../../src/diverge_internals_struct.h"
// #include "../catch.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/misc/generate_symmetries.h"
#include "../../src/misc/shared_mem.h"

CT_TEST(toymodel) {
    site_descr_t sites[1];

    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;

    index_t n_spin = 1;

    diverge_model_t* model = diverge_model_init();
    model->nk[0] = 6, model->nk[1] = 6;
    model->nkf[0] = 1, model->nkf[1] = 1;
    model->n_orb = 1;
    model->SU2 = true;
    model->lattice[0][0] = 1., model->lattice[0][1] = 0.0, model->lattice[0][2] = 0.0;
    model->lattice[1][0] = 0.0, model->lattice[1][1] = 1., model->lattice[1][2] = 0.0;
    model->lattice[2][0] = 0.0, model->lattice[2][1] = 0.0, model->lattice[2][2] = 1.0;
    // s, p+ and p- hubbard model with C4

    model->orb_symmetries = (complex128_t*)calloc(1*4,sizeof(complex128_t));
    sym_op_t curr_symm;
    curr_symm.type = 'E'; curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 0.;curr_symm.normal_vector[2] = 1.;
    model->n_sym = 4;
    diverge_generate_symm_trafo(n_spin,sites, 1,&curr_symm, 1,
                            &(model->rs_symmetries[0][0][0]),model->orb_symmetries);
    curr_symm.type = 'R';
    curr_symm.angle = 90;
    diverge_generate_symm_trafo(n_spin,sites, 1,&curr_symm, 1,
                            &(model->rs_symmetries[1][0][0]),model->orb_symmetries+1);
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(n_spin,sites, 1,&curr_symm, 1,
                            &(model->rs_symmetries[2][0][0]),model->orb_symmetries+2);
    curr_symm.angle = 270;
    diverge_generate_symm_trafo(n_spin,sites, 1,&curr_symm, 1,
                            &(model->rs_symmetries[3][0][0]),model->orb_symmetries+3);
    model->n_hop = 2;
    model->hop = (rs_hopping_t*)calloc(model->n_hop,sizeof(rs_hopping_t));
    model->hop[0].R[0] =  1;
    model->hop[1].R[0] =  -1;
    model->hop[0].t = 1.;
    model->hop[1].t = 1.;
    diverge_model_internals_common( model );

    mpi_log_printf("generate second model\n")
    diverge_model_t* model2 = diverge_model_init();
    model2->nk[0] = 2, model2->nk[1] = 2;
    model2->nkf[0] = 3, model2->nkf[1] = 3;
    model2->n_orb = 1;
    model2->SU2 = true;
    model2->lattice[0][0] = 1., model2->lattice[0][1] = 0.0, model2->lattice[0][2] = 0.0;
    model2->lattice[1][0] = 0.0, model2->lattice[1][1] = 1., model2->lattice[1][2] = 0.0;
    model2->lattice[2][0] = 0.0, model2->lattice[2][1] = 0.0, model2->lattice[2][2] = 1.0;
    // s, p+ and p- hubbard model with C4

    model2->orb_symmetries = (complex128_t*)calloc(1*4,sizeof(complex128_t));

    curr_symm.type = 'E'; curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 0.;curr_symm.normal_vector[2] = 1.;
    model2->n_sym = 4;
    diverge_generate_symm_trafo(n_spin,sites, 1,&curr_symm, 1,
                            &(model2->rs_symmetries[0][0][0]),model2->orb_symmetries);
    curr_symm.type = 'R';
    curr_symm.angle = 90;
    diverge_generate_symm_trafo(n_spin,sites, 1,&curr_symm, 1,
                            &(model2->rs_symmetries[1][0][0]),model2->orb_symmetries+1);
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(n_spin,sites, 1,&curr_symm, 1,
                            &(model2->rs_symmetries[2][0][0]),model2->orb_symmetries+2);
    curr_symm.angle = 270;
    diverge_generate_symm_trafo(n_spin,sites, 1,&curr_symm, 1,
                            &(model2->rs_symmetries[3][0][0]),model2->orb_symmetries+3);
    model2->n_hop = 2;
    model2->hop = (rs_hopping_t*)calloc(model2->n_hop,sizeof(rs_hopping_t));
    model2->hop[0].R[1] =  1;
    model2->hop[1].R[1] =  -1;
    model2->hop[1].t = 1.;
    model2->hop[0].t = 1.;
    diverge_model_internals_common( model2 );

    if (shared_exclusive_enter( model->internals->ham ) == 1)
        diverge_symmetrize_2pt_coarse( model, model->internals->ham, NULL );
    shared_exclusive_wait( model->internals->ham );

    if (shared_exclusive_enter( model2->internals->ham ) == 1)
        diverge_symmetrize_2pt_fine( model2, model2->internals->ham, NULL );
    shared_exclusive_wait( model2->internals->ham );

    for(index_t i = 0; i < 6*6; ++i) {
        CT_CHECK(std::abs(model->internals->ham[i] - model2->internals->ham[i]) < 1e-10);
    }
    diverge_model_free(model);
    diverge_model_free(model2);
}

CT_TEST(honeycomb) {
    mpi_log_printf("symm_honey \n");
    site_descr_t sites[2];

    sites[0].n_functions = 2;
    sites[1].n_functions = 2;

    sites[0].amplitude[0] = 1./sqrt(2.);
    sites[0].amplitude[1] = I128/sqrt(2.);
    sites[0].function[0] = orb_px;
    sites[0].function[1] = orb_py;

    sites[1].amplitude[0] = 1./sqrt(2.);
    sites[1].amplitude[1] = -I128/sqrt(2.);
    sites[1].function[0] = orb_px;
    sites[1].function[1] = orb_py;

    index_t n_spin = 1;

    diverge_model_t* model = diverge_model_init();
    model->nk[0] = 2, model->nk[1] = 2;
    model->nkf[0] = 3, model->nkf[1] = 3;
    model->n_orb = 2;
    model->SU2 = true;
    model->lattice[0][0] = 1., model->lattice[0][1] = 0.0, model->lattice[0][2] = 0.0;
    model->lattice[1][0] = 0.0, model->lattice[1][1] = 1., model->lattice[1][2] = 0.0;
    model->lattice[2][0] = 0.0, model->lattice[2][1] = 0.0, model->lattice[2][2] = 1.0;
    // s, p+ and p- hubbard model with C4

    model->orb_symmetries = (complex128_t*)calloc(4*4,sizeof(complex128_t));
    sym_op_t curr_symm;
    curr_symm.type = 'E'; curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 0.;curr_symm.normal_vector[2] = 1.;
    model->n_sym = 4;
    diverge_generate_symm_trafo(n_spin,sites, 2,&curr_symm, 1,
                            &(model->rs_symmetries[0][0][0]),model->orb_symmetries);
    curr_symm.type = 'R';
    curr_symm.angle = 90;
    diverge_generate_symm_trafo(n_spin,sites, 2,&curr_symm, 1,
                            &(model->rs_symmetries[1][0][0]),model->orb_symmetries+4);
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(n_spin,sites, 2,&curr_symm, 1,
                            &(model->rs_symmetries[2][0][0]),model->orb_symmetries+8);
    curr_symm.angle = 270;
    diverge_generate_symm_trafo(n_spin,sites, 2,&curr_symm, 1,
                            &(model->rs_symmetries[3][0][0]),model->orb_symmetries+12);
    model->n_hop = 4;
    model->hop = (rs_hopping_t*)calloc(model->n_hop,sizeof(rs_hopping_t));
    model->hop[0].R[0] =  1;model->hop[0].o1 = 0;model->hop[0].o2 = 1;
    model->hop[1].R[0] = -1;model->hop[1].o1 = 0;model->hop[1].o2 = 1;
    model->hop[2].R[0] =  1;model->hop[2].o1 = 1;model->hop[2].o2 = 0;
    model->hop[3].R[0] = -1;model->hop[3].o1 = 1;model->hop[3].o2 = 0;
    model->hop[0].t = 1.0;
    model->hop[1].t = 1.0;
    model->hop[2].t = 1.0;
    model->hop[3].t = 1.0;
    diverge_model_internals_common( model );

    diverge_model_t* model2 = diverge_model_init();
    model2->nk[0] = 6, model2->nk[1] = 6;
    model2->nkf[0] = 1, model2->nkf[1] = 1;
    model2->n_orb = 2;
    model2->SU2 = true;
    model2->lattice[0][0] = 1., model2->lattice[0][1] = 0.0, model2->lattice[0][2] = 0.0;
    model2->lattice[1][0] = 0.0, model2->lattice[1][1] = 1., model2->lattice[1][2] = 0.0;
    model2->lattice[2][0] = 0.0, model2->lattice[2][1] = 0.0, model2->lattice[2][2] = 1.0;

    model2->orb_symmetries = (complex128_t*)calloc(4*4,sizeof(complex128_t));

    curr_symm.type = 'E'; curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 0.;curr_symm.normal_vector[2] = 1.;
    model2->n_sym = 4;
    diverge_generate_symm_trafo(n_spin,sites, 2,&curr_symm, 1,
                            &(model2->rs_symmetries[0][0][0]),model2->orb_symmetries);
    curr_symm.type = 'R';
    curr_symm.angle = 90;
    diverge_generate_symm_trafo(n_spin,sites, 2,&curr_symm, 1,
                            &(model2->rs_symmetries[1][0][0]),model2->orb_symmetries+4);
    curr_symm.angle = 180;
    diverge_generate_symm_trafo(n_spin,sites, 2,&curr_symm, 1,
                            &(model2->rs_symmetries[2][0][0]),model2->orb_symmetries+8);
    curr_symm.angle = 270;
    diverge_generate_symm_trafo(n_spin,sites, 2,&curr_symm, 1,
                            &(model2->rs_symmetries[3][0][0]),model2->orb_symmetries+12);
    model2->n_hop = 4;
    model2->hop = (rs_hopping_t*)calloc(model2->n_hop,sizeof(rs_hopping_t));
    model2->hop[0].R[1] =  1;model2->hop[0].o1 = 0;model2->hop[0].o2 = 1;
    model2->hop[1].R[1] = -1;model2->hop[1].o1 = 0;model2->hop[1].o2 = 1;
    model2->hop[2].R[1] =  1;model2->hop[2].o1 = 1;model2->hop[2].o2 = 0;
    model2->hop[3].R[1] = -1;model2->hop[3].o1 = 1;model2->hop[3].o2 = 0;
    model2->hop[0].t = -1.;
    model2->hop[1].t = -1.;
    model2->hop[2].t = -1.;
    model2->hop[3].t = -1.;
    diverge_model_internals_common( model2 );

    if (shared_exclusive_enter( model->internals->ham ) == 1)
        diverge_symmetrize_2pt_fine( model, model->internals->ham, NULL );
    shared_exclusive_wait( model->internals->ham );

    if (shared_exclusive_enter( model2->internals->ham ) == 1)
        diverge_symmetrize_2pt_coarse( model2, model2->internals->ham, NULL );
    shared_exclusive_wait( model2->internals->ham );

    for(index_t i = 0; i < 6*6*4; ++i) {
        CT_CHECK(std::abs(model->internals->ham[i] - model2->internals->ham[i]) < 1e-10);
    }
    diverge_model_free(model);
    diverge_model_free(model2);
}

#endif // DIVERGE_SKIP_TESTS
