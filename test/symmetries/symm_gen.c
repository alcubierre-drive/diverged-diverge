#ifndef DIVERGE_SKIP_TESTS

#include "../helper_functions.h"
#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/misc/generate_symmetries.h"
#include "../../src/misc/linalg.h"
LINALG_IMPLEMENT

#define CT_SUITE_NAME symmetry_generator
#include "../ctest.h"

CT_TEST( SU2 ) {
    sym_op_t curr_symm[5] = {0};
    site_descr_t sites[3] = {0};
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;
    sites[0].n_functions = 1;

    sites[1].n_functions = 2;
    sites[2].n_functions = 2;

    sites[1].amplitude[0] = 1./sqrt(2.);
    sites[1].amplitude[1] = I128/sqrt(2.);

    sites[2].amplitude[0] = 1./sqrt(2.);
    sites[2].amplitude[1] = -I128/sqrt(2.);
    sites[1].function[0] = orb_px;
    sites[1].function[1] = orb_py;

    sites[2].function[0] = orb_px;
    sites[2].function[1] = orb_py;

    claM3d_t m_rs = {0};
    claM3cd_t m_orb = {0};

    // Mirror plane along y
    index_t len_sym = 1;
    curr_symm[0].type = 'M'; curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 0.;
    curr_symm[0].angle = 0;

    diverge_generate_symm_trafo( 1, sites, 3, curr_symm, len_sym, m_rs.m[0], m_orb.m[0] );

    claM3d_t test_rs = {{{-1.,0.,0.},
                         { 0.,1.,0.},
                         { 0.,0.,1.}}};
    claM3cd_t test_orb = {{{1., 0., 0.},
                           {0., 0.,-1.},
                           {0.,-1.0,0.}}};
    CT_CHECK(claM3d_norm(claM3d_sub(m_rs,test_rs)) < 1e-6);
    CT_CHECK(claM3cd_norm(claM3cd_sub(m_orb,test_orb)) < 1e-6);

    // 90 degree rotation around z
    len_sym = 1;
    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[0].angle = 90;

    diverge_generate_symm_trafo( 1, sites, 3, curr_symm, len_sym, m_rs.m[0], m_orb.m[0] );

    test_rs = (claM3d_t){{{0.,-1.,0.},
                          {1., 0.,0.},
                          {0., 0.,1.}}};
    test_orb = (claM3cd_t){{{1., 0.,0.},
                            {0.,-I, 0.},
                            {0., 0.,I}}};
    CT_CHECK(claM3d_norm(claM3d_sub(m_rs,test_rs)) < 1e-6);
    CT_CHECK(claM3cd_norm(claM3cd_sub(m_orb,test_orb)) < 1e-6);

    len_sym = 3;
    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[0].angle = -45;

    curr_symm[1].type = 'M'; curr_symm[1].normal_vector[0] = 1.;
    curr_symm[1].normal_vector[1] = 0.;curr_symm[1].normal_vector[2] = 0.;
    curr_symm[1].angle = 90;

    curr_symm[2].type = 'R'; curr_symm[2].normal_vector[0] = 0.;
    curr_symm[2].normal_vector[1] = 0.;curr_symm[2].normal_vector[2] = 1.;
    curr_symm[2].angle = 45;

    diverge_generate_symm_trafo( 1, sites, 3, curr_symm, len_sym, m_rs.m[0], m_orb.m[0] );

    test_rs = (claM3d_t){{{0.,1.,0.},
                          {1.,0.,0.},
                          {0.,0.,1.}}};
    test_orb = (claM3cd_t){{{1.,0., 0.},
                            {0.,0.,-I },
                            {0.,I , 0.}}};
    CT_CHECK(claM3d_norm(claM3d_sub(m_rs,test_rs)) < 1e-6);
    CT_CHECK(claM3cd_norm(claM3cd_sub(m_orb,test_orb)) < 1e-6);
}

CT_TEST( nonSU2 ) {
    sym_op_t curr_symm[5] = {0};
    site_descr_t sites[1] = {0};
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;
    sites[0].n_functions = 1;

    claM3d_t m_rs = {0};
    claM2cd_t m_orb = {0};
    // Mirror plane along y
    index_t len_sym = 1;
    curr_symm[0].type = 'M'; curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 0.;
    curr_symm[0].angle = 0;

    diverge_generate_symm_trafo( 2, sites, 1, curr_symm, len_sym, m_rs.m[0], m_orb.m[0] );

    claM2cd_t test_orb = {{{ 0.,-I },
                           {-I , 0.}}};
    CT_CHECK(claM2cd_norm(claM2cd_sub(m_orb,test_orb)) < 1e-6);

    // rotation 90 z
    len_sym = 1;
    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[0].angle = 90;

    diverge_generate_symm_trafo( 2, sites, 1, curr_symm, len_sym, m_rs.m[0], m_orb.m[0] );

    test_orb = (claM2cd_t){{{(1.-I128)/sqrt(2),              0.0},
                            {              0.0,(1.+I128)/sqrt(2)}}};
    CT_CHECK(claM2cd_norm(claM2cd_sub(m_orb,test_orb)) < 1e-6);

    // mirror 45
    len_sym = 3;
    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[0].angle = -45;

    curr_symm[1].type = 'M'; curr_symm[1].normal_vector[0] = 1.;
    curr_symm[1].normal_vector[1] = 0.;curr_symm[1].normal_vector[2] = 0.;
    curr_symm[1].angle = 0;

    curr_symm[2].type = 'R'; curr_symm[2].normal_vector[0] = 0.;
    curr_symm[2].normal_vector[1] = 0.;curr_symm[2].normal_vector[2] = 1.;
    curr_symm[2].angle = 45;

    diverge_generate_symm_trafo( 2, sites, 1, curr_symm, len_sym, m_rs.m[0], m_orb.m[0] );

    test_orb = (claM2cd_t){{{               0.0,(1.-I128)/sqrt(2)},
                            {(-1.-I128)/sqrt(2),              0.0}}};
    CT_CHECK(claM2cd_norm(claM2cd_sub(m_orb,test_orb)) < 1e-6);


    len_sym = 1;
    curr_symm[0].type = 'M'; curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = -1.;curr_symm[0].normal_vector[2] = 0.;
    curr_symm[0].angle = 0;

    diverge_generate_symm_trafo( 2, sites, 1, curr_symm, len_sym, m_rs.m[0], m_orb.m[0] );

    test_orb = (claM2cd_t){{{               0.0,(1.-I128)/sqrt(2)},
                            {(-1.-I128)/sqrt(2),              0.0}}};
    CT_CHECK(claM2cd_norm(claM2cd_sub(m_orb,test_orb)) < 1e-6);
}

#endif // DIVERGE_SKIP_TESTS
