#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
// #include "../catch.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/misc/generate_symmetries.h"
#include "../../src/diverge_Eigen3.hpp"

#define CT_SUITE_NAME symmetry_generator
#include "../ctest.h"

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;
typedef Eigen::Matrix<complex128_t,-1,-1,Eigen::RowMajor> CMatXcd;

CT_TEST( SU2 ) {
    sym_op_t curr_symm[5];
    site_descr_t sites[3];
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

    index_t n_spin = 1;
    double* rst = (double*)malloc(3*3*sizeof(double));
    complex128_t* orb_trafo = (complex128_t*)malloc(3*3*n_spin*n_spin
                                                        *sizeof(complex128_t));

    Map<CMat3d> m_rs (rst,3,3);
    Map<CMatXcd> m_orb(orb_trafo,3,3);

    // Mirror plane along y
    index_t len_sym = 1;
    curr_symm[0].type = 'M'; curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 0.;
    curr_symm[0].angle = 0;

    diverge_generate_symm_trafo( n_spin, sites, 3, curr_symm, len_sym, rst, orb_trafo );

    Mat3d test_rs; test_rs << -1.,0.,0.,
                              0.,1.,0.,
                              0.,0.,1.0;
    Mat3cd test_orb; test_orb << 1.0,0.0,0.0,
                                 0.0,0.0,-1.0,
                                 0.0,-1.0,0.0;
    CT_CHECK((m_rs-test_rs).lpNorm<Eigen::Infinity>() < 1e-10);
    CT_CHECK((m_orb-test_orb).lpNorm<Eigen::Infinity>() < 1e-10);


    // 90 degree rotation around z
    len_sym = 1;
    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[0].angle = 90;

    diverge_generate_symm_trafo( n_spin, sites, 3, curr_symm, len_sym, rst, orb_trafo );

    test_rs << 0.,-1.,0.,
               1.,0.,0.,
               0.,0.,1.0;
    test_orb << 1.0,0.0,0.0,
                0.0,-I128,0.0,
                0.0,0.0,I128;
    CT_CHECK((m_rs-test_rs).lpNorm<Eigen::Infinity>() < 1e-10);
    CT_CHECK((m_orb-test_orb).lpNorm<Eigen::Infinity>() < 1e-10);

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

    diverge_generate_symm_trafo( n_spin, sites, 3, curr_symm, len_sym, rst, orb_trafo );

    test_rs << 0.,1.,0.,
               1.,0.,0.,
               0.,0.,1.0;
    test_orb << 1.0,0.0,0.0,
                0.0,0.,-I128,
                0.0,I128,0.0;
    CT_CHECK((m_rs-test_rs).lpNorm<Eigen::Infinity>() < 1e-10);
    CT_CHECK((m_orb-test_orb).lpNorm<Eigen::Infinity>() < 1e-10);


    free(rst);
    free(orb_trafo);
}

CT_TEST( nonSU2 ) {
    sym_op_t curr_symm[5];
    site_descr_t sites[1];
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;
    sites[0].n_functions = 1;

    index_t n_spin = 2;
    complex128_t* orb_trafo = (complex128_t*)malloc(n_spin*n_spin
                                                        *sizeof(complex128_t));
    double* rst = (double*)malloc(3*3*sizeof(double));
    Map<CMatXcd> m_orb(orb_trafo,2,2);
    // Mirror plane along y
    index_t len_sym = 1;
    curr_symm[0].type = 'M'; curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 0.;
    curr_symm[0].angle = 0;

    diverge_generate_symm_trafo( n_spin, sites, 1, curr_symm, len_sym, rst, orb_trafo );

    Mat2cd test_orb; test_orb << 0.0,-I128,
                                 -I128,0.0;
    CT_CHECK((m_orb-test_orb).lpNorm<Eigen::Infinity>() < 1e-10);

    // rotation 90 z
    len_sym = 1;
    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[0].angle = 90;

    diverge_generate_symm_trafo( n_spin, sites, 1, curr_symm, len_sym, rst, orb_trafo );

    test_orb << (1.-I128)/sqrt(2),0.0,
                0.0,(1.+I128)/sqrt(2);
    CT_CHECK((m_orb-test_orb).lpNorm<Eigen::Infinity>() < 1e-10);


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

    diverge_generate_symm_trafo( n_spin, sites, 1, curr_symm, len_sym, rst, orb_trafo );

    test_orb << 0.0,(1.-I128)/sqrt(2),
                (-1.-I128)/sqrt(2),0.0;
    CT_CHECK((m_orb-test_orb).lpNorm<Eigen::Infinity>() < 1e-10);


    len_sym = 1;
    curr_symm[0].type = 'M'; curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = -1.;curr_symm[0].normal_vector[2] = 0.;
    curr_symm[0].angle = 0;

    diverge_generate_symm_trafo( n_spin, sites, 1, curr_symm, len_sym, rst, orb_trafo );

    test_orb << 0.0,(1.-I128)/sqrt(2),
                (-1.-I128)/sqrt(2),0.0;
    CT_CHECK((m_orb-test_orb).lpNorm<Eigen::Infinity>() < 1e-10);

    free(rst);
    free(orb_trafo);
}

#endif // DIVERGE_SKIP_TESTS
