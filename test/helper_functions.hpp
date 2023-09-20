#ifndef DIVERGE_SKIP_TESTS

#pragma once

#include "../src/diverge_common.h"
#include "../src/diverge_model.h"

bool check_hermicity(complex128_t* data, index_t linsize,index_t batches);
complex128_t trace(complex128_t* data, index_t linsize, index_t batches);
complex128_t trace_batch_lin(complex128_t* data, index_t linsize, index_t batches);
double trace(double* data, index_t linsize, index_t batches);
bool check_equal(complex128_t* data1, complex128_t* data2, index_t size, double errbnd = 1e-12);
bool check_equal(double* data1, double* data2, index_t size, double errbnd = 1e-12);
bool check_abs_sorted_equal(complex128_t* data1, complex128_t* data2, index_t size);
bool check_abs_sorted_equal(double* data1, double* data2, index_t size);


inline complex128_t ff(const double* k, const double* r){
    return exp(-I128*(k[0]*r[0]+k[1]*r[1]+k[2]*r[2]));
}

inline void fill_Rvec(const diverge_model_t* model, double* Rvec) {
    const tu_formfactor_t* tu_ff = model->tu_ff;
    for(index_t i = 0; i < model->n_tu_ff; ++i) {
        Rvec[3*i+0] = tu_ff[i].R[0] * model->lattice[0][0]
                            +tu_ff[i].R[1] * model->lattice[1][0]
                            +tu_ff[i].R[2] * model->lattice[2][0];
        Rvec[3*i+1] = tu_ff[i].R[0] * model->lattice[0][1]
                            +tu_ff[i].R[1] * model->lattice[1][1]
                            +tu_ff[i].R[2] * model->lattice[2][1];
        Rvec[3*i+2] = tu_ff[i].R[0] * model->lattice[0][2]
                            +tu_ff[i].R[1] * model->lattice[1][2]
                            +tu_ff[i].R[2] * model->lattice[2][2];
    }
}

inline void fill_Rvec_from_vec(const diverge_model_t* model, index_t* R, double* Rvec) {
    for(index_t i = 0; i < model->n_tu_ff; ++i) {
        Rvec[3*i+0] = R[3*i+0] * model->lattice[0][0]
                            +R[3*i+1] * model->lattice[1][0]
                            +R[3*i+2] * model->lattice[2][0];
        Rvec[3*i+1] = R[3*i+0] * model->lattice[0][1]
                            +R[3*i+1] * model->lattice[1][1]
                            +R[3*i+2] * model->lattice[2][1];
        Rvec[3*i+2] = R[3*i+0] * model->lattice[0][2]
                            +R[3*i+1] * model->lattice[1][2]
                            +R[3*i+2] * model->lattice[2][2];
    }
}

#endif // DIVERGE_SKIP_TESTS
