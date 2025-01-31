#include "invert.h"
#include "../diverge_Eigen3.hpp"

void matrix_invert( const double* A, index_t sz, double* Ainv ) {
    Map<cMatXd>(Ainv, sz, sz) = Map<const cMatXd>(A, sz, sz).inverse();
}

void cmatrix_invert( const complex128_t* A, index_t sz, complex128_t* Ainv ) {
    Map<cMatXcd>(Ainv, sz, sz) = Map<const cMatXcd>(A, sz, sz).inverse();
}

void cfmatrix_invert_F( const gf_complex_t* A, index_t sz, gf_complex_t* Ainv ) {
    typedef Eigen::Matrix<gf_complex_t,-1,-1> MatXcf;
    Map<MatXcf>(Ainv, sz, sz) = Map<const MatXcf>(A, sz, sz).inverse();
}
