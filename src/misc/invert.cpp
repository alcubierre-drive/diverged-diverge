#include "invert.h"
#include "../diverge_Eigen3.hpp"

typedef Eigen::Matrix<complex128_t,-1,-1,Eigen::RowMajor> CMatXcd;
typedef Eigen::Matrix<gf_complex_t,-1,-1,Eigen::RowMajor> CMatXcf;
typedef Eigen::Matrix<double,-1,-1,Eigen::RowMajor> CMatXd;

void matrix_invert( const double* A, index_t sz, double* Ainv ) {
    Map<const CMatXd> mA(A, sz, sz);
    Map<CMatXd>(Ainv, sz, sz) = mA.inverse();
}

void cmatrix_invert( const complex128_t* A, index_t sz, complex128_t* Ainv ) {
    Map<const CMatXcd> mA(A, sz, sz);
    Map<CMatXcd>(Ainv, sz, sz) = mA.inverse();
}

void cfmatrix_invert( const gf_complex_t* A, index_t sz, gf_complex_t* Ainv ) {
    Map<const CMatXcf> mA(A, sz, sz);
    Map<CMatXcf>(Ainv, sz, sz) = mA.inverse();
}
