#include "phonons.h"
#include "../diverge_Eigen3.hpp"

typedef Eigen::Matrix<complex128_t,-1,-1,Eigen::RowMajor> cMatXcd;

void phonon_gemm( const complex128_t* g_, const complex128_t* D_,
        complex128_t* V_, complex128_t* tmp_, index_t linsz, index_t nO ) {
    Map<const cMatXcd> g(g_, linsz, nO),
                       D(D_, nO, nO);
    Map<cMatXcd> V(V_, linsz, linsz),
                 tmp(tmp_, linsz, nO);
    tmp = g * D;
    V = tmp * g.adjoint();
}

