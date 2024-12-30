#ifndef DIVERGE_SKIP_TESTS
#ifndef DIVERGE_SKIP_TESTS_LINALG

#include "t_linalg_cla_solvers.hpp"

static void eigensolve_Xd( const MatXd& H, MatXd& U, VecXd& E ) {
    Eigen::SelfAdjointEigenSolver<MatXd> S(H);
    U = S.eigenvectors();
    E = S.eigenvalues();
}

static void eigensolve_Xcd( const MatXcd& H, MatXcd& U, VecXd& E ) {
    Eigen::SelfAdjointEigenSolver<MatXcd> S(H);
    U = S.eigenvectors();
    E = S.eigenvalues();
}

#define EIGENSOLVE_IMPL( dim, typ ) \
EigenSol##dim##typ eigensolve_##dim##typ( CMat##dim##typ H ) { \
    MatX##typ HX = H; \
    MatX##typ UX; \
    VecXd EX; \
    eigensolve_X##typ( HX, UX, EX ); \
    CMat##dim##typ U = UX; \
    Vec##dim##d E = EX; \
    return { U, E }; \
}

EIGENSOLVE_IMPL( 2, d )
EIGENSOLVE_IMPL( 3, d )
EIGENSOLVE_IMPL( 4, d )
EIGENSOLVE_IMPL( 2, cd )
EIGENSOLVE_IMPL( 3, cd )
EIGENSOLVE_IMPL( 4, cd )

static void svd_Xd( const MatXd& H, VecXd& E ) {
    auto S = H.bdcSvd();
    E = S.singularValues();
}

static void svd_Xcd( const MatXcd& H, VecXd& E ) {
    auto S = H.bdcSvd();
    E = S.singularValues();
}


#define SVD_IMPL( dim, typ ) \
Vec##dim##d svd_##dim##typ( CMat##dim##typ H ) { \
    MatX##typ HX = H; \
    VecXd EX; \
    svd_X##typ( HX, EX ); \
    Vec##dim##d E = EX; \
    return E; \
}

SVD_IMPL( 2, d )
SVD_IMPL( 3, d )
SVD_IMPL( 4, d )
SVD_IMPL( 2, cd )
SVD_IMPL( 3, cd )
SVD_IMPL( 4, cd )

#endif // DIVERGE_SKIP_TESTS_LINALG
#endif // DIVERGE_SKIP_TESTS
