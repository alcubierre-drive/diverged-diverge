/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_svd.h"
#include "batched_svd_private.h"
#include "../diverge_Eigen3.hpp"

typedef Eigen::SelfAdjointEigenSolver<MatXcd> Eigen_t;
typedef Eigen::JacobiSVD<MatXcd> SVD_t;

void* batched_svd_cpu( void* data ) {
    batched_svd_cfg_t* cfg = (batched_svd_cfg_t*)data;
    MatXcd A_help = MatXcd::Zero( cfg->dim, cfg->dim );
    void* solver = NULL;
    if (cfg->do_eigen) {
        Eigen_t* sol = new Eigen_t( A_help );
        solver = (void*)sol;
    } else {
        SVD_t* sol = new SVD_t( A_help,
            Eigen::DecompositionOptions::ComputeFullU |
            Eigen::DecompositionOptions::ComputeFullV );
        solver = (void*)sol;
    }
    SVD_t* svd = (SVD_t*)solver;
    Eigen_t* eig = (Eigen_t*)solver;

    for (index_t i=0; i<cfg->cnt; ++i) {
        Map<MatXcd> A(cfg->A + POW2(cfg->dim)*i, cfg->dim, cfg->dim);

        Map<VecXd> S(cfg->S + cfg->dim*i, cfg->dim);

        if (cfg->do_eigen) eig->compute( A.transpose() );
        else               svd->compute( A.transpose() ); // Row/Column

        if (cfg->U != NULL) {
            Map<MatXcd> U(cfg->U + POW2(cfg->dim)*i, cfg->dim, cfg->dim);
            if (!cfg->do_eigen)
                U = svd->matrixU().transpose();
            else
                U = eig->eigenvectors(); // no transpose here to be consistent
        }
        if (cfg->V != NULL) {
            Map<MatXcd> V(cfg->V + POW2(cfg->dim)*i, cfg->dim, cfg->dim);
            if (!cfg->do_eigen)
                V = svd->matrixV();
        }
        S = cfg->do_eigen ? eig->eigenvalues() : svd->singularValues();
    }

    if (cfg->do_eigen)
        delete eig;
    else
        delete svd;

    return NULL;
}
