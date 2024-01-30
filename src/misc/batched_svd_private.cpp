/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "batched_svd.h"
#include "batched_svd_private.h"
#include "../diverge_Eigen3.hpp"

void* batched_svd_cpu( void* data ) {
    batched_svd_cfg_t* cfg = (batched_svd_cfg_t*)data;
    MatXcd A_help = MatXcd::Zero( cfg->dim, cfg->dim );
    Eigen::JacobiSVD<MatXcd> SVD( A_help,
            Eigen::DecompositionOptions::ComputeFullU |
            Eigen::DecompositionOptions::ComputeFullV );

    for (index_t i=0; i<cfg->cnt; ++i) {
        Map<MatXcd> A(cfg->A + POW2(cfg->dim)*i, cfg->dim, cfg->dim);

        Map<VecXd> S(cfg->S + cfg->dim*i, cfg->dim);
        SVD.compute( A.transpose() ); // Row/Column
        if (cfg->U != NULL) {
            Map<MatXcd> U(cfg->U + POW2(cfg->dim)*i, cfg->dim, cfg->dim);
            U = SVD.matrixU().transpose();
        }
        if (cfg->V != NULL) {
            Map<MatXcd> V(cfg->V + POW2(cfg->dim)*i, cfg->dim, cfg->dim);
            V = SVD.matrixV();
        }
        S = SVD.singularValues();
    }
    return NULL;
}
