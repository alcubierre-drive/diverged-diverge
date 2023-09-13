/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_filling.h"
#include "diverge_internals_struct.h"
#include <algorithm>

double diverge_model_get_filling( diverge_model_t* model, const double* E, index_t nb ) {
    index_t nk = kdimtot(model->nk, model->nkf);
    if (E == NULL) {
        nb = model->n_orb * model->n_spin;
        E = model->internals->E;
    }
    index_t zero_E = 0;
    #pragma omp parallel for reduction(+:zero_E)
    for (index_t kb=0; kb<nk*nb; ++kb)
        zero_E += E[kb] < 0.0;
    return (double)zero_E / (double)(nk*nb);
}

void diverge_model_set_chempot( diverge_model_t* model, double* E, index_t nb, double mu ) {
    index_t nk = kdimtot(model->nk, model->nkf);
    if (E == NULL) {
        nb = model->n_orb * model->n_spin;
        E = model->internals->E;
    }
    #pragma omp parallel for
    for (index_t kb=0; kb<nk*nb; ++kb)
        E[kb] -= mu;
    model->internals->mu = mu;
}

double diverge_model_set_filling( diverge_model_t* model, double* E, index_t nb, double nu ) {
    index_t nk = kdimtot(model->nk, model->nkf);
    if (E == NULL) {
        nb = model->n_orb * model->n_spin;
        E = model->internals->E;
    }
    double* E_cpy = (double*)malloc(sizeof(double)*nk*nb);
    memcpy( E_cpy, E, sizeof(double)*nk*nb );
    std::sort( E_cpy, E_cpy+nk*nb );
    index_t idx = lround(nu * (double)(nk*nb));
    idx = MIN(idx,nk*nb);
    idx = MAX(idx,0);
    double mu = E_cpy[idx];
    diverge_model_set_chempot( model, E, nb, mu );
    free( E_cpy );
    return mu;
}
