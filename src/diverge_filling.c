/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_filling.h"
#include "diverge_internals_struct.h"
#include "misc/mpi_functions.h"
#include "misc/shared_mem.h"
#include "misc/mergesort.h"

double diverge_model_get_filling( diverge_model_t* model, const double* E, index_t nb ) {
    index_t nk = kdimtot(model->nk, model->nkf);
    if (E == NULL) {
        nb = model->n_orb * model->n_spin;
        E = model->internals->E;
    }
    index_t zero_E = 0;
    #pragma omp parallel for reduction(+:zero_E) num_threads(diverge_omp_num_threads())
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
    // this should cover all cases (custom E and internal E)
    int shared = shared_exclusive_enter(E);
    if (shared == -1 || shared == 1) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t kb=0; kb<nk*nb; ++kb)
            E[kb] -= mu;
    }
    if (shared != -1) shared_exclusive_wait(E);
    model->internals->mu = mu;
}

static inline int compare_dbl( const void* a_, const void* b_ ) {
    const double *a = a_, *b = b_;
    if (*a < *b) return -1;
    else if (*a > *b) return 1;
    else return 0;
}

MERGESORT_IMPL(dblsort, double)
double diverge_model_set_filling( diverge_model_t* model, double* E, index_t nb, double nu ) {
    index_t nk = kdimtot(model->nk, model->nkf);
    if (E == NULL) {
        nb = model->n_orb * model->n_spin;
        E = model->internals->E;
    }
    double* E_cpy = (double*)malloc(sizeof(double)*nk*nb);
    memcpy( E_cpy, E, sizeof(double)*nk*nb );
    dblsort( E_cpy, nk*nb, diverge_omp_num_threads(), 0 );

    double idx_frac = nu * (double)(nk*nb) - 0.5;
    index_t idx_0 = (index_t)idx_frac;
    index_t idx_1 = idx_0 + 1;

    double mu;
    if (idx_1 <= 0) {
        mu = E_cpy[0];
    } else if (idx_0 >= (nk*nb-1)) {
        mu = E_cpy[nk*nb-1];
    } else {
        mu = E_cpy[idx_0] + (E_cpy[idx_1] - E_cpy[idx_0]) * (idx_frac - idx_0);
    }

    diverge_model_set_chempot( model, E, nb, mu );
    free( E_cpy );
    return mu;
}
