/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "mpi_functions.h"
#include "mergesort.h"
#include "hartree_fock.h"
#include "batched_eigen.h"
#include "../diverge_model.h"
#include "../diverge_flow_step.h"
#include "../diverge_internals_struct.h"
#include "../diverge_model_internals.h"
#include "../diverge_hack.h"

struct diverge_hartree_fock_t {
    double nu;
    index_t nk;
    index_t nb;

    double* E;
    complex128_t* U;
    const complex128_t* H;

    complex128_t* flow_step_sigma;
    complex128_t* internal_sigma;

    void* buf;

    diverge_flow_step_t* step;
};

static inline double fermi( double E, double T ) {
    return 1./(exp(E/T) + 1.);
}
MERGESORT_IMPL( dblsort, double )

static greensfunc_op_t hf_data_gfill( const diverge_model_t* m, complex128_t Lambda, gf_complex_t* buf );

diverge_hartree_fock_t* diverge_hartree_fock_init( diverge_model_t* model, double nu, const char* chans ) {
    diverge_hartree_fock_t* m = calloc(1, sizeof*m);

    m->nu = nu;
    m->nk = kdimtot(model->nk, model->nkf);
    m->nb = model->n_orb * model->n_spin;

    m->E = diverge_model_internals_get_E(model);
    m->U = diverge_model_internals_get_U(model);
    m->H = diverge_model_internals_get_H(model);

    m->internal_sigma = calloc( m->nk * POW2(m->nb), sizeof*m->internal_sigma );

    m->buf = calloc( m->nk * POW2(m->nb), sizeof(complex128_t) );

    // we force the user to use these two
    diverge_model_hack( model, "tu_gf_skip_selfenergy", "1" );
    diverge_model_hack( model, "tu_selfenergy_flow", "1" );

    char xchans[1024] = {0};
    strncpy( xchans, chans, 1022 );
    strcat( xchans, "S" );
    m->step = diverge_flow_step_init( model, "tu", xchans );
    m->flow_step_sigma = diverge_flow_step_vertex( m->step, 'S' ).ary;

    // add the HF pointer to the model
    model->internals->hf_data = m;

    // and set the GF generator
    model->gfill = &hf_data_gfill;

    return m;
}

void diverge_hartree_fock_free( diverge_hartree_fock_t* m ) {
    diverge_flow_step_free( m->step );
    free( m->internal_sigma );
    free( m->buf );
    free( m );
}

void diverge_hartree_fock_exec( diverge_hartree_fock_t* m, gf_complex_t* dest, complex128_t Lambda ) {
    const index_t nk_nb2 = m->nk * POW2(m->nb);
    complex128_t* cbuf = m->buf;
    double* dbuf = m->buf;

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk_nb2; ++k) cbuf[k] = m->H[k] + m->flow_step_sigma[k];

    // eigensolve
    batched_eigen_r( NULL, -2, cbuf, m->E, m->nb, m->nk );

    // copy eigenvectors over
    memcpy( m->U, cbuf, sizeof*m->U * nk_nb2 );

    // copy eigenvalues in order to sort them and find the chempot corresponding
    // to the requested filling
    memcpy( dbuf, m->E, sizeof*m->E * m->nk*m->nb );
    dblsort( dbuf, m->nk*m->nb, diverge_omp_num_threads(), 0 );

    double which_E_dbl = (double)(m->nk*m->nb) * m->nu - 0.5;
    index_t which_E_start = (index_t)(which_E_dbl);
    double rest = which_E_dbl - which_E_start;
    index_t which_E_stop = which_E_start + 1;

    double mu;
    if (which_E_start >= (m->nk*m->nb-1)) {
        mu = dbuf[m->nk*m->nb-1];
    } else if (which_E_stop <= 0) {
        mu = dbuf[0];
    } else {
        mu = dbuf[which_E_start] + (dbuf[which_E_stop] - dbuf[which_E_start]) * rest;
    }
    // apply the filling
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<m->nk*m->nb; ++i) m->E[i] -= mu;

    // calculate the Fermi function and put it into dbuf
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<m->nk*m->nb; ++i) dbuf[i] = fermi(m->E[i], cabs(Lambda));

    // calculate the 'orbital space' Fermi function (code from default gfill)
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<m->nk; ++k)
    for (index_t o=0; o<m->nb; ++o)
    for (index_t p=0; p<m->nb; ++p) {
        complex128_t tmp = 0.0;
        for (index_t b=0; b<m->nb; ++b)
            tmp += m->U[IDX3(k,b,o,m->nb,m->nb)] * conj(m->U[IDX3(k,b,p,m->nb,m->nb)]) * dbuf[IDX2(k,b,m->nb)];
        dest[IDX3(k,o,p, m->nb,m->nb)] = tmp;
    }

    // and copy it over to the -Lambda GF
    memcpy( dest + nk_nb2, dest, sizeof*dest * nk_nb2 );
}

static greensfunc_op_t hf_data_gfill( const diverge_model_t* m, complex128_t Lambda, gf_complex_t* buf ) {
    diverge_hartree_fock_exec( m->internals->hf_data, buf, Lambda );
    return greensfunc_op_cpu;
}

void diverge_hartree_fock_step( diverge_hartree_fock_t* m, double T ) {
    const index_t nk_nb2 = m->nk * POW2(m->nb);
    memcpy( m->internal_sigma, m->flow_step_sigma, sizeof*m->flow_step_sigma * nk_nb2 );
    diverge_flow_step_euler( m->step, T, -1.0 );
    for (index_t i=0; i<nk_nb2; ++i) m->flow_step_sigma[i] -= m->internal_sigma[i];
}

complex128_t* diverge_hartree_fock_get_sigma( diverge_hartree_fock_t* h ) {
    return h->flow_step_sigma;
}

complex128_t* diverge_hartree_fock_get_sigma_prev( diverge_hartree_fock_t* h ) {
    return h->internal_sigma;
}
