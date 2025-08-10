#define CT_SUITE_NAME hamilton
#include "ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../src/diverge.h"
#include "models/all.h"

static inline index_t array_equal( const complex128_t* A, const complex128_t* B, index_t sz ) {
    index_t nequal = 0;
    for (index_t i=0; i<sz; ++i) nequal += (cabs(A[i]-B[i]) < 1.e-9);
    return nequal == sz;
}

CT_TEST(generator_fine) {
    diverge_model_t* m = KWARGS(gen_kagome_model, .nkx=12, .nky=12);

    double* kfmesh = diverge_model_internals_get_kfmesh( m );
    complex128_t* H_buf = diverge_model_internals_get_H( m );

    complex128_t* H_kpt = (complex128_t*)calloc(POW2(m->n_orb*m->n_spin), sizeof*H_kpt);
    complex128_t* H_idx = (complex128_t*)calloc(POW2(m->n_orb*m->n_spin), sizeof*H_idx);

    index_t nsame_idx = 0, nsame_kpt = 0, ntot = 0;
    for (index_t k=0; k<kdimtot(m->nk,m->nkf); ++k) {
        diverge_model_ham_at_kpt( m, kfmesh+3*k, H_kpt );
        diverge_model_ham_at_kfidx( m, k, H_idx );
        complex128_t* H_check = H_buf + k*POW2(m->n_orb*m->n_spin);
        nsame_idx += array_equal( H_check, H_idx, POW2(m->n_orb*m->n_spin) );
        nsame_kpt += array_equal( H_check, H_kpt, POW2(m->n_orb*m->n_spin) );
        ntot++;
    }

    CT_CHECK( nsame_idx == ntot );
    CT_CHECK( nsame_kpt == ntot );

    free( H_kpt );
    free( H_idx );

    diverge_model_free( m );
}

CT_TEST(generator_coarse) {
    diverge_model_t* m = KWARGS(gen_kagome_model, .nkx=12, .nky=12);

    double* kmesh = diverge_model_internals_get_kmesh( m );
    complex128_t* H_buf = diverge_model_internals_get_H( m );

    complex128_t* H_kpt = (complex128_t*)calloc(POW2(m->n_orb*m->n_spin), sizeof*H_kpt);
    complex128_t* H_idx = (complex128_t*)calloc(POW2(m->n_orb*m->n_spin), sizeof*H_idx);

    index_t nsame_idx = 0, nsame_kpt = 0, ntot = 0;
    for (index_t k=0; k<kdim(m->nk); ++k) {
        diverge_model_ham_at_kpt( m, kmesh+3*k, H_kpt );
        diverge_model_ham_at_kcidx( m, k, H_idx );
        complex128_t* H_check = H_buf + kidxc2f(k,m->nk,m->nkf)*POW2(m->n_orb*m->n_spin);
        nsame_idx += array_equal( H_check, H_idx, POW2(m->n_orb*m->n_spin) );
        nsame_kpt += array_equal( H_check, H_kpt, POW2(m->n_orb*m->n_spin) );
        ntot++;
    }

    CT_CHECK( nsame_idx == ntot );
    CT_CHECK( nsame_kpt == ntot );

    free( H_kpt );
    free( H_idx );

    diverge_model_free( m );
}

#endif
