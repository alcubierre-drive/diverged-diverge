#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME greens
#include "../ctest.h"

#include "../model_examples/model_constructor.h"
#include "../helper_functions.hpp"

#include "../../src/diverge.h"
#include "../../src/misc/subspace_gf.h"
#include "../../src/misc/largemat_gf.h"

CT_TEST(subspace_full) {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;

    diverge_model_t* m = C_gen_kagome_model(6,6);
    complex128_t* gbuf = (complex128_t*)diverge_model_internals_get_greens(m);
    const complex128_t Lambda(0.0,1.0);
    m->gfill(m, Lambda, (complex128_t*)gbuf);

    const index_t orbs[] = {0,1,2};
    subspace_gfill_multi_t* s = subspace_gfill_multi_init(
            diverge_model_internals_get_E(m),
            diverge_model_internals_get_U(m),
            kdimtot(m->nk, m->nkf), orbs, 3, 3, 1);
    complex128_t* xbuf = (complex128_t*)calloc(kdimtot(m->nk,m->nkf)*9*2, sizeof(*xbuf));
    subspace_gfill_multi_exec(s, xbuf, Lambda);
    subspace_gfill_multi_free(s);

    CT_CHECK( check_equal(xbuf, gbuf, kdimtot(m->nk,m->nkf)*9*2) );

    free(xbuf);
    diverge_model_free(m);
}

CT_TEST(subspace_partial) {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;

    diverge_model_t* m = C_gen_kagome_model(6,6);
    complex128_t* gbuf = (complex128_t*)diverge_model_internals_get_greens(m);
    const complex128_t Lambda(0.0,1.0);
    m->gfill(m, Lambda, (complex128_t*)gbuf);

    const index_t orbs[] = {2,0};
    subspace_gfill_multi_t* s = subspace_gfill_multi_init(
            diverge_model_internals_get_E(m),
            diverge_model_internals_get_U(m),
            kdimtot(m->nk, m->nkf), orbs, 2, 3, 1);
    complex128_t* xbuf = (complex128_t*)calloc(kdimtot(m->nk,m->nkf)*4*2, sizeof(*xbuf));
    subspace_gfill_multi_exec(s, xbuf, Lambda);
    subspace_gfill_multi_free(s);

    double maxerr = 0.0;
    index_t nk = kdimtot(m->nk,m->nkf);
    for (index_t l=0; l<2; ++l)
    for (index_t k=0; k<nk; ++k)
    for (index_t io1=0; io1<2; ++io1)
    for (index_t io2=0; io2<2; ++io2)
        maxerr = MAX(maxerr, std::abs(xbuf[IDX4(l,k,io1,io2,nk,2,2)] - gbuf[IDX4(l,k,orbs[io1],orbs[io2],nk,3,3)]));

    CT_CHECK( maxerr < 1e-12 );

    free(xbuf);
    diverge_model_free(m);
}

CT_TEST(largemat) {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;
    const complex128_t Lambda(0.0,1.0);

    diverge_model_t* m = C_gen_kagome_model(6,6);
    m->gfill(m, Lambda, diverge_model_internals_get_greens(m));

    diverge_model_t* l = C_gen_kagome_model(6,6);
    diverge_largemat_gf(l, Lambda, (gf_complex_t*)diverge_model_internals_get_greens(l));

    /*
     * CT_CHECK( check_equal(diverge_model_internals_get_greens(m),
     *                       diverge_model_internals_get_greens(l),
     *                       kdimtot(m->nk,m->nkf)*9*2) );
     */
    double maxerr = 0.0;
    complex128_t *gbuf = (complex128_t*)diverge_model_internals_get_greens(m),
                 *xbuf = (complex128_t*)diverge_model_internals_get_greens(l);
    index_t nk = kdimtot(m->nk,m->nkf);
    for (index_t l=0; l<2; ++l)
    for (index_t k=0; k<nk; ++k)
    for (index_t o1=0; o1<3; ++o1)
    for (index_t o2=0; o2<3; ++o2)
        maxerr = MAX(maxerr, std::abs(xbuf[IDX4(l,k,o1,o2,nk,3,3)] - gbuf[IDX4(l,k,o1,o2,nk,3,3)]));
    CT_CHECK( maxerr < 1e-12 );


    diverge_model_free(m);
    diverge_model_free(l);
}

#endif
