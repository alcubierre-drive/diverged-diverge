#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../helper_functions.hpp"

#include "../../src/diverge.h"
#include "../../src/misc/subspace_gf.h"

TEST_CASE("greensfunction subspace full", "[greens][subspace-full]") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;

    diverge_model_t* m = gen_kagome_model(6,6);
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

    CHECK( check_equal(xbuf, gbuf, kdimtot(m->nk,m->nkf)*9) );

    free(xbuf);
    diverge_model_free(m);
}

TEST_CASE("greensfunction subspace partial", "[greens][subspace]") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;

    diverge_model_t* m = gen_kagome_model(6,6);
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
    for (index_t k=0; k<kdimtot(m->nk,m->nkf); ++k)
    for (index_t io1=0; io1<2; ++io1)
    for (index_t io2=0; io2<2; ++io2)
        maxerr = MAX(maxerr, std::abs(xbuf[IDX3(k,io1,io2,2,2)] - gbuf[IDX3(k,orbs[io1],orbs[io2],3,3)]));

    CHECK( maxerr < 1e-12 );

    free(xbuf);
    diverge_model_free(m);
}

#endif
