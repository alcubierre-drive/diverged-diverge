#define CT_SUITE_NAME tu_prop_katanin
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_hack.h"
#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/propagator_details.h"
#include "../../src/tu/projection.h"
#include "../../src/tu/vertex.h"
#include "../../src/tu/vertex_details.h"
#include "../../src/tu/flow_stepper.h"
#include "../../src/tu/flow_stepper_details.h"
#include "../helper_functions.h"
#include "../models/all.h"

#define KAT_LAMBDA 0.3
#define KAT_MU (-0.4)

static diverge_model_t* kat_model( double mu, bool katanin ) {
    diverge_model_t* m = KWARGS(gen_square_lat_hub, .nkx=6, .nky=6, .lx=1, .ly=1, .mu=mu);
    diverge_model_internals_tu(m, 1.1);
    if (katanin) diverge_model_hack(m, "tu_katanin", "1");
    return m;
}

static index_t kat_loopsize( const tu_prop_t* t ) {
    return t->my_nk * POW2(t->n_orbff*POW2(t->n_spin));
}

static complex128_t* kat_const_self( const diverge_model_t* m, double val ) {
    const index_t nb = m->n_orb*m->n_spin;
    const index_t nktot = kdimtot(m->nk, m->nkf);
    complex128_t* self = calloc( nktot*POW2(nb), sizeof*self );
    for (index_t k=0; k<nktot; ++k)
    for (index_t b=0; b<nb; ++b)
        self[IDX2(k, b+nb*b, POW2(nb))] = val;
    return self;
}

CT_TEST_NAME(tu_prop_katanin_off, "tu_prop,katanin:off") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return; // no GF floats

    diverge_model_t* m = kat_model( KAT_MU, false );
    tu_prop_t* t = tu_prop_init(m);

    CT_CHECK( !tu_prop_katanin(t) );
    CT_CHECK( t->GF_fft_plus_kat == t->GF_fft_plus );
    CT_CHECK( t->GF_fft_conj_plus_kat == t->GF_fft_conj_plus );

    // the swap must not move anything
    complex128_t* before = t->GF_fft_plus;
    tu_prop_katanin_swap(t);
    CT_CHECK( t->GF_fft_plus == before );

    tu_prop_free(t);
    diverge_model_free(m);
}

CT_TEST_NAME(tu_prop_katanin_same, "tu_prop,katanin:same") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return; // no GF floats

    diverge_model_t* m = kat_model( KAT_MU, true );
    tu_prop_t* t = tu_prop_init(m);
    CT_CHECK( tu_prop_katanin(t) );

    complex128_t* self = kat_const_self( m, 0.17 );
    const index_t sz = kat_loopsize(t);
    complex128_t* Lplain = calloc(sz, sizeof*Lplain);
    complex128_t* Lkat = calloc(sz, sizeof*Lkat);

    tu_prop_GF_zero_T( t, I128*KAT_LAMBDA, self );
    tu_prop_GF_zero_T_kat( t, I128*KAT_LAMBDA, self );

    for (int pp=0; pp<2; ++pp) {
        void (*loop)(tu_prop_t*, complex128_t*, complex128_t*) =
            pp ? &tu_prop_pp_loop : &tu_prop_ph_loop;
        (*loop)( t, Lplain, NULL );
        tu_prop_katanin_swap(t);
        (*loop)( t, Lkat, NULL );
        tu_prop_katanin_swap(t);
        CT_CHECK( check_equal_cd( Lplain, Lkat, sz, 0.0 ) );
    }

    // swapping twice must restore the plain buffers
    complex128_t* before = t->GF_fft_plus;
    tu_prop_katanin_swap(t);
    tu_prop_katanin_swap(t);
    CT_CHECK( t->GF_fft_plus == before );

    free(Lplain); free(Lkat); free(self);
    tu_prop_free(t);
    diverge_model_free(m);
}

// \Sigma = \delta \cdot 1, s.t., [G^{-1} - \delta]^{-1} = G with mu -> mu + \delta.
// Checks whether this holds
CT_TEST_NAME(tu_prop_katanin_mushift, "tu_prop,katanin:mushift") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return; // no GF floats

    const double delta = 0.11;

    diverge_model_t* mA = kat_model( KAT_MU, true );
    diverge_model_t* mB = kat_model( KAT_MU + delta, false );
    tu_prop_t* tA = tu_prop_init(mA);
    tu_prop_t* tB = tu_prop_init(mB);
    CT_CHECK( tu_prop_katanin(tA) );
    CT_CHECK( tA->n_orbff == tB->n_orbff );

    complex128_t* self = kat_const_self( mA, delta );
    const index_t sz = kat_loopsize(tA);
    complex128_t* LA = calloc(sz, sizeof*LA);
    complex128_t* LB = calloc(sz, sizeof*LB);

    tu_prop_GF_zero_T( tA, I128*KAT_LAMBDA, NULL );      // plain: G
    tu_prop_GF_zero_T_kat( tA, I128*KAT_LAMBDA, self );  // kat:   G_eps
    tu_prop_GF_zero_T( tB, I128*KAT_LAMBDA, NULL );      // reference

    for (int pp=0; pp<2; ++pp) {
        void (*loop)(tu_prop_t*, complex128_t*, complex128_t*) =
            pp ? &tu_prop_pp_loop : &tu_prop_ph_loop;
        tu_prop_katanin_swap(tA);
        (*loop)( tA, LA, NULL );
        tu_prop_katanin_swap(tA);
        (*loop)( tB, LB, NULL );
        CT_CHECK( check_equal_cd( LA, LB, sz, 1e-12 ) );
    }

    free(LA); free(LB); free(self);
    tu_prop_free(tA); tu_prop_free(tB);
    diverge_model_free(mA); diverge_model_free(mB);
}

CT_TEST_NAME(tu_prop_katanin_linear, "tu_prop,katanin:linear") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return; // no GF floats

    diverge_model_t* m = kat_model( KAT_MU, true );
    tu_prop_t* t = tu_prop_init(m);

    const index_t sz = kat_loopsize(t);
    complex128_t* L0 = calloc(sz, sizeof*L0);
    complex128_t* L1 = calloc(sz, sizeof*L1);
    double resid[2] = {0.0, 0.0};

    tu_prop_GF_zero_T( t, I128*KAT_LAMBDA, NULL );
    tu_prop_ph_loop( t, L0, NULL );

    // D(eps)/eps must converge; compare D(eps)/eps against D(eps/2)/(eps/2)
    complex128_t* Dprev = calloc(sz, sizeof*Dprev);
    for (int step=0; step<3; ++step) {
        const double eps = 1e-2 / (double)(1 << step);
        complex128_t* self = kat_const_self( m, eps );
        tu_prop_GF_zero_T_kat( t, I128*KAT_LAMBDA, self );
        tu_prop_katanin_swap(t);
        tu_prop_ph_loop( t, L1, NULL );
        tu_prop_katanin_swap(t);

        double diff = 0.0;
        for (index_t i=0; i<sz; ++i) {
            complex128_t d = (L1[i] - L0[i]) / eps;
            if (step > 0) diff += POW2(cabs(d - Dprev[i]));
            Dprev[i] = d;
        }
        if (step > 0) resid[step-1] = sqrt(diff);
        free(self);
    }

    // halving eps has to shrink the residual by roughly a factor of two
    CT_CHECK( resid[0] > 0.0 );
    CT_CHECK( resid[1] < 0.6*resid[0] );

    free(L0); free(L1); free(Dprev);
    tu_prop_free(t);
    diverge_model_free(m);
}

CT_TEST_NAME(tu_prop_katanin_step, "tu_prop,katanin:step") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return; // no GF floats

    double nrm[2] = {0.0, 0.0};
    index_t nonzero = 0;

    for (int half=0; half<2; ++half) {
        const double dlambda = -0.02 / (double)(1 << half);

        diverge_model_t* m = kat_model( KAT_MU, true );
        tu_prop_t* prop = tu_prop_init(m);
        tu_projection_t* proj = tu_projection_init(m);
        tu_vertex_t* vert = tu_vertex_init( m, proj, prop, 1,1,1, /*self=*/1 );
        tu_stepper_t* step = tu_stepper_init( proj, prop, vert );
        CT_CHECK( step->katanin );

        double maxs[6] = {0};
        tu_stepper_euler( step, vert, KAT_LAMBDA, dlambda, maxs, NULL );

        // \mathcal{K} after exactly one shell
        for (index_t i=0; i<step->full_vert_size; ++i)
            nrm[half] += POW2(cabs(step->kat_ph[i]));
        nrm[half] = sqrt(nrm[half]);
        if (half == 0) nonzero = (nrm[0] > 0.0);

        // the flow must stay finite with Katanin on
        CT_CHECK( isfinite(maxs[0]) && isfinite(maxs[1]) && isfinite(maxs[2]) );

        tu_stepper_free(step);
        tu_vertex_free(vert);
        tu_projection_free(proj);
        tu_prop_free(prop);
        diverge_model_free(m);
    }

    CT_CHECK( nonzero );                         // Katanin does something
    CT_CHECK( fabs(nrm[0]/nrm[1] - 2.0) < 0.1 ); // and it is linear in d\Lambda
}

#endif // DIVERGE_SKIP_TESTS
