#define CT_SUITE_NAME tu_prop_gemm
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

// TU propagator algorithm tests. Greedy vs. GEMM, and so on.

#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_hack.h"
#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/propagator_details.h"
#include "../helper_functions.h"
#include "../models/all.h"

#define GEMM_LAMBDA 0.3
#define GEMM_TOL 1e-12

static void set_fine( diverge_model_t* m, index_t nkfx, index_t nkfy ) {
    diverge_model_internals_reset( m );
    m->nkf[0] = nkfx, m->nkf[1] = nkfy;
    diverge_model_internals_common( m );
}

static complex128_t* random_self( const diverge_model_t* m, double scale ) {
    const index_t nb = m->n_orb*m->n_spin;
    const index_t nktot = kdimtot(m->nk, m->nkf);
    complex128_t* self = calloc( nktot*POW2(nb), sizeof*self );
    random_seed( 4242 );
    for (index_t i=0; i<nktot*POW2(nb); ++i) self[i] = scale*random_cmplx();
    return self;
}

static double maxabs( const complex128_t* x, index_t n ) {
    double m = 0.0;
    for (index_t i=0; i<n; ++i) m = MAX(m, cabs(x[i]));
    return m;
}

// loop of either algorithm (reduced or simple), unscaled if !rescale
static void loop_algo( tu_prop_t* t, tu_fft_algorithm_t algo, bool reduced, bool pp,
        bool subtract, complex128_t* L, index_t ntasks, index_t taskoff, bool rescale ) {
    if (reduced) {
        t->which_fft_red = algo;
        S_loop_static_red( t, pp, subtract, L, ntasks, taskoff, rescale );
    } else {
        t->which_fft_simple = algo;
        S_loop_static_simple( t, pp, subtract, L, ntasks, taskoff, rescale );
    }
}

static void gemm_vs_default( diverge_model_t* m ) {
    tu_data_t* tu_data = (tu_data_t*)m->internals->tu_data;
    tu_prop_t* t = tu_prop_init( m );
    const tu_fft_algorithm_t default_red = t->which_fft_red,
                             default_simple = t->which_fft_simple;
    const index_t sz = t->my_nk*POW2(t->n_orbff*POW2(t->n_spin));
    complex128_t* L1 = calloc( sz, sizeof*L1 );
    complex128_t* L2 = calloc( sz, sizeof*L2 );
    complex128_t* self = random_self( m, 0.3 );

    for (int use_self=0; use_self<2; ++use_self) {
        tu_prop_GF_zero_T( t, I128*GEMM_LAMBDA, use_self ? self : NULL );
        for (int reduced=0; reduced<2; ++reduced) {
            if (reduced && !tu_data->use_reduced_loop) continue;
            const tu_fft_algorithm_t default_algo = reduced ? default_red : default_simple;
            const index_t ntot = reduced ? tu_data->unique_mi_l_sz : POW2(t->n_orbff);
            for (int pp=0; pp<2; ++pp) {
                ZERO_OUT(L1, sz); ZERO_OUT(L2, sz);
                loop_algo( t, default_algo, reduced, pp, false, L1, -1, 0, true );
                loop_algo( t, tu_fft_algorithm_gemm, reduced, pp, false, L2, -1, 0, true );
                double tol = GEMM_TOL*MAX(1.0, maxabs(L1, sz));
                CT_CHECK( maxabs(L1, sz) > 0.0 );
                CT_CHECK( check_equal_cd( L1, L2, sz, tol ) );

                // force no memory => row-wise calculation
                double mem = tu_data->gemm_loop_mem_GiB;
                tu_data->gemm_loop_mem_GiB = 1.e-12;
                ZERO_OUT(L2, sz);
                loop_algo( t, tu_fft_algorithm_gemm, reduced, pp, false, L2, -1, 0, true );
                tu_data->gemm_loop_mem_GiB = mem;
                CT_CHECK( check_equal_cd( L1, L2, sz, tol ) );

                // now with parallel calculation
                ZERO_OUT(L1, sz); ZERO_OUT(L2, sz);
                loop_algo( t, default_algo, reduced, pp, false, L1, -1, 0, false );
                index_t half = ntot/2;
                loop_algo( t, tu_fft_algorithm_gemm, reduced, pp, false, L2, half, 0, false );
                loop_algo( t, tu_fft_algorithm_gemm, reduced, pp, false, L2, ntot-half, half, false );
                CT_CHECK( check_equal_cd( L1, L2, sz, tol ) );
                loop_algo( t, tu_fft_algorithm_gemm, reduced, pp, true, L1, -1, 0, false );
                CT_CHECK( maxabs(L1, sz) < tol );
            }
        }
    }
    t->which_fft_red = default_red;
    t->which_fft_simple = default_simple;

    // we also check the permuted GEMM layout for the GF buffers
    {
        const tu_fft_algorithm_t red0 = tu_data->which_fft_red, simple0 = tu_data->which_fft_simple;
        tu_data->which_fft_red = tu_data->which_fft_simple = tu_fft_algorithm_gemm;
        tu_prop_t* tg = tu_prop_init( m );
        tu_data->which_fft_red = red0, tu_data->which_fft_simple = simple0;
        CT_CHECK( tg->gemm_gf_layout == (t->nktot > 1) );
        for (int use_self=0; use_self<2; ++use_self) {
            tu_prop_GF_zero_T( t, I128*GEMM_LAMBDA, use_self ? self : NULL );
            tu_prop_GF_zero_T( tg, I128*GEMM_LAMBDA, use_self ? self : NULL );
            for (int reduced=0; reduced<2; ++reduced) {
                if (reduced && !tu_data->use_reduced_loop) continue;
                for (int pp=0; pp<2; ++pp) {
                    ZERO_OUT(L1, sz); ZERO_OUT(L2, sz);
                    loop_algo( t, reduced ? default_red : default_simple, reduced, pp, false, L1, -1, 0, true );
                    loop_algo( tg, tu_fft_algorithm_gemm, reduced, pp, false, L2, -1, 0, true );
                    CT_CHECK( check_equal_cd( L1, L2, sz, GEMM_TOL*MAX(1.0, maxabs(L1, sz)) ) );
                }
            }
        }
        tu_prop_free( tg );
    }

    free( self );
    free( L2 );
    free( L1 );
    tu_prop_free( t );
}

CT_TEST_NAME(tu_prop_gemm_random, "tu_prop,gemm,random") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return; // no GF floats
    diverge_model_t* m = KWARGS(gen_random, .nkx=4, .nky=4);
    set_fine( m, 3, 2 );
    diverge_model_internals_tu( m, 2.1 );
    gemm_vs_default( m );
    diverge_model_free( m );
}

CT_TEST_NAME(tu_prop_gemm_random_coarse, "tu_prop,gemm,random:coarse") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;
    diverge_model_t* m = KWARGS(gen_random, .nkx=6, .nky=4);
    diverge_model_internals_tu( m, 1.5 );
    gemm_vs_default( m );
    diverge_model_free( m );
}

CT_TEST_NAME(tu_prop_gemm_rashba_sym, "tu_prop,gemm,rashba:sym") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;
    diverge_model_t* m = KWARGS(gen_square_lat_rashba_sym, .nkx=6, .nky=6, .nkxf=3, .nkyf=3);
    diverge_model_internals_tu( m, 2.1 );
    gemm_vs_default( m );
    diverge_model_free( m );
}

CT_TEST_NAME(tu_prop_gemm_honeycomb, "tu_prop,gemm,honeycomb") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;
    diverge_model_t* m = KWARGS(gen_honeycomb_lat_hub_sym, .nkx=6, .nky=6, .nkxf=2, .nkyf=3);
    diverge_model_internals_tu( m, 1.5 );
    gemm_vs_default( m );
    diverge_model_free( m );
}

CT_TEST_NAME(tu_prop_gemm_square, "tu_prop,gemm,square") {
    if (sizeof(gf_complex_t) != sizeof(complex128_t)) return;
    diverge_model_t* m = KWARGS(gen_square_lat_hub, .nkx=8, .nky=6, .lx=1, .ly=1, .nkxf=3, .nkyf=1);
    diverge_model_internals_tu( m, 2.1 );
    gemm_vs_default( m );
    diverge_model_free( m );
}

#endif // DIVERGE_SKIP_TESTS
