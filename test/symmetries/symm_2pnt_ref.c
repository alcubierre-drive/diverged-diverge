#define CT_SUITE_NAME twopoint_sym_ref
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

// symmetrization checks against the previous version

#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_internals_struct.h"
#include "../../src/diverge_symmetrize.h"
#include "../../src/misc/mpi_functions.h"
#include "../helper_functions.h"
#include "../models/all.h"

static double ref_dot( const double* A, const double* B ) {
    return A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
}

// the original implementation (direct sum)
static void symmetrize_2pt_ref( diverge_model_t* model, complex128_t* buf, const complex128_t* aux,
        index_t nk, const index_t* symm_map_mom, const double* kmesh ) {
    index_t n_os = model->n_orb * model->n_spin;
    index_t n_spin = model->n_spin;
    index_t n_orb = model->n_orb;
    index_t n_sym = model->n_sym;
    double norm = 1./(double) n_sym;
    double* symm_beyond_UC = model->internals->symm_beyond_UC;
    index_t* symm_map_orb = model->internals->symm_map_orb;
    index_t* symm_orb_off = model->internals->symm_orb_off;
    index_t* symm_orb_len = model->internals->symm_orb_len;
    complex128_t* symm_pref = model->internals->symm_pref;
    memset( buf, 0, sizeof(complex128_t)*nk*POW2(n_os) );

    #pragma omp parallel for collapse(5) num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
    for (index_t oo1=0; oo1<n_orb; ++oo1)
    for (index_t ss1=0; ss1<n_spin; ++ss1)
    for (index_t oo2=0; oo2<n_orb; ++oo2)
    for (index_t ss2=0; ss2<n_spin; ++ss2) {
        index_t o1 = oo1 + n_orb * ss1;
        index_t o2 = oo2 + n_orb * ss2;
        for (index_t s=0; s<n_sym; ++s) {
            index_t osym1 = IDX2( o1, s, n_sym ),
                    osym2 = IDX2( o2, s, n_sym ),
                    ksym = symm_map_mom[IDX2(k,s,n_sym)];
            for (index_t i=0; i<symm_orb_len[osym1]; ++i)
            for (index_t j=0; j<symm_orb_len[osym2]; ++j) {
                index_t xo1 = symm_orb_off[osym1] + i,
                        xo2 = symm_orb_off[osym2] + j;
                double kdotR11 = ref_dot(symm_beyond_UC+3*IDX2(oo1,s,n_sym), kmesh+3*ksym);
                double kdotR22 = ref_dot(symm_beyond_UC+3*IDX2(oo2,s,n_sym), kmesh+3*ksym);
                buf[IDX3(k,o1,o2,n_os,n_os)] += aux[IDX3(ksym,symm_map_orb[xo1],symm_map_orb[xo2], n_os,n_os)] *
                                                (symm_pref[xo1] * (cos(kdotR11) - I * sin(kdotR11))) *
                                            conj(symm_pref[xo2] * (cos(kdotR22) - I * sin(kdotR22))) * norm;
            }
        }
    }
}

static double maxabs_diff( const complex128_t* a, const complex128_t* b, index_t n, double* amax ) {
    double d = 0.0, m = 0.0;
    for (index_t i=0; i<n; ++i) { d = MAX(d, cabs(a[i]-b[i])); m = MAX(m, cabs(a[i])); }
    if (amax) *amax = m;
    return d;
}

static void compare_with_ref( diverge_model_t* model ) {
    CT_CHECK( model->n_sym > 0 );
    diverge_generate_symm_maps( model );
    const index_t n_os = model->n_orb*model->n_spin;
    random_seed( 777 );
    for (int fine=0; fine<2; ++fine) {
        const index_t nk = fine ? kdimtot(model->nk, model->nkf) : kdim(model->nk);
        const index_t sz = nk*POW2(n_os);
        complex128_t* in = calloc( sz, sizeof*in );
        complex128_t* ref = calloc( sz, sizeof*ref );
        complex128_t* out = calloc( sz, sizeof*out );
        for (index_t i=0; i<sz; ++i) in[i] = random_cmplx();

        symmetrize_2pt_ref( model, ref, in, nk,
                fine ? model->internals->symm_map_mom_fine : model->internals->symm_map_mom_crs,
                fine ? model->internals->kfmesh : model->internals->kmesh );

        memcpy( out, in, sizeof(complex128_t)*sz );
        if (fine) diverge_symmetrize_2pt_fine( model, out, NULL );
        else      diverge_symmetrize_2pt_coarse( model, out, NULL );
        double amax = 0.0, d = maxabs_diff( ref, out, sz, &amax );
        CT_CHECK( amax > 0.0 );
        CT_CHECK( d <= 1e-13*amax );

        memcpy( in, out, sizeof(complex128_t)*sz );
        symmetrize_2pt_ref( model, ref, in, nk,
                fine ? model->internals->symm_map_mom_fine : model->internals->symm_map_mom_crs,
                fine ? model->internals->kfmesh : model->internals->kmesh );
        if (fine) diverge_symmetrize_2pt_fine( model, out, NULL );
        else      diverge_symmetrize_2pt_coarse( model, out, NULL );
        CT_CHECK( maxabs_diff( ref, out, sz, NULL ) <= 1e-13*amax );

        if (fine && sizeof(gf_complex_t) == sizeof(complex128_t)) {
            for (index_t i=0; i<sz; ++i) in[i] = random_cmplx();
            memcpy( out, in, sizeof(complex128_t)*sz );
            symmetrize_2pt_ref( model, ref, in, nk, model->internals->symm_map_mom_fine,
                    model->internals->kfmesh );
            diverge_symmetrize_greens( model, (gf_complex_t*)out, NULL );
            CT_CHECK( maxabs_diff( ref, out, sz, NULL ) <= 1e-13*amax );
        }
        free( out );
        free( ref );
        free( in );
    }
}

CT_TEST_NAME(twopoint_sym_ref_rashba, "symmetrize,2pt,ref,rashba") {
    diverge_model_t* m = KWARGS(gen_square_lat_rashba_sym, .nkx=6, .nky=6, .nkxf=3, .nkyf=3);
    strcpy( m->name, "rashba_sym" );
    compare_with_ref( m );
    diverge_model_free( m );
}

CT_TEST_NAME(twopoint_sym_ref_honeycomb, "symmetrize,2pt,ref,honeycomb") {
    diverge_model_t* m = KWARGS(gen_honeycomb_lat_hub_sym, .nkx=6, .nky=6, .nkxf=2, .nkyf=2);
    strcpy( m->name, "honeycomb_sym" );
    compare_with_ref( m );
    diverge_model_free( m );
}

CT_TEST_NAME(twopoint_sym_ref_pxpy, "symmetrize,2pt,ref,pxpy") {
    diverge_model_t* m = KWARGS(gen_triangular_pxpy_sym, .nkx=6, .nky=6);
    strcpy( m->name, "triangular_pxpy_sym" );
    compare_with_ref( m );
    diverge_model_free( m );
}

#endif // DIVERGE_SKIP_TESTS
