#define CT_SUITE_NAME tu_reco_ft
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/vertex.h"
#include "../../src/tu/vertex_details.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/projection.h"
#include "../../src/tu/vertex_reco_fused.h"
#include "../../src/tu/diverge_interface_symmstruct.h"
#include "../../src/diverge_hack.h"
#include "../models/all.h"
#include "../helper_functions.h"
#include "../../src/misc/mpi_functions.h"

// Average the IBZ vertex over the little group of each of its momenta, using the
// same relation the reconstruction inverts,
//   V(q,R,L) = sum_{l,r} pref(l,q) conj(pref(r,q)) V(q, r, l),
// with l,r running over the images of L,R, so that random test data becomes a
// legal IBZ vertex. Multi dimensional orbital irreps have more than one image
// per leg, which is exactly the case the fused kernel has to cope with.
static void symmetrize_ibz_vertex( diverge_model_t* model, tu_vertex_t* V,
        complex128_t* X, bool use_in_in ) {
    tu_data_t* tu = model->internals->tu_data;
    const tu_symmstruct_t* symm = tu->symm;
    if (!symm->use_symmetries) return;
    const index_t n_sym = model->n_sym, n_spin = model->n_spin, n_orbff = tu->n_orbff;
    const index_t legdim = POW2(n_spin)*n_orbff, rest = POW2(legdim);
    const index_t* mom = model->internals->symm_map_mom_crs;
    const complex128_t* cc = use_in_in ? symm->mi_prefac_in_in : symm->mi_prefac_in_out;

    // entries of the symmetry list that share a momentum action are duplicates
    // and must not be counted twice, or the average is not a projector
    const index_t nk_mom = symm->nk;
    double* wsym = calloc( n_sym, sizeof(double) );
    index_t* cls = calloc( n_sym, sizeof(index_t) );
    index_t* crep = calloc( n_sym, sizeof(index_t) );
    index_t* csz = calloc( n_sym, sizeof(index_t) );
    index_t ncls = 0;
    for (index_t sy=0; sy<n_sym; ++sy) {
        cls[sy] = -1;
        for (index_t c=0; c<ncls; ++c) {
            bool same = true;
            for (index_t k=0; k<nk_mom; ++k)
                if (mom[sy+n_sym*k] != mom[crep[c]+n_sym*k]) { same = false; break; }
            if (same) { cls[sy] = c; break; }
        }
        if (cls[sy] < 0) { crep[ncls] = sy; cls[sy] = ncls++; }
        csz[cls[sy]]++;
    }
    for (index_t sy=0; sy<n_sym; ++sy) wsym[sy] = 1.0/(double)csz[cls[sy]];

    complex128_t* acc = calloc( rest, sizeof(complex128_t) );
    // indexed by the global map entry, not by the leg
    complex128_t* pref = calloc( symm->mi_maps_to_sz, sizeof(complex128_t) );
    for (index_t q=0; q<V->my_nk; ++q) {
        const index_t q_in_full = symm->idx_ibz_in_fullmesh[q + tu->my_nk_off];
        const double* kp = model->internals->kmesh + 3*q_in_full;
        complex128_t* Xq = X + rest*q;
        ZERO_OUT( acc, rest );
        double nstab = 0.0;
        for (index_t sy=0; sy<n_sym; ++sy) {
            if (mom[sy + n_sym*q_in_full] != q_in_full) continue;
            nstab += wsym[sy];
            const index_t isym = symm->inv_symmetry[sy];
            for (index_t l=0; l<legdim; ++l) {
                const index_t idx = l+legdim*isym;
                for (index_t e=0; e<symm->mi_map_len[idx]; ++e) {
                    const index_t ent = symm->mi_map_off[idx]+e;
                    double kd = 0.0;
                    for (int c3=0; c3<3; ++c3) {
                        double d = 0.0;
                        for (int g=0; g<3; ++g)
                            d += model->rs_symmetries[sy][c3][g]*symm->beyond_UC[3*ent+g];
                        kd += kp[c3]*d;
                    }
                    pref[ent] = cc[ent]*cexp(-I*kd);
                }
            }
            for (index_t R=0; R<legdim; ++R)
            for (index_t L=0; L<legdim; ++L) {
                const index_t Lo = symm->mi_map_off[L+legdim*isym], Ln = symm->mi_map_len[L+legdim*isym];
                const index_t Ro = symm->mi_map_off[R+legdim*isym], Rn = symm->mi_map_len[R+legdim*isym];
                complex128_t sum = 0.0;
                for (index_t li=0; li<Ln; ++li)
                for (index_t ri=0; ri<Rn; ++ri)
                    sum += pref[Lo+li]*conj(pref[Ro+ri])*
                           Xq[IDX2(symm->mi_maps_to[Ro+ri],symm->mi_maps_to[Lo+li],legdim)];
                acc[IDX2(R,L,legdim)] += wsym[sy]*sum;
            }
        }
        for (index_t i=0; i<rest; ++i) Xq[i] = acc[i]/nstab;
    }
    free(acc); free(pref); free(wsym); free(cls); free(crep); free(csz);
}

// A/B check of diverge_tu_vertex_reco_ft() -- the default two-step path, or the
// fused one if hack_fused enables it -- against diverge_tu_vertex_reco_ft_ref().
// The fused kernel is compared to the reference directly either way.
static void compare_reco_ft( diverge_model_t* model, double ffcut, int nrep, const char* tag,
        bool hack_fused ) {
    diverge_model_internals_tu( model, ffcut );
    // the fused kernel is opt-in: only with the hack does the public entry point
    // use it, otherwise it takes the two-step path
    if (hack_fused) diverge_model_hack( model, "tu_reco_fused", "1" );
    tu_projection_t* proj = tu_projection_init( model );
    tu_prop_t* loop = tu_prop_init( model );
    tu_vertex_t* V = tu_vertex_init( model, proj, loop, 1,1,1,0 );

    const index_t rest = POW2(V->n_orbff*POW2(V->n_spin));
    const index_t nr = proj->nr;
    CT_CHECK( (V->reco_fused != NULL) == (hack_fused && V->symm->use_symmetries) );

    index_t maxmaplen = 0;
    if (V->symm->use_symmetries)
        for (index_t i=0; i<V->symm->mi_cnt; ++i)
            maxmaplen = MAX( maxmaplen, V->symm->mi_map_len[i] );
    CT_PRINTF( "%s: max leg map length %li (>1 exercises the general path)\n",
            tag, (long)maxmaplen );

    random_seed( 0xbeef );
    for (index_t i=0; i<V->full_vert_size; ++i) {
        V->Pch[i] = random_cmplx();
        V->Cch[i] = random_cmplx();
        V->Dch[i] = random_cmplx();
    }
    // the IBZ vertex has to be invariant under the little group of each of its
    // momenta -- random data is not, so project it onto that subspace
    symmetrize_ibz_vertex( model, V, V->Pch, true );
    { complex128_t* chk = malloc(V->full_vert_size*sizeof(complex128_t));
      memcpy(chk, V->Pch, V->full_vert_size*sizeof(complex128_t));
      symmetrize_ibz_vertex( model, V, V->Pch, true );
      double d = 0, m = 0;
      for (index_t i=0;i<V->full_vert_size;++i){ d=MAX(d,cabs(chk[i]-V->Pch[i])); m=MAX(m,cabs(chk[i])); }
      CT_PRINTF("%s: symmetrizer idempotency dev=%.3e\n", tag, d/m);
      CT_CHECK( d <= 1e-12*m ); free(chk); }
    symmetrize_ibz_vertex( model, V, V->Cch, false );
    symmetrize_ibz_vertex( model, V, V->Dch, false );

    complex128_t* ref = calloc( rest*nr, sizeof(complex128_t) );
    complex128_t* new = calloc( rest*nr, sizeof(complex128_t) );
    complex128_t* scratch = calloc( MAX(V->my_nk,V->symm->my_nk_in_bz)*rest, sizeof(complex128_t) );

    complex128_t* fus = calloc( rest*nr, sizeof(complex128_t) );
    tu_reco_fused_t* fused = tu_reco_fused_init( model, proj );
    double t_ref = 0.0, t_new = 0.0, t_fus = 0.0;
    const char channels[3] = { 'P', 'C', 'D' };
    for (int c=0; c<3; ++c) {
        for (int rep=0; rep<nrep; ++rep) {
            double t0 = diverge_mpi_wtime();
            diverge_tu_vertex_reco_ft_ref( V, channels[c], ref, proj->my_prefac_reco, nr, scratch );
            double t1 = diverge_mpi_wtime();
            diverge_tu_vertex_reco_ft( V, channels[c], new, proj->my_prefac_reco, nr, scratch );
            double t2 = diverge_mpi_wtime();
            t_ref += t1-t0; t_new += t2-t1;
        }

        double maxdiff = 0.0, maxval = 0.0;
        for (index_t i=0; i<rest*nr; ++i) {
            maxdiff = MAX( maxdiff, cabs(ref[i]-new[i]) );
            maxval = MAX( maxval, cabs(ref[i]) );
        }
        CT_PRINTF( "%s %c: nr=%li rest=%li my_nk=%li my_nk_in_bz=%li rel.dev=%.3e\n",
                tag, channels[c], (long)nr, (long)rest, (long)V->my_nk,
                (long)V->symm->my_nk_in_bz, maxdiff/maxval );
        CT_CHECK( maxdiff <= 1e-13*maxval );

        if (fused) {
            double t3 = diverge_mpi_wtime();
            for (int rep=0; rep<nrep; ++rep) tu_reco_fused_ft( fused, V, channels[c], fus );
            t_fus += diverge_mpi_wtime()-t3;
            double fd = 0.0;
            for (index_t i=0; i<rest*nr; ++i) fd = MAX( fd, cabs(ref[i]-fus[i]) );
            CT_PRINTF( "%s %c: fused nu=%li rel.dev=%.3e\n", tag, channels[c],
                    (long)tu_reco_fused_nu(fused), fd/maxval );
            CT_CHECK( fd <= 1e-12*maxval );
        }
    }
    CT_PRINTF( "%s timing: ref=%.4fs public=%.4fs (%.2fx) fused=%.4fs (%.2fx)\n", tag,
            t_ref, t_new, t_ref/t_new, t_fus, t_fus > 0.0 ? t_ref/t_fus : 0.0 );

    tu_reco_fused_free(fused);
    free(ref); free(new); free(scratch); free(fus);
    tu_vertex_free(V); tu_prop_free(loop); tu_projection_free(proj);
    diverge_model_free(model);
}

CT_TEST_NAME( nosym_spin1, "reco_ft,nosym:nspin1" ) {
    compare_reco_ft( KWARGS(gen_square_lat_hub, .nkx=12, .nky=12), 1.2, 1, "nosym,nspin1", false );
}
CT_TEST_NAME( sym_spin1, "reco_ft,sym:nspin1" ) {
    compare_reco_ft( KWARGS(gen_square_lat_hub_sym, .nkx=12, .nky=12), 1.2, 1, "sym,nspin1", false );
}
CT_TEST_NAME( nosym_spin2, "reco_ft,nosym:nspin2" ) {
    compare_reco_ft( KWARGS(gen_square_lat_rashba, .nkx=12, .nky=12), 1.2, 1, "nosym,nspin2", false );
}
CT_TEST_NAME( sym_spin2, "reco_ft,sym:nspin2" ) {
    compare_reco_ft( KWARGS(gen_square_lat_rashba_sym, .nkx=12, .nky=12), 1.2, 1, "sym,nspin2", false );
}

CT_TEST_NAME( sym_honeycomb, "reco_ft,sym:honeycomb" ) {
    compare_reco_ft( KWARGS(gen_honeycomb_lat_hub_sym, .nkx=12, .nky=12), 1.6, 1, "sym,honeycomb", false );
}
CT_TEST_NAME( sym_kagome, "reco_ft,sym:kagome" ) {
    compare_reco_ft( KWARGS(gen_kagome_model, .nkx=12, .nky=12), 1.2, 1, "sym,kagome", false );
}

/* p_x/p_y on the triangular lattice: the C3/C6 rotations mix the two orbitals,
 * so a leg maps onto a superposition and the leg maps are not one-to-one. In the
 * p_+/p_- basis the same model diagonalises the rotations and the maps collapse
 * back to one image per leg -- both must reproduce the reference. */
CT_TEST_NAME( sym_pxpy, "reco_ft,sym:pxpy" ) {
    compare_reco_ft( KWARGS(gen_triangular_pxpy_sym, .nkx=12, .nky=12, .to_ppm=-1), 1.2, 1, "sym,pxpy", false );
}
CT_TEST_NAME( sym_ppm, "reco_ft,sym:ppm" ) {
    compare_reco_ft( KWARGS(gen_triangular_pxpy_sym, .nkx=12, .nky=12, .to_ppm= 1), 1.2, 1, "sym,ppm", false );
}

/* With the hack the public entry point routes through the fused kernel, so the
 * same comparison against the reference now tests the fused path end to end --
 * once monomial, once with the multi-image leg maps of p_x/p_y. */
CT_TEST_NAME( hack_sym_spin1, "reco_ft,hack:nspin1" ) {
    compare_reco_ft( KWARGS(gen_square_lat_hub_sym, .nkx=12, .nky=12), 1.2, 1, "hack,sym,nspin1", true );
}
CT_TEST_NAME( hack_sym_pxpy, "reco_ft,hack:pxpy" ) {
    compare_reco_ft( KWARGS(gen_triangular_pxpy_sym, .nkx=12, .nky=12, .to_ppm=-1), 1.2, 1, "hack,sym,pxpy", true );
}

// production-sized timing comparison, hidden (run explicitly)
CT_TEST_NAME_HIDDEN( speed_nosym_spin1, "reco_ft,speed,nosym,nspin1" ) {
    compare_reco_ft( KWARGS(gen_square_lat_hub, .nkx=64, .nky=64), 3.1, 20, "speed,nosym,nspin1", false );
}
CT_TEST_NAME_HIDDEN( speed_sym_spin1, "reco_ft,speed,sym,nspin1" ) {
    compare_reco_ft( KWARGS(gen_square_lat_hub_sym, .nkx=64, .nky=64), 3.1, 20, "speed,sym,nspin1", false );
}
CT_TEST_NAME_HIDDEN( speed_nosym_spin2, "reco_ft,speed,nosym,nspin2" ) {
    compare_reco_ft( KWARGS(gen_square_lat_rashba, .nkx=64, .nky=64), 2.1, 5, "speed,nosym,nspin2", false );
}
CT_TEST_NAME_HIDDEN( speed_sym_spin2, "reco_ft,speed,sym,nspin2" ) {
    compare_reco_ft( KWARGS(gen_square_lat_rashba_sym, .nkx=64, .nky=64), 2.1, 5, "speed,sym,nspin2", false );
}
CT_TEST_NAME_HIDDEN( speed_sym_hc, "reco_ft,speed,sym,honeycomb" ) {
    compare_reco_ft( KWARGS(gen_honeycomb_lat_hub_sym, .nkx=48, .nky=48), 2.1, 5, "speed,sym,honeycomb", false );
}
CT_TEST_NAME_HIDDEN( speed_sym_kag, "reco_ft,speed,sym,kagome" ) {
    compare_reco_ft( KWARGS(gen_kagome_model, .nkx=48, .nky=48), 1.6, 5, "speed,sym,kagome", false );
}
CT_TEST_NAME_HIDDEN( speed_sym_pxpy, "reco_ft,speed,sym,pxpy" ) {
    compare_reco_ft( KWARGS(gen_triangular_pxpy_sym, .nkx=48, .nky=48, .to_ppm=-1), 2.1, 5, "speed,sym,pxpy", false );
}

/* Survey of the leg maps the model zoo produces. A leg map with more than one
 * image is what separates the general reconstruction from a monomial one; this
 * records which models (if any) reach it. */
static void survey_map_lengths( diverge_model_t* model, double ffcut, const char* tag ) {
    diverge_model_internals_tu( model, ffcut );
    tu_data_t* tu = model->internals->tu_data;
    const tu_symmstruct_t* symm = tu->symm;
    index_t maxlen = 0, nmulti = 0;
    if (symm->use_symmetries)
        for (index_t i=0; i<symm->mi_cnt; ++i) {
            maxlen = MAX( maxlen, symm->mi_map_len[i] );
            if (symm->mi_map_len[i] > 1) nmulti++;
        }
    tu_projection_t* proj = tu_projection_init( model );
    // building the tables exercises the closed-group check inside
    tu_reco_fused_t* fused = tu_reco_fused_init( model, proj );
    CT_PRINTF( "%-22s n_sym=%-3li legdim*n_sym=%-6li max map len %li, %li multi-image legs\n",
            tag, (long)model->n_sym, (long)symm->mi_cnt, (long)maxlen, (long)nmulti );
    tu_reco_fused_free( fused );
    tu_projection_free( proj );
    diverge_model_free( model );
}

CT_TEST_NAME_HIDDEN( survey, "reco_ft,survey" ) {
    survey_map_lengths( KWARGS(gen_square_lat_hub_sym, .nkx=12, .nky=12), 1.2, "square" );
    survey_map_lengths( KWARGS(gen_square_lat_rashba_sym, .nkx=12, .nky=12), 1.2, "rashba" );
    survey_map_lengths( KWARGS(gen_honeycomb_lat_hub_sym, .nkx=12, .nky=12), 1.6, "honeycomb" );
    survey_map_lengths( KWARGS(gen_kagome_model, .nkx=12, .nky=12), 1.2, "kagome" );
    survey_map_lengths( KWARGS(gen_triangular_pxpy_sym, .nkx=12, .nky=12, .to_ppm=-1), 1.2, "pxpy" );
    survey_map_lengths( KWARGS(gen_triangular_pxpy_sym, .nkx=12, .nky=12, .to_ppm= 1), 1.2, "p+p-" );
    survey_map_lengths( KWARGS(gen_triangular_lat_rashba, .nkx=12, .nky=12), 1.2, "tri-rashba" );
    survey_map_lengths( KWARGS(gen_honeycomb_lat_rash, .nkx=12, .nky=12), 1.6, "hc-rashba" );
    // larger form factor cutoffs pull in bonds whose images can collide
    survey_map_lengths( KWARGS(gen_honeycomb_lat_hub_sym, .nkx=12, .nky=12), 3.1, "honeycomb ff3.1" );
    survey_map_lengths( KWARGS(gen_kagome_model, .nkx=12, .nky=12), 2.6, "kagome ff2.6" );
    survey_map_lengths( KWARGS(gen_triangular_pxpy_sym, .nkx=12, .nky=12, .to_ppm=-1), 3.1, "pxpy ff3.1" );
    CT_CHECK( 1 );
}

#endif // DIVERGE_SKIP_TESTS
