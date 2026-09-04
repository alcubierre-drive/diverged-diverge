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
#include "../models/all.h"
#include "../helper_functions.h"
#include "../../src/misc/mpi_functions.h"

// Average the IBZ vertex over the little group of each of its momenta, using the
// same relation the reconstruction inverts,
//   V(q,R,L) = pref(L,q) conj(pref(R,q)) V(q, sigma(R), sigma(L)),
// so that random test data becomes a legal IBZ vertex.
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
    complex128_t* pref = calloc( legdim, sizeof(complex128_t) );
    index_t* map = calloc( legdim, sizeof(index_t) );
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
                const index_t off = symm->mi_map_off[l+legdim*isym];
                double kd = 0.0;
                for (int e=0; e<3; ++e) {
                    double d = 0.0;
                    for (int g=0; g<3; ++g)
                        d += model->rs_symmetries[sy][e][g]*symm->beyond_UC[3*(l+legdim*isym)+g];
                    kd += kp[e]*d;
                }
                pref[l] = cc[off]*cexp(-I*kd);
                map[l] = symm->mi_maps_to[off];
            }
            for (index_t R=0; R<legdim; ++R)
            for (index_t L=0; L<legdim; ++L)
                acc[IDX2(R,L,legdim)] += wsym[sy]*pref[L]*conj(pref[R])*Xq[IDX2(map[R],map[L],legdim)];
        }
        for (index_t i=0; i<rest; ++i) Xq[i] = acc[i]/nstab;
    }
    free(acc); free(pref); free(map); free(wsym); free(cls); free(crep); free(csz);
}

// A/B check of the fused diverge_tu_vertex_reco_ft() against the two-step
// reference implementation diverge_tu_vertex_reco_ft_ref()
static void compare_reco_ft( diverge_model_t* model, double ffcut, int nrep, const char* tag ) {
    diverge_model_internals_tu( model, ffcut );
    tu_projection_t* proj = tu_projection_init( model );
    tu_prop_t* loop = tu_prop_init( model );
    tu_vertex_t* V = tu_vertex_init( model, proj, loop, 1,1,1,0 );

    const index_t rest = POW2(V->n_orbff*POW2(V->n_spin));
    const index_t nr = proj->nr;

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
            CT_PRINTF( "%s %c: fused nu=%li nslow=%li/%li rel.dev=%.3e\n", tag, channels[c],
                    (long)tu_reco_fused_nu(fused), (long)tu_reco_fused_nslow(fused),
                    (long)V->symm->my_nk_in_bz, fd/maxval );
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
    compare_reco_ft( KWARGS(gen_square_lat_hub, .nkx=12, .nky=12), 1.2, 1, "nosym,nspin1" );
}
CT_TEST_NAME( sym_spin1, "reco_ft,sym:nspin1" ) {
    compare_reco_ft( KWARGS(gen_square_lat_hub_sym, .nkx=12, .nky=12), 1.2, 1, "sym,nspin1" );
}
CT_TEST_NAME( nosym_spin2, "reco_ft,nosym:nspin2" ) {
    compare_reco_ft( KWARGS(gen_square_lat_rashba, .nkx=12, .nky=12), 1.2, 1, "nosym,nspin2" );
}
CT_TEST_NAME( sym_spin2, "reco_ft,sym:nspin2" ) {
    compare_reco_ft( KWARGS(gen_square_lat_rashba_sym, .nkx=12, .nky=12), 1.2, 1, "sym,nspin2" );
}

CT_TEST_NAME( sym_honeycomb, "reco_ft,sym:honeycomb" ) {
    compare_reco_ft( KWARGS(gen_honeycomb_lat_hub_sym, .nkx=12, .nky=12), 1.6, 1, "sym,honeycomb" );
}
CT_TEST_NAME( sym_kagome, "reco_ft,sym:kagome" ) {
    compare_reco_ft( KWARGS(gen_kagome_model, .nkx=12, .nky=12), 1.2, 1, "sym,kagome" );
}

// production-sized timing comparison, hidden (run explicitly)
CT_TEST_NAME_HIDDEN( speed_nosym_spin1, "reco_ft,speed,nosym,nspin1" ) {
    compare_reco_ft( KWARGS(gen_square_lat_hub, .nkx=64, .nky=64), 3.1, 20, "speed,nosym,nspin1" );
}
CT_TEST_NAME_HIDDEN( speed_sym_spin1, "reco_ft,speed,sym,nspin1" ) {
    compare_reco_ft( KWARGS(gen_square_lat_hub_sym, .nkx=64, .nky=64), 3.1, 20, "speed,sym,nspin1" );
}
CT_TEST_NAME_HIDDEN( speed_nosym_spin2, "reco_ft,speed,nosym,nspin2" ) {
    compare_reco_ft( KWARGS(gen_square_lat_rashba, .nkx=64, .nky=64), 2.1, 5, "speed,nosym,nspin2" );
}
CT_TEST_NAME_HIDDEN( speed_sym_spin2, "reco_ft,speed,sym,nspin2" ) {
    compare_reco_ft( KWARGS(gen_square_lat_rashba_sym, .nkx=64, .nky=64), 2.1, 5, "speed,sym,nspin2" );
}
CT_TEST_NAME_HIDDEN( speed_sym_hc, "reco_ft,speed,sym,honeycomb" ) {
    compare_reco_ft( KWARGS(gen_honeycomb_lat_hub_sym, .nkx=48, .nky=48), 2.1, 5, "speed,sym,honeycomb" );
}
CT_TEST_NAME_HIDDEN( speed_sym_kag, "reco_ft,speed,sym,kagome" ) {
    compare_reco_ft( KWARGS(gen_kagome_model, .nkx=48, .nky=48), 1.6, 5, "speed,sym,kagome" );
}

#endif // DIVERGE_SKIP_TESTS
