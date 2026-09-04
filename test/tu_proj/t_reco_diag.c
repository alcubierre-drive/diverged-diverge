#define CT_SUITE_NAME reco_diag
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/vertex.h"
#include "../../src/tu/vertex_details.h"
#include "../../src/tu/propagator.h"
#include "../../src/tu/projection.h"
#include "../../src/tu/diverge_interface_symmstruct.h"
#include "../models/all.h"

// is the set existing_r closed under the point group, and how often is the
// beyond_UC displacement of the symmetry maps non-zero?
static void diag( diverge_model_t* model, const char* tag ) {
    diverge_model_internals_tu( model, 3.1 );
    tu_projection_t* proj = tu_projection_init( model );
    tu_data_t* tu = model->internals->tu_data;
    tu_symmstruct_t* symm = tu->symm;

    const index_t nr = proj->nr, n_sym = model->n_sym;
    CT_PRINTF( "%s: n_sym=%li nr=%li nk=%li nkibz=%li n_orbff=%li\n", tag,
            (long)n_sym, (long)nr, (long)tu->nk, (long)tu->nkibz, (long)tu->n_orbff );

    // cartesian coordinates of the existing_r, folded into [0,nk)
    const index_t* nk = model->nk;
    // M = lattice^T maps lattice coordinates to cartesian: R_cart = M r
    double M[3][3], Minv[3][3];
    for (int e=0;e<3;++e) for (int d=0;d<3;++d) M[e][d] = model->lattice[d][e];
    double det = M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
               - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
               + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        int i1=(i+1)%3,i2=(i+2)%3,j1=(j+1)%3,j2=(j+2)%3;
        Minv[i][j] = (M[j1][i1]*M[j2][i2]-M[j1][i2]*M[j2][i1])/det;
    }

    index_t closed = 0, total = 0;
    for (index_t s=0; s<n_sym; ++s)
    for (index_t rv=0; rv<nr; ++rv) {
        double rc[3] = {0,0,0}, gr[3] = {0,0,0};
        for (int e=0; e<3; ++e) for (int d=0; d<3; ++d)
            rc[e] += M[e][d]*(double)proj->existing_r[rv].v[d];
        for (int e=0; e<3; ++e) for (int f=0; f<3; ++f)
            gr[e] += model->rs_symmetries[s][e][f]*rc[f];
        index_t gl[3]; bool integer = true;
        for (int d=0; d<3; ++d) {
            double v = 0;
            for (int e=0; e<3; ++e) v += Minv[d][e]*gr[e];
            double vr = round(v);
            if (fabs(v-vr) > 1e-6) integer = false;
            gl[d] = (((index_t)vr) % nk[d] + nk[d]) % nk[d];
        }
        total++;
        if (!integer) continue;
        for (index_t rw=0; rw<nr; ++rw)
            if (proj->existing_r[rw].v[0] == gl[0] &&
                proj->existing_r[rw].v[1] == gl[1] &&
                proj->existing_r[rw].v[2] == gl[2]) { closed++; break; }
    }
    CT_PRINTF( "%s: existing_r closed under point group: %li/%li\n", tag,
            (long)closed, (long)total );

    // size of the r-set needed by the fused operator: u = g^-1 R_r + d_R - d_L
    // (in lattice coordinates, modulo the k-mesh)
    index_t nd = symm->beyond_UC_sz/3;
    index_t* dlat = calloc(3*nd, sizeof(index_t));
    index_t ndist = 0;
    index_t* dist = calloc(3*nd, sizeof(index_t));
    for (index_t i=0; i<nd; ++i) {
        for (int d=0; d<3; ++d) {
            double v = 0;
            for (int e=0; e<3; ++e) v += Minv[d][e]*symm->beyond_UC[3*i+e];
            dlat[3*i+d] = (index_t)round(v);
        }
        bool seen = false;
        for (index_t j=0; j<ndist; ++j)
            if (dist[3*j+0]==dlat[3*i+0] && dist[3*j+1]==dlat[3*i+1] && dist[3*j+2]==dlat[3*i+2])
                { seen = true; break; }
        if (!seen) { for (int d=0;d<3;++d) dist[3*ndist+d]=dlat[3*i+d]; ndist++; }
    }
    CT_PRINTF( "%s: distinct beyond_UC displacements: %li\n", tag, (long)ndist );

    // enumerate u over all symmetries, all r and all pairs of displacements
    index_t ucap = 1<<22, usz = 0;
    char* seen_u = calloc( nk[0]*nk[1]*nk[2], 1 );
    (void)ucap;
    for (index_t sy=0; sy<n_sym; ++sy) {
        double GL[3][3];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
            double v = 0;
            for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                v += Minv[i][a]*model->rs_symmetries[sy][a][b]*M[b][j];
            GL[i][j] = v;
        }
        for (index_t rv=0; rv<nr; ++rv)
        for (index_t a=0; a<ndist; ++a)
        for (index_t b=0; b<ndist; ++b) {
            index_t u[3];
            for (int d=0; d<3; ++d) {
                double v = 0;
                for (int e=0; e<3; ++e) v += GL[d][e]*(double)proj->existing_r[rv].v[e];
                v += (double)(dist[3*a+d] - dist[3*b+d]);
                u[d] = (((index_t)llround(v)) % nk[d] + nk[d]) % nk[d];
            }
            index_t key = IDX3(u[0],u[1],u[2],nk[1],nk[2]);
            if (!seen_u[key]) { seen_u[key] = 1; usz++; }
        }
    }
    CT_PRINTF( "%s: |U| = %li  (nr=%li, n_sym*nr=%li) -> gemm flop factor %.2fx\n", tag,
            (long)usz, (long)nr, (long)(n_sym*nr), (double)(n_sym*nr)/(double)usz );
    free(seen_u); free(dlat); free(dist);

    // how many of the beyond_UC displacements are non-zero?
    index_t nz = 0, nb = symm->beyond_UC_sz/3;
    for (index_t i=0; i<nb; ++i) {
        double n2 = 0; for (int d=0; d<3; ++d) n2 += POW2(symm->beyond_UC[3*i+d]);
        if (n2 > 1e-16) nz++;
    }
    CT_PRINTF( "%s: beyond_UC non-zero: %li/%li\n", tag, (long)nz, (long)nb );

    // multiplicity of the one-to-many symmetry maps
    index_t maxlen = 0, sumlen = 0;
    for (index_t i=0; i<symm->o2m_map_sz; ++i) {
        maxlen = MAX(maxlen, symm->o2m_map_len[i]);
        sumlen += symm->o2m_map_len[i];
    }
    CT_PRINTF( "%s: o2m map len: max=%li avg=%.3f\n", tag, (long)maxlen,
            (double)sumlen/(double)symm->o2m_map_sz );

    tu_projection_free(proj);
    diverge_model_free(model);
}

CT_TEST_NAME_HIDDEN( sq, "diag,square" ) {
    diag( KWARGS(gen_square_lat_hub_sym, .nkx=24, .nky=24), "square" );
}
CT_TEST_NAME_HIDDEN( hc, "diag,honeycomb" ) {
    diag( KWARGS(gen_honeycomb_lat_hub_sym, .nkx=24, .nky=24), "honeycomb" );
}
CT_TEST_NAME_HIDDEN( kag, "diag,kagome" ) {
    diag( KWARGS(gen_kagome_model, .nkx=24, .nky=24), "kagome" );
}


// element-wise check of the fused kernel against the reference kernel:
//   E(q,r) pref(L,q) conj(pref(R,q))  ==  Etil(q_ibz,u)/w * c(L,s) conj(c(R,s))
static void kernel_check( diverge_model_t* model, const char* tag ) {
    diverge_model_internals_tu( model, 1.2 );
    tu_projection_t* proj = tu_projection_init( model );
    tu_data_t* tu = model->internals->tu_data;
    tu_symmstruct_t* symm = tu->symm;
    const index_t n_orbff = tu->n_orbff, n_spin = model->n_spin;
    const index_t legdim = POW2(n_spin)*n_orbff, nr = proj->nr;
    const index_t my_nk_in_bz = symm->my_nk_in_bz, my_nk_in_bz_off = symm->my_nk_in_bz_off;

    double M[3][3];
    for (int e=0;e<3;++e) for (int d=0;d<3;++d) M[e][d] = model->lattice[d][e];

    double worst_inv = 0.0, worst_fwd = 0.0;
    index_t legmiss = 0;
    for (index_t q=0; q<my_nk_in_bz; ++q) {
        const index_t q_in_full = symm->from_ibz_2_bz[q + my_nk_in_bz_off];
        const index_t qibz = symm->kmaps_to[q_in_full];
        const index_t sy = symm->which_symmetry_used[q_in_full].sym;
        const index_t isym = symm->which_symmetry_used[q_in_full].isym;
        const double* kibz = model->internals->kmesh + 3*symm->idx_ibz_in_fullmesh[qibz];

        for (index_t R=0; R<legdim; ++R)
        for (index_t L=0; L<legdim; ++L) {
            const index_t offR = symm->o2m_map_off[R*my_nk_in_bz+q];
            const index_t offL = symm->o2m_map_off[L*my_nk_in_bz+q];
            if (symm->idx_map_symm[offR] != symm->mi_maps_to[symm->mi_map_off[R+legdim*isym]]) legmiss++;
            if (symm->idx_map_symm[offL] != symm->mi_maps_to[symm->mi_map_off[L+legdim*isym]]) legmiss++;
            const complex128_t cR = symm->mi_prefac_in_in[symm->mi_map_off[R+legdim*isym]];
            const complex128_t cL = symm->mi_prefac_in_in[symm->mi_map_off[L+legdim*isym]];

            for (index_t rv=0; rv<nr; rv+=7) {
                const complex128_t lhs = proj->my_prefac_reco[rv + nr*q] *
                    symm->prefactor_symmP[offL] * conj(symm->prefactor_symmP[offR]);
                // u = G^{+-1} R_r + d_R - d_L, both directions tried
                for (int dir=0; dir<2; ++dir) {
                    double uc[3] = {0,0,0};
                    for (int e=0;e<3;++e) for (int f=0;f<3;++f) {
                        double Sef = dir ? model->rs_symmetries[sy][e][f] : model->rs_symmetries[sy][f][e];
                        double Rc = 0; for (int d=0;d<3;++d) Rc += M[f][d]*(double)proj->existing_r[rv].v[d];
                        uc[e] += Sef*Rc;
                    }
                    for (int e=0;e<3;++e)
                        uc[e] += symm->beyond_UC[3*(R+legdim*isym)+e] - symm->beyond_UC[3*(L+legdim*isym)+e];
                    double kr = 0; for (int e=0;e<3;++e) kr += kibz[e]*uc[e];
                    const complex128_t rhs = cexp(I*kr)*cL*conj(cR);
                    double dev = cabs(lhs-rhs);
                    if (dir) worst_fwd = MAX(worst_fwd,dev); else worst_inv = MAX(worst_inv,dev);
                }
            }
        }
    }
    CT_PRINTF( "kernel %s: legmap mismatches=%li  worst dev: S^-1=%.3e  S=%.3e\n",
            tag, (long)legmiss, worst_inv, worst_fwd );
    tu_projection_free(proj); diverge_model_free(model);
}

// is the symmetry list a closed group action on the k-mesh, i.e. is every point
// of an orbit hit by the same number of symmetries?
static void orbit_check( diverge_model_t* model, const char* tag ) {
    diverge_model_internals_tu( model, 1.2 );
    tu_data_t* tu = model->internals->tu_data;
    const tu_symmstruct_t* symm = tu->symm;
    const index_t n_sym = model->n_sym, nk = kdim(model->nk);
    const index_t* mom = model->internals->symm_map_mom_crs;
    index_t* cnt = calloc(nk, sizeof(index_t));
    index_t bad = 0, checked = 0, free_pts = 0;
    for (index_t qi=0; qi<tu->nkibz; ++qi) {
        const index_t q_in_full = symm->idx_ibz_in_fullmesh[qi];
        ZERO_OUT( cnt, nk );
        for (index_t sy=0; sy<n_sym; ++sy) cnt[mom[sy + n_sym*q_in_full]]++;
        index_t ref = -1, norbit = 0;
        for (index_t p=0; p<nk; ++p) if (cnt[p]) {
            norbit++;
            if (ref < 0) ref = cnt[p];
            else if (cnt[p] != ref) bad++;
        }
        checked++;
        if (norbit != symm->from_ibz_2_bz_len[qi]) bad++;
        if (norbit == n_sym && ref == 1) free_pts++;
    }
    CT_PRINTF( "orbits %s: n_sym=%li inconsistent=%li/%li  free(|orbit|==n_sym, hit once)=%li/%li = %.1f%%\n",
            tag, (long)n_sym, (long)bad, (long)checked, (long)free_pts, (long)checked,
            100.0*(double)free_pts/(double)checked );
    free(cnt);
    diverge_model_free(model);
}

// how many of the listed symmetries share a momentum action, and is the orbit
// multiplicity uniform once the duplicates are collapsed?
static void dup_check( diverge_model_t* model, const char* tag ) {
    diverge_model_internals_tu( model, 1.2 );
    tu_data_t* tu = model->internals->tu_data;
    const tu_symmstruct_t* symm = tu->symm;
    const index_t n_sym = model->n_sym, nk = kdim(model->nk);
    const index_t* mom = model->internals->symm_map_mom_crs;

    index_t* rep = calloc(n_sym, sizeof(index_t));   // representative of each class
    index_t* mult = calloc(n_sym, sizeof(index_t));  // size of the class
    index_t ndist = 0;
    for (index_t sy=0; sy<n_sym; ++sy) {
        rep[sy] = -1;
        for (index_t t=0; t<sy; ++t) {
            bool same = true;
            for (index_t k=0; k<nk; ++k)
                if (mom[sy+n_sym*k] != mom[t+n_sym*k]) { same = false; break; }
            if (same) { rep[sy] = rep[t]; break; }
        }
        if (rep[sy] < 0) rep[sy] = ndist++;
    }
    for (index_t sy=0; sy<n_sym; ++sy) mult[rep[sy]]++;

    // with duplicates collapsed: is every orbit point hit equally often?
    index_t* cnt = calloc(nk, sizeof(index_t));
    index_t bad = 0;
    for (index_t qi=0; qi<tu->nkibz; ++qi) {
        const index_t q_in_full = symm->idx_ibz_in_fullmesh[qi];
        ZERO_OUT( cnt, nk );
        for (index_t sy=0; sy<n_sym; ++sy) cnt[mom[sy+n_sym*q_in_full]] += 1;
        index_t ref = -1;
        for (index_t p=0; p<nk; ++p) if (cnt[p]) {
            // divide out the class size of the symmetries that reach p
            if (ref < 0) ref = cnt[p]; else if (cnt[p] != ref) bad++;
        }
    }
    CT_PRINTF( "dups %s: n_sym=%li distinct momentum actions=%li  class sizes:", tag,
            (long)n_sym, (long)ndist );
    for (index_t c=0; c<ndist; ++c) CT_PRINTF( " %li", (long)mult[c] );
    CT_PRINTF( "  non-uniform orbits (raw)=%li/%li\n", (long)bad, (long)tu->nkibz );
    free(rep); free(mult); free(cnt);
    diverge_model_free(model);
}

// do symmetries sharing a momentum action also share leg map, prefactor and
// unit cell displacement?  (if not, they cannot simply be averaged)
static void dupleg_check( diverge_model_t* model, const char* tag ) {
    diverge_model_internals_tu( model, 1.2 );
    tu_data_t* tu = model->internals->tu_data;
    const tu_symmstruct_t* symm = tu->symm;
    const index_t n_sym = model->n_sym, nk = kdim(model->nk);
    const index_t n_spin = model->n_spin, n_orbff = tu->n_orbff;
    const index_t legdim = POW2(n_spin)*n_orbff;
    const index_t* mom = model->internals->symm_map_mom_crs;

    index_t legdiff = 0, dvecdiff = 0, npairs = 0;
    double cinmax = 0.0, coutmax = 0.0;
    for (index_t a=0; a<n_sym; ++a)
    for (index_t b=a+1; b<n_sym; ++b) {
        bool same = true;
        for (index_t k=0; k<nk; ++k)
            if (mom[a+n_sym*k] != mom[b+n_sym*k]) { same = false; break; }
        if (!same) continue;
        npairs++;
        const index_t ia = symm->inv_symmetry[a], ib = symm->inv_symmetry[b];
        for (index_t l=0; l<legdim; ++l) {
            const index_t oa = symm->mi_map_off[l+legdim*ia], ob = symm->mi_map_off[l+legdim*ib];
            if (symm->mi_maps_to[oa] != symm->mi_maps_to[ob]) legdiff++;
            cinmax  = MAX(cinmax,  cabs(symm->mi_prefac_in_in[oa]  - symm->mi_prefac_in_in[ob]));
            coutmax = MAX(coutmax, cabs(symm->mi_prefac_in_out[oa] - symm->mi_prefac_in_out[ob]));
            for (int d=0; d<3; ++d)
                if (fabs(symm->beyond_UC[3*(l+legdim*ia)+d]-symm->beyond_UC[3*(l+legdim*ib)+d])>1e-9)
                    dvecdiff++;
        }
    }
    CT_PRINTF( "dupleg %s: pairs=%li legmap diffs=%li  d diffs=%li  max|dc_in|=%.3e  max|dc_out|=%.3e\n",
            tag, (long)npairs, (long)legdiff, (long)dvecdiff, cinmax, coutmax );
    diverge_model_free(model);
}

CT_TEST_NAME_HIDDEN( duplg, "diag,duplegs" ) {
    dupleg_check( KWARGS(gen_square_lat_rashba_sym, .nkx=12, .nky=12), "rashba" );
}

CT_TEST_NAME_HIDDEN( dup, "diag,dups" ) {
    dup_check( KWARGS(gen_square_lat_rashba_sym, .nkx=12, .nky=12), "rashba" );
    dup_check( KWARGS(gen_square_lat_hub_sym, .nkx=12, .nky=12), "hub" );
}

CT_TEST_NAME_HIDDEN( orb, "diag,orbits" ) {
    orbit_check( KWARGS(gen_square_lat_hub_sym, .nkx=64, .nky=64), "hub" );
    orbit_check( KWARGS(gen_square_lat_rashba_sym, .nkx=64, .nky=64), "rashba" );
    orbit_check( KWARGS(gen_honeycomb_lat_hub_sym, .nkx=48, .nky=48), "honeycomb" );
    orbit_check( KWARGS(gen_kagome_model, .nkx=48, .nky=48), "kagome" );
}

CT_TEST_NAME_HIDDEN( kern, "diag,kernel" ) {
    kernel_check( KWARGS(gen_square_lat_hub_sym, .nkx=6, .nky=6), "hub" );
    kernel_check( KWARGS(gen_square_lat_rashba_sym, .nkx=6, .nky=6), "rashba" );
}

#endif
