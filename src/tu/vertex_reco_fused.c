/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

/*
 * Fusion of the symmetry reconstruction into the real space Fourier transform.
 *
 * The two-step implementation reconstructs the vertex on the full BZ and then
 * contracts it with exp(i k_q R_r). Written out, with R,L the two composite legs
 * (spin,spin,bond), q = S_s q_ibz a BZ point and s the symmetry that generated
 * it,
 *
 *   buf(R,L,r) = 1/nk sum_q exp(i k_q R_r) pref(L,q) conj(pref(R,q))
 *                            X(q_ibz, sigma_s(R), sigma_s(L)),
 *   pref(l,q)  = c(l,s) exp(-i k_q S_s d(l,s)).
 *
 * Since the symmetries are orthogonal, k_q S_s d = (S_s k_ibz)(S_s d) = k_ibz d,
 * and exp(i k_q R_r) = exp(i k_ibz S_s^-1 R_r). The whole q-dependence of the
 * prefactors therefore collapses onto the IBZ momentum, and all three factors
 * combine into a *single* plane wave of the IBZ momentum,
 *
 *   buf(R,L,r) = sum_s sum_{l,r'} c(l,s) conj(c(r',s)) F( u(s,r,R,L), r', l )
 *   F(u,R',L') = 1/nk sum_{q_ibz} w(q_ibz) exp(i k_ibz R_u) X(q_ibz,R',L')
 *   u(s,r,R,L) = S_s^-1 R_r - d(L,s) + d(R,s),
 *
 * where l and r' run over the images of L and R under s. For a one dimensional
 * orbital irrep there is exactly one of each, and A_s is a monomial matrix on
 * the leg index; multi dimensional ones (p_x/p_y, spin mixing) send a leg to a
 * superposition of several, and the only thing that changes is that the real
 * space gather sums over the images. F never sees the symmetry maps, so the
 * GEMM -- where the entire gain sits -- is the same either way, and so is the
 * set of u: the displacement d is a property of the leg, not of the individual
 * image (see tu_reco_fused_init).
 *
 * with w(q_ibz) = |orbit(q_ibz)|/n_sym the weight that turns the sum over the
 * orbit into a sum over the full point group. F is one GEMM over the *IBZ*
 * vertex, and the symmetry sum is left as a gather over the (small) real space
 * result. Its cost is |U| nk_ibz rest instead of nr nk_bz rest, i.e. smaller by
 * n_sym nr / |U|, and the reconstructed full-BZ vertex is never formed.
 *
 * u only ever leaves the set of real space vectors of the projection because of
 * the unit cell displacements d, of which there are very few distinct ones, so
 * |U| stays close to nr.
 */

// this must precede the other includes to pre-define the types
#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

#include "vertex_reco_fused.h"
#include "vertex_details.h"
#include "projection.h"
#include "diverge_interface_symmstruct.h"
#include "../diverge_model_internals.h"
#include "../misc/gemm.h"
#include "../misc/mpi_functions.h"
#include "../misc/invert.h"
#include <stddef.h>

struct tu_reco_fused_t {
    index_t nu;         // number of real space vectors of the fused GEMM
    index_t nr;         // real space vectors of the projection (output)
    index_t n_sym;
    index_t legdim;     // n_spin^2 * n_orbff
    index_t rest;       // legdim^2
    index_t my_nk;
    index_t ndmax;      // max number of distinct unit cell displacements
    index_t ncls;       // number of distinct momentum actions in the symmetry list

    complex128_t* Etil; // (my_nk, nu)   fused exponentials, weights folded in
    complex128_t* F;    // (rest, nu)    GEMM result

    /* The symmetry acts on the composite leg index as a sparse unitary A_s. For
     * one dimensional orbital irreps that is a monomial matrix -- one image per
     * leg -- but multi dimensional ones (p_x/p_y, spin mixing) send a leg to a
     * superposition of several. The images are therefore kept as a CSR list over
     * (symmetry, leg) instead of a dense per-leg table. */
    index_t n_ent;      // total number of map entries, i.e. sum of mlen
    index_t* moff;      // (n_sym, legdim)              offset into the entry lists
    index_t* mlen;      // (n_sym, legdim)              number of images of l
    index_t* ment;      // (n_ent,)                     image legs sigma_s(l)
    complex128_t* cP;   // (n_ent,)                     P prefactors
    complex128_t* cCD;  // (n_ent,)                     C/D prefactors
    index_t* dcls;      // (n_sym, legdim)              displacement class of l
    index_t* uidx;      // (n_sym, ndmax, ndmax, nr)    index of u into Etil
    double* wsym;       // (n_sym,)   1/(number of entries sharing a momentum action)
};

index_t tu_reco_fused_nu( const tu_reco_fused_t* f ) { return f ? f->nu : -1; }
index_t tu_reco_fused_nr( const tu_reco_fused_t* f ) { return f ? f->nr : -1; }

void tu_reco_fused_free( tu_reco_fused_t* f ) {
    if (!f) return;
    free(f->Etil); free(f->F); free(f->moff); free(f->mlen); free(f->ment);
    free(f->dcls); free(f->uidx); free(f->cP); free(f->cCD); free(f->wsym);
    free(f);
}

tu_reco_fused_t* tu_reco_fused_init( const diverge_model_t* model, const tu_projection_t* proj ) {
    const tu_data_t* tu = model->internals->tu_data;
    const tu_symmstruct_t* symm = tu->symm;
    if (!symm->use_symmetries || model->n_sym < 1) return NULL;

    const index_t n_sym = model->n_sym, n_spin = model->n_spin, n_orbff = tu->n_orbff;
    const index_t legdim = POW2(n_spin)*n_orbff, nr = proj->nr;
    const index_t nkx = model->nk[0], nky = model->nk[1], nkz = model->nk[2];
    const index_t nk_tot = nkx*nky*nkz;

    const claM3d_t M = claM3d_transpose( claM3d_map(model->lattice[0]) ),
                   Minv = claM3d_scale( claM3d_map(model->internals->rlattice.v[0]), 1./(2.*M_PI) );

    tu_reco_fused_t* f = calloc(1, sizeof*f);
    f->n_sym = n_sym; f->legdim = legdim; f->rest = POW2(legdim);
    f->nr = nr; f->my_nk = tu->my_nk;
    f->moff = calloc(n_sym*legdim, sizeof(index_t));
    f->mlen = calloc(n_sym*legdim, sizeof(index_t));
    f->dcls = calloc(n_sym*legdim, sizeof(index_t));

    // the maps are stored per (leg, symmetry) with the leg index running
    // fastest, exactly as the composite leg index does
    #define MI_IDX(s,l) ((l) + legdim*symm->inv_symmetry[s])

    // lay out the CSR image lists
    for (index_t s=0; s<n_sym; ++s)
    for (index_t l=0; l<legdim; ++l) {
        const index_t len = symm->mi_map_len[MI_IDX(s,l)];
        if (len < 1)
            mpi_err_printf( "fused reco_ft: leg %li has no image under symmetry %li\n",
                    (long)l, (long)s );
        f->moff[IDX2(s,l,legdim)] = f->n_ent;
        f->mlen[IDX2(s,l,legdim)] = len;
        f->n_ent += len;
    }
    f->ment = calloc(f->n_ent, sizeof(index_t));
    f->cP   = calloc(f->n_ent, sizeof(complex128_t));
    f->cCD  = calloc(f->n_ent, sizeof(complex128_t));

    // per symmetry: leg maps, prefactors and the distinct unit cell displacements
    claV3ll_t* dset = calloc(n_sym*legdim, sizeof(claV3ll_t)); // distinct displacements
    index_t* nd     = calloc(n_sym, sizeof(index_t));
    for (index_t s=0; s<n_sym; ++s) {
        for (index_t l=0; l<legdim; ++l) {
            const index_t idx = MI_IDX(s,l);
            const index_t off = symm->mi_map_off[idx], len = symm->mi_map_len[idx];
            const index_t dst = f->moff[IDX2(s,l,legdim)];
            for (index_t e=0; e<len; ++e) {
                f->ment[dst+e] = symm->mi_maps_to[off+e];
                f->cP[dst+e]   = symm->mi_prefac_in_in[off+e];
                f->cCD[dst+e]  = symm->mi_prefac_in_out[off+e];
            }

            /* beyond_UC is pushed once per map entry, but the value pushed is
             * beyond_UC_in[(s, o2(l))] (diverge_interface_symmstruct.c) and the
             * orbital o2 is fixed by the leg alone -- so all images of a leg
             * share its displacement even when there are several of them. That
             * is what keeps the set of fused real space vectors, and with it the
             * width of the GEMM, independent of the size of the leg maps. */
            const claV3d_t d0 = claV3d_map( symm->beyond_UC + 3*off );
            for (index_t e=1; e<len; ++e)
                if (claV3d_norm( claV3d_sub( claV3d_map(symm->beyond_UC + 3*(off+e)), d0 ) ) > 1e-9)
                    mpi_err_printf( "fused reco_ft: images of leg %li under symmetry %li "
                            "disagree on the unit cell displacement\n", (long)l, (long)s );

            const claV3d_t dc = claM3d_matvec( Minv, d0 );
            claV3ll_t dl;
            for (int d=0; d<3; ++d) dl.v[d] = lroundl( dc.v[d] );

            index_t cls = -1;
            for (index_t c=0; c<nd[s]; ++c)
                if (claV3ll_equal( dset[IDX2(s,c,legdim)], dl )) { cls = c; break; }
            if (cls < 0) {
                cls = nd[s]++;
                dset[IDX2(s,cls,legdim)] = dl;
            }
            f->dcls[IDX2(s,l,legdim)] = cls;
        }
    }
    #undef MI_IDX
    index_t ndmax = 0;
    for (index_t s=0; s<n_sym; ++s) ndmax = MAX(ndmax, nd[s]);
    f->ndmax = ndmax;

    // u(s,r,a,b) = S_s^-1 R_r + d_a - d_b, in lattice coordinates modulo the mesh
    index_t* u2i = NULL;
    {
        f->uidx = calloc(n_sym*ndmax*ndmax*nr, sizeof(index_t));
        u2i = malloc(nk_tot*sizeof(index_t));
        for (index_t i=0; i<nk_tot; ++i) u2i[i] = -1;
        claV3ll_t* ulist = calloc(n_sym*ndmax*ndmax*nr, sizeof(claV3ll_t));
        const claV3ll_t nkv = {{ nkx, nky, nkz }};

        for (index_t s=0; s<n_sym; ++s) {
            // S_s^-1 in lattice coordinates -- integral, the symmetries are orthogonal
            const claM3d_t GLd = claM3d_matmul( Minv,
                    claM3d_matmul( claM3d_transpose( claM3d_map(model->rs_symmetries[s][0]) ), M ) );
            claM3ll_t GL;
            for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) GL.m[i][j] = lroundl( GLd.m[i][j] );

            for (index_t a=0; a<nd[s]; ++a)
            for (index_t b=0; b<nd[s]; ++b)
            for (index_t rv=0; rv<nr; ++rv) {
                const claV3i_t ri = proj->existing_r[rv];
                const claV3ll_t Rr = {{ ri.v[0], ri.v[1], ri.v[2] }};
                claV3ll_t u = claV3ll_add( claM3ll_matvec( GL, Rr ),
                        claV3ll_sub( dset[IDX2(s,a,legdim)], dset[IDX2(s,b,legdim)] ) );
                for (int d=0; d<3; ++d) u.v[d] = ((u.v[d] % nkv.v[d]) + nkv.v[d]) % nkv.v[d];
                const index_t key = IDX3(u.v[0],u.v[1],u.v[2],nky,nkz);
                if (u2i[key] < 0) { u2i[key] = f->nu; ulist[f->nu] = u; f->nu++; }
                f->uidx[IDX4(s,a,b,rv,ndmax,ndmax,nr)] = u2i[key];
            }
        }

        {
            const index_t* mom = model->internals->symm_map_mom_crs;
            const index_t nk_tot_mom = symm->nk;

            /* The symmetry list may contain several entries with the same action
             * on the momenta -- divERGe appends inverses that spin-1/2 rotations
             * do not match up to the sign of their SU(2) representative. Their
             * contributions have to be averaged, not summed, so each entry is
             * weighted with the inverse size of its class. */
            index_t* cls = calloc(n_sym, sizeof(index_t));
            index_t* csz = calloc(n_sym, sizeof(index_t));
            index_t* crep = calloc(n_sym, sizeof(index_t));
            index_t ncls = 0;
            for (index_t sy=0; sy<n_sym; ++sy) {
                cls[sy] = -1;
                for (index_t c=0; c<ncls; ++c) {
                    bool same = true;
                    for (index_t k=0; k<nk_tot_mom; ++k)
                        if (mom[sy+n_sym*k] != mom[crep[c]+n_sym*k]) { same = false; break; }
                    if (same) { cls[sy] = c; break; }
                }
                if (cls[sy] < 0) { crep[ncls] = sy; cls[sy] = ncls++; }
                csz[cls[sy]]++;
            }
            f->wsym = calloc(n_sym, sizeof(double));
            for (index_t sy=0; sy<n_sym; ++sy) f->wsym[sy] = 1.0/(double)csz[cls[sy]];

            /* Replacing the sum over the orbit of an IBZ momentum by a sum over
             * the (deduplicated) symmetry list is exact iff every point of that
             * orbit is reached equally often. The symmetry list is a group --
             * divERGe completes it with the identity and the missing inverses
             * (ensure_inverse_symmetries_and_id) -- and the IBZ reduction assigns
             * every BZ point to the representative that reaches it in one step
             * (diverge_interface_symmstruct.c), so the images of an IBZ momentum
             * are its recorded orbit and the stabiliser hits each of them equally
             * often. There is no reconstruction path besides the fused one, so
             * the invariant is checked rather than assumed. */
            index_t* img = calloc(ncls, sizeof(index_t));
            index_t* cnt = calloc(ncls, sizeof(index_t));
            for (index_t q=0; q<f->my_nk; ++q) {
                const index_t q_in_full = symm->idx_ibz_in_fullmesh[q + tu->my_nk_off];
                index_t nimg = 0;
                for (index_t c=0; c<ncls; ++c) {
                    const index_t p = mom[crep[c] + n_sym*q_in_full];
                    index_t seen = -1;
                    for (index_t i=0; i<nimg; ++i) if (img[i] == p) { seen = i; break; }
                    if (seen < 0) { img[nimg] = p; cnt[nimg] = 1; nimg++; }
                    else cnt[seen]++;
                }
                bool uniform = (nimg == symm->from_ibz_2_bz_len[q + tu->my_nk_off]);
                for (index_t i=1; i<nimg && uniform; ++i) if (cnt[i] != cnt[0]) uniform = false;
                if (!uniform)
                    mpi_err_printf( "fused reco_ft: the symmetry list does not hit the orbit "
                            "of IBZ momentum %li uniformly -- it is not closed under "
                            "composition\n", (long)(q + tu->my_nk_off) );
            }
            free(img); free(cnt); free(cls); free(csz); free(crep);
            f->ncls = ncls;
        }

        // fused exponentials for the generic momenta
        {
            mpi_log_large_alloc( (f->my_nk + f->rest)*f->nu*(index_t)sizeof(complex128_t),
                    "the fused reco_ft tables" );
            f->Etil = calloc(f->my_nk*f->nu, sizeof(complex128_t));
            const double* kmesh = model->internals->kmesh;
            #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<f->my_nk; ++q) {
                const index_t qibz = q + tu->my_nk_off;
                const double* kp = kmesh + 3*symm->idx_ibz_in_fullmesh[qibz];
                const double w = (double)symm->from_ibz_2_bz_len[qibz]/(double)f->ncls;
                for (index_t u=0; u<f->nu; ++u) {
                    const claV3d_t ul = {{ (double)ulist[u].v[0], (double)ulist[u].v[1],
                                           (double)ulist[u].v[2] }};
                    const double kr = claV3d_dot( claV3d_map(kp), claM3d_matvec( M, ul ) );
                    f->Etil[IDX2(q,u,f->nu)] = w*cexp(I*kr);
                }
            }
            f->F = calloc(f->rest*f->nu, sizeof(complex128_t));
        }
        free(ulist);
    }

    free(dset); free(nd); free(u2i);
    mpi_vrb_printf("fused reco_ft: nu=%li (nr=%li, n_sym*nr=%li), %li entries for "
            "%li*%li legs\n", (long)f->nu, (long)nr, (long)(n_sym*nr), (long)f->n_ent,
            (long)n_sym, (long)legdim);
    return f;
}

void tu_reco_fused_ft( const tu_reco_fused_t* f, const tu_vertex_t* V, char chan,
        complex128_t* buf ) {
    const complex128_t* Xch = chan == 'P' ? V->Pch : chan == 'C' ? V->Cch : V->Dch;
    const complex128_t* c = chan == 'P' ? f->cP : f->cCD;
    const index_t nu = f->nu, nr = f->nr, legdim = f->legdim, rest = f->rest,
                  n_sym = f->n_sym, ndmax = f->ndmax;

    // F(u,R',L') = 1/nk sum_q exp(i k_q R_u) X(q,R',L')
    double t0 = diverge_mpi_wtime();
    single_gemm_Xt_Afortran( (single_gemm_Xt_t*)((ptrdiff_t)-1), Xch, f->Etil, f->F,
            nu, rest, f->my_nk, 1./(double)V->nk, 0.0 );
    double t1 = diverge_mpi_wtime();

    // symmetry sum, now in real space. The channel only decides which spin
    // labels sit on which leg, i.e. it is a permutation of the output slot that
    // the gather can write to directly.
    const index_t n_spin = V->n_spin, n_orbff = V->n_orbff;
    #pragma omp parallel for collapse(6) schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t a4=0; a4<n_spin; ++a4)
    for (index_t a3=0; a3<n_spin; ++a3)
    for (index_t mo3=0; mo3<n_orbff; ++mo3)
    for (index_t a2=0; a2<n_spin; ++a2)
    for (index_t a1=0; a1<n_spin; ++a1)
    for (index_t mo1=0; mo1<n_orbff; ++mo1) {
        const index_t R = IDX3(a4,a3,mo3,n_spin,n_orbff), L = IDX3(a2,a1,mo1,n_spin,n_orbff);
        // 'C': (s4,s3,mo3,s2,s1,mo1) <- (s2,s3,mo3,s4,s1,mo1)
        // 'D': (s4,s3,mo3,s2,s1,mo1) <- (s2,s4,mo3,s3,s1,mo1)
        index_t Rd = R, Ld = L;
        if (chan == 'C') { Rd = IDX3(a2,a3,mo3,n_spin,n_orbff); Ld = IDX3(a4,a1,mo1,n_spin,n_orbff); }
        if (chan == 'D') { Rd = IDX3(a3,a2,mo3,n_spin,n_orbff); Ld = IDX3(a4,a1,mo1,n_spin,n_orbff); }
        complex128_t* restrict out = buf + IDX2(IDX2(Rd,Ld,legdim),0,nr);
        for (index_t r=0; r<nr; ++r) out[r] = 0.0;
        for (index_t s=0; s<n_sym; ++s) {
            /* The exponent u = S_s^-1 R_r - d(L,s) + d(R,s) depends on the legs
             * only through their displacements, which all images of a leg share,
             * so the u lookup stays outside the sum over images. */
            const index_t* restrict ui = f->uidx + IDX4(s, f->dcls[IDX2(s,R,legdim)],
                    f->dcls[IDX2(s,L,legdim)], 0, ndmax, ndmax, nr);
            const index_t Lo = f->moff[IDX2(s,L,legdim)], Ln = f->mlen[IDX2(s,L,legdim)];
            const index_t Ro = f->moff[IDX2(s,R,legdim)], Rn = f->mlen[IDX2(s,R,legdim)];
            for (index_t li=0; li<Ln; ++li)
            for (index_t ri=0; ri<Rn; ++ri) {
                const complex128_t pref = f->wsym[s]*c[Lo+li]*conj(c[Ro+ri]);
                const complex128_t* restrict Fs =
                    f->F + nu*IDX2(f->ment[Ro+ri], f->ment[Lo+li], legdim);
                for (index_t r=0; r<nr; ++r) out[r] += pref*Fs[ui[r]];
            }
        }
    }
    double t2 = diverge_mpi_wtime();
#ifdef TU_PROJ_TIMING
    mpi_tim_printf( "reco_ft[fused] %c: gemm:%7.3fs gather:%7.3fs tot:%7.3fs "
            "(gemm %li x %li x %li, gather rest=%li n_sym=%li n_ent=%li nr=%li)\n",
            chan, t1-t0, t2-t1, t2-t0, (long)nu, (long)rest, (long)f->my_nk,
            (long)rest, (long)n_sym, (long)f->n_ent, (long)nr );
#else
    (void)t0; (void)t1; (void)t2;
#endif
}
