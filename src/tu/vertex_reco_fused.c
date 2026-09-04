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
 *   buf(R,L,r) = sum_s c(L,s) conj(c(R,s)) F( u(s,r,R,L), sigma_s(R), sigma_s(L) )
 *   F(u,R',L') = 1/nk sum_{q_ibz} w(q_ibz) exp(i k_ibz R_u) X(q_ibz,R',L')
 *   u(s,r,R,L) = S_s^-1 R_r - d(L,s) + d(R,s),
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

#include "vertex_reco_fused.h"
#include "vertex_details.h"
#include "projection.h"
#include "diverge_interface_symmstruct.h"
#include "../diverge_model_internals.h"
#include "../misc/gemm.h"
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

    index_t* legmap;    // (n_sym, legdim)              sigma_s(l)
    index_t* dcls;      // (n_sym, legdim)              displacement class of l
    index_t* uidx;      // (n_sym, ndmax, ndmax, nr)    index of u into Etil
    complex128_t* cP;   // (n_sym, legdim)              P prefactors
    complex128_t* cCD;  // (n_sym, legdim)              C/D prefactors
    double* wsym;       // (n_sym,)   1/(number of entries sharing a momentum action)

    // high symmetry momenta, whose orbit is not hit uniformly by the symmetry
    // list, are left to the two-step reconstruction
    index_t nslow;      // number of BZ points on the two-step path
    index_t* slowq;     // (nslow,)         BZ points, local indexing
    complex128_t* Eslow;// (nslow, nr)      their exponentials
    complex128_t* Qslow;// (nslow, rest)    their reconstructed vertex
};

index_t tu_reco_fused_nu( const tu_reco_fused_t* f ) { return f ? f->nu : -1; }
index_t tu_reco_fused_nr( const tu_reco_fused_t* f ) { return f ? f->nr : -1; }
index_t tu_reco_fused_nslow( const tu_reco_fused_t* f ) { return f ? f->nslow : -1; }

void tu_reco_fused_free( tu_reco_fused_t* f ) {
    if (!f) return;
    free(f->Etil); free(f->F); free(f->legmap); free(f->dcls);
    free(f->uidx); free(f->cP); free(f->cCD); free(f->wsym);
    free(f->slowq); free(f->Eslow); free(f->Qslow); free(f);
}

// 3x3 inverse, returns false if singular
static bool mat3_inv( const double A[3][3], double Ainv[3][3] );
static bool mat3_inv( const double A[3][3], double Ainv[3][3] ) {
    const double det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                     - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                     + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if (fabs(det) < 1e-12) return false;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
        const int i1=(i+1)%3, i2=(i+2)%3, j1=(j+1)%3, j2=(j+2)%3;
        Ainv[i][j] = (A[j1][i1]*A[j2][i2] - A[j1][i2]*A[j2][i1])/det;
    }
    return true;
}

// round to integer, reporting whether the value actually was one
static index_t to_int( double v, bool* ok ) {
    const double r = round(v);
    if (fabs(v-r) > 1e-6) *ok = false;
    return (index_t)r;
}

tu_reco_fused_t* tu_reco_fused_init( const diverge_model_t* model, const tu_projection_t* proj ) {
    const tu_data_t* tu = model->internals->tu_data;
    const tu_symmstruct_t* symm = tu->symm;
    if (!symm->use_symmetries || model->n_sym < 1) return NULL;

    const index_t n_sym = model->n_sym, n_spin = model->n_spin, n_orbff = tu->n_orbff;
    const index_t legdim = POW2(n_spin)*n_orbff, nr = proj->nr;
    const index_t nkx = model->nk[0], nky = model->nk[1], nkz = model->nk[2];
    const index_t nk_tot = nkx*nky*nkz;

    // M maps lattice coordinates to cartesian ones
    double M[3][3], Minv[3][3];
    for (int e=0; e<3; ++e) for (int d=0; d<3; ++d) M[e][d] = model->lattice[d][e];
    if (!mat3_inv((const double(*)[3])M, Minv)) return NULL;

    bool ok = true;
    tu_reco_fused_t* f = calloc(1, sizeof*f);
    f->n_sym = n_sym; f->legdim = legdim; f->rest = POW2(legdim);
    f->nr = nr; f->my_nk = tu->my_nk;
    f->legmap = calloc(n_sym*legdim, sizeof(index_t));
    f->dcls   = calloc(n_sym*legdim, sizeof(index_t));
    f->cP     = calloc(n_sym*legdim, sizeof(complex128_t));
    f->cCD    = calloc(n_sym*legdim, sizeof(complex128_t));

    // per symmetry: leg map, prefactors and the distinct unit cell displacements
    index_t* dlat = calloc(n_sym*legdim*3, sizeof(index_t)); // displacement per leg
    index_t* dset = calloc(n_sym*legdim*3, sizeof(index_t)); // distinct ones per symmetry
    index_t* nd   = calloc(n_sym, sizeof(index_t));
    for (index_t s=0; s<n_sym && ok; ++s) {
        const index_t isym = symm->inv_symmetry[s];
        for (index_t l=0; l<legdim; ++l) {
            // the maps are stored per (leg, symmetry) with the leg index running
            // fastest, exactly as the composite leg index does
            const index_t idx = l + legdim*isym;
            if (symm->mi_map_len[idx] != 1) { ok = false; break; } // not a plain permutation
            const index_t off = symm->mi_map_off[idx];
            f->legmap[IDX2(s,l,legdim)] = symm->mi_maps_to[off];
            f->cP[IDX2(s,l,legdim)]  = symm->mi_prefac_in_in[off];
            f->cCD[IDX2(s,l,legdim)] = symm->mi_prefac_in_out[off];

            index_t dl[3];
            for (int d=0; d<3; ++d) {
                double v = 0.0;
                for (int e=0; e<3; ++e) v += Minv[d][e]*symm->beyond_UC[3*idx+e];
                dl[d] = to_int(v, &ok);
            }
            for (int d=0; d<3; ++d) dlat[IDX3(s,l,d,legdim,3)] = dl[d];

            index_t cls = -1;
            for (index_t c=0; c<nd[s]; ++c)
                if (dset[IDX3(s,c,0,legdim,3)] == dl[0] && dset[IDX3(s,c,1,legdim,3)] == dl[1] &&
                    dset[IDX3(s,c,2,legdim,3)] == dl[2]) { cls = c; break; }
            if (cls < 0) {
                cls = nd[s]++;
                for (int d=0; d<3; ++d) dset[IDX3(s,cls,d,legdim,3)] = dl[d];
            }
            f->dcls[IDX2(s,l,legdim)] = cls;
        }
    }
    index_t ndmax = 0;
    for (index_t s=0; s<n_sym; ++s) ndmax = MAX(ndmax, nd[s]);
    f->ndmax = ndmax;

    // u(s,r,a,b) = S_s^-1 R_r + d_a - d_b, in lattice coordinates modulo the mesh
    index_t* u2i = NULL;
    if (ok) {
        f->uidx = calloc(n_sym*ndmax*ndmax*nr, sizeof(index_t));
        u2i = malloc(nk_tot*sizeof(index_t));
        for (index_t i=0; i<nk_tot; ++i) u2i[i] = -1;
        index_t* ulist = calloc(n_sym*ndmax*ndmax*nr, sizeof(index_t));

        for (index_t s=0; s<n_sym && ok; ++s) {
            // S_s^-1 in lattice coordinates (the symmetries are orthogonal)
            double GL[3][3];
            for (int i=0; i<3 && ok; ++i) for (int j=0; j<3; ++j) {
                double v = 0.0;
                for (int a=0; a<3; ++a) for (int b=0; b<3; ++b)
                    v += Minv[i][a]*model->rs_symmetries[s][b][a]*M[b][j];
                GL[i][j] = v;
                if (fabs(v-round(v)) > 1e-6) ok = false;
            }
            for (index_t a=0; a<nd[s]; ++a)
            for (index_t b=0; b<nd[s]; ++b)
            for (index_t rv=0; rv<nr; ++rv) {
                index_t u[3];
                for (int d=0; d<3; ++d) {
                    double v = (double)(dset[IDX3(s,a,d,legdim,3)] - dset[IDX3(s,b,d,legdim,3)]);
                    for (int e=0; e<3; ++e) v += GL[d][e]*(double)proj->existing_r[rv].v[e];
                    u[d] = (index_t)llround(v);
                }
                u[0] = ((u[0] % nkx) + nkx) % nkx;
                u[1] = ((u[1] % nky) + nky) % nky;
                u[2] = ((u[2] % nkz) + nkz) % nkz;
                const index_t key = IDX3(u[0],u[1],u[2],nky,nkz);
                if (u2i[key] < 0) { u2i[key] = f->nu; ulist[f->nu] = key; f->nu++; }
                f->uidx[IDX4(s,a,b,rv,ndmax,ndmax,nr)] = u2i[key];
            }
        }

        // Replacing the sum over the orbit of an IBZ momentum by an unweighted
        // sum over the whole symmetry list is only exact if the list maps that
        // momentum onto n_sym *distinct* BZ points. That holds for the generic
        // momenta; the ones on symmetry lines (and models whose symmetry list is
        // not closed under composition) are collected for the two-step path.
        bool* is_free = calloc(f->my_nk, sizeof(bool));
        if (ok) {
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
             * orbit is reached equally often. That is the case for a genuine
             * group action; momenta where it is not are left to the two-step
             * path. */
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
                is_free[q] = uniform;
                if (!uniform) f->nslow += symm->from_ibz_2_bz_len[q + tu->my_nk_off];
            }
            free(img); free(cnt); free(cls); free(csz); free(crep);
            f->ncls = ncls;

            f->slowq = calloc(MAX(f->nslow,1), sizeof(index_t));
            f->Eslow = calloc(MAX(f->nslow,1)*nr, sizeof(complex128_t));
            f->Qslow = calloc(MAX(f->nslow,1)*f->rest, sizeof(complex128_t));
            index_t k = 0;
            for (index_t q=0; q<f->my_nk; ++q) {
                if (is_free[q]) continue;
                const index_t qibz = q + tu->my_nk_off;
                const index_t beg = symm->from_ibz_2_bz_off[qibz] - symm->my_nk_in_bz_off;
                for (index_t i=0; i<symm->from_ibz_2_bz_len[qibz]; ++i, ++k) {
                    f->slowq[k] = beg + i;
                    for (index_t rv=0; rv<nr; ++rv)
                        f->Eslow[IDX2(k,rv,nr)] = proj->my_prefac_reco[rv + nr*(beg+i)];
                }
            }
        }

        // fused exponentials for the generic momenta
        if (ok) {
            f->Etil = calloc(f->my_nk*f->nu, sizeof(complex128_t));
            const double* kmesh = model->internals->kmesh;
            #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<f->my_nk; ++q) {
                const index_t qibz = q + tu->my_nk_off;
                const double* kp = kmesh + 3*symm->idx_ibz_in_fullmesh[qibz];
                const double w = is_free[q] ?
                    (double)symm->from_ibz_2_bz_len[qibz]/(double)f->ncls : 0.0;
                for (index_t u=0; u<f->nu; ++u) {
                    const index_t key = ulist[u];
                    const index_t ul[3] = { key/(nky*nkz), (key/nkz)%nky, key%nkz };
                    double kr = 0.0;
                    for (int e=0; e<3; ++e) {
                        double Rc = 0.0;
                        for (int d=0; d<3; ++d) Rc += M[e][d]*(double)ul[d];
                        kr += kp[e]*Rc;
                    }
                    f->Etil[IDX2(q,u,f->nu)] = w*cexp(I*kr);
                }
            }
            f->F = calloc(f->rest*f->nu, sizeof(complex128_t));
        }
        free(is_free);
        free(ulist);
    }

    free(dlat); free(dset); free(nd); free(u2i);
    // nothing to gain if every momentum ends up on the two-step path
    if (!ok || !f->F || f->nslow >= symm->my_nk_in_bz) { tu_reco_fused_free(f); return NULL; }
    mpi_vrb_printf("fused reco_ft: nu=%li (nr=%li, n_sym*nr=%li), %li/%li BZ points "
            "left to the two-step path\n", (long)f->nu, (long)nr, (long)(n_sym*nr),
            (long)f->nslow, (long)symm->my_nk_in_bz);
    return f;
}

/* canonical (P-like) symmetry reconstruction, restricted to the BZ points of the
 * high symmetry momenta that the fused path cannot take */
static void reco_slow( const tu_reco_fused_t* f, const tu_vertex_t* V,
        const complex128_t* restrict Xch, const complex128_t* restrict pref ) {
    const index_t legdim = f->legdim, rest = f->rest, my_nk_in_bz = V->symm->my_nk_in_bz;
    const index_t my_nk_in_bz_off = V->symm->my_nk_in_bz_off, my_nk_off = V->tu_data->my_nk_off;
    const index_t* restrict from_ibz_2_bz = V->symm->from_ibz_2_bz;
    const index_t* restrict kmaps_to = V->symm->kmaps_to;
    const index_t* restrict map_len = V->symm->o2m_map_len;
    const index_t* restrict map_off = V->symm->o2m_map_off;
    const index_t* restrict idx_map = V->symm->idx_map_symm;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<f->nslow; ++k)
    for (index_t R=0; R<legdim; ++R) {
        const index_t q = f->slowq[k];
        const complex128_t* restrict Xq =
            Xch + rest*(kmaps_to[from_ibz_2_bz[q + my_nk_in_bz_off]] - my_nk_off);
        const index_t ridx = R*my_nk_in_bz + q;
        const index_t rlen = map_len[ridx], roff = map_off[ridx];
        complex128_t* restrict out = f->Qslow + k*rest + R*legdim;
        for (index_t L=0; L<legdim; ++L) {
            const index_t lidx = L*my_nk_in_bz + q;
            const index_t llen = map_len[lidx], loff = map_off[lidx];
            complex128_t sum = 0.0;
            for (index_t l=0; l<llen; ++l)
            for (index_t r=0; r<rlen; ++r)
                sum += pref[loff+l] * conj(pref[roff+r]) *
                       Xq[IDX2(idx_map[roff+r], idx_map[loff+l], legdim)];
            out[L] = sum;
        }
    }
}

void tu_reco_fused_ft( const tu_reco_fused_t* f, const tu_vertex_t* V, char chan,
        complex128_t* buf ) {
    const complex128_t* Xch = chan == 'P' ? V->Pch : chan == 'C' ? V->Cch : V->Dch;
    const complex128_t* c = chan == 'P' ? f->cP : f->cCD;
    const index_t nu = f->nu, nr = f->nr, legdim = f->legdim, rest = f->rest,
                  n_sym = f->n_sym, ndmax = f->ndmax;

    // F(u,R',L') = 1/nk sum_q exp(i k_q R_u) X(q,R',L')
    single_gemm_Xt_Afortran( (single_gemm_Xt_t*)((ptrdiff_t)-1), Xch, f->Etil, f->F,
            nu, rest, f->my_nk, 1./(double)V->nk, 0.0 );

    // symmetry sum, now in real space. The channel only decides which spin
    // labels sit on which leg, i.e. it is a permutation of the output slot that
    // the gather can write to directly.
    const index_t n_spin = V->n_spin, n_orbff = V->n_orbff;
    #pragma omp parallel for collapse(2) schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t R=0; R<legdim; ++R)
    for (index_t L=0; L<legdim; ++L) {
        const index_t a4 = R/(n_spin*n_orbff), a3 = (R/n_orbff)%n_spin, mo3 = R%n_orbff,
                      a2 = L/(n_spin*n_orbff), a1 = (L/n_orbff)%n_spin, mo1 = L%n_orbff;
        index_t Rd = R, Ld = L;
        if (chan == 'C') { Rd = IDX3(a2,a3,mo3,n_spin,n_orbff); Ld = IDX3(a4,a1,mo1,n_spin,n_orbff); }
        if (chan == 'D') { Rd = IDX3(a3,a2,mo3,n_spin,n_orbff); Ld = IDX3(a4,a1,mo1,n_spin,n_orbff); }
        complex128_t* restrict out = buf + IDX2(IDX2(Rd,Ld,legdim),0,nr);
        for (index_t r=0; r<nr; ++r) out[r] = 0.0;
        for (index_t s=0; s<n_sym; ++s) {
            const complex128_t pref = f->wsym[s]*c[IDX2(s,L,legdim)]*conj(c[IDX2(s,R,legdim)]);
            const complex128_t* restrict Fs = f->F + nu*IDX2(f->legmap[IDX2(s,R,legdim)],
                                                             f->legmap[IDX2(s,L,legdim)], legdim);
            const index_t* restrict ui = f->uidx + IDX4(s, f->dcls[IDX2(s,R,legdim)],
                    f->dcls[IDX2(s,L,legdim)], 0, ndmax, ndmax, nr);
            for (index_t r=0; r<nr; ++r) out[r] += pref*Fs[ui[r]];
        }
    }

    // the high symmetry momenta, the old way, added on top
    if (f->nslow > 0) {
        reco_slow( f, V, Xch, c == f->cP ? V->symm->prefactor_symmP : V->symm->prefactor_symmCD );
        single_gemm_Xt_Afortran( (single_gemm_Xt_t*)((ptrdiff_t)-1), f->Qslow, f->Eslow,
                f->F, nr, rest, f->nslow, 1./(double)V->nk, 0.0 );
        #pragma omp parallel for collapse(2) schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t R=0; R<legdim; ++R)
        for (index_t L=0; L<legdim; ++L) {
            const index_t a4 = R/(n_spin*n_orbff), a3 = (R/n_orbff)%n_spin, mo3 = R%n_orbff,
                          a2 = L/(n_spin*n_orbff), a1 = (L/n_orbff)%n_spin, mo1 = L%n_orbff;
            index_t Rd = R, Ld = L;
            if (chan == 'C') { Rd = IDX3(a2,a3,mo3,n_spin,n_orbff); Ld = IDX3(a4,a1,mo1,n_spin,n_orbff); }
            if (chan == 'D') { Rd = IDX3(a3,a2,mo3,n_spin,n_orbff); Ld = IDX3(a4,a1,mo1,n_spin,n_orbff); }
            complex128_t* restrict out = buf + IDX2(IDX2(Rd,Ld,legdim),0,nr);
            const complex128_t* restrict in = f->F + IDX2(IDX2(R,L,legdim),0,nr);
            for (index_t r=0; r<nr; ++r) out[r] += in[r];
        }
    }
}
