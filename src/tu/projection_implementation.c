/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "projection_implementation.h"
#include "../misc/gemm.h"
#include <stddef.h>

/* upper bound for the memory of one block of the CPU projection (gathered
 * real space vertex and its GEMM result) */
#ifndef TU_PROJ_BLOCK_BYTES
#define TU_PROJ_BLOCK_BYTES (256l<<20)
#endif

/* CPU projection for nk > 1. For every target orbital pair om, spin s and IBZ
 * momentum q the projection reads
 *
 *   out(q,om,s) += sum_{e in om} I(r1(e), from(e), s) * P(q, r2(e)),
 *
 * with I the real space vertex (nr fastest) and P = proj_prefac (my_nk, nr).
 * Only P depends on q, so the sum is split into a sparse gather that collects
 * the entries of every (om,s) by their plane wave index r2,
 *
 *   G(om,s, r2) = sum_{e in om, r2(e) == r2} I(r1(e), from(e), s),
 *
 * and a ZGEMM out(q,om,s) += sum_r2 P(q,r2) G(om,s,r2). I is read once instead
 * of once per q. The target orbital pairs are processed in blocks so that G
 * and the GEMM result never exceed TU_PROJ_BLOCK_BYTES. The channels only
 * differ in which spin labels the target index combines (chan = 'C','P','D'). */
static void to_X_gemm( char chan, complex128_t* restrict proj_vertex,
        const complex128_t* restrict I_channel, const claV2ll_t* restrict rsmap_to,
        index_t n_om, const index_t* restrict rsmap_from, const index_t* restrict mom,
        const index_t* restrict rsmap_off, const index_t* restrict rsmap_len,
        const complex128_t* restrict proj_prefac, const index_t n_orbff,
        const index_t ns, const index_t my_nk, const index_t nr ) {
    if (n_om < 1 || my_nk < 1 || nr < 1) return;
    const index_t ns2 = ns*ns, ns4 = ns2*ns2;
    const index_t rest = POW2(n_orbff)*ns4;
    const index_t leg2 = ns2*n_orbff; // stride of the second orbital of a pair

    // spin part of the source and target index for each (s1,s2,s3,s4)
    index_t sfrom[ns4], sto[ns4];
    for (index_t s1=0; s1<ns; ++s1)
    for (index_t s2=0; s2<ns; ++s2)
    for (index_t s3=0; s3<ns; ++s3)
    for (index_t s4=0; s4<ns; ++s4) {
        const index_t si = IDX4(s1,s2,s3,s4,ns,ns,ns);
        index_t sa, sb;
        switch (chan) {
            case 'C': sa = s1+s4*ns; sb = s3+s2*ns; break;
            case 'D': sa = s1+s3*ns; sb = s4+s2*ns; break;
            default:
            case 'P': sa = s1+s2*ns; sb = s3+s4*ns; break;
        }
        sfrom[si] = n_orbff*(s1+s2*ns) + leg2*n_orbff*(s3+s4*ns);
        sto[si] = n_orbff*sa + leg2*n_orbff*sb;
    }

    const index_t bytes_per_om = ns4*(nr+my_nk)*(index_t)sizeof(complex128_t);
    const index_t nb_max = MIN(n_om, MAX(1, TU_PROJ_BLOCK_BYTES/bytes_per_om));
    mpi_log_large_alloc( nb_max*bytes_per_om, "the CPU TU projection block" );
    // G(col, r) and C(col, q) with col = lom + nb*si, both column major
    complex128_t* G = malloc( nb_max*ns4*nr*sizeof(complex128_t) );
    complex128_t* C = malloc( nb_max*ns4*my_nk*sizeof(complex128_t) );

    for (index_t om0=0; om0<n_om; om0+=nb_max) {
        const index_t nb = MIN(nb_max, n_om-om0), ncols = nb*ns4;

        #pragma omp parallel num_threads(diverge_omp_num_threads())
        {
            #pragma omp for schedule(static)
            for (index_t i=0; i<ncols*nr; ++i) G[i] = 0.0;
            #pragma omp for schedule(dynamic,16)
            for (index_t lom=0; lom<nb; ++lom) {
                const index_t om = om0+lom;
                for (index_t e=rsmap_off[om]; e<rsmap_off[om]+rsmap_len[om]; e+=2) {
                    const complex128_t* restrict Ie = I_channel + mom[e] +
                        nr*(rsmap_from[e] + leg2*rsmap_from[e+1]);
                    complex128_t* restrict Ge = G + lom + ncols*mom[e+1];
                    for (index_t si=0; si<ns4; ++si) Ge[nb*si] += Ie[nr*sfrom[si]];
                }
            }
        }

        // C(col,q) = sum_r G(col,r) P(q,r)
        single_gemm_Xt_Afortran( (single_gemm_Xt_t*)((ptrdiff_t)-1), proj_prefac, G, C,
                ncols, my_nk, nr, 1.0, 0.0 );

        #pragma omp parallel for collapse(2) schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t q=0; q<my_nk; ++q)
        for (index_t si=0; si<ns4; ++si) {
            complex128_t* restrict out = proj_vertex + rest*q + sto[si];
            const complex128_t* restrict Cq = C + ncols*q + nb*si;
            for (index_t lom=0; lom<nb; ++lom) {
                const claV2ll_t t = rsmap_to[om0+lom];
                out[t.v[0] + leg2*t.v[1]] += Cq[lom];
            }
        }
    }
    free(G);
    free(C);
}

void _to_C( complex128_t* proj_vertex, const complex128_t* I_channel,
        const claV2ll_t* rsmap_to, index_t rsmap_to_sz, const index_t* rsmap_from,
        const index_t* mom, const index_t* rsmap_off, const index_t* rsmap_len,
        const complex128_t* proj_prefac, const index_t realsp_size,
        const index_t num_spin, const index_t nk,
        const index_t my_nk, const index_t nr ) {
    const index_t mapped_indices = rsmap_to_sz;
    if (nk == 1) {
        #pragma omp parallel for collapse(5) schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t s1=0; s1<num_spin; ++s1)
        for (index_t s2=0; s2<num_spin; ++s2)
        for (index_t s3=0; s3<num_spin; ++s3)
        for (index_t s4=0; s4<num_spin; ++s4)
        for (index_t orb_map=0; orb_map<mapped_indices; ++orb_map) {
            const index_t idx_FROM = rsmap_from[orb_map*2]+realsp_size*(
                (s1+s2*num_spin)+POW2(num_spin)*(
                    rsmap_from[orb_map*2+1]+realsp_size*(
                        (s3+s4*num_spin))));

            const index_t idx_TO = rsmap_to[orb_map].v[0]+realsp_size*(
            (s1+s4*num_spin)+POW2(num_spin)*(
                rsmap_to[orb_map].v[1]+realsp_size*(
                    (s3+s2*num_spin))));
            proj_vertex[idx_TO] += I_channel[idx_FROM];
        }
    } else {
        to_X_gemm( 'C', proj_vertex, I_channel, rsmap_to, mapped_indices, rsmap_from,
                mom, rsmap_off, rsmap_len, proj_prefac, realsp_size, num_spin, my_nk, nr );
    }
}

void _to_P( complex128_t* proj_vertex, const complex128_t* I_channel,
        const claV2ll_t* rsmap_to, index_t rsmap_to_sz, const index_t* rsmap_from,
        const index_t* mom, const index_t* rsmap_off, const index_t* rsmap_len,
        const complex128_t* proj_prefac, const index_t realsp_size,
        const index_t num_spin, const index_t nk,
        const index_t my_nk, const index_t nr ) {
    const index_t mapped_indices = rsmap_to_sz;
    if (nk == 1) {
        #pragma omp parallel for collapse(5) schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t s1=0; s1<num_spin; ++s1)
        for (index_t s2=0; s2<num_spin; ++s2)
        for (index_t s3=0; s3<num_spin; ++s3)
        for (index_t s4=0; s4<num_spin; ++s4)
        for (index_t orb_map=0; orb_map<mapped_indices; ++orb_map) {
                const index_t idx_FROM = rsmap_from[orb_map*2]+realsp_size*(
                    (s1+s2*num_spin)+POW2(num_spin)*(
                        rsmap_from[orb_map*2+1]+realsp_size*(
                            (s3+s4*num_spin))));

                const index_t idx_TO = rsmap_to[orb_map].v[0]+realsp_size*(
                (s1+s2*num_spin)+POW2(num_spin)*(
                    rsmap_to[orb_map].v[1]+realsp_size*(
                        (s3+s4*num_spin))));
                proj_vertex[idx_TO] += I_channel[idx_FROM];
        }
    } else {
        to_X_gemm( 'P', proj_vertex, I_channel, rsmap_to, mapped_indices, rsmap_from,
                mom, rsmap_off, rsmap_len, proj_prefac, realsp_size, num_spin, my_nk, nr );
    }
}

void _to_D( complex128_t* proj_vertex, const complex128_t* I_channel,
        const claV2ll_t* rsmap_to, index_t rsmap_to_sz, const index_t* rsmap_from,
        const index_t* mom, const index_t* rsmap_off, const index_t* rsmap_len,
        const complex128_t* proj_prefac, const index_t realsp_size,
        const index_t num_spin, const index_t nk,
        const index_t my_nk, const index_t nr ) {
    const index_t mapped_indices = rsmap_to_sz;
    if (nk==1) {
        #pragma omp parallel for collapse(5) schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t s1=0; s1<num_spin; ++s1)
        for (index_t s2=0; s2<num_spin; ++s2)
        for (index_t s3=0; s3<num_spin; ++s3)
        for (index_t s4=0; s4<num_spin; ++s4)
        for (index_t orb_map=0; orb_map<mapped_indices; ++orb_map) {
            const index_t idx_FROM = rsmap_from[orb_map*2]+realsp_size*(
                (s1+s2*num_spin)+POW2(num_spin)*(
                    rsmap_from[orb_map*2+1]+realsp_size*(
                        (s3+s4*num_spin))));

            const index_t idx_TO = rsmap_to[orb_map].v[0]+realsp_size*(
            (s1+s3*num_spin)+POW2(num_spin)*(
                rsmap_to[orb_map].v[1]+realsp_size*(
                    (s4+s2*num_spin))));
            proj_vertex[idx_TO] += I_channel[idx_FROM];
        }
    } else {
        to_X_gemm( 'D', proj_vertex, I_channel, rsmap_to, mapped_indices, rsmap_from,
                mom, rsmap_off, rsmap_len, proj_prefac, realsp_size, num_spin, my_nk, nr );
    }
}

