/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "kernels_loop_cpu.h"
#include "kernels_helpers.h"
#include "../diverge_internals_struct.h"
#include "../misc/mpi_functions.h"

void* npatch_loop_gen_cpu( void* data ) {
    double tick = diverge_mpi_wtime();
    npatch_loop_kernel_info_t* loop_info = (npatch_loop_kernel_info_t*)data;
    npatch_loop_t* L = loop_info->L;

    complex128_t * restrict Lp = L->Lp,
                 * restrict Lm = L->Lm;
    const gf_complex_t * restrict Gp = L->Gp,
                       * restrict Gm = L->Gm;
    const index_t nb = L->nb,
                  np = L->np;
    const index_t * restrict nk_flat = L->nk_flat;
    const index_t * restrict p_idx = L->p_idx,
                  * restrict p_cnt = L->p_cnt,
                  * restrict p_dis = L->p_dis,
                  * restrict p_map = L->p_map;
    const double * restrict f_wgh = L->f_wgh;
    const double global_norm = 0.5/M_PI;
    const index_t L0 = loop_info->start,
                  L1 = loop_info->stop;

    #pragma omp parallel for
    for (index_t L_idx = L0; L_idx < L1; ++L_idx) {
        UNRAVEL_V( L_idx, p, p1, p2, o1, o2, o3, o4 )
        index_t k = p_idx[p],
                k1 = p_idx[p1],
                k2 = p_idx[p2],
                n_k = p_cnt[p],
                d_k = p_dis[p];
        const index_t *kf_ary = p_map + d_k;
        const double *kf_wgh = f_wgh + d_k;

        complex128_t _Lp = 0.0, _Lm = 0.0;
        for (index_t kf=0; kf<n_k; ++kf) {
            index_t kf_idx = kf_ary[kf];
            index_t _kl = k1p2(k, kf_idx, nk_flat),
                    _krp = k1m2(k1p2(k1, k2, nk_flat), _kl, nk_flat),
                    _krc = k1p2(k1m2(k1, k2, nk_flat), _kl, nk_flat);
            double kf_norm = global_norm * kf_wgh[kf];
            _Lp += (Gm[_kl*nb*nb+o1*nb+o3] * Gm[_krc*nb*nb+o2*nb+o4] +
                    Gp[_kl*nb*nb+o1*nb+o3] * Gp[_krc*nb*nb+o2*nb+o4]) * kf_norm;
            _Lm += (Gm[_kl*nb*nb+o1*nb+o3] * Gp[_krp*nb*nb+o2*nb+o4] +
                    Gp[_kl*nb*nb+o1*nb+o3] * Gm[_krp*nb*nb+o2*nb+o4]) * kf_norm;
        }
        Lp[L_idx] = _Lp;
        Lm[L_idx] = _Lm;
    }
    double tock = diverge_mpi_wtime();
    loop_info->wtime += (tock-tick);

    return NULL;
}
