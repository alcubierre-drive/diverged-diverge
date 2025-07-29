/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct npatch_vertex_t npatch_vertex_t;
typedef struct npatch_loop_t npatch_loop_t;

/* header is index_t[128], with all displ/size information in units of 64bit
 * (i.e., double). It contains:
 *
 * header[0] - VERTEX_FILE_MAGIC_NUMBER
 * header[125-126] - file format version string [displ, size]. if both are zero,
 *                   default to v0.4 for diverge.output.read().
 *
 * header[1-2] - vertex [displ, size]
 * header[3-4] - Lp [displ, size]
 * header[5-6] - Lm [displ, size]
 * header[7-8] - dV [displ, size]
 * header[9] - channel 1
 * header[10] - use dV (channel 1)
 * header[11] - nq (channel 1)
 * header[12-13] - serial q matrices (channel 1) [displ, size]
 * header[14] - channel 2
 * header[15] - use dV (channel 2)
 * header[16] - nq (channel 2)
 * header[17-18] - serial q matrices (channel 2) [displ, size]
 * header[19] - channel 3
 * header[20] - use dV (channel 3)
 * header[21] - nq (channel 3)
 * header[22-23] - serial q matrices (channel 3) [displ, size]
 * …
 *
 * the serial_qmat data contains (in index_t if not indicated differently):
 *      n_k_kp
 *      nb
 *      q
 *      eigen
 *      nv
 *      k_kp_descr[n_k_kp]
 *      IF EIGEN:
 *         values[nv] as (double)
 *         vectors[nv * n_k_kp * nb*nb * 2] as (double), (real, imag)
 *      ELSE:
 *         vertex[POW2(n_k_kp * nb*nb) * 2] as (double), (real, imag)
 */
struct npatch_output_post_config_t {
    bool q_matrices;
    bool q_matrices_use_dV;
    int q_matrices_nv;
    double q_matrices_max_rel;
    char q_matrices_eigen_which;

    bool V;
    bool dV;
    bool Lp;
    bool Lm;
};
typedef struct npatch_output_post_config_t npatch_output_post_config_t;
int npatch_output_post( npatch_vertex_t* v, npatch_loop_t* l, const npatch_output_post_config_t* ocfg, const char* fname );

#ifdef __cplusplus
}
#endif
