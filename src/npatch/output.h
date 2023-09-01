#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct npatch_vertex_t npatch_vertex_t;
typedef struct npatch_loop_t npatch_loop_t;

/* header is index_t[128], and contains:
 * VERTEX_FILE_MAGIC_NUMBER
 * vertex_displ
 * vertex_count
 * Lp_displ
 * Lp_count
 * Lm_displ
 * Lm_count
 * dV_displ
 * dV_count
 * chan_1
 * use_dV_1
 * nq_1
 * serial_qmat_displ_1
 * serial_qmat_count_1
 * chan_2
 * use_dV_2
 * nq_2
 * serial_qmat_displ_2
 * serial_qmat_count_2
 * chan_3
 * use_dV_3
 * nq_3
 * serial_qmat_displ_3
 * serial_qmat_count_3
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
