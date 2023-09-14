/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "vertex.h"

#ifdef __cplusplus
extern "C" {
#endif

struct qkkp_t {
    index_t q, k, kp;
    index_t p1, p2, p3;
    char chan;
};
typedef struct qkkp_t qkkp_t;

void vhelper_vertex_in_q_channel( const npatch_vertex_t* vert,
        const npatch_loop_t* loop, char chan, bool dV, complex128_t** pvert,
        qkkp_t** pvindex, index_t** pqcounts, index_t* pnq );
// outputs are allocated and need to be freed by the user.
// pvert: the vertex in the respective channel as function of q, k, k' and the
// orbitals ordered with np**3 * no**4 entries
// pvindex: the array of q, k, k' and the corresponding patch indices with
// np**3 entries
// pqcounts: how many vertex entries exist for each value of q with at
// maximum np**2 entries
// pnq: size of the array returned in pqcounts

index_t* vhelper_qkkp_to_ary( const qkkp_t* q, index_t nq );
// output is allocated and needs to be freed by the user.
// returns the (nq,3) array of (q,k,k') for each entry in the qkkp_t* q array

npatch_qvert_mat_t* vhelper_q_vertex_matrix_from_vertex(
        const complex128_t* q_vertex, const qkkp_t* q_vindex,
        const index_t* q_counts, index_t nq, index_t nb );

// helper function written in c++
void vhelper_vertex_sort_in_q( complex128_t* vertex, qkkp_t* vindex, index_t np,
        index_t nb, index_t** pqcounts, index_t* pnq );

// allocates a buffer of size Nmax*Nmax*nq and fills it with the matrices
// extended by zeros as needed, so to prepare for the batched GEMM
complex128_t* vhelper_q_vertex_matrices_to_buffer( npatch_qvert_mat_t* qvm, index_t nq, index_t Nmax );

// sorts the array with respect to the absolute norm of each q matrix, and cuts
// off the elements that are smaller than max_rel * ||qvm||_{max, q}
void vhelper_q_vertex_sort_max( npatch_qvert_mat_t* qvm, index_t* pnq, double max_rel );

#ifdef __cplusplus
}
#endif

