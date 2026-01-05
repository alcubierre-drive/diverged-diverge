/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;
typedef struct npatch_loop_t npatch_loop_t;

struct npatch_vertex_t {
    index_t size;
    complex128_t* V;
    complex128_t* dV;
};
typedef struct npatch_vertex_t npatch_vertex_t;

npatch_vertex_t* npatch_vertex_init( diverge_model_t* dmod );
void npatch_vertex_free( npatch_vertex_t* vert );

// structure that is taylored to store the vertex of a single transfer momenutm
// (wrt to a single channel) and do the eigenspace transform
struct npatch_qvert_mat_t {
    complex128_t* data;
    index_t* k_kp_descr;
    index_t n_k_kp;
    index_t nb;
    index_t q;

    bool eigen;
    double* values;
    index_t nv;
};
typedef struct npatch_qvert_mat_t npatch_qvert_mat_t;

// generate q matrices from vertex, including allocation.
// sorts with absolute norm of each of the q vertices. the number of vertices
// smaller than 'max_q_rel' is stored in *pnq_sorted, the total number of
// vertices in *pnq
npatch_qvert_mat_t* npatch_qvert_mat_from_vertex( const npatch_vertex_t* vert,
        const npatch_loop_t* loop, char chan, bool dV, double max_q_rel,
        index_t* pnq, index_t* pnq_sorted );

// transform the q matrices to their eigenspace and only consider the nv first
// eigenvectors/values (determined by 'which'. defaults to 'M', which is
// magnitude). If called with nv<0 do nothing, for nv=0 take *all*
// eigenvectors/values.
void npatch_qvert_mat_eigen( npatch_qvert_mat_t* qvm, index_t nq, index_t nv, char which );

// serialize a single q matrix pointed to by qvm. return its freshly allocated
// memory and store the size in *psize
// memory layout (interpreted as *signed* index_t if not stated differently)
// -------------
// n_k_kp
// nb
// q
// eigen
// nv
// k_kp_descr[n_k_kp]
// IF EIGEN:
//    values[nv] as (double)
//    vectors[nv * n_k_kp * nb*nb * 2] as (double), (real, imag)
// ELSE:
//    vertex[POW2(n_k_kp * nb*nb) * 2] as (double), (real, imag)
uindex_t* npatch_qvert_mat_serialize( npatch_qvert_mat_t* qvm, index_t* psize );

// deserialize a single q matrix including its allocation
npatch_qvert_mat_t* npatch_qvert_mat_deserialize( uindex_t* mem );

// free the q matrix array
void npatch_qvert_mat_free( npatch_qvert_mat_t* qvm, index_t nq );

#ifdef __cplusplus
}
#endif
