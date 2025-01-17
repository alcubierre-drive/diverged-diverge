/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

typedef struct {
    int current_rank;
    int target_rank;
    index_t current_index;
    index_t target_index;
    complex128_t phase;
} grid_symm_op_t;

typedef struct {
    index_t* reorder_before_send; // _loop_size
    complex128_t* phase_before_send; // _loop_size
    index_t *send_counts, *send_displs, *recv_counts, *recv_displs;
    index_t* reorder_after_recv; // _loop_size

    complex128_t* orbital_matrix; // (nb, nb) or NULL
    int orbital_matrix_free;
    index_t nb;
    char chan;
} grid_symm_comm_t;

void grid_symm_comm_free( grid_symm_comm_t S );

typedef struct {
    int idx_pre_symm;
    int idx_post_symm;
    int R1, R2, R3;
} grid_symm_prop_t;

typedef struct grid_symm_transform_t grid_symm_transform_t;
grid_symm_transform_t* grid_symm_transform_init( diverge_model_t* model );

index_t grid_symm_transform_nsym( const grid_symm_transform_t* s );
index_t grid_symm_transform_loopsize( const grid_symm_transform_t* s );

void grid_symm_transform_get_trafo( grid_symm_transform_t* s, index_t sym, index_t qdispl,
        index_t qcount, complex128_t* phasesP /*(qcount, nk, POW4(nb))*/,
        complex128_t* phasesC, index_t* mapP /*(qcount, nk, POW4(nb))*/, index_t* mapC );
complex128_t* grid_symm_transform_orbmatrix_copy( const grid_symm_transform_t* s, index_t sym );
complex128_t* grid_symm_transform_orbmatrix_ref( const grid_symm_transform_t* s, index_t sym );
void grid_symm_transform_free( grid_symm_transform_t* s );

void grid_symm_transform_init_communicator( grid_symm_transform_t* s,
        index_t qdispl, index_t qcount );
grid_symm_comm_t grid_symm_transform_comm( grid_symm_transform_t* s, index_t symm, char chan );


#ifdef __cplusplus
}
#endif
