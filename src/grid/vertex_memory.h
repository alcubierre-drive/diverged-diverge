/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "typedefs.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dist_info_t dist_info_t;
typedef struct grid_vertex_t grid_vertex_t;

grid_vertex_t* grid_vertex_init( index_t nk, index_t nb, index_t* k1m2, index_t* k1p2, bool SU2 );
void grid_vertex_from_qbuf( grid_vertex_t* v, const complex128_t* qbuf_in, char chan_in, index_t nspin );
// modify transformations
void grid_vertex_overwrite_transform( grid_vertex_t* v, int send, int recv, const char* new_name );
// anti-symmetrizes vertex. factor 0.25 means we average. 0.5 means we
// 'extend' the C/D channel vertex to all channels
void grid_vertex_exchange_symmetry( grid_vertex_t* v, double factor/*=0.5*/ );
void grid_vertex_to_file( grid_vertex_t* v, const char* fname, char chan );
void grid_vertex_get_chan_allranks( grid_vertex_t* v, complex128_t* buf, char chan );
void grid_vertex_free( grid_vertex_t* v );

struct grid_vertex_t {
    // elements: v, dv, left, right.
    complex128_t* vertex_buffers[4];
    // communication buffers
    complex128_t *send_pack, *recv_pack;

    bool SU2;
    index_t size, nk, nb;

    int sub_D[3]; // = (1,1,1)

    // index distribution info, just an alias for the content in dist
    index_t myrank, nranks;
    index_t *qcounts, *qdispls, qdispl, qcount, *counts, *displs;
    dist_info_t* dist;

    // and the communication infrastructure
    int n_send_recv_maps;
    index_t* send_recv_maps[8];
    int *send_recv_displs[8], *send_recv_counts[8];
    char send_recv_descr[8][64];

    // momentum add/sub
    const index_t *k1m2, *k1p2;
};

// the 'chans' function creates trafo if no transform is found and then resets
void grid_vertex_project_chans( grid_vertex_t* v, const complex128_t* in, complex128_t* out,
        char chan_in, char chan_out, bool add, double scale );
void grid_vertex_project_name( grid_vertex_t* v, const complex128_t* in, complex128_t* out,
        const char* name, bool add, double scale );
void grid_vertex_copy( grid_vertex_t* v, const complex128_t* in, complex128_t* out, bool add, double scale );

#ifdef __cplusplus
}
#endif
