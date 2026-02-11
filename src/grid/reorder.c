/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "reorder.h"
#include "helpers.h"

dist_info_t* dist_info_init( index_t nk, index_t nb ) {
    dist_info_t* d = calloc(1, sizeof*d);
    d->myrank = diverge_mpi_comm_rank();
    d->nranks = diverge_mpi_comm_size();
    d->qcounts = diverge_mpi_distribute(nk);
    d->qdispls = d->qcounts + d->nranks;
    d->counts = calloc(d->nranks, sizeof*d->qcounts * 2);
    for (int i=0; i<d->nranks*2; ++i) d->counts[i] = d->qcounts[i] * POW2(nk*POW2(nb));
    d->displs = d->counts + d->nranks;
    return d;
}

void dist_info_free( dist_info_t* d ) {
    free( d->qcounts );
    free( d->counts );
    free( d );
}

static void gen_send_pack( const complex128_t* vertex, const index_t* send_map,
        const dist_info_t* dist, complex128_t* pack );
static void communicate_packs( const complex128_t* send_pack, const int* send_displs,
        const int* send_counts, complex128_t* recv_pack, const int* recv_displs,
        const int* recv_counts );
static void write_recv_pack( complex128_t* vertex_T, const index_t* recv_map,
        const complex128_t* pack, const dist_info_t* dist, bool add, double scale );
static inline int rank_of_index( index_t index, const dist_info_t* dist );

static void gen_send_pack( const complex128_t* vertex, const index_t* send_map,
        const dist_info_t* dist, complex128_t* pack ) {
    const index_t displ = dist->displs[dist->myrank], count = dist->counts[dist->myrank];
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t loc=0; loc<count; ++loc)
        pack[loc] = vertex[send_map[loc] - displ];
}

static void communicate_packs( const complex128_t* send_pack, const int* send_displs, const int* send_counts,
                        complex128_t* recv_pack, const int* recv_displs, const int* recv_counts ) {
    diverge_mpi_alltoallv_complex( send_pack, send_counts, send_displs, recv_pack, recv_counts, recv_displs );
}

static void write_recv_pack( complex128_t* vertex_T, const index_t* recv_map,
        const complex128_t* pack, const dist_info_t* dist, bool add, double scale ) {
    const index_t displ = dist->displs[dist->myrank], count = dist->counts[dist->myrank];
    if (add) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t loc=0; loc<count; ++loc)
            vertex_T[recv_map[loc] - displ] += scale*pack[loc];
    } else {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t loc=0; loc<count; ++loc)
            vertex_T[recv_map[loc] - displ] = scale*pack[loc];
    }
}

void full_pack_exchange( const complex128_t* vertex, complex128_t* vertex_T,
        const index_t* send_map, complex128_t* send_pack, const int* send_displs, const int* send_counts,
        const index_t* recv_map, complex128_t* recv_pack, const int* recv_displs, const int* recv_counts,
        const dist_info_t* dist, bool add, double scale ) {
    gen_send_pack( vertex, send_map, dist, send_pack );
    communicate_packs( send_pack, send_displs, send_counts,
                       recv_pack, recv_displs, recv_counts );
    write_recv_pack( vertex_T, recv_map, recv_pack, dist, add, scale );
}

