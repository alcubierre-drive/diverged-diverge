/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "index_generator.hpp"

namespace grid {

IndexGenerator::IndexGenerator( index_t _nk, index_t _nb, MPI_Comm _comm ) {
    comm = _comm;
    nk = _nk;
    nb = _nb;
    nranks = diverge_mpi_comm_size();
    myrank = diverge_mpi_comm_rank();
    starts = new index_t[nranks];
    stops = new index_t[nranks];
    sizes = new index_t[nranks];
    index_t full_size = nk*nk*nk * nb*nb*nb*nb;
    index_t minimum_batch_size = nk*nk * nb*nb*nb*nb;

    index_t batches_per_rank = nk/(index_t)nranks;

    qstarts = new index_t[nranks];
    qstops = new index_t[nranks];

    for (int i=0; i<nranks; ++i) {
        starts[i] = batches_per_rank*minimum_batch_size*i;
        qstarts[i] = batches_per_rank*i;
    }

    for (int i=0; i<nranks-1; ++i) {
        stops[i] = starts[i+1];
        qstops[i] = qstarts[i+1];
    }
    stops[nranks-1] = full_size;
    qstops[nranks-1] = nk;

    for (int i=0; i<nranks; ++i) {
        sizes[i] = stops[i] - starts[i];
    }

    size = sizes[myrank];

    qstart = qstarts[myrank];
    qstop = qstops[myrank];
}

IndexGenerator::~IndexGenerator() {
    delete[] starts;
    delete[] stops;
    delete[] sizes;
    delete[] qstarts;
    delete[] qstops;
}

}
