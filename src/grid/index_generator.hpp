/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "typedefs.hpp"

namespace grid {

class IndexGenerator {
    public:
        IndexGenerator( index_t _nk, index_t _nb, MPI_Comm _comm );
        ~IndexGenerator();
        index_t* starts;
        index_t* stops;
        index_t* sizes;
        index_t size;

        index_t* qstarts;
        index_t* qstops;
        index_t qstart, qstop;

        int getRank() const { return myrank; }
        int getNRanks() const { return nranks; }

        index_t get_nk() const { return nk; }
        index_t get_nb() const { return nb; }
        MPI_Comm get_comm() const { return comm; }
    private:
        index_t nk, nb;
        MPI_Comm comm;
        int nranks;
        int myrank;
};

}
