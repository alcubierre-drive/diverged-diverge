#pragma once

#include "../diverge_common.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "global.hpp"
#include "../misc/mpi_functions.h"
#include "diverge_interface_symmstruct.hpp"
#include <cassert>

struct tu_data_t {
    bool use_reduced_loop;
    index_t nk;
    index_t nkf;
    index_t nktot;
    index_t nkibz;

    index_t my_nk;
    index_t my_nk_off;

    double offload_cpu;

    // reduction of computational cost of the loop
    vector<Vec3i> existing_ff;
    vector<Vec3i> existing_ff_fine;

    vector<Vec3i> unique_ff_combi;
    index_t* ff_idx_to_unique_ff_elem;

    vector<index_t> unique_mi_l;
    vector<index_t> unique_mi_r;
    vector<index_t> lmap_l;
    vector<index_t> lmap_r;
    vector<index_t> lmap_off;
    vector<index_t> lmap_len;

    index_t n_orbff;
    index_t n_bonds;
    index_t* bond_sizes;
    index_t* bond_offsets;
    index_t* ob_to_orbff;
    symmstruct_t* symm;

    index_t* mi_to_ofrom;
    index_t* mi_to_oto;
    index_t* mi_to_ffidx;
    index_t* mi_to_R;

    index_t n_frq;
    vector<double> bosonic_freq;
    double doping_per_UC = 0.0;
};

// tu info struct contains all relevant informations for the TU code
// which do not strictly belong to the model
void init_MPI_dist(diverge_model_t* model);
void init_ff_struct(const double max_dist, diverge_model_t* model);
void init_symmetries(diverge_model_t* model);
void generate_reduced_loop_maps(diverge_model_t* model);
void init_bos_freqs(tu_data_t* tu_data, index_t n_frq);
// helper function for fixed particle number flows
void init_doping_per_UC(diverge_model_t* model);



void tu_data_destructor(void* model);
