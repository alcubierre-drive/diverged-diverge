/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "diverge_interface.hpp"
#include "vertex.hpp"
#include "projection_wrapper.hpp"

#include <vector>
using std::vector;

void _to_C(complex128_t* proj_vertex, const complex128_t* const I_channel,
            const vector<array<index_t, 2>>& rsmap_to,
            const vector<index_t>& rsmap_from,
            const vector<index_t>& mom,
            const vector<index_t>& rsmap_off,
            const vector<index_t>& rsmap_len,
            const complex128_t* const proj_prefac,
            const index_t realsp_size,
            const index_t num_spin,const index_t nkpts_vertex,
            const index_t my_nkpts_vertex,
            const index_t nr);

void _to_D(complex128_t* proj_vertex, const complex128_t* const I_channel,
            const vector<array<index_t, 2>>& rsmap_to,
            const vector<index_t>& rsmap_from,
            const vector<index_t>& mom,
            const vector<index_t>& rsmap_off,
            const vector<index_t>& rsmap_len,
            const complex128_t* const proj_prefac,
            const index_t realsp_size,
            const index_t num_spin,const index_t nkpts_vertex,
            const index_t my_nkpts_vertex,
            const index_t nr);


void _to_P(complex128_t* proj_vertex, const complex128_t* const I_channel,
            const vector<array<index_t, 2>>& rsmap_to,
            const vector<index_t>& rsmap_from,
            const vector<index_t>& mom,
            const vector<index_t>& rsmap_off,
            const vector<index_t>& rsmap_len,
            const complex128_t* const proj_prefac,
            const index_t realsp_size,
            const index_t num_spin,const index_t nkpts_vertex,
            const index_t my_nkpts_vertex,
            const index_t nr);

#ifdef USE_CUDA
void GPUproj_handler_ft(const cuda_proj_descr_t* cp,const char channel, complex128_t* proj_vertex,
                    const complex128_t* const I_channel,
                    const vector<index_t>& rsmap_to,
                    const vector<index_t>& rsmap_from,
                    const vector<index_t>& mom_in,
                    const complex128_t* const proj_prefac,
                    const index_t realsp_size,
                    const index_t num_spin,
                    const index_t my_nkpts_vertex,
                    const size_t n_projpref);
#endif
