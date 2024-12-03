/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "global.hpp"
#include "../diverge_model.h"
#include "../diverge_flow_step.h"
#include "vertex.hpp"
#include "projection_wrapper.hpp"
#include "projection_handler.hpp"

#ifndef TU_MAGIC_NUMBER_POST_PROCESSING
#define TU_MAGIC_NUMBER_POST_PROCESSING 'T'
#endif

class tu_postproduction
{
public:
    tu_postproduction(diverge_flow_step_t* st, bool tu_skip_channel_calc);
    ~tu_postproduction();

    void run_and_save(string filename, bool lingaps, bool susc_ff, bool susc_full,
            bool lingap_atscale, bool save_self, bool save_full, bool save_symmetries);
    void lingap_ff(bool atscale);
    void calc_susc(bool susz_ff, bool susz_full);
    void save_channel_leading_ev();

    index_t tu_n_decomp_values;
    index_t tu_n_singular_values;
    char algo;
    double cutoff;
    bool cutoff_rel;
    char tu_lingap_solver_mode[4] = "sss";
    bool bare_susceptibilities = false;
    bool channel_calc_project = false;

    index_t* lenP;
    index_t* lenC;
    index_t* lenD;
    index_t* offP;
    index_t* offC;
    index_t* offD;
    char* Ptype;
    char* Ctype;
    char* Dtype;
    double* Pval;
    double* Cval;
    double* Dval;
    complex128_t* Pvec;
    complex128_t* Cvec;
    complex128_t* Dvec;

    void to_file(string filename, bool save_self, bool save_full, bool save_symmetries);

    double* Sgap_sc;
    double* Sgap_mag;
    double* Sgap_charge;
    complex128_t* Ugap_sc;
    complex128_t* Ugap_mag;
    complex128_t* Ugap_charge;
    complex128_t* Vgap_sc;
    complex128_t* Vgap_mag;
    complex128_t* Vgap_charge;

    complex128_t* suscPp;
    complex128_t* suscCh;
    complex128_t* suscMa;

    complex128_t* suscPp_ff;
    complex128_t* suscCh_ff;
    complex128_t* suscMa_ff;
private:
    index_t tu_n_singular_valuesP=0;
    index_t tu_n_singular_valuesC=0;
    index_t tu_n_singular_valuesD=0;

    void construct_multiindex_maps(const diverge_model_t* model);

    std::vector<index_t> mi_map_idx;
    std::vector<complex128_t> mi_map_pref_in_in;
    std::vector<complex128_t> mi_map_pref_in_out;

    index_t tu_suscP=0;
    index_t tu_suscC=0;
    index_t tu_suscD=0;

    index_t tu_suscP_ff=0;
    index_t tu_suscC_ff=0;
    index_t tu_suscD_ff=0;
    bool called_lingap = false;
    bool called_suscep_ff = false;
    bool called_suscep_full = false;
    bool skip_channel_calc;
    Vertex* vertex;
    Vertex* internal_vertex;
    Projection* proj;
    const diverge_model_t* model;
    void _save_channel_leading_ev(complex128_t* channel, complex128_t* vec,
                    double* val, char* type, index_t* len, index_t* off);
};

