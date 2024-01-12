/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "../diverge_common.h"
#include "../misc/mpi_functions.h"
#include "diverge_interface.hpp"
#include "projection_wrapper.hpp"
#include "propagator.hpp"



class Vertex
{
public:
    Vertex(diverge_model_t* model, Projection* proj, tu_loop_t* loop, bool P = true,
            bool C = true, bool D = true, bool self = false);
    ~Vertex(void);


    Vertex(const Vertex& other);
    Vertex& operator=(const Vertex& other);

    template<char channel>
    void reco(complex128_t* const buf, const complex128_t* inp = nullptr) const;

    template<char channel>
    void reco_ft(complex128_t* const buf, const complex128_t * const existing_r_exp, const uint nr,
                            complex128_t * bufMPI = nullptr) const;

    void init_U_from_fullV(diverge_model_t* model, Projection* proj);
    void project_fullV(const diverge_model_t* const mod_ptr, complex128_t*,
                        const double * frq = nullptr) const;

    void initialize_GPU_memory(Projection* proj, tu_loop_t* loop);

    void init_U(bool enforce_exchange);
    void unset_U();
    
    complex128_t* Pch = nullptr;
    complex128_t* Cch = nullptr;
    complex128_t* Dch = nullptr;

    complex128_t* pp_bubble_int = nullptr;
    complex128_t* ph_bubble_int = nullptr;

    complex128_t* self_en = nullptr;

    const complex128_t* get_P_ptr_cst()const{return Pch;}
    const complex128_t* get_C_ptr_cst()const{return Cch;}
    const complex128_t* get_D_ptr_cst()const{return Dch;}
    const complex128_t* get_E_ptr_cst()const{return self_en;}

    const complex128_t* get_PP_bubble_cst()const{return pp_bubble_int;}
    const complex128_t* get_PH_bubble_cst()const{return ph_bubble_int;}

    diverge_model_t* model;
    const tu_data_t* const tu_data;
    const index_t n_orbff;
    const index_t n_orb;
    const index_t n_bonds;
    const index_t n_spin;
    const index_t nk;
    const index_t nkf;
    const index_t my_nk;
    const index_t my_nk_off;
    const bool SU2;
    const index_t full_vert_size;
    const index_t full_bubble_size;
    const index_t selfenergy_size;
    const symmstruct_t* symm;

    char* GPU_memory_pool[32];
    char* GPU_memory_pool_managed[32];
    char* GPU_gemm_buffer[32][3];
    bool has_gpu_memory = false;

    bool P_flow;
    bool C_flow;
    bool D_flow;
    const bool use_self;

    // helper arrays
    complex128_t* bubble_helper;
    complex128_t* projection_helper;
    complex128_t* product_helper;
private:
    bool owns;
};


void add(Vertex &result, const Vertex &inp, const double factor) ;
void set(Vertex &f,const Vertex &initial,const Vertex &inp, const double factor);
void multiply(Vertex &f,const Vertex &inp, const double factor);
double infinity_norm(const Vertex &inp);
vector<double> max(const Vertex &inp);

#ifndef SKIP_VERTEX_RECO_INCLUDE
#include "vertex_reco.hpp"
#endif
