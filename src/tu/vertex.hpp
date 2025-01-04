/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../diverge_common.h"
#include "../misc/mpi_functions.h"
#include "diverge_interface.h"

// this is a temporary interface for the C-style selfenergy
#ifdef __cplusplus
extern "C" {
#endif
diverge_model_t* Vertex_get_model( void* V_ );
const tu_data_t* Vertex_get_tu_data( void* V_ );
index_t Vertex_get_n_orb( void* V_ );
index_t Vertex_get_n_spin( void* V_ );
index_t Vertex_get_n_orbff( void* V_ );
index_t Vertex_get_nk( void* V_ );
index_t Vertex_get_nkf( void* V_ );
index_t Vertex_get_my_nk( void* V_ );
index_t Vertex_get_my_nk_off( void* V_ );
const symmstruct_t* Vertex_get_symm( void* V_ );
bool Vertex_get_P_flow( void* V_ );
bool Vertex_get_C_flow( void* V_ );
bool Vertex_get_D_flow( void* V_ );
complex128_t* Vertex_get_product_helper( void* V_ );
complex128_t* Vertex_get_bubble_helper( void* V_ );
#ifdef __cplusplus
}

#include "../diverge_Eigen3.hpp"

#include "projection_wrapper.h"
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
    void reco_ft(complex128_t* const buf, const complex128_t * const existing_r_exp, const uindex_t nr,
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

    bool C_to_P = true;
    bool D_to_P = true;
    bool P_to_C = true;
    bool D_to_C = true;
    bool P_to_D = true;
    bool C_to_D = true;

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

#endif // __cplusplus
