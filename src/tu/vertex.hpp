/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../diverge_common.h"
#include "../misc/mpi_functions.h"
#include "diverge_interface.h"
#include "projection.h"

// this is a temporary interface for the C-style selfenergy
#ifdef __cplusplus
typedef class Vertex Vertex;
typedef class tu_prop_t tu_prop_t;
extern "C" {
#else
typedef struct Vertex Vertex;
typedef struct tu_prop_t tu_prop_t;
#endif

Vertex* Vertex_init( diverge_model_t* modelin, projection_t* proj, tu_prop_t* loop,
        bool P, bool C, bool D, bool self );
Vertex* Vertex_copy( const Vertex* other );
void Vertex_free( Vertex* V );

diverge_model_t* Vertex_get_model( const Vertex* V );
const tu_data_t* Vertex_get_tu_data( const Vertex* V );
index_t Vertex_get_n_orb( const Vertex* V );
index_t Vertex_get_n_spin( const Vertex* V );
index_t Vertex_get_n_orbff( const Vertex* V );
index_t Vertex_get_nk( const Vertex* V );
index_t Vertex_get_nkf( const Vertex* V );
index_t Vertex_get_my_nk( const Vertex* V );
index_t Vertex_get_my_nk_off( const Vertex* V );
const symmstruct_t* Vertex_get_symm( const Vertex* V );
bool Vertex_get_P_flow( const Vertex* V );
bool Vertex_get_C_flow( const Vertex* V );
bool Vertex_get_D_flow( const Vertex* V );
complex128_t* Vertex_get_product_helper( const Vertex* V );
complex128_t* Vertex_get_bubble_helper( const Vertex* V );
void Vertex_reco( const Vertex* V, char chan, complex128_t* buf,
        const complex128_t* inp/*=NULL*/ );
void Vertex_reco_ft( const Vertex* V, char chan, complex128_t* buf,
        const complex128_t* existing_r_exp, index_t nr, complex128_t* bufMPI/*=NULL*/ );
int Vertex_get_X_flow( const Vertex* V, char X );
int Vertex_get_X_to_Y_proj( const Vertex* V, char X, char Y );
complex128_t* Vertex_get_chan( const Vertex* V, char X );
index_t Vertex_get_full_vert_size( const Vertex* V );
#ifdef __cplusplus
}

#include "../diverge_Eigen3.hpp"
#include <vector>
using std::vector;

class Vertex
{
public:
    Vertex(diverge_model_t* model, projection_t* proj, tu_prop_t* loop, bool P = true,
            bool C = true, bool D = true, bool self = false);
    ~Vertex(void);


    Vertex(const Vertex& other);
    Vertex& operator=(const Vertex& other);

    void reco( const char channel, complex128_t* const buf, const complex128_t* inp = nullptr) const;

    void reco_ft( const char channel, complex128_t* const buf, const complex128_t * const existing_r_exp, const uindex_t nr,
                            complex128_t * bufMPI = nullptr) const;

    void project_fullV(const diverge_model_t* const mod_ptr, complex128_t*,
                        const double * frq = nullptr) const;

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
    void init_U_from_fullV(diverge_model_t* model, projection_t* proj);
    void initialize_GPU_memory(projection_t* proj, tu_prop_t* loop);
    void init_U(bool enforce_exchange);
};


void add(Vertex &result, const Vertex &inp, const double factor) ;
void set(Vertex &f,const Vertex &initial,const Vertex &inp, const double factor);
void multiply(Vertex &f,const Vertex &inp, const double factor);
double infinity_norm(const Vertex &inp);
vector<double> max(const Vertex &inp);

#endif // __cplusplus
