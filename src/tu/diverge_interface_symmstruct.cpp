/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_interface_symmstruct.hpp"

template<typename T>
char* push_value( char* ptr, uindex_t* p_sz, const T& val ) {
    uindex_t sz = *p_sz;
    uindex_t add = sizeof(T);
    ptr = (char*)realloc(ptr, sz + add);
    memcpy( ptr + sz, &val, add );
    *p_sz += add;
    return ptr;
}

template<typename T>
char* push_vector( char* ptr, uindex_t* p_sz, const vector<T>& vec ) {
    uindex_t sz = *p_sz;
    uindex_t add = sizeof(T) * vec.size() + sizeof(uindex_t);
    ptr = (char*)realloc(ptr, sz + add);

    uindex_t vec_sz = vec.size();
    memcpy( ptr + sz, &vec_sz, sizeof(uindex_t) );
    memcpy( ptr + sz + sizeof(uindex_t), vec.data(), sizeof(T) * vec.size() );

    *p_sz += add;
    return ptr;
}

void symmstruct_serialize( const symmstruct_t* symm, char** ptr_, uindex_t* nbytes_ ) {
    char* ptr = NULL;
    uindex_t size = 0;
    ptr = push_value( ptr, &size, symm->use_symmetries );
    ptr = push_vector( ptr, &size, symm->inv_symmetry );
    ptr = push_vector( ptr, &size, symm->mi_prefac_in_out );
    ptr = push_vector( ptr, &size, symm->mi_prefac_in_in );
    ptr = push_vector( ptr, &size, symm->mi_maps_to );
    ptr = push_vector( ptr, &size, symm->mi_map_len );
    ptr = push_vector( ptr, &size, symm->mi_map_off );
    ptr = push_vector( ptr, &size, symm->beyond_UC );
    ptr = push_vector( ptr, &size, symm->kmaps_to );
    ptr = push_vector( ptr, &size, symm->idx_ibz_in_fullmesh );
    ptr = push_vector( ptr, &size, symm->which_symmetry_used );
    ptr = push_vector( ptr, &size, symm->prefactor_symmP );
    ptr = push_vector( ptr, &size, symm->prefactor_symmCD );
    ptr = push_vector( ptr, &size, symm->idx_map_symm );
    ptr = push_vector( ptr, &size, symm->o2m_map_len );
    ptr = push_vector( ptr, &size, symm->o2m_map_off );
    ptr = push_vector( ptr, &size, symm->from_ibz_2_bz );
    ptr = push_vector( ptr, &size, symm->from_ibz_2_bz_off );
    ptr = push_vector( ptr, &size, symm->from_ibz_2_bz_len );
    ptr = push_value( ptr, &size, symm->my_nk_in_bz );
    ptr = push_value( ptr, &size, symm->my_nk_in_bz_off );
    *ptr_ = ptr;
    *nbytes_ = size;
}

template<typename T>
T pop_value( const char* ptr, uindex_t* p_off ) {
    T val;
    memcpy( &val, ptr + *p_off, sizeof(T) );
    *p_off += sizeof(T);
    return val;
}

template<typename T>
vector<T> pop_vector( const char* ptr, uindex_t* p_off ) {
    uindex_t vec_sz = pop_value<uindex_t>( ptr, p_off );
    vector<T> vec(vec_sz);
    memcpy( (void*)vec.data(), ptr + *p_off, sizeof(T)*vec_sz );
    *p_off += sizeof(T)*vec_sz;
    return vec;
}

symmstruct_t* symmstruct_deserialize( const char* ptr ) {
    symmstruct_t* symm = new symmstruct_t;
    uindex_t off = 0;
    symm->use_symmetries = pop_value<bool>( ptr, &off );
    symm->inv_symmetry = pop_vector<index_t>( ptr, &off );
    symm->mi_prefac_in_out = pop_vector<complex128_t>( ptr, &off );
    symm->mi_prefac_in_in = pop_vector<complex128_t>( ptr, &off );
    symm->mi_maps_to = pop_vector<index_t>( ptr, &off );
    symm->mi_map_len = pop_vector<index_t>( ptr, &off );
    symm->mi_map_off = pop_vector<index_t>( ptr, &off );
    symm->beyond_UC = pop_vector<Vec3d>( ptr, &off );
    symm->kmaps_to = pop_vector<index_t>( ptr, &off );
    symm->idx_ibz_in_fullmesh = pop_vector<index_t>( ptr, &off );
    symm->which_symmetry_used = pop_vector<array<index_t,2>>( ptr, &off );
    symm->prefactor_symmP = pop_vector<complex128_t>( ptr, &off );
    symm->prefactor_symmCD = pop_vector<complex128_t>( ptr, &off );
    symm->idx_map_symm = pop_vector<index_t>( ptr, &off );
    symm->o2m_map_len = pop_vector<index_t>( ptr, &off );
    symm->o2m_map_off = pop_vector<index_t>( ptr, &off );
    symm->from_ibz_2_bz = pop_vector<index_t>( ptr, &off );
    symm->from_ibz_2_bz_off = pop_vector<index_t>( ptr, &off );
    symm->from_ibz_2_bz_len = pop_vector<index_t>( ptr, &off );
    symm->my_nk_in_bz = pop_value<index_t>( ptr, &off );
    symm->my_nk_in_bz_off = pop_value<index_t>( ptr, &off );
    return symm;
}

#define VECTOR_CMP_ERROR_SIZE 12345

static inline int operator!=( const vector<index_t>& va, const vector<index_t>& vb ) {
    int not_equals = 0;
    if (va.size() != vb.size()) return VECTOR_CMP_ERROR_SIZE;
    for (uindex_t i=0; i<va.size(); ++i)
        not_equals += va[i] != vb[i];
    return not_equals;
}

static inline int operator!=( const vector<complex128_t>& va, const vector<complex128_t>& vb ) {
    int not_equals = 0;
    if (va.size() != vb.size()) return VECTOR_CMP_ERROR_SIZE;
    for (uindex_t i=0; i<va.size(); ++i)
        not_equals += std::abs(va[i] - vb[i]) > 1.e-5;
    return not_equals;
}

static inline int operator!=( const vector<Vec3d>& va, const vector<Vec3d>& vb ) {
    int not_equals = 0;
    if (va.size() != vb.size()) return VECTOR_CMP_ERROR_SIZE;
    for (uindex_t i=0; i<va.size(); ++i)
        not_equals += (va[i] - vb[i]).norm() > 1.e-5;
    return not_equals;
}

static inline int operator!=( const array<index_t,2>& aa, const array<index_t,2>& ab ) {
    return (aa[0] != ab[0]) || (aa[1] != ab[1]);
}

static inline int operator!=( const vector<array<index_t,2>>& va, const vector<array<index_t,2>>& vb ) {
    int not_equals = 0;
    if (va.size() != vb.size()) return VECTOR_CMP_ERROR_SIZE;
    for (uindex_t i=0; i<va.size(); ++i)
        not_equals += va[i] != vb[i];
    return not_equals;
}

int symmstruct_equal( const symmstruct_t* A, const symmstruct_t* B ) {
    int errors = 0;
    errors += A->use_symmetries != B->use_symmetries;
    errors += A->inv_symmetry != B->inv_symmetry;

    errors += A->mi_prefac_in_out != B->mi_prefac_in_out;
    errors += A->mi_prefac_in_in != B->mi_prefac_in_in;
    errors += A->mi_maps_to != B->mi_maps_to;
    errors += A->mi_map_len != B->mi_map_len;
    errors += A->mi_map_off != B->mi_map_off;
    errors += A->beyond_UC != B->beyond_UC;
    errors += A->kmaps_to != B->kmaps_to;
    errors += A->idx_ibz_in_fullmesh != B->idx_ibz_in_fullmesh;
    errors += A->which_symmetry_used != B->which_symmetry_used;

    /* those are MPI specific (due to my_nk)
    errors += A->prefactor_symmP != B->prefactor_symmP;
    errors += A->prefactor_symmCD != B->prefactor_symmCD;
    errors += A->idx_map_symm != B->idx_map_symm;
    errors += A->o2m_map_len != B->o2m_map_len;
    errors += A->o2m_map_off != B->o2m_map_off;
    */

    errors += A->from_ibz_2_bz != B->from_ibz_2_bz;
    errors += A->from_ibz_2_bz_off != B->from_ibz_2_bz_off;
    errors += A->from_ibz_2_bz_len != B->from_ibz_2_bz_len;

    return errors;
}
