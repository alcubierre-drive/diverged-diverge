/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "alloc.h"

rs_hopping_t* diverge_mem_alloc_rs_hopping_t( index_t num ) {
    return calloc(num, sizeof(rs_hopping_t));
}

rs_vertex_t* diverge_mem_alloc_rs_vertex_t( index_t num ) {
    return calloc(num, sizeof(rs_vertex_t));
}

tu_formfactor_t* diverge_mem_alloc_tu_formfactor_t( index_t num ) {
    return calloc(num, sizeof(tu_formfactor_t));
}

double* diverge_mem_alloc_complex128_t( index_t num ) {
    return calloc(num, sizeof(complex128_t));
}

void diverge_mem_free( void* addr ) {
    free(addr);
}

