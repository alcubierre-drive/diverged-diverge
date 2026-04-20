/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

// a shadow memory implementation of dynamic arrays based on malloc/realloc,
// wrapped into macros to be type agnostic. aligned to 512bit for avx extensions
// on vector's data.
//
// usage in functions works, also when passed as "pointer to array", e.g.,
//
// void do_complicated_thing(int** vector, ...) {
//      dv_vec_resize(*vector, 10);
//      dv_vec_push(*vector, -1);
//      dv_vec_push(*vector, -1);
//      dv_vec_push(*vector, -1);
//      for (int i=0; i<dv_vec_sz(*vector); ++i) {
//      }
//      dv_vec_free(*vector);
// }

#pragma once

#include <stdint.h>
#include <stdlib.h>

// make this 512bit padded for alignment of vector's memory
#ifndef DV_VEC_HEADER_PADDING_NBYTES
#define DV_VEC_HEADER_PADDING_NBYTES() (sizeof(int64_t)*(2+4))
#endif

#ifndef DV_VEC_INITIAL_CAP
#define DV_VEC_INITIAL_CAP 128
#endif

typedef struct {
    int64_t sz, cap;
    #if DV_VEC_HEADER_PADDING_NBYTES != 0
    char _padding[DV_VEC_HEADER_PADDING_NBYTES];
    #endif
} _dv_vec_header_t;

#define _dv_vec_get_header(ptr) ((ptr) ? (_dv_vec_header_t*)(ptr)-1 : NULL)

#define dv_vec_sz(ptr)  (_dv_vec_get_header(ptr) ? _dv_vec_get_header(ptr)->sz  : 0)
#define dv_vec_cap(ptr) (_dv_vec_get_header(ptr) ? _dv_vec_get_header(ptr)->cap : 0)

#define dv_vec_init_cap(ptr, CAP) do { \
    _dv_vec_header_t* _header = calloc(1, sizeof*_header + sizeof*ptr * CAP); \
    _header->cap = CAP; \
    ptr = (void*)(_header + 1); \
} while (0)

#define dv_vec_init(ptr) dv_vec_init_cap(ptr, DV_VEC_INITIAL_CAP)

#define dv_vec_push(ptr, ...) do { \
    _dv_vec_header_t* _header = _dv_vec_get_header(ptr); \
    if (!_header) { \
        dv_vec_init(ptr); \
        _header = _dv_vec_get_header(ptr); \
    } else if (_header->sz >= _header->cap) { \
        _header->cap *= 2; \
        _header = realloc(_header, sizeof*_header + sizeof*ptr * _header->cap); \
        ptr = (void*)(_header+1); \
    } \
    ptr[_header->sz++] = __VA_ARGS__; \
} while (0)

#define dv_vec_reserve(ptr, CAP) do { \
    if ((CAP) == 0) { \
        dv_vec_free(ptr); \
        (ptr) = NULL; \
    } else { \
        _dv_vec_header_t* _header = _dv_vec_get_header(ptr); \
        if (!_header) { \
            dv_vec_init_cap(ptr, CAP); \
            _header = _dv_vec_get_header(ptr); \
        } else { \
            _header->cap = CAP; \
            _header = realloc(_header, sizeof*_header + sizeof*ptr * _header->cap); \
            ptr = (void*)(_header+1); \
        } \
    } \
} while (0)

#define dv_vec_resize_lazy(ptr, SZ) do { \
    _dv_vec_header_t* _header = _dv_vec_get_header(ptr); \
    if (_header) { \
        _header->sz = SZ; \
    } \
} while (0)

#define dv_vec_resize(ptr, SZ) do { \
    dv_vec_reserve(ptr, SZ); \
    _dv_vec_header_t* _header = _dv_vec_get_header(ptr); \
    if (_header) { \
        _header->sz = (SZ); \
    } \
} while (0)

#define dv_vec_shrink(ptr) do { \
    _dv_vec_header_t* _header = _dv_vec_get_header(ptr); \
    if (_header) { \
        _header->cap = _header->sz; \
        _header = realloc(_header, sizeof*_header + sizeof*ptr * _header->sz); \
        if (!_header) { \
            (ptr) = NULL; \
        } else { \
            ptr = (void*)(_header+1); \
        } \
    } \
} while (0)

#define dv_vec_free(ptr) free(_dv_vec_get_header(ptr))
