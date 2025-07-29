/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    btree_left = 0,
    btree_right,
} btree_insertion_sel;

// very barebones implementation
typedef struct btree_t btree_t;
struct btree_t {
    btree_t* parent;
    btree_t* children[2];
    index_t key;
    index_t val;
};
btree_t* btree_insert( btree_t* head, index_t key );
btree_t* btree_find( btree_t* head, index_t key );
static inline void btree_iter( btree_t* head, void (*iter)(btree_t*, void*), void* ctx ) {
    if (head) {
        btree_iter( head->children[btree_left], iter, ctx );
        btree_iter( head->children[btree_right], iter, ctx );
        (*iter)( head, ctx );
    }
}
void btree_free( btree_t* head );

btree_t* btree_minimum( btree_t* head );
btree_t* btree_next( btree_t* node );

// a bit more, can do find_or_insert and operates on contiguous memory
typedef struct xbtree_t xbtree_t;
struct xbtree_t {
    btree_t* head;
    btree_t** mem;

    size_t n_mem;
    size_t i_mem;

    size_t mem_sz;
    size_t mem_idx;
};

xbtree_t* xbtree_init( size_t nreserve );
void xbtree_free( xbtree_t* x );

btree_t* xbtree_find( xbtree_t* x, index_t key );
btree_t* xbtree_insert( xbtree_t* x, index_t key );
btree_t* xbtree_find_or_insert( xbtree_t* x, index_t key );
static inline void xbtree_iter( xbtree_t* x, void (*iter)(btree_t*, void*), void* ctx ) {
    btree_t* head = x->head;
    if (head) {
        btree_iter( head->children[btree_left], iter, ctx );
        btree_iter( head->children[btree_right], iter, ctx );
        (*iter)( head, ctx );
    }
}
static inline void xbtree_iter_ascending( xbtree_t* x, void (*iter)(btree_t*, void*), void* ctx ) {
    for (btree_t* start=btree_minimum(x->head); start!=NULL; start=btree_next(start))
        (*iter)( start, ctx );
}

#ifdef __cplusplus
}
#endif
