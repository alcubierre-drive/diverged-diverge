/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include <string.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef vtree_key_t
#define vtree_key_t int
#endif
#ifndef vtree_val_t
#define vtree_val_t int
#endif
#ifndef vtree_padding_bytes
#define VTREE_PADDING_DISABLED
#define vtree_padding_bytes 0
#endif

// API
typedef struct vtree_node_t vtree_node_t;
struct vtree_node_t {
    vtree_key_t key;
    vtree_val_t val;
    vtree_node_t* parent;
#ifndef VTREE_PADDING_DISABLED
    char _padding[vtree_padding_bytes];
#endif
    vtree_node_t* children[2];
};

#ifndef VTREE_PADDING_DISABLED
#define vtree_padding_offset offsetof(vtree_node_t, _padding)
#else // VTREE_PADDING_DISABLED
#define vtree_padding_offset offsetof(vtree_node_t, children)
#endif // VTREE_PADDING_DISABLED

typedef struct xvtree_t xvtree_t;
static inline xvtree_t* xvtree_init( size_t nreserve );
static inline void xvtree_free( xvtree_t* x );
static inline vtree_node_t* xvtree_find( xvtree_t* x, vtree_key_t key );
static inline vtree_node_t* xvtree_insert( xvtree_t* x, vtree_key_t key );
static inline vtree_node_t* xvtree_find_or_insert( xvtree_t* x, vtree_key_t key );
static inline void xvtree_iter( xvtree_t* x, void (*iter)(vtree_node_t*, void*), void* ctx );
static inline void xvtree_iter_ascending( xvtree_t* x, void (*iter)(vtree_node_t*, void*), void* ctx );

// IMPLEMENTATION
#ifndef vtree_key_cmp
static inline int vtree_key_cmp( vtree_key_t a, vtree_key_t b ) { return memcmp( &a, &b, sizeof(a) ); }
#endif
typedef enum { vtree_left = 0, vtree_right, } vtree_insertion_sel;

static inline void vtree_node_iter( vtree_node_t* head, void (*iter)(vtree_node_t*, void*), void* ctx ) {
    if (head) {
        vtree_node_iter( head->children[vtree_left], iter, ctx );
        vtree_node_iter( head->children[vtree_right], iter, ctx );
        (*iter)( head, ctx );
    }
}

static inline vtree_node_t* vtree_node_minimum( vtree_node_t* head ) {
    if (!head) return NULL;
    vtree_node_t* tmp = head;
    while (tmp->children[vtree_left]) tmp = tmp->children[vtree_left];
    return tmp;
}

static inline vtree_node_t* vtree_node_next( vtree_node_t* node ) {
    if (!node) return NULL;
    vtree_node_t* next;
    if (!node->children[vtree_right]) {
        next = node;
        while (next->parent && next == next->parent->children[vtree_right])
            next = next->parent;
        next = next->parent;
    } else {
        next = node->children[vtree_right];
        while (next->children[vtree_left]) next = next->children[vtree_left];
    }
    return next;
}

typedef struct xvtree_t xvtree_t;
struct xvtree_t {
    vtree_node_t* head;
    vtree_node_t** mem;

    size_t n_mem;
    size_t i_mem;

    size_t mem_sz;
    size_t mem_idx;
};

static inline xvtree_t* xvtree_init( size_t nreserve ) {
    xvtree_t* x = calloc(1, sizeof*x);

    mpi_vrb_printf( "vtree_node_t alignment: "
            "key@%li(+%li), val@%li(+%li), parent@%li(+%li), "
            "padding@%li(+%li), children@%li(+%li) ==> size=%li\n" ,
            offsetof(vtree_node_t, key), sizeof(x->head->key),
            offsetof(vtree_node_t, val), sizeof(x->head->val),
            offsetof(vtree_node_t, parent), sizeof(x->head->parent),
            vtree_padding_offset, vtree_padding_bytes,
            offsetof(vtree_node_t, children), sizeof(x->head->children),
            sizeof(*x->head) );

    x->n_mem = 1;
    x->i_mem = 0;
    x->mem = calloc(1, sizeof*(x->mem));
    x->mem[0] = calloc(nreserve, sizeof(vtree_node_t));
    x->mem_sz = nreserve;
    x->mem_idx = 0;
    return x;
}

static inline vtree_node_t* xvtree_next_slot( xvtree_t* x ) {
    if (++(x->mem_idx) < x->mem_sz) {
        return x->mem[x->i_mem] + x->mem_idx;
    } else {
        x->mem = realloc(x->mem, (sizeof(*(x->mem)) * ++x->n_mem));
        x->mem[++x->i_mem] = calloc(x->mem_sz, sizeof(vtree_node_t));
        return x->mem[x->i_mem] + (x->mem_idx = 0);
    }
}

static inline void xvtree_free( xvtree_t* x ) {
    for (size_t i=0; i<x->n_mem; ++i)
        free( x->mem[i] );
    free( x->mem );
    free( x );
}

static inline vtree_node_t* xvtree_find( xvtree_t* x, vtree_key_t key ) {
    vtree_node_t* tmp = x->head;
    while (tmp) {
        if (!vtree_key_cmp(tmp->key,key)) return tmp;
        tmp = tmp->children[vtree_key_cmp(key,tmp->key) < 0 ? vtree_left : vtree_right];
    }
    return tmp;
}

static inline vtree_node_t* xvtree_insert( xvtree_t* x, vtree_key_t key ) {
    vtree_node_t* node = xvtree_next_slot( x );
    node->key = key;

    if (!x->head)
        return (x->head = node);

    vtree_node_t* tmp = x->head;
    vtree_node_t* tmp_parent = NULL;
    vtree_insertion_sel where;

    while (tmp) {
        tmp_parent = tmp;
        tmp = tmp->children[(where = vtree_key_cmp(key,tmp->key) < 0 ? vtree_left : vtree_right)];
    }

    tmp_parent->children[where] = node;
    node->parent = tmp_parent;
    return node;
}

static inline vtree_node_t* xvtree_find_or_insert( xvtree_t* x, vtree_key_t key ) {
    if (!x->head)
        return xvtree_insert( x, key );

    vtree_node_t* tmp = x->head;
    vtree_node_t* tmp_parent = NULL;
    vtree_insertion_sel where;

    while (tmp) {
        if (!vtree_key_cmp(tmp->key,key)) return tmp;
        tmp_parent = tmp;
        tmp = tmp->children[(where = vtree_key_cmp(key,tmp->key) < 0 ? vtree_left : vtree_right)];
    }

    vtree_node_t* node = xvtree_next_slot(x);
    tmp_parent->children[where] = node;
    node->key = key;
    node->parent = tmp_parent;
    return node;
}

static inline void xvtree_iter( xvtree_t* x, void (*iter)(vtree_node_t*, void*), void* ctx ) {
    vtree_node_t* head = x->head;
    if (head) {
        vtree_node_iter( head->children[vtree_left], iter, ctx );
        vtree_node_iter( head->children[vtree_right], iter, ctx );
        (*iter)( head, ctx );
    }
}

static inline void xvtree_iter_ascending( xvtree_t* x, void (*iter)(vtree_node_t*, void*), void* ctx ) {
    for (vtree_node_t* start=vtree_node_minimum(x->head); start!=NULL; start=vtree_node_next(start))
        (*iter)( start, ctx );
}

#ifdef __cplusplus
}
#endif
