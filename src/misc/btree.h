/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct btree_t btree_t;
struct btree_t {
    btree_t* parent;
    btree_t* children[2];
    index_t key;
    index_t val;
};

btree_t* btree_insert( btree_t* head, index_t key );
btree_t* btree_find( btree_t* head, index_t key );
void btree_iter( btree_t* head, void (*iter)(btree_t*, void*), void* ctx );
void btree_free( btree_t* head );

#ifdef __cplusplus
}
#endif
