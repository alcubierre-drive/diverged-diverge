/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hashtable_t hashtable_t;

hashtable_t* hashtable_init( index_t width, index_t depth );
hashtable_t* hashtable_init_fg( index_t width, index_t depth, index_t hashtable_empty_value );
void hashtable_free( hashtable_t* t );

void hashtable_insert( hashtable_t* t, index_t key, void* val );
void hashtable_insert_safe( hashtable_t* t, index_t key, void* val );
void hashtable_iterate( hashtable_t* t, void (*func)(index_t, void*, void*), void* ctx );
void hashtable_iterate_addr( hashtable_t* t, void (*func)(index_t*, void**, void*), void* ctx );
void* hashtable_find( hashtable_t* t, index_t key );
void* hashtable_pop( hashtable_t* t, index_t key );
void hashtable_delete( hashtable_t* t, index_t key );

#ifdef __cplusplus
}
#endif
