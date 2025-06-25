/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "hashtable.h"

#define HASHTABLE_EMPTY -1

typedef struct {
    index_t key;
    void* val;
} map_t;

struct hashtable_t {
    map_t* tab;
    index_t width;
    index_t depth;
};

uint32_t RSHash(const uint8_t* data, size_t size) {
    uint32_t a = 63689, b = 378551;
    uint32_t hash = 0;
    for (uint32_t i=0; i<size; i++) {
       hash = hash * a + data[i];
       a *= b;
    }
    return hash;
}
static index_t hash( index_t key ) { return RSHash((const uint8_t*)&key, sizeof(key)); }

hashtable_t* hashtable_init( index_t width, index_t depth ) {
    hashtable_t* t = calloc(1, sizeof*t);
    t->tab = malloc( width*depth * sizeof*t->tab );
    for (index_t i=0; i<width*depth; ++i) t->tab[i] = (map_t){.key=HASHTABLE_EMPTY, .val=NULL};
    t->width = width;
    t->depth = depth;
    return t;
}

static void hashtable_increase_depth( hashtable_t* t, index_t depth ) {
    printf( "increase depth\n" );
    t->tab = realloc( t->tab, sizeof*t->tab * t->width * depth );
    for (; t->depth<depth; ++(t->depth))
    for (index_t i=0; i<t->width; ++i)
        t->tab[IDX2(t->depth,i,t->width)] = (map_t){.key=HASHTABLE_EMPTY, .val=NULL};
}

void hashtable_free( hashtable_t* t ) {
    free( t->tab );
    free( t );
}

void hashtable_insert( hashtable_t* t, index_t key, void* val ) {
    index_t h = hash(key) % t->width;
    index_t d = 0;
    while (d < t->depth) {
        if (t->tab[IDX2(d,h,t->width)].key == HASHTABLE_EMPTY) break;
        d++;
    }
    if (d == t->depth)
        hashtable_increase_depth( t, t->depth*2 );
    t->tab[IDX2(d,h,t->width)] = (map_t){.key=key, .val=val};
}

void hashtable_insert_safe( hashtable_t* t, index_t key, void* val ) {
    index_t h = hash(key) % t->width;
    index_t d = 0;
    while (d < t->depth) {
        if (t->tab[IDX2(d,h,t->width)].key == HASHTABLE_EMPTY ||
            t->tab[IDX2(d,h,t->width)].key == key) break;
        d++;
    }
    if (d == t->depth)
        hashtable_increase_depth( t, t->depth*2 );
    t->tab[IDX2(d,h,t->width)] = (map_t){.key=key, .val=val};
}

void hashtable_iterate( hashtable_t* t, void (*func)(index_t, void*, void*), void* ctx ) {
    for (index_t i=0; i<t->width*t->depth; ++i)
        if (t->tab[i].key != HASHTABLE_EMPTY)
            (*func)( t->tab[i].key, t->tab[i].val, ctx );
}

void hashtable_iterate_addr( hashtable_t* t, void (*func)(index_t*, void**, void*), void* ctx ) {
    for (index_t i=0; i<t->width*t->depth; ++i)
        if (t->tab[i].key != HASHTABLE_EMPTY)
            (*func)( &t->tab[i].key, &t->tab[i].val, ctx );
}

void* hashtable_find( hashtable_t* t, index_t key ) {
    index_t h = hash(key) % t->width;
    for (index_t d=0; d<t->depth; ++d) {
        map_t c = t->tab[IDX2(d,h,t->width)];
        if (c.key == key) return c.val;
    }
    return NULL;
}

void* hashtable_pop( hashtable_t* t, index_t key ) {
    index_t h = hash(key) % t->width;
    for (index_t d=0; d<t->depth; ++d) {
        map_t* c = &t->tab[IDX2(d,h,t->width)];
        if (c->key == key) {
            c->key = HASHTABLE_EMPTY;
            return c->val;
        }
    }
    return NULL;
}

void hashtable_delete( hashtable_t* t, index_t key ) {
    hashtable_pop( t, key );
}

