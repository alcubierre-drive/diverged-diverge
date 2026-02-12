/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#ifndef vhash_key_t
#define vhash_key_t index_t
#endif//vhash_key_t
#ifndef vhash_key_empty
#define vhash_key_empty -1
#endif

#ifndef vhash_val_t
typedef void* void_ptr_t;
#define vhash_val_t void_ptr_t
#endif//vhash_val_t
#ifndef vhash_val_empty
#define vhash_val_empty NULL
#endif

#define VHASH_IMPLEMENT(PREFIX) \
 \
typedef struct { \
    vhash_key_t key; \
    vhash_val_t val; \
} PREFIX##map_t; \
 \
typedef struct { \
    PREFIX##map_t* tab; \
    index_t width; \
    index_t depth; \
} PREFIX##table_t; \
 \
static inline uint64_t PREFIX##splitmix64(uint64_t x) { \
    x += 0x9E3779B97F4A7C15ULL; \
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; \
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; \
    x = x ^ (x >> 31); \
    x &= ~(1ULL << 63); /* clear the sign bit */ \
    return x; \
} \
 \
static inline uint64_t PREFIX##sp64_bytes_load_u64(const uint8_t* ptr, uint64_t ptr_nbytes) { \
    uint64_t x = 0; \
    memcpy(&x, ptr, MIN(ptr_nbytes,8)); \
    return x; \
} \
 \
static inline uint64_t PREFIX##sp64_bytes(const uint8_t *data, uint64_t nbytes) { \
    uint64_t h = 0x9E3779B97F4A7C15ULL; /* seed */ \
    for (uint64_t i=0; i<nbytes; i+=8) { \
        h += PREFIX##sp64_bytes_load_u64(data + i, nbytes-i); \
        h = PREFIX##splitmix64(h); \
    } \
    return h; \
} \
 \
static inline PREFIX##table_t* PREFIX##table_init( index_t width, index_t depth ) { \
    PREFIX##table_t* t = calloc(1, sizeof*t); \
    t->tab = malloc( width*depth * sizeof*t->tab ); \
    for (index_t i=0; i<width*depth; ++i) t->tab[i] = (PREFIX##map_t){.key=vhash_key_empty, .val=vhash_val_empty}; \
    t->width = width; \
    t->depth = depth; \
    return t; \
} \
 \
static inline void PREFIX##table_increase_depth( PREFIX##table_t* t, index_t depth ) { \
    printf( "PREFIX##table increase depth (%li->%li)\n", t->depth, depth ); \
    t->tab = realloc( t->tab, sizeof*t->tab * t->width * depth ); \
    for (; t->depth<depth; ++(t->depth)) \
    for (index_t i=0; i<t->width; ++i) \
        t->tab[IDX2(t->depth,i,t->width)] = (PREFIX##map_t){.key=vhash_key_empty, .val=vhash_val_empty}; \
} \
 \
static inline void PREFIX##table_free( PREFIX##table_t* t ) { \
    free( t->tab ); \
    free( t ); \
} \
 \
static inline void PREFIX##table_insert( PREFIX##table_t* t, vhash_key_t key, vhash_val_t val ) { \
    index_t h = PREFIX##sp64_bytes((const uint8_t*)&key, sizeof(key)) % t->width; \
    index_t d = 0; \
    while (d < t->depth) { \
        if (t->tab[IDX2(d,h,t->width)].key == vhash_key_empty) break; \
        d++; \
    } \
    if (d == t->depth) \
        PREFIX##table_increase_depth( t, t->depth*2 ); \
    t->tab[IDX2(d,h,t->width)] = (PREFIX##map_t){.key=key, .val=val}; \
} \
 \
static inline void PREFIX##table_insert_safe( PREFIX##table_t* t, vhash_key_t key, vhash_val_t val ) { \
    index_t h = PREFIX##sp64_bytes((const uint8_t*)&key, sizeof(key)) % t->width; \
    index_t d = 0; \
    while (d < t->depth) { \
        if (t->tab[IDX2(d,h,t->width)].key == vhash_key_empty || \
            t->tab[IDX2(d,h,t->width)].key == key) break; \
        d++; \
    } \
    if (d == t->depth) \
        PREFIX##table_increase_depth( t, t->depth*2 ); \
    t->tab[IDX2(d,h,t->width)] = (PREFIX##map_t){.key=key, .val=val}; \
} \
 \
static inline void PREFIX##table_iterate( PREFIX##table_t* t, void (*func)(vhash_key_t, vhash_val_t, void*), void* ctx ) { \
    for (index_t i=0; i<t->width*t->depth; ++i) \
        if (t->tab[i].key != vhash_key_empty) \
            (*func)( t->tab[i].key, t->tab[i].val, ctx ); \
} \
 \
static inline void PREFIX##table_iterate_addr( PREFIX##table_t* t, void (*func)(vhash_key_t*, vhash_val_t*, void*), void* ctx ) { \
    for (index_t i=0; i<t->width*t->depth; ++i) \
        if (t->tab[i].key != vhash_key_empty) \
            (*func)( &t->tab[i].key, &t->tab[i].val, ctx ); \
} \
 \
static inline vhash_val_t PREFIX##table_find( PREFIX##table_t* t, vhash_key_t key ) { \
    index_t h = PREFIX##sp64_bytes((const uint8_t*)&key, sizeof(key)) % t->width; \
    for (index_t d=0; d<t->depth; ++d) { \
        PREFIX##map_t c = t->tab[IDX2(d,h,t->width)]; \
        if (c.key == key) return c.val; \
    } \
    return vhash_val_empty; \
} \
 \
static inline vhash_val_t PREFIX##table_pop( PREFIX##table_t* t, vhash_key_t key ) { \
    index_t h = PREFIX##sp64_bytes((const uint8_t*)&key, sizeof(key)) % t->width; \
    for (index_t d=0; d<t->depth; ++d) { \
        PREFIX##map_t* c = &t->tab[IDX2(d,h,t->width)]; \
        if (c->key == key) { \
            c->key = vhash_key_empty; \
            return c->val; \
        } \
    } \
    return vhash_val_empty; \
} \
 \
static inline void PREFIX##table_delete( PREFIX##table_t* t, vhash_key_t key ) { \
    PREFIX##table_pop( t, key ); \
}
