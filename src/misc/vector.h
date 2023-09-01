#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vector_t {
    index_t* data;
    index_t size;
    index_t _data_sz;
};
typedef struct vector_t vector_t;

void vector_init( vector_t* v, index_t size );
void vector_set( vector_t* v, index_t i, index_t val );
index_t vector_get( vector_t* v, index_t i );
void vector_push_back( vector_t* v, index_t val );
void vector_free( vector_t* v );
void vector_print( vector_t* v );

#ifdef __cplusplus
}
#endif
