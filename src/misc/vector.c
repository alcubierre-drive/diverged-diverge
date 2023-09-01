#include "vector.h"

void vector_init( vector_t* v, index_t size ) {
    v->_data_sz = 256;
    if ( size > v->_data_sz ) {
        v->_data_sz = size;
    }
    v->data = malloc(sizeof(index_t)*v->_data_sz);
    v->size = size;
}

void vector_set( vector_t* v, index_t i, index_t val ) {
    v->data[i] = val;
}

index_t vector_get( vector_t* v, index_t i ) {
    return v->data[i];
}

void vector_push_back( vector_t* v, index_t val ) {
    if (v->_data_sz <= v->size+1) {
        v->data = realloc(v->data, sizeof(index_t)*v->_data_sz*2);
        v->_data_sz *= 2;
    }
    v->data[v->size++] = val;
}

void vector_free( vector_t* v ) {
    free(v->data);
}

void vector_print( vector_t* v ) {
    for (index_t i=0; i<v->size; ++i) {
        printf("%li ", v->data[i]);
    }
}

