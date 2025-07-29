#include "btree.h"

btree_t* btree_insert( btree_t* head, index_t key ) {
    btree_t* node = calloc(1, sizeof*node);
    node->key = key;

    if (!head)
        return node;

    btree_t* tmp = head;
    btree_t* tmp_parent = NULL;
    btree_insertion_sel where;

    while (tmp) {
        tmp_parent = tmp;
        tmp = tmp->children[(where = key < tmp->key ? btree_left : btree_right)];
    }

    tmp_parent->children[where] = node;
    node->parent = tmp_parent;
    return node;
}

btree_t* btree_find( btree_t* head, index_t key ) {
    btree_t* tmp = head;
    while (tmp) {
        if (tmp->key == key) return tmp;
        tmp = tmp->children[key < tmp->key ? btree_left : btree_right];
    }
    return tmp;
}

static inline void btree_iter_free(btree_t* node, void* ctx) {
    (void)ctx;
    free(node);
}

void btree_free( btree_t* head ) {
    btree_iter( head, &btree_iter_free, NULL );
}

btree_t* btree_minimum( btree_t* head ) {
    if (!head) return NULL;
    btree_t* tmp = head;
    while (tmp->children[btree_left]) tmp = tmp->children[btree_left];
    return tmp;
}

btree_t* btree_next( btree_t* node ) {
    if (!node) return NULL;
    btree_t* next;
    if (!node->children[btree_right]) {
        next = node;
        while (next->parent && next == next->parent->children[btree_right])
            next = next->parent;
        next = next->parent;
    } else {
        next = node->children[btree_right];
        while (next->children[btree_left]) next = next->children[btree_left];
    }
    return next;
}

xbtree_t* xbtree_init( size_t nreserve ) {
    xbtree_t* x = calloc(1, sizeof*x);
    x->n_mem = 1;
    x->i_mem = 0;
    x->mem = calloc(1, sizeof*(x->mem));
    x->mem[0] = calloc(nreserve, sizeof(btree_t));
    x->mem_sz = nreserve;
    x->mem_idx = 0;
    return x;
}

static inline btree_t* xbtree_next_slot( xbtree_t* x ) {
    if (++(x->mem_idx) < x->mem_sz) {
        return x->mem[x->i_mem] + x->mem_idx;
    } else {
        x->mem = realloc(x->mem, (sizeof(*(x->mem)) * ++x->n_mem));
        x->mem[++x->i_mem] = calloc(x->mem_sz, sizeof(btree_t));
        return x->mem[x->i_mem] + (x->mem_idx = 0);
    }
}

void xbtree_free( xbtree_t* x ) {
    for (size_t i=0; i<x->n_mem; ++i)
        free( x->mem[i] );
    free( x->mem );
    free( x );
}

btree_t* xbtree_find( xbtree_t* x, index_t key ) {
    btree_t* tmp = x->head;
    while (tmp) {
        if (tmp->key == key) return tmp;
        tmp = tmp->children[key < tmp->key ? btree_left : btree_right];
    }
    return tmp;
}
btree_t* xbtree_insert( xbtree_t* x, index_t key ) {
    btree_t* node = xbtree_next_slot( x );
    node->key = key;

    if (!x->head)
        return (x->head = node);

    btree_t* tmp = x->head;
    btree_t* tmp_parent = NULL;
    btree_insertion_sel where;

    while (tmp) {
        tmp_parent = tmp;
        tmp = tmp->children[(where = key < tmp->key ? btree_left : btree_right)];
    }

    tmp_parent->children[where] = node;
    node->parent = tmp_parent;
    return node;
}
btree_t* xbtree_find_or_insert( xbtree_t* x, index_t key ) {
    if (!x->head)
        return xbtree_insert( x, key );

    btree_t* tmp = x->head;
    btree_t* tmp_parent = NULL;
    btree_insertion_sel where;

    while (tmp) {
        if (tmp->key == key) return tmp;
        tmp_parent = tmp;
        tmp = tmp->children[(where = key < tmp->key ? btree_left : btree_right)];
    }

    btree_t* node = xbtree_next_slot(x);
    tmp_parent->children[where] = node;
    node->key = key;
    node->parent = tmp_parent;
    return node;
}
