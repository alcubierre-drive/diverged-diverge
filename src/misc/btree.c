#include "btree.h"

typedef enum {
    btree_left = 0,
    btree_right,
} btree_insertion_sel;

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

void btree_iter( btree_t* head, void (*iter)(btree_t*, void*), void* ctx ) {
    if (head) {
        btree_iter( head->children[btree_left], iter, ctx );
        btree_iter( head->children[btree_right], iter, ctx );
        (*iter)( head, ctx );
    }
}

static inline void btree_iter_free(btree_t* node, void* ctx) {
    (void)ctx;
    free(node);
}

void btree_free( btree_t* head ) {
    btree_iter( head, &btree_iter_free, NULL );
}

