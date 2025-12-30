#define CT_SUITE_NAME dicts
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_common.h"
#include "../../src/misc/btree.h"
#include "../../src/misc/hashtable.h"

typedef void* void_ptr;
#define vtree_key_t index_t
#define vtree_val_t void_ptr
#include "../../src/misc/vtree.h"

CT_TEST_NAME( btree_vtree, "btree,vtree" ) {
    xbtree_t* xb = xbtree_init( 16 );
    xvtree_t* xv = xvtree_init( 16 );

    index_t n_insert = 1024;
    index_t max_val = 64;
    for (index_t i=0; i<n_insert; ++i) {
        index_t key = rand() % max_val;
        xbtree_find_or_insert( xb, key )->val = i;
        xvtree_find_or_insert( xv, key )->val = (void_ptr)i;
    }

    // iterate, but without function
    for (vtree_node_t* xn=vtree_node_minimum(xv->head); xn!=NULL; xn=vtree_node_next(xn)) {
        btree_t* bn = xbtree_find( xb, xn->key );
        CT_CHECK( bn != NULL );
        CT_CHECK( bn->key == xn->key );
        CT_CHECK( bn->val == ((index_t)xn->val) );
    }

    xbtree_free(xb);
    xvtree_free(xv);
}

CT_TEST_NAME( vtree_hashtable, "vtree,hashtable" ) {
    xvtree_t* xv = xvtree_init( 16 );
    void_ptr ht_empty = (void_ptr)-1;
    hashtable_t* ht = hashtable_init_fg( 16, 2, (index_t)ht_empty );

    index_t n_insert = 1024;
    index_t max_val = 64;
    for (index_t i=0; i<n_insert; ++i) {
        index_t key = rand() % max_val;
        void_ptr val = (void_ptr)i;
        xvtree_find_or_insert( xv, key )->val = val;
        hashtable_insert_safe( ht, key, val );
    }

    // iterate, but without function
    for (vtree_node_t* xn=vtree_node_minimum(xv->head); xn!=NULL; xn=vtree_node_next(xn)) {
        void* ht_val = hashtable_find(ht, xn->key);
        CT_CHECK( ht_val != ht_empty );
        CT_CHECK( ht_val == xn->val );
    }

    xvtree_free(xv);
    hashtable_free(ht);
}

#endif // DIVERGE_SKIP_TESTS
