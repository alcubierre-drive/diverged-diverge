#define CT_SUITE_NAME containers
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_common.h"
#include "../../src/misc/hashtable.h"
#include "../../src/misc/dv_vector.h"

/*
typedef void* void_ptr;
#define vtree_key_t index_t
#define vtree_val_t void_ptr
#include "../../src/misc/vtree.h"

CT_TEST_NAME( dict_vtree_hashtable, "dict,vtree,hashtable" ) {
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
*/

CT_TEST(dv_vector_init_and_basic_properties) {
    int *v = NULL;

    CT_CHECK(dv_vec_sz(v) == 0);
    CT_CHECK(dv_vec_cap(v) == 0);

    dv_vec_init(v);

    CT_CHECK(dv_vec_cap(v) > 0);
    CT_CHECK(dv_vec_sz(v) == 0);

    dv_vec_free(v);
}

CT_TEST(dv_vector_push_elements) {
    int *v = NULL;

    for (int i=0; i<100; ++i) dv_vec_push(v, i);

    CT_CHECK(dv_vec_sz(v) == 100);
    for (int i=0; i<100; ++i) CT_CHECK(v[i] == i);

    dv_vec_free(v);
}

CT_TEST(dv_vector_automatic_growth) {
    int *v = NULL;

    int initial_cap = DV_VEC_INITIAL_CAP;

    for (int i=0; i<initial_cap+10; ++i) dv_vec_push(v, i);

    CT_CHECK(dv_vec_cap(v) >= (initial_cap * 2));
    CT_CHECK(dv_vec_sz(v) == (initial_cap + 10));

    dv_vec_free(v);
}

CT_TEST(dv_vector_reserve_and_resize) {
    int *v = NULL;

    dv_vec_reserve(v, 50);

    CT_CHECK(dv_vec_cap(v) == 50);

    dv_vec_resize(v, 20);

    CT_CHECK(dv_vec_sz(v) == 20);
    CT_CHECK(dv_vec_cap(v) == 20);

    dv_vec_free(v);
}

CT_TEST(dv_vector_resize_lazy) {
    int *v = NULL;

    dv_vec_reserve(v, 100);
    dv_vec_resize_lazy(v, 10);

    CT_CHECK(dv_vec_sz(v) == 10);
    CT_CHECK(dv_vec_cap(v) == 100);

    dv_vec_free(v);
}

CT_TEST(dv_vector_shrink) {
    int *v = NULL;

    for (int i = 0; i < 50; ++i) {
        dv_vec_push(v, i);
    }

    dv_vec_resize_lazy(v, 10);
    dv_vec_shrink(v);

    CT_CHECK(dv_vec_sz(v) == 10);
    CT_CHECK(dv_vec_cap(v) == 10);

    dv_vec_free(v);
}

CT_TEST(dv_vector_reserve_zero_frees) {
    int *v = NULL;

    dv_vec_push(v, 1);
    CT_CHECK(v != NULL);

    dv_vec_reserve(v, 0);
    CT_CHECK(v == NULL);
}

CT_TEST(dv_vector_multiple_types) {
    double *vd = NULL;

    for (int i=0; i<10; ++i) dv_vec_push(vd, (double)i * 0.5);

    CT_CHECK(dv_vec_sz(vd) == 10);
    for (int i=0; i<10; ++i) CT_CHECK(vd[i] == (double)i * 0.5);

    dv_vec_free(vd);
}

CT_TEST(dv_vector_pointer_stability_after_realloc) {
    int *v = NULL;

    for (int i=0; i<1000; ++i) dv_vec_push(v, i);

    CT_CHECK(dv_vec_sz(v) == 1000);
    for (int i=0; i<1000; ++i) CT_CHECK(v[i] == i);

    dv_vec_free(v);
}

/*
// TODO
VECTOR_DECL
VECTOR_DECL_INIT_CAP
VECTOR_INIT_CAP
VECTOR_DECL_INIT
VECTOR_INIT
VECTOR_PUSH_BACK
VECTOR_RESERVE
VECTOR_RESIZE
VECTOR_SHRINK_TO_FIT
VECTOR_FREE
VECTOR_COPY
*/

#endif // DIVERGE_SKIP_TESTS
