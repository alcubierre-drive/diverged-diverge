/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "dependency_tree.h"
#include "hashtable.h"
#include "../diverge_flow_step.h"
#include "../diverge_flow_step_internal.h"
#include <pthread.h>

typedef struct {
    hashtable_t* mod;
    hashtable_t* step;
    pthread_mutex_t mtx;
} deptree_t;

static inline void printsteps( index_t key, void* val, void* ctx ) {
    (void)val; (void)ctx; void* step = (void*)key;
    mpi_wrn_printf( "flow step %p dangling, please free resources\n", step );
}

static inline void printmods( index_t key, void* val, void* ctx ) {
    (void)val; (void)ctx; void* mod = (void*)key;
    mpi_wrn_printf( "model %p dangling, please free resources\n", mod );
}

static void deptree_free( deptree_t* d ) {
    pthread_mutex_lock( &d->mtx );
    hashtable_iterate( d->step, &printsteps, NULL );
    hashtable_free( d->step );
    hashtable_iterate( d->mod, &printmods, NULL );
    hashtable_free( d->mod );
    pthread_mutex_unlock( &d->mtx );
}

static void deptree_init( deptree_t* d ) {
    pthread_mutex_lock( &d->mtx );
    d->mod = hashtable_init(32, 4);
    d->step = hashtable_init(32, 4);
    pthread_mutex_unlock( &d->mtx );
}

typedef struct { diverge_model_t* mod; diverge_flow_step_t* found; } findmodel_t;
static inline void findmodel( index_t key, void* val, void* ctx ) {
    findmodel_t* f = ctx;
    if (f->mod == val) f->found = (diverge_flow_step_t*)key;
}
static diverge_flow_step_t* deptree_model_locked( deptree_t* d, diverge_model_t* mod ) {
    pthread_mutex_lock( &d->mtx );
    findmodel_t f = { mod, 0 };
    hashtable_iterate( d->step, &findmodel, &f );
    pthread_mutex_unlock( &d->mtx );
    return f.found;
}

void deptree_insert_model( deptree_t* d, diverge_model_t* mod ) {
    pthread_mutex_lock( &d->mtx );
    hashtable_insert_safe( d->mod, (index_t)mod, (void*)-1 );
    pthread_mutex_unlock( &d->mtx );
}

void deptree_delete_model( deptree_t* d, diverge_model_t* mod ) {
    diverge_flow_step_t* step = deptree_model_locked(d, mod);
    if (step) {
        mpi_err_printf( "model %p locked by step %p <- free first\n", mod, step );
        return;
    }
    pthread_mutex_lock( &d->mtx );
    hashtable_delete( d->mod, (index_t)mod );
    pthread_mutex_unlock( &d->mtx );
}

void deptree_insert_flow( deptree_t* d, diverge_flow_step_t* step ) {
    pthread_mutex_lock( &d->mtx );
    hashtable_insert_safe( d->step, (index_t)step, step->mod );
    pthread_mutex_unlock( &d->mtx );
}

void deptree_delete_flow( deptree_t* d, diverge_flow_step_t* step ) {
    pthread_mutex_lock( &d->mtx );
    hashtable_delete( d->step, (index_t)step );
    pthread_mutex_unlock( &d->mtx );
}

static deptree_t deptree = { NULL, NULL, PTHREAD_MUTEX_INITIALIZER, };

void dependency_tree_init( void ) { deptree_init( &deptree ); }
void dependency_tree_free( void ) { deptree_free( &deptree ); }
void dependency_tree_insert_model( diverge_model_t* mod ) { deptree_insert_model( &deptree, mod ); }
void dependency_tree_remove_model( diverge_model_t* mod ) { deptree_delete_model( &deptree, mod ); }
void dependency_tree_insert_flow( diverge_flow_step_t* step ) { deptree_insert_flow( &deptree, step ); }
void dependency_tree_remove_flow( diverge_flow_step_t* step ) { deptree_delete_flow( &deptree, step ); }

