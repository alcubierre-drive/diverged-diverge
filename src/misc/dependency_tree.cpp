/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "dependency_tree.h"
#include "dependency_tree.hpp"
#include "../diverge_flow_step_internal.hpp"

#include <mutex>

std::mutex mtx;

void dependency_tree::insert( diverge_model_t* m ) {
    mtx.lock();
    models.insert( m );
    mtx.unlock();
}

void dependency_tree::insert( diverge_flow_step_t* s ) {
    mtx.lock();
    flows.insert( { s, s->mod } );
    mtx.unlock();
}

bool dependency_tree::is_frozen( diverge_model_t* m ) {
    mtx.lock();
    int counts = 0;
    for (locked_flow_model_ptr_t sm : flows) {
        counts += (sm.m == m);
    }
    mtx.unlock();
    return counts;
}

void dependency_tree::remove( diverge_model_t* m ) {
    if (this->is_frozen(m)) {
        mpi_err_printf( "model %x locked by flow step. free flow step first.\n", m );
        return;
    }
    mtx.lock();
    auto it = models.find( m );
    if (it != models.end()) {
        models.erase( it );
    } else {
        mpi_err_printf( "couldnt find model %x in dependency tree\n", m );
    }
    mtx.unlock();
}

void dependency_tree::remove( diverge_flow_step_t* s ) {
    mtx.lock();
    locked_flow_model_ptr_t sm = {s, s->mod};
    auto it = flows.find( sm );
    if (it != flows.end()) {
        flows.erase( it );
    } else {
        mpi_err_printf( "couldn't find flow %x in dependency tree\n", s );
    }
    mtx.unlock();
}

dependency_tree::~dependency_tree() {
    if (flows.size() > 0)
        mpi_wrn_printf( "%lu stale diverge_flow_step_t allocations\n", flows.size() );
    if (models.size() > 0)
        mpi_wrn_printf( "%lu stale diverge_model_t allocations\n", models.size() );
}

static dependency_tree* h_tree = NULL;
void dependency_tree_init( void ) {
    mtx.lock();
    if (h_tree)
        mpi_err_printf( "h_tree != NULL\n" );
    h_tree = new dependency_tree();
    mtx.unlock();
}

void dependency_tree_insert_model( void* ptr ) {
    h_tree->insert( (diverge_model_t*)ptr );
}
void dependency_tree_remove_model( void* ptr ) {
    h_tree->remove( (diverge_model_t*)ptr );
}
void dependency_tree_insert_flow( void* ptr ) {
    h_tree->insert( (diverge_flow_step_t*)ptr );
}
void dependency_tree_remove_flow( void* ptr ) {
    h_tree->remove( (diverge_flow_step_t*)ptr );
}
int dependency_tree_is_model_frozen( void* ptr ) {
    return h_tree->is_frozen( (diverge_model_t*)ptr );
}

void dependency_tree_free( void ) {
    mtx.lock();
    delete h_tree;
    h_tree = NULL;
    mtx.unlock();
}
