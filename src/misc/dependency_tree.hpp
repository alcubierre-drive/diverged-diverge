/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include <set>

typedef struct diverge_model_t diverge_model_t;
typedef struct diverge_flow_step_t diverge_flow_step_t;

typedef struct {
    diverge_flow_step_t* s;
    diverge_model_t* m;
} locked_flow_model_ptr_t;

inline bool operator<( locked_flow_model_ptr_t A, locked_flow_model_ptr_t B ) {
    if (A.s == B.s)
        return A.m < B.m;
    else
        return A.s < B.s;
}

typedef std::set<diverge_model_t*> modptr_set_t;
typedef std::set<locked_flow_model_ptr_t> flowptr_set_t;

struct dependency_tree {
    modptr_set_t models;
    flowptr_set_t flows;

    bool is_frozen( diverge_model_t* m );

    void insert( diverge_model_t* m );
    void insert( diverge_flow_step_t* s );

    void remove( diverge_model_t* m );
    void remove( diverge_flow_step_t* s );

    ~dependency_tree();
};
