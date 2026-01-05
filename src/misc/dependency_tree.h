/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;
typedef struct diverge_flow_step_t diverge_flow_step_t;

void dependency_tree_init( void );
void dependency_tree_free( void );
void dependency_tree_insert_model( diverge_model_t* mod );
void dependency_tree_remove_model( diverge_model_t* mod );
void dependency_tree_insert_flow( diverge_flow_step_t* step );
void dependency_tree_remove_flow( diverge_flow_step_t* step );

#ifdef __cplusplus
}
#endif

