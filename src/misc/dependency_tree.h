/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void dependency_tree_init( void );
void dependency_tree_insert_model( void* ptr );
void dependency_tree_remove_model( void* ptr );
void dependency_tree_insert_flow( void* ptr );
void dependency_tree_remove_flow( void* ptr );
int dependency_tree_is_model_frozen( void* ptr );
void dependency_tree_free( void );

#ifdef __cplusplus
}
#endif
