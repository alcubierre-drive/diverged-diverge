/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

// Function: diverge_model_hack
// enable unsupported features. depending on the key, different stages of
// initialization of the model are required.
void diverge_model_hack( diverge_model_t* m, const char* key, const char* val );

// Function: diverge_model_print_hacks
// print which keys are available
void diverge_model_print_hacks( void );

#ifdef __cplusplus
}
#endif
