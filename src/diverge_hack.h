/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

/**
 * enable unsupported features. depending on the key, different stages of
 * initialization of the model are required.
 *
 * *usage is discouraged but this function exists in order to have an easy way
 * to switch to non-default options without recompiling the library. Have a
 * look at the (cpp) source file to see what the available hacks actually do.*
 */
void diverge_model_hack( diverge_model_t* m, const char* key, const char* val );

/** print which keys are available */
void diverge_model_print_hacks( void );

#ifdef __cplusplus
}
#endif
