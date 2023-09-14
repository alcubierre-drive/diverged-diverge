/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

void diverge_model_hack( diverge_model_t* m, const char* key, const char* val );
void diverge_model_print_hacks( void );

#ifdef __cplusplus
}
#endif
