/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t digest[16];
} uint128_t;

uint128_t md5sum_div_stack( const char* buf, size_t size );
char* md5sum_div( const char* buf, size_t size );

#ifdef __cplusplus
}
#endif
