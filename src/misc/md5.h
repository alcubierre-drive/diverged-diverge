/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

char* md5sum_div( const char* buf, size_t size );

#ifdef __cplusplus
}
#endif
