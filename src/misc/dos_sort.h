/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void dos_sort_doubles( double* start, index_t size );
void dos_sort_doubles_indirect( index_t* start, index_t size, const double* vals );

#ifdef __cplusplus
}
#endif

