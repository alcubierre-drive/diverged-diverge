/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    diverge_flow_type_invalid = -1,
    diverge_patch,
    diverge_grid,
    diverge_tu,
    diverge_tu_phonons,
    DIVERGE_FLOW_TYPE_COUNT
} diverge_flow_type;

diverge_flow_type diverge_flow_type_fromstr( const char* mode );

// DEVELOPERS
//
// For a new backend, you must add a new flow_type in the above enum and modify
// the function diverge_flow_type_fromstr.

#ifdef __cplusplus
}
#endif
