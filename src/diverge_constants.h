/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t _MAX_N_ORBS;
    int64_t _USE_GF_FLOATS;
    int64_t _MAX_NAME_LENGTH;
    int64_t _MAX_N_SYM;
    int64_t _MAX_ORBS_PER_SITE;
    double  _DIVERGE_EPS_MESH;
} diverge_constants_t;

diverge_constants_t diverge_constants( void );
void diverge_constants_print( void );

#ifdef __cplusplus
}
#endif
