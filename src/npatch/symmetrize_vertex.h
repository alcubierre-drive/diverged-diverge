/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

// symmetrizes buffer buf, which must be a buffer of length
// (np, np, np, nb, nb, nb, nb). Here, the first three indices are mapped to
// coarse momentum indices using the map, i.e., k = map[p] with p=0..np-1
// aux is an auxiliary buffer that may be either NULL or of size
// sizeof(complex128)*POW3(np)*POW4(nb) + sizeof(index_t)*(kdim(model->nk)+1)
void symmetrize_npatch_vertex( diverge_model_t* model, complex128_t* buf, const index_t* map, index_t np, complex128_t* aux );

#ifdef __cplusplus
}
#endif
