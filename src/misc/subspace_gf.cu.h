/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "subspace_gf.h"

#ifdef __cplusplus
extern "C" {
#endif

subspace_gfill_multi_t* subspace_gfill_multi_init_cu( double* E, complex128_t* U,
        index_t nk, const index_t* orbs, index_t n_orbs, index_t nb, index_t b_per_bb );
void subspace_gfill_multi_free_cu( subspace_gfill_multi_t* m );
void subspace_gfill_multi_exec_cu( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda, double mu_add );

#ifdef __cplusplus
}
#endif
