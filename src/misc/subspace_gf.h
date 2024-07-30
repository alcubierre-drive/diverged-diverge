#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct subspace_gfill_multi_t subspace_gfill_multi_t;

subspace_gfill_multi_t* subspace_gfill_multi_init( double* E, complex128_t* U,
        index_t nk, const index_t* orbs, index_t n_orbs, index_t nb, index_t b_per_bb );
void subspace_gfill_multi_free( subspace_gfill_multi_t* m );
void subspace_gfill_multi_exec( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda );

#ifdef __cplusplus
}
#endif
