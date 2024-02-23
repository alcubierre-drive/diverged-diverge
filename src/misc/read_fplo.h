#pragma once

#include "../diverge_common.h"
#include "../diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

rs_hopping_t* diverge_read_fplo( const char* fname, index_t* n_hop, index_t* n_orb );

#ifdef __cplusplus
}
#endif
