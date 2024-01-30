/* Copyright (C) 2024 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// function: rs_hopping_to_supercell
// map hopping parameters to a cubic supercell of extent (nr[0], nr[1], nr[2]).
// outputs number of new hopping parameters to *n_hop and *allocates* hopping
// parameters that are returned. the new number of orbitals is prod(nr) * n_orb.
rs_hopping_t* rs_hopping_to_supercell( const rs_hopping_t* h, index_t* n_hop,
                                       index_t no, const index_t nr[3] );

#ifdef __cplusplus
}
#endif
