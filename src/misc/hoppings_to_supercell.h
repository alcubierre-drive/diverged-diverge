/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

typedef struct diverge_model_t diverge_model_t;

#ifdef __cplusplus
extern "C" {
#endif

// function: rs_hopping_to_supercell
// map model to a cubic supercell of extent (nr[0], nr[1], nr[2]).
void rs_hopping_to_supercell( diverge_model_t* m, int* nr );

// function: rs_hopping_to_fractcell
// map hopping parameters and model to supercell and allow matrix supercells
// wrt. to the old lattice vectors:
//
// S1 = m->lattice[0] * N1[0] + m->lattice[1] * N1[1] + m->lattice[2] * N1[2]
// S2 = m->lattice[0] * N2[0] + m->lattice[1] * N2[1] + m->lattice[2] * N2[2]
// ...
//
// modify model accordingly; operate on hopping parameters, lattice, orbitals,
// and positions; leave anything else as-is.
//
// In case N2 == N3 == NULL, make call to <rs_hopping_to_supercell> with N1 as
// nr.
void rs_hopping_to_fractcell( diverge_model_t* m, int* N1, int* N2, int* N3 );

#ifdef __cplusplus
}
#endif
