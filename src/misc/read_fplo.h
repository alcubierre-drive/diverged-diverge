/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

// function: diverge_read_fplo
// read FPLO Hamiltonian files (+hamdata) into <diverge_model_t> objects.
//
// Behavior:
// - initializes the <diverge_model_t> via <diverge_model_init>
// - sets the hopping parameters (<diverge_model_t.hop> and <diverge_model_t.n_hop>)
// - sets the number of orbitals (<diverge_model_t.n_orb>)
// - sets the number of spins (<diverge_model_t.n_spin>)
// - sets the real space lattice (<diverge_model_t.lattice>) and positions (<diverge_model_t.positions>)
//
// *DOES NOT SET ANY OTHER VARIABLES IN THE MODEL*
diverge_model_t* diverge_read_fplo( const char* fname );

#ifdef __cplusplus
}
#endif
