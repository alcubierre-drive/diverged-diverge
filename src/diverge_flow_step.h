#pragma once

#include "diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Typedef: diverge_flow_step_t
typedef struct diverge_flow_step_t diverge_flow_step_t;

// Function: diverge_flow_step_init
// internals of the model must be set up at the time of calling this function
// therefore, the model already holds all vertex data etc. returns NULL upon
// error.
// all channels present in the string (P,C,D) are selected, all other ones deselected.
diverge_flow_step_t* diverge_flow_step_init( diverge_model_t* mod, const char* method,
        const char* channels );

// Function: diverge_flow_step_vertmax
void diverge_flow_step_vertmax( diverge_flow_step_t* st, double vertmax[1] );
// Function: diverge_flow_step_loopmax
void diverge_flow_step_loopmax( diverge_flow_step_t* st, double loopmax[2] );
// Function: diverge_flow_step_chanmax
void diverge_flow_step_chanmax( diverge_flow_step_t* st, double chanmax[3] );

// Function: diverge_flow_step_euler
// Do an euler step starting at Lambda ending at Lambda+dLambda
void diverge_flow_step_euler( diverge_flow_step_t* st, double Lambda, double dLambda );

// Function: diverge_flow_step_niter
// return the number of iterations carried out by the <diverge_flow_step_t>
index_t diverge_flow_step_niter( diverge_flow_step_t* st );
// Function: diverge_flow_step_ntimings
// return the length of the timing list attached to <diverge_flow_step_t>
index_t diverge_flow_step_ntimings( diverge_flow_step_t* st );
// Function: diverge_flow_step_timing
// return the timing list element at idx attached to <diverge_flow_step_t>
double diverge_flow_step_timing( diverge_flow_step_t* st, index_t idx );
// Function: diverge_flow_step_timing_descr
// return the timing list element name at idx attached to <diverge_flow_step_t>
const char* diverge_flow_step_timing_descr( diverge_flow_step_t* st, index_t idx );

// Function: diverge_flow_step_timing_vec
// returns all timings as static vector (that must not be freed), maximal size
// 128. last element [128] is zerod.
double* diverge_flow_step_timing_vec( diverge_flow_step_t* st );
// Function: diverge_flow_step_timing_descr_vec
// returns all timing descriptions as static vector (that must not be freed),
// maximal size 128. last element [128] is zerod.
char** diverge_flow_step_timing_descr_vec( diverge_flow_step_t* st );

// Function: diverge_flow_step_lambda
// get the current Lambda
double diverge_flow_step_lambda( diverge_flow_step_t* st );
// Function: diverge_flow_step_dlambda
// get the current dLambda
double diverge_flow_step_dlambda( diverge_flow_step_t* st );

// TODO(@jh): how do we need to implement the routines to capture both
// Bogaki-Shampine and Adaptive Euler?

// Function: diverge_flow_step_free
void diverge_flow_step_free( diverge_flow_step_t* st );


// Struct: diverge_flow_step_vertex_t
typedef struct {
    // Var: ary
    // pointer to the raw memory behind the vertex
    complex128_t* ary;
    // Var: q_0, q_1
    // start and stop of the 'q' dimension ('k1' in the case of grid-FRG and
    // chan == 'V')
    index_t q_0, q_1;
    // Var: nk
    // number of k points
    index_t nk;
    // Var: n_orbff
    // number of orbitals*formfactors, returns number of orbitals for patch and
    // grid
    index_t n_orbff;
    // Var: n_spin
    // number of spins (==1 if SU2)
    index_t n_spin;
    // Var: backend
    // holds 'T'U, 'G'rid, 'P'atch, or 'I'nvalid
    char backend;
    // Var: channel
    // holds the channel.
    char channel;
} diverge_flow_step_vertex_t;


// Function: diverge_flow_step_vertex
// returns pointer to vertex object. behaves differently for the various
// backends:
//
// TUFRG - chan can be 'P', 'C', or 'D', both upper and lower case. For TUFRG,
//         the case is ignored. The actual channel specific vertex object is
//         returned, ordered as (q, s4,s3,ob2, s2,s1,ob1). Note that q is
//         symmetry reduced as well as MPI distributed.
//         Additionally, chan can be 'S' to yield the self-energy on the fine
//         mesh (if used). Then, n_orbff contains the number of orbitals and nk
//         the number of total (refined * coarse) kpts.
// grid FRG - chan can be 'P', 'C', 'D', or 'V', both upper and lower case. In
//            general, lower case references the differential vertex and upper
//            case the full vertex. The three channels are returned in
//            (q, k,O1,O2, k',O3,O4) order. 'V' stands for
//            (k1,k2,k3, O1,O2,O3,O4) ordering. In case of multiple MPI ranks,
//            only the locally accessible part of the vertex is referenced in
//            the returned array. For grid, the indices Oi are actually (oi, si).
// patch FRG - chan may be 'V' (full vertex) or 'v' (differential vertex). The
//             vertex in (p1,p2,p3, O1,O2,O3,O4) order is returned. As in the
//             grid case, the indices Oi are actually (oi, si).
diverge_flow_step_vertex_t diverge_flow_step_vertex( diverge_flow_step_t* st, char chan );

// Function: diverge_flow_step_refill
// adjusts the filling value of the non-interacting Hamiltonian such that *with*
// self-energy, it is set to nu. Works only if the standard Hamiltonian- &
// Greensfunction-generators are used as it operates on the internal
// energy/hamiltonian/orbital2band arrays.
//
// if workspace != NULL, use this space as scratch memory. size must be at least
// sizeof(complex128_t) * (nktot, nb, nb) + sizeof(double) * (nktot, nb)
void diverge_flow_step_refill( diverge_flow_step_t* s, double nu, void* workspace );

// Function: diverge_flow_step_refill_Hself generates the Hamiltonian +
// self-energy and diagonalizes it in the workspace buffer (same size as above,
// energies after matrices). Finds the filling and returns the chemical
// potential needed to obtain this filling value
double diverge_flow_step_refill_Hself( diverge_flow_step_t* s, double nu, void* workspace );

#ifdef __cplusplus
}
#endif
