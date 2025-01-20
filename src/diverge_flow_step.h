/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_model.h"

// sort of a private include
#include "diverge_flow_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * opaque typedef because none of the internal structure is needed by users.
 * Pointers to this structure must be obtained through
 * :c:func:`diverge_flow_step_init` (or :c:func:`diverge_flow_step_init_any`)
 * and free'd with :c:func:`diverge_flow_step_free`.
 */
typedef struct diverge_flow_step_t diverge_flow_step_t;

/** flow step timings given by :c:func:`diverge_flow_step_timings` */
typedef struct {
    /** time in seconds */
    double t;
    /** name given for this timing */
    char nam[56];
} diverge_timing_t;

/**
 * internals of the ``model`` must be set up at the time of calling this
 * function. Therefore, the model already holds all vertex data etc. returns
 * NULL upon error.
 *
 * :param diverge_model_t model: model to base the flow step on
 * :param const char* method: string that can be "patch", "tu", or "grid";
 *                            selecting one of the backends.
 * :param const char* channels: all channels present in the string (P,C,D) are
 *                              selected, all other ones deselected.
 *
 * :returns: an pointer to a :c:type:`diverge_flow_step_t` object.
 */
diverge_flow_step_t* diverge_flow_step_init( diverge_model_t* mod, const char* method,
        const char* channels );

// undocumented functions, they are needed for some C interfacing, but not
// explicitly exposed to users.
diverge_flow_type diverge_flow_step_type( const diverge_flow_step_t* st );
void diverge_flow_step_get_channels( const diverge_flow_step_t* st, int chans[4] );

/**
 * Allows for a selective turning off of the feedback in between the channels,
 * before or at any point during the flow. The format of the Feedback char is
 * "C|D|CP" which means that in the P channel only the C channel, in the C
 * channel only the D channel and in the D channel both of the other channels
 * are included. The separators "|" are reqired!
 */
void diverge_flow_step_set_interchannel( diverge_flow_step_t* st, const char* feedback );

/**
 * initialize the flow step instance from a model instance that has its
 * internals set to one method. Everything else works as in
 * :c:func:`diverge_flow_step_init`.
 */
diverge_flow_step_t* diverge_flow_step_init_any( diverge_model_t* mod, const char* channels );

/** outputs the maximum vertex element in ``vertmax`` */
void diverge_flow_step_vertmax( diverge_flow_step_t* st, double vertmax[1] );
/** outputs the maximum loop elements in ``loopmax`` */
void diverge_flow_step_loopmax( diverge_flow_step_t* st, double loopmax[2] );
/** outputs the maximum channel vertex elements in ``chanmax`` */
void diverge_flow_step_chanmax( diverge_flow_step_t* st, double chanmax[3] );

/** Do an euler step starting at :math:`\Lambda` ending at :math:`\Lambda+d_\Lambda` */
void diverge_flow_step_euler( diverge_flow_step_t* st, double Lambda, double dLambda );

/** return the number of iterations carried out by the :c:type:`diverge_flow_step_t` */
index_t diverge_flow_step_niter( diverge_flow_step_t* st );

/** return the length of the timing list attached to :c:type:`diverge_flow_step_t` */
index_t diverge_flow_step_ntimings( diverge_flow_step_t* st );
/**
 * return timing vector (type :c:struct:`diverge_timing_t`, length given by
 * :c:func:`diverge_flow_step_ntimings`) taken during the flow
 */
diverge_timing_t* diverge_flow_step_timings( diverge_flow_step_t* st );

/**
 * .. deprecated:: 0.9
 * use :c:func:`diverge_flow_step_timings` instead.
 *
 * return the timing list element at ``idx`` attached to
 * :c:type:`diverge_flow_step_t`
 */
double diverge_flow_step_timing( diverge_flow_step_t* st, index_t idx );
/**
 * .. deprecated:: 0.9
 * use :c:func:`diverge_flow_step_timings` instead.
 *
 * the timing list element name at ``idx`` attached to
 * :c:type:`diverge_flow_step_t`
 */
const char* diverge_flow_step_timing_descr( diverge_flow_step_t* st, index_t idx );
/**
 * .. deprecated:: 0.9
 * use :c:func:`diverge_flow_step_timings` instead.
 *
 * returns all timings as static vector (that must not be freed), maximal size
 * 128. last element [128] is zerod.
 *
 * cf. :c:func:`diverge_flow_step_timing`
 */
double* diverge_flow_step_timing_vec( diverge_flow_step_t* st );
/**
 * .. deprecated:: 0.9
 * use :c:func:`diverge_flow_step_timings` instead.
 *
 * returns all timing descriptions as static vector (that must not be freed),
 * maximal size 128. last element [128] is zerod.
 *
 * cf. :c:func:`diverge_flow_step_timing_descr`
 */
char** diverge_flow_step_timing_descr_vec( diverge_flow_step_t* st );

/** get the current :math:`\Lambda` */
double diverge_flow_step_lambda( diverge_flow_step_t* st );
/** get the current :math:`d_\Lambda` */
double diverge_flow_step_dlambda( diverge_flow_step_t* st );

/** free the resources attached to a :c:func:`diverge_flow_step_t`. */
void diverge_flow_step_free( diverge_flow_step_t* st );

/**
 * Vertices
 * ========
 *
 * For maximum flexibility, we expose the memory of the vertex objects that are
 * allocated for the FRG flow to users.
 */

/**
 * Structure that can hold all vertex objects used in divERGe. depending on the
 * backend and channel, the content differs.
 */
typedef struct {
    /** pointer to the raw memory behind the vertex */
    complex128_t* ary;
    /** start of the 'q' dimension ('k1' in the case of grid-FRG and chan == 'V') */
    index_t q_0;
    /** stop of the 'q' dimension ('k1' in the case of grid-FRG and chan == 'V') */
    index_t q_1;
    /** number of k points, not used for TUFRG vertices */
    index_t nk;
    /** number of orbitals & formfactors, returns number of orbitals for patch and grid */
    index_t n_orbff;
    /** number of spins (==1 if SU2) */
    index_t n_spin;
    /** holds 'T'U, 'G'rid, 'P'atch, or 'I'nvalid */
    char backend;
    /** holds the channel. */
    char channel;
} diverge_flow_step_vertex_t;


/**
 * returns a vertex object defined by :c:struct:`diverge_flow_step_vertex_t`.
 * behaves differently for the various backends:
 *
 * :TUFRG: chan can be 'P', 'C', or 'D', both upper and lower case. For TUFRG,
 *         the case determines whether the vertex is just the channel (upper
 *         case), the full vertex projected to the respective channel (lower
 *         case), or a physical channel (su'p'erconducting, 'm'agnetic,
 *         c'h'arge). In case phonons are included, the phonon contribution to
 *         the channels can be activated by using ``-chan``. The actual channel
 *         specific vertex object is returned, ordered as (q, s4,s3,ob2,
 *         s2,s1,ob1). Note that q is symmetry reduced as well as MPI
 *         distributed. Additionally, chan can be 'S' to yield the self-energy
 *         on the fine mesh (if used). Then, n_orbff contains the number of
 *         orbitals and nk the number of total (refined * coarse) kpts.
 * :grid FRG: chan can be 'P', 'C', 'D', or 'V', both upper and lower case. In
 *            general, lower case references the differential vertex and upper
 *            case the full vertex. The three channels are returned in
 *            (q, k,O1,O2, k',O3,O4) order. 'V' stands for
 *            (k1,k2,k3, O1,O2,O3,O4) ordering. In case of multiple MPI ranks,
 *            only the locally accessible part of the vertex is referenced in
 *            the returned array. For grid, the indices Oi are actually (oi, si).
 * :patch FRG: chan may be 'V' (full vertex) or 'v' (differential vertex). The
 *             vertex in (p1,p2,p3, O1,O2,O3,O4) order is returned. As in the
 *             grid case, the indices Oi are actually (oi, si).
 */
diverge_flow_step_vertex_t diverge_flow_step_vertex( diverge_flow_step_t* st, char chan );

/**
 * Filling
 * =======
 *
 * For quasi-canonical calculations, where the filling :math:`\nu` is forced to
 * be fixed instead of the chemical potential :math:`\mu`, we provide functions
 * that adjust the model's kinetics depending on the self-energy.
 */

/**
 * adjusts the filling value of the non-interacting Hamiltonian such that *with*
 * self-energy, it is set to :math:`\nu`. Works only if the standard
 * Hamiltonian- & Greensfunction-generators are used as it operates on the
 * internal energy/hamiltonian/orbital2band arrays.
 *
 * if ``workspace != NULL``, use this space as scratch memory. size must be at
 * least ``sizeof(complex128_t) * (nktot * nb * nb) + sizeof(double) * (nktot *
 * nb)``. On return (if ``workspace != NULL``) ``workspace`` contains the
 * Hamiltonian matrices as :c:type:`complex128_t` array (nktot, nb, nb) followed
 * by the dispersion vectors as ``double`` array (nktot, nb), i.e.:
 *
 * .. sourcecode:: C
 *
 *  complex128_t* Utmp = (complex128_t*)workspace;
 *  double* Etmp = (double*)( Utmp + nk*nb*nb );
 */
void diverge_flow_step_refill( diverge_flow_step_t* s, double nu, void* workspace );

/**
 * generates the Hamiltonian + self-energy and diagonalizes it in the workspace
 * buffer (same size as :c:func:`diverge_flow_step_refill`, energies after
 * matrices). Finds the filling and returns the chemical potential needed to
 * obtain this filling value
 */
double diverge_flow_step_refill_Hself( diverge_flow_step_t* s, double nu, void* workspace );

/**
 * generates the Hamiltonian + self-energy and diagonalizes it in the workspace
 * buffer (same size as :c:func:`diverge_flow_step_refill`, energies after
 * matrices). Calculates the current filling
 */
double diverge_flow_step_get_filling_Hself( diverge_flow_step_t* s, void* workspace );

#ifdef __cplusplus
}
#endif
