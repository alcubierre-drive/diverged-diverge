.. _Advanced Features:

Advanced Features
=================

Green's functions for systems with many orbitals
------------------------------------------------

.. c:autodoc:: misc/largemat_gf.h

Green's functions for subspace systems
--------------------------------------

When only a certain orbital subspace is considered as "correlated", a custom
Green's function generator can be defined to only calculate the Green's function
for those orbitals. We provide a GPU (and CPU) subspace GF generator that is
most conveniently used when both the full system and the subspace system are
separate :c:type:`diverge_model_t` instances.

Due to the GPU capability, the functions provided here must operate on a handle
(:c:type:`subspace_gfill_multi_t`). The actual Green's function calculation is
carried out using :c:func:`subspace_gfill_multi_exec`; the documentation
includes an example of how to use it as a :c:type:`greensfunc_generator_t`.

.. c:autodoc:: misc/subspace_gf.h

Unique Distances
----------------

Sometimes it may be useful to generate all uniquely defined distances in a
divERGe model, i.e., in order to set on-site, nearest-neighbor,
next-nearest-neighbor, etc. interactions. We provide a function to do so in the
unstable interface, which can be included with ``#include
<misc/unique_distances.h>``. This header comprises two functions with the
following signatures:

.. c:autodoc:: misc/unique_distances.h

Hopping Parameters from Hamiltonians
------------------------------------

The inverse Fourier transform to get hopping parameters from Hamiltonian arrays
is implemented in the following function:

.. c:autodoc:: misc/ham2hop.h

Hamiltonians at Specific Momentum Points
----------------------------------------

To obtain a tight-binding Hamiltonian (from hopping parameters
:c:member:`diverge_model_t.hop`) at a specific point in momentum space *without*
setting it up in the entire BZ (as done in :c:member:`diverge_model_t.hfill`),
one can use the function :c:func:`diverge_model_ham_at_kpt`. In addition, we
provide wrappers for the coarse and fine mesh indices---these wrappers rely on
the model internals structure to be set up (via
:c:func:`diverge_model_internals_common`).

.. c:autodoc:: misc/ham_at_k.h
