Meshing
=======

divERGe strictly operates on regular momentum meshes along the reciprocal
lattice vectors. In the :ref:`Model`, a coarse and a fine momentum mesh can be
set (:c:member:`diverge_model_t.nk` and :c:member:`diverge_model_t.nkf`). To
access the generated meshes, use the functions provided in the :ref:`Internals`
section. To generate meshes or reciprocal lattice vectors, use the functions
given here.

.. c:autodoc:: diverge_momentum_gen.h
.. c:autodoc:: misc/kmesh_to_bands.h
