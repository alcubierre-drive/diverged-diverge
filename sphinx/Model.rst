.. _Model:

Model
=====

The model structure :c:struct:`diverge_model_t` represents the center of the
divERGe library. Users typically allocate a model with
:c:func:`diverge_model_init`. There are some required parameters that must be
set, and thereafter one can generate common internal structures via
:c:func:`diverge_model_internals_common` (see :ref:`Internals`). Finally, the
model **must** be free'd using the :c:func:`diverge_model_free` function to
appropriately release all allocated memory.

.. c:autodoc:: diverge_model.h

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Internals
   Meshing
   Wannier90
   FPLO
   Patching
   Symmetries
   Symmetrization
   Filling
   Berry Curvature
   Supercells
   Output
   Hacking
   Advanced Features
