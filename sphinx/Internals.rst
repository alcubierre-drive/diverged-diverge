.. _Internals:

Internals
=========

Functions to generate and access internal structures. Kinetics are initalized
with :c:func:`diverge_model_internals_common`. Backend-specific interaction
internals with :c:func:`diverge_model_internals_tu`,
:c:func:`diverge_model_internals_patch`, :c:func:`diverge_model_internals_grid`,
or :c:func:`diverge_model_internals_any`.

Internals can be accessed through several getter functions that *may break when
using non-standard models* (custom Hamiltonians etc.):

* :c:func:`diverge_model_internals_get_E`
* :c:func:`diverge_model_internals_get_U`
* :c:func:`diverge_model_internals_get_H`
* :c:func:`diverge_model_internals_get_kmesh`
* :c:func:`diverge_model_internals_get_kfmesh`
* :c:func:`diverge_model_internals_get_greens`

.. c:autodoc:: diverge_model_internals.h
