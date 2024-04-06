Flow
====

divERGe does *not* include an automatic flow integrator. To achieve
quasi-automatic integration, users are expected to

- create a :c:type:`diverge_flow_step_t` object (see :ref:`Flow Step Object`)
- create a :c:struct:`diverge_euler_t` object (see :ref:`Flow Integration`)
- wrap this in a ``do {} while()`` loop (see :ref:`Flow Integration`) until a
  stopping criterion is reached.
- call postprocessing routines and save the output to disk (see
  :ref:`Postprocessing and Output`)

Those steps are summarized in this document (:ref:`Flow Step Object`) and the
liked pages below:

.. toctree::
   Flow Integration
   Postprocessing and Output

.. _Flow Step Object:

Flow Step Object
----------------

Given the heterogenous nature of the three backends that can be chosen in
divERGe, the flow step internals are *not* exposed to the user. The API is
therefore private by construction and everything is done over the
:c:type:`diverge_flow_step_t` handle.

.. c:autodoc:: diverge_flow_step.h

