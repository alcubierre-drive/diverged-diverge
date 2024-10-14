.. _Flow:

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

Those steps are summarized in the liked pages below:

.. toctree::
   Flow Step Object
   Flowing Vertices
   Flow Integration
   Postprocessing and Output

