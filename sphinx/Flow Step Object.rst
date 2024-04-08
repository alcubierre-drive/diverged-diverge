.. _Flow Step Object:

Flow Step Object
================

Given the heterogenous nature of the three backends that can be chosen in
divERGe, the flow step internals are *not* exposed to the user. The API is
therefore private by construction and everything is done over the
:c:type:`diverge_flow_step_t` handle.

.. c:autodoc:: diverge_flow_step.h

