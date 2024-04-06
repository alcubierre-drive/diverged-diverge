.. _Flow Integration:

Flow Integration
================

The FRG flow is integrated by successive calls to
:c:func:`diverge_flow_step_euler`. Before carrying out this integration, users
must allocate a :c:type:`diverge_flow_step_t` object using
:c:func:`diverge_flow_step_init`. We explicitly leave the construction of the
(Euler) integration loop to the users, in order to adjust step sizes etc. by
hand.

.. c:autodoc:: diverge_euler.h
