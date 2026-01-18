.. _Flowing Vertices:

Flowing Vertices
================

It can be useful to extract certain components of the three (:math:`P`,
:math:`C`, :math:`D`) diagrammatic channels over the course of the FRG flow in
order to analyze them in post-processing. For example, one may be interested to
calculate expectation values like

.. math::
   \langle \Delta^\Lambda \rangle = \Delta_{12} X^\Lambda_{1234} \Delta_{34}

for a certain bilinear :math:`\Delta` (for specific diagrammatic channels
:math:`X`) at each scale :math:`\Lambda` of the flow. Plotting these expectation
values can yield information on which fluctuation drives or suppresses a certain
instability.

Because storing the full vertices (even in TU approximation) is not feasible, we
here provide functionality to store the vertices of *some* momentum points, such
that features specific to a model can be tracked.

In practice, if one has a :c:type:`diverge_flow_step_t` object all set up, one
can do the following during the flow loop:

.. sourcecode:: C

   // these are a prerequisite
   // diverge_flow_step_t* step = ...;
   // index_t* qpts = ...;
   // index_t nqpts = ...;

   // initialize the tu_vertex_storage_t object for the 'm'agnetic channel
   tu_vertex_storage_t* mag = tu_vertex_storage_init( step, 'm', qpts, nqpts );
   do {
       // here we store (magnetic) vertices
       tu_vertex_storage_snap( mag );
       diverge_flow_step_euler( step, eu.Lambda, eu.dLambda );
       diverge_flow_step_vertmax( step, &vmax );
       diverge_flow_step_chanmax( step, cmax );
   } while (diverge_euler_next( &eu, vmax ));
   // and the last one
   tu_vertex_storage_snap( mag );

   // we can get a specific vertex (step #1, qpt #0)...
   complex128_t* mvert = tu_vertex_storage_get( mag, 1, 0 );
   // or save them to a file:
   tu_vertex_storage_to_file( mag, "magnetic_vertices.dvg" );

   // finally, we have to cleanup
   tu_vertex_storage_free( mag );

.. c:autodoc:: misc/vertex_store.h
