.. _Flowing Vertices:

Flowing Vertices
================

.. note::
   This feature is in a very early stage, so please consider it 'alpha' at best.

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
can do the following inside/before the flow loop:

.. sourcecode:: C

   // these are a prerequisite
   // diverge_flow_step_t* step = ...;
   // index_t* qpts = ...;
   // index_t nqpts = ...;

   // initialize the tu_highsym_vertex_t object
   tu_highsym_vertex_t* qverts = tu_highsym_vertex_store( step, qpts,
       nqpts, NULL );
   do {
       diverge_flow_step_euler( step, eu.Lambda, eu.dLambda );
       diverge_flow_step_vertmax( step, &vmax );
       diverge_flow_step_chanmax( step, cmax );
       // here we store vertices
       tu_highsym_vertex_store( step, qpts, nqpts, qverts );
   } while (diverge_euler_next( &eu, vmax ));

   // now save those vertices to file
   tu_highsym_vertex_to_file( qverts, "highsym_vertices.qvx" );

   // and do cleanup
   tu_highsym_vertex_free( qverts );

.. c:autodoc:: misc/vertex_store.h
