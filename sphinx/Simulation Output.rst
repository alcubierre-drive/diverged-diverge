.. _Simulation Output:

Simulation Output
=================

When you've run an FRG simulation, you will usually have a model output file
(from a call to, e.g., :c:func:`diverge_model_to_file`) ``model.dvg`` and a
postprocessing output file (from a call to, e.g.,
:c:func:`diverge_postprocess_and_write`) ``post.dvg``. The python library
``diverge.output`` is the suggested way to read these files, though file format
specifications are included in the documentation, so you can build your own
tools. To read both the model and the postprocessing file to memory, simply use
the :func:`diverge.output.read` function:

.. sourcecode:: python

   import diverge.output as do
   model = do.read( 'model.dvg' )
   post = do.read( 'post.dvg' )

Depending on the contents of the ``.dvg`` file, different python classes may be
returned. For models, :func:`diverge.output.read` returns an instance of
:class:`diverge.output.diverge_model`; for postprocessing files, an instance of
either :class:`diverge.output.diverge_post_tu`,
:class:`diverge.output.diverge_post_grid`, or
:class:`diverge.output.diverge_post_patch` is returned, depending on the backend
chosen. We chose to provide three different classes here due to the heterogenous
nature of simulation output depending on backends.

The model output class :class:`diverge.output.diverge_model` contains all the
information held by the :c:struct:`diverge_model_t` structure. Depending on the
output options requested at the time of saving the ``model.dvg`` file (see
:c:func:`diverge_model_to_file_finegrained`,
:c:struct:`diverge_model_output_conf_t`, and :func:`diverge.model_to_file_PY`),
some other information like the band structure, momentum meshes, eigenvalues on
the mesh, eigenvectors on the mesh, etc. are accessible.

Bandstructures in particular receive some more support in the output library, as
they are frequently plotted. We provide the convenience functions

    * :func:`diverge.output.bandstructure_bands`
    * :func:`diverge.output.bandstructure_kpts`
    * :func:`diverge.output.bandstructure_ticks`
    * :func:`diverge.output.bandstructure_xvals`

that all take an instance of :class:`diverge.output.diverge_model` as argument,
i.e., the object you receive from a call to :func:`diverge.output.read` on a
``model.dvg`` file. With the returned arrays, plotting bands through
``matplotlib`` works like a breeze:

.. sourcecode:: python

   import matplotlib.pyplot as plt
   xvals = do.bandstructure_xvals(model)
   bands = do.bandstructure_bands(model)
   plt.plot( xvals, bands, c='black' )
   plt.show()

TODO(LK) examples for postprocessing output with TU/patch?

.. toctree::
   :maxdepth: 1

   Simulation Output Module Spec
