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

Analyzing the postprocessing data that comes out of the flow can turn out to be
a bit more involved. Depending on the backend, some of the following
"observables" are returned:

    * susceptibilities in each of the interaction channels
    * solutions to linearized gap equations (at :math:`\boldsymbol q = 0`) in
      each of the interaction channels
    * actual vertices, either per channel or full ones
    * eigenvalues/eigenvectors or singular values/vectors of vertices in
      each of the interaction channels

It can turn out to be convenient to plot :math:`\boldsymbol q` dependent
quantities (such as susceptibilities or channel eigenvalues) on irreducible
paths instead of scatter/color plots. To generate such graphical
representations, either use the function :c:func:`diverge_kmesh_to_bands_crs`
with an existing :c:struct:`diverge_model_t` instance, or include this data in
your model output file through the key
:c:member:`diverge_model_output_conf_t.kc_ibz_path` (using the
:c:func:`finegrained inteface <diverge_model_to_file_finegrained>` or the
:func:`python interface <diverge.model_to_file_PY>`). Then, the *model* output
file will have coarse mesh indices corresponding to the IBZ path saved in
:attr:`kc_ibz_path <diverge.output.diverge_model.kc_ibz_path>`. With, say, a
TUFRG postprocessing saved to ``post.dvg``, one could then do

.. sourcecode:: python

   path_segment_len, path_indices, _ = model.kc_ibz_path
   ibz_map = post.kmaps_to
   ibz_indices = ibz_map[path_indices]

   suscep = post.Csusc.sum( axis=(0,1,2,3) ).real
   suscep_path = suscep[ibz_indices]

   plt.plot( suscep_path, c='black',
             label=r'$\chi_\mathrm{mag}(\boldsymbol q)$' )
   plt.show()

Notice how we must take care of the fact that the TUFRG output is saved on the
*irreducible BZ*, not the *full BZ*; those differ as soon as the model includes
symmetries. We therefore have to find the IBZ indices as a function of BZ
indices via :attr:`kmaps_to <diverge.output.diverge_post_tu.kmaps_to>` and take
those *IBZ* indices that lie on the path indices (that refer to the *full BZ*).

.. toctree::
   :maxdepth: 1

   Simulation Output Module Spec
