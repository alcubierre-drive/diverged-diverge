.. _Python Interface:

Python Interface
================

Wrapper for the :ref:`C Interface` ``diverge.h``, generated with some help of
ctypesgen. It is heavily based on wrapping C arrays as numpy structured arrays.
Therefore, some documentation for these (and how to use them from the
``diverge`` python module) is included in the section :ref:`Structured Arrays`.

The documentation only includes those functions that exist *on top* of the
:ref:`C Interface`. Everything else is documented *there*. For usage examples,
see the python examples shipped with :release_version:`diverge.tar.gz
<divERGe.tar.gz>` or found in the :release_tree:`examples/ <examples>`
directory of the repo . They generically work similar to the following skeleton:

.. sourcecode:: python

   import diverge
   diverge.init()

   model = diverge.model_init()
   model.contents.MEMBER = VALUE
   # and so on

   diverge.model_internals_common( model )
   diverge.model_internals_tu( model, 2.01 )

   step = diverge.flow_step_init_any( model, "PCD" )
   # some flow loop and do whatever you like with step/model

   diverge.flow_step_free( step )
   diverge.model_free( model )

   diverge.finalize()

Notice how the ``diverge_`` prefix disappeared into the module name. This holds
for all functions that hold this prefix. Others *keep* their name.

.. note:: Installation and Extension

    On GNU/Linux, the python library will search for ``libdivERGe.so`` in your
    ``LD_LIBRARY_PATH`` *with precedence* over the shipped version (located
    somewhere in your ``PYTHONPATH``). This makes it easy to swap out the backend by
    pointing ``LD_LIBRARY_PATH`` to a location with the desired ``libdivERGe.so``:

    .. sourcecode:: bash

        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/gpu/version/ python script.py

.. toctree::
   :maxdepth: 1

   Autoflow
   MPI Output
   Structured Arrays
   Full Python Module Specification

