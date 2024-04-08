.. _Tutorial:

Tutorial
========

This tutorial is given in a similar fashion in the example
:release_tree:`hubbard.c <examples/c_cpp/hubbard.c>`, including parsing of
command line arguments etc.

To gain access to all diverge functions, include the ``diverge.h`` header:

.. sourcecode:: C

    #include <diverge.h>

We refer the reader to the ``Makefile`` and ``Makefile.local`` in the
:release_tree:`examples/ <examples>` directory of :release_version:`divERGe.tar.gz
<divERGe.tar.gz>` to understand how C/C++ code using ``libdivERGe.so`` is
compiled and linked.

To initialize the library, you must either call :c:func:`diverge_init` or
:c:func:`diverge_embed`.

.. sourcecode:: C

    diverge_init( NULL, NULL );

Thereafter, you may continue with the central building block of divERGe, i.e.,
the :c:struct:`diverge_model_t` instance (see :ref:`Model`).

Creating a new ``diverge_model_t`` instance
-------------------------------------------

Call :c:func:`diverge_model_init`:

.. sourcecode:: C

    diverge_model_t* mod = diverge_model_init();

If desired, set the model name:

.. sourcecode:: C

    sprintf(mod->name, "My Fancy Model");

Set the dimension and the number of coarse and fine k points. Here, we use a 6×6
mesh with 5×5 refinement. As we don't touch mod->nk[2] and mod->nkf[2] (they are
zero by default) the library understands that the model is two dimensional.

.. sourcecode:: C

    mod->nk[0] = 6;
    mod->nk[1] = 6;
    mod->nkf[0] = 5;
    mod->nkf[1] = 5;

Set the lattice vectors. By default, all parameters that are left untouched are
zero.

.. sourcecode:: C

    mod->lattice[0][0] = 1.0;
    mod->lattice[1][1] = 1.0;
    mod->lattice[2][2] = 1.0;

Set the number of orbitals involved. Here, we keep it simple and use a
single-orbital model. For multiple orbitals, their positions must be indicated
in :c:member:`diverge_model_t.positions`.

.. sourcecode:: C

    mod->n_orb = 1;

Set the number of hopping elements, allocate the hopping elements (via
:c:func:`diverge_mem_alloc_rs_hopping_t` or plain malloc/calloc) and fill each
:c:struct:`rs_hopping_t` element of the array :c:member:`diverge_model_t.hop`.

.. sourcecode:: C

    mod->n_hop = 4;
    // alternatively, use an appropriate call to malloc()
    mod->hop = diverge_mem_alloc_rs_hopping_t(4);
    // structure: { {Rx,Ry,Rz}, o1,o2, s1,s2, t }
    mod->hop[0] = {{ 1,0,0}, 0,0,0,0, 1};
    mod->hop[1] = {{-1,0,0}, 0,0,0,0, 1};
    mod->hop[2] = {{0, 1,0}, 0,0,0,0, 1};
    mod->hop[3] = {{0,-1,0}, 0,0,0,0, 1};

We want to make use of :math:`SU(2)` symmetry and therefore only consider 'one'
spin. :math:`n_\mathrm{spin}` generally corresponds to :math:`2S+1`, i.e., the
number of possible spin z values (see :c:member:`diverge_model_t.n_spin`).

.. sourcecode:: C

    mod->SU2 = true;
    mod->n_spin = 1;

The :ref:`point group symmetry setup <Symmetries>` is a bit more complicated. It
can be neglected in many cases and usually merely improves runtimes. Now we come
to the vertex setup. Since we do a Hubbard model, this is extremely simple. One
vertex element of type :c:struct:`rs_vertex_t` put into the array
:c:member:`diverge_model_t.vert` that must be allocated (again, you *may* use
:c:func:`diverge_mem_alloc_rs_vertex_t` for convenience or plain malloc/calloc):

.. sourcecode:: C

    mod->n_vert = 1;
    mod->vert = diverge_mem_alloc_rs_vertex_t(1);
    // structure: { channel, {Rx,Ry,Rz}, o1,o2, s1,s2,s3,s4, V }
    mod->vert[0] = {'D', {0,0,0}, 0,0,0,0,0,0, 3};

It is *strongly advised* to validate whether the model input is correctly
interpreted by the library by calling :c:func:`diverge_model_validate`:

.. sourcecode:: C

    if (diverge_model_validate( mod ))
        diverge_mpi_exit(EXIT_FAILURE);

Optionally we can make the code calculate a non-interacting band structure by
defining the points of the irreducible BZ path in relative reciprocal lattice
vector coordinates (reciprocal crystal coordinates):

.. sourcecode:: C

    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 0.5;
    mod->ibz_path[2][1] = 0.5;

It is *mandatory* to call :c:func:`diverge_model_internals_common` on the
:c:struct:`diverge_model_t` in order to initialize internal data structures:

.. sourcecode:: C

    diverge_model_internals_common( mod );

Grid FRG
--------
Thereafter we have several choices. If we want the model to be prepared for a
*grid FRG calculation*, we call :c:func:`diverge_model_internals_grid`:

.. sourcecode:: C

    diverge_model_internals_grid( mod );

N-patch FRG
-----------
For an *N-patch FRG calculation*, we are missing the definition of a
:c:struct:`patching <mom_patching_t>`. Note that for N-patch, ``nkf[0] = nkf[1]
= 1`` is expected, as well as dimension 2 (set via ``nk[2] = nkf[2] = 0``).
``nk[0]`` and ``nk[1]`` should be large (compared to one). We can try to
automatically generate a patching by directly calling

.. sourcecode:: C

    diverge_model_internals_patch( mod, 6 );

where the second parameter of :c:func:`diverge_model_internals_patch` determines
the number of patches to be found in the IBZ. Fine-grained control over the
patching setup is provided through other functions documented :ref:`here
<Patching>`.

Truncated Unity FRG
-------------------
For a *Truncated Unity FRG calculation*, we can either define formfactors right
in the model by allocating and filling the :c:struct:`tu_formfactor_t` array
:c:member:`diverge_model_t.tu_ff` and setting its length, or we can
automatically search for formfactors when this array points to ``NULL`` (and is
of zero length). :c:func:`diverge_model_internals_tu` takes as second parameter
the maximum distance up to which formfactors shall be included, here 2.01:

.. sourcecode:: C

    diverge_model_internals_tu( mod, 2.01 );

Writing the ``diverge_model_t`` to a file
-----------------------------------------
As we might want to understand whether our model implementation is correct
without doing the numerically expensive FRG flow, we can save it to a file using
:c:func:`diverge_model_to_file`:

.. sourcecode:: C

    char* md5 = diverge_model_to_file( mod, "model.diverge" );

The md5sum of the model is saved in the string ``md5`` (which points to static
memory, i.e., doesn't need to be free'd). From python, we can read the
file ``model.diverge`` using the interface given in ``diverge.output`` (see
:ref:`Simulation Output`):

.. sourcecode:: python

    import diverge.output as ro
    M = ro.read('model.diverge')
    help(M)

Initializing the Flow Step Instance ``diverge_flow_step_t``
-----------------------------------------------------------
We must make a choice here: Which channels should be simulated and which backend
should be used. Note that the setup of the :c:struct:`diverge_model_t` internals
(using the functions :c:func:`diverge_model_internals_patch`,
:c:func:`diverge_model_internals_grid`, :c:func:`diverge_model_internals_tu`)
must coincide with your choice of method: To stick to the example above, we call
:c:func:`diverge_flow_step_init` as follows:

.. sourcecode:: C

    diverge_flow_step_t* st = diverge_flow_step_init( mod, "grid", "PCD" );
    if (!st) {
        mpi_err_printf("error in initializing the flow step\n");
        diverge_mpi_exit(EXIT_FAILURE);
    }

Users are alwys expected to write their own integrator (with some help given
through the structure :c:struct:`diverge_euler_t` and the function
:c:func:`diverge_euler_next`) and perform the flow using
:c:func:`diverge_flow_step_euler`. In practice, this looks like:

.. sourcecode:: C

    double Lambda = 10.0;
    for (int i=0; i<10; ++i) {

        diverge_flow_step_euler( st, Lambda, -1.0 );

        double vertmax;    diverge_flow_step_vertmax( st, &vertmax );
        double loopmax[2]; diverge_flow_step_loopmax( st, loopmax );
        double chanmax[3]; diverge_flow_step_chanmax( st, chanmax );

        mpi_eprintf( "%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
                Lambda, -1.0, loopmax[0], loopmax[1], chanmax[0],
                chanmax[1], chanmax[2], vertmax );

        Lambda -= 1.0;
    }

Notice the usage of the convenience function :c:func:`mpi_eprintf`, which allows
to print to ``stderr`` only on one of all MPI ranks. The functions
:c:func:`diverge_flow_step_vertmax`, :c:func:`diverge_flow_step_loopmax` and
:c:func:`diverge_flow_step_chanmax` return the vertex maximum, the loop maxima,
and the cahnnel maxima (here called after each flow step).

Postprocessing
--------------
The postprocessing procedure is fairly simple. We run
:c:func:`diverge_postprocess_and_write` and obtain output in the filename we
pass (``post.diverge``):

.. sourcecode:: C

    diverge_postprocess_and_write( st, "post.diverge" );

Again, we can use the Python module ``diverge.output`` (see :ref:`Simulation
Output`) to load the data into python:

.. sourcecode:: python

    import diverge.output as ro
    P = ro.read('post.diverge')
    help(P)

Finalize
--------
We must free the resources allocated during the setup and run. For the
:c:type:`diverge_flow_step_t` instance, simply call
:c:func:`diverge_flow_step_free`. For the :c:struct:`diverge_model_t` instance
and all attached memory, use :c:func:`diverge_model_free`. Finally, the MPI and
internal setup must be teared down. To do so, call :c:func:`diverge_finalize`
(if the library has been initialized with :c:func:`diverge_embed`, call
:c:func:`diverge_reset` instead):

.. sourcecode:: C

    diverge_flow_step_free( st );
    diverge_model_free( mod );
    diverge_finalize();

That's it; these were the steps required to write an FRG code. To see how to
extend this minimal example, take a look around this documentation or the
:release_tree:`examples in the git repo <examples>`.
