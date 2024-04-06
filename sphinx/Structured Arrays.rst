.. _Structured Arrays:

Structured Arrays
=================

DivERGe defines several primitive datatypes (structures) that can be
interfaced to from this python library:

    * :c:type:`complex128_t`
    * :c:struct:`rs_hopping_t`
    * :c:struct:`rs_vertex_t`
    * :c:struct:`tu_formfactor_t`
    * :c:struct:`sym_op_t`
    * :c:struct:`site_descr_t`

Allocation
----------

To allocate an array that is *not* garbage collected from python (and can thus
be free'd from within divERGe or via :c:func:`diverge_mem_free` (*note: the
python name for this function is diverge.mem_free*),
call :func:`diverge.alloc_array`. The datatype is passed as
*string* to this function, in contrast to what one may be used to from numpy.
For standard, garbage-collected allocation, use ``diverge.zeros``, which is a
wrapper for ``numpy.zeros``:

.. sourcecode:: python

    hopping_gc = diverge.zeros( (128,), dtype=diverge.rs_hopping_t )
    hopping_non_gc = diverge.alloc_array( (128,), dtype="rs_hopping_t" )

View
----

A numpy structured array can be used just as any other numpy array (shapes,
views, etc). To conveniently use memory owned by divERGe in a pythonic way, we
added :func:`diverge.view_array`:

.. sourcecode:: python

    positions = diverge.view_array( model.contents.positions, dtype=np.float64, shape=(12,3) )
    positions[:] += 1 # now positions are updated!

Access
------

Now to the usage of structured arrays (or array views). If one or more full
element(s) is (are) to be set, a (list of) tuple(s) can be passed as so:

.. sourcecode:: python

    hopping = diverge.alloc_array( (128,), dtype="rs_hopping_t" )
    hopping[0] = ( [0,0,0], 0,0, 0,0, (1.0,0.0) )
    hopping[1:3] = [ ( [0,0,0], 1,1, 0,0, (0.2,0.0) ),
                     ( [0,0,0], 2,2, 0,0, (0.2,0.0) ) ]

This works because the inserted tuples are understood as compatible with the
:c:struct:`rs_hopping_t` data type:

.. sourcecode:: C

    // C declaration
    struct rs_hopping_t {
      index_t R[3];
      index_t o1,o2, s1,s2;
      complex128_t t;
    };

If only a single member of the structure should be modified or accessed, a
dictionary-like syntax is supported:

.. sourcecode:: python

    vertex = diverge.alloc_array( (128,), dtype="rs_vertex_t" )
    vertex[0]['chan'] = 'D'
    vertex[0]['V'] = (3.0,0.0)

The keys of this dictionary-like access resemble the names of the members in
the respective C declaration.

