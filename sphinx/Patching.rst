.. _Patching:

Patching
========
Advanced npatch setup in divERGe

We can call :cpp:func:`diverge_patching_find_fs_pts` (or
:c:func:`diverge_patching_find_fs_pts_C` in plain C) to find the Fermi surface
and store it in a vector. Note that we can provide NULL as the energy to the
Fermi surface routine and make the code use its internals. To illustrate the
usage of the extended :c:struct:`mom_patching_t` struct generation, we add some
high symmetry points after finding the Fermi surface:

.. sourcecode:: C++

    vector<index_t> patches = diverge_patching_find_fs_pts( mod, NULL, 1, 4, 100 );
    // adding some high symmetry points
    patches.push_back(0);
    patches.push_back(mod->nk[0]/2*mod->nk[1]+mod->nk[1]/2);
    patches.push_back(mod->nk[0]/2*mod->nk[1]);
    patches.push_back(mod->nk[1]/2);

Thereafter, we must allocate and fill the :c:struct:`mom_patching_t` struct
:c:struct:`diverge_model_t` by calling :c:func:`diverge_patching_from_indices`:

.. sourcecode:: C++

    mod->patching = diverge_patching_from_indices( mod, patches.data(), patches.size() );

We can find an automatically optimized integration refinement with
:c:func:`diverge_patching_autofine` for each of the patches. Again, passing NULL
for the energies makes the code use its internals:

.. sourcecode:: C++

    diverge_patching_autofine( mod, mod->patching, NULL, 1, 40, 0.1, 2.0, 0.5 );

Notice that in the function calls to :c:func:`diverge_patching_autofine` and
:c:func:`diverge_patching_find_fs_pts` we put 1 as the number of bands. Passing
NULL as the energy array makes the library use its internals. If using autofine,
it is **strongly recommended** to also symmetrize the refinement with
:c:func:`diverge_patching_symmetrize_refinement` – for this to work you must
implement all the relevant :ref:`point-group symmetries <Symmetries>`.

.. sourcecode:: C++

    diverge_patching_symmetrize_refinement( mod, mod->patching );

Finally, we can call the patching internals :c:func:`diverge_model_internals_patch`:

.. sourcecode:: C++

    diverge_model_internals_patch( mod, -1 );

As ``mod->patching`` is now set, the second parameter is ignored in the above
call to :c:func:`diverge_model_internals_patch`.

.. cpp:autodoc:: diverge_patching.h

In summary, calling of the above functions in the model would look like:

.. sourcecode:: C

    index_t *patches, npatches;
    diverge_patching_find_fs_pts_C( mod, E, nb, np_ibz, np_ibz*10,
        &patches, &npatches );
    mom_patching_t* pat = diverge_patching_from_indices( mod, patches,
        npatches );
    free(patches);
    diverge_patching_autofine( mod, pat, E, nb, 10, 0.5, 4.0, 0.4 );
    diverge_patching_symmetrize_refinement( mod, pat );
    mod->patching = pat;


