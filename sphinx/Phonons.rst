.. _Phonons:

Phonons
-------

We include facilities to implment the flow equations (in static TU
approximation) under the influence of non-interacting phonons coupled to the
electronic system. Even though only the phonon hopping and electron-phonon
vertices are required (see :c:func:`diverge_model_internals_tu_phonons`), it is
recommended to build a :c:type:`diverge_model_t` instance for the *phonons* as
well---that way, the phonon model can easily be checked.

Creating the lattice-spring model for the phonon dispersion is very similar to
creating tight-binding models for electrons. divERGe includes a function to make
the phonon model comply with the acoustic sum rule;
:c:func:`diverge_phonons_asr`.

The electron-phonon coupling is given in real-space (similar to the file format
in `EPW <https://epw-code.org/>`_).

.. c:autodoc:: diverge_phonons.h

.. c:autodoc:: misc/wigner.h
