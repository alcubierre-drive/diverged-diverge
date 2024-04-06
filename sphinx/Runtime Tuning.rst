.. _Runtime Tuning:

Runtime Tuning
==============

Besides the obvious tuning parameters exposed through the compile time options
(see :ref:`Compilation DEFINES`, :ref:`CUDA Compilation`, and :ref:`MPI
Compilation`), divERGe enables fine-grained CPU parallelization control using
the following environment variables:

    :``DIVERGE_OMP_NUM_THREADS=VAL``: set the (maximum) number of OpenMP threads
        that execute parallel regions in the core components of divERGe to
        ``VAR``. This specifically excludes parallelism in components like FFTW,
        OpenBLAS, etc. All loops within the source code are equipped with a
        ``num_threads()`` OpenMP clause. For user programs, a function that
        returns the number of threads is exported as
        :c:func:`diverge_omp_num_threads`.
    :``DIVERGE_FFTW_NUM_THREADS=VAL``: set the number of FFTW threads to ``VAL``.
    :``DIVERGE_OMP_IMPLICIT_NUM_THREADS=VAL``: set the number of OpenMP threads
        to ``VAL`` for all OpenMP regions that do not specify ``num_threads()``.
        Especially useful if linking to, e.g., OpenBLAS, where thread control is
        *not* possible apart from this global, implicit OpenMP setting.
    :``DIVERGE_SHARED_MALLOC_EXTRA_COLORS=VAL``: boolean flag to control
        additional splitting of shared memory communicators. may be benificial
        for performance in some cases; and very useful for debugging.
