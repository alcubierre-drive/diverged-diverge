Linear Algebra
==============

divERGe exports some of its linear algebra functions to the C interface. They
are useful, because they provide convenient wrappers around annoying CUDA
functions when compiling with GPU support.

.. c:autodoc:: misc/batched_eigen.h
.. c:autodoc:: misc/batched_svd.h
.. c:autodoc:: misc/batched_gemms.h
.. c:autodoc:: misc/batched_gemm_small.h
.. c:autodoc:: misc/batched_zgemd.h
.. c:autodoc:: misc/eigen.h
.. c:autodoc:: misc/gemm.h
.. c:autodoc:: misc/invert.h
.. c:autodoc:: misc/linalg.h
