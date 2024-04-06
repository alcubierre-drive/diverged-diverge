Logging
=======

divERGe includes a C style logging facility with a bunch of macros
``mpi_???_printf`` and a loglevel setter :c:func:`mpi_loglevel_set`. For usage
from python, we provide :c:func:`mpi_py_print` and friends.

.. c:autodoc:: misc/mpi_log.h
