.. _Compilation:

Compilation
===========

Compilation of the :ref:`C Interface` ``libdivERGe.so``.

Minimal Requirements
--------------------
 - up-to-date C/C++ compiler (that can handle c++17 and c11)
 - a BLAS and LAPACK implementation including the CBLAS and LAPACKE interfaces
 - FFTW3, including its parallel (OpenMP or (p)threads) version
 - a linux machine is favorable (WSL 'just works', MacOS with some tweaks)

Make
----
To compile the library, navigate into the diverge directory (the git repo's
root) and execute make:

.. sourcecode:: bash

    cd /path/to/diverge/
    make -jXX

This will compile the :ref:`C Interface` with XX threads in parallel.
For installation of the python library (see :ref:`Simulation Output` and
:ref:`Python Interface`), execute

.. sourcecode:: bash

    pip install .

in the git repo's root. Note that you might want to create a virtual environment
first, or change the above pip call to

.. sourcecode:: bash

    pip install --editable . --config-settings editable_mode=strict --break-system-packages

The makefile exposes several targets to build specific components of divERGe:

+---------------+----------------------------------------------------------------------+
| Target        | Effect                                                               |
+===============+======================================================================+
|   ``lib``     | build the library, specifically, libdivERGe.so.                      |
+---------------+----------------------------------------------------------------------+
|    ``test``   | build an executable (divERGe) that does nothing but running          |
|               | :c:func:`diverge_run_tests`.                                         |
+---------------+----------------------------------------------------------------------+
|    ``dist``   | pack the divERGe.tgz file with the shared library, header files,     |
|               | python library, and examples                                         |
+---------------+----------------------------------------------------------------------+
|    ``clean``  | remove all build time files (object files .o, dependency files .d, …)|
+---------------+----------------------------------------------------------------------+

Changing the defaults
---------------------
To compile the code with different options/compiler/etc., you can create a file
'Makefile.local' in the top level directory of the git repository. This file is
sourced by the Makefile and therefore allows for full control.

To add, e.g., the non-standard FFTW3 path /opt/fftw/ to the include and link
locations, put

.. sourcecode:: make

    INCLUDES += -I/opt/fftw/include/
    LIBS += -L/opt/fftw/lib/

into the 'Makefile.local' (*this is an example, actual content will differ!*)

Examples for Makefile.local
---------------------------
The `make.local/ directory
<https://codeberg.org/lennart_klebl/divERGe/src/branch/master/make.local/>`_ of the git
repo contains various examples of how to compile divERGe on certain
machines/configurations.

Variables defined and used in 'Makefile'
----------------------------------------
Compilers etc.

.. sourcecode:: make

    CC := gcc # C compiler
    LD := g++ # linker
    CXX := g++ # C++ compiler
    CUCC := nvcc # CUDA compiler, only used when USE_CUDA := true
    AR := ar # if you don't know this you don't need to change it
    STRIP := strip # if you don't know this you don't need to change it

*usual suspects*

.. sourcecode:: make

    PREFLAGS = # flags that should come before anything else in CFLAGS/CXXFLAGS
    INCLUDES = # additional -I/some/where/ to find library headers
    DEFINES = # additional -DSOME_THING to play with features
    LIBS = -llapacke -lcblas -lfftw3 -lfftw3_omp # libraries

.. _MPI Compilation:

MPI Compilation
---------------

in your 'Makefile.local', put

.. sourcecode:: make

    DEFINES += -DUSE_MPI
    CC := mpicc
    CXX := mpicxx
    LD := mpicxx

Further optimizations for MPI parallelism can be enabled, they are toggled via
DEFINE keys (i.e. adding ``DEFINES += -DKEY`` to Makefile.local, see
:ref:`Compilation DEFINES` for a full list). MPI related :ref:`Compilation
DEFINES` are, in addition, listed here:

+----------------------------------+----------------------------------------------------------------+
| Compilation DEFINE               | Explanation                                                    |
+==================================+================================================================+
| ``USE_SHARED_MEM``               | experimental support for putting some objects                  |
|                                  | (Hamiltonian, Orbital/Band matrices) into                      |
|                                  | :ref:`node-global shared memory <Shared Memory>`,              |
|                                  | i.e., creating a shared memory pool that is used by            |
|                                  | all MPI ranks on a single node. This flag requires some        |
|                                  | more recent linux kernel features (``memfd_create``,           |
|                                  | ``mmap``).                                                     |
+----------------------------------+----------------------------------------------------------------+
| ``USE_MPI_FFTW``                 | use the MPI backend for FFTW in the TU propagator calculation. |
|                                  | this usually comes with an enormeous speedup                   |
+----------------------------------+----------------------------------------------------------------+
| ``MPI_VERBOSE_ALLOCATION_PRINT`` | be extra verbose on TU propagator                              |
|                                  | allocations in the new CPU MPI algorithm                       |
+----------------------------------+----------------------------------------------------------------+
| ``TRANSPOSE_MPI_FFT_BUFFERS``    | try to optimize memory access and usage by                     |
|                                  | defining some tensors in different ordering.                   |
|                                  | Only enabled when switching the                                |
|                                  | (undocumented)                                                 |
|                                  | tu_which_fft_{greens,simple,red} to the old                    |
|                                  | MPI FFT algorithm (see below...)                               |
+----------------------------------+----------------------------------------------------------------+

Some other experimental (*UNSUPPORTED*) MPI-related features are not enabled at
compile time but rather at runtime; and steered using
:c:func:`diverge_model_hack`. This includes the following config keys:

+------------------------------------+--------------------------------------------------------------+
| Config Key                         | Explanation                                                  |
+====================================+==============================================================+
|``int tu_which_fft_greens``         | swap the FFT algorithm for Green's function generation in the|
|                                    | TU propagator (int values see enumeration in source)         |
+------------------------------------+--------------------------------------------------------------+
|``int tu_which_fft_simple``         | swap the FFT algorithm for the simple TU propagator          |
|                                    | generation                                                   |
+------------------------------------+--------------------------------------------------------------+
|``int tu_which_fft_red``            | swap the FFT algorithm for the reduced TU propagator         |
|                                    | generation                                                   |
+------------------------------------+--------------------------------------------------------------+
|``model_shared_gf``                 | put the Green's function in node-global shared memory and    |
|                                    | overwrite the :c:member:`Green's function generator          |
|                                    | <diverge_model_t.gfill>` with a node-global shared memory    |
|                                    | variant. Look at the source code if you want to extend on    |
|                                    | this feature. We are not willing to properly support this    |
|                                    | feature (reasons: self-energy, consistency between backends, |
|                                    | user-defined Green's function generators, …) but see the     |
|                                    | potential benefits in terms of memory when using many MPI    |
|                                    | ranks on a single node (looking at you, AMD EPYC...)         |
+------------------------------------+--------------------------------------------------------------+
|``model_free_hamiltonian``          | free the hamiltonian array s.t. it doesn't consume valuable  |
|                                    | RAM. Most meaningful when, again, using many ranks.          |
+------------------------------------+--------------------------------------------------------------+
|``int tu_extra_propagator_timings`` | if !0, enable verbose timing of the (CPU/MPI) algorithms in  |
|                                    | the TU propagator                                            |
+------------------------------------+--------------------------------------------------------------+
|``int tu_mpifft_chunksize``         | set the number of FFTs executed in parallel in the CPU/MPI   |
|                                    | algorithm of the TU propagator to the given value; default is|
|                                    | the value of ``OMP_NUM_THREADS``.                            |
+------------------------------------+--------------------------------------------------------------+

.. _CUDA Compilation:

CUDA Compilation
----------------

in your 'Makefile.local', put

.. sourcecode:: make

    USE_CUDA := true

depending on your infrastructure, you might need to adjust ``CULIBS`` and
``INCLUDES`` to allow for finding the CUDA libraries and inclusion of the right
headers. Examples are found in the `make.local/ directory
<https://codeberg.org/lennart_klebl/divERGe/src/branch/master/make.local/>`_ of the git
repo.

.. _Compiling Tests:

Compiling Tests
---------------

The production builds usually include divERGe's test suite (based on
catch2.hpp). If you wish to get an executable usable to run the tests, compile
using

.. sourcecode:: bash

    make test -j

To run the tests call the executable from the command line with

.. sourcecode:: bash

    ./divERGe

.. _Advanced Compilation Features:

Advanced Compilation Features
-----------------------------

C/C++/CUDA flags are usually not changed manually, but still exposed through the
Makefile.local like this:

.. sourcecode:: make

    # usually no need to change directly
    CXXFLAGS = $(PREFLAGS) -std=c++17 -fPIC -fopenmp -MMD -Wall -Wextra -pedantic $(DEFINES) $(INCLUDES)
    CFLAGS = $(PREFLAGS) -std=c11 -fPIC -fopenmp -MMD -Wall -Wextra -pedantic $(DEFINES) $(INCLUDES)
    LDFLAGS = -fPIC -fopenmp -Wall -Wextra -pedantic $(LIBS)
    CUFLAGS = $(DEFINES) $(INCLUDES) $(DEVICES) -Xcompiler="-fPIC -fopenmp" -rdc=true
    CULDFLAGS = -Xcompiler="-fPIC"

Cuda libraries and device configuration options are set in the Makefile.local
like:

.. sourcecode:: make

    CULIBS = -lcudart -lcuda -lcufft -lcublas -lcusolver -lcusolverMg
    DEVICES = -gencode arch=compute_70,code=sm_70 \
    	  -gencode arch=compute_75,code=sm_75 \
    	  -gencode arch=compute_80,code=sm_80

Advanced linker options for both the library libdivERGe.so as well as the
testing executable divERGe can be set through:

.. sourcecode:: make

    # for linkage:
    # EXEFLAGS: linking of the main executable
    # SOFLAGS: linking of the library
    # LDSOFLAGS: LDFLAGS used for linking of the library
    EXEFLAGS = 
    SOFLAGS = -Wl,--version-script=libdivERGe.map -shared
    LDSOFLAGS = $(LDFLAGS)

.. _Compilation DEFINES:

Compilation DEFINES
-------------------

Various features are enabled/disabled using compile-time flags ("DEFINES").
Those mostly enable experimental, untested, or deprecated features. In general,
there exist compile time variables that toggle behavior when *defined* and other
ones that may *carry a value*. In the former case, you can extend the
Makefile.local with

.. sourcecode:: make

    DEFINES += -DVARIABLE_NAME

to have the variable ``VARIABLE_NAME`` defined. In the latter case, use

.. sourcecode:: make

    DEFINES += -DVARIABLE_NAME=VARIABLE_VAL

which sets the variable ``VARIABLE_NAME`` to ``VARIABLE_VAL``. An incomplete
list of compile time variables is given below, where either only the name is
written (i.e., the variable can be either *defined* or *not*), or
``VARIABLE_NAME=VARIABLE_VAL`` including an explanation (in which case the
*value* of the variable ``VARIABLE_VAL`` has an effect).

+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
| Compilation DEFINE                               | Explanation                                                                                       |
+==================================================+===================================================================================================+
|``BATCHED_EIGEN_NCHUNKS_AUTONUM=NUM``             | set default number of chunks in GPU batched eigensolver to NUM                                    |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``BATCHED_EIGEN_NCHUNKS_AUTOSIZE_GB=GB``          | set maximum size of chunk in GB for GPU batched eigensolver to GB                                 |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``BATCHED_GEMM_N_LL_NUM_EXHAUSTIVE_VARIANTS=VAL`` | compile all batched GEMMs where the matrix dimension is (much) smaller than the number of         |
|                                                  | executions up to matrix size VAL explicitly. default: 4.                                          |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``CUBLAS_BATCHED_GEMM_CALL=VAL``                  | use this function name for batched cublas gemm instead of the default                             |
|                                                  | "cublasZgemmStridedBatched_64" (may not be there in older CUDA versions)                          |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``CUDA_CIRCUMVENT_CONSTEXPR``                     | on old CUDA versions, constexpr may not be supported. define to circumvent this issue.            |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``CUDA_DEFINE_ATOMIC_ADD``                        | on old CUDA versions (or when compiling for old GPU architectures) atomicAdd is not defined.      |
|                                                  | defining this variable adds an implementation (that is usually slower than the actual one on      |
|                                                  | recent hardware)                                                                                  |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``CUDA_VERSION_STR=VAL``                          | set the CUDA version string to VAL (used in release builds)                                       |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_EPS_MESH=VAL``                          | set the difference at which two momentum mesh points are considered to be equal                   |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_GRID_PARALLEL_BATCHED_GEMM``            | if defined, do the grid FRG batched GEMM in parallel                                              |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_FLOW_STEP_EIGCHAN_PITER_ARPACK=VAL``    | integer constant that defines how to select arpack backend in :c:func:`diverge_flow_step_eigchan` |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_FLOW_STEP_EIGCHAN_PITER_MAXITER=VAL``   | integer constant that sets the default number of maximum iterations in the power iteration        |
|                                                  | backend of :c:func:`diverge_flow_step_eigchan`                                                    |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_FLOW_STEP_EIGCHAN_PITER_TOL=VAL``       | floating point constant that sets the default tolerance for power iteration backend of            |
|                                                  | :c:func:`diverge_flow_step_eigchan`                                                               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_LOG_COLORLESS``                         | if defined, do not use fancy colors in the log messages, but only bright white                    |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_LOG_GRAYSCALE``                         | if defined, use grayscale instead of fancy colors in the log messages. cleaner, more modern look  |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_LOG_NAMELESS``                          | do not print the log names                                                                        |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_LOG_PREFIX=VAL``                        | set the log prefix to VAL                                                                         |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_MODEL_MAGIC_NUMBER=VAL``                | change the magic number for the output files generated by :c:func:`diverge_model_to_file` to VAL  |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_MODEL_NBANDSTRUCTURE=VAL``              | change the default number of band structure points per segment to VAL                             |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_NO_ABORT_ON_GPU_ERROR``                 | if defined, diverge will not abort the program upon encountering a CUDA error.                    |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_NO_REMOVE_COLORS_FILE_OUTPUT``          | if defined, color escape sequences are not removed from the output even if it may be redirected   |
|                                                  | to a file                                                                                         |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_OPENBLAS_LIMIT_THREADS=VAL``            | limit the number of threads to be used in OpenBLAS calls to VAL. useful when the OpenBLAS library |
|                                                  | is compiled for a certain maximum number of threads that can be exceeded by modern cluster        |
|                                                  | architectures.                                                                                    |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_USE_ZHEEVD``                            | use LAPACK's ``zheevd`` algorithm instead of the newer ``zheevr`` in :c:func:`single_eigen`       |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``BATCHED_SVD_THREAD_CNT_MAX=VAL``                | limit the number of threads in :c:func:`batched_svd`                                              |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_GETLINE_MAX=VAL``                       | set the maximum number of characters in a line parsed with the (internal) text_io.h parser        |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_SKIP_TESTS``                            | do not compile :c:func:`tests <diverge_run_tests>` into the library                               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_QNAN_DISABLE_CLANG_OPTS``               | disable compiler optimizations for :ref:`IEEE 754 Quiet NaN`, may be needed for ``CC := clang``   |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_TU_INPLACE_PERMUTE``                    | enable in-place permutation algorithm for TU projection initialization by default (slower,        |
|                                                  | deprecated)                                                                                       |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_TU_GF_COARSIFY_FFT``                    | set the coarsification of the Green's function to happen via FFTs (in the selfenergy kernel).     |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_UNIQUE_ORBFF_FIND_HASHTABLE``           | set the default algorithm for finding unique orbital/formfactor combinations to an O(1) one which |
|                                                  | relies on a custom hashtable implementation (splitmix64)                                          |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``TU_POST_USE_BATCHED_SVD_U``                     | use the U-matrix of the SVD instead of the V-matrix for vertex diagonalization in TU post         |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``TU_PROJ_TIMING``                                | enable fine projection timing in TU                                                               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``TU_VERTEX_RECO_GPU_ZGEMM``                      | use ``cublasXt_zgemm`` for the optimized TU vertex reco, only faster for large formfactor counts  |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``LDOS_KERNEL_GPU_DEV_MAXNOS=VAL``                | set the maximum number of bands considered in a single CUDA thread for the LDOS GPU kernels       |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``LDOS_KERNEL_GPU_DEV_NTHREADS=VAL``              | set the number of threads per block to use in the LDOS GPU kernels                                |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``CT_SKIP_INITIALIZERS``                          | the `ctest.h <https://github.com/alcubierre-drive/ctest>`_ library relies on the compiler's       |
|                                                  | ability to add startup code to C functions (non-standard). This can be circumvented by defining   |
|                                                  | the initializer skip. Note that most reasonable compilers (gcc, clang, ...) support initializer   |
|                                                  | code, though.                                                                                     |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``CLOG_MISSING``                                  | provide a declaration of the ``math.h`` function ``clog`` that's missing on some platforms        |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``GIT_VERSION=VAL``                               | set the git version string to VAL                                                                 |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``GIT_VERSION_BRANCH=VAL``                        | set the git branch string to VAL                                                                  |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``GRID_FF_SHELL_DISTANCE=VAL``                    | set the maximum distance of formfactors to be generated in the grid FRG code to VAL               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``LEAVE_EIGEN_BROKEN``                            | do not turn off Eigen optimizations. They are nice in general, but do not allow to operate on the |
|                                                  | underlying memory transparently. Thus they are turned off by default.                             |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_POST_MAGIC_NUMBER_GRID=VAL``            | set the grid FRG post processing magic number to VAL                                              |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_POST_MAGIC_NUMBER_NPATCH=VAL``          | set the npatch FRG post processing magic number to VAL                                            |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MAX_NAME_LENGTH=VAL``                           | define the maximum length of the character array holding :c:member:`diverge_model_t.name`         |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MAX_N_ORBS=VAL``                                | define the maximum number of orbitals to be used in :c:struct:`diverge_model_t`                   |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MAX_N_SYM=VAL``                                 | define the maximum length of symmetry arrays being used in :c:struct:`diverge_model_t`            |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MAX_ORBS_PER_SITE=VAL``                         | define the maximum number of orbitals on a single site used in :c:struct:`site_descr_t`           |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MAX_THREADS_PER_BLOCK=VAL``                     | set the maximum number of CUDA threads in a CUDA block for the npatch GPU kernels                 |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MKL_FORCE_AVX_KERNEL_ON_EPYC``                  | if defined explicitly force MKL to use AVX kernels on AMD EPYC CPUs. Might improve performance    |
|                                                  | (*MAYBE OUTDATED*)                                                                                |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``NDEBUG``                                        | standard DEFINE that has some implications in divERGe as well (smaller messages, ...)             |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``NDEBUG_SKIP_ERROR_LINES``                       | skip file and line information in error messages                                                  |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``NPATCH_ALMOST_ZERO=VAL``                        | define VAL that is seen to be zero for floating point comparisons in the npatch code              |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``NPATCH_PERIODIC_NORM_NG=VAL``                   | for momentum additions/subtractions that extend over more than one primitive zone, define the     |
|                                                  | range in G vectors that is searched                                                               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``SKIP_EIGEN3_INCLUDE_PREFIX``                    | some clusters install eigen3 in a way equivalent to /usr/include/Eigen instead of                 |
|                                                  | /usr/include/eigen3/Eigen. This flag is there to enable the former.                               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``SKIP_FLOW_TIMING``                              | do not time the flow step in the grid FRG code                                                    |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``TAG_VERSION=VAL``                               | set the git tag version string to VAL                                                             |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``DIVERGE_POST_MAGIC_NUMBER_TU=VAL``              | set the TU backend post processing magic number (relevant for output) to VAL                      |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_ARPACK_EIGENSOLVER``                        | compile an ARPACK-based eigensolver that can be used for :c:func:`diverge_flow_step_eigchan`.     |
|                                                  | Linkage to ``-larpack`` has to be ensured in ``Makefile.local``.                                  |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_COMPRESSED_OUTPUT``                         | compress the output files using Zstandard. Linkage to ``-lzstd`` has to be ensured in             |
|                                                  | ``Makefile.local``.                                                                               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_DYNAMIC_RECONSTRUCT_SCHEDULE``              | use ``omp parallel for schedule(dynamic)`` for the TU propagator reconstruction loop              |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MERGESORT_DEFAULT_TASK_SIZE=VAL``               | in the parallel mergesort implementation shipped with divERGe set the default *serial* task size  |
|                                                  | to ``VAL``.                                                                                       |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_ALTERNATIVE_TIME``                          | use an alternative method of measuring time when compiling without MPI                            |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_EIGEN_BATCHED_GEMM``                        | use Eigen's matrix multiplication in batched gemms (CPU)                                          |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_EIGEN3_CPU_EIGENSOLVER``                    | default to a CPU version of the batched eigensolver, built with Eigen3. Can be turned off at run  |
|                                                  | time via :c:func:`batched_eigen_r_set_eigen3_mode`.                                               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_EIGEN3_CPU_EIGENSOLVER_EXTENSIVE=VAL``      | compile extensive fixed-size versions (up to dimension ``VAL`` up to ``VAL = 16`` for the CPU     |
|                                                  | eigensolver built with Eigen3. May be faster for small-size matrices, but compilation times are   |
|                                                  | terrible.                                                                                         |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_GF_FLOATS``                                 | save Green's function in single point precision. should work for the npatch backend.              |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_MKL``                                       | use ``#include <mkl.h>`` instead of ``#include <lapacke.h>`` and ``#include <cblas.h>``           |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_MPI``                                       | enable MPI parallelization (:ref:`MPI Compilation`). Not recommended in conjunction with GPUs     |
|                                                  | (:ref:`CUDA Compilation`) as GPU/MPI algorithms are scarcely available and therefore not          |
|                                                  | implemented.                                                                                      |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_NO_BLAS_VERTEX_LOOP``                       | in the vertex/loop batched GEMMs of the grid FRG code, skip BLAS calls and do the multiplication  |
|                                                  | by hand.                                                                                          |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_NO_LAPACKE``                                | do not use LAPACK functions (or, their C wrappers) but Eigen.                                     |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_SERIAL_FFTW``                               | do not enable multithreaded FFTW.                                                                 |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_SHARED_MEM``                                | experimental support for putting some objects (Hamiltonian, Orbital/Band matrices) into           |
|                                                  | node-global shared memory, i.e., creating a shared memory pool that is used by all MPI ranks on a |
|                                                  | single node. This flag requires some more recent linux kernel features (``memfd_create``,         |
|                                                  | ``mmap``).                                                                                        |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``SHARED_MEM_MPI_IMPL``                           | use ``MPI_Win_XXX`` (and friends) for the implementation of the shared memory. Might behave       |
|                                                  | strangely on clusters where pinning to certain sockets is enforced.                               |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``SHARED_MEM_POSIX_IMPL``                         | use the POSIX functions ``shm_open`` and ``shm_unlink`` to create shared memory regions. In       |
|                                                  | Linux, these are then typically located in ``/dev/shm/``.                                         |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MEMFD_NOT_SUPPORTED``                           | use a syscall to ``SYS_memfd_create`` instead of the userspace API, required on old kernels       |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``USE_MPI_FFTW``                                  | use the MPI backend for FFTW in the TU propagator calculation. this usually comes with an         |
|                                                  | enormeous speedup. Users are responsible for adding the corresponding linker flags to the         |
|                                                  | Makefile.local.                                                                                   |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``PACK_MPI_MULTIPLIER=VAL``                       | set the MPI packing size to ``VAL`` in the tu mpi propagator kernels. default: 128. required      |
|                                                  | because MPI is a 32bit API and some arrays exceed this limit.                                     |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``TU_GPU_MPI_ALLOC_PRINT``                        | be verbose in TU GPU/MPI kernels about memory allocations.                                        |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``CUDA_MEM_ALIGN=VAL``                            | algin host memory in TU propagator kernels to ``VAL`` bytes for fast CUDA transfer                |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``MPI_VERBOSE_ALLOCATION_PRINT``                  | be extra verbose on TU propagator allocations in the new CPU/MPI algorithms                       |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
|``TRANSPOSE_MPI_FFT_BUFFERS``                     | try to optimize memory access and usage by defining some tensors in different ordering. Only      |
|                                                  | enabled when switching the TU propagator algorithms via :c:func:`diverge_model_hack`              |
|                                                  | (*unsupported*)                                                                                   |
+--------------------------------------------------+---------------------------------------------------------------------------------------------------+
