MPI Output
==========

For easier MPI output from within Python, we provide the class
:class:`diverge.mpi_stdout_logger` that replaces the standard ``stdout`` by a
class that only writes if the MPI rank is 0. Notably, this class writes to both
a file and stdout, therefore usage is as follows:

.. sourcecode:: Python

   import diverge
   import sys
   stdout_ = sys.stdout
   sys.stdout = mpi_stdout_logger( "output_file.dat" )
   print( "some stuff" )
   sys.stdout = stdout_
