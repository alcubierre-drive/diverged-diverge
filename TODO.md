# TODOs
- doc: structured arrays in python (done?), struct datatype usage example
  symmetries from python (emery)
- code: include static self-energies in both grid and npatch, with fourier
  selection and cutoff in npatch
- checkpointing of flow step
- c library functions for reading model files

# CHANGES
- Makefile now uses targets:
  - lib
  - test
  - dist
  - clean
- Makefile now includes Makefile.local and Makefile.inc as depencencies for
  source files
- tests are now toggled with a single DEFINE, i.e. `-DDIVERGE_SKIP_TESTS`
- BHK defaults changed to something that runs on a laptop with reasonable time

- need to adjust documentation accordingly...

# POSSIBLE PITFALLS
- Hamiltonian is actually in C order, but our Eigensolvers assume F order. ->
  Eigenvectors are actually their conjugate. GF is also in F order (or, so to
  say, we effectively always calculate GF^T instead of GF).
