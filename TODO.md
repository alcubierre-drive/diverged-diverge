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
- _need to adjust documentation accordingly..._

- diverge_postprocess_config_t -> diverge_postprocess_conf_t
- diverge_postprocess_config_default -> diverge_postprocess_conf_defaults
