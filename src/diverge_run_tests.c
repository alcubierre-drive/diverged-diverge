#include "diverge_run_tests.h"
#include "misc/mpi_log.h"

#if !defined(INC_TEST) || !INC_TEST
int diverge_run_tests( int argc, char** argv ) {
    (void)argc;
    (void)argv;
    mpi_wrn_printf("tests not compiled in. consider recompilation with INC_TEST := true (Makefile.local)\n");
    return 1;
}
#endif

