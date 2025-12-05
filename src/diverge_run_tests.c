/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_run_tests.h"
#include "misc/compilation_status.h"
#include "misc/mpi_log.h"
#include "misc/quotes.h"
#include "misc/init_internal_libs.h"

#define CT_MAIN
#include "../test/ctest.h"
#include "../test/ctest_register.h"

int diverge_run_tests(int argc, char** argv ) {
    #ifndef DIVERGE_SKIP_TESTS
        if (!environ_loglevel_set()) mpi_loglevel_set(5);
        #ifdef CT_SKIP_INITIALIZERS
            ct__register__all();
        #endif
        return ct_list_or_run(argc, argv);
    #else // DIVERGE_SKIP_TESTS
        (void)argc; (void)argv;
        mpi_wrn_printf("Tests not compiled in. Here's a nice quote instead:\n%s",
            random_quote());
        return 1;
    #endif // DIVERGE_SKIP_TESTS
}
