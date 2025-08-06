/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_run_tests.h"
#include "misc/compilation_status.h"
#include "misc/mpi_log.h"
#include "misc/quotes.h"

#ifndef DIVERGE_SKIP_TESTS

// #define CATCH_CONFIG_RUNNER
// #include "../test/catch.hpp"

// for some reason this function is not defined but used in CATCH. Everything
// seems to work without putting anything useful into this function.
// #ifdef __ANDROID__
// extern "C" {
// int __android_log_write(int prio, const char* tag, const char* text) {
//     (void)prio; (void)tag; (void)text;
//     return 0;
// }
// }
// #endif

#define CT_MAIN
#define CT_PRINTF mpi_printf
#include "../test/ctest.h"
#include "../test/ctest_register.h"

int diverge_run_tests(int argc, char** argv ) {
    mpi_loglevel_set(5);
    #ifdef CT_SKIP_INITIALIZERS
    ct__register__all();
    #endif
    return ct_list_or_run(argc, argv);
    // Catch::Session session;
    // int returnCode = session.applyCommandLine( argc, argv );
    // if( returnCode != 0 )
    //     return returnCode;
    // int numFailed = session.run();
    // return numFailed;
}

#else // DIVERGE_SKIP_TESTS

int diverge_run_tests( int argc, char** argv ) {
    (void)argc;
    (void)argv;
    mpi_wrn_printf("Tests not compiled in. Here's a nice quote instead:\n%s",
            random_quote());
    return 1;
}

#endif // DIVERGE_SKIP_TESTS
