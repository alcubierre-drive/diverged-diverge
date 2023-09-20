/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_run_tests.h"
#include "misc/compilation_status.h"
#include "misc/mpi_log.h"
#include "misc/quotes.h"

#ifndef DIVERGE_SKIP_TESTS

#define CATCH_CONFIG_RUNNER
#include "../test/catch.hpp"

// for some reason this function is not defined but used in CATCH. Everything
// seems to work without putting anything useful into this function.
#ifdef __ANDROID__
extern "C" {
int __android_log_write(int prio, const char* tag, const char* text) {
    (void)prio; (void)tag; (void)text;
    return 0;
}
}
#endif

int diverge_run_tests(int argc, char** argv ) {
    mpi_loglevel_set(5);
    Catch::Session session;
    int returnCode = session.applyCommandLine( argc, argv );
    if( returnCode != 0 )
        return returnCode;
    int numFailed = session.run();
    return numFailed;
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
