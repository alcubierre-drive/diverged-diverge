#include "../src/diverge_run_tests.h"
#include "../src/misc/compilation_status.h"
#include <fftw3.h>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

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
    diverge_compilation_status();
    Catch::Session session;
    int returnCode = session.applyCommandLine( argc, argv );
    if( returnCode != 0 )
        return returnCode;
    int numFailed = session.run();
    fftw_cleanup();
    return numFailed;
}
