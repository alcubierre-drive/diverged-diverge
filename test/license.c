#define CT_SUITE_NAME misc
#include "ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../src/misc/license.h"
#include "../src/misc/quotes.h"

CT_TEST_HIDDEN(license) {
    CT_CHECK( diverge_license() != NULL );
    diverge_license_print();
}

CT_TEST_HIDDEN(quote) {
    const char* quot = diverge_random_quote();
    CT_CHECK( quot != NULL );
    mpi_printf( "%s\n", quot );
}

#endif
