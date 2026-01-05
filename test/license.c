#define CT_SUITE_NAME license
#include "ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../src/misc/license.h"

CT_TEST(license) {
    diverge_license_print();
}

#endif
