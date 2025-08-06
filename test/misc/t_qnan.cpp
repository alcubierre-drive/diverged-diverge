#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME quiet_nan
#include "../ctest.h"
// #include "../catch.hpp"
#include "../../src/misc/qnan.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"

CT_TEST( int ) {

    uint32_t payload = clock();
    double qnan_with_payload = qnan_gen( payload );
    double regular_number = payload;

    mpi_log_printf( "quiet NaN unsigned payload=%lu\n", payload );
    CT_CHECK( qnan_isnan( qnan_with_payload ) );
    CT_CHECK( !qnan_isnan( regular_number ) );
    CT_CHECK( qnan_get( qnan_with_payload ) == payload );
}

CT_TEST( float ) {

    float payload = -(float)clock() / 1234.;
    double fnan_with_payload = fnan_gen( payload );
    double regular_number = payload;

    mpi_log_printf( "quiet NaN floating point payload=%.5e\n", payload );
    CT_CHECK( fnan_isnan( fnan_with_payload ) );
    CT_CHECK( !fnan_isnan( regular_number ) );
    CT_CHECK( fnan_get( fnan_with_payload ) == payload );
}

#endif
