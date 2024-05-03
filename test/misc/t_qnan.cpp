#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../../src/misc/qnan.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"

TEST_CASE("quiet NaN, integers", "[qnan][int]") {

    uint32_t payload = clock();
    double qnan_with_payload = qnan_gen( payload );
    double regular_number = payload;

    mpi_log_printf( "quiet NaN unsigned payload=%lu\n", payload );
    CHECK( qnan_isnan( qnan_with_payload ) );
    CHECK( !qnan_isnan( regular_number ) );
    CHECK( qnan_get( qnan_with_payload ) == payload );
    CHECK( std::isnan(qnan_with_payload) );
}

TEST_CASE("quiet NaN, floats", "[qnan][float]") {

    float payload = -(float)clock() / 1234.;
    double fnan_with_payload = fnan_gen( payload );
    double regular_number = payload;

    mpi_log_printf( "quiet NaN floating point payload=%.5e\n", payload );
    CHECK( fnan_isnan( fnan_with_payload ) );
    CHECK( !fnan_isnan( regular_number ) );
    CHECK( fnan_get( fnan_with_payload ) == payload );
    CHECK( std::isnan(fnan_with_payload) );
}

#endif
