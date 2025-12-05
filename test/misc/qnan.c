#define CT_SUITE_NAME qnan
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/misc/qnan.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"
#include <time.h>

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
    double payload = -(double)clock() / 1234.;
    double fnan_with_payload = fnan_gen( payload );
    double regular_number = payload;

    mpi_log_printf( "quiet NaN floating point payload=%.5e\n", payload );
    CT_CHECK( fnan_isnan( fnan_with_payload ) );
    CT_CHECK( !fnan_isnan( regular_number ) );

    double payload_trunc = fnan_get(fnan_with_payload);
    mpi_log_printf( "truncated vs full payload: %.18e %.18e\n", payload, payload_trunc );
    double denom = fabs(payload);
    if (denom < 1.e-3) denom = 1.;
    CT_CHECK( fabs(payload_trunc-payload)/denom < 1.e-4 );
}

CT_TEST( int32 ) {
    uint16_t payload = clock();
    float qnan_with_payload = qnanf_gen( payload );
    float regular_number = payload;

    mpi_log_printf( "quiet NaNf unsigned payload=%u\n", payload );
    CT_CHECK( qnanf_isnan( qnan_with_payload ) );
    CT_CHECK( !qnanf_isnan( regular_number ) );
    CT_CHECK( qnanf_get( qnan_with_payload ) == payload );
}

CT_TEST( float32 ) {
    float payload = -(float)clock() / 1234.;
    float fnan_with_payload = fnanf_gen( payload );
    float regular_number = payload;

    mpi_log_printf( "quiet NaNf floating point payload=%.5e\n", payload );
    CT_CHECK( fnanf_isnan( fnan_with_payload ) );
    CT_CHECK( !fnanf_isnan( regular_number ) );

    float payload_trunc = fnanf_get(fnan_with_payload);
    mpi_log_printf( "truncated vs full payload: %.18e %.18e\n", payload, payload_trunc );
    float denom = fabsf(payload);
    if (denom < 1.e-3) denom = 1.;
    CT_CHECK( fabsf(payload_trunc-payload)/denom < 1.e-4 );
}

#endif
